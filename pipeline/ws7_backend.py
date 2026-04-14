"""
pipeline/ws7_backend.py
------------------------
WS7: FastAPI backend — exposes the WS6 pipeline as a REST API.

Endpoints
---------
  POST /analyze
      Upload a resume (PDF/DOCX) + user inputs.
      Runs the full WS6 LangGraph pipeline.
      Returns ranked ScoredJob list.

  GET  /jobs
      Browse all scraped jobs from the silver layer.
      Supports filtering by company and target-role flag.
      Supports pagination.

  GET  /company/{name}/sponsorship
      Returns PERM + USCIS sponsorship history for a company.

  POST /refresh
      Triggers WS1 (scraper) + WS2 (sponsorship) + WS4 (index build)
      as a background task. Returns 202 immediately.

  GET  /health
      Liveness check. Returns data layer status.

Run
---
  cd /path/to/ADSFinal_VisaSystem
  /opt/anaconda3/bin/python -m uvicorn pipeline.ws7_backend:app --reload --port 8000

  Docs available at: http://localhost:8000/docs
"""

import logging
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.candidate_profile import UserInputs
from models.scored_job import ScoredJob
from pipeline.ws2_build_sponsorship import normalize_company
from pipeline.ws4_job_matcher import EMBEDDINGS_PATH, JOB_INDEX_PATH, JOBS_SILVER
from pipeline.ws5_confidence_scorer import SPONSORSHIP_PARQUET
from pipeline.ws6_langgraph_pipeline import run_pipeline

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "VisaMatch API",
    description = "AI-powered job matching for international candidates (F-1 OPT / H-1B)",
    version     = "0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten when deploying
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Background refresh state ───────────────────────────────────────────────────

_refresh_lock   = threading.Lock()
_refresh_status = {"running": False, "last_run": None, "last_error": None}


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeResponse(BaseModel):
    total:   int
    results: list[ScoredJob]


class JobSummary(BaseModel):
    job_id:         str
    company_slug:   str
    title:          str
    location:       str
    apply_url:      str
    posted_at:      str
    is_active:      bool
    is_target_role: bool


class JobListResponse(BaseModel):
    total:    int
    page:     int
    per_page: int
    results:  list[JobSummary]


class SponsorshipResponse(BaseModel):
    company_name_norm:     str
    company_name_raw:      Optional[str]   = None
    total_perm_certified:  int             = 0
    total_perm_denied:     int             = 0
    recent_perm_certified: int             = 0
    recent_perm_denied:    int             = 0
    perm_approval_rate:    Optional[float] = None
    recent_approval_rate:  Optional[float] = None
    trend_direction:       Optional[str]   = None
    data_freshness_score:  Optional[float] = None
    uscis_total_approvals: int             = 0
    uscis_total_denials:   int             = 0
    perm_by_year:          Optional[str]   = None


class RefreshResponse(BaseModel):
    accepted: bool
    message:  str


class HealthResponse(BaseModel):
    status:              str
    jobs_silver_exists:  bool
    sponsorship_exists:  bool
    index_exists:        bool
    refresh_running:     bool
    last_refresh:        Optional[str]
    last_error:          Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_jobs_df() -> pd.DataFrame:
    if not JOBS_SILVER.exists():
        raise HTTPException(
            status_code=503,
            detail="Job silver layer not found. POST /refresh to build it.",
        )
    return pd.read_parquet(JOBS_SILVER)


def _load_sponsorship_df() -> pd.DataFrame:
    if not SPONSORSHIP_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail="Sponsorship data not found. POST /refresh to build it.",
        )
    return pd.read_parquet(SPONSORSHIP_PARQUET)


def _run_refresh_background(force: bool = True) -> None:
    """Runs WS1 + WS2 + WS4 index build in a background thread."""
    global _refresh_status
    with _refresh_lock:
        if _refresh_status["running"]:
            return
        _refresh_status["running"]    = True
        _refresh_status["last_error"] = None

    try:
        log.info("[WS7/refresh] Starting background data refresh...")

        # ── WS1: scrape Greenhouse ─────────────────────────────────────────────
        from Scrapers.Job_Scraping import save_bronze, scrape_all
        from Scrapers.config import COMPANIES
        from pipeline.ws1_run_scraper import build_silver, save_silver

        run_time = datetime.now(timezone.utc)
        records  = scrape_all(companies=COMPANIES, run_time=run_time)
        save_bronze(records, run_time=run_time)
        df_silver = build_silver()
        if not df_silver.empty:
            save_silver(df_silver)
        log.info("[WS7/refresh] WS1 done — %d jobs", len(df_silver))

        # ── WS2: rebuild sponsorship history ───────────────────────────────────
        from pipeline.ws2_build_sponsorship import (
            SILVER_CSV, SILVER_DIR, SILVER_PARQUET,
            aggregate_perm, load_all_perm, load_uscis,
        )
        perm_raw = load_all_perm()
        perm_agg = aggregate_perm(perm_raw)
        uscis    = load_uscis()
        combined = perm_agg.merge(uscis, on="company_name_norm", how="outer")
        for col in ("uscis_total_approvals", "uscis_total_denials"):
            combined[col] = combined[col].fillna(0).astype(int)
        combined = combined.sort_values(
            "total_perm_certified", ascending=False
        ).reset_index(drop=True)
        for col in ("naics", "state", "company_name_raw"):
            combined[col] = combined[col].astype(str)
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(SILVER_PARQUET, index=False)
        combined.to_csv(SILVER_CSV, index=False, encoding="utf-8-sig")
        log.info("[WS7/refresh] WS2 done — %d companies", len(combined))

        # ── WS4: rebuild embedding index ───────────────────────────────────────
        from pipeline.ws4_job_matcher import build_index
        build_index(force=True)
        log.info("[WS7/refresh] WS4 index built")

        _refresh_status["last_run"] = datetime.now(timezone.utc).isoformat()
        log.info("[WS7/refresh] Refresh complete")

    except Exception as exc:
        log.error("[WS7/refresh] Failed: %s", exc)
        _refresh_status["last_error"] = str(exc)
    finally:
        _refresh_status["running"] = False


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── POST /analyze ─────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model = AnalyzeResponse,
    summary        = "Match a resume against scraped jobs",
)
async def analyze(
    resume:               UploadFile = File(..., description="Resume PDF or DOCX"),
    visa_status:          str        = Form(default="F-1 OPT"),
    requires_sponsorship: bool       = Form(default=True),
    target_roles:         str        = Form(
                              default="Data Engineer,Analytics Engineer",
                              description="Comma-separated list of target roles",
                          ),
    preferred_locations:  str        = Form(
                              default="Remote",
                              description="Comma-separated list of preferred locations",
                          ),
    open_to_relocation:   bool       = Form(default=True),
    top_n:                int        = Form(default=15, ge=1, le=50),
):
    """
    Upload a resume (PDF or DOCX) with your job preferences.

    Returns a ranked list of matching jobs with:
    - Semantic match score
    - Visa sponsorship confidence (for international candidates)
    - Recency score
    - Plain-English reasoning per job
    """
    # Validate file type
    suffix = Path(resume.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a PDF or DOCX.",
        )

    # Check job index is ready
    if not EMBEDDINGS_PATH.exists() or not JOB_INDEX_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Job index not built yet. POST /refresh first, then retry.",
        )

    # Save upload to a temp file, run pipeline, clean up
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(resume.file, tmp)
            tmp_path = Path(tmp.name)

        user_inputs = UserInputs(
            visa_status          = visa_status,
            requires_sponsorship = requires_sponsorship,
            target_roles         = [r.strip() for r in target_roles.split(",") if r.strip()],
            preferred_locations  = [l.strip() for l in preferred_locations.split(",") if l.strip()],
            open_to_relocation   = open_to_relocation,
        )

        log.info("[WS7/analyze] Pipeline starting for '%s'", resume.filename)
        scored = run_pipeline(tmp_path, user_inputs, top_n=top_n)
        log.info("[WS7/analyze] Done — %d results returned", len(scored))

        return AnalyzeResponse(total=len(scored), results=scored)

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


# ── GET /jobs ─────────────────────────────────────────────────────────────────

@app.get(
    "/jobs",
    response_model = JobListResponse,
    summary        = "Browse all scraped jobs",
)
def list_jobs(
    company:     Optional[str]  = None,
    target_role: Optional[bool] = None,
    active_only: bool           = True,
    page:        int            = 1,
    per_page:    int            = 25,
):
    """
    Returns paginated jobs from the silver layer.

    - `company` — filter by company slug (e.g. `stripe`, `airbnb`)
    - `target_role=true` — only Data Engineer / Analyst / Scientist roles
    - `active_only=true` — only jobs seen in the last 14 days
    """
    df = _load_jobs_df()

    if active_only and "is_active" in df.columns:
        df = df[df["is_active"] == True]

    if company:
        df = df[df["company_slug"].str.contains(company.lower(), case=False, na=False)]

    if target_role is not None and "is_target_role" in df.columns:
        df = df[df["is_target_role"] == target_role]

    total   = len(df)
    offset  = (page - 1) * per_page
    page_df = df.iloc[offset : offset + per_page]

    results = [
        JobSummary(
            job_id         = str(row.get("job_id", "")),
            company_slug   = str(row.get("company_slug", "")),
            title          = str(row.get("title", "")),
            location       = str(row.get("location", "")),
            apply_url      = str(row.get("apply_url", "")),
            posted_at      = str(row.get("posted_at", "")),
            is_active      = bool(row.get("is_active", False)),
            is_target_role = bool(row.get("is_target_role", False)),
        )
        for _, row in page_df.iterrows()
    ]

    return JobListResponse(total=total, page=page, per_page=per_page, results=results)


# ── GET /company/{name}/sponsorship ───────────────────────────────────────────

@app.get(
    "/company/{name}/sponsorship",
    response_model = SponsorshipResponse,
    summary        = "Get sponsorship history for a company",
)
def company_sponsorship(name: str):
    """
    Returns PERM + USCIS visa sponsorship history for a company.

    `name` can be a slug (`stripe`) or display name (`Stripe Inc`).
    Uses the same 3-tier fuzzy match as the scoring pipeline:
    exact → prefix → substring.
    """
    df   = _load_sponsorship_df()
    norm = normalize_company(name)

    match_df = df[df["company_name_norm"] == norm]
    if match_df.empty:
        match_df = df[df["company_name_norm"].str.startswith(norm, na=False)]
    if match_df.empty and len(norm) > 4:
        match_df = df[df["company_name_norm"].str.contains(norm, na=False, regex=False)]

    if match_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No sponsorship data found for '{name}'. "
                "Try a shorter name or check /jobs for valid company slugs."
            ),
        )

    row = match_df.iloc[0]

    def _int(col):   return int(row[col]) if col in row.index and pd.notna(row[col]) else 0
    def _float(col): return float(row[col]) if col in row.index and pd.notna(row[col]) else None
    def _str(col):   return str(row[col]) if col in row.index and pd.notna(row[col]) else None

    return SponsorshipResponse(
        company_name_norm     = str(row["company_name_norm"]),
        company_name_raw      = _str("company_name_raw"),
        total_perm_certified  = _int("total_perm_certified"),
        total_perm_denied     = _int("total_perm_denied"),
        recent_perm_certified = _int("recent_perm_certified"),
        recent_perm_denied    = _int("recent_perm_denied"),
        perm_approval_rate    = _float("perm_approval_rate"),
        recent_approval_rate  = _float("recent_approval_rate"),
        trend_direction       = _str("trend_direction"),
        data_freshness_score  = _float("data_freshness_score"),
        uscis_total_approvals = _int("uscis_total_approvals"),
        uscis_total_denials   = _int("uscis_total_denials"),
        perm_by_year          = _str("perm_by_year"),
    )


# ── POST /refresh ─────────────────────────────────────────────────────────────

@app.post(
    "/refresh",
    response_model = RefreshResponse,
    status_code    = 202,
    summary        = "Trigger background data refresh",
)
def trigger_refresh(background_tasks: BackgroundTasks):
    """
    Starts a background job that re-runs:
    1. WS1 — scrapes Greenhouse for fresh job listings
    2. WS2 — rebuilds sponsorship_history.parquet
    3. WS4 — rebuilds the job embedding index

    Returns 202 immediately. Poll `GET /health` to check when complete.
    """
    if _refresh_status["running"]:
        return RefreshResponse(
            accepted=False,
            message="A refresh is already running. Check GET /health for status.",
        )

    background_tasks.add_task(_run_refresh_background, True)
    return RefreshResponse(
        accepted=True,
        message="Refresh started. Poll GET /health to check completion.",
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Liveness and data layer status check",
)
def health():
    """
    Returns liveness status and whether all data dependencies are ready.

    All three of `jobs_silver_exists`, `sponsorship_exists`, and
    `index_exists` must be `true` before `/analyze` will work.
    """
    return HealthResponse(
        status             = "ok",
        jobs_silver_exists = JOBS_SILVER.exists(),
        sponsorship_exists = SPONSORSHIP_PARQUET.exists(),
        index_exists       = EMBEDDINGS_PATH.exists() and JOB_INDEX_PATH.exists(),
        refresh_running    = _refresh_status["running"],
        last_refresh       = _refresh_status["last_run"],
        last_error         = _refresh_status["last_error"],
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "pipeline.ws7_backend:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        app_dir = str(_ROOT),
    )
