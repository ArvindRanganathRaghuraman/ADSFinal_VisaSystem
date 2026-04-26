"""
pipeline/ws6_langgraph_pipeline.py
------------------------------------
WS6: Master LangGraph orchestration layer.

Wires ALL workstreams into a single typed graph. Each workstream
becomes a named agent node. State flows linearly through the graph;
every agent is self-aware and skips work it doesn't need to redo.

Agent nodes (in execution order)
---------------------------------
  ScraperAgent           — WS1: scrape Greenhouse → bronze/silver jobs
  SponsorshipAgent       — WS2: PERM + USCIS → sponsorship_history.parquet
  IndexBuilderAgent      — WS4 (build): embed target-role jobs → index
  ResumeParserAgent      — WS3: PDF/DOCX + user inputs → CandidateProfile
  MatchingAgent          — WS4 (query): CandidateProfile → top-N JobMatches
  SignalExtractionAgent  — WS5 partial: scan JD text → per-job sponsorship signal
  HistoricalEvidenceAgent— WS5 partial: query PERM silver → per-company evidence
  AuditorAgent           — WS5 final: cross-validate, assign confidence, rank

Graph layout
------------
  [scraper] → [sponsorship] → [index_builder]
      → [resume_parser] → [matching]
      → [signal_extraction] → [historical_evidence]
      → [auditor] → END

  Each node short-circuits to END on error.
  Data-pipeline nodes (scraper, sponsorship, index_builder) are
  skipped automatically when their outputs already exist, unless
  --refresh is passed.

Two public entry points
-----------------------
  run_pipeline(resume_path, user_inputs, refresh=False)
      Full pipeline. WS1→WS2→WS3→WS4→WS5.

  run_pipeline_from_profile(profile, user_inputs, refresh=False)
      Skip WS3. Inject a pre-built CandidateProfile → WS4→WS5.
      Use for testing the graph without a resume PDF.

CLI
---
  # Full pipeline with a real resume
  python -m pipeline.ws6_langgraph_pipeline --resume path/to/resume.pdf

  # Demo mode — built-in sample profile, no PDF needed
  python -m pipeline.ws6_langgraph_pipeline --demo

  # Force re-scrape + rebuild everything first
  python -m pipeline.ws6_langgraph_pipeline --demo --refresh
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, StateGraph

# ── WS1 ───────────────────────────────────────────────────────────────────────
from Scrapers.Job_Scraping import save_bronze, scrape_all
from Scrapers.config import COMPANIES
from pipeline.ws1_run_scraper import build_silver, save_silver

# ── WS2 ───────────────────────────────────────────────────────────────────────
from pipeline.ws2_build_sponsorship import (
    SILVER_CSV        as SPONSORSHIP_CSV,
    SILVER_DIR        as SPONSORSHIP_SILVER_DIR,
    SILVER_PARQUET    as SPONSORSHIP_PARQUET,
    aggregate_perm,
    load_all_perm,
    load_uscis,
    normalize_company,
)

# ── WS3 ───────────────────────────────────────────────────────────────────────
from pipeline.ws3_resume_parser import parse_resume

# ── WS4 ───────────────────────────────────────────────────────────────────────
from pipeline.ws4_job_matcher import (
    EMBEDDINGS_PATH,
    JOB_INDEX_PATH,
    JOBS_SILVER,
    build_index,
    match,
)

# ── WS5 ───────────────────────────────────────────────────────────────────────
from pipeline.ws5_confidence_scorer import (
    W_DOM_MATCH,
    W_DOM_RECENCY,
    W_INT_MATCH,
    W_INT_RECENCY,
    W_INT_VISA,
    _build_lookup,
    _find_company,
    _load_sponsorship_df,
    _perm_base_score,
    _recency_score,
    _scan_jd,
)

# ── Models ─────────────────────────────────────────────────────────────────────
from models.candidate_profile import CandidateProfile, UserInputs
from models.job_match import JobMatch
from models.scored_job import ScoredJob

log = logging.getLogger(__name__)


# ── Demo profile (no resume PDF needed) ───────────────────────────────────────

DEMO_PROFILE = CandidateProfile(
    name                 = "Demo Candidate",
    email                = "demo@example.com",
    technical_skills     = ["Python", "SQL", "Apache Spark", "dbt", "Airflow",
                            "AWS", "Snowflake", "BigQuery", "Docker", "Kafka"],
    years_of_experience  = 3.0,
    job_titles           = ["Data Engineer", "Analytics Engineer"],
    highest_degree       = "Master's",
    field_of_study       = "Computer Science",
    best_fit_roles       = ["Data Engineer", "Analytics Engineer"],
    seniority_level      = "Mid",
    profile_summary      = (
        "Mid-level Data Engineer with 3 years of experience building "
        "scalable data pipelines using Spark, Airflow, and dbt. "
        "Strong background in cloud data warehousing on AWS and Snowflake."
    ),
    visa_status          = "F-1 OPT",
    requires_sponsorship = True,
    target_roles         = ["Data Engineer", "Analytics Engineer"],
    preferred_locations  = ["Remote", "San Francisco, CA"],
    open_to_relocation   = True,
)


# ── Pipeline state ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    resume_path:      Optional[str]   # None when profile is injected directly
    user_inputs:      UserInputs
    top_n:            int
    candidate_top_n:  int             # WS4 → WS5 hand-off size (default 50)
    refresh_data:     bool            # force re-run WS1 + WS2 + WS4 index build

    # ── Agent outputs ─────────────────────────────────────────────────────────
    profile:          Optional[CandidateProfile]        # ResumeParserAgent (WS3)
    matches:          Optional[list[JobMatch]]           # MatchingAgent (WS4)
    jd_signals:       Optional[dict]                    # SignalExtractionAgent
    company_evidence: Optional[dict]                    # HistoricalEvidenceAgent
    scored:           Optional[list[ScoredJob]]          # AuditorAgent

    # ── Control ───────────────────────────────────────────────────────────────
    error:  Optional[str]
    status: str


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT NODES
# ══════════════════════════════════════════════════════════════════════════════

# ── ScraperAgent (WS1) ────────────────────────────────────────────────────────

def scraper_agent(state: PipelineState) -> PipelineState:
    """
    WS1: Scrape Greenhouse job boards → bronze layer → silver layer.
    Skipped if jobs_all.parquet already exists and refresh_data is False.
    """
    if JOBS_SILVER.exists() and not state["refresh_data"]:
        log.info("[WS6:ScraperAgent] Silver jobs exist — skipping scrape")
        return {**state, "status": "scraper_skipped"}

    log.info("[WS6:ScraperAgent] Scraping %d companies...", len(COMPANIES))
    try:
        run_time = datetime.now(timezone.utc)
        records  = scrape_all(companies=COMPANIES, run_time=run_time)
        save_bronze(records, run_time=run_time)
        df_silver = build_silver()
        if not df_silver.empty:
            save_silver(df_silver)
        log.info("[WS6:ScraperAgent] Done — %d jobs in silver layer", len(df_silver))
        return {**state, "status": "scraped"}
    except Exception as exc:
        log.error("[WS6:ScraperAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── SponsorshipAgent (WS2) ────────────────────────────────────────────────────

def sponsorship_agent(state: PipelineState) -> PipelineState:
    """
    WS2: Load DOL PERM + USCIS data → sponsorship_history.parquet.
    Skipped if the parquet already exists and refresh_data is False.
    """
    if SPONSORSHIP_PARQUET.exists() and not state["refresh_data"]:
        log.info("[WS6:SponsorshipAgent] Sponsorship silver exists — skipping build")
        return {**state, "status": "sponsorship_skipped"}

    log.info("[WS6:SponsorshipAgent] Building sponsorship history...")
    try:
        perm_raw = load_all_perm()
        perm_agg = aggregate_perm(perm_raw)
        uscis    = load_uscis()
        combined = perm_agg.merge(uscis, on="company_name_norm", how="outer")

        for col in ("uscis_total_approvals", "uscis_total_denials"):
            combined[col] = combined[col].fillna(0).astype(int)
        combined = combined.sort_values("total_perm_certified", ascending=False).reset_index(drop=True)
        for col in ("naics", "state", "company_name_raw"):
            combined[col] = combined[col].astype(str)

        SPONSORSHIP_SILVER_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(SPONSORSHIP_PARQUET, index=False)
        combined.to_csv(SPONSORSHIP_CSV, index=False, encoding="utf-8-sig")
        log.info("[WS6:SponsorshipAgent] Done — %d companies", len(combined))
        return {**state, "status": "sponsorship_built"}
    except Exception as exc:
        log.error("[WS6:SponsorshipAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── IndexBuilderAgent (WS4 build) ─────────────────────────────────────────────

def index_builder_agent(state: PipelineState) -> PipelineState:
    """
    WS4 (build): Embed all target-role jobs → job_embeddings.npy + job_index.parquet.
    Skipped if index already exists and refresh_data is False.
    """
    if EMBEDDINGS_PATH.exists() and JOB_INDEX_PATH.exists() and not state["refresh_data"]:
        log.info("[WS6:IndexBuilderAgent] Job index exists — skipping build")
        return {**state, "status": "index_skipped"}

    if not JOBS_SILVER.exists():
        return {
            **state,
            "error": "jobs_all.parquet not found. ScraperAgent must run first.",
            "status": "failed",
        }

    log.info("[WS6:IndexBuilderAgent] Building job embedding index...")
    try:
        build_index(force=state["refresh_data"])
        log.info("[WS6:IndexBuilderAgent] Done")
        return {**state, "status": "index_built"}
    except Exception as exc:
        log.error("[WS6:IndexBuilderAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── ResumeParserAgent (WS3) ───────────────────────────────────────────────────

def resume_parser_agent(state: PipelineState) -> PipelineState:
    """
    WS3: PDF/DOCX + user inputs → CandidateProfile.
    Skipped when profile is already injected (run_pipeline_from_profile).
    """
    if state.get("profile") is not None:
        log.info("[WS6:ResumeParserAgent] Profile pre-loaded — skipping parse")
        return {**state, "status": "parsed"}

    if not state.get("resume_path"):
        return {**state, "error": "No resume path provided.", "status": "failed"}

    log.info("[WS6:ResumeParserAgent] Parsing resume: %s", state["resume_path"])
    try:
        profile = parse_resume(state["resume_path"], state["user_inputs"])
        log.info(
            "[WS6:ResumeParserAgent] Done — %s | %s | %.1f yrs",
            profile.name or "—",
            profile.seniority_level or "—",
            profile.years_of_experience or 0,
        )
        return {**state, "profile": profile, "status": "parsed"}
    except Exception as exc:
        log.error("[WS6:ResumeParserAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── MatchingAgent (WS4 query) ─────────────────────────────────────────────────

def matching_agent(state: PipelineState) -> PipelineState:
    """
    WS4 (query): Embed CandidateProfile → cosine similarity vs job index
    → weighted match score (semantic + skill overlap + role + location).
    """
    log.info("[WS6:MatchingAgent] Matching profile against job index (top %d)...",
             state["candidate_top_n"])
    try:
        matches = match(state["profile"], top_n=state["candidate_top_n"])
        log.info(
            "[WS6:MatchingAgent] Done — %d matches | best=%.4f (%s @ %s)",
            len(matches),
            matches[0].match_score if matches else 0,
            matches[0].title       if matches else "—",
            matches[0].company_slug if matches else "—",
        )
        return {**state, "matches": matches, "status": "matched"}
    except Exception as exc:
        log.error("[WS6:MatchingAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── SignalExtractionAgent ─────────────────────────────────────────────────────

def signal_extraction_agent(state: PipelineState) -> PipelineState:
    """
    Scans each matched job description for sponsorship phrases using the
    prioritised regex cascade from WS5.

    Output: jd_signals = {job_id: {"signal": str, "delta": float}}

    Skipped for domestic candidates (requires_sponsorship=False).
    """
    if not state["user_inputs"].requires_sponsorship:
        log.info("[WS6:SignalExtractionAgent] Domestic candidate — skipping JD scan")
        return {**state, "jd_signals": {}, "status": "signals_extracted"}

    log.info("[WS6:SignalExtractionAgent] Scanning %d job descriptions...",
             len(state["matches"]))
    try:
        signals: dict = {}
        for job in state["matches"]:
            sig, delta = _scan_jd(job.description_text)
            signals[job.job_id] = {"signal": sig, "delta": delta}

        counts: dict = {}
        for v in signals.values():
            counts[v["signal"]] = counts.get(v["signal"], 0) + 1
        log.info("[WS6:SignalExtractionAgent] Done — distribution: %s", counts)
        return {**state, "jd_signals": signals, "status": "signals_extracted"}
    except Exception as exc:
        log.error("[WS6:SignalExtractionAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── HistoricalEvidenceAgent ───────────────────────────────────────────────────

def historical_evidence_agent(state: PipelineState) -> PipelineState:
    """
    Queries the WS2 sponsorship silver layer for every unique company
    in the match list.

    Output: company_evidence = {
        slug: {base_score, perm_total, trend, freshness}
    }

    Skipped for domestic candidates.
    """
    if not state["user_inputs"].requires_sponsorship:
        log.info("[WS6:HistoricalEvidenceAgent] Domestic candidate — skipping PERM lookup")
        return {**state, "company_evidence": {}, "status": "evidence_gathered"}

    unique_slugs = {job.company_slug for job in state["matches"]}
    log.info("[WS6:HistoricalEvidenceAgent] Looking up %d companies in PERM/USCIS data...",
             len(unique_slugs))
    try:
        df     = _load_sponsorship_df()
        lookup = _build_lookup(df)

        evidence: dict = {}
        for slug in unique_slugs:
            row = _find_company(slug, lookup, df)
            if row is None:
                evidence[slug] = {
                    "base_score": 0.40,
                    "perm_total": 0,
                    "trend":      "unknown",
                    "freshness":  0.0,
                }
            else:
                evidence[slug] = {
                    "base_score": _perm_base_score(row),
                    "perm_total": int(row.get("total_perm_certified", 0) or 0),
                    "trend":      str(row.get("trend_direction", "insufficient_data")),
                    "freshness":  float(row.get("data_freshness_score", 0) or 0),
                }

        found = sum(1 for e in evidence.values() if e["perm_total"] > 0)
        log.info("[WS6:HistoricalEvidenceAgent] Done — %d/%d found in PERM data",
                 found, len(unique_slugs))
        return {**state, "company_evidence": evidence, "status": "evidence_gathered"}
    except Exception as exc:
        log.error("[WS6:HistoricalEvidenceAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── AuditorAgent ──────────────────────────────────────────────────────────────

def _detect_contradiction(signal: str, ev: dict) -> tuple[str, float]:
    """Cross-validate JD signal vs PERM history. Returns (note, confidence_adj)."""
    perm_total = ev["perm_total"]
    trend      = ev["trend"]

    if signal == "positive" and perm_total == 0:
        return (
            "Contradiction: JD offers sponsorship but company has no PERM history — verify independently.",
            -0.10,
        )
    if signal == "negative" and perm_total >= 100:
        return (
            "Contradiction: JD discourages sponsorship but company has strong PERM record — may be role-specific.",
            +0.05,
        )
    if signal in ("positive", "likely_positive") and trend == "decreasing_sharply":
        return (
            "Caution: JD implies sponsorship but PERM filings have dropped sharply.",
            -0.08,
        )
    if signal == "negative" and trend == "increasing":
        return (
            "Note: JD discourages sponsorship but PERM filings are increasing — check current policy.",
            0.0,
        )
    return "", 0.0


def _build_reasoning(
    profile: CandidateProfile,
    job: JobMatch,
    rec_score: float,
    days_old: int,
    signal: Optional[str],
    perm_total: Optional[int],
    trend: Optional[str],
    visa_conf: Optional[float],
    contradiction_note: str,
    needs_sponsor: bool,
) -> str:
    parts: list[str] = []

    fit = ("Strong match" if job.match_score >= 0.70 else
           "Good match"   if job.match_score >= 0.50 else
           "Moderate match" if job.match_score >= 0.35 else "Weak match")
    roles = ", ".join(profile.target_roles or profile.best_fit_roles or ["your target role"])
    parts.append(f"{fit} for {roles}.")

    n = len(job.matched_skills)
    if n > 0:
        top  = ", ".join(job.matched_skills[:5])
        tail = ", ..." if n > 5 else ""
        parts.append(f"{n}/{max(len(profile.technical_skills),1)} skills matched ({top}{tail}).")
    else:
        parts.append("No listed skills found directly in job description.")

    if job.role_match:
        parts.append("Job title aligns with target roles.")
    if job.location_match or "remote" in job.location.lower():
        parts.append("Remote or preferred location.")

    if days_old < 0:   parts.append("Posting date unknown.")
    elif days_old == 0: parts.append("Posted today.")
    elif days_old == 1: parts.append("Posted yesterday.")
    elif days_old <= 7: parts.append(f"Posted {days_old}d ago.")
    elif days_old <= 30: parts.append(f"Posted ~{days_old//7}w ago.")
    else:               parts.append(f"Posted {days_old}d ago — may already be filled.")

    if needs_sponsor and visa_conf is not None:
        jd_msgs = {
            "positive":        "JD explicitly states visa sponsorship is available.",
            "likely_positive": "JD suggests openness to international candidates.",
            "negative":        "JD explicitly states no visa sponsorship.",
            "likely_negative": "JD may require existing work authorization.",
            "unknown":         "JD does not mention visa sponsorship.",
        }
        parts.append(jd_msgs.get(signal or "unknown", ""))

        if perm_total == 0:       parts.append("No PERM history found.")
        elif perm_total < 5:      parts.append(f"Minimal PERM history ({perm_total} cases).")
        elif perm_total < 20:     parts.append(f"Limited PERM history ({perm_total} certifications).")
        elif perm_total < 100:    parts.append(f"{perm_total} PERM certifications on record.")
        else:                     parts.append(f"Strong PERM history ({perm_total}+ certifications).")

        trend_msgs = {
            "increasing":         "Sponsorship filings trending upward.",
            "stable":             "Sponsorship volume is stable.",
            "decreasing":         "Sponsorship filings have declined recently.",
            "decreasing_sharply": "Sponsorship filings dropped sharply — verify.",
            "insufficient_data":  "Insufficient data for trend.",
        }
        if trend in trend_msgs:
            parts.append(trend_msgs[trend])

        if contradiction_note:
            parts.append(contradiction_note)

    return " ".join(p for p in parts if p)


def auditor_agent(state: PipelineState) -> PipelineState:
    """
    Cross-validates JD signals (SignalExtractionAgent) against PERM evidence
    (HistoricalEvidenceAgent). Detects contradictions, assigns visa confidence,
    computes final weighted score, generates plain-English reasoning per job.

    International: final = 0.40×match + 0.35×visa + 0.25×recency
    Domestic:      final = 0.70×match + 0.30×recency
    """
    log.info("[WS6:AuditorAgent] Auditing and scoring %d jobs...", len(state["matches"]))
    try:
        profile          = state["profile"]
        jd_signals       = state.get("jd_signals") or {}
        company_evidence = state.get("company_evidence") or {}
        needs_sponsor    = state["user_inputs"].requires_sponsorship

        results: list[ScoredJob] = []
        for job in state["matches"]:
            rec_score, days_old = _recency_score(job.posted_at)

            if needs_sponsor:
                sig_data = jd_signals.get(job.job_id, {"signal": "unknown", "delta": 0.0})
                ev_data  = company_evidence.get(
                    job.company_slug,
                    {"base_score": 0.40, "perm_total": 0, "trend": "unknown", "freshness": 0.0}
                )
                signal     = sig_data["signal"]
                delta      = sig_data["delta"]
                perm_total = ev_data["perm_total"]
                trend      = ev_data["trend"]

                contradiction_note, contra_adj = _detect_contradiction(signal, ev_data)
                visa_conf = max(0.05, min(0.95, ev_data["base_score"] + delta + contra_adj))
                final     = W_INT_MATCH * job.match_score + W_INT_VISA * visa_conf + W_INT_RECENCY * rec_score
            else:
                signal = "n/a"; perm_total = 0; trend = "n/a"
                visa_conf = None; contradiction_note = ""
                final = W_DOM_MATCH * job.match_score + W_DOM_RECENCY * rec_score

            reasoning = _build_reasoning(
                profile, job, rec_score, days_old,
                signal, perm_total, trend, visa_conf,
                contradiction_note, needs_sponsor,
            )
            results.append(ScoredJob(
                **job.model_dump(),
                recency_score      = round(rec_score, 4),
                days_since_posted  = days_old,
                visa_confidence    = round(visa_conf, 4) if visa_conf is not None else None,
                sponsorship_signal = signal,
                perm_filings_total = perm_total,
                sponsorship_trend  = trend,
                final_score        = round(final, 4),
                reasoning          = reasoning,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        top = results[: state["top_n"]]
        if top:
            log.info("[WS6:AuditorAgent] Top: %.4f — %s @ %s",
                     top[0].final_score, top[0].title, top[0].company_slug)
        return {**state, "scored": top, "status": "scored"}
    except Exception as exc:
        log.error("[WS6:AuditorAgent] Failed: %s", exc)
        return {**state, "error": str(exc), "status": "failed"}


# ── Routing ────────────────────────────────────────────────────────────────────

def _ok(next_node: str):
    def _route(state: PipelineState) -> str:
        return "end" if state.get("error") else next_node
    return _route


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("scraper",            scraper_agent)
    g.add_node("sponsorship",        sponsorship_agent)
    g.add_node("index_builder",      index_builder_agent)
    g.add_node("resume_parser",      resume_parser_agent)
    g.add_node("matching",           matching_agent)
    g.add_node("signal_extraction",  signal_extraction_agent)
    g.add_node("historical_evidence",historical_evidence_agent)
    g.add_node("auditor",            auditor_agent)

    g.set_entry_point("scraper")

    g.add_conditional_edges("scraper",             _ok("sponsorship"),         {"sponsorship":         "sponsorship",         "end": END})
    g.add_conditional_edges("sponsorship",         _ok("index_builder"),       {"index_builder":       "index_builder",       "end": END})
    g.add_conditional_edges("index_builder",       _ok("resume_parser"),       {"resume_parser":       "resume_parser",       "end": END})
    g.add_conditional_edges("resume_parser",       _ok("matching"),            {"matching":            "matching",            "end": END})
    g.add_conditional_edges("matching",            _ok("signal_extraction"),   {"signal_extraction":   "signal_extraction",   "end": END})
    g.add_conditional_edges("signal_extraction",   _ok("historical_evidence"), {"historical_evidence": "historical_evidence", "end": END})
    g.add_conditional_edges("historical_evidence", _ok("auditor"),             {"auditor":             "auditor",             "end": END})
    g.add_edge("auditor", END)

    return g


# ── Shared invoke ──────────────────────────────────────────────────────────────

def _invoke(initial_state: PipelineState) -> list[ScoredJob]:
    graph = build_graph().compile()
    log.info("=== WS6: LangGraph Master Pipeline ===")
    final_state: PipelineState = graph.invoke(initial_state)
    log.info("=== WS6 complete — status: %s ===", final_state["status"])
    if final_state.get("error"):
        raise RuntimeError(f"Pipeline failed: {final_state['error']}")
    return final_state["scored"] or []


def _base_state(top_n: int, candidate_top_n: int, refresh: bool) -> dict:
    return {
        "top_n":            top_n,
        "candidate_top_n":  candidate_top_n,
        "refresh_data":     refresh,
        "matches":          None,
        "jd_signals":       None,
        "company_evidence": None,
        "scored":           None,
        "error":            None,
        "status":           "pending",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_pipeline(
    resume_path:     str | Path,
    user_inputs:     UserInputs,
    top_n:           int  = 30,
    candidate_top_n: int  = 150,
    refresh:         bool = False,
) -> list[ScoredJob]:
    """
    Full pipeline: WS1 → WS2 → WS4-build → WS3 → WS4-match → WS5.
    Called by WS7 (FastAPI) with the uploaded file path.
    Set refresh=True to force re-scrape and re-index.
    """
    return _invoke({
        **_base_state(top_n, candidate_top_n, refresh),
        "resume_path": str(resume_path),
        "user_inputs": user_inputs,
        "profile":     None,
    })


def run_pipeline_from_profile(
    profile:         CandidateProfile,
    user_inputs:     UserInputs,
    top_n:           int  = 30,
    candidate_top_n: int  = 150,
    refresh:         bool = False,
) -> list[ScoredJob]:
    """
    Profile-first pipeline: skip WS3, inject CandidateProfile → WS4 → WS5.
    Used for testing without a resume PDF.
    """
    return _invoke({
        **_base_state(top_n, candidate_top_n, refresh),
        "resume_path": None,
        "user_inputs": user_inputs,
        "profile":     profile,
    })


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_results(scored: list[ScoredJob], top_n: int, needs_sponsor: bool) -> None:
    label = ("INTERNATIONAL (with sponsorship)" if needs_sponsor
             else "DOMESTIC (no sponsorship needed)")
    print(f"\n{'='*70}")
    print(f"  TOP {top_n} JOBS  —  {label}")
    print(f"{'='*70}\n")
    for i, s in enumerate(scored[:top_n], 1):
        visa_str = (f"  visa={s.visa_confidence:.2f}  signal={s.sponsorship_signal}"
                    if s.visa_confidence is not None else "")
        print(f"{i:2}. {s.title}")
        print(f"    Company  : {s.company_slug}")
        print(f"    Location : {s.location}")
        print(f"    Posted   : {s.posted_at}  ({s.days_since_posted}d ago)")
        print(f"    Score    : {s.final_score:.3f}  "
              f"[match={s.match_score:.3f}  recency={s.recency_score:.2f}" + visa_str + "]")
        print(f"    Why      : {s.reasoning}")
        print(f"    Apply    : {s.apply_url}")
        print()
    print(f"{'='*70}")
    print(f"  Showing {min(top_n, len(scored))} of {len(scored)} scored jobs.")
    print(f"{'='*70}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="WS6: Master LangGraph pipeline (WS1→WS2→WS3→WS4→WS5)"
    )
    parser.add_argument("--resume",     default=None,
                        help="Path to resume PDF or DOCX. Omit to use --demo.")
    parser.add_argument("--demo",       action="store_true",
                        help="Run with built-in demo profile (no resume file needed).")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-scrape (WS1), rebuild sponsorship (WS2), and rebuild index (WS4).")
    parser.add_argument("--visa",       default="F-1 OPT")
    parser.add_argument("--no-sponsor", action="store_true")
    parser.add_argument("--roles",      default="Data Engineer,Analytics Engineer")
    parser.add_argument("--locations",  default="Remote")
    parser.add_argument("--top",        type=int, default=15)
    args = parser.parse_args()

    # Default to demo mode when run with no arguments (e.g. VS Code play button)
    use_demo = args.demo or not args.resume

    inputs = UserInputs(
        visa_status          = args.visa,
        requires_sponsorship = not args.no_sponsor,
        target_roles         = [r.strip() for r in args.roles.split(",")],
        preferred_locations  = [l.strip() for l in args.locations.split(",")],
        open_to_relocation   = True,
    )

    if use_demo:
        log.info("DEMO mode — using built-in sample profile")
        scored = run_pipeline_from_profile(DEMO_PROFILE, inputs,
                                           top_n=args.top, refresh=args.refresh)
    else:
        scored = run_pipeline(args.resume, inputs,
                              top_n=args.top, refresh=args.refresh)

    _print_results(scored, args.top, inputs.requires_sponsorship)


if __name__ == "__main__":
    main()
