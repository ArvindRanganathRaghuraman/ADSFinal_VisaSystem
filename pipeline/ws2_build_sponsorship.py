"""
pipeline/ws2_build_sponsorship.py
----------------------------------
WS2: Build the sponsorship history silver layer.

Sources:
  Data/PERM/PERM_Disclosure_Data_FY20XX*.xlsx  — DOL PERM case-level records
  Data/USCIS/Employer Information.csv           — USCIS H-1B petition aggregates

Output:
  Data/silver/sponsorship_history.parquet   — one row per normalized company name
  Data/silver/sponsorship_history.csv

The silver table is keyed on company_name_norm (normalized/lowercased company
name with legal suffixes stripped).  Downstream confidence scoring queries this
table by normalized name to get per-company sponsorship signals.

Usage:
  python -m pipeline.ws2_build_sponsorship
"""

import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PERM_DIR        = _ROOT / "Data" / "PERM"
USCIS_PATH      = _ROOT / "Data" / "USCIS" / "Employer Information.csv"
SILVER_DIR      = _ROOT / "Data" / "silver"
SILVER_PARQUET  = SILVER_DIR / "sponsorship_history.parquet"
SILVER_CSV      = SILVER_DIR / "sponsorship_history.csv"

# How far back counts as "recent" (fiscal years)
RECENT_YEARS = 2

# Columns we want from the two PERM schemas
# Old schema: FY2020 – FY2024
_OLD_PERM_COLS = {
    "company_name": "EMPLOYER_NAME",
    "state":        "EMPLOYER_STATE_PROVINCE",
    "naics":        "NAICS_CODE",
    "fein":         None,   # not present in old schema
}
# New schema: FY2025+
_NEW_PERM_COLS = {
    "company_name": "EMP_BUSINESS_NAME",
    "state":        "EMP_STATE",
    "naics":        "EMP_NAICS",
    "fein":         "EMP_FEIN",
}
_PERM_COMMON   = ["CASE_STATUS", "DECISION_DATE", "JOB_TITLE"]

# CASE_STATUS values
_APPROVED_STATUSES = {"Certified", "Certified-Expired"}
_DENIED_STATUSES   = {"Denied"}

# USCIS approval/denial column pairs
_USCIS_APPROVAL_COLS = [
    "New Employment Approval",
    "Continuation Approval",
    "Change with Same Employer Approval",
    "New Concurrent Approval",
    "Change of Employer Approval",
    "Amended Approval",
]
_USCIS_DENIAL_COLS = [
    "New Employment Denial",
    "Continuation Denial",
    "Change with Same Employer Denial",
    "New Concurrent Denial",
    "Change of Employer Denial",
    "Amended Denial",
]


# ── Company name normalization ─────────────────────────────────────────────────

_LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"inc|incorporated|llc|corp|corporation|ltd|limited|lp|llp|plc"
    r"|co|company|holdings|group|services|technologies|technology"
    r"|solutions|systems|international|global|us|usa"
    r")\.?\b",
    re.IGNORECASE,
)


def normalize_company(name) -> str:
    """Lowercase, strip punctuation and common legal suffixes."""
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)          # replace punctuation with space
    s = _LEGAL_SUFFIX_RE.sub(" ", s)         # strip legal suffixes
    s = re.sub(r"\s+", " ", s).strip()       # collapse whitespace
    return s


# ── DOL fiscal year helper ─────────────────────────────────────────────────────

def to_fiscal_year(dt) -> int | None:
    """
    Convert a date to DOL fiscal year (Oct 1 – Sep 30).
    FY2025 = Oct 1 2024 – Sep 30 2025.
    """
    if pd.isna(dt):
        return None
    try:
        month = dt.month
        year  = dt.year
        return year + 1 if month >= 10 else year
    except Exception:
        return None


# ── PERM loader ────────────────────────────────────────────────────────────────

def _load_perm_file(path: Path) -> pd.DataFrame:
    """
    Load one PERM Excel file, detect schema, return a normalised DataFrame
    with columns: company_name_raw, state, naics, fein, case_status,
                  decision_date, job_title, fiscal_year, is_approved, is_denied
    """
    log.info("Loading PERM: %s", path.name)

    # Peek at header to detect schema
    header_df = pd.read_excel(path, nrows=0)
    is_new_schema = "EMP_BUSINESS_NAME" in header_df.columns
    mapping = _NEW_PERM_COLS if is_new_schema else _OLD_PERM_COLS

    # Build the usecols list (only the columns we actually need)
    usecols = list(_PERM_COMMON)
    for canonical, raw_col in mapping.items():
        if raw_col is not None:
            usecols.append(raw_col)

    df = pd.read_excel(path, usecols=usecols)

    # Rename to canonical names
    rename = {v: k for k, v in mapping.items() if v is not None}
    rename.update({c: c.lower() for c in _PERM_COMMON})
    df = df.rename(columns=rename)

    # Fill missing canonical columns
    for col in ("company_name", "state", "naics", "fein"):
        if col not in df.columns:
            df[col] = None

    # Parse decision date
    df["decision_date"] = pd.to_datetime(df.get("decision_date"), errors="coerce")

    # Fiscal year
    df["fiscal_year"] = df["decision_date"].apply(to_fiscal_year)

    # Sponsorship outcome flags
    df["is_approved"] = df["case_status"].isin(_APPROVED_STATUSES)
    df["is_denied"]   = df["case_status"].isin(_DENIED_STATUSES)

    # Normalise company name
    df["company_name_norm"] = df["company_name"].apply(normalize_company)
    df["company_name_raw"]  = df["company_name"].astype(str)

    keep = [
        "company_name_norm", "company_name_raw", "state", "naics", "fein",
        "case_status", "decision_date", "job_title",
        "fiscal_year", "is_approved", "is_denied",
    ]
    return df[keep].dropna(subset=["company_name_norm"]).query("company_name_norm != ''")


def load_all_perm() -> pd.DataFrame:
    """Load and concatenate all PERM Excel files."""
    files = sorted(PERM_DIR.glob("PERM_Disclosure_Data_FY*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No PERM Excel files found in {PERM_DIR}")

    parts = [_load_perm_file(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    log.info("PERM combined: %d records from %d files", len(df), len(files))
    return df


# ── USCIS loader ───────────────────────────────────────────────────────────────

def load_uscis() -> pd.DataFrame:
    """
    Load USCIS employer petition data and return per-company aggregates.
    Columns: company_name_norm, uscis_total_approvals, uscis_total_denials,
             uscis_last_fiscal_year
    """
    log.info("Loading USCIS: %s", USCIS_PATH.name)
    df = pd.read_csv(USCIS_PATH, encoding="utf-16", sep="\t", low_memory=False)

    # Strip whitespace from column names (some have trailing spaces)
    df.columns = df.columns.str.strip()

    df["company_name_norm"] = df["Employer (Petitioner) Name"].apply(normalize_company)
    df = df[df["company_name_norm"] != ""]

    # Parse fiscal year
    df["uscis_fiscal_year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")

    # Approval / denial totals (fill NaN with 0)
    approval_cols = [c for c in _USCIS_APPROVAL_COLS if c in df.columns]
    denial_cols   = [c for c in _USCIS_DENIAL_COLS   if c in df.columns]
    df["total_approvals"] = df[approval_cols].fillna(0).sum(axis=1)
    df["total_denials"]   = df[denial_cols].fillna(0).sum(axis=1)

    # Aggregate per company (sum across all fiscal years in this dataset)
    agg = (
        df.groupby("company_name_norm")
        .agg(
            uscis_total_approvals=("total_approvals", "sum"),
            uscis_total_denials=("total_denials", "sum"),
            uscis_last_fiscal_year=("uscis_fiscal_year", "max"),
        )
        .reset_index()
    )
    log.info("USCIS: %d unique companies after aggregation", len(agg))
    return agg


# ── PERM aggregation ───────────────────────────────────────────────────────────

def aggregate_perm(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw PERM records build a company-level summary table.
    """
    current_fy = _current_fiscal_year()
    recent_cutoff = current_fy - RECENT_YEARS  # FYs strictly after this are "recent"

    # Per company per year counts
    yearly = (
        df.groupby(["company_name_norm", "fiscal_year"])
        .agg(
            certified=("is_approved", "sum"),
            denied=("is_denied", "sum"),
        )
        .reset_index()
    )

    # Build per-year dict stored as JSON string
    by_year = (
        yearly.groupby("company_name_norm")[["fiscal_year", "certified", "denied"]]
        .apply(
            lambda g: json.dumps(
                {
                    str(int(row["fiscal_year"])): {
                        "certified": int(row["certified"]),
                        "denied": int(row["denied"]),
                    }
                    for _, row in g.iterrows()
                    if row["fiscal_year"] is not None and not np.isnan(row["fiscal_year"])
                }
            ),
            include_groups=False,
        )
        .rename("perm_by_year")
        .reset_index()
    )

    # Totals
    totals = (
        df.groupby("company_name_norm")
        .agg(
            total_perm_certified=("is_approved", "sum"),
            total_perm_denied=("is_denied", "sum"),
            last_perm_date=("decision_date", "max"),
            # Most common raw name
            company_name_raw=("company_name_raw", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
            # Most common state
            state=("state", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
            # Most common NAICS
            naics=("naics", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
        )
        .reset_index()
    )

    # Recent counts (last RECENT_YEARS fiscal years)
    recent = (
        df[df["fiscal_year"] > recent_cutoff]
        .groupby("company_name_norm")
        .agg(
            recent_perm_certified=("is_approved", "sum"),
            recent_perm_denied=("is_denied", "sum"),
        )
        .reset_index()
    )

    # Merge
    result = totals.merge(by_year, on="company_name_norm", how="left")
    result = result.merge(recent, on="company_name_norm", how="left")
    result["recent_perm_certified"] = result["recent_perm_certified"].fillna(0).astype(int)
    result["recent_perm_denied"]    = result["recent_perm_denied"].fillna(0).astype(int)

    # Approval rates
    total = result["total_perm_certified"] + result["total_perm_denied"]
    result["perm_approval_rate"] = np.where(
        total > 0, result["total_perm_certified"] / total, np.nan
    )

    recent_total = result["recent_perm_certified"] + result["recent_perm_denied"]
    result["recent_approval_rate"] = np.where(
        recent_total > 0, result["recent_perm_certified"] / recent_total, np.nan
    )

    # Trend direction
    result["trend_direction"] = result.apply(
        lambda r: _compute_trend(r["perm_by_year"], current_fy, RECENT_YEARS), axis=1
    )

    # Data freshness score (0–1 based on how recent the last filing is)
    result["data_freshness_score"] = result["last_perm_date"].apply(
        lambda d: _freshness_score(d, current_fy)
    )

    return result


def _current_fiscal_year() -> int:
    now = datetime.now(timezone.utc)
    return now.year + 1 if now.month >= 10 else now.year


def _compute_trend(by_year_json: str, current_fy: int, recent_years: int) -> str:
    """
    Derive trend from a JSON string of {year: {certified, denied}}.
    Returns one of: increasing | decreasing | stable | insufficient_data
    """
    try:
        by_year = json.loads(by_year_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return "insufficient_data"

    if not by_year:
        return "insufficient_data"

    # Build sorted (year, certified) pairs
    pairs = sorted(
        [(int(y), v.get("certified", 0)) for y, v in by_year.items()],
        key=lambda x: x[0],
    )
    if len(pairs) < 3:
        return "insufficient_data"

    cutoff = current_fy - recent_years
    recent_vals = [c for y, c in pairs if y > cutoff]
    older_vals  = [c for y, c in pairs if y <= cutoff]

    if not older_vals:
        return "insufficient_data"

    recent_avg  = sum(recent_vals) / len(recent_vals) if recent_vals else 0.0
    older_avg   = sum(older_vals)  / len(older_vals)

    if older_avg == 0:
        return "increasing" if recent_avg > 0 else "insufficient_data"

    ratio = recent_avg / older_avg
    if ratio >= 1.2:
        return "increasing"
    if ratio <= 0.4:
        return "decreasing_sharply"
    if ratio <= 0.75:
        return "decreasing"
    return "stable"


def _freshness_score(last_date, current_fy: int) -> float:
    """Score 0–1 based on recency of last PERM filing."""
    if pd.isna(last_date):
        return 0.0
    last_fy = to_fiscal_year(last_date)
    if last_fy is None:
        return 0.0
    age = current_fy - last_fy
    if age <= 1:
        return 1.0
    if age <= 2:
        return 0.75
    if age <= 3:
        return 0.5
    if age <= 4:
        return 0.25
    return 0.1


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("=== WS2: Build Sponsorship Silver Layer ===")

    # Load & aggregate PERM
    perm_raw = load_all_perm()
    perm_agg = aggregate_perm(perm_raw)
    log.info("PERM aggregated: %d unique companies", len(perm_agg))

    # Load USCIS
    uscis_agg = load_uscis()

    # Merge PERM + USCIS on normalised name
    combined = perm_agg.merge(uscis_agg, on="company_name_norm", how="outer")

    # Fill missing USCIS columns for PERM-only companies
    for col in ("uscis_total_approvals", "uscis_total_denials"):
        combined[col] = combined[col].fillna(0).astype(int)

    # Final sort — highest total certified first
    combined = combined.sort_values("total_perm_certified", ascending=False).reset_index(drop=True)

    # Ensure mixed-type columns are strings (parquet requires homogeneous types)
    for col in ("naics", "state", "company_name_raw"):
        combined[col] = combined[col].astype(str)

    # Save
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SILVER_PARQUET, index=False)
    log.info("Silver parquet → %s  (%d rows)", SILVER_PARQUET, len(combined))

    combined.to_csv(SILVER_CSV, index=False, encoding="utf-8-sig")
    log.info("Silver CSV    → %s", SILVER_CSV)

    # Quick sanity print for the companies in our scraper
    _spot_check(combined)

    log.info("=== WS2 complete ===")


def _spot_check(df: pd.DataFrame) -> None:
    """Print sponsorship summary for the 15 scraped companies."""
    from Scrapers.config import COMPANY_DISPLAY_NAMES
    log.info("\n── Spot check: scraped companies ──")
    for slug, display in COMPANY_DISPLAY_NAMES.items():
        norm = normalize_company(display)
        match = df[df["company_name_norm"].str.contains(norm, na=False, regex=False)]
        if match.empty:
            log.info("  %-15s → NOT FOUND in PERM/USCIS data", display)
        else:
            row = match.iloc[0]
            log.info(
                "  %-15s → certified=%d  trend=%-20s  freshness=%.2f",
                display,
                int(row.get("total_perm_certified", 0) or 0),
                str(row.get("trend_direction", "?")),
                float(row.get("data_freshness_score", 0) or 0),
            )


if __name__ == "__main__":
    run()
