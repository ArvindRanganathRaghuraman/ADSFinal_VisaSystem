"""
pipeline/ws5_confidence_scorer.py
-----------------------------------
WS5: Final confidence scoring.

Combines job-match quality (WS4), posting recency, and — for international
candidates — visa sponsorship confidence into a single ranked list with
plain-English reasoning for each result.

Scoring weights
---------------
International students (requires_sponsorship=True):
    final_score = 0.40 × match_score + 0.35 × visa_confidence + 0.25 × recency_score

Domestic / no sponsorship needed:
    final_score = 0.70 × match_score + 0.30 × recency_score

Recency bands
-------------
  0–3 days  → 1.00
  4–7 days  → 0.85
  8–14 days → 0.70
  15–30 days → 0.50
  31–60 days → 0.30
  60+ days  → 0.10

Visa confidence components
--------------------------
  PERM history — total certified filings, trend direction, data freshness
  JD scan      — regex phrases signalling intent to sponsor OR refusal

Usage
-----
  from pipeline.ws4_job_matcher import match
  from pipeline.ws5_confidence_scorer import score

  matches = match(profile, top_n=50)
  scored  = score(matches, profile)
  for s in scored:
      print(f"{s.final_score:.3f}  {s.title} @ {s.company_slug}")
      print(s.reasoning)
"""

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.candidate_profile import CandidateProfile
from models.job_match import JobMatch
from models.scored_job import ScoredJob

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SPONSORSHIP_PARQUET = _ROOT / "Data" / "silver" / "sponsorship_history.parquet"

# ── Final score weights ────────────────────────────────────────────────────────
W_INT_MATCH   = 0.40   # international: job fit
W_INT_VISA    = 0.35   # international: visa confidence
W_INT_RECENCY = 0.25   # international: posting recency

W_DOM_MATCH   = 0.70   # domestic: job fit
W_DOM_RECENCY = 0.30   # domestic: posting recency


# ── JD sponsorship phrase lists ────────────────────────────────────────────────
# Checked in priority order: pos_strong → neg_strong → pos_weak → neg_weak

_JD_POSITIVE_STRONG = [
    r"will\s+sponsor",
    r"visa\s+sponsorship\s+(is\s+|will\s+be\s+|)provided",
    r"sponsorship\s+(is\s+|will\s+be\s+)?available",
    r"h[\-\s]?1[\-\s]?b\s+sponsor",
    r"able\s+to\s+sponsor",
    r"provide[sd]?\s+sponsorship",
    r"sponsor\s+work\s+(visa|authorization|permit)",
    r"open\s+to\s+(visa\s+)?sponsoring",
    r"we\s+(do\s+|will\s+)?sponsor",
    r"sponsorship\s+for\s+(this\s+)?role",
    r"sponsor\s+h[\-\s]?1[\-\s]?b",
    r"support\s+(visa|h[\-\s]?1[\-\s]?b)\s+transfer",
]
_JD_POSITIVE_WEAK = [
    r"open\s+to\s+sponsorship",
    r"sponsorship\s+considered",
    r"international\s+candidates\s+welcome",
    r"relocation\s+(and\s+|or\s+)?visa",
]
_JD_NEGATIVE_STRONG = [
    r"no\s+(visa\s+)?sponsorship",
    r"not\s+(able|eligible)\s+to\s+sponsor",
    r"cannot\s+sponsor",
    r"unable\s+to\s+sponsor",
    r"sponsorship\s+(is\s+)?not\s+available",
    r"does\s+not\s+(offer\s+|provide\s+|)sponsor",
    r"without\s+(the\s+need\s+(of|for)\s+)?sponsorship",
    r"must\s+be\s+(a\s+)?u\.?s\.?\s+citizen",
    r"u\.?s\.?\s+citizens?\s+only",
    r"must\s+be\s+(legally\s+)?authorized\s+to\s+work\s+in\s+the\s+u\.?s\.?",
    r"no\s+work\s+visa",
    r"(not\s+)?require.*sponsorship\s+to\s+work",
]
_JD_NEGATIVE_WEAK = [
    r"must\s+be\s+authorized\s+to\s+work",
    r"authorized\s+to\s+work\s+in\s+the\s+(us|united\s+states)",
    r"work\s+authorization\s+required",
    r"eligible\s+to\s+work\s+in\s+the\s+u\.?s\.?\s+without",
]


def _compile(patterns: list[str]):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_POS_STRONG_RE = _compile(_JD_POSITIVE_STRONG)
_POS_WEAK_RE   = _compile(_JD_POSITIVE_WEAK)
_NEG_STRONG_RE = _compile(_JD_NEGATIVE_STRONG)
_NEG_WEAK_RE   = _compile(_JD_NEGATIVE_WEAK)


# ── Recency scoring ────────────────────────────────────────────────────────────

def _recency_score(posted_at: str) -> tuple[float, int]:
    """Parse posted_at string, return (score 0-1, days_since_posted)."""
    try:
        dt = pd.to_datetime(posted_at, utc=True)
        now = datetime.now(timezone.utc)
        days = max(0, (now - dt.to_pydatetime()).days)
    except Exception:
        return 0.50, -1   # unknown date → neutral score

    if days <= 3:
        return 1.00, days
    if days <= 7:
        return 0.85, days
    if days <= 14:
        return 0.70, days
    if days <= 30:
        return 0.50, days
    if days <= 60:
        return 0.30, days
    return 0.10, days


# ── JD sponsorship scan ────────────────────────────────────────────────────────

def _scan_jd(text: str) -> tuple[str, float]:
    """
    Scan job description for sponsorship signals.
    Positive-strong patterns take priority over negative-strong to handle JDs
    that say both "must be authorised" generically but also "will sponsor H-1B".
    Returns (signal_label, score_delta).
    """
    t = text.lower()
    if any(r.search(t) for r in _POS_STRONG_RE):
        return "positive", +0.30
    if any(r.search(t) for r in _NEG_STRONG_RE):
        return "negative", -0.40
    if any(r.search(t) for r in _POS_WEAK_RE):
        return "likely_positive", +0.15
    if any(r.search(t) for r in _NEG_WEAK_RE):
        return "likely_negative", -0.20
    return "unknown", 0.0


# ── PERM / USCIS lookup ────────────────────────────────────────────────────────

def _load_sponsorship_df() -> pd.DataFrame:
    if not SPONSORSHIP_PARQUET.exists():
        raise FileNotFoundError(
            f"Sponsorship silver not found at {SPONSORSHIP_PARQUET}. Run WS2 first."
        )
    return pd.read_parquet(SPONSORSHIP_PARQUET)


def _build_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    """O(1) exact lookup by normalised company name."""
    return {str(row["company_name_norm"]): row for _, row in df.iterrows()}


def _find_company(slug: str, lookup: dict, df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Match company_slug (from job index) to a row in the sponsorship table.
    Strategy:
      1. Exact match (company_name_norm == slug)
      2. One is a prefix of the other
      3. Slug appears as a substring (only if slug len > 4 to avoid false matches)
    """
    if slug in lookup:
        return lookup[slug]
    for norm, row in lookup.items():
        if norm.startswith(slug) or slug.startswith(norm):
            return row
    if len(slug) > 4:
        for norm, row in lookup.items():
            if slug in norm:
                return row
    return None


def _perm_base_score(row: pd.Series) -> float:
    """Convert PERM/USCIS filing history into a 0-1 base score."""
    total  = int(row.get("total_perm_certified", 0) or 0)
    uscis  = int(row.get("uscis_total_approvals", 0) or 0)
    trend  = str(row.get("trend_direction", "insufficient_data"))
    fresh  = float(row.get("data_freshness_score", 0) or 0)

    # Volume-based base score (use max of PERM certified or USCIS approvals)
    combined = max(total, uscis)
    if combined == 0:
        base = 0.30
    elif combined >= 200:
        base = 0.85
    elif combined >= 100:
        base = 0.78
    elif combined >= 50:
        base = 0.68
    elif combined >= 20:
        base = 0.56
    elif combined >= 5:
        base = 0.44
    else:
        base = 0.35

    # Trend adjustment
    trend_adj = {
        "increasing":          +0.10,
        "stable":               0.00,
        "decreasing":          -0.05,
        "decreasing_sharply":  -0.15,
        "insufficient_data":    0.00,
    }
    base += trend_adj.get(trend, 0.0)

    # Freshness discount: stale data is less reliable
    # freshness 1.0 → full score; freshness 0.1 → 55% of score
    base = base * (0.45 + 0.55 * fresh)

    return max(0.05, min(0.95, base))


def _visa_confidence(
    company_slug: str,
    description: str,
    lookup: dict,
    df: pd.DataFrame,
) -> tuple[float, str, int, str]:
    """
    Returns (visa_conf, jd_signal, perm_total, trend).
    """
    jd_signal, jd_delta = _scan_jd(description)

    comp_row = _find_company(company_slug, lookup, df)
    if comp_row is None:
        base       = 0.40   # unknown company → cautiously neutral
        perm_total = 0
        trend      = "unknown"
    else:
        base       = _perm_base_score(comp_row)
        perm_total = int(comp_row.get("total_perm_certified", 0) or 0)
        trend      = str(comp_row.get("trend_direction", "insufficient_data"))

    visa_conf = max(0.05, min(0.95, base + jd_delta))
    return visa_conf, jd_signal, perm_total, trend


# ── Reasoning builder ──────────────────────────────────────────────────────────

def _build_reasoning(
    profile: CandidateProfile,
    job: JobMatch,
    recency_score: float,
    days_old: int,
    visa_conf: Optional[float],
    jd_signal: Optional[str],
    perm_total: Optional[int],
    trend: Optional[str],
) -> str:
    parts = []

    # ── Job fit ──────────────────────────────────────────────────────────────
    if job.match_score >= 0.70:
        fit_label = "Strong match"
    elif job.match_score >= 0.50:
        fit_label = "Good match"
    elif job.match_score >= 0.35:
        fit_label = "Moderate match"
    else:
        fit_label = "Weak match"

    roles = ", ".join(profile.target_roles or profile.best_fit_roles or ["your target role"])
    parts.append(f"{fit_label} for {roles}.")

    n_matched = len(job.matched_skills)
    n_total   = max(len(profile.technical_skills), 1)
    if n_matched > 0:
        top_skills = ", ".join(job.matched_skills[:5])
        tail = ", ..." if n_matched > 5 else ""
        parts.append(
            f"{n_matched} of your {n_total} skills matched in the job description "
            f"({top_skills}{tail})."
        )
    else:
        parts.append("None of your listed skills appeared directly in the job description.")

    if job.role_match:
        parts.append("Job title aligns with your target roles.")

    if job.location_match:
        parts.append("Location matches your preferences.")
    elif "remote" in job.location.lower():
        parts.append("Remote position.")

    # ── Recency ──────────────────────────────────────────────────────────────
    if days_old < 0:
        parts.append("Posting date unknown.")
    elif days_old == 0:
        parts.append("Posted today.")
    elif days_old == 1:
        parts.append("Posted yesterday.")
    elif days_old <= 7:
        parts.append(f"Posted {days_old} days ago.")
    elif days_old <= 30:
        weeks = days_old // 7
        parts.append(f"Posted about {weeks} week{'s' if weeks > 1 else ''} ago.")
    else:
        parts.append(f"Posted {days_old} days ago — may be an older listing.")

    # ── Visa / sponsorship ────────────────────────────────────────────────────
    if visa_conf is not None:
        # JD signal
        jd_msgs = {
            "positive":        "Job description explicitly mentions visa sponsorship is available.",
            "likely_positive": "Job description suggests openness to international candidates.",
            "negative":        "Job description explicitly states no visa sponsorship.",
            "likely_negative": "Job description may require existing work authorization.",
            "unknown":         "Job description does not mention visa sponsorship.",
        }
        parts.append(jd_msgs.get(jd_signal or "unknown", ""))

        # PERM history
        if perm_total == 0:
            parts.append("No PERM sponsorship history found for this company.")
        elif perm_total < 5:
            parts.append(f"Company has minimal PERM history ({perm_total} certified case{'s' if perm_total != 1 else ''}).")
        elif perm_total < 20:
            parts.append(f"Company has filed {perm_total} PERM certifications (limited history).")
        elif perm_total < 100:
            parts.append(f"Company has {perm_total} PERM certifications on record.")
        else:
            parts.append(f"Company has a strong PERM history ({perm_total}+ certifications).")

        # Trend
        trend_msgs = {
            "increasing":          "Sponsorship filings are trending upward.",
            "stable":              "Sponsorship filing volume is stable.",
            "decreasing":          "Sponsorship filings have declined recently.",
            "decreasing_sharply":  "Sponsorship filings have dropped sharply — proceed with caution.",
            "insufficient_data":   "Insufficient data to determine sponsorship trend.",
            "unknown":             "",
        }
        msg = trend_msgs.get(trend or "unknown", "")
        if msg:
            parts.append(msg)

    return " ".join(p for p in parts if p)


# ── Main public function ───────────────────────────────────────────────────────

def score(
    matches: list[JobMatch],
    profile: CandidateProfile,
) -> list[ScoredJob]:
    """
    Takes WS4 JobMatch results + CandidateProfile.
    Returns ScoredJob list sorted by final_score descending.

    For international candidates (requires_sponsorship=True) visa confidence
    is computed and contributes 35% of the final score.
    For domestic candidates, final score = 70% match + 30% recency.
    """
    needs_sponsorship = profile.requires_sponsorship
    log.info(
        "WS5 scoring %d jobs (requires_sponsorship=%s)",
        len(matches), needs_sponsorship,
    )

    sponsorship_df = _load_sponsorship_df()
    lookup         = _build_lookup(sponsorship_df)

    results: list[ScoredJob] = []

    for job in matches:
        # Recency
        rec_score, days_old = _recency_score(job.posted_at)

        # Visa confidence
        if needs_sponsorship:
            visa_conf, jd_signal, perm_total, trend = _visa_confidence(
                job.company_slug, job.description_text, lookup, sponsorship_df
            )
        else:
            visa_conf, jd_signal, perm_total, trend = None, None, None, None

        # Final score
        if needs_sponsorship:
            final = (
                W_INT_MATCH   * job.match_score
                + W_INT_VISA  * visa_conf
                + W_INT_RECENCY * rec_score
            )
        else:
            final = (
                W_DOM_MATCH   * job.match_score
                + W_DOM_RECENCY * rec_score
            )

        reasoning = _build_reasoning(
            profile, job, rec_score, days_old,
            visa_conf, jd_signal, perm_total, trend,
        )

        results.append(ScoredJob(
            **job.model_dump(),
            recency_score      = round(rec_score, 4),
            days_since_posted  = days_old,
            visa_confidence    = round(visa_conf, 4) if visa_conf is not None else None,
            sponsorship_signal = jd_signal or "n/a",
            perm_filings_total = perm_total or 0,
            sponsorship_trend  = trend or "n/a",
            final_score        = round(final, 4),
            reasoning          = reasoning,
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    if results:
        log.info(
            "Top result: %.4f — %s @ %s",
            results[0].final_score, results[0].title, results[0].company_slug,
        )
    return results
