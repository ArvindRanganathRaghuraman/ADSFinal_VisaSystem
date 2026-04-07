"""
models/scored_job.py
---------------------
Final output schema from WS5 (confidence scoring).
Extends JobMatch with recency, visa confidence, and plain-English reasoning.
Passed to WS6 orchestration and the WS7/WS8 frontend.
"""

from typing import Optional
from models.job_match import JobMatch


class ScoredJob(JobMatch):
    # ── Recency ───────────────────────────────────────────────────────────────
    recency_score:     float   # 0-1, higher = more recent posting
    days_since_posted: int     # -1 if date could not be parsed

    # ── Visa confidence (None when sponsorship not required) ──────────────────
    visa_confidence:    Optional[float] = None
    # positive / likely_positive / unknown / likely_negative / negative / n/a
    sponsorship_signal: str = "n/a"
    perm_filings_total: int  = 0      # total PERM certifications found
    sponsorship_trend:  str  = "n/a"  # increasing/stable/decreasing/...

    # ── Final combined score ──────────────────────────────────────────────────
    final_score: float   # main ranking score used to sort results

    # ── Plain-English explanation ─────────────────────────────────────────────
    reasoning: str
