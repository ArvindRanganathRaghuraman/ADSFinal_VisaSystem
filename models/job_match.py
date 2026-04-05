"""
models/job_match.py
--------------------
Output schema from WS4 (job matching).
Passed downstream to WS5 (confidence scoring) and the frontend.
"""

from pydantic import BaseModel


class JobMatch(BaseModel):
    # ── Job identity ──────────────────────────────────────────────────────────
    job_id:           str
    company_slug:     str
    title:            str
    location:         str
    apply_url:        str
    posted_at:        str

    # ── Description (kept for WS5 sponsorship signal scan) ────────────────────
    description_text: str

    # ── Match scores ──────────────────────────────────────────────────────────
    match_score:          float   # 0-1 final weighted score
    semantic_score:       float   # cosine similarity
    skill_overlap_score:  float   # % of resume skills found in job text
    role_match:           bool    # job title aligns with candidate's target roles
    location_match:       bool    # job location matches candidate's preferences

    # ── Evidence ──────────────────────────────────────────────────────────────
    matched_skills: list[str]     # which resume skills appeared in the job text
