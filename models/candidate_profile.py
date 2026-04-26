"""
models/candidate_profile.py
----------------------------
Pydantic schemas shared across WS3 (extraction), WS4 (matching), WS5 (scoring).
"""

from typing import Optional
from pydantic import BaseModel, Field


class UserInputs(BaseModel):
    """
    What the user tells us directly via the UI form.
    These cannot be reliably inferred from the resume alone.
    """
    visa_status: str = Field(
        description="Current visa/work-auth status",
        examples=["F-1 OPT", "F-1 CPT", "H-1B", "Green Card", "US Citizen", "Other"],
    )
    requires_sponsorship: bool = Field(
        description="Whether the candidate needs employer sponsorship"
    )
    target_roles: list[str] = Field(
        default_factory=list,
        description="Role types the candidate is interested in, e.g. ['Data Engineer', 'Data Analyst']",
    )
    preferred_locations: list[str] = Field(
        default_factory=list,
        description="Preferred cities/states/remote, e.g. ['San Francisco, CA', 'Remote']",
    )
    open_to_relocation: bool = Field(default=False)


class CandidateProfile(BaseModel):
    """
    Fully structured candidate profile produced by WS3.
    Combines LLM-extracted resume features + user-provided inputs.
    """

    # ── Identity (best-effort from resume) ────────────────────────────────────
    name:  Optional[str] = None
    email: Optional[str] = None

    # ── Skills ────────────────────────────────────────────────────────────────
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Programming languages, frameworks, tools, platforms",
        examples=[["Python", "SQL", "Spark", "dbt", "Airflow", "AWS"]],
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description="Communication, leadership, cross-functional collaboration, etc.",
    )

    # ── Experience ────────────────────────────────────────────────────────────
    years_of_experience: Optional[float] = Field(
        default=None,
        description="Total years of professional work experience (excluding internships)",
    )
    job_titles: list[str] = Field(
        default_factory=list,
        description="All job titles held, most recent first",
        examples=[["Senior Data Engineer", "Data Engineer", "Software Engineer"]],
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Industries or domains worked in",
        examples=[["Fintech", "E-commerce", "Healthcare"]],
    )

    # ── Education ─────────────────────────────────────────────────────────────
    highest_degree: Optional[str] = Field(
        default=None,
        description="Highest academic degree",
        examples=["Bachelor's", "Master's", "PhD"],
    )
    field_of_study: Optional[str] = Field(
        default=None,
        examples=["Computer Science", "Information Systems", "Data Science"],
    )
    university: Optional[str] = None

    # ── Inferred role alignment ────────────────────────────────────────────────
    best_fit_roles: list[str] = Field(
        default_factory=list,
        description="Role types the resume most strongly aligns with (LLM inferred)",
        examples=[["Data Engineer", "Analytics Engineer"]],
    )
    seniority_level: Optional[str] = Field(
        default=None,
        description="Inferred seniority based on experience",
        examples=["Entry", "Mid", "Senior", "Staff", "Principal"],
    )

    # ── Profile summary ────────────────────────────────────────────────────────
    profile_summary: str = Field(
        default="",
        description="2-3 sentence plain-English summary of the candidate's profile",
    )

    # ── User-provided inputs (merged in) ──────────────────────────────────────
    visa_status: str = ""
    requires_sponsorship: bool = False
    target_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    open_to_relocation: bool = False

    @classmethod
    def from_extraction_and_inputs(
        cls,
        extracted: dict,
        user_inputs: "UserInputs",
    ) -> "CandidateProfile":
        """Merge LLM-extracted fields with user-provided inputs."""
        data = {**extracted}
        data["visa_status"]          = user_inputs.visa_status
        data["requires_sponsorship"] = user_inputs.requires_sponsorship
        data["target_roles"]         = user_inputs.target_roles or extracted.get("best_fit_roles", [])
        data["preferred_locations"]  = user_inputs.preferred_locations
        data["open_to_relocation"]   = user_inputs.open_to_relocation
        return cls(**data)
