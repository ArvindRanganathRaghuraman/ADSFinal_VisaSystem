"""
pipeline/ws4_job_matcher.py
----------------------------
WS4: Semantic job matching using sentence-transformers (all-MiniLM-L6-v2).
Completely free — model runs locally, no API calls.

Two operations:
  build_index()  — embed all target-role jobs from the silver layer, save to disk.
                   Run once after WS1, then re-run whenever new jobs are scraped.

  match()        — embed a CandidateProfile, compute similarity against stored
                   job embeddings, apply rule-based boosting, return ranked list.

Scoring (final_score = weighted sum):
  50%  semantic_score      — cosine similarity between resume and job embeddings
  30%  skill_overlap_score — fraction of resume skills found in job description
  10%  role_match          — job title aligns with candidate's target roles
  10%  location_match      — job location matches candidate's preferred locations

Usage:
  # Build index (run after WS1 scrape)
  python -m pipeline.ws4_job_matcher --build

  # Match a profile against the index
  from pipeline.ws4_job_matcher import match
  from pipeline.ws3_resume_parser import parse_resume
  from models.candidate_profile import UserInputs

  profile = parse_resume("resume.pdf", UserInputs(...))
  results = match(profile, top_n=20)
  for r in results:
      print(r.match_score, r.title, r.company_slug)
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.candidate_profile import CandidateProfile
from models.job_match import JobMatch

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 80 MB, fast, free
JOBS_SILVER      = _ROOT / "Scrapers" / "data" / "silver" / "jobs_all.parquet"
INDEX_DIR        = _ROOT / "Scrapers" / "data" / "silver"
EMBEDDINGS_PATH  = INDEX_DIR / "job_embeddings.npy"
JOB_INDEX_PATH   = INDEX_DIR / "job_index.parquet"

# Scoring weights — must sum to 1.0
W_SEMANTIC  = 0.50
W_SKILL     = 0.30
W_ROLE      = 0.10
W_LOCATION  = 0.10

# Only embed the first N chars of each description (speed + context quality)
DESC_CHARS = 1500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove any residual HTML tags from description text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _job_text(row: pd.Series) -> str:
    """Construct the text string we embed for a job."""
    desc = _strip_html(str(row.get("description_text", "")))[:DESC_CHARS]
    return f"{row['title']} at {row['company_slug']}. {desc}"


def _resume_text(profile: CandidateProfile) -> str:
    """Construct the text string we embed for a candidate."""
    skills   = " ".join(profile.technical_skills)
    titles   = " ".join(profile.job_titles)
    roles    = " ".join(profile.target_roles or profile.best_fit_roles)
    exp      = f"{int(profile.years_of_experience)} years experience" if profile.years_of_experience else ""
    degree   = profile.highest_degree or ""
    summary  = profile.profile_summary or ""
    return f"{roles} {titles} {exp} {degree} {skills} {summary}"


def _skill_overlap(resume_skills: list[str], job_text: str) -> tuple[float, list[str]]:
    """
    Fraction of resume skills that appear in the job text.
    Returns (score 0-1, list of matched skill strings).
    """
    if not resume_skills:
        return 0.0, []
    job_lower = job_text.lower()
    matched = [s for s in resume_skills if s.lower() in job_lower]
    return len(matched) / len(resume_skills), matched


def _role_match(job_title: str, target_roles: list[str]) -> bool:
    """True if the job title contains any of the candidate's target role keywords."""
    if not target_roles:
        return False
    title_lower = job_title.lower()
    role_keywords = {
        "Data Engineer":       ["data engineer", "data engineering"],
        "Data Scientist":      ["data scientist", "data science"],
        "Data Analyst":        ["data analyst"],
        "Analytics Engineer":  ["analytics engineer"],
        "ML Engineer":         ["machine learning engineer", "ml engineer"],
        "Software Engineer":   ["software engineer"],
    }
    for role in target_roles:
        keywords = role_keywords.get(role, [role.lower()])
        if any(kw in title_lower for kw in keywords):
            return True
    return False


def _location_match(job_location: str, preferred: list[str]) -> bool:
    """True if the job location overlaps with any of the candidate's preferred locations."""
    if not preferred:
        return False
    loc_lower = job_location.lower()
    if "remote" in loc_lower:
        return True
    for pref in preferred:
        # Match on city name or state abbreviation
        parts = [p.strip().lower() for p in pref.replace(",", " ").split()]
        if any(p in loc_lower for p in parts if len(p) > 2):
            return True
    return False


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    """
    Embed all target-role jobs from the silver layer and save:
      job_embeddings.npy  — float32 array (n_jobs × 384)
      job_index.parquet   — metadata aligned with the embedding rows
    """
    if EMBEDDINGS_PATH.exists() and JOB_INDEX_PATH.exists() and not force:
        log.info("Job index already exists. Pass force=True to rebuild.")
        return

    if not JOBS_SILVER.exists():
        raise FileNotFoundError(
            f"Silver jobs not found at {JOBS_SILVER}. Run WS1 first."
        )

    df = pd.read_parquet(JOBS_SILVER)
    jobs = df[df["is_target_role"]].copy().reset_index(drop=True)
    log.info("Building index for %d target-role jobs...", len(jobs))

    # Build text for each job
    texts = [_job_text(row) for _, row in jobs.iterrows()]

    # Load model (downloads ~80 MB on first run)
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # pre-normalise so dot product = cosine sim
        convert_to_numpy=True,
    )

    # Save embeddings
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    log.info("Embeddings saved → %s  shape=%s", EMBEDDINGS_PATH, embeddings.shape)

    # Save metadata index (no large text columns)
    index_cols = [
        "job_id", "company_slug", "title", "location",
        "apply_url", "posted_at", "is_active",
    ]
    # Keep description for WS5 signal extraction — truncated to save space
    jobs["description_clean"] = jobs["description_text"].apply(
        lambda t: _strip_html(str(t))[:3000]
    )
    index_cols.append("description_clean")

    jobs[index_cols].to_parquet(JOB_INDEX_PATH, index=False)
    log.info("Job index saved → %s  (%d rows)", JOB_INDEX_PATH, len(jobs))


# ── Matcher ───────────────────────────────────────────────────────────────────

def match(
    profile: CandidateProfile,
    top_n: int = 20,
) -> list[JobMatch]:
    """
    Match a CandidateProfile against the pre-built job index.

    Returns a list of JobMatch objects sorted by match_score descending.
    """
    if not EMBEDDINGS_PATH.exists() or not JOB_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Job index not found. Run: python -m pipeline.ws4_job_matcher --build"
        )

    # Load index
    job_index  = pd.read_parquet(JOB_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)   # shape: (n_jobs, 384)

    # Embed resume
    model       = SentenceTransformer(EMBEDDING_MODEL)
    resume_text = _resume_text(profile)
    resume_emb  = model.encode(
        [resume_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]   # shape: (384,)

    # Semantic scores — dot product of normalised vectors = cosine similarity
    semantic_scores = (embeddings @ resume_emb).astype(float)   # shape: (n_jobs,)

    # Build full score for each job
    results: list[JobMatch] = []
    for i, row in job_index.iterrows():
        desc   = str(row.get("description_clean", ""))
        sem    = float(semantic_scores[i])
        skill_score, matched = _skill_overlap(profile.technical_skills, desc)
        role   = _role_match(row["title"], profile.target_roles or profile.best_fit_roles)
        loc    = _location_match(row["location"], profile.preferred_locations)

        final = (
            W_SEMANTIC * sem
            + W_SKILL   * skill_score
            + W_ROLE    * (1.0 if role else 0.0)
            + W_LOCATION * (1.0 if loc else 0.0)
        )

        results.append(JobMatch(
            job_id           = str(row["job_id"]),
            company_slug     = str(row["company_slug"]),
            title            = str(row["title"]),
            location         = str(row["location"]),
            apply_url        = str(row["apply_url"]),
            posted_at        = str(row["posted_at"]),
            description_text = desc,
            match_score          = round(final, 4),
            semantic_score       = round(sem, 4),
            skill_overlap_score  = round(skill_score, 4),
            role_match           = role,
            location_match       = loc,
            matched_skills       = matched,
        ))

    results.sort(key=lambda r: r.match_score, reverse=True)
    log.info("Matched %d jobs → returning top %d", len(results), top_n)
    return results[:top_n]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="WS4: Job matcher")
    parser.add_argument("--build", action="store_true", help="Build/rebuild job embedding index")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    if args.build:
        build_index(force=args.force)
    else:
        parser.print_help()
