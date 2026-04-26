"""
pipeline/ws3_resume_parser.py
------------------------------
WS3: Resume ingestion + feature extraction.

Primary:  Groq API (Llama 3.3 70B) — handles any resume format/layout
Fallback: Rule-based NLP            — used if Groq is unavailable or fails

Steps:
  1. Extract raw text from PDF or DOCX (pdfplumber / python-docx)
  2. Try Groq LLM extraction → structured JSON
     If Groq fails → fall back to regex + skills taxonomy
  3. Merge with user-provided inputs (visa status, target roles, locations)
  4. Return validated CandidateProfile

Usage (CLI):
  python -m pipeline.ws3_resume_parser \
      --resume path/to/resume.pdf \
      --visa "F-1 OPT" \
      --roles "Data Engineer" "Data Scientist" \
      --locations "San Francisco, CA" "Remote"

Usage (import):
  from pipeline.ws3_resume_parser import parse_resume
  from models.candidate_profile import UserInputs

  profile = parse_resume("resume.pdf", UserInputs(
      visa_status="F-1 OPT",
      requires_sponsorship=True,
      target_roles=["Data Engineer"],
      preferred_locations=["Remote"],
  ))
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

import pdfplumber
from groq import Groq

from models.candidate_profile import CandidateProfile, UserInputs

log = logging.getLogger(__name__)

MAX_RESUME_CHARS = 12_000
GROQ_MODEL       = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Skills taxonomy  (used by rule-based fallback + to enrich Groq output)
# ─────────────────────────────────────────────────────────────────────────────
SKILLS_TAXONOMY: dict[str, list[str]] = {
    "languages": [
        "Python", "SQL", "R", "Java", "Scala", "Go", "Bash", "Shell",
        "C++", "C#", "JavaScript", "TypeScript", "Ruby", "Rust", "Julia",
        "MATLAB", "SAS", "Perl", "Swift", "Kotlin",
    ],
    "data_processing": [
        "Apache Spark", "PySpark", "Spark", "Hadoop", "HDFS", "MapReduce",
        "Apache Kafka", "Kafka", "Apache Flink", "Flink",
        "Apache Airflow", "Airflow", "dbt", "Dagster", "Prefect", "Luigi",
        "Pandas", "NumPy", "Polars", "Dask", "Ray",
        "ETL", "ELT", "Data Pipeline", "Data Warehouse", "Data Lake",
        "Data Lakehouse", "Delta Lake", "Apache Iceberg", "Iceberg", "Apache Hudi",
    ],
    "ml_ai": [
        "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
        "Computer Vision", "Reinforcement Learning",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost", "LightGBM",
        "CatBoost", "Hugging Face", "Transformers", "BERT",
        "MLflow", "Kubeflow", "SageMaker", "Vertex AI",
        "Feature Engineering", "Model Training", "Model Deployment",
        "A/B Testing", "Statistical Analysis", "Time Series",
    ],
    "databases": [
        "PostgreSQL", "MySQL", "SQLite", "Oracle", "SQL Server",
        "MongoDB", "Cassandra", "DynamoDB", "Redis", "Elasticsearch",
        "Neo4j", "ClickHouse", "Druid", "Presto", "Trino", "Hive",
        "Snowflake", "BigQuery", "Redshift", "Azure Synapse", "Databricks",
    ],
    "cloud": [
        "AWS", "Amazon Web Services", "GCP", "Google Cloud", "Azure",
        "S3", "EC2", "Lambda", "RDS", "EMR", "CloudFormation",
        "Cloud Storage", "Cloud Run", "Pub/Sub", "Dataflow", "Dataproc",
    ],
    "visualization": [
        "Tableau", "Power BI", "Looker", "Metabase", "Grafana",
        "Matplotlib", "Seaborn", "Plotly", "D3.js", "Superset",
    ],
    "devops_infra": [
        "Docker", "Kubernetes", "Terraform", "Ansible", "Helm",
        "Git", "GitHub", "GitLab", "CI/CD", "Jenkins", "GitHub Actions",
        "Linux", "Unix", "REST API", "GraphQL", "Microservices",
    ],
    "analytics": [
        "Data Analysis", "Business Intelligence", "BI", "Reporting",
        "Dashboard", "KPI", "Excel", "Google Sheets",
        "Statistical Modeling", "Regression", "Hypothesis Testing",
        "Cohort Analysis", "Product Analytics",
    ],
}

ALL_SKILLS: list[str] = [s for grp in SKILLS_TAXONOMY.values() for s in grp]
_SHORT_SKILLS = {"R", "Go", "C", "SAS", "BI", "AWS", "GCP"}

ROLE_SIGNALS: dict[str, list[str]] = {
    "Data Engineer": [
        "Spark", "PySpark", "Airflow", "dbt", "Kafka", "ETL", "ELT",
        "Data Pipeline", "Data Warehouse", "Data Lake", "Hadoop",
        "Flink", "Dagster", "Delta Lake", "Redshift", "BigQuery",
        "Snowflake", "Databricks", "Python", "SQL", "Scala",
    ],
    "Data Scientist": [
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "scikit-learn", "XGBoost", "NLP", "Computer Vision",
        "Statistical Analysis", "Feature Engineering", "A/B Testing",
        "Python", "R",
    ],
    "Data Analyst": [
        "SQL", "Tableau", "Power BI", "Looker", "Excel", "Dashboard",
        "KPI", "Reporting", "Business Intelligence", "Data Analysis",
        "Python", "Statistical Modeling",
    ],
    "Analytics Engineer": [
        "dbt", "SQL", "Data Warehouse", "Snowflake", "BigQuery",
        "Redshift", "Databricks", "Looker", "ETL", "ELT",
    ],
    "ML Engineer": [
        "MLflow", "Kubeflow", "SageMaker", "Vertex AI", "Docker",
        "Kubernetes", "Model Deployment", "TensorFlow", "PyTorch",
        "Python", "Machine Learning", "CI/CD",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(path: Path) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(path)
    elif suffix in (".docx", ".doc"):
        text = extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or DOCX.")
    if not text.strip():
        raise ValueError(f"No text extracted from {path.name}.")
    if len(text) > MAX_RESUME_CHARS:
        log.warning("Truncating resume from %d to %d chars", len(text), MAX_RESUME_CHARS)
        text = text[:MAX_RESUME_CHARS]
    log.info("Extracted %d chars from %s", len(text), path.name)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Groq extraction  (primary)
# ─────────────────────────────────────────────────────────────────────────────

_GROQ_PROMPT = """\
You are a precise resume parser. Extract structured information from the resume below and return ONLY valid JSON — no markdown, no explanation, nothing else.

Return exactly this JSON structure (use null for missing fields, empty arrays for missing lists):

{{
  "name": "string or null",
  "email": "string or null",
  "technical_skills": ["list of all technical skills, tools, languages, frameworks"],
  "soft_skills": ["list of soft skills if mentioned"],
  "years_of_experience": number or null,
  "job_titles": ["all job titles held, most recent first"],
  "industries": ["industries or domains worked in"],
  "highest_degree": "Bachelor's or Master's or PhD or null",
  "field_of_study": "string or null",
  "university": "string or null",
  "best_fit_roles": ["roles this resume fits best, choose from: Data Engineer, Data Scientist, Data Analyst, Analytics Engineer, ML Engineer, Software Engineer"],
  "seniority_level": "Entry or Mid or Senior or Staff or Principal or null",
  "profile_summary": "2-3 sentence plain-English summary of the candidate"
}}

Resume:
{resume_text}
"""


def _extract_with_groq(resume_text: str) -> dict:
    """Call Groq Llama 3.3 to extract structured fields. Returns raw dict."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)
    prompt = _GROQ_PROMPT.format(resume_text=resume_text)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback  (used if Groq fails)
# ─────────────────────────────────────────────────────────────────────────────

_DEGREE_PATTERNS = [
    (re.compile(r'\b(ph\.?d\.?|doctor(?:ate)?)\b', re.I),             "PhD"),
    (re.compile(r'\b(m\.?s\.?|master(?:\'?s)?|m\.?eng\.?)\b', re.I), "Master's"),
    (re.compile(r'\b(b\.?s\.?|b\.?e\.?|bachelor(?:\'?s)?|b\.?tech\.?)\b', re.I), "Bachelor's"),
]
_FIELD_PATTERNS = [
    (re.compile(r'computer science', re.I),       "Computer Science"),
    (re.compile(r'data science', re.I),            "Data Science"),
    (re.compile(r'information system', re.I),      "Information Systems"),
    (re.compile(r'software engineering', re.I),    "Software Engineering"),
    (re.compile(r'electrical engineering', re.I),  "Electrical Engineering"),
    (re.compile(r'mathematics|math\b', re.I),      "Mathematics"),
    (re.compile(r'statistics', re.I),              "Statistics"),
    (re.compile(r'business analytics', re.I),      "Business Analytics"),
    (re.compile(r'machine learning', re.I),        "Machine Learning"),
]
_YOE_RE      = re.compile(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:professional\s+)?experience', re.I)
_DATE_RANGE  = re.compile(r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|present|current|now)', re.I)
_EMAIL_RE    = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')


def _rule_based_extract(text: str) -> dict:
    """Pure regex/taxonomy extraction — no external calls."""
    # Skills
    skills = []
    for skill in ALL_SKILLS:
        if skill in _SHORT_SKILLS:
            pattern = r'(?<![A-Za-z])' + re.escape(skill) + r'(?![A-Za-z])'
        else:
            pattern = r'(?i)\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            skills.append(skill)

    # Email
    em = _EMAIL_RE.search(text)
    email = em.group(0) if em else None

    # Name — first short alphabetic line
    name = None
    for line in text.splitlines()[:4]:
        words = line.strip().split()
        if 2 <= len(words) <= 4 and all(re.match(r"[A-Za-z\-'\.]+$", w) for w in words):
            name = line.strip()
            break

    # Degree
    degree = None
    for pat, label in _DEGREE_PATTERNS:
        if pat.search(text):
            degree = label
            break

    # Field
    field = None
    for pat, label in _FIELD_PATTERNS:
        if pat.search(text):
            field = label
            break

    # University
    university = None
    for line in text.splitlines():
        if re.search(r'\b(university|institute of technology|college)\b', line, re.I):
            university = line.strip()[:120]
            break

    # Years of experience
    yoe = None
    m = _YOE_RE.search(text)
    if m:
        yoe = float(m.group(1))
    else:
        years = []
        for m2 in _DATE_RANGE.finditer(text):
            years.append(int(m2.group(1)))
            end = m2.group(2).lower()
            years.append(2026 if end in ("present", "current", "now") else int(end))
        if len(years) >= 2:
            yoe = float(max(0, min(2026 - min(years), 40)))

    # Job titles
    title_patterns = [
        r"(?:senior\s+)?data\s+engineer(?:ing)?",
        r"(?:senior\s+)?data\s+scientist",
        r"(?:senior\s+)?data\s+analyst",
        r"analytics\s+engineer",
        r"(?:machine\s+learning|ml)\s+engineer",
        r"software\s+engineer",
        r"platform\s+engineer",
        r"backend\s+engineer",
        r"data\s+architect",
    ]
    titles = []
    seen_t: set[str] = set()
    for p in title_patterns:
        mt = re.search(p, text, re.I)
        if mt:
            t = " ".join(w.capitalize() for w in mt.group(0).split())
            if t.lower() not in seen_t:
                titles.append(t)
                seen_t.add(t.lower())

    # Industries
    industry_map = {
        "Fintech":     ["fintech", "payment", "banking", "trading", "lending"],
        "Healthtech":  ["healthcare", "medical", "clinical", "pharma", "biotech"],
        "E-commerce":  ["e-commerce", "ecommerce", "retail", "marketplace"],
        "SaaS":        ["saas", "b2b software"],
        "Data/AI":     ["data platform", "machine learning platform"],
    }
    industries = [ind for ind, kws in industry_map.items()
                  if any(kw in text.lower() for kw in kws)]

    # Best-fit roles
    skills_lower = {s.lower() for s in skills}
    role_scores = {
        role: sum(1 for s in signals if s.lower() in skills_lower)
        for role, signals in ROLE_SIGNALS.items()
    }
    best_roles = sorted([r for r, sc in role_scores.items() if sc >= 2],
                        key=lambda r: role_scores[r], reverse=True)

    # Seniority
    seniority = None
    text_lower = text.lower()
    for level, keywords in [
        ("Principal", ["principal", "distinguished"]),
        ("Staff",     ["staff"]),
        ("Senior",    ["senior", "sr.", "lead", "manager"]),
        ("Mid",       ["mid-level", " ii ", " iii "]),
        ("Entry",     ["junior", "jr.", "associate", "entry", "intern"]),
    ]:
        if any(kw in text_lower for kw in keywords):
            seniority = level
            break
    if seniority is None and yoe is not None:
        seniority = "Entry" if yoe < 2 else "Mid" if yoe < 5 else "Senior"

    # Summary
    who = name or "The candidate"
    yoe_str = f"{int(yoe)} years of experience" if yoe else "professional experience"
    top_skills = ", ".join(skills[:5]) if skills else "various technical skills"
    role_str = " and ".join(best_roles[:2]) if best_roles else "data"
    summary = (f"{who} is a {role_str} professional with {yoe_str}. "
               f"Core skills include {top_skills}."
               + (f" Holds a {degree}." if degree else ""))

    return {
        "name": name, "email": email,
        "technical_skills": skills, "soft_skills": [],
        "years_of_experience": yoe,
        "job_titles": titles, "industries": industries,
        "highest_degree": degree, "field_of_study": field, "university": university,
        "best_fit_roles": best_roles, "seniority_level": seniority,
        "profile_summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Groq output normalisation  (clean up before passing to Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(raw: dict) -> dict:
    """
    Groq sometimes returns extra fields or wrong types.
    Sanitise before constructing CandidateProfile.
    """
    def _list(v) -> list:
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    def _str(v) -> str | None:
        return str(v).strip() if v else None

    def _float(v) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    valid_seniority = {"Entry", "Mid", "Senior", "Staff", "Principal"}
    seniority = raw.get("seniority_level")
    if seniority not in valid_seniority:
        seniority = None

    return {
        "name":               _str(raw.get("name")),
        "email":              _str(raw.get("email")),
        "technical_skills":   _list(raw.get("technical_skills")),
        "soft_skills":        _list(raw.get("soft_skills")),
        "years_of_experience": _float(raw.get("years_of_experience")),
        "job_titles":         _list(raw.get("job_titles")),
        "industries":         _list(raw.get("industries")),
        "highest_degree":     _str(raw.get("highest_degree")),
        "field_of_study":     _str(raw.get("field_of_study")),
        "university":         _str(raw.get("university")),
        "best_fit_roles":     _list(raw.get("best_fit_roles")),
        "seniority_level":    seniority,
        "profile_summary":    _str(raw.get("profile_summary")) or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume(
    resume_path: str | Path,
    user_inputs: UserInputs,
) -> CandidateProfile:
    """
    Full WS3 pipeline: file → text → extraction → CandidateProfile.

    Tries Groq first; falls back to rule-based if Groq is unavailable.
    """
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {path}")

    log.info("=== WS3: Resume Parser ===")
    text = extract_resume_text(path)

    # Primary: Groq
    extracted = None
    try:
        log.info("Extracting with Groq (%s)...", GROQ_MODEL)
        raw = _extract_with_groq(text)
        extracted = _normalise(raw)
        log.info("Groq extraction successful")
    except Exception as e:
        log.warning("Groq extraction failed (%s) — falling back to rule-based", e)

    # Fallback: rule-based
    if extracted is None:
        log.info("Using rule-based extraction")
        extracted = _rule_based_extract(text)

    profile = CandidateProfile.from_extraction_and_inputs(extracted, user_inputs)

    log.info("Name      : %s", profile.name or "—")
    log.info("Skills    : %s", ", ".join(profile.technical_skills[:8]))
    log.info("Roles     : %s", ", ".join(profile.best_fit_roles))
    log.info("Seniority : %s  |  Exp: %s yrs", profile.seniority_level, profile.years_of_experience)
    log.info("=== WS3 complete ===")
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WS3: Parse resume → CandidateProfile")
    p.add_argument("--resume",     required=True)
    p.add_argument("--visa",       required=True, help='e.g. "F-1 OPT"')
    p.add_argument("--sponsor",    action="store_true", default=True)
    p.add_argument("--roles",      nargs="*", default=[])
    p.add_argument("--locations",  nargs="*", default=[])
    p.add_argument("--relocation", action="store_true", default=False)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_parser().parse_args()
    user_inputs = UserInputs(
        visa_status=args.visa,
        requires_sponsorship=args.sponsor,
        target_roles=args.roles,
        preferred_locations=args.locations,
        open_to_relocation=args.relocation,
    )
    profile = parse_resume(resume_path=args.resume, user_inputs=user_inputs)
    print("\n" + "=" * 60)
    print(json.dumps(profile.model_dump(), indent=2, default=str))
