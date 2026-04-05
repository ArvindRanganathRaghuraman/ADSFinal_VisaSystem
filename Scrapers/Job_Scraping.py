"""
Scrapers/Job_Scraping.py
------------------------
Greenhouse ATS job scraper.

Can be run directly:
    python Scrapers/Job_Scraping.py

Or imported by the pipeline:
    from Scrapers.Job_Scraping import scrape_all, save_bronze
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from Scrapers.config import (
    COMPANIES,
    HEADERS,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    TARGET_KEYWORDS,
)

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "data"

# ── US state lookup ────────────────────────────────────────────────────────────
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    # abbreviations
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "washington d.c.", "washington dc",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_target_role(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in TARGET_KEYWORDS)


def is_us_location(location: str) -> bool:
    if not location:
        return False
    loc = location.lower()
    if "remote" in loc and any(x in loc for x in ("us", "united states", "america")):
        return True
    if loc.strip() == "remote":
        return True
    if "united states" in loc or ", usa" in loc or " usa" in loc:
        return True
    parts = [p.strip().rstrip(".") for p in re.split(r"[,/|]", loc)]
    return any(part in US_STATES for part in parts)


# ── API ────────────────────────────────────────────────────────────────────────

def fetch_jobs_for_company(slug: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            log.warning("[%s] No Greenhouse board found — skipping", slug)
            return []
        resp.raise_for_status()
        return resp.json().get("jobs", [])
    except requests.exceptions.RequestException as e:
        log.error("[%s] Request failed: %s", slug, e)
        return []


def parse_job(raw: dict, company_slug: str, run_time: datetime) -> dict:
    loc = raw.get("location", {})
    location_str = loc.get("name", "") if isinstance(loc, dict) else str(loc)
    depts = [d.get("name", "") for d in raw.get("departments", [])]

    return {
        "job_id":           str(raw.get("id", "")),
        "company_slug":     company_slug,
        "title":            raw.get("title", ""),
        "is_target_role":   is_target_role(raw.get("title", "")),
        "location":         location_str,
        "posted_at":        raw.get("updated_at", ""),
        "scraped_at":       run_time.isoformat(),
        "description_html": raw.get("content", ""),
        "description_text": clean_html(raw.get("content", "")),
        "departments":      ", ".join(depts),
        "apply_url":        raw.get("absolute_url", ""),
        "source_ats":       "greenhouse",
        "raw_json":         json.dumps(raw),
    }


# ── Scraper ────────────────────────────────────────────────────────────────────

def scrape_all(
    companies: list[str] | None = None,
    run_time: datetime | None = None,
) -> list[dict]:
    """
    Scrape all companies and return a list of parsed job records.

    Args:
        companies:  list of Greenhouse slugs; defaults to config.COMPANIES
        run_time:   UTC datetime stamped on each record; defaults to now
    """
    if companies is None:
        companies = COMPANIES
    if run_time is None:
        run_time = datetime.now(timezone.utc)

    all_records = []
    for slug in companies:
        log.info("Scraping [%s] ...", slug)
        raw_jobs = fetch_jobs_for_company(slug)
        parsed = [parse_job(j, slug, run_time) for j in raw_jobs]
        parsed = [p for p in parsed if is_us_location(p["location"])]
        target_count = sum(1 for p in parsed if p["is_target_role"])
        log.info("  → %d US jobs  |  %d target-role jobs", len(parsed), target_count)
        all_records.extend(parsed)
        time.sleep(REQUEST_DELAY)

    return all_records


# ── Storage ────────────────────────────────────────────────────────────────────

def save_bronze(records: list[dict], run_time: datetime | None = None) -> None:
    """
    Persist scraped records to the bronze layer.

    Writes:
      bronze/greenhouse/greenhouse_YYYY-MM-DD.csv   (no raw_json / HTML)
      bronze/greenhouse/greenhouse_YYYY-MM-DD.json  (full fidelity)
      bronze/greenhouse/manifest.json               (run log)
    """
    if not records:
        log.warning("No records to save.")
        return

    if run_time is None:
        run_time = datetime.now(timezone.utc)

    date_str = run_time.strftime("%Y-%m-%d")
    bronze_dir = BASE_DIR / "bronze" / "greenhouse"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    csv_df = df.drop(columns=["raw_json", "description_html"], errors="ignore")
    csv_path = bronze_dir / f"greenhouse_{date_str}.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("CSV  → %s  (%d rows)", csv_path, len(csv_df))

    json_path = bronze_dir / f"greenhouse_{date_str}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    log.info("JSON → %s", json_path)

    # Append to manifest
    manifest_path = bronze_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest.append({
        "run_date":       date_str,
        "scraped_at":     run_time.isoformat(),
        "csv_path":       str(csv_path.relative_to(BASE_DIR)),
        "json_path":      str(json_path.relative_to(BASE_DIR)),
        "total_jobs":     len(df),
        "target_jobs":    int(df["is_target_role"].sum()),
        "companies_hit":  int(df["company_slug"].nunique()),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "\n── Run Summary (%s) ──\n"
        "  Total jobs   : %d\n"
        "  Target roles : %d\n"
        "  Companies    : %d\n"
        "  Output dir   : %s",
        date_str, len(df), int(df["is_target_role"].sum()),
        df["company_slug"].nunique(), BASE_DIR,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _run_time = datetime.now(timezone.utc)
    _records  = scrape_all(run_time=_run_time)
    save_bronze(_records, run_time=_run_time)
