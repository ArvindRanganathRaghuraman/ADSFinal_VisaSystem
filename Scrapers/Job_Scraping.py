"""
Scrapers/Job_Scraping.py
------------------------
Multi-ATS job scraper: Greenhouse, Lever, Ashby.

To add a new company, edit Scrapers/companies.json — set its "ats" field to
"greenhouse", "lever", or "ashby".

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
from typing import Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from Scrapers.config import (
    COMPANIES,
    COMPANY_ATS,
    HEADERS,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    TARGET_KEYWORDS,
)

log = logging.getLogger(__name__)

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
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "washington d.c.", "washington dc",
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


def _get(url: str, slug: str, **kwargs) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
        if resp.status_code == 404:
            log.warning("[%s] 404 — board not found, skipping", slug)
            return None
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        log.error("[%s] Request failed: %s", slug, e)
        return None


# ── Greenhouse ─────────────────────────────────────────────────────────────────

def fetch_greenhouse_jobs(slug: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    resp = _get(url, slug)
    return resp.json().get("jobs", []) if resp else []


def parse_greenhouse_job(raw: dict, slug: str, run_time: datetime) -> dict:
    loc = raw.get("location", {})
    location_str = loc.get("name", "") if isinstance(loc, dict) else str(loc)
    depts = [d.get("name", "") for d in raw.get("departments", [])]
    return {
        "job_id":           str(raw.get("id", "")),
        "company_slug":     slug,
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


# ── Lever ──────────────────────────────────────────────────────────────────────

def fetch_lever_jobs(slug: str) -> list[dict]:
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    resp = _get(url, slug)
    if not resp:
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


def parse_lever_job(raw: dict, slug: str, run_time: datetime) -> dict:
    cats = raw.get("categories", {})
    location_str = cats.get("location", "") or cats.get("allLocations", [""])[0] if isinstance(cats.get("allLocations"), list) else cats.get("location", "")
    created_ms = raw.get("createdAt", 0)
    posted_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).isoformat() if created_ms else ""
    html = raw.get("description", "") + "".join(
        f"<h3>{lst.get('text','')}</h3><ul>{''.join(f'<li>{i}</li>' for i in lst.get('content',[]))}</ul>"
        for lst in raw.get("lists", [])
    )
    return {
        "job_id":           str(raw.get("id", "")),
        "company_slug":     slug,
        "title":            raw.get("text", ""),
        "is_target_role":   is_target_role(raw.get("text", "")),
        "location":         location_str,
        "posted_at":        posted_at,
        "scraped_at":       run_time.isoformat(),
        "description_html": html,
        "description_text": clean_html(html),
        "departments":      cats.get("department", "") or cats.get("team", ""),
        "apply_url":        raw.get("hostedUrl", ""),
        "source_ats":       "lever",
        "raw_json":         json.dumps(raw),
    }


# ── Ashby ──────────────────────────────────────────────────────────────────────

def fetch_ashby_jobs(slug: str) -> list[dict]:
    url = "https://api.ashbyhq.com/posting-public/job-board.list"
    try:
        resp = requests.post(
            url,
            json={"organizationHostedJobsPageName": slug},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            log.warning("[%s] 404 — Ashby board not found, skipping", slug)
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) if data.get("success") else []
    except requests.exceptions.RequestException as e:
        log.error("[%s] Ashby request failed: %s", slug, e)
        return []


def parse_ashby_job(raw: dict, slug: str, run_time: datetime) -> dict:
    dept = (raw.get("department") or {}).get("name", "")
    team = (raw.get("team") or {}).get("name", "")
    html = raw.get("descriptionHtml", "")
    return {
        "job_id":           str(raw.get("id", "")),
        "company_slug":     slug,
        "title":            raw.get("title", ""),
        "is_target_role":   is_target_role(raw.get("title", "")),
        "location":         raw.get("locationName", ""),
        "posted_at":        raw.get("publishedDate", ""),
        "scraped_at":       run_time.isoformat(),
        "description_html": html,
        "description_text": clean_html(html),
        "departments":      dept or team,
        "apply_url":        raw.get("applicationFormUrl", ""),
        "source_ats":       "ashby",
        "raw_json":         json.dumps(raw),
    }


# ── SmartRecruiters ────────────────────────────────────────────────────────────

def fetch_smartrecruiters_jobs(slug: str) -> list[dict]:
    url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
    resp = _get(url, slug, params={"limit": 100})
    if not resp:
        return []
    data = resp.json()
    return data.get("content", [])


def parse_smartrecruiters_job(raw: dict, slug: str, run_time: datetime) -> dict:
    loc = raw.get("location", {})
    location_str = ", ".join(filter(None, [
        loc.get("city", ""), loc.get("region", ""), loc.get("country", "")
    ]))
    dept = (raw.get("department") or {}).get("label", "")
    return {
        "job_id":           str(raw.get("id", "")),
        "company_slug":     slug,
        "title":            raw.get("name", ""),
        "is_target_role":   is_target_role(raw.get("name", "")),
        "location":         location_str,
        "posted_at":        raw.get("releasedDate", ""),
        "scraped_at":       run_time.isoformat(),
        "description_html": "",
        "description_text": "",
        "departments":      dept,
        "apply_url":        raw.get("ref", ""),
        "source_ats":       "smartrecruiters",
        "raw_json":         json.dumps(raw),
    }


# ── Workable ───────────────────────────────────────────────────────────────────

def fetch_workable_jobs(slug: str) -> list[dict]:
    url = f"https://apply.workable.com/api/v3/accounts/{slug}/jobs"
    try:
        resp = requests.post(
            url,
            json={"limit": 50, "details": False},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            log.warning("[%s] 404 — Workable board not found, skipping", slug)
            return []
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.exceptions.RequestException as e:
        log.error("[%s] Workable request failed: %s", slug, e)
        return []


def parse_workable_job(raw: dict, slug: str, run_time: datetime) -> dict:
    loc = raw.get("location", {})
    location_str = ", ".join(filter(None, [
        loc.get("city", ""), loc.get("region", ""), loc.get("country", "")
    ]))
    return {
        "job_id":           str(raw.get("shortcode", raw.get("id", ""))),
        "company_slug":     slug,
        "title":            raw.get("title", ""),
        "is_target_role":   is_target_role(raw.get("title", "")),
        "location":         location_str,
        "posted_at":        raw.get("published_on", ""),
        "scraped_at":       run_time.isoformat(),
        "description_html": "",
        "description_text": "",
        "departments":      raw.get("department", ""),
        "apply_url":        raw.get("url", ""),
        "source_ats":       "workable",
        "raw_json":         json.dumps(raw),
    }


# ── ATS dispatcher ─────────────────────────────────────────────────────────────

_ATS_SCRAPERS: dict[str, tuple[Callable, Callable]] = {
    "greenhouse":      (fetch_greenhouse_jobs,      parse_greenhouse_job),
    "lever":           (fetch_lever_jobs,           parse_lever_job),
    "ashby":           (fetch_ashby_jobs,           parse_ashby_job),
    "smartrecruiters": (fetch_smartrecruiters_jobs, parse_smartrecruiters_job),
    "workable":        (fetch_workable_jobs,         parse_workable_job),
}


def scrape_company(slug: str, run_time: datetime) -> list[dict]:
    known_ats = COMPANY_ATS.get(slug)
    platforms = (
        [(known_ats, _ATS_SCRAPERS[known_ats])]
        if known_ats and known_ats in _ATS_SCRAPERS
        else list(_ATS_SCRAPERS.items())
    )

    for ats, (fetch_fn, parse_fn) in platforms:
        raw_jobs = fetch_fn(slug)
        if raw_jobs:
            parsed = [parse_fn(j, slug, run_time) for j in raw_jobs]
            parsed = [p for p in parsed if is_us_location(p["location"])]
            target_count = sum(1 for p in parsed if p["is_target_role"])
            log.info("  [%s/%s] → %d US jobs | %d target-role", slug, ats, len(parsed), target_count)
            return parsed

    log.warning("  [%s] no jobs found on any platform", slug)
    return []


# ── Scraper ────────────────────────────────────────────────────────────────────

def scrape_all(
    companies: list[str] | None = None,
    run_time: datetime | None = None,
) -> list[dict]:
    """
    Scrape all companies (across all ATS platforms) and return parsed job records.

    Args:
        companies:  list of company slugs; defaults to config.COMPANIES
        run_time:   UTC datetime stamped on each record; defaults to now
    """
    if companies is None:
        companies = COMPANIES
    if run_time is None:
        run_time = datetime.now(timezone.utc)

    all_records: list[dict] = []
    for slug in companies:
        log.info("Scraping [%s] ...", slug)
        all_records.extend(scrape_company(slug, run_time))
        time.sleep(REQUEST_DELAY)

    return all_records


# ── Storage ────────────────────────────────────────────────────────────────────

def save_bronze(records: list[dict], run_time: datetime | None = None) -> None:
    """
    Persist scraped records (all ATS combined) to the bronze layer.

    Writes:
      bronze/jobs/jobs_YYYY-MM-DD.csv   (no raw_json / HTML)
      bronze/jobs/jobs_YYYY-MM-DD.json  (full fidelity)
      bronze/jobs/manifest.json         (run log)
    """
    if not records:
        log.warning("No records to save.")
        return

    if run_time is None:
        run_time = datetime.now(timezone.utc)

    date_str = run_time.strftime("%Y-%m-%d")
    bronze_dir = BASE_DIR / "bronze" / "jobs"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    csv_path = bronze_dir / f"jobs_{date_str}.csv"
    df.drop(columns=["raw_json", "description_html"], errors="ignore").to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    log.info("CSV  → %s  (%d rows)", csv_path, len(df))

    json_path = bronze_dir / f"jobs_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    log.info("JSON → %s", json_path)

    manifest_path = bronze_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    ats_counts = df.groupby("source_ats").size().to_dict()
    manifest.append({
        "run_date":       date_str,
        "scraped_at":     run_time.isoformat(),
        "csv_path":       str(csv_path.relative_to(BASE_DIR)),
        "json_path":      str(json_path.relative_to(BASE_DIR)),
        "total_jobs":     len(df),
        "target_jobs":    int(df["is_target_role"].sum()),
        "companies_hit":  int(df["company_slug"].nunique()),
        "by_ats":         ats_counts,
    })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "\n── Run Summary (%s) ──\n"
        "  Total jobs   : %d\n"
        "  Target roles : %d\n"
        "  Companies    : %d\n"
        "  By ATS       : %s\n"
        "  Output dir   : %s",
        date_str, len(df), int(df["is_target_role"].sum()),
        df["company_slug"].nunique(), ats_counts, BASE_DIR,
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
