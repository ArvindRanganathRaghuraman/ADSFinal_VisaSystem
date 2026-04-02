

import requests
import json
import time
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_KEYWORDS = {"data engineer", "data analyst", "data scientist"}

COMPANIES = [
    "stripe", "airbnb", "lyft", "pinterest", "robinhood",
    "coinbase", "databricks", "snowflake", "figma", "notion",
    "airtable", "brex", "rippling", "gusto", "lattice"
]

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolves to wherever you run the script from — just keep data/ next to it
BASE_DIR = Path(__file__).parent / "data"
BASE_DIR.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc)

DATE_STR   = now.strftime("%Y-%m-%d")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY   = 1.2
REQUEST_TIMEOUT = 15


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_target_role(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in TARGET_KEYWORDS)


US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new hampshire","new jersey","new mexico","new york","north carolina",
    "north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming",
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia",
    "ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt",
    "va","wa","wv","wi","wy","dc","washington d.c.","washington dc",
}

def is_us_location(location: str) -> bool:
    if not location:
        return False
    loc = location.lower()
    if "remote" in loc and ("us" in loc or "united states" in loc or "america" in loc):
        return True
    # bare "remote" with no country hint — include it (likely US-only boards)
    if loc.strip() == "remote":
        return True
    if "united states" in loc or ", usa" in loc or " usa" in loc:
        return True
    # check for ", ST" pattern (city, State) or state name anywhere
    parts = [p.strip().rstrip(".") for p in re.split(r"[,/|]", loc)]
    for part in parts:
        if part in US_STATES:
            return True
    return False


def fetch_jobs_for_company(slug: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            log.warning(f"[{slug}] No Greenhouse board found — skipping")
            return []
        resp.raise_for_status()
        return resp.json().get("jobs", [])
    except requests.exceptions.RequestException as e:
        log.error(f"[{slug}] Request failed: {e}")
        return []


def parse_job(raw: dict, company_slug: str) -> dict:
    loc  = raw.get("location", {})
    location_str = loc.get("name", "") if isinstance(loc, dict) else str(loc)
    depts = [d.get("name", "") for d in raw.get("departments", [])]

    return {
        "job_id"           : str(raw.get("id", "")),
        "company_slug"     : company_slug,
        "title"            : raw.get("title", ""),
        "is_target_role"   : is_target_role(raw.get("title", "")),
        "location"         : location_str,
        "posted_at"        : raw.get("updated_at", ""),
        "scraped_at"       : now.isoformat(),
        "description_html" : raw.get("content", ""),
        "description_text" : clean_html(raw.get("content", "")),
        "departments"      : ", ".join(depts),
        "apply_url"        : raw.get("absolute_url", ""),
        "source_ats"       : "greenhouse",
        "raw_json"         : json.dumps(raw),
    }


# ── Scraper ────────────────────────────────────────────────────────────────────
def scrape_all(companies: list[str]) -> list[dict]:
    all_records = []
    for slug in companies:
        log.info(f"Scraping [{slug}] ...")
        raw_jobs = fetch_jobs_for_company(slug)
        parsed   = [parse_job(j, slug) for j in raw_jobs]
        parsed   = [p for p in parsed if is_us_location(p["location"])]
        target_count = sum(1 for p in parsed if p["is_target_role"])
        log.info(f"  → {len(parsed)} US jobs  |  {target_count} target-role jobs")
        all_records.extend(parsed)
        time.sleep(REQUEST_DELAY)
    return all_records


# ── Storage ────────────────────────────────────────────────────────────────────
def save_bronze(records: list[dict]) -> None:
    if not records:
        log.warning("No records to save.")
        return

    bronze_dir = BASE_DIR / "bronze" / "greenhouse"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    # Drop raw_json — it contains commas/newlines that make CSV messy
    # It's still in the JSON backup below if you ever need it
    csv_df = df.drop(columns=["raw_json", "description_html"])

    csv_path = bronze_dir / f"greenhouse_{DATE_STR}.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"CSV  → {csv_path}  ({len(csv_df)} rows)")

    # JSON — keeps full fidelity including raw_json, easy to inspect manually
    json_path = bronze_dir / f"greenhouse_{DATE_STR}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    log.info(f"JSON → {json_path}")

    # manifest.json — tracks every run, useful when you migrate to PostgreSQL
    manifest_path = bronze_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest.append({
        "run_date"      : DATE_STR,
        "scraped_at"    : now.isoformat(),
        "csv_path"      : str(csv_path.relative_to(BASE_DIR)),
        "json_path"     : str(json_path.relative_to(BASE_DIR)),
        "total_jobs"    : len(df),
        "target_jobs"   : int(df["is_target_role"].sum()),
        "companies_hit" : int(df["company_slug"].nunique()),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest updated → {manifest_path}")

    # Summary
    log.info(
        f"\n── Run Summary ({DATE_STR}) ──\n"
        f"  Total jobs    : {len(df)}\n"
        f"  Target roles  : {int(df['is_target_role'].sum())}\n"
        f"  Companies     : {df['company_slug'].nunique()}\n"
        f"  Output dir    : {BASE_DIR}"
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = scrape_all(COMPANIES)
    save_bronze(records)