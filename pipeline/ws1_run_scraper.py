"""
pipeline/ws1_run_scraper.py
---------------------------
WS1: Daily multi-ATS job scraper pipeline (Greenhouse, Lever, Ashby).

Steps:
  1. Scrape all configured companies across ATS platforms → bronze layer
     (date-stamped JSON + CSV per run)
  2. Merge all bronze JSON files → silver layer (deduplicated parquet/CSV)
     - Deduplicates by job_id (keeps the most-recent scrape record)
     - Tracks first_seen / last_seen / times_seen per job
     - Marks jobs as active (seen within ACTIVE_JOB_DAYS)

Usage:
  # From project root:
  python -m pipeline.ws1_run_scraper

  # Cron (daily at 06:00 UTC):
  0 6 * * * cd /path/to/ADSFinal_VisaSystem && python -m pipeline.ws1_run_scraper >> logs/ws1.log 2>&1
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Allow running as `python pipeline/ws1_run_scraper.py` from any cwd
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from Scrapers.Job_Scraping import save_bronze, scrape_all
from Scrapers.config import COMPANIES

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BRONZE_DIR       = _ROOT / "Scrapers" / "data" / "bronze" / "jobs"
SILVER_DIR       = _ROOT / "Scrapers" / "data" / "silver"
SILVER_PARQUET   = SILVER_DIR / "jobs_all.parquet"
SILVER_CSV       = SILVER_DIR / "jobs_all.csv"

# Jobs not seen in this many days are marked inactive
ACTIVE_JOB_DAYS = 14


# ── Silver builder ─────────────────────────────────────────────────────────────

def build_silver() -> pd.DataFrame:
    """
    Read every bronze JSON snapshot, deduplicate by job_id, and return
    a single DataFrame enriched with tracking columns.
    """
    all_records: list[dict] = []

    json_files = sorted(BRONZE_DIR.glob("jobs_*.json"))
    if not json_files:
        log.warning("No bronze JSON files found in %s", BRONZE_DIR)
        return pd.DataFrame()

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            records = json.load(f)
        all_records.extend(records)

    log.info("Loaded %d raw records from %d bronze files", len(all_records), len(json_files))

    df = pd.DataFrame(all_records)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True)

    # Per-job tracking stats across all scrape runs
    tracking = (
        df.groupby("job_id")
        .agg(
            first_seen=("scraped_at", "min"),
            last_seen=("scraped_at", "max"),
            times_seen=("scraped_at", "count"),
        )
        .reset_index()
    )

    # Keep only the most recent record for each job_id
    df_latest = (
        df.sort_values("scraped_at")
        .groupby("job_id", as_index=False)
        .last()
    )

    df_silver = df_latest.merge(tracking, on="job_id", how="left")

    # Active = seen in the last ACTIVE_JOB_DAYS days
    cutoff = datetime.now(timezone.utc) - timedelta(days=ACTIVE_JOB_DAYS)
    df_silver["is_active"] = df_silver["last_seen"] >= cutoff

    # Drop raw_json (large, already stored per-day in bronze JSON)
    df_silver = df_silver.drop(columns=["raw_json"], errors="ignore")

    log.info(
        "Silver summary: %d unique jobs | %d active | %d target-role",
        len(df_silver),
        df_silver["is_active"].sum(),
        df_silver["is_target_role"].sum(),
    )
    return df_silver


def save_silver(df: pd.DataFrame) -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(SILVER_PARQUET, index=False)
    log.info("Silver parquet → %s", SILVER_PARQUET)

    # CSV excludes HTML for readability
    df.drop(columns=["description_html"], errors="ignore").to_csv(
        SILVER_CSV, index=False, encoding="utf-8-sig"
    )
    log.info("Silver CSV    → %s", SILVER_CSV)


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("=== WS1: Daily Scraper Pipeline ===")
    run_time = datetime.now(timezone.utc)

    # Step 1 — scrape → bronze
    log.info("Step 1/2: Scraping %d companies (Greenhouse / Lever / Ashby)...", len(COMPANIES))
    records = scrape_all(companies=COMPANIES, run_time=run_time)
    save_bronze(records, run_time=run_time)

    # Step 2 — merge bronze → silver
    log.info("Step 2/2: Building silver jobs table...")
    df_silver = build_silver()
    if not df_silver.empty:
        save_silver(df_silver)

    log.info("=== WS1 complete ===")


if __name__ == "__main__":
    run()
