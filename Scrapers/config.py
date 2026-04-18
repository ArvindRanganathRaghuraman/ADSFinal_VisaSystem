# ── Scraper Configuration ──────────────────────────────────────────────────────
# To add or remove companies, edit Scrapers/companies.json — no Python changes needed.

import json
from pathlib import Path

# Role keywords used to flag "target" jobs in the scraped data.
# Matching is case-insensitive substring match on job title.
TARGET_KEYWORDS = {"data engineer", "data analyst", "data scientist"}

# Load companies from the companion JSON file
_COMPANIES_FILE = Path(__file__).parent / "companies.json"
_company_list: list[dict] = json.loads(_COMPANIES_FILE.read_text(encoding="utf-8"))

# All company slugs — derived from companies.json
COMPANIES: list[str] = [c["slug"] for c in _company_list]

# Human-readable canonical names for each slug (used for display + PERM matching)
COMPANY_DISPLAY_NAMES: dict[str, str] = {c["slug"]: c["display_name"] for c in _company_list}

# HTTP settings
REQUEST_DELAY   = 1.2   # seconds between requests per company
REQUEST_TIMEOUT = 15    # seconds per HTTP call

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
