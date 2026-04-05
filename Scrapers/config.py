# ── Scraper Configuration ──────────────────────────────────────────────────────
# Edit this file to add companies, keywords, or tune request settings.

# Role keywords used to flag "target" jobs in the scraped data.
# Matching is case-insensitive substring match on job title.
TARGET_KEYWORDS = {"data engineer", "data analyst", "data scientist"}

# Greenhouse ATS board slugs for companies to scrape.
# Add more slugs here — Greenhouse API requires no auth.
COMPANIES = [
    "stripe",
    "airbnb",
    "lyft",
    "pinterest",
    "robinhood",
    "coinbase",
    "databricks",
    "snowflake",
    "figma",
    "notion",
    "airtable",
    "brex",
    "rippling",
    "gusto",
    "lattice",
]

# Human-readable canonical names for each slug (used for display + PERM matching)
COMPANY_DISPLAY_NAMES = {
    "stripe":      "Stripe",
    "airbnb":      "Airbnb",
    "lyft":        "Lyft",
    "pinterest":   "Pinterest",
    "robinhood":   "Robinhood",
    "coinbase":    "Coinbase",
    "databricks":  "Databricks",
    "snowflake":   "Snowflake",
    "figma":       "Figma",
    "notion":      "Notion",
    "airtable":    "Airtable",
    "brex":        "Brex",
    "rippling":    "Rippling",
    "gusto":       "Gusto",
    "lattice":     "Lattice",
}

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
