#!/usr/bin/env bash
# scripts/daily_refresh.sh
# Runs WS1 (job scraper) then WS2 (sponsorship builder) with basic error checking.
# Intended to be called by cron — see scripts/setup_cron.sh to install.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

PYTHON="${PYTHON:-python3}"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

log() { echo "[$TIMESTAMP] $*"; }

log "=== Daily refresh started ==="

log "--- WS1: Scraping jobs ---"
if ! "$PYTHON" -m pipeline.ws1_run_scraper; then
    log "ERROR: WS1 failed — aborting refresh"
    exit 1
fi

log "--- WS2: Building sponsorship table ---"
if ! "$PYTHON" -m pipeline.ws2_build_sponsorship; then
    log "ERROR: WS2 failed"
    exit 1
fi

log "=== Daily refresh complete ==="
