#!/usr/bin/env bash
# scripts/setup_cron.sh
# Installs the daily refresh cron job into the current user's crontab.
# Safe to run multiple times — won't add duplicate entries.
#
# Usage:
#   bash scripts/setup_cron.sh
#   bash scripts/setup_cron.sh --remove   # removes the installed job

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PROJECT_ROOT/scripts/daily_refresh.sh"
LOG="$PROJECT_ROOT/logs/daily_refresh.log"
MARKER="# ADSFinal_VisaSystem daily refresh"

CRON_LINE="0 14 * * * cd \"$PROJECT_ROOT\" && bash \"$SCRIPT\" >> \"$LOG\" 2>&1  $MARKER"

remove_job() {
    crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
    echo "Removed cron job."
}

install_job() {
    chmod +x "$SCRIPT"
    mkdir -p "$PROJECT_ROOT/logs"

    # Check for existing entry
    if crontab -l 2>/dev/null | grep -qF "$MARKER"; then
        echo "Cron job already installed — run with --remove to uninstall first."
        exit 0
    fi

    # Append to existing crontab
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Cron job installed. Runs daily at 09:00 EST (14:00 UTC)."
    echo "Logs → $LOG"
    echo ""
    echo "Current crontab:"
    crontab -l
}

case "${1:-}" in
    --remove) remove_job ;;
    *)        install_job ;;
esac
