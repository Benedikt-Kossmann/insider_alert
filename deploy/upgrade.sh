#!/usr/bin/env bash
# deploy/upgrade.sh — Upgrade insider_alert on the production server
# Usage: sudo bash deploy/upgrade.sh
# Run from /opt/insider_alert (or the script detects WORKDIR automatically)
set -euo pipefail

WORKDIR="/opt/insider_alert"
VENV="$WORKDIR/.venv"
SERVICE="insider-alert"

cd "$WORKDIR"

echo "=== insider_alert upgrade ==="
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
git pull --ff-only
echo ""

# 2. Check if requirements changed → install deps
echo "[2/4] Checking dependencies..."
if git diff HEAD~1 --name-only 2>/dev/null | grep -qE 'requirements\.txt'; then
    echo "  → requirements.txt changed, installing dependencies..."
    "$VENV/bin/pip" install -r requirements.txt --quiet
    echo "  → Done."
else
    echo "  → No dependency changes."
fi
echo ""

# 3. Restart service
echo "[3/4] Restarting $SERVICE..."
systemctl restart "$SERVICE"
echo ""

# 4. Check status
echo "[4/4] Service status:"
systemctl status "$SERVICE" --no-pager -l | head -20
echo ""
echo "=== Recent logs (last 15 lines) ==="
journalctl -u "$SERVICE" --no-pager -n 15
echo ""
echo "=== Upgrade complete ==="
