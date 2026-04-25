#!/bin/bash
# Activate Tailscale via HTTP — for headless Pi over LAN HTTP
#
# Prerequisites:
#   1. Pi's HTTP (8766) is reachable: curl -s http://192.168.1.165:8766/health
#   2. Code with setup_tailscale is pushed to git
#   3. Get auth key: https://login.tailscale.com/admin/settings/keys (reusable, 90 days)
#
# Usage: TAILSCALE_AUTH_KEY=tskey-auth-xxx ./scripts/setup_tailscale_via_http.sh
#   Or: ./scripts/setup_tailscale_via_http.sh tskey-auth-xxx

set -e

PI_URL="${LUMEN_URL:-http://192.168.1.165:8766}"
AUTH_KEY="${TAILSCALE_AUTH_KEY:-$1}"

if [ -z "$AUTH_KEY" ]; then
    echo "=== Tailscale via HTTP ==="
    echo ""
    echo "Usage: TAILSCALE_AUTH_KEY=tskey-auth-xxx ./scripts/setup_tailscale_via_http.sh"
    echo "   Or: ./scripts/setup_tailscale_via_http.sh tskey-auth-xxx"
    echo ""
    echo "Get auth key: https://login.tailscale.com/admin/settings/keys"
    echo "Pi URL: $PI_URL"
    exit 1
fi

echo "=== Activating Tailscale via HTTP ==="
echo "Pi URL: $PI_URL"
echo ""

call_tool() {
    local name="$1"
    local args="$2"
    curl -s -X POST "$PI_URL/v1/tools/call" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$name\", \"arguments\": $args}" \
        --connect-timeout 10 --max-time 120
}

result=$(call_tool "setup_tailscale" "{\"auth_key\": \"$AUTH_KEY\"}" 2>/dev/null || true)

if echo "$result" | grep -q '"success":true'; then
    echo "   ✅ Tailscale active!"
    echo ""
    echo "$result" | python3 -m json.tool 2>/dev/null || echo "$result"
    echo ""
    echo "Update Cursor MCP (~/.cursor/mcp.json) with tailscale_ip from above."
else
    echo "   Response: $result"
    echo ""
    echo "If setup_tailscale not found: push this repo to git, ensure anima.service restarted."
fi
