#!/bin/bash
# Update all Mac-side configs with the current Pi Tailscale IP.
# Run this after reflash + Tailscale auth if restore_lumen.sh couldn't detect the IP yet.
# Usage: ./scripts/update_pi_ip.sh [pi-host]

PI_USER="unitares-anima"
PI_HOST="${1:-lumen.local}"
SSH_KEY="${HOME}/.ssh/id_ed25519_pi"
SSH_OPTS="-i ${SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
ANIMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

NEW_TS_IP=$(ssh $SSH_OPTS "$PI_USER@$PI_HOST" "tailscale ip -4 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
if [ -z "$NEW_TS_IP" ]; then
    echo "Could not get Tailscale IP from Pi at $PI_HOST"
    echo "Is Tailscale authenticated? Run: ssh -i $SSH_KEY $PI_USER@$PI_HOST 'sudo tailscale up --hostname=lumen'"
    exit 1
fi

log "Pi Tailscale IP: $NEW_TS_IP"

CONFIGS=(
    "$HOME/.claude.json"
    "$HOME/.cursor/mcp.json"
)
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$cfg" ]; then
        sed -i '' -E "s|http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:8766|http://${NEW_TS_IP}:8766|g" "$cfg" && \
            log "Updated $cfg" || log "Could not update $cfg"
    fi
done

# Update SSH config (lumen host entries)
SSH_CFG="$HOME/.ssh/config"
if [ -f "$SSH_CFG" ]; then
    # Replace old Tailscale IP in Host line and HostName for lumen entries
    sed -i '' -E "/^Host lumen lumen-tailscale/{ s|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|${NEW_TS_IP}|g; }" "$SSH_CFG"
    sed -i '' -E "/^Host lumen lumen-tailscale/{ n; s|HostName [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|HostName ${NEW_TS_IP}|; }" "$SSH_CFG" && \
        log "Updated $SSH_CFG" || log "Could not update $SSH_CFG"
fi

MEMORY="$HOME/.claude/projects/-Users-cirwel/memory/MEMORY.md"
if [ -f "$MEMORY" ]; then
    sed -i '' -E "s|http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:8766/mcp/.*\(Tailscale|http://${NEW_TS_IP}:8766/mcp/    (Tailscale|" "$MEMORY" && \
        log "Updated MEMORY.md"
fi

CLAUDEMD="$ANIMA_DIR/CLAUDE.md"
if [ -f "$CLAUDEMD" ]; then
    sed -i '' -E "s|\b(100\.[0-9]+\.[0-9]+\.[0-9]+):8766|${NEW_TS_IP}:8766|g" "$CLAUDEMD" && \
        log "Updated CLAUDE.md"
fi

log "Done. All configs point to $NEW_TS_IP:8766"
log "Restart Claude Code / Cursor to pick up new MCP URL."
