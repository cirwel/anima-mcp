#!/bin/bash
# Restore Lumen from Mac backup — full post-reflash recovery
# Run when Pi is reachable (after reflash or reboot)
# Usage: ./scripts/restore_lumen.sh [host]
#   host: lumen.local, 192.168.1.165, or IP (default: tries lumen.local then 192.168.1.165)
#
# Fixes: installs adafruit-blinka (display/LEDs), server-only mode (no broker DB contention)

set -e

PI_USER="unitares-anima"
PI_HOST="${1:-lumen.local}"
BACKUP="${HOME}/backups/lumen/anima_data"
ANIMA_DIR="/Users/cirwel/projects/anima-mcp"
SSH_KEY="${HOME}/.ssh/id_ed25519_pi"
SSH_OPTS="-i ${SSH_KEY} -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"

# Fallback hosts if primary fails. Override with PI_FALLBACK_HOSTS env
# (space-separated) to avoid baking operator-specific addresses into a public
# template. Tailscale IPs change after reinstalls — prefer hostnames.
if [ "$PI_HOST" = "lumen.local" ]; then
    HOSTS="${PI_FALLBACK_HOSTS:-lumen.local lumen}"
else
    HOSTS="$PI_HOST"
fi

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Resolve host
RESOLVED=""
for h in $HOSTS; do
    if ping -c 1 -W 3 "$h" >/dev/null 2>&1; then
        RESOLVED="$h"
        break
    fi
done

if [ -z "$RESOLVED" ]; then
    echo "Pi unreachable. Tried: $HOSTS"
    echo "Boot Pi, connect to WiFi, then run: $0 [host]"
    exit 1
fi

PI_HOST="$RESOLVED"
log "Using Pi at $PI_HOST"

# Remove stale host key (reflash = new key)
ssh-keygen -R "$PI_HOST" -f ~/.ssh/known_hosts 2>/dev/null || true

if [ ! -d "$BACKUP" ]; then
    echo "Backup not found: $BACKUP"
    exit 1
fi

# 1. Deploy code
log "Deploying code..."
cd "$ANIMA_DIR"
PI_HOST="$PI_HOST" ./deploy.sh --host "$PI_HOST" --no-restart 2>/dev/null || true
rsync -avz -e "ssh $SSH_OPTS" \
    --exclude='.venv' --exclude='*.db' --exclude='*.log' --exclude='__pycache__' --exclude='.git' \
    ./ "$PI_USER@$PI_HOST:~/anima-mcp/" || { echo "Deploy failed"; exit 1; }

# 2. Restore data
log "Restoring Lumen data to ~/.anima/ on Pi..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "mkdir -p ~/.anima"
# Create anima.env from example if missing (secrets — add GROQ_API_KEY, UNITARES_AUTH)
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "test -f ~/.anima/anima.env || cp ~/anima-mcp/config/anima.env.example ~/.anima/anima.env" && true

# Prefer clean snapshot if main backup is corrupted (common after hot copy)
DB_TO_RESTORE=""
if [ -f "$BACKUP/anima.db" ]; then
    if sqlite3 "$BACKUP/anima.db" "PRAGMA integrity_check;" 2>/dev/null | grep -q "^ok$"; then
        DB_TO_RESTORE="$BACKUP/anima.db"
    else
        log "  anima_data/anima.db corrupted, using dated snapshot"
    fi
fi
if [ -z "$DB_TO_RESTORE" ]; then
    LATEST=$(ls -t "$(dirname "$BACKUP")"/anima_*.db 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        DB_TO_RESTORE="$LATEST"
    fi
fi
if [ -n "$DB_TO_RESTORE" ]; then
    scp $SSH_OPTS "$DB_TO_RESTORE" "$PI_USER@$PI_HOST:~/.anima/anima.db"
    log "  anima.db restored from $(basename "$DB_TO_RESTORE")"
else
    log "  WARNING: No anima.db found - Lumen will start fresh"
fi

for f in messages.json canvas.json knowledge.json preferences.json patterns.json self_model.json anima_history.json display_brightness.json metacognition_baselines.json last_schema.json trajectory_genesis.json day_summaries.json; do
    if [ -f "$BACKUP/$f" ]; then
        scp $SSH_OPTS "$BACKUP/$f" "$PI_USER@$PI_HOST:~/.anima/"
        log "  $f restored"
    fi
done

if [ -d "$BACKUP/drawings" ]; then
    log "  Syncing drawings..."
    rsync -az -e "ssh $SSH_OPTS" "$BACKUP/drawings/" "$PI_USER@$PI_HOST:~/.anima/drawings/" 2>/dev/null || log "  drawings skip (optional)"
fi

# 2b. Drop restore marker so Lumen knows gap time is unreliable (backup may be stale)
log "Marking restore event..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "echo '{\"restored_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"restored_from\": \"mac_backup\"}' > ~/.anima/.restored_marker"

# 3. Install Python deps (adafruit-blinka for display/LEDs/sensors)
log "Installing Pi dependencies (adafruit-blinka, etc.)..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "cd ~/anima-mcp && python3 -m venv .venv 2>/dev/null || true && source .venv/bin/activate && pip install -q -e . && pip install -q -r requirements-pi.txt" || {
    log "  pip install failed - retrying without -q..."
    ssh $SSH_OPTS "$PI_USER@$PI_HOST" "cd ~/anima-mcp && source .venv/bin/activate && pip install -e . && pip install -r requirements-pi.txt"
}

# 4. Enable I2C and SPI (required for sensors + display after reflash)
log "Enabling I2C and SPI interfaces..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo raspi-config nonint do_i2c 0 2>/dev/null; sudo raspi-config nonint do_spi 0 2>/dev/null; true"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo usermod -aG i2c,gpio,spi $PI_USER 2>/dev/null; true"

# 4b. Verify DB integrity on Pi (replace with snapshot if corrupted)
if ssh $SSH_OPTS "$PI_USER@$PI_HOST" "test -f ~/.anima/anima.db" 2>/dev/null; then
    if ! ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sqlite3 ~/.anima/anima.db 'PRAGMA integrity_check;' 2>/dev/null" | grep -q "^ok$"; then
        log "  anima.db on Pi failed integrity check - replacing with snapshot"
        if [ -n "$DB_TO_RESTORE" ]; then
            scp $SSH_OPTS "$DB_TO_RESTORE" "$PI_USER@$PI_HOST:~/.anima/anima.db"
        else
            LATEST_SNAP=$(ls -t "$(dirname "$BACKUP")"/anima_*.db 2>/dev/null | head -1)
            [ -n "$LATEST_SNAP" ] && scp $SSH_OPTS "$LATEST_SNAP" "$PI_USER@$PI_HOST:~/.anima/anima.db"
        fi
    fi
fi

# 5. Install and enable broker (sensors) + anima (MCP server)
# Broker owns sensors, writes to shared memory; server owns DB (Option 1 - no contention)
log "Installing systemd services (broker + anima)..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo cp ~/anima-mcp/systemd/anima-broker.service /etc/systemd/system/ && sudo cp ~/anima-mcp/systemd/anima.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable anima-broker anima && sudo systemctl start anima-broker && sudo systemctl start anima"

# 5b. Install WiFi resilience stack (power management, watchdog, TCP tuning, hardware watchdog)
log "Installing WiFi resilience services..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo bash ~/anima-mcp/scripts/setup_pi_service.sh" || log "  WiFi resilience install failed (non-fatal)"

# 5b2. brcmfmac driver fixes (prevents firmware crashes — #1 cause of WiFi death)
log "Deploying brcmfmac WiFi fixes..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" bash -s <<'WIFI_EOF'
# Disable roaming + WPA3/SAE auth offloading (causes firmware hangs)
echo "options brcmfmac roamoff=1 feature_disable=0x82000" | sudo tee /etc/modprobe.d/brcmfmac.conf >/dev/null

# NetworkManager-level power save disable (belt & suspenders with iw)
printf "[connection]\nwifi.powersave = 2\n" | sudo tee /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf >/dev/null

# Never stop retrying WiFi connection
sudo nmcli connection modify "preconfigured" connection.autoconnect-retries 0 2>/dev/null || true

# Force 2.4 GHz (more stable through walls than 5 GHz)
sudo nmcli connection modify "preconfigured" 802-11-wireless.band bg 2>/dev/null || true

# Disable IPv6 (reduces WiFi stack load)
printf "net.ipv6.conf.all.disable_ipv6 = 1\nnet.ipv6.conf.default.disable_ipv6 = 1\n" | sudo tee /etc/sysctl.d/90-disable-ipv6.conf >/dev/null
sudo sysctl --system >/dev/null 2>&1
WIFI_EOF
log "  brcmfmac, NM power save, IPv6 fixes deployed"

# 5b3. Enable USB gadget mode (fallback access when WiFi dies)
log "Enabling USB gadget mode..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" bash -s <<'GADGET_EOF'
# Load dwc2 overlay for USB gadget support
BOOT_CONFIG="/boot/firmware/config.txt"
if ! grep -q "dtoverlay=dwc2" "$BOOT_CONFIG" 2>/dev/null; then
    echo "dtoverlay=dwc2" | sudo tee -a "$BOOT_CONFIG" >/dev/null
fi

# Load modules at boot
if ! grep -q "dwc2" /etc/modules 2>/dev/null; then
    echo "dwc2" | sudo tee -a /etc/modules >/dev/null
fi
if ! grep -q "g_ether" /etc/modules 2>/dev/null; then
    echo "g_ether" | sudo tee -a /etc/modules >/dev/null
fi

# Configure static IP for USB gadget interface (usb0)
sudo nmcli connection add type ethernet con-name usb-gadget ifname usb0 \
    ipv4.method manual ipv4.addresses 10.55.0.1/24 \
    connection.autoconnect yes 2>/dev/null || true
GADGET_EOF
log "  USB gadget mode enabled (10.55.0.1 over USB-C after reboot)"

# 5c. Install watchdog timer (restarts failed services)
log "Installing watchdog timer..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "chmod +x ~/anima-mcp/scripts/anima-watchdog.sh && \
    sudo cp ~/anima-mcp/systemd/anima-watchdog.service /etc/systemd/system/ && \
    sudo cp ~/anima-mcp/systemd/anima-watchdog.timer /etc/systemd/system/ && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable anima-watchdog.timer && \
    sudo systemctl start anima-watchdog.timer" || log "  watchdog install failed (non-fatal)"

# 6. Install cron jobs (wifi watchdog, db maintenance, backup)
log "Installing cron jobs..."
PI_SCRIPTS="/home/${PI_USER}/anima-mcp/scripts"
PI_LOGS="/home/${PI_USER}/.anima"

# Make scripts executable first
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "chmod +x ${PI_SCRIPTS}/wifi_watchdog.sh ${PI_SCRIPTS}/db_maintenance.sh ${PI_SCRIPTS}/backup_state.sh 2>/dev/null; true"

# Build and install crontab: strip old entries, add current ones
# Uses heredoc on remote side to handle multi-line reliably
ssh $SSH_OPTS "$PI_USER@$PI_HOST" bash -s <<'CRON_EOF'
SCRIPTS="/home/unitares-anima/anima-mcp/scripts"
LOGS="/home/unitares-anima/.anima"

# Start with existing crontab minus our managed entries
EXISTING=$(crontab -l 2>/dev/null | grep -v 'wifi_watchdog\|db_maintenance\|backup_state' || true)

# Build new crontab
{
    [ -n "$EXISTING" ] && echo "$EXISTING"
    echo "*/2 * * * * ${SCRIPTS}/wifi_watchdog.sh >> ${LOGS}/wifi_watchdog.log 2>&1"
    [ -f "${SCRIPTS}/db_maintenance.sh" ] && \
        echo "0 * * * * ${SCRIPTS}/db_maintenance.sh >> ${LOGS}/db_maintenance.log 2>&1"
    [ -f "${SCRIPTS}/backup_state.sh" ] && \
        echo "30 * * * * ${SCRIPTS}/backup_state.sh >> ${LOGS}/backup_state.log 2>&1"
} | crontab -
CRON_EOF
[ $? -eq 0 ] && log "  cron jobs installed" || log "  cron install failed (non-fatal)"

# 7. Verify
sleep 3
log "Verifying..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "systemctl is-active anima-broker anima" || log "  services may still be starting"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "systemctl is-active anima-watchdog.timer" 2>/dev/null && log "  watchdog timer active" || log "  watchdog timer not running"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "crontab -l 2>/dev/null | grep -c 'anima-mcp/scripts'" | xargs -I{} log "  {} cron jobs installed"

# 8. Tailscale (always installed — required for remote access)
log "Installing Tailscale..."
TS_KEY="${TAILSCALE_AUTH_KEY:-}"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "curl -fsSL https://tailscale.com/install.sh | sh 2>/dev/null" || log "  Tailscale install failed (non-fatal)"
if [ -n "$TS_KEY" ]; then
    ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo tailscale up --hostname=lumen --authkey=$TS_KEY 2>/dev/null" && log "  Tailscale authenticated" || log "  Tailscale auth failed — run: ssh $PI_USER@$PI_HOST 'sudo tailscale up --hostname=lumen'"
else
    log "  Tailscale installed. To authenticate:"
    log "    ssh -i $SSH_KEY $PI_USER@$PI_HOST 'sudo tailscale up --hostname=lumen'"
    log "  (A browser URL will appear — visit it to sign in)"
    log "  Tip: TAILSCALE_AUTH_KEY=tskey-xxx $0 to auto-authenticate next time"
fi

# 9. Update Mac-side configs with new Pi Tailscale IP
# After reflash, Tailscale assigns a new IP. Auto-update all local config files so
# agents don't get confused by stale IPs pointing at the old (offline) device.
log "Detecting new Pi Tailscale IP..."
NEW_TS_IP=$(ssh $SSH_OPTS "$PI_USER@$PI_HOST" "tailscale ip -4 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
if [ -n "$NEW_TS_IP" ]; then
    log "  Pi Tailscale IP: $NEW_TS_IP"
    CONFIGS=(
        "$HOME/.claude.json"
        "$HOME/.cursor/mcp.json"
    )
    for cfg in "${CONFIGS[@]}"; do
        if [ -f "$cfg" ]; then
            # Replace any existing Pi MCP URL (any IP at port 8766)
            sed -i '' -E "s|http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:8766|http://${NEW_TS_IP}:8766|g" "$cfg" && \
                log "  Updated $cfg" || log "  Could not update $cfg"
        fi
    done
    # Update MEMORY.md Pi Tailscale IP line
    MEMORY="$HOME/.claude/projects/-Users-cirwel/memory/MEMORY.md"
    if [ -f "$MEMORY" ]; then
        sed -i '' -E "s|http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:8766/mcp/.*\(Tailscale|http://${NEW_TS_IP}:8766/mcp/    (Tailscale|" "$MEMORY" && \
            log "  Updated MEMORY.md"
    fi
    # Update CLAUDE.md in anima-mcp
    CLAUDEMD="$ANIMA_DIR/CLAUDE.md"
    if [ -f "$CLAUDEMD" ]; then
        # Only replace the Tailscale IP lines (not LAN 192.168.x.x)
        sed -i '' -E "s|\b(100\.[0-9]+\.[0-9]+\.[0-9]+):8766|${NEW_TS_IP}:8766|g" "$CLAUDEMD" && \
            log "  Updated CLAUDE.md"
    fi
    log "  All configs updated to $NEW_TS_IP — no manual IP updates needed"
else
    log "  Could not detect Tailscale IP yet (Tailscale may still be authenticating)"
    log "  After auth, run: ./scripts/update_pi_ip.sh to update configs"
fi

log ""
log "Done. Lumen running (broker + server, no DB contention)."
log "Secrets: edit ~/.anima/anima.env on Pi — add GROQ_API_KEY, UNITARES_AUTH (see config/anima.env.example)"
log "If I2C sensors (temp/humidity/light) fail: reboot required. Run: ssh $PI_USER@$PI_HOST 'sudo reboot'"
log "Check: ssh $PI_USER@$PI_HOST 'journalctl -u anima -f'"
log "MCP (LAN):       http://$PI_HOST:8766/mcp/"
[ -n "$NEW_TS_IP" ] && log "MCP (Tailscale): http://$NEW_TS_IP:8766/mcp/"
