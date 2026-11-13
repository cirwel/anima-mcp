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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANIMA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_ROOT="${LUMEN_BACKUP_DIR:-${HOME}/backups/lumen}"
BACKUP="${BACKUP_ROOT}/anima_data"
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

sqlite_ok() {
    python3 - "$1" <<'PY'
import sqlite3
import sys

try:
    with sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True) as connection:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
except sqlite3.Error:
    raise SystemExit(1)
raise SystemExit(0 if rows == [("ok",)] else 1)
PY
}

json_ok() {
    python3 - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        json.load(handle)
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

# Resolve host
RESOLVED=""
for h in $HOSTS; do
    # ICMP can be blocked while SSH is healthy. Probe the service without
    # trusting/storing a possibly stale post-reflash host key; the selected
    # host is enrolled normally immediately below.
    if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "$PI_USER@$h" true >/dev/null 2>&1; then
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
ssh-keygen -R "$PI_HOST" -f "$HOME/.ssh/known_hosts" 2>/dev/null || true

if [ ! -d "$BACKUP" ]; then
    if [ -d "${BACKUP}.previous" ]; then
        BACKUP="${BACKUP}.previous"
        log "Primary learned-state mirror absent; using preserved previous generation"
    else
        echo "Backup not found: $BACKUP"
        exit 1
    fi
fi

# Validate a complete recovery set before stopping services or replacing code.
for required in anima_config.json preferences.json self_model.json patterns.json metacognition_baselines.json; do
    if [ ! -f "$BACKUP/$required" ]; then
        log "ERROR: learned-state recovery point is missing $required"
        exit 1
    fi
done
for source in "$BACKUP"/*.json; do
    [ -e "$source" ] || break
    if ! json_ok "$source"; then
        log "ERROR: invalid learned-state JSON: $(basename "$source")"
        exit 1
    fi
done
if [ -e "$BACKUP/oauth.db" ] && ! sqlite_ok "$BACKUP/oauth.db"; then
    log "ERROR: OAuth persistence database failed integrity_check"
    exit 1
fi

# Prefer the newest locally verified snapshot. Python's sqlite3 module is
# available everywhere this service runs; the sqlite3 CLI is not installed on
# a fresh Pi and must not be a hidden restore dependency.
DB_TO_RESTORE=""
if [ -f "$BACKUP/anima.db" ]; then
    if sqlite_ok "$BACKUP/anima.db"; then
        DB_TO_RESTORE="$BACKUP/anima.db"
    else
        log "  anima_data/anima.db corrupted, using dated snapshot"
    fi
fi
if [ -z "$DB_TO_RESTORE" ]; then
    while IFS= read -r candidate; do
        if sqlite_ok "$candidate"; then
            DB_TO_RESTORE="$candidate"
            break
        fi
        log "  skipping corrupt snapshot: $(basename "$candidate")"
    done < <(ls -t "$BACKUP_ROOT"/anima_*.db 2>/dev/null || true)
fi
if [ -z "$DB_TO_RESTORE" ]; then
    log "  ERROR: no verified anima.db recovery point found; refusing a silent fresh identity"
    exit 1
fi

# Freeze the mutable nightly mirror and selected DB into one local staging
# generation. A launchd backup can publish a new anima_data directory while a
# restore is running; the Pi must receive exactly the files validated here.
RESTORE_STAGE=$(mktemp -d "${TMPDIR:-/tmp}/lumen-restore.XXXXXX") || {
    log "ERROR: could not allocate local restore staging directory"
    exit 1
}
cleanup_restore_stage() { rm -rf "$RESTORE_STAGE"; }
trap cleanup_restore_stage EXIT
mkdir -p "$RESTORE_STAGE/anima_data"
rsync -a --exclude='*.db*' --exclude='anima.env*' \
    "$BACKUP/" "$RESTORE_STAGE/anima_data/" || {
        log "ERROR: could not freeze learned-state recovery generation"
        exit 1
    }
if [ -e "$BACKUP/oauth.db" ]; then
    cp "$BACKUP/oauth.db" "$RESTORE_STAGE/anima_data/oauth.db" || {
        log "ERROR: could not stage OAuth persistence database"
        exit 1
    }
    sqlite_ok "$RESTORE_STAGE/anima_data/oauth.db" || {
        log "ERROR: staged OAuth database failed integrity_check"
        exit 1
    }
fi
cp "$DB_TO_RESTORE" "$RESTORE_STAGE/anima.db" || {
    log "ERROR: could not stage verified anima.db"
    exit 1
}
sqlite_ok "$RESTORE_STAGE/anima.db" || {
    log "ERROR: staged anima.db failed integrity_check"
    exit 1
}
for required in anima_config.json preferences.json self_model.json patterns.json metacognition_baselines.json; do
    [ -f "$RESTORE_STAGE/anima_data/$required" ] || {
        log "ERROR: staged learned state lost $required"
        exit 1
    }
done
for source in "$RESTORE_STAGE/anima_data"/*.json; do
    [ -e "$source" ] || break
    json_ok "$source" || {
        log "ERROR: staged learned-state JSON is invalid: $(basename "$source")"
        exit 1
    }
done
BACKUP="$RESTORE_STAGE/anima_data"
DB_TO_RESTORE="$RESTORE_STAGE/anima.db"
log "Recovery set verified and frozen before mutation: DB + learned state"

# Stop every persistent-state writer before changing code or restoring files.
# On a fresh reflash the units may not exist yet; that is safely inactive.
log "Quiescing Lumen before restore..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" '
    sudo systemctl stop anima anima-broker 2>/dev/null || true
    for unit in anima anima-broker; do
        state=$(systemctl is-active "$unit" 2>/dev/null || true)
        if [ "$state" = active ] || [ "$state" = activating ]; then
            echo "$unit failed to stop" >&2
            exit 1
        fi
    done
' || { echo "Could not quiesce Lumen; restore aborted"; exit 1; }

# 1. Deploy code. Restore already owns the explicit no-backup exception: the
# source state is the selected recovery point, not the device being rebuilt.
log "Deploying code..."
cd "$ANIMA_DIR"
PI_HOST="$PI_HOST" PI_USER="$PI_USER" SSH_KEY="$SSH_KEY" \
    ./deploy.sh --host "$PI_HOST" --no-restart --skip-backup || {
        echo "Deploy failed; services remain stopped"
        exit 1
    }

# 2. Restore data
log "Restoring Lumen data to ~/.anima/ on Pi..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "mkdir -p ~/.anima"
# Create anima.env from example if missing. The template ships every key EMPTY
# — anima.env is excluded from backups because it holds secrets, so a reflash
# always lands here with no values. The post-restore check below reports which
# ones are still blank rather than letting them be discovered later.
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "test -f ~/.anima/anima.env || cp ~/anima-mcp/config/anima.env.example ~/.anima/anima.env" && true

# Upload every recovery component under temporary names first. Nothing below
# becomes authoritative until the remote validation succeeds and anima.db is
# moved last as the transaction's commit point.
ssh $SSH_OPTS "$PI_USER@$PI_HOST" \
    "rm -rf ~/.anima/.restore-state && mkdir -p ~/.anima/.restore-state/json ~/.anima/.restore-state/learning_inbox ~/.anima/.restore-state/drawings"
scp $SSH_OPTS "$DB_TO_RESTORE" "$PI_USER@$PI_HOST:~/.anima/.restore-anima.db"
if [ -e "$BACKUP/oauth.db" ]; then
    scp $SSH_OPTS "$BACKUP/oauth.db" \
        "$PI_USER@$PI_HOST:~/.anima/.restore-state/oauth.db"
fi

for source in "$BACKUP"/*.json; do
    [ -e "$source" ] || break
    f=$(basename "$source")
    case "$f" in
        *[!A-Za-z0-9._-]*)
            log "  ERROR: unsafe learned-state filename: $f"
            exit 1
            ;;
    esac
    scp $SSH_OPTS "$source" "$PI_USER@$PI_HOST:~/.anima/.restore-state/json/$f"
done

if [ -d "$BACKUP/learning_inbox" ]; then
    log "  Staging learning inbox..."
    rsync -az --delete -e "ssh $SSH_OPTS" \
        "$BACKUP/learning_inbox/" "$PI_USER@$PI_HOST:~/.anima/.restore-state/learning_inbox/"
fi

if [ -d "$BACKUP/drawings" ]; then
    log "  Staging drawings..."
    rsync -az --delete -e "ssh $SSH_OPTS" \
        "$BACKUP/drawings/" "$PI_USER@$PI_HOST:~/.anima/.restore-state/drawings/"
fi

log "  Validating and committing staged identity state..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" bash -s <<'RESTORE_COMMIT'
set -e
ANIMA="$HOME/.anima"
STATE="$ANIMA/.restore-state"
DB_STAGE="$ANIMA/.restore-anima.db"

python3 - "$STATE/json" "$DB_STAGE" <<'PY'
import json
import os
import sqlite3
import sys
from pathlib import Path

json_root = Path(sys.argv[1])
db_path = Path(sys.argv[2])
oauth_path = json_root.parent / "oauth.db"
required = {
    "anima_config.json",
    "preferences.json",
    "self_model.json",
    "patterns.json",
    "metacognition_baselines.json",
}
missing = sorted(name for name in required if not (json_root / name).is_file())
if missing:
    print("remote learned-state stage missing: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
json_paths = list(json_root.glob("*.json"))
try:
    for path in json_paths:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
except (OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error) as exc:
    print(f"remote restore validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
if rows != [("ok",)]:
    print(f"remote DB integrity_check failed: {rows[:3]}", file=sys.stderr)
    raise SystemExit(1)
if oauth_path.exists():
    try:
        with sqlite3.connect(f"file:{oauth_path}?mode=ro", uri=True) as connection:
            oauth_rows = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error as exc:
        print(f"remote OAuth validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if oauth_rows != [("ok",)]:
        print(f"remote OAuth integrity_check failed: {oauth_rows[:3]}", file=sys.stderr)
        raise SystemExit(1)
durable_paths = [db_path]
durable_paths.extend(path for path in json_root.parent.rglob("*") if path.is_file())
for path in durable_paths:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
PY

# Remove the DB commit point before changing the learned snapshot. If power is
# lost in this window, the boot continuity gate sees a missing DB and retries a
# complete restore rather than accepting a hybrid state.
rm -f "$ANIMA/.pre-restore-anima.db"
if [ -e "$ANIMA/anima.db" ]; then
    mv "$ANIMA/anima.db" "$ANIMA/.pre-restore-anima.db"
fi
rm -f "$ANIMA/anima.db-wal" "$ANIMA/anima.db-shm"

find "$ANIMA" -maxdepth 1 -type f -name '*.json' -delete
mv "$STATE"/json/*.json "$ANIMA/"
rm -rf "$ANIMA/learning_inbox" "$ANIMA/drawings"
mv "$STATE/learning_inbox" "$ANIMA/learning_inbox"
mv "$STATE/drawings" "$ANIMA/drawings"
rm -f "$ANIMA/oauth.db" "$ANIMA/oauth.db-wal" "$ANIMA/oauth.db-shm"
if [ -e "$STATE/oauth.db" ]; then
    mv "$STATE/oauth.db" "$ANIMA/oauth.db"
    chmod 600 "$ANIMA/oauth.db"
fi

# Final commit point.
mv "$DB_STAGE" "$ANIMA/anima.db"
python3 - "$ANIMA" <<'PY'
import os
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
db = root / "anima.db"
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
    rows = connection.execute("PRAGMA integrity_check").fetchall()
if rows != [("ok",)]:
    raise SystemExit(1)
with db.open("rb") as handle:
    os.fsync(handle.fileno())
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
rm -f "$ANIMA/.pre-restore-anima.db"
rm -rf "$STATE"
RESTORE_COMMIT
log "  anima.db and learned state committed from frozen recovery generation"

# The retired broker agency store — see #123. The server is the sole active
# action learner; this separate table is restored only as rollback/history
# state and remains unused unless ANIMA_BROKER_AGENCY_ENABLED=true is set.
#
# Deliberately NOT matched by the anima_*.db glob above: it is a ~100KB agency
# store and must never be selectable as the 240MB main database.
AGENCY_BACKUP=$(ls -t "$BACKUP_ROOT"/agency_*.db 2>/dev/null | head -1)
if [ -n "$AGENCY_BACKUP" ] && sqlite_ok "$AGENCY_BACKUP"; then
    scp $SSH_OPTS "$AGENCY_BACKUP" "$PI_USER@$PI_HOST:~/anima-mcp/.restore-agency.db"
    ssh $SSH_OPTS "$PI_USER@$PI_HOST" \
        "mv ~/anima-mcp/.restore-agency.db ~/anima-mcp/anima.db && rm -f ~/anima-mcp/anima.db-wal ~/anima-mcp/anima.db-shm"
    log "  broker agency store restored from $(basename "$AGENCY_BACKUP") [#123]"
else
    log "  NOTE: no agency_*.db rollback history found (#123)"
fi

# 2b. Drop restore marker so Lumen knows gap time is unreliable (backup may be stale)
log "Marking restore event..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path.home() / ".anima" / ".restored_marker"
temporary = path.with_name(path.name + ".tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump({
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "restored_from": "mac_backup",
    }, handle)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

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

# 4b. Verify the installed DB once more before any service may open it.
REMOTE_DB_CHECK="import sqlite3,sys;p=\"/home/${PI_USER}/.anima/anima.db\";c=sqlite3.connect(p);r=c.execute(\"PRAGMA integrity_check\").fetchall();c.close();sys.exit(0 if r==[(\"ok\",)] else 1)"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "python3 -c '$REMOTE_DB_CHECK'" || {
    log "  ERROR: installed anima.db failed integrity_check; services remain stopped"
    exit 1
}

# 5. Install and enable every configured runtime owner. The deploy above used
# --no-restart while state was being restored, so the Elixir helper now consumes
# any pending release marker and verifies fresh shadow state before the Python
# body and mind start.
log "Installing systemd services (sensor owner + broker + anima)..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "set -e; sudo install -m 0644 ~/anima-mcp/systemd/anima-restore.service /etc/systemd/system/anima-restore.service; sudo install -m 0644 ~/anima-mcp/systemd/anima-broker.service /etc/systemd/system/anima-broker.service; sudo install -m 0644 ~/anima-mcp/systemd/anima.service /etc/systemd/system/anima.service; sudo systemctl daemon-reload; bash ~/anima-mcp/scripts/deploy_elixir_broker.sh; sudo systemctl enable anima-restore anima-broker anima; sudo systemctl start anima-broker; sudo systemctl start anima"

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

# IPv6 must stay ENABLED. Disabling it caused a total internet outage on
# 2026-07-23: the router's IPv4 WAN died (IPv6-only WAN), and the Pi — v6
# disabled — lost pypi, GitHub, and the Tailscale control plane ("logged out"
# for weeks). The WiFi-stack-load win is not worth losing the only WAN path.
sudo rm -f /etc/sysctl.d/90-disable-ipv6.conf
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

# 5d. Install the outbound heartbeat (dead-man's switch to an off-host monitor).
# Every other liveness surface Lumen has runs on the Mac and cannot report that
# the Mac itself is down. This one leaves both machines. It stays inert until
# ANIMA_HEARTBEAT_URL is set in anima.env — which a reflash deliberately does not
# restore, so the pings stop and the provider alerts. See docs/operations/HEARTBEAT.md.
log "Installing outbound heartbeat timer..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "chmod +x ~/anima-mcp/scripts/lumen-heartbeat.sh && \
    sudo cp ~/anima-mcp/systemd/lumen-heartbeat.service /etc/systemd/system/ && \
    sudo cp ~/anima-mcp/systemd/lumen-heartbeat.timer /etc/systemd/system/ && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable lumen-heartbeat.timer && \
    sudo systemctl start lumen-heartbeat.timer" || log "  heartbeat install failed (non-fatal)"

# 5e. Install the journal archive (volatile journald -> ~/.anima -> Mac mirror).
# The Pi's journal lives in RAM and high-churn logging rotates it in under a
# day; without this export a quiet failure investigated later has no
# operational record to read (#188 could not be root-caused for exactly this
# reason). The hourly export is what makes post-hoc forensics possible.
log "Installing journal archive timer..."
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "sudo usermod -aG adm $PI_USER && \
    chmod +x ~/anima-mcp/scripts/journal-archive.sh && \
    sudo cp ~/anima-mcp/systemd/lumen-journal-archive.service /etc/systemd/system/ && \
    sudo cp ~/anima-mcp/systemd/lumen-journal-archive.timer /etc/systemd/system/ && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable lumen-journal-archive.timer && \
    sudo systemctl start lumen-journal-archive.timer" || log "  journal archive install failed (non-fatal)"

# 6. Install scheduled maintenance (wifi watchdog, db maintenance, db + state
# backups) as systemd timers, and retire any crontab lines left by an older
# restore. This used to write a crontab that also scheduled the WiFi watchdog
# already enabled as wifi-watchdog.timer in step 5b, so two copies ran.
log "Installing scheduled jobs (systemd timers)..."
PI_SCRIPTS="/home/${PI_USER}/anima-mcp/scripts"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "chmod +x ${PI_SCRIPTS}/storage_maintenance.py 2>/dev/null; bash ${PI_SCRIPTS}/install_scheduled_jobs.sh" \
    && log "  scheduled jobs installed" || log "  scheduled job install failed (non-fatal)"

# 7. Verify the embodied boundary, not merely that systemd accepted Start=.
log "Verifying..."
VERIFY_RUNTIME_CODE='import json,os,sys,time;p="/dev/shm/anima_state.json";started=time.time();end=started+60
while time.time()<end:
 try:
  envelope=json.load(open(p));data=envelope.get("data",{});fresh=os.path.getmtime(p)>=started
  if fresh and isinstance(data.get("readings"),dict) and isinstance(data.get("anima"),dict):sys.exit(0)
 except Exception:pass
 time.sleep(1)
sys.exit(1)'
VERIFY_SHADOW_CODE='import json,os,sys,time;p="/dev/shm/anima_state.shadow.json";end=time.time()+15
while time.time()<end:
 try:
  envelope=json.load(open(p));data=envelope.get("data",{});fresh=time.time()-os.path.getmtime(p)<15
  if fresh and isinstance(data.get("readings"),dict):sys.exit(0)
 except Exception:pass
 time.sleep(1)
sys.exit(1)'
ssh $SSH_OPTS "$PI_USER@$PI_HOST" \
    "set -e; systemctl is-active --quiet anima-broker; systemctl is-active --quiet anima; systemctl show anima -p Environment --value | tr ' ' '\n' | grep -qx 'ANIMA_SENSORS_BACKEND=shm'; if grep -Eq '^[[:space:]]*ANIMA_ENV_SENSORS_FROM_SHM=[^[:space:]#]+' ~/.anima/anima.env 2>/dev/null; then systemctl is-active --quiet anima-broker-ex; python3 -c '$VERIFY_SHADOW_CODE'; fi; python3 -c '$VERIFY_RUNTIME_CODE'" || {
        log "  ERROR: restored services did not publish fresh broker state"
        exit 1
    }
log "  configured sensor owner + broker + mind active; fresh shared state verified"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "systemctl is-active anima-watchdog.timer" 2>/dev/null && log "  watchdog timer active" || log "  watchdog timer not running"
ssh $SSH_OPTS "$PI_USER@$PI_HOST" "systemctl is-active anima-storage-maintenance.timer" 2>/dev/null && log "  storage maintenance timer active" || log "  storage maintenance timer not running"
for t in wifi-watchdog anima-db-maintenance anima-db-backup anima-state-backup; do
    ssh $SSH_OPTS "$PI_USER@$PI_HOST" "systemctl is-active $t.timer" >/dev/null 2>&1 && log "  $t timer active" || log "  $t timer not running"
done

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

# Secrets never ride along with a restore (anima.env is excluded from backups),
# so report exactly which ones came back empty. Values are never printed — only
# whether each key has one. ANIMA_ADMIN_SECRET is called out separately because
# an empty value now disables the destructive tools instead of ungating them.
MISSING_SECRETS="$(ssh $SSH_OPTS "$PI_USER@$PI_HOST" '
    for k in GROQ_API_KEY UNITARES_AUTH ANIMA_ADMIN_SECRET ANIMA_HEARTBEAT_URL WIFI_SSID WIFI_PASSWORD; do
        v="$(sed -nE "s/^[[:space:]]*${k}=(.*)$/\1/p" ~/.anima/anima.env 2>/dev/null | tail -1)"
        [ -z "$v" ] && printf "%s " "$k"
    done' 2>/dev/null || true)"

if [ -n "${MISSING_SECRETS// /}" ]; then
    log ""
    log "⚠️  UNSET in ~/.anima/anima.env: ${MISSING_SECRETS}"
    log "    Edit it on the Pi, then: sudo systemctl restart anima anima-broker"
    case "$MISSING_SECRETS" in
        *ANIMA_ADMIN_SECRET*)
            log ""
            log "⚠️  ANIMA_ADMIN_SECRET is empty — the six destructive MCP tools"
            log "    (git_pull, deploy_from_github, system_service, system_power,"
            log "    fix_ssh_port, setup_tailscale) will REFUSE to run until it is set."
            log "    This is deliberate: unset means the server cannot authenticate"
            log "    the caller, so the gate fails closed rather than open."
            log "    Generate: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
            log "    Then set the same value in your MCP client's X-Anima-Admin header."
            ;;
    esac
    case "$MISSING_SECRETS" in
        *ANIMA_HEARTBEAT_URL*)
            log ""
            log "⚠️  ANIMA_HEARTBEAT_URL is empty — the outbound dead-man's switch is INERT."
            log "    Provision and acceptance-test it per docs/operations/HEARTBEAT.md."
            log "    No service restart is required; the next timer run reads anima.env."
            ;;
    esac
else
    log "Secrets: all expected keys present in ~/.anima/anima.env"
fi
log "Secrets reference: config/anima.env.example"
log "If I2C sensors (temp/humidity/light) fail: reboot required. Run: ssh $PI_USER@$PI_HOST 'sudo reboot'"
log "Check: ssh $PI_USER@$PI_HOST 'journalctl -u anima -f'"
log "MCP (LAN):       http://$PI_HOST:8766/mcp/"
[ -n "$NEW_TS_IP" ] && log "MCP (Tailscale): http://$NEW_TS_IP:8766/mcp/"
