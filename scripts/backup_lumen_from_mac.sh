#!/bin/bash
# CANONICAL COPY. This runs from ~/scripts/backup_lumen.sh on the operator's
# Mac via launchd (com.unitares.lumen-backup, daily 03:00). It lived ONLY there
# and was never version controlled — including the 2026-07-30 hang fix below,
# which existed on exactly one disk. Committed here so it survives that disk.
# If you edit the live copy, mirror it here (and vice versa).
# Backup Lumen's data from Pi to Mac
# Run daily via launchd: com.unitares.lumen-backup.plist
#
# Purpose: recover Lumen's accumulated DB (identity/memory/EISV history) after
# an SD-card or box failure. The DB is the irreplaceable artifact; the rest of
# ~/.anima is largely regenerable.
#
# RESTORE: copy the newest verified snapshot back to the Pi, e.g.
#   scp $BACKUP_DIR/anima_<latest>.db unitares-anima@lumen-local:~/.anima/anima.db
# (stop the anima service first; the snapshot is a consistent sqlite3 .backup).
# If the Mac is also gone, pull the atomic off-site bundle from HF instead:
#   hf download hikewa/lumen-db-backups daily/lumen_recovery_<Dow>.tar --repo-type dataset
# (private dataset; each bundle contains the matching DB + learned-state archive;
# day-of-week rotation holds the last 7 complete daily recovery points).

BACKUP_DIR="/Users/cirwel/backups/lumen"
DATE=$(date +%Y%m%d_%H%M)
PI_HOST="lumen-local"
SSH_KEY="/Users/cirwel/.ssh/id_ed25519_pi"
# ServerAlive* bounds a STALLED transfer, which ConnectTimeout does not:
# ConnectTimeout only covers session setup, so an rsync whose peer goes away
# mid-stream can block forever. The 2026-07 log is full of "Connection closed
# by remote host" / "hangup on receiver" from exactly that. 15s x 4 = the
# session dies within ~60s instead of hanging the whole run.
SSH_OPTS="-i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
LOG="/Users/cirwel/backups/lumen_backup.log"

# Off-site copy (3-2-1): push one bundle containing the verified DB plus its
# sanitized learned-state archive to a private HF dataset so a Mac-disk or
# physical failure can't take the Pi and its only backup at once. A single Hub
# artifact prevents a partial two-upload run from publishing mismatched halves.
# Off-site push is best-effort: a failure is logged but does not fail the run,
# because the local backup is already captured and verified.
HF_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/hf"
HF_REPO="hikewa/lumen-db-backups"
HF_BACKUP="${HF_BACKUP:-1}"   # set HF_BACKUP=0 to skip the off-site push
PI_BACKUP_MAX_AGE_MINUTES="${PI_BACKUP_MAX_AGE_MINUTES:-180}"
# Retention is in snapshots, not hours. This launchd job is daily plus
# RunAtLoad; 21 snapshots is roughly three weeks. The old hard-coded 48 was
# written for an hourly cadence and could silently consume ~12 GB on the Mac.
ANIMA_DB_RETAIN="${ANIMA_DB_RETAIN:-21}"
case "$PI_BACKUP_MAX_AGE_MINUTES" in
    ''|*[!0-9]*) PI_BACKUP_MAX_AGE_MINUTES=180 ;;
esac

mkdir -p "$BACKUP_DIR"

# Stricter exit policy: a run is only "ok" if it produces a fresh, consistent
# backup this invocation. Any path that fails to capture one sets FAILED=1 and
# the script exits non-zero, so the launchd outcome wrapper records a failure
# instead of a false "ok". See exit at end of script.
FAILED=0
FORENSICS_MIRRORED=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG"
}

validate_learned_state() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = {
    "anima_config.json",
    "preferences.json",
    "self_model.json",
    "patterns.json",
    "metacognition_baselines.json",
}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    print("missing required snapshots: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
for path in root.glob("*.json"):
    try:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"invalid {path.name}: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
}

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

hf_upload_with_watchdog() {
    LOCAL_FILE="$1"
    REMOTE_PATH="$2"
    LABEL="$3"
    HF_TIMEOUT="${HF_TIMEOUT:-900}"

    "$HF_BIN" upload "$HF_REPO" "$LOCAL_FILE" "$REMOTE_PATH" \
        --repo-type dataset \
        --commit-message "off-site ${LABEL} backup ${DATE}" >/dev/null 2>&1 &
    HF_PID=$!

    ( sleep "$HF_TIMEOUT"
      kill -TERM "$HF_PID" 2>/dev/null
      sleep 10
      kill -KILL "$HF_PID" 2>/dev/null ) >/dev/null 2>&1 &
    HF_WATCHDOG=$!

    if wait "$HF_PID" 2>/dev/null; then
        HF_RC=0
    else
        HF_RC=$?
    fi
    kill "$HF_WATCHDOG" 2>/dev/null
    wait "$HF_WATCHDOG" 2>/dev/null

    if [ "$HF_RC" -eq 0 ]; then
        log "Off-site ${LABEL} pushed to HF: $HF_REPO/$REMOTE_PATH"
    elif [ "$HF_RC" -eq 143 ] || [ "$HF_RC" -eq 137 ]; then
        log "WARNING: off-site ${LABEL} push exceeded ${HF_TIMEOUT}s and was killed (local backup is OK)"
    else
        log "WARNING: off-site ${LABEL} push failed rc=${HF_RC} (local backup is OK; check hf auth/network)"
    fi
    return 0
}

log "Starting Lumen backup..."

# Test connection (try LAN first, then Tailscale)
if ! ssh $SSH_OPTS unitares-anima@$PI_HOST "echo ok" >/dev/null 2>&1; then
    PI_HOST="lumen"  # Tailscale fallback (hostname, won't break on IP change)
    if ! ssh $SSH_OPTS unitares-anima@$PI_HOST "echo ok" >/dev/null 2>&1; then
        log "Pi not reachable on LAN or Tailscale, skipping backup"

        # Staleness alert: check how old the last successful backup is
        STALE_FILE="$BACKUP_DIR/.last_success"
        if [ -f "$STALE_FILE" ]; then
            LAST_SUCCESS=$(cat "$STALE_FILE")
            NOW=$(date +%s)
            AGE_HOURS=$(( (NOW - LAST_SUCCESS) / 3600 ))
            if [ "$AGE_HOURS" -ge 48 ]; then
                ALERT_MSG="LUMEN BACKUP STALE: No successful backup in ${AGE_HOURS} hours. Pi unreachable."
                log "ALERT: $ALERT_MSG"
                # macOS notification
                osascript -e "display notification \"$ALERT_MSG\" with title \"Lumen Backup Alert\"" 2>/dev/null || true
            fi
        fi

        # Pi unreachable => no backup captured this run => report failure.
        exit 1
    fi
    log "Using Tailscale connection"
fi

# Slim state mirror for full restore. restore_lumen.sh rebuilds Lumen from
# $BACKUP_DIR/anima_data/: the small JSON files there are the SOURCE OF TRUTH
# for Lumen's learned self (self_model.json beliefs, knowledge.json Q&A insights,
# patterns.json, preferences.json, anima_config.json calibration, anima_history.json,
# last_schema.json) — they
# are NOT stored in anima.db. We mirror ~/.anima EXCEPT the heavy regenerable
# dirs: backups/ (Pi's own rotation, ~5G — the scan that caused rsync 255) and
# schema_renders/ (~1G, re-renderable), plus the live *.db* (captured consistently
# below). Secret-bearing anima.env* files are intentionally excluded. The broad
# *.db* rule also removes old ``anima.db.corrupted.*`` files; ``*.db`` alone
# left 1.5 GB of those inside the supposedly slim mirror. Footprint stays
# ~10-20M so the run is fast and reliable.
log "Mirroring slim ~/.anima state (for restore)..."
MIRROR_CAPTURED=0
MIRROR_STAGE=$(mktemp -d "$BACKUP_DIR/.anima_data.staging.XXXXXX") || {
    log "ERROR: could not create learned-state staging directory"
    exit 1
}
rsync -az --timeout=120 \
    --exclude='backups/' --exclude='schema_renders/' \
    --exclude='*.db*' --exclude='anima.env*' \
    --exclude='*.tmp' --exclude='*.log' \
    -e "ssh $SSH_OPTS" "unitares-anima@$PI_HOST:~/.anima/" "$MIRROR_STAGE/"
MIRROR_STATUS=$?
MIRROR_VALIDATION=""
OAUTH_CAPTURED=1
ssh $SSH_OPTS unitares-anima@$PI_HOST "test -s ~/.anima/oauth.db" >/dev/null 2>&1
OAUTH_PRESENT_STATUS=$?
if [ $OAUTH_PRESENT_STATUS -eq 0 ]; then
    PI_OAUTH_TMP="/tmp/anima_oauth_snap_${DATE}.db"
    if ssh $SSH_OPTS unitares-anima@$PI_HOST \
        "python3 -c \"import sqlite3,os;s=sqlite3.connect(os.path.expanduser('~/.anima/oauth.db'));d=sqlite3.connect('${PI_OAUTH_TMP}');s.backup(d);d.close();s.close()\"" 2>/dev/null && \
       rsync -az --timeout=60 -e "ssh $SSH_OPTS" \
        "unitares-anima@$PI_HOST:${PI_OAUTH_TMP}" "$MIRROR_STAGE/oauth.db" && \
       sqlite_ok "$MIRROR_STAGE/oauth.db"; then
        if chmod 600 "$MIRROR_STAGE/oauth.db"; then
            log "OAuth client/token database captured consistently"
        else
            OAUTH_CAPTURED=0
            log "ERROR: could not protect captured OAuth database permissions"
        fi
    else
        OAUTH_CAPTURED=0
        log "ERROR: OAuth database exists but could not be captured consistently"
    fi
    ssh $SSH_OPTS unitares-anima@$PI_HOST "rm -f ${PI_OAUTH_TMP}" 2>/dev/null
elif [ $OAUTH_PRESENT_STATUS -gt 1 ]; then
    OAUTH_CAPTURED=0
    log "ERROR: could not determine whether OAuth persistence requires backup"
fi

if { [ $MIRROR_STATUS -eq 0 ] || [ $MIRROR_STATUS -eq 24 ]; } && \
   [ $OAUTH_CAPTURED -eq 1 ] && \
   MIRROR_VALIDATION=$(validate_learned_state "$MIRROR_STAGE" 2>&1); then
    # Publish only a completed staging tree. Keep one prior generation so a
    # power loss between the two directory renames remains recoverable.
    MIRROR_SWAP_READY=1
    if ! rm -rf "$BACKUP_DIR/anima_data.previous"; then
        MIRROR_SWAP_READY=0
        FAILED=1
        log "ERROR: could not retire prior learned-state generation"
    fi
    if [ $MIRROR_SWAP_READY -eq 1 ] && [ -e "$BACKUP_DIR/anima_data" ]; then
        if ! mv "$BACKUP_DIR/anima_data" "$BACKUP_DIR/anima_data.previous"; then
            MIRROR_SWAP_READY=0
            FAILED=1
            log "ERROR: could not preserve prior learned-state mirror"
        fi
    fi
    if [ $MIRROR_SWAP_READY -eq 1 ] && mv "$MIRROR_STAGE" "$BACKUP_DIR/anima_data"; then
        MIRROR_CAPTURED=1
        log "Slim state mirror ok: $(du -sh "$BACKUP_DIR/anima_data/" 2>/dev/null | cut -f1)"
    else
        FAILED=1
        log "ERROR: could not publish learned-state mirror"
        if [ ! -e "$BACKUP_DIR/anima_data" ] && [ -e "$BACKUP_DIR/anima_data.previous" ]; then
            mv "$BACKUP_DIR/anima_data.previous" "$BACKUP_DIR/anima_data" 2>/dev/null
        fi
        rm -rf "$MIRROR_STAGE"
    fi
else
    FAILED=1
    rm -rf "$MIRROR_STAGE"
    if [ $MIRROR_STATUS -eq 0 ] || [ $MIRROR_STATUS -eq 24 ]; then
        log "ERROR: slim state validation failed (${MIRROR_VALIDATION:-unknown error}) — this run is not a complete restore point"
    else
        log "ERROR: slim state mirror failed (status $MIRROR_STATUS) — this run is not a complete restore point"
    fi
fi

# Package the sanitized learned-state generation for the off-site half of the
# recovery point. anima.env* and *.db* never entered this tree, so the archive
# contains continuity data without API keys or stale/corrupt database copies.
STATE_ARCHIVE=""
if [ $MIRROR_CAPTURED -eq 1 ]; then
    STATE_ARCHIVE_TMP="$BACKUP_DIR/.anima_state_${DATE}.tar.gz.tmp"
    STATE_ARCHIVE_CANDIDATE="$BACKUP_DIR/anima_state_${DATE}.tar.gz"
    if tar -C "$BACKUP_DIR/anima_data" --exclude='./oauth.db' \
       -czf "$STATE_ARCHIVE_TMP" . && \
       tar -tzf "$STATE_ARCHIVE_TMP" >/dev/null 2>&1; then
        mv "$STATE_ARCHIVE_TMP" "$STATE_ARCHIVE_CANDIDATE"
        STATE_ARCHIVE="$STATE_ARCHIVE_CANDIDATE"
        log "Learned-state archive captured: $(basename "$STATE_ARCHIVE") ($(du -h "$STATE_ARCHIVE" | cut -f1))"
        ls -t "$BACKUP_DIR"/anima_state_*.tar.gz 2>/dev/null | tail -n +15 | xargs rm -f 2>/dev/null
    else
        rm -f "$STATE_ARCHIVE_TMP"
        log "WARNING: could not package learned state for off-site copy (local mirror is OK)"
    fi
fi

# Preserve Lumen's lossless forensic archives off-device. Never use --delete:
# the Pi is allowed to expire a locally archived incident only after a receipt
# proves an earlier Mac run had already seen it.
FORENSICS_REMOTE_STATE=$(ssh $SSH_OPTS unitares-anima@$PI_HOST \
    "if [ -d ~/.anima/backups/forensics ]; then echo present; else echo absent; fi" \
    2>/dev/null)
if [ "$FORENSICS_REMOTE_STATE" = "absent" ]; then
    FORENSICS_MIRRORED=1
elif [ "$FORENSICS_REMOTE_STATE" = "present" ]; then
    mkdir -p "$BACKUP_DIR/forensics"
    rsync -az --timeout=120 -e "ssh $SSH_OPTS" \
        "unitares-anima@$PI_HOST:~/.anima/backups/forensics/" \
        "$BACKUP_DIR/forensics/"
    FORENSICS_STATUS=$?
    if [ $FORENSICS_STATUS -eq 0 ] || [ $FORENSICS_STATUS -eq 24 ]; then
        FORENSICS_MIRRORED=1
        log "Forensic archives mirrored off-device"
    else
        log "WARNING: forensic archive mirror failed (status $FORENSICS_STATUS)"
    fi
else
    log "WARNING: could not determine forensic archive state on Pi"
fi

# Capture a consistent DB snapshot. anima.db holds identity, state_history,
# drawing_history, system_metrics — the heavy irreplaceable core. (The learned
# self-model lives in the JSON mirrored above, NOT here.) Pulled as a consistent
# sqlite .backup and pushed off-site. We do NOT rsync the full ~/.anima tree
# (that ~64G scan dropped the Pi connection with rsync 255).
DB_CAPTURED=0

# Primary: a recent Pi hourly snapshot (backup_db.sh via the sqlite backup API).
# Never advance today's success marker from an arbitrarily old file: if the
# hourly job stalled, mint a fresh snapshot from the live database below.
PI_BACKUP=$(ssh $SSH_OPTS unitares-anima@$PI_HOST \
    "find ~/.anima/backups -maxdepth 1 -type f -name 'anima_*.db' -mmin -${PI_BACKUP_MAX_AGE_MINUTES} -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-" \
    2>/dev/null)
if [ -n "$PI_BACKUP" ]; then
    rsync -az --timeout=120 -e "ssh $SSH_OPTS" "unitares-anima@$PI_HOST:$PI_BACKUP" "$BACKUP_DIR/anima_${DATE}.db"
    DB_RSYNC_STATUS=$?
    if { [ $DB_RSYNC_STATUS -eq 0 ] || [ $DB_RSYNC_STATUS -eq 24 ]; } && [ -s "$BACKUP_DIR/anima_${DATE}.db" ]; then
        PRIMARY_INTEGRITY=$(sqlite3 "$BACKUP_DIR/anima_${DATE}.db" "PRAGMA integrity_check;" 2>&1 | head -1)
        if [ "$PRIMARY_INTEGRITY" = "ok" ]; then
            DB_CAPTURED=1
            log "DB snapshot from recent Pi backup: anima_${DATE}.db ($(du -h "$BACKUP_DIR/anima_${DATE}.db" | cut -f1))"
        else
            log "WARNING: recent Pi snapshot failed integrity_check ($PRIMARY_INTEGRITY) — trying remote .backup fallback"
            rm -f "$BACKUP_DIR/anima_${DATE}.db" 2>/dev/null
        fi
    else
        log "WARNING: Pi-snapshot rsync failed (status $DB_RSYNC_STATUS) — trying remote .backup fallback"
        rm -f "$BACKUP_DIR/anima_${DATE}.db" 2>/dev/null
    fi
else
    log "No Pi snapshot newer than ${PI_BACKUP_MAX_AGE_MINUTES} minutes — minting a fresh remote .backup"
fi

# Fallback: ask the Pi to mint a fresh consistent snapshot via sqlite3 .backup,
# pull it, then clean it up. No dependency on any local mirror; stays WAL-consistent.
# NOTE: this uses python3's sqlite3 module, NOT the sqlite3(1) CLI. The CLI is
# NOT installed on the Pi (`command -v sqlite3` -> nothing), so the original
# `ssh ... "sqlite3 ... .backup"` could never succeed. It was dead code that
# would have failed exactly when it was needed — when the Pi's own hourly
# snapshot is missing, which is the only path that reaches here. python3 is
# present (it runs both services) and Connection.backup() is equally
# WAL-consistent.
if [ $DB_CAPTURED -eq 0 ]; then
    PI_TMP="/tmp/anima_snap_${DATE}.db"
    if ssh $SSH_OPTS unitares-anima@$PI_HOST "python3 -c \"import sqlite3,os;s=sqlite3.connect(os.path.expanduser('~/.anima/anima.db'));d=sqlite3.connect('${PI_TMP}');s.backup(d);d.close();s.close()\"" 2>/dev/null; then
        rsync -az --timeout=120 -e "ssh $SSH_OPTS" "unitares-anima@$PI_HOST:${PI_TMP}" "$BACKUP_DIR/anima_${DATE}.db"
        DB_RSYNC_STATUS=$?
        ssh $SSH_OPTS unitares-anima@$PI_HOST "rm -f ${PI_TMP}" 2>/dev/null
        if { [ $DB_RSYNC_STATUS -eq 0 ] || [ $DB_RSYNC_STATUS -eq 24 ]; } && [ -s "$BACKUP_DIR/anima_${DATE}.db" ]; then
            DB_CAPTURED=1
            log "DB snapshot from remote .backup fallback: anima_${DATE}.db ($(du -h "$BACKUP_DIR/anima_${DATE}.db" | cut -f1))"
        else
            log "ERROR: fallback .backup rsync failed (status $DB_RSYNC_STATUS) — no consistent DB captured"
            rm -f "$BACKUP_DIR/anima_${DATE}.db" 2>/dev/null
        fi
    else
        log "ERROR: remote sqlite3 .backup failed — no consistent DB captured"
    fi
fi

# Recovery is the whole point: a present-but-corrupt DB is worse than a
# reported failure. Verify the captured snapshot is actually restorable.
if [ $DB_CAPTURED -eq 1 ]; then
    INTEGRITY=$(sqlite3 "$BACKUP_DIR/anima_${DATE}.db" "PRAGMA integrity_check;" 2>&1 | head -1)
    if [ "$INTEGRITY" != "ok" ]; then
        log "ERROR: captured DB failed integrity_check ($INTEGRITY) — discarding"
        rm -f "$BACKUP_DIR/anima_${DATE}.db" 2>/dev/null
        DB_CAPTURED=0
    fi
fi

# The retired broker agency store — retained as rollback/history state.
#
# The server is the sole active action learner. The historical broker table is
# disabled by default, but preserving its 1.6M updates keeps rollback and audit
# history recoverable. It sits outside ~/.anima, which is all the block above
# covers.
BROKER_DB="$BACKUP_DIR/agency_${DATE}.db"
PI_AGENCY_TMP="/tmp/anima_agency_snap_${DATE}.db"
if ssh $SSH_OPTS unitares-anima@$PI_HOST "test -f ~/anima-mcp/anima.db && python3 -c \"import sqlite3,os;s=sqlite3.connect(os.path.expanduser('~/anima-mcp/anima.db'));d=sqlite3.connect('${PI_AGENCY_TMP}');s.backup(d);d.close();s.close()\"" 2>/dev/null; then
    rsync -az --timeout=60 -e "ssh $SSH_OPTS" "unitares-anima@$PI_HOST:${PI_AGENCY_TMP}" "$BROKER_DB" 2>/dev/null
    AGENCY_RSYNC=$?
    ssh $SSH_OPTS unitares-anima@$PI_HOST "rm -f ${PI_AGENCY_TMP}" 2>/dev/null
    if { [ $AGENCY_RSYNC -eq 0 ] || [ $AGENCY_RSYNC -eq 24 ]; } && [ -s "$BROKER_DB" ]; then
        AGENCY_ROWS=$(sqlite3 "$BROKER_DB" "SELECT COALESCE(SUM(count),0) FROM agency_values;" 2>/dev/null)
        log "Broker agency store captured: agency_${DATE}.db (${AGENCY_ROWS:-?} updates) [#123]"
        # Keep only the last 14 — this file is small but the run is nightly.
        ls -t "$BACKUP_DIR"/agency_*.db 2>/dev/null | tail -n +15 | xargs rm -f 2>/dev/null
    else
        log "WARNING: broker agency store rsync failed (status $AGENCY_RSYNC) — see #123"
        rm -f "$BROKER_DB" 2>/dev/null
    fi
else
    log "WARNING: could not snapshot ~/anima-mcp/anima.db (broker agency store, #123)"
fi

if [ $DB_CAPTURED -eq 1 ]; then
    # A recovery point is complete only when both the verified database and
    # learned JSON/event state were refreshed in this run.  Still retain and
    # off-site the DB when the mirror failed, but never advance last_success.
    if [ $MIRROR_CAPTURED -eq 1 ]; then
        if ! date +%s > "$BACKUP_DIR/.last_success"; then
            FAILED=1
            log "ERROR: could not update backup success marker"
        fi
    fi

    # Build one off-site recovery unit. The learned archive and DB are already
    # independently validated; tar -tf verifies the final container before it
    # can replace a weekday slot on the Hub.
    RECOVERY_BUNDLE=""
    if [ $MIRROR_CAPTURED -eq 1 ] && [ -n "$STATE_ARCHIVE" ] && [ -s "$STATE_ARCHIVE" ]; then
        BUNDLE_TMP="$BACKUP_DIR/.lumen_recovery_${DATE}.tar.tmp"
        BUNDLE_CANDIDATE="$BACKUP_DIR/lumen_recovery_${DATE}.tar"
        if tar -C "$BACKUP_DIR" -cf "$BUNDLE_TMP" \
            "anima_${DATE}.db" "$(basename "$STATE_ARCHIVE")" && \
           tar -tf "$BUNDLE_TMP" >/dev/null 2>&1; then
            mv "$BUNDLE_TMP" "$BUNDLE_CANDIDATE"
            RECOVERY_BUNDLE="$BUNDLE_CANDIDATE"
            log "Complete off-site bundle captured: $(basename "$RECOVERY_BUNDLE") ($(du -h "$RECOVERY_BUNDLE" | cut -f1))"
            ls -t "$BACKUP_DIR"/lumen_recovery_*.tar 2>/dev/null | tail -n +15 | xargs rm -f 2>/dev/null
        else
            rm -f "$BUNDLE_TMP"
            log "WARNING: could not package complete off-site recovery bundle"
        fi
    fi

    # Off-site copy to HF (best-effort; does not affect local run success/failure)
    #
    # HARD TIMEOUT, and it is load-bearing. On 2026-07-24 this upload hung for
    # 6d14h (14 min of CPU, then sleeping forever). launchd will not start a
    # second instance while one is running, so every nightly backup from
    # 07-24 to 07-30 silently never fired — no error, no alert, and
    # `unitares-automations census` still read "last=running". The comment
    # above says this step is best-effort, but a HANG IS NOT A FAILURE: it
    # never returns, so the best-effort branch never gets reached.
    #
    # macOS ships no timeout(1)/gtimeout, so supervise with a watchdog. Do not
    # poll `kill -0` instead — a finished child stays a zombie until reaped, so
    # `kill -0` keeps succeeding and the poll would run the full timeout anyway.
    if [ "$HF_BACKUP" = "1" ] && [ -x "$HF_BIN" ]; then
        DAY_NAME=$(date +%a)
        if [ -n "$RECOVERY_BUNDLE" ] && [ -s "$RECOVERY_BUNDLE" ]; then
            hf_upload_with_watchdog \
                "$RECOVERY_BUNDLE" \
                "daily/lumen_recovery_${DAY_NAME}.tar" \
                "complete recovery bundle"
        else
            log "WARNING: off-site upload skipped because no complete recovery bundle was built"
        fi
    fi
else
    FAILED=1
fi

# Publish a machine-readable receipt back to the Pi only after this run has a
# verified database, learned-state mirror, and complete restore bundle. Storage
# maintenance treats a missing/stale receipt as a hard stop for DB artifacts.
if [ $FAILED -eq 0 ] && [ $DB_CAPTURED -eq 1 ] && \
   [ $MIRROR_CAPTURED -eq 1 ] && [ -n "${RECOVERY_BUNDLE:-}" ] && \
   [ -s "$RECOVERY_BUNDLE" ]; then
    RECEIPT_LOCAL=$(mktemp "$BACKUP_DIR/.offdevice-recovery-receipt.XXXXXX")
    RECEIPT_REMOTE="/home/unitares-anima/.anima/backups/.offdevice-recovery-receipt.tmp"
    python3 - "$RECEIPT_LOCAL" "$DATE" "$FORENSICS_MIRRORED" \
        "$(basename "$RECOVERY_BUNDLE")" <<'PY'
import json
import sys
import time

path, generation, mirrored, bundle = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema": "anima.offdevice-recovery.v1",
            "captured_at_epoch": time.time(),
            "generation": generation,
            "database_integrity": "ok",
            "learned_state_valid": True,
            "restore_bundle_verified": True,
            "forensics_mirrored": mirrored == "1",
            "bundle": bundle,
        },
        handle,
        sort_keys=True,
    )
PY
    if scp $SSH_OPTS "$RECEIPT_LOCAL" \
        "unitares-anima@$PI_HOST:$RECEIPT_REMOTE" >/dev/null 2>&1 && \
       ssh $SSH_OPTS unitares-anima@$PI_HOST \
        "chmod 600 '$RECEIPT_REMOTE' && mv '$RECEIPT_REMOTE' ~/.anima/backups/offdevice-recovery-receipt.json" \
        >/dev/null 2>&1; then
        log "Off-device recovery receipt published to Pi"
    else
        log "WARNING: recovery receipt publish failed; Pi forensic pruning remains gated"
    fi
    rm -f "$RECEIPT_LOCAL"
fi

# Keep a bounded number of Mac database snapshots. HF independently keeps the
# seven weekday recovery bundles.
ls -t "$BACKUP_DIR"/anima_*.db 2>/dev/null | \
    tail -n +$((ANIMA_DB_RETAIN + 1)) | xargs rm -f 2>/dev/null

# Keep log from growing too large
if [ -f "$LOG" ]; then
    tail -500 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
fi

if [ $FAILED -eq 0 ]; then
    log "Backup complete"
    exit 0
else
    log "Backup FAILED — required restore state was incomplete this run"
    exit 1
fi
