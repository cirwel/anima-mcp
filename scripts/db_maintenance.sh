#!/bin/bash
# DB maintenance — WAL checkpoint + integrity check.
# Run hourly via cron: 0 * * * * /home/unitares-anima/anima-mcp/scripts/db_maintenance.sh
#
# Uses Python's sqlite3 module instead of the sqlite3 CLI. The CLI isn't
# installed on the Pi; the prior shell version silently failed both ops and
# mis-diagnosed "command not found" as integrity failure, creating 16 days
# of false-alarm .corrupted DB copies (~195MB each).

set -u

DB="/home/unitares-anima/.anima/anima.db"
LOGFILE="/home/unitares-anima/.anima/db_maintenance.log"
PYTHON="/home/unitares-anima/anima-mcp/.venv/bin/python3"

# Fall back to system python3 if the venv isn't there (fresh install)
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3 || true)"
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" >> "$LOGFILE"
}

if [ ! -f "$DB" ]; then
    log "ERROR: DB not found at $DB"
    exit 1
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    log "ERROR: python3 not found — cannot run maintenance"
    exit 1
fi

HOUR=$(date +%H)

# Run both ops in one Python invocation. Prints one of:
#   OK:<wal_before>:<wal_after>[:integrity_ok|integrity_<detail>]
#   FAIL:<exception_type>:<message>
RESULT=$("$PYTHON" - "$DB" "$HOUR" <<'PY' 2>&1
import os, sqlite3, sys

db_path, hour = sys.argv[1], sys.argv[2]
wal_path = db_path + "-wal"


def wal_size():
    try:
        return os.path.getsize(wal_path)
    except OSError:
        return 0


try:
    wal_before = wal_size()
    wal_after = wal_before
    if wal_before > 1_048_576:  # >1MB
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        wal_after = wal_size()

    integrity = ""
    if hour == "00":
        with sqlite3.connect(db_path, timeout=60.0) as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
        status = row[0] if row else "unknown"
        integrity = ":integrity_ok" if status == "ok" else f":integrity_{status}"

    print(f"OK:{wal_before}:{wal_after}{integrity}")
except sqlite3.DatabaseError as e:
    # Real SQLite error — this IS corruption or schema damage, not a
    # tooling failure. Route it to the INTEGRITY FAILURE branch so the
    # forensic copy gets made.
    print(f"CORRUPT:{type(e).__name__}:{e}")
except Exception as e:
    # Everything else (ImportError, OSError, permission denied, ...) is a
    # tooling failure. Must NOT be treated as corruption — that was the
    # 2026-04-03 through 2026-04-19 false-alarm bug.
    print(f"FAIL:{type(e).__name__}:{e}")
PY
)

case "$RESULT" in
    OK:*)
        IFS=: read -r _ WAL_BEFORE WAL_AFTER REST <<< "$RESULT"
        if [ "$WAL_BEFORE" != "$WAL_AFTER" ]; then
            log "WAL checkpoint: ${WAL_BEFORE} -> ${WAL_AFTER} bytes"
        fi
        if [ "$REST" = "integrity_ok" ]; then
            log "Integrity check: OK"
        elif [ -n "$REST" ]; then
            log "INTEGRITY FAILURE: $REST"
            cp "$DB" "${DB}.corrupted.$(date +%Y%m%d_%H%M)"
            log "Corrupted DB saved for analysis"
        fi
        ;;
    CORRUPT:*)
        # Real SQLite-level corruption — make forensic copy
        log "INTEGRITY FAILURE: $RESULT"
        cp "$DB" "${DB}.corrupted.$(date +%Y%m%d_%H%M)"
        log "Corrupted DB saved for analysis"
        ;;
    FAIL:*)
        # Tooling failure (python missing, perms, etc.) — NOT corruption.
        # That mis-classification was the 16-day false-alarm bug.
        log "Maintenance tooling error (not corruption): $RESULT"
        ;;
    *)
        log "Unexpected maintenance output: $RESULT"
        ;;
esac

# Keep log bounded
if [ -f "$LOGFILE" ]; then
    tail -200 "$LOGFILE" > "${LOGFILE}.tmp" && mv "${LOGFILE}.tmp" "$LOGFILE"
fi
