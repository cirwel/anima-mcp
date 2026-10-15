#!/bin/bash
# Hourly consistent backup of anima.db (sqlite backup API, WAL-safe).
# Keeps last 24 backups (1 day of hourly snapshots).
# Installed via crontab: 15 * * * * /home/unitares-anima/anima-mcp/scripts/backup_db.sh
#
# Why two stages (2026-10-15): copying the ~280MB DB straight onto the SD card
# left the whole copy as dirty page cache. Lumen's own small commits then had
# to wait behind it, and the broker went silent for the ~26s the backup took,
# every hour at :15 (server log: "Too many consecutive errors"). ionice cannot
# help because buffered writeback is not charged to this process.
#
# Stage 1 takes the consistent snapshot into RAM (/dev/shm), so the backup
# API's single read transaction never competes with SD writes. Stage 2 writes
# it to the card in 4MB chunks, flushing each one, so at most one chunk of
# backup data is ever waiting in front of Lumen's writes.

set -u

DB_PATH="$HOME/.anima/anima.db"
BACKUP_DIR="$HOME/.anima/backups"
TIMESTAMP=$(date +%Y%m%d_%H)
FINAL="$BACKUP_DIR/anima_${TIMESTAMP}.db"
STAGE_DIR="${ANIMA_BACKUP_STAGE_DIR:-/dev/shm}"
STAGE="$STAGE_DIR/anima_backup_$$.db"

trap 'rm -f "$STAGE" "$FINAL.partial"' EXIT

mkdir -p "$BACKUP_DIR"
[ -f "$DB_PATH" ] || exit 0

# RAM staging needs room for the whole DB plus headroom; otherwise stage
# beside the destination (the old behaviour, stall included) rather than fail.
DB_KB=$(( $(stat -c %s "$DB_PATH") / 1024 ))
FREE_KB=$(df -Pk "$STAGE_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -z "$FREE_KB" ] || [ "$FREE_KB" -lt $(( DB_KB * 3 / 2 )) ]; then
    STAGE="$FINAL.partial.stage"
    trap 'rm -f "$STAGE" "$FINAL.partial"' EXIT
    echo "[$(date)] Staging on disk: not enough room in $STAGE_DIR"
fi

# Use Python's sqlite3.backup() (sqlite3 CLI not installed).
nice -n 19 python3 - "$DB_PATH" "$STAGE" "$FINAL.partial" <<'PY'
import os, sqlite3, sys, time

src_path, stage_path, partial_path = sys.argv[1:4]
CHUNK = 4 * 1024 * 1024

src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
dst = sqlite3.connect(stage_path)
src.backup(dst)
dst.close()
src.close()

with open(stage_path, "rb") as fin, open(partial_path, "wb") as fout:
    while True:
        block = fin.read(CHUNK)
        if not block:
            break
        fout.write(block)
        fout.flush()
        os.fdatasync(fout.fileno())
        time.sleep(0.02)
PY
status=$?

if [ $status -eq 0 ]; then
    mv -f "$FINAL.partial" "$FINAL"
    echo "[$(date)] Backup OK: anima_${TIMESTAMP}.db"
else
    echo "[$(date)] Backup FAILED (python exit $status); no snapshot this hour"
    exit 1
fi

# Keep only last 24 backups
ls -1t "$BACKUP_DIR"/anima_*.db 2>/dev/null | tail -n +25 | xargs -r rm
