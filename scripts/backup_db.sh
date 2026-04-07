#!/bin/bash
# Hourly consistent backup of anima.db using sqlite3 .backup (handles WAL safely)
# Keeps last 24 backups (1 day of hourly snapshots)
# Installed via crontab: 0 * * * * /home/unitares-anima/anima-mcp/scripts/backup_db.sh

DB_PATH="$HOME/.anima/anima.db"
BACKUP_DIR="$HOME/.anima/backups"
TIMESTAMP=$(date +%Y%m%d_%H)

mkdir -p "$BACKUP_DIR"

if [ -f "$DB_PATH" ]; then
    sqlite3 "$DB_PATH" ".backup '$BACKUP_DIR/anima_${TIMESTAMP}.db'" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[$(date)] Backup OK: anima_${TIMESTAMP}.db"
    else
        # Fall back to cp if sqlite3 .backup fails
        cp "$DB_PATH" "$BACKUP_DIR/anima_${TIMESTAMP}.db"
        echo "[$(date)] Backup via cp (sqlite3 failed): anima_${TIMESTAMP}.db"
    fi

    # Keep only last 24 backups
    ls -1t "$BACKUP_DIR"/anima_*.db 2>/dev/null | tail -n +25 | xargs -r rm
fi
