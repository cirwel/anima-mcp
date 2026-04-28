#!/bin/bash
# Auto-restore Lumen's data on boot if ~/.anima/anima.db is missing or corrupt.
# Called by anima-restore.service (oneshot, before anima-broker and anima).
# Never overwrites existing good data.

set -euo pipefail

ANIMA_DIR="/home/unitares-anima/.anima"
DB_PATH="$ANIMA_DIR/anima.db"
SSH_KEY="/home/unitares-anima/.ssh/id_ed25519"
SSH_OPTS="-i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
BACKUP_USER="${BACKUP_USER:-cirwel}"
BACKUP_DIR="backups/lumen"
MARKER="$ANIMA_DIR/.restored_marker"
LOG_TAG="anima-restore"

# Mac connection targets (tried in order). Override with BACKUP_MAC_HOSTS env
# (space-separated) to avoid baking operator-specific addresses into a public
# template. Tailscale IPs change after reinstalls — prefer hostnames.
if [ -n "${BACKUP_MAC_HOSTS:-}" ]; then
    # shellcheck disable=SC2206
    MAC_HOSTS=( $BACKUP_MAC_HOSTS )
else
    MAC_HOSTS=(
        "lumen-mac"            # Tailscale hostname (set in your tailnet)
    )
fi

log() {
    logger -t "$LOG_TAG" "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Guard: never overwrite good data ---
if [ -f "$DB_PATH" ] && [ "$(stat -c%s "$DB_PATH" 2>/dev/null || echo 0)" -gt 1024 ]; then
    log "DB exists and is >1KB, skipping restore"
    exit 0
fi

log "DB missing or trivial — starting restore"
mkdir -p "$ANIMA_DIR"

# --- Check SSH key exists ---
if [ ! -f "$SSH_KEY" ]; then
    log "No SSH key at $SSH_KEY — cannot restore from Mac, starting fresh"
    exit 0
fi

# --- Find reachable Mac ---
MAC_HOST=""
for host in "${MAC_HOSTS[@]}"; do
    if ssh $SSH_OPTS "$BACKUP_USER@$host" "echo ok" >/dev/null 2>&1; then
        MAC_HOST="$host"
        log "Mac reachable at $host"
        break
    fi
done

if [ -z "$MAC_HOST" ]; then
    log "Mac not reachable on any address — starting fresh"
    exit 0
fi

REMOTE="$BACKUP_USER@$MAC_HOST"

# --- Restore DB (try live backup first, then dated snapshots) ---
DB_RESTORED=false

# Try the live rsync copy
log "Pulling anima.db from Mac backup..."
rsync -az -e "ssh $SSH_OPTS" "$REMOTE:~/$BACKUP_DIR/anima_data/anima.db" "$DB_PATH" 2>/dev/null

if [ -f "$DB_PATH" ] && [ "$(stat -c%s "$DB_PATH" 2>/dev/null || echo 0)" -gt 1024 ]; then
    # Verify integrity
    if sqlite3 "$DB_PATH" "PRAGMA integrity_check;" 2>/dev/null | grep -q "^ok$"; then
        log "DB restored and integrity verified"
        DB_RESTORED=true
    else
        log "DB integrity check failed, trying dated snapshots..."
        rm -f "$DB_PATH"
    fi
fi

# Fall back to most recent dated snapshot
if [ "$DB_RESTORED" = false ]; then
    log "Trying dated snapshots..."
    LATEST_SNAPSHOT=$(ssh $SSH_OPTS "$REMOTE" "ls -t ~/$BACKUP_DIR/anima_*.db 2>/dev/null | head -1" 2>/dev/null || true)

    if [ -n "$LATEST_SNAPSHOT" ]; then
        rsync -az -e "ssh $SSH_OPTS" "$REMOTE:$LATEST_SNAPSHOT" "$DB_PATH" 2>/dev/null

        if [ -f "$DB_PATH" ] && sqlite3 "$DB_PATH" "PRAGMA integrity_check;" 2>/dev/null | grep -q "^ok$"; then
            log "DB restored from snapshot: $(basename "$LATEST_SNAPSHOT")"
            DB_RESTORED=true
        else
            log "Snapshot also failed integrity check"
            rm -f "$DB_PATH"
        fi
    else
        log "No dated snapshots found on Mac"
    fi
fi

# --- Restore JSON files and drawings ---
log "Pulling JSON configs and drawings..."
rsync -az -e "ssh $SSH_OPTS" \
    --include='*.json' \
    --include='drawings/***' \
    --include='schema_renders/***' \
    --exclude='*.db' \
    --exclude='*.db-wal' \
    --exclude='*.db-shm' \
    "$REMOTE:~/$BACKUP_DIR/anima_data/" "$ANIMA_DIR/" 2>/dev/null || log "JSON/drawings sync failed (non-fatal)"

# --- Write restore marker for kintsugi gap detection ---
if [ "$DB_RESTORED" = true ]; then
    cat > "$MARKER" <<EOF
{
    "restored_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "source": "$MAC_HOST",
    "db_restored": true,
    "reason": "boot_auto_restore"
}
EOF
    log "Restore complete — marker written"
else
    cat > "$MARKER" <<EOF
{
    "restored_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "source": "$MAC_HOST",
    "db_restored": false,
    "reason": "boot_auto_restore_db_failed"
}
EOF
    log "Restore partial — DB not recovered, JSON/drawings may be present"
fi

# Fix ownership (script runs as root, files must be owned by anima user)
chown -R unitares-anima:unitares-anima "$ANIMA_DIR"

log "Auto-restore finished"
