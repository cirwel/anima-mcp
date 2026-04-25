#!/bin/bash
# Anima Watchdog - Restart failed services with rate limiting
# Run via systemd timer every 5 minutes (see systemd/anima-watchdog.timer)
#
# Checks anima-broker.service and anima.service.
# Only restarts if a service has been down for >5 minutes.
# Rate-limits restarts to once per 10 minutes per service.

LOGFILE="/home/unitares-anima/.anima/watchdog.log"
STATE_DIR="/tmp/anima-watchdog"
MIN_DOWN_SECONDS=300    # 5 minutes before we act
MIN_RESTART_GAP=600     # 10 minutes between restart attempts

# Display-silent-failure: broker stays "active" but ST7789 has latched off.
# Broker logs `[Errno 5] Input/output error` ~every 2s; healthy = 0.
# Threshold 10 in 60s is well above any transient SPI hiccup.
DISPLAY_ERROR_WINDOW=60
DISPLAY_ERROR_THRESHOLD=10

mkdir -p "$STATE_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" >> "$LOGFILE"
    logger -t anima-watchdog "$1"
}

# Get how long a service has been inactive (seconds), or 0 if active
service_down_seconds() {
    local svc="$1"
    local state
    state=$(systemctl is-active "$svc" 2>/dev/null)

    if [ "$state" = "active" ]; then
        echo 0
        return
    fi

    # Get the timestamp of the last state change
    local ts
    ts=$(systemctl show "$svc" --property=ActiveExitTimestamp --value 2>/dev/null)
    if [ -z "$ts" ] || [ "$ts" = "" ]; then
        # Never started or no timestamp — treat as just now
        echo 0
        return
    fi

    local exit_epoch
    exit_epoch=$(date -d "$ts" +%s 2>/dev/null || echo 0)
    local now
    now=$(date +%s)

    if [ "$exit_epoch" -eq 0 ]; then
        echo 0
        return
    fi

    echo $((now - exit_epoch))
}

# Check if we restarted this service too recently
can_restart() {
    local svc="$1"
    local last_file="${STATE_DIR}/${svc}.last_restart"

    if [ ! -f "$last_file" ]; then
        return 0  # No record, OK to restart
    fi

    local last_restart
    last_restart=$(cat "$last_file" 2>/dev/null || echo 0)
    local now
    now=$(date +%s)
    local elapsed=$((now - last_restart))

    if [ "$elapsed" -ge "$MIN_RESTART_GAP" ]; then
        return 0  # Enough time has passed
    fi

    return 1  # Too soon
}

record_restart() {
    local svc="$1"
    date +%s > "${STATE_DIR}/${svc}.last_restart"
}

check_and_restart() {
    local svc="$1"
    local state
    state=$(systemctl is-active "$svc" 2>/dev/null)

    if [ "$state" = "active" ]; then
        return 0
    fi

    local down_secs
    down_secs=$(service_down_seconds "$svc")

    if [ "$down_secs" -lt "$MIN_DOWN_SECONDS" ]; then
        log "WARN: $svc is $state (down ${down_secs}s) - waiting for ${MIN_DOWN_SECONDS}s threshold"
        return 0
    fi

    if ! can_restart "$svc"; then
        log "WARN: $svc is $state (down ${down_secs}s) - skipping, restarted too recently"
        return 0
    fi

    log "RESTART: $svc has been $state for ${down_secs}s - attempting restart"
    if sudo systemctl restart "$svc" 2>&1; then
        record_restart "$svc"
        sleep 5
        local new_state
        new_state=$(systemctl is-active "$svc" 2>/dev/null)
        if [ "$new_state" = "active" ]; then
            log "OK: $svc restarted successfully"
        else
            log "FAIL: $svc restart attempted but state is now: $new_state"
        fi
    else
        record_restart "$svc"  # Record even on failure to prevent rapid retries
        log "FAIL: $svc restart command failed"
    fi
}

check_display_silent_failure() {
    local svc="anima-broker"
    local state
    state=$(systemctl is-active "$svc" 2>/dev/null)

    # If broker isn't active, service-down path above handles it.
    if [ "$state" != "active" ]; then
        return 0
    fi

    local err_count
    err_count=$(journalctl -u "$svc" --since "${DISPLAY_ERROR_WINDOW} seconds ago" --no-pager 2>/dev/null \
        | grep -c "\[Errno 5\] Input/output error")
    # grep -c exits 1 on zero matches; ensure numeric.
    err_count=${err_count:-0}

    if [ "$err_count" -lt "$DISPLAY_ERROR_THRESHOLD" ]; then
        return 0
    fi

    if ! can_restart "$svc"; then
        log "WARN: display Errno 5 flood (${err_count} in ${DISPLAY_ERROR_WINDOW}s) - skipping, restarted too recently"
        return 0
    fi

    log "RESTART: display silent failure (Errno 5 x${err_count} in ${DISPLAY_ERROR_WINDOW}s) - restarting anima-broker + anima"
    if sudo systemctl restart anima-broker anima 2>&1; then
        record_restart "anima-broker"
        record_restart "anima"
        log "OK: display-triggered restart fired"
    else
        record_restart "anima-broker"
        record_restart "anima"
        log "FAIL: display-triggered restart command failed"
    fi
}

# --- Main ---

check_and_restart "anima-broker"
check_and_restart "anima"
check_display_silent_failure

# Keep log from growing (keep last 200 lines)
if [ -f "$LOGFILE" ]; then
    tail -200 "$LOGFILE" > "${LOGFILE}.tmp" && mv "${LOGFILE}.tmp" "$LOGFILE"
fi
