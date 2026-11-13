#!/bin/bash
# Install Lumen's scheduled maintenance jobs as systemd timers and retire the
# crontab lines they replace. Run ON THE PI as unitares-anima (uses sudo).
# Idempotent: deploy.sh runs it on every deploy, restore_lumen.sh after a
# reflash.
#
# Why (2026-11-13): jobs were split between systemd timers and a user crontab,
# and the WiFi watchdog was on BOTH — restore_lumen.sh enabled
# wifi-watchdog.timer (via setup_pi_service.sh) and then also wrote a */2 cron
# line, so two copies ran every 2 minutes, one as root and one as the user,
# sharing a /tmp fail counter. One scheduler means one place to look:
# `systemctl list-timers 'anima-*' 'lumen-*' 'wifi-*'`.
#
# Order matters: every timer is installed and enabled BEFORE its cron line is
# removed, so a failure part-way leaves a job double-scheduled (harmless for
# one hour) rather than unscheduled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UNIT_DIR=/etc/systemd/system

# timer → the crontab pattern it replaces
TIMERS=(
    "wifi-watchdog:wifi_watchdog.sh"
    "anima-db-maintenance:db_maintenance.sh"
    "anima-db-backup:backup_db.sh"
    "anima-state-backup:backup_state.sh"
)

chmod +x "$SCRIPT_DIR"/wifi_watchdog.sh "$SCRIPT_DIR"/db_maintenance.sh \
    "$SCRIPT_DIR"/backup_db.sh "$SCRIPT_DIR"/backup_state.sh

changed=0
for entry in "${TIMERS[@]}"; do
    name="${entry%%:*}"
    for unit in "$name.service" "$name.timer"; do
        if ! cmp -s "$PROJECT_DIR/systemd/$unit" "$UNIT_DIR/$unit"; then
            sudo install -m 0644 "$PROJECT_DIR/systemd/$unit" "$UNIT_DIR/$unit"
            changed=1
        fi
    done
done
if [ "$changed" -eq 1 ]; then
    sudo systemctl daemon-reload
fi

retired=()
for entry in "${TIMERS[@]}"; do
    name="${entry%%:*}"
    sudo systemctl enable --now "$name.timer" >/dev/null
    systemctl is-active --quiet "$name.timer" || {
        echo "✗ $name.timer did not start; leaving its cron line in place" >&2
        exit 1
    }
    retired+=("${entry#*:}")
done

# Strip only the lines for jobs that now have a running timer; every other
# crontab entry is kept untouched.
if current=$(crontab -l 2>/dev/null); then
    pattern=$(IFS='|'; echo "${retired[*]}" | sed 's/\./\\./g')
    kept=$(printf '%s\n' "$current" | grep -Ev "anima-mcp/scripts/($pattern)" || true)
    if [ "$kept" != "$current" ]; then
        removed=$(( $(printf '%s\n' "$current" | wc -l) - $(printf '%s' "$kept" | grep -c '' || true) ))
        if [ -n "$kept" ]; then
            printf '%s\n' "$kept" | crontab -
        else
            crontab -r
        fi
        echo "✓ Retired $removed crontab line(s) now handled by systemd timers"
    fi
fi

echo "✓ Scheduled jobs on systemd timers: ${TIMERS[*]%%:*}"
