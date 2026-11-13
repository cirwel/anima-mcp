#!/bin/bash
# WiFi Watchdog - Auto-reconnect if WiFi drops
# Scheduled by systemd/wifi-watchdog.timer every 2 minutes, as root. Do not
# also add a crontab line: until 2026-11-13 restore_lumen.sh did, and two
# copies ran (root + user) sharing the /tmp fail counter.
#
# Escalation levels:
#   1. nmcli reconnect (soft)
#   2. rfkill cycle + driver reload (hardware reset)
#   3. Full reboot (nuclear)

LOGFILE="/home/unitares-anima/.anima/wifi_watchdog.log"
FAIL_COUNT_FILE="/tmp/wifi_watchdog_fails"
MAX_SOFT_RETRIES=3  # After 3 soft fails (6 min), escalate to hardware reset
MAX_HARD_RETRIES=5  # After 5 total fails (10 min), reboot

# Load WiFi credentials from env file (never hardcode)
ENV_FILE="/home/unitares-anima/.anima/anima.env"
WIFI_SSID=""
WIFI_PASSWORD=""
if [ -f "$ENV_FILE" ]; then
    WIFI_SSID=$(grep -E '^WIFI_SSID=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'" | head -1)
    WIFI_PASSWORD=$(grep -E '^WIFI_PASSWORD=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'" | head -1)
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" >> "$LOGFILE"
}

# Boot grace period — don't run aggressive recovery within 3 minutes of boot.
# The brcmfmac driver and NetworkManager need time to initialize after a reboot.
# Without this, we can boot-loop: reboot → driver slow → watchdog reboots again.
UPTIME_SECONDS=$(awk '{print int($1)}' /proc/uptime)
if [ "$UPTIME_SECONDS" -lt 180 ]; then
    log "Boot grace period (uptime ${UPTIME_SECONDS}s < 180s) — skipping"
    exit 0
fi

# Read consecutive fail count
FAIL_COUNT=0
if [ -f "$FAIL_COUNT_FILE" ]; then
    FAIL_COUNT=$(cat "$FAIL_COUNT_FILE" 2>/dev/null || echo 0)
fi

# Check if wlan0 even exists as an interface
if ! ip link show wlan0 > /dev/null 2>&1; then
    log "CRITICAL: wlan0 interface missing! Driver may have crashed."
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "$FAIL_COUNT" > "$FAIL_COUNT_FILE"
    # Try to reload the driver
    modprobe -r brcmfmac 2>/dev/null
    sleep 3
    modprobe brcmfmac 2>/dev/null
    sleep 5
    # If interface is still missing after driver reload, reboot (but only once —
    # fail count persists across reboots via /tmp which is tmpfs, cleared on reboot,
    # so this won't loop)
    if ! ip link show wlan0 > /dev/null 2>&1; then
        log "CRITICAL: wlan0 still missing after driver reload. REBOOTING."
        sync
        sleep 1
        reboot
    fi
    log "brcmfmac driver reload restored wlan0"
    exit 1
fi

# Check if we can reach the gateway
GATEWAY=$(ip route | grep default | awk '{print $3}')

# Also try router directly if no gateway in route table
if [ -z "$GATEWAY" ]; then
    GATEWAY="192.168.1.1"
fi

# Test connectivity
CONNECTED=false
if ping -c 2 -W 5 "$GATEWAY" > /dev/null 2>&1; then
    CONNECTED=true
fi

# Also test internet (in case gateway responds but upstream is down)
if [ "$CONNECTED" = true ]; then
    if ! ping -c 1 -W 5 8.8.8.8 > /dev/null 2>&1; then
        # Gateway reachable but no internet - still count as connected
        # (router might be up but ISP down, not our problem)
        true
    fi
fi

if [ "$CONNECTED" = true ]; then
    # Success - reset fail counter
    if [ "$FAIL_COUNT" -gt 0 ]; then
        log "WiFi recovered after $FAIL_COUNT failures (gateway: $GATEWAY)"
    fi
    echo 0 > "$FAIL_COUNT_FILE"

    # Log once per hour
    CURRENT_HOUR=$(date +"%Y-%m-%d %H")
    LAST_OK=$(grep "WiFi OK" "$LOGFILE" 2>/dev/null | tail -1 | cut -d: -f1-3 | xargs -I{} date -d "{}" +"%Y-%m-%d %H" 2>/dev/null)
    if [ "$LAST_OK" != "$CURRENT_HOUR" ]; then
        SIGNAL=$(iwconfig wlan0 2>/dev/null | grep -o 'Signal level=.*' | head -1)
        log "WiFi OK (gateway: $GATEWAY, $SIGNAL)"
    fi
    exit 0
fi

# --- WiFi is DOWN ---
FAIL_COUNT=$((FAIL_COUNT + 1))
echo "$FAIL_COUNT" > "$FAIL_COUNT_FILE"

if [ "$FAIL_COUNT" -le "$MAX_SOFT_RETRIES" ]; then
    # Level 1: Soft reconnect
    log "WARN: WiFi down (attempt $FAIL_COUNT/$MAX_SOFT_RETRIES) - soft reconnect"
    nmcli device wifi rescan 2>/dev/null
    sleep 3
    if [ -n "$WIFI_SSID" ] && [ -n "$WIFI_PASSWORD" ]; then
        nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PASSWORD" 2>> "$LOGFILE" || \
            wpa_cli -i wlan0 reconfigure 2>> "$LOGFILE"
    else
        log "WARN: WIFI_SSID/WIFI_PASSWORD not set in $ENV_FILE — trying wpa_cli reconfigure only"
        wpa_cli -i wlan0 reconfigure 2>> "$LOGFILE"
    fi

elif [ "$FAIL_COUNT" -le "$MAX_HARD_RETRIES" ]; then
    # Level 2: Hardware reset - rfkill cycle + driver reload
    log "WARN: WiFi down (attempt $FAIL_COUNT) - HARDWARE RESET"

    # Bring interface down
    ip link set wlan0 down 2>/dev/null
    sleep 1

    # Unblock wifi radio
    rfkill unblock wifi 2>/dev/null
    sleep 1

    # Reload the brcmfmac driver (fixes firmware crashes)
    modprobe -r brcmfmac 2>/dev/null
    sleep 3
    modprobe brcmfmac 2>/dev/null
    sleep 5

    # Bring interface back up
    ip link set wlan0 up 2>/dev/null
    sleep 2

    # Restart NetworkManager to pick up the interface
    systemctl restart NetworkManager 2>/dev/null
    sleep 5

    # Try to connect
    if [ -n "$WIFI_SSID" ] && [ -n "$WIFI_PASSWORD" ]; then
        nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PASSWORD" 2>> "$LOGFILE"
    else
        wpa_cli -i wlan0 reconfigure 2>> "$LOGFILE"
    fi
    log "Hardware reset complete, reconnect attempted"

else
    # Level 3: Nuclear - reboot
    log "CRITICAL: WiFi down for $((FAIL_COUNT * 2)) minutes. REBOOTING."
    sync
    sleep 1
    reboot
fi

# Keep log file from growing too large (keep last 200 lines)
if [ -f "$LOGFILE" ]; then
    tail -200 "$LOGFILE" > "${LOGFILE}.tmp" && mv "${LOGFILE}.tmp" "$LOGFILE"
fi
