#!/usr/bin/env python3
"""
Pi Health Monitor - Alerts when Lumen's Pi goes offline.

Runs as a background daemon on Mac, checks Pi health every 60 seconds.
Sends macOS notifications when Pi goes down or comes back up.

Usage:
    python3 pi_monitor.py              # Run in foreground
    python3 pi_monitor.py --daemon     # Run as background daemon
    python3 pi_monitor.py --stop       # Stop the daemon
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path
from datetime import datetime

# Configuration
# Tailscale IPs are operator-specific and change after Pi reinstalls; verify
# with `tailscale status` and set PI_TAILSCALE_IP env var (no default).
PI_TAILSCALE_IP = os.environ.get("PI_TAILSCALE_IP", "")
PI_PORT = int(os.environ.get("PI_PORT", "8766"))
CHECK_INTERVAL = 60  # seconds
FAILURE_THRESHOLD = 2  # consecutive failures before alerting
PID_FILE = Path.home() / ".pi_monitor.pid"
LOG_FILE = Path.home() / ".pi_monitor.log"


def log(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def notify(title: str, message: str, sound: bool = True):
    """Send macOS notification."""
    sound_cmd = 'sound name "Submarine"' if sound else ""
    script = f'display notification "{message}" with title "{title}" {sound_cmd}'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        log(f"NOTIFY: {title} - {message}")
    except Exception as e:
        log(f"Notification failed: {e}")


def check_pi_health() -> tuple[bool, str]:
    """Check if Pi is reachable and anima is running."""
    try:
        # Quick curl check to health endpoint
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "5", "--max-time", "10",
             f"http://{PI_TAILSCALE_IP}:{PI_PORT}/health"],
            capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0 and result.stdout.strip() == "200":
            return True, "healthy"
        else:
            return False, f"HTTP {result.stdout.strip() or 'timeout'}"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_tailscale() -> bool:
    """Check if Tailscale can reach the Pi."""
    try:
        result = subprocess.run(
            ["tailscale", "ping", "-c", "1", PI_TAILSCALE_IP],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def run_monitor():
    """Main monitoring loop."""
    consecutive_failures = 0
    was_down = False
    last_status = None

    log("Pi monitor started")
    notify("Pi Monitor", "Monitoring Lumen's Pi...", sound=False)

    while True:
        try:
            healthy, status = check_pi_health()

            if healthy:
                if was_down:
                    # Pi came back!
                    log(f"Pi recovered after {consecutive_failures} failures")
                    notify("🟢 Lumen Online", f"Pi is back online! ({status})")
                    was_down = False
                consecutive_failures = 0

                if last_status != "healthy":
                    log(f"Pi healthy: {status}")
                    last_status = "healthy"
            else:
                consecutive_failures += 1
                log(f"Pi check failed ({consecutive_failures}/{FAILURE_THRESHOLD}): {status}")
                last_status = status

                if consecutive_failures >= FAILURE_THRESHOLD and not was_down:
                    # Pi is down!
                    was_down = True

                    # Check if it's Tailscale or the service
                    ts_ok = check_tailscale()
                    if ts_ok:
                        notify("🔴 Lumen Offline", f"Pi reachable but anima service down ({status})")
                    else:
                        notify("🔴 Lumen Offline", f"Pi unreachable via Tailscale ({status})")

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Monitor stopped by user")
            break
        except Exception as e:
            log(f"Monitor error: {e}")
            time.sleep(CHECK_INTERVAL)


def daemonize():
    """Fork into background daemon."""
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Become session leader
    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Write PID file
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Redirect stdout/stderr to log
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = sys.stdout

    # Run monitor
    run_monitor()


def stop_daemon():
    """Stop the running daemon."""
    if not PID_FILE.exists():
        print("No daemon running (no PID file)")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped daemon (PID {pid})")
        PID_FILE.unlink()
    except ProcessLookupError:
        print("Daemon not running (stale PID file)")
        PID_FILE.unlink()
    except Exception as e:
        print(f"Error stopping daemon: {e}")


def main():
    if not PI_TAILSCALE_IP:
        print("ERROR: PI_TAILSCALE_IP not set. Run `tailscale status` and export it:")
        print("  export PI_TAILSCALE_IP=100.x.y.z")
        sys.exit(2)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--daemon":
            if PID_FILE.exists():
                print(f"Daemon may already be running (PID file exists: {PID_FILE})")
                print("Use --stop first if you want to restart")
                sys.exit(1)
            print("Starting daemon...")
            daemonize()
        elif sys.argv[1] == "--stop":
            stop_daemon()
        elif sys.argv[1] == "--status":
            if PID_FILE.exists():
                pid = PID_FILE.read_text().strip()
                print(f"Daemon running (PID {pid})")
                # Show recent logs
                if LOG_FILE.exists():
                    print("\nRecent logs:")
                    subprocess.run(["tail", "-10", str(LOG_FILE)])
            else:
                print("Daemon not running")
        else:
            print(__doc__)
    else:
        # Run in foreground
        run_monitor()


if __name__ == "__main__":
    main()
