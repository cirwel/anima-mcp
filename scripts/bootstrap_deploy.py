#!/usr/bin/env python3
"""
One-time bootstrap: deploy anima-mcp from GitHub zip. No git needed.
Run on Pi when rsync/SSH unavailable: curl -s https://raw.githubusercontent.com/CIRWEL/anima-mcp/main/scripts/bootstrap_deploy.py | python3
"""
import urllib.request
import zipfile
import shutil
import subprocess
import sys
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path("/home/unitares-anima/anima-mcp")
URL = "https://github.com/CIRWEL/anima-mcp/archive/refs/heads/main.zip"

def main():
    print("Downloading from GitHub...")
    zip_path = Path("/tmp/anima-mcp-main.zip")
    ext_path = Path("/tmp/anima-mcp-main")

    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ext_path.parent)
    zip_path.unlink(missing_ok=True)

    # The downloaded generation contains the backup implementation that will
    # accompany the new code. Run it before copying or restarting so changed
    # persistence code cannot be the first process to touch the only state.
    sys.path.insert(0, str(ext_path / "src"))
    try:
        from anima_mcp.state_snapshot import create_local_state_snapshot

        snapshot = create_local_state_snapshot()
        print("State recovery point:", snapshot)
    except Exception as exc:
        allow_fresh = os.environ.get("ANIMA_ALLOW_FRESH_START", "false").lower()
        if allow_fresh not in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "pre-restart state snapshot failed; set "
                "ANIMA_ALLOW_FRESH_START=true only for an intentional new identity"
            ) from exc
        print("WARNING: fresh-start override accepted; no prior state snapshot")

    print("Deploying to", REPO_ROOT)
    skip = {".venv", ".git", "__pycache__", ".env"}
    for item in ext_path.iterdir():
        if item.name in skip or item.name.endswith(".db"):
            continue
        dst = REPO_ROOT / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(
                item, dst,
                ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.db", ".env")
            )
        else:
            shutil.copy2(item, dst)
    shutil.rmtree(ext_path, ignore_errors=True)

    print("Synchronizing core systemd units...")
    for unit in (
        "anima-restore.service",
        "anima-broker.service",
        "anima.service",
        "anima-storage-maintenance.service",
        "anima-storage-maintenance.timer",
    ):
        subprocess.run(
            ["sudo", "install", "-m", "0644", str(REPO_ROOT / "systemd" / unit),
             str(Path("/etc/systemd/system") / unit)],
            timeout=15,
            check=True,
        )
    subprocess.run(
        ["sudo", "systemctl", "daemon-reload"], timeout=15, check=True
    )
    subprocess.run(
        [
            "sudo",
            "systemctl",
            "enable",
            "--now",
            "anima-storage-maintenance.timer",
        ],
        timeout=15,
        check=True,
    )

    # Scheduled maintenance timers (and retirement of their old crontab
    # lines). Non-fatal: a failure leaves the cron lines in place.
    print("Synchronizing scheduled jobs...")
    jobs = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "install_scheduled_jobs.sh")],
        timeout=120,
        check=False,
    )
    if jobs.returncode != 0:
        print("  scheduled job sync failed; cron entries left in place")

    # The Elixir sensor owner executes a compiled OTP release, not the synced
    # source tree. The helper is change-aware, skips unconfigured hosts, and
    # verifies fresh shadow state before returning after a rebuild/restart.
    print("Synchronizing Elixir sensor broker release...")
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "deploy_elixir_broker.sh")],
        timeout=600,
        check=True,
    )

    print("Restarting broker and anima services...")
    verification_started = time.time()
    subprocess.run(
        ["sudo", "systemctl", "restart", "anima-broker", "anima"],
        timeout=45,
        check=True,
    )
    for unit in ("anima-broker", "anima"):
        subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            timeout=15,
            check=True,
        )
    environment = subprocess.run(
        ["systemctl", "show", "anima", "-p", "Environment", "--value"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout.split()
    if "ANIMA_SENSORS_BACKEND=shm" not in environment:
        raise RuntimeError("anima service is not pinned to the SHM sensor backend")

    shm_path = Path("/dev/shm/anima_state.json")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            envelope = json.loads(shm_path.read_text())
            data = envelope.get("data", {})
            if (
                shm_path.stat().st_mtime >= verification_started
                and isinstance(data.get("readings"), dict)
                and isinstance(data.get("anima"), dict)
            ):
                break
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError("services started but no fresh broker state was published")

    print("Done. Services and fresh broker state verified.")

if __name__ == "__main__":
    main()
    sys.exit(0)
