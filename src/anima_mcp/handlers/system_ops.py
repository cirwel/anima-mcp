"""System operations handlers — zero global state dependencies.

These handlers only use subprocess, pathlib, and stdlib modules.
They manage deployment, service control, networking, and power.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

from mcp.types import TextContent

from ..admin_auth import require_admin


RESTART_LOCKFILE = Path("/tmp/anima-restarting")
RESTART_WAIT_SECONDS = 120  # Callers must wait this long before retrying


def _pi_ssh_host() -> str:
    """Return Pi host/IP for user-facing SSH instructions."""
    return os.environ.get("ANIMA_PI_HOST", "100.78.71.1")


async def _delayed_restart():
    """Restart both anima services after a delay that lets the response complete.

    Writes a lockfile BEFORE dispatching, so any caller that reconnects early
    can detect that a restart is in progress and back off.

    Dispatches 'systemctl restart anima' to systemd via Popen.
    This is an explicit restart, so PartOf=anima.service on the broker's
    unit file causes the broker to also restart. systemd (PID 1) receives
    the D-Bus command before it kills our cgroup, so the restart proceeds
    even after our process dies.
    """
    # Write lockfile before restart — survives the process dying
    try:
        import time
        RESTART_LOCKFILE.write_text(json.dumps({
            "restart_at": time.time(),
            "wait_seconds": RESTART_WAIT_SECONDS,
        }))
    except Exception:
        pass  # Best-effort — don't block restart if this fails

    # 5-second delay ensures the MCP response is fully sent before we die
    await asyncio.sleep(5)
    subprocess.Popen(
        ["sudo", "systemctl", "restart", "anima"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _sync_systemd_services(repo_root: Path) -> list[str]:
    """Sync systemd service/timer/mount files from repo to /etc/systemd/system/ if changed."""
    synced = []
    systemd_dir = repo_root / "systemd"
    target_dir = Path("/etc/systemd/system")
    if not systemd_dir.exists():
        return synced

    # Units in this set are auto-installed even if they don't exist yet in /etc/systemd/system/
    AUTO_INSTALL = {"anima-restore.service"}

    for pattern in ("*.service", "*.timer", "*.mount"):
        for unit_file in systemd_dir.glob(pattern):
            target = target_dir / unit_file.name
            try:
                repo_content = unit_file.read_text()

                if not target.exists():
                    # Only auto-install whitelisted units
                    if unit_file.name not in AUTO_INSTALL:
                        continue
                    result = subprocess.run(
                        ["sudo", "cp", str(unit_file), str(target)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        # Enable the new unit
                        subprocess.run(
                            ["sudo", "systemctl", "enable", unit_file.name],
                            capture_output=True, text=True, timeout=10
                        )
                        synced.append(f"{unit_file.name} (installed)")
                    continue

                target_content = target.read_text()
                if repo_content != target_content:
                    result = subprocess.run(
                        ["sudo", "cp", str(unit_file), str(target)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        synced.append(unit_file.name)
            except Exception:
                continue

    if synced:
        subprocess.run(
            ["sudo", "systemctl", "daemon-reload"],
            capture_output=True, timeout=10
        )
    return synced


async def handle_git_pull(arguments: dict) -> list[TextContent]:
    """
    Pull latest code from git and optionally restart.
    Enables remote deployments via MCP without SSH.
    """
    if err := require_admin():
        return err
    restart = arguments.get("restart", False)
    stash = arguments.get("stash", False)  # Stash local changes before pull
    force = arguments.get("force", False)  # Hard reset to remote (DANGER: loses local changes)

    # Find repo root (where .git is)
    repo_root = Path(__file__).parent.parent.parent.parent  # anima-mcp/
    git_dir = repo_root / ".git"

    if not git_dir.exists():
        # Bootstrap: deploy from GitHub zip (no git needed — for Pi set up via rsync without .git)
        try:
            import urllib.request
            import zipfile
            import shutil

            url = "https://github.com/CIRWEL/anima-mcp/archive/refs/heads/main.zip"
            zip_path = Path("/tmp") / "anima-mcp-main.zip"
            ext_path = Path("/tmp") / "anima-mcp-main"

            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(ext_path.parent)
            zip_path.unlink(missing_ok=True)

            src = ext_path
            skip = {".venv", ".git", "__pycache__", ".env"}
            for item in src.iterdir():
                if item.name in skip or item.name.endswith(".db"):
                    continue
                dst = repo_root / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(item, dst, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.db", ".env"))
                else:
                    shutil.copy2(item, dst)
            shutil.rmtree(ext_path, ignore_errors=True)

            # Sync systemd service files if changed
            synced_services = _sync_systemd_services(repo_root)

            output = {"success": True, "bootstrap": "Deployed from GitHub zip", "repo": str(repo_root)}
            if synced_services:
                output["synced_services"] = synced_services
            if restart:
                output["restart"] = "Restarting in ~5 seconds."
                output["wait_seconds"] = RESTART_WAIT_SECONDS
                output["warning"] = (
                    f"Do NOT attempt SSH or MCP contact for {RESTART_WAIT_SECONDS} seconds. "
                    "This response confirms the restart was scheduled successfully. "
                    "Any 'fetch failed' or timeout after this is expected — the server is restarting."
                )
                asyncio.create_task(_delayed_restart())
            return [TextContent(type="text", text=json.dumps(output, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Bootstrap (zip deploy) failed: {e}",
                "repo": str(repo_root),
            }))]

    try:
        # Stash local changes if requested (only when .git exists)
        if stash:
            subprocess.run(
                ["git", "stash", "push", "-m", "Auto-stash before git_pull"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            # Continue even if stash fails (might be nothing to stash)

        # Hard reset if force requested (DANGER)
        if force:
            subprocess.run(
                ["git", "fetch", "origin", "main"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )

        # Git fetch + pull
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip() if result.stderr else None,
            "repo": str(repo_root),
        }

        if result.returncode == 0:
            # Check what changed
            diff_result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            output["latest_commit"] = diff_result.stdout.strip()

            # Sync systemd service files if changed
            synced_services = _sync_systemd_services(repo_root)
            if synced_services:
                output["synced_services"] = synced_services

            if restart:
                output["restart"] = "Restarting in ~5 seconds."
                output["wait_seconds"] = RESTART_WAIT_SECONDS
                output["warning"] = (
                    f"Do NOT attempt SSH or MCP contact for {RESTART_WAIT_SECONDS} seconds. "
                    "This response confirms the restart was scheduled successfully. "
                    "Any 'fetch failed' or timeout after this is expected — the server is restarting."
                )
                asyncio.create_task(_delayed_restart())
            else:
                output["note"] = "Changes pulled. Use restart=true to apply, or manually restart."

        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "error": "Git pull timed out"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Git pull failed: {e}"
        }))]


async def handle_system_service(arguments: dict) -> list[TextContent]:
    """
    Manage system services (systemctl).
    Enables remote control of rpi-connect, anima, and other services.
    """
    if err := require_admin():
        return err
    service = arguments.get("service")
    action = arguments.get("action", "status")

    if not service:
        return [TextContent(type="text", text=json.dumps({
            "error": "service parameter required"
        }))]

    # Whitelist of allowed services for security
    ALLOWED_SERVICES = [
        "rpi-connect",
        "rpi-connect-wayvnc",
        "anima",
        "anima-broker",
        "anima-mcp",
        "ssh",
        "sshd",
    ]

    if service not in ALLOWED_SERVICES:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Service '{service}' not in allowed list",
            "allowed": ALLOWED_SERVICES
        }))]

    ALLOWED_ACTIONS = ["status", "start", "stop", "restart", "enable", "disable"]
    if action not in ALLOWED_ACTIONS:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Action '{action}' not allowed",
            "allowed": ALLOWED_ACTIONS
        }))]

    try:
        # For rpi-connect, use the rpi-connect CLI for some actions
        if service == "rpi-connect" and action in ["start", "restart"]:
            # Try rpi-connect on first
            rpi_result = subprocess.run(
                ["rpi-connect", "on"],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = {
                "success": rpi_result.returncode == 0,
                "service": service,
                "action": "rpi-connect on",
                "stdout": rpi_result.stdout.strip(),
                "stderr": rpi_result.stderr.strip() if rpi_result.stderr else None,
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        # Standard systemctl for other cases
        cmd = ["systemctl", action, service]
        if action != "status":
            cmd = ["sudo", "systemctl", action, service]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = {
            "success": result.returncode == 0,
            "service": service,
            "action": action,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip() if result.stderr else None,
        }

        # For status, also check if service is active
        if action == "status":
            is_active = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True,
                text=True,
                timeout=10
            )
            output["is_active"] = is_active.stdout.strip() == "active"

        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Command timed out for {service}"
        }))]
    except FileNotFoundError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Command not found: {e}"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"System service command failed: {e}"
        }))]


async def handle_fix_ssh_port(arguments: dict) -> list[TextContent]:
    """
    Switch SSH to port 2222/22222 (headless fix when port 22 is blocked), or reset to port 22.
    Call via HTTP when SSH times out: avoids need for keyboard/monitor.
    Use port=22 to remove alternate port lines and restore default (22).
    """
    if err := require_admin():
        return err
    port = arguments.get("port", 2222)
    if port not in (22, 2222, 22222):
        pi_host = _pi_ssh_host()
        return [TextContent(type="text", text=json.dumps({
            "error": "port must be 22, 2222, or 22222",
            "usage_2222": f"ssh -p 2222 -i ~/.ssh/id_ed25519_pi unitares-anima@{pi_host}",
            "usage_22": f"ssh -i ~/.ssh/id_ed25519_pi unitares-anima@{pi_host}",
        }))]

    try:
        if port == 22:
            pi_host = _pi_ssh_host()
            # Reset to default: remove Port 2222 and Port 22222 lines from sshd_config
            sed = subprocess.run(
                ["sudo", "sed", "-i.bak", "/^Port 2222$/d; /^Port 22222$/d", "/etc/ssh/sshd_config"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if sed.returncode != 0:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": f"Failed to edit sshd_config: {sed.stderr}"
                }))]
            restart = subprocess.run(
                ["sudo", "systemctl", "restart", "ssh"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return [TextContent(type="text", text=json.dumps({
                "success": restart.returncode == 0,
                "port": 22,
                "message": "SSH reset to port 22 (default). Connect with:",
                "connect": f"ssh -i ~/.ssh/id_ed25519_pi unitares-anima@{pi_host}",
                "stderr": restart.stderr.strip() if restart.stderr else None,
            }))]

        # Switch to 2222 or 22222
        check = subprocess.run(
            ["grep", "-q", f"^Port {port}", "/etc/ssh/sshd_config"],
            capture_output=True,
            timeout=5,
        )
        if check.returncode == 0:
            subprocess.run(
                ["sudo", "systemctl", "restart", "ssh"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            pi_host = _pi_ssh_host()
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "message": f"SSH already on port {port}, restarted",
                "connect": f"ssh -p {port} -i ~/.ssh/id_ed25519_pi unitares-anima@{pi_host}",
            }))]

        echo = subprocess.run(
            ["sh", "-c", f"echo 'Port {port}' | sudo tee -a /etc/ssh/sshd_config"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if echo.returncode != 0:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Failed to update sshd_config: {echo.stderr}"
            }))]

        restart = subprocess.run(
            ["sudo", "systemctl", "restart", "ssh"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        pi_host = _pi_ssh_host()
        return [TextContent(type="text", text=json.dumps({
            "success": restart.returncode == 0,
            "port": port,
            "message": f"SSH now on port {port}. Connect with:",
            "connect": f"ssh -p {port} -i ~/.ssh/id_ed25519_pi unitares-anima@{pi_host}",
            "stderr": restart.stderr.strip() if restart.stderr else None,
        }))]
    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "Command timed out"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e)
        }))]


async def handle_deploy_from_github(arguments: dict) -> list[TextContent]:
    """
    Deploy latest code from GitHub via zip download. No git required.
    Use when git_pull fails (no .git) or to force-refresh from main.
    """
    if err := require_admin():
        return err
    import urllib.request
    import zipfile
    import shutil

    restart = arguments.get("restart", True)
    repo_root = Path(__file__).parent.parent.parent.parent  # anima-mcp/

    try:
        url = "https://github.com/CIRWEL/anima-mcp/archive/refs/heads/main.zip"
        zip_path = Path("/tmp") / "anima-mcp-main.zip"
        ext_path = Path("/tmp") / "anima-mcp-main"

        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ext_path.parent)
        zip_path.unlink(missing_ok=True)

        src = ext_path
        skip = {".venv", ".git", "__pycache__", ".env"}
        for item in src.iterdir():
            if item.name in skip or item.name.endswith(".db"):
                continue
            dst = repo_root / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.db", ".env"))
            else:
                shutil.copy2(item, dst)
        shutil.rmtree(ext_path, ignore_errors=True)

        output = {"success": True, "message": "Deployed from GitHub", "repo": str(repo_root)}
        if restart:
            output["restart"] = "Restarting in ~5 seconds."
            output["wait_seconds"] = RESTART_WAIT_SECONDS
            output["warning"] = (
                f"Do NOT attempt SSH or MCP contact for {RESTART_WAIT_SECONDS} seconds. "
                "This response confirms the restart was scheduled successfully. "
                "Any 'fetch failed' or timeout after this is expected — the server is restarting."
            )
            asyncio.create_task(_delayed_restart())
        return [TextContent(type="text", text=json.dumps(output, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e),
            "repo": str(repo_root),
        }))]


async def handle_setup_tailscale(arguments: dict) -> list[TextContent]:
    """
    Install and activate Tailscale on Pi for direct VPN access.
    Call via HTTP when SSH unavailable. Requires auth_key for headless.
    Get key: https://login.tailscale.com/admin/settings/keys
    """
    if err := require_admin():
        return err
    auth_key = arguments.get("auth_key", "").strip()
    if not auth_key:
        return [TextContent(type="text", text=json.dumps({
            "error": "auth_key required for headless setup",
            "hint": "Get at https://login.tailscale.com/admin/settings/keys (reusable, 90 days)",
            "usage": "Call with auth_key=tskey-auth-xxx"
        }))]

    if not auth_key.startswith("tskey-"):
        return [TextContent(type="text", text=json.dumps({
            "error": "Invalid auth_key format (should start with tskey-)"
        }))]

    try:
        # Install Tailscale
        install = subprocess.run(
            ["sh", "-c", "curl -fsSL https://tailscale.com/install.sh | sh"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if install.returncode != 0:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Install failed: {install.stderr or install.stdout}"
            }))]

        # Activate with auth key
        up = subprocess.run(
            ["sudo", "tailscale", "up", "--authkey=" + auth_key],
            capture_output=True,
            text=True,
            timeout=60
        )

        if up.returncode != 0:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": up.stderr.strip() or up.stdout.strip() or "tailscale up failed",
                "hint": "Auth key may be expired or invalid"
            }))]

        # Get Tailscale IP
        ip_result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=10
        )
        ts_ip = ip_result.stdout.strip().split("\n")[0] if ip_result.stdout else None

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": "Tailscale active. Use 100.x.x.x for MCP/SSH.",
            "tailscale_ip": ts_ip,
            "mcp_url": f"http://{ts_ip}:8766/mcp/" if ts_ip else None,
            "connect": f"ssh -i ~/.ssh/id_ed25519_pi unitares-anima@{ts_ip}" if ts_ip else None,
        }))]
    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "Command timed out"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e)
        }))]


async def handle_system_power(arguments: dict) -> list[TextContent]:
    """
    Reboot or shutdown the Pi remotely.
    Useful for recovery when services are stuck.
    """
    if err := require_admin():
        return err
    action = arguments.get("action", "status")
    confirm = arguments.get("confirm", False)

    ALLOWED_ACTIONS = ["status", "reboot", "shutdown"]
    if action not in ALLOWED_ACTIONS:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Action '{action}' not allowed",
            "allowed": ALLOWED_ACTIONS
        }))]

    try:
        if action == "status":
            # Get uptime and load
            uptime = subprocess.run(
                ["uptime"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return [TextContent(type="text", text=json.dumps({
                "action": "status",
                "uptime": uptime.stdout.strip(),
            }, indent=2))]

        # Reboot and shutdown require confirmation
        if not confirm:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Action '{action}' requires confirm=true",
                "warning": "This will disconnect all sessions. Are you sure?",
                "hint": f"Call again with confirm=true to {action}"
            }, indent=2))]

        if action == "reboot":
            # Schedule reboot in 5 seconds to allow response to be sent
            subprocess.Popen(
                ["sudo", "shutdown", "-r", "+0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "action": "reboot",
                "message": "Rebooting now. Pi will be back in ~2 minutes.",
                "wait_seconds": RESTART_WAIT_SECONDS,
                "warning": (
                    f"Do NOT attempt SSH or MCP contact for {RESTART_WAIT_SECONDS} seconds. "
                    "Any connection attempt during reboot can destabilize WiFi."
                ),
            }, indent=2))]

        elif action == "shutdown":
            subprocess.Popen(
                ["sudo", "shutdown", "-h", "+0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "action": "shutdown",
                "message": "Shutting down. Manual power cycle required to restart."
            }, indent=2))]

    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "error": "Command timed out"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Power command failed: {e}"
        }))]
