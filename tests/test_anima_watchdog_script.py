"""Tests for scripts/anima-watchdog.sh.

Pins the display-silent-failure path added after the 2026-04-24 incident,
where anima-broker stayed `active` but the ST7789 latched off, logging
`[Errno 5] Input/output error` every ~2s. The prior watchdog only checked
`systemctl is-active` and missed it.

Strategy: rewrite LOGFILE/STATE_DIR to tmp, then stub systemctl/journalctl/
sudo via a PATH overlay so the script's real behavior runs unchanged.
"""
from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "anima-watchdog.sh"


def _rewrite_paths(tmp_path: Path, state_dir: Path, log: Path) -> Path:
    local = tmp_path / "anima-watchdog.sh"
    text = SCRIPT.read_text()
    text = text.replace(
        'LOGFILE="/home/unitares-anima/.anima/watchdog.log"',
        f'LOGFILE="{log}"',
    ).replace(
        'STATE_DIR="/tmp/anima-watchdog"',
        f'STATE_DIR="{state_dir}"',
    )
    local.write_text(text)
    local.chmod(0o755)
    return local


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/bash\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_stubs(
    bin_dir: Path,
    *,
    is_active: str = "active",
    journal_errors: int = 0,
    restart_exit: int = 0,
    sudo_calls: Path | None = None,
) -> None:
    """Create systemctl/journalctl/sudo/logger stubs on PATH."""
    bin_dir.mkdir(parents=True, exist_ok=True)

    sudo_log = sudo_calls or bin_dir / "sudo_calls.log"

    # systemctl: used for `is-active` and `show --property=ActiveExitTimestamp`
    _write_stub(
        bin_dir,
        "systemctl",
        f"""
if [ "$1" = "is-active" ]; then
    echo "{is_active}"
    [ "{is_active}" = "active" ] && exit 0 || exit 3
fi
if [ "$1" = "show" ]; then
    echo ""
    exit 0
fi
if [ "$1" = "restart" ]; then
    # `sudo systemctl restart ...` — sudo stub dispatches to this branch.
    echo "restart $@" >> "{sudo_log}"
    exit {restart_exit}
fi
exit 0
""".strip(),
    )

    # journalctl: emit N synthetic error lines
    _write_stub(
        bin_dir,
        "journalctl",
        f"""
for i in $(seq 1 {journal_errors}); do
    echo "Apr 24 20:29:20 LUMEN anima-broker[1614455]: [SafeCall] Error: [Errno 5] Input/output error"
done
exit 0
""".strip(),
    )

    # sudo: log the invocation, then exec the rest
    _write_stub(
        bin_dir,
        "sudo",
        f"""
echo "$@" >> "{sudo_log}"
exec "$@"
""".strip(),
    )

    # logger: no-op (script calls it for syslog)
    _write_stub(bin_dir, "logger", "exit 0")


def _run(script: Path, bin_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )


@pytest.fixture
def setup(tmp_path):
    state_dir = tmp_path / "state"
    log = tmp_path / "watchdog.log"
    bin_dir = tmp_path / "bin"
    script = _rewrite_paths(tmp_path, state_dir, log)
    return {"script": script, "state_dir": state_dir, "log": log, "bin": bin_dir}


class TestDisplaySilentFailure:
    def test_healthy_broker_no_errors_does_nothing(self, setup):
        _make_stubs(setup["bin"], is_active="active", journal_errors=0)
        result = _run(setup["script"], setup["bin"])
        assert result.returncode == 0

        sudo_log = setup["bin"] / "sudo_calls.log"
        assert not sudo_log.exists(), "should not restart a healthy broker"

    def test_error_flood_triggers_restart(self, setup):
        _make_stubs(setup["bin"], is_active="active", journal_errors=30)
        result = _run(setup["script"], setup["bin"])
        assert result.returncode == 0

        sudo_log = setup["bin"] / "sudo_calls.log"
        assert sudo_log.exists(), "flood should invoke sudo systemctl restart"
        content = sudo_log.read_text()
        assert "systemctl restart anima-broker anima" in content

        log_text = setup["log"].read_text()
        assert "display silent failure" in log_text
        assert "Errno 5 x30" in log_text

    def test_below_threshold_does_nothing(self, setup):
        # 9 < threshold (10) — should not restart
        _make_stubs(setup["bin"], is_active="active", journal_errors=9)
        result = _run(setup["script"], setup["bin"])
        assert result.returncode == 0

        sudo_log = setup["bin"] / "sudo_calls.log"
        assert not sudo_log.exists(), "below-threshold flood must not restart"

    def test_rate_limit_blocks_restart(self, setup):
        """A restart within MIN_RESTART_GAP (600s) must be skipped."""
        setup["state_dir"].mkdir(parents=True, exist_ok=True)
        # Simulate a restart that happened 60s ago
        (setup["state_dir"] / "anima-broker.last_restart").write_text(
            str(int(time.time()) - 60)
        )
        (setup["state_dir"] / "anima.last_restart").write_text(
            str(int(time.time()) - 60)
        )

        _make_stubs(setup["bin"], is_active="active", journal_errors=30)
        result = _run(setup["script"], setup["bin"])
        assert result.returncode == 0

        sudo_log = setup["bin"] / "sudo_calls.log"
        assert not sudo_log.exists(), "rate-limited flood must not restart"

        log_text = setup["log"].read_text()
        assert "restarted too recently" in log_text

    def test_inactive_broker_skipped_by_display_check(self, setup):
        """When broker is inactive, service-down path handles it — the display
        check must not double-fire."""
        _make_stubs(setup["bin"], is_active="inactive", journal_errors=30)
        result = _run(setup["script"], setup["bin"])
        assert result.returncode == 0

        # The inactive-service branch has its own rate-limit state handling;
        # what we assert here is that the display check alone didn't fire a
        # restart from the Errno 5 path. The log message for silent-failure
        # must not appear.
        log_text = setup["log"].read_text() if setup["log"].exists() else ""
        assert "display silent failure" not in log_text

    def test_script_has_executable_permission(self):
        mode = SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "anima-watchdog.sh must be executable"
