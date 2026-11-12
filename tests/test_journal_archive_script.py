"""Tests for scripts/journal-archive.sh — the volatile-journal escape hatch.

journald on the Pi is RAM-backed and high-churn logging rotates it in under a
day, which is why the 2026-03-28 day-summary writer death (#188) could not be
root-caused five months later. The script exports new entries hourly into
~/.anima/journal-archive/ so the existing slim backup mirror carries them
off-host.

Strategy mirrors test_lumen_heartbeat_script.py: the script takes every path
and binary from the environment, so no rewriting is needed. journalctl is
stubbed with a recorder that emits configured output and honors a failure
switch; the externally visible behaviour that matters is which files exist,
that they are zcat-readable, and that failure exits non-zero.
"""
from __future__ import annotations

import gzip
import os
import subprocess
import time
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "journal-archive.sh"


@pytest.fixture
def rig(tmp_path):
    """Env + a journalctl stub that records its args and emits JOURNAL_LINES."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "journalctl-calls.txt"
    stub = bin_dir / "journalctl"
    # Emits the contents of $JOURNAL_LINES_FILE (empty if unset/missing),
    # records argv, fails on JOURNALCTL_SHOULD_FAIL. Touches the path given
    # via --cursor-file= the way real journalctl updates it on success.
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{calls}"\n'
        '[ -n "${JOURNALCTL_SHOULD_FAIL:-}" ] && exit 1\n'
        'for a in "$@"; do case "$a" in --cursor-file=*)\n'
        '    echo "s=fakecursor" > "${a#--cursor-file=}";; esac; done\n'
        'if [ -n "${JOURNAL_LINES_FILE:-}" ] && [ -f "$JOURNAL_LINES_FILE" ]; then\n'
        '    cat "$JOURNAL_LINES_FILE"\nfi\nexit 0\n'
    )
    stub.chmod(0o755)

    archive = tmp_path / "journal-archive"
    lines_file = tmp_path / "journal-lines.txt"

    def run(extra_env=None):
        env = dict(os.environ)
        env["JOURNAL_ARCHIVE_DIR"] = str(archive)
        env["JOURNAL_ARCHIVE_JOURNALCTL"] = str(stub)
        env["JOURNAL_LINES_FILE"] = str(lines_file)
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
        )

    class Rig:
        pass

    r = Rig()
    r.tmp = tmp_path
    r.archive = archive
    r.calls = calls
    r.lines_file = lines_file
    r.run = run
    r.day_file = archive / f"journal-{date.today().isoformat()}.log.gz"
    return r


def read_gz(path: Path) -> str:
    """Read a possibly multi-member gzip file the way zcat does."""
    with gzip.open(path, "rt") as fh:
        return fh.read()


class TestExport:
    def test_new_entries_land_in_day_file(self, rig):
        rig.lines_file.write_text("Aug 27 03:00:00 LUMEN kernel: hello\n")
        result = rig.run()
        assert result.returncode == 0, result.stderr
        assert rig.day_file.exists()
        assert "kernel: hello" in read_gz(rig.day_file)

    def test_empty_export_appends_nothing(self, rig):
        rig.lines_file.write_text("")
        result = rig.run()
        assert result.returncode == 0, result.stderr
        assert not rig.day_file.exists()

    def test_two_runs_append_two_readable_members(self, rig):
        rig.lines_file.write_text("first window\n")
        assert rig.run().returncode == 0
        rig.lines_file.write_text("second window\n")
        assert rig.run().returncode == 0
        content = read_gz(rig.day_file)
        assert "first window" in content
        assert "second window" in content

    def test_cursor_advances_only_via_scratch_promotion(self, rig):
        """journalctl runs against a scratch cursor; the real one is promoted
        after the append lands. The stub writes s=fakecursor to whatever
        --cursor-file= path it is handed."""
        rig.lines_file.write_text("x\n")
        cursor = rig.tmp / "custom.cursor"
        cursor.write_text("s=old\n")
        result = rig.run({"JOURNAL_ARCHIVE_CURSOR": str(cursor)})
        assert result.returncode == 0, result.stderr
        # journalctl was NOT handed the real cursor path...
        assert f"--cursor-file={cursor}" not in rig.calls.read_text()
        assert "--cursor-file=" in rig.calls.read_text()
        # ...but the real cursor carries the advanced value after promotion.
        assert cursor.read_text() == "s=fakecursor\n"


class TestFailureDirection:
    def test_journalctl_failure_exits_nonzero_without_partial_append(self, rig):
        rig.lines_file.write_text("would be lost\n")
        result = rig.run({"JOURNALCTL_SHOULD_FAIL": "1"})
        assert result.returncode != 0
        assert "journalctl export failed" in result.stderr
        assert not rig.day_file.exists()

    def test_append_failure_does_not_advance_cursor(self, rig):
        """The finding-1 scenario: journalctl succeeds (scratch cursor written)
        but the append fails — the real cursor must NOT advance, so the window
        is retried next run instead of permanently skipped."""
        rig.archive.mkdir(parents=True)
        # Make the append fail in a way no uid can bypass: `>>` onto a
        # directory is EISDIR for everyone. This used to chmod the archive dir
        # to 0555, which root ignores — so the test failed in any container
        # that runs as root while passing on CI (2026-11-12).
        rig.day_file.mkdir()
        cursor = rig.tmp / "cursor"
        cursor.write_text("s=old\n")
        rig.lines_file.write_text("entries that must not be skipped\n")
        result = rig.run({"JOURNAL_ARCHIVE_CURSOR": str(cursor)})
        assert result.returncode != 0
        assert "append" in result.stderr
        assert cursor.read_text() == "s=old\n"

    def test_failure_writes_marker_and_success_clears_it(self, rig):
        rig.lines_file.write_text("x\n")
        result = rig.run({"JOURNALCTL_SHOULD_FAIL": "1"})
        assert result.returncode != 0
        marker = rig.archive / ".last_failure"
        assert marker.exists()
        assert "journalctl export failed" in marker.read_text()
        result = rig.run()
        assert result.returncode == 0, result.stderr
        assert not marker.exists()


class TestRetention:
    def test_old_files_pruned_recent_kept(self, rig):
        rig.archive.mkdir(parents=True)
        old = rig.archive / "journal-2026-01-01.log.gz"
        old.write_bytes(gzip.compress(b"ancient\n"))
        stale = time.time() - 20 * 86400
        os.utime(old, (stale, stale))
        recent = rig.archive / "journal-2026-08-20.log.gz"
        recent.write_bytes(gzip.compress(b"recent\n"))
        rig.lines_file.write_text("")
        assert rig.run().returncode == 0
        assert not old.exists()
        assert recent.exists()

    def test_size_cap_prunes_oldest_first_and_announces(self, rig):
        rig.archive.mkdir(parents=True)
        oldest = rig.archive / "journal-2026-08-01.log.gz"
        newest = rig.archive / "journal-2026-08-20.log.gz"
        # ~600KB each: together they exceed the 1MB cap, but the survivor
        # plus directory overhead stays under it (du block-rounds upward).
        oldest.write_bytes(os.urandom(600 * 1024))
        newest.write_bytes(os.urandom(600 * 1024))
        rig.lines_file.write_text("")
        result = rig.run({"JOURNAL_ARCHIVE_MAX_MB": "1"})
        assert result.returncode == 0, result.stderr
        assert not oldest.exists()
        assert newest.exists()
        assert "size cap" in result.stderr

    def test_todays_file_is_never_pruned(self, rig):
        """Today's file may be the sole copy the daily mirror has not yet
        carried off-host; a storm day exceeds the cap loudly instead."""
        rig.archive.mkdir(parents=True)
        rig.day_file.write_bytes(os.urandom(2 * 1024 * 1024))
        rig.lines_file.write_text("")
        result = rig.run({"JOURNAL_ARCHIVE_MAX_MB": "1"})
        assert result.returncode == 0, result.stderr
        assert rig.day_file.exists()
        assert "sole un-mirrored copy" in result.stderr

    def test_stuck_delete_exits_loud_instead_of_spinning(self, rig):
        rig.archive.mkdir(parents=True)
        victim = rig.archive / "journal-2026-08-01.log.gz"
        # `rm -f` refuses a non-empty directory for every uid, root included;
        # a 0555 archive dir (the old simulation) does not stop root.
        victim.mkdir()
        (victim / "payload").write_bytes(os.urandom(600 * 1024))
        rig.lines_file.write_text("")
        result = rig.run({
            "JOURNAL_ARCHIVE_MAX_MB": "0",
            "JOURNAL_ARCHIVE_CURSOR": str(rig.tmp / "cursor"),
        })
        assert result.returncode != 0
        assert "could not delete" in result.stderr
        assert victim.exists()
