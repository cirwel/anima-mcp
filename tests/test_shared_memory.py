"""Tests for SharedMemoryClient — read/write cycles, error handling, atomicity."""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from anima_mcp.shared_memory import SharedMemoryClient


class TestSharedMemoryInit:
    """Test initialization and backend selection."""

    def test_default_init_read_mode(self, tmp_path):
        """Default mode is read, backend is file."""
        filepath = tmp_path / "state.json"
        client = SharedMemoryClient(mode="read", filepath=filepath)
        assert client.mode == "read"
        assert client.backend == "file"
        assert client.filepath == filepath

    def test_write_mode(self, tmp_path):
        """Write mode is accepted."""
        filepath = tmp_path / "state.json"
        client = SharedMemoryClient(mode="write", filepath=filepath)
        assert client.mode == "write"

    def test_backend_always_file(self, tmp_path):
        """Any backend string resolves to 'file'."""
        filepath = tmp_path / "state.json"
        client = SharedMemoryClient(backend="auto", filepath=filepath)
        assert client.backend == "file"

        client2 = SharedMemoryClient(backend="posix_ipc", filepath=filepath)
        assert client2.backend == "file"

    def test_creates_parent_directory(self, tmp_path):
        """Init creates parent directory if it does not exist."""
        filepath = tmp_path / "sub" / "dir" / "state.json"
        SharedMemoryClient(filepath=filepath)
        assert filepath.parent.exists()

    def test_existing_directory_ok(self, tmp_path):
        """Init succeeds when parent directory already exists."""
        filepath = tmp_path / "state.json"
        # tmp_path already exists
        client = SharedMemoryClient(filepath=filepath)
        assert client.filepath == filepath


class TestSharedMemoryWrite:
    """Test write operations."""

    def test_basic_write_read_cycle(self, tmp_path):
        """Write data, read it back, verify round-trip."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        data = {"warmth": 0.5, "clarity": 0.8}
        result = writer.write(data)

        assert result is True
        read_back = reader.read()
        assert read_back == data

    def test_write_complex_nested_data(self, tmp_path):
        """Write nested dicts, lists, mixed types."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        data = {
            "readings": {"cpu_temp_c": 55.0, "eeg_delta_power": 0.3},
            "anima": {"warmth": 0.36, "clarity": 0.73, "stability": 0.5, "presence": 0.8},
            "activity": {"level": "active", "reason": "engaged"},
            "learning": {
                "preferences": {"satisfaction": 0.87},
                "self_beliefs": {"stability_recovery": {"confidence": 0.68}},
                "agency": {"action_values": {"focus_attention": 0.22}},
            },
            "tags": ["sensor", "neural", "ambient"],
            "count": 42,
            "active": True,
            "nothing": None,
        }

        assert writer.write(data) is True
        read_back = reader.read()
        assert read_back == data

    def test_write_creates_envelope(self, tmp_path):
        """Written file contains envelope with updated_at, pid, data."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        writer.write({"key": "value"})

        raw = json.loads(filepath.read_text())
        assert "updated_at" in raw
        assert "pid" in raw
        assert raw["pid"] == os.getpid()
        assert raw["data"] == {"key": "value"}

    def test_write_overwrites_previous(self, tmp_path):
        """Second write replaces first write completely."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        writer.write({"version": 1})
        writer.write({"version": 2})

        assert reader.read() == {"version": 2}

    def test_read_only_client_cannot_write(self, tmp_path):
        """Read-only client raises PermissionError on write."""
        filepath = tmp_path / "state.json"
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        with pytest.raises(PermissionError, match="read-only mode"):
            reader.write({"data": "forbidden"})

    def test_write_returns_false_on_filesystem_error(self, tmp_path):
        """Write returns False (not crash) on filesystem errors."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        # Make directory read-only to trigger write failure
        with patch("builtins.open", side_effect=PermissionError("disk full")):
            result = writer.write({"data": 1})
            assert result is False


class TestSharedMemoryRead:
    """Test read operations."""

    def test_read_nonexistent_file_returns_none(self, tmp_path):
        """Reading when file does not exist returns None, not crash."""
        filepath = tmp_path / "nonexistent.json"
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        result = reader.read()
        assert result is None

    def test_read_malformed_json_returns_none(self, tmp_path):
        """Corrupt JSON file returns None, not crash."""
        filepath = tmp_path / "state.json"
        filepath.write_text("this is not json {{{")

        reader = SharedMemoryClient(mode="read", filepath=filepath)
        # Create the lock file so read doesn't fail on lock
        filepath.with_suffix(".lock").touch()

        result = reader.read()
        assert result is None

    def test_read_empty_file_returns_none(self, tmp_path):
        """Empty file returns None."""
        filepath = tmp_path / "state.json"
        filepath.write_text("")

        reader = SharedMemoryClient(mode="read", filepath=filepath)
        filepath.with_suffix(".lock").touch()

        result = reader.read()
        assert result is None

    def test_read_valid_envelope_no_data_key(self, tmp_path):
        """JSON file without 'data' key returns None (via .get)."""
        filepath = tmp_path / "state.json"
        filepath.write_text('{"updated_at": "2026-01-01", "pid": 1}')

        reader = SharedMemoryClient(mode="read", filepath=filepath)
        filepath.with_suffix(".lock").touch()

        result = reader.read()
        assert result is None

    def test_read_extracts_data_from_envelope(self, tmp_path):
        """Read returns just the 'data' portion of the envelope."""
        filepath = tmp_path / "state.json"
        envelope = {
            "updated_at": "2026-02-22T10:00:00",
            "pid": 12345,
            "data": {"warmth": 0.7, "clarity": 0.9}
        }
        filepath.write_text(json.dumps(envelope))

        reader = SharedMemoryClient(mode="read", filepath=filepath)
        filepath.with_suffix(".lock").touch()

        result = reader.read()
        assert result == {"warmth": 0.7, "clarity": 0.9}

    def test_read_retries_on_json_decode_error(self, tmp_path):
        """Read retries up to 3 times on JSONDecodeError."""
        filepath = tmp_path / "state.json"
        filepath.write_text("invalid json")
        filepath.with_suffix(".lock").touch()

        reader = SharedMemoryClient(mode="read", filepath=filepath)

        # With all 3 retries hitting bad JSON, should return None
        with patch("anima_mcp.shared_memory.time.sleep") as mock_sleep:
            result = reader.read()
            assert result is None
            # Should have slept between retries (exponential backoff)
            assert mock_sleep.call_count >= 1


class TestAtomicWriteCorrectness:
    """Test that writes use the atomic temp-file-then-rename pattern."""

    def test_temp_file_cleaned_up_on_success(self, tmp_path):
        """After successful write, no .tmp file remains."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        writer.write({"key": "value"})

        tmp_file = filepath.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert filepath.exists()

    def test_target_file_has_correct_data(self, tmp_path):
        """After write, the target file contains valid JSON with correct data."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        data = {"sensor": "temperature", "value": 23.5}
        writer.write(data)

        raw = json.loads(filepath.read_text())
        assert raw["data"] == data

    def test_write_uses_fsync(self, tmp_path):
        """Write calls fsync for durability."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        with patch("anima_mcp.shared_memory.os.fsync") as mock_fsync:
            writer.write({"data": 1})
            assert mock_fsync.called

    def test_write_uses_replace(self, tmp_path):
        """Write uses Path.replace (atomic rename) pattern."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        original_replace = Path.replace
        replace_calls = []

        def tracking_replace(self_path, target_path):
            replace_calls.append((str(self_path), str(target_path)))
            return original_replace(self_path, target_path)

        with patch.object(Path, "replace", tracking_replace):
            writer.write({"data": 1})

        assert len(replace_calls) == 1
        assert replace_calls[0][0].endswith(".tmp")
        assert replace_calls[0][1] == str(filepath)

    def test_lock_file_created_during_write(self, tmp_path):
        """Write creates a .lock file for coordination."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        writer.write({"data": 1})

        lock_file = filepath.with_suffix(".lock")
        assert lock_file.exists()


class TestClear:
    """Test clear operation."""

    def test_clear_removes_file(self, tmp_path):
        """Clear deletes the shared memory file."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        writer.write({"data": 1})
        assert filepath.exists()

        writer.clear()
        assert not filepath.exists()

    def test_clear_nonexistent_file_ok(self, tmp_path):
        """Clearing a nonexistent file does not crash."""
        filepath = tmp_path / "nonexistent.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        # Should not raise
        writer.clear()

    def test_clear_ignored_in_read_mode(self, tmp_path):
        """Clear is a no-op for read-mode clients (mode check in clear)."""
        filepath = tmp_path / "state.json"
        filepath.write_text('{"data": "keep"}')

        reader = SharedMemoryClient(mode="read", filepath=filepath)
        reader.clear()

        # File should still exist since mode is read
        assert filepath.exists()


class TestStaleData:
    """Test staleness detection via timestamp in envelope."""

    def test_written_envelope_has_recent_timestamp(self, tmp_path):
        """Written envelope has a timestamp close to now."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)

        before = time.time()
        writer.write({"data": 1})
        after = time.time()

        raw = json.loads(filepath.read_text())
        # Parse the ISO timestamp
        from datetime import datetime
        ts = datetime.fromisoformat(raw["updated_at"])
        ts_epoch = ts.timestamp()

        assert before - 1 <= ts_epoch <= after + 2  # tolerance for slow CI / clock jitter

    def test_stale_threshold_concept(self, tmp_path):
        """Verify stale threshold constant exists for consumer use."""
        from anima_mcp.server_state import SHM_STALE_THRESHOLD_SECONDS
        assert isinstance(SHM_STALE_THRESHOLD_SECONDS, (int, float))
        assert SHM_STALE_THRESHOLD_SECONDS > 0


class TestConcurrentAccess:
    """Test file locking behavior."""

    def test_read_write_different_clients(self, tmp_path):
        """Multiple clients can read/write to same file."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        reader1 = SharedMemoryClient(mode="read", filepath=filepath)
        reader2 = SharedMemoryClient(mode="read", filepath=filepath)

        writer.write({"iteration": 1})
        assert reader1.read() == {"iteration": 1}
        assert reader2.read() == {"iteration": 1}

        writer.write({"iteration": 2})
        assert reader1.read() == {"iteration": 2}
        assert reader2.read() == {"iteration": 2}

    def test_read_returns_none_when_lock_held_after_retries(self, tmp_path):
        """If lock cannot be acquired after retries, read returns None."""
        filepath = tmp_path / "state.json"
        filepath.write_text('{"data": {"ok": true}}')

        reader = SharedMemoryClient(mode="read", filepath=filepath)

        # Simulate blocked lock by patching fcntl.flock to always raise
        with patch("anima_mcp.shared_memory.fcntl.flock", side_effect=BlockingIOError):
            with patch("anima_mcp.shared_memory.time.sleep"):
                result = reader.read()
                assert result is None
