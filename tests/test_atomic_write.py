"""Tests for atomic JSON write utility."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from anima_mcp.atomic_write import atomic_json_write


class TestAtomicJsonWrite:
    """Test atomic_json_write function."""

    def test_basic_write(self, tmp_path):
        """Write data and read it back."""
        target = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        atomic_json_write(target, data)

        assert target.exists()
        with open(target) as f:
            result = json.load(f)
        assert result == data

    def test_write_with_indent(self, tmp_path):
        """Indented output is formatted."""
        target = tmp_path / "test.json"
        data = {"a": 1}
        atomic_json_write(target, data, indent=2)

        text = target.read_text()
        assert "  " in text  # Has indentation
        assert json.loads(text) == data

    def test_overwrites_existing(self, tmp_path):
        """Atomic write replaces existing file."""
        target = tmp_path / "test.json"
        target.write_text('{"old": true}')

        atomic_json_write(target, {"new": True})
        result = json.loads(target.read_text())
        assert result == {"new": True}

    def test_creates_parent_directories(self, tmp_path):
        """Creates parent dirs if they don't exist."""
        target = tmp_path / "sub" / "dir" / "test.json"
        atomic_json_write(target, {"nested": True})

        assert target.exists()
        assert json.loads(target.read_text()) == {"nested": True}

    def test_no_tmp_file_on_success(self, tmp_path):
        """Tmp file is cleaned up after successful write."""
        target = tmp_path / "test.json"
        atomic_json_write(target, {"data": 1})

        tmp_file = target.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_no_tmp_file_on_error(self, tmp_path):
        """Tmp file is cleaned up on serialization error."""
        target = tmp_path / "test.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": Unserializable()})

        tmp_file = target.with_suffix(".tmp")
        assert not tmp_file.exists()
        # Original file should not exist (was never written)
        assert not target.exists()

    def test_preserves_original_on_error(self, tmp_path):
        """If write fails, the original file is preserved."""
        target = tmp_path / "test.json"
        target.write_text('{"original": true}')

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": Unserializable()})

        # Original should be untouched
        result = json.loads(target.read_text())
        assert result == {"original": True}

    def test_accepts_string_path(self, tmp_path):
        """Works with string paths, not just Path objects."""
        target = str(tmp_path / "test.json")
        atomic_json_write(target, [1, 2, 3])

        with open(target) as f:
            assert json.load(f) == [1, 2, 3]

    def test_calls_fsync(self, tmp_path):
        """Verifies fsync is called on the file descriptor."""
        target = tmp_path / "test.json"

        with patch("anima_mcp.atomic_write.os.fsync") as mock_fsync:
            atomic_json_write(target, {"data": 1})
            assert mock_fsync.called

    def test_uses_rename(self, tmp_path):
        """Verifies the rename (replace) pattern is used."""
        target = tmp_path / "test.json"

        # Track if tmp file exists before rename
        original_replace = Path.replace
        replace_called = []

        def tracking_replace(self_path, target_path):
            replace_called.append((str(self_path), str(target_path)))
            return original_replace(self_path, target_path)

        with patch.object(Path, "replace", tracking_replace):
            atomic_json_write(target, {"data": 1})

        assert len(replace_called) == 1
        assert replace_called[0][0].endswith(".tmp")
