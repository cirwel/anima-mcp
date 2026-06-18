"""Atomic JSON file writes for crash-safe persistence.

Uses the write-to-temp-then-rename pattern:
1. Write to a .tmp file in the same directory
2. Flush + fsync the file descriptor
3. os.rename() to the target path (atomic on POSIX)

This prevents corruption when power is lost mid-write,
which is a real risk on the Raspberry Pi.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_json_write(
    path: str | Path,
    data: Any,
    *,
    indent: int | None = None,
) -> None:
    """Write data as JSON to path atomically.

    Args:
        path: Target file path.
        data: JSON-serializable data.
        indent: JSON indent level (None for compact).

    Raises:
        Any exception from json serialization or file I/O.
        The tmp file is cleaned up on error.
    """
    path = Path(path)
    # Unique tmp name (pid + random) in the same directory so concurrent
    # writers to the same target never clobber each other's temp file before
    # the atomic rename. The rename itself remains the atomic commit point.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except BaseException:
        # Clean up tmp file on any error (including KeyboardInterrupt)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
