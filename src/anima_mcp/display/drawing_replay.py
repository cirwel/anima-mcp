"""Process recording for Lumen's drawings: the piece as it was made.

A saved PNG is the canvas's final pixel dict. It cannot say in what order the
marks landed, how long Lumen paused between them, or that a region was drawn
over three times. This module keeps that history as an append-only log beside
the canvas, so a piece can be replayed (see scripts/render_drawing_replay.py)
instead of only shown.

Observation only: nothing here feeds a drawing decision, and every public method
swallows its own errors, because a broken recorder must never stop Lumen drawing.

Format (v1): JSON Lines, one record per line.

    {"type": "header", "v": 1, "piece": "p1731...", "w": 240, "h": 240,
     "era": "gestural", "started": <piece clear time>, "opened": <t>}
    {"type": "mark", "t": <t>, "px": [x, y, r, g, b, ...], "g": "<gesture>",
     "s": [E, I, S, V, C], "a": "<arc phase>"}
    {"type": "base", "t": <t>, "px": [...], "erase": [x, y, ...], "why": "..."}
    {"type": "resume", "t": <t>}
    {"type": "era", "t": <t>, "era": "field"}
    {"type": "off", "t": <t>, "why": "..."}
    {"type": "end", "t": <t>, "png": "<file name>", "reason": ..., "manual": bool}

"px" is flattened (x, y, r, g, b) quintuples applied in order; later writes to
the same pixel replace earlier ones, exactly as CanvasState.draw_pixel does.

"base" records explain pixels the log did not see being drawn. why="pre-existing"
means the piece was already on the canvas when recording started (a piece begun
before this recorder existed, or one whose log was lost); the in-progress log
keeps it so the open piece can still be watched, but such a piece gets no
finished replay, because a replay is only written for a piece recorded from its
first mark. why="reconcile" means a restart found the log and the persisted
canvas disagreeing (a crash between their writes, at most one persist interval
of marks); the record brings the log back to the canvas, so the replay always
ends on the real piece.
"resume" marks a process restart: the gap before it is downtime, not a pause.
"era" marks a forced mid-piece era switch; the header names the starting era.
"off" replaces the whole log when recording stops partway (size cap, a disk
that will not take writes): that piece then gets no replay rather than one
with a hole in it, and a restart reads the marker and stays off.

Times are wall-clock seconds (time.time()), rounded to 10 ms.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

FORMAT_VERSION = 1

# The in-progress log of a pathological piece must not fill the SD card. A
# typical 8-hour piece is a few thousand marks (well under 1 MB); past this
# the recorder stops for the rest of the piece and says so once.
MAX_IN_PROGRESS_BYTES = 32 * 1024 * 1024
MAX_BUFFERED_RECORDS = 5000

REPLAY_SUFFIX = ".replay.jsonl.gz"

Pixel = Tuple[int, int, int]
PixelMap = Dict[Tuple[int, int], Pixel]


def replay_path_for(png_path: Path) -> Path:
    """lumen_drawing_X.png -> lumen_drawing_X.replay.jsonl.gz (same directory)."""
    png_path = Path(png_path)
    return png_path.with_name(png_path.stem + REPLAY_SUFFIX)


def _flat(pixels: Iterable[Tuple[int, int, Pixel]]) -> List[int]:
    out: List[int] = []
    for x, y, (r, g, b) in pixels:
        out.extend((int(x), int(y), int(r), int(g), int(b)))
    return out


def _log(msg: str) -> None:
    print(f"[Replay] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def iter_records(path: Path) -> Iterator[dict]:
    """Yield records from a replay log (.jsonl or .jsonl.gz).

    A torn final line (the process died mid-write) is skipped, not fatal.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def apply_record(pixels: PixelMap, rec: dict) -> None:
    """Apply one record's pixel effect to `pixels` in place."""
    erase = rec.get("erase") or []
    for i in range(0, len(erase) - 1, 2):
        pixels.pop((erase[i], erase[i + 1]), None)
    px = rec.get("px") or []
    for i in range(0, len(px) - 4, 5):
        pixels[(px[i], px[i + 1])] = (px[i + 2], px[i + 3], px[i + 4])


def final_pixels(records: Iterable[dict]) -> PixelMap:
    """The canvas a log describes once every record is applied."""
    pixels: PixelMap = {}
    for rec in records:
        apply_record(pixels, rec)
    return pixels


def _repair_tail(path: Path) -> None:
    """End the file on a newline. A process killed mid-write leaves a torn
    last line; appending straight onto it would merge the next record into
    that fragment and the reader would drop both."""
    try:
        if path.stat().st_size == 0:
            return
    except FileNotFoundError:
        return
    with open(path, "rb+") as fh:
        fh.seek(-1, 2)
        if fh.read(1) != b"\n":
            fh.write(b"\n")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class ReplayRecorder:
    """Append-only recorder for the piece currently on the canvas.

    Wiring (DrawingEngine): `pending` is installed as CanvasState._replay_tap so
    draw_pixel appends to it; record_mark() turns what accumulated into one mark
    record; flush() rides the canvas persistence cadence; finalize() writes the
    finished log beside the PNG; start_piece() begins a new log at clear.
    """

    def __init__(self, path_fn: Callable[[], Path]):
        self._path_fn = path_fn
        self.pending: List[Tuple[int, int, Pixel]] = []
        self._buf: List[str] = []
        self._piece: Optional[str] = None
        self._header: Optional[dict] = None
        self._off = False  # recording stopped for the rest of this piece

    # -- paths ---------------------------------------------------------------

    def path(self) -> Path:
        return self._path_fn()

    # -- lifecycle -----------------------------------------------------------

    def open(self, canvas, era_name: str) -> None:
        """Attach to whatever piece is on the canvas at process start."""
        try:
            piece = canvas.piece_uid()
            path = self.path()
            records = list(iter_records(path)) if path.exists() else []
            header = records[0] if records and records[0].get("type") == "header" else None
            if canvas.pixels and header and header.get("piece") == piece:
                self._piece = piece
                self._header = header
                if any(r.get("type") == "off" for r in records):
                    # Recording stopped earlier in this piece; a restart must
                    # not quietly resume it into a replay with a hole in it.
                    self._off = True
                    return
                # Resume first, so a renderer drops the downtime before it
                # meets the reconcile record stamped after it.
                self._emit({"type": "resume", "t": self._now()})
                for rec in self._reconcile_record(records, canvas):
                    self._emit(rec)
            else:
                self._begin(canvas, era_name, pre_existing=bool(canvas.pixels))
            self.flush()
        except Exception as e:
            _log(f"open failed, recording off for this piece: {e}")
            self._go_off("open failed")

    def start_piece(self, canvas, era_name: str) -> None:
        """A fresh canvas: discard the previous log and begin a new one."""
        try:
            self.pending.clear()
            self._buf.clear()
            self._off = False
            self._begin(canvas, era_name, pre_existing=bool(canvas.pixels), truncate=True)
            self.flush()
        except Exception as e:
            _log(f"start_piece failed, recording off for this piece: {e}")
            self._go_off("start failed")

    # -- recording -----------------------------------------------------------

    def record_mark(self, *, gesture: Optional[str], eisv: Iterable[float],
                    arc_phase: Optional[str], now: Optional[float] = None) -> None:
        """Turn the pixels draw_pixel tapped since the last mark into a record."""
        if not self.pending:
            return
        try:
            px = _flat(self.pending)
            self.pending.clear()
            if self._off:
                return
            rec = {
                "type": "mark",
                "t": self._now(now),
                "px": px,
                "g": gesture,
                "s": [round(float(v), 4) for v in eisv],
                "a": arc_phase,
            }
            self._emit(rec)
        except Exception as e:
            _log(f"record_mark failed: {e}")
            self._go_off("record failed")  # a dropped mark is a hole

    def note_era(self, era_name: str) -> None:
        """The era changed without a new piece (a forced switch)."""
        try:
            if not self._off:
                self._emit({"type": "era", "t": self._now(), "era": era_name})
        except Exception as e:
            _log(f"note_era failed: {e}")

    def flush(self) -> None:
        if not self._buf or self._off:
            self._buf.clear()
            return
        try:
            path = self.path()
            path.parent.mkdir(parents=True, exist_ok=True)
            _repair_tail(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("".join(self._buf))
            self._buf.clear()
            if path.stat().st_size > MAX_IN_PROGRESS_BYTES:
                _log(f"in-progress log over {MAX_IN_PROGRESS_BYTES} bytes; "
                     f"recording off for piece {self._piece}")
                self._go_off("size cap")
        except Exception as e:
            _log(f"flush failed: {e}")
            # Keep the buffer for the next attempt, but not without bound: a
            # disk that stays unwritable must not grow memory all day.
            if len(self._buf) > MAX_BUFFERED_RECORDS:
                _log("flush keeps failing; recording off for this piece")
                self._go_off("flush failing")

    def finalize(self, png_path: Path, canvas, *, reason: Optional[str] = None,
                 manual: bool = False) -> Optional[Path]:
        """Write the finished log, compressed, beside the saved PNG.

        The written copy is reconciled against `canvas` first, so a replay
        always ends on exactly the pixels of the PNG it sits beside, even if
        a flush was lost. A manual save is a snapshot of a piece that keeps
        going, so the in-progress log stays open either way; start_piece()
        is what ends it. A piece whose recording was switched off gets no
        replay rather than a wrong one.
        """
        try:
            self.flush()
            src = self.path()
            if self._off or not src.exists():
                return None
            _repair_tail(src)
            records = list(iter_records(src))
            if any(r.get("type") == "base" and r.get("why") == "pre-existing" for r in records):
                return None  # recording never saw this piece's opening
            dest = replay_path_for(png_path)
            tail = self._reconcile_record(records, canvas)
            tail.append({
                "type": "end",
                "t": self._now(),
                "png": Path(png_path).name,
                "reason": reason,
                "manual": bool(manual),
            })
            tmp = dest.with_name(dest.name + ".tmp")
            with open(src, "rb") as fin, gzip.open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout)
                for rec in tail:
                    fout.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))
            tmp.rename(dest)
            return dest
        except Exception as e:
            _log(f"finalize failed: {e}")
            return None

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _now(now: Optional[float] = None) -> float:
        return round(time.time() if now is None else now, 2)

    def _emit(self, rec: dict) -> None:
        self._buf.append(json.dumps(rec, separators=(",", ":")) + "\n")

    def _begin(self, canvas, era_name: str, *, pre_existing: bool,
               truncate: bool = True) -> None:
        path = self.path()
        if truncate and path.exists():
            path.unlink()
        self._piece = canvas.piece_uid()
        self._header = {
            "type": "header",
            "v": FORMAT_VERSION,
            "piece": self._piece,
            "w": canvas.width,
            "h": canvas.height,
            "era": era_name,
            "started": round(float(canvas.last_clear_time), 2),
            "opened": self._now(),
        }
        self._emit(self._header)
        if pre_existing:
            self._emit({
                "type": "base",
                "t": self._now(),
                "px": _flat((x, y, c) for (x, y), c in canvas.pixels.items()),
                "why": "pre-existing",
            })

    def _go_off(self, why: str) -> None:
        """Stop recording for the rest of this piece, durably.

        The log is replaced by its header plus an "off" record: that frees
        the disk at once, and a restart reads the marker and stays off
        instead of resuming into a replay with a hole in it.
        """
        self._off = True
        self._buf.clear()
        self.pending.clear()
        try:
            path = self.path()
            if self._header is None:
                if path.exists():
                    path.unlink()
                return
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for rec in (self._header, {"type": "off", "t": self._now(), "why": why}):
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            tmp.replace(path)
        except Exception as e:
            # No marker means a restart would resume this log with a hole in
            # it. Without the log, a restart begins a fresh one whose opening
            # is an explicit "pre-existing" base, which hides nothing.
            _log(f"could not mark recording off on disk ({e}); removing the log")
            try:
                self.path().unlink()
            except Exception as e2:
                _log(f"could not remove the log either: {e2}")

    def stop(self, why: str) -> None:
        """Stop recording for the rest of this piece (see _go_off)."""
        if not self._off:
            _log(f"recording off for piece {self._piece}: {why}")
            self._go_off(why)

    def _reconcile_record(self, records: Iterable[dict], canvas) -> List[dict]:
        """[] when the log already ends on the canvas, else one base record
        that brings it there (pixels added or recoloured, pixels erased)."""
        logged = final_pixels(records)
        add = []
        for (x, y), c in canvas.pixels.items():
            c = tuple(int(v) for v in c)
            if logged.get((x, y)) != c:
                add.append((x, y, c))
        erase: List[int] = []
        for (x, y) in logged:
            if (x, y) not in canvas.pixels:
                erase.extend((x, y))
        if not add and not erase:
            return []
        return [{
            "type": "base",
            "t": self._now(),
            "px": _flat(add),
            "erase": erase,
            "why": "reconcile",
        }]
