#!/usr/bin/env python3
"""Render a Lumen drawing replay as a GIF, a video, or PNG frames.

Lumen writes `lumen_drawing_<ts>_<era>.replay.jsonl.gz` beside each saved
drawing (format: src/anima_mcp/display/drawing_replay.py). The Mac's hourly
mirror carries them to ~/backups/lumen/anima_data/drawings/. The open piece's
log so far is `canvas.replay.jsonl` beside canvas.json, and renders the same way.

    python3 scripts/render_drawing_replay.py PIECE.replay.jsonl.gz            # PIECE.gif
    python3 scripts/render_drawing_replay.py PIECE.replay.jsonl.gz -o p.mp4   # needs ffmpeg
    python3 scripts/render_drawing_replay.py PIECE.replay.jsonl.gz -o frames/ # PNG frames

Clocks. A piece takes hours, so time is always compressed into --seconds:
  wall   (default) real time, with any pause longer than --max-gap shortened
         to it. Stillness stays visible as stillness, but an hour of rest
         does not eat the video. Process downtime (a "resume" record) is
         removed entirely: Lumen was not pausing, Lumen was not running.
  marks  every mark gets equal time. Shows the order of making, not its rhythm.

The strip under the canvas plots the drawing state recorded with each mark
(E, I, S, V and behavioral coherence C). Each line is scaled to its own range
within the piece, so it shows movement, not level. Dim ticks mark a restart
(resume) or a reconcile; --no-strip renders the canvas alone.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from anima_mcp.display.drawing_replay import apply_record, iter_records  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SERIES = (  # (label, index into "s", colour)
    ("E", 0, (235, 140, 60)),
    ("I", 1, (90, 160, 240)),
    ("S", 2, (150, 150, 150)),
    ("V", 3, (110, 200, 120)),
    ("C", 4, (235, 235, 235)),
)
STRIP_H = 44  # unscaled pixels
BG = (0, 0, 0)


def build_timeline(records: List[dict], clock: str, max_gap: float) -> Tuple[List[Tuple[float, dict]], List[float], dict]:
    """Assign each mark/base/era record a playback position.

    Returns (events, ticks, header): events as (position, record) in log order,
    ticks as positions of resume/reconcile markers. On the marks clock only
    pixel-bearing records advance the position; an era switch takes no time.
    """
    header = next((r for r in records if r.get("type") == "header"), {})
    events: List[Tuple[float, dict]] = []
    ticks: List[float] = []
    pos = 0.0
    last_t: Optional[float] = None
    seen_pixels = False
    for rec in records:
        kind = rec.get("type")
        t = rec.get("t")
        if kind == "resume":
            last_t = t  # downtime is not a pause: drop the gap entirely
            ticks.append(pos)
            continue
        if kind not in ("mark", "base", "era"):
            continue
        if clock == "marks":
            if kind != "era":
                if seen_pixels:
                    pos += 1.0
                seen_pixels = True
        elif t is not None:
            if last_t is not None:
                pos += min(max(0.0, t - last_t), max_gap)
            last_t = t
        events.append((pos, rec))
        if kind == "base" and rec.get("why") == "reconcile":
            ticks.append(pos)
    return events, ticks, header


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _series_ranges(events) -> List[Tuple[float, float]]:
    ranges = []
    for _, idx, _ in SERIES:
        vals = [r["s"][idx] for _, r in events if r.get("type") == "mark" and len(r.get("s") or []) > idx]
        ranges.append((min(vals), max(vals)) if vals else (0.0, 1.0))
    return ranges


def render_frames(records: List[dict], *, seconds: float, fps: int, clock: str,
                  max_gap: float, scale: int, strip: bool, hold: float):
    """Yield RGB frames (PIL Images) for the replay."""
    events, ticks, header = build_timeline(records, clock, max_gap)
    w, h = int(header.get("w", 240)), int(header.get("h", 240))
    W, H = w * scale, h * scale
    sh = STRIP_H * scale if strip else 0
    # Video encoders want even dimensions.
    out_w, out_h = W + (W % 2), H + sh + ((H + sh) % 2)
    total = events[-1][0] if events else 0.0
    n = max(2, int(round(seconds * fps)))
    ranges = _series_ranges(events)
    # Elapsed is measured from the piece's start, or from the first record if
    # the log began later than that is consistent with (a clock that disagrees).
    first_t = next((r.get("t") for _, r in events if r.get("t") is not None), None)
    started = header.get("started")
    if started is None or (first_t is not None and started > first_t):
        started = first_t
    era = header.get("era") or ""
    try:
        font = ImageFont.load_default(size=9 * scale)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    pixels: dict = {}
    canvas = Image.new("RGB", (w, h), BG)
    cpx = canvas.load()
    applied = 0
    points: List[Tuple[float, List[float], float]] = []  # (pos, state, t)

    def xpos(p: float) -> float:
        return (p / total) * (W - 1) if total > 0 else 0.0

    for k in range(n):
        playhead = total * k / (n - 1)
        while applied < len(events) and events[applied][0] <= playhead:
            pos, rec = events[applied]
            apply_record(pixels, rec)
            erase = rec.get("erase") or []
            for i in range(0, len(erase) - 1, 2):
                cpx[erase[i], erase[i + 1]] = BG
            px = rec.get("px") or []
            for i in range(0, len(px) - 4, 5):
                cpx[px[i], px[i + 1]] = (px[i + 2], px[i + 3], px[i + 4])
            if rec.get("type") == "era" and rec.get("era"):
                era = rec["era"]
            if rec.get("type") == "mark" and rec.get("s"):
                points.append((pos, rec["s"], rec.get("t") or 0.0))
            applied += 1

        frame = Image.new("RGB", (out_w, out_h), BG)
        frame.paste(canvas.resize((W, H), Image.NEAREST), (0, 0))
        if strip:
            d = ImageDraw.Draw(frame)
            top = H
            pad = 3 * scale
            plot_top, plot_bot = top + pad, top + sh - 12 * scale
            for tp in ticks:
                if tp <= playhead:
                    x = xpos(tp)
                    d.line([(x, plot_top), (x, plot_bot)], fill=(60, 60, 60))
            for (label, idx, colour), (lo, hi) in zip(SERIES, ranges):
                span = (hi - lo) or 1.0
                line = [
                    (xpos(p), plot_bot - (s[idx] - lo) / span * (plot_bot - plot_top))
                    for p, s, _ in points if len(s) > idx
                ]
                if len(line) >= 2:
                    d.line(line, fill=colour, width=max(1, scale // 2))
            x = xpos(playhead)
            d.line([(x, plot_top), (x, plot_bot)], fill=(120, 120, 120))
            label_y = top + sh - 11 * scale
            lx = pad
            for label, _, colour in SERIES:
                d.text((lx, label_y), label, fill=colour, font=font)
                lx += 12 * scale
            elapsed = ""
            if started is not None and points:
                elapsed = _fmt_elapsed(points[-1][2] - started)
            info = f"{era}  {elapsed}  {len(pixels)}px"
            d.text((W - pad - d.textlength(info, font=font), label_y), info,
                   fill=(170, 170, 170), font=font)
        yield frame

    for _ in range(int(round(hold * fps))):
        yield frame


def write_output(frames, out: Path, fps: int) -> int:
    count = 0
    suffix = out.suffix.lower()
    if suffix == ".gif":
        frames = list(frames)
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        return len(frames)
    if suffix in (".mp4", ".webm", ".mov"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg not found on PATH; use -o something.gif or a frames/ directory")
        it = iter(frames)
        first = next(it)
        codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30"] if suffix == ".webm" else \
                ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{first.width}x{first.height}", "-r", str(fps), "-i", "-", *codec, str(out)],
            stdin=subprocess.PIPE)
        for f in (first, *it):
            proc.stdin.write(f.tobytes())
            count += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {out}")
        return count
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(out / f"frame_{i:05d}.png")
        count += 1
    return count


def default_output(src: Path) -> Path:
    name = src.name
    for suf in (".replay.jsonl.gz", ".replay.jsonl", ".jsonl.gz", ".jsonl"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return Path.cwd() / f"{name}.gif"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", type=Path)
    ap.add_argument("-o", "--out", type=Path, help=".gif, .mp4, .webm, .mov, or a directory for PNG frames")
    ap.add_argument("--seconds", type=float, default=20.0, help="playback length before the hold (default 20)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--clock", choices=("wall", "marks"), default="wall")
    ap.add_argument("--max-gap", type=float, default=120.0,
                    help="wall clock: longest pause kept at full length, seconds (default 120)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--no-strip", action="store_true", help="canvas only, no state strip")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the finished piece")
    args = ap.parse_args(argv)

    records = list(iter_records(args.replay))
    if not any(r.get("type") in ("mark", "base") for r in records):
        print(f"{args.replay}: no marks recorded", file=sys.stderr)
        return 1
    out = args.out or default_output(args.replay)
    frames = render_frames(
        records, seconds=args.seconds, fps=args.fps, clock=args.clock,
        max_gap=args.max_gap, scale=max(1, args.scale), strip=not args.no_strip,
        hold=args.hold)
    n = write_output(frames, out, args.fps)
    marks = sum(1 for r in records if r.get("type") == "mark")
    print(f"{out}  ({n} frames, {marks} marks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
