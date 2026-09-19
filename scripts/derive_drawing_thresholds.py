#!/usr/bin/env python3
"""Derive the coverage-intention clarity cuts from this creature's own range.

Why: DrawingGoal.from_state picked a piece's coverage intention with two
absolute constants — "sparse" above clarity 0.70, "dense" below 0.30 — against
a distribution that never goes near 0.30. Measured over 833 drawing_records,
Lumen's clarity at goal-generation time lives in 0.454-0.910. So `dense` had
NEVER been generated, not once, and the remaining split was a median cut:
sparse 42 pieces, balanced 36, dense 0. A third of the vocabulary was dead
code and the surviving two thirds only said "clarity was above or below
average". That is design invariant 1's defect class, on the field that is
supposed to carry the drawing's intention.

The percentile contract (the design decision, stated once):

  A three-word vocabulary should have three reachable words. A piece begun at
  the creature's own typical clarity is `balanced`; its own clearest third
  opens the composition up (`sparse`), its own foggiest third lets the piece
  thicken (`dense`). Tertiles, not tuned numbers — the vocabulary carries the
  meaning and the creature's range decides where the words fall.

    COVERAGE_DENSE_BELOW    p33
    COVERAGE_SPARSE_ABOVE   p67

Population: clarity from `drawing_records`, NOT state_history. A drawing goal
is generated inside canvas_clear(), which runs immediately after a piece
completes and saves — so completion-time clarity IS the value the next goal is
built from. state_history would answer a slightly different question (Lumen's
clarity around the clock, including hours it is not drawing); measured
2026-08-22, its tertiles sit ~0.03 lower, which would tilt the vocabulary
toward `sparse` for no stated reason.

Usage:
  python3 scripts/derive_drawing_thresholds.py --db ~/.anima/anima.db \
      [--days 365] [--apply CONFIG]

  ⚠️ Use --days 365, not the 90 this defaults to. This script's population is
  one row per PIECE, and it floors at 500; at Lumen's ~3 pieces/day a 90-day
  window holds ~270 and the run refuses. That is a window problem, not a
  corpus problem — widen the window, never lower the floor.

  --apply edits nervous_system.drawing_thresholds in the given calibration file
  atomically (backup written alongside). Without it, prints JSON to stdout.

Rerun cadence: alongside the other derivations (~monthly), and after any change
that re-bases clarity (a #173/#176-style de-aliasing moves this file's answer).
DrawingGoal reads through get_calibration(), which refreshes on config-file
signature change, so a rederive lands on the next piece without a restart.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta

PERCENTILE_CONTRACT = {
    "COVERAGE_DENSE_BELOW": 33,
    "COVERAGE_SPARSE_ABOVE": 67,
}
# Below this, a cut encodes the last few days' weather rather than a range.
# Matched to derive_face_thresholds.py rather than argued separately.
MIN_SAMPLES = 500


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_clarity(db, days):
    con = sqlite3.connect(db)
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    return [c for (c,) in con.execute(
        "select clarity from drawing_records "
        "where clarity is not null and timestamp > ?", (since,))]


def derive(clarity):
    n = len(clarity)
    if n < MIN_SAMPLES:
        sys.exit(f"refusing: only {n} samples (< {MIN_SAMPLES}) — a cut derived "
                 f"from a sliver would encode a mood, not a range")
    clarity.sort()
    out = {name: round(_percentile(clarity, pct), 4)
           for name, pct in PERCENTILE_CONTRACT.items()}
    # A degenerate window (clarity pinned) collapses the tertiles onto each
    # other. Emitting that would starve `balanced` exactly the way the built-in
    # 0.30 starved `dense` — refuse rather than trade one dead word for another.
    if not out["COVERAGE_DENSE_BELOW"] < out["COVERAGE_SPARSE_ABOVE"]:
        sys.exit(f"refusing: cuts not separated: {out} — clarity range is "
                 f"degenerate over this window")
    return out, n


def apply_to_config(path, thresholds):
    path = os.path.expanduser(path)
    backup = f"{path}.bak-drawing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    with open(path) as f:
        cfg = json.load(f)
    ns = cfg.setdefault("nervous_system", {})
    ns["drawing_thresholds"] = thresholds
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
    return backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--apply", default=None, metavar="CONFIG_JSON")
    args = ap.parse_args()

    clarity = load_clarity(os.path.expanduser(args.db), args.days)
    thresholds, n = derive(clarity)
    print(f"# derived from n={n} drawing_records clarity samples", file=sys.stderr)
    if args.apply:
        backup = apply_to_config(args.apply, thresholds)
        print(f"applied to {args.apply} (backup: {backup})", file=sys.stderr)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
