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

  --days defaults to 365. This script's population is one row per PIECE, and
  it floors at 500; at Lumen's ~3 pieces/day a 90-day window (the old default)
  holds ~270 and the run refuses. That is a window problem, not a corpus
  problem — widen the window, never lower the floor.

  --apply MERGES the COVERAGE_* keys into nervous_system.drawing_thresholds in
  the given calibration file atomically (backup written alongside), preserving
  the curiosity derivation's CURIOSITY_PIVOT_* keys. Without it, prints JSON.

Cadence: the running server now applies this itself, weekly
(anima_mcp/self_derivation.py). This script remains the operator's path — to
inspect, to force a rerun after a change that re-bases clarity (a
#173/#176-style de-aliasing moves this file's answer), or on a device where
ANIMA_SELF_DERIVATION=false.

The contract, the refusals and the merge rule live in
anima_mcp.drawing_derivation, shared with the server so they cannot drift.
DrawingGoal reads through get_calibration(), which refreshes on config-file
signature change, so a rederive lands on the next piece without a restart.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from anima_mcp.drawing_derivation import (  # noqa: E402
    COVERAGE_DAYS, coverage_report, merge_coverage,
)


def apply_to_config(path, thresholds):
    path = os.path.expanduser(path)
    backup = f"{path}.bak-drawing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    with open(path) as f:
        cfg = json.load(f)
    ns = cfg.setdefault("nervous_system", {})
    existing = ns.get("drawing_thresholds")
    ns["drawing_thresholds"] = merge_coverage(
        existing if isinstance(existing, dict) else {}, thresholds)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
    return backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--days", type=int, default=COVERAGE_DAYS)
    ap.add_argument("--apply", default=None, metavar="CONFIG_JSON")
    args = ap.parse_args()

    report = coverage_report(os.path.expanduser(args.db), days=args.days)
    if not report.get("available"):
        sys.exit(f"refusing: {report.get('reason')}")
    if report.get("refused"):
        sys.exit(f"refusing: {report['refused']}")
    thresholds = report["thresholds"]
    print(f"# derived from n={report['samples']} drawing_records clarity samples",
          file=sys.stderr)
    if args.apply:
        backup = apply_to_config(args.apply, thresholds)
        print(f"applied to {args.apply} (backup: {backup})", file=sys.stderr)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
