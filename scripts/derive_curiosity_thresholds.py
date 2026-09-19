#!/usr/bin/env python3
"""Derive the per-era curiosity pivot from this creature's own coherence range.

Why: `_update_attention` split "still exploring" (curiosity drains) from
"pattern found" (curiosity regenerates) with one absolute constant, C < 0.4,
against a behavioural coherence distribution that differs per era. Measured
2026-08-02 on a resonance piece, live C sat in [0.377, 0.498] mean 0.458 — so
~95% of ticks took the REGENERATING branch and curiosity rose monotonically
into its 1.0 clamp. `attention_exhausted()` (curiosity < 0.15) and
`earned_composition` (curiosity < 0.2) were therefore not badly tuned, they
were unreachable: 26 of the first 34 instrumented completions landed on the
8-hour cap and none was earned. An era reaching C ~ 0.8 is split fairly by that
same 0.4, which is exactly why one number cannot serve every era. Design
invariant 1, on the signal meant to end a piece.

The percentile contract, the refusal semantics and the replay all live in
`anima_mcp.drawing_derivation` — imported here rather than restated, and shared
with the `diagnostics` tool so the report is readable on a device with no shell.
This file is the ACTING half: it is the only path that writes calibration.

Usage:
  python3 scripts/derive_curiosity_thresholds.py --db ~/.anima/anima.db \
      [--days 90] [--apply CONFIG]

  Unlike derive_drawing_thresholds.py, 90 days is usually enough here: the
  population is ~96 trajectory intervals per piece rather than one row per
  piece, so the same 500 floor is cleared by far fewer pieces. --apply MERGES
  into drawing_thresholds, preserving that script's COVERAGE_* keys, so run
  order does not matter.

  Without --apply it prints the report and JSON and changes nothing. Absent
  keys fall back to the built-in 0.4, so an un-applied derivation moves no mark.

Rerun cadence: alongside the other derivations (~monthly), and after any change
that re-bases behavioural C. `_curiosity_pivot()` reads through
get_calibration(), which refreshes on config-file signature change, so a
rederive lands without a restart.
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
from anima_mcp.drawing_derivation import derive_report  # noqa: E402


def apply_to_config(path, thresholds):
    """Merge CURIOSITY_PIVOT_* into drawing_thresholds, preserving neighbours.

    Merged, not replaced: derive_drawing_thresholds.py owns the COVERAGE_* keys
    in this same dict and overwriting it whole would silently revert them.
    """
    path = os.path.expanduser(path)
    backup = f"{path}.bak-curiosity-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    with open(path) as f:
        cfg = json.load(f)
    ns = cfg.setdefault("nervous_system", {})
    existing = ns.get("drawing_thresholds")
    if not isinstance(existing, dict):
        existing = {}
    # Drop stale pivots for eras this run did not emit, so a since-retired or
    # since-refused era stops being steered by a number nothing re-verified.
    existing = {k: v for k, v in existing.items()
                if not k.startswith("CURIOSITY_PIVOT_")}
    existing.update(thresholds)
    ns["drawing_thresholds"] = existing
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

    report = derive_report(os.path.expanduser(args.db), days=args.days)
    if not report.get("available"):
        sys.exit(f"refusing: {report.get('reason')}")
    print(f"# derived from n={report['usable_intervals']} usable "
          f"drawing_trajectory intervals across {len(report['eras'])} eras",
          file=sys.stderr)
    print(json.dumps({"report": report["eras"]}, indent=2), file=sys.stderr)

    refused = report["eras"].get("_refused")
    if refused:
        sys.exit(f"refusing: {refused}")
    thresholds = report["thresholds"]
    if not thresholds:
        sys.exit("refusing: no era produced a verifiable pivot — see the report "
                 "above; the built-in 0.4 keeps serving every era")
    if args.apply:
        backup = apply_to_config(args.apply, thresholds)
        print(f"applied to {args.apply} (backup: {backup})", file=sys.stderr)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
