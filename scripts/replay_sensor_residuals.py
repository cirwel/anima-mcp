#!/usr/bin/env python3
"""Offline SensorResidualTracker replay & Stage-2 readiness evaluator CLI (#79, #175).

Replays historical rows from Lumen's ``state_history`` SQLite table strictly
read-only (``?mode=ro`` by default; optional ``--immutable`` for frozen backups)
through the real ``SensorResidualTracker``, or evaluates an existing persisted
``sensor_residuals.json`` snapshot, and prints a Stage-2 readiness report
(Markdown to stdout, optional JSON output).

Usage:
  PYTHONPATH=src python3 scripts/replay_sensor_residuals.py \\
      --db ~/.anima/anima.db --days 14 --json-out /tmp/report.json

  PYTHONPATH=src python3 scripts/replay_sensor_residuals.py \\
      --snapshot ~/.anima/sensor_residuals.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from anima_mcp.residual_backtest import (  # noqa: E402
    DEFAULT_DB_PATH,
    evaluate_snapshot,
    format_markdown_report,
    replay_state_history,
)


def _paths_refer_to_same_file(a: Path, b: Path) -> bool:
    """Return True if two paths resolve to the same path or same inode."""
    pa = a.expanduser()
    pb = b.expanduser()
    try:
        if os.path.samefile(pa, pb):
            return True
    except OSError:
        pass
    try:
        return pa.resolve() == pb.resolve()
    except OSError:
        return os.path.abspath(str(pa)) == os.path.abspath(str(pb))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay historical state_history rows through SensorResidualTracker "
            "or evaluate a persisted sensor_residuals.json snapshot for Stage-2 readiness."
        )
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Path to SQLite anima.db (read-only; default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Path to a persisted sensor_residuals.json snapshot to evaluate read-only.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help=(
            "Restrict SQLite replay to the last N days anchored to the latest "
            "valid timestamp in state_history (omit for full history)."
        ),
    )
    parser.add_argument(
        "--immutable",
        action="store_true",
        default=False,
        help=(
            "Open SQLite with immutable=1 (for frozen backup copies on read-only media; "
            "do not use on live WAL-mode databases)."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "Override SensorResidualTracker min_samples (default: production 43200). "
            "Clearly labelled as custom override in the report."
        ),
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=None,
        help=(
            "Override SensorResidualTracker window_samples (default: production 302400)."
        ),
    )
    parser.add_argument(
        "--frozen-run-threshold",
        type=int,
        default=None,
        help=(
            "Override SensorResidualTracker frozen_run_threshold (default: production 60)."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full JSON readiness report.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Stdout output format (default: markdown).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.db is not None and args.snapshot is not None:
        parser.error("Specify either --db or --snapshot, not both.")
    if args.snapshot is not None and args.days is not None:
        parser.error("--days applies only to SQLite --db replay, not --snapshot.")
    if args.snapshot is not None and args.immutable:
        parser.error("--immutable applies only to SQLite --db replay, not --snapshot.")

    input_path = (
        Path(args.snapshot)
        if args.snapshot is not None
        else Path(args.db if args.db is not None else str(DEFAULT_DB_PATH))
    )
    if args.json_out is not None:
        out_path = Path(args.json_out)
        if _paths_refer_to_same_file(out_path, input_path):
            print(
                f"error: --json-out ({out_path}) must not refer to the same file as the input ({input_path})",
                file=sys.stderr,
            )
            return 2

    try:
        if args.snapshot is not None:
            report = evaluate_snapshot(
                args.snapshot,
                min_samples=args.min_samples,
                window_samples=args.window_samples,
                frozen_run_threshold=args.frozen_run_threshold,
            )
        else:
            report = replay_state_history(
                str(input_path),
                days=args.days,
                min_samples=args.min_samples,
                window_samples=args.window_samples,
                frozen_run_threshold=args.frozen_run_threshold,
                immutable=args.immutable,
            )

        if args.json_out is not None:
            out_path = Path(args.json_out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(format_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
