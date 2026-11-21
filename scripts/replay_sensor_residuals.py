#!/usr/bin/env python3
"""Offline SensorResidualTracker replay & Stage-2 readiness evaluator CLI (#79, #175).

Replays historical rows from Lumen's ``state_history`` SQLite table strictly
read-only (``?mode=ro&immutable=1``) through the real ``SensorResidualTracker``,
or evaluates an existing persisted ``sensor_residuals.json`` snapshot, and
prints a Stage-2 readiness report (Markdown to stdout, optional JSON output).

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
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from anima_mcp.residual_backtest import (  # noqa: E402
    DEFAULT_DB_PATH,
    evaluate_snapshot,
    format_markdown_report,
    replay_state_history,
)


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
            "timestamp in state_history (omit for full history)."
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

    try:
        if args.snapshot is not None:
            report = evaluate_snapshot(
                args.snapshot,
                min_samples=args.min_samples,
                window_samples=args.window_samples,
                frozen_run_threshold=args.frozen_run_threshold,
            )
        else:
            db_path = args.db if args.db is not None else str(DEFAULT_DB_PATH)
            report = replay_state_history(
                db_path,
                days=args.days,
                min_samples=args.min_samples,
                window_samples=args.window_samples,
                frozen_run_threshold=args.frozen_run_threshold,
            )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(format_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
