"""Tests for offline SensorResidualTracker replay & Stage-2 readiness evaluator (#79, #175)."""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import sqlite3
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest

from anima_mcp.residual_backtest import (
    DEFAULT_MIN_SAMPLES,
    evaluate_snapshot,
    format_markdown_report,
    load_state_history_tuples,
    reconstruct_state_history_row,
    replay_state_history,
)
from anima_mcp.sensor_residuals import (
    CHANNEL_SPECS,
    COMPARISON_VERSION,
    SCHEMA_VERSION as RESIDUALS_SCHEMA_VERSION,
    OnlineCorrelation,
    SensorResidualTracker,
)
from anima_mcp.sensors.base import SensorReadings


def _create_state_history_db(
    db_path: Path,
    rows: Iterable[Tuple[Any, Any, Any, Any, Any, Any]],
) -> Path:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                warmth REAL,
                clarity REAL,
                stability REAL,
                presence REAL,
                sensors TEXT DEFAULT '{}',
                epoch INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            "CREATE INDEX idx_state_history_time ON state_history(timestamp)"
        )
        conn.executemany(
            """
            INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            list(rows),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _synthetic_rows(
    count: int,
    *,
    start: datetime = datetime(2026, 11, 1, 0, 0, 0),
    interval_s: float = 240.0,
    flat_memory: bool = False,
    flat_stability_channels: bool = False,
    include_external_light: bool = True,
) -> list[Tuple[str, float, float, float, float, str]]:
    rows = []
    for i in range(count):
        ts = start + timedelta(seconds=i * interval_s)
        w1 = math.sin(i * 0.25)
        w2 = math.cos(i * 0.17)
        amb = 25.0 + 2.5 * w1
        cpu_t = 54.0 + 3.5 * w1 + 1.5 * w2
        if flat_stability_channels:
            hum = 35.0
            pres = 820.0
            mem = 20.0
            delta = 0.90
            theta = 0.02
        else:
            hum = 35.0 + 6.0 * w1
            pres = 820.0 + 3.0 * w2
            mem = 20.0 + (0.05 * w1 if flat_memory else 2.5 * w2)
            delta = 0.88 + 0.06 * w1
            theta = 0.08 + 0.05 * w2

        # Daytime-only gated external light every other sample
        if include_external_light and i % 2 == 0:
            ext_lux = max(1.0, 120.0 + 80.0 * w1)
            light_status = "ready_shadow"
        else:
            ext_lux = None
            light_status = "inconclusive"

        # Legacy anima correlated positively with warmth & light, negatively with deviation magnitude
        warmth = max(0.0, min(1.0, 0.55 + 0.18 * w1 + 0.05 * w2))
        dev_mag = abs(w1) + abs(w2)
        stability = max(0.0, min(1.0, 0.90 - 0.12 * dev_mag))
        clarity = max(0.0, min(1.0, 0.70 + 0.15 * w1))
        presence = 0.72

        sensors: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "ambient_temp_c": amb,
            "cpu_temp_c": cpu_t,
            "humidity_pct": hum,
            "pressure_hpa": pres,
            "light_lux": 900.0 + 100.0 * w2,  # raw lux must be ignored by tracker
            "external_light_lux": ext_lux,
            "light_attribution_status": light_status,
            "memory_percent": mem,
            "cpu_percent": 6.0 + 3.0 * abs(w2),
            "disk_percent": 65.0 + 0.001 * i,
            "eeg_delta_power": delta,
            "eeg_theta_power": theta,
            "hearing_available": False,
        }
        rows.append((ts.isoformat(), warmth, clarity, stability, presence, json.dumps(sensors)))
    return rows


# ---------------------------------------------------------------------------
# 1. Read-only SQLite reconstruction & Invariant 2 (fail toward unknown)
# ---------------------------------------------------------------------------


def test_reconstruct_row_handles_nulls_malformed_json_and_light_gate():
    # Malformed JSON -> valid_sensors_json=False, all sensor fields None
    rec_bad = reconstruct_state_history_row(
        "2026-11-10T12:00:00",
        None,
        float("nan"),
        0.8,
        float("inf"),
        "{not valid json",
    )
    assert rec_bad is not None
    assert rec_bad.valid_sensors_json is False
    assert rec_bad.readings.ambient_temp_c is None
    assert rec_bad.readings.memory_percent is None
    assert rec_bad.external_light_lux is None
    assert rec_bad.anima.warmth is None
    assert rec_bad.anima.clarity is None
    assert rec_bad.anima.stability == pytest.approx(0.8)
    assert rec_bad.anima.presence is None

    # Non-ready attribution status blocks external_light_lux even if numeric value is present
    payload = json.dumps(
        {
            "ambient_temp_c": 24.5,
            "cpu_temp_c": True,  # bool must be rejected as None
            "humidity_pct": float("nan"),
            "light_lux": 777.0,
            "external_light_lux": 123.0,
            "light_attribution_status": "inconclusive",
            "memory_percent": None,
        },
        allow_nan=True,
    )
    rec_gated = reconstruct_state_history_row(
        "2026-11-10T12:04:00Z", 0.5, 0.7, 0.8, 0.75, payload
    )
    assert rec_gated is not None
    assert rec_gated.valid_sensors_json is True
    assert rec_gated.readings.ambient_temp_c == pytest.approx(24.5)
    assert rec_gated.readings.cpu_temp_c is None
    assert rec_gated.readings.humidity_pct is None
    assert rec_gated.readings.light_lux == pytest.approx(777.0)
    assert rec_gated.external_light_lux is None  # gated out by inconclusive status

    # Bad timestamp returns None
    assert reconstruct_state_history_row("not-a-timestamp", 0.5, 0.5, 0.5, 0.5, "{}") is None
    assert reconstruct_state_history_row(None, 0.5, 0.5, 0.5, 0.5, "{}") is None


def test_load_state_history_tuples_readonly_and_chronological(tmp_path: Path):
    rows = _synthetic_rows(5, start=datetime(2026, 11, 10, 0, 0, 0), interval_s=240.0)
    # Insert out of order to verify ORDER BY timestamp ASC
    db_path = _create_state_history_db(tmp_path / "ro_test.db", list(reversed(rows)))
    # Mark DB file strictly read-only (0o444) to prove ?mode=ro&immutable=1 writes nothing
    os.chmod(db_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        tuples = load_state_history_tuples(db_path)
        assert len(tuples) == 5
        timestamps = [t[0] for t in tuples]
        assert timestamps == sorted(timestamps)
        for ts, readings, anima in tuples:
            assert isinstance(ts, datetime)
            assert isinstance(readings, SensorReadings)
            assert anima.readings is readings
        # No WAL/SHM files created
        assert not (tmp_path / "ro_test.db-wal").exists()
        assert not (tmp_path / "ro_test.db-shm").exists()
    finally:
        os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# 2. Replay through real SensorResidualTracker + Stage-2 readiness evaluation
# ---------------------------------------------------------------------------


def test_replay_state_history_metrics_correlations_and_sigma_floor_watch(tmp_path: Path):
    # 120 rows at 240s spacing (~8 hours), flat memory_percent -> 100% sigma-floored
    rows = _synthetic_rows(
        120,
        start=datetime(2026, 11, 10, 0, 0, 0),
        interval_s=240.0,
        flat_memory=True,
    )
    # Inject 2 malformed/missing rows to verify hygiene & Invariant 2 handling
    rows.append((
        (datetime(2026, 11, 10, 0, 0, 0) + timedelta(seconds=120 * 240.0)).isoformat(),
        None,
        None,
        None,
        None,
        "{broken json",
    ))
    rows.append((
        "bad-ts",
        0.5,
        0.5,
        0.5,
        0.5,
        "{}",
    ))
    db_path = _create_state_history_db(tmp_path / "replay.db", rows)

    report = replay_state_history(db_path, min_samples=20, window_samples=80)
    assert report["available"] is True
    assert report["config"]["mode"] == "custom_override"
    assert report["config"]["using_production_defaults"] is False
    assert report["source"]["rows_replayed"] == 121
    assert report["source"]["data_hygiene"]["malformed_sensor_json_rows"] == 1
    assert report["source"]["data_hygiene"]["skipped_bad_timestamp_rows"] == 1
    assert report["source"]["data_hygiene"]["null_anima_rows"] == 1

    # Sampling interval & broker cadence comparison
    assert report["sampling"]["median_seconds"] == pytest.approx(240.0, abs=0.01)
    assert report["sampling"]["cadence_ratio_vs_broker"] == pytest.approx(120.0, abs=0.1)
    assert report["sampling"]["implied_24h_rows"] == 360

    # Per-channel stats
    amb = report["channels"]["ambient_temp_c"]
    assert amb["n"] == 120
    assert amb["warm_ticks"] == 100
    assert amb["missing_ticks"] == 1
    assert amb["p05"] < amb["p50"] < amb["p95"]
    assert amb["sd"] > 0.2
    assert amb["sigma_floored_fraction"] == 0.0

    # memory_percent is flat (sd ~0.035 < min_sigma=1.0) -> 100% sigma floored and highlighted
    mem = report["channels"]["memory_percent"]
    assert mem["sigma_floor_watch"] is True
    assert mem["sigma_floored_fraction"] == pytest.approx(1.0)
    assert mem["current_sigma_floored"] is True

    # external_light_lux was valid on ~50% of ticks (60/121), warm after 20 valid samples -> 40 warm
    light = report["channels"]["external_light_lux"]
    assert light["n"] == 60
    assert light["warm_ticks"] == 40
    assert light["valid_fraction"] == pytest.approx(60 / 121, abs=1e-3)

    # Composites & shadow-vs-legacy correlations
    assert report["composites"]["warmth"]["n"] == 100
    assert report["composites"]["stability"]["n"] == 100
    assert report["composites"]["stability"]["sigma_floored_fraction"] == pytest.approx(1.0)
    assert report["comparisons"]["warmth"]["pearson_r"] > 0.7
    assert report["comparisons"]["stability"]["pearson_r"] < -0.2
    assert report["comparisons"]["stability"]["shadow_variance"] > 1e-4

    # Stage-2 gates
    gates = {g["id"]: g for g in report["stage2_readiness"]["gates"]}
    assert report["stage2_readiness"]["stage2_ready"] is False
    assert gates["seven_day_live_warm_coverage"]["status"] == "fail"
    assert gates["non_degenerate_stability_variance"]["status"] == "pass"
    assert gates["non_degenerate_warmth_variance"]["status"] == "pass"
    assert gates["sigma_floor_watch_channels"]["status"] == "warn"
    assert gates["python_elixir_parity_checklist"]["status"] == "manual_required"

    # Markdown output renders cleanly and highlights watch channels + checklist
    md = format_markdown_report(report)
    assert "CUSTOM OVERRIDE" in md
    assert "memory_percent" in md
    assert "eeg_delta_power" in md
    assert "eeg_theta_power" in md
    assert "eisv_mapper.ex" in md
    assert "240.00s" in md


def test_replay_days_window_and_production_defaults_stay_honest(tmp_path: Path):
    # 20 days of rows at 1-hour intervals (480 rows)
    rows = _synthetic_rows(480, start=datetime(2026, 11, 1, 0, 0, 0), interval_s=3600.0)
    db_path = _create_state_history_db(tmp_path / "window.db", rows)

    # --days 7 anchored to max DB timestamp -> ~169 rows (inclusive boundary)
    report = replay_state_history(db_path, days=7)
    assert report["config"]["mode"] == "production_defaults"
    assert report["config"]["using_production_defaults"] is True
    assert report["config"]["min_samples"] == DEFAULT_MIN_SAMPLES
    assert report["source"]["rows_replayed"] == 169
    assert report["source"]["wall_clock_span_days"] == pytest.approx(7.0, abs=0.01)

    # With production min_samples=43200 and only 169 rows, every channel is warming (n_warm=0)
    assert report["channels"]["ambient_temp_c"]["warm_ticks"] == 0
    assert report["channels"]["ambient_temp_c"]["p50"] is None
    assert report["channels"]["ambient_temp_c"]["sigma_floored_fraction"] is None
    gates = {g["id"]: g for g in report["stage2_readiness"]["gates"]}
    assert gates["non_degenerate_stability_variance"]["status"] == "unknown"
    assert gates["seven_day_live_warm_coverage"]["status"] == "fail"


def test_replay_detects_degenerate_stability_variance(tmp_path: Path):
    # All 5 stability channels completely flat -> z=0 -> stability.mean_abs_z variance == 0
    # Use frozen_run_threshold high enough that humidity/pressure aren't frozen yet
    rows = _synthetic_rows(
        40,
        start=datetime(2026, 11, 1, 0, 0, 0),
        interval_s=60.0,
        flat_stability_channels=True,
    )
    db_path = _create_state_history_db(tmp_path / "flat.db", rows)

    report = replay_state_history(
        db_path, min_samples=10, window_samples=40, frozen_run_threshold=100
    )
    stab_comp = report["comparisons"]["stability"]
    assert stab_comp["n"] == 30
    assert stab_comp["shadow_variance"] == pytest.approx(0.0)
    gates = {g["id"]: g for g in report["stage2_readiness"]["gates"]}
    assert gates["non_degenerate_stability_variance"]["status"] == "fail"


def test_replay_empty_and_missing_table(tmp_path: Path):
    empty_db = _create_state_history_db(tmp_path / "empty.db", [])
    report = replay_state_history(empty_db)
    assert report["source"]["rows_replayed"] == 0
    assert report["channels"]["ambient_temp_c"]["warm_fraction"] is None
    assert report["stage2_readiness"]["stage2_ready"] is False

    no_table = tmp_path / "notable.db"
    sqlite3.connect(str(no_table)).close()
    with pytest.raises(ValueError, match="state_history"):
        replay_state_history(no_table)


# ---------------------------------------------------------------------------
# 3. Persisted sensor_residuals.json snapshot evaluation
# ---------------------------------------------------------------------------


def test_evaluate_snapshot_warming_and_full_7d_live(tmp_path: Path):
    # 3a. Real warming snapshot saved by SensorResidualTracker
    snap_path = tmp_path / "warming_sr.json"
    tracker = SensorResidualTracker(snap_path, min_samples=DEFAULT_MIN_SAMPLES, auto_load=False)
    for i in range(15):
        w = math.sin(i * 0.4)
        tracker.observe(
            SensorReadings(
                timestamp=datetime(2026, 11, 19, 12, 0, 0) + timedelta(seconds=i * 2),
                ambient_temp_c=24.0 + w,
                cpu_temp_c=54.0 + 2.0 * w,
                humidity_pct=35.0 + 2.0 * w,
                pressure_hpa=820.0 + w,
                memory_percent=20.0 + 0.1 * w,
                cpu_percent=6.0 + w,
                disk_percent=65.0,
                eeg_delta_power=0.88 + 0.05 * w,
                eeg_theta_power=0.05 + 0.04 * w,
            ),
            external_light_lux=100.0 + 20.0 * w,
            current_warmth=0.55 + 0.05 * w,
            current_stability=0.82 - 0.03 * abs(w),
            current_clarity=0.75 + 0.04 * w,
            now=1_795_000_000.0 + i * 2.0,
        )
    assert tracker.save(now=1_795_000_050.0)

    warm_rep = evaluate_snapshot(snap_path)
    assert warm_rep["source"]["kind"] == "persisted_snapshot"
    assert warm_rep["source"]["observed_ticks"] == 15
    assert warm_rep["channels"]["memory_percent"]["current_sigma_floored"] is True
    assert warm_rep["channels"]["ambient_temp_c"]["p50"] is None  # histogram only
    gates = {g["id"]: g for g in warm_rep["stage2_readiness"]["gates"]}
    assert gates["seven_day_live_warm_coverage"]["status"] == "fail"
    assert gates["non_degenerate_stability_variance"]["status"] == "unknown"

    # 3b. Synthetic full 7-day live warm snapshot (345,600+ ticks) -> all automated gates pass
    corr_warmth = OnlineCorrelation()
    corr_stab = OnlineCorrelation()
    for i in range(100):
        w = math.sin(i * 0.2)
        corr_warmth.update(w, 0.5 + 0.1 * w)
        corr_stab.update(abs(w) + 0.2, 0.85 - 0.1 * abs(w))

    full_channels = {}
    for spec in CHANNEL_SPECS:
        # Make variance strictly above min_sigma^2
        var_val = (spec.min_sigma * 2.0) ** 2
        full_channels[spec.name] = {
            "n": 350_000,
            "mean": 10.0,
            "var": var_val,
            "observed_ticks": 350_000,
            "valid_ticks": 350_000,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": 1000,
            "status_counts": {
                "warm": 306_800,
                "warming": 43_200,
                "missing": 0,
                "frozen": 0,
            },
            "z_abs_histogram": {"lt_1": 200_000, "1_to_2": 80_000, "2_to_3": 20_000, "ge_3": 6_800},
        }
    full_payload = {
        "schema_version": RESIDUALS_SCHEMA_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "saved_at": 1_795_700_000.0,
        "observed_ticks": 350_000,
        "channels": full_channels,
        "correlations": {
            "warmth_signed_z_vs_current_warmth": corr_warmth.to_dict(),
            "stability_mean_abs_z_vs_current_stability": corr_stab.to_dict(),
            "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
        },
        "composites": {
            "warmth": {"status": "warm"},
            "stability": {"status": "warm"},
            "clarity_light": {"status": "warm"},
        },
    }
    full_path = tmp_path / "full_7d_sr.json"
    full_path.write_text(json.dumps(full_payload))

    full_rep = evaluate_snapshot(full_path)
    full_gates = {g["id"]: g for g in full_rep["stage2_readiness"]["gates"]}
    assert full_gates["seven_day_live_warm_coverage"]["status"] == "pass"
    assert full_gates["non_degenerate_stability_variance"]["status"] == "pass"
    assert full_gates["non_degenerate_warmth_variance"]["status"] == "pass"
    assert full_gates["sigma_floor_watch_channels"]["status"] == "pass"
    assert full_gates["python_elixir_parity_checklist"]["status"] == "manual_required"
    assert full_rep["stage2_readiness"]["blocking_gates"] == ["python_elixir_parity_checklist"]
    assert full_rep["comparisons"]["warmth"]["pearson_r"] == pytest.approx(1.0)
    assert full_rep["comparisons"]["stability"]["pearson_r"] == pytest.approx(-1.0)

    # Custom min_samples override on snapshot must NOT pass production 7-day soak gate
    override_rep = evaluate_snapshot(full_path, min_samples=100)
    override_gates = {g["id"]: g for g in override_rep["stage2_readiness"]["gates"]}
    assert override_gates["seven_day_live_warm_coverage"]["status"] == "fail"

    # Corrupt/invalid schema snapshot fails cleanly without modifying file on disk
    bad_snap = tmp_path / "bad_sr.json"
    bad_snap.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema"):
        evaluate_snapshot(bad_snap)
    assert bad_snap.exists()
    assert not (tmp_path / "bad_sr.json.corrupt").exists()


# ---------------------------------------------------------------------------
# 4. CLI script integration + isolation guardrails
# ---------------------------------------------------------------------------


def _load_cli_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "replay_sensor_residuals.py"
    )
    spec = importlib.util.spec_from_file_location("replay_sensor_residuals", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_db_replay_snapshot_and_argument_validation(tmp_path: Path, capsys):
    cli = _load_cli_module()
    rows = _synthetic_rows(60, start=datetime(2026, 11, 10, 0, 0, 0), interval_s=240.0)
    db_path = _create_state_history_db(tmp_path / "cli.db", rows)
    json_out = tmp_path / "out" / "report.json"

    rc = cli.main(
        [
            "--db",
            str(db_path),
            "--days",
            "1",
            "--min-samples",
            "10",
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "# Sensor Residual Backtest & Stage-2 Readiness Report" in captured.out
    assert json_out.exists()
    saved = json.loads(json_out.read_text())
    assert saved["source"]["rows_replayed"] == 60
    assert saved["config"]["min_samples"] == 10

    # Snapshot mode with --format json
    snap_path = tmp_path / "snap.json"
    tracker = SensorResidualTracker(snap_path, min_samples=5, auto_load=False)
    tracker.save(now=1_795_000_000.0)
    rc_snap = cli.main(["--snapshot", str(snap_path), "--format", "json"])
    assert rc_snap == 0
    snap_out = json.loads(capsys.readouterr().out)
    assert snap_out["source"]["kind"] == "persisted_snapshot"

    # Reject specifying both --db and --snapshot
    with pytest.raises(SystemExit) as exc_both:
        cli.main(["--db", str(db_path), "--snapshot", str(snap_path)])
    assert exc_both.value.code == 2

    # Reject --days with --snapshot
    with pytest.raises(SystemExit) as exc_days:
        cli.main(["--snapshot", str(snap_path), "--days", "7"])
    assert exc_days.value.code == 2

    # Missing file returns exit code 2 cleanly
    assert cli.main(["--db", str(tmp_path / "missing.db")]) == 2


def test_residual_backtest_isolation_guardrails():
    src_root = Path(__file__).resolve().parents[1] / "src" / "anima_mcp"
    backtest_tree = ast.parse((src_root / "residual_backtest.py").read_text())
    for node in ast.walk(backtest_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("save", "maybe_save"):
                raise AssertionError(
                    f"residual_backtest.py must not call .{node.func.attr}()"
                )

    # Runtime sensing/governance modules must never import residual_backtest
    for rel in ("anima.py", "eisv_mapper.py", "unitares_bridge.py"):
        tree = ast.parse((src_root / rel).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "residual_backtest" in node.module
            ):
                raise AssertionError(f"{rel} must not import residual_backtest")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "residual_backtest" in alias.name:
                        raise AssertionError(f"{rel} must not import residual_backtest")
