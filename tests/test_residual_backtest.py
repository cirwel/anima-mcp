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
    iter_state_history_records,
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

    # Missing light_attribution_status blocks external_light_lux (strict ready_shadow gate)
    rec_missing_status = reconstruct_state_history_row(
        "2026-11-10T12:08:00",
        0.5,
        0.7,
        0.8,
        0.75,
        json.dumps({"external_light_lux": 140.0, "light_lux": 500.0}),
    )
    assert rec_missing_status is not None
    assert rec_missing_status.external_light_lux is None

    # String or bool external_light_lux under ready_shadow must be rejected
    for bad_lux in ("140.0", True, False, None):
        rec_bad_lux = reconstruct_state_history_row(
            "2026-11-10T12:12:00",
            0.5,
            0.7,
            0.8,
            0.75,
            json.dumps(
                {
                    "external_light_lux": bad_lux,
                    "light_attribution_status": "ready_shadow",
                }
            ),
        )
        assert rec_bad_lux is not None
        assert rec_bad_lux.external_light_lux is None

    # Finite numeric external_light_lux with ready_shadow passes through
    rec_ready = reconstruct_state_history_row(
        "2026-11-10T12:16:00",
        0.5,
        0.7,
        0.8,
        0.75,
        json.dumps(
            {
                "external_light_lux": 140.5,
                "light_attribution_status": "ready_shadow",
            }
        ),
    )
    assert rec_ready is not None
    assert rec_ready.external_light_lux == pytest.approx(140.5)

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

    # 3b. Snapshot with per-channel warm >= 302,400 but low paired correlation count (100):
    # simultaneous contributor overlap is NOT proven -> seven_day_live_warm_coverage == "unknown"
    corr_warmth_small = OnlineCorrelation()
    corr_stab_small = OnlineCorrelation()
    for i in range(100):
        w = math.sin(i * 0.2)
        corr_warmth_small.update(w, 0.5 + 0.1 * w)
        corr_stab_small.update(abs(w) + 0.2, 0.85 - 0.1 * abs(w))

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
    unproven_payload = {
        "schema_version": RESIDUALS_SCHEMA_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "saved_at": 1_795_700_000.0,
        "observed_ticks": 350_000,
        "channels": full_channels,
        "correlations": {
            "warmth_signed_z_vs_current_warmth": corr_warmth_small.to_dict(),
            "stability_mean_abs_z_vs_current_stability": corr_stab_small.to_dict(),
            "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
        },
        "composites": {
            "warmth": {"status": "warm"},
            "stability": {"status": "warm"},
            "clarity_light": {"status": "warm"},
        },
    }
    unproven_path = tmp_path / "unproven_7d_sr.json"
    unproven_path.write_text(json.dumps(unproven_payload))
    unproven_rep = evaluate_snapshot(unproven_path)
    unproven_gates = {g["id"]: g for g in unproven_rep["stage2_readiness"]["gates"]}
    assert unproven_gates["seven_day_live_warm_coverage"]["status"] == "unknown"

    # 3c. Synthetic full 7-day live warm snapshot with proven >=302,400 paired simultaneous warm ticks
    n_full = 306_800
    full_payload = dict(unproven_payload)
    full_payload["correlations"] = {
        "warmth_signed_z_vs_current_warmth": {
            "n": n_full,
            "mean_x": 0.1,
            "mean_y": 0.55,
            "m2_x": 0.25 * n_full,
            "m2_y": 0.01 * n_full,
            "c_xy": 0.04 * n_full,  # r = 0.04 / sqrt(0.0025) = +0.8
        },
        "stability_mean_abs_z_vs_current_stability": {
            "n": n_full,
            "mean_x": 0.5,
            "mean_y": 0.80,
            "m2_x": 0.25 * n_full,
            "m2_y": 0.01 * n_full,
            "c_xy": -0.025 * n_full,  # r = -0.5
        },
        "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
    }
    full_path = tmp_path / "full_7d_sr.json"
    full_path.write_text(json.dumps(full_payload))

    full_rep = evaluate_snapshot(full_path)
    full_gates = {g["id"]: g for g in full_rep["stage2_readiness"]["gates"]}
    assert full_gates["seven_day_live_warm_coverage"]["status"] == "pass"
    assert full_gates["non_degenerate_stability_variance"]["status"] == "pass"
    assert full_gates["non_degenerate_warmth_variance"]["status"] == "pass"
    assert full_gates["shadow_legacy_correlation_alignment"]["status"] == "pass"
    assert full_gates["sigma_floor_watch_channels"]["status"] == "pass"
    assert full_gates["python_elixir_parity_checklist"]["status"] == "manual_required"
    assert full_rep["stage2_readiness"]["blocking_gates"] == ["python_elixir_parity_checklist"]
    assert full_rep["comparisons"]["warmth"]["pearson_r"] == pytest.approx(0.8)
    assert full_rep["comparisons"]["stability"]["pearson_r"] == pytest.approx(-0.5)

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


def test_snapshot_sigma_floor_watch_unwarm_channel_and_small_variance(tmp_path: Path):
    # Case A: 2 watch channels warm & low-floored, 1 watch channel (eeg_theta_power) unwarm -> "unknown"
    # Plus variance = 1.21e-6 (would round to 1e-6 at 6 decimals) must PASS unrounded > 1e-6 check
    channels = {}
    for spec in CHANNEL_SPECS:
        warm_ticks = 0 if spec.name == "eeg_theta_power" else 500
        channels[spec.name] = {
            "n": 1000 if warm_ticks else 10,
            "mean": 1.0,
            "var": (spec.min_sigma * 2.0) ** 2,
            "observed_ticks": 1000,
            "valid_ticks": 1000 if warm_ticks else 10,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": 0,
            "status_counts": {
                "warm": warm_ticks,
                "warming": 1000 - warm_ticks,
                "missing": 0,
                "frozen": 0,
            },
            "z_abs_histogram": {"lt_1": warm_ticks, "1_to_2": 0, "2_to_3": 0, "ge_3": 0},
        }
    n_pair = 100
    payload = {
        "schema_version": RESIDUALS_SCHEMA_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "saved_at": 1_795_700_000.0,
        "observed_ticks": 1000,
        "channels": channels,
        "correlations": {
            "warmth_signed_z_vs_current_warmth": {
                "n": n_pair,
                "mean_x": 0.0,
                "mean_y": 0.5,
                "m2_x": 1.21e-6 * n_pair,
                "m2_y": 0.01 * n_pair,
                "c_xy": 0.5 * math.sqrt(1.21e-6 * 0.01) * n_pair,  # r = +0.5
            },
            "stability_mean_abs_z_vs_current_stability": {
                "n": n_pair,
                "mean_x": 0.5,
                "mean_y": 0.8,
                "m2_x": 1.21e-6 * n_pair,
                "m2_y": 0.01 * n_pair,
                "c_xy": -0.5 * math.sqrt(1.21e-6 * 0.01) * n_pair,  # r = -0.5
            },
            "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
        },
        "composites": {},
    }
    snap_a = tmp_path / "unwarm_watch.json"
    snap_a.write_text(json.dumps(payload))
    rep_a = evaluate_snapshot(snap_a)
    gates_a = {g["id"]: g for g in rep_a["stage2_readiness"]["gates"]}
    assert gates_a["sigma_floor_watch_channels"]["status"] == "unknown"
    assert gates_a["non_degenerate_stability_variance"]["status"] == "pass"
    assert gates_a["non_degenerate_warmth_variance"]["status"] == "pass"
    assert gates_a["shadow_legacy_correlation_alignment"]["status"] == "pass"

    # Case B: one watch channel breaches >=50% floored while another is still unwarm -> "warn"
    payload["channels"]["memory_percent"]["floored_ticks"] = 400
    snap_b = tmp_path / "warn_with_unwarm.json"
    snap_b.write_text(json.dumps(payload))
    rep_b = evaluate_snapshot(snap_b)
    gates_b = {g["id"]: g for g in rep_b["stage2_readiness"]["gates"]}
    assert gates_b["sigma_floor_watch_channels"]["status"] == "warn"

    # Case C: snapshot variance scope explicitly reports paired warm samples (never "all warm ticks")
    assert gates_a["non_degenerate_stability_variance"]["evidence"]["sample_scope"] == "paired_warm_ticks"
    assert "paired warm samples" in gates_a["non_degenerate_stability_variance"]["reason"]
    assert "all warm ticks" not in gates_a["non_degenerate_stability_variance"]["reason"]
    assert gates_a["non_degenerate_warmth_variance"]["evidence"]["sample_scope"] == "paired_warm_ticks"
    assert "paired warm samples" in gates_a["non_degenerate_warmth_variance"]["reason"]

    # Case D: missing floored_ticks key on a watched channel in JSON -> "unknown", never "pass"
    missing_ft_channels = {}
    for spec in CHANNEL_SPECS:
        ch = {
            "n": 1000,
            "mean": 1.0,
            "var": (spec.min_sigma * 2.0) ** 2,
            "observed_ticks": 1000,
            "valid_ticks": 1000,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": 0,
            "status_counts": {"warm": 500, "warming": 500, "missing": 0, "frozen": 0},
            "z_abs_histogram": {"lt_1": 500, "1_to_2": 0, "2_to_3": 0, "ge_3": 0},
        }
        if spec.name == "eeg_delta_power":
            del ch["floored_ticks"]
        missing_ft_channels[spec.name] = ch
    missing_ft_payload = dict(payload, channels=missing_ft_channels)
    snap_c = tmp_path / "missing_floored_ticks.json"
    snap_c.write_text(json.dumps(missing_ft_payload))
    rep_c = evaluate_snapshot(snap_c)
    assert rep_c["channels"]["eeg_delta_power"]["floored_ticks"] is None
    assert rep_c["channels"]["eeg_delta_power"]["floored_ticks_known"] is False
    assert rep_c["channels"]["eeg_delta_power"]["sigma_floored_fraction"] is None
    gates_c = {g["id"]: g for g in rep_c["stage2_readiness"]["gates"]}
    assert gates_c["sigma_floor_watch_channels"]["status"] == "unknown"

    # Case E: rotating 3-of-5 stability contributors (<302,400 per channel, >=302,400 composite warm) -> "pass"
    n_rot = 306_800
    rot_channels = {}
    for spec in CHANNEL_SPECS:
        warm_t = 250_000 if spec.composite == "stability" else 306_800
        rot_channels[spec.name] = {
            "n": 350_000,
            "mean": 10.0,
            "var": (spec.min_sigma * 2.0) ** 2,
            "observed_ticks": 350_000,
            "valid_ticks": 350_000,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": 0,
            "status_counts": {"warm": warm_t, "warming": 43_200, "missing": 0, "frozen": 0},
            "z_abs_histogram": {"lt_1": warm_t, "1_to_2": 0, "2_to_3": 0, "ge_3": 0},
        }
    rot_payload = {
        "schema_version": RESIDUALS_SCHEMA_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "saved_at": 1_795_700_000.0,
        "observed_ticks": 350_000,
        "channels": rot_channels,
        "correlations": {
            "warmth_signed_z_vs_current_warmth": {
                "n": n_rot,
                "mean_x": 0.1,
                "mean_y": 0.55,
                "m2_x": 0.25 * n_rot,
                "m2_y": 0.01 * n_rot,
                "c_xy": 0.04 * n_rot,
            },
            "stability_mean_abs_z_vs_current_stability": {
                "n": n_rot,
                "mean_x": 0.5,
                "mean_y": 0.80,
                "m2_x": 0.25 * n_rot,
                "m2_y": 0.01 * n_rot,
                "c_xy": -0.025 * n_rot,
            },
            "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
        },
        "composites": {},
    }
    snap_rot = tmp_path / "rotating_3_of_5.json"
    snap_rot.write_text(json.dumps(rot_payload))
    rep_rot = evaluate_snapshot(snap_rot)
    gates_rot = {g["id"]: g for g in rep_rot["stage2_readiness"]["gates"]}
    assert gates_rot["seven_day_live_warm_coverage"]["status"] == "pass"

    # Case F: skewed 3-of-5 stability channel counts [306800, 306800, 300000, 0, 0]
    # Naive sum // 3 = 304,533 >= 302,400, but exact tight bound is 300,000 < 302,400 -> "fail"
    skewed_counts = {
        "humidity_pct": 306_800,
        "pressure_hpa": 306_800,
        "memory_percent": 300_000,
        "eeg_delta_power": 0,
        "eeg_theta_power": 0,
    }
    skewed_channels = {}
    for spec in CHANNEL_SPECS:
        warm_t = skewed_counts.get(spec.name, 306_800)
        skewed_channels[spec.name] = {
            "n": 350_000,
            "mean": 10.0,
            "var": (spec.min_sigma * 2.0) ** 2,
            "observed_ticks": 350_000,
            "valid_ticks": 350_000,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": 0,
            "status_counts": {"warm": warm_t, "warming": 350_000 - warm_t, "missing": 0, "frozen": 0},
            "z_abs_histogram": {"lt_1": warm_t, "1_to_2": 0, "2_to_3": 0, "ge_3": 0},
        }
    skewed_payload = {
        "schema_version": RESIDUALS_SCHEMA_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "saved_at": 1_795_700_000.0,
        "observed_ticks": 350_000,
        "channels": skewed_channels,
        "correlations": {
            "warmth_signed_z_vs_current_warmth": OnlineCorrelation().to_dict(),
            "stability_mean_abs_z_vs_current_stability": OnlineCorrelation().to_dict(),
            "clarity_light_z_vs_current_clarity": OnlineCorrelation().to_dict(),
        },
        "composites": {},
    }
    snap_skewed = tmp_path / "skewed_3_of_5.json"
    snap_skewed.write_text(json.dumps(skewed_payload))
    rep_skewed = evaluate_snapshot(snap_skewed)
    gates_skewed = {g["id"]: g for g in rep_skewed["stage2_readiness"]["gates"]}
    assert gates_skewed["seven_day_live_warm_coverage"]["evidence"]["stability_simultaneous_upper_bound"] == 300_000
    assert gates_skewed["seven_day_live_warm_coverage"]["status"] == "fail"

    # Case G: non-finite floored_ticks (Infinity / 1e309) must not raise OverflowError; degrades to "unknown"
    inf_channels = {
        spec.name: {
            "n": 1000,
            "mean": 1.0,
            "var": (spec.min_sigma * 2.0) ** 2,
            "observed_ticks": 1000,
            "valid_ticks": 1000,
            "last_valid_at": 1_795_700_000.0,
            "floored_ticks": float("inf") if spec.name == "memory_percent" else 0,
            "status_counts": {"warm": 500, "warming": 500, "missing": 0, "frozen": 0},
            "z_abs_histogram": {"lt_1": 500, "1_to_2": 0, "2_to_3": 0, "ge_3": 0},
        }
        for spec in CHANNEL_SPECS
    }
    inf_payload = dict(payload, channels=inf_channels)
    snap_inf = tmp_path / "inf_floored_ticks.json"
    snap_inf.write_text(json.dumps(inf_payload, allow_nan=True))
    rep_inf = evaluate_snapshot(snap_inf)
    assert rep_inf["channels"]["memory_percent"]["floored_ticks"] is None
    assert rep_inf["channels"]["memory_percent"]["floored_ticks_known"] is False
    assert rep_inf["channels"]["memory_percent"]["sigma_floored_fraction"] is None
    assert rep_inf["channels"]["memory_percent"]["observed_ticks"] == 1000
    gates_inf = {g["id"]: g for g in rep_inf["stage2_readiness"]["gates"]}
    assert gates_inf["sigma_floor_watch_channels"]["status"] == "unknown"


def test_single_read_transaction_isolates_concurrent_wal_mutations(tmp_path: Path):
    # Create >500 rows (spans >1 batch in pass 2) in WAL mode
    wal_db = tmp_path / "txn_wal.db"
    rows = _synthetic_rows(550, start=datetime(2026, 11, 1, 0, 0, 0), interval_s=60.0)
    conn = sqlite3.connect(str(wal_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            warmth REAL,
            clarity REAL,
            stability REAL,
            presence REAL,
            sensors TEXT DEFAULT '{}'
        )
        """
    )
    conn.executemany(
        "INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    try:
        it = iter_state_history_records(wal_db)
        first = next(it)
        assert first.anima.warmth == pytest.approx(rows[0][1])

        # Mutate rows in the second batch (rowid > 500) on a separate WAL writer
        # while the read iterator is mid-stream; single BEGIN snapshot must isolate it!
        conn.execute("UPDATE state_history SET warmth = 0.123 WHERE id > 500")
        conn.commit()

        remaining = list(it)
        assert len(remaining) == 549
        # Row 550 must still reflect the original pre-mutation warmth, not 0.123
        assert remaining[-1].anima.warmth == pytest.approx(rows[-1][1])
    finally:
        conn.close()


def test_wal_readonly_days_anchor_mixed_tz_and_null_legacy_shadow_variance(tmp_path: Path):
    # 1. WAL-backed DB with uncommitted WAL frames must be visible under default ?mode=ro
    wal_db = tmp_path / "wal.db"
    conn = sqlite3.connect(str(wal_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            warmth REAL,
            clarity REAL,
            stability REAL,
            presence REAL,
            sensors TEXT DEFAULT '{}'
        )
        """
    )
    # Mix timezone offsets (+02:00 vs Z) out of lexical order AND include a "zzzz-bad-ts" row
    # that lexically sorts AFTER every ISO timestamp!
    base_rows = _synthetic_rows(50, start=datetime(2026, 11, 1, 0, 0, 0), interval_s=3600.0)
    # Replace all legacy anima values with NULL on one replay copy later; first test WAL + bad-ts + mixed TZ
    conn.executemany(
        "INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors) VALUES (?, ?, ?, ?, ?, ?)",
        base_rows,
    )
    # Add mixed-TZ rows representing 2026-11-05T08:00:00Z (+02:00 = 10:00) and 2026-11-05T09:00:00Z
    sample_sensors = base_rows[-1][5]
    conn.execute(
        "INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-11-05T09:00:00Z", 0.6, 0.7, 0.8, 0.7, sample_sensors),
    )
    conn.execute(
        "INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors) VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-11-05T10:00:00+02:00", 0.6, 0.7, 0.8, 0.7, sample_sensors),
    )
    # Lexically larger invalid timestamp must NOT break --days anchoring
    conn.execute(
        "INSERT INTO state_history (timestamp, warmth, clarity, stability, presence, sensors) VALUES (?, ?, ?, ?, ?, ?)",
        ("zzzz-bad-ts", 0.5, 0.5, 0.5, 0.5, sample_sensors),
    )
    conn.commit()
    # Keep writer open (WAL frames uncheckpointed) while reading strictly read-only
    try:
        tuples = load_state_history_tuples(wal_db, days=1)
        # Only the 2 rows within 1 day of 2026-11-05T09:00:00Z should be returned, in UTC order
        assert len(tuples) == 2
        assert tuples[0][0].isoformat() == "2026-11-05T10:00:00+02:00"  # 08:00Z first
        assert tuples[1][0].isoformat() == "2026-11-05T09:00:00+00:00"  # 09:00Z second

        rep_wal = replay_state_history(wal_db, days=1)
        assert rep_wal["source"]["sqlite_immutable"] is False
        assert rep_wal["source"]["rows_replayed"] == 2
        assert rep_wal["source"]["data_hygiene"]["skipped_bad_timestamp_rows"] == 1
    finally:
        conn.close()

    # Validate days up front even on empty DB
    empty_db = _create_state_history_db(tmp_path / "empty_days.db", [])
    for bad_days in (0, -5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="days"):
            replay_state_history(empty_db, days=bad_days)  # type: ignore[arg-type]

    # 2. Full-shadow variance works even when ALL legacy anima columns are NULL
    null_legacy_rows = [
        (ts, None, None, None, None, sensors)
        for ts, _, _, _, _, sensors in _synthetic_rows(60, interval_s=240.0)
    ]
    null_db = _create_state_history_db(tmp_path / "null_legacy.db", null_legacy_rows)
    rep_null = replay_state_history(null_db, min_samples=15, window_samples=50)
    gates_null = {g["id"]: g for g in rep_null["stage2_readiness"]["gates"]}
    assert rep_null["comparisons"]["warmth"]["n"] == 0
    assert rep_null["comparisons"]["stability"]["n"] == 0
    assert gates_null["non_degenerate_warmth_variance"]["status"] == "pass"
    assert gates_null["non_degenerate_stability_variance"]["status"] == "pass"
    assert gates_null["shadow_legacy_correlation_alignment"]["status"] == "unknown"

    # 3. Wrong-sign correlation (inverted legacy warmth & stability) fails alignment gate
    inverted_rows = [
        (ts, 1.0 - w, c, 1.0 - s, p, sensors)
        for ts, w, c, s, p, sensors in _synthetic_rows(80, interval_s=240.0)
    ]
    inv_db = _create_state_history_db(tmp_path / "inverted.db", inverted_rows)
    rep_inv = replay_state_history(inv_db, min_samples=15, window_samples=60)
    gates_inv = {g["id"]: g for g in rep_inv["stage2_readiness"]["gates"]}
    assert rep_inv["comparisons"]["warmth"]["pearson_r"] < 0
    assert rep_inv["comparisons"]["stability"]["pearson_r"] > 0
    assert gates_inv["shadow_legacy_correlation_alignment"]["status"] == "fail"
    assert "shadow_legacy_correlation_alignment" in rep_inv["stage2_readiness"]["blocking_gates"]


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

    # Reject --days and --immutable with --snapshot
    with pytest.raises(SystemExit) as exc_days:
        cli.main(["--snapshot", str(snap_path), "--days", "7"])
    assert exc_days.value.code == 2

    with pytest.raises(SystemExit) as exc_imm:
        cli.main(["--snapshot", str(snap_path), "--immutable"])
    assert exc_imm.value.code == 2

    # Missing file returns exit code 2 cleanly
    assert cli.main(["--db", str(tmp_path / "missing.db")]) == 2

    # [P1] --json-out must NEVER overwrite --db or --snapshot (direct, symlink, or hardlink)
    db_bytes_before = db_path.read_bytes()
    snap_bytes_before = snap_path.read_bytes()
    assert cli.main(["--db", str(db_path), "--json-out", str(db_path)]) == 2
    assert db_path.read_bytes() == db_bytes_before

    db_symlink = tmp_path / "db_link.db"
    db_symlink.symlink_to(db_path)
    assert cli.main(["--db", str(db_path), "--json-out", str(db_symlink)]) == 2
    assert db_path.read_bytes() == db_bytes_before

    db_hardlink = tmp_path / "db_hard.db"
    os.link(db_path, db_hardlink)
    assert cli.main(["--db", str(db_path), "--json-out", str(db_hardlink)]) == 2
    assert db_path.read_bytes() == db_bytes_before

    assert cli.main(["--snapshot", str(snap_path), "--json-out", str(snap_path)]) == 2
    assert snap_path.read_bytes() == snap_bytes_before

    # [P1] --json-out must NEVER overwrite SQLite sidecar files (-wal, -shm, -journal)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{db_path}{suffix}")
        assert cli.main(["--db", str(db_path), "--json-out", str(sidecar)]) == 2
        assert not sidecar.exists()

    wal_sidecar = Path(f"{db_path}-wal")
    wal_sidecar.write_bytes(b"simulated-wal-bytes")
    wal_symlink = tmp_path / "wal_link"
    wal_symlink.symlink_to(wal_sidecar)
    try:
        assert cli.main(["--db", str(db_path), "--json-out", str(wal_sidecar)]) == 2
        assert cli.main(["--db", str(db_path), "--json-out", str(wal_symlink)]) == 2
        assert wal_sidecar.read_bytes() == b"simulated-wal-bytes"
    finally:
        wal_symlink.unlink()
        wal_sidecar.unlink()

    # Corrupt SQLite DB and unwritable --json-out return exit code 2 without raw tracebacks
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_text("not a sqlite file at all")
    assert cli.main(["--db", str(corrupt_db)]) == 2
    assert "error:" in capsys.readouterr().err

    blocker = tmp_path / "file_blocker"
    blocker.write_text("x")
    assert cli.main(["--db", str(db_path), "--json-out", str(blocker / "out.json")]) == 2
    assert "error:" in capsys.readouterr().err

    # --immutable opt-in works cleanly on frozen backup DBs
    assert cli.main(["--db", str(db_path), "--immutable", "--format", "json"]) == 0
    imm_out = json.loads(capsys.readouterr().out)
    assert imm_out["source"]["sqlite_immutable"] is True


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
