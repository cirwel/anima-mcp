"""Tests for shadow-only per-sensor self-relative residuals (#79, #175)."""

from __future__ import annotations

import ast
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.anima import sense_self
from anima_mcp.eisv_mapper import anima_to_body_eisv_projection
from anima_mcp.sensor_residuals import (
    AUTHORITY,
    CHANNEL_SPEC_MAP,
    COMPARISON_VERSION,
    MOVES_NO_GATE,
    SCHEMA_VERSION,
    OnlineCorrelation,
    ResidualBand,
    SensorResidualTracker,
    load_persisted_sensor_residuals,
)
from anima_mcp.sensors.base import SensorReadings


def _make_readings(**overrides) -> SensorReadings:
    defaults = dict(
        timestamp=datetime(2026, 11, 19, 12, 0, 0),
        cpu_temp_c=52.0,
        ambient_temp_c=23.0,
        humidity_pct=45.0,
        light_lux=500.0,
        pressure_hpa=1013.0,
        cpu_percent=18.0,
        memory_percent=36.0,
        disk_percent=42.0,
        eeg_delta_power=0.80,
        eeg_theta_power=0.15,
    )
    defaults.update(overrides)
    return SensorReadings(**defaults)


def _feed_varying_cycle(
    tracker: SensorResidualTracker,
    count: int,
    *,
    t0: float = 1_700_000_000.0,
    light_override: float | None = 120.0,
) -> dict:
    snap = {}
    for i in range(count):
        wave = math.sin(i * 0.4)
        wave2 = math.cos(i * 0.3)
        readings = _make_readings(
            ambient_temp_c=23.0 + 1.5 * wave,
            cpu_temp_c=52.0 + 3.0 * wave + 0.5 * wave2,
            humidity_pct=45.0 + 4.0 * wave,
            pressure_hpa=1013.0 + 2.0 * wave2,
            light_lux=900.0 + 50.0 * wave,  # raw lux must be ignored
            memory_percent=36.0 + 2.0 * wave,
            cpu_percent=18.0 + 5.0 * wave2,
            disk_percent=42.0 + 0.01 * i,
            eeg_delta_power=0.75 + 0.08 * wave,
            eeg_theta_power=0.20 + 0.06 * wave2,
        )
        ext_lux = None if light_override is None else max(0.0, light_override + 30.0 * wave)
        snap = tracker.observe(
            readings,
            external_light_lux=ext_lux,
            current_warmth=0.55 + 0.1 * wave,
            current_stability=0.85 - 0.1 * abs(wave),
            current_clarity=0.70 + 0.08 * wave,
            now=t0 + i * 2.0,
        )
    return snap


# ---------------------------------------------------------------------------
# 1-2. Warm-up -> warm transition, pre-update z scoring, Welford -> EW handoff
# ---------------------------------------------------------------------------


def test_warmup_transition_and_pre_update_z_scoring(tmp_path: Path):
    spec = CHANNEL_SPEC_MAP["cpu_percent"]  # physical_frozen_guard=False, min_sigma=2.0
    band = ResidualBand(spec=spec, min_samples=4, window_samples=10)

    samples = [10.0, 20.0, 10.0, 20.0]  # mean=15.0, pop std=5.0
    for idx, val in enumerate(samples):
        res = band.observe(val, now=100.0 + idx)
        assert res["status"] == "warming"
        assert res["z"] is None
        assert res["samples"] == idx + 1

    # Next sample must be scored BEFORE being folded into mean/var
    prior_mean = band.mean
    prior_sigma = band.effective_sigma
    assert prior_mean == pytest.approx(15.0)
    assert prior_sigma == pytest.approx(5.0)

    res = band.observe(30.0, now=104.0)
    assert res["status"] == "warm"
    assert res["z"] == pytest.approx((30.0 - 15.0) / 5.0)
    assert res["sigma_floored"] is False
    # Post-update mean includes 30.0
    assert band.mean == pytest.approx(18.0)


def test_welford_to_ew_handoff_seamless_and_adapts(tmp_path: Path):
    spec = CHANNEL_SPEC_MAP["cpu_percent"]
    band = ResidualBand(spec=spec, min_samples=5, window_samples=20)

    # First 20 samples around 10.0 +/- 3.0 -> exact Welford window boundary
    for i in range(20):
        band.observe(10.0 + (3.0 if i % 2 == 0 else -3.0), now=float(i))

    assert band.n == 20
    assert band.mean == pytest.approx(10.0)
    assert band.raw_sigma == pytest.approx(3.0)

    # Sample 21 continues seamlessly with alpha = 1/20 without variance jump
    pre_sigma = band.raw_sigma
    res21 = band.observe(13.0, now=20.0)
    assert res21["status"] == "warm"
    assert res21["z"] == pytest.approx((13.0 - 10.0) / pre_sigma)

    # Regime shift to ~50.0 adapts exponentially instead of freezing on lifetime mean
    for i in range(21, 160):
        band.observe(50.0 + (2.0 if i % 2 == 0 else -2.0), now=float(i))
    assert band.mean > 49.0
    assert abs(band.raw_sigma - 2.0) < 0.6


# ---------------------------------------------------------------------------
# 3. Sigma floor transparency
# ---------------------------------------------------------------------------


def test_sigma_floored_transparency_and_tick_counter():
    spec = CHANNEL_SPEC_MAP["memory_percent"]  # min_sigma = 1.0
    band = ResidualBand(spec=spec, min_samples=3, window_samples=10)

    for i in range(3):
        band.observe(40.0, now=float(i))

    assert band.raw_sigma == pytest.approx(0.0)
    assert band.effective_sigma == pytest.approx(spec.min_sigma)

    res = band.observe(42.0, now=3.0)
    assert res["status"] == "warm"
    assert res["sigma_floored"] is True
    # Scored pre-update against prior mean=40.0 and floored sigma=1.0
    assert res["z"] == pytest.approx((42.0 - 40.0) / spec.min_sigma)
    assert band.floored_ticks == 1
    # Post-update sigma now reflects 42.0 but stays below min_sigma
    assert band.raw_sigma == pytest.approx(math.sqrt(0.75))
    assert band.sigma_floored is True


# ---------------------------------------------------------------------------
# 4-5. Deferred frozen-run guard vs non-physical channels
# ---------------------------------------------------------------------------


def test_deferred_frozen_guard_never_collapses_sigma_with_stuck_ticks(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=10,
        window_samples=50,
        frozen_run_threshold=60,
        auto_load=False,
    )
    # Build healthy variance on ambient_temp_c (varying 20..26)
    _feed_varying_cycle(tracker, 20, light_override=None)
    band = tracker.bands["ambient_temp_c"]
    baseline_sigma_before_stuck = band.raw_sigma
    assert baseline_sigma_before_stuck > 0.5

    # Feed 59 identical values on ambient_temp_c while changing other channels:
    # tick 1 folds immediately, ticks 2..59 are held (<60), NOT folded yet.
    for k in range(59):
        readings = _make_readings(
            ambient_temp_c=23.0,
            cpu_temp_c=50.0 + 0.1 * k,
            humidity_pct=45.0 + 0.2 * math.sin(k),
            pressure_hpa=1013.0 + 0.1 * math.cos(k),
        )
        res = tracker.observe(readings, now=2_000_000_000.0 + k * 2.0)
        assert res["channels"]["ambient_temp_c"]["status"] == "warm"
        assert res["channels"]["ambient_temp_c"]["z"] is not None

    # Tick 60 hits threshold -> marks frozen and discards the 58 held repeats
    res60 = tracker.observe(_make_readings(ambient_temp_c=23.0, cpu_temp_c=56.0), now=2_000_000_200.0)
    assert res60["channels"]["ambient_temp_c"]["status"] == "frozen"
    assert res60["channels"]["ambient_temp_c"]["z"] is None
    assert band.held_repeats == 0
    # Variance was NOT collapsed by 58 identical repeats
    assert band.raw_sigma >= baseline_sigma_before_stuck * 0.90

    # Tick 61 unsticks -> resumes warm immediately without flushing stuck repeats
    res61 = tracker.observe(_make_readings(ambient_temp_c=23.5, cpu_temp_c=56.2), now=2_000_000_202.0)
    assert res61["channels"]["ambient_temp_c"]["status"] == "warm"
    assert res61["channels"]["ambient_temp_c"]["z"] is not None


def test_short_identical_run_under_threshold_flushes_on_change():
    spec = CHANNEL_SPEC_MAP["ambient_temp_c"]
    band = ResidualBand(spec=spec, min_samples=3, window_samples=100, frozen_run_threshold=60)
    band.observe(20.0, now=1.0)
    band.observe(24.0, now=2.0)
    # 5 identical readings of 22.0 (< 60)
    for k in range(5):
        band.observe(22.0, now=3.0 + k)
    assert band.n == 3  # first 22.0 folded, 4 held
    # Change to 25.0 -> flushes all 4 copies of 22.0 before folding 25.0
    res = band.observe(25.0, now=10.0)
    assert res["status"] == "warm"
    assert band.n == 8


def test_non_physical_channels_exempt_from_frozen_guard(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=5,
        window_samples=50,
        frozen_run_threshold=60,
        auto_load=False,
    )
    snap = {}
    for k in range(100):
        readings = _make_readings(
            ambient_temp_c=22.0 + 0.5 * math.sin(k),
            cpu_temp_c=50.0 + 0.5 * math.cos(k),
            humidity_pct=45.0 + math.sin(k),
            pressure_hpa=1013.0 + math.cos(k),
            memory_percent=35.0,
            cpu_percent=12.0,
            disk_percent=42.0,
            eeg_delta_power=0.80,
            eeg_theta_power=0.10,
        )
        snap = tracker.observe(readings, now=1000.0 + k)

    for ch in ("memory_percent", "cpu_percent", "disk_percent", "eeg_delta_power", "eeg_theta_power"):
        assert snap["channels"][ch]["status"] == "warm"
        assert snap["channels"][ch]["samples"] == 100
        assert snap["channels"][ch]["z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6-9. Derived diff inheritance, log1p gated light, disk excluded, composites
# ---------------------------------------------------------------------------


def test_derived_cpu_above_ambient_inherits_parent_missing_and_frozen(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=5,
        window_samples=50,
        frozen_run_threshold=5,
        auto_load=False,
    )
    _feed_varying_cycle(tracker, 10)

    # Missing ambient_temp_c forces derived channel missing even when cpu_temp_c is finite
    snap_missing = tracker.observe(
        _make_readings(ambient_temp_c=None, cpu_temp_c=55.0),
        now=1_800_000_000.0,
    )
    assert snap_missing["channels"]["ambient_temp_c"]["status"] == "missing"
    assert snap_missing["channels"]["cpu_temp_above_ambient_c"]["status"] == "missing"
    assert snap_missing["channels"]["cpu_temp_above_ambient_c"]["z"] is None
    assert snap_missing["composites"]["warmth"]["status"] == "missing"

    # Stuck ambient_temp_c (frozen) freezes derived channel even if cpu_temp_c varies
    snap_frozen = {}
    for k in range(5):
        snap_frozen = tracker.observe(
            _make_readings(ambient_temp_c=23.0, cpu_temp_c=50.0 + 0.3 * k),
            now=1_800_000_010.0 + k,
        )
    assert snap_frozen["channels"]["ambient_temp_c"]["status"] == "frozen"
    assert snap_frozen["channels"]["cpu_temp_above_ambient_c"]["status"] == "frozen"
    assert snap_frozen["channels"]["cpu_temp_above_ambient_c"]["z"] is None
    assert snap_frozen["composites"]["warmth"]["status"] == "frozen"

    # When parent recovers, derived channel recovers cleanly immediately
    snap_recover = tracker.observe(
        _make_readings(ambient_temp_c=23.4, cpu_temp_c=52.0),
        now=1_800_000_020.0,
    )
    assert snap_recover["channels"]["ambient_temp_c"]["status"] == "warm"
    assert snap_recover["channels"]["cpu_temp_above_ambient_c"]["status"] == "warm"
    assert snap_recover["composites"]["warmth"]["status"] == "warm"


def test_external_light_uses_gated_log1p_only_never_raw_lux(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=3,
        window_samples=50,
        auto_load=False,
    )
    # Raw light_lux varies wildly while gated external_light_lux is None (unattributable)
    for k in range(5):
        snap = tracker.observe(
            _make_readings(light_lux=100.0 * (k + 1)),
            external_light_lux=None,
            now=500.0 + k,
        )
        assert snap["channels"]["external_light_lux"]["status"] == "missing"
        assert snap["channels"]["external_light_lux"]["samples"] == 0
        assert snap["channels"]["external_light_lux"]["valid_fraction"] == 0.0

    # Feed finite gated external light (negative clamped to 0, transformed via log1p)
    snap1 = tracker.observe(_make_readings(light_lux=9999.0), external_light_lux=0.0, now=510.0)
    assert snap1["channels"]["external_light_lux"]["conditional"] == "daytime_attributable_only"
    tracker.observe(_make_readings(), external_light_lux=math.expm1(2.0), now=511.0)
    tracker.observe(_make_readings(), external_light_lux=math.expm1(4.0), now=512.0)

    band = tracker.bands["external_light_lux"]
    assert band.mean == pytest.approx(2.0)
    snap_warm = tracker.observe(_make_readings(), external_light_lux=math.expm1(6.0), now=513.0)
    assert snap_warm["channels"]["external_light_lux"]["status"] == "warm"
    assert snap_warm["composites"]["clarity_light"]["conditional"] == "daytime_attributable_only"
    assert snap_warm["composites"]["clarity_light"]["z"] > 0.0
    assert snap_warm["composites"]["clarity_light"]["valid_fraction"] == pytest.approx(4 / 9, abs=1e-4)


def test_disk_percent_excluded_from_stability_and_composites(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=5,
        window_samples=50,
        auto_load=False,
    )
    _feed_varying_cycle(tracker, 10)
    # Huge spike on disk_percent must NOT affect stability composite at all
    snap_a = tracker.observe(_make_readings(disk_percent=42.0), now=1_800_000_000.0)
    # Use a fresh identical tracker and inject disk_percent=99.9
    tracker_b = SensorResidualTracker(
        tmp_path / "sr_b.json",
        min_samples=5,
        window_samples=50,
        auto_load=False,
    )
    _feed_varying_cycle(tracker_b, 10)
    snap_b = tracker_b.observe(_make_readings(disk_percent=99.9), now=1_800_000_000.0)

    assert snap_a["channels"]["disk_percent"]["non_stationary"] is True
    assert snap_b["channels"]["disk_percent"]["z"] > 10.0
    assert snap_a["composites"]["stability"] == snap_b["composites"]["stability"]
    assert "disk_percent" in snap_b["composites"]["stability"]["excluded_channels"]
    assert "disk_percent" not in snap_b["composites"]["stability"]["contributors"]


def test_stability_composite_requires_min_contributors(tmp_path: Path):
    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=5,
        window_samples=50,
        auto_load=False,
    )
    _feed_varying_cycle(tracker, 10)

    # Drop 3 of the 5 stability channels -> only 2 contributors (< 3) -> insufficient
    snap = tracker.observe(
        _make_readings(
            humidity_pct=None,
            pressure_hpa=None,
            eeg_delta_power=None,
            memory_percent=36.0,
            eeg_theta_power=0.18,
        ),
        now=1_800_000_000.0,
    )
    stab = snap["composites"]["stability"]
    assert stab["status"] == "insufficient"
    assert stab["contributor_count"] == 2
    assert stab["mean_abs_z"] is None
    assert stab["overlap_note"]


# ---------------------------------------------------------------------------
# 10. Online Welford Pearson r and histogram
# ---------------------------------------------------------------------------


def test_online_correlation_and_histogram(tmp_path: Path):
    corr = OnlineCorrelation()
    for i in range(10):
        corr.update(float(i), 2.0 * i + 1.0)
    assert corr.pearson_r == pytest.approx(1.0)

    corr_neg = OnlineCorrelation()
    for i in range(10):
        corr_neg.update(float(i), -3.0 * i + 5.0)
    assert corr_neg.pearson_r == pytest.approx(-1.0)

    tracker = SensorResidualTracker(
        tmp_path / "sr.json",
        min_samples=5,
        window_samples=50,
        auto_load=False,
    )
    _feed_varying_cycle(tracker, 30)
    assert tracker.save(now=1_800_000_000.0)
    detail = load_persisted_sensor_residuals(tmp_path / "sr.json", now=1_800_000_005.0)
    assert detail["available"] is True
    assert detail["snapshot_age_s"] == pytest.approx(5.0)
    assert sum(detail["channels"]["ambient_temp_c"]["z_abs_histogram"].values()) == 25
    assert detail["correlations"]["warmth_signed_z_vs_current_warmth"]["n"] == 25
    assert detail["correlations"]["warmth_signed_z_vs_current_warmth"]["pearson_r"] is not None


# ---------------------------------------------------------------------------
# 11. Persistence round-trip, negative clock step, corrupt recovery, read-only
# ---------------------------------------------------------------------------


def test_persistence_roundtrip_clock_step_and_corrupt_quarantine(tmp_path: Path):
    p = tmp_path / "sensor_residuals.json"
    tracker = SensorResidualTracker(p, min_samples=5, window_samples=50, auto_load=True)
    _feed_varying_cycle(tracker, 12, t0=2_000.0)
    assert tracker.save(now=2_050.0)

    # Reload cleanly and preserve baseline + comparison stats
    reloaded = SensorResidualTracker(p, min_samples=5, window_samples=50, auto_load=True)
    assert reloaded.observed_ticks == 12
    assert reloaded.bands["ambient_temp_c"].mean == pytest.approx(
        tracker.bands["ambient_temp_c"].mean
    )
    assert reloaded.bands["ambient_temp_c"].z_abs_histogram == (
        tracker.bands["ambient_temp_c"].z_abs_histogram
    )

    # Negative clock step must not break maybe_save or produce negative ages
    assert reloaded.maybe_save(now=1_500.0) is False
    detail = load_persisted_sensor_residuals(p, now=1_000.0)
    assert detail["snapshot_age_s"] == 0.0
    assert detail["channels"]["ambient_temp_c"]["last_valid_age_s"] == 0.0

    # Bumping comparison_version keeps Welford baselines but resets comparison stats
    saved_obj = json.loads(p.read_text())
    assert saved_obj["comparison_version"] == COMPARISON_VERSION
    saved_obj["comparison_version"] = COMPARISON_VERSION + 1
    p.write_text(json.dumps(saved_obj))
    restarted_comp = SensorResidualTracker(p, min_samples=5, window_samples=50, auto_load=True)
    assert restarted_comp.bands["ambient_temp_c"].mean == pytest.approx(
        tracker.bands["ambient_temp_c"].mean
    )
    assert sum(restarted_comp.bands["ambient_temp_c"].z_abs_histogram.values()) == 0
    assert restarted_comp.correlations["warmth_signed_z_vs_current_warmth"].n == 0

    # Corrupt file is quarantined to .corrupt and cold-starts safely
    p.write_text("{corrupt json")
    recovered = SensorResidualTracker(p, min_samples=5, window_samples=50, auto_load=True)
    assert recovered.observed_ticks == 0
    assert tmp_path.joinpath("sensor_residuals.json.corrupt").exists()

    # Schema mismatch on read-only loader never writes or renames files
    p.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 99}))
    ro = load_persisted_sensor_residuals(p)
    assert ro["available"] is False
    assert ro["reason"] == "schema_mismatch"
    assert p.exists()  # read-only loader must not rename/quarantine

    missing_ro = load_persisted_sensor_residuals(tmp_path / "does_not_exist.json")
    assert missing_ro["available"] is False
    assert not (tmp_path / "does_not_exist.json").exists()


# ---------------------------------------------------------------------------
# 12. Zero-steering and writer isolation guards
# ---------------------------------------------------------------------------


def test_zero_steering_bit_identical_sense_self_and_eisv_projection(tmp_path: Path):
    readings = _make_readings()
    anima_before = sense_self(readings, external_light_lux=120.0)
    eisv_before = anima_to_body_eisv_projection(anima_before, readings)

    tracker = SensorResidualTracker(tmp_path / "sr.json", min_samples=3, auto_load=False)
    _feed_varying_cycle(tracker, 20)

    anima_after = sense_self(readings, external_light_lux=120.0)
    eisv_after = anima_to_body_eisv_projection(anima_after, readings)

    assert anima_after.to_dict() == anima_before.to_dict()
    assert anima_after.clarity_attribution == anima_before.clarity_attribution
    assert eisv_after.to_dict() == eisv_before.to_dict()


def test_import_isolation_only_broker_writes_sensor_residuals():
    src_root = Path(__file__).resolve().parents[1] / "src" / "anima_mcp"
    for rel in ("anima.py", "eisv_mapper.py", "unitares_bridge.py"):
        tree = ast.parse((src_root / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "sensor_residuals" in node.module:
                raise AssertionError(f"{rel} must not import sensor_residuals")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "sensor_residuals" in alias.name:
                        raise AssertionError(f"{rel} must not import sensor_residuals")

    # display_ops.py may import AUTHORITY / MOVES_NO_GATE / load_persisted_sensor_residuals
    # read-only, never SensorResidualTracker
    disp_text = (src_root / "handlers" / "display_ops.py").read_text()
    assert "SensorResidualTracker" not in disp_text


# ---------------------------------------------------------------------------
# 13-14. Broker wiring + diagnostics MCP readout
# ---------------------------------------------------------------------------


def test_broker_snapshot_and_shutdown_wire_sensor_residuals():
    from anima_mcp.stable_creature import StableCreature
    from anima_mcp.anima import Anima

    rt = StableCreature()
    readings = _make_readings()
    anima = Anima(warmth=0.6, clarity=0.7, stability=0.8, presence=0.9, readings=readings)
    body_proj = anima_to_body_eisv_projection(anima, readings)
    shm_client = MagicMock()
    sr_snap = {"schema_version": SCHEMA_VERSION, "authority": AUTHORITY, "moves_no_gate": MOVES_NO_GATE}

    with patch("anima_mcp.creature_runtime.read_shadow_governance", return_value=None):
        rt.build_and_write_shm_snapshot(
            shm_client,
            anima,
            readings,
            None,
            [],
            body_proj,
            SimpleNamespace(creature_id="id", name="Lumen", total_awakenings=1),
            SimpleNamespace(surprise=0.0, surprise_sources=[]),
            SimpleNamespace(_cumulative_surprise=0.0),
            SimpleNamespace(confidence=0.9),
            None,
            "led_hardware_controller",
            None,
            None,
            None,
            210.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {},
            1.0,
            sensor_residuals=sr_snap,
        )

    written = shm_client.write.call_args[0][0]
    assert written["sensor_residuals"] == sr_snap
    assert written["body_anima"] == anima.to_dict()
    assert written["body_eisv_projection"] == body_proj.to_dict()

    sr_writer = MagicMock()
    sr_writer.save.return_value = True
    loop = MagicMock(is_running=lambda: False)
    rt.shutdown_creature_resources(
        None, None, None, None, None, None, None, None, None,
        loop, MagicMock(), MagicMock(), shm_client,
        sensor_residuals=sr_writer,
    )
    sr_writer.save.assert_called_once()


@pytest.mark.asyncio
async def test_handle_diagnostics_sensor_residuals_states(tmp_path: Path):
    from anima_mcp.handlers.display_ops import handle_diagnostics
    from anima_mcp.tool_registry import TOOLS

    tool = next(t for t in TOOLS if t.name == "diagnostics")
    assert "sensor_residuals_detail" in tool.inputSchema["properties"]

    sensors = SimpleNamespace(is_pi=lambda: False, available_sensors=lambda: ["mock"])
    fresh_ts = datetime.now().isoformat()
    stale_ts = (datetime.now() - timedelta(seconds=120)).isoformat()

    # Fresh SHM snapshot surfaces available=True
    fresh_shm = SimpleNamespace(
        read=lambda: {
            "timestamp": fresh_ts,
            "sensor_residuals": {
                "schema_version": SCHEMA_VERSION,
                "authority": AUTHORITY,
                "moves_no_gate": MOVES_NO_GATE,
                "observed_ticks": 42,
                "channels": {},
                "composites": {},
            },
        }
    )
    with patch("anima_mcp.accessors._get_leds", return_value=None), \
         patch("anima_mcp.accessors._get_display", return_value=None), \
         patch("anima_mcp.accessors._get_display_update_task", return_value=None), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
         patch("anima_mcp.accessors._get_shm_client", return_value=fresh_shm):
        out = json.loads((await handle_diagnostics({}))[0].text)

    assert out["sensor_residuals"]["available"] is True
    assert out["sensor_residuals"]["authority"] == AUTHORITY
    assert out["sensor_residuals"]["moves_no_gate"] is True
    assert out["sensor_residuals"]["observed_ticks"] == 42
    assert "sensor_residuals_detail" not in out

    # Stale SHM fails toward unknown
    stale_shm = SimpleNamespace(read=lambda: {"timestamp": stale_ts, "sensor_residuals": {"observed_ticks": 99}})
    with patch("anima_mcp.accessors._get_leds", return_value=None), \
         patch("anima_mcp.accessors._get_display", return_value=None), \
         patch("anima_mcp.accessors._get_display_update_task", return_value=None), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
         patch("anima_mcp.accessors._get_shm_client", return_value=stale_shm):
        out_stale = json.loads((await handle_diagnostics({}))[0].text)

    assert out_stale["sensor_residuals"]["available"] is False
    assert out_stale["sensor_residuals"]["status"] == "unknown"
    assert out_stale["sensor_residuals"]["reason"] == "shm_stale"

    # Opt-in detail reads persisted JSON read-only, while "false" string does not enable it
    persisted = tmp_path / "sr_detail.json"
    tracker = SensorResidualTracker(persisted, min_samples=3, auto_load=False)
    _feed_varying_cycle(tracker, 5)
    tracker.save()
    with patch("anima_mcp.accessors._get_leds", return_value=None), \
         patch("anima_mcp.accessors._get_display", return_value=None), \
         patch("anima_mcp.accessors._get_display_update_task", return_value=None), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
         patch("anima_mcp.accessors._get_shm_client", return_value=fresh_shm), \
         patch("anima_mcp.sensor_residuals.DEFAULT_PERSISTENCE_PATH", persisted):
        out_false = json.loads((await handle_diagnostics({"sensor_residuals_detail": "false"}))[0].text)
        out_detail = json.loads((await handle_diagnostics({"sensor_residuals_detail": True}))[0].text)

    assert "sensor_residuals_detail" not in out_false
    assert out_detail["sensor_residuals_detail"]["available"] is True
    assert "ambient_temp_c" in out_detail["sensor_residuals_detail"]["channels"]

    # Stale snapshot observed_at inside fresh SHM envelope fails toward unknown
    stale_snap_shm = SimpleNamespace(
        read=lambda: {
            "timestamp": fresh_ts,
            "sensor_residuals": {
                "observed_at": stale_ts,
                "observed_ticks": 5,
            },
        }
    )
    with patch("anima_mcp.accessors._get_leds", return_value=None), \
         patch("anima_mcp.accessors._get_display", return_value=None), \
         patch("anima_mcp.accessors._get_display_update_task", return_value=None), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
         patch("anima_mcp.accessors._get_shm_client", return_value=stale_snap_shm):
        out_snap_stale = json.loads((await handle_diagnostics({}))[0].text)
    assert out_snap_stale["sensor_residuals"]["available"] is False
    assert out_snap_stale["sensor_residuals"]["status"] == "unknown"
    assert out_snap_stale["sensor_residuals"]["reason"] == "snapshot_stale"

    # Spoofed authority/moves_no_gate/available in SHM cannot override invariant fields
    spoof_shm = SimpleNamespace(
        read=lambda: {
            "timestamp": fresh_ts,
            "sensor_residuals": {
                "available": False,
                "authority": "authoritative_gate",
                "moves_no_gate": False,
                "observed_ticks": 7,
            },
        }
    )
    with patch("anima_mcp.accessors._get_leds", return_value=None), \
         patch("anima_mcp.accessors._get_display", return_value=None), \
         patch("anima_mcp.accessors._get_display_update_task", return_value=None), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
         patch("anima_mcp.accessors._get_shm_client", return_value=spoof_shm):
        out_spoof = json.loads((await handle_diagnostics({}))[0].text)
    assert out_spoof["sensor_residuals"]["available"] is True
    assert out_spoof["sensor_residuals"]["authority"] == AUTHORITY
    assert out_spoof["sensor_residuals"]["moves_no_gate"] is True


def test_non_finite_bool_and_interleaved_missing_and_boundary_flush(tmp_path: Path):
    # Non-finite and bool external_light_lux inputs map to missing and leave baseline intact
    tracker = SensorResidualTracker(tmp_path / "sr.json", min_samples=3, window_samples=8, auto_load=False)
    for bad in (None, float("nan"), float("inf"), float("-inf"), True):
        snap = tracker.observe(_make_readings(light_lux=1000.0), external_light_lux=bad, now=100.0)
        assert snap["channels"]["external_light_lux"]["status"] == "missing"
        assert snap["channels"]["external_light_lux"]["z"] is None
        assert snap["channels"]["external_light_lux"]["samples"] == 0

    # Held repeats crossing window_samples match exact sequential folding
    from dataclasses import replace as dc_replace

    spec = CHANNEL_SPEC_MAP["ambient_temp_c"]
    seq = ResidualBand(
        spec=dc_replace(spec, physical_frozen_guard=False),
        min_samples=3,
        window_samples=6,
    )
    held = ResidualBand(spec=spec, min_samples=3, window_samples=6, frozen_run_threshold=60)
    values = [20.0, 24.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 25.0]
    for idx, v in enumerate(values):
        res_seq = seq.observe(v, now=float(idx))
        res_held = held.observe(v, now=float(idx))
    assert held.n == seq.n
    assert held.mean == pytest.approx(seq.mean)
    assert held.var == pytest.approx(seq.var)
    assert res_held["z"] == pytest.approx(res_seq["z"])

    # Interleaved None breaks a repeated run cleanly before frozen threshold
    interleaved = ResidualBand(spec=spec, min_samples=2, window_samples=20, frozen_run_threshold=5)
    for i in range(4):
        assert interleaved.observe(21.0, now=float(i))["status"] != "frozen"
    assert interleaved.observe(None, now=5.0)["status"] == "missing"
    for i in range(4):
        assert interleaved.observe(21.0, now=6.0 + i)["status"] != "frozen"

    # Boolean / NaN corrupted fields in JSON reject cleanly without crashing
    assert ResidualBand.from_dict(spec, {"n": True, "mean": 1.0, "var": 0.1}, min_samples=3, window_samples=10, frozen_run_threshold=60).n == 0
    assert OnlineCorrelation.from_dict({"n": True, "mean_x": 0.0, "mean_y": 0.0, "m2_x": 1.0, "m2_y": 1.0, "c_xy": 0.0}).n == 0

    # Read-only loader leaves a corrupt JSON file untouched (no quarantine rename)
    bad_file = tmp_path / "corrupt_only.json"
    bad_file.write_text("{broken")
    ro = load_persisted_sensor_residuals(bad_file)
    assert ro["available"] is False
    assert bad_file.exists()
    assert not tmp_path.joinpath("corrupt_only.json.corrupt").exists()

    # Save failure in unwritable target returns False without disrupting observe()
    unwritable = tmp_path / "as_file"
    unwritable.write_text("blocker")
    isolated = SensorResidualTracker(unwritable / "child.json", min_samples=2, auto_load=False)
    snap_ok = isolated.observe(_make_readings(), now=500.0)
    assert snap_ok["observed_ticks"] == 1
    assert isolated.save(now=500.0) is False

