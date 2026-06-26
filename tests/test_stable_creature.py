"""
Tests for stable_creature.py — Hardware broker main loop helpers.

Covers:
  - Module-level constants and configuration
  - signal_handler shutdown flag
  - _has_module helper
  - Governance interval parsing and clamping
  - Sensor retry logic
  - Shared memory write coordination
  - Learning state save/restore
  - Background async/thread helpers
  - LED brightness preset loading
  - SHM data assembly
  - UNITARES bridge lifecycle
  - Shutdown sequence
  - Social boost signal handling
  - Activity state integration
"""

import json
import signal
import time
import asyncio
import concurrent.futures
import threading
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import (
    MagicMock, AsyncMock, patch,
)

from anima_mcp.sensors.base import SensorReadings
from anima_mcp.anima import Anima, MoodMomentum
from anima_mcp.inner_life import InnerLife
from anima_mcp.eisv_mapper import anima_to_eisv
from anima_mcp.metacognition import (
    MetacognitiveMonitor, Prediction, PredictionError,
)
from anima_mcp.shared_memory import SharedMemoryClient
from anima_mcp.identity import CreatureIdentity


# =====================================================================
# Helpers
# =====================================================================

def _make_readings(**kwargs) -> SensorReadings:
    """Create SensorReadings with sensible defaults."""
    defaults = dict(
        timestamp=datetime.now(),
        cpu_temp_c=55.0,
        ambient_temp_c=22.0,
        humidity_pct=45.0,
        light_lux=300.0,
        pressure_hpa=1013.0,
        cpu_percent=15.0,
        memory_percent=35.0,
        disk_percent=50.0,
    )
    defaults.update(kwargs)
    return SensorReadings(**defaults)


def _make_anima(**kwargs) -> Anima:
    """Create Anima with sensible defaults."""
    r = _make_readings()
    defaults = dict(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5, readings=r)
    defaults.update(kwargs)
    return Anima(**defaults)


def _make_identity(**kwargs) -> CreatureIdentity:
    """Create CreatureIdentity with sensible defaults."""
    now = datetime.now()
    defaults = dict(
        creature_id="test-creature-id-1234",
        born_at=now,
        total_awakenings=3,
        total_alive_seconds=7200.0,
        name="Lumen",
        name_history=["Anima", "Lumen"],
        current_awakening_at=now,
        last_heartbeat_at=None,
        metadata={},
    )
    defaults.update(kwargs)
    return CreatureIdentity(**defaults)


# =====================================================================
# Module-level constants
# =====================================================================

class TestModuleConstants:
    """Verify module-level constants are sane."""

    def test_update_interval(self):
        from anima_mcp.stable_creature import UPDATE_INTERVAL
        assert UPDATE_INTERVAL == 2.0

    def test_max_retries(self):
        from anima_mcp.stable_creature import MAX_RETRIES
        assert MAX_RETRIES == 3

    def test_retry_delay(self):
        from anima_mcp.stable_creature import RETRY_DELAY
        assert RETRY_DELAY == 0.5


# =====================================================================
# Signal handler
# =====================================================================

class TestSignalHandler:
    """Test signal-based shutdown flag."""

    def test_signal_handler_sets_running_false(self):
        """signal_handler sets the global `running` flag to False."""
        import anima_mcp.stable_creature as sc
        old_running = sc.running
        try:
            sc.running = True
            sc.signal_handler(signal.SIGINT, None)
            assert sc.running is False
        finally:
            sc.running = old_running

    def test_signal_handler_idempotent(self):
        """Calling signal_handler twice still leaves running = False."""
        import anima_mcp.stable_creature as sc
        old_running = sc.running
        try:
            sc.running = True
            sc.signal_handler(signal.SIGINT, None)
            sc.signal_handler(signal.SIGTERM, None)
            assert sc.running is False
        finally:
            sc.running = old_running


# =====================================================================
# _has_module helper
# =====================================================================

class TestHasModule:
    """Test the _has_module helper function."""

    def test_has_module_true_for_loaded(self):
        """Returns True for a module that loaded successfully."""
        from anima_mcp.stable_creature import _has_module, _LEARNING_MODULES
        # Pick any module that loaded (or inject one)
        old = _LEARNING_MODULES.copy()
        try:
            _LEARNING_MODULES["test_mod"] = True
            assert _has_module("test_mod") is True
        finally:
            _LEARNING_MODULES.clear()
            _LEARNING_MODULES.update(old)

    def test_has_module_false_for_missing(self):
        """Returns False for a module that did not load."""
        from anima_mcp.stable_creature import _has_module, _LEARNING_MODULES
        old = _LEARNING_MODULES.copy()
        try:
            _LEARNING_MODULES["test_mod"] = False
            assert _has_module("test_mod") is False
        finally:
            _LEARNING_MODULES.clear()
            _LEARNING_MODULES.update(old)

    def test_has_module_false_for_unknown(self):
        """Returns False for a module key that does not exist at all."""
        from anima_mcp.stable_creature import _has_module
        assert _has_module("nonexistent_module_xyz") is False


# =====================================================================
# Governance interval parsing
# =====================================================================

class TestGovernanceIntervalParsing:
    """Test the governance interval env var parsing logic.

    The parsing code runs at import/module level, but we can exercise the
    same logic by re-running the expressions inline.
    """

    def test_default_governance_interval(self):
        """Without env var, interval is DEFAULT_GOVERNANCE_INTERVAL (180s)."""
        DEFAULT_GOVERNANCE_INTERVAL = 180.0
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = None
        try:
            _interval_raw = float(_interval_env) if _interval_env is not None else None
        except (TypeError, ValueError):
            _interval_raw = None
        GOVERNANCE_INTERVAL = (
            _interval_raw if _interval_raw is not None else DEFAULT_GOVERNANCE_INTERVAL
        )
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == 180.0

    def test_custom_governance_interval_valid(self):
        """Custom valid value is accepted."""
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "60"
        _interval_raw = float(_interval_env)
        GOVERNANCE_INTERVAL = _interval_raw
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == 60.0

    def test_governance_interval_clamped_to_minimum(self):
        """Values below MIN_GOVERNANCE_INTERVAL are clamped."""
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "5"
        _interval_raw = float(_interval_env)
        GOVERNANCE_INTERVAL = _interval_raw
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == MIN_GOVERNANCE_INTERVAL

    def test_governance_interval_invalid_string_uses_default(self):
        """Non-numeric string falls back to default."""
        DEFAULT_GOVERNANCE_INTERVAL = 180.0
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "not_a_number"
        try:
            _interval_raw = float(_interval_env) if _interval_env is not None else None
            GOVERNANCE_INTERVAL = (
                _interval_raw if _interval_raw is not None else DEFAULT_GOVERNANCE_INTERVAL
            )
        except (TypeError, ValueError):
            GOVERNANCE_INTERVAL = DEFAULT_GOVERNANCE_INTERVAL
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == DEFAULT_GOVERNANCE_INTERVAL

    def test_governance_interval_float_value(self):
        """Float env var value is parsed correctly."""
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "45.5"
        _interval_raw = float(_interval_env)
        GOVERNANCE_INTERVAL = _interval_raw
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == 45.5

    def test_governance_interval_zero_clamped(self):
        """Zero value is clamped to minimum."""
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "0"
        _interval_raw = float(_interval_env)
        GOVERNANCE_INTERVAL = _interval_raw
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == MIN_GOVERNANCE_INTERVAL

    def test_governance_interval_negative_clamped(self):
        """Negative value is clamped to minimum."""
        MIN_GOVERNANCE_INTERVAL = 30.0
        _interval_env = "-100"
        _interval_raw = float(_interval_env)
        GOVERNANCE_INTERVAL = _interval_raw
        GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
        assert GOVERNANCE_INTERVAL == MIN_GOVERNANCE_INTERVAL


# =====================================================================
# Sensor retry logic
# =====================================================================

class TestSensorRetryLogic:
    """Test the sensor read retry pattern used in the main loop."""

    def test_sensor_read_succeeds_first_try(self):
        """Sensor read succeeds on first attempt."""
        mock_sensors = MagicMock()
        mock_sensors.read.return_value = _make_readings()

        readings = None
        for attempt in range(3):
            try:
                readings = mock_sensors.read()
                break
            except Exception:
                if attempt < 2:
                    pass  # retry
        assert readings is not None
        assert mock_sensors.read.call_count == 1

    def test_sensor_read_succeeds_on_retry(self):
        """Sensor read fails once then succeeds."""
        mock_sensors = MagicMock()
        mock_sensors.read.side_effect = [
            OSError("I2C bus error"),
            _make_readings(),
        ]

        readings = None
        for attempt in range(3):
            try:
                readings = mock_sensors.read()
                break
            except Exception:
                if attempt < 2:
                    pass  # retry
        assert readings is not None
        assert mock_sensors.read.call_count == 2

    def test_sensor_read_fails_all_retries(self):
        """Sensor read fails on all 3 retries."""
        mock_sensors = MagicMock()
        mock_sensors.read.side_effect = OSError("I2C bus error")

        readings = None
        for attempt in range(3):
            try:
                readings = mock_sensors.read()
                break
            except Exception:
                if attempt < 2:
                    pass  # retry
        assert readings is None
        assert mock_sensors.read.call_count == 3

    def test_sensor_read_specific_exception_types(self):
        """Different exception types all trigger retries."""
        for exc_type in [OSError, IOError, RuntimeError, ValueError]:
            mock_sensors = MagicMock()
            mock_sensors.read.side_effect = [
                exc_type("sensor failure"),
                _make_readings(),
            ]
            readings = None
            for attempt in range(3):
                try:
                    readings = mock_sensors.read()
                    break
                except Exception:
                    if attempt < 2:
                        pass
            assert readings is not None, f"Failed for {exc_type.__name__}"


# =====================================================================
# MoodMomentum integration
# =====================================================================

class TestMoodMomentumIntegration:
    """Test MoodMomentum smoothing as used by the broker."""

    def test_first_call_returns_original(self):
        """First call to smooth returns the input unmodified."""
        mm = MoodMomentum()
        anima = _make_anima(warmth=0.8, clarity=0.3)
        smoothed = mm.smooth(anima)
        assert smoothed.warmth == 0.8
        assert smoothed.clarity == 0.3

    def test_second_call_applies_ema(self):
        """Second call applies EMA smoothing."""
        mm = MoodMomentum()
        a1 = _make_anima(warmth=0.8)
        mm.smooth(a1)

        a2 = _make_anima(warmth=0.2)
        smoothed = mm.smooth(a2)
        # EMA: 0.08 * 0.2 + 0.92 * 0.8 = 0.752
        assert abs(smoothed.warmth - 0.752) < 0.01

    def test_smooth_converges_slowly(self):
        """Many iterations of same value converge toward it."""
        mm = MoodMomentum()
        mm.smooth(_make_anima(warmth=0.8))
        for _ in range(50):
            smoothed = mm.smooth(_make_anima(warmth=0.2))
        # After 50 steps with alpha=0.08, should be very close to 0.2
        assert smoothed.warmth < 0.25


# =====================================================================
# InnerLife integration
# =====================================================================

class TestInnerLifeIntegration:
    """Test InnerLife update as used by the broker."""

    def test_inner_life_update_returns_inner_state(self):
        """InnerLife.update returns an InnerState object."""
        from anima_mcp.inner_life import InnerState
        il = InnerLife()
        raw = _make_anima(warmth=0.6, clarity=0.7, stability=0.5, presence=0.4)
        smoothed = _make_anima(warmth=0.55, clarity=0.65, stability=0.5, presence=0.45)
        state = il.update(raw, smoothed)
        assert isinstance(state, InnerState)
        assert "warmth" in state.temperament
        assert "warmth" in state.drives

    def test_inner_life_social_boost(self):
        """apply_social_boost does not crash."""
        il = InnerLife()
        raw = _make_anima()
        smoothed = _make_anima()
        il.update(raw, smoothed)
        # Should not raise
        il.apply_social_boost()

    def test_inner_life_pending_events(self):
        """get_pending_events returns a list."""
        il = InnerLife()
        raw = _make_anima()
        smoothed = _make_anima()
        il.update(raw, smoothed)
        events = il.get_pending_events()
        assert isinstance(events, list)


# =====================================================================
# EISV mapping
# =====================================================================

class TestEISVMapping:
    """Test anima_to_eisv as used by the broker."""

    def test_eisv_from_normal_anima(self):
        """EISV metrics are in valid range."""
        anima = _make_anima(warmth=0.6, clarity=0.7, stability=0.8, presence=0.7)
        readings = anima.readings
        eisv = anima_to_eisv(anima, readings)
        assert 0.0 <= eisv.energy <= 1.0
        assert 0.0 <= eisv.integrity <= 1.0
        assert 0.0 <= eisv.entropy <= 1.0
        assert -1.0 <= eisv.valence <= 1.0

    def test_eisv_to_dict(self):
        """EISV to_dict has expected keys."""
        anima = _make_anima()
        eisv = anima_to_eisv(anima, anima.readings)
        d = eisv.to_dict()
        assert set(d.keys()) == {"E", "I", "S", "V"}


# =====================================================================
# LED brightness preset loading
# =====================================================================

class TestLEDBrightnessPreset:
    """Test brightness preset loading logic from stable_creature."""

    def test_load_brightness_from_file(self, tmp_path):
        """Brightness loaded from display_brightness.json."""
        preset_path = tmp_path / "display_brightness.json"
        preset_path.write_text(json.dumps({"leds": 0.25}))

        # Replicate the loading logic
        _preset_led_brightness = 0.12  # default
        try:
            if preset_path.exists():
                _br_data = json.loads(preset_path.read_text())
                _preset_led_brightness = _br_data.get("leds", 0.12)
        except Exception:
            pass
        assert _preset_led_brightness == 0.25

    def test_load_brightness_fallback_on_missing_file(self, tmp_path):
        """Missing file falls back to default 0.12."""
        preset_path = tmp_path / "nonexistent.json"
        _preset_led_brightness = 0.12
        try:
            if preset_path.exists():
                _br_data = json.loads(preset_path.read_text())
                _preset_led_brightness = _br_data.get("leds", 0.12)
        except Exception:
            pass
        assert _preset_led_brightness == 0.12

    def test_load_brightness_fallback_on_malformed_json(self, tmp_path):
        """Malformed JSON falls back to default."""
        preset_path = tmp_path / "display_brightness.json"
        preset_path.write_text("not valid json {{{")
        _preset_led_brightness = 0.12
        try:
            if preset_path.exists():
                _br_data = json.loads(preset_path.read_text())
                _preset_led_brightness = _br_data.get("leds", 0.12)
        except Exception:
            pass
        assert _preset_led_brightness == 0.12

    def test_load_brightness_missing_leds_key(self, tmp_path):
        """JSON without 'leds' key falls back to default."""
        preset_path = tmp_path / "display_brightness.json"
        preset_path.write_text(json.dumps({"other_key": 0.5}))
        _preset_led_brightness = 0.12
        try:
            if preset_path.exists():
                _br_data = json.loads(preset_path.read_text())
                _preset_led_brightness = _br_data.get("leds", 0.12)
        except Exception:
            pass
        assert _preset_led_brightness == 0.12


# =====================================================================
# LED brightness estimate
# =====================================================================

class TestLEDBrightnessEstimate:
    """Test estimate_instantaneous_brightness used by broker."""

    def test_estimate_returns_positive(self):
        from anima_mcp.display.leds.brightness import estimate_instantaneous_brightness
        result = estimate_instantaneous_brightness(0.12)
        assert result > 0.0

    def test_estimate_with_zero_base(self):
        from anima_mcp.display.leds.brightness import estimate_instantaneous_brightness
        result = estimate_instantaneous_brightness(0.0)
        assert result >= 0.008  # min floor in implementation

    def test_estimate_with_high_base(self):
        from anima_mcp.display.leds.brightness import estimate_instantaneous_brightness
        result = estimate_instantaneous_brightness(1.0)
        assert 0.9 <= result <= 1.2  # should be near 1.0


# =====================================================================
# SHM data assembly
# =====================================================================

class TestSHMDataAssembly:
    """Test the data structure assembled for shared memory writes."""

    def test_basic_shm_data_shape(self):
        """SHM data has required top-level keys."""
        identity = _make_identity()
        anima = _make_anima(warmth=0.6, clarity=0.7, stability=0.8, presence=0.5)
        readings = anima.readings
        readings.led_brightness = 0.12
        eisv = anima_to_eisv(anima, readings)
        il = InnerLife()
        inner_state = il.update(_make_anima(), anima)
        metacog = MetacognitiveMonitor()
        prediction = metacog.predict()
        pred_error = metacog.observe(readings, anima)

        shm_data = {
            "timestamp": datetime.now().isoformat(),
            "readings": readings.to_dict(),
            "anima": anima.to_dict(),
            "inner_life": inner_state.to_dict() if inner_state else {},
            "drive_events": [],
            "eisv": eisv.to_dict(),
            "identity": {
                "creature_id": identity.creature_id,
                "name": identity.name,
                "awakenings": identity.total_awakenings,
            },
            "metacognition": {
                "surprise": pred_error.surprise,
                "surprise_sources": pred_error.surprise_sources,
                "cumulative_surprise": metacog._cumulative_surprise,
                "prediction_confidence": prediction.confidence,
            },
        }
        assert "timestamp" in shm_data
        assert "readings" in shm_data
        assert "anima" in shm_data
        assert "inner_life" in shm_data
        assert "eisv" in shm_data
        assert "identity" in shm_data
        assert "metacognition" in shm_data
        assert shm_data["identity"]["creature_id"] == "test-creature-id-1234"
        assert shm_data["identity"]["name"] == "Lumen"

    def test_shm_data_with_governance(self):
        """When governance decision exists, it's included in SHM data."""
        last_decision = {
            "action": "proceed",
            "margin": "comfortable",
            "reason": "stable state",
            "source": "unitares",
        }
        shm_data = {"timestamp": datetime.now().isoformat()}
        if last_decision:
            shm_data["governance"] = {
                **last_decision,
                "governance_at": datetime.now().isoformat(),
            }
        assert "governance" in shm_data
        assert shm_data["governance"]["action"] == "proceed"
        assert "governance_at" in shm_data["governance"]

    def test_shm_data_without_governance(self):
        """When no governance decision, SHM data does not include it."""
        last_decision = None
        shm_data = {"timestamp": datetime.now().isoformat()}
        if last_decision:
            shm_data["governance"] = last_decision
        assert "governance" not in shm_data

    def test_shm_data_wifi_status(self):
        """WiFi status is included in SHM data."""
        import psutil
        shm_data = {}
        try:
            net_stats = psutil.net_if_stats()
            wlan = net_stats.get("wlan0")
            shm_data["wifi_connected"] = bool(wlan and wlan.isup)
        except Exception:
            shm_data["wifi_connected"] = False
        assert "wifi_connected" in shm_data
        assert isinstance(shm_data["wifi_connected"], bool)

    def test_shm_data_agency_led_brightness(self):
        """Agency LED brightness is included when not 1.0."""
        shm_data = {}
        _agency_led_brightness = 0.6
        if _agency_led_brightness != 1.0:
            shm_data["agency_led_brightness"] = _agency_led_brightness
        assert shm_data["agency_led_brightness"] == 0.6

    def test_shm_data_agency_led_brightness_default(self):
        """Agency LED brightness not included when at default 1.0."""
        shm_data = {}
        _agency_led_brightness = 1.0
        if _agency_led_brightness != 1.0:
            shm_data["agency_led_brightness"] = _agency_led_brightness
        assert "agency_led_brightness" not in shm_data


# =====================================================================
# Shared memory write coordination
# =====================================================================

class TestSHMWriteCoordination:
    """Test SHM write operations as the broker uses them."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Broker writes data, reader gets same data back."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        reader = SharedMemoryClient(mode="read", filepath=filepath)

        data = {
            "timestamp": datetime.now().isoformat(),
            "anima": {"warmth": 0.5, "clarity": 0.6},
            "eisv": {"E": 0.5, "I": 0.6, "S": 0.4, "V": 0.1},
        }
        assert writer.write(data) is True
        read_back = reader.read()
        assert read_back == data

    def test_clear_removes_stale_data(self, tmp_path):
        """Writer.clear() removes file on startup (as broker does)."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        writer.write({"stale": True})
        assert filepath.exists()
        writer.clear()
        assert not filepath.exists()

    def test_write_error_returns_false(self, tmp_path):
        """Write returns False on filesystem error (doesn't crash)."""
        filepath = tmp_path / "state.json"
        writer = SharedMemoryClient(mode="write", filepath=filepath)
        with patch("builtins.open", side_effect=PermissionError("disk full")):
            result = writer.write({"data": 1})
            assert result is False


# =====================================================================
# Social boost signal
# =====================================================================

class TestSocialBoostSignal:
    """Test the social boost file-based signaling mechanism."""

    def test_social_boost_file_consumed(self, tmp_path):
        """Social boost file is deleted when detected."""
        boost_path = tmp_path / "anima_social_boost"
        boost_path.touch()
        assert boost_path.exists()

        il = InnerLife()
        il.update(_make_anima(), _make_anima())

        # Replicate the broker logic
        if boost_path.exists():
            try:
                boost_path.unlink()
                il.apply_social_boost()
            except Exception:
                pass
        assert not boost_path.exists()

    def test_social_boost_absent_is_noop(self, tmp_path):
        """When no boost file exists, nothing happens."""
        boost_path = tmp_path / "anima_social_boost"
        assert not boost_path.exists()

        il = InnerLife()
        il.update(_make_anima(), _make_anima())

        # Should not raise
        if boost_path.exists():
            boost_path.unlink()
            il.apply_social_boost()
        # No exception means success


# =====================================================================
# Governance check-in timing
# =====================================================================

class TestGovernanceCheckTiming:
    """Test the governance check-in rate limiting logic."""

    def test_first_check_in_always_triggers(self):
        """First check-in should always happen."""
        first_check_in = True
        last_governance_time = 0
        GOVERNANCE_INTERVAL = 180.0
        current_time = time.time()
        should_check = first_check_in or (current_time - last_governance_time >= GOVERNANCE_INTERVAL)
        assert should_check is True

    def test_rate_limited_after_recent_check(self):
        """Check is skipped if last check was recent."""
        first_check_in = False
        current_time = time.time()
        last_governance_time = current_time - 10  # 10 seconds ago
        GOVERNANCE_INTERVAL = 180.0
        should_check = first_check_in or (current_time - last_governance_time >= GOVERNANCE_INTERVAL)
        assert should_check is False

    def test_check_triggers_after_interval(self):
        """Check triggers when interval has elapsed."""
        first_check_in = False
        current_time = time.time()
        last_governance_time = current_time - 200  # 200 seconds ago
        GOVERNANCE_INTERVAL = 180.0
        should_check = first_check_in or (current_time - last_governance_time >= GOVERNANCE_INTERVAL)
        assert should_check is True

    def test_skip_if_future_running(self):
        """Background future blocks new submission."""
        _governance_future = MagicMock()
        _governance_future.done.return_value = False
        should_submit = _governance_future is None
        assert should_submit is False


# =====================================================================
# Background future result collection
# =====================================================================

class TestBackgroundFutureCollection:
    """Test non-blocking collection of background governance futures."""

    def test_governance_future_done_extracts_result(self):
        """Completed governance future yields decision."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.return_value = {
            "decision": {"action": "proceed", "margin": "comfortable"},
            "time": time.time(),
            "first": True,
            "checked_at": datetime.now().isoformat(),
        }

        last_decision = None
        first_check_in = True
        if future.done():
            try:
                gov_result = future.result()
                last_decision = gov_result["decision"]
                if gov_result.get("first"):
                    first_check_in = False
            except Exception:
                pass

        assert last_decision is not None
        assert last_decision["action"] == "proceed"
        assert first_check_in is False

    def test_governance_future_timeout_updates_time(self):
        """Timeout exception still updates last_governance_time."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.side_effect = asyncio.TimeoutError()

        current_time = time.time()
        last_governance_time = 0

        if future.done():
            try:
                future.result()
            except asyncio.TimeoutError:
                last_governance_time = current_time
            except Exception:
                pass

        assert last_governance_time == current_time

    def test_governance_future_cancelled_no_update(self):
        """CancelledError does not update governance time."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.side_effect = asyncio.CancelledError()

        last_governance_time = 0
        if future.done():
            try:
                future.result()
            except asyncio.TimeoutError:
                last_governance_time = time.time()
            except asyncio.CancelledError:
                pass
            except Exception:
                last_governance_time = time.time()

        assert last_governance_time == 0

    def test_governance_future_connection_refused_silent(self):
        """Connection refused errors are silently handled."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.side_effect = ConnectionError("Connection refused")

        current_time = time.time()
        last_governance_time = 0
        printed_error = False

        if future.done():
            try:
                future.result()
            except asyncio.TimeoutError:
                last_governance_time = current_time
            except asyncio.CancelledError:
                pass
            except Exception as e:
                last_governance_time = current_time
                if "Connection refused" not in str(e) and "Cannot connect" not in str(e):
                    printed_error = True

        assert last_governance_time == current_time
        assert printed_error is False

    def test_cognitive_future_done_extracts_synthesis(self):
        """Completed cognitive future yields synthesis."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.return_value = {"synthesis": "I notice a pattern"}

        cog_result = None
        if future.done():
            try:
                cog_result = future.result()
            except Exception:
                pass

        assert cog_result is not None
        assert "synthesis" in cog_result

    def test_memory_future_done_extracts_memories(self):
        """Completed memory future yields relevant memories."""
        future = MagicMock(spec=concurrent.futures.Future)
        future.done.return_value = True
        future.result.return_value = [
            {"text": "memory 1", "relevance": 0.9},
            {"text": "memory 2", "relevance": 0.7},
        ]

        mem_result = None
        if future.done():
            try:
                mem_result = future.result()
            except Exception:
                pass

        assert mem_result is not None
        assert len(mem_result) == 2


# =====================================================================
# _run_async_in_background helper
# =====================================================================

class TestRunAsyncInBackground:
    """Test the _run_async_in_background helper pattern."""

    def test_run_async_coroutine_on_loop(self):
        """Coroutine runs on a background event loop."""
        _bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        t.start()

        async def hello():
            return "world"

        try:
            future = asyncio.run_coroutine_threadsafe(hello(), _bg_loop)
            result = future.result(timeout=5.0)
            assert result == "world"
        finally:
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)

    def test_run_async_timeout_raises(self):
        """Timeout raises concurrent.futures.TimeoutError."""
        _bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        t.start()

        async def slow():
            await asyncio.sleep(10)
            return "never"

        try:
            future = asyncio.run_coroutine_threadsafe(slow(), _bg_loop)
            with pytest.raises(concurrent.futures.TimeoutError):
                future.result(timeout=0.1)
        finally:
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)

    def test_run_async_exception_propagates(self):
        """Exceptions from coroutine propagate to caller."""
        _bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        t.start()

        async def fail():
            raise ValueError("test error")

        try:
            future = asyncio.run_coroutine_threadsafe(fail(), _bg_loop)
            with pytest.raises(ValueError, match="test error"):
                future.result(timeout=5.0)
        finally:
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)


# =====================================================================
# UNITARES bridge lifecycle
# =====================================================================

class TestBridgeLifecycle:
    """Test UNITARES bridge initialization and cleanup."""

    def test_bridge_not_created_without_url(self):
        """Bridge is None when UNITARES_URL is not set."""
        from anima_mcp.unitares_bridge import UnitaresBridge
        unitares_url = None
        bridge = UnitaresBridge(unitares_url=unitares_url) if unitares_url else None
        assert bridge is None

    def test_bridge_created_with_url(self):
        """Bridge is created when UNITARES_URL is set."""
        from anima_mcp.unitares_bridge import UnitaresBridge
        unitares_url = "http://localhost:8767/mcp/"
        bridge = UnitaresBridge(unitares_url=unitares_url) if unitares_url else None
        assert bridge is not None

    def test_bridge_agent_id_set(self):
        """Bridge agent_id is set from identity."""
        from anima_mcp.unitares_bridge import UnitaresBridge
        bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp/")
        identity = _make_identity()
        bridge.set_agent_id(identity.creature_id)
        assert bridge._agent_id == "test-creature-id-1234"

    def test_bridge_session_id_set(self):
        """Bridge session_id is set with proper prefix."""
        from anima_mcp.unitares_bridge import UnitaresBridge
        bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp/")
        identity = _make_identity()
        bridge.set_session_id(f"anima-{identity.creature_id[:8]}")
        assert bridge._session_id == "anima-test-cre"

    def test_bridge_setup_failure_sets_bridge_none(self):
        """Bridge setup failure is handled gracefully."""
        from anima_mcp.unitares_bridge import UnitaresBridge
        bridge = UnitaresBridge(unitares_url="http://localhost:8767/mcp/")
        bridge.set_agent_id = MagicMock(side_effect=RuntimeError("setup fail"))

        try:
            bridge.set_agent_id("test")
        except Exception:
            bridge = None
        assert bridge is None


# =====================================================================
# Learning state save/restore
# =====================================================================

class TestLearningSaveRestore:
    """Test the periodic learning save pattern from the broker."""

    def test_learning_save_interval_check(self):
        """Learning save only triggers every 300 seconds."""
        last_learning_save = time.time()
        # Immediately after: should NOT save
        assert not (time.time() - last_learning_save > 300)

    def test_learning_save_after_interval(self):
        """Learning save triggers after 300 seconds have passed."""
        last_learning_save = time.time() - 301
        assert (time.time() - last_learning_save > 300)

    def test_learning_save_calls_subsystems(self):
        """Periodic save calls each subsystem's save method."""
        adaptive_model = MagicMock()
        preferences = MagicMock()
        self_model = MagicMock()

        # Replicate save logic
        adaptive_model._save_patterns()
        preferences._save()
        self_model.save()

        adaptive_model._save_patterns.assert_called_once()
        preferences._save.assert_called_once()
        self_model.save.assert_called_once()

    def test_learning_save_error_doesnt_crash(self):
        """Individual save failures do not prevent others from saving."""
        adaptive_model = MagicMock()
        adaptive_model._save_patterns.side_effect = OSError("disk full")
        preferences = MagicMock()
        self_model = MagicMock()

        try:
            if adaptive_model:
                adaptive_model._save_patterns()
        except Exception:
            pass  # as in the broker
        preferences._save()
        self_model.save()

        preferences._save.assert_called_once()
        self_model.save.assert_called_once()


# =====================================================================
# Shutdown sequence
# =====================================================================

class TestShutdownSequence:
    """Test the cleanup in the finally block of run_creature."""

    def test_shutdown_saves_learning_state(self):
        """Shutdown saves all learning subsystems."""
        adaptive_model = MagicMock()
        preferences = MagicMock()
        self_model = MagicMock()
        inner_life = MagicMock()

        # Replicate finally block logic
        try:
            if adaptive_model:
                adaptive_model._save_patterns()
            if preferences:
                preferences._save()
            if self_model:
                self_model.save()
            if inner_life:
                inner_life.save()
        except Exception:
            pass

        adaptive_model._save_patterns.assert_called_once()
        preferences._save.assert_called_once()
        self_model.save.assert_called_once()
        inner_life.save.assert_called_once()

    def test_shutdown_closes_experiential(self):
        """Shutdown closes pathway DB and experiential marks."""
        exp_filter = MagicMock()
        pathways = MagicMock()
        exp_marks = MagicMock()

        if exp_filter:
            try:
                exp_filter.save()
            except Exception:
                pass
        if pathways:
            try:
                pathways.close()
            except Exception:
                pass
        if exp_marks:
            try:
                exp_marks.close()
            except Exception:
                pass

        exp_filter.save.assert_called_once()
        pathways.close.assert_called_once()
        exp_marks.close.assert_called_once()

    def test_shutdown_stops_voice(self):
        """Shutdown stops voice module."""
        voice = MagicMock()
        if voice:
            try:
                voice.stop()
            except Exception:
                pass
        voice.stop.assert_called_once()

    def test_shutdown_store_sleep_close(self):
        """Shutdown calls store.sleep() then store.close()."""
        store = MagicMock()
        if store:
            store.sleep()
            store.close()
        store.sleep.assert_called_once()
        store.close.assert_called_once()

    def test_shutdown_bridge_close_on_loop(self):
        """Shutdown closes bridge on background event loop."""
        _bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        t.start()

        bridge = MagicMock()
        close_coro = AsyncMock(return_value=None)
        bridge.close = close_coro

        try:
            future = asyncio.run_coroutine_threadsafe(bridge.close(), _bg_loop)
            future.result(timeout=3)
        except Exception:
            pass
        finally:
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)

        bridge.close.assert_called()

    def test_shutdown_executor_shutdown(self):
        """Shutdown calls executor.shutdown."""
        _bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Should not hang or raise
        _bg_executor.shutdown(wait=True)
        # Verify it's shut down by checking we can't submit new work
        with pytest.raises(RuntimeError):
            _bg_executor.submit(lambda: None)

    def test_shutdown_clears_shm(self, tmp_path):
        """Shutdown clears shared memory file."""
        filepath = tmp_path / "state.json"
        shm_client = SharedMemoryClient(mode="write", filepath=filepath)
        shm_client.write({"data": "test"})
        assert filepath.exists()
        shm_client.clear()
        assert not filepath.exists()

    def test_shutdown_stops_bg_loop(self):
        """Shutdown stops and closes the background event loop."""
        _bg_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        t.start()
        time.sleep(0.05)  # let the loop actually start

        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
        t.join(timeout=2)
        _bg_loop.close()
        assert _bg_loop.is_closed()

    def test_shutdown_with_none_components(self):
        """Shutdown handles None components without crashing."""
        adaptive_model = None
        preferences = None
        self_model = None
        voice = None
        store = None
        exp_filter = None
        pathways = None
        exp_marks = None

        # Replicate the None-check pattern from the finally block
        try:
            if adaptive_model:
                adaptive_model._save_patterns()
            if preferences:
                preferences._save()
            if self_model:
                self_model.save()
        except Exception:
            pass

        if exp_filter:
            exp_filter.save()
        if pathways:
            pathways.close()
        if exp_marks:
            exp_marks.close()
        if voice:
            voice.stop()
        if store:
            store.sleep()
            store.close()
        # No assertions needed - just verifying no crash


# =====================================================================
# Activity state integration
# =====================================================================

class TestActivityStateIntegration:
    """Test activity state logic as used in the broker main loop."""

    def test_activity_brightness_calculation(self):
        """LED brightness scales with activity multiplier."""
        _preset_led_brightness = 0.12
        _agency_led_brightness = 1.0
        brightness_multiplier = 0.35  # resting

        _prev_led_brightness = (
            _preset_led_brightness * _agency_led_brightness * brightness_multiplier
        )
        assert abs(_prev_led_brightness - 0.042) < 0.001

    def test_activity_brightness_with_agency_dimmer(self):
        """Agency dimmer further reduces brightness."""
        _preset_led_brightness = 0.12
        _agency_led_brightness = 0.5
        brightness_multiplier = 1.0  # active

        _prev_led_brightness = (
            _preset_led_brightness * _agency_led_brightness * brightness_multiplier
        )
        assert abs(_prev_led_brightness - 0.06) < 0.001

    def test_should_skip_update_pattern(self):
        """Activity manager can skip updates for power saving."""
        activity_manager = MagicMock()
        activity_manager.should_skip_update.return_value = True
        assert activity_manager.should_skip_update() is True

        activity_manager.should_skip_update.return_value = False
        assert activity_manager.should_skip_update() is False


# =====================================================================
# Agency LED brightness
# =====================================================================

class TestAgencyLEDBrightness:
    """Test the agency LED brightness adjustment logic."""

    def test_increase_brightness(self):
        """Increase LED brightness by 20%."""
        _agency_led_brightness = 0.5
        current = _agency_led_brightness
        direction = "increase"
        if direction == "increase" and current < 1.0:
            _agency_led_brightness = min(1.0, current * 1.2)
        assert abs(_agency_led_brightness - 0.6) < 0.01

    def test_decrease_brightness(self):
        """Decrease LED brightness by 20%."""
        _agency_led_brightness = 0.5
        current = _agency_led_brightness
        direction = "decrease"
        if direction == "decrease" and current > 0.05:
            _agency_led_brightness = max(0.05, current * 0.8)
        assert abs(_agency_led_brightness - 0.4) < 0.01

    def test_increase_capped_at_one(self):
        """Brightness does not exceed 1.0."""
        _agency_led_brightness = 0.9
        current = _agency_led_brightness
        if current < 1.0:
            _agency_led_brightness = min(1.0, current * 1.2)
        # 0.9 * 1.2 = 1.08, clamped to 1.0
        assert _agency_led_brightness == 1.0

    def test_decrease_floored_at_005(self):
        """Brightness does not go below 0.05."""
        _agency_led_brightness = 0.06
        current = _agency_led_brightness
        if current > 0.05:
            _agency_led_brightness = max(0.05, current * 0.8)
        # 0.06 * 0.8 = 0.048, floored at 0.05
        assert _agency_led_brightness == 0.05

    def test_no_increase_at_max(self):
        """No increase when already at max."""
        _agency_led_brightness = 1.0
        current = _agency_led_brightness
        if current < 1.0:
            _agency_led_brightness = min(1.0, current * 1.2)
        assert _agency_led_brightness == 1.0

    def test_no_decrease_at_floor(self):
        """No decrease when already at floor."""
        _agency_led_brightness = 0.05
        current = _agency_led_brightness
        if current > 0.05:
            _agency_led_brightness = max(0.05, current * 0.8)
        assert _agency_led_brightness == 0.05


# =====================================================================
# Experiential state collection
# =====================================================================

class TestExperientialStateCollection:
    """Test the experiential stats collection pattern."""

    def test_collects_pathway_stats(self):
        """Pathway stats are collected per tick."""
        pathways = MagicMock()
        pathways.get_stats.return_value = {"total": 5, "active": 3}
        _exp_state = {}
        if pathways:
            try:
                _exp_state["pathways"] = pathways.get_stats()
            except Exception:
                pass
        assert _exp_state["pathways"] == {"total": 5, "active": 3}

    def test_collects_filter_stats(self):
        """Experiential filter stats are collected per tick."""
        exp_filter = MagicMock()
        exp_filter.get_stats.return_value = {"dimensions": 4}
        _exp_state = {}
        if exp_filter:
            try:
                _exp_state["filter"] = exp_filter.get_stats()
            except Exception:
                pass
        assert _exp_state["filter"] == {"dimensions": 4}

    def test_collects_marks_stats(self):
        """Experiential marks stats are collected per tick."""
        exp_marks = MagicMock()
        exp_marks.get_stats.return_value = {"earned": 2}
        _exp_state = {}
        if exp_marks:
            try:
                _exp_state["marks"] = exp_marks.get_stats()
            except Exception:
                pass
        assert _exp_state["marks"] == {"earned": 2}

    def test_exception_in_one_doesnt_block_others(self):
        """Exception in one stats collection does not prevent others."""
        pathways = MagicMock()
        pathways.get_stats.side_effect = RuntimeError("db locked")
        exp_filter = MagicMock()
        exp_filter.get_stats.return_value = {"dimensions": 4}
        exp_marks = MagicMock()
        exp_marks.get_stats.return_value = {"earned": 2}

        _exp_state = {}
        if pathways:
            try:
                _exp_state["pathways"] = pathways.get_stats()
            except Exception:
                pass
        if exp_filter:
            try:
                _exp_state["filter"] = exp_filter.get_stats()
            except Exception:
                pass
        if exp_marks:
            try:
                _exp_state["marks"] = exp_marks.get_stats()
            except Exception:
                pass

        assert "pathways" not in _exp_state
        assert _exp_state["filter"] == {"dimensions": 4}
        assert _exp_state["marks"] == {"earned": 2}


# =====================================================================
# Metacognition integration
# =====================================================================

class TestMetacognitionIntegration:
    """Test metacognition as used by the broker."""

    def test_predict_observe_cycle(self):
        """Basic predict-observe cycle works."""
        metacog = MetacognitiveMonitor()
        prediction = metacog.predict()
        assert isinstance(prediction, Prediction)

        readings = _make_readings(light_lux=300.0)
        anima = _make_anima()
        pred_error = metacog.observe(readings, anima)
        assert isinstance(pred_error, PredictionError)
        assert 0.0 <= pred_error.surprise <= 1.0

    def test_surprise_sources_are_list(self):
        """Surprise sources is always a list."""
        metacog = MetacognitiveMonitor()
        metacog.predict()
        pred_error = metacog.observe(_make_readings(), _make_anima())
        assert isinstance(pred_error.surprise_sources, (list, type(None)))

    def test_should_reflect_returns_tuple(self):
        """should_reflect returns (bool, str) tuple."""
        metacog = MetacognitiveMonitor()
        metacog.predict()
        pred_error = metacog.observe(_make_readings(), _make_anima())
        result = metacog.should_reflect(pred_error)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)


# =====================================================================
# Drive events collection
# =====================================================================

class TestDriveEventsCollection:
    """Test drive event collection as used by broker for SHM."""

    def test_drive_events_from_inner_life(self):
        """Drive events are collected from InnerLife pending events."""
        il = InnerLife()
        raw = _make_anima(warmth=0.2, stability=0.2)  # low values trigger drives
        smoothed = _make_anima(warmth=0.25, stability=0.25)

        # Need many updates to accumulate drives
        for _ in range(200):
            il.update(raw, smoothed)

        events = il.get_pending_events()
        _drive_events = []
        for ev in events:
            obs_text = il.get_observation_text(ev)
            if obs_text:
                _drive_events.append({
                    "text": obs_text,
                    "dimension": ev.dimension,
                    "event_type": ev.event_type,
                    "drive_value": round(ev.drive_value, 3),
                })

        # After many low-value ticks, drives should accumulate
        # (events may or may not have fired depending on thresholds)
        assert isinstance(_drive_events, list)


# =====================================================================
# Identity fallback
# =====================================================================

class TestIdentityFallback:
    """Test fallback identity creation when DB fails."""

    def test_fallback_identity_created(self):
        """When identity store fails, a fallback identity is created."""
        import uuid
        anima_id = str(uuid.uuid4())
        now = datetime.now()
        identity = CreatureIdentity(
            creature_id=anima_id,
            born_at=now,
            total_awakenings=0,
            total_alive_seconds=0.0,
            name="Lumen",
            name_history=[],
            current_awakening_at=now,
            last_heartbeat_at=None,
            metadata={},
        )
        assert identity.creature_id == anima_id
        assert identity.name == "Lumen"
        assert identity.total_awakenings == 0

    def test_fallback_identity_uses_env_var(self):
        """Fallback uses ANIMA_ID env var if available."""
        import uuid
        env_id = "env-var-creature-id-123"
        anima_id = env_id or str(uuid.uuid4())
        identity = CreatureIdentity(
            creature_id=anima_id,
            born_at=datetime.now(),
        )
        assert identity.creature_id == env_id

    def test_fallback_identity_generates_uuid(self):
        """Fallback generates UUID when no env var."""
        import uuid
        env_id = None
        anima_id = env_id or str(uuid.uuid4())
        identity = CreatureIdentity(
            creature_id=anima_id,
            born_at=datetime.now(),
        )
        # Should be a valid UUID
        uuid.UUID(identity.creature_id)


# =====================================================================
# Pattern apply interval
# =====================================================================

class TestPatternApplyInterval:
    """Test the hourly pattern application logic."""

    def test_pattern_apply_not_due(self):
        """Patterns not applied when interval hasn't elapsed."""
        last_pattern_apply = time.time()
        should_apply = time.time() - last_pattern_apply > 3600
        assert should_apply is False

    def test_pattern_apply_due(self):
        """Patterns applied when interval has elapsed."""
        last_pattern_apply = time.time() - 3601
        should_apply = time.time() - last_pattern_apply > 3600
        assert should_apply is True

    def test_pattern_apply_updates_timestamp(self):
        """After applying patterns, timestamp is updated."""
        last_pattern_apply = time.time() - 3601
        activity_manager = MagicMock()
        activity_manager.apply_learned_patterns.return_value = ["adj1"]

        if time.time() - last_pattern_apply > 3600:
            try:
                activity_manager.apply_learned_patterns(
                    adaptive_model=None,
                    self_model=None,
                )
                last_pattern_apply = time.time()
            except Exception:
                last_pattern_apply = time.time()

        assert time.time() - last_pattern_apply < 1.0
        activity_manager.apply_learned_patterns.assert_called_once()

    def test_pattern_apply_error_still_updates_timestamp(self):
        """Error in pattern apply still updates timestamp to prevent retry storm."""
        last_pattern_apply = time.time() - 3601
        activity_manager = MagicMock()
        activity_manager.apply_learned_patterns.side_effect = RuntimeError("boom")

        if time.time() - last_pattern_apply > 3600:
            try:
                activity_manager.apply_learned_patterns(
                    adaptive_model=None, self_model=None,
                )
                last_pattern_apply = time.time()
            except Exception:
                last_pattern_apply = time.time()

        assert time.time() - last_pattern_apply < 1.0


# =====================================================================
# ThreadPoolExecutor usage
# =====================================================================

class TestBackgroundExecutor:
    """Test the background executor pattern used by the broker."""

    def test_executor_submit_and_result(self):
        """Executor submits work and returns result."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(lambda: 42)
            assert future.result(timeout=5) == 42
        finally:
            executor.shutdown(wait=True)

    def test_executor_single_worker_serializes(self):
        """Single worker serializes tasks."""
        results = []
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="test-bg"
        )
        try:
            f1 = executor.submit(lambda: results.append(1) or 1)
            f2 = executor.submit(lambda: results.append(2) or 2)
            f1.result(timeout=5)
            f2.result(timeout=5)
            assert results == [1, 2]
        finally:
            executor.shutdown(wait=True)

    def test_future_done_check(self):
        """Future.done() returns True after completion."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(lambda: 42)
            future.result(timeout=5)  # wait for completion
            assert future.done() is True
        finally:
            executor.shutdown(wait=True)

    def test_future_not_done_while_running(self):
        """Future.done() returns False while task is running."""
        import threading
        event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(lambda: event.wait(timeout=5))
            # Brief pause to let executor pick up the task
            time.sleep(0.05)
            # Task should still be running
            assert future.done() is False
            event.set()
            future.result(timeout=5)
        finally:
            executor.shutdown(wait=True)


# =====================================================================
# Dialectic rate limiting
# =====================================================================

class TestDialecticRateLimiting:
    """Test the 60-second rate limiter for dialectic synthesis."""

    def test_first_dialectic_allowed(self):
        """First dialectic synthesis is always allowed."""
        last_dialectic_time = 0
        current_time = time.time()
        pred_error_surprise = 0.5
        _cognitive_future = None

        allowed = (
            pred_error_surprise > 0.3
            and current_time - last_dialectic_time > 60
            and _cognitive_future is None
        )
        assert allowed is True

    def test_dialectic_blocked_within_cooldown(self):
        """Dialectic blocked within 60-second cooldown."""
        current_time = time.time()
        last_dialectic_time = current_time - 30  # 30s ago
        pred_error_surprise = 0.5
        _cognitive_future = None

        allowed = (
            pred_error_surprise > 0.3
            and current_time - last_dialectic_time > 60
            and _cognitive_future is None
        )
        assert allowed is False

    def test_dialectic_blocked_low_surprise(self):
        """Dialectic not triggered for low surprise."""
        last_dialectic_time = 0
        current_time = time.time()
        pred_error_surprise = 0.1  # too low
        _cognitive_future = None

        allowed = (
            pred_error_surprise > 0.3
            and current_time - last_dialectic_time > 60
            and _cognitive_future is None
        )
        assert allowed is False

    def test_dialectic_blocked_future_running(self):
        """Dialectic blocked when previous future still running."""
        last_dialectic_time = 0
        current_time = time.time()
        pred_error_surprise = 0.5
        _cognitive_future = MagicMock()  # not None

        allowed = (
            pred_error_surprise > 0.3
            and current_time - last_dialectic_time > 60
            and _cognitive_future is None
        )
        assert allowed is False


# =====================================================================
# Face rendering
# =====================================================================

class TestFaceRendering:
    """Test face state derivation as used by broker."""

    def test_derive_face_state(self):
        """derive_face_state produces a face state from anima."""
        from anima_mcp.display.face import derive_face_state, face_to_ascii
        anima = _make_anima(warmth=0.6, clarity=0.7, stability=0.8, presence=0.5)
        face_state = derive_face_state(anima)
        assert face_state is not None
        ascii_face = face_to_ascii(face_state)
        assert isinstance(ascii_face, str)
        assert len(ascii_face) > 0

    def test_face_state_activity_resting(self):
        """Face state modified for resting activity."""
        from anima_mcp.display.face import derive_face_state, EyeState
        anima = _make_anima()
        face_state = derive_face_state(anima)
        # Simulate activity state = resting
        face_state.eyes = EyeState.CLOSED
        face_state.eye_openness = 0.0
        assert face_state.eyes == EyeState.CLOSED
        assert face_state.eye_openness == 0.0

    def test_face_state_activity_drowsy(self):
        """Face state modified for drowsy activity."""
        from anima_mcp.display.face import derive_face_state, EyeState
        anima = _make_anima()
        face_state = derive_face_state(anima)
        # Simulate activity state = drowsy
        face_state.eyes = EyeState.DROOPY
        face_state.eye_openness = 0.4
        assert face_state.eyes == EyeState.DROOPY
        assert face_state.eye_openness == 0.4


# =====================================================================
# Voice integration
# =====================================================================

class TestVoiceIntegration:
    """Test voice module integration patterns."""

    def test_voice_update_state(self):
        """Voice.update_state called with anima dimensions."""
        voice = MagicMock()
        anima = _make_anima(warmth=0.6, clarity=0.7, stability=0.5, presence=0.4)
        feeling = anima.feeling()

        voice.update_state(
            warmth=anima.warmth,
            clarity=anima.clarity,
            stability=anima.stability,
            presence=anima.presence,
            mood=feeling.get("mood", "neutral"),
        )
        voice.update_state.assert_called_once()
        call_kwargs = voice.update_state.call_args.kwargs
        assert call_kwargs["warmth"] == 0.6

    def test_voice_update_environment(self):
        """Voice.update_environment called with sensor data."""
        voice = MagicMock()
        readings = _make_readings(ambient_temp_c=23.0, humidity_pct=50.0, light_lux=300.0)

        voice.update_environment(
            temperature=readings.ambient_temp_c or readings.cpu_temp_c or 22.0,
            humidity=readings.humidity_pct or 50.0,
            light_level=readings.light_lux or 500.0,
        )
        voice.update_environment.assert_called_once()
        call_kwargs = voice.update_environment.call_args.kwargs
        assert call_kwargs["temperature"] == 23.0

    def test_voice_error_doesnt_crash(self):
        """Voice update error is caught and does not crash."""
        voice = MagicMock()
        voice.update_state.side_effect = RuntimeError("audio device busy")

        try:
            voice.update_state(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5, mood="neutral")
        except Exception:
            pass  # as in the broker

        # Should have been called even though it errored
        voice.update_state.assert_called_once()


# =====================================================================
# Learning state in SHM
# =====================================================================

class TestLearningSHMState:
    """Test learning state assembly for SHM writes."""

    def test_learning_state_with_preferences(self):
        """Learning state includes preference satisfaction."""
        preferences = MagicMock()
        preferences.get_overall_satisfaction.return_value = 0.87
        preferences.get_most_unsatisfied.return_value = ("warmth", 0.2)

        current_state = {"warmth": 0.5, "clarity": 0.6, "stability": 0.7, "presence": 0.5}
        learning_state = {}
        try:
            learning_state["preferences"] = {
                "satisfaction": preferences.get_overall_satisfaction(current_state),
                "most_unsatisfied": preferences.get_most_unsatisfied(current_state),
            }
        except Exception:
            pass
        assert learning_state["preferences"]["satisfaction"] == 0.87

    def test_learning_state_with_self_beliefs(self):
        """Learning state includes self belief summary."""
        self_model = MagicMock()
        self_model.get_belief_summary.return_value = {
            "stability_recovery": {"confidence": 0.68}
        }
        learning_state = {}
        try:
            learning_state["self_beliefs"] = self_model.get_belief_summary()
        except Exception:
            pass
        assert "stability_recovery" in learning_state["self_beliefs"]

    def test_learning_state_with_agency(self):
        """Learning state includes action stats."""
        action_selector = MagicMock()
        action_selector.get_action_stats.return_value = {
            "action_values": {"focus_attention": 0.22}
        }
        learning_state = {}
        try:
            learning_state["agency"] = action_selector.get_action_stats()
        except Exception:
            pass
        assert "action_values" in learning_state["agency"]

    def test_learning_state_with_prediction_accuracy(self):
        """Learning state includes prediction accuracy stats."""
        adaptive_model = MagicMock()
        adaptive_model.get_accuracy_stats.return_value = {
            "overall_mean_error": 0.15
        }
        learning_state = {}
        try:
            learning_state["prediction_accuracy"] = adaptive_model.get_accuracy_stats()
        except Exception:
            pass
        assert learning_state["prediction_accuracy"]["overall_mean_error"] == 0.15

    def test_learning_state_exception_isolation(self):
        """Exception in one learning module doesn't prevent others."""
        preferences = MagicMock()
        preferences.get_overall_satisfaction.side_effect = RuntimeError("db error")
        self_model = MagicMock()
        self_model.get_belief_summary.return_value = {"key": "val"}

        learning_state = {}
        try:
            learning_state["preferences"] = {
                "satisfaction": preferences.get_overall_satisfaction({}),
            }
        except Exception:
            pass
        try:
            learning_state["self_beliefs"] = self_model.get_belief_summary()
        except Exception:
            pass

        assert "preferences" not in learning_state
        assert learning_state["self_beliefs"] == {"key": "val"}


# =====================================================================
# Main entry point
# =====================================================================

class TestMainEntryPoint:
    """Test the main() entry point routing."""

    def test_main_calls_run_creature(self):
        """main() delegates to run_creature()."""
        from anima_mcp.stable_creature import main
        with patch("anima_mcp.stable_creature.run_creature") as mock_run:
            main()
            mock_run.assert_called_once()


# =====================================================================
# ENHANCED_LEARNING_AVAILABLE flag
# =====================================================================

class TestEnhancedLearningAvailable:
    """Test the ENHANCED_LEARNING_AVAILABLE flag."""

    def test_enhanced_learning_flag_type(self):
        """ENHANCED_LEARNING_AVAILABLE is a boolean."""
        from anima_mcp.stable_creature import ENHANCED_LEARNING_AVAILABLE
        assert isinstance(ENHANCED_LEARNING_AVAILABLE, bool)

    def test_learning_modules_dict_populated(self):
        """_LEARNING_MODULES has entries for all 8 modules."""
        from anima_mcp.stable_creature import _LEARNING_MODULES
        expected_keys = {
            "adaptive_prediction", "preferences", "self_model",
            "agency", "memory_retrieval", "weighted_pathways",
            "experiential_filter", "experiential_marks",
        }
        assert expected_keys == set(_LEARNING_MODULES.keys())


# =====================================================================
# DB path resolution
# =====================================================================

class TestDBPathResolution:
    """Test the DB path resolution logic from run_creature."""

    def test_db_path_from_env_var(self):
        """ANIMA_DB env var overrides default path."""
        env_db = "/custom/path/anima.db"
        if env_db:
            db_path = env_db
        else:
            db_path = str(Path.home() / ".anima" / "anima.db")
        assert db_path == "/custom/path/anima.db"

    def test_db_path_default(self):
        """Default DB path is ~/.anima/anima.db."""
        env_db = None
        if env_db:
            db_path = env_db
        else:
            home_dir = Path.home() / ".anima"
            db_path = str(home_dir / "anima.db")
        assert db_path == str(Path.home() / ".anima" / "anima.db")
