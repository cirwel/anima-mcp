"""
Tests for anima calculations - catch bugs like inverted neural math.

Run with: pytest tests/test_anima.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from anima_mcp.anima import (
    sense_self, _sense_warmth, _sense_clarity,
    _sense_stability, _sense_presence
)
from anima_mcp.sensors.base import SensorReadings
from anima_mcp.config import NervousSystemCalibration


@pytest.fixture
def default_calibration():
    return NervousSystemCalibration()


@pytest.fixture
def now():
    return datetime.now()


@pytest.fixture
def normal_readings(now):
    """Typical room conditions."""
    return SensorReadings(
        timestamp=now,
        cpu_temp_c=55.0,
        ambient_temp_c=25.0,
        humidity_pct=40.0,
        light_lux=300.0,
        pressure_hpa=1013.0,
        cpu_percent=10.0,
        memory_percent=30.0,
        disk_percent=50.0,
    )


@pytest.fixture
def extreme_readings(now):
    """Extreme conditions to test edge cases."""
    return SensorReadings(
        timestamp=now,
        cpu_temp_c=85.0,  # Hot CPU
        ambient_temp_c=35.0,  # Hot room
        humidity_pct=90.0,  # Very humid
        light_lux=10000.0,  # Bright sunlight
        pressure_hpa=950.0,  # Low pressure (storm)
        cpu_percent=95.0,  # High load
        memory_percent=90.0,  # High memory
        disk_percent=95.0,  # Almost full
    )


class TestAnimaRanges:
    """All anima values should be in [0, 1] range."""

    def test_normal_readings_in_range(self, normal_readings, default_calibration):
        anima = sense_self(normal_readings, default_calibration)

        assert 0 <= anima.warmth <= 1, f"warmth={anima.warmth} out of range"
        assert 0 <= anima.clarity <= 1, f"clarity={anima.clarity} out of range"
        assert 0 <= anima.stability <= 1, f"stability={anima.stability} out of range"
        assert 0 <= anima.presence <= 1, f"presence={anima.presence} out of range"

    def test_extreme_readings_in_range(self, extreme_readings, default_calibration):
        anima = sense_self(extreme_readings, default_calibration)

        assert 0 <= anima.warmth <= 1, f"warmth={anima.warmth} out of range"
        assert 0 <= anima.clarity <= 1, f"clarity={anima.clarity} out of range"
        assert 0 <= anima.stability <= 1, f"stability={anima.stability} out of range"
        assert 0 <= anima.presence <= 1, f"presence={anima.presence} out of range"

    def test_missing_sensors_in_range(self, now, default_calibration):
        """Even with missing data, values should be valid."""
        sparse_readings = SensorReadings(timestamp=now, cpu_temp_c=50.0)
        anima = sense_self(sparse_readings, default_calibration)

        assert 0 <= anima.warmth <= 1
        assert 0 <= anima.clarity <= 1
        assert 0 <= anima.stability <= 1
        assert 0 <= anima.presence <= 1

    def test_authoritative_unknown_does_not_read_process_local_prediction_model(
        self, normal_readings, default_calibration
    ):
        with patch(
            "anima_mcp.anima._get_prediction_accuracy",
            side_effect=AssertionError("stale process-local reader used"),
        ):
            anima = sense_self(
                normal_readings,
                default_calibration,
                prediction_accuracy=None,
            )

        assert 0 <= anima.clarity <= 1


class TestAnimaNotExtreme:
    """Values shouldn't be stuck at extremes under normal conditions."""

    def test_stability_not_always_high(self, now, default_calibration):
        """Bug check: stability was stuck at 98% due to inverted neural calc."""
        readings = SensorReadings(
            timestamp=now,
            cpu_temp_c=55.0,
            ambient_temp_c=25.0,
            humidity_pct=40.0,
            memory_percent=30.0,
            pressure_hpa=1013.0,
        )
        stability = _sense_stability(readings, default_calibration)

        # Should be reasonable, not pinned at top
        assert stability < 0.95, f"stability={stability} suspiciously high - check neural calc"
        assert stability > 0.3, f"stability={stability} suspiciously low"

    def test_presence_not_always_high(self, now, default_calibration):
        """Bug check: presence was stuck at 98% due to inverted neural calc."""
        readings = SensorReadings(
            timestamp=now,
            disk_percent=20.0,
            memory_percent=30.0,
            cpu_percent=10.0,
            light_lux=300.0,
        )
        presence = _sense_presence(readings, default_calibration)

        # Should be reasonable, not pinned at top
        assert presence < 0.95, f"presence={presence} suspiciously high - check neural calc"
        assert presence > 0.3, f"presence={presence} suspiciously low"

    def test_warmth_varies_with_temperature(self, now, default_calibration):
        """Warmth should respond to temperature changes."""
        cold = SensorReadings(timestamp=now, cpu_temp_c=40.0, ambient_temp_c=15.0)
        hot = SensorReadings(timestamp=now, cpu_temp_c=80.0, ambient_temp_c=35.0)

        warmth_cold = _sense_warmth(cold, default_calibration)
        warmth_hot = _sense_warmth(hot, default_calibration)

        assert warmth_hot > warmth_cold, "Hot should feel warmer than cold"
        assert warmth_hot - warmth_cold > 0.2, "Temperature should have meaningful impact"

    def test_clarity_varies_with_prediction_accuracy(self, now, default_calibration):
        """Clarity should respond to prediction accuracy (internal seeing).

        Note: Light was removed from clarity calculation because LEDs affect
        the light sensor, creating a feedback loop. Clarity now measures
        self-prediction accuracy - genuine internal awareness.
        """
        readings = SensorReadings(timestamp=now, light_lux=100.0)

        # Low prediction accuracy = low clarity (confused about own state)
        clarity_low = _sense_clarity(readings, default_calibration, prediction_accuracy=0.2)
        # High prediction accuracy = high clarity (understands own state)
        clarity_high = _sense_clarity(readings, default_calibration, prediction_accuracy=0.9)

        assert clarity_high > clarity_low, "Better prediction accuracy should mean clearer internal seeing"
        assert clarity_high - clarity_low > 0.2, "Prediction accuracy should have meaningful impact"


class TestAnimaMath:
    """Verify the math is correct (not inverted)."""

    def test_high_resource_usage_reduces_presence(self, now, default_calibration):
        """High disk/memory/cpu usage should reduce presence."""
        low_usage = SensorReadings(
            timestamp=now,
            disk_percent=10.0,
            memory_percent=10.0,
            cpu_percent=5.0,
            light_lux=300.0,
        )
        high_usage = SensorReadings(
            timestamp=now,
            disk_percent=90.0,
            memory_percent=90.0,
            cpu_percent=90.0,
            light_lux=300.0,
        )

        presence_low = _sense_presence(low_usage, default_calibration)
        presence_high = _sense_presence(high_usage, default_calibration)

        assert presence_low > presence_high, "High resource usage should reduce presence"

    def test_high_memory_reduces_stability(self, now, default_calibration):
        """High memory usage should reduce stability."""
        low_mem = SensorReadings(timestamp=now, memory_percent=10.0, humidity_pct=50.0, pressure_hpa=1013.0)
        high_mem = SensorReadings(timestamp=now, memory_percent=90.0, humidity_pct=50.0, pressure_hpa=1013.0)

        stability_low = _sense_stability(low_mem, default_calibration)
        stability_high = _sense_stability(high_mem, default_calibration)

        assert stability_low > stability_high, "High memory usage should reduce stability"


class TestNeuralContribution:
    """Verify neural simulation contributes correctly (not inverted)."""

    def test_neural_adds_to_instability_when_low(self, now, default_calibration):
        """Low neural groundedness should increase instability, not decrease it."""
        # This test would have caught the stability bug
        readings = SensorReadings(
            timestamp=now,
            humidity_pct=50.0,
            memory_percent=10.0,
            pressure_hpa=1013.0,
            light_lux=500.0,  # Moderate light = moderate neural
        )
        stability = _sense_stability(readings, default_calibration)

        # With corrected math, stability should be moderate, not 98%+
        assert 0.5 < stability < 0.95, f"stability={stability} - neural contribution may be wrong"

    def test_neural_adds_to_void_when_low(self, now, default_calibration):
        """High neural gamma (ctx switching) should increase void (scattered)."""
        # This test would have caught the presence bug
        readings = SensorReadings(
            timestamp=now,
            disk_percent=20.0,
            memory_percent=20.0,
            cpu_percent=5.0,
            light_lux=100.0,  # Dim light = low gamma
        )
        presence = _sense_presence(readings, default_calibration)

        # With corrected math, presence should be moderate, not 98%+
        assert 0.4 < presence < 0.95, f"presence={presence} - neural contribution may be wrong"


# ==================== per-channel sensor liveness ====================

class TestChannelLiveness:
    """A sensor that is present but frozen is not informing anything.

    The BMP280 has returned 682.5015433175248 byte-identical since Lumen woke
    from the July blackout, and nothing in the stack could see it: every other
    freshness gate reads the envelope timestamp, which stays current while an
    individual channel is dead for days. sensor_coverage sat at exactly 1.0 for
    30 days with sd 0.00000, and missing_sensors at 0 — so 35% of the weight
    across clarity (0.15) and stability (0.20) was a constant.
    """

    def setup_method(self):
        from anima_mcp.anima import _reset_channel_liveness
        _reset_channel_liveness()

    def _readings(self, now, i=0, **kw):
        """Non-target channels jitter, so only the channel under test can freeze."""
        base = dict(
            cpu_temp_c=55.0 + i * 0.01,
            ambient_temp_c=28.0 + i * 0.003,
            humidity_pct=38.0 + i * 0.01,
            light_lux=12.0 + i * 0.05,
            pressure_hpa=830.0 + i * 0.02,
        )
        base.update(kw)
        return SensorReadings(timestamp=now + timedelta(seconds=i * 2), **base)

    def _repeat(self, now, times, **kw):
        from anima_mcp.anima import _frozen_channel_count
        n = 0
        for i in range(times):
            n = _frozen_channel_count(self._readings(now, i=i, **kw))
        return n

    def test_frozen_channel_is_detected_after_threshold(self, now):
        from anima_mcp.anima import _FROZEN_REPEAT_THRESHOLD as T
        # T calls leave the repeat counter at T-1 — one short of the gate.
        assert self._repeat(now, T, pressure_hpa=682.5015433175248) == 0
        assert self._repeat(now, 1, pressure_hpa=682.5015433175248) == 1

    def test_a_calm_room_is_not_a_dead_sensor(self, now):
        """The false positive that would matter: low movement is still movement."""
        from anima_mcp.anima import _frozen_channel_count, _FROZEN_REPEAT_THRESHOLD as T
        worst = 0
        for i in range(T + 20):
            worst = max(worst, _frozen_channel_count(SensorReadings(
                timestamp=now + timedelta(seconds=i * 2),
                cpu_temp_c=55.0 + (i % 3) * 0.001,
                ambient_temp_c=28.0 + (i % 5) * 0.001,
                humidity_pct=38.0 + (i % 7) * 0.001,
                light_lux=12.0 + (i % 2) * 0.001,
                pressure_hpa=830.0 + (i % 4) * 0.001,
            )))
        assert worst == 0

    def test_recovery_when_the_channel_moves_again(self, now):
        from anima_mcp.anima import _frozen_channel_count, _FROZEN_REPEAT_THRESHOLD as T
        self._repeat(now, T + 1, pressure_hpa=682.5)
        assert _frozen_channel_count(self._readings(now, i=999, pressure_hpa=682.5)) == 1
        assert _frozen_channel_count(self._readings(now, i=1000, pressure_hpa=831.2)) == 0

    def test_absent_channel_is_not_also_counted_frozen(self, now):
        """missing and frozen must not double-count the same channel."""
        from anima_mcp.anima import _FROZEN_REPEAT_THRESHOLD as T
        assert self._repeat(now, T + 2, pressure_hpa=None) == 0

    def test_frozen_sensor_lowers_stability(self, now, default_calibration):
        """The point: a dead sense should register as instability."""
        from anima_mcp.anima import _sense_stability, _FROZEN_REPEAT_THRESHOLD as T
        healthy = _sense_stability(self._readings(now, i=0), default_calibration)
        for i in range(T + 2):
            _sense_stability(self._readings(now, i=i, pressure_hpa=682.5), default_calibration)
        degraded = _sense_stability(
            self._readings(now, i=T + 3, pressure_hpa=682.5), default_calibration
        )
        assert degraded < healthy, "a frozen channel must cost stability"

    def test_frozen_sensor_lowers_clarity_coverage(self, now, default_calibration):
        from anima_mcp.anima import _sense_clarity, _FROZEN_REPEAT_THRESHOLD as T
        healthy = _sense_clarity(self._readings(now, i=0), default_calibration)
        for i in range(T + 2):
            _sense_clarity(self._readings(now, i=i, pressure_hpa=682.5), default_calibration)
        degraded = _sense_clarity(
            self._readings(now, i=T + 3, pressure_hpa=682.5), default_calibration
        )
        assert degraded < healthy, "a frozen channel must cost coverage"

    def test_one_sense_call_advances_liveness_once(self, now, default_calibration):
        from anima_mcp.anima import (
            _FROZEN_REPEAT_THRESHOLD as T,
            _channel_repeat_count,
            sense_self,
        )
        def _sample(index):
            return SensorReadings(
                timestamp=now + timedelta(seconds=index * 2),
                cpu_temp_c=55.0,
                ambient_temp_c=28.0,
                humidity_pct=38.0,
                light_lux=12.0,
                pressure_hpa=830.0,
                cpu_percent=20.0,
                memory_percent=30.0,
                disk_percent=40.0,
                eeg_alpha_power=0.8,
                eeg_beta_power=0.2,
                eeg_gamma_power=0.1,
                eeg_theta_power=0.4,
                eeg_delta_power=0.6,
            )

        for index in range(T):
            sense_self(_sample(index), default_calibration, prediction_accuracy=0.5)

        assert _channel_repeat_count["pressure_hpa"] == T - 1
        sense_self(_sample(T), default_calibration, prediction_accuracy=0.5)
        assert _channel_repeat_count["pressure_hpa"] == T

    def test_reusing_one_physical_sample_is_idempotent(self, now, default_calibration):
        from anima_mcp.anima import _channel_repeat_count, sense_self

        readings = self._readings(now, i=1, pressure_hpa=682.5)
        sense_self(readings, default_calibration, prediction_accuracy=0.5)
        sense_self(readings, default_calibration, prediction_accuracy=0.5)

        assert _channel_repeat_count["pressure_hpa"] == 0

    def test_all_channels_absent_stays_in_range(self, now, default_calibration):
        """missing + frozen must never exceed the channel count."""
        from anima_mcp.anima import _sense_stability
        s = _sense_stability(
            SensorReadings(timestamp=now, cpu_temp_c=None, ambient_temp_c=None,
                           humidity_pct=None, light_lux=None, pressure_hpa=None),
            default_calibration,
        )
        assert 0.0 <= s <= 1.0
