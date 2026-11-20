"""
Tests for metacognition.py — prediction state machine, observation,
surprise detection, reflection, curiosity, and persistence.

Covers:
  - Prediction generation (baseline, LED proprioception, diurnal, trend)
  - Observation and surprise calculation
  - should_reflect gating (surprise, cooldown, sustained, multi-source)
  - reflect() assembly
  - Curiosity question generation
  - Surprise trend and prediction accuracy
  - Baseline persistence (save/load)
"""

import pytest
import random
import json
from datetime import datetime

from anima_mcp.metacognition import (
    MetacognitiveMonitor, Prediction, PredictionError, Reflection,
    SURPRISE_CHANNELS,
)
from anima_mcp.sensors.base import SensorReadings
from anima_mcp.anima import Anima


# ==================== Helpers ====================

def make_readings(**kwargs) -> SensorReadings:
    """Create SensorReadings with defaults."""
    defaults = dict(
        timestamp=datetime.now(),
        ambient_temp_c=22.0,
        humidity_pct=45.0,
        light_lux=100.0,
        pressure_hpa=1013.0,
        cpu_temp_c=50.0,
    )
    defaults.update(kwargs)
    return SensorReadings(**defaults)


def make_anima(**kwargs) -> Anima:
    """Create Anima with defaults."""
    defaults = dict(
        warmth=0.5, clarity=0.5, stability=0.5, presence=0.5,
        readings=make_readings(),
    )
    defaults.update(kwargs)
    return Anima(**defaults)


@pytest.fixture
def mm(tmp_path):
    """Create MetacognitiveMonitor with temp data dir."""
    return MetacognitiveMonitor(data_dir=str(tmp_path))


# ==================== Prediction ====================

class TestPredict:
    """Test prediction generation."""

    def test_first_prediction_has_no_baselines(self, mm):
        """Fresh monitor has None baselines, prediction uses None values."""
        pred = mm.predict()
        assert isinstance(pred, Prediction)
        assert pred.ambient_temp_c is None
        assert pred.confidence >= 0.3  # minimum confidence

    def test_prediction_after_observation(self, mm):
        """After observing readings, baselines are learned and used in prediction."""
        readings = make_readings(ambient_temp_c=25.0, light_lux=200.0)
        anima = make_anima()
        mm.observe(readings, anima)
        pred = mm.predict()
        assert pred.ambient_temp_c is not None
        assert abs(pred.ambient_temp_c - 25.0) < 1.0

    def test_light_prediction_from_baseline(self, mm):
        """Light prediction uses learned baseline (no glow decomposition)."""
        # Establish a baseline
        readings = make_readings(light_lux=100.0)
        mm.observe(readings, make_anima())

        pred = mm.predict(led_brightness=0.5)
        # Prediction comes from baseline, not LED model
        assert pred.light_lux is not None
        assert abs(pred.light_lux - 100.0) < 10  # Should be near baseline

    def test_confidence_increases_with_history(self, mm):
        """Confidence grows as more history is accumulated."""
        readings = make_readings()
        anima = make_anima()
        pred0 = mm.predict()
        conf0 = pred0.confidence

        for _ in range(25):
            mm.observe(readings, anima)
        pred25 = mm.predict()
        assert pred25.confidence > conf0

    def test_anima_prediction_from_history(self, mm):
        """After observing anima state, prediction includes anima values."""
        readings = make_readings()
        anima = make_anima(warmth=0.8, clarity=0.3)
        mm.observe(readings, anima)
        pred = mm.predict()
        assert pred.warmth == pytest.approx(0.8)
        assert pred.clarity == pytest.approx(0.3)

    def test_diurnal_pattern(self, mm):
        """Diurnal temperature pattern influences prediction at same hour."""
        now = datetime.now()
        hour = now.hour
        # Populate diurnal data for this hour
        mm._diurnal_temp[hour] = [30.0, 30.0, 30.0]
        mm._baseline_ambient_temp = 20.0

        pred = mm.predict(current_time=now)
        # Should blend diurnal (30) with baseline (20): 0.6*30 + 0.4*20 = 26
        assert pred.ambient_temp_c is not None
        assert pred.ambient_temp_c > 20.0  # Pulled toward diurnal pattern


# ==================== Observe ====================

class TestObserve:
    """Test observation and surprise computation."""

    def test_observe_returns_prediction_error(self, mm):
        """observe() returns PredictionError with surprise value."""
        readings = make_readings()
        anima = make_anima()
        error = mm.observe(readings, anima)
        assert isinstance(error, PredictionError)
        assert error.surprise >= 0.0

    def test_observe_updates_baselines(self, mm):
        """After observing, baselines are initialized."""
        assert mm._baseline_ambient_temp is None
        readings = make_readings(ambient_temp_c=22.0)
        mm.observe(readings, make_anima())
        assert mm._baseline_ambient_temp is not None
        assert abs(mm._baseline_ambient_temp - 22.0) < 0.5

    def test_no_surprise_when_stable(self, mm):
        """Repeated identical observations produce low surprise."""
        readings = make_readings(ambient_temp_c=22.0, light_lux=100.0)
        anima = make_anima()
        # Build history so predictions stabilize
        for _ in range(10):
            mm.predict()
            mm.observe(readings, anima)
        # Now another identical reading should have low surprise
        mm.predict()
        error = mm.observe(readings, anima)
        assert error.surprise < 0.15  # Should be very low

    def test_high_surprise_on_large_delta(self, mm):
        """A sudden large temperature change produces high surprise."""
        readings = make_readings(ambient_temp_c=22.0)
        anima = make_anima()
        for _ in range(5):
            mm.predict()
            mm.observe(readings, anima)
        # Sudden jump
        mm.predict()
        shock = make_readings(ambient_temp_c=35.0)
        error = mm.observe(shock, anima)
        assert error.error_ambient_temp > 0.5  # Large temperature error
        assert "ambient_temp" in error.surprise_sources

    def test_observe_appends_to_history(self, mm):
        """Each observation adds to sensor and anima history."""
        assert len(mm._sensor_history) == 0
        mm.observe(make_readings(), make_anima())
        assert len(mm._sensor_history) == 1
        mm.observe(make_readings(), make_anima())
        assert len(mm._sensor_history) == 2

    def test_observe_increments_save_counter(self, mm):
        """Each observation increments _save_counter."""
        assert mm._save_counter == 0
        mm.observe(make_readings(), make_anima())
        assert mm._save_counter == 1


# ==================== LED-stable light gating ====================

class TestLightSurpriseLedGating:
    """The VEML7700 is self-lit. 'light' only enters surprise_sources when LEDs are quiet."""

    def _seed_baseline(self, mm, lux=100.0, led=0.05, ticks=20):
        """Establish a stable light baseline with quiet LEDs."""
        anima = make_anima()
        for _ in range(ticks):
            mm.predict()
            mm.observe(make_readings(light_lux=lux, led_brightness=led), anima)

    def test_light_surprise_passes_when_leds_stable(self, mm):
        mm._led_stable_min_samples = 5  # tighten for fast test
        self._seed_baseline(mm, lux=100.0, led=0.05, ticks=10)
        mm.predict()
        shock = make_readings(light_lux=1000.0, led_brightness=0.05)  # 1-decade jump, LEDs quiet
        error = mm.observe(shock, make_anima())
        assert error.error_light > 0.2
        assert "light" in error.surprise_sources

    def test_light_surprise_blocked_when_leds_dancing(self, mm):
        mm._led_stable_min_samples = 5
        anima = make_anima()
        # Seed with LED swings between 0.05 and 0.8 — clearly unstable
        for i in range(10):
            led = 0.05 if i % 2 == 0 else 0.8
            mm.predict()
            mm.observe(make_readings(light_lux=100.0, led_brightness=led), anima)
        mm.predict()
        shock = make_readings(light_lux=1000.0, led_brightness=0.8)
        error = mm.observe(shock, anima)
        assert error.error_light > 0.2
        assert "light" not in error.surprise_sources

    def test_light_surprise_blocked_without_enough_history(self, mm):
        # Default _led_stable_min_samples=15; only a handful of observations.
        anima = make_anima()
        for _ in range(3):
            mm.predict()
            mm.observe(make_readings(light_lux=100.0, led_brightness=0.05), anima)
        mm.predict()
        shock = make_readings(light_lux=1000.0, led_brightness=0.05)
        error = mm.observe(shock, anima)
        assert error.error_light > 0.2
        assert "light" not in error.surprise_sources

    def test_light_surprise_blocked_when_led_data_missing(self, mm):
        mm._led_stable_min_samples = 5
        anima = make_anima()
        # led_brightness=None across the window — treat as unknown, stay closed
        for _ in range(10):
            mm.predict()
            mm.observe(make_readings(light_lux=100.0, led_brightness=None), anima)
        mm.predict()
        shock = make_readings(light_lux=1000.0, led_brightness=None)
        error = mm.observe(shock, anima)
        assert error.error_light > 0.2
        assert "light" not in error.surprise_sources

    def test_led_is_stable_requires_min_samples(self, mm):
        mm._led_stable_min_samples = 10
        for _ in range(9):
            mm._led_brightness_history.append(0.05)
        assert mm._led_is_stable() is False
        mm._led_brightness_history.append(0.05)
        assert mm._led_is_stable() is True

    def test_led_is_stable_checks_range_not_mean(self, mm):
        mm._led_stable_min_samples = 5
        mm._led_stable_range_threshold = 0.10
        # Mean is low but range is wide — not stable
        for v in [0.0, 0.05, 0.0, 0.20, 0.05]:
            mm._led_brightness_history.append(v)
        assert mm._led_is_stable() is False
        # Reset with a tight range
        mm._led_brightness_history.clear()
        for v in [0.05, 0.06, 0.05, 0.07, 0.06]:
            mm._led_brightness_history.append(v)
        assert mm._led_is_stable() is True


# ==================== PredictionError ====================

class TestPredictionError:
    """Test PredictionError data structure."""

    def test_to_dict(self):
        """to_dict returns serializable dict with errors and sources."""
        pred = Prediction(timestamp=datetime.now(), ambient_temp_c=22.0)
        error = PredictionError(
            timestamp=datetime.now(), prediction=pred,
            error_ambient_temp=0.5, surprise=0.3,
            surprise_sources=["ambient_temp"],
        )
        d = error.to_dict()
        assert d["surprise"] == 0.3
        assert "ambient_temp" in d["surprise_sources"]
        assert d["errors"]["ambient_temp"] == 0.5

    def test_surprise_sources_empty_by_default(self):
        """New PredictionError has empty surprise sources."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
        )
        assert error.surprise_sources == []
        assert error.surprise == 0.0


# ==================== should_reflect ====================

class TestShouldReflect:
    """Test reflection trigger logic."""

    def test_high_surprise_triggers(self, mm):
        """Surprise above threshold triggers reflection."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.5,  # Above default threshold (0.25)
        )
        should, reason = mm.should_reflect(error)
        assert should is True
        assert "high_surprise" in reason

    def test_low_surprise_no_trigger(self, mm):
        """Surprise below threshold does not trigger reflection."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.1,
        )
        should, reason = mm.should_reflect(error)
        assert should is False
        assert reason == "normal"

    def test_cooldown_prevents_reflection(self, mm):
        """Within cooldown period, should_reflect returns False."""
        mm._last_reflection_time = datetime.now()
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.5,
        )
        should, reason = mm.should_reflect(error)
        assert should is False
        assert reason == "cooldown"

    def test_sustained_surprise_triggers(self, mm):
        """Sustained elevated cumulative surprise triggers reflection."""
        mm._cumulative_surprise = mm.surprise_threshold * 0.9  # Above 0.8 * threshold
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.1,  # Low immediate surprise
        )
        should, reason = mm.should_reflect(error)
        assert should is True
        assert "sustained" in reason

    def test_multiple_sources_triggers(self, mm):
        """Three or more surprise sources triggers reflection."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.15,  # Below threshold
            surprise_sources=["ambient_temp", "warmth", "clarity"],
        )
        should, reason = mm.should_reflect(error)
        assert should is True
        assert "multiple" in reason


# ==================== reflect() ====================

class TestReflect:
    """Test reflection generation."""

    def test_reflect_returns_reflection(self, mm):
        """reflect() returns Reflection with correct trigger."""
        readings = make_readings()
        anima = make_anima()
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.3,
        )
        reflection = mm.reflect(error, anima, readings, trigger="surprise")
        assert isinstance(reflection, Reflection)
        assert reflection.trigger == "surprise"
        assert reflection.felt_state is not None
        assert reflection.sensor_state is not None

    def test_reflect_sets_last_reflection_time(self, mm):
        """reflect() updates _last_reflection_time."""
        assert mm._last_reflection_time is None
        readings = make_readings()
        anima = make_anima()
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
        )
        mm.reflect(error, anima, readings)
        assert mm._last_reflection_time is not None

    def test_reflect_observation_from_surprise_sources(self, mm):
        """Reflection observation describes surprise sources."""
        readings = make_readings(ambient_temp_c=30.0)
        anima = make_anima()
        pred = Prediction(timestamp=datetime.now(), ambient_temp_c=22.0)
        error = PredictionError(
            timestamp=datetime.now(), prediction=pred,
            surprise=0.3,
            surprise_sources=["ambient_temp"],
            actual_ambient_temp_c=30.0,
        )
        reflection = mm.reflect(error, anima, readings)
        assert "Temperature" in reflection.observation

    def test_reflect_to_dict(self, mm):
        """Reflection.to_dict serializes correctly."""
        readings = make_readings()
        anima = make_anima()
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.2,
        )
        reflection = mm.reflect(error, anima, readings)
        d = reflection.to_dict()
        assert "trigger" in d
        assert "observation" in d
        assert d["surprise"] == 0.2

    def test_manual_reflection(self, mm):
        """trigger_manual_reflection creates button-triggered reflection."""
        readings = make_readings()
        anima = make_anima()
        reflection = mm.trigger_manual_reflection(anima, readings)
        assert reflection.trigger == "button"

    def test_discrepancy_detection(self, mm):
        """Large warmth-temperature discrepancy is detected."""
        # Temp = 35°C implies warmth = (35-15)/20 = 1.0
        # But anima.warmth = 0.2 → large discrepancy
        readings = make_readings(ambient_temp_c=35.0)
        anima = make_anima(warmth=0.2, readings=readings)
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
        )
        reflection = mm.reflect(error, anima, readings)
        assert reflection.discrepancy > 0.3
        assert "cooler" in reflection.discrepancy_description.lower()


# ==================== Curiosity ====================

class TestCuriosity:
    """Test curiosity question generation."""

    def test_no_curiosity_low_surprise(self, mm):
        """Low surprise produces no curiosity question."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.1,
        )
        q = mm.generate_curiosity_question(error)
        assert q is None

    def test_curiosity_from_temperature_surprise(self, mm):
        """Temperature surprise generates temp-related question."""
        pred = Prediction(timestamp=datetime.now(), ambient_temp_c=22.0)
        error = PredictionError(
            timestamp=datetime.now(), prediction=pred,
            surprise=0.4,
            surprise_sources=["ambient_temp"],
            actual_ambient_temp_c=30.0,
        )
        q = mm.generate_curiosity_question(error)
        assert q is not None
        assert "temperature" in q.lower() or "warmer" in q.lower() or "°C" in q

    def test_curiosity_from_warmth_surprise(self, mm):
        """Warmth surprise generates warmth-related question."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now(), warmth=0.5),
            surprise=0.3,
            surprise_sources=["warmth"],
            actual_warmth=0.8,
        )
        q = mm.generate_curiosity_question(error)
        assert q is not None
        assert "warmth" in q.lower() or "warm" in q.lower() or "comfort" in q.lower()

    def test_curiosity_multiple_sources(self, mm):
        """Multiple surprise sources generate connection question."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.3,
            surprise_sources=["ambient_temp", "warmth"],
            actual_ambient_temp_c=30.0,
            actual_warmth=0.8,
        )
        # With multiple sources, should get a question (either specific or connection)
        q = mm.generate_curiosity_question(error)
        assert q is not None

    def test_generic_curiosity_high_surprise(self, mm):
        """High surprise with no specific sources generates generic question."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            surprise=0.5,
            surprise_sources=[],  # No specific sources
        )
        q = mm.generate_curiosity_question(error)
        assert q is not None

    def test_get_domain_error_honest_none_and_measured_zero(self, mm):
        """Unmeasured channels return None even when error_* defaults to 0.0;
        measured exact matches return 0.0."""
        blank_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
        )
        for ch in SURPRISE_CHANNELS:
            assert mm._get_domain_error(ch, blank_error) is None
        assert mm._get_domain_error("cpu_temp", blank_error) is None

        # Only actual present (prediction None) still returns None.
        half_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            actual_pressure_hpa=1013.0,
            error_pressure=0.0,
        )
        assert mm._get_domain_error("pressure", half_error) is None

        # Zero lux cannot compute log10 error in observe(), so treat as unmeasured.
        zero_lux_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now(), light_lux=0.0),
            actual_light_lux=10.0,
            error_light=0.0,
        )
        assert mm._get_domain_error("light", zero_lux_error) is None

        # Exact measured match (predicted == actual == 22.0) returns 0.0.
        exact_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now(), ambient_temp_c=22.0),
            actual_ambient_temp_c=22.0,
            error_ambient_temp=0.0,
        )
        assert mm._get_domain_error("ambient_temp", exact_error) == 0.0

    def test_curiosity_handles_zero_celsius_temp(self, mm):
        """0.0°C is a valid temperature, not missing data."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now(), ambient_temp_c=5.0),
            actual_ambient_temp_c=0.0,
            surprise=0.22,
            surprise_sources=["ambient_temp"],
        )
        q = mm.generate_curiosity_question(error)
        assert q is not None and "0.0°C" in q

    def test_surprise_channel_parity_across_observe_phrasing_and_templates(self, tmp_path):
        """Every channel observe() can tag has phrasing in _SURPRISE_PHRASING and
        a channel-specific question template in generate_curiosity_question()."""
        from anima_mcp.loop_phases import _SURPRISE_PHRASING

        assert set(_SURPRISE_PHRASING.keys()) == set(SURPRISE_CHANNELS)

        observed_channels = set()
        channel_spikes = [
            ("ambient_temp", make_readings(ambient_temp_c=30.0), {}),
            ("humidity", make_readings(humidity_pct=75.0), {}),
            ("light", make_readings(light_lux=5000.0, led_brightness=0.5), {}),
            ("pressure", make_readings(pressure_hpa=1030.0), {}),
            ("warmth", make_readings(), {"warmth": 0.9}),
            ("clarity", make_readings(), {"clarity": 0.2}),
            ("stability", make_readings(), {"stability": 0.2}),
            ("presence", make_readings(), {"presence": 0.2}),
        ]
        for ch, spike_readings, anima_overrides in channel_spikes:
            mon = MetacognitiveMonitor(data_dir=str(tmp_path / ch))
            base_readings = make_readings(led_brightness=0.5)
            base_anima = make_anima()
            for _ in range(mon._led_stable_min_samples):
                mon.predict(led_brightness=0.5)
                mon.observe(base_readings, base_anima)
            mon.predict(led_brightness=0.5)
            err = mon.observe(
                spike_readings,
                make_anima(**anima_overrides) if anima_overrides else base_anima,
            )
            assert ch in err.surprise_sources
            observed_channels.add(ch)

        assert observed_channels == set(SURPRISE_CHANNELS)

        # Every channel produces a question at surprise=0.22 (< 0.25 generic fallback)
        # both with full values and with value-free fallback.
        pred = Prediction(
            timestamp=datetime.now(),
            ambient_temp_c=22.0,
            humidity_pct=45.0,
            light_lux=100.0,
            pressure_hpa=1013.0,
            warmth=0.5,
            clarity=0.7,
            stability=0.7,
            presence=0.7,
        )
        for ch in SURPRISE_CHANNELS:
            populated = PredictionError(
                timestamp=datetime.now(),
                prediction=pred,
                actual_ambient_temp_c=28.0,
                actual_humidity_pct=60.0,
                actual_light_lux=500.0,
                actual_pressure_hpa=1025.0,
                actual_warmth=0.8,
                actual_clarity=0.3,
                actual_stability=0.3,
                actual_presence=0.3,
                surprise=0.22,
                surprise_sources=[ch],
            )
            q_pop = MetacognitiveMonitor(data_dir=str(tmp_path / f"q_{ch}")).generate_curiosity_question(populated)
            assert q_pop is not None, f"missing populated template for {ch}"
            if ch == "presence":
                assert any(w in q_pop.lower() for w in ("presence", "headroom", "capacity"))
                assert "room" not in q_pop.lower()

            value_free = PredictionError(
                timestamp=datetime.now(),
                prediction=Prediction(timestamp=datetime.now()),
                surprise=0.22,
                surprise_sources=[ch],
            )
            q_vf = MetacognitiveMonitor(data_dir=str(tmp_path / f"vf_{ch}")).generate_curiosity_question(value_free)
            assert q_vf is not None, f"missing value-free fallback for {ch}"


# ==================== Surprise Trend & Accuracy ====================

class TestSurpriseTrendAndAccuracy:
    """Test surprise trend and prediction accuracy metrics."""

    def test_surprise_trend_empty(self, mm):
        """Empty history returns 0 surprise trend."""
        assert mm.get_surprise_trend() == 0.0

    def test_surprise_trend_computed(self, mm):
        """Surprise trend averages recent error surprises."""
        for i in range(5):
            err = PredictionError(
                timestamp=datetime.now(),
                prediction=Prediction(timestamp=datetime.now()),
                surprise=0.1 * (i + 1),  # 0.1, 0.2, 0.3, 0.4, 0.5
            )
            mm._error_history.append(err)
        trend = mm.get_surprise_trend(window=5)
        assert trend == pytest.approx(0.3)  # Average of 0.1..0.5

    def test_prediction_accuracy_insufficient(self, mm):
        """With < 5 errors, returns insufficient_data flag."""
        acc = mm.get_prediction_accuracy()
        assert acc.get("insufficient_data") is True

    def test_prediction_accuracy_computed(self, mm):
        """With enough errors, returns accuracy metrics."""
        for i in range(10):
            err = PredictionError(
                timestamp=datetime.now(),
                prediction=Prediction(timestamp=datetime.now()),
                surprise=0.2,
                error_ambient_temp=0.1,
                error_warmth=0.05,
                error_clarity=0.03,
            )
            mm._error_history.append(err)
        acc = mm.get_prediction_accuracy()
        assert "mean_surprise" in acc
        assert acc["mean_surprise"] == pytest.approx(0.2)
        assert acc["mean_temp_error"] == pytest.approx(0.1)
        assert "reflection_count" in acc
        assert "domain_weights" not in acc
        assert "pending_curiosity_evals" not in acc


# ==================== Persistence ====================

class TestPersistence:
    """Test baseline save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        """Baselines survive save/load cycle."""
        mm1 = MetacognitiveMonitor(data_dir=str(tmp_path))
        mm1._baseline_ambient_temp = 22.5
        mm1._baseline_humidity = 45.0
        mm1._baseline_light = 100.0
        mm1._baseline_pressure = 1013.0
        mm1._diurnal_temp[14] = [25.0, 26.0]
        mm1._save_counter = 17
        assert mm1.save() is True

        mm2 = MetacognitiveMonitor(data_dir=str(tmp_path))
        assert mm2._baseline_ambient_temp == pytest.approx(22.5)
        assert mm2._baseline_humidity == pytest.approx(45.0)
        assert mm2._baseline_light == pytest.approx(100.0)
        assert mm2._diurnal_temp[14] == [25.0, 26.0]
        assert mm2._save_counter == 17

    def test_load_tolerates_legacy_curiosity_keys_and_drops_on_save(self, tmp_path):
        """Legacy domain_weights and curiosity_log load cleanly and drop on save."""
        data_file = tmp_path / "metacognition_baselines.json"
        data_file.write_text(json.dumps({
            "baseline_ambient_temp": 22.5,
            "domain_weights": {"light": 1.5, "ambient_temp": 0.8},
            "curiosity_log": [{"domains": ["light"], "errors_at_time": {"light": 0.4}, "obs_count": 12}],
            "observation_count": 17,
        }))

        mm = MetacognitiveMonitor(data_dir=str(tmp_path))
        assert mm._baseline_ambient_temp == pytest.approx(22.5)
        assert mm._save_counter == 17
        assert mm.save() is True

        saved = json.loads(data_file.read_text())
        assert "domain_weights" not in saved
        assert "curiosity_log" not in saved

    def test_read_only_observer_never_overwrites_snapshot(self, tmp_path):
        writer = MetacognitiveMonitor(data_dir=str(tmp_path))
        writer._baseline_ambient_temp = 22.5
        assert writer.save() is True
        before = (tmp_path / "metacognition_baselines.json").read_bytes()

        observer = MetacognitiveMonitor(data_dir=str(tmp_path), read_only=True)
        observer._baseline_ambient_temp = 99.0

        assert observer.is_writable is False
        assert observer.save() is False
        assert (tmp_path / "metacognition_baselines.json").read_bytes() == before

    def test_load_missing_file(self, tmp_path):
        """Loading with no file doesn't crash, uses None baselines."""
        mm = MetacognitiveMonitor(data_dir=str(tmp_path / "nonexistent"))
        assert mm._baseline_ambient_temp is None

    def test_periodic_save(self, tmp_path):
        """Every 100 observations, baselines are auto-saved."""
        mm = MetacognitiveMonitor(data_dir=str(tmp_path))
        readings = make_readings(ambient_temp_c=22.0)
        anima = make_anima()

        data_file = tmp_path / "metacognition_baselines.json"
        assert not data_file.exists()

        for _ in range(100):
            mm.observe(readings, anima)

        assert data_file.exists()
        data = json.loads(data_file.read_text())
        assert data["baseline_ambient_temp"] is not None


# ==================== get_recent_reflections ====================

class TestGetRecentReflections:
    """Test reflection history retrieval."""

    def test_empty_returns_empty(self, mm):
        assert mm.get_recent_reflections() == []

    def test_returns_most_recent(self, mm):
        """Returns last N reflections."""
        readings = make_readings()
        anima = make_anima()
        for i in range(3):
            error = PredictionError(
                timestamp=datetime.now(),
                prediction=Prediction(timestamp=datetime.now()),
                surprise=0.3 + i * 0.1,
            )
            mm.reflect(error, anima, readings, trigger=f"test_{i}")

        recent = mm.get_recent_reflections(count=2)
        assert len(recent) == 2
        assert recent[-1].trigger == "test_2"


# ==================== Contemplative rotation ====================

class TestContemplativeRotation:
    """The stillness-born question pool must ROTATE, not clump.

    The board already refuses a same-question re-ask inside
    ``QUESTION_EXPIRY_SECONDS``, and that guard works — measured on the live
    board, zero repeat pairs fell under 4 h. But expiry and dedup are the same
    constant, so a prompt becomes eligible again the moment it expires, and an
    ANSWERED prompt is treated exactly like one nobody looked at. Observed gaps
    clustered at 4.00/4.04/4.11/4.14/4.16/4.27/4.29/4.44 h: the window
    functioning as a clock, the same failure shape as the drawing engine's 8 h
    cap.

    Rotation alone is not the fix — simulated at the live rate it lifts the
    minimum gap but raises volume by a third. Retirement of answered prompts
    is the load-bearing half; rotation covers the remainder evenly.
    """

    def _pool(self, mm):
        """Read the pool directly. Sampling for it burned tens of thousands of
        global-RNG draws per run and perturbed unrelated suites."""
        from anima_mcp.metacognition import CONTEMPLATIVE_PROMPTS
        mm._contemplative_curiosity_rate = 1.0
        return set(CONTEMPLATIVE_PROMPTS)

    def test_no_record_still_produces_a_question(self, mm):
        """Cold start (empty board) keeps today's behavior."""
        mm._contemplative_curiosity_rate = 1.0
        assert mm.generate_contemplative_question(recently_asked=[]) in self._pool(mm)
        assert mm.generate_contemplative_question(recently_asked=None) in self._pool(mm)

    def test_rate_limit_still_gates(self, mm):
        """Rotation must not turn an occasional wondering into a flood."""
        mm._contemplative_curiosity_rate = 0.0
        assert mm.generate_contemplative_question(recently_asked=[]) is None

    def test_never_asked_prompt_wins(self, mm):
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        asked, held_back = pool[:-1], pool[-1]
        for _ in range(30):
            assert mm.generate_contemplative_question(recently_asked=asked) == held_back

    def test_picks_the_longest_unasked(self, mm):
        """All nine used: the one asked longest ago comes back next."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        record = list(pool)  # oldest first
        for _ in range(30):
            assert mm.generate_contemplative_question(recently_asked=record) == record[0]

    def test_repeats_only_after_the_whole_pool(self, mm):
        """The regression that matters: driving the generator from its own
        record must space a prompt out by the full pool, not by the expiry
        window. Nine draws must yield nine distinct prompts."""
        mm._contemplative_curiosity_rate = 1.0
        pool = self._pool(mm)
        record = []
        for _ in range(len(pool)):
            q = mm.generate_contemplative_question(recently_asked=record)
            assert q not in record, f"{q!r} repeated before the pool was exhausted"
            record.append(q)
        assert set(record) == pool

    def test_unrelated_questions_do_not_disturb_rotation(self, mm):
        """Agency/curiosity questions share the board but not the pool."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        asked, held_back = pool[:-1], pool[-1]
        noisy = []
        for q in asked:
            noisy += [q, "why does the air is dry affect me?"]
        assert mm.generate_contemplative_question(recently_asked=noisy) == held_back

    def test_duplicate_heavy_record_uses_the_latest_appearance(self, mm):
        """The real input shape: the live board held 36 contemplative questions
        drawn from 9 strings, so every prompt appears several times. Only the
        MOST RECENT appearance should count — an old copy must not make a
        just-asked prompt look cold."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        # Everything asked once long ago, then all but `stale` asked again.
        recent_again = [p for p in pool if p != pool[0]]
        record = list(pool) + recent_again
        for _ in range(30):
            assert mm.generate_contemplative_question(recently_asked=record) == pool[0]

        # And the reverse: the prompt whose only appearance is the newest one
        # must not be chosen while anything older is available.
        record2 = list(pool[:-1]) + [pool[-1]]
        assert mm.generate_contemplative_question(recently_asked=record2) == pool[0]

    def test_answered_prompts_retire(self, mm):
        """A question that got a real answer is finished. Re-asking it is
        reciting, not wondering."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        retired, live = pool[:-1], pool[-1]
        for _ in range(30):
            q = mm.generate_contemplative_question(already_answered=retired)
            assert q == live

    def test_exhausted_pool_stays_quiet(self, mm):
        """Every prompt answered -> honest silence, not a tenth variation.
        This is the behavioural consequence to weigh: in a long stationary
        stretch the contemplative channel goes quiet by design."""
        mm._contemplative_curiosity_rate = 1.0
        pool = self._pool(mm)
        for _ in range(30):
            assert mm.generate_contemplative_question(already_answered=pool) is None

    def test_unanswered_prompts_keep_coming_back(self, mm):
        """The other half of the contract: retirement is earned by an ANSWER,
        never by an expiry. Anything nobody replied to stays in rotation —
        the same principle that leaves an unanswered drive: request visible."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        # Asked many times over, answered never.
        record = pool * 4
        q = mm.generate_contemplative_question(recently_asked=record, already_answered=set())
        assert q in pool

    def test_retirement_and_rotation_compose(self, mm):
        """Retired prompts are excluded from the rotation, not merely ranked
        down — the coldest ANSWERED prompt must never win."""
        mm._contemplative_curiosity_rate = 1.0
        pool = sorted(self._pool(mm))
        coldest_but_answered = pool[0]
        record = [coldest_but_answered] + pool[1:] * 3
        for _ in range(30):
            q = mm.generate_contemplative_question(
                recently_asked=record, already_answered={coldest_but_answered}
            )
            assert q != coldest_but_answered
            assert q in pool[1:]


class TestClarityQuestionCoverage:
    """Lumen has to be able to name what surprised it.

    `generate_curiosity_question` branches on clarity at 0.6 / 0.4. Warmth has
    always carried an `else` for the band between; clarity did not, so a clarity
    surprise landing there produced no clarity-specific question. Measured at
    c=0.50 before the fix:

        sources=["clarity"]            6 questions, 0 naming clarity
                                       (fell through to the generic
                                       "surprise > 0.25" pool)
        sources=["clarity","stability"] 11 questions, 0 naming clarity
                                       (that fallback is gated on
                                       `not questions`, and stability had
                                       already filled the list)

    The band is wider than the cuts suggest: clarity's lived range is
    0.454-0.910 over 833 drawing_records, so the `c < 0.4` arm sits below the
    whole distribution and the missing `else` covered everything under 0.6.

    These tests assert clarity is NAMED, not merely that some question exists —
    an earlier version asserted the latter and passed against the unfixed code,
    because the generic pool answered for it.
    """

    def _error(self, clarity, sources=("clarity",)):
        from anima_mcp.metacognition import Prediction, PredictionError
        return PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            actual_clarity=clarity,
            error_clarity=0.4,
            surprise=0.5,
            surprise_sources=list(sources),
        )

    def _names_clarity(self, mc, clarity, sources=("clarity",), draws=300):
        """Sample the random choice; return whether clarity is ever named."""
        random.seed(1234)
        for _ in range(draws):
            q = mc.generate_curiosity_question(self._error(clarity, sources))
            if q and "clarity" in q.lower():
                return True
        return False

    @pytest.mark.parametrize("c", [0.40, 0.45, 0.50, 0.55, 0.60])
    def test_midrange_clarity_is_named(self, c):
        assert self._names_clarity(MetacognitiveMonitor(), c), (
            f"clarity {c} surprised Lumen but no question said so")

    def test_named_even_when_another_source_fills_the_list(self):
        """The stricter half: the generic fallback cannot cover for clarity
        here, because it only fires when nothing else produced questions."""
        assert self._names_clarity(
            MetacognitiveMonitor(), 0.50, ("clarity", "stability"))

    def test_whole_lived_range_names_clarity(self):
        """Sweep the measured 0.454-0.910 range: no unnamed bands left."""
        mc = MetacognitiveMonitor()
        for i in range(24):
            c = 0.454 + (0.910 - 0.454) * i / 23
            assert self._names_clarity(mc, c, draws=60), f"unnamed at {c:.3f}"

    def test_high_clarity_keeps_its_own_vocabulary(self):
        """The fall-through must not swallow the arm that already worked."""
        random.seed(7)
        mc = MetacognitiveMonitor()
        seen = {mc.generate_curiosity_question(self._error(0.85))
                for _ in range(80)}
        assert seen - {None}
        assert not any(q and "shifted to" in q for q in seen)

    def test_below_the_surprise_floor_still_asks_nothing(self):
        """Unchanged: a small error is not wonder."""
        from anima_mcp.metacognition import Prediction, PredictionError
        mc = MetacognitiveMonitor()
        quiet = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            actual_clarity=0.5, surprise=0.1, surprise_sources=["clarity"],
        )
        assert mc.generate_curiosity_question(quiet) is None
