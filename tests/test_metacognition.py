"""
Tests for metacognition.py — prediction state machine, observation,
surprise detection, reflection, curiosity, and persistence.

Covers:
  - Prediction generation (baseline, LED proprioception, diurnal, trend)
  - Observation and surprise calculation
  - should_reflect gating (surprise, cooldown, sustained, multi-source)
  - reflect() assembly
  - Curiosity question generation and domain weighting
  - Curiosity effectiveness tracking (_evaluate_curiosity_outcomes)
  - Surprise trend and prediction accuracy
  - Baseline persistence (save/load)
"""

import pytest
import json
from datetime import datetime

from anima_mcp.metacognition import (
    MetacognitiveMonitor, Prediction, PredictionError, Reflection,
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
    """Test curiosity question generation and domain weighting."""

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

    def test_record_curiosity(self, mm):
        """record_curiosity stores domain error snapshot in log."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            error_light=0.3, error_ambient_temp=0.1,
        )
        mm.record_curiosity(["light", "ambient_temp"], error)
        assert len(mm._curiosity_log) == 1
        assert "light" in mm._curiosity_log[0]["domains"]

    def test_evaluate_curiosity_outcomes_reward(self, mm):
        """When prediction improves after curiosity, domain weight increases."""
        # Record curiosity with high initial error
        initial_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            error_ambient_temp=0.5,
        )
        mm._save_counter = 10
        mm.record_curiosity(["ambient_temp"], initial_error)

        # Fast forward observations past eval horizon
        mm._save_counter = 10 + mm._eval_horizon + 1

        # Current error is much lower — improvement
        current_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            error_ambient_temp=0.1,
        )
        mm._evaluate_curiosity_outcomes(current_error)

        # Domain weight should have increased
        assert mm._domain_weights.get("ambient_temp", 1.0) > 1.0

    def test_evaluate_curiosity_outcomes_penalty(self, mm):
        """When prediction worsens after curiosity, domain weight decreases."""
        initial_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            error_ambient_temp=0.1,
        )
        mm._save_counter = 10
        mm.record_curiosity(["ambient_temp"], initial_error)

        mm._save_counter = 10 + mm._eval_horizon + 1

        # Current error is higher — prediction worsened
        current_error = PredictionError(
            timestamp=datetime.now(),
            prediction=Prediction(timestamp=datetime.now()),
            error_ambient_temp=0.5,
        )
        mm._evaluate_curiosity_outcomes(current_error)

        # Domain weight should have decreased
        assert mm._domain_weights.get("ambient_temp", 1.0) < 1.0


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
        assert "domain_weights" in acc


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
        mm1._domain_weights = {"light": 1.5, "ambient_temp": 0.8}
        mm1.save()

        mm2 = MetacognitiveMonitor(data_dir=str(tmp_path))
        assert mm2._baseline_ambient_temp == pytest.approx(22.5)
        assert mm2._baseline_humidity == pytest.approx(45.0)
        assert mm2._baseline_light == pytest.approx(100.0)
        assert mm2._diurnal_temp[14] == [25.0, 26.0]
        assert mm2._domain_weights.get("light") == pytest.approx(1.5)
        assert mm2._domain_weights.get("ambient_temp") == pytest.approx(0.8)

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
