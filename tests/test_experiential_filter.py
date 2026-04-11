"""
Tests for ExperientialFilter — Layer 2 of Experiential Accumulation.

Salience weights per sensor dimension drift based on surprise and
dissatisfaction, amplifying signals the creature should pay attention to.
"""

import json
import time
import pytest
from datetime import datetime

from anima_mcp.experiential_filter import (
    ExperientialFilter,
    SalienceWeight,
    DIMENSIONS,
    SALIENCE_MAX,
    SALIENCE_MIN,
    PREF_TO_SENSOR,
    get_experiential_filter,
)
from anima_mcp.anima import sense_self
from anima_mcp.sensors.base import SensorReadings
from anima_mcp.config import NervousSystemCalibration


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton between tests."""
    import anima_mcp.experiential_filter as ef
    ef._instance = None
    yield
    ef._instance = None


@pytest.fixture
def ef(tmp_path):
    """ExperientialFilter backed by a temporary file."""
    path = str(tmp_path / "experiential_filter.json")
    return ExperientialFilter(persistence_path=path)


@pytest.fixture
def now():
    return datetime.now()


@pytest.fixture
def normal_readings(now):
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


class TestDefaultSalience:
    def test_default_salience_is_neutral(self, ef):
        """All dimensions start at 1.0 (neutral)."""
        for dim in DIMENSIONS:
            assert ef.get_salience(dim) == 1.0

    def test_get_all_saliences(self, ef):
        """get_all_saliences returns a dict of all dimensions at 1.0."""
        saliences = ef.get_all_saliences()
        assert len(saliences) == len(DIMENSIONS)
        for dim in DIMENSIONS:
            assert saliences[dim] == 1.0


class TestSurpriseAmplification:
    def test_surprise_amplifies_salience(self, ef):
        """Surprise on 'light' increases light salience."""
        before = ef.get_salience("light")
        ef.update_from_surprise(["light"], 0.8)
        after = ef.get_salience("light")
        assert after > before
        # 0.02 * 0.8 = 0.016
        assert abs(after - (1.0 + 0.016)) < 0.001

    def test_surprise_maps_sources_correctly(self, ef):
        """Verify source_to_dim mapping works for all known sources."""
        # "light_lux" should map to "light"
        ef.update_from_surprise(["light_lux"], 1.0)
        assert ef.get_salience("light") > 1.0

        # "temperature" should map to "ambient_temp"
        ef.update_from_surprise(["temperature"], 1.0)
        assert ef.get_salience("ambient_temp") > 1.0

        # "humidity_pct" should map to "humidity"
        ef.update_from_surprise(["humidity_pct"], 1.0)
        assert ef.get_salience("humidity") > 1.0

        # Unknown source should be ignored without error
        ef.update_from_surprise(["unknown_sensor"], 1.0)

    def test_multiple_sources_in_one_call(self, ef):
        """Multiple surprise sources amplify their respective dimensions."""
        ef.update_from_surprise(["light", "cpu_temp", "pressure"], 0.5)
        assert ef.get_salience("light") > 1.0
        assert ef.get_salience("cpu_temp") > 1.0
        assert ef.get_salience("pressure") > 1.0
        # Unrelated dimension unaffected
        assert ef.get_salience("humidity") == 1.0

    def test_temp_dampening_reduces_temp_amplification(self, ef):
        """temp_dampening reduces surprise amplification for temperature dims."""
        # Without dampening
        ef.update_from_surprise(["cpu_temp"], 0.8, temp_dampening=0.0)
        undampened = ef.get_salience("cpu_temp")

        # Reset
        ef._weights["cpu_temp"].weight = 1.0

        # With 10% dampening
        ef.update_from_surprise(["cpu_temp"], 0.8, temp_dampening=0.10)
        dampened = ef.get_salience("cpu_temp")

        assert dampened < undampened
        assert dampened > 1.0  # Still amplified, just less

    def test_temp_dampening_doesnt_affect_non_temp(self, ef):
        """temp_dampening only affects ambient_temp and cpu_temp dimensions."""
        ef.update_from_surprise(["light"], 0.8, temp_dampening=0.10)
        light_val = ef.get_salience("light")
        # Should be same as without dampening: 1.0 + 0.02 * 0.8 = 1.016
        assert abs(light_val - 1.016) < 0.001


class TestDissatisfactionAmplification:
    def test_dissatisfaction_amplifies(self, ef):
        """'warmth' dissatisfaction increases ambient_temp salience."""
        before = ef.get_salience("ambient_temp")
        ef.update_from_dissatisfaction("warmth")
        after = ef.get_salience("ambient_temp")
        assert after > before
        assert abs(after - 1.001) < 0.0001

    def test_dissatisfaction_maps_all_prefs(self, ef):
        """All preference-to-sensor mappings work."""
        for pref, sensor in PREF_TO_SENSOR.items():
            before = ef.get_salience(sensor)
            ef.update_from_dissatisfaction(pref)
            assert ef.get_salience(sensor) > before

    def test_unknown_dissatisfaction_ignored(self, ef):
        """Unknown preference dimensions are silently ignored."""
        saliences_before = ef.get_all_saliences()
        ef.update_from_dissatisfaction("nonexistent")
        saliences_after = ef.get_all_saliences()
        assert saliences_before == saliences_after


class TestDecay:
    def test_decay_toward_neutral(self, ef):
        """Salience > 1.0 decays back toward 1.0."""
        ef.update_from_surprise(["light"], 1.0)
        initial = ef.get_salience("light")
        assert initial > 1.0

        # Decay many times
        for _ in range(1000):
            ef.tick()

        after = ef.get_salience("light")
        assert after < initial
        assert after > 1.0  # Still above neutral (slow decay)

    def test_decay_below_neutral(self, tmp_path):
        """Salience < 1.0 (manually set) decays up toward 1.0."""
        path = str(tmp_path / "ef.json")
        ef = ExperientialFilter(persistence_path=path)
        # Manually set weight below neutral
        ef._weights["light"].weight = 0.8
        initial = ef.get_salience("light")
        assert initial < 1.0

        for _ in range(1000):
            ef.tick()

        after = ef.get_salience("light")
        assert after > initial
        assert after < 1.0  # Still below neutral (slow decay)

    def test_decay_rate(self):
        """Verify the decay math: excess *= 0.9995 per tick."""
        sw = SalienceWeight(dimension="test", weight=1.5)
        sw.decay_toward_neutral()
        expected = 1.0 + 0.5 * 0.9995
        assert abs(sw.weight - expected) < 0.0001


class TestBounds:
    def test_salience_bounds_max(self, ef):
        """Salience never exceeds 2.0."""
        # Hammer surprise many times
        for _ in range(200):
            ef.update_from_surprise(["light"], 1.0)
        assert ef.get_salience("light") <= SALIENCE_MAX

    def test_salience_bounds_min(self, tmp_path):
        """Salience never falls below 0.5."""
        path = str(tmp_path / "ef.json")
        ef = ExperientialFilter(persistence_path=path)
        # Manually set very low
        ef._weights["light"].weight = 0.1
        ef.tick()  # Decay should enforce minimum
        assert ef.get_salience("light") >= SALIENCE_MIN

    def test_surprise_respects_max(self):
        """SalienceWeight.amplify_from_surprise respects max bound."""
        sw = SalienceWeight(dimension="test", weight=1.99)
        sw.amplify_from_surprise(1.0)
        assert sw.weight <= SALIENCE_MAX

    def test_dissatisfaction_respects_max(self):
        """SalienceWeight.amplify_from_dissatisfaction respects max bound."""
        sw = SalienceWeight(dimension="test", weight=1.999)
        sw.amplify_from_dissatisfaction()
        assert sw.weight <= SALIENCE_MAX


class TestPersistence:
    def test_persist_and_reload(self, tmp_path):
        """JSON round-trip preserves state."""
        path = str(tmp_path / "experiential_filter.json")
        ef1 = ExperientialFilter(persistence_path=path)

        # Modify some weights
        ef1.update_from_surprise(["light", "cpu_temp"], 0.5)
        ef1.update_from_dissatisfaction("warmth")
        ef1.save()

        # Reload
        ef2 = ExperientialFilter(persistence_path=path)
        assert abs(ef2.get_salience("light") - ef1.get_salience("light")) < 0.001
        assert abs(ef2.get_salience("cpu_temp") - ef1.get_salience("cpu_temp")) < 0.001
        assert abs(ef2.get_salience("ambient_temp") - ef1.get_salience("ambient_temp")) < 0.001

    def test_save_interval(self, tmp_path):
        """Doesn't save every tick, only every 120s."""
        path = str(tmp_path / "experiential_filter.json")
        ef = ExperientialFilter(persistence_path=path)
        ef.save()  # Initial save

        # Record initial mtime
        import os
        initial_mtime = os.path.getmtime(path)

        # Tick without enough time passing — should NOT save
        ef.tick()
        # File should not have been updated (no 120s elapsed)
        assert os.path.getmtime(path) == initial_mtime

    def test_save_interval_elapsed(self, tmp_path):
        """Save happens when 120s have elapsed."""
        path = str(tmp_path / "experiential_filter.json")
        ef = ExperientialFilter(persistence_path=path)
        ef.save()  # Initial save

        # Fake time elapsed
        ef._last_save_time = time.time() - 130  # 130s ago

        ef.tick()  # Should trigger save

        # Verify file was written with updated data
        with open(path) as f:
            data = json.load(f)
        assert "weights" in data

    def test_load_missing_file(self, tmp_path):
        """Loading from nonexistent file starts fresh."""
        path = str(tmp_path / "nonexistent.json")
        ef = ExperientialFilter(persistence_path=path)
        for dim in DIMENSIONS:
            assert ef.get_salience(dim) == 1.0

    def test_load_corrupt_file(self, tmp_path):
        """Loading from corrupt file starts fresh."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        ef = ExperientialFilter(persistence_path=str(path))
        for dim in DIMENSIONS:
            assert ef.get_salience(dim) == 1.0


class TestGetStats:
    def test_get_stats(self, ef):
        """Returns correct summary with biased dimensions."""
        # Initially, nothing biased
        stats = ef.get_stats()
        assert stats["dimensions"] == len(DIMENSIONS)
        assert stats["biased_count"] == 0
        assert stats["biased_dimensions"] == {}
        assert abs(stats["mean_salience"] - 1.0) < 0.001

        # After surprise, light becomes biased
        ef.update_from_surprise(["light"], 1.0)
        stats = ef.get_stats()
        assert stats["biased_count"] == 1
        assert "light" in stats["biased_dimensions"]
        assert stats["mean_salience"] > 1.0


class TestSalienceWeightDataclass:
    def test_to_dict_roundtrip(self):
        """SalienceWeight serializes and deserializes correctly."""
        sw = SalienceWeight(
            dimension="light",
            weight=1.5,
            surprise_accumulator=0.3,
            dissatisfaction_ticks=5,
        )
        d = sw.to_dict()
        sw2 = SalienceWeight.from_dict(d)
        assert sw2.dimension == sw.dimension
        assert sw2.weight == sw.weight
        assert sw2.surprise_accumulator == sw.surprise_accumulator
        assert sw2.dissatisfaction_ticks == sw.dissatisfaction_ticks


class TestIntegrationSenseSelf:
    def test_integration_sense_self(self, normal_readings):
        """Salience actually changes anima output values."""
        cal = NervousSystemCalibration()

        # Compute with no salience (neutral)
        anima_neutral = sense_self(normal_readings, cal)

        # Compute with amplified cpu_temp salience (should increase warmth)
        salience = {"cpu_temp": 1.5, "ambient_temp": 1.5}
        anima_amplified = sense_self(normal_readings, cal, salience_weights=salience)

        # Warmth should differ: higher salience on thermal sensors
        # The exact direction depends on the sensor values, but they should differ
        assert anima_amplified.warmth != anima_neutral.warmth

    def test_integration_clarity_with_salience(self, normal_readings):
        """Light salience changes clarity."""
        cal = NervousSystemCalibration()

        anima_neutral = sense_self(normal_readings, cal)

        # Amplify light salience
        salience = {"light": 1.8}
        anima_amplified = sense_self(normal_readings, cal, salience_weights=salience)

        # Clarity should differ
        assert anima_amplified.clarity != anima_neutral.clarity

    def test_integration_stability_with_salience(self, normal_readings):
        """Humidity/pressure salience changes stability."""
        cal = NervousSystemCalibration()

        anima_neutral = sense_self(normal_readings, cal)

        # Amplify humidity and pressure salience
        salience = {"humidity": 1.8, "pressure": 1.8, "memory": 1.5}
        anima_amplified = sense_self(normal_readings, cal, salience_weights=salience)

        # Stability should differ
        assert anima_amplified.stability != anima_neutral.stability

    def test_integration_presence_with_salience(self, normal_readings):
        """Memory/CPU salience changes presence."""
        cal = NervousSystemCalibration()

        anima_neutral = sense_self(normal_readings, cal)

        # Amplify memory and cpu salience
        salience = {"memory": 1.8, "cpu": 1.8}
        anima_amplified = sense_self(normal_readings, cal, salience_weights=salience)

        # Presence should differ (higher salience on void signals = more void = less presence)
        assert anima_amplified.presence != anima_neutral.presence

    def test_neutral_salience_no_change(self, normal_readings):
        """Salience all at 1.0 should produce identical results to no salience."""
        import anima_mcp.computational_neural as cn

        cal = NervousSystemCalibration()

        # sense_self() mutates the global neural EMA state on every call.
        # Reset between comparisons so neutral salience is measured from the
        # same initial neural state rather than from sequential drift.
        cn._sensor = None
        anima_none = sense_self(normal_readings, cal)
        cn._sensor = None
        salience = {dim: 1.0 for dim in DIMENSIONS}
        anima_neutral = sense_self(normal_readings, cal, salience_weights=salience)

        assert abs(anima_none.warmth - anima_neutral.warmth) < 0.05
        assert abs(anima_none.clarity - anima_neutral.clarity) < 0.05
        assert abs(anima_none.stability - anima_neutral.stability) < 0.05
        assert abs(anima_none.presence - anima_neutral.presence) < 0.05

    def test_all_values_still_in_range(self, normal_readings):
        """Even with extreme salience, anima stays in [0, 1]."""
        cal = NervousSystemCalibration()
        salience = {dim: SALIENCE_MAX for dim in DIMENSIONS}
        anima = sense_self(normal_readings, cal, salience_weights=salience)
        assert 0 <= anima.warmth <= 1
        assert 0 <= anima.clarity <= 1
        assert 0 <= anima.stability <= 1
        assert 0 <= anima.presence <= 1


class TestSingleton:
    def test_singleton_returns_same_instance(self, tmp_path):
        """get_experiential_filter returns the same instance."""
        path = str(tmp_path / "ef.json")
        ef1 = get_experiential_filter(persistence_path=path)
        ef2 = get_experiential_filter()
        assert ef1 is ef2
