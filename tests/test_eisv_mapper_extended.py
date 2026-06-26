"""Extended tests for eisv_mapper.py — covers uncovered branches.

Supplements test_eisv_mapper.py with tests for:
- compute_ethical_drift (all branches)
- compute_confidence (with/without prev_anima)
- generate_status_text (without readings/eisv)
- EISVMetrics.__repr__
"""

from datetime import datetime

from anima_mcp.eisv_mapper import (
    EISVMetrics,
    compute_ethical_drift,
    compute_confidence,
    generate_status_text,
)
from anima_mcp.anima import Anima
from anima_mcp.sensors.base import SensorReadings


def _anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5) -> Anima:
    readings = SensorReadings(
        timestamp=datetime.now(),
        cpu_temp_c=45.0,
        ambient_temp_c=22.0,
        humidity_pct=50.0,
        light_lux=300.0,
        cpu_percent=50.0,
        memory_percent=50.0,
        disk_percent=50.0,
    )
    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings,
    )


def _readings(**kwargs) -> SensorReadings:
    defaults = dict(
        timestamp=datetime.now(),
        cpu_temp_c=45.0,
        ambient_temp_c=22.0,
        humidity_pct=50.0,
        light_lux=300.0,
        cpu_percent=50.0,
        memory_percent=50.0,
        disk_percent=50.0,
    )
    defaults.update(kwargs)
    return SensorReadings(**defaults)


# ── EISVMetrics.__repr__ ──

class TestEISVMetricsRepr:
    def test_repr_format(self):
        m = EISVMetrics(energy=0.5, integrity=0.7, entropy=0.3, valence=0.1)
        r = repr(m)
        assert "EISV(" in r
        assert "E=0.50" in r
        assert "I=0.70" in r
        assert "S=0.30" in r
        assert "V=+0.10" in r  # signed valence


# ── compute_ethical_drift ──

class TestComputeEthicalDrift:
    def test_no_prev_returns_zeros(self):
        drift = compute_ethical_drift(_anima(), None)
        assert drift == [0.0, 0.0, 0.0]

    def test_identical_states_returns_zeros(self):
        a = _anima(0.5, 0.5, 0.5, 0.5)
        drift = compute_ethical_drift(a, a)
        assert all(abs(d) < 1e-9 for d in drift)

    def test_warmth_increase_positive_drift(self):
        prev = _anima(warmth=0.4)
        curr = _anima(warmth=0.6)
        drift = compute_ethical_drift(curr, prev)
        assert drift[0] > 0  # emotional drift positive

    def test_clarity_decrease_negative_drift(self):
        prev = _anima(clarity=0.7)
        curr = _anima(clarity=0.3)
        drift = compute_ethical_drift(curr, prev)
        assert drift[1] < 0  # epistemic drift negative

    def test_stability_change_behavioral_drift(self):
        prev = _anima(stability=0.8)
        curr = _anima(stability=0.4)
        drift = compute_ethical_drift(curr, prev)
        assert drift[2] < 0  # behavioral drift negative

    def test_drift_clamped(self):
        # Extreme change should be clamped to [-0.5, 0.5]
        prev = _anima(warmth=0.0, clarity=0.0, stability=0.0)
        curr = _anima(warmth=1.0, clarity=1.0, stability=1.0)
        drift = compute_ethical_drift(curr, prev)
        for d in drift:
            assert -0.5 <= d <= 0.5

    def test_temperature_amplification(self):
        prev = _anima(warmth=0.4, stability=0.4)
        curr = _anima(warmth=0.5, stability=0.5)
        prev_r = _readings(ambient_temp_c=20.0)
        curr_r = _readings(ambient_temp_c=26.0)  # 6°C change > 2°C threshold

        drift_no_env = compute_ethical_drift(curr, prev)
        drift_with_env = compute_ethical_drift(curr, prev, curr_r, prev_r)

        # Emotional drift (index 0) should be amplified
        assert abs(drift_with_env[0]) > abs(drift_no_env[0])
        # Epistemic drift (index 1) is NOT amplified by env
        assert abs(drift_with_env[1] - drift_no_env[1]) < 1e-9

    def test_light_amplification(self):
        prev = _anima(warmth=0.4)
        curr = _anima(warmth=0.5)
        # Large light change (>30% of world light)
        prev_r = _readings(light_lux=500.0)
        prev_r.led_brightness = 0.0
        curr_r = _readings(light_lux=100.0)
        curr_r.led_brightness = 0.0

        drift_no_env = compute_ethical_drift(curr, prev)
        drift_with_env = compute_ethical_drift(curr, prev, curr_r, prev_r)

        # Should be amplified
        assert abs(drift_with_env[0]) >= abs(drift_no_env[0])

    def test_small_temperature_change_no_amplification(self):
        prev = _anima(warmth=0.4)
        curr = _anima(warmth=0.5)
        prev_r = _readings(ambient_temp_c=22.0)
        curr_r = _readings(ambient_temp_c=22.5)  # 0.5°C < 2°C threshold

        drift_no_env = compute_ethical_drift(curr, prev)
        drift_with_env = compute_ethical_drift(curr, prev, curr_r, prev_r)

        # No amplification — should be same
        assert abs(drift_with_env[0] - drift_no_env[0]) < 1e-9

    def test_small_light_change_no_amplification(self):
        prev = _anima(warmth=0.4)
        curr = _anima(warmth=0.5)
        prev_r = _readings(light_lux=500.0)
        prev_r.led_brightness = 0.0
        curr_r = _readings(light_lux=480.0)  # 4% change < 30% threshold
        curr_r.led_brightness = 0.0

        drift_no_env = compute_ethical_drift(curr, prev)
        drift_with_env = compute_ethical_drift(curr, prev, curr_r, prev_r)

        assert abs(drift_with_env[0] - drift_no_env[0]) < 1e-9


# ── compute_confidence ──

class TestComputeConfidence:
    def test_no_prev(self):
        a = _anima(clarity=0.8, stability=0.7, presence=0.6)
        c = compute_confidence(a)
        expected = 0.8 * 0.5 + 0.7 * 0.3 + 0.6 * 0.2  # 0.4 + 0.21 + 0.12 = 0.73
        assert abs(c - expected) < 1e-9

    def test_high_clarity_high_confidence(self):
        c = compute_confidence(_anima(clarity=1.0, stability=1.0, presence=1.0))
        assert c == 1.0

    def test_low_clarity_low_confidence(self):
        c = compute_confidence(_anima(clarity=0.0, stability=0.0, presence=0.0))
        assert c == 0.05  # clamped minimum

    def test_rapid_transition_penalizes(self):
        prev = _anima(warmth=0.2, clarity=0.2, stability=0.2)
        curr = _anima(warmth=0.8, clarity=0.8, stability=0.8)
        c_with_prev = compute_confidence(curr, prev_anima=prev)
        c_without = compute_confidence(curr)
        assert c_with_prev < c_without

    def test_small_transition_no_penalty(self):
        prev = _anima(warmth=0.5, clarity=0.5, stability=0.5)
        curr = _anima(warmth=0.52, clarity=0.51, stability=0.50)
        c_with_prev = compute_confidence(curr, prev_anima=prev)
        c_without = compute_confidence(curr)
        # Total delta = 0.02 + 0.01 + 0.0 = 0.03, below 0.15 threshold
        assert abs(c_with_prev - c_without) < 1e-9

    def test_confidence_always_at_least_0_05(self):
        prev = _anima(warmth=0.0, clarity=0.0, stability=0.0)
        curr = _anima(warmth=1.0, clarity=0.0, stability=0.0)
        c = compute_confidence(curr, prev_anima=prev)
        assert c >= 0.05


# ── generate_status_text ──

class TestGenerateStatusText:
    def test_basic_no_readings_no_eisv(self):
        text = generate_status_text(_anima())
        assert "Anima state:" in text
        assert "Warmth:" in text
        assert "Clarity:" in text
        assert "Stability:" in text
        assert "Presence:" in text
        # Should NOT have Neural or EISV
        assert "Neural:" not in text
        assert "EISV:" not in text

    def test_with_readings_adds_neural(self):
        r = _readings()
        r.eeg_alpha_power = 0.6
        r.eeg_beta_power = 0.4
        r.eeg_gamma_power = 0.2
        text = generate_status_text(_anima(), readings=r)
        assert "Neural:" in text
        assert "Alpha=0.60" in text
        assert "Beta=0.40" in text
        assert "Gamma=0.20" in text

    def test_with_eisv_adds_eisv_line(self):
        eisv = EISVMetrics(energy=0.5, integrity=0.7, entropy=0.3, valence=0.1)
        text = generate_status_text(_anima(), eisv=eisv)
        assert "EISV:" in text
        assert "E=0.50" in text

    def test_readings_without_neural_no_neural_line(self):
        r = _readings()
        # No eeg_* attributes set
        text = generate_status_text(_anima(), readings=r)
        assert "Neural:" not in text

    def test_partial_neural(self):
        r = _readings()
        r.eeg_alpha_power = 0.5
        # beta and gamma not set
        text = generate_status_text(_anima(), readings=r)
        assert "Neural:" in text
        assert "Alpha=0.50" in text
        assert "Beta" not in text
