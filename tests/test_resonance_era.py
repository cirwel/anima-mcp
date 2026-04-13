"""Tests for resonance era — memory field core."""
import numpy as np
from anima_mcp.display.eras.resonance import (
    _deposit, _decay, _diffuse, _gradient_at, FIELD_SIZE, DECAY_RATE,
)


class TestDeposit:
    def test_deposit_adds_value_at_position(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.8)
        assert field[24, 24] > 0.0

    def test_deposit_accumulates(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.5)
        _deposit(field, pixel_x=120, pixel_y=120, value=0.3)
        assert abs(field[24, 24] - 0.8) < 0.01

    def test_deposit_clamps_to_canvas(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _deposit(field, pixel_x=0, pixel_y=0, value=0.5)
        assert field[0, 0] > 0.0
        _deposit(field, pixel_x=239, pixel_y=239, value=0.5)
        assert field[FIELD_SIZE - 1, FIELD_SIZE - 1] > 0.0


class TestDecay:
    def test_decay_reduces_field(self):
        field = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _decay(field)
        assert field[0, 0] < 1.0
        assert abs(field[0, 0] - DECAY_RATE) < 0.001

    def test_decay_preserves_zeros(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        _decay(field)
        assert field.sum() == 0.0


class TestDiffuse:
    def test_diffuse_spreads_point_source(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        diffused = _diffuse(field, sigma=1.0)
        assert diffused[24, 24] < 1.0
        assert diffused[24, 25] > 0.0
        assert diffused[25, 24] > 0.0

    def test_diffuse_conserves_energy_approximately(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        before_sum = field.sum()
        diffused = _diffuse(field, sigma=1.0)
        after_sum = diffused.sum()
        assert abs(before_sum - after_sum) < 0.1

    def test_higher_sigma_spreads_more(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        low_sigma = _diffuse(field.copy(), sigma=0.5)
        high_sigma = _diffuse(field.copy(), sigma=2.0)
        assert high_sigma[24, 24] < low_sigma[24, 24]


class TestGradient:
    def test_gradient_zero_on_uniform_field(self):
        field = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=np.float32) * 0.5
        gx, gy, mag = _gradient_at(field, 24, 24)
        assert abs(mag) < 0.01

    def test_gradient_nonzero_at_edge_of_deposit(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[24, 24] = 1.0
        gx, gy, mag = _gradient_at(field, 25, 24)
        assert mag > 0.0

    def test_gradient_at_boundary_does_not_crash(self):
        field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        field[0, 0] = 1.0
        _gradient_at(field, 0, 0)
        _gradient_at(field, FIELD_SIZE - 1, FIELD_SIZE - 1)


# ---------------------------------------------------------------------------
# ResonanceState + ResonanceEra tests
# ---------------------------------------------------------------------------
from anima_mcp.display.eras.resonance import ResonanceEra, ResonanceState


class TestResonanceState:
    def test_create_state_has_zeroed_field(self):
        era = ResonanceEra()
        state = era.create_state()
        assert state.field.shape == (FIELD_SIZE, FIELD_SIZE)
        assert state.field.sum() == 0.0

    def test_intentionality_range(self):
        state = ResonanceState()
        state.gesture_remaining = 0
        assert 0.0 <= state.intentionality() <= 1.0
        state.gesture_remaining = 20
        assert state.intentionality() > 0.1

    def test_gestures_vocabulary(self):
        state = ResonanceState()
        assert "sediment" in state.gestures()
        assert "flow" in state.gestures()
        assert "scratch" in state.gestures()


class TestChooseGesture:
    def test_low_gradient_selects_sediment(self):
        era = ResonanceEra()
        state = era.create_state()
        # Field all zeros -> zero gradient -> sediment
        era.choose_gesture(state, clarity=0.5, stability=0.5, presence=0.5, coherence=0.5)
        assert state.gesture == "sediment"
        assert state.gesture_remaining > 0

    def test_high_gradient_selects_scratch(self):
        era = ResonanceEra()
        state = era.create_state()
        # Sharp edge: one cell high, neighbors zero
        state.field[24, 24] = 5.0
        state._focus_cx = 25
        state._focus_cy = 24
        era.choose_gesture(state, clarity=0.5, stability=0.5, presence=0.5, coherence=0.5)
        assert state.gesture == "scratch"
