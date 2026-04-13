"""Tests for resonance era — memory field core."""
import math
import numpy as np
from anima_mcp.display.eras.resonance import (
    _deposit, _decay, _diffuse, _gradient_at, FIELD_SIZE, CELL_SIZE, DECAY_RATE,
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


# ---------------------------------------------------------------------------
# generate_color tests
# ---------------------------------------------------------------------------
import colorsys


class TestGenerateColor:
    def test_high_warmth_produces_warm_hue(self):
        era = ResonanceEra()
        state = era.create_state()
        color, category = era.generate_color(state, warmth=0.9, clarity=0.7, stability=0.7, presence=0.7)
        h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        hue_deg = h * 360
        assert hue_deg < 100 or hue_deg > 340, f"High warmth hue {hue_deg:.0f} should be warm"

    def test_low_warmth_produces_cool_hue(self):
        era = ResonanceEra()
        state = era.create_state()
        color, category = era.generate_color(state, warmth=0.1, clarity=0.7, stability=0.7, presence=0.7)
        h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        hue_deg = h * 360
        assert 150 < hue_deg < 270, f"Low warmth hue {hue_deg:.0f} should be cool"

    def test_high_clarity_produces_vivid_color(self):
        era = ResonanceEra()
        state = era.create_state()
        color, _ = era.generate_color(state, warmth=0.5, clarity=0.9, stability=0.5, presence=0.5)
        _, s, _ = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        assert s > 0.6, f"High clarity should produce saturation > 0.6, got {s:.2f}"

    def test_low_clarity_produces_washed_color(self):
        era = ResonanceEra()
        state = era.create_state()
        color, _ = era.generate_color(state, warmth=0.5, clarity=0.1, stability=0.5, presence=0.5)
        _, s, _ = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
        assert s < 0.6, f"Low clarity should produce saturation < 0.6, got {s:.2f}"

    def test_field_warmth_bias(self):
        era = ResonanceEra()
        state = era.create_state()
        color_cold, _ = era.generate_color(state, warmth=0.5, clarity=0.7, stability=0.7, presence=0.7)
        h_cold = colorsys.rgb_to_hsv(color_cold[0]/255, color_cold[1]/255, color_cold[2]/255)[0] * 360
        state.field[state._focus_cx, state._focus_cy] = 5.0
        color_hot, _ = era.generate_color(state, warmth=0.5, clarity=0.7, stability=0.7, presence=0.7)
        h_hot = colorsys.rgb_to_hsv(color_hot[0]/255, color_hot[1]/255, color_hot[2]/255)[0] * 360
        assert h_cold != h_hot, "Field warmth bias should shift hue"

    def test_returns_valid_rgb_and_category(self):
        era = ResonanceEra()
        state = era.create_state()
        color, cat = era.generate_color(state, 0.5, 0.5, 0.5, 0.5)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        assert cat in ("warm", "cool", "neutral")

    def test_light_regime_shifts(self):
        era = ResonanceEra()
        state = era.create_state()
        color_dim, _ = era.generate_color(state, 0.5, 0.5, 0.5, 0.5, light_regime="dim")
        state2 = era.create_state()
        color_dark, _ = era.generate_color(state2, 0.5, 0.5, 0.5, 0.5, light_regime="dark")
        assert color_dim != color_dark


# ---------------------------------------------------------------------------
# place_mark tests
# ---------------------------------------------------------------------------
import random


class FakeCanvas:
    def __init__(self):
        self.pixels = {}

    def draw_pixel(self, x, y, color):
        if 0 <= x < 240 and 0 <= y < 240:
            self.pixels[(x, y)] = color


class TestPlaceMark:
    def test_sediment_draws_pixels(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "sediment"
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert len(canvas.pixels) > 0

    def test_flow_draws_pixels(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "flow"
        state._grad_gx = 1.0
        state._grad_gy = 0.0
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert len(canvas.pixels) > 0

    def test_scratch_draws_elongated_mark(self):
        random.seed(42)
        era = ResonanceEra()
        state_s = era.create_state()
        state_s.gesture = "sediment"
        canvas_s = FakeCanvas()
        era.place_mark(state_s, canvas_s, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))

        random.seed(42)
        state_x = era.create_state()
        state_x.gesture = "scratch"
        state_x._grad_gx = 0.0
        state_x._grad_gy = 1.0
        canvas_x = FakeCanvas()
        era.place_mark(state_x, canvas_x, 120.0, 120.0, 0.0, 0.5, (255, 0, 0))
        assert len(canvas_x.pixels) >= len(canvas_s.pixels)

    def test_place_mark_deposits_to_field(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "sediment"
        canvas = FakeCanvas()
        assert state.field.sum() == 0.0
        era.place_mark(state, canvas, 120.0, 120.0, 0.0, 0.5, (255, 128, 0))
        assert state.field.sum() > 0.0

    def test_marks_stay_within_canvas(self):
        era = ResonanceEra()
        state = era.create_state()
        state.gesture = "scratch"
        state._grad_gx = 1.0
        state._grad_gy = 0.0
        canvas = FakeCanvas()
        era.place_mark(state, canvas, 5.0, 5.0, 0.0, 0.8, (255, 255, 255))
        for (x, y) in canvas.pixels:
            assert 0 <= x < 240 and 0 <= y < 240


# ---------------------------------------------------------------------------
# drift_focus tests
# ---------------------------------------------------------------------------


class TestDriftFocus:
    def test_drift_stays_in_bounds(self):
        random.seed(42)
        era = ResonanceEra()
        state = era.create_state()
        fx, fy, d = 200.0, 200.0, 0.5
        for _ in range(100):
            fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5)
            assert 0 <= fx <= 240 and 0 <= fy <= 240, f"Focus escaped: ({fx}, {fy})"

    def test_drift_with_gradient_influences_direction(self):
        random.seed(42)
        era = ResonanceEra()
        state = era.create_state()
        # Create gradient pointing right (positive x)
        for i in range(FIELD_SIZE):
            state.field[i, :] = float(i) / FIELD_SIZE
        fx, fy = 120.0, 120.0
        d = math.pi / 2  # Initially pointing up
        positions = []
        for _ in range(50):
            fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5, canvas=None)
            positions.append(fx)
        avg_x = sum(positions) / len(positions)
        assert avg_x > 120.0, f"Expected rightward drift, avg_x={avg_x:.1f}"

    def test_drift_updates_focus_cell(self):
        era = ResonanceEra()
        state = era.create_state()
        fx, fy, d = era.drift_focus(state, 60.0, 60.0, 0.0, 0.5, 0.5, 0.5, 0.5)
        expected_cx = min(max(0, int(fx) // CELL_SIZE), FIELD_SIZE - 1)
        expected_cy = min(max(0, int(fy) // CELL_SIZE), FIELD_SIZE - 1)
        assert state._focus_cx == expected_cx
        assert state._focus_cy == expected_cy
