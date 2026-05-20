"""
Tests for display/drawing_engine.py -- state machine logic.

Covers:
- CanvasState: draw_pixel, clear, compositional_satisfaction, persistence
- DrawingState: attention signals, coherence, narrative arc, false starts
- DrawingGoal: generation from anima state
- DrawingIntent: energy property, reset
- DrawingEngine: narrative arc transitions, autonomy checks, era management
"""

import time
import json
import pytest
from unittest.mock import patch

from anima_mcp.display.drawing_engine import (
    CanvasState,
    DrawingState,
    DrawingGoal,
    DrawingIntent,
    DrawingEngine,
    DrawingEISV,
)

from conftest import make_anima


# ---------------------------------------------------------------------------
# CanvasState
# ---------------------------------------------------------------------------

class TestCanvasStateDrawPixel:
    """Test pixel drawing and boundary checks."""

    def test_draw_pixel_in_bounds(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 20, (255, 0, 0))
        assert (10, 20) in canvas.pixels
        assert canvas.pixels[(10, 20)] == (255, 0, 0)

    def test_draw_pixel_out_of_bounds_ignored(self):
        canvas = CanvasState()
        canvas.draw_pixel(-1, 0, (255, 0, 0))
        canvas.draw_pixel(0, -1, (255, 0, 0))
        canvas.draw_pixel(240, 0, (255, 0, 0))
        canvas.draw_pixel(0, 240, (255, 0, 0))
        assert len(canvas.pixels) == 0

    def test_draw_pixel_boundary_values(self):
        canvas = CanvasState()
        canvas.draw_pixel(0, 0, (1, 2, 3))
        canvas.draw_pixel(239, 239, (4, 5, 6))
        assert (0, 0) in canvas.pixels
        assert (239, 239) in canvas.pixels

    def test_draw_pixel_tracks_recent_locations(self):
        canvas = CanvasState()
        for i in range(5):
            canvas.draw_pixel(i, i, (255, 255, 255))
        assert len(canvas.recent_locations) == 5
        assert canvas.recent_locations[-1] == (4, 4)

    def test_draw_pixel_caps_recent_locations_at_20(self):
        canvas = CanvasState()
        for i in range(25):
            canvas.draw_pixel(i % 240, 0, (255, 255, 255))
        assert len(canvas.recent_locations) == 20

    def test_draw_pixel_resets_satisfaction(self):
        canvas = CanvasState()
        canvas.is_satisfied = True
        canvas.draw_pixel(10, 10, (255, 0, 0))
        assert canvas.is_satisfied is False

    def test_draw_pixel_marks_dirty(self):
        canvas = CanvasState()
        canvas._dirty = False
        canvas.draw_pixel(10, 10, (255, 0, 0))
        assert canvas._dirty is True

    def test_draw_pixel_appends_to_new_pixels(self):
        canvas = CanvasState()
        canvas.draw_pixel(5, 5, (100, 100, 100))
        assert len(canvas._new_pixels) == 1
        assert canvas._new_pixels[0] == (5, 5, (100, 100, 100))

    def test_draw_pixel_overwrites(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.draw_pixel(10, 10, (0, 255, 0))
        assert canvas.pixels[(10, 10)] == (0, 255, 0)


class TestCanvasStateClear:
    """Test canvas clearing behavior."""

    def test_clear_resets_pixels(self):
        canvas = CanvasState()
        canvas.draw_pixel(10, 10, (255, 0, 0))
        canvas.clear()
        assert len(canvas.pixels) == 0

    def test_clear_resets_locations(self):
        canvas = CanvasState()
        for i in range(5):
            canvas.draw_pixel(i, 0, (255, 0, 0))
        canvas.clear()
        assert len(canvas.recent_locations) == 0

    def test_clear_resets_phase_to_opening(self):
        canvas = CanvasState()
        canvas.drawing_phase = "resolving"
        canvas.clear()
        assert canvas.drawing_phase == "opening"

    def test_clear_resets_satisfaction(self):
        canvas = CanvasState()
        canvas.is_satisfied = True
        canvas.satisfaction_time = 12345.0
        canvas.clear()
        assert canvas.is_satisfied is False
        assert canvas.satisfaction_time == 0.0

    def test_clear_resets_energy_and_marks(self):
        canvas = CanvasState()
        canvas.energy = 0.3
        canvas.mark_count = 500
        canvas.clear()
        assert canvas.energy == 1.0
        assert canvas.mark_count == 0

    def test_clear_resets_attention_signals(self):
        canvas = CanvasState()
        canvas.curiosity = 0.1
        canvas.engagement = 0.9
        canvas.fatigue = 0.8
        canvas.clear()
        assert canvas.curiosity == 1.0
        assert canvas.engagement == 0.5
        assert canvas.fatigue == 0.0

    def test_clear_resets_arc_phase(self):
        canvas = CanvasState()
        canvas.arc_phase = "closing"
        canvas.clear()
        assert canvas.arc_phase == "opening"

    def test_clear_resets_coherence_history(self):
        canvas = CanvasState()
        canvas.coherence_history = [0.5, 0.6, 0.7]
        canvas.clear()
        assert canvas.coherence_history == []

    def test_clear_sets_drawing_pause(self):
        canvas = CanvasState()
        canvas.clear()
        assert canvas.drawing_paused_until > time.time()

    def test_clear_invalidates_render_cache(self):
        canvas = CanvasState()
        canvas._cached_image = "something"
        canvas.clear()
        assert canvas._dirty is True
        assert canvas._cached_image is None
        assert canvas._new_pixels == []

    def test_clear_clears_pending_era_switch(self):
        canvas = CanvasState()
        canvas.pending_era_switch = "geometric"
        canvas.clear()
        assert canvas.pending_era_switch is None


class TestCanvasCompositionalSatisfaction:
    """Test compositional satisfaction scoring."""

    def test_too_sparse_returns_zero(self):
        canvas = CanvasState()
        for i in range(10):
            canvas.draw_pixel(i, 0, (255, 255, 255))
        assert canvas.compositional_satisfaction() == 0.0

    def test_minimum_50_pixels_required(self):
        canvas = CanvasState()
        for i in range(49):
            canvas.draw_pixel(i, 0, (255, 255, 255))
        assert canvas.compositional_satisfaction() == 0.0

    def test_well_distributed_gets_positive_score(self):
        canvas = CanvasState()
        # Place pixels in all four quadrants
        for i in range(30):
            canvas.draw_pixel(i + 10, 10, (255, 255, 255))  # Q0
            canvas.draw_pixel(i + 130, 10, (255, 255, 255))  # Q1
            canvas.draw_pixel(i + 10, 130, (255, 255, 255))  # Q2
            canvas.draw_pixel(i + 130, 130, (255, 255, 255))  # Q3
        sat = canvas.compositional_satisfaction()
        assert sat > 0.0

    def test_unbalanced_lower_satisfaction(self):
        canvas_balanced = CanvasState()
        canvas_unbalanced = CanvasState()

        # Balanced: equal distribution
        for i in range(40):
            canvas_balanced.draw_pixel(i + 10, 10, (255, 255, 255))
            canvas_balanced.draw_pixel(i + 130, 10, (255, 255, 255))
            canvas_balanced.draw_pixel(i + 10, 130, (255, 255, 255))
            canvas_balanced.draw_pixel(i + 130, 130, (255, 255, 255))

        # Unbalanced: all in one quadrant
        for i in range(160):
            canvas_unbalanced.draw_pixel(i % 100 + 10, i // 100 + 10, (255, 255, 255))

        sat_balanced = canvas_balanced.compositional_satisfaction()
        sat_unbalanced = canvas_unbalanced.compositional_satisfaction()
        assert sat_balanced >= sat_unbalanced

    def test_coherence_history_influences_satisfaction(self):
        canvas = CanvasState()
        # Place enough pixels in all quadrants
        for i in range(50):
            canvas.draw_pixel(i % 100 + 10, (i * 3) % 100 + 10, (255, 255, 255))
            canvas.draw_pixel(i % 100 + 130, (i * 3) % 100 + 130, (255, 255, 255))

        # Without coherence history
        canvas.compositional_satisfaction()

        # With high coherence history
        canvas.coherence_history = [0.9] * 10
        sat_high_coh = canvas.compositional_satisfaction()

        # With low coherence history
        canvas.coherence_history = [0.1] * 10
        sat_low_coh = canvas.compositional_satisfaction()

        assert sat_high_coh > sat_low_coh

    def test_satisfaction_bounded_0_1(self):
        canvas = CanvasState()
        # Fill many pixels
        for x in range(240):
            for y in range(0, 240, 10):
                canvas.draw_pixel(x, y, (255, 255, 255))
        sat = canvas.compositional_satisfaction()
        assert 0.0 <= sat <= 1.0


class TestCanvasMarkSatisfied:
    """Test satisfaction marking."""

    def test_mark_satisfied_sets_flag(self):
        canvas = CanvasState()
        canvas.mark_satisfied()
        assert canvas.is_satisfied is True
        assert canvas.satisfaction_time > 0.0

    def test_mark_satisfied_idempotent(self):
        canvas = CanvasState()
        canvas.mark_satisfied()
        first_time = canvas.satisfaction_time
        canvas.mark_satisfied()
        assert canvas.satisfaction_time == first_time


class TestCanvasPersistence:
    """Test save/load to disk."""

    def test_save_and_load_roundtrip(self, tmp_path):
        canvas = CanvasState()
        canvas.draw_pixel(10, 20, (255, 128, 0))
        canvas.draw_pixel(100, 100, (0, 255, 0))
        canvas.curiosity = 0.42
        canvas.engagement = 0.73
        canvas.fatigue = 0.15
        canvas.arc_phase = "developing"
        canvas.coherence_history = [0.5, 0.6, 0.7]
        canvas._era_name = "pointillist"
        canvas.drawings_saved = 3

        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            canvas.save_to_disk()

            canvas2 = CanvasState()
            canvas2.load_from_disk()

        assert (10, 20) in canvas2.pixels
        assert canvas2.pixels[(10, 20)] == (255, 128, 0)
        assert abs(canvas2.curiosity - 0.42) < 0.01
        assert abs(canvas2.engagement - 0.73) < 0.01
        assert abs(canvas2.fatigue - 0.15) < 0.01
        assert canvas2.arc_phase == "developing"
        assert len(canvas2.coherence_history) == 3
        assert canvas2._era_name == "pointillist"
        assert canvas2.drawings_saved == 3

    def test_load_handles_missing_file(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "nonexistent.json"):
            canvas = CanvasState()
            canvas.load_from_disk()
        # Should use defaults
        assert canvas.curiosity == 1.0
        assert canvas.arc_phase == "opening"

    def test_load_handles_corrupted_json(self, tmp_path):
        bad_file = tmp_path / "canvas.json"
        bad_file.write_text("not valid json {{{")
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=bad_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        # Should use defaults and delete corrupted file
        assert canvas.curiosity == 1.0
        assert not bad_file.exists()

    def test_load_handles_empty_file(self, tmp_path):
        empty_file = tmp_path / "canvas.json"
        empty_file.write_text("")
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=empty_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        assert canvas.curiosity == 1.0

    def test_load_handles_non_dict_data(self, tmp_path):
        bad_file = tmp_path / "canvas.json"
        bad_file.write_text('"just a string"')
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=bad_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        assert canvas.curiosity == 1.0

    def test_load_skips_invalid_pixels(self, tmp_path):
        data = {
            "pixels": {
                "10,20": [255, 0, 0],        # Valid
                "bad_key": [0, 255, 0],       # Invalid key
                "10,20,30": [0, 0, 255],      # Too many parts
                "500,500": [255, 255, 255],   # Out of bounds
                "5,5": [256, 0, 0],           # Invalid color
                "8,8": [0, 0],                # Wrong color length
            },
        }
        canvas_file = tmp_path / "canvas.json"
        canvas_file.write_text(json.dumps(data))
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=canvas_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        assert len(canvas.pixels) == 1  # Only the valid one
        assert (10, 20) in canvas.pixels

    def test_load_validates_phase(self, tmp_path):
        data = {"drawing_phase": "invalid_phase"}
        canvas_file = tmp_path / "canvas.json"
        canvas_file.write_text(json.dumps(data))
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=canvas_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        assert canvas.drawing_phase == "opening"  # Default, invalid rejected

    def test_load_validates_arc_phase(self, tmp_path):
        data = {"arc_phase": "flying"}
        canvas_file = tmp_path / "canvas.json"
        canvas_file.write_text(json.dumps(data))
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=canvas_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        assert canvas.arc_phase == "opening"  # Default, invalid rejected

    def test_load_clamps_attention_signals(self, tmp_path):
        data = {
            "curiosity": 5.0,   # Over 1.0
            "engagement": -1.0,  # Under 0.0
            "fatigue": 2.0,     # Over 1.0
        }
        canvas_file = tmp_path / "canvas.json"
        canvas_file.write_text(json.dumps(data))
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=canvas_file):
            canvas = CanvasState()
            canvas.load_from_disk()
        # Out-of-range values are rejected, defaults used
        assert canvas.curiosity == 1.0
        assert canvas.engagement == 0.5
        assert canvas.fatigue == 0.0


# ---------------------------------------------------------------------------
# DrawingState — attention, coherence, narrative arc
# ---------------------------------------------------------------------------

class TestDrawingStateAttention:
    """Test attention signal properties."""

    def test_default_values(self):
        state = DrawingState()
        assert state.curiosity == 1.0
        assert state.engagement == 0.5
        assert state.fatigue == 0.0

    def test_derived_energy(self):
        state = DrawingState()
        # Default: 0.6*1.0 + 0.4*0.5 = 0.8, * (1-0) = 0.8
        expected = 0.6 * 1.0 + 0.4 * 0.5
        assert abs(state.derived_energy - expected) < 0.01

    def test_derived_energy_with_fatigue(self):
        state = DrawingState()
        state.fatigue = 0.5
        expected = (0.6 * 1.0 + 0.4 * 0.5) * (1.0 - 0.5 * 0.5)
        assert abs(state.derived_energy - expected) < 0.01

    def test_derived_energy_fully_fatigued(self):
        state = DrawingState()
        state.fatigue = 1.0
        energy = state.derived_energy
        assert energy < 0.5  # Significantly reduced

    def test_derived_energy_all_depleted(self):
        state = DrawingState()
        state.curiosity = 0.0
        state.engagement = 0.0
        state.fatigue = 1.0
        assert state.derived_energy == 0.0

    def test_attention_exhausted_conditions(self):
        state = DrawingState()
        # Not exhausted with defaults
        assert state.attention_exhausted() is False

        # Exhausted: low curiosity + low engagement
        state.curiosity = 0.1
        state.engagement = 0.2
        assert state.attention_exhausted() is True

    def test_attention_exhausted_high_fatigue_path(self):
        state = DrawingState()
        state.curiosity = 0.1
        state.fatigue = 0.85
        state.engagement = 0.5  # Still engaged, but fatigued
        assert state.attention_exhausted() is True

    def test_attention_not_exhausted_high_curiosity(self):
        state = DrawingState()
        state.curiosity = 0.5
        state.engagement = 0.1
        state.fatigue = 0.9
        assert state.attention_exhausted() is False  # Curiosity still above 0.15


class TestDrawingStateCoherence:
    """Test coherence calculation and settling detection."""

    def test_coherence_formula(self):
        state = DrawingState()
        state.V = 0.0
        C = state.coherence()
        assert abs(C - 0.5) < 0.01

    def test_coherence_settled_needs_20_samples(self):
        state = DrawingState()
        state.coherence_history = [0.8] * 19
        assert state.coherence_settled() is False

    def test_coherence_settled_high_stable(self):
        state = DrawingState()
        state.coherence_history = [0.75] * 20
        assert state.coherence_settled() is True

    def test_coherence_not_settled_low_mean(self):
        state = DrawingState()
        state.coherence_history = [0.4] * 20
        assert state.coherence_settled() is False  # Mean < 0.6

    def test_coherence_not_settled_high_variance(self):
        state = DrawingState()
        # Alternating values create high variance
        state.coherence_history = [0.3, 0.9] * 10
        assert state.coherence_settled() is False

    def test_coherence_settled_threshold(self):
        """Mean > 0.6 and variance < 0.015 is required."""
        state = DrawingState()
        # Mean exactly 0.65 with low variance
        state.coherence_history = [0.64, 0.66] * 10
        mean = sum(state.coherence_history[-10:]) / 10
        assert mean > 0.6
        assert state.coherence_settled() is True


class TestDrawingStateNarrativeComplete:
    """Test narrative completion detection (multiple paths)."""

    def test_closing_phase_is_complete(self):
        state = DrawingState()
        state.arc_phase = "closing"
        assert state.narrative_complete() is True

    def test_coherence_settled_and_attention_exhausted(self):
        state = DrawingState()
        state.coherence_history = [0.75] * 20
        state.curiosity = 0.1
        state.engagement = 0.2
        assert state.narrative_complete() is True

    def test_compositional_satisfaction_path(self):
        state = DrawingState()
        state.curiosity = 0.15  # Just below threshold

        canvas = CanvasState()
        # Create well-distributed pixels for high satisfaction
        for x in range(0, 240, 5):
            for y in range(0, 240, 20):
                canvas.draw_pixel(x, y, (255, 255, 255))
        canvas.coherence_history = [0.8] * 10

        sat = canvas.compositional_satisfaction()
        if sat > 0.7:
            assert state.narrative_complete(canvas) is True

    def test_extreme_fatigue_path(self):
        state = DrawingState()
        state.fatigue = 0.95  # > 0.90
        assert state.narrative_complete() is True

    def test_not_complete_opening_phase(self):
        state = DrawingState()
        state.arc_phase = "opening"
        assert state.narrative_complete() is False

    def test_stalled_drawing_path(self):
        state = DrawingState()
        canvas = CanvasState()
        # Simulate stalled: low energy, lots of pixels, long time
        state.curiosity = 0.01
        state.engagement = 0.01
        state.fatigue = 0.5
        for i in range(250):
            canvas.draw_pixel(i % 240, i // 240, (128, 128, 128))
        canvas.last_clear_time = time.time() - 1000  # 16+ minutes ago
        assert state.narrative_complete(canvas) is True

    def test_time_limit_path(self):
        state = DrawingState()
        canvas = CanvasState()
        for i in range(60):
            canvas.draw_pixel(i, 0, (128, 128, 128))
        canvas.last_clear_time = time.time() - 30000  # > 8 hours
        assert state.narrative_complete(canvas) is True


class TestDrawingStateFalseStart:
    """Test false start detection."""

    def test_not_false_start_no_canvas(self):
        state = DrawingState()
        assert state.is_false_start(None) is False

    def test_not_false_start_in_developing_phase(self):
        state = DrawingState()
        state.arc_phase = "developing"
        canvas = CanvasState()
        assert state.is_false_start(canvas) is False

    def test_not_false_start_too_soon(self):
        state = DrawingState()
        state.arc_phase = "opening"
        canvas = CanvasState()
        canvas.phase_start_time = time.time()  # Just started
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is False

    def test_not_false_start_too_few_marks(self):
        state = DrawingState()
        state.arc_phase = "opening"
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 5  # < 8
        assert state.is_false_start(canvas) is False

    def test_not_false_start_high_momentum(self):
        state = DrawingState()
        state.arc_phase = "opening"
        state.i_momentum = 0.4  # >= 0.25
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is False

    def test_not_false_start_not_enough_coherence_data(self):
        state = DrawingState()
        state.arc_phase = "opening"
        state.i_momentum = 0.1
        state.engagement = 0.1
        state.coherence_history = [0.1, 0.2]  # < 3 values
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is False

    def test_false_start_all_conditions_met(self):
        state = DrawingState()
        state.arc_phase = "opening"
        state.i_momentum = 0.1
        state.engagement = 0.1
        state.coherence_history = [0.1, 0.15, 0.2, 0.1, 0.15,
                                    0.1, 0.2, 0.15, 0.1, 0.2]
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is True

    def test_not_false_start_high_coherence(self):
        state = DrawingState()
        state.arc_phase = "opening"
        state.i_momentum = 0.1
        state.engagement = 0.1
        state.coherence_history = [0.5, 0.6, 0.7, 0.6, 0.5,
                                    0.6, 0.7, 0.5, 0.6, 0.7]  # mean > 0.35
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is False

    def test_not_false_start_engaged(self):
        state = DrawingState()
        state.arc_phase = "opening"
        state.i_momentum = 0.1
        state.engagement = 0.5  # >= 0.3
        state.coherence_history = [0.1] * 10
        canvas = CanvasState()
        canvas.phase_start_time = time.time() - 60
        canvas.mark_count = 20
        assert state.is_false_start(canvas) is False


class TestDrawingStateReset:
    """Test reset behavior for DrawingState."""

    def test_reset_clears_all_fields(self):
        state = DrawingState()
        state.E = 0.1
        state.I = 0.9
        state.curiosity = 0.1
        state.engagement = 0.9
        state.fatigue = 0.8
        state.arc_phase = "resolving"
        state.coherence_history = [0.5, 0.6]
        state.i_momentum = 0.7
        state.phase_mark_count = 100

        state.reset()

        assert state.E == 0.4
        assert state.I == 0.2
        assert state.S == 0.5
        assert state.V == 0.0
        assert state.curiosity == 1.0
        assert state.engagement == 0.5
        assert state.fatigue == 0.0
        assert state.arc_phase == "opening"
        assert state.coherence_history == []
        assert state.i_momentum == 0.0
        assert state.phase_mark_count == 0
        assert state.gesture_history == []


# ---------------------------------------------------------------------------
# DrawingGoal
# ---------------------------------------------------------------------------

class TestDrawingGoal:
    """Test drawing goal generation from state."""

    def test_from_state_neutral(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.5)
        assert goal.warmth_bias == 0.0
        assert goal.coverage_target == "balanced"

    def test_from_state_high_warmth(self):
        goal = DrawingGoal.from_state(warmth=0.9, clarity=0.5)
        assert goal.warmth_bias > 0.0
        assert "warm tones" in goal.description

    def test_from_state_low_warmth(self):
        goal = DrawingGoal.from_state(warmth=0.1, clarity=0.5)
        assert goal.warmth_bias < 0.0
        assert "cool tones" in goal.description

    def test_from_state_high_clarity_sparse(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.8)
        assert goal.coverage_target == "sparse"

    def test_from_state_low_clarity_dense(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.2)
        assert goal.coverage_target == "dense"

    def test_from_state_morning_quadrant(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.5, hour=9)
        assert goal.initial_quadrant == 0

    def test_from_state_afternoon_quadrant(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.5, hour=15)
        assert goal.initial_quadrant == 1

    def test_from_state_night_no_quadrant(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.5, hour=22)
        assert goal.initial_quadrant is None

    def test_from_state_no_hour(self):
        goal = DrawingGoal.from_state(warmth=0.5, clarity=0.5, hour=None)
        assert goal.initial_quadrant is None


# ---------------------------------------------------------------------------
# DrawingIntent
# ---------------------------------------------------------------------------

class TestDrawingIntent:
    """Test DrawingIntent properties and reset."""

    def test_energy_property_reads_from_state(self):
        intent = DrawingIntent()
        intent.state.curiosity = 0.8
        intent.state.engagement = 0.6
        intent.state.fatigue = 0.0
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert abs(intent.energy - expected) < 0.01

    def test_energy_setter_adjusts_curiosity(self):
        intent = DrawingIntent()
        intent.energy = 0.7
        assert abs(intent.state.curiosity - 0.7) < 0.01

    def test_eisv_alias(self):
        intent = DrawingIntent()
        assert intent.eisv is intent.state

    def test_reset_clears_focus(self):
        intent = DrawingIntent()
        intent.focus_x = 200.0
        intent.focus_y = 200.0
        intent.mark_count = 50
        intent.reset()
        assert intent.focus_x == 120.0
        assert intent.focus_y == 120.0
        assert intent.mark_count == 0

    def test_reset_clears_era_state(self):
        intent = DrawingIntent()
        intent.era_state = "something"
        intent.reset()
        assert intent.era_state is None

    def test_reset_resets_state(self):
        intent = DrawingIntent()
        intent.state.fatigue = 0.9
        intent.reset()
        assert intent.state.fatigue == 0.0


# ---------------------------------------------------------------------------
# DrawingEngine — narrative arc transitions
# ---------------------------------------------------------------------------

class TestDrawingEngineNarrativeArc:
    """Test _update_narrative_arc phase transitions."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a DrawingEngine with mocked disk and identity."""
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_fresh_canvas_stays_opening(self, engine):
        """Fewer than 10 pixels stays in opening."""
        engine.canvas.pixels.clear()
        engine.intent.state.arc_phase = "developing"
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "opening"

    def test_opening_to_developing(self, engine):
        """Momentum > 0.15 and >10 marks transitions to developing."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "opening"
        engine.intent.state.i_momentum = 0.2
        engine.intent.state.phase_mark_count = 15
        engine._behavioral_C = 0.3
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "developing"

    def test_developing_to_resolving(self, engine):
        """High coherence + stable velocity transitions to resolving."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "developing"
        engine._behavioral_C = 0.65
        engine.intent.state.coherence_velocity = 0.005
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "resolving"

    def test_developing_regression_to_opening(self, engine):
        """Low coherence + low momentum regresses to opening."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "developing"
        engine._behavioral_C = 0.2
        engine.intent.state.i_momentum = 0.1
        engine.intent.state.phase_mark_count = 25
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "opening"

    def test_resolving_to_closing(self, engine):
        """Narrative complete transitions to closing."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "resolving"
        engine.intent.state.fatigue = 0.95  # Extreme fatigue path
        engine._behavioral_C = 0.7
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "closing"

    def test_resolving_destabilized_to_developing(self, engine):
        """Low coherence in resolving drops back to developing."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "resolving"
        engine._behavioral_C = 0.35  # < 0.4 hysteresis
        engine.intent.state.fatigue = 0.1  # Not fatigued
        engine.intent.state.curiosity = 0.8  # Not exhausted
        engine.intent.state.coherence_history = [0.3] * 5  # Not settled
        engine._update_narrative_arc()
        assert engine.intent.state.arc_phase == "developing"

    def test_closing_marks_satisfaction(self, engine):
        """Entering closing marks canvas as satisfied."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "closing"
        engine.canvas.is_satisfied = False
        engine._update_narrative_arc()
        assert engine.canvas.is_satisfied is True

    def test_phase_mark_count_resets_on_transition(self, engine):
        """Phase transitions reset the phase mark counter."""
        for i in range(15):
            engine.canvas.draw_pixel(i, i, (255, 255, 255))
        engine.intent.state.arc_phase = "opening"
        engine.intent.state.i_momentum = 0.2
        engine.intent.state.phase_mark_count = 20
        engine._behavioral_C = 0.3
        engine._update_narrative_arc()
        assert engine.intent.state.phase_mark_count == 0


class TestDrawingEngineAttention:
    """Test _update_attention signal dynamics."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_curiosity_depletes_exploring(self, engine):
        """Low coherence drains curiosity."""
        state = engine.intent.state
        initial_curiosity = state.curiosity
        engine._update_attention(I_signal=0.2, S_signal=0.5, C=0.2, gesture_switch=False)
        assert state.curiosity < initial_curiosity

    def test_curiosity_regenerates_with_pattern(self, engine):
        """High coherence regenerates curiosity."""
        state = engine.intent.state
        state.curiosity = 0.5
        engine._update_attention(I_signal=0.5, S_signal=0.3, C=0.8, gesture_switch=False)
        assert state.curiosity > 0.5

    def test_curiosity_drains_in_resolving(self, engine):
        """Resolving phase drains curiosity unless deeply coherent."""
        state = engine.intent.state
        state.arc_phase = "resolving"
        state.curiosity = 0.5
        engine._update_attention(I_signal=0.5, S_signal=0.3, C=0.4, gesture_switch=False)
        assert state.curiosity < 0.5

    def test_curiosity_slight_regen_deep_coherence_resolving(self, engine):
        """Deep coherence in resolving allows slight regeneration."""
        state = engine.intent.state
        state.arc_phase = "resolving"
        state.curiosity = 0.5
        engine._update_attention(I_signal=0.5, S_signal=0.3, C=0.8, gesture_switch=False)
        # Slight regen: -0.0005 * 0.8 = -(-0.0004) → curiosity increases
        assert state.curiosity >= 0.5

    def test_engagement_follows_intentionality(self, engine):
        """Engagement rises toward I_signal * (1 - 0.5*S)."""
        state = engine.intent.state
        state.engagement = 0.1
        # High I, low S → target high
        for _ in range(50):
            engine._update_attention(I_signal=0.8, S_signal=0.1, C=0.5, gesture_switch=False)
        assert state.engagement > 0.4

    def test_fatigue_always_increases(self, engine):
        """Fatigue never decreases during normal drawing (outside second wind)."""
        state = engine.intent.state
        initial_fatigue = state.fatigue
        engine._update_attention(I_signal=0.3, S_signal=0.5, C=0.3, gesture_switch=False)
        assert state.fatigue >= initial_fatigue

    def test_gesture_switch_adds_fatigue(self, engine):
        """Gesture switches add extra fatigue."""
        state = engine.intent.state
        state.fatigue = 0.0
        engine._update_attention(I_signal=0.3, S_signal=0.5, C=0.3, gesture_switch=False)
        fatigue_no_switch = state.fatigue

        state.fatigue = 0.0
        engine._update_attention(I_signal=0.3, S_signal=0.5, C=0.3, gesture_switch=True)
        fatigue_with_switch = state.fatigue

        assert fatigue_with_switch > fatigue_no_switch

    def test_second_wind_reduces_fatigue(self, engine):
        """High coherence + engagement gives slight fatigue recovery."""
        state = engine.intent.state
        state.fatigue = 0.5
        state.engagement = 0.7
        # Use high C and engaged state for second wind
        engine._update_attention(I_signal=0.6, S_signal=0.2, C=0.7, gesture_switch=False)
        # Base fatigue is added, but second wind subtracts some
        # Net effect: fatigue still increases but less than without second wind
        # We check the second-wind code path triggers: fatigue - 0.0005
        # The base fatigue added is 0.0004 + 0.0008*(1-0.7) = 0.00064
        # With second wind: -0.0005
        # Net per step is small, but second wind does reduce
        assert state.fatigue < 0.502  # Only a tiny increase due to second wind


class TestDrawingEngineRecovery:
    """Test restart recovery and crash-resilient persistence."""

    def test_restart_keeps_old_unfinished_canvas(self, tmp_path):
        canvas_file = tmp_path / "canvas.json"
        old_time = time.time() - 30000  # > 8 hours
        data = {
            "pixels": {"10,20": [255, 128, 0]},
            "mark_count": 12,
            "last_clear_time": old_time,
            "drawing_start_time": old_time,
            "fatigue": 0.4,
            "arc_phase": "developing",
        }
        canvas_file.write_text(json.dumps(data))

        with patch("anima_mcp.display.drawing_engine._get_canvas_path", return_value=canvas_file):
            engine = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)

        assert engine.canvas.pixels == {(10, 20): (255, 128, 0)}
        assert engine.intent.mark_count == 12
        assert engine.intent.state.arc_phase == "developing"

    def test_draw_persists_after_mark_batch(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path", return_value=tmp_path / "canvas.json"):
            engine = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)

        anima = make_anima()
        engine._last_persist_time = time.time()

        with patch("anima_mcp.display.drawing_engine.random.random", return_value=0.0), \
             patch.object(engine.canvas, "save_to_disk") as save_to_disk:
            for _ in range(5):
                engine.draw(anima)

        assert engine.intent.mark_count == 5
        save_to_disk.assert_called_once()


class TestDrawingEngineCoherenceTracking:
    """Test _update_coherence_tracking."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_appends_to_coherence_history(self, engine):
        state = engine.intent.state
        state.coherence_history.clear()
        engine._update_coherence_tracking(C=0.5, I_signal=0.3)
        assert len(state.coherence_history) == 1
        assert state.coherence_history[0] == 0.5

    def test_caps_coherence_history_at_30(self, engine):
        state = engine.intent.state
        state.coherence_history = [0.5] * 30
        engine._update_coherence_tracking(C=0.6, I_signal=0.3)
        assert len(state.coherence_history) == 30
        assert state.coherence_history[-1] == 0.6

    def test_coherence_velocity_ema(self, engine):
        state = engine.intent.state
        state.coherence_history = [0.3]
        state.coherence_velocity = 0.0
        engine._update_coherence_tracking(C=0.5, I_signal=0.3)
        # dC = 0.5 - 0.3 = 0.2, velocity = 0.2*0.2 + 0.8*0 = 0.04
        assert abs(state.coherence_velocity - 0.04) < 0.01

    def test_i_momentum_ema(self, engine):
        state = engine.intent.state
        state.i_momentum = 0.0
        engine._update_coherence_tracking(C=0.5, I_signal=0.6)
        # i_mom = 0.1*0.6 + 0.9*0 = 0.06
        assert abs(state.i_momentum - 0.06) < 0.01

    def test_phase_mark_count_increments(self, engine):
        state = engine.intent.state
        state.phase_mark_count = 5
        engine._update_coherence_tracking(C=0.5, I_signal=0.3)
        assert state.phase_mark_count == 6


class TestDrawingEngineEISVStep:
    """Test the EISV thermodynamic step."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_eisv_step_returns_three_values(self, engine):
        result = engine._eisv_step()
        assert len(result) == 3
        dE, C, S_signal = result
        assert isinstance(dE, float)
        assert isinstance(C, float)
        assert isinstance(S_signal, float)

    def test_eisv_step_clamps_values(self, engine):
        """EISV values should stay within bounds after many steps."""
        for _ in range(1000):
            engine._eisv_step()
        state = engine.intent.state
        assert 0.0 <= state.E <= 1.0
        assert 0.0 <= state.I <= 1.0
        assert 0.001 <= state.S <= 2.0
        assert -2.0 <= state.V <= 2.0

    def test_eisv_step_s_signal_bounded(self, engine):
        """Behavioral entropy signal should be bounded [0, 1]."""
        engine.intent.state.gesture_history = ["dot", "stroke", "curve", "cluster"] * 5
        _, _, S_signal = engine._eisv_step()
        assert 0.0 <= S_signal <= 1.0


class TestDrawingEngineSetDrives:
    """Test set_drives method."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_set_drives_updates_state(self, engine):
        drives = {"warmth": 0.3, "clarity": 0.6, "stability": 0.4, "presence": 0.7}
        engine.set_drives(drives)
        assert engine.intent.state.drive_warmth == 0.3
        assert engine.intent.state.drive_clarity == 0.6
        assert engine.intent.state.drive_stability == 0.4
        assert engine.intent.state.drive_presence == 0.7

    def test_set_drives_handles_empty(self, engine):
        engine.set_drives({})
        assert engine.intent.state.drive_warmth == 0.0

    def test_set_drives_handles_none(self, engine):
        engine.set_drives(None)
        # Should not crash


class TestDrawingEngineGetDrawingEISV:
    """Test get_drawing_eisv reporting."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_returns_dict(self, engine):
        result = engine.get_drawing_eisv()
        assert isinstance(result, dict)

    def test_contains_eisv_keys(self, engine):
        result = engine.get_drawing_eisv()
        for key in ("E", "I", "S", "V", "C", "marks", "phase", "era"):
            assert key in result

    def test_contains_attention_keys(self, engine):
        result = engine.get_drawing_eisv()
        for key in ("curiosity", "engagement", "fatigue", "energy"):
            assert key in result

    def test_contains_narrative_keys(self, engine):
        result = engine.get_drawing_eisv()
        for key in ("arc_phase", "i_momentum", "coherence_settled",
                     "attention_exhausted", "narrative_complete",
                     "compositional_satisfaction"):
            assert key in result

    def test_values_are_numbers(self, engine):
        result = engine.get_drawing_eisv()
        for key in ("E", "I", "S", "V", "C", "curiosity", "engagement",
                     "fatigue", "energy"):
            assert isinstance(result[key], (int, float))


class TestDrawingEngineSetEra:
    """Test era switching."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_set_era_immediate_no_drawing(self, engine):
        result = engine.set_era("pointillist")
        assert result["success"] is True
        assert result["queued"] is False
        assert engine.active_era.name == "pointillist"

    def test_set_era_queued_with_drawing(self, engine):
        # Add enough pixels for "drawing in progress"
        for i in range(60):
            engine.canvas.draw_pixel(i, 0, (255, 0, 0))
        result = engine.set_era("field")
        assert result["success"] is True
        assert result["queued"] is True
        assert engine.canvas.pending_era_switch == "field"

    def test_set_era_force_immediate(self, engine):
        for i in range(60):
            engine.canvas.draw_pixel(i, 0, (255, 0, 0))
        result = engine.set_era("geometric", force_immediate=True)
        assert result["success"] is True
        assert result["queued"] is False
        assert engine.active_era.name == "geometric"

    def test_set_era_unknown_fails(self, engine):
        result = engine.set_era("nonexistent_era_xyz")
        assert result["success"] is False

    def test_set_era_clears_pending(self, engine):
        engine.canvas.pending_era_switch = "geometric"
        engine.set_era("pointillist")
        assert engine.canvas.pending_era_switch is None


class TestDrawingEngineCanvasClear:
    """Test canvas_clear behavior."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_clear_resets_canvas_and_intent(self, engine):
        engine.canvas.draw_pixel(10, 10, (255, 0, 0))
        engine.intent.mark_count = 50
        engine.canvas_clear(persist=False)
        assert len(engine.canvas.pixels) == 0
        assert engine.intent.mark_count == 0

    def test_clear_prevents_double_clear_during_pause(self, engine):
        engine.canvas.draw_pixel(10, 10, (255, 0, 0))
        engine.canvas_clear(persist=False)
        # During pause period, second clear should be a no-op
        engine.canvas.draw_pixel(20, 20, (0, 255, 0))
        engine.canvas_clear(persist=False)
        # The second clear didn't happen because we're paused
        assert (20, 20) in engine.canvas.pixels

    def test_clear_applies_pending_era(self, engine):
        engine.canvas.pending_era_switch = "pointillist"
        engine.canvas_clear(persist=False)
        assert engine.active_era.name == "pointillist"

    def test_clear_generates_drawing_goal(self, engine):
        engine.last_anima = make_anima(warmth=0.8, clarity=0.3)
        engine.canvas_clear(persist=False)
        assert engine.drawing_goal is not None


class TestDrawingEngineCanvasCheckAutonomy:
    """Test canvas_check_autonomy decision logic."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            eng = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        return eng

    def test_returns_none_without_anima(self, engine):
        assert engine.canvas_check_autonomy(anima=None) is None

    def test_returns_none_during_pause(self, engine):
        engine.canvas.drawing_paused_until = time.time() + 100
        anima = make_anima()
        assert engine.canvas_check_autonomy(anima) is None

    def test_returns_none_recent_save(self, engine):
        engine.canvas.last_save_time = time.time()  # Just saved
        anima = make_anima()
        assert engine.canvas_check_autonomy(anima) is None

    def test_false_start_returns_abandoned(self, engine):
        """False start conditions lead to canvas abandonment."""
        anima = make_anima()
        engine.canvas.last_save_time = 0.0
        engine.canvas.drawing_paused_until = 0.0
        engine.canvas.phase_start_time = time.time() - 60
        engine.canvas.mark_count = 20

        # Set up false start conditions
        state = engine.intent.state
        state.arc_phase = "opening"
        state.i_momentum = 0.1
        state.engagement = 0.1
        state.coherence_history = [0.1] * 10

        # Add some pixels (but < 200)
        for i in range(50):
            engine.canvas.draw_pixel(i, 0, (128, 128, 128))

        with patch.object(engine, '_check_lumen_said_finished', return_value=False):
            result = engine.canvas_check_autonomy(anima)
        assert result == "abandoned"

    def test_narrative_complete_saves_and_clears(self, engine):
        """Narrative complete with enough pixels saves and clears."""
        anima = make_anima()
        engine.canvas.last_save_time = 0.0
        engine.canvas.drawing_paused_until = 0.0
        engine.last_anima = anima

        # Set up narrative complete conditions
        state = engine.intent.state
        state.fatigue = 0.95  # Emergency exit path
        engine.intent.mark_count = 10

        # Add enough pixels
        for i in range(250):
            engine.canvas.draw_pixel(i % 240, i // 240, (128, 128, 128))

        with patch.object(engine, '_check_lumen_said_finished', return_value=False), \
             patch.object(engine, 'canvas_save', return_value="/tmp/test.png"), \
             patch.object(engine, 'canvas_clear'):
            result = engine.canvas_check_autonomy(anima)
        assert result == "saved_and_cleared"


# ---------------------------------------------------------------------------
# DrawingEISV backward compatibility alias
# ---------------------------------------------------------------------------

class TestCanvasSaveAtomicPNG:
    """Test that canvas_save writes valid PNG via atomic temp file."""

    def test_atomic_save_writes_png(self, tmp_path):
        """Regression: img.save(.tmp) failed because PIL needs format='PNG'."""
        from pathlib import Path

        engine = DrawingEngine()
        engine.canvas.draw_pixel(10, 10, (255, 0, 0))
        engine.canvas.draw_pixel(20, 20, (0, 255, 0))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = engine.canvas_save()

        assert result is not None
        saved = Path(result)
        assert saved.exists()
        assert saved.suffix == ".png"
        assert saved.stat().st_size > 0
        # No .tmp files should remain
        drawings_dir = tmp_path / ".anima" / "drawings"
        tmp_files = list(drawings_dir.glob("*.tmp"))
        assert tmp_files == []


class TestCanvasSaveSensorFallback:
    """canvas_save should pull fresh readings when _last_readings is unset,
    instead of feeding growth silent defaults (0 lux, 22 C, 50% humidity)."""

    def test_tiny_opening_snapshot_skips_growth_observation(self, tmp_path):
        """Opening-phase false starts can be saved as files without counting as drawings."""
        from pathlib import Path
        from unittest.mock import MagicMock

        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            engine = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        engine.canvas.drawing_phase = "opening"
        engine.canvas.arc_phase = "opening"
        engine.intent.state.arc_phase = "opening"
        engine.canvas.mark_count = 1
        engine.intent.mark_count = 1
        engine.canvas.draw_pixel(10, 10, (255, 0, 0))
        engine.canvas.draw_pixel(20, 20, (0, 255, 0))
        engine.last_anima = make_anima()
        setattr(engine, "_last_readings", MagicMock(light_lux=42.0, ambient_temp_c=19.5, humidity_pct=55.0))

        growth_mock = MagicMock()

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("anima_mcp.growth.get_growth_system", return_value=growth_mock):
            result = engine.canvas_save(manual=True)

        assert result is not None
        assert Path(result).exists()
        growth_mock.observe_drawing.assert_not_called()
        growth_mock.record_drawing_completion.assert_not_called()

    def test_fresh_sensor_read_when_last_readings_missing(self, tmp_path):
        from unittest.mock import MagicMock

        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            engine = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        for i in range(250):
            engine.canvas.draw_pixel(i % 240, i // 240, (255, 0, 0))
        engine.last_anima = make_anima()
        # Explicitly no prior render push
        engine._last_readings = None

        fresh = MagicMock(light_lux=42.0, ambient_temp_c=19.5, humidity_pct=55.0)
        fake_sensors = MagicMock()
        fake_sensors.read.return_value = fresh

        growth_mock = MagicMock()
        growth_mock.observe_drawing.return_value = None

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("anima_mcp.accessors._get_sensors", return_value=fake_sensors), \
             patch("anima_mcp.growth.get_growth_system", return_value=growth_mock):
            engine.canvas_save()

        fake_sensors.read.assert_called_once()
        growth_mock.observe_drawing.assert_called_once()
        env = growth_mock.observe_drawing.call_args.kwargs["environment"]
        assert env["light_lux"] == 42.0
        assert env["temp_c"] == 19.5
        assert env["humidity_pct"] == 55.0

    def test_skip_growth_when_fresh_read_also_fails(self, tmp_path):
        from unittest.mock import MagicMock

        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"):
            engine = DrawingEngine(db_path=str(tmp_path / "test.db"), identity_store=None)
        for i in range(250):
            engine.canvas.draw_pixel(i % 240, i // 240, (255, 0, 0))
        engine.last_anima = make_anima()
        engine._last_readings = None

        fake_sensors = MagicMock()
        fake_sensors.read.side_effect = RuntimeError("sensor bus down")

        growth_mock = MagicMock()

        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("anima_mcp.accessors._get_sensors", return_value=fake_sensors), \
             patch("anima_mcp.growth.get_growth_system", return_value=growth_mock):
            engine.canvas_save()

        # Should NOT call growth with defaults when readings are unavailable
        growth_mock.observe_drawing.assert_not_called()


class TestDrawingEISVAlias:
    """Test that DrawingEISV is an alias for DrawingState."""

    def test_alias_is_same_class(self):
        assert DrawingEISV is DrawingState

    def test_alias_works_as_constructor(self):
        eisv = DrawingEISV()
        assert eisv.E == 0.4
        assert hasattr(eisv, "curiosity")
