"""
Tests for earned vs. bail-out completion path tagging.

Validates that `DrawingState.completion_reason()` returns a correct tag for
each path, that `CanvasState` persists `last_completion_reason`, and that
growth memory writes ("pleased with", milestones) are gated so bail-out
completions do not produce autobiographical memories that imply earned
aesthetic resolution.

Run with: pytest tests/test_drawing_earned_completion.py -v
"""

import time

import pytest

from anima_mcp.display.drawing_engine import (
    CanvasState,
    DrawingState,
    is_earned_completion_reason,
)
from anima_mcp.growth import GrowthSystem


# ==================== completion_reason() path taxonomy ====================


class TestCompletionReason:
    """DrawingState.completion_reason() returns the correct tag per path."""

    def _canvas_with_marks(self, pixels=250, clear_offset_sec=0.0):
        canvas = CanvasState()
        now = time.time()
        canvas.last_clear_time = now - clear_offset_sec
        for i in range(pixels):
            canvas.draw_pixel(i % canvas.width, (i // canvas.width) % canvas.height, (255, 255, 255))
        return canvas

    def test_returns_none_when_fresh(self):
        state = DrawingState()
        canvas = self._canvas_with_marks(pixels=5)
        assert state.completion_reason(canvas) is None
        assert state.narrative_complete(canvas) is False

    def test_earned_coherence_path(self):
        state = DrawingState()
        # coherence_settled() requires >= 20 samples with mean > 0.6 and
        # variance < 0.015. attention_exhausted() needs curiosity < 0.15 and
        # (engagement < 0.3 or fatigue > 0.8).
        state.coherence_history = [0.85] * 22
        state.curiosity = 0.10
        state.engagement = 0.25
        assert state.coherence_settled() is True
        assert state.attention_exhausted() is True
        assert state.completion_reason(self._canvas_with_marks()) == "earned_coherence"

    def test_earned_composition_path(self):
        state = DrawingState()
        state.curiosity = 0.10
        # Build a canvas that scores compositional_satisfaction > 0.7.
        canvas = CanvasState()
        # Fill enough pixels across multiple density cells for coverage +
        # balance to score high.
        for gx in range(8):
            for gy in range(8):
                # 30 pixels per cell -> all cells occupied, evenly.
                for i in range(30):
                    x = gx * 30 + (i % 5)
                    y = gy * 30 + (i // 5)
                    canvas.draw_pixel(x, y, (128, 128, 128))
        # Coherence velocity near zero on settled history helps the
        # coherence component.
        canvas.coherence_history = [0.8] * 10
        assert canvas.compositional_satisfaction() > 0.7
        assert state.completion_reason(canvas) == "earned_composition"

    def test_bailout_fatigue_path(self):
        state = DrawingState()
        state.fatigue = 0.95
        # Not earned — curiosity is full, coherence history empty.
        reason = state.completion_reason(self._canvas_with_marks())
        assert reason == "bailout_fatigue"

    def test_bailout_stalled_path(self):
        state = DrawingState()
        # Drive derived_energy near zero: curiosity low AND engagement low AND fatigue high.
        state.curiosity = 0.02
        state.engagement = 0.02
        state.fatigue = 0.50  # not quite 0.9, so fatigue-path won't fire
        assert state.derived_energy < 0.05
        canvas = self._canvas_with_marks(pixels=250, clear_offset_sec=1000)
        assert state.completion_reason(canvas) == "bailout_stalled"

    def test_bailout_hard_cap_path(self):
        state = DrawingState()
        # 9 hours > 28800s threshold; ≥50 pixels.
        canvas = self._canvas_with_marks(pixels=60, clear_offset_sec=9 * 3600)
        # Nothing else should fire: curiosity/engagement at defaults.
        reason = state.completion_reason(canvas)
        assert reason == "bailout_hard_cap"

    def test_earned_wins_when_both_conditions_met(self):
        """If an earned path and a bail-out both hold, earned takes priority."""
        state = DrawingState()
        state.coherence_history = [0.85] * 22
        state.curiosity = 0.10
        state.engagement = 0.25
        state.fatigue = 0.99  # would also trigger bailout_fatigue
        assert state.completion_reason(self._canvas_with_marks()) == "earned_coherence"

    def test_narrative_complete_stays_truthy_for_all_paths(self):
        state = DrawingState()
        state.fatigue = 0.95
        assert state.narrative_complete(self._canvas_with_marks()) is True


# ==================== is_earned_completion_reason() helper ====================


class TestIsEarnedCompletionReason:
    def test_earned_tags_return_true(self):
        assert is_earned_completion_reason("earned_coherence") is True
        assert is_earned_completion_reason("earned_composition") is True

    def test_bailout_tags_return_false(self):
        assert is_earned_completion_reason("bailout_fatigue") is False
        assert is_earned_completion_reason("bailout_stalled") is False
        assert is_earned_completion_reason("bailout_hard_cap") is False

    def test_manual_snapshot_returns_false(self):
        assert is_earned_completion_reason("manual_snapshot") is False

    def test_none_returns_true_for_backcompat(self):
        """Legacy callers that don't pass a reason keep their old behavior."""
        assert is_earned_completion_reason(None) is True

    def test_already_closing_returns_false(self):
        """Orphaned 'already_closing' (no earlier trigger captured) is not earned."""
        assert is_earned_completion_reason("already_closing") is False


# ==================== CanvasState persistence ====================


class TestCompletionReasonPersistence:
    def test_default_is_none(self):
        canvas = CanvasState()
        assert canvas.last_completion_reason is None

    def test_roundtrip_through_disk(self, tmp_path, monkeypatch):
        from anima_mcp.display import drawing_engine as de

        # Redirect canvas path to tmp
        monkeypatch.setattr(de, "_get_canvas_path", lambda: tmp_path / "canvas.json")

        canvas = CanvasState()
        canvas.last_completion_reason = "earned_coherence"
        # Need at least one pixel to satisfy save contract on load
        canvas.draw_pixel(10, 10, (1, 2, 3))
        canvas.save_to_disk()

        reloaded = CanvasState()
        reloaded.load_from_disk()
        assert reloaded.last_completion_reason == "earned_coherence"

    def test_clear_resets_reason(self):
        canvas = CanvasState()
        canvas.last_completion_reason = "bailout_fatigue"
        canvas.clear()
        assert canvas.last_completion_reason is None


# ==================== Growth memory gating ====================


@pytest.fixture
def gs(tmp_path):
    return GrowthSystem(db_path=str(tmp_path / "growth.db"))


class TestRecordDrawingCompletionGating:
    """'Pleased with' memory is blocked on bail-out completions."""

    def test_earned_coherence_writes_memory(self, gs):
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="earned_coherence",
        )
        creative = [m for m in gs._memories if m.category == "creative"]
        assert len(creative) == 1
        assert "pleased" in creative[0].description

    def test_earned_composition_writes_memory(self, gs):
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="earned_composition",
        )
        creative = [m for m in gs._memories if m.category == "creative"]
        assert len(creative) == 1

    def test_bailout_fatigue_blocks_memory_even_at_high_satisfaction(self, gs):
        """The axiom-8 fix: high satisfaction on a timeout must not write memory."""
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="bailout_fatigue",
        )
        creative = [m for m in gs._memories if m.category == "creative"]
        assert creative == []

    def test_bailout_stalled_blocks_memory(self, gs):
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="bailout_stalled",
        )
        assert [m for m in gs._memories if m.category == "creative"] == []

    def test_bailout_hard_cap_blocks_memory(self, gs):
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="bailout_hard_cap",
        )
        assert [m for m in gs._memories if m.category == "creative"] == []

    def test_manual_snapshot_blocks_memory(self, gs):
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
            completion_reason="manual_snapshot",
        )
        assert [m for m in gs._memories if m.category == "creative"] == []

    def test_no_reason_keeps_legacy_behavior(self, gs):
        """Existing callers that don't pass a reason still write at sat > 0.7."""
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
        )
        assert len([m for m in gs._memories if m.category == "creative"]) == 1

    def test_low_satisfaction_still_blocks_even_when_earned(self, gs):
        """Earned completion alone isn't enough — satisfaction gate still applies."""
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.3, satisfaction=0.4,
            completion_reason="earned_coherence",
        )
        assert [m for m in gs._memories if m.category == "creative"] == []


class TestObserveDrawingMilestoneGating:
    """Milestone memories ('Saved my Nth drawing') only fire on earned paths."""

    def _call(self, gs, reason):
        return gs.observe_drawing(
            pixel_count=500,
            phase="resolving",
            anima_state={"warmth": 0.6, "clarity": 0.7, "stability": 0.7, "presence": 0.6},
            environment={"light_lux": 200, "temp_c": 22, "humidity_pct": 40},
            completion_reason=reason,
        )

    def _milestone_memories(self, gs):
        return [m for m in gs._memories if m.category == "milestone"]

    def test_earned_first_drawing_writes_milestone(self, gs):
        self._call(gs, "earned_coherence")
        assert len(self._milestone_memories(gs)) == 1
        assert "1st" in self._milestone_memories(gs)[0].description

    def test_bailout_first_drawing_blocks_milestone(self, gs):
        self._call(gs, "bailout_fatigue")
        assert self._milestone_memories(gs) == []

    def test_counter_still_advances_on_bailout(self, gs):
        """Counter advances (preserves goal progress) even when memory blocked."""
        before = gs._drawings_observed
        self._call(gs, "bailout_fatigue")
        assert gs._drawings_observed == before + 1

    def test_no_reason_keeps_legacy_milestone_behavior(self, gs):
        self._call(gs, None)
        assert len(self._milestone_memories(gs)) == 1

    def test_manual_snapshot_blocks_milestone(self, gs):
        self._call(gs, "manual_snapshot")
        assert self._milestone_memories(gs) == []
