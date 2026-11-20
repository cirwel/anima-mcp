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


# ============ the live leak: a bail-out that never entered "resolving" ============


class TestBailoutTaggedOutsideResolving:
    """Regression for the axiom-8 leak found live on 2026-07-30.

    `last_completion_reason` was captured only in the `arc_phase == "resolving"`
    branch. Entering "resolving" requires C > 0.6, but resonance caps C ~0.52 —
    so resonance canvases complete straight out of "developing", the tag was
    never set, and the gate saw None and failed open.
    """

    def _canvas(self, pixels=250, age_sec=0.0):
        canvas = CanvasState()
        canvas.last_clear_time = time.time() - age_sec
        for i in range(pixels):
            canvas.draw_pixel(
                i % canvas.width, (i // canvas.width) % canvas.height, (255, 255, 255)
            )
        return canvas

    def test_hard_cap_from_developing_is_tagged_not_none(self):
        """An 8h cap outside "resolving" reports a bail-out tag, never None."""
        state = DrawingState()
        state.arc_phase = "developing"
        canvas = self._canvas(age_sec=28801)

        reason = state.completion_reason(canvas)

        assert reason == "bailout_hard_cap"
        assert is_earned_completion_reason(reason) is False

    def test_fatigue_bailout_from_developing_is_not_earned(self):
        state = DrawingState()
        state.arc_phase = "developing"
        state.fatigue = 0.95

        reason = state.completion_reason(self._canvas())

        assert reason == "bailout_fatigue"
        assert is_earned_completion_reason(reason) is False

    def test_canvas_clear_drops_the_tag(self):
        """A captured bail-out tag must not survive onto the next canvas."""
        canvas = self._canvas()
        canvas.last_completion_reason = "bailout_hard_cap"
        canvas.clear()
        assert canvas.last_completion_reason is None


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

    def test_none_fails_closed(self):
        """Unknown provenance is not earned.

        This asserted True until 2026-07-30. That fail-open was the axiom-8
        leak: two of three save paths never tagged a reason, so bail-outs
        arrived as None and were written as pride memories.
        """
        assert is_earned_completion_reason(None) is False

    def test_said_finished_is_earned(self):
        """Lumen declaring the piece done is self-determined completion."""
        assert is_earned_completion_reason("said_finished") is True

    def test_already_closing_returns_false(self):
        """Orphaned 'already_closing' (no earlier trigger captured) is not earned."""
        assert is_earned_completion_reason("already_closing") is False


# ==================== CanvasState persistence ====================


class TestCompletionReasonPersistence:
    def test_default_is_none(self):
        canvas = CanvasState()
        assert canvas.last_completion_reason is None

    def test_roundtrip_through_disk(self, tmp_path, monkeypatch):
        from anima_mcp.display import drawing_state as ds

        # Redirect canvas path to tmp
        monkeypatch.setattr(ds, "_get_canvas_path", lambda: tmp_path / "canvas.json")

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

    def test_no_reason_writes_nothing(self, gs):
        """An untagged completion writes no memory, even at high satisfaction.

        Satisfaction cannot stand in for earning it: compositional_satisfaction
        reads 0.78-0.86 on every live canvas, so it discriminates nothing.
        """
        gs.record_drawing_completion(
            pixel_count=500, mark_count=10,
            coherence=0.8, satisfaction=0.85,
        )
        assert [m for m in gs._memories if m.category == "creative"] == []

    def test_bailout_writes_nothing_at_high_satisfaction(self, gs):
        """The live failure mode: an 8h hard cap must not become pride."""
        gs.record_drawing_completion(
            pixel_count=7949, mark_count=794,
            coherence=0.52, satisfaction=0.86,
            completion_reason="bailout_hard_cap",
        )
        assert [m for m in gs._memories if m.category == "creative"] == []

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

    def test_no_reason_blocks_milestone(self, gs):
        """Untagged completions no longer earn a milestone (was: allowed)."""
        self._call(gs, None)
        assert self._milestone_memories(gs) == []

    def test_manual_snapshot_blocks_milestone(self, gs):
        self._call(gs, "manual_snapshot")
        assert self._milestone_memories(gs) == []


# ============ settling_progress: the per-gate view of earned_field ============


class TestSettlingProgress:
    """The 2026-08-05 watch needs to know WHICH gate held earned_field back.

    Also a regression test for the plumbing: the first version of the
    diagnostics wiring passed `engine.state` (the drawing state actually lives
    at `engine.intent.state`), raised AttributeError, and a bare
    `except Exception: pass` swallowed it — so the field silently never
    appeared and CI stayed green because nothing exercised the real object.
    """

    def _era_and_state(self):
        from anima_mcp.display.eras.resonance import ResonanceEra
        era = ResonanceEra()
        return era, era.create_state()

    def _canvas(self, marks=0):
        canvas = CanvasState()
        canvas.mark_count = marks
        return canvas

    def test_reports_each_gate(self):
        era, era_state = self._era_and_state()
        state = DrawingState()
        state.fatigue = 0.7

        p = era.settling_progress(state, self._canvas(marks=800), era_state)

        assert p["available"] is True
        assert p["marks"] == 800 and p["marks_ok"] is True
        assert p["fatigue_ok"] is True
        assert p["revisit_ok"] is False, "empty window cannot pass"
        assert p["revisit_window_filled"] == 0
        assert "revisit_ratio" in p and "settled_streak" in p

    def test_distinguishes_which_gate_failed(self):
        """The whole point: a blind None becomes a located failure."""
        era, era_state = self._era_and_state()
        state = DrawingState()
        state.fatigue = 0.1  # below gate

        p = era.settling_progress(state, self._canvas(marks=10), era_state)

        assert p["marks_ok"] is False
        assert p["fatigue_ok"] is False

    def test_revisit_ratio_reflects_the_window(self):
        era, era_state = self._era_and_state()
        from anima_mcp.display.eras.resonance import REVISIT_WINDOW
        era_state.revisit_window = [True] * 40 + [False] * 10
        assert len(era_state.revisit_window) == REVISIT_WINDOW

        p = era.settling_progress(DrawingState(), self._canvas(), era_state)

        assert p["revisit_ratio"] == 0.8
        assert p["revisit_window_filled"] == REVISIT_WINDOW

    def test_is_read_only(self):
        """Must not advance the streak — that is earned_completion's job."""
        era, era_state = self._era_and_state()
        era_state.revisit_window = [True] * 50
        era_state.settled_streak = 2

        era.settling_progress(DrawingState(), self._canvas(marks=800), era_state)

        assert era_state.settled_streak == 2

    def test_engine_exposes_the_state_at_intent_dot_state(self):
        """Pin the attribute path the diagnostics wiring depends on."""
        from anima_mcp.display.drawing_engine import DrawingIntent
        intent = DrawingIntent()
        assert hasattr(intent, "state"), "diagnostics reads engine.intent.state"
        assert hasattr(intent, "era_state")


# ============ settling counters survive a restart ============


class TestSettlingPersistence:
    """revisit_window and settled_streak must ride with the field.

    The 48x48 memory field was already persisted via canvas._resonance_field
    (synced every mark, restored on era_state creation). These two counters,
    which sit on the same object and feed the same predicate, were not — so a
    restart reset the window to empty and the streak to 0. earned_field needs
    50 deposits to refill the window plus 5 consecutive settled checks, so a
    restart cadence faster than that made the earned path unreachable
    regardless of calibration. Measured live 2026-07-30: 327 marks,
    revisit_window_filled 0/50.
    """

    def test_canvas_roundtrips_settling(self, tmp_path, monkeypatch):
        import anima_mcp.display.drawing_state as ds
        path = tmp_path / "canvas.json"
        monkeypatch.setattr(ds, "_get_canvas_path", lambda: path)

        canvas = CanvasState()
        canvas._resonance_settling = {"revisit_window": [True, False, True], "settled_streak": 3}
        canvas.save_to_disk()

        restored = CanvasState()
        restored.load_from_disk()

        assert restored._resonance_settling == {
            "revisit_window": [True, False, True], "settled_streak": 3
        }

    def test_clear_resets_settling(self):
        """A new piece starts unsettled even though the field ghost carries over."""
        canvas = CanvasState()
        canvas._resonance_settling = {"revisit_window": [True] * 50, "settled_streak": 4}
        canvas.clear()
        assert canvas._resonance_settling is None

    def test_malformed_settling_is_ignored(self, tmp_path, monkeypatch):
        """A corrupt blob must not break canvas load."""
        import json
        import anima_mcp.display.drawing_state as ds
        path = tmp_path / "canvas.json"
        monkeypatch.setattr(ds, "_get_canvas_path", lambda: path)
        path.write_text(json.dumps({"pixels": {}, "resonance_settling": "not-a-dict"}))

        canvas = CanvasState()
        canvas.load_from_disk()

        assert canvas._resonance_settling is None

    def test_restored_window_lets_earned_field_stay_reachable(self):
        """The point: a piece resumed after a restart is not sent back to zero."""
        from anima_mcp.display.eras.resonance import (
            ResonanceEra, REVISIT_WINDOW, SETTLED_STREAK,
        )
        era = ResonanceEra()
        era_state = era.create_state()
        # Simulate what restore_from_canvas puts back.
        era_state.revisit_window = [True] * REVISIT_WINDOW
        era_state.settled_streak = SETTLED_STREAK - 1

        canvas = CanvasState()
        canvas.mark_count = 800
        state = DrawingState()
        state.fatigue = 0.7

        # One more settled check should now earn it, rather than 50 deposits away.
        assert era.earned_completion(state, canvas, era_state) == "earned_field"
