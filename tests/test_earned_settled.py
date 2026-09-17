"""earned_settled: self-relative completion for a piece that stopped changing.

Derived from the first 27 cap-length pieces of drawing_trajectory
(2026-08-02..08-11). The corpus splits three ways and each way is a test
family here:

  field       — plateaus settled (marks continue, novelty ~0): MUST fire
  gestural    — keeps producing novelty to the cap:            MUST NOT fire
  geometric   — freezes idle (marks stop entirely):            MUST NOT fire

Evidence semantics under test: active-below advances the streak, active-above
resets it, idle/unknowable holds it.
"""

import time

from anima_mcp.display.drawing_engine import (
    MIN_RECORDED_DRAWING_PIXELS,
    SETTLED_MIN_AGE_SECONDS,
    SETTLED_MIN_MARKS,
    SETTLED_SMOOTH_WINDOW,
    SETTLED_STREAK_SAMPLES,
    TRAJECTORY_SAMPLE_INTERVAL,
    CanvasState,
    DrawingState,
    is_earned_completion_reason,
)


def make_engine_stub(canvas):
    """The tracker only touches self.canvas — a stub keeps the test honest
    about that boundary."""
    from anima_mcp.display.drawing_engine import DrawingEngine

    class _Stub:
        pass

    stub = _Stub()
    stub.canvas = canvas
    stub._update_novelty_settling = (
        DrawingEngine._update_novelty_settling.__get__(stub)
    )
    return stub


def paint(canvas, n, start=0):
    """Add n distinct pixels (and one mark per call site's bookkeeping)."""
    for i in range(start, start + n):
        canvas.draw_pixel(i % 240, i // 240, (10, 20, 30))


def advance(stub, t, novel_px, marks_delta, px_offset):
    """Simulate one sample interval: paint novel_px pixels, bump marks."""
    canvas = stub.canvas
    if novel_px:
        paint(canvas, novel_px, start=px_offset)
    canvas.mark_count += marks_delta
    stub._update_novelty_settling(t)
    return px_offset + novel_px


class TestTrackerSemantics:
    def _fresh(self):
        canvas = CanvasState()
        canvas.pixels = {}
        canvas.mark_count = 0
        stub = make_engine_stub(canvas)
        # init sample (creates the dict, establishes baseline)
        paint(canvas, 5)
        canvas.mark_count = 1
        stub._update_novelty_settling(1000.0)
        return stub, canvas

    def _run(self, stub, samples, t0=1000.0, px0=5):
        """samples: list of (novel_px, marks_delta). Returns tracker dict."""
        t, px = t0, px0
        for novel, marks in samples:
            t += TRAJECTORY_SAMPLE_INTERVAL + 1
            px = advance(stub, t, novel, marks, px)
        return stub.canvas._novelty_settling

    def test_field_shape_accumulates_streak(self):
        """High novelty then low-with-marks — the settled signature."""
        stub, canvas = self._fresh()
        ramp = [(60, 5)] * 6                        # peak-setting active work
        # First SMOOTH_WINDOW-1 low samples still average with the ramp and
        # correctly read above-threshold; the streak starts after that.
        settled = [(2, 3)] * (SETTLED_STREAK_SAMPLES + SETTLED_SMOOTH_WINDOW)
        ns = self._run(stub, ramp + settled)
        assert ns["streak"] >= SETTLED_STREAK_SAMPLES
        assert ns["peak"] > 0

    def test_gestural_shape_never_accumulates(self):
        """Steady novelty near its own peak — streak stays at zero."""
        stub, canvas = self._fresh()
        ns = self._run(stub, [(50, 8)] * 30)
        assert ns["streak"] == 0

    def test_active_above_threshold_resets_streak(self):
        stub, canvas = self._fresh()
        ns = self._run(stub, [(60, 5)] * 6 + [(2, 3)] * 5 + [(55, 5)])
        assert ns["streak"] == 0

    def test_idle_holds_streak(self):
        """A geometric-style freeze (marks stop) neither advances nor resets."""
        stub, canvas = self._fresh()
        ns = self._run(stub, [(60, 5)] * 6 + [(2, 3)] * 5)
        streak_before = ns["streak"]
        assert streak_before > 0
        ns = self._run(stub, [(0, 0)] * 20, t0=ns["last_t"],
                       px0=ns["last_px"])
        assert ns["streak"] == streak_before

    def test_idle_forever_never_fires(self):
        """The geometric freeze: 7h of marks_delta=0 accumulates nothing."""
        stub, canvas = self._fresh()
        ns = self._run(stub, [(200, 10)] * 4 + [(0, 0)] * 84)
        assert ns["streak"] == 0

    def test_sub_interval_ticks_do_not_advance(self):
        """Retry ticks between sample intervals must not feed the tracker."""
        stub, canvas = self._fresh()
        ns0 = dict(canvas._novelty_settling)
        stub._update_novelty_settling(1000.0 + 5)
        stub._update_novelty_settling(1000.0 + 10)
        ns = canvas._novelty_settling
        assert ns["last_t"] == ns0["last_t"]

    def test_corrupt_tracker_restarts_clean(self):
        stub, canvas = self._fresh()
        canvas._novelty_settling = {"last_t": "not-a-number"}
        stub._update_novelty_settling(99999.0)
        assert canvas._novelty_settling is None
        stub._update_novelty_settling(100400.0)
        assert isinstance(canvas._novelty_settling, dict)

    def test_corrupt_peak_and_streak_self_heal(self):
        """A guard covering only SOME fields would raise every 300s forever on
        the rest. Every corrupt field must reset the tracker, not throw."""
        for bad_field, bad_value in (("peak", "garbage"), ("streak", "NaN"),
                                     ("recent", [1.0, "x", 2.0])):
            stub, canvas = self._fresh()
            ns = canvas._novelty_settling
            ns[bad_field] = bad_value
            # Enough active samples to reach the corrupted field's code path
            self._run(stub, [(60, 5)] * 4)
            ns = canvas._novelty_settling
            # Either healed by reset (fresh dict) or the corrupt value was
            # overwritten by a clean one — never a raise, never stuck.
            assert ns is None or isinstance(ns, dict)
            # And the tracker keeps functioning afterwards:
            if ns is None:
                stub._update_novelty_settling(999999.0)
                assert isinstance(canvas._novelty_settling, dict)

    def test_single_mark_trickle_holds_not_advances(self):
        """A RESTING-hour mark landing on painted territory is not 'working'.
        marks_delta=1 must hold the streak exactly like idle does."""
        stub, canvas = self._fresh()
        ns = self._run(stub, [(60, 5)] * 6 + [(2, 3)] * 5)
        streak_before = ns["streak"]
        assert streak_before > 0
        ns = self._run(stub, [(0, 1)] * 20, t0=ns["last_t"], px0=ns["last_px"])
        assert ns["streak"] == streak_before


class TestCompletionPredicate:
    def _settled_canvas(self, age_s=SETTLED_MIN_AGE_SECONDS + 600):
        canvas = CanvasState()
        paint(canvas, MIN_RECORDED_DRAWING_PIXELS + 50)
        canvas.mark_count = SETTLED_MIN_MARKS + 50
        canvas.last_clear_time = time.time() - age_s
        canvas._novelty_settling = {
            "last_t": time.time(), "last_px": 250, "last_marks": 150,
            "recent": [1.0, 0.0, 2.0], "peak": 55.0,
            "streak": SETTLED_STREAK_SAMPLES,
        }
        return canvas

    def test_fires_when_settled_with_investment(self):
        state = DrawingState()
        assert state.completion_reason(self._settled_canvas()) == "earned_settled"

    def test_streak_short_by_one_does_not_fire(self):
        canvas = self._settled_canvas()
        canvas._novelty_settling["streak"] = SETTLED_STREAK_SAMPLES - 1
        assert DrawingState().completion_reason(canvas) is None

    def test_min_age_blocks_early_fire(self):
        canvas = self._settled_canvas(age_s=SETTLED_MIN_AGE_SECONDS - 100)
        assert DrawingState().completion_reason(canvas) is None

    def test_min_marks_blocks_thin_piece(self):
        canvas = self._settled_canvas()
        canvas.mark_count = SETTLED_MIN_MARKS - 1
        assert DrawingState().completion_reason(canvas) is None

    def test_no_tracker_no_fire(self):
        canvas = self._settled_canvas()
        canvas._novelty_settling = None
        assert DrawingState().completion_reason(canvas) is None

    def test_zero_peak_no_fire(self):
        """A piece that never established a peak has nothing to be relative to."""
        canvas = self._settled_canvas()
        canvas._novelty_settling["peak"] = 0.0
        assert DrawingState().completion_reason(canvas) is None

    def test_earned_takes_priority_over_hard_cap(self):
        canvas = self._settled_canvas(age_s=29000)  # past the 8h cap too
        assert DrawingState().completion_reason(canvas) == "earned_settled"

    def test_already_closing_outranks_settled(self):
        """The originating reason is captured on the closing transition; the
        caller's `!= "already_closing"` guard protects it from overwrite. A
        settled tracker landing mid-close must NOT relabel that capture —
        that would re-introduce the pride-memory bug through a new door."""
        canvas = self._settled_canvas()
        state = DrawingState()
        state.arc_phase = "closing"
        assert state.completion_reason(canvas) == "already_closing"

    def test_forced_era_switch_resets_tracker(self):
        """Peak belongs to the era that set it. A forced mid-piece switch must
        not carry a geometric-sized peak into a small-mark era."""
        from anima_mcp.display.drawing_engine import DrawingEngine

        class _Stub:
            pass

        stub = _Stub()
        canvas = CanvasState()
        paint(canvas, 60)  # drawing_in_progress threshold is 50px
        canvas._novelty_settling = {"peak": 2500.0, "streak": 8}
        canvas.save_to_disk = lambda: True

        class _Intent:
            era_state = None

        stub.canvas = canvas
        stub.intent = _Intent()
        # set_era now asks the engine which recent dispositions belong to the
        # era it is switching to, so the partial stub needs that method too.
        stub._recent_dispositions_for = (
            DrawingEngine._recent_dispositions_for.__get__(stub)
        )
        stub.set_era = DrawingEngine.set_era.__get__(stub)
        result = stub.set_era("field", force_immediate=True)
        assert result["success"] is True
        assert canvas._novelty_settling is None


class TestPersistence:
    def test_round_trip(self, tmp_path, monkeypatch):
        from anima_mcp.display import drawing_engine as de
        monkeypatch.setattr(de, "_get_canvas_path",
                            lambda: tmp_path / "canvas.json")
        canvas = CanvasState()
        paint(canvas, 10)
        tracker = {"last_t": 123.0, "last_px": 10, "last_marks": 4,
                   "recent": [3.0, 1.0], "peak": 40.0, "streak": 7}
        canvas._novelty_settling = dict(tracker)
        assert canvas.save_to_disk()
        restored = CanvasState()
        monkeypatch.setattr(de, "_get_canvas_path",
                            lambda: tmp_path / "canvas.json")
        restored.load_from_disk()
        assert restored._novelty_settling == tracker

    def test_clear_resets_tracker(self):
        canvas = CanvasState()
        canvas._novelty_settling = {"streak": 99}
        canvas.clear()
        assert canvas._novelty_settling is None


class TestReasonTaxonomy:
    def test_earned_settled_is_earned(self):
        assert is_earned_completion_reason("earned_settled")
