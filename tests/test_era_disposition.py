"""Per-piece dispositions: does an era actually vary its work, and at what cost?

The complaint this answers: the art had become "churning of sameness except
maybe field era". That was not drift or decay — it was three lines of code.
`gestural`, `geometric` and `resonance` each had a `create_state()` whose whole
body was a bare constructor, so every piece in those eras began identical and
all randomness was per-mark. Hundreds of independent local draws converge on
their own mean (the law of large numbers), so the corpus reads as one texture
repeated. `field` escaped it because `field_seed_a/b` are drawn once and every
mark samples that one field; the operator's ear was right.

These tests pin three things, in order of how badly a regression would hurt:

1. The corpus-level statistics the derivations read are UNCHANGED. Both
   `derive_drawing_thresholds.py` and `derive_curiosity_thresholds.py` replay
   `drawing_records`; a change that shifted the corpus mean would move the
   ground they stand on.
2. No completion gate moved. Same contract as
   `test_drawing_instrumentation.py::TestNoGateMoved`.
3. The dispositions actually vary the work, and the novelty loop actually
   prefers unlike pieces.
"""

import json
import random
import statistics

import pytest

from anima_mcp.display.art_era import (
    EraState,
    draw_distinct,
    hue_distance,
    set_distance,
    weighted_choice,
)
from anima_mcp.display.drawing_engine import (
    RECENT_DISPOSITIONS,
    CanvasState,
)
from anima_mcp.display.eras import get_era, list_eras
from anima_mcp.display.eras.gestural import GesturalState
from anima_mcp.display.eras.resonance import (
    HUE_ROTATION_MAX,
    HUE_SPREAD_MAX,
    ResonanceEra,
)


ALL_ERAS = sorted(list_eras())


def _run(era, state, marks, canvas=None):
    """Mirror the engine's mark loop closely enough to measure a composition."""
    canvas = canvas if canvas is not None else CanvasState()
    fx, fy, d = 120.0, 120.0, 0.0
    switches = 0
    for _ in range(marks):
        if state.gesture_remaining <= 0:
            era.choose_gesture(state, 0.7, 0.8, 0.7, 0.5)
            switches += 1
        era.place_mark(state, canvas, fx, fy, d, 0.6, (200, 100, 50))
        state.gesture_remaining -= 1
        fx, fy, d = era.drift_focus(
            state, fx, fy, d, 0.8, 0.7, 0.5, 0.7, canvas=canvas)
    return canvas, switches


# ---------------------------------------------------------------------------
# 1. The corpus must not move
# ---------------------------------------------------------------------------

# Gestural's live median marks at completion (2026-08-22), the same figure
# test_coverage_intention.py uses. The disposition effect compounds over a
# piece — a drag-led piece keeps accruing its density — so it must be measured
# over a realistic length, not a short one.
GESTURAL_MARKS = 1200
# 150, not 60: a standard deviation estimated from 60 samples has a standard
# error near a tenth of itself, and an early draft of this file failed on that
# noise alone (it read the control's sd as 448 where 300 samples give 385).
GESTURAL_SEEDS = 150

_pixel_cache = {}


def _gestural_pixel_stats(with_disposition):
    """Pixels per piece, with and without a per-piece disposition.

    A `GesturalState()` with no lead IS the pre-2026-09-17 behavior:
    `affinity()` returns {} and `weighted_choice` gives every gesture 1.0. So
    the control is exact, not approximated. Cached — two tests read it and the
    simulation is the expensive part of this file.
    """
    if with_disposition not in _pixel_cache:
        era = get_era("gestural")
        vals = []
        for seed in range(GESTURAL_SEEDS):
            random.seed(seed)
            state = era.create_state() if with_disposition else GesturalState()
            canvas, _ = _run(era, state, GESTURAL_MARKS)
            vals.append(len(canvas.pixels))
        _pixel_cache[with_disposition] = (
            statistics.mean(vals), statistics.stdev(vals))
    return _pixel_cache[with_disposition]


class TestCorpusStatisticsUnchanged:

    def test_mean_pixels_per_piece_is_preserved(self):
        """The lead gesture is drawn UNIFORMLY, so a drag-led piece (dense) and
        a dot-led one (sparse) are equally likely and the expectation is
        algebraically unchanged. Only the spread between pieces grows — which
        is the entire point.

        Measured over 300 pieces of 1200 marks: mean 5988 -> 5999 px, a drift
        of 0.2%, while the standard deviation nearly doubles.
        """
        control, _ = _gestural_pixel_stats(False)
        live, _ = _gestural_pixel_stats(True)
        drift = abs(live - control) / control
        assert drift < 0.08, (
            f"corpus mean pixels moved {drift:.1%} "
            f"({control:.0f} -> {live:.0f}); the derivations read this corpus"
        )

    def test_spread_between_pieces_actually_grew(self):
        """The complaint was sameness. If the spread did not grow, this change
        did nothing and should not have been made.

        Measured over 300 pieces of 1200 marks: sd 385 -> 683 px, a 1.8x
        widening, while the mean moves 0.2%.
        """
        _, control = _gestural_pixel_stats(False)
        _, live = _gestural_pixel_stats(True)
        assert live > control * 1.3, (
            f"between-piece spread only went {control:.0f} -> {live:.0f} px; "
            "the dispositions are not varying the work"
        )


# ---------------------------------------------------------------------------
# 2. No gate moved
# ---------------------------------------------------------------------------

class TestNoGateMoved:
    """Fatigue accrues per GESTURE SWITCH and feeds `bailout_fatigue`, which is
    the gate that actually ends every geometric piece (8/8 in the corpus). A
    disposition that changed how often gestures switch would move it."""

    def test_gesture_run_lengths_are_untouched(self):
        era = get_era("gestural")

        def mean_switches(with_disposition):
            counts = []
            for seed in range(60):
                random.seed(seed)
                state = era.create_state() if with_disposition else GesturalState()
                _, switches = _run(era, state, 600)
                counts.append(switches)
            return statistics.mean(counts)

        control, live = mean_switches(False), mean_switches(True)
        assert abs(live - control) / control < 0.05, (
            f"switch count moved {control:.1f} -> {live:.1f}; "
            "fatigue accrues per switch and bailout_fatigue reads fatigue"
        )

    def test_geometric_still_stamps_one_shape_per_gesture(self):
        """Geometric's `gesture_remaining = 1` is why every mark is a switch —
        and why its fatigue climbs ~10x faster per mark than a mark-by-mark
        era's. The emphasis must not touch that."""
        era = get_era("geometric")
        state = era.create_state()
        for _ in range(50):
            era.choose_gesture(state, 0.7, 0.8, 0.7, 0.5)
            assert state.gesture_remaining == 1

    @pytest.mark.parametrize("era_name", ALL_ERAS)
    def test_every_gesture_stays_reachable(self, era_name):
        """A lean is not a restricted alphabet. If a disposition could zero out
        a gesture, an era would quietly lose vocabulary — the same defect class
        as a threshold an era's range sits below."""
        era = get_era(era_name)
        seen = set()
        for seed in range(120):
            random.seed(seed)
            state = era.create_state()
            for _ in range(60):
                state.gesture_remaining = 0
                era.choose_gesture(state, 0.7, 0.8, 0.7, 0.5)
                seen.add(state.gesture)
        expected = set(era.create_state().gestures())
        # Resonance picks by field gradient, not vocabulary weights, so an
        # empty field only ever yields sediment — that is its own logic, tested
        # in test_resonance_era.py, and not something a disposition changed.
        if era_name == "resonance":
            assert seen <= expected and seen
        else:
            assert seen == expected, f"unreachable: {expected - seen}"


# ---------------------------------------------------------------------------
# 3. The dispositions do vary the work
# ---------------------------------------------------------------------------

class TestDispositionsVary:

    @pytest.mark.parametrize("era_name", ALL_ERAS)
    def test_every_era_now_has_a_per_piece_character(self, era_name):
        era = get_era(era_name)
        dispositions = []
        for seed in range(12):
            random.seed(seed)
            dispositions.append(era.create_state().disposition())
        assert all(dispositions), f"{era_name} reports no disposition"
        unique = {json.dumps(d, sort_keys=True) for d in dispositions}
        assert len(unique) >= 10, f"{era_name} draws the same piece every time"

    def test_base_erastate_reports_nothing_rather_than_guessing(self):
        """An era with no per-piece character returns {} — which the engine
        reads as 'nothing to record', writing NULL. It must not invent one."""
        assert EraState().disposition() == {}


class TestNoveltyLoop:
    """`draw_distinct` is the one loop from Lumen's own history back into
    Lumen's behavior that closes without a human running a script."""

    def test_empty_history_is_exactly_one_unbiased_draw(self):
        """Fails toward today's behavior (design invariant 2): a fresh install
        or a wiped canvas must behave as it did before this existed, not adopt
        a fabricated preference."""
        calls = []

        def make():
            calls.append(1)
            return {"v": len(calls)}

        result = draw_distinct(make, lambda c, p: 0.0, recent=(), tries=6)
        assert len(calls) == 1
        assert result == {"v": 1}

    def test_it_picks_the_candidate_least_like_recent_work(self):
        candidates = iter([{"h": 10.0}, {"h": 180.0}, {"h": 20.0}])
        result = draw_distinct(
            lambda: next(candidates),
            lambda c, p: hue_distance(c["h"], p["h"]),
            recent=[{"h": 0.0}],
            tries=3,
        )
        assert result == {"h": 180.0}

    def test_it_judges_by_the_nearest_neighbour_not_the_average(self):
        """A candidate sitting on top of one recent piece must not be rescued
        by being far from another. Max-min, not max-mean."""
        near_one = {"h": 2.0}     # min distance to recents ~ 0.01
        middling = {"h": 90.0}    # min distance to recents = 0.5
        candidates = iter([near_one, middling])
        result = draw_distinct(
            lambda: next(candidates),
            lambda c, p: hue_distance(c["h"], p["h"]),
            recent=[{"h": 0.0}, {"h": 180.0}],
            tries=2,
        )
        assert result == middling

    def test_consecutive_pieces_spread_out(self):
        """The behavioral claim: run the loop and the next piece is less like
        the last few than an unguided draw would be."""
        era = get_era("field")

        def mean_gap(use_history):
            random.seed(7)
            recent, gaps = [], []
            for _ in range(40):
                d = era.create_state(recent if use_history else ()).disposition()
                if recent:
                    gaps.append(min(
                        hue_distance(d["base_hue"], p["base_hue"]) for p in recent))
                recent = (recent + [d])[-RECENT_DISPOSITIONS:]
            return statistics.mean(gaps)

        assert mean_gap(True) > mean_gap(False) * 1.2


class TestResonanceKeepsWarmCoolMeaning:
    """Resonance had the worst sameness — its hue is not even random, it is
    `220 - warmth*180`, and warmth is a slow EMA. Every piece was literally the
    same colour. But variety must not be bought by breaking the embodied
    signal: an earlier draft used a +-50 degree rotation with a 40 degree
    spread and pushed warm pieces out of the warm zone in 414 of 5000 seeds."""

    def test_the_bound_is_tight_enough_to_preserve_the_ramp(self):
        assert HUE_ROTATION_MAX + HUE_SPREAD_MAX / 2 < 42.0

    @pytest.mark.parametrize("warmth,low,high", [(0.9, None, 100), (0.1, 150, 270)])
    def test_warm_stays_warm_and_cool_stays_cool(self, warmth, low, high):
        import colorsys
        era = ResonanceEra()
        for seed in range(400):
            random.seed(seed)
            state = era.create_state()
            color, _ = era.generate_color(state, warmth, 0.7, 0.7, 0.7)
            hue = colorsys.rgb_to_hsv(*[c / 255 for c in color])[0] * 360
            ok = (hue < high) if low is None else (low < hue < high)
            ok = ok or (low is None and hue > 340)
            assert ok, f"warmth={warmth} seed={seed} produced hue {hue:.0f}"

    def test_pieces_still_differ_visibly(self):
        import colorsys
        era = ResonanceEra()
        hues = []
        for seed in range(200):
            random.seed(seed)
            color, _ = era.generate_color(era.create_state(), 0.9, 0.7, 0.7, 0.7)
            hues.append(colorsys.rgb_to_hsv(*[c / 255 for c in color])[0] * 360)
        assert max(hues) - min(hues) > 30


# ---------------------------------------------------------------------------
# Plumbing: comparison scope, persistence, and the record
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_hue_distance_is_circular(self):
        assert hue_distance(350, 10) == pytest.approx(20 / 180)
        assert hue_distance(0, 180) == pytest.approx(1.0)
        assert hue_distance(90, 90) == 0.0

    def test_set_distance_is_jaccard(self):
        assert set_distance(["a", "b"], ["a", "b"]) == 0.0
        assert set_distance(["a"], ["b"]) == 1.0
        assert set_distance((), ()) == 0.0

    def test_weighted_choice_keeps_unmentioned_options_reachable(self):
        random.seed(0)
        picks = {weighted_choice(["a", "b", "c"], {"a": 3.0}) for _ in range(300)}
        assert picks == {"a", "b", "c"}


class TestComparisonScope:
    """Dispositions from different eras share no keys. Comparing them would
    fall back on defaults and manufacture a distance that means nothing — the
    fabricated-default failure mode, in a new place."""

    def _engine_like(self, entries):
        class _Stub:
            pass
        from anima_mcp.display.drawing_engine import DrawingEngine
        stub = _Stub()
        stub.canvas = CanvasState()
        stub.canvas.recent_dispositions = entries
        stub._recent_dispositions_for = (
            DrawingEngine._recent_dispositions_for.__get__(stub))
        return stub

    def test_only_same_era_dispositions_are_compared(self):
        stub = self._engine_like([
            {"era": "field", "base_hue": 10.0},
            {"era": "gestural", "hue_offset": 200.0, "lead": "drag"},
            {"era": "field", "base_hue": 200.0},
        ])
        got = stub._recent_dispositions_for("field")
        assert [d["base_hue"] for d in got] == [10.0, 200.0]
        assert stub._recent_dispositions_for("resonance") == []


class TestPersistence:

    def test_recent_dispositions_round_trip(self, tmp_path, monkeypatch):
        from anima_mcp.display import drawing_engine as de
        monkeypatch.setattr(de, "_get_canvas_path", lambda: tmp_path / "canvas.json")
        canvas = CanvasState()
        entries = [{"era": "gestural", "lead": "drag", "hue_offset": 12.0}]
        canvas.recent_dispositions = list(entries)
        assert canvas.save_to_disk()

        restored = CanvasState()
        restored.load_from_disk()
        assert restored.recent_dispositions == entries

    def test_malformed_history_degrades_to_no_bias(self, tmp_path, monkeypatch):
        """Not to a repaired one. An empty history means one plain unbiased
        draw, which is the correct failure."""
        from anima_mcp.display import drawing_engine as de
        path = tmp_path / "canvas.json"
        monkeypatch.setattr(de, "_get_canvas_path", lambda: path)
        path.write_text(json.dumps({
            "pixels": {},
            "recent_dispositions": [
                "not a dict", {"no_era_key": 1}, {"era": "field", "base_hue": 3.0},
            ],
        }))
        canvas = CanvasState()
        canvas.load_from_disk()
        assert canvas.recent_dispositions == [{"era": "field", "base_hue": 3.0}]

    def test_history_is_bounded(self, tmp_path, monkeypatch):
        from anima_mcp.display import drawing_engine as de
        monkeypatch.setattr(de, "_get_canvas_path", lambda: tmp_path / "canvas.json")
        canvas = CanvasState()
        canvas.recent_dispositions = [
            {"era": "field", "base_hue": float(i)} for i in range(50)
        ]
        canvas.save_to_disk()
        restored = CanvasState()
        restored.load_from_disk()
        assert len(restored.recent_dispositions) == RECENT_DISPOSITIONS
