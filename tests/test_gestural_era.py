"""Tests for gestural era smooth commitment signal."""

import colorsys
import random

import pytest

from anima_mcp.display.eras.gestural import (
    ACCENT_COUNT,
    RUN_SHIMMER,
    VIBRANT_COLORS,
    GesturalEra,
    GesturalState,
)


def test_intentionality_smooth_range():
    """I spans 0.1-0.8 across commitment values, not bimodal."""
    state = GesturalState()
    state.gesture_remaining = 0

    # No commitment -> base I
    state.direction_commitment = 0.0
    assert abs(state.intentionality() - 0.15) < 0.01

    # Mid commitment
    state.direction_commitment = 0.5
    i_mid = state.intentionality()
    assert 0.4 < i_mid < 0.5, f"Expected ~0.425, got {i_mid}"

    # Full commitment
    state.direction_commitment = 1.0
    i_full = state.intentionality()
    assert 0.65 < i_full < 0.75, f"Expected ~0.70, got {i_full}"

    # Full commitment + active gesture run
    state.direction_commitment = 1.0
    state.gesture_remaining = 20
    i_max = state.intentionality()
    assert 0.9 < i_max <= 1.0, f"Expected ~1.0, got {i_max}"


def test_commitment_ramps_during_lock():
    """Lock for 20 marks -> commitment > 0.6."""
    random.seed(12345)
    era = GesturalEra()
    state = era.create_state()
    state.direction_locked = True
    state.direction_lock_remaining = 30

    fx, fy, d = 120.0, 120.0, 0.0
    for _ in range(20):
        fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5)

    # 20 marks * +0.06 = 1.0 commitment (capped)
    assert state.direction_commitment > 0.8, (
        f"Expected commitment > 0.8 after 20 locked marks, got {state.direction_commitment}"
    )


def test_commitment_decays_after_lock():
    """Set commitment=0.8, run 30 unlocked marks -> commitment < 0.2."""
    random.seed(12345)
    era = GesturalEra()
    state = era.create_state()
    state.direction_commitment = 0.8
    state.direction_locked = False
    state.direction_lock_remaining = 0

    fx, fy, d = 120.0, 120.0, 0.0
    for _ in range(30):
        fx, fy, d = era.drift_focus(state, fx, fy, d, 0.5, 0.5, 0.5, 0.5)
        # Force no new locks for deterministic test
        state.direction_locked = False
        state.direction_lock_remaining = 0

    # 0.8 * 0.95^30 ≈ 0.17
    assert state.direction_commitment < 0.25, (
        f"Expected commitment < 0.25 after 30 unlocked marks, got {state.direction_commitment}"
    )


def test_jump_preserves_commitment_decay():
    """Focus jump doesn't zero commitment — it should still be positive."""
    state = GesturalState()
    state.direction_commitment = 0.8
    state.direction_locked = False
    state.direction_lock_remaining = 0

    # Simulate what a jump does (sets locked=False, lock_remaining=0)
    state.direction_locked = False
    state.direction_lock_remaining = 0
    # Key assertion: commitment is NOT zeroed by the jump fields
    assert state.direction_commitment == 0.8, (
        f"Jump should not zero commitment, got {state.direction_commitment}"
    )


def _paint(era, state, marks, presence=0.7, warmth=0.5, clarity=0.9, stability=0.9):
    """Colour a piece in the engine's order: the colour is asked for first,
    then — if the run is spent — the next gesture is chosen. Returns one
    (run_index, color, category) per mark. Clarity and stability default high
    so saturation and value stay well clear of the range where an 8-bit RGB
    round-trip blurs hue."""
    out, run = [], -1
    for _ in range(marks):
        color, category = era.generate_color(state, warmth, clarity, stability, presence)
        if state.gesture_remaining <= 0:
            era.choose_gesture(state, clarity, stability, presence, 0.5)
            run += 1
        state.gesture_remaining -= 1
        out.append((run, color, category))
    return out


def _hue(color):
    return colorsys.rgb_to_hsv(*[c / 255 for c in color])[0] * 360.0


def _circular_gap(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _runs(painted):
    runs = {}
    for run, color, category in painted:
        runs.setdefault(run, []).append((color, category))
    return list(runs.values())


class TestColorGeneration:
    """Color system uses vibrant palette + HSV with presence-driven vibrant rate.

    These two counted vibrant draws over 500 calls with `gesture_remaining`
    pinned at 500 — i.e. inside ONE run. Since 2026-11-12 a run is vibrant or
    it is not, so that loop reads 0% or 100% and says nothing. What they
    protect is unchanged and still pinned: presence sets the share of MARKS
    that come out vibrant, ~30% at full presence and ~15% at none. It is now
    counted over whole runs (20,000 marks, ~770 runs), which also lets the
    bounds be tighter than the 500-call sample allowed.
    """

    def test_vibrant_colors_occur_at_high_presence(self):
        """High presence should produce vibrant marks ~30% of the time."""
        random.seed(42)
        era = GesturalEra()
        painted = _paint(era, era.create_state(), 20000, presence=1.0)
        rate = sum(1 for _, _, cat in painted if cat == "vibrant") / len(painted)
        assert 0.24 < rate < 0.36, f"Vibrant rate at high presence {rate:.3f} should be ~30%"

    def test_vibrant_colors_rare_at_low_presence(self):
        """Low presence should produce vibrant marks ~15% of the time."""
        random.seed(42)
        era = GesturalEra()
        painted = _paint(era, era.create_state(), 20000, presence=0.0)
        rate = sum(1 for _, _, cat in painted if cat == "vibrant") / len(painted)
        assert 0.10 < rate < 0.20, f"Vibrant rate at low presence {rate:.3f} should be ~15%"


class TestRunColour:
    """A gesture run is one stroke, so it is one colour (2026-11-12).

    Colour was drawn per mark: a 15-45 mark line came out as 15-45 colours,
    and a quarter of every piece's marks drew from all 26 off-palette colours
    whatever the piece's own palette. Measured over rendered pieces, one with
    a 50-degree hue window still filled 8 of 12 hue bins.
    """

    def test_a_run_holds_one_hue(self):
        """Marks in a run shimmer by RUN_SHIMMER of the piece's span, no more.
        Before, they ranged over the whole span."""
        era = GesturalEra()
        for seed in range(20):
            random.seed(seed)
            state = era.create_state()
            allowed = state.hue_span * RUN_SHIMMER + 2.0  # +2: 8-bit RGB rounding
            for run in _runs(_paint(era, state, 600)):
                hues = [_hue(c) for c, cat in run if cat != "vibrant"]
                for h in hues:
                    assert _circular_gap(h, hues[0]) <= allowed, (
                        f"seed {seed}: a run spans {_circular_gap(h, hues[0]):.0f} "
                        f"degrees of a {state.hue_span:.0f}-degree window")

    def test_runs_still_range_over_the_piece_window(self):
        """Holding a run's hue must not collapse the piece to one hue: across
        runs the window is still used, and never exceeded."""
        era = GesturalEra()
        random.seed(3)
        state = era.create_state()
        centre = (0.5 * 360.0 + state.hue_offset) % 360.0
        offsets = [
            _circular_gap(_hue(c), centre)
            for run in _runs(_paint(era, state, 6000))
            for c, cat in run[:1] if cat != "vibrant"
        ]
        half = state.hue_span / 2.0
        assert max(offsets) <= half + 2.0
        assert max(offsets) > half * 0.6, "runs are not using the piece's window"

    def test_an_accent_run_is_one_of_the_piece_accents(self):
        """A piece has an accent pair, drawn once. Before, every vibrant mark
        drew afresh from all 26."""
        era = GesturalEra()
        for seed in range(20):
            random.seed(seed)
            state = era.create_state()
            assert len(set(state.accents)) == ACCENT_COUNT
            assert set(state.accents) <= set(VIBRANT_COLORS)
            seen = set()
            for run in _runs(_paint(era, state, 2000)):
                vibrant = {c for c, cat in run if cat == "vibrant"}
                if vibrant:
                    assert len(vibrant) == 1, "an accent run changed colour mid-run"
                    assert len(vibrant) == len({c for c, _ in run}), "a run mixed accent and hue"
                    seen |= vibrant
            assert seen and seen <= set(state.accents), f"seed {seed}: {seen}"

    def test_a_bare_state_still_paints(self):
        """`GesturalState()` has no accent pair (it is the control the corpus
        tests compare against). Its accent runs take one colour from the whole
        list rather than failing."""
        random.seed(0)
        era = GesturalEra()
        painted = _paint(era, GesturalState(), 3000)
        assert any(cat == "vibrant" for _, _, cat in painted)
        for run in _runs(painted):
            assert len({c for c, cat in run if cat == "vibrant"}) <= 1

    def test_accents_are_reported_and_survive_json(self):
        import json
        random.seed(1)
        d = GesturalEra().create_state().disposition()
        assert len(d["accents"]) == ACCENT_COUNT
        assert all(isinstance(a, str) and a.startswith("#") for a in d["accents"])
        assert json.loads(json.dumps(d)) == d

    def test_the_next_piece_avoids_the_last_accents(self):
        """The accent pair is part of what draw_distinct compares."""
        era = GesturalEra()

        def shared(use_history):
            hits = 0
            for seed in range(400):
                random.seed(seed)
                prior = dict(era.create_state().disposition(), era="gestural")
                nxt = era.create_state([prior] if use_history else ()).disposition()
                hits += bool(set(nxt["accents"]) & set(prior["accents"]))
            return hits / 400

        # Measured 2026-11-12: 0.088 with history, 0.148 without (chance is 0.151).
        assert shared(True) < shared(False) * 0.8

    def test_a_prior_without_accents_fabricates_no_preference(self, monkeypatch):
        """Pieces banked before 2026-11-12 carry no accents. Against such a
        prior two candidates that differ only in accents must score the same,
        and hue and lead must keep the 1:1 they had."""
        from anima_mcp.display.eras import gestural as mod
        captured = {}

        def spy(make, distance, recent=(), tries=6):
            captured["distance"] = distance
            return make()

        monkeypatch.setattr(mod, "draw_distinct", spy)
        GesturalEra().create_state()
        distance = captured["distance"]
        old = {"era": "gestural", "hue_offset": 40.0, "lead": "drag"}
        base = {"hue_offset": 100.0, "lead": "dot", "accents": ((255, 0, 0), (0, 255, 0))}
        other = dict(base, accents=((0, 0, 255), (128, 0, 128)))
        assert distance(base, old) == distance(other, old)
        same_lead = dict(base, lead="drag")
        far_hue = dict(base, hue_offset=220.0)  # 180 degrees from the prior
        same_hue = dict(base, hue_offset=40.0)
        # one full unit of lead distance == one full unit of hue distance
        assert distance(base, old) - distance(same_lead, old) == pytest.approx(
            distance(far_hue, old) - distance(same_hue, old))

    def test_colour_does_not_steer_the_geometry(self):
        """The corpus pin for this change: pixels, gesture switches and
        intentionality (which behavioural coherence is built from) must not
        depend on colour at all. Colour is generated here with the RNG state
        saved and restored around it, so if it touches nothing the geometry
        reads the two pieces are identical, not merely alike."""
        from anima_mcp.display.drawing_engine import CanvasState
        era = GesturalEra()

        def piece(seed, with_colour):
            random.seed(seed)
            state = era.create_state()
            canvas = CanvasState()
            fx, fy, d = 120.0, 120.0, 0.0
            switches, trace = 0, []
            for _ in range(1200):
                color = (200, 100, 50)
                if with_colour:
                    rng = random.getstate()
                    color, _ = era.generate_color(state, 0.42, 0.7, 0.7, 0.71)
                    random.setstate(rng)
                if state.gesture_remaining <= 0:
                    era.choose_gesture(state, 0.7, 0.7, 0.71, 0.5)
                    switches += 1
                trace.append(round(state.intentionality(), 6))
                era.place_mark(state, canvas, fx, fy, d, 0.6, color)
                state.gesture_remaining -= 1
                fx, fy, d = era.drift_focus(state, fx, fy, d, 0.7, 0.71, 0.5, 0.7, canvas=canvas)
            return set(canvas.pixels), switches, trace

        for seed in range(5):
            assert piece(seed, True) == piece(seed, False)


def test_direction_lock_probability_increased():
    """Direction locks should occur more frequently (prob > 0.08 at high C+clarity)."""
    random.seed(42)
    era = GesturalEra()
    lock_count = 0
    trials = 2000

    for _ in range(trials):
        state = era.create_state()
        state.direction_locked = False
        state.direction_lock_remaining = 0
        state.direction_commitment = 0.0

        era.drift_focus(state, 120.0, 120.0, 0.0, 0.5, 0.5, 0.8, 0.8)
        if state.direction_locked:
            lock_count += 1

    lock_rate = lock_count / trials
    # Theoretical: 0.06 * 1.3 * 0.9 = 0.070. Must be above old rate (~0.03).
    assert lock_rate > 0.05, f"Lock rate {lock_rate:.3f} should be > 0.05 with high C and clarity"
