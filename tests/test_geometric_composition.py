"""
A geometric piece has a vocabulary and an arrangement of its own (2026-11-12).

The operator's read was "kitchen sink", and it was exact. Measured through the
real engine before this change, over 120 pieces: a ~70-stamp geometric piece
used 15.1 of the 16 templates, its four commonest shapes carried 52% of the
stamps, and every stamp either took a shape-wide random step from the last or
jumped to the canvas centre — so sixteen kinds of form buried each other in one
heap. The 2026-09-17 emphasis (4 of 16 weights x2.0-3.5, the other 12 left at
full weight) had moved the census from 15.7 templates per piece to 15.3.

After: 6.7 templates per piece, the vocabulary carrying ~94% of the stamps, on
one of four arrangements drawn once per piece. These tests pin that, and pin
what it was not allowed to cost:

* nothing is favoured across the corpus (vocabulary and layout are uniform
  draws) and no template is unreachable inside a piece;
* clarity still means "tight vs scattered";
* piece LENGTH does not depend on how varied a piece's shapes are — the gate
  that ends every geometric piece, ``bailout_fatigue``, reads fatigue, and
  fatigue used to accrue only when a stamp's template NAME changed.
"""

import math
import random
import statistics
from collections import Counter
from unittest.mock import patch

import pytest

from anima_mcp.display.drawing_engine import DrawingEngine
from anima_mcp.display.drawing_state import CanvasState
from anima_mcp.display.eras import get_era
from anima_mcp.display.eras.geometric import (
    ACCENT_MAX,
    LAYOUT_MARGIN,
    LAYOUTS,
    VOCABULARY_MAX,
    VOCABULARY_MIN,
    GeometricEra,
    GeometricState,
    make_layout,
)

# Geometric's lived length: bailout_fatigue lands near 70 stamps.
PIECE_STAMPS = 70


def _census(state, stamps=PIECE_STAMPS, clarity=0.7, stability=0.8):
    era = get_era("geometric")
    counts = Counter()
    for _ in range(stamps):
        era.choose_gesture(state, clarity, stability, 0.7, 0.5)
        counts[state.gesture] += 1
    return counts


class TestAPieceHasAVocabulary:

    def test_the_vocabulary_carries_the_piece(self):
        """The defect was that it did not: 52% from the top four of sixteen."""
        era = get_era("geometric")
        shares, distinct = [], []
        for seed in range(120):
            random.seed(seed)
            state = era.create_state()
            counts = _census(state)
            shares.append(sum(counts[g] for g in state.emphasis) / PIECE_STAMPS)
            distinct.append(len(counts))
        assert statistics.mean(shares) > 0.85, statistics.mean(shares)
        # 15.1 before. A vocabulary of 2-4 plus a few strays lands near 6-7.
        assert statistics.mean(distinct) < 9, statistics.mean(distinct)

    def test_the_bare_state_is_the_old_alphabet(self):
        """A `GeometricState()` with no vocabulary is exactly the pre-change
        control — the whole alphabet — which is what makes the comparison
        above a comparison and not a description."""
        distinct = []
        for seed in range(60):
            random.seed(seed)
            distinct.append(len(_census(GeometricState())))
        assert statistics.mean(distinct) > 14

    def test_vocabulary_size_is_bounded_and_varies(self):
        era = get_era("geometric")
        sizes = set()
        for seed in range(200):
            random.seed(seed)
            state = era.create_state()
            assert VOCABULARY_MIN <= len(state.emphasis) <= VOCABULARY_MAX
            assert len(set(state.emphasis)) == len(state.emphasis)
            assert 0.0 <= state.accent <= ACCENT_MAX
            sizes.add(len(state.emphasis))
        assert sizes == set(range(VOCABULARY_MIN, VOCABULARY_MAX + 1))

    def test_no_template_is_favoured_across_the_corpus(self):
        """The vocabulary is a UNIFORM sample, so the narrowing is per piece and
        never a narrowing of what Lumen draws overall. A biased sample would
        quietly retire templates from the era."""
        era = get_era("geometric")
        shapes = GeometricState().gestures()
        membership = Counter()
        pieces = 4000
        for seed in range(pieces):
            random.seed(seed)
            membership.update(era.create_state().emphasis)
        expected = pieces * ((VOCABULARY_MIN + VOCABULARY_MAX) / 2) / len(shapes)
        for shape in shapes:
            assert abs(membership[shape] - expected) / expected < 0.15, (
                shape, membership[shape], expected)

    def test_embodied_weighting_still_acts_inside_the_vocabulary(self):
        """Clarity and stability steered shape choice before; they must still.
        With `organic` and `rectangle` both in the vocabulary, low stability
        favours the organic form and high stability the structured one."""
        def organic_share(stability):
            random.seed(3)
            state = GeometricState()
            state.emphasis = ["organic", "rectangle"]
            state.accent = 0.0
            counts = _census(state, stamps=4000, stability=stability)
            return counts["organic"] / 4000

        assert organic_share(0.3) > organic_share(0.8) + 0.15

    def test_a_strict_piece_never_strays(self):
        random.seed(5)
        state = GeometricState()
        state.emphasis = ["rings", "line"]
        state.accent = 0.0
        assert set(_census(state, stamps=500)) == {"rings", "line"}


class TestAPieceHasAnArrangement:

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_anchors_stay_on_the_canvas(self, layout):
        for seed in range(200):
            random.seed(seed)
            anchors = make_layout(layout)
            assert anchors, layout
            for x, y in anchors:
                assert LAYOUT_MARGIN <= x <= 240 - LAYOUT_MARGIN
                assert LAYOUT_MARGIN <= y <= 240 - LAYOUT_MARGIN

    def test_an_unknown_layout_is_no_arrangement_not_a_guess(self):
        assert make_layout("") == []
        assert make_layout("spiral-of-doom") == []

    def test_every_layout_is_drawn_and_none_is_favoured(self):
        era = get_era("geometric")
        seen = Counter()
        for seed in range(2000):
            random.seed(seed)
            seen[era.create_state().layout] += 1
        assert set(seen) == set(LAYOUTS)
        for layout in LAYOUTS:
            assert abs(seen[layout] - 500) < 90, seen

    def test_a_band_is_a_line(self):
        """Unclamped interior anchors are collinear — it reads as a band
        because it is one."""
        random.seed(2)
        for _ in range(50):
            pts = make_layout("band")
            (x0, y0), (x1, y1) = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
            length = math.hypot(x1 - x0, y1 - y0)
            assert length > 0
            inner = [p for p in pts
                     if LAYOUT_MARGIN < p[0] < 240 - LAYOUT_MARGIN
                     and LAYOUT_MARGIN < p[1] < 240 - LAYOUT_MARGIN]
            for x, y in inner:
                off = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / length
                assert off < 1e-6

    def test_constellation_groups_keep_apart(self):
        """Best-candidate placement, so the groups are separated relative to
        each other without a minimum-distance constant."""
        gaps = []
        for seed in range(200):
            random.seed(seed)
            pts = make_layout("constellation")
            gaps.append(min(
                math.hypot(a[0] - b[0], a[1] - b[1])
                for i, a in enumerate(pts) for b in pts[i + 1:]))
        # Uniform placement of 3-6 points in this square gives a median nearest
        # gap near 30px; best-of-12 roughly doubles it.
        assert statistics.median(gaps) > 45, statistics.median(gaps)

    def test_stamps_land_on_the_arrangement(self):
        era = get_era("geometric")
        for layout in LAYOUTS:
            random.seed(9)
            state = GeometricState()
            state.layout = layout
            state.anchors = make_layout(layout)
            dists = []
            for _ in range(400):
                x, y, _d = era.drift_focus(state, 120, 120, 0.0, 0.8, 0.7, 0.5, 0.9)
                dists.append(min(math.hypot(x - ax, y - ay) for ax, ay in state.anchors))
            # Jitter sd at clarity 0.9 is 19/sqrt(n) px; the mean radial miss
            # of a 2-D gaussian is ~1.25 sd.
            assert statistics.mean(dists) < 30, (layout, statistics.mean(dists))

    def test_clarity_still_means_tight_versus_scattered(self):
        """drift_focus's contract: 'clarity: higher = tighter focus, lower =
        more scattered'. The arrangement changed what it is tight AROUND, not
        that."""
        era = get_era("geometric")

        def miss(clarity, seed):
            random.seed(seed)
            state = GeometricState()
            state.anchors = make_layout("constellation")
            total = 0.0
            for _ in range(300):
                x, y, _d = era.drift_focus(state, 120, 120, 0.0, 0.8, 0.7, 0.5, clarity)
                total += min(math.hypot(x - ax, y - ay) for ax, ay in state.anchors)
            return total / 300

        for seed in range(20):
            assert miss(0.55, seed) > miss(0.95, seed)

    def test_a_state_with_no_arrangement_keeps_the_old_drift(self):
        """Fail toward the previous behavior: a bare state (or one restored
        without anchors) random-walks at high clarity exactly as before."""
        era = get_era("geometric")
        state = GeometricState()
        random.seed(4)
        x, y, _d = era.drift_focus(state, 100.0, 100.0, 0.0, 0.8, 0.7, 0.5, 0.9)
        random.seed(4)
        spread = 15 + (1.0 - 0.9) * 40
        assert x == pytest.approx(100.0 + random.gauss(0, spread))
        assert y == pytest.approx(100.0 + random.gauss(0, spread))

    def test_pieces_no_longer_share_one_footprint(self):
        """The heap was the same heap every time. Different arrangements put
        the work in different places: the spread of each piece's centre of mass
        is the between-piece variety a viewer actually sees."""
        era = get_era("geometric")

        def centres(with_layout):
            out = []
            for seed in range(80):
                random.seed(seed)
                state = era.create_state()
                if not with_layout:
                    state.anchors = []
                canvas = CanvasState()
                fx, fy, d = 120.0, 120.0, 0.0
                for _ in range(PIECE_STAMPS):
                    era.choose_gesture(state, 0.55, 0.8, 0.7, 0.5)
                    era.place_mark(state, canvas, fx, fy, d, 0.6, (200, 100, 50))
                    # clarity 0.55 takes the old code's "jump around centre"
                    # branch, the one that always heaped mid-canvas.
                    fx, fy, d = era.drift_focus(state, fx, fy, d, 0.8, 0.7, 0.5, 0.55)
                xs = [p[0] for p in canvas.pixels]
                ys = [p[1] for p in canvas.pixels]
                out.append((statistics.mean(xs), statistics.mean(ys)))
            return (statistics.pstdev(c[0] for c in out)
                    + statistics.pstdev(c[1] for c in out))

        assert centres(True) > 2.0 * centres(False)

    def test_disposition_reports_what_was_drawn(self):
        random.seed(1)
        state = get_era("geometric").create_state()
        d = state.disposition()
        assert d["emphasis"] == state.emphasis
        assert d["layout"] in LAYOUTS
        assert d["anchor_count"] == len(state.anchors)
        assert 0.0 <= d["accent"] <= ACCENT_MAX

    def test_a_pre_layout_history_biases_nothing(self):
        """Dispositions banked before 2026-11-12 carry no `layout` (and an
        `emphasis_weight` nothing reads now). They must compare as equally
        distant from every candidate layout — no fabricated preference."""
        era = get_era("geometric")
        old = [{"emphasis": ["circle", "rings", "line", "wave"],
                "emphasis_weight": 2.7, "hue_pos": 0.5, "hue_span_frac": 0.6}]
        seen = Counter()
        for seed in range(1200):
            random.seed(seed)
            seen[era.create_state(old).layout] += 1
        for layout in LAYOUTS:
            assert abs(seen[layout] - 300) < 75, seen


class TestPieceLengthDoesNotDependOnVariety:
    """`bailout_fatigue` ends every geometric piece, and fatigue accrued per
    gesture switch DETECTED BY NAME. While every piece sampled all 16 templates
    that was a switch on ~91% of stamps. Give a piece a vocabulary and the name
    repeats — measured through the engine, 70% — so pieces ran ~25% longer
    (68 -> 85 stamps) purely because their shapes agreed with each other.

    A stamp era now counts every stamp as the gesture it is. Measured over 120
    pieces each: median 71 stamps before this change, 67 after."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("anima_mcp.display.drawing_engine._get_canvas_path",
                   return_value=tmp_path / "canvas.json"), \
                patch.object(CanvasState, "save_to_disk", lambda self: True):
            eng = DrawingEngine(db_path=str(tmp_path / "t.db"), identity_store=None)
            eng.active_era = get_era("geometric")
            yield eng

    def _fatigue_after(self, engine, vocabulary, stamps=40):
        state = engine.intent.state
        state.fatigue = 0.0
        state.engagement = 0.4
        state.gesture_history.clear()
        era_state = GeometricState()
        era_state.emphasis = list(vocabulary)
        era_state.accent = 0.0
        engine.intent.era_state = era_state
        for _ in range(stamps):
            engine.active_era.choose_gesture(era_state, 0.7, 0.8, 0.7, 0.5)
            state.gesture_history.append(era_state.gesture)
            history = state.gesture_history
            switch = len(history) >= 2 and history[-1] != history[-2]
            if getattr(engine.active_era, "each_mark_is_a_gesture", False):
                switch = len(history) >= 2
            engine._update_attention(0.6, 0.5, 0.45, switch)
        return state.fatigue

    def test_the_era_declares_itself(self):
        assert GeometricEra.each_mark_is_a_gesture is True
        for name in ("gestural", "pointillist", "field", "resonance"):
            assert not getattr(get_era(name), "each_mark_is_a_gesture", False), name

    def test_one_shape_tires_as_fast_as_sixteen(self, engine):
        random.seed(0)
        narrow = self._fatigue_after(engine, ["circle"])
        random.seed(0)
        wide = self._fatigue_after(engine, GeometricState().gestures())
        assert narrow == pytest.approx(wide, rel=1e-9)

    def test_the_engine_reads_the_declaration(self, engine):
        """Drive the real `draw()` with a one-shape vocabulary. By-name
        detection would see no switch at all and fatigue would crawl at the
        base rate; the declaration makes it climb as a stamp era's must."""
        from anima_mcp.anima import Anima

        random.seed(1)
        era_state = GeometricState()
        era_state.emphasis = ["circle"]
        era_state.accent = 0.0
        era_state.anchors = make_layout("constellation")
        engine.intent.era_state = era_state
        engine.intent.state.fatigue = 0.0
        anima = Anima(warmth=0.5, clarity=0.8, stability=0.8, presence=0.7,
                      readings=None)
        start = engine.intent.mark_count
        for _ in range(200000):
            engine.draw(anima)
            if engine.intent.mark_count - start >= 30:
                break
        assert engine.intent.mark_count - start >= 30
        assert set(engine.intent.state.gesture_history) == {"circle"}
        # 29 switches x 0.006 x fatigue_rate 2.0 = 0.348 from switches alone;
        # by-name detection would have left only the base term (< 0.08).
        assert engine.intent.state.fatigue > 0.30, engine.intent.state.fatigue
