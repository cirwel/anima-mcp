"""A drawing's stated coverage intention has to steer the drawing.

`coverage_target` was generated per piece from clarity, described to the
operator, persisted since #128 — and read by nothing. Measured over the
instrumented corpus that showed up exactly as you would expect: within every
era, "sparse" and "balanced" pieces land at the same density (gestural 9.4% vs
10.1%, pointillist 2.26% vs 2.13%, resonance 14.8% vs 16.0%), while the spread
BETWEEN eras is 1.4%-23.9%. Era decided everything; the intention decided
nothing. Its vocabulary was also one word short: `dense` needs clarity < 0.30
against a lived range of 0.454-0.910, so it had never once been generated.

These tests pin both halves — the bias that makes the intention real, and the
calibration plumbing that makes its vocabulary reachable.
"""
from __future__ import annotations

import json
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_mcp.display.drawing_engine import (
    COVERAGE_BIAS_MARGIN,
    COVERAGE_BIAS_STRENGTH,
    CanvasState,
    DrawingEngine,
    DrawingGoal,
    _DEFAULT_COVERAGE_CUTS,
    _coverage_cuts,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "derive_drawing_thresholds.py"


def _engine(coverage_target, canvas=None):
    """An engine stub carrying just what _apply_coverage_bias reads.

    Built without __init__ deliberately: the bias is a pure function of the
    goal and the canvas grid, and standing up a real engine would drag a DB
    and an identity store into a geometry test.
    """
    eng = DrawingEngine.__new__(DrawingEngine)
    eng.canvas = canvas if canvas is not None else CanvasState()
    eng.drawing_goal = (DrawingGoal(coverage_target=coverage_target)
                        if coverage_target else None)
    return eng


class _Gesture:
    """Minimal era_state stand-in — the bias only reads gesture_remaining."""

    def __init__(self, remaining=0):
        self.gesture_remaining = remaining


def _fill_cell(canvas, gx, gy, n):
    for i in range(n):
        canvas.draw_pixel(gx * 30 + (i % 30), gy * 30 + (i // 30) % 30, (1, 2, 3))


# ---------------------------------------------------------------------------
# The bias itself
# ---------------------------------------------------------------------------

class TestCoverageBias:

    def test_balanced_moves_nothing(self):
        """`balanced` must be byte-identical to the pre-change behavior — half
        the corpus keeps its old dynamics, which is what makes the A/B honest."""
        canvas = CanvasState()
        _fill_cell(canvas, 0, 0, 200)
        eng = _engine("balanced", canvas)
        assert eng._apply_coverage_bias(120.0, 120.0, _Gesture()) == (120.0, 120.0)

    def test_no_goal_moves_nothing(self):
        eng = _engine(None)
        _fill_cell(eng.canvas, 0, 0, 200)
        assert eng._apply_coverage_bias(120.0, 120.0, _Gesture()) == (120.0, 120.0)

    def test_sparse_leans_toward_the_emptiest_region(self):
        canvas = CanvasState()
        # Everything painted except the far corner: one unambiguous minimum.
        for gx in range(8):
            for gy in range(8):
                if (gx, gy) != (7, 7):
                    _fill_cell(canvas, gx, gy, 10)
        eng = _engine("sparse", canvas)
        fx, fy = eng._apply_coverage_bias(120.0, 120.0, _Gesture())
        assert fx > 120.0 and fy > 120.0  # toward cell (7,7) at (225, 225)

    def test_dense_leans_toward_the_fullest_region(self):
        canvas = CanvasState()
        for gx in range(8):
            for gy in range(8):
                _fill_cell(canvas, gx, gy, 5)
        _fill_cell(canvas, 0, 0, 400)  # one unambiguous maximum
        eng = _engine("dense", canvas)
        fx, fy = eng._apply_coverage_bias(120.0, 120.0, _Gesture())
        assert fx < 120.0 and fy < 120.0  # toward cell (0,0) at (15, 15)

    def test_mid_stroke_is_left_alone(self):
        """Each era's character lives in its sustained gestures — gestural locks
        direction for 15-45 marks to get long lines. A per-mark positional pull
        would bow those into arcs, so the bias waits for a gesture boundary."""
        canvas = CanvasState()
        for gx in range(8):
            for gy in range(8):
                if (gx, gy) != (7, 7):
                    _fill_cell(canvas, gx, gy, 10)
        eng = _engine("sparse", canvas)
        assert eng._apply_coverage_bias(120.0, 120.0, _Gesture(remaining=7)) == (120.0, 120.0)

    def test_empty_grid_carries_no_direction(self):
        eng = _engine("sparse")
        assert eng.canvas.coverage_bias_cell("sparse") is None
        assert eng._apply_coverage_bias(120.0, 120.0, _Gesture()) == (120.0, 120.0)

    def test_bias_is_a_fraction_not_a_teleport(self):
        canvas = CanvasState()
        for gx in range(8):
            for gy in range(8):
                if (gx, gy) != (7, 7):
                    _fill_cell(canvas, gx, gy, 10)
        eng = _engine("sparse", canvas)
        fx, _ = eng._apply_coverage_bias(120.0, 120.0, _Gesture())
        expected = 120.0 + (225 - 120.0) * COVERAGE_BIAS_STRENGTH
        assert fx == pytest.approx(expected)

    def test_clamped_inside_the_widest_era_margin(self):
        """Era focus margins run 15-25px. Repeated leaning must never park the
        focus in a band some era treats as off-canvas."""
        canvas = CanvasState()
        for gx in range(8):
            for gy in range(8):
                _fill_cell(canvas, gx, gy, 5)
        _fill_cell(canvas, 0, 0, 400)
        eng = _engine("dense", canvas)
        fx, fy = 120.0, 120.0
        for _ in range(200):
            fx, fy = eng._apply_coverage_bias(fx, fy, _Gesture())
        assert fx >= COVERAGE_BIAS_MARGIN and fy >= COVERAGE_BIAS_MARGIN

    def test_ties_do_not_collapse_to_the_top_left(self):
        """sparsest_cell() scans in fixed order and resolves ties to the first
        minimum, so on a mostly-empty grid it always answers (0,0). Reused for a
        per-gesture bias that would drag every `sparse` piece into the top-left
        corner and call it an intention. Ties must break uniformly."""
        canvas = CanvasState()
        _fill_cell(canvas, 4, 4, 50)  # 63 cells tied at zero
        assert canvas.sparsest_cell() == (0, 0)  # the old behavior, documented
        random.seed(11)
        picks = {canvas.coverage_bias_cell("sparse") for _ in range(200)}
        assert len(picks) > 20, f"tie-breaking collapsed to {picks}"
        assert (4, 4) not in picks


# ---------------------------------------------------------------------------
# What the bias does to a real era over a real piece
# ---------------------------------------------------------------------------

def _run_piece(era, coverage_target, seed, marks=600):
    """Mirror the engine's mark loop closely enough to measure composition."""
    random.seed(seed)
    canvas = CanvasState()
    eng = _engine(coverage_target, canvas)
    state = era.create_state()
    fx, fy, direction = 120.0, 120.0, 0.0
    for _ in range(marks):
        if state.gesture_remaining <= 0:
            era.choose_gesture(state, 0.7, 0.8, 0.7, 0.5)
        era.place_mark(state, canvas, fx, fy, direction, 0.6, (200, 100, 50))
        state.gesture_remaining -= 1
        fx, fy, direction = era.drift_focus(
            state, fx, fy, direction, 0.8, 0.7, 0.5, 0.7, canvas=canvas)
        fx, fy = eng._apply_coverage_bias(fx, fy, state)
    return canvas


SEEDS = range(32)
# Marks per piece, from the live per-era medians at completion (2026-08-22).
PIECE_MARKS = {"gestural": 1200, "pointillist": 1200, "field": 450,
               "resonance": 750, "geometric": 70}


def _mean_entropy(era_name, target):
    from anima_mcp.display.eras import get_era
    marks = PIECE_MARKS[era_name]
    vals = [_run_piece(get_era(era_name), target, s, marks).grid_entropy()
            for s in SEEDS]
    return sum(vals) / len(vals)


def _paired_gap(era_name, target, against="balanced"):
    """Mean of the PER-SEED entropy difference, not the difference of means.

    `_run_piece` seeds the RNG, so the same seed gives both runs the same era
    disposition and the same drift — the comparison is naturally paired, and
    pairing removes the between-piece variance that has nothing to do with the
    intention. Unpaired means needed that variance to be small; since eras
    draw a per-piece disposition (2026-09-17) it no longer is, and it was
    never small for `geometric`, whose 70 whole-shape stamps swing entropy far
    more than any bias does.

    This is why `test_sparse_spreads[geometric]` passed for a month on a gap it
    did not have: at 16 unpaired seeds it was +0.005 against a balanced spread
    of 0.062 — under a tenth of a standard deviation. Measured paired over 64
    seeds, on the code as it stood BEFORE dispositions existed, geometric's
    sparse direction is -0.006 and wins 27 of 64 — no effect, and slightly the
    wrong way. Pairing makes the test able to say so.
    """
    from anima_mcp.display.eras import get_era
    era = get_era(era_name)
    marks = PIECE_MARKS[era_name]
    diffs = [
        _run_piece(era, target, s, marks).grid_entropy()
        - _run_piece(era, against, s, marks).grid_entropy()
        for s in SEEDS
    ]
    return sum(diffs) / len(diffs)


class TestIntentionChangesTheComposition:
    """The claim this PR has to earn: within one era, the same creature drawing
    under different intentions produces measurably different compositions.

    Measured on grid_entropy — which #128 already records per piece AND per
    300s sample, so the identical statistic validates this on the Pi without
    new instrumentation. occupied_cells is deliberately NOT the metric: gestural
    reaches ~60 of 64 cells whatever it intends, so cell count cannot see the
    difference there even when entropy can.

    Measured effect over 20 seeds, quoted so nobody has to re-derive it:

        era          sparse           balanced         dense
        gestural     0.918 +- .010    0.924 +- .010    0.902 +- .015
        pointillist  0.750 +- .020    0.734 +- .020    0.635 +- .029
        field        0.711 +- .061    0.616 +- .087    0.612 +- .097
        resonance    0.901 +- .018    0.872 +- .051    0.850 +- .028
        geometric    0.816 +- .029    0.811 +- .062    0.697 +- .030
    """

    @pytest.mark.parametrize("era_name", list(PIECE_MARKS))
    def test_dense_concentrates(self, era_name):
        """`dense` must lower entropy everywhere it has room to — the one era
        that already concentrates (field) is exempted below, on purpose."""
        if era_name == "field":
            pytest.skip("field already concentrates; see test_the_two_no_ops")
        assert _paired_gap(era_name, "dense") < 0

    @pytest.mark.parametrize("era_name", list(PIECE_MARKS))
    def test_sparse_spreads(self, era_name):
        if era_name in ("gestural", "geometric"):
            pytest.skip(f"{era_name} does not respond to sparse; see test_the_no_ops")
        assert _paired_gap(era_name, "sparse") > 0

    def test_the_no_ops_are_real_and_must_not_be_tuned_away(self):
        """Three era/direction pairs do not respond, and all three are correct.

        Gestural sweeps the whole canvas by construction (~60 of 64 cells),
        so there is nothing left for `sparse` to open up. Field already works
        in a tight region, so `dense` has nothing left to concentrate.
        Geometric stamps ~70 whole shapes over a whole piece, so a bias that
        leans each successive gesture boundary has very few boundaries to lean
        — `dense` still lands (the stamps pile up), `sparse` does not.

        ⚠️ Geometric's `sparse` was asserted as a real effect from 2026-08-22
        until 2026-09-17 and never was one. Measured paired over 64 seeds on
        the pre-disposition code: -0.006, winning 27 of 64 — no effect, and
        slightly the wrong way. The original 16-seed unpaired sample read
        +0.005 against a balanced spread of 0.062 and called it a pass. It is
        recorded here rather than quietly deleted: a test that passed by luck
        for a month is a fact about this suite worth keeping.

        Each era still responds to at least one direction; none responds to
        neither.

        This is pinned as a TEST, not a comment, because the tempting fix is to
        raise COVERAGE_BIAS_STRENGTH until every cell of the table moves — which
        buys those two pairs by turning `dense` into a single-cell blob in the
        eras that already work (measured: strength 0.6 puts 38% of a
        pointillist piece's pixels in one cell of 64). The intention is a bias
        on an era, not an override of it.
        """
        assert abs(_paired_gap("gestural", "sparse")) < 0.03
        assert abs(_paired_gap("field", "dense")) < 0.03
        assert abs(_paired_gap("geometric", "sparse")) < 0.03

        # ...and each era must still answer to SOMETHING, or the intention is
        # decorative for it.
        assert _paired_gap("geometric", "dense") < -0.05
        assert _paired_gap("gestural", "dense") < 0
        assert _paired_gap("field", "sparse") > 0


# ---------------------------------------------------------------------------
# Reachable vocabulary: the calibration plumbing
# ---------------------------------------------------------------------------

class TestCoverageCuts:

    def test_defaults_match_the_historical_constants(self):
        """A fresh install must generate goals exactly as before this existed."""
        assert _DEFAULT_COVERAGE_CUTS["COVERAGE_DENSE_BELOW"] == 0.30
        assert _DEFAULT_COVERAGE_CUTS["COVERAGE_SPARSE_ABOVE"] == 0.70

    def test_same_clarity_two_calibrations_two_intentions(self, monkeypatch):
        """The point of the change: clarity 0.66 is `balanced` on the fleet
        default and `dense` against Lumen's own measured range, where 0.66 sits
        in its foggiest third."""
        import anima_mcp.config as config_mod

        class _Cal:
            drawing_thresholds = {}

        monkeypatch.setattr(config_mod, "get_calibration", lambda: _Cal())
        assert DrawingGoal.from_state(0.5, 0.66).coverage_target == "balanced"

        _Cal.drawing_thresholds = {"COVERAGE_DENSE_BELOW": 0.6935,
                                   "COVERAGE_SPARSE_ABOVE": 0.7326}
        assert DrawingGoal.from_state(0.5, 0.66).coverage_target == "dense"
        assert DrawingGoal.from_state(0.5, 0.71).coverage_target == "balanced"
        assert DrawingGoal.from_state(0.5, 0.80).coverage_target == "sparse"

    def test_broken_calibration_fails_open(self, monkeypatch):
        import anima_mcp.config as config_mod
        monkeypatch.setattr(config_mod, "get_calibration",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert _coverage_cuts() == (0.30, 0.70)

    def test_garbage_value_keeps_the_good_half(self, monkeypatch):
        import anima_mcp.config as config_mod

        class _Cal:
            drawing_thresholds = {"COVERAGE_DENSE_BELOW": "not-a-number",
                                  "COVERAGE_SPARSE_ABOVE": 0.75}

        monkeypatch.setattr(config_mod, "get_calibration", lambda: _Cal())
        assert _coverage_cuts() == (0.30, 0.75)

    def test_non_monotone_pair_is_rejected_whole(self, monkeypatch):
        """Cuts that cross would starve `balanced` exactly the way the built-in
        0.30 starved `dense` — one dead word traded for another."""
        import anima_mcp.config as config_mod

        class _Cal:
            drawing_thresholds = {"COVERAGE_DENSE_BELOW": 0.80,
                                  "COVERAGE_SPARSE_ABOVE": 0.60}

        monkeypatch.setattr(config_mod, "get_calibration", lambda: _Cal())
        assert _coverage_cuts() == (0.30, 0.70)


# ---------------------------------------------------------------------------
# The derivation script
# ---------------------------------------------------------------------------

class TestDerivationScript:

    def _make_db(self, tmp_path, n=600, spread=0.2):
        db = tmp_path / "anima.db"
        con = sqlite3.connect(db)
        con.execute("create table drawing_records (timestamp text, clarity real)")
        base = datetime.now() - timedelta(days=10)
        rows = [((base + timedelta(minutes=5 * i)).isoformat(timespec="seconds"),
                 0.7 + spread * ((i % 21) - 10) / 10.0) for i in range(n)]
        con.executemany("insert into drawing_records values (?,?)", rows)
        con.commit()
        return db

    def _run(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, timeout=60)

    def test_derives_tertiles_that_leave_all_three_words_reachable(self, tmp_path):
        db = self._make_db(tmp_path)
        r = self._run("--db", str(db), "--days", "90")
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        lo, hi = out["COVERAGE_DENSE_BELOW"], out["COVERAGE_SPARSE_ABOVE"]
        assert lo < hi
        # Roughly a third of the sample must fall on each side, or the word is
        # decorative — the defect this script exists to fix.
        vals = [0.7 + 0.2 * ((i % 21) - 10) / 10.0 for i in range(600)]
        assert 0.2 < sum(v < lo for v in vals) / len(vals) < 0.45
        assert 0.2 < sum(v > hi for v in vals) / len(vals) < 0.45

    def test_refuses_a_sliver_of_data(self, tmp_path):
        r = self._run("--db", str(self._make_db(tmp_path, n=50)))
        assert r.returncode != 0
        assert "refusing" in r.stderr

    def test_refuses_a_degenerate_range(self, tmp_path):
        """Clarity pinned flat collapses the tertiles onto each other."""
        r = self._run("--db", str(self._make_db(tmp_path, spread=0.0)))
        assert r.returncode != 0
        assert "degenerate" in r.stderr

    def test_apply_edits_config_atomically_with_backup(self, tmp_path):
        db = self._make_db(tmp_path)
        cfg = tmp_path / "anima_config.json"
        cfg.write_text(json.dumps({"nervous_system": {"cpu_temp_min": 40.0}}))
        r = self._run("--db", str(db), "--apply", str(cfg))
        assert r.returncode == 0, r.stderr
        saved = json.loads(cfg.read_text())
        assert "COVERAGE_DENSE_BELOW" in saved["nervous_system"]["drawing_thresholds"]
        assert saved["nervous_system"]["cpu_temp_min"] == 40.0  # untouched
        assert list(tmp_path.glob("anima_config.json.bak-drawing-*"))
