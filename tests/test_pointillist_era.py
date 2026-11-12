"""Pointillist colour: a zone holds a hue, and a piece keeps its anchor.

The era's only per-piece character was `color_anchor_hue`, and it could not be
seen. Two things erased it (measured 2026-11-12 over rendered pieces):

* every DOT drew its own role — anchor 45%, complement 30%, split-complement
  25% — so all four hues of the scheme were spread evenly through every zone.
  Those four sit 0/120/180/240 degrees round the wheel, so whatever the anchor
  the piece covered the whole wheel: hue resultant length 0.03, the lowest
  possible reading of "has a dominant hue".
* the anchor itself random-walked 2 degrees a mark, ~75 degrees over a piece.

So `draw_distinct` was choosing between pieces nobody could tell apart. Now a
zone draws its role once, the anchor is tethered to the piece's own, and a
per-piece share of each zone's dots take the zone hue's complement — the
side-by-side pairs the era is named for.

Colour only: nothing here may move a pixel, a gesture switch or coherence.
"""
from __future__ import annotations

import colorsys
import json
import math
import random
import statistics

import pytest

from anima_mcp.display.drawing_engine import CanvasState
from anima_mcp.display.eras import pointillist as mod
from anima_mcp.display.eras.pointillist import (
    COMPLEMENT_SHARE_MAX,
    COMPLEMENT_SHARE_MIN,
    PointillistEra,
    PointillistState,
    draw_zone_role,
)

# Lumen's live median marks per pointillist piece (2026-08-22).
PIECE_MARKS = 1200


def _hue(color):
    return colorsys.rgb_to_hsv(*[c / 255 for c in color])[0] * 360.0


def _gap(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _resultant(hues, harmonic=1):
    """Mean resultant length of a set of hues, 0..1. 0 = spread evenly round
    the wheel, 1 = one hue. `harmonic=2` folds a hue onto its complement, so it
    measures concentration on an AXIS (a complementary pair counts as one)."""
    x = sum(math.cos(math.radians(harmonic * h)) for h in hues) / len(hues)
    y = sum(math.sin(math.radians(harmonic * h)) for h in hues) / len(hues)
    return math.hypot(x, y)


def _piece(seed, marks=PIECE_MARKS):
    """One piece's dots in the engine's order, as (zone_index, hue, role)."""
    random.seed(seed)
    era = PointillistEra()
    state = era.create_state()
    fx, fy, d = 120.0, 120.0, 0.0
    zone, last = 0, (state.zone_center_x, state.zone_center_y)
    dots = []
    for _ in range(marks):
        color, _ = era.generate_color(state, 0.42, 0.7, 0.7, 0.71)
        if state.gesture_remaining <= 0:
            era.choose_gesture(state, 0.7, 0.7, 0.71, 0.47)
        state.gesture_remaining -= 1
        dots.append((zone, _hue(color), getattr(state, "zone_role", None)))
        fx, fy, d = era.drift_focus(state, fx, fy, d, 0.7, 0.71, 0.47, 0.7)
        here = (state.zone_center_x, state.zone_center_y)
        if here != last:
            zone, last = zone + 1, here
    return state, dots


def _zones(dots):
    zones = {}
    for zone, hue, role in dots:
        zones.setdefault(zone, []).append((hue, role))
    return list(zones.values())


class TestZoneHoldsAHue:

    def test_the_role_changes_only_when_the_zone_does(self):
        for seed in range(10):
            _, dots = _piece(seed)
            for zone in _zones(dots):
                assert len({role for _, role in zone}) == 1, (
                    f"seed {seed}: a zone changed hue role mid-zone")

    def test_roles_keep_the_scheme_the_era_always_had(self):
        """Anchor / complement / split-complement at 45/30/25 — the split is
        untouched, only where it is drawn moved."""
        random.seed(0)
        roles = [draw_zone_role() for _ in range(20000)]
        assert roles.count(0.0) / len(roles) == pytest.approx(0.45, abs=0.02)
        assert roles.count(180.0) / len(roles) == pytest.approx(0.30, abs=0.02)
        assert roles.count(120.0) / len(roles) == pytest.approx(0.125, abs=0.015)
        assert roles.count(-120.0) / len(roles) == pytest.approx(0.125, abs=0.015)

    def test_a_zone_reads_as_one_colour(self):
        """Within a zone the dots concentrate on its hue. Measured: 0.11
        before (every zone held the whole scheme), 0.59 after."""
        within = []
        for seed in range(20):
            _, dots = _piece(seed)
            within += [_resultant([h for h, _ in z]) for z in _zones(dots) if len(z) >= 20]
        assert statistics.mean(within) > 0.4


class TestPieceKeepsItsAnchor:

    def test_the_anchor_is_the_one_the_piece_was_given(self):
        """It used to random-walk ~75 degrees over a piece, so the disposition
        banked at the end was not the palette the piece began with."""
        for seed in range(10):
            random.seed(seed)
            given = PointillistEra().create_state().color_anchor_hue
            state, _ = _piece(seed)
            assert state.color_anchor_hue == given

    def test_the_wander_stays_on_its_tether(self):
        random.seed(5)
        era = PointillistEra()
        state = era.create_state()
        wander = []
        for _ in range(20000):
            era.generate_color(state, 0.42, 0.7, 0.7, 0.71)
            wander.append(state.anchor_wander)
        # 2 degrees a step at 0.98 retention settles near 10 degrees of spread;
        # an untethered walk would be ~280 degrees out by now.
        assert statistics.pstdev(wander) < 15.0
        assert abs(statistics.mean(wander)) < 5.0

    def test_a_piece_has_a_dominant_hue(self):
        """The disposition is visible now. Honest about how far: the scheme
        itself balances round the wheel (0.45 - 0.30 - 0.25/2 ~ 0.03), so the
        resultant rises through each piece's own ~19 zones rather than to
        near 1. Measured over 60 pieces: resultant 0.026 -> 0.11; on the
        complementary axis 0.36 -> 0.56."""
        first, axial = [], []
        for seed in range(40):
            _, dots = _piece(seed)
            hues = [h for _, h, _ in dots]
            first.append(_resultant(hues))
            axial.append(_resultant(hues, harmonic=2))
        assert statistics.mean(first) > 0.07
        assert statistics.mean(axial) > 0.48


class TestComplementShare:

    def test_it_is_drawn_once_reported_and_in_range(self):
        shares = set()
        for seed in range(12):
            random.seed(seed)
            d = PointillistEra().create_state().disposition()
            assert COMPLEMENT_SHARE_MIN <= d["complement_share"] <= COMPLEMENT_SHARE_MAX
            assert json.loads(json.dumps(d)) == d
            shares.add(d["complement_share"])
        assert len(shares) >= 8

    @pytest.mark.parametrize("share", [0.0, 0.1, 0.3])
    def test_that_share_of_a_zone_takes_the_complement(self, share):
        random.seed(2)
        era = PointillistEra()
        state = era.create_state()
        state.complement_share = share
        state.zone_role = 0.0
        partner = (state.color_anchor_hue + 180.0) % 360.0
        hues = [_hue(era.generate_color(state, 0.42, 0.7, 0.7, 0.71)[0]) for _ in range(6000)]
        got = sum(1 for h in hues if _gap(h, partner) < 60.0) / len(hues)
        assert got == pytest.approx(share, abs=0.03)

    def test_the_next_piece_differs_in_share_as_well_as_anchor(self):
        era = PointillistEra()

        def mean_gap(use_history):
            gaps = []
            for seed in range(300):
                random.seed(seed)
                prior = dict(era.create_state().disposition(), era="pointillist")
                nxt = era.create_state([prior] if use_history else ()).disposition()
                gaps.append(abs(nxt["complement_share"] - prior["complement_share"]))
            return statistics.mean(gaps)

        assert mean_gap(True) > mean_gap(False) * 1.1

    def test_a_prior_without_a_share_fabricates_no_preference(self, monkeypatch):
        """Pieces banked before 2026-11-12 carry only the anchor. Against one,
        candidates that differ only in share must score the same — a missing
        key is not a share of zero (or of anything)."""
        captured = {}

        def spy(make, distance, recent=(), tries=6):
            captured["distance"] = distance
            return make()

        monkeypatch.setattr(mod, "draw_distinct", spy)
        PointillistEra().create_state()
        distance = captured["distance"]
        old = {"era": "pointillist", "color_anchor_hue": 30.0}
        low = {"color_anchor_hue": 200.0, "complement_share": COMPLEMENT_SHARE_MIN}
        high = {"color_anchor_hue": 200.0, "complement_share": COMPLEMENT_SHARE_MAX}
        assert distance(low, old) == distance(high, old)
        for junk in (None, "n/a"):
            assert distance(low, dict(old, complement_share=junk)) == distance(low, old)
        # and with a share on record it does discriminate
        assert distance(high, dict(old, complement_share=COMPLEMENT_SHARE_MIN)) > \
            distance(low, dict(old, complement_share=COMPLEMENT_SHARE_MIN))


class TestCorpusDoesNotMove:

    def test_colour_does_not_steer_the_geometry(self):
        """Pixels, gesture switches and intentionality (which behavioural
        coherence is built from) must not depend on colour at all. Colour is
        generated with the RNG state saved and restored around it, so if it
        touches nothing the geometry reads the two pieces are identical, not
        merely alike."""
        era = PointillistEra()

        def piece(seed, with_colour):
            random.seed(seed)
            state = era.create_state()
            canvas = CanvasState()
            fx, fy, d = 120.0, 120.0, 0.0
            switches, trace = 0, []
            for _ in range(PIECE_MARKS):
                color = (200, 100, 50)
                if with_colour:
                    rng = random.getstate()
                    color, _ = era.generate_color(state, 0.42, 0.7, 0.7, 0.71)
                    random.setstate(rng)
                if state.gesture_remaining <= 0:
                    era.choose_gesture(state, 0.7, 0.7, 0.71, 0.47)
                    switches += 1
                trace.append(round(state.intentionality(), 6))
                era.place_mark(state, canvas, fx, fy, d, 0.6, color)
                state.gesture_remaining -= 1
                fx, fy, d = era.drift_focus(state, fx, fy, d, 0.7, 0.71, 0.47, 0.7, canvas=canvas)
            return set(canvas.pixels), switches, trace

        for seed in range(5):
            assert piece(seed, True) == piece(seed, False)

    def test_a_bare_state_still_paints(self):
        """`PointillistState()` — no share, anchor role — must not fail."""
        color, category = PointillistEra().generate_color(PointillistState(), 0.5, 0.7, 0.7, 0.7)
        assert len(color) == 3 and category in ("warm", "cool", "neutral")
