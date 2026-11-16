"""Tests for inner life drive system — mood relief dampening."""

from unittest.mock import MagicMock

import pytest

from anima_mcp.inner_life import InnerLife


def _make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5):
    a = MagicMock()
    a.warmth = warmth
    a.clarity = clarity
    a.stability = stability
    a.presence = presence
    return a


def _make_inner_life(temperament, drives=None):
    il = InnerLife.__new__(InnerLife)
    il._temperament = dict(temperament)
    il._drives = drives or {d: 0.0 for d in temperament}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {d: 0.0 for d in temperament}
    il._pending_events = []
    il._last_save = 0.0
    il._saturated_since = {d: None for d in temperament}
    il._last_request_at = {d: 0.0 for d in temperament}
    il._active_requests = {}
    return il


def test_elapsed_time_matches_equivalent_temperament_ticks():
    initial = {"warmth": 0.3, "clarity": 0.3, "stability": 0.3, "presence": 0.3}
    one_long_tick = _make_inner_life(initial)
    five_reference_ticks = _make_inner_life(initial)
    anima = _make_anima(warmth=0.8, clarity=0.8, stability=0.8, presence=0.8)

    one_long_tick.update(anima, anima, elapsed_seconds=10.0)
    for _ in range(5):
        five_reference_ticks.update(anima, anima, elapsed_seconds=2.0)

    for dimension in initial:
        assert one_long_tick._temperament[dimension] == pytest.approx(
            five_reference_ticks._temperament[dimension], abs=1e-12,
        )


def test_drive_dampened_when_mood_above_comfort():
    """Drive accumulation slows when current mood exceeds comfort threshold."""
    il = InnerLife.__new__(InnerLife)
    il._temperament = {"warmth": 0.30, "clarity": 0.70, "stability": 0.70, "presence": 0.70}
    il._drives = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._pending_events = []
    il._last_save = 0.0
    il._saturated_since = {d: None for d in il._drives}
    il._last_request_at = {d: 0.0 for d in il._drives}
    il._active_requests = {}

    # Mood warmth well above comfort (0.40) — should dampen accumulation
    anima_high_mood = _make_anima(warmth=0.60)
    il.update(anima_high_mood, anima_high_mood)
    drive_dampened = il._drives["warmth"]

    # Reset and test without dampening (mood below comfort)
    il._drives = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._temperament = {"warmth": 0.30, "clarity": 0.70, "stability": 0.70, "presence": 0.70}
    anima_low_mood = _make_anima(warmth=0.30)
    il.update(anima_low_mood, anima_low_mood)
    drive_undampened = il._drives["warmth"]

    assert drive_dampened < drive_undampened, (
        f"Dampened ({drive_dampened}) should be less than undampened ({drive_undampened})"
    )


def test_drive_still_accumulates_when_dampened():
    """Drive doesn't stop entirely — dampening floors at 10% rate."""
    il = InnerLife.__new__(InnerLife)
    il._temperament = {"warmth": 0.30, "clarity": 0.70, "stability": 0.70, "presence": 0.70}
    il._drives = {"warmth": 0.5, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._pending_events = []
    il._last_save = 0.0
    il._saturated_since = {d: None for d in il._drives}
    il._last_request_at = {d: 0.0 for d in il._drives}
    il._active_requests = {}

    # Even with high mood, drive should still inch up (10% floor)
    anima = _make_anima(warmth=0.80)
    il.update(anima, anima)

    assert il._drives["warmth"] > 0.5, (
        f"Drive should still accumulate (got {il._drives['warmth']})"
    )


def test_no_dampening_when_mood_below_comfort():
    """Mood below comfort threshold — no dampening, full accumulation rate."""
    il = InnerLife.__new__(InnerLife)
    il._temperament = {"warmth": 0.30, "clarity": 0.70, "stability": 0.70, "presence": 0.70}
    il._drives = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {"warmth": 0.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._pending_events = []
    il._last_save = 0.0
    il._saturated_since = {d: None for d in il._drives}
    il._last_request_at = {d: 0.0 for d in il._drives}
    il._active_requests = {}

    # Mood at 0.35 — below comfort (0.40), no dampening
    anima = _make_anima(warmth=0.35)
    il.update(anima, anima)
    drive_after = il._drives["warmth"]

    # Full rate: 0.003 * (1 + 0.10*2) = 0.0036
    assert drive_after > 0.003, f"Expected full accumulation, got {drive_after}"


def test_saturated_drive_decays_when_mood_good():
    """A saturated drive (1.0) with mood above comfort should not keep climbing."""
    il = InnerLife.__new__(InnerLife)
    il._temperament = {"warmth": 0.35, "clarity": 0.70, "stability": 0.70, "presence": 0.70}
    il._drives = {"warmth": 1.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._prev_drives = dict(il._drives)
    il._crossed_thresholds = {"warmth": 0.5, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
    il._pending_events = []
    il._last_save = 0.0
    il._saturated_since = {d: None for d in il._drives}
    il._last_request_at = {d: 0.0 for d in il._drives}
    il._active_requests = {}

    # Run 50 ticks with high mood — temperament should rise, drive should eventually decay
    anima = _make_anima(warmth=0.60)
    for _ in range(50):
        il.update(anima, anima)

    assert il._drives["warmth"] < 1.0, (
        f"Drive should have started decaying after 50 ticks of good mood, got {il._drives['warmth']}"
    )


# ==================== Self-relative comfort band ====================

from anima_mcp import inner_life as _il_mod  # noqa: E402


def _settle(il, value, ticks):
    """Feed `ticks` updates with temperament held at `value` on every dim."""
    for _ in range(ticks):
        il._temperament = {d: value for d in il._drives}
        il._observe_temperament(il._temperament)


def test_comfort_falls_back_to_absolute_before_enough_samples():
    il = _make_inner_life({d: 0.8 for d in ("warmth", "clarity", "stability", "presence")})
    c = il.comfort_threshold("clarity")
    assert c["source"] == "absolute_fallback"
    assert c["threshold"] == _il_mod.DRIVE_COMFORT["clarity"]


def test_clarity_drive_can_arise_against_its_own_band():
    """Lived case: clarity never fell below 0.45, so its drive was dead.

    Against its own band, a dip from a 0.80 baseline to 0.65 is longing.
    """
    dims = ("warmth", "clarity", "stability", "presence")
    il = _make_inner_life({d: 0.8 for d in dims})
    # A baseline that wanders 0.75-0.85 for an hour+.
    for i in range(_il_mod.COMFORT_MIN_SAMPLES + 200):
        v = 0.75 if i % 2 else 0.85
        il._observe_temperament({d: v for d in dims})
    c = il.comfort_threshold("clarity")
    assert c["source"] == "self_relative"
    assert 0.74 < c["threshold"] < 0.76  # mean 0.80 - 1 sigma (0.05)

    il._temperament = {"warmth": 0.8, "clarity": 0.65, "stability": 0.8, "presence": 0.8}
    low = _make_anima(warmth=0.8, clarity=0.65, stability=0.8, presence=0.8)
    il.update(low, low)
    assert il._drives["clarity"] > 0.0
    assert 0.65 > _il_mod.DRIVE_COMFORT["clarity"]  # the old cut would never have fired
    assert il._drives["warmth"] == 0.0


def test_min_sigma_keeps_tiny_noise_from_becoming_a_want():
    """Presence barely moves; a 0.005 wobble must not register as longing."""
    dims = ("warmth", "clarity", "stability", "presence")
    il = _make_inner_life({d: 0.71 for d in dims})
    for i in range(_il_mod.COMFORT_MIN_SAMPLES + 10):
        v = 0.712 if i % 2 else 0.708
        il._observe_temperament({d: v for d in dims})
    c = il.comfort_threshold("presence")
    assert c["sigma"] == _il_mod.COMFORT_MIN_SIGMA
    assert c["threshold"] < 0.705


def test_band_window_lets_the_baseline_drift():
    dims = ("warmth", "clarity", "stability", "presence")
    il = _make_inner_life({d: 0.5 for d in dims})
    il._comfort_band = {d: {"n": float(_il_mod.COMFORT_WINDOW_SAMPLES), "mean": 0.5, "var": 0.0001}
                        for d in dims}
    # One window of a new operating point moves the mean ~63% of the way
    # (EMA behaviour), where plain Welford over the same history moves half.
    for _ in range(_il_mod.COMFORT_WINDOW_SAMPLES):
        il._observe_temperament({d: 0.7 for d in dims})
    assert il._comfort_band["clarity"]["mean"] > 0.62
    assert il._comfort_band["clarity"]["n"] == _il_mod.COMFORT_WINDOW_SAMPLES


def test_comfort_band_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(_il_mod, "_PERSISTENCE_PATH", tmp_path / "inner_life.json")
    il = _il_mod.InnerLife()
    il._temperament = {d: 0.8 for d in _il_mod.DIMENSIONS}
    for _ in range(_il_mod.COMFORT_MIN_SAMPLES):
        il._observe_temperament(il._temperament)
    il.save()
    again = _il_mod.InnerLife()
    assert again.comfort_threshold("clarity")["source"] == "self_relative"
    assert again._comfort_band["clarity"]["n"] == _il_mod.COMFORT_MIN_SAMPLES


def test_update_publishes_comfort_source():
    il = _make_inner_life({d: 0.5 for d in ("warmth", "clarity", "stability", "presence")})
    a = _make_anima()
    state = il.update(a, a)
    assert state.comfort["warmth"]["source"] == "absolute_fallback"
    assert "comfort" in state.to_dict()
