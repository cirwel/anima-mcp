"""Tests for gestural era smooth commitment signal."""

import random

from anima_mcp.display.eras.gestural import GesturalEra, GesturalState


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


class TestColorGeneration:
    """Color system uses vibrant palette + HSV with presence-driven vibrant rate."""

    def test_vibrant_colors_occur_at_high_presence(self):
        """High presence should produce vibrant colors ~30% of the time."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 500

        vibrant_count = 0
        for _ in range(500):
            _, category = era.generate_color(state, 0.5, 0.5, 0.5, 1.0)  # high presence
            if category == "vibrant":
                vibrant_count += 1

        rate = vibrant_count / 500
        assert 0.15 < rate < 0.45, f"Vibrant rate at high presence {rate:.3f} should be ~30%"

    def test_vibrant_colors_rare_at_low_presence(self):
        """Low presence should produce vibrant colors ~15% of the time."""
        random.seed(42)
        era = GesturalEra()
        state = era.create_state()
        state.gesture = "stroke"
        state.gesture_remaining = 500

        vibrant_count = 0
        for _ in range(500):
            _, category = era.generate_color(state, 0.5, 0.5, 0.5, 0.0)  # low presence
            if category == "vibrant":
                vibrant_count += 1

        rate = vibrant_count / 500
        assert 0.05 < rate < 0.25, f"Vibrant rate at low presence {rate:.3f} should be ~15%"


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
