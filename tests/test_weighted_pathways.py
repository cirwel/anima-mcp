"""
Tests for weighted_pathways module.

Validates context discretization, pathway reinforcement/decay,
SQLite persistence, and integration with the action selector.
"""

import pytest
import time
import json
import sqlite3
from unittest.mock import patch

from anima_mcp.weighted_pathways import (
    SurpriseBucket,
    SatisfactionBucket,
    DriveBucket,
    ActivityBucket,
    discretize_surprise,
    discretize_satisfaction,
    discretize_drive,
    discretize_activity,
    discretize_context,
    Pathway,
    WeightedPathways,
    PATHWAY_REWARD_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_pathways.db")


@pytest.fixture
def wp(tmp_db):
    """Create a WeightedPathways instance with a fresh temp database."""
    w = WeightedPathways(db_path=tmp_db)
    try:
        yield w
    finally:
        w.close()


# ---------------------------------------------------------------------------
# Context discretization
# ---------------------------------------------------------------------------

class TestDiscretizeContext:

    def test_surprise_low(self):
        assert discretize_surprise(0.0) == SurpriseBucket.LOW
        assert discretize_surprise(0.14) == SurpriseBucket.LOW

    def test_surprise_moderate(self):
        assert discretize_surprise(0.15) == SurpriseBucket.MODERATE
        assert discretize_surprise(0.25) == SurpriseBucket.MODERATE
        assert discretize_surprise(0.35) == SurpriseBucket.MODERATE

    def test_surprise_high(self):
        assert discretize_surprise(0.36) == SurpriseBucket.HIGH
        assert discretize_surprise(1.0) == SurpriseBucket.HIGH

    def test_satisfaction_unsatisfied(self):
        assert discretize_satisfaction(0.0) == SatisfactionBucket.UNSATISFIED
        assert discretize_satisfaction(0.34) == SatisfactionBucket.UNSATISFIED

    def test_satisfaction_neutral(self):
        assert discretize_satisfaction(0.35) == SatisfactionBucket.NEUTRAL
        assert discretize_satisfaction(0.5) == SatisfactionBucket.NEUTRAL
        assert discretize_satisfaction(0.65) == SatisfactionBucket.NEUTRAL

    def test_satisfaction_satisfied(self):
        assert discretize_satisfaction(0.66) == SatisfactionBucket.SATISFIED
        assert discretize_satisfaction(1.0) == SatisfactionBucket.SATISFIED

    def test_drive_calm(self):
        assert discretize_drive(0.0) == DriveBucket.CALM
        assert discretize_drive(0.19) == DriveBucket.CALM

    def test_drive_wanting(self):
        assert discretize_drive(0.2) == DriveBucket.WANTING
        assert discretize_drive(0.35) == DriveBucket.WANTING
        assert discretize_drive(0.5) == DriveBucket.WANTING

    def test_drive_urgent(self):
        assert discretize_drive(0.51) == DriveBucket.URGENT
        assert discretize_drive(1.0) == DriveBucket.URGENT

    def test_activity_active(self):
        assert discretize_activity("active") == ActivityBucket.ACTIVE
        assert discretize_activity("Active") == ActivityBucket.ACTIVE

    def test_activity_drowsy(self):
        assert discretize_activity("drowsy") == ActivityBucket.DROWSY

    def test_activity_resting(self):
        assert discretize_activity("resting") == ActivityBucket.RESTING

    def test_activity_unknown_defaults_resting(self):
        assert discretize_activity("unknown") == ActivityBucket.RESTING

    def test_discretize_context_produces_key(self):
        key = discretize_context(
            surprise=0.1,
            satisfaction=0.7,
            drive=0.1,
            activity="active",
        )
        assert key == "low|sat|calm|act"

    def test_discretize_context_boundaries(self):
        """Verify all bucket thresholds produce correct keys."""
        # All LOW/UNSATISFIED/CALM/RESTING
        assert discretize_context(0.0, 0.0, 0.0, "resting") == "low|unsat|calm|rest"
        # All HIGH/SATISFIED/URGENT/ACTIVE
        assert discretize_context(0.5, 0.8, 0.6, "active") == "hi|sat|urg|act"
        # All MODERATE/NEUTRAL/WANTING/DROWSY
        assert discretize_context(0.25, 0.5, 0.35, "drowsy") == "mod|neut|want|drow"


# ---------------------------------------------------------------------------
# Pathway dataclass
# ---------------------------------------------------------------------------

class TestPathway:

    def test_pathway_default_strength(self):
        """A new pathway has strength 0.5."""
        pw = Pathway(context_key="low|sat|calm|act", action_key="ask_question")
        assert pw.strength == 0.5
        assert pw.use_count == 0
        assert pw.total_reward == 0.0

    def test_reinforce_positive(self):
        """Positive outcome increases strength."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.5)
        pw.reinforce(0.8, time.time())
        assert pw.strength > 0.5
        assert pw.use_count == 1
        assert pw.total_reward == pytest.approx(0.8)

    def test_reinforce_negative(self):
        """Negative outcome decreases strength."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.5)
        pw.reinforce(-0.6, time.time())
        assert pw.strength < 0.5
        assert pw.use_count == 1
        assert pw.total_reward == pytest.approx(-0.6)

    def test_reinforcement_bounds_upper(self):
        """Strength cannot exceed 5.0."""
        pw = Pathway(context_key="ctx", action_key="act", strength=4.95)
        pw.reinforce(1.0, time.time())
        assert pw.strength == pytest.approx(5.0)
        # Even more reinforcement stays at 5.0
        pw.reinforce(1.0, time.time())
        assert pw.strength == pytest.approx(5.0)

    def test_reinforcement_bounds_lower(self):
        """Strength cannot go below 0.01."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.05)
        pw.reinforce(-1.0, time.time())
        assert pw.strength == pytest.approx(0.01)
        # Further negative reinforcement stays at 0.01
        pw.reinforce(-1.0, time.time())
        assert pw.strength == pytest.approx(0.01)

    def test_reinforce_clamps_quality(self):
        """Quality values beyond [-1, 1] are clamped."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.5)
        pw.reinforce(5.0, time.time())  # clamped to 1.0
        # strength = 0.5 + 0.15 * 1.0 = 0.65
        assert pw.strength == pytest.approx(0.65)

    def test_decay_over_time(self):
        """Decay reduces strength based on elapsed time."""
        pw = Pathway(context_key="ctx", action_key="act", strength=1.0, last_used=1000.0)

        # Mock 10 hours later
        now = 1000.0 + 10 * 3600
        pw.decay(now)

        # strength *= 0.999 ^ 10 = 0.999^10 ~ 0.99004
        expected = 1.0 * (0.999 ** 10)
        assert pw.strength == pytest.approx(expected, rel=1e-4)

    def test_decay_large_gap(self):
        """Heavy decay over a very long time."""
        pw = Pathway(context_key="ctx", action_key="act", strength=1.0, last_used=1.0)

        # 10000 hours later
        now = 1.0 + 10000 * 3600
        pw.decay(now)

        # 0.999 ^ 10000 is extremely small, should be clamped to 0.01
        assert pw.strength == pytest.approx(0.01)

    def test_decay_preserves_minimum(self):
        """Heavily decayed pathway stays at 0.01."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.02, last_used=0.0)
        # Decay over a long time
        pw.decay(1000000.0)
        assert pw.strength >= 0.01

    def test_decay_no_last_used(self):
        """If last_used is 0, decay does nothing."""
        pw = Pathway(context_key="ctx", action_key="act", strength=0.8, last_used=0.0)
        pw.decay(time.time())
        assert pw.strength == 0.8

    def test_reinforce_with_lr_bonus(self):
        """lr_bonus scales the learning rate."""
        pw_normal = Pathway(context_key="ctx", action_key="act", strength=0.5)
        pw_bonus = Pathway(context_key="ctx", action_key="act", strength=0.5)

        now = time.time()
        pw_normal.reinforce(0.8, now, lr_bonus=0.0)
        pw_bonus.reinforce(0.8, now, lr_bonus=0.10)

        # Normal: 0.5 + 0.15 * 0.8 = 0.62
        assert pw_normal.strength == pytest.approx(0.62)
        # Bonus: 0.5 + 0.15 * 1.10 * 0.8 = 0.632
        assert pw_bonus.strength == pytest.approx(0.632)
        assert pw_bonus.strength > pw_normal.strength

    def test_non_finite_quality_cannot_corrupt_pathway(self):
        pw = Pathway(context_key="ctx", action_key="act", strength=0.5)

        with pytest.raises(ValueError, match="finite"):
            pw.reinforce(float("nan"), time.time())

        assert pw.strength == 0.5
        assert pw.use_count == 0

    def test_negative_bonus_cannot_reverse_learning_direction(self):
        pw = Pathway(context_key="ctx", action_key="act", strength=0.5)

        pw.reinforce(0.8, time.time(), lr_bonus=-2.0)

        assert pw.strength == 0.5
        assert pw.use_count == 1


# ---------------------------------------------------------------------------
# WeightedPathways — core operations
# ---------------------------------------------------------------------------

class TestWeightedPathways:

    def test_get_strength_default(self, wp):
        """Unknown pathway returns 0.5."""
        s = wp.get_strength("some|context", "some_action")
        assert s == pytest.approx(0.5)

    def test_reinforce_and_get(self, wp):
        """Reinforcing a pathway changes its strength."""
        wp.reinforce("ctx", "act", 0.5)
        s = wp.get_strength("ctx", "act")
        assert s > 0.5

    def test_reinforce_negative_decreases(self, wp):
        """Negative reinforcement decreases strength."""
        wp.reinforce("ctx", "act", -0.5)
        s = wp.get_strength("ctx", "act")
        assert s < 0.5

    def test_get_all_strengths_prefix_scan(self, wp):
        """get_all_strengths returns only pathways for the matching context."""
        wp.reinforce("ctx_a", "act1", 0.5)
        wp.reinforce("ctx_a", "act2", 0.3)
        wp.reinforce("ctx_b", "act1", 0.8)

        result = wp.get_all_strengths("ctx_a")
        assert "act1" in result
        assert "act2" in result
        assert len(result) == 2

        result_b = wp.get_all_strengths("ctx_b")
        assert "act1" in result_b
        assert len(result_b) == 1

    def test_get_all_strengths_empty(self, wp):
        """Empty result for unknown context."""
        result = wp.get_all_strengths("nonexistent")
        assert result == {}

    def test_get_stats_empty(self, wp):
        """Stats for empty pathways."""
        stats = wp.get_stats()
        assert stats["total_pathways"] == 0
        assert stats["unique_contexts"] == 0
        assert stats["unique_actions"] == 0
        assert stats["avg_strength"] == 0.5
        assert stats["total_reinforcements"] == 0

    def test_get_stats_populated(self, wp):
        """Stats after some reinforcements."""
        wp.reinforce("ctx_a", "act1", 0.5)
        wp.reinforce("ctx_a", "act2", 0.3)
        wp.reinforce("ctx_b", "act1", 0.8)

        stats = wp.get_stats()
        assert stats["total_pathways"] == 3
        assert stats["unique_contexts"] == 2
        assert stats["unique_actions"] == 2
        assert stats["total_reinforcements"] == 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_persist_and_reload(self, tmp_db):
        """Pathways survive across instances with the same DB."""
        wp1 = WeightedPathways(db_path=tmp_db)
        wp1.reinforce("ctx", "act", 0.7)
        strength_before = wp1.get_strength("ctx", "act")
        wp1.close()

        wp2 = WeightedPathways(db_path=tmp_db)
        strength_after = wp2.get_strength("ctx", "act")
        wp2.close()

        # Should be very close (only tiny decay between close and reload)
        assert strength_after == pytest.approx(strength_before, abs=0.01)

    def test_persist_multiple_pathways(self, tmp_db):
        """Multiple pathways all persist."""
        wp1 = WeightedPathways(db_path=tmp_db)
        wp1.reinforce("c1", "a1", 0.8)
        wp1.reinforce("c1", "a2", -0.3)
        wp1.reinforce("c2", "a1", 0.5)
        wp1.close()

        wp2 = WeightedPathways(db_path=tmp_db)
        stats = wp2.get_stats()
        assert stats["total_pathways"] == 3
        wp2.close()

    def test_init_creates_table(self, tmp_db):
        """Initialization creates the pathways table."""
        import sqlite3
        wp = WeightedPathways(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "pathways" in table_names
        conn.close()
        wp.close()

    def test_rewardless_legacy_pathways_are_preserved_then_neutralized(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE pathways (
            context_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            strength REAL DEFAULT 0.5,
            use_count INTEGER DEFAULT 0,
            last_used REAL DEFAULT 0,
            total_reward REAL DEFAULT 0,
            PRIMARY KEY (context_key, action_key)
        )""")
        conn.execute(
            "INSERT INTO pathways VALUES (?, ?, ?, ?, ?, ?)",
            ("hi|unsat|urg|act", "face_expression", 0.01, 9000, 123.0, 0.0),
        )
        conn.commit()
        conn.close()

        wp = WeightedPathways(db_path=tmp_db)

        assert wp.get_strength("hi|unsat|urg|act", "face_expression") == 0.5
        assert wp._pathways["hi|unsat|urg|act|face_expression"].use_count == 0
        conn = sqlite3.connect(tmp_db)
        version = json.loads(conn.execute(
            "SELECT data FROM pathways_state WHERE key='reward_model_version'"
        ).fetchone()[0])
        legacy = json.loads(conn.execute(
            "SELECT data FROM pathways_state WHERE key='legacy_rewardless_v1'"
        ).fetchone()[0])
        conn.close()
        wp.close()
        assert version == PATHWAY_REWARD_VERSION
        assert legacy[0]["use_count"] == 9000


# ---------------------------------------------------------------------------
# Decay with mocked time
# ---------------------------------------------------------------------------

class TestDecayWithMockedTime:

    def test_decay_over_time_mocked(self):
        """Mock time.time() to test exact decay math."""
        pw = Pathway(context_key="ctx", action_key="act", strength=1.0, last_used=1000.0)

        # 5 hours later
        with patch("anima_mcp.weighted_pathways.time") as mock_time:
            mock_time.time.return_value = 1000.0 + 5 * 3600
            # Manually call decay with mocked time
            pw.decay(mock_time.time())

        expected = 1.0 * (0.999 ** 5)
        assert pw.strength == pytest.approx(expected, rel=1e-4)

    def test_get_strength_applies_decay(self, tmp_db):
        """get_strength applies decay based on current time."""
        wp = WeightedPathways(db_path=tmp_db)
        # Directly set a pathway with old last_used
        pw = wp._get_or_create("ctx", "act")
        pw.strength = 2.0
        pw.last_used = time.time() - 100 * 3600  # 100 hours ago

        s = wp.get_strength("ctx", "act")
        expected = 2.0 * (0.999 ** 100)
        assert s == pytest.approx(expected, rel=0.01)
        wp.close()


# ---------------------------------------------------------------------------
# Integration with ActionSelector
# ---------------------------------------------------------------------------

class TestIntegrationWithActionSelector:

    def test_pathway_strength_changes_action_winner(self, tmp_path):
        """Pathway strengths actually influence which action is selected."""
        from anima_mcp.agency import ActionSelector, ActionType

        db_path = str(tmp_path / "test_integration.db")
        selector = ActionSelector(db_path=db_path)

        # Disable exploration to get deterministic results
        selector._exploration_rate = 0.0

        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}

        # With high surprise, ASK_QUESTION and FOCUS_ATTENTION are candidates.
        # Give ASK_QUESTION a very strong pathway and FOCUS_ATTENTION a very weak one.
        strong_strengths = {"ask_question": 3.0, "focus_attention": 0.05, "stay_quiet": 0.05}

        # Run many trials and count how often ASK_QUESTION wins
        ask_count = 0
        trials = 50
        for _ in range(trials):
            action = selector.select_action(
                state,
                surprise_level=0.5,
                surprise_sources=["temperature"],
                pathway_strengths=strong_strengths,
            )
            if action.action_type == ActionType.ASK_QUESTION:
                ask_count += 1

        # With a 6x multiplier on ask_question, it should dominate
        assert ask_count > trials * 0.5, f"ASK_QUESTION won {ask_count}/{trials} times (expected majority)"

    def test_pathway_strengths_none_no_effect(self, tmp_path):
        """When pathway_strengths is None, behavior is unchanged."""
        from anima_mcp.agency import ActionSelector

        db_path = str(tmp_path / "test_none.db")
        selector = ActionSelector(db_path=db_path)
        selector._exploration_rate = 0.0

        state = {"warmth": 0.5, "clarity": 0.5}

        # Should not raise
        action = selector.select_action(state, surprise_level=0.0, pathway_strengths=None)
        assert action is not None

    def test_pathway_multiplier_bounds(self, tmp_path):
        """Pathway multiplier is bounded between 0.25 and 4.0."""
        from anima_mcp.agency import ActionSelector

        db_path = str(tmp_path / "test_bounds.db")
        selector = ActionSelector(db_path=db_path)
        selector._exploration_rate = 0.0

        state = {"warmth": 0.5, "clarity": 0.5}

        # Very extreme pathway strength (10.0) should be clamped to 4.0 multiplier
        extreme_strengths = {"stay_quiet": 10.0}
        action = selector.select_action(
            state, surprise_level=0.0, pathway_strengths=extreme_strengths,
        )
        assert action is not None

        # Very low pathway strength (0.01) should be clamped to 0.25 multiplier
        weak_strengths = {"stay_quiet": 0.01}
        action = selector.select_action(
            state, surprise_level=0.0, pathway_strengths=weak_strengths,
        )
        assert action is not None
