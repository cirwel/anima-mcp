"""
Tests for agency module.

Validates action selection, value learning, and SQLite persistence.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.agency import (
    AGENCY_REWARD_VERSION,
    Action,
    ActionSelector,
    ActionType,
    build_agency_inputs,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_agency.db")


@pytest.fixture
def selector(tmp_db):
    """Create an ActionSelector with a fresh temp database."""
    return ActionSelector(db_path=tmp_db)


class TestAgencyPersistence:
    """Test SQLite persistence of action values and exploration rate."""

    def test_init_creates_tables(self, tmp_db):
        """Test that init creates the required tables."""
        import sqlite3
        ActionSelector(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        # Check tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "agency_values" in table_names
        assert "agency_state" in table_names
        conn.close()

    def test_persist_and_reload_action_values(self, tmp_db):
        """Test that action values survive across instances."""
        sel1 = ActionSelector(db_path=tmp_db)

        # Manually set some action values and persist
        sel1._action_values["ask_question"] = 0.75
        sel1._action_counts["ask_question"] = 5
        sel1._persist_action("ask_question")

        sel1._action_values["stay_quiet"] = 0.3
        sel1._action_counts["stay_quiet"] = 12
        sel1._persist_action("stay_quiet")

        # Create a new instance from same DB
        sel2 = ActionSelector(db_path=tmp_db)

        assert sel2._action_values["ask_question"] == pytest.approx(0.75)
        assert sel2._action_counts["ask_question"] == 5
        assert sel2._action_values["stay_quiet"] == pytest.approx(0.3)
        assert sel2._action_counts["stay_quiet"] == 12

    def test_persist_and_reload_exploration_rate(self, tmp_db):
        """Test that exploration rate persists across instances."""
        sel1 = ActionSelector(db_path=tmp_db)
        sel1._exploration_rate = 0.123

        # Persist via any action
        sel1._action_values["test_action"] = 0.5
        sel1._action_counts["test_action"] = 1
        sel1._persist_action("test_action")

        # Reload
        sel2 = ActionSelector(db_path=tmp_db)
        assert sel2._exploration_rate == pytest.approx(0.123)

    def test_learn_from_outcome_persists(self, tmp_db):
        """Test that record_outcome persists learned values."""
        sel1 = ActionSelector(db_path=tmp_db)

        # Record a positive outcome (increased satisfaction → positive reward)
        action = Action(ActionType.ASK_QUESTION, motivation="test")
        sel1._action_counts["ask_question"] = 1
        sel1.record_outcome(
            action,
            state_before={"warmth": 0.5, "clarity": 0.5},
            state_after={"warmth": 0.7, "clarity": 0.7},
            preference_satisfaction_before=0.3,
            preference_satisfaction_after=0.8,  # Big increase → positive reward
            surprise_after=0.2,
        )

        # Value should have shifted from default 0.5
        assert sel1._action_values["ask_question"] != 0.5

        # Reload and verify persistence
        sel2 = ActionSelector(db_path=tmp_db)
        assert "ask_question" in sel2._action_values

    def test_exploration_rate_decays_on_outcome(self, tmp_db):
        """Test that exploration rate decays when learning."""
        sel = ActionSelector(db_path=tmp_db)
        initial_rate = sel._exploration_rate

        action = Action(ActionType.STAY_QUIET, motivation="test")
        sel.record_outcome(
            action,
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.5},
            preference_satisfaction_before=0.5,
            preference_satisfaction_after=0.5,
            surprise_after=0.2,
        )

        assert sel._exploration_rate < initial_rate
        assert sel._exploration_rate >= 0.05  # Minimum floor

    def test_exploration_floor_reduction(self, tmp_db):
        """exploration_floor_reduction lowers the minimum exploration floor."""
        sel = ActionSelector(db_path=tmp_db)
        sel._exploration_rate = 0.06  # Just above default floor

        # Decay without reduction — floor is 0.05
        sel.record_outcome(
            Action(ActionType.STAY_QUIET, motivation="test"),
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.5},
            preference_satisfaction_before=0.5,
            preference_satisfaction_after=0.5,
            surprise_after=0.1,
        )
        assert sel._exploration_rate >= 0.05

        # Now with reduction of 0.01 — floor drops to 0.04
        sel._exploration_rate = 0.03  # Below old floor
        sel.record_outcome(
            Action(ActionType.STAY_QUIET, motivation="test"),
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.5},
            preference_satisfaction_before=0.5,
            preference_satisfaction_after=0.5,
            surprise_after=0.1,
            exploration_floor_reduction=0.01,
        )
        assert sel._exploration_rate >= 0.04  # New lower floor
        assert sel._exploration_rate < 0.05  # Below old floor

    def test_fresh_db_has_defaults(self, tmp_db):
        """Test that a fresh database starts with default values."""
        sel = ActionSelector(db_path=tmp_db)
        assert sel._exploration_rate == 0.2
        assert len(sel._action_values) == 0
        assert len(sel._action_counts) == 0

    def test_question_feedback_persists(self, tmp_db):
        """Test that question feedback learning persists."""
        sel1 = ActionSelector(db_path=tmp_db)

        # Record good feedback
        sel1.record_question_feedback("test question", {
            "score": 0.9,
            "signals": ["engagement"],
        })

        # ask_question value should increase from default 0.5
        assert sel1._action_values.get("ask_question", 0.5) > 0.5

        # Reload
        sel2 = ActionSelector(db_path=tmp_db)
        assert sel2._action_values.get("ask_question", 0.5) > 0.5

    def test_legacy_false_success_values_are_preserved_then_reset(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        conn.executescript("""
            CREATE TABLE agency_values (
                action_key TEXT PRIMARY KEY, value REAL, count INTEGER
            );
            CREATE TABLE agency_state (key TEXT PRIMARY KEY, data TEXT);
            INSERT INTO agency_values VALUES ('face_expression', 0.91, 400);
            INSERT INTO agency_state VALUES ('exploration_rate', '0.03');
        """)
        conn.commit()
        conn.close()

        selector = ActionSelector(db_path=tmp_db)

        assert selector._action_values == {}
        assert selector._action_counts == {}
        assert selector._exploration_rate == 0.2
        conn = sqlite3.connect(tmp_db)
        version = json.loads(conn.execute(
            "SELECT data FROM agency_state WHERE key='reward_model_version'"
        ).fetchone()[0])
        legacy = json.loads(conn.execute(
            "SELECT data FROM agency_state WHERE key='legacy_values_v1'"
        ).fetchone()[0])
        conn.close()
        assert version == AGENCY_REWARD_VERSION
        assert legacy["action_values"][0]["action_key"] == "face_expression"


class TestActionSelection:
    """Test action selection logic."""

    def test_select_action_returns_action(self, selector):
        """Test that select_action returns a valid Action."""
        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}
        action = selector.select_action(state, surprise_level=0.3)
        assert isinstance(action, Action)
        assert isinstance(action.action_type, ActionType)

    def test_high_surprise_favors_question(self, selector):
        """Test that high surprise makes question-asking more likely."""
        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}

        # Run many trials with high surprise
        question_count = 0
        trials = 100
        for _ in range(trials):
            action = selector.select_action(
                state, surprise_level=0.8, surprise_sources=["temperature"]
            )
            if action.action_type == ActionType.ASK_QUESTION:
                question_count += 1

        # Should ask questions at least sometimes with high surprise
        assert question_count > 0

    def test_low_surprise_favors_quiet(self, selector):
        """Test that low surprise results in quiet behavior."""
        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}

        # Low surprise, no exploration
        selector._exploration_rate = 0.0  # Disable exploration noise
        action = selector.select_action(state, surprise_level=0.0)

        # With no surprise, should default to STAY_QUIET (it's the only candidate)
        assert action.action_type == ActionType.STAY_QUIET

    def test_action_count_increments(self, selector):
        """Test that selecting an action increments its count."""
        state = {"warmth": 0.5, "clarity": 0.5}
        selector._exploration_rate = 0.0

        action = selector.select_action(state, surprise_level=0.0)
        key = action.action_type.value
        assert selector._action_counts.get(key, 0) >= 1

    def test_sensitivity_modifier_changes_surprise_threshold(self, selector):
        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}
        selector._exploration_rate = 0.0
        available = {ActionType.STAY_QUIET, ActionType.ASK_QUESTION}

        with patch("anima_mcp.agency.random.gauss", return_value=0.0):
            quiet = selector.select_action(
                state, surprise_level=0.18, available_actions=available,
            )
            selector.adjust_sensitivity("increase")
            question = selector.select_action(
                state, surprise_level=0.18, available_actions=available,
            )

        assert quiet.action_type == ActionType.STAY_QUIET
        assert question.action_type == ActionType.ASK_QUESTION

    def test_exploration_only_selects_executable_action_with_valid_parameters(self, selector):
        selector._exploration_rate = 1.0
        available = {ActionType.FOCUS_ATTENTION}
        state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}

        with patch("anima_mcp.agency.random.gauss", return_value=0.0):
            action = selector.select_action(
                state,
                surprise_level=0.5,
                surprise_sources=["ambient_temp"],
                available_actions=available,
            )

        assert action.action_type == ActionType.FOCUS_ATTENTION
        assert action.action_type != ActionType.EXPLORE
        assert action.parameters["sensor"] == "ambient_temp"


class TestRecordOutcome:
    """Test value learning from outcomes."""

    def test_positive_reward_increases_value(self, selector):
        """Test that increased satisfaction → positive reward → higher value."""
        action = Action(ActionType.ASK_QUESTION, motivation="test")
        selector._action_values["ask_question"] = 0.5
        selector._action_counts["ask_question"] = 1

        selector.record_outcome(
            action,
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.7},
            preference_satisfaction_before=0.2,
            preference_satisfaction_after=0.9,  # Large increase
            surprise_after=0.2,
        )
        assert selector._action_values["ask_question"] > 0.5

    def test_negative_reward_decreases_value(self, selector):
        """Test that decreased satisfaction → negative reward → lower value."""
        action = Action(ActionType.STAY_QUIET, motivation="test")
        selector._action_values["stay_quiet"] = 0.5
        selector._action_counts["stay_quiet"] = 1

        selector.record_outcome(
            action,
            state_before={"warmth": 0.7},
            state_after={"warmth": 0.3},
            preference_satisfaction_before=0.8,
            preference_satisfaction_after=0.2,  # Large decrease
            surprise_after=0.8,  # Very high surprise = penalty
        )
        assert selector._action_values["stay_quiet"] < 0.5

    def test_neutral_outcome_minimal_change(self, selector):
        """Test that no satisfaction change → near-zero reward → minimal value change."""
        action = Action(ActionType.STAY_QUIET, motivation="test")
        selector._action_values["stay_quiet"] = 0.5
        selector._action_counts["stay_quiet"] = 1

        selector.record_outcome(
            action,
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.5},
            preference_satisfaction_before=0.5,
            preference_satisfaction_after=0.5,  # No change
            surprise_after=0.2,  # Optimal surprise
        )
        # Should stay close to 0.5
        assert abs(selector._action_values["stay_quiet"] - 0.5) < 0.1

    def test_failed_execution_is_penalized_and_cannot_claim_state_change(self, selector):
        action = Action(
            ActionType.ASK_QUESTION,
            execution_succeeded=False,
            execution_detail="board suppressed question",
        )

        outcome = selector.record_outcome(
            action,
            state_before={"warmth": 0.2, "last_surprise": 0.8},
            state_after={"warmth": 0.9},
            preference_satisfaction_before=0.1,
            preference_satisfaction_after=0.9,
            surprise_after=0.2,
        )

        assert outcome.reward == -0.1
        assert outcome.goal_achieved is False

    def test_focus_needs_measured_surprise_baseline_for_goal_reward(self, selector):
        action = Action(ActionType.FOCUS_ATTENTION, execution_succeeded=True)

        outcome = selector.record_outcome(
            action,
            state_before={"warmth": 0.5},
            state_after={"warmth": 0.5},
            preference_satisfaction_before=0.5,
            preference_satisfaction_after=0.5,
            surprise_after=0.0,
        )

        assert outcome.goal_achieved is False
        assert outcome.reward == pytest.approx(0.05)


def test_build_agency_inputs_wires_broker_drives_predictions_pathways_and_capabilities():
    preferences = MagicMock()
    preferences.get_overall_satisfaction.return_value = 0.6
    self_model = MagicMock()
    self_model.predict_own_response.return_value = {"surprise_likelihood": 0.8}
    pathways = MagicMock()
    pathways.get_all_strengths.return_value = {"ask_question": 0.7}

    result = build_agency_inputs(
        current_state={"warmth": 0.4, "clarity": 0.7, "stability": 0.6, "presence": 0.8},
        surprise_level=0.4,
        surprise_sources=["light_lux"],
        preferences=preferences,
        shm_data={
            "inner_life": {"drives": {"warmth": 0.3, "clarity": 0.1}},
            "activity": {"level": "drowsy"},
        },
        self_model=self_model,
        pathways=pathways,
        can_focus=True,
        can_reflect=True,
        can_led=False,
    )

    assert result["drives"] == {"warmth": 0.3, "clarity": 0.1}
    assert result["self_predictions"] == {"surprise_likelihood": 0.8}
    assert result["pathway_context"] == "hi|neut|want|drow"
    assert result["pathway_strengths"] == {"ask_question": 0.7}
    assert ActionType.FOCUS_ATTENTION in result["available_actions"]
    assert ActionType.REQUEST_REFLECTION in result["available_actions"]
    assert ActionType.LED_BRIGHTNESS not in result["available_actions"]
    assert ActionType.EXPLORE not in result["available_actions"]
