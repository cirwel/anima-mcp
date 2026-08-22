"""
Tests for preferences.py — satisfaction RL loop, experience recording,
preference learning, optimal range adaptation, and persistence.

Covers:
  - Preference.current_satisfaction (optimal range, boundary behavior)
  - Preference.update_from_experience (positive/negative valence, range expansion)
  - PreferenceSystem.record_state and record_event
  - PreferenceSystem._learn_from_experience
  - get_overall_satisfaction (weighted average, empty state)
  - get_most_unsatisfied (confidence threshold, correct selection)
  - get_preferred_direction (below/above/within optimal range)
  - Persistence (save/load round-trip)
  - describe_preferences (natural language output)
"""

import pytest
from datetime import datetime, timedelta

from anima_mcp.preferences import (
    PreferenceSystem, Preference,
)


@pytest.fixture
def ps(tmp_path):
    """Create PreferenceSystem with temp persistence path."""
    return PreferenceSystem(persistence_path=tmp_path / "prefs.json")


def default_state(**overrides):
    """Create default anima-like state dict."""
    state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}
    state.update(overrides)
    return state


# ==================== Preference.current_satisfaction ====================

class TestCurrentSatisfaction:
    """Test satisfaction calculation for different value ranges."""

    def test_in_optimal_range_high(self):
        """Value in optimal range gives high satisfaction."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        sat = pref.current_satisfaction(0.5)
        assert sat > 0.8  # Center of range → high satisfaction

    def test_at_center_max_satisfaction(self):
        """Value at center of optimal range gives max satisfaction."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        sat = pref.current_satisfaction(0.5)
        assert sat == pytest.approx(1.0)

    def test_below_optimal_low_satisfaction(self):
        """Value below optimal range gives reduced satisfaction."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        sat = pref.current_satisfaction(0.1)
        assert sat < 0.7  # Well below optimal

    def test_above_optimal_high_low_satisfaction(self):
        """Value above optimal range gives reduced satisfaction."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        sat = pref.current_satisfaction(0.95)
        assert sat < 0.7  # Well above optimal

    def test_extreme_low_zero_satisfaction(self):
        """Very extreme values can hit zero satisfaction."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        sat = pref.current_satisfaction(0.0)
        assert sat >= 0.0  # Never negative
        assert sat < 0.5

    def test_satisfaction_bounded(self):
        """Satisfaction is always in [0, 1]."""
        pref = Preference(dimension="warmth")
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sat = pref.current_satisfaction(v)
            assert 0.0 <= sat <= 1.0

    def test_satisfaction_is_continuous_at_optimal_boundaries(self):
        pref = Preference(
            dimension="warmth",
            optimal_low=0.3,
            optimal_high=0.7,
            optimal_center=0.6,
        )

        assert pref.current_satisfaction(0.3 - 1e-9) == pytest.approx(
            pref.current_satisfaction(0.3), abs=1e-8,
        )
        assert pref.current_satisfaction(0.7 + 1e-9) == pytest.approx(
            pref.current_satisfaction(0.7), abs=1e-8,
        )


# ==================== Preference.update_from_experience ====================

class TestUpdateFromExperience:
    """Test preference learning from positive/negative experiences."""

    def test_positive_experience_increases_valence(self):
        """Good outcome with active dimension increases valence."""
        pref = Preference(dimension="clarity", valence=0.0)
        pref.update_from_experience(state_value=0.8, outcome_valence=0.5)
        assert pref.valence > 0.0

    def test_negative_experience_decreases_valence(self):
        """Bad outcome decreases valence."""
        pref = Preference(dimension="clarity", valence=0.0)
        pref.update_from_experience(state_value=0.8, outcome_valence=-0.5)
        assert pref.valence < 0.0

    def test_valence_clamped(self):
        """Valence stays within [-1, 1]."""
        pref = Preference(dimension="clarity", valence=0.9)
        for _ in range(50):
            pref.update_from_experience(state_value=1.0, outcome_valence=1.0)
        assert pref.valence <= 1.0
        assert pref.valence >= -1.0

    def test_positive_experience_expands_optimal_range(self):
        """Good outcome with value outside range expands the range."""
        pref = Preference(dimension="warmth", optimal_low=0.3, optimal_high=0.7)
        # Good outcome at 0.9 (above optimal_high)
        pref.update_from_experience(state_value=0.9, outcome_valence=0.5)
        assert pref.optimal_high > 0.7  # Range expanded upward

    def test_negative_experience_contracts_range(self):
        """Bad outcome contracts optimal range."""
        pref = Preference(dimension="warmth", optimal_low=0.2, optimal_high=0.8)
        original_high = pref.optimal_high
        # Bad outcome at high value
        pref.update_from_experience(state_value=0.9, outcome_valence=-0.5)
        assert pref.optimal_high <= original_high  # Range contracted

    def test_confidence_grows(self):
        """Confidence increases with experience count, capped at 1.0."""
        pref = Preference(dimension="warmth")
        assert pref.confidence == 0.0
        for _ in range(25):
            pref.update_from_experience(state_value=0.5, outcome_valence=0.3)
        assert pref.confidence == pytest.approx(1.0)  # 25/20 = 1.25, capped at 1.0


# ==================== record_state and record_event ====================

class TestRecordStateAndEvent:
    """Test state recording and event-based learning."""

    def test_record_state(self, ps):
        """record_state stores state in history."""
        ps.record_state(default_state())
        assert len(ps._state_history) == 1
        assert ps._last_state is not None

    def test_record_event_no_prior_state(self, ps):
        """record_event with no prior state is a no-op (can't learn)."""
        ps.record_event("calm", valence=0.3)
        assert len(ps._recent_experiences) == 0

    def test_record_event_learns(self, ps):
        """record_event with prior state creates experience and learns."""
        state = default_state(warmth=0.7, clarity=0.8)
        ps.record_state(state)
        # Set timestamp far enough back to be found
        ps._state_history[-1]["timestamp"] = datetime.now() - timedelta(seconds=10)
        ps.record_event("calm", valence=0.5, current_state=state)
        assert len(ps._recent_experiences) == 1

    def test_positive_event_shapes_preferences(self, ps):
        """Positive events shift preference valence positively."""
        state = default_state(warmth=0.8)
        ps.record_state(state)
        ps._state_history[-1]["timestamp"] = datetime.now() - timedelta(seconds=10)

        warmth_before = ps._preferences["warmth"].valence
        ps.record_event("calm", valence=0.5, current_state=state)
        assert ps._preferences["warmth"].valence > warmth_before


# ==================== get_overall_satisfaction ====================

class TestOverallSatisfaction:
    """Test weighted satisfaction calculation."""

    def test_default_preferences_moderate(self, ps):
        """Fresh preferences with no experience return 0.5 (neutral)."""
        sat = ps.get_overall_satisfaction(default_state())
        assert sat == pytest.approx(0.5)  # Zero confidence → default 0.5

    def test_all_optimal_high_satisfaction(self, ps):
        """All values in optimal range with confidence → high satisfaction."""
        # Build confidence
        for pref in ps._preferences.values():
            pref.confidence = 0.8
            pref.valence = 0.5
        state = default_state()  # 0.5 is center of default optimal range
        sat = ps.get_overall_satisfaction(state)
        assert sat > 0.8

    def test_empty_state_returns_default(self, ps):
        """Empty state dict returns 0.5."""
        sat = ps.get_overall_satisfaction({})
        assert sat == 0.5

    def test_influence_weight_changes_overall_satisfaction(self, ps):
        warmth = ps._preferences["warmth"]
        clarity = ps._preferences["clarity"]
        for pref in (warmth, clarity):
            pref.confidence = 1.0
            pref.valence = 0.0
        warmth.optimal_low, warmth.optimal_high = 0.8, 1.0
        clarity.optimal_low, clarity.optimal_high = 0.4, 0.6
        state = {"warmth": 0.0, "clarity": 0.5}

        warmth.influence_weight, clarity.influence_weight = 3.0, 0.3
        warmth_weighted = ps.get_overall_satisfaction(state)
        warmth.influence_weight, clarity.influence_weight = 0.3, 3.0
        clarity_weighted = ps.get_overall_satisfaction(state)

        assert warmth_weighted < clarity_weighted


# ==================== get_most_unsatisfied ====================

class TestMostUnsatisfied:
    """Test finding least satisfied preference."""

    def test_all_satisfied_returns_high(self, ps):
        """When all are satisfied, worst is still relatively high."""
        for pref in ps._preferences.values():
            pref.confidence = 0.5
        dim, sat = ps.get_most_unsatisfied(default_state())
        # Default state is center of optimal range → high satisfaction
        assert sat > 0.5

    def test_finds_worst_dimension(self, ps):
        """Correctly identifies the least satisfied dimension."""
        ps._preferences["warmth"].confidence = 0.5
        ps._preferences["warmth"].optimal_low = 0.7
        ps._preferences["warmth"].optimal_high = 0.9
        ps._preferences["clarity"].confidence = 0.5

        state = default_state(warmth=0.1)  # Way below warmth's optimal
        dim, sat = ps.get_most_unsatisfied(state)
        assert dim == "warmth"
        assert sat < 0.5

    def test_ignores_low_confidence(self, ps):
        """Preferences with confidence < 0.2 are ignored."""
        ps._preferences["warmth"].confidence = 0.1  # Too low
        ps._preferences["warmth"].optimal_low = 0.9
        ps._preferences["warmth"].optimal_high = 1.0
        dim, sat = ps.get_most_unsatisfied(default_state(warmth=0.1))
        assert dim != "warmth"  # Should be skipped


# ==================== get_preferred_direction ====================

class TestPreferredDirection:
    """Test direction-of-change guidance."""

    def test_below_optimal_wants_increase(self, ps):
        """Value below optimal range → positive direction (increase)."""
        ps._preferences["warmth"].optimal_low = 0.4
        ps._preferences["warmth"].optimal_high = 0.7
        direction = ps.get_preferred_direction("warmth", 0.2)
        assert direction == 1.0

    def test_above_optimal_wants_decrease(self, ps):
        """Value above optimal range → negative direction (decrease)."""
        ps._preferences["warmth"].optimal_low = 0.3
        ps._preferences["warmth"].optimal_high = 0.6
        direction = ps.get_preferred_direction("warmth", 0.9)
        assert direction == -1.0

    def test_in_optimal_slight_pull_to_center(self, ps):
        """Value in optimal range → slight pull toward center."""
        ps._preferences["warmth"].optimal_low = 0.3
        ps._preferences["warmth"].optimal_high = 0.7
        # Value at 0.6, center is 0.5 → slight negative
        direction = ps.get_preferred_direction("warmth", 0.6)
        assert direction < 0.0
        assert abs(direction) < 0.2  # Gentle pull

    def test_unknown_dimension_returns_zero(self, ps):
        """Unknown dimension returns 0.0."""
        assert ps.get_preferred_direction("nonexistent", 0.5) == 0.0

    def test_in_range_direction_uses_learned_optimal_center(self, ps):
        pref = ps._preferences["warmth"]
        pref.optimal_low = 0.3
        pref.optimal_high = 0.7
        pref.optimal_center = 0.65

        assert ps.get_preferred_direction("warmth", 0.55) > 0.0


# ==================== Persistence ====================

class TestPersistence:
    """Test preference save/load round-trip."""

    def test_save_and_load(self, tmp_path):
        """Preferences survive save/load cycle."""
        path = tmp_path / "prefs.json"
        ps1 = PreferenceSystem(persistence_path=path)
        ps1._preferences["warmth"].valence = 0.6
        ps1._preferences["warmth"].confidence = 0.8
        ps1._preferences["warmth"].optimal_low = 0.4
        ps1._preferences["warmth"].optimal_high = 0.8
        ps1._preferences["warmth"].experience_count = 15
        ps1._save()

        ps2 = PreferenceSystem(persistence_path=path)
        assert ps2._preferences["warmth"].valence == pytest.approx(0.6)
        assert ps2._preferences["warmth"].confidence == pytest.approx(0.8)
        assert ps2._preferences["warmth"].optimal_low == pytest.approx(0.4)
        assert ps2._preferences["warmth"].experience_count == 15

    def test_load_missing_file_no_crash(self, tmp_path):
        """Loading from nonexistent file doesn't crash."""
        ps = PreferenceSystem(persistence_path=tmp_path / "missing.json")
        assert ps._preferences["warmth"].valence == 0.0  # Default

    def test_read_only_snapshot_refreshes_without_writing(self, tmp_path):
        """Server readers follow broker snapshots but cannot overwrite them."""
        path = tmp_path / "prefs.json"
        writer = PreferenceSystem(persistence_path=path)
        writer._preferences["warmth"].valence = 0.25
        writer._preferences["warmth"].optimal_center = 0.61
        writer._save()

        reader = PreferenceSystem(persistence_path=path, read_only=True)
        assert reader._preferences["warmth"].valence == pytest.approx(0.25)
        assert reader._preferences["warmth"].optimal_center == pytest.approx(0.61)

        reader._preferences["warmth"].valence = -1.0
        reader._save()
        assert PreferenceSystem(persistence_path=path)._preferences["warmth"].valence == pytest.approx(0.25)

        writer._preferences["warmth"].valence = 0.75
        writer._save()
        assert reader.refresh_if_changed(force=True) is True
        assert reader._preferences["warmth"].valence == pytest.approx(0.75)


# ==================== describe_preferences ====================

class TestDescribePreferences:
    """Test natural language preference description."""

    def test_no_confidence_returns_developing(self, ps):
        """With no confident preferences, returns 'developing' message."""
        desc = ps.describe_preferences()
        assert "developing" in desc.lower()

    def test_valued_dimension_described(self, ps):
        """High-valence confident preference is described."""
        ps._preferences["warmth"].valence = 0.5
        ps._preferences["warmth"].confidence = 0.5
        desc = ps.describe_preferences()
        assert "warmth" in desc.lower()
        assert "values" in desc.lower()

    def test_summary_dict(self, ps):
        """get_preference_summary returns structured data."""
        summary = ps.get_preference_summary()
        assert "warmth" in summary
        assert "valence" in summary["warmth"]
        assert "confidence" in summary["warmth"]
