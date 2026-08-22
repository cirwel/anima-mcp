"""
Tests for preference meta-learning — influence weights, trajectory health,
and the meta-learning update cycle.

Covers:
  - Preference.influence_weight (default, floor enforcement, JSON persistence)
  - PreferenceSystem.enforce_weight_conservation (sum constraint, ratio preservation)
  - compute_trajectory_health (bounds, high satisfaction, variance penalty)
  - meta_learning_update (boost, reduce, floor, conservation)
"""

import math

import pytest

from anima_mcp.preferences import Preference, PreferenceSystem


class TestInfluenceWeight:
    def test_default_weight_is_one(self):
        p = Preference(dimension="warmth")
        assert p.influence_weight == 1.0

    def test_weight_floor_enforced(self):
        p = Preference(dimension="warmth", influence_weight=0.1)
        p.enforce_floor()
        assert p.influence_weight >= 0.3

    def test_weight_persists_in_json(self):
        p = Preference(dimension="warmth", influence_weight=0.8)
        d = p.to_dict()
        p2 = Preference.from_dict(d)
        assert abs(p2.influence_weight - 0.8) < 0.001


class TestConservation:
    def test_weights_sum_to_four(self):
        ps = PreferenceSystem.__new__(PreferenceSystem)
        ps._preferences = {
            "warmth": Preference("warmth", influence_weight=2.0),
            "clarity": Preference("clarity", influence_weight=1.5),
            "stability": Preference("stability", influence_weight=1.0),
            "presence": Preference("presence", influence_weight=0.5),
        }
        ps.enforce_weight_conservation()
        total = sum(p.influence_weight for p in ps._preferences.values())
        assert abs(total - 4.0) < 0.01

    def test_conservation_preserves_ratios(self):
        ps = PreferenceSystem.__new__(PreferenceSystem)
        ps._preferences = {
            "warmth": Preference("warmth", influence_weight=2.0),
            "clarity": Preference("clarity", influence_weight=2.0),
            "stability": Preference("stability", influence_weight=2.0),
            "presence": Preference("presence", influence_weight=2.0),
        }
        ps.enforce_weight_conservation()
        for p in ps._preferences.values():
            assert abs(p.influence_weight - 1.0) < 0.01

    def test_conservation_keeps_floor_and_exact_sum_under_extreme_weights(self):
        ps = PreferenceSystem.__new__(PreferenceSystem)
        ps._preferences = {
            "warmth": Preference("warmth", influence_weight=1000.0),
            "clarity": Preference("clarity", influence_weight=0.0),
            "stability": Preference("stability", influence_weight=-5.0),
            "presence": Preference("presence", influence_weight=0.01),
        }

        ps.enforce_weight_conservation()

        values = [pref.influence_weight for pref in ps._preferences.values()]
        assert sum(values) == pytest.approx(4.0)
        assert min(values) >= 0.3


class TestTrajectoryHealth:
    def test_health_bounded_zero_one(self):
        from anima_mcp.preferences import compute_trajectory_health
        h = compute_trajectory_health(
            satisfaction_history=[0.5] * 20,
            action_efficacy=0.8,
            prediction_accuracy_trend=0.1,
        )
        assert 0.0 <= h <= 1.0

    def test_high_satisfaction_high_health(self):
        from anima_mcp.preferences import compute_trajectory_health
        h = compute_trajectory_health(
            satisfaction_history=[0.9] * 20,
            action_efficacy=0.9,
            prediction_accuracy_trend=0.1,
        )
        assert h > 0.7

    def test_high_variance_lowers_health(self):
        from anima_mcp.preferences import compute_trajectory_health
        h_stable = compute_trajectory_health(
            satisfaction_history=[0.6] * 20,
            action_efficacy=0.5,
            prediction_accuracy_trend=0.0,
        )
        h_volatile = compute_trajectory_health(
            satisfaction_history=[0.2, 0.9] * 10,
            action_efficacy=0.5,
            prediction_accuracy_trend=0.0,
        )
        assert h_stable > h_volatile


class TestMetaLearningCycle:
    def test_positive_correlation_boosts_weight(self):
        from anima_mcp.preferences import meta_learning_update
        weights = {"warmth": 1.0, "clarity": 1.0, "stability": 1.0, "presence": 1.0}
        correlations = {"warmth": 0.5, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
        new_weights = meta_learning_update(weights, correlations, beta=0.005)
        assert new_weights["warmth"] > 1.0

    def test_negative_correlation_reduces_weight(self):
        from anima_mcp.preferences import meta_learning_update
        weights = {"warmth": 1.0, "clarity": 1.0, "stability": 1.0, "presence": 1.0}
        correlations = {"warmth": -0.5, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
        new_weights = meta_learning_update(weights, correlations, beta=0.005)
        assert new_weights["warmth"] < 1.0

    def test_floor_preserved_after_update(self):
        from anima_mcp.preferences import meta_learning_update
        weights = {"warmth": 0.3, "clarity": 1.0, "stability": 1.0, "presence": 1.0}
        correlations = {"warmth": -1.0, "clarity": 0.0, "stability": 0.0, "presence": 0.0}
        new_weights = meta_learning_update(weights, correlations, beta=0.005)
        assert new_weights["warmth"] >= 0.3

    def test_conservation_after_update(self):
        from anima_mcp.preferences import meta_learning_update
        weights = {"warmth": 1.0, "clarity": 1.0, "stability": 1.0, "presence": 1.0}
        correlations = {"warmth": 0.8, "clarity": -0.3, "stability": 0.1, "presence": -0.1}
        new_weights = meta_learning_update(weights, correlations, beta=0.005)
        total = sum(new_weights.values())
        assert abs(total - 4.0) < 0.01

    def test_floor_and_conservation_hold_simultaneously_at_extremes(self):
        from anima_mcp.preferences import meta_learning_update

        new_weights = meta_learning_update(
            {"warmth": 100.0, "clarity": 0.3, "stability": 0.3, "presence": 0.3},
            {"warmth": 1.0, "clarity": -1.0, "stability": -1.0, "presence": -1.0},
            beta=1.0,
        )

        assert sum(new_weights.values()) == pytest.approx(4.0)
        assert min(new_weights.values()) >= 0.3

    def test_non_finite_proposal_is_neutralized_at_floor(self):
        from anima_mcp.preferences import meta_learning_update

        new_weights = meta_learning_update(
            {"warmth": float("inf"), "clarity": 1.0, "stability": 1.0},
            {},
            beta=0.0,
        )

        assert all(math.isfinite(weight) for weight in new_weights.values())
        assert new_weights["warmth"] == pytest.approx(0.3)
        assert sum(new_weights.values()) == pytest.approx(4.0)
