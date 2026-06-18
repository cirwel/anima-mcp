"""
Tests for self-model module.

Validates belief updates, predictions, correlation testing, and recovery profiles.
"""

import pytest

from anima_mcp.self_model import SelfBelief, SelfModel


class TestSelfBelief:
    """Test the SelfBelief dataclass."""

    def test_supporting_evidence_increases_confidence(self):
        """Test that supporting evidence raises confidence."""
        belief = SelfBelief(belief_id="test", description="test belief", confidence=0.5)
        initial = belief.confidence
        belief.update_from_evidence(supports=True, strength=1.0)
        assert belief.confidence > initial

    def test_contradicting_evidence_decreases_confidence(self):
        """Test that contradicting evidence lowers confidence."""
        belief = SelfBelief(belief_id="test", description="test belief", confidence=0.7)
        initial = belief.confidence
        belief.update_from_evidence(supports=False, strength=1.0)
        assert belief.confidence < initial

    def test_supporting_evidence_increases_value(self):
        """Test that supporting evidence raises value."""
        belief = SelfBelief(belief_id="test", description="test", confidence=0.5, value=0.5)
        belief.update_from_evidence(supports=True, strength=1.0)
        assert belief.value > 0.5

    def test_contradicting_evidence_moves_value_toward_half(self):
        """Test that contradicting evidence pulls value toward 0.5."""
        belief = SelfBelief(belief_id="test", description="test", confidence=0.8, value=0.9)
        belief.update_from_evidence(supports=False, strength=1.0)
        assert belief.value < 0.9

    def test_confidence_clamped_to_zero_one(self):
        """Test confidence stays in [0, 1]."""
        belief = SelfBelief(belief_id="test", description="test", confidence=0.99)
        for _ in range(50):
            belief.update_from_evidence(supports=True, strength=1.0)
        assert belief.confidence <= 1.0

        belief2 = SelfBelief(belief_id="test", description="test", confidence=0.01)
        for _ in range(50):
            belief2.update_from_evidence(supports=False, strength=1.0)
        assert belief2.confidence >= 0.0

    def test_update_bonus_speeds_learning(self):
        """update_bonus increases the belief learning rate."""
        b_normal = SelfBelief(belief_id="t1", description="test", confidence=0.5)
        b_bonus = SelfBelief(belief_id="t2", description="test", confidence=0.5)

        b_normal.update_from_evidence(supports=True, strength=1.0, update_bonus=0.0)
        b_bonus.update_from_evidence(supports=True, strength=1.0, update_bonus=0.15)

        assert b_bonus.confidence > b_normal.confidence

    def test_evidence_counts_tracked(self):
        """Test that evidence counts are tracked."""
        belief = SelfBelief(belief_id="test", description="test")
        belief.update_from_evidence(supports=True)
        belief.update_from_evidence(supports=True)
        belief.update_from_evidence(supports=False)
        assert belief.supporting_count == 2
        assert belief.contradicting_count == 1

    def test_last_tested_updated(self):
        """Test that last_tested is set on update."""
        belief = SelfBelief(belief_id="test", description="test")
        assert belief.last_tested is None
        belief.update_from_evidence(supports=True)
        assert belief.last_tested is not None

    def test_belief_strength_uncertain_with_few_samples(self):
        """Test uncertain when < 3 evidence points."""
        belief = SelfBelief(belief_id="test", description="test")
        belief.update_from_evidence(supports=True)
        assert belief.get_belief_strength() == "uncertain"

    def test_belief_strength_categories(self):
        """Test belief strength categories based on confidence."""
        belief = SelfBelief(belief_id="test", description="test",
                           supporting_count=5, contradicting_count=5)

        belief.confidence = 0.2
        assert belief.get_belief_strength() == "doubtful"

        belief.confidence = 0.5
        assert belief.get_belief_strength() == "moderate"

        belief.confidence = 0.7
        assert belief.get_belief_strength() == "confident"

        belief.confidence = 0.9
        assert belief.get_belief_strength() == "very confident"


class TestSelfModel:
    """Test the SelfModel (beliefs, predictions, observations)."""

    @pytest.fixture
    def model(self, tmp_path):
        """Create a SelfModel with temp persistence."""
        return SelfModel(persistence_path=tmp_path / "self_model.json")

    def test_initial_beliefs_exist(self, model):
        """Test that model initializes with expected beliefs."""
        summary = model.get_belief_summary()
        assert "light_sensitive" in summary
        assert "stability_recovery" in summary
        assert "interaction_clarity_boost" in summary

    def test_observe_surprise_updates_light_belief(self, model):
        """Test that light surprise updates light sensitivity belief."""
        initial = model._beliefs["light_sensitive"].confidence
        model.observe_surprise(0.8, ["light"])
        assert model._beliefs["light_sensitive"].confidence != initial

    def test_observe_surprise_updates_temp_belief(self, model):
        """Test that temp surprise updates temp sensitivity belief."""
        initial = model._beliefs["temp_sensitive"].confidence
        model.observe_surprise(0.5, ["ambient_temp"])
        assert model._beliefs["temp_sensitive"].confidence != initial

    def test_observe_surprise_ignores_unrelated_sources(self, model):
        """Test that unrelated sources don't affect sensitivity beliefs."""
        light_before = model._beliefs["light_sensitive"].confidence
        temp_before = model._beliefs["temp_sensitive"].confidence
        model.observe_surprise(0.5, ["cpu"])
        assert model._beliefs["light_sensitive"].confidence == light_before
        assert model._beliefs["temp_sensitive"].confidence == temp_before

    def test_observe_interaction_positive(self, model):
        """Test that clarity increase supports interaction belief."""
        model.observe_interaction(clarity_before=0.4, clarity_after=0.7)
        assert model._beliefs["interaction_clarity_boost"].supporting_count == 1

    def test_observe_interaction_negative(self, model):
        """Test that clarity decrease contradicts interaction belief."""
        model.observe_interaction(clarity_before=0.7, clarity_after=0.4)
        assert model._beliefs["interaction_clarity_boost"].contradicting_count == 1

    def test_observe_time_pattern_evening(self, model):
        """Test evening warmth observation."""
        model.observe_time_pattern(hour=20, warmth=0.8, clarity=0.5)
        assert model._beliefs["evening_warmth_increase"].supporting_count == 1

    def test_observe_time_pattern_morning(self, model):
        """Test morning clarity observation."""
        model.observe_time_pattern(hour=8, warmth=0.5, clarity=0.8)
        assert model._beliefs["morning_clarity"].supporting_count == 1


class TestPredictions:
    """Test self-model predictions."""

    @pytest.fixture
    def model(self, tmp_path):
        return SelfModel(persistence_path=tmp_path / "self_model.json")

    def test_predict_light_change(self, model):
        """Test prediction for light change context."""
        predictions = model.predict_own_response("light_change")
        assert "surprise_likelihood" in predictions
        assert "warmth_change" in predictions

    def test_predict_temp_change(self, model):
        """Test prediction for temp change context."""
        predictions = model.predict_own_response("temp_change")
        assert "surprise_likelihood" in predictions
        assert "clarity_change" in predictions

    def test_predict_stability_drop(self, model):
        """Test prediction for stability drop context."""
        predictions = model.predict_own_response("stability_drop")
        assert "fast_recovery" in predictions

    def test_predict_unknown_context(self, model):
        """Unknown context returns empty predictions."""
        predictions = model.predict_own_response("unknown_thing")
        assert predictions == {}

    def test_verify_prediction_accurate(self, model):
        """Accurate prediction boosts belief confidence."""
        initial_conf = model._beliefs["light_sensitive"].confidence
        prediction = model.predict_own_response("light_change")
        # Actual matches prediction closely
        actual = {k: v for k, v in prediction.items()}
        model.verify_prediction("light_change", prediction, actual)
        assert model._beliefs["light_sensitive"].confidence >= initial_conf

    def test_verify_prediction_inaccurate(self, model):
        """Inaccurate prediction reduces belief confidence and nudges value."""
        # Set belief to a non-default value so prediction differs from actual
        model._beliefs["light_sensitive"].value = 0.8
        initial_conf = model._beliefs["light_sensitive"].confidence
        prediction = model.predict_own_response("light_change")
        # Actual surprise is much lower than predicted
        actual = {"surprise_likelihood": 0.2, "warmth_change": 0.1}
        model.verify_prediction("light_change", prediction, actual)
        assert model._beliefs["light_sensitive"].confidence <= initial_conf


class TestRecoveryProfile:
    """Test recovery profile / tau estimation."""

    @pytest.fixture
    def model(self, tmp_path):
        return SelfModel(persistence_path=tmp_path / "self_model.json")

    def test_no_episodes_returns_none(self, model):
        """Test empty recovery profile when no episodes."""
        profile = model.get_recovery_profile()
        assert profile["tau_estimate"] is None
        assert profile["n_episodes"] == 0

    def test_recovery_episode_tracked(self, model):
        """Test that stability drop + recovery creates an episode."""
        # Simulate drop
        model.observe_stability_change(0.8, 0.4, duration_seconds=10)
        assert len(model._stability_episodes) == 1
        assert model._stability_episodes[0]["recovered"] is False

    def test_recovery_completes_episode(self, model):
        """Test that recovery after a drop completes the episode."""
        model.observe_stability_change(0.8, 0.4, duration_seconds=10)
        model.observe_stability_change(0.4, 0.7, duration_seconds=30)
        assert model._stability_episodes[0]["recovered"] is True

    def test_recovery_bonus_widens_threshold(self, model):
        """recovery_bonus makes more recoveries count as 'fast'."""
        # Drop stability
        model.observe_stability_change(0.8, 0.4, duration_seconds=10)
        initial_confidence = model._beliefs["stability_recovery"].confidence

        # Recover with bonus — the wider threshold should make this count as fast
        model.observe_stability_change(0.4, 0.7, duration_seconds=30,
                                       recovery_bonus=0.30)

        # With 30% bonus, threshold is 600 * 1.30 = 780s per unit
        # Recovery: time / amount = ~few seconds / 0.3 — well under threshold
        # So it should count as supporting evidence
        assert model._beliefs["stability_recovery"].confidence > initial_confidence


class TestPersistence:
    """Test model save/load."""

    def test_save_and_load(self, tmp_path):
        """Test that beliefs persist across instances."""
        path = tmp_path / "self_model.json"

        model1 = SelfModel(persistence_path=path)
        # Build up some evidence
        for _ in range(5):
            model1.observe_surprise(0.8, ["light"])
        model1.save()

        # Reload
        model2 = SelfModel(persistence_path=path)
        assert model2._beliefs["light_sensitive"].supporting_count == 5
        assert model2._beliefs["light_sensitive"].confidence > 0.5


class TestSelfDescription:
    """Test natural language self-description."""

    def test_low_confidence_returns_learning(self, tmp_path):
        """Test that all-low confidence gives 'still learning' message."""
        model = SelfModel(persistence_path=tmp_path / "m.json")
        # Reset all beliefs to low confidence so none pass the 0.4 threshold
        for belief in model._beliefs.values():
            belief.confidence = 0.2
        desc = model.get_self_description()
        assert "learning" in desc.lower()

    def test_confident_belief_appears_in_description(self, tmp_path):
        """Test that confident beliefs appear in description."""
        model = SelfModel(persistence_path=tmp_path / "m.json")
        # Build up light sensitivity belief
        model._beliefs["light_sensitive"].confidence = 0.8
        model._beliefs["light_sensitive"].value = 0.9
        model._beliefs["light_sensitive"].supporting_count = 10
        model._beliefs["light_sensitive"].contradicting_count = 1

        desc = model.get_self_description()
        assert "light" in desc.lower()


class TestBeliefSignature:
    """Test trajectory signature extraction."""

    def test_belief_signature_structure(self, tmp_path):
        """Test belief signature has expected fields."""
        model = SelfModel(persistence_path=tmp_path / "m.json")
        sig = model.get_belief_signature()
        assert "values" in sig
        assert "confidences" in sig
        assert "labels" in sig
        assert "total_evidence" in sig
        assert "avg_confidence" in sig
        assert sig["n_beliefs"] == len(model._beliefs)
