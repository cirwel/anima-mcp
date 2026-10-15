"""
Tests for self-model module.

Validates belief updates, predictions, correlation testing, and recovery profiles.
"""

import json
from datetime import datetime, timedelta

import pytest

from anima_mcp.self_model import SelfBelief, SelfModel


def _live_day(model, *, warmth=lambda h: 0.5, clarity=lambda h: 0.5,
              skip_hours=frozenset(), samples_per_hour=3):
    """Feed 2026-10-12 12:00 .. 2026-10-13 23:59 hour by hour.

    Only the 2026-10-13 windows (morning closes 11:00, evening 23:00) have
    enough reference hours behind them, so each belief is judged once.
    """
    start = datetime(2026, 10, 12, 12)
    for offset in range(36):
        when = start + timedelta(hours=offset)
        if when.hour in skip_hours:
            continue
        for _ in range(samples_per_hour):
            model.observe_time_pattern(
                when.hour, warmth=warmth(when.hour),
                clarity=clarity(when.hour), now=when)


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

    def test_zero_strength_is_not_counted_as_evidence(self):
        """A neutral reading cannot inflate the evidence denominator."""
        belief = SelfBelief(belief_id="test", description="test")
        belief.update_from_evidence(supports=False, strength=0.0)
        assert belief.supporting_count == 0
        assert belief.contradicting_count == 0
        assert belief.last_tested is None

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
        """Warmer evenings than the rest of the day support the belief."""
        _live_day(model, warmth=lambda h: 0.7 if 18 <= h <= 22 else 0.4 + 0.01 * (h % 3))
        assert model._beliefs["evening_warmth_increase"].supporting_count == 1
        assert model._beliefs["evening_warmth_increase"].contradicting_count == 0

    def test_observe_time_pattern_morning(self, model):
        """Clearer mornings than the rest of the day support the belief."""
        _live_day(model, clarity=lambda h: 0.9 if 6 <= h <= 10 else 0.6 + 0.01 * (h % 3))
        assert model._beliefs["morning_clarity"].supporting_count == 1

    def test_high_clarity_all_day_is_not_morning_clarity(self, model):
        """The retired test supported whenever morning clarity exceeded 0.5.
        Clarity that is high everywhere, and a little lower in the morning,
        now contradicts instead."""
        _live_day(model, clarity=lambda h: 0.70 if 6 <= h <= 10 else 0.75 + 0.01 * (h % 3))
        belief = model._beliefs["morning_clarity"]
        assert belief.supporting_count == 0
        assert belief.contradicting_count == 1

    def test_window_is_judged_once_after_it_closes(self, model):
        start = datetime(2026, 10, 13, 11)
        belief = model._beliefs["morning_clarity"]
        for offset in range(31):  # 10-13 11:00 .. 10-14 17:00
            when = start + timedelta(hours=offset)
            clarity = 0.9 if 6 <= when.hour <= 10 else 0.6 + 0.01 * (when.hour % 3)
            model.observe_time_pattern(when.hour, warmth=0.5, clarity=clarity, now=when)
            if when < datetime(2026, 10, 14, 11):
                assert (belief.supporting_count, belief.contradicting_count) == (0, 0)
        assert (belief.supporting_count, belief.contradicting_count) == (1, 0)

    def test_sparse_hours_are_unknown(self, model):
        """A broker that saw only two morning hours has no verdict."""
        _live_day(model, clarity=lambda h: 0.9 if 6 <= h <= 10 else 0.6 + 0.01 * (h % 3),
                  skip_hours={6, 7, 8})
        belief = model._beliefs["morning_clarity"]
        assert (belief.supporting_count, belief.contradicting_count) == (0, 0)

    def test_hourly_means_survive_restart(self, tmp_path):
        path = tmp_path / "self_model.json"
        model = SelfModel(persistence_path=path)
        day = datetime(2026, 10, 13, 11)
        for offset in range(20):  # 11:00 .. 06:00 next day
            when = day + timedelta(hours=offset)
            model.observe_time_pattern(when.hour, warmth=0.5,
                                       clarity=0.6 + 0.01 * (when.hour % 3), now=when)
        model._save()

        restarted = SelfModel(persistence_path=path)
        for hour in range(7, 12):  # 07:00 .. 11:00
            when = datetime(2026, 10, 14, hour)
            restarted.observe_time_pattern(hour, warmth=0.5, clarity=0.9, now=when)
        assert restarted.beliefs["morning_clarity"].supporting_count == 1


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
        assert model._stability_episodes[0]["recovery_seconds"] == pytest.approx(30.0)

    def test_recovery_accumulates_elapsed_time_across_partial_steps(self, model):
        model.observe_stability_change(0.8, 0.4, duration_seconds=10)
        model.observe_stability_change(0.4, 0.4, duration_seconds=20)
        model.observe_stability_change(0.4, 0.45, duration_seconds=30)
        model.observe_stability_change(0.45, 0.7, duration_seconds=10)

        episode = model._stability_episodes[0]
        assert episode["recovered"] is True
        assert episode["recovery_seconds"] == pytest.approx(60.0)

    def test_gradual_drop_is_one_episode_at_the_deepest_trough(self, model):
        model.observe_stability_change(0.8, 0.75, duration_seconds=2)
        model.observe_stability_change(0.75, 0.7, duration_seconds=2)
        model.observe_stability_change(0.7, 0.65, duration_seconds=2)

        assert len(model._stability_episodes) == 1
        episode = model._stability_episodes[0]
        assert episode["initial"] == pytest.approx(0.8)
        assert episode["dropped_to"] == pytest.approx(0.65)

        model.observe_stability_change(0.65, 0.76, duration_seconds=12)
        assert episode["recovered"] is True
        assert episode["recovery_seconds"] == pytest.approx(12.0)

    def test_recovery_bonus_widens_threshold(self, model):
        """recovery_bonus makes more recoveries count as 'fast'."""
        # Drop stability
        model.observe_stability_change(0.8, 0.4, duration_seconds=10)
        initial_confidence = model._beliefs["stability_recovery"].confidence

        # Recover with bonus — the wider threshold should make this count as fast
        model.observe_stability_change(0.4, 0.7, duration_seconds=30,
                                       recovery_bonus=0.30)

        # With 30% bonus, threshold is 600 * 1.30 = 780s per unit
        # Recovery: time / amount = 30 / 0.3 = 100 — well under threshold
        # So it should count as supporting evidence
        assert model._beliefs["stability_recovery"].confidence > initial_confidence


class TestPersistence:
    """Test model save/load."""

    def test_save_and_load(self, tmp_path):
        """Test that beliefs persist across instances."""
        path = tmp_path / "self_model.json"

        model1 = SelfModel(persistence_path=path)
        # Repeated ticks inside one five-minute episode count once.
        for _ in range(5):
            model1.observe_surprise(0.8, ["light"])
        model1.save()

        # Reload
        model2 = SelfModel(persistence_path=path)
        assert model2._beliefs["light_sensitive"].supporting_count == 1
        assert model2._beliefs["light_sensitive"].confidence > 0.5

    def test_read_only_reader_refreshes_without_writing(self, tmp_path):
        path = tmp_path / "self_model.json"
        writer = SelfModel(persistence_path=path)
        writer.apply_evidence("light_sensitive", supports=True, strength=1.0)
        writer.save()

        reader = SelfModel(persistence_path=path, read_only=True)
        before = reader.beliefs["light_sensitive"].supporting_count
        reader.apply_evidence("light_sensitive", supports=True, strength=1.0)
        reader.save()
        assert reader.beliefs["light_sensitive"].supporting_count == before

        writer.apply_evidence("light_sensitive", supports=True, strength=1.0)
        writer.save()
        assert reader.refresh_if_changed()
        assert reader.beliefs["light_sensitive"].supporting_count == before + 1

    def test_cadence_corrupted_beliefs_cold_start_once(self, tmp_path):
        path = tmp_path / "self_model.json"
        path.write_text(json.dumps({
            "beliefs": {
                "morning_clarity": {
                    "confidence": 1.0,
                    "value": 1.0,
                    "supporting_count": 387287,
                    "contradicting_count": 0,
                },
            },
            "_migrated_noise_reset": True,
        }))

        model = SelfModel(persistence_path=path)
        belief = model.beliefs["morning_clarity"]
        assert (belief.confidence, belief.value) == (0.3, 0.5)
        assert belief.supporting_count == 0
        assert json.loads(path.read_text())["_migrated_episode_evidence_v2"] is True

        model.apply_evidence("morning_clarity", supports=True, strength=1.0)
        model.save()
        restarted = SelfModel(persistence_path=path)
        assert restarted.beliefs["morning_clarity"].supporting_count == 1

    def test_raw_light_warmth_belief_is_cold_started_for_residual_v4(self, tmp_path):
        path = tmp_path / "self_model.json"
        path.write_text(json.dumps({
            "beliefs": {
                "light_warmth_correlation": {
                    "confidence": 0.99,
                    "value": 0.9,
                    "supporting_count": 400,
                    "contradicting_count": 2,
                },
            },
            "_migrated_noise_reset": True,
            "_migrated_episode_evidence_v2": True,
            "_migrated_dead_channel_reset_v3": True,
        }))

        model = SelfModel(persistence_path=path)
        belief = model.beliefs["light_warmth_correlation"]
        assert (belief.confidence, belief.value) == (0.5, 0.5)
        assert (belief.supporting_count, belief.contradicting_count) == (0, 0)

        saved = json.loads(path.read_text())
        audit = saved["_migrated_light_attribution_reset_v4"]
        assert audit["light_warmth_correlation"]["supporting_count"] == 400

    def test_clarity_target_beliefs_cold_start_after_equation_change_v5(self, tmp_path):
        path = tmp_path / "self_model.json"
        path.write_text(json.dumps({
            "beliefs": {
                "temp_clarity_correlation": {
                    "confidence": 0.95,
                    "value": 0.8,
                    "supporting_count": 80,
                    "contradicting_count": 4,
                },
                "interaction_clarity_boost": {
                    "confidence": 0.9,
                    "value": 0.9,
                    "supporting_count": 20,
                    "contradicting_count": 1,
                },
                "morning_clarity": {
                    "confidence": 0.88,
                    "value": 0.85,
                    "supporting_count": 30,
                    "contradicting_count": 2,
                },
            },
            "_migrated_noise_reset": True,
            "_migrated_episode_evidence_v2": True,
            "_migrated_dead_channel_reset_v3": True,
            "_migrated_light_attribution_reset_v4": True,
        }))

        model = SelfModel(persistence_path=path)

        expected = {
            "temp_clarity_correlation": (0.5, 0.5),
            "interaction_clarity_boost": (0.5, 0.7),
            "morning_clarity": (0.3, 0.5),
        }
        for belief_id, prior in expected.items():
            belief = model.beliefs[belief_id]
            assert (belief.confidence, belief.value) == prior
            assert (belief.supporting_count, belief.contradicting_count) == (0, 0)

        saved = json.loads(path.read_text())
        audit = saved["_migrated_clarity_semantics_reset_v5"]
        assert audit["temp_clarity_correlation"]["supporting_count"] == 80


class TestEpisodeEvidence:
    def test_time_pattern_counts_once_per_day(self, tmp_path):
        model = SelfModel(persistence_path=tmp_path / "self_model.json")
        _live_day(model, clarity=lambda h: 0.9 if 6 <= h <= 10 else 0.6 + 0.01 * (h % 3),
                  samples_per_hour=40)
        assert model.beliefs["morning_clarity"].supporting_count == 1

    def test_v7_migration_retires_absolute_time_evidence(self, tmp_path):
        path = tmp_path / "self_model.json"
        path.write_text(json.dumps({
            "beliefs": {
                "morning_clarity": {
                    "confidence": 0.86, "value": 0.79,
                    "supporting_count": 42, "contradicting_count": 0,
                },
                "evening_warmth_increase": {
                    "confidence": 0.4, "value": 0.45,
                    "supporting_count": 3, "contradicting_count": 9,
                },
            },
            "evidence_buckets": {
                "time:morning_clarity": "2026-10-14",
                "time:evening_warmth": "2026-10-13",
            },
            "_migrated_noise_reset": True,
            "_migrated_episode_evidence_v2": True,
            "_migrated_dead_channel_reset_v3": True,
            "_migrated_light_attribution_reset_v4": True,
            "_migrated_clarity_semantics_reset_v5": True,
            "_migrated_led_causal_evidence_reset_v6": True,
        }))

        model = SelfModel(persistence_path=path)

        for belief_id in ("morning_clarity", "evening_warmth_increase"):
            belief = model.beliefs[belief_id]
            assert (belief.confidence, belief.value) == (0.3, 0.5)
            assert (belief.supporting_count, belief.contradicting_count) == (0, 0)
        assert "time:morning_clarity" not in model._evidence_buckets
        assert "time:evening_warmth" not in model._evidence_buckets

        audit = json.loads(path.read_text())["_migrated_time_pattern_relative_reset_v7"]
        assert audit["morning_clarity"] == {
            "confidence": 0.86, "value": 0.79,
            "supporting_count": 42, "contradicting_count": 0,
            "retired_evidence": "absolute_threshold_in_window",
        }

        # A second load must not reset again.
        model.apply_evidence("morning_clarity", supports=True)
        model._save()
        again = SelfModel(persistence_path=path)
        assert again.beliefs["morning_clarity"].supporting_count == 1

    def test_temperament_counts_once_per_hour(self, tmp_path):
        model = SelfModel(persistence_path=tmp_path / "self_model.json")
        for _ in range(100):
            model.observe_temperament({"warmth": 0.2, "presence": 0.2})
        assert model.beliefs["warmth_baseline_low"].supporting_count == 1
        assert model.beliefs["presence_baseline_low"].supporting_count == 1


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
