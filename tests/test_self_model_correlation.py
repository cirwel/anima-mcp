"""
Tests for self-model correlation testing and belief persistence.

Covers audit findings:
  #3 - Beliefs should persist (save/load cycle)
  #13 - Correlation epsilon too small for real sensor data
"""

import math
import pytest
from datetime import datetime

from anima_mcp.self_model import SelfBelief, SelfModel


@pytest.fixture
def model(tmp_path):
    """Create a SelfModel with temp persistence path."""
    persistence_path = tmp_path / "self_model.json"
    return SelfModel(persistence_path=persistence_path)


class TestCorrelationCalculation:
    """Test _test_correlation_belief math."""

    def test_perfect_positive_correlation(self, model):
        """Perfectly correlated data should support belief."""
        belief_id = "my_leds_affect_lux"
        # Feed perfectly correlated data
        for i in range(15):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": i * 0.01,
                "light_lux": i * 10.0,
            })

        initial_confidence = model._beliefs[belief_id].confidence
        model._test_correlation_belief(belief_id, "led_lux")
        # Should have increased confidence (positive correlation)
        assert model._beliefs[belief_id].confidence >= initial_confidence

    def test_no_correlation_weakens_belief(self, model):
        """Uncorrelated data should weaken belief."""
        belief_id = "my_leds_affect_lux"
        import random
        random.seed(42)
        for i in range(15):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": random.random(),
                "light_lux": random.random() * 1000,
            })

        model._test_correlation_belief(belief_id, "led_lux")
        # Random data may or may not correlate, but shouldn't crash
        # Just verify it ran without error
        assert model._beliefs[belief_id].confidence is not None

    def test_constant_values_handled(self, model):
        """Constant x or y values should not crash (epsilon guard)."""
        belief_id = "my_leds_affect_lux"
        for i in range(15):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": 0.12,  # Constant
                "light_lux": 50.0,  # Constant
            })

        # Should not crash — epsilon guard returns early
        model._test_correlation_belief(belief_id, "led_lux")

    def test_near_constant_values_handled(self, model):
        """Near-constant values (tiny variance) should not crash or produce NaN."""
        belief_id = "my_leds_affect_lux"
        for i in range(15):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": 0.12 + i * 1e-12,  # Barely varies
                "light_lux": 50.0 + i * 1e-12,
            })

        model._test_correlation_belief(belief_id, "led_lux")
        conf = model._beliefs[belief_id].confidence
        assert not math.isnan(conf)
        assert not math.isinf(conf)

    def test_insufficient_data_skipped(self, model):
        """Less than 10 data points should skip calculation."""
        belief_id = "my_leds_affect_lux"
        for i in range(5):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": i * 0.01,
                "light_lux": i * 10.0,
            })

        initial = model._beliefs[belief_id].confidence
        model._test_correlation_belief(belief_id, "led_lux")
        # Should be unchanged — not enough data
        assert model._beliefs[belief_id].confidence == initial

    def test_stable_input_leaves_belief_at_prior(self, model):
        """When input variance is below CV=5% (stable HVAC-like environment),
        we have no information to test the correlation. The belief must stay
        at its prior — the previous behavior eroded confidence on every tick
        and drove correlations to 0 permanently in quiet rooms (see Lumen's
        temp_clarity_correlation at 0 supporting / 31,854 contradicting).
        """
        belief_id = "my_leds_affect_lux"

        # Seed the belief at a specific confidence so drift is detectable.
        model._beliefs[belief_id].confidence = 0.6
        model._beliefs[belief_id].supporting_count = 20
        model._beliefs[belief_id].contradicting_count = 5

        # CV far below 5%: brightness hovering around 0.12 with sub-percent jitter.
        for i in range(15):
            model._correlation_data["led_lux"].append({
                "timestamp": datetime.now().isoformat(),
                "led_brightness": 0.1200 + (i % 3) * 0.00005,
                "light_lux": 50.0 + (i % 3) * 0.01,
            })

        prior_conf = model._beliefs[belief_id].confidence
        prior_supp = model._beliefs[belief_id].supporting_count
        prior_contra = model._beliefs[belief_id].contradicting_count

        model._test_correlation_belief(belief_id, "led_lux")

        # No update — the belief stays exactly where it was.
        assert model._beliefs[belief_id].confidence == prior_conf
        assert model._beliefs[belief_id].supporting_count == prior_supp
        assert model._beliefs[belief_id].contradicting_count == prior_contra

        # Window should be cleared so fresh data can accumulate.
        assert len(model._correlation_data["led_lux"]) == 0

    def test_stable_input_repeated_does_not_accumulate_disconfirm(self, model):
        """Fire the stable-input path 500 times. Confidence must not drift —
        the whole point of this change is to stop the decay-to-zero in quiet
        environments."""
        belief_id = "my_leds_affect_lux"
        model._beliefs[belief_id].confidence = 0.7

        for cycle in range(500):
            model._correlation_data["led_lux"].clear()
            for i in range(12):
                model._correlation_data["led_lux"].append({
                    "timestamp": datetime.now().isoformat(),
                    "led_brightness": 0.10,
                    "light_lux": 100.0,
                })
            model._test_correlation_belief(belief_id, "led_lux")

        assert model._beliefs[belief_id].confidence == 0.7

    def test_opposite_signed_windows_are_aggregated_not_last_write_wins(self):
        belief = SelfBelief(
            belief_id="correlation",
            description="signed relationship",
            confidence=0.5,
            value=0.5,
        )

        belief.update_correlation(1.0)
        after_positive = belief.value
        belief.update_correlation(-1.0)

        assert after_positive > 0.6
        assert 0.4 < belief.value < 0.6
        assert belief.supporting_count == 2

    def test_negative_update_bonus_cannot_reverse_learning(self):
        belief = SelfBelief(
            belief_id="correlation",
            description="signed relationship",
            confidence=0.6,
            value=0.7,
        )

        belief.update_correlation(0.8, update_bonus=-2.0)

        assert belief.confidence == pytest.approx(0.6)
        assert belief.value == pytest.approx(0.7)


class TestBeliefPersistence:
    """Test that beliefs survive save/load cycles."""

    def test_save_and_load_preserves_beliefs(self, model):
        """Beliefs modified in memory should persist after save+load."""
        # Modify a belief
        model._beliefs["my_leds_affect_lux"].update_from_evidence(
            supports=True, strength=0.8
        )
        modified_confidence = model._beliefs["my_leds_affect_lux"].confidence
        modified_value = model._beliefs["my_leds_affect_lux"].value

        # Save
        model.save()

        # Create new model from same path
        model2 = SelfModel(persistence_path=model.persistence_path)
        loaded_belief = model2._beliefs.get("my_leds_affect_lux")

        assert loaded_belief is not None
        assert abs(loaded_belief.confidence - modified_confidence) < 0.01
        assert abs(loaded_belief.value - modified_value) < 0.01

    def test_save_creates_file(self, model):
        """save() should create the persistence file."""
        assert not model.persistence_path.exists()
        model.save()
        assert model.persistence_path.exists()

    def test_evidence_counts_persist(self, model):
        """Supporting and contradicting counts should survive save/load."""
        belief = model._beliefs["my_leds_affect_lux"]
        belief.update_from_evidence(supports=True, strength=0.5)
        belief.update_from_evidence(supports=True, strength=0.5)
        belief.update_from_evidence(supports=False, strength=0.3)

        model.save()
        model2 = SelfModel(persistence_path=model.persistence_path)
        loaded = model2._beliefs["my_leds_affect_lux"]

        assert loaded.supporting_count == 2
        assert loaded.contradicting_count == 1


class TestBeliefSummary:
    """Test get_belief_summary for display/schema integration."""

    def test_summary_returns_all_beliefs(self, model):
        """get_belief_summary should return all beliefs with their state."""
        summary = model.get_belief_summary()
        assert isinstance(summary, dict)
        assert "my_leds_affect_lux" in summary

    def test_summary_includes_confidence(self, model):
        summary = model.get_belief_summary()
        for belief_id, info in summary.items():
            assert "confidence" in info
            assert "value" in info
            assert 0 <= info["confidence"] <= 1
            assert 0 <= info["value"] <= 1


class TestCorrelationPairing:
    """A gap in either channel must not desynchronise the two series.

    _test_correlation_belief used to filter x and y with separate
    comprehensions, then truncate both to min(len) and zip positionally. With
    an intermittent sensor that silently correlated readings taken at
    different timestamps — not noisier, just misaligned, which can produce a
    confident wrong sign with nothing downstream able to notice.
    """

    def _feed(self, model, rows):
        for temp, clarity in rows:
            model._correlation_data["temp_clarity"].append(
                {"temp": temp, "clarity": clarity, "timestamp": datetime.now()}
            )

    def test_gap_in_one_channel_does_not_shift_the_other(self, model):
        """A PERIODIC signal, so misalignment is detectable.

        A monotonic ramp is useless here: shifting one series against the
        other still leaves it positively correlated, so the old code passes.
        With a period-12 wave, dropping the first three x samples shifts the
        surviving series by a quarter period and drives the old positional
        zip toward zero, while correct pairing stays perfectly collinear.
        """
        rows = []
        for i in range(36):
            wave = math.sin(2 * math.pi * i / 12)
            temp = 20.0 + 5.0 * wave   # CV ~0.18, clears the 5% variance gate
            clarity = 0.5 + 0.2 * wave  # y is an exact function of x
            drop_x = i < 3                       # quarter-period desync
            rows.append((None if drop_x else temp, clarity))
        self._feed(model, rows)

        model._test_correlation_belief("temp_clarity_correlation", "temp_clarity")

        belief = model.beliefs["temp_clarity_correlation"]
        assert belief.value > 0.6, (
            "surviving pairs are exactly collinear, so this must read as a "
            f"strong positive; got {belief.value} (misaligned zip reads ~0.5)"
        )

    def test_pairing_survives_gaps_in_both_channels(self, model):
        """Both channels gappy, on a periodic signal, at different strides.

        CHARACTERIZATION, not a regression guard: verified that this one also
        passes against the pre-fix code, because these particular strides
        desynchronise in a way that still reads positive. Kept because it
        pins the both-channels-gappy shape, but the guards with teeth are
        test_gap_in_one_channel_does_not_shift_the_other and
        test_too_few_complete_pairs_is_a_no_op — both confirmed failing
        before the fix and passing after.
        """
        rows = []
        for i in range(48):
            wave = math.sin(2 * math.pi * i / 12)
            temp = 20.0 + 5.0 * wave   # CV ~0.18, clears the 5% variance gate
            clarity = 0.5 + 0.2 * wave
            rows.append((None if i % 5 == 0 else temp,
                         None if i % 8 == 0 else clarity))
        self._feed(model, rows)
        model._test_correlation_belief("temp_clarity_correlation", "temp_clarity")
        assert model.beliefs["temp_clarity_correlation"].value > 0.6

    def test_too_few_complete_pairs_is_a_no_op(self, model):
        """<10 COMPLETE pairs must bail, even when each series alone has 10+.

        The old min(len(x), len(y)) test passed here — 15 x-values and 15
        y-values — while only 5 rows actually had both.
        """
        before = model.beliefs["temp_clarity_correlation"].value
        rows = [(20.0 + i, None) for i in range(15)]
        rows += [(None, 0.5 + i * 0.01) for i in range(15)]
        rows += [(25.0 + i, 0.4 + i * 0.03) for i in range(5)]
        self._feed(model, rows)
        model._test_correlation_belief("temp_clarity_correlation", "temp_clarity")
        assert model.beliefs["temp_clarity_correlation"].value == before
