"""
Integration tests for the experiential accumulation system.

Tests that all three layers (pathways, filter, marks) work together
and integrate correctly with existing systems.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.weighted_pathways import (
    WeightedPathways,
    discretize_context,
)
from anima_mcp.experiential_filter import ExperientialFilter, DIMENSIONS
from anima_mcp.experiential_marks import ExperientialMarks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    return str(tmp_path / "test_integration.db")


@pytest.fixture
def tmp_filter_path(tmp_path):
    """Create a temporary path for filter persistence."""
    return str(tmp_path / "test_filter.json")


@pytest.fixture
def pathways(tmp_db):
    pw = WeightedPathways(db_path=tmp_db)
    yield pw
    pw.close()


@pytest.fixture
def exp_filter(tmp_filter_path):
    return ExperientialFilter(persistence_path=tmp_filter_path)


@pytest.fixture
def marks(tmp_db):
    m = ExperientialMarks(db_path=tmp_db)
    yield m
    m.close()


# ---------------------------------------------------------------------------
# Layer 1: Pathway reinforcement across iterations
# ---------------------------------------------------------------------------


class TestPathwayReinforcementLoop:
    """Simulate multiple iterations of context -> action -> outcome -> reinforce."""

    def test_pathway_strengthens_with_positive_outcomes(self, pathways):
        ctx = discretize_context(surprise=0.1, satisfaction=0.7, drive=0.1, activity="active")
        initial = pathways.get_all_strengths(ctx).get("focus_attention", 0.5)

        for _ in range(10):
            pathways.reinforce(ctx, "focus_attention", 0.3)

        final = pathways.get_all_strengths(ctx)["focus_attention"]
        assert final > initial, "Pathway should strengthen with positive outcomes"

    def test_pathway_weakens_with_negative_outcomes(self, pathways):
        ctx = discretize_context(surprise=0.5, satisfaction=0.3, drive=0.6, activity="active")
        for _ in range(5):
            pathways.reinforce(ctx, "adjust_sensitivity", 0.5)
        mid = pathways.get_all_strengths(ctx)["adjust_sensitivity"]

        for _ in range(10):
            pathways.reinforce(ctx, "adjust_sensitivity", -0.5)
        final = pathways.get_all_strengths(ctx)["adjust_sensitivity"]
        assert final < mid, "Pathway should weaken with negative outcomes"

    def test_different_contexts_diverge(self, pathways):
        ctx_calm = discretize_context(surprise=0.05, satisfaction=0.8, drive=0.1, activity="active")
        ctx_stressed = discretize_context(surprise=0.5, satisfaction=0.2, drive=0.7, activity="active")

        for _ in range(10):
            pathways.reinforce(ctx_calm, "rest", 0.5)
            pathways.reinforce(ctx_stressed, "rest", -0.3)

        calm_strength = pathways.get_all_strengths(ctx_calm).get("rest", 0.5)
        stressed_strength = pathways.get_all_strengths(ctx_stressed).get("rest", 0.5)
        assert calm_strength > stressed_strength


# ---------------------------------------------------------------------------
# Layer 2: Filter changes anima output
# ---------------------------------------------------------------------------


class TestFilterChangesAnima:
    """Verify that salience weights change anima computation."""

    def test_salience_weights_accepted_by_sense_self(self, exp_filter):
        """sense_self should accept salience_weights without error."""
        from anima_mcp.anima import sense_self
        from anima_mcp.sensors.base import SensorReadings

        readings = SensorReadings(
            timestamp=datetime.now(),
            cpu_temp_c=55.0,
            ambient_temp_c=24.0,
            humidity_pct=50.0,
            light_lux=10.0,
            pressure_hpa=1013.0,
            cpu_percent=30.0,
            memory_percent=40.0,
        )

        # Call without weights
        baseline = sense_self(readings)
        assert 0.0 <= baseline.warmth <= 1.0

        # Call with weights — should succeed and return valid anima
        exp_filter._weights["light"].value = 2.0
        exp_filter._weights["cpu_temp"].value = 0.5
        exp_filter._weights["humidity"].value = 2.0
        saliences = exp_filter.get_all_saliences()
        weighted = sense_self(readings, salience_weights=saliences)
        assert 0.0 <= weighted.warmth <= 1.0
        assert 0.0 <= weighted.clarity <= 1.0
        assert 0.0 <= weighted.stability <= 1.0
        assert 0.0 <= weighted.presence <= 1.0

    def test_salience_dict_structure(self, exp_filter):
        """ExperientialFilter produces correct dict for sense_self consumption."""
        saliences = exp_filter.get_all_saliences()
        assert isinstance(saliences, dict)
        assert "light" in saliences
        assert "cpu_temp" in saliences
        assert "humidity" in saliences
        # All start neutral
        for v in saliences.values():
            assert abs(v - 1.0) < 0.001

    def test_surprise_amplifies_dimension(self, exp_filter):
        """Surprise on a sensor should increase its salience."""
        initial = exp_filter.get_all_saliences()["light"]
        exp_filter.update_from_surprise(["light"], 0.5)
        after = exp_filter.get_all_saliences()["light"]
        assert after > initial

    def test_decay_toward_neutral(self, exp_filter):
        """Salience should decay toward 1.0 over time."""
        exp_filter._weights["cpu_temp"].value = 1.8
        for _ in range(1000):
            exp_filter.tick()
        val = exp_filter.get_all_saliences()["cpu_temp"]
        assert val < 1.8, "Should have decayed"
        assert val >= 1.0, "Should not decay below neutral"


# ---------------------------------------------------------------------------
# Layer 3: Marks modify system parameters
# ---------------------------------------------------------------------------


class TestMarksModifyParameters:
    """Verify that earned marks produce concrete effects."""

    def test_pathway_lr_bonus_from_infant_mark(self, marks):
        assert marks.get_effect("pathway_lr_bonus") == 0.0
        marks.check_and_earn(observation_count=1000)
        assert marks.get_effect("pathway_lr_bonus") > 0.0

    def test_stability_recovery_bonus_stacks(self, marks):
        marks.check_and_earn(awakenings=2)
        bonus_2 = marks.get_effect("stability_recovery_bonus")
        marks.check_and_earn(awakenings=10)
        bonus_10 = marks.get_effect("stability_recovery_bonus")
        assert bonus_10 > bonus_2, "Veteran should stack on First Return"

    def test_marks_are_permanent(self, tmp_db):
        m1 = ExperientialMarks(db_path=tmp_db)
        m1.check_and_earn(awakenings=2)
        assert m1.get_effect("stability_recovery_bonus") > 0
        m1.close()

        m2 = ExperientialMarks(db_path=tmp_db)
        assert m2.get_effect("stability_recovery_bonus") > 0
        m2.close()


# ---------------------------------------------------------------------------
# Cross-layer integration
# ---------------------------------------------------------------------------


class TestCrossLayerIntegration:
    """Test that all three layers appear in shared memory and schema."""

    def test_all_layers_in_shared_memory_format(self, pathways, exp_filter, marks):
        """Simulate the shm_data construction from stable_creature."""
        ctx = discretize_context(0.1, 0.7, 0.1, "active")
        pathways.reinforce(ctx, "focus_attention", 0.3)

        # Amplify light enough to register as biased (abs > 0.01)
        for _ in range(5):
            exp_filter.update_from_surprise(["light"], 0.5)

        marks.check_and_earn(awakenings=2)

        exp_state = {}
        exp_state["pathways"] = pathways.get_stats()
        exp_state["filter"] = exp_filter.get_stats()
        exp_state["marks"] = marks.get_stats()

        assert exp_state["pathways"]["total_pathways"] >= 1
        assert exp_state["filter"]["biased_count"] >= 1
        assert exp_state["marks"]["total_marks"] >= 1

        # Should be JSON-serializable
        json_str = json.dumps(exp_state)
        assert isinstance(json_str, str)

    def test_schema_hub_includes_experiential_nodes(self):
        """SchemaHub compose_schema should include experiential nodes."""
        from anima_mcp.schema_hub import SchemaHub

        # The imports happen inside _inject_experiential_accumulation via
        # `from .experiential_marks import get_experiential_marks` etc.
        # We patch at the module level where they'll be imported from.
        with patch("anima_mcp.experiential_marks.get_experiential_marks") as mock_marks_fn, \
             patch("anima_mcp.experiential_filter.get_experiential_filter") as mock_filter_fn, \
             patch("anima_mcp.weighted_pathways.get_weighted_pathways") as mock_pw_fn:

            mock_marks_inst = MagicMock()
            mock_marks_inst.get_stats.return_value = {
                "total_marks": 1,
                "mark_names": ["First Return"],
                "categories": ["resilience"],
                "active_effects": {"stability_recovery_bonus": 0.05},
            }
            mock_marks_inst.get_all_earned.return_value = [{
                "mark_id": "resilience_first_return",
                "name": "First Return",
                "description": "Survived restart",
                "category": "resilience",
                "effect_key": "stability_recovery_bonus",
                "effect_value": 0.05,
                "effect_description": "+5% stability recovery",
                "earned_at": "2026-03-17T12:00:00",
                "trigger_context": "awakenings=2",
            }]
            mock_marks_fn.return_value = mock_marks_inst

            mock_filter_inst = MagicMock()
            mock_filter_inst.get_stats.return_value = {
                "dimensions": 7,
                "biased_count": 1,
                "biased_dimensions": {"light": 1.15},
                "mean_salience": 1.02,
            }
            mock_filter_fn.return_value = mock_filter_inst

            mock_pw_inst = MagicMock()
            mock_pw_inst.get_stats.return_value = {
                "total_pathways": 3,
                "unique_contexts": 2,
                "unique_actions": 2,
                "avg_strength": 0.65,
                "total_reinforcements": 10,
            }
            mock_pw_fn.return_value = mock_pw_inst

            hub = SchemaHub()
            schema = hub.compose_schema()

            node_types = {n.node_type for n in schema.nodes}
            node_ids = {n.node_id for n in schema.nodes}

            assert "mark" in node_types, "Should have mark nodes"
            assert "experiential" in node_types, "Should have experiential nodes"
            assert "mark_resilience_first_return" in node_ids
            assert "experiential_filter_bias" in node_ids
            assert "experiential_pathway_density" in node_ids

    def test_eisv_status_text_includes_experiential(self):
        """generate_status_text should include experiential summary."""
        from anima_mcp.anima import Anima
        from anima_mcp.sensors.base import SensorReadings
        from anima_mcp.eisv_mapper import generate_status_text

        readings = SensorReadings(timestamp=datetime.now())
        anima = Anima(warmth=0.5, clarity=0.6, stability=0.7, presence=0.8, readings=readings)
        exp_summary = {
            "marks": {"total_marks": 2},
            "filter": {"biased_count": 1},
            "pathways": {"total_pathways": 5, "avg_strength": 0.72},
        }

        text = generate_status_text(anima, experiential_summary=exp_summary)
        assert "Experience:" in text
        assert "2 marks" in text
        assert "5 pathways" in text


# ---------------------------------------------------------------------------
# Full loop simulation
# ---------------------------------------------------------------------------


class TestFullLoopSimulation:
    """Simulate the stable_creature main loop with all three layers."""

    def test_full_iteration_cycle(self, tmp_db, tmp_filter_path):
        pw = WeightedPathways(db_path=tmp_db)
        ef = ExperientialFilter(persistence_path=tmp_filter_path)
        em = ExperientialMarks(db_path=tmp_db)

        for i in range(20):
            saliences = ef.get_all_saliences()
            assert all(0.5 <= v <= 2.0 for v in saliences.values())

            ctx = discretize_context(
                surprise=0.1 + (i % 5) * 0.1,
                satisfaction=0.5 + (i % 3) * 0.1,
                drive=0.2,
                activity="active",
            )

            outcome = 0.1 if i % 3 == 0 else -0.05
            pw.reinforce(ctx, "focus_attention", outcome)

            if i % 5 == 0:
                ef.update_from_surprise(["light", "cpu_temp"], 0.3)
            ef.tick()

        em.check_and_earn(awakenings=3, observation_count=20)

        assert pw.get_stats()["total_pathways"] > 0
        assert pw.get_stats()["total_reinforcements"] >= 20
        assert em.get_stats()["total_marks"] >= 1

        ef.save()
        ef2 = ExperientialFilter(persistence_path=tmp_filter_path)
        assert ef2.get_stats()["dimensions"] == len(DIMENSIONS)

        pw.close()
        em.close()
