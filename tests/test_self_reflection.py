"""
Tests for self_reflection.py — insight persistence, pattern analysis,
preference/belief/drawing analyzers, and reflect() orchestration.
"""

import json

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from anima_mcp.self_reflection import (
    SelfReflectionSystem, SelfInsight, InsightCategory, StatePattern,
)


@pytest.fixture
def srs(tmp_path):
    """Create SelfReflectionSystem with temp database."""
    system = SelfReflectionSystem(db_path=str(tmp_path / "test_reflect.db"))
    # state_history is created by the server, not SelfReflectionSystem — create it here
    conn = system._connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            timestamp TEXT, warmth REAL, clarity REAL,
            stability REAL, presence REAL, sensors TEXT
        )
    """)
    conn.commit()
    return system


# ==================== Insight Data Class ====================

class TestInsight:
    """Test Insight strength calculation."""

    def test_strength_new_insight(self):
        """New insight (no validations) has moderate strength."""
        i = SelfInsight(
            id="test", category=InsightCategory.TEMPORAL,
            description="test", confidence=0.8, sample_count=10,
            discovered_at=datetime.now(), last_validated=datetime.now(),
            validation_count=0, contradiction_count=0,
        )
        # No validations → strength = confidence * 0.5
        assert i.strength() == pytest.approx(0.4)

    def test_strength_validated(self):
        """Validated insight has higher strength."""
        i = SelfInsight(
            id="test", category=InsightCategory.TEMPORAL,
            description="test", confidence=0.8, sample_count=10,
            discovered_at=datetime.now(), last_validated=datetime.now(),
            validation_count=5, contradiction_count=0,
        )
        # All validations, no contradictions → strength = 0.8 * (5/5) = 0.8
        assert i.strength() == pytest.approx(0.8)

    def test_strength_contradicted(self):
        """Contradicted insight has lower strength."""
        i = SelfInsight(
            id="test", category=InsightCategory.TEMPORAL,
            description="test", confidence=0.8, sample_count=10,
            discovered_at=datetime.now(), last_validated=datetime.now(),
            validation_count=1, contradiction_count=3,
        )
        # 1/(1+3) = 0.25 → 0.8 * 0.25 = 0.2
        assert i.strength() == pytest.approx(0.2)

    def test_to_dict(self):
        i = SelfInsight(
            id="test_id", category=InsightCategory.ENVIRONMENT,
            description="light insight", confidence=0.7, sample_count=50,
            discovered_at=datetime.now(), last_validated=datetime.now(),
        )
        d = i.to_dict()
        assert d["id"] == "test_id"
        assert d["category"] == "environment"
        assert "strength" in d


# ==================== Persistence ====================

class TestInsightPersistence:
    """Test save/load round-trip via SQLite."""

    def test_save_and_reload(self, tmp_path):
        db = str(tmp_path / "persist.db")
        srs1 = SelfReflectionSystem(db_path=db)
        now = datetime.now()
        insight = SelfInsight(
            id="persist_test", category=InsightCategory.TEMPORAL,
            description="warmth peaks at night", confidence=0.85,
            sample_count=100, discovered_at=now, last_validated=now,
            validation_count=3, contradiction_count=0,
        )
        srs1._save_insight(insight)
        srs1.close()

        srs2 = SelfReflectionSystem(db_path=db)
        assert "persist_test" in srs2._insights
        loaded = srs2._insights["persist_test"]
        assert loaded.description == "warmth peaks at night"
        assert loaded.confidence == pytest.approx(0.85)
        assert loaded.validation_count == 3
        srs2.close()

    def test_dedup_by_id(self, srs):
        """Saving same ID twice overwrites, not duplicates."""
        now = datetime.now()
        for i in range(3):
            insight = SelfInsight(
                id="dedup_test", category=InsightCategory.WELLNESS,
                description=f"version {i}", confidence=0.5 + i * 0.1,
                sample_count=10, discovered_at=now, last_validated=now,
            )
            srs._save_insight(insight)
        assert len([k for k in srs._insights if k == "dedup_test"]) == 1
        assert srs._insights["dedup_test"].description == "version 2"

    def test_raw_light_insight_migration_retracts_only_derived_pattern(self, tmp_path):
        db = str(tmp_path / "raw-light.db")
        first = SelfReflectionSystem(db_path=db)
        now = datetime.now()
        derived = SelfInsight(
            id="high_light_higher_clarity",
            category=InsightCategory.ENVIRONMENT,
            description="I feel more clarity in bright light",
            confidence=0.95,
            sample_count=500,
            discovered_at=now,
            last_validated=now,
        )
        qa = SelfInsight(
            id="qa_light_question",
            category=InsightCategory.ENVIRONMENT,
            description="A visitor said light may matter",
            confidence=0.6,
            sample_count=1,
            discovered_at=now,
            last_validated=now,
        )
        first._save_insight(derived)
        first._save_insight(qa)
        first._connect().execute(
            "DELETE FROM reflection_state WHERE key = ?",
            ("migration_external_light_insights_v1",),
        )
        first._connect().commit()
        first.close()

        second = SelfReflectionSystem(db_path=db)

        assert second._insights[derived.id].active is False
        assert second._insights[qa.id].active is True
        audit = second._get_reflection_state(
            "migration_external_light_insights_v1"
        )
        assert derived.id in audit
        second.close()

    def test_inverted_low_sensor_migration_retracts_only_buggy_ids(self, tmp_path):
        db = str(tmp_path / "inverted-sensor.db")
        first = SelfReflectionSystem(db_path=db)
        now = datetime.now()

        inverted = SelfInsight(
            id="low_humidity_lower_stability",
            category=InsightCategory.ENVIRONMENT,
            description="I feel more lower stability when the air is dry",
            confidence=1.0,
            sample_count=287,
            discovered_at=now,
            last_validated=now,
        )
        valid = SelfInsight(
            id="high_humidity_lower_stability",
            category=InsightCategory.ENVIRONMENT,
            description="I feel less stability when it's humid",
            confidence=0.8,
            sample_count=40,
            discovered_at=now,
            last_validated=now,
        )
        first._save_insight(inverted)
        first._save_insight(valid)
        first._connect().execute(
            "DELETE FROM reflection_state WHERE key = ?",
            ("migration_inverted_low_sensor_insights_v1",),
        )
        first._connect().commit()
        first.close()

        second = SelfReflectionSystem(db_path=db)

        assert second._insights[inverted.id].active is False
        assert second._insights[valid.id].active is True
        audit = json.loads(second._get_reflection_state(
            "migration_inverted_low_sensor_insights_v1"
        ))
        assert audit["reason"] == "negative_high_minus_low_branch_was_inverted"
        assert audit["rows"][inverted.id]["corrected_claim_id"] == valid.id
        second.close()


class TestReflectionEpisodePersistence:
    """Test reflection episode persistence and broker drain idempotency."""

    def test_broker_drain_is_idempotent_and_persists_watermark(self, tmp_path):
        db = str(tmp_path / "reflection.db")
        shm_event = {
            "timestamp": "2026-04-05T12:00:00",
            "metacognition": {
                "last_reflection": {
                    "event_id": "broker-metacog:2026-04-05T12:00:00",
                    "timestamp": "2026-04-05T12:00:00",
                    "kind": "metacog",
                    "source": "broker",
                    "trigger": "high_surprise",
                    "topic_tags": ["warmth"],
                    "observation": "Felt warmth differed from expected",
                    "surprise": 0.52,
                    "discrepancy": 0.18,
                    "belief_snapshot": {"warmth_baseline_low": {"value": 0.4, "confidence": 0.6}},
                    "preference_snapshot": {"warmth": {"valence": 0.5, "confidence": 0.5, "influence_weight": 1.0}},
                },
            },
        }

        srs1 = SelfReflectionSystem(db_path=db)
        assert srs1.drain_broker_reflection(shm_event) is True
        assert srs1.drain_broker_reflection(shm_event) is False
        count = srs1._connect().execute("SELECT COUNT(*) AS c FROM reflection_episodes").fetchone()["c"]
        assert count == 1
        srs1.close()

        srs2 = SelfReflectionSystem(db_path=db)
        assert srs2.drain_broker_reflection(shm_event) is False
        count = srs2._connect().execute("SELECT COUNT(*) AS c FROM reflection_episodes").fetchone()["c"]
        assert count == 1
        srs2.close()

    def test_reflect_records_analytic_episode(self, srs):
        with patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(_preferences={}, _drawings_observed=0)), \
             patch("anima_mcp.self_model.get_self_model", return_value=MagicMock(beliefs={})):
            srs.reflect()

        episodes = srs.get_recent_reflection_episodes(limit=5, kind="analytic")
        assert episodes
        assert episodes[0].kind == "analytic"


# ==================== should_reflect ====================

class TestShouldReflect:
    """Test reflection timing gate."""

    def test_true_on_first_call(self, srs):
        assert srs.should_reflect() is True

    def test_false_within_interval(self, srs):
        srs._last_analysis_time = datetime.now()
        assert srs.should_reflect() is False

    def test_true_after_interval(self, srs):
        srs._last_analysis_time = datetime.now() - timedelta(hours=2)
        assert srs.should_reflect() is True


# ==================== analyze_patterns ====================

class TestAnalyzePatterns:
    """Test state history pattern analysis."""

    def test_empty_db_returns_empty(self, srs):
        """No state_history rows → empty patterns."""
        patterns = srs.analyze_patterns(hours=24)
        assert patterns == []

    def test_with_state_history(self, tmp_path):
        """Populated state_history produces patterns."""
        db = str(tmp_path / "patterns.db")
        srs = SelfReflectionSystem(db_path=db)
        conn = srs._connect()
        # Create state_history table and insert 50 rows
        conn.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                timestamp TEXT, warmth REAL, clarity REAL,
                stability REAL, presence REAL, sensors TEXT
            )
        """)
        import json
        base = datetime.now() - timedelta(hours=12)
        for i in range(50):
            ts = base + timedelta(minutes=i * 15)
            # Create varying gated external-light levels to produce a pattern
            light = 50 + i * 10  # increasing
            warmth = 0.3 + (i / 50) * 0.4  # correlates with light
            conn.execute(
                "INSERT INTO state_history VALUES (?, ?, ?, ?, ?, ?)",
                (ts.isoformat(), warmth, 0.5, 0.5, 0.5,
                 json.dumps({
                     "external_light_lux": light,
                     "light_attribution_status": "ready_shadow",
                     "ambient_temp_c": 22,
                 }))
            )
        conn.commit()

        patterns = srs.analyze_patterns(hours=24)
        # Should find at least temporal patterns (50 rows across time)
        assert isinstance(patterns, list)
        srs.close()


# ==================== _analyze_conjunctive_patterns ====================


class TestAnalyzeConjunctivePatterns:
    """Conjunctive patterns expand the space past single-axis saturation.

    Lumen accumulated 176 single-axis insights in two bursts early-on; after
    that, re-detecting them only bumped validation counts. Conjunctive
    patterns (two inputs jointly) open a new tier so new_insight_ids can
    keep populating when the world has structure in pairs, not just axes.
    """

    def _rows_with_quadrant(self, clarity_in_hh: float, n: int = 60):
        """Build synthetic rows where the (high_light, high_temp) quadrant
        has clarity at `clarity_in_hh` and all other quadrants at 0.4.
        Returns sqlite3.Row-like dicts the analyzer accepts.
        """
        import json
        rows = []
        for i in range(n):
            # Cycle through quadrants evenly so median-splits land cleanly.
            quad = i % 4
            if quad == 0:
                light, temp, clarity = 100, 20, 0.4
            elif quad == 1:
                light, temp, clarity = 100, 30, 0.4
            elif quad == 2:
                light, temp, clarity = 500, 20, 0.4
            else:
                light, temp, clarity = 500, 30, clarity_in_hh
            rows.append({
                "warmth": 0.5,
                "clarity": clarity,
                "stability": 0.5,
                "presence": 0.5,
                "sensors": json.dumps({
                    "external_light_lux": light,
                    "ambient_temp_c": temp,
                    "humidity_pct": 40,
                    "interaction_level": 0.0,
                }),
            })
        return rows

    def test_strong_conjunction_emits_pattern(self, srs):
        """When the (high_light, high_temp) quadrant has clarity 0.8 vs overall
        ~0.5, the analyzer should emit a pattern naming both conditions."""
        rows = self._rows_with_quadrant(clarity_in_hh=0.8)
        patterns = srs._analyze_conjunctive_patterns(rows)
        light_temp = [
            p for p in patterns
            if "light" in p.condition and "temperature" in p.condition
        ]
        assert light_temp, f"expected a light+temperature pattern, got {patterns}"
        p = light_temp[0]
        assert "high light" in p.condition and "high temperature" in p.condition
        assert "higher clarity" in p.outcome
        assert p.correlation > 0.15

    def test_no_conjunction_when_within_threshold(self, srs):
        """Quadrant deviation of only 0.05 should not emit."""
        rows = self._rows_with_quadrant(clarity_in_hh=0.45)  # only 0.05 above 0.4
        patterns = srs._analyze_conjunctive_patterns(rows)
        light_temp = [
            p for p in patterns
            if "light" in p.condition and "temperature" in p.condition
        ]
        assert light_temp == []

    def test_empty_when_too_few_samples(self, srs):
        """Fewer than 40 samples cannot form meaningful quadrants."""
        rows = self._rows_with_quadrant(clarity_in_hh=0.8, n=20)
        assert srs._analyze_conjunctive_patterns(rows) == []

    def test_cap_at_three_patterns(self, srs):
        """Even when many pairs show deviations, emit at most 3."""
        import json
        # Build rows where EVERY pair has a big high/high deviation by
        # making clarity jump whenever any two inputs are in their upper half.
        rows = []
        inputs = [(100, 20, 40, 0.0), (500, 30, 70, 0.8)]
        for i in range(80):
            low_high = inputs[i % 2]
            light, t, h, a = low_high
            # All four inputs correlated — all pairs will show high/high
            # quadrant with elevated clarity.
            clarity = 0.8 if i % 2 == 1 else 0.4
            rows.append({
                "warmth": 0.5,
                "clarity": clarity,
                "stability": 0.5,
                "presence": 0.5,
                "sensors": json.dumps({
                "external_light_lux": light,
                    "ambient_temp_c": t,
                    "humidity_pct": h,
                    "interaction_level": a,
                }),
            })
        patterns = srs._analyze_conjunctive_patterns(rows)
        assert len(patterns) <= 3

    def test_tolerates_missing_sensor_keys(self, srs):
        """Rows with partial sensor data shouldn't crash the analyzer."""
        import json
        rows = []
        for i in range(60):
            sensors = {
                "external_light_lux": 100 + i * 5
            }  # only one input available
            rows.append({
                "warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5,
                "sensors": json.dumps(sensors),
            })
        # Should not raise; most pairs won't have both values → empty result ok
        patterns = srs._analyze_conjunctive_patterns(rows)
        assert isinstance(patterns, list)

    def test_insight_id_distinct_from_single_axis(self, srs):
        """Conjunctive pattern's derived insight_id must not collide with a
        single-axis finding — `generate_insights` builds insight_id from
        `condition_outcome`, and the `_and_` joiner keeps them distinct."""
        rows = self._rows_with_quadrant(clarity_in_hh=0.8)
        patterns = srs._analyze_conjunctive_patterns(rows)
        if not patterns:
            pytest.skip("no conjunctive pattern emitted — covered by other test")
        for p in patterns:
            # The id format used by generate_insights:
            derived_id = f"{p.condition}_{p.outcome}".replace(" ", "_").lower()
            assert "_and_" in derived_id, f"conjunctive id should contain '_and_', got {derived_id}"


# ==================== generate_insights ====================

class TestGenerateInsights:
    """Test converting StatePatterns into Insights."""

    def test_temporal_pattern_creates_temporal_insight(self, srs):
        pattern = StatePattern(
            condition="the morning", outcome="highest clarity",
            correlation=0.25, sample_count=30,
            avg_warmth=0.5, avg_clarity=0.7, avg_stability=0.5, avg_presence=0.5,
        )
        insights = srs.generate_insights([pattern])
        assert len(insights) == 1
        assert insights[0].category == InsightCategory.TEMPORAL
        assert "morning" in insights[0].description.lower()

    def test_environment_pattern(self, srs):
        pattern = StatePattern(
            condition="low light", outcome="higher stability",
            correlation=0.3, sample_count=80,
            avg_warmth=0.5, avg_clarity=0.5, avg_stability=0.7, avg_presence=0.5,
        )
        insights = srs.generate_insights([pattern])
        assert len(insights) == 1
        assert insights[0].category == InsightCategory.ENVIRONMENT

    def test_causal_pattern(self, srs):
        pattern = StatePattern(
            condition="warmth rises", outcome="presence falls",
            correlation=-0.15, sample_count=20,
            avg_warmth=0.0, avg_clarity=0.0, avg_stability=0.0, avg_presence=0.0,
        )
        insights = srs.generate_insights([pattern])
        assert len(insights) == 1
        assert insights[0].category == InsightCategory.WELLNESS

    def test_existing_insight_validated(self, srs):
        """Re-encountering a pattern validates existing insight, doesn't duplicate."""
        pattern = StatePattern(
            condition="the night", outcome="highest warmth",
            correlation=0.3, sample_count=50,
            avg_warmth=0.7, avg_clarity=0.5, avg_stability=0.5, avg_presence=0.5,
        )
        srs.generate_insights([pattern])
        insight_id = "the_night_highest_warmth"
        # Same analysis window: the same hours, so no second count.
        srs.generate_insights([pattern])
        assert srs._insights[insight_id].validation_count == 1
        # A full window later the data is independent and counts.
        srs._insights[insight_id].last_judged -= timedelta(hours=25)
        srs.generate_insights([pattern])
        assert srs._insights[insight_id].validation_count == 2


# ==================== Preference Insights ====================

class TestPreferenceInsights:
    """Test _analyze_preference_insights."""

    def _mock_growth(self):
        """Create mock growth system with known preferences."""
        from anima_mcp.growth import GrowthPreference, PreferenceCategory
        mock = MagicMock()
        mock._preferences = {
            "night_calm": GrowthPreference(
                category=PreferenceCategory.TEMPORAL, name="night_calm",
                description="The quiet of night calms me",
                value=0.9, confidence=0.9, observation_count=100,
                first_noticed=datetime.now(), last_confirmed=datetime.now(),
            ),
            "low_conf": GrowthPreference(
                category=PreferenceCategory.ENVIRONMENT, name="low_conf",
                description="I like quiet", value=0.5, confidence=0.3,
                observation_count=2,
                first_noticed=datetime.now(), last_confirmed=datetime.now(),
            ),
        }
        return mock

    def test_high_confidence_preference_creates_insight(self, srs):
        mock_growth = self._mock_growth()
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            insights = srs._analyze_preference_insights()
        # night_calm has confidence=0.9, obs=100 → above thresholds (0.8, 10)
        assert len(insights) >= 1
        descs = [i.description for i in insights]
        assert any("night" in d.lower() for d in descs)

    def test_low_confidence_skipped(self, srs):
        mock_growth = self._mock_growth()
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            insights = srs._analyze_preference_insights()
        # low_conf has confidence=0.3 → below threshold
        ids = [i.id for i in insights]
        assert "pref_low_conf" not in ids

    def test_existing_insight_is_not_revalidated_from_same_samples(self, srs):
        mock_growth = self._mock_growth()
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            srs._analyze_preference_insights()
            validation_count = srs._insights["pref_night_calm"].validation_count
            insights2 = srs._analyze_preference_insights()
        # Second call should validate, not create new
        assert len(insights2) == 0
        assert srs._insights["pref_night_calm"].validation_count == validation_count

    def test_source_retraction_deactivates_existing_insight(self, srs):
        mock_growth = self._mock_growth()
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            srs._analyze_preference_insights()
            insight = srs._insights["pref_night_calm"]
            assert insight.active is True

            mock_growth._preferences["night_calm"].confidence = 0.5
            srs._analyze_preference_insights()
            contradiction_count = insight.contradiction_count
            srs._analyze_preference_insights()

        assert insight.active is False
        assert insight.strength() == 0.0
        assert insight.contradiction_count == contradiction_count
        assert insight not in srs.get_insights()
        assert insight in srs.get_insights(include_inactive=True)
        row = srs._connect().execute(
            "SELECT active FROM insights WHERE id = 'pref_night_calm'"
        ).fetchone()
        assert row[0] == 0

    def test_retracted_insight_reactivates_only_after_source_reearns_gate(self, srs):
        mock_growth = self._mock_growth()
        pref = mock_growth._preferences["night_calm"]
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            srs._analyze_preference_insights()
            pref.confidence = 0.5
            srs._analyze_preference_insights()
            pref.confidence = 0.75
            srs._analyze_preference_insights()
            assert srs._insights["pref_night_calm"].active is False
            pref.confidence = 0.8
            srs._analyze_preference_insights()

        assert srs._insights["pref_night_calm"].active is True


# ==================== Belief Insights ====================

class TestBeliefInsights:
    """Test _analyze_belief_insights."""

    def _mock_self_model(self, beliefs):
        mock = MagicMock()
        mock.beliefs = beliefs
        return mock

    def test_well_tested_belief_creates_insight(self, srs):
        belief = MagicMock()
        belief.supporting_count = 15
        belief.contradicting_count = 2
        belief.confidence = 0.8
        belief.description = "light affects my clarity"
        belief.get_belief_strength.return_value = "fairly confident"

        mock_sm = self._mock_self_model({"b1": belief})
        with patch("anima_mcp.self_model.get_self_model", return_value=mock_sm):
            insights = srs._analyze_belief_insights()
        assert len(insights) == 1
        assert "light" in insights[0].description.lower()

    def test_low_evidence_skipped(self, srs):
        belief = MagicMock()
        belief.supporting_count = 3
        belief.contradicting_count = 1
        belief.confidence = 0.8
        belief.description = "not enough data"

        mock_sm = self._mock_self_model({"b1": belief})
        with patch("anima_mcp.self_model.get_self_model", return_value=mock_sm):
            insights = srs._analyze_belief_insights()
        assert len(insights) == 0


# ==================== Drawing Insights ====================

class TestDrawingInsights:
    """Test _analyze_drawing_insights."""

    def test_insufficient_drawings_returns_empty(self, srs):
        mock_growth = MagicMock()
        mock_growth._drawings_observed = 2
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            insights = srs._analyze_drawing_insights()
        assert insights == []

    def test_drawing_wellbeing_insight(self, srs):
        from anima_mcp.growth import GrowthPreference, PreferenceCategory
        mock_growth = MagicMock()
        mock_growth._drawings_observed = 10
        mock_growth._preferences = {
            "drawing_wellbeing": GrowthPreference(
                category=PreferenceCategory.ACTIVITY, name="drawing_wellbeing",
                description="I feel good when I draw",
                value=0.8, confidence=0.7, observation_count=8,
                first_noticed=datetime.now(), last_confirmed=datetime.now(),
            ),
        }
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth):
            insights = srs._analyze_drawing_insights()
        assert len(insights) >= 1
        assert any("draw" in i.description.lower() for i in insights)


# ==================== reflect() ====================

class TestReflect:
    """Test the main reflect() orchestrator."""

    def test_returns_string_or_none(self, srs):
        """reflect() returns Optional[str]."""
        result = srs.reflect()
        assert result is None or isinstance(result, str)

    def test_sets_last_analysis_time(self, srs):
        assert srs._last_analysis_time is None
        srs.reflect()
        assert srs._last_analysis_time is not None

    def test_returns_strongest_new_insight(self, srs):
        """If new insights discovered, returns description of strongest."""
        # Inject a high-confidence preference that will trigger insight
        from anima_mcp.growth import GrowthPreference, PreferenceCategory
        mock_growth = MagicMock()
        mock_growth._preferences = {
            "night_calm": GrowthPreference(
                category=PreferenceCategory.TEMPORAL, name="night_calm",
                description="The quiet of night calms me",
                value=0.9, confidence=0.95, observation_count=200,
                first_noticed=datetime.now(), last_confirmed=datetime.now(),
            ),
        }
        mock_growth._drawings_observed = 0
        mock_sm = MagicMock()
        mock_sm.beliefs = {}
        with patch("anima_mcp.growth.get_growth_system", return_value=mock_growth), \
             patch("anima_mcp.self_model.get_self_model", return_value=mock_sm):
            result = srs.reflect()
        # Should mention the new insight
        if result:
            assert "noticed" in result.lower() or "night" in result.lower() or "know" in result.lower()


class TestReflectionDynamics:
    """Test same-kind rumination/update detection and summary behavior."""

    @staticmethod
    def _record_episode(srs, *, kind, timestamp, topics, surprise, discrepancy, belief_value, pref_value):
        srs.record_episode(
            kind=kind,
            source="test",
            trigger="unit",
            event_timestamp=timestamp,
            topic_tags=topics,
            observation="test reflection",
            surprise=surprise,
            discrepancy=discrepancy,
            belief_snapshot={"warmth_baseline_low": {"value": belief_value, "confidence": 0.6}},
            preference_snapshot={"warmth": {"valence": pref_value, "confidence": 0.5, "influence_weight": 1.0}},
        )

    def test_same_kind_rumination_detected(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=1), topics=["warmth"], surprise=0.58, discrepancy=0.24, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=2), topics=["warmth"], surprise=0.56, discrepancy=0.22, belief_value=0.4, pref_value=0.4)

        new_insights = srs._analyze_reflection_episode_insights()
        assert any("without updating" in insight.description.lower() for insight in new_insights)

        summary = srs.get_reflection_summary()
        assert summary["dominant_focus"]["tag"] == "warmth"
        assert summary["rumination"]["count"] == 2

    def test_cross_kind_overlap_does_not_count_as_rumination(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="analytic", timestamp=base + timedelta(seconds=1), topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.4, pref_value=0.4)

        new_insights = srs._analyze_reflection_episode_insights()
        assert new_insights == []

    def test_rumination_window_ignores_old_same_kind_episode(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.4, pref_value=0.4)
        for idx in range(10):
            self._record_episode(
                srs,
                kind="metacog",
                timestamp=base + timedelta(seconds=idx + 1),
                topics=[f"other_{idx}"],
                surprise=0.3,
                discrepancy=0.1,
                belief_value=0.4,
                pref_value=0.4,
            )
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=20), topics=["warmth"], surprise=0.56, discrepancy=0.21, belief_value=0.4, pref_value=0.4)

        new_insights = srs._analyze_reflection_episode_insights()
        assert new_insights == []

    def test_similarity_epsilon_blocks_false_rumination(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.2, discrepancy=0.1, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=1), topics=["warmth"], surprise=0.35, discrepancy=0.25, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=2), topics=["warmth"], surprise=0.5, discrepancy=0.4, belief_value=0.4, pref_value=0.4)

        new_insights = srs._analyze_reflection_episode_insights()
        assert new_insights == []

    def test_productive_update_detected(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.3, pref_value=0.3)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=1), topics=["warmth"], surprise=0.54, discrepancy=0.21, belief_value=0.45, pref_value=0.45)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=2), topics=["warmth"], surprise=0.53, discrepancy=0.19, belief_value=0.6, pref_value=0.6)

        new_insights = srs._analyze_reflection_episode_insights()
        assert any("change what i know" in insight.description.lower() for insight in new_insights)

    @staticmethod
    def _record_analytic(srs, *, timestamp, topics, belief_value, pref_value):
        """Analytic episodes have no surprise/discrepancy — they're interval-driven summaries."""
        srs.record_episode(
            kind="analytic",
            source="self_reflection",
            trigger="interval",
            event_timestamp=timestamp,
            topic_tags=topics,
            observation="periodic reflection",
            belief_snapshot={"warmth_baseline_low": {"value": belief_value, "confidence": 0.6}},
            preference_snapshot={"warmth": {"valence": pref_value, "confidence": 0.5, "influence_weight": 1.0}},
        )

    def test_analytic_overlap_does_not_trigger_rumination(self, srs):
        """Regression: analytic episodes with overlapping tags and stable beliefs must NOT be
        flagged as rumination.

        Analytic reflection is interval-driven pattern summarization. In a healthy stable
        system the same patterns recur every cycle with unchanged beliefs — that's normal,
        not rumination. Before this fix, `_intensity_is_similar` returned True vacuously
        for analytic (no surprise/discrepancy data) and the detector labeled every stable
        analytic cycle as rumination. Rumination classification is now scoped to metacog.
        """
        base = datetime(2026, 4, 5, 12, 0, 0)
        # Three analytic episodes, same topic, stable beliefs — the exact steady-state case
        self._record_analytic(srs, timestamp=base, topics=["warmth"], belief_value=0.4, pref_value=0.4)
        self._record_analytic(srs, timestamp=base + timedelta(seconds=1), topics=["warmth"], belief_value=0.4, pref_value=0.4)
        self._record_analytic(srs, timestamp=base + timedelta(seconds=2), topics=["warmth"], belief_value=0.4, pref_value=0.4)

        new_insights = srs._analyze_reflection_episode_insights()
        assert all(
            "without updating" not in insight.description.lower()
            and "keep reflecting" not in insight.description.lower()
            for insight in new_insights
        ), f"analytic episodes were incorrectly flagged as rumination: {[i.description for i in new_insights]}"

        summary = srs.get_reflection_summary()
        # Overlap is still counted as "repeated" so the summary stays honest about recurrence,
        # but it must not roll up into rumination.
        assert summary["rumination"]["count"] == 0
        assert summary["learning_yield"]["repeated"] == 2  # 2 repeat pairs across 3 episodes

    def test_productive_dominance_suppresses_rumination_insight(self, srs):
        """When a topic bucket has both productive and rumination pairs in the window,
        the dominant signal wins. Productive-dominant → only the learning insight fires,
        the rumination insight is suppressed.
        """
        base = datetime(2026, 4, 5, 12, 0, 0)
        # First pair: stable beliefs (ruminative)
        self._record_episode(srs, kind="metacog", timestamp=base, topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.4, pref_value=0.4)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=1), topics=["warmth"], surprise=0.56, discrepancy=0.21, belief_value=0.4, pref_value=0.4)
        # Next three pairs: belief/pref movement (productive)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=2), topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.55, pref_value=0.55)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=3), topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.7, pref_value=0.7)
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=4), topics=["warmth"], surprise=0.55, discrepancy=0.2, belief_value=0.85, pref_value=0.85)

        new_insights = srs._analyze_reflection_episode_insights()

        # Productive count (3) > rumination count (1), so only the learning insight should fire
        rumination_insights = [i for i in new_insights if "without updating" in i.description.lower()]
        productive_insights = [i for i in new_insights if "change what i know" in i.description.lower()]
        assert len(productive_insights) == 1
        assert len(rumination_insights) == 0, (
            f"productive-dominant bucket should suppress rumination insight; "
            f"got: {[i.description for i in new_insights]}"
        )

    def test_rumination_and_server_broker_share_bucket(self, srs):
        """Server-origin and broker-origin metacog episodes on the same topic must
        bucket together (same kind) and contribute to the same rumination count.

        This is the regression guard for the server-side recording hook at
        server.py:392. Before that hook existed, server reflections never made it
        into the table; now they do, and the detector must treat them as same-kind
        peers of broker events.
        """
        base = datetime(2026, 4, 5, 12, 0, 0)
        # Mixed source, same kind, same topic, stable beliefs
        srs.record_episode(
            kind="metacog", source="broker", trigger="high_surprise",
            event_timestamp=base,
            topic_tags=["warmth"],
            observation="broker reflection",
            surprise=0.55, discrepancy=0.2,
            belief_snapshot={"warmth_baseline_low": {"value": 0.4, "confidence": 0.6}},
            preference_snapshot={"warmth": {"valence": 0.4, "confidence": 0.5, "influence_weight": 1.0}},
            event_id="broker-metacog:1",
        )
        srs.record_episode(
            kind="metacog", source="server", trigger="high_surprise",
            event_timestamp=base + timedelta(seconds=1),
            topic_tags=["warmth"],
            observation="server reflection",
            surprise=0.56, discrepancy=0.21,
            belief_snapshot={"warmth_baseline_low": {"value": 0.4, "confidence": 0.6}},
            preference_snapshot={"warmth": {"valence": 0.4, "confidence": 0.5, "influence_weight": 1.0}},
            event_id="server-metacog:1",
        )
        srs.record_episode(
            kind="metacog", source="broker", trigger="high_surprise",
            event_timestamp=base + timedelta(seconds=2),
            topic_tags=["warmth"],
            observation="broker reflection",
            surprise=0.55, discrepancy=0.2,
            belief_snapshot={"warmth_baseline_low": {"value": 0.4, "confidence": 0.6}},
            preference_snapshot={"warmth": {"valence": 0.4, "confidence": 0.5, "influence_weight": 1.0}},
            event_id="broker-metacog:2",
        )

        summary = srs.get_reflection_summary()
        # Three episodes, same kind (metacog), same topic, stable beliefs → 2 ruminative pairs
        assert summary["learning_yield"]["repeated"] == 2
        assert summary["rumination"]["count"] == 2
        assert summary["dominant_focus"]["tag"] == "warmth"
        assert summary["dominant_focus"]["kind"] == "metacog"

        new_insights = srs._analyze_reflection_episode_insights()
        assert any("without updating" in i.description.lower() for i in new_insights), (
            "mixed broker/server metacog episodes on same topic should fire rumination"
        )


# ==================== get_insights ====================

class TestGetInsights:
    """Test insight retrieval and filtering."""

    def test_empty_returns_empty(self, srs):
        assert srs.get_insights() == []

    def test_filter_by_category(self, srs):
        now = datetime.now()
        srs._save_insight(SelfInsight(
            id="t1", category=InsightCategory.TEMPORAL,
            description="time", confidence=0.8, sample_count=10,
            discovered_at=now, last_validated=now,
        ))
        srs._save_insight(SelfInsight(
            id="e1", category=InsightCategory.ENVIRONMENT,
            description="env", confidence=0.6, sample_count=10,
            discovered_at=now, last_validated=now,
        ))
        temporal = srs.get_insights(category=InsightCategory.TEMPORAL)
        assert len(temporal) == 1
        assert temporal[0].id == "t1"

    def test_sorted_by_strength(self, srs):
        now = datetime.now()
        srs._save_insight(SelfInsight(
            id="weak", category=InsightCategory.TEMPORAL,
            description="weak", confidence=0.3, sample_count=5,
            discovered_at=now, last_validated=now, validation_count=1,
        ))
        srs._save_insight(SelfInsight(
            id="strong", category=InsightCategory.TEMPORAL,
            description="strong", confidence=0.9, sample_count=50,
            discovered_at=now, last_validated=now, validation_count=10,
        ))
        insights = srs.get_insights()
        assert insights[0].id == "strong"


# ==================== Contradiction Detection ====================

class TestContradictionDetection:
    """Test _find_contradicting_insights and _extract_condition_from_id."""

    def test_extract_condition_highest(self):
        assert SelfReflectionSystem._extract_condition_from_id("the_night_highest_warmth") == "the_night"

    def test_extract_condition_lowest(self):
        assert SelfReflectionSystem._extract_condition_from_id("the_morning_lowest_presence") == "the_morning"

    def test_extract_condition_higher(self):
        assert SelfReflectionSystem._extract_condition_from_id("low_light_higher_stability") == "low_light"

    def test_extract_condition_lower(self):
        assert SelfReflectionSystem._extract_condition_from_id("high_temp_lower_clarity") == "high_temp"

    def test_extract_condition_causal_rises(self):
        assert SelfReflectionSystem._extract_condition_from_id("warmth_rises_presence_falls") == "warmth_rises"

    def test_extract_condition_causal_falls(self):
        assert SelfReflectionSystem._extract_condition_from_id("clarity_falls_stability_rises") == "clarity_falls"

    def test_extract_condition_unknown_format(self):
        assert SelfReflectionSystem._extract_condition_from_id("something_completely_different") == ""

    def test_temporal_contradictions_detected(self, srs):
        """'warmth best at night' and 'warmth best in afternoon' should contradict."""
        # Seed the 'night' insight first
        night_pattern = StatePattern(
            condition="the night", outcome="highest warmth",
            correlation=0.3, sample_count=200,
            avg_warmth=0.7, avg_clarity=0.5, avg_stability=0.5, avg_presence=0.5,
        )
        srs.generate_insights([night_pattern])
        night_insight = srs._insights["the_night_highest_warmth"]
        assert night_insight.confidence == 1.0  # High sample count → max confidence

        # Now generate the contradicting 'afternoon' insight
        afternoon_pattern = StatePattern(
            condition="the afternoon", outcome="highest warmth",
            correlation=0.25, sample_count=150,
            avg_warmth=0.6, avg_clarity=0.5, avg_stability=0.5, avg_presence=0.5,
        )
        srs.generate_insights([afternoon_pattern])

        # The afternoon insight should have been penalized
        afternoon = srs._insights["the_afternoon_highest_warmth"]
        assert afternoon.contradiction_count == 1
        assert afternoon.confidence < 1.0  # Penalized by 50%

        # The existing night insight should also have been penalized
        night_after = srs._insights["the_night_highest_warmth"]
        assert night_after.contradiction_count == 1
        assert night_after.confidence < 1.0  # Penalized by 30%

    def test_environment_contradictions_still_work(self, srs):
        """Verify _higher_/_lower_ environment contradictions still detected."""
        low_light_pattern = StatePattern(
            condition="low light", outcome="higher stability",
            correlation=0.3, sample_count=100,
            avg_warmth=0.5, avg_clarity=0.5, avg_stability=0.7, avg_presence=0.5,
        )
        srs.generate_insights([low_light_pattern])

        high_light_pattern = StatePattern(
            condition="high light", outcome="higher stability",
            correlation=0.2, sample_count=80,
            avg_warmth=0.5, avg_clarity=0.5, avg_stability=0.6, avg_presence=0.5,
        )
        srs.generate_insights([high_light_pattern])

        # Both should exist with contradictions noted
        assert "low_light_higher_stability" in srs._insights
        assert "high_light_higher_stability" in srs._insights
        low = srs._insights["low_light_higher_stability"]
        high = srs._insights["high_light_higher_stability"]
        assert low.contradiction_count == 1
        assert high.contradiction_count == 1

    def test_causal_contradictions_detected(self, srs):
        """'warmth rises → presence falls' vs 'warmth rises → presence rises' should contradict."""
        pattern1 = StatePattern(
            condition="warmth rises", outcome="presence falls",
            correlation=-0.15, sample_count=100,
            avg_warmth=0.0, avg_clarity=0.0, avg_stability=0.0, avg_presence=0.0,
        )
        srs.generate_insights([pattern1])
        assert "warmth_rises_presence_falls" in srs._insights

        pattern2 = StatePattern(
            condition="warmth rises", outcome="presence rises",
            correlation=0.15, sample_count=80,
            avg_warmth=0.0, avg_clarity=0.0, avg_stability=0.0, avg_presence=0.0,
        )
        srs.generate_insights([pattern2])

        falls = srs._insights["warmth_rises_presence_falls"]
        rises = srs._insights["warmth_rises_presence_rises"]
        assert falls.contradiction_count == 1
        assert rises.contradiction_count == 1


# ==================== Evidence counts once and can lose (2026-10-15) ====================

class TestMetaInsightsJudgeOnlyNewPairs:
    _record_episode = staticmethod(TestReflectionDynamics._record_episode)

    def _ruminate(self, srs, base):
        for i in range(3):
            self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=i),
                                 topics=["light"], surprise=0.55, discrepancy=0.2,
                                 belief_value=0.4, pref_value=0.4)

    def test_rereading_the_same_episodes_does_not_validate(self, srs):
        """The live failure: 'validated 86 times' from re-reading 50 episodes hourly."""
        self._ruminate(srs, datetime(2026, 4, 5, 12, 0, 0))
        [created] = srs._analyze_reflection_episode_insights()
        for _ in range(5):
            srs._analyze_reflection_episode_insights()
        stored = srs._insights[created.id]
        assert (stored.validation_count, stored.contradiction_count) == (1, 0)

    def test_new_rumination_validates_once(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._ruminate(srs, base)
        [created] = srs._analyze_reflection_episode_insights()
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=3),
                             topics=["light"], surprise=0.56, discrepancy=0.21,
                             belief_value=0.4, pref_value=0.4)
        srs._analyze_reflection_episode_insights()
        srs._analyze_reflection_episode_insights()
        assert srs._insights[created.id].validation_count == 2

    def test_a_productive_pair_contradicts_rumination(self, srs):
        base = datetime(2026, 4, 5, 12, 0, 0)
        self._ruminate(srs, base)
        [created] = srs._analyze_reflection_episode_insights()
        self._record_episode(srs, kind="metacog", timestamp=base + timedelta(seconds=3),
                             topics=["light"], surprise=0.4, discrepancy=0.2,
                             belief_value=0.7, pref_value=0.7)
        srs._analyze_reflection_episode_insights()
        stored = srs._insights[created.id]
        assert (stored.validation_count, stored.contradiction_count) == (1, 1)


class TestPatternInsightsCanLose:
    def _store(self, srs, claim_id="the_night_highest_warmth"):
        pattern = StatePattern(
            condition="the night", outcome="highest warmth", correlation=0.3,
            sample_count=50, avg_warmth=0.7, avg_clarity=0.5,
            avg_stability=0.5, avg_presence=0.5,
        )
        srs.generate_insights([pattern])
        insight = srs._insights[claim_id]
        insight.last_judged -= timedelta(hours=25)
        return insight

    def test_measured_but_faded_pattern_is_contradicted(self, srs):
        insight = self._store(srs)
        srs._pattern_measurements = {insight.id: (0.02, 0.1)}
        srs.generate_insights([])
        assert (insight.validation_count, insight.contradiction_count) == (1, 1)

    def test_measured_claim_that_still_holds_validates_without_being_strongest(self, srs):
        insight = self._store(srs)
        srs._pattern_measurements = {insight.id: (0.15, 0.1)}
        srs.generate_insights([])
        assert (insight.validation_count, insight.contradiction_count) == (2, 0)

    def test_unmeasured_pattern_is_unknown(self, srs):
        insight = self._store(srs)
        srs._pattern_measurements = {}
        srs.generate_insights([])
        assert (insight.validation_count, insight.contradiction_count) == (1, 0)

    def test_judged_at_most_once_per_window(self, srs):
        insight = self._store(srs)
        srs._pattern_measurements = {insight.id: (0.02, 0.1)}
        for _ in range(5):
            srs.generate_insights([])
        assert insight.contradiction_count == 1

    def test_sensor_analyzer_measures_every_dimension(self, srs):
        rows = [
            {"timestamp": "2026-10-15T00:00:00", "warmth": 0.3 + 0.03 * i,
             "clarity": 0.2 + 0.05 * i, "stability": 0.5, "presence": 0.5,
             "sensors": json.dumps({"ambient_temp_c": 18 + i})}
            for i in range(12)
        ]
        srs._pattern_measurements = {}
        pattern = srs._analyze_sensor_correlation(rows, "ambient_temp_c", "Temperature")
        assert pattern.outcome == "higher clarity"
        effect, cut = srs._pattern_measurements["high_temperature_higher_warmth"]
        assert effect > cut  # warmth holds too, though clarity was strongest
        effect, _ = srs._pattern_measurements["high_temperature_lower_warmth"]
        assert effect < 0


class TestDrawingInsightsCountNewEvidence:
    def _growth(self, *, evidence, confidence=0.9, value=0.8):
        from anima_mcp.growth import GrowthPreference, PreferenceCategory
        growth = MagicMock()
        growth._drawings_observed = 10
        pref = {}
        for name in ("drawing_night", "drawing_wellbeing"):
            p = GrowthPreference(
                category=PreferenceCategory.ACTIVITY, name=name, description=name,
                value=value, confidence=confidence, observation_count=evidence,
                first_noticed=datetime.now(), last_confirmed=datetime.now(),
            )
            p.evidence_count = evidence
            pref[name] = p
        growth._preferences = pref
        return growth

    def _run(self, srs, growth):
        with patch("anima_mcp.growth.get_growth_system", return_value=growth):
            srs._analyze_drawing_insights()

    def test_unchanged_preference_does_not_revalidate(self, srs):
        growth = self._growth(evidence=8)
        for _ in range(4):
            self._run(srs, growth)
        assert srs._insights["drawing_night"].validation_count == 1
        growth._preferences["drawing_night"].evidence_count = 9
        self._run(srs, growth)
        assert srs._insights["drawing_night"].validation_count == 2

    def test_wellness_retires_and_flips(self, srs):
        self._run(srs, self._growth(evidence=8))
        wellness = srs._insights["drawing_wellness"]
        assert wellness.description == "drawing seems to help me feel better"
        self._run(srs, self._growth(evidence=8, confidence=0.3))
        assert wellness.active is False
        self._run(srs, self._growth(evidence=9, value=0.2))
        assert wellness.active is True
        assert wellness.contradiction_count == 1
        assert wellness.description == "my drawings don't always reflect how i feel"


def test_recount_collapse_migration(tmp_path):
    db = str(tmp_path / "recount.db")
    first = SelfReflectionSystem(db_path=db)
    conn = first._connect()
    conn.execute("DELETE FROM reflection_state WHERE key = 'migration_insight_recount_collapse_v1'")
    now = datetime.now()
    for iid, v, c in [("drawing_night", 959, 0), ("clarity_rises_stability_rises", 1, 108),
                      ("reflect_rumination_metacog_light", 92, 0), ("pref_warm_temp", 1351, 0)]:
        first._save_insight(SelfInsight(
            id=iid, category=InsightCategory.BEHAVIORAL, description=iid,
            confidence=0.9, sample_count=1, discovered_at=now, last_validated=now,
            validation_count=v, contradiction_count=c,
        ))
    conn.commit()

    second = SelfReflectionSystem(db_path=db)
    counts = {i.id: (i.validation_count, i.contradiction_count)
              for i in second._insights.values()}
    assert counts["drawing_night"] == (1, 0)
    assert counts["clarity_rises_stability_rises"] == (1, 1)
    assert counts["reflect_rumination_metacog_light"] == (1, 0)
    assert counts["pref_warm_temp"] == (1351, 0)  # mirrors a source that can fall
    audit = json.loads(second._get_reflection_state("migration_insight_recount_collapse_v1"))
    assert audit["rows"]["drawing_night"]["validation_count"] == 959
