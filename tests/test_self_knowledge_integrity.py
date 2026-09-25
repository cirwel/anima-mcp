"""
Tests for the 2026-08-21 self-knowledge integrity pass.

Covers the false-knowledge generators found by the self-model audit:
rowid-stable insight saves, deliberate tie-break ordering, belief-insight
retraction and fresh-evidence validation, the zero-variance sensor guard,
sign-aware Q&A verification, trend staleness (fail toward unknown), per-pass
qa confidence re-sync, the knowledge v3 external-confidence rescale, and the
self-model dead-channel cold start.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.self_reflection import (
    SelfReflectionSystem, SelfInsight, InsightCategory,
)


@pytest.fixture
def srs(tmp_path):
    system = SelfReflectionSystem(db_path=str(tmp_path / "test_reflect.db"))
    conn = system._connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            timestamp TEXT, warmth REAL, clarity REAL,
            stability REAL, presence REAL, sensors TEXT
        )
    """)
    conn.commit()
    return system


def _mk_insight(iid, confidence=1.0, validation_count=1, category=InsightCategory.WELLNESS,
                last_validated=None, active=True, sample_count=10):
    return SelfInsight(
        id=iid, category=category, description=f"desc {iid}",
        confidence=confidence, sample_count=sample_count,
        discovered_at=datetime.now(),
        last_validated=last_validated or datetime.now(),
        validation_count=validation_count, contradiction_count=0,
        active=active,
    )


class TestRowidStability:
    def test_save_preserves_rowid(self, srs):
        ins = _mk_insight("pref_test")
        srs._save_insight(ins)
        conn = srs._connect()
        rowid_before = conn.execute(
            "SELECT rowid FROM insights WHERE id='pref_test'").fetchone()[0]
        ins.validation_count += 5
        srs._save_insight(ins)
        rowid_after = conn.execute(
            "SELECT rowid FROM insights WHERE id='pref_test'").fetchone()[0]
        assert rowid_after == rowid_before
        assert conn.execute(
            "SELECT validation_count FROM insights WHERE id='pref_test'"
        ).fetchone()[0] == ins.validation_count


class TestTieBreakOrdering:
    def test_self_derived_beats_external_on_tie(self, srs):
        # Same strength; qa_ (external) loaded/created FIRST so a naive
        # stable sort would surface it first.
        srs._save_insight(_mk_insight("qa_external"))
        srs._save_insight(_mk_insight("pref_derived"))
        ordered = srs.get_insights()
        ids = [i.id for i in ordered]
        assert ids.index("pref_derived") < ids.index("qa_external")

    def test_recency_breaks_ties_within_source(self, srs):
        old = _mk_insight("pref_old", last_validated=datetime.now() - timedelta(days=30))
        new = _mk_insight("pref_new", last_validated=datetime.now())
        srs._save_insight(old)
        srs._save_insight(new)
        ids = [i.id for i in srs.get_insights()]
        assert ids.index("pref_new") < ids.index("pref_old")

    def test_strength_still_dominates(self, srs):
        srs._save_insight(_mk_insight("qa_strong", confidence=1.0))
        srs._save_insight(_mk_insight("pref_weak", confidence=0.4))
        ids = [i.id for i in srs.get_insights()]
        assert ids.index("qa_strong") < ids.index("pref_weak")


class TestBeliefRetraction:
    def _mock_sm(self, confidence, supporting=20, contradicting=200):
        belief = MagicMock()
        belief.confidence = confidence
        belief.supporting_count = supporting
        belief.contradicting_count = contradicting
        belief.description = "My baseline warmth tends to stay low"
        belief.get_belief_strength.return_value = "doubtful"
        sm = MagicMock()
        sm.beliefs = {"warmth_baseline_low": belief}
        return sm

    def test_below_threshold_belief_retracts_its_insight(self, srs):
        srs._save_insight(_mk_insight("belief_warmth_baseline_low", confidence=1.0))
        with patch("anima_mcp.self_model.get_self_model",
                   return_value=self._mock_sm(confidence=0.0001)):
            srs._analyze_belief_insights()
        stored = srs._insights["belief_warmth_baseline_low"]
        assert stored.active is False
        assert stored.confidence == pytest.approx(0.0001)
        assert stored.contradiction_count == 1

    def test_retraction_only_fires_once(self, srs):
        srs._save_insight(_mk_insight("belief_warmth_baseline_low", confidence=1.0))
        with patch("anima_mcp.self_model.get_self_model",
                   return_value=self._mock_sm(confidence=0.0001)):
            srs._analyze_belief_insights()
            srs._analyze_belief_insights()
        assert srs._insights["belief_warmth_baseline_low"].contradiction_count == 1

    def test_validation_requires_new_evidence(self, srs):
        sm = self._mock_sm(confidence=0.9, supporting=90, contradicting=10)
        with patch("anima_mcp.self_model.get_self_model", return_value=sm):
            srs._analyze_belief_insights()  # creates, sample_count=100
            before = srs._insights["belief_warmth_baseline_low"].validation_count
            srs._analyze_belief_insights()  # same counts -> no bump
            mid = srs._insights["belief_warmth_baseline_low"].validation_count
            sm.beliefs["warmth_baseline_low"].supporting_count = 95  # new evidence
            srs._analyze_belief_insights()
            after = srs._insights["belief_warmth_baseline_low"].validation_count
        assert mid == before
        assert after == before + 1

    def test_confidence_resyncs_from_live_belief(self, srs):
        sm = self._mock_sm(confidence=0.9, supporting=90, contradicting=10)
        with patch("anima_mcp.self_model.get_self_model", return_value=sm):
            srs._analyze_belief_insights()
            sm.beliefs["warmth_baseline_low"].confidence = 0.75
            srs._analyze_belief_insights()
        assert srs._insights["belief_warmth_baseline_low"].confidence == pytest.approx(0.75)

    def test_insufficient_evidence_suspends_without_contradiction(self, srs):
        # A cold-start reset (counts -> 0) is "unknown", not "refuted":
        # the insight deactivates but earns no contradiction mark.
        srs._save_insight(_mk_insight("belief_warmth_baseline_low", confidence=1.0))
        sm = self._mock_sm(confidence=0.9, supporting=2, contradicting=1)
        with patch("anima_mcp.self_model.get_self_model", return_value=sm):
            srs._analyze_belief_insights()
        stored = srs._insights["belief_warmth_baseline_low"]
        assert stored.active is False
        assert stored.contradiction_count == 0
        assert stored.confidence == pytest.approx(0.9)

    def test_contradicting_evidence_does_not_validate(self, srs):
        sm = self._mock_sm(confidence=0.9, supporting=90, contradicting=10)
        with patch("anima_mcp.self_model.get_self_model", return_value=sm):
            srs._analyze_belief_insights()
            before = srs._insights["belief_warmth_baseline_low"].validation_count
            # New evidence arrives, but it LOWERED the belief's confidence:
            # that is not corroboration.
            sm.beliefs["warmth_baseline_low"].contradicting_count = 11
            sm.beliefs["warmth_baseline_low"].confidence = 0.85
            srs._analyze_belief_insights()
        stored = srs._insights["belief_warmth_baseline_low"]
        assert stored.validation_count == before
        assert stored.confidence == pytest.approx(0.85)


class TestZeroVarianceGuard:
    def _rows(self, srs, sensor_value_fn):
        conn = srs._connect()
        base = datetime.now() - timedelta(hours=20)
        for i in range(30):
            # Clarity varies with time; the sensor value comes from the fn.
            conn.execute(
                "INSERT INTO state_history VALUES (?, ?, ?, ?, ?, ?)",
                ((base + timedelta(minutes=30 * i)).isoformat(),
                 0.5, 0.5 + (0.3 if i >= 15 else 0.0), 0.8, 0.7,
                 json.dumps({"interaction_level": sensor_value_fn(i)})))
        conn.commit()
        return conn.execute("SELECT * FROM state_history").fetchall()

    def test_constant_sensor_yields_no_pattern(self, srs):
        rows = self._rows(srs, lambda i: 0.0)
        assert srs._analyze_sensor_correlation(
            rows, "interaction_level", "Interaction") is None

    def test_varying_sensor_still_detected(self, srs):
        # Sensor tracks the clarity split, so a real pattern exists.
        rows = self._rows(srs, lambda i: 1.0 if i >= 15 else 0.0)
        pattern = srs._analyze_sensor_correlation(
            rows, "interaction_level", "Interaction")
        assert pattern is not None
        assert "clarity" in pattern.outcome

    def test_negative_difference_stays_anchored_to_high_sensor_bucket(self, srs):
        # Clarity rises over time while the sensor falls, so high sensor values
        # correspond to the lower-clarity bucket.
        rows = self._rows(srs, lambda i: 0.0 if i >= 15 else 1.0)
        pattern = srs._analyze_sensor_correlation(
            rows, "interaction_level", "Interaction"
        )

        assert pattern is not None
        assert pattern.condition == "high interaction"
        assert pattern.outcome == "lower clarity"
        assert pattern.avg_clarity == pytest.approx(0.5)


class TestSignAwareVerification:
    def _seed_positive_light_clarity(self, srs):
        conn = srs._connect()
        base = datetime.now() - timedelta(hours=100)
        for i in range(60):
            lux = 10.0 + i * 5.0
            clarity = 0.4 + (i / 60) * 0.4  # rises with lux
            conn.execute(
                "INSERT INTO state_history VALUES (?, ?, ?, ?, ?, ?)",
                ((base + timedelta(hours=i)).isoformat(),
                 0.5, clarity, 0.8, 0.7,
                 json.dumps({"external_light_lux": lux})))
        conn.commit()

    def test_wrong_direction_contradicted(self, srs):
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "light reduces my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is False
        assert "opposes" in result.detail

    def test_right_direction_supported(self, srs):
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "light increases my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is True

    def test_anti_pole_wording_falls_back_to_magnitude(self, srs):
        # "dim light" claims the LOW pole of lux; nothing binds the marker
        # to a pole, so the signed check must stand down. With a real
        # positive light->clarity relationship the magnitude check supports.
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "dim light increases my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is True
        assert "opposes" not in result.detail

    def test_substring_marker_is_not_a_direction(self, srs):
        # "pointless" contains "less"; word-boundary matching must not read
        # it as a decrease claim (no real marker => unverifiable, which is
        # the pre-existing behavior for markerless text).
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "light is pointless for my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is None

    def test_direction_marker_inside_question_is_ignored(self, srs):
        # The kb text embeds the original question; a direction word there
        # is not a claim. Only the answer segment carries direction.
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "When I asked 'does light reduce clarity?', I learned: "
            "light increases my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is True

    def test_direction_marker_inside_question_is_ignored_told_wording(self, srs):
        # Same guard for the current minting wording. Without the "i was
        # told:" cut, "reduce" inside the question would be read as the claim
        # and mint a CONTRADICTED verdict against a correct answer.
        self._seed_positive_light_clarity(srs)
        result = srs._verify_qa_insight(
            "When I asked 'does light reduce clarity?', I was told: "
            "light increases my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is True

    def test_constant_sensor_is_unverifiable(self, srs):
        conn = srs._connect()
        base = datetime.now() - timedelta(hours=100)
        for i in range(60):
            conn.execute(
                "INSERT INTO state_history VALUES (?, ?, ?, ?, ?, ?)",
                ((base + timedelta(hours=i)).isoformat(),
                 0.5, 0.4 + (i / 60) * 0.4, 0.8, 0.7,
                 json.dumps({"external_light_lux": 4.0})))
        conn.commit()
        result = srs._verify_qa_insight(
            "light increases my clarity", InsightCategory.ENVIRONMENT)
        assert result.verified is None
        assert "no variance" in result.detail


class TestTrendStaleness:
    def _history_with_summaries(self, tmp_path, newest_age_days):
        from anima_mcp.anima_history import AnimaHistory
        history = AnimaHistory(persistence_path=tmp_path / "anima_history.json")
        summaries = []
        for i in range(7):
            day = datetime.now() - timedelta(days=newest_age_days + (6 - i))
            summaries.append({
                "date": day.isoformat(),
                "center": [0.5, 0.5 + i * 0.02, 0.8, 0.7],
                "variance": [0.01] * 4, "n_obs": 100, "hours": 8.0,
                "perturbations": 0, "trends": {},
            })
        (tmp_path / "day_summaries.json").write_text(
            json.dumps({"summaries": summaries}))
        return history

    def test_fresh_summaries_trend_detected(self, tmp_path):
        history = self._history_with_summaries(tmp_path, newest_age_days=0)
        trend = history.detect_long_term_trend("clarity")
        assert trend is not None
        assert trend["direction"] == "increasing"

    def test_stale_summaries_return_none(self, tmp_path):
        history = self._history_with_summaries(tmp_path, newest_age_days=60)
        assert history.detect_long_term_trend("clarity") is None

    def test_one_fresh_summary_cannot_launder_stale_ones(self, tmp_path):
        # One current summary + six 90-day-old ones must not read as a
        # seven-summary "recent" trend: only in-window summaries count,
        # and fewer than 3 of them is no trend data.
        from anima_mcp.anima_history import AnimaHistory
        history = AnimaHistory(persistence_path=tmp_path / "anima_history.json")
        rows = []
        for i in range(6):
            day = datetime.now() - timedelta(days=90 + i)
            rows.append({"date": day.isoformat(),
                         "center": [0.5, 0.3 + i * 0.05, 0.8, 0.7],
                         "variance": [0.01] * 4, "n_obs": 100, "hours": 8.0,
                         "perturbations": 0, "trends": {}})
        rows.append({"date": datetime.now().isoformat(),
                     "center": [0.5, 0.9, 0.8, 0.7],
                     "variance": [0.01] * 4, "n_obs": 100, "hours": 8.0,
                     "perturbations": 0, "trends": {}})
        (tmp_path / "day_summaries.json").write_text(json.dumps({"summaries": rows}))
        assert history.detect_long_term_trend("clarity") is None

    def test_unparsable_date_is_dropped_not_fatal(self, tmp_path):
        history = self._history_with_summaries(tmp_path, newest_age_days=0)
        data = json.loads((tmp_path / "day_summaries.json").read_text())
        data["summaries"][0]["date"] = "not-a-date"
        (tmp_path / "day_summaries.json").write_text(json.dumps(data))
        trend = history.detect_long_term_trend("clarity")
        assert trend is not None
        assert trend["n_summaries"] == 6

    def test_stale_trend_insight_deactivates(self, srs):
        srs._save_insight(_mk_insight("trend_clarity_increasing"))
        history = MagicMock()
        history.detect_long_term_trend.return_value = None
        with patch("anima_mcp.anima_history.get_anima_history", return_value=history):
            srs._analyze_long_term_trends()
        stored = srs._insights["trend_clarity_increasing"]
        assert stored.active is False
        # Unknown is not refuted: no contradiction mark for staleness.
        assert stored.contradiction_count == 0


class TestQaResync:
    def _kb_with(self, *insights):
        kb = MagicMock()
        kb.get_all_insights.return_value = list(insights)
        return kb

    def _qa(self, iid, confidence, text="pattern recognition connects things",
            ts=None):
        qa = MagicMock()
        qa.insight_id = iid
        qa.confidence = confidence
        qa.text = text
        qa.category = "general"
        qa.references = 2
        qa.timestamp = ts if ts is not None else time.time() - 86400 * 30
        qa.source_author = "Claude"
        return qa

    def test_confidence_drop_propagates(self, srs):
        stale = datetime.now() - timedelta(days=45)
        srs._save_insight(_mk_insight("qa_alpha", confidence=1.0, last_validated=stale))
        with patch("anima_mcp.knowledge.get_knowledge",
                   return_value=self._kb_with(self._qa("alpha", 0.85))):
            srs.sync_from_qa_knowledge()
        stored = srs._insights["qa_alpha"]
        assert stored.confidence == pytest.approx(0.85)
        assert stored.active is True
        # A resync is a fresh check: last_validated must move, or the recency
        # tie-break reads a just-rechecked row as stale.
        assert stored.last_validated > stale + timedelta(days=1)

    def test_below_floor_deactivates_without_contradiction_mark(self, srs):
        # A resync is a mirror pass, not an evidence event: a policy change
        # in the kb (e.g. the v3 rescale) must not stamp ~1,779 rows as
        # "contradicted".
        srs._save_insight(_mk_insight("qa_alpha", confidence=1.0))
        with patch("anima_mcp.knowledge.get_knowledge",
                   return_value=self._kb_with(self._qa("alpha", 0.4))):
            srs.sync_from_qa_knowledge()
        stored = srs._insights["qa_alpha"]
        assert stored.active is False
        assert stored.confidence == pytest.approx(0.4)
        assert stored.contradiction_count == 0

    def test_verification_penalty_survives_resync(self, srs):
        # Mint-time verification-CONTRADICTED signature: 0 validations,
        # >=1 contradiction. The kb's raw confidence must not resurrect it.
        contradicted = _mk_insight("qa_wrongway", confidence=0.36)
        contradicted.validation_count = 0
        contradicted.contradiction_count = 1
        srs._save_insight(contradicted)
        with patch("anima_mcp.knowledge.get_knowledge",
                   return_value=self._kb_with(self._qa("wrongway", 0.9))):
            srs.sync_from_qa_knowledge()
        stored = srs._insights["qa_wrongway"]
        assert stored.confidence == pytest.approx(0.9 * 0.4)
        assert stored.active is False  # 0.36 < the 0.6 sync floor

    def test_lumen_authored_qa_row_not_demoted(self, srs):
        lumen_qa = _mk_insight("qa_selfanswer")
        lumen_qa.source_author = "lumen"
        srs._save_insight(lumen_qa)
        srs._save_insight(_mk_insight("qa_external"))
        srs._save_insight(_mk_insight("pref_derived"))
        ids = [i.id for i in srs.get_insights()]
        # Lumen's own qa-bridged answer ranks with the self-derived rows,
        # ahead of external qa rows of equal strength.
        assert ids.index("qa_selfanswer") < ids.index("qa_external")

    def test_new_sync_uses_kb_timestamp(self, srs):
        learned = time.time() - 86400 * 90
        with patch("anima_mcp.knowledge.get_knowledge",
                   return_value=self._kb_with(self._qa("beta", 0.9, ts=learned))):
            srs.sync_from_qa_knowledge()
        stored = srs._insights["qa_beta"]
        assert abs(stored.discovered_at.timestamp() - learned) < 5


class TestKnowledgeV3Rescale:
    def _write_kb(self, path, rows, version=2):
        path.write_text(json.dumps({"schema_version": version, "insights": rows}))

    def _row(self, iid, author, confidence, reconverged=0.0, references=0):
        return {
            "insight_id": iid, "text": f"text {iid}", "source_question": "q",
            "source_answer": "a", "source_author": author,
            "timestamp": time.time() - 86400 * 100, "category": "general",
            "confidence": confidence, "references": references,
            "last_reconverged_at": reconverged,
        }

    def test_unearned_external_rescaled(self, tmp_path):
        kb_file = tmp_path / "knowledge.json"
        self._write_kb(kb_file, [
            self._row("ext_unearned", "Claude", 1.0),
            self._row("ext_earned", "Claude", 0.9, reconverged=time.time()),
            self._row("self_derived", "lumen", 1.0),
            self._row("ext_at_floor", "Kenny", 0.5),
        ])
        with patch("anima_mcp.knowledge._get_knowledge_path", return_value=kb_file):
            from anima_mcp.knowledge import KnowledgeBase
            kb = KnowledgeBase()
        by_id = {i.insight_id: i for i in kb.get_all_insights()}
        assert by_id["ext_unearned"].confidence == pytest.approx(0.5)
        assert by_id["ext_unearned"].legacy_confidence == pytest.approx(1.0)
        assert by_id["ext_earned"].confidence == pytest.approx(0.9)
        assert by_id["self_derived"].confidence == pytest.approx(1.0)
        assert by_id["ext_at_floor"].confidence == pytest.approx(0.5)
        assert by_id["ext_at_floor"].legacy_confidence is None
        saved = json.loads(kb_file.read_text())
        assert saved["schema_version"] >= 3
        # Whole-file sidecar written before the rescale.
        sidecar = kb_file.with_suffix(".pre-v3.json")
        assert sidecar.exists()
        assert json.loads(sidecar.read_text())["schema_version"] == 2

    def test_operator_authored_rows_are_not_exempt(self, tmp_path):
        # Authorship is not an exemption (#121): a Kenny-authored row at 1.0
        # is external prose and rescales like any other unearned external.
        kb_file = tmp_path / "knowledge.json"
        self._write_kb(kb_file, [self._row("kenny_taught", "Kenny", 1.0)])
        with patch("anima_mcp.knowledge._get_knowledge_path", return_value=kb_file):
            from anima_mcp.knowledge import KnowledgeBase
            kb = KnowledgeBase()
        row = kb.get_all_insights()[0]
        assert row.confidence == pytest.approx(0.5)
        assert row.legacy_confidence == pytest.approx(1.0)

    def test_v2_not_reapplied_to_post_v2_rows(self, tmp_path):
        kb_file = tmp_path / "knowledge.json"
        # A version-2 file whose row EARNED 5 references under the gated
        # logic (legacy_references None). The v3 bump must not re-compress.
        self._write_kb(kb_file, [
            self._row("honest_refs", "lumen", 0.7, references=5),
        ], version=2)
        with patch("anima_mcp.knowledge._get_knowledge_path", return_value=kb_file):
            from anima_mcp.knowledge import KnowledgeBase
            kb = KnowledgeBase()
        row = kb.get_all_insights()[0]
        assert row.references == 5
        assert row.legacy_references is None

    def test_migration_runs_once(self, tmp_path):
        kb_file = tmp_path / "knowledge.json"
        self._write_kb(kb_file, [self._row("ext_unearned", "Claude", 1.0)])
        with patch("anima_mcp.knowledge._get_knowledge_path", return_value=kb_file):
            from anima_mcp.knowledge import KnowledgeBase
            kb1 = KnowledgeBase()
            # Simulate post-migration confidence growth, persist, reload.
            kb1._insights[0].confidence = 0.8
            kb1._save()
            kb2 = KnowledgeBase()
        assert kb2.get_all_insights()[0].confidence == pytest.approx(0.8)


class TestSelfModelColdStart:
    def _seed(self, path, beliefs):
        data = {
            "beliefs": beliefs,
            "_migrated_noise_reset": True,
            "_migrated_episode_evidence_v2": True,
        }
        path.write_text(json.dumps(data))

    def test_dead_channel_beliefs_reset(self, tmp_path):
        from anima_mcp.self_model import SelfModel
        path = tmp_path / "self_model.json"
        self._seed(path, {
            "warmth_recovery": {"confidence": 0.51, "value": 0.895,
                                "supporting_count": 1626, "contradicting_count": 457},
            "stability_recovery": {"confidence": 0.639, "value": 0.516,
                                   "supporting_count": 2395, "contradicting_count": 1000},
            "interaction_clarity_boost": {"confidence": 0.502, "value": 0.703,
                                          "supporting_count": 1, "contradicting_count": 4},
            "question_asking_tendency": {"confidence": 0.49, "value": 0.698,
                                         "supporting_count": 0, "contradicting_count": 2},
            "light_sensitive": {"confidence": 0.48, "value": 0.5,
                                "supporting_count": 0, "contradicting_count": 5},
        })
        sm = SelfModel(persistence_path=path, read_only=False)
        # Each belief cold-starts to its own constructor prior — the 0.7
        # hypothesis seeds are design priors and survive the reset.
        expected = {
            "warmth_recovery": (0.5, 0.5),
            "stability_recovery": (0.5, 0.5),
            "interaction_clarity_boost": (0.5, 0.7),
            "question_asking_tendency": (0.5, 0.7),
        }
        for bid, (conf, value) in expected.items():
            b = sm.beliefs[bid]
            assert (b.confidence, b.value) == (conf, value), bid
            assert b.supporting_count == 0 and b.contradicting_count == 0, bid
        # Untouched belief keeps its state.
        assert sm.beliefs["light_sensitive"].contradicting_count == 5
        # The flag carries the pre-reset state as an audit trail.
        flag = json.loads(path.read_text())["_migrated_dead_channel_reset_v3"]
        assert flag["warmth_recovery"]["supporting_count"] == 1626
        assert flag["warmth_recovery"]["value"] == pytest.approx(0.895)

    def test_cold_start_runs_once(self, tmp_path):
        from anima_mcp.self_model import SelfModel
        path = tmp_path / "self_model.json"
        self._seed(path, {
            "warmth_recovery": {"confidence": 0.51, "value": 0.895,
                                "supporting_count": 10, "contradicting_count": 4},
        })
        SelfModel(persistence_path=path, read_only=False)
        data = json.loads(path.read_text())
        assert data["_migrated_dead_channel_reset_v3"]
        # Post-migration learning must survive a reload.
        data["beliefs"]["warmth_recovery"]["supporting_count"] = 3
        path.write_text(json.dumps(data))
        sm2 = SelfModel(persistence_path=path, read_only=False)
        assert sm2.beliefs["warmth_recovery"].supporting_count == 3

    def test_read_only_never_migrates(self, tmp_path):
        from anima_mcp.self_model import SelfModel
        path = tmp_path / "self_model.json"
        self._seed(path, {
            "warmth_recovery": {"confidence": 0.51, "value": 0.895,
                                "supporting_count": 10, "contradicting_count": 4},
        })
        SelfModel(persistence_path=path, read_only=True)
        assert "_migrated_dead_channel_reset_v3" not in json.loads(path.read_text())
