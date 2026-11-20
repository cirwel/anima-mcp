"""Tests for EISV trajectory awareness integration."""

import os
import sqlite3
import tempfile

import pytest

from anima_mcp.eisv.mapping import (
    anima_to_eisv, compute_trajectory_window, classify_trajectory,
    TrajectoryShape, compute_derivatives,
)
from anima_mcp.eisv.expression import (
    ExpressionGenerator, translate_expression, generate_lumen_expression,
    TOKEN_MAP, ALL_TOKENS, LUMEN_TOKENS,
)
from anima_mcp.eisv.awareness import (
    TrajectoryAwareness,
    compute_expression_coherence,
    compute_suggestion_recall,
)


class TestMapping:
    def test_anima_to_eisv_basic(self):
        result = anima_to_eisv(0.8, 0.7, 0.9, 0.5)
        assert result["E"] == 0.8
        assert result["I"] == 0.7
        assert abs(result["S"] - 0.1) < 1e-9
        assert abs(result["V"] - 0.1) < 1e-9

    def test_anima_to_eisv_clamping(self):
        result = anima_to_eisv(1.5, -0.5, 0.0, 2.0)
        assert result["E"] == 1.0
        assert result["I"] == 0.0
        assert result["S"] == 1.0
        assert result["V"] == 1.0

    def test_eisv_values_in_range(self):
        for w in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5, 1.0]:
                for s in [0.0, 0.5, 1.0]:
                    for p in [0.0, 0.5, 1.0]:
                        r = anima_to_eisv(w, c, s, p)
                        for k in ("E", "I", "S"):
                            assert 0.0 <= r[k] <= 1.0
                        assert -1.0 <= r["V"] <= 1.0

    def test_mapping_matches_canonical_governance_mapping_without_neural_data(self):
        from anima_mcp.eisv_mapper import anima_components_to_eisv

        trajectory = anima_to_eisv(0.41, 0.78, 0.66, 0.92)
        governance = anima_components_to_eisv(0.41, 0.78, 0.66, 0.92).to_dict()

        assert trajectory == governance
        assert abs(trajectory["V"] + 0.37) < 1e-12


class TestDerivatives:
    def test_compute_derivatives_basic(self):
        states = [
            {"t": 0.0, "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1},
            {"t": 1.0, "E": 0.6, "I": 0.5, "S": 0.3, "V": 0.1},
            {"t": 2.0, "E": 0.7, "I": 0.5, "S": 0.3, "V": 0.1},
        ]
        derivs = compute_derivatives(states)
        assert len(derivs) == 2
        assert abs(derivs[0]["dE"] - 0.1) < 1e-9

    def test_trajectory_window_structure(self):
        states = [{"t": float(i), "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1} for i in range(5)]
        window = compute_trajectory_window(states)
        assert "states" in window
        assert "derivatives" in window
        assert "second_derivatives" in window


class TestShapeClassifier:
    def test_settled_presence(self):
        states = [{"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1} for i in range(10)]
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.SETTLED_PRESENCE

    def test_rising_entropy(self):
        states = [{"t": float(i), "E": 0.5, "I": 0.5, "S": 0.2 + i * 0.08, "V": 0.1} for i in range(10)]
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.RISING_ENTROPY

    @pytest.mark.parametrize("sample_interval", [0.5, 2.0, 10.0])
    def test_rising_entropy_uses_whole_window_change(self, sample_interval):
        """The same lived trajectory has one shape at every sample cadence."""
        states = []
        for index in range(30):
            progress = index / 29
            states.append({
                "t": index * sample_interval,
                "E": 0.5,
                "I": 0.5,
                "S": 0.2 + 0.12 * progress,
                "V": 0.0,
            })

        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.RISING_ENTROPY

    def test_falling_energy_uses_whole_window_change_at_live_cadence(self):
        states = []
        for index in range(30):
            progress = index / 29
            energy = 0.6 - 0.12 * progress
            states.append({
                "t": index * 2.0,
                "E": energy,
                "I": 0.6,
                "S": 0.2,
                "V": energy - 0.6,
            })

        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.FALLING_ENERGY

    def test_valence_rising_uses_signed_window_change_at_live_cadence(self):
        states = []
        for index in range(30):
            progress = index / 29
            integrity = 0.7 - 0.12 * progress
            states.append({
                "t": index * 2.0,
                "E": 0.5,
                "I": integrity,
                "S": 0.2,
                "V": 0.5 - integrity,
            })

        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.VALENCE_RISING

    def test_basin_transition_down(self):
        states = [{"t": float(i), "E": 0.8 - i * 0.04, "I": 0.5, "S": 0.2, "V": 0.1} for i in range(10)]
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.BASIN_TRANSITION_DOWN

    def test_convergence(self):
        # Small decaying oscillation
        states = []
        for i in range(10):
            amp = 0.01 * (0.8 ** i)
            states.append({"t": float(i), "E": 0.5 + amp, "I": 0.5 - amp, "S": 0.3, "V": 0.1})
        window = compute_trajectory_window(states)
        assert classify_trajectory(window) == TrajectoryShape.CONVERGENCE


class TestExpressionGenerator:
    def test_generates_valid_tokens(self):
        gen = ExpressionGenerator(seed=42)
        for shape in TrajectoryShape:
            tokens = gen.generate(shape.value)
            assert len(tokens) >= 1
            assert all(t in ALL_TOKENS for t in tokens)

    def test_deterministic_with_seed(self):
        gen1 = ExpressionGenerator(seed=42)
        gen2 = ExpressionGenerator(seed=42)
        for shape in TrajectoryShape:
            assert gen1.generate(shape.value) == gen2.generate(shape.value)

    def test_weight_update(self):
        gen = ExpressionGenerator(seed=42)
        before = gen.get_weights("settled_presence").copy()
        gen.update_weights("settled_presence", ["~stillness~"], 0.9)
        after = gen.get_weights("settled_presence")
        assert after["~stillness~"] > before["~stillness~"]


class TestBridge:
    def test_token_map_completeness(self):
        assert set(TOKEN_MAP.keys()) == set(ALL_TOKENS)
        for mapped in TOKEN_MAP.values():
            assert all(t in LUMEN_TOKENS for t in mapped)

    def test_translate_expression(self):
        result = translate_expression(["~warmth~", "~curiosity~"])
        assert len(result) <= 3
        assert all(t in LUMEN_TOKENS for t in result)

    def test_translate_empty(self):
        assert translate_expression([]) == []

    def test_translate_caps_at_3(self):
        result = translate_expression(["~warmth~", "~curiosity~", "~resonance~", "~stillness~"])
        assert len(result) <= 3

    def test_generate_lumen_expression_pipeline(self):
        result = generate_lumen_expression("settled_presence", {"E": 0.7, "I": 0.7, "S": 0.1, "V": 0.05})
        assert "shape" in result
        assert "suggested_tokens" not in result  # This is in awareness, not here
        assert "lumen_tokens" in result
        assert "eisv_tokens" in result
        assert all(t in LUMEN_TOKENS for t in result["lumen_tokens"])


class TestTrajectoryAwareness:
    def test_insufficient_data_returns_none(self):
        ta = TrajectoryAwareness(buffer_size=30)
        # Only 3 states, need 5
        for i in range(3):
            ta._buffer.append({"t": float(i), "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1})
        assert ta.get_trajectory_suggestion() is None

    def test_sufficient_data_returns_suggestion(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        for i in range(10):
            ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})
        result = ta.get_trajectory_suggestion()
        assert result is not None
        assert "shape" in result
        assert "suggested_tokens" in result
        assert "eisv_tokens" in result
        assert "trigger" in result
        assert result["shape"] == "settled_presence"

    def test_record_state_subsampling(self):
        ta = TrajectoryAwareness(buffer_size=30)
        ta._last_record_time = 0  # Reset
        ta.record_state(0.5, 0.5, 0.5, 0.5)
        assert len(ta._buffer) == 1
        # Immediately recording again should be subsampled away
        ta.record_state(0.6, 0.6, 0.6, 0.6)
        assert len(ta._buffer) == 1  # Still 1

    def test_record_state_accepts_exact_operational_eisv(self):
        ta = TrajectoryAwareness(buffer_size=30)
        operational = {"E": 0.31, "I": 0.72, "S": 0.18, "V": -0.41}

        ta.record_state(0.8, 0.2, 0.3, 0.4, eisv=operational)

        assert {k: ta._buffer[0][k] for k in operational} == operational

    def test_record_state_produces_governance_scaled_ethical_drift(self):
        ta = TrajectoryAwareness(buffer_size=30)
        ta._last_record_time = 0
        ta.record_state(0.5, 0.5, 0.7, 0.5)
        ta._last_record_time = 0
        ta.record_state(0.62, 0.5, 0.7, 0.5)

        assert ta._buffer[0]["ethical_drift"] == 0.0
        assert ta._buffer[1]["ethical_drift"] == pytest.approx(0.36)
        window = compute_trajectory_window(list(ta._buffer))
        assert classify_trajectory(window) == TrajectoryShape.DRIFT_DISSONANCE

    def test_caching(self):
        ta = TrajectoryAwareness(buffer_size=30, cache_seconds=60.0, seed=42)
        for i in range(10):
            ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})
        r1 = ta.get_trajectory_suggestion()
        r2 = ta.get_trajectory_suggestion()
        assert r1 is r2  # Same object (cached)

    def test_current_shape_property(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        assert ta.current_shape is None
        for i in range(10):
            ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})
        ta.get_trajectory_suggestion()
        assert ta.current_shape == "settled_presence"

    def test_eisv_weight_feedback_preserves_update_behavior(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        for i in range(10):
            ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})
        ta.get_trajectory_suggestion()
        before = ta._generator.get_weights("settled_presence").copy()
        matched_count = ta.record_eisv_weight_feedback(["~stillness~"], 0.9)
        after = ta._generator.get_weights("settled_presence")
        assert matched_count == 1
        assert after["~stillness~"] == pytest.approx(
            before["~stillness~"] + 0.064
        )

    def test_suggestion_recall_recording_does_not_update_weights(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()
        before = ta._generator.get_weights("settled_presence")

        score = ta.record_suggestion_recall(
            ["quiet", "here"],
            ["quiet", "soft"],
        )

        assert score == 0.5
        assert ta._generator.get_weights("settled_presence") == before

    def test_bootstrap_from_history(self):
        ta = TrajectoryAwareness(buffer_size=30)
        records = [
            {"timestamp": f"2026-01-01T00:0{i}:00", "warmth": 0.7, "clarity": 0.7, "stability": 0.8, "presence": 0.5}
            for i in range(5)
        ]
        added = ta.bootstrap_from_history(records)
        assert added == 5
        assert ta.buffer_size == 5

    def test_bootstrap_reconstructs_ethical_drift(self):
        ta = TrajectoryAwareness(buffer_size=30)
        records = [
            {
                "timestamp": "2026-01-01T00:00:00",
                "warmth": 0.5,
                "clarity": 0.5,
                "stability": 0.7,
                "presence": 0.5,
            },
            {
                "timestamp": "2026-01-01T00:00:02",
                "warmth": 0.5,
                "clarity": 0.38,
                "stability": 0.7,
                "presence": 0.5,
            },
        ]

        assert ta.bootstrap_from_history(records) == 2
        assert ta._buffer[-1]["ethical_drift"] == pytest.approx(0.36)

    def test_graceful_failure(self):
        ta = TrajectoryAwareness(buffer_size=30)
        # Add corrupted data
        for i in range(10):
            ta._buffer.append({"t": float(i)})  # Missing E, I, S, V
        # Should return None, not raise
        assert ta.get_trajectory_suggestion() is None


def _make_settled_buffer(ta, n=10):
    """Helper: fill buffer with settled_presence data."""
    for i in range(n):
        ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})


class TestPersistence:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name

    def teardown_method(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_init_db_creates_table(self):
        TrajectoryAwareness(buffer_size=30, db_path=self.db_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trajectory_events'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_db_migrates_legacy_schema_idempotently(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE trajectory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                shape TEXT,
                eisv_state TEXT,
                derivatives TEXT,
                suggested_tokens TEXT,
                expression_tokens TEXT,
                coherence_score REAL,
                cache_hit INTEGER DEFAULT 0,
                buffer_size INTEGER
            )"""
        )
        conn.execute(
            """INSERT INTO trajectory_events
               (timestamp, event_type, shape, suggested_tokens,
                expression_tokens, coherence_score)
               VALUES (?, 'feedback', 'settled_presence', ?, ?, 0.75)""",
            (
                "2026-01-01T00:00:00+00:00",
                '[ "warm", "feel" ]',
                '[ "warm", "cold" ]',
            ),
        )
        conn.commit()
        conn.close()

        first = TrajectoryAwareness(buffer_size=30, db_path=self.db_path)
        assert first.record_suggestion_recall(
            ["quiet"],
            ["quiet", "soft"],
            shape="settled_presence",
        ) == 1.0
        first.close()
        second = TrajectoryAwareness(buffer_size=30, db_path=self.db_path)
        second.close()

        conn = sqlite3.connect(self.db_path)
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(trajectory_events)")
        ]
        legacy_row = conn.execute(
            "SELECT suggested_tokens, expression_tokens, coherence_score, "
            "suggestion_recall, suggested_count, actual_count, "
            "eisv_weight_feedback_score "
            "FROM trajectory_events WHERE timestamp = ?",
            ("2026-01-01T00:00:00+00:00",),
        ).fetchone()
        typed_row = conn.execute(
            "SELECT coherence_score, suggestion_recall, suggested_count, "
            "actual_count FROM trajectory_events "
            "WHERE suggestion_recall IS NOT NULL"
        ).fetchone()
        conn.close()

        assert columns.count("suggestion_recall") == 1
        assert columns.count("suggested_count") == 1
        assert columns.count("actual_count") == 1
        assert columns.count("eisv_weight_feedback_score") == 1
        assert legacy_row == (
            '[ "warm", "feel" ]',
            '[ "warm", "cold" ]',
            0.75,
            None,
            None,
            None,
            None,
        )
        assert typed_row == (None, 1.0, 1, 2)

    def test_init_db_completes_partial_suggestion_recall_migration(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE trajectory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                shape TEXT,
                eisv_state TEXT,
                derivatives TEXT,
                suggested_tokens TEXT,
                expression_tokens TEXT,
                coherence_score REAL,
                cache_hit INTEGER DEFAULT 0,
                buffer_size INTEGER,
                suggestion_recall REAL
            )"""
        )
        conn.commit()
        conn.close()

        awareness = TrajectoryAwareness(
            buffer_size=30,
            db_path=self.db_path,
        )
        awareness.close()

        conn = sqlite3.connect(self.db_path)
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(trajectory_events)")
        ]
        conn.close()

        assert columns.count("suggestion_recall") == 1
        assert columns.count("suggested_count") == 1
        assert columns.count("actual_count") == 1
        assert columns.count("eisv_weight_feedback_score") == 1

    def test_log_event_writes_row(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        ta._log_event(
            event_type="test",
            shape="settled_presence",
            eisv_state={"E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1},
        )
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM trajectory_events").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_log_event_no_db_path_is_noop(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        # Should not raise even without db_path
        ta._log_event(event_type="test", shape="settled_presence")

    def test_suggestion_logs_event(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        result = ta.get_trajectory_suggestion()
        assert result is not None

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT event_type, shape FROM trajectory_events"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "classification"
        assert rows[0][1] == "settled_presence"

    def test_eisv_weight_update_logs_only_matched_feedback(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        assert ta.record_eisv_weight_feedback(["~stillness~"], 0.9) == 1

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT event_type, eisv_weight_feedback_score "
            "FROM trajectory_events ORDER BY id"
        ).fetchall()
        conn.close()
        assert ("eisv_weight_update", 0.9) in rows

    def test_deprecated_feedback_wrapper_keeps_zero_match_filter(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        assert ta.record_feedback(["quiet"], 0.9) == 0

    def test_lumen_weight_feedback_is_a_true_noop(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()
        before_weights = ta._generator.get_weights("settled_presence")
        conn = sqlite3.connect(self.db_path)
        before_events = conn.execute(
            "SELECT COUNT(*) FROM trajectory_events"
        ).fetchone()[0]
        conn.close()

        assert ta.record_eisv_weight_feedback(["quiet", "warm"], 0.9) == 0

        conn = sqlite3.connect(self.db_path)
        after_events = conn.execute(
            "SELECT COUNT(*) FROM trajectory_events"
        ).fetchone()[0]
        conn.close()
        state = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert ta._generator.get_weights("settled_presence") == before_weights
        assert after_events == before_events
        assert state == {
            "scope": "process_lifetime",
            "persisted": False,
            "fallback_only": True,
            "student_active": False,
            "update_count": 0,
            "matched_token_count": 0,
            "positive_updates": 0,
            "negative_updates": 0,
            "net_reward": None,
            "skipped": {
                "not_fallback_generated": 0,
                "degenerate_stratum": 0,
                "unknown_bounds": 0,
                "floor_violation": 0,
                "ceiling_violation": 0,
                "no_baseline": 0,
                "zero_delta": 0,
                "saturated": 0,
                "no_shape": 0,
                "update_error": 0,
            },
        }

    def test_suggestion_recall_uses_new_nullable_fields(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)

        assert ta.record_suggestion_recall(
            ["quiet", "here"],
            ["quiet"],
            shape="settled_presence",
        ) == 0.5

        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT suggestion_recall, suggested_count, actual_count, "
            "coherence_score FROM trajectory_events WHERE event_type='suggestion'"
        ).fetchone()
        conn.close()
        assert row == (0.5, 2, 1, None)

    def test_cache_hit_does_not_log_again(self):
        ta = TrajectoryAwareness(
            buffer_size=30, db_path=self.db_path, cache_seconds=60.0, seed=42
        )
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()  # Fresh classification -> logs
        ta.get_trajectory_suggestion()  # Cache hit -> should NOT log

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM trajectory_events").fetchall()
        conn.close()
        assert len(rows) == 1  # Only the first classification

    @pytest.mark.parametrize(
        "suggested_count,actual_count,score",
        [
            (1, 1, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 1, 0.5),
            (1, 2, 0.0),
        ],
    )
    def test_anchored_recall_skips_degenerate_strata_without_mutating_weights(
        self, suggested_count, actual_count, score
    ):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()
        before = ta._generator.get_weights("settled_presence").copy()
        conn = sqlite3.connect(self.db_path)
        before_rows = conn.execute("SELECT COUNT(*) FROM trajectory_events").fetchone()[0]
        conn.close()

        updated = ta.record_eisv_weight_feedback(
            ["~stillness~"],
            score,
            suggested_count=suggested_count,
            actual_count=actual_count,
            generator_source="fallback",
            anchored_recall=True,
        )

        assert updated == 0
        assert ta._generator.get_weights("settled_presence") == before
        updates = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert updates["update_count"] == 0
        assert updates["skipped"]["degenerate_stratum"] == 1
        conn = sqlite3.connect(self.db_path)
        after_rows = conn.execute("SELECT COUNT(*) FROM trajectory_events").fetchone()[0]
        conn.close()
        assert after_rows == before_rows

    def test_anchored_recall_skips_student_generated_output(self, tmp_path):
        from pathlib import Path

        model_dir = Path(__file__).resolve().parent.parent / "data" / "student_model"
        if not model_dir.exists():
            pytest.skip("student_model artifacts absent")
        ta = TrajectoryAwareness(
            buffer_size=30,
            db_path=str(tmp_path / "traj.db"),
            seed=42,
            student_model_dir=str(model_dir),
        )
        _make_settled_buffer(ta)
        suggestion = ta.get_trajectory_suggestion()
        assert suggestion["generator_source"] == "student"
        before = ta._generator.get_weights("settled_presence").copy()

        # Explicit "student", omitted (None), and unrecognized sources all skip (affirmative gate).
        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            1.0,
            suggested_count=2,
            actual_count=3,
            generator_source=suggestion["generator_source"],
            anchored_recall=True,
        ) == 0
        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            1.0,
            suggested_count=2,
            actual_count=3,
            anchored_recall=True,
        ) == 0
        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            1.0,
            suggested_count=2,
            actual_count=3,
            generator_source="unknown_source",
            anchored_recall=True,
        ) == 0

        assert ta._generator.get_weights("settled_presence") == before
        updates = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert updates["student_active"] is True
        assert updates["update_count"] == 0
        assert updates["skipped"]["not_fallback_generated"] == 3
        ta.close()

        # Also verify omitted/unknown provenance fails closed even when student is NOT loaded.
        ta_fb = TrajectoryAwareness(buffer_size=30, seed=42)
        _make_settled_buffer(ta_fb)
        ta_fb.get_trajectory_suggestion()
        assert ta_fb.record_eisv_weight_feedback(
            ["~stillness~"],
            1.0,
            suggested_count=2,
            actual_count=3,
            anchored_recall=True,
        ) == 0
        assert ta_fb.get_state()["expression_generator"]["eisv_weight_updates"]["skipped"]["not_fallback_generated"] == 1

    def test_anchored_recall_skips_floor_and_ceiling_violations_as_anomalies(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()
        before = ta._generator.get_weights("settled_presence").copy()

        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            0.0,
            suggested_count=2,
            actual_count=2,
            generator_source="fallback",
            anchored_recall=True,
        ) == 0
        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            0.75,
            suggested_count=4,
            actual_count=2,
            generator_source="fallback",
            anchored_recall=True,
        ) == 0

        assert ta._generator.get_weights("settled_presence") == before
        updates = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert updates["skipped"]["floor_violation"] == 1
        assert updates["skipped"]["ceiling_violation"] == 1
        assert ta._free_recall_baseline == {}

    def test_anchored_recall_welford_centering_prevents_ratchet_at_realistic_hit_rate(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()
        initial_w = ta._generator.get_weights("settled_presence")["~stillness~"]

        # First informative sample seeds baseline without updating.
        assert ta.record_eisv_weight_feedback(
            ["~stillness~"],
            0.5,
            suggested_count=2,
            actual_count=3,
            generator_source="fallback",
            anchored_recall=True,
        ) == 0
        assert ta._generator.get_weights("settled_presence")["~stillness~"] == initial_w

        # Simulate realistic 20% full-recall (score=1.0) / 80% floor (score=0.5) stream.
        stream = [1.0 if i % 5 == 0 else 0.5 for i in range(40)]
        for score in stream:
            ta.record_eisv_weight_feedback(
                ["~stillness~"],
                score,
                suggested_count=2,
                actual_count=3,
                generator_source="fallback",
                anchored_recall=True,
            )

        updates = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert updates["skipped"]["no_baseline"] == 1
        assert updates["positive_updates"] > 0
        assert updates["negative_updates"] > 0
        assert updates["update_count"] == (
            updates["positive_updates"] + updates["negative_updates"]
        )
        # Centered around running hit rate (~0.2), weights stay near initial instead of ratcheting to 0.1 or 10.0.
        final_w = ta._generator.get_weights("settled_presence")["~stillness~"]
        assert abs(final_w - initial_w) < 0.25
        assert abs(updates["net_reward"]) < 1.5

    def test_zero_delta_saturated_shape_routing_and_error_buckets(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        # Missing shape increments no_shape
        assert ta.record_eisv_weight_feedback(["~stillness~"], 0.9) == 0

        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        # score == 0.5 has reward 0 -> skipped as zero_delta
        assert ta.record_eisv_weight_feedback(["~stillness~"], 0.5) == 0

        # Clamp token to ceiling 10.0, then try pushing further up -> skipped as saturated
        ta._generator._token_weights["settled_presence"]["~stillness~"] = 10.0
        assert ta.record_eisv_weight_feedback(["~stillness~"], 1.0) == 0

        # Explicit shape routes credit to the suggestion's shape even if _current_shape changed.
        ta._current_shape = "convergence"
        before_converge = ta._generator.get_weights("convergence")["~resonance~"]
        before_settled = ta._generator.get_weights("settled_presence")["~resonance~"]
        assert ta.record_eisv_weight_feedback(
            ["~resonance~"],
            1.0,
            shape="settled_presence",
        ) == 1
        assert ta._generator.get_weights("settled_presence")["~resonance~"] > before_settled
        assert ta._generator.get_weights("convergence")["~resonance~"] == before_converge

        # Exception during generator update increments update_error and leaves baseline untouched.
        ta._free_recall_baseline[("settled_presence", 2, 3)] = (1, 0.0)
        original_update = ta._generator.update_weights
        def _boom(*args, **kwargs):
            raise RuntimeError("boom")
        ta._generator.update_weights = _boom
        assert ta.record_eisv_weight_feedback(
            ["~resonance~"],
            1.0,
            shape="settled_presence",
            suggested_count=2,
            actual_count=3,
            generator_source="fallback",
            anchored_recall=True,
        ) == 0
        ta._generator.update_weights = original_update
        assert ta._free_recall_baseline[("settled_presence", 2, 3)] == (1, 0.0)

        updates = ta.get_state()["expression_generator"]["eisv_weight_updates"]
        assert updates["update_count"] == 1
        assert updates["skipped"]["no_shape"] == 1
        assert updates["skipped"]["zero_delta"] == 1
        assert updates["skipped"]["saturated"] == 1
        assert updates["skipped"]["update_error"] == 1


class TestGetState:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name

    def teardown_method(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_get_state_empty_buffer(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        state = ta.get_state()
        assert state["current_shape"] is None
        assert state["current_eisv"] is None
        assert state["derivatives"] is None
        assert state["buffer"]["size"] == 0
        assert state["buffer"]["capacity"] == 30

    def test_get_state_with_data(self):
        ta = TrajectoryAwareness(
            buffer_size=30, cache_seconds=60.0, db_path=self.db_path, seed=42
        )
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        state = ta.get_state()
        assert state["current_shape"] == "settled_presence"
        assert state["current_eisv"] is not None
        assert state["current_eisv"]["E"] == 0.7
        assert state["buffer"]["size"] == 10
        assert state["cache"]["shape"] == "settled_presence"
        assert state["expression_generator"]["total_generations"] == 1
        assert "feedback_count" not in state["expression_generator"]
        assert "mean_coherence" not in state["expression_generator"]
        assert state["expression_generator"]["suggestion_recall"] == {
            "scope": "persisted_history",
            "strata": [],
        }

    def test_get_state_with_recent_events(self):
        ta = TrajectoryAwareness(
            buffer_size=30, db_path=self.db_path, seed=42
        )
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        state = ta.get_state()
        assert len(state["recent_events"]) == 1
        assert state["recent_events"][0]["event_type"] == "classification"
        assert "legacy_mixed_score" in state["recent_events"][0]
        assert state["recent_events"][0]["legacy_score_kind"] is None
        assert "coherence_score" not in state["recent_events"][0]
        assert state["shape_distribution"]["settled_presence"] >= 1

    def test_get_state_labels_legacy_score_units_conservatively(self):
        ta = TrajectoryAwareness(
            buffer_size=30, db_path=self.db_path, seed=42
        )
        ta._log_event(event_type="suggestion", coherence_score=0.8)
        ta._log_event(event_type="feedback", coherence_score=0.6)

        events = ta.get_state()["recent_events"]

        assert [event["legacy_mixed_score"] for event in events] == [0.8, 0.6]
        assert [event["legacy_score_kind"] for event in events] == [
            "suggestion_recall",
            "untyped_feedback_score",
        ]

    def test_get_state_uses_persisted_suggestion_recall_strata(self):
        ta = TrajectoryAwareness(
            buffer_size=30, db_path=self.db_path, seed=42
        )
        ta.record_suggestion_recall(["quiet"], ["quiet"])
        ta.record_suggestion_recall(["quiet", "here"], ["quiet"])
        ta.record_suggestion_recall(["quiet", "here"], ["quiet", "soft"])

        recall = ta.get_state()["expression_generator"]["suggestion_recall"]

        assert recall == {
            "scope": "persisted_history",
            "strata": [
                {
                    "suggested_count": 1,
                    "actual_count": 1,
                    "observation_count": 1,
                    "mean_suggestion_recall": 1.0,
                    "anchored_floor": 1.0,
                    "achievable_ceiling": 1.0,
                    "informative": False,
                },
                {
                    "suggested_count": 2,
                    "actual_count": 1,
                    "observation_count": 1,
                    "mean_suggestion_recall": 0.5,
                    "anchored_floor": 0.5,
                    "achievable_ceiling": 0.5,
                    "informative": False,
                },
                {
                    "suggested_count": 2,
                    "actual_count": 2,
                    "observation_count": 1,
                    "mean_suggestion_recall": 0.5,
                    "anchored_floor": 0.5,
                    "achievable_ceiling": 1.0,
                    "informative": True,
                },
            ],
        }

    def test_get_state_labels_in_memory_recall_process_lifetime(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        ta.record_suggestion_recall(["quiet"], ["quiet", "soft"])

        recall = ta.get_state()["expression_generator"]["suggestion_recall"]

        assert recall == {
            "scope": "process_lifetime",
            "strata": [
                {
                    "suggested_count": 1,
                    "actual_count": 2,
                    "observation_count": 1,
                    "mean_suggestion_recall": 1.0,
                    "anchored_floor": 1.0,
                    "achievable_ceiling": 1.0,
                    "informative": False,
                }
            ],
        }

    def test_get_state_window_seconds(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        # Two states 60 seconds apart
        ta._buffer.append({"t": 1000.0, "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1})
        ta._buffer.append({"t": 1060.0, "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1})

        state = ta.get_state()
        assert state["buffer"]["window_seconds"] == 60.0


class TestSuggestionRecall:
    def test_single_suggestion_remains_perfect_with_extra_output(self):
        """The forced anchor makes this 1.0; do not change the denominator."""
        assert compute_suggestion_recall(
            ["warm"],
            ["warm", "cold"],
        ) == 1.0

    def test_full_overlap(self):
        assert compute_suggestion_recall(["warm", "feel"], ["warm", "feel"]) == 1.0

    def test_no_overlap(self):
        assert compute_suggestion_recall(["warm", "feel"], ["cold", "dim"]) == 0.0

    def test_partial_overlap(self):
        assert compute_suggestion_recall(["warm", "feel"], ["warm", "cold"]) == 0.5

    def test_none_suggested(self):
        assert compute_suggestion_recall(None, ["warm"]) is None

    def test_empty_suggested(self):
        assert compute_suggestion_recall([], ["warm"]) is None

    def test_deprecated_name_is_a_pure_alias(self):
        assert compute_expression_coherence(
            ["warm", "feel"], ["warm"]
        ) == compute_suggestion_recall(["warm", "feel"], ["warm"])


class TestLEDShapeBias:
    def test_settled_presence_warm(self):
        from anima_mcp.display.leds import get_shape_color_bias
        bias = get_shape_color_bias("settled_presence")
        assert bias[0] > 0  # Warmer red
        assert bias[2] <= 0  # Less blue

    def test_convergence_warm(self):
        from anima_mcp.display.leds import get_shape_color_bias
        bias = get_shape_color_bias("convergence")
        assert bias[0] >= 0  # Warm: non-negative red
        assert bias[2] <= 0  # Warm: non-positive blue

    def test_unknown_shape_zero(self):
        from anima_mcp.display.leds import get_shape_color_bias
        bias = get_shape_color_bias("not_a_shape")
        assert bias == (0, 0, 0)

    def test_none_shape_zero(self):
        from anima_mcp.display.leds import get_shape_color_bias
        bias = get_shape_color_bias(None)
        assert bias == (0, 0, 0)

    def test_all_shapes_small_magnitude(self):
        """All biases should be subtle (<=15 per channel)."""
        from anima_mcp.display.leds import get_shape_color_bias
        from anima_mcp.eisv.mapping import TrajectoryShape
        for shape in TrajectoryShape:
            bias = get_shape_color_bias(shape.value)
            assert all(abs(c) <= 15 for c in bias), f"{shape.value}: {bias} too large"
