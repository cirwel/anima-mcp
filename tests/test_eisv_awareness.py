"""Tests for EISV trajectory awareness integration."""

import os
import sqlite3
import tempfile
from anima_mcp.eisv.mapping import (
    anima_to_eisv, compute_trajectory_window, classify_trajectory,
    TrajectoryShape, compute_derivatives,
)
from anima_mcp.eisv.expression import (
    ExpressionGenerator, translate_expression, generate_lumen_expression,
    TOKEN_MAP, ALL_TOKENS, LUMEN_TOKENS,
)
from anima_mcp.eisv.awareness import TrajectoryAwareness, compute_expression_coherence


class TestMapping:
    def test_anima_to_eisv_basic(self):
        result = anima_to_eisv(0.8, 0.7, 0.9, 0.5)
        assert result["E"] == 0.8
        assert result["I"] == 0.7
        assert abs(result["S"] - 0.1) < 1e-9
        assert abs(result["V"] - 0.15) < 1e-9

    def test_anima_to_eisv_clamping(self):
        result = anima_to_eisv(1.5, -0.5, 0.0, 2.0)
        assert result["E"] == 1.0
        assert result["I"] == 0.0
        assert result["S"] == 1.0
        assert result["V"] == 0.0

    def test_eisv_values_in_range(self):
        for w in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5, 1.0]:
                for s in [0.0, 0.5, 1.0]:
                    for p in [0.0, 0.5, 1.0]:
                        r = anima_to_eisv(w, c, s, p)
                        for k in ("E", "I", "S", "V"):
                            assert 0.0 <= r[k] <= 1.0


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

    def test_feedback_forwarding(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        for i in range(10):
            ta._buffer.append({"t": float(i), "E": 0.7, "I": 0.7, "S": 0.2, "V": 0.1})
        ta.get_trajectory_suggestion()
        before = ta._generator.get_weights("settled_presence").copy()
        ta.record_feedback(["~stillness~"], 0.9)
        after = ta._generator.get_weights("settled_presence")
        assert after["~stillness~"] > before["~stillness~"]

    def test_bootstrap_from_history(self):
        ta = TrajectoryAwareness(buffer_size=30)
        records = [
            {"timestamp": f"2026-01-01T00:0{i}:00", "warmth": 0.7, "clarity": 0.7, "stability": 0.8, "presence": 0.5}
            for i in range(5)
        ]
        added = ta.bootstrap_from_history(records)
        assert added == 5
        assert ta.buffer_size == 5

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

    def test_feedback_logs_event(self):
        ta = TrajectoryAwareness(buffer_size=30, db_path=self.db_path, seed=42)
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        ta.record_feedback(["~stillness~"], 0.9)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT event_type FROM trajectory_events ORDER BY id"
        ).fetchall()
        conn.close()
        event_types = [r[0] for r in rows]
        assert "feedback" in event_types

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
        assert state["expression_generator"]["feedback_count"] == 0

    def test_get_state_with_recent_events(self):
        ta = TrajectoryAwareness(
            buffer_size=30, db_path=self.db_path, seed=42
        )
        _make_settled_buffer(ta)
        ta.get_trajectory_suggestion()

        state = ta.get_state()
        assert len(state["recent_events"]) == 1
        assert state["recent_events"][0]["event_type"] == "classification"
        assert state["shape_distribution"]["settled_presence"] >= 1

    def test_get_state_window_seconds(self):
        ta = TrajectoryAwareness(buffer_size=30, seed=42)
        # Two states 60 seconds apart
        ta._buffer.append({"t": 1000.0, "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1})
        ta._buffer.append({"t": 1060.0, "E": 0.5, "I": 0.5, "S": 0.3, "V": 0.1})

        state = ta.get_state()
        assert state["buffer"]["window_seconds"] == 60.0


class TestCoherence:
    def test_full_overlap(self):
        assert compute_expression_coherence(["warm", "feel"], ["warm", "feel"]) == 1.0

    def test_no_overlap(self):
        assert compute_expression_coherence(["warm", "feel"], ["cold", "dim"]) == 0.0

    def test_partial_overlap(self):
        assert compute_expression_coherence(["warm", "feel"], ["warm", "cold"]) == 0.5

    def test_none_suggested(self):
        assert compute_expression_coherence(None, ["warm"]) is None

    def test_empty_suggested(self):
        assert compute_expression_coherence([], ["warm"]) is None


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
