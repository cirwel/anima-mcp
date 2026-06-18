"""Tests for eisv/expression.py — expression generator and Lumen bridge."""

from anima_mcp.eisv.expression import (
    ExpressionGenerator,
    StudentExpressionGenerator,
    translate_expression,
    shape_to_lumen_trigger,
    generate_lumen_expression,
    ALL_TOKENS,
    SHAPE_TOKEN_AFFINITY,
    SHAPE_PATTERN_WEIGHTS,
    TOKEN_MAP,
    LUMEN_TOKENS,
)


# ── ExpressionGenerator ──

class TestExpressionGeneratorInit:
    def test_init_populates_weights(self):
        gen = ExpressionGenerator(seed=42)
        assert len(gen._token_weights) > 0

    def test_weights_have_all_tokens(self):
        gen = ExpressionGenerator(seed=42)
        for shape, weights in gen._token_weights.items():
            for token in ALL_TOKENS:
                assert token in weights

    def test_affine_tokens_weighted_higher(self):
        gen = ExpressionGenerator(seed=42)
        for shape, affine_tokens in SHAPE_TOKEN_AFFINITY.items():
            weights = gen._token_weights[shape]
            for token in affine_tokens:
                assert weights[token] == 3.0

    def test_non_affine_tokens_weighted_default(self):
        gen = ExpressionGenerator(seed=42)
        shape = "settled_presence"
        affine = set(SHAPE_TOKEN_AFFINITY[shape])
        weights = gen._token_weights[shape]
        for token in ALL_TOKENS:
            if token not in affine:
                assert weights[token] == 1.0


class TestExpressionGeneratorGenerate:
    def test_deterministic_with_seed(self):
        gen1 = ExpressionGenerator(seed=42)
        gen2 = ExpressionGenerator(seed=42)
        for shape in SHAPE_PATTERN_WEIGHTS:
            assert gen1.generate(shape) == gen2.generate(shape)

    def test_returns_list_of_strings(self):
        gen = ExpressionGenerator(seed=42)
        for shape in SHAPE_PATTERN_WEIGHTS:
            result = gen.generate(shape)
            assert isinstance(result, list)
            assert all(isinstance(t, str) for t in result)

    def test_tokens_from_vocabulary(self):
        gen = ExpressionGenerator(seed=42)
        for shape in SHAPE_PATTERN_WEIGHTS:
            result = gen.generate(shape)
            for token in result:
                assert token in ALL_TOKENS

    def test_all_shapes_generate(self):
        gen = ExpressionGenerator(seed=0)
        for shape in SHAPE_PATTERN_WEIGHTS:
            result = gen.generate(shape)
            assert len(result) >= 1
            assert len(result) <= 3

    def test_single_pattern(self):
        # Run many times to find a single pattern
        found_single = False
        for _ in range(100):
            gen_test = ExpressionGenerator(seed=_)
            result = gen_test.generate("settled_presence")
            if len(result) == 1:
                found_single = True
                break
        assert found_single

    def test_pair_pattern_unique_tokens(self):
        found_pair = False
        for i in range(100):
            gen_test = ExpressionGenerator(seed=i)
            result = gen_test.generate("rising_entropy")
            if len(result) == 2 and result[0] != result[1]:
                found_pair = True
                break
        assert found_pair

    def test_repetition_pattern_same_token(self):
        found_rep = False
        for i in range(200):
            gen_test = ExpressionGenerator(seed=i)
            result = gen_test.generate("falling_energy")
            if len(result) == 2 and result[0] == result[1]:
                found_rep = True
                break
        assert found_rep

    def test_unknown_shape_uses_fallback(self):
        gen = ExpressionGenerator(seed=42)
        result = gen.generate("totally_unknown_shape")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestExpressionGeneratorWeights:
    def test_update_weights_positive_score(self):
        gen = ExpressionGenerator(seed=42)
        shape = "settled_presence"
        token = ALL_TOKENS[0]
        original = gen.get_weights(shape)[token]
        gen.update_weights(shape, [token], 1.0)  # max score
        assert gen.get_weights(shape)[token] > original

    def test_update_weights_negative_score(self):
        gen = ExpressionGenerator(seed=42)
        shape = "settled_presence"
        token = ALL_TOKENS[0]
        original = gen.get_weights(shape)[token]
        gen.update_weights(shape, [token], 0.0)  # min score
        assert gen.get_weights(shape)[token] < original

    def test_update_weights_clamped_low(self):
        gen = ExpressionGenerator(seed=42)
        shape = "settled_presence"
        token = ALL_TOKENS[0]
        # Drive weight down many times
        for _ in range(1000):
            gen.update_weights(shape, [token], 0.0)
        assert gen.get_weights(shape)[token] >= 0.1

    def test_update_weights_clamped_high(self):
        gen = ExpressionGenerator(seed=42)
        shape = "settled_presence"
        token = ALL_TOKENS[0]
        for _ in range(1000):
            gen.update_weights(shape, [token], 1.0)
        assert gen.get_weights(shape)[token] <= 10.0

    def test_update_unknown_shape_noop(self):
        gen = ExpressionGenerator(seed=42)
        # Should not crash
        gen.update_weights("nonexistent_shape", [ALL_TOKENS[0]], 0.8)

    def test_get_weights_returns_copy(self):
        gen = ExpressionGenerator(seed=42)
        w1 = gen.get_weights("settled_presence")
        w2 = gen.get_weights("settled_presence")
        assert w1 == w2
        w1[ALL_TOKENS[0]] = 999.0
        assert gen.get_weights("settled_presence")[ALL_TOKENS[0]] != 999.0

    def test_get_weights_unknown_shape_empty(self):
        gen = ExpressionGenerator(seed=42)
        assert gen.get_weights("nonexistent") == {}


# ── StudentExpressionGenerator ──

class TestStudentExpressionGenerator:
    def test_fallback_when_no_model_dir(self, tmp_path):
        gen = StudentExpressionGenerator(str(tmp_path / "nonexistent"), fallback_seed=42)
        assert not gen.is_loaded

    def test_fallback_generates(self, tmp_path):
        gen = StudentExpressionGenerator(str(tmp_path / "nonexistent"), fallback_seed=42)
        result = gen.generate("settled_presence")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_fallback_without_window(self, tmp_path):
        gen = StudentExpressionGenerator(str(tmp_path / "nonexistent"), fallback_seed=42)
        result = gen.generate("rising_entropy", window=None)
        assert isinstance(result, list)

    def test_update_weights_delegates_to_fallback(self, tmp_path):
        gen = StudentExpressionGenerator(str(tmp_path / "nonexistent"), fallback_seed=42)
        shape = "settled_presence"
        token = ALL_TOKENS[0]
        original = gen.get_weights(shape)[token]
        gen.update_weights(shape, [token], 1.0)
        assert gen.get_weights(shape)[token] > original

    def test_get_weights_delegates(self, tmp_path):
        gen = StudentExpressionGenerator(str(tmp_path / "nonexistent"), fallback_seed=42)
        w = gen.get_weights("settled_presence")
        assert isinstance(w, dict)
        assert len(w) > 0


# ── translate_expression ──

class TestTranslateExpression:
    def test_known_token(self):
        result = translate_expression(["~warmth~"])
        assert len(result) >= 1
        assert all(t in LUMEN_TOKENS for t in result)

    def test_unknown_token_skipped(self):
        result = translate_expression(["~nonexistent~"])
        assert result == []

    def test_mixed_known_and_unknown(self):
        result = translate_expression(["~nonexistent~", "~warmth~"])
        assert len(result) >= 1

    def test_dedup(self):
        # Both map to tokens that may overlap
        result = translate_expression(["~warmth~", "~reflection~"])
        # Should have unique tokens
        assert len(result) == len(set(result))

    def test_max_three_tokens(self):
        # Use many tokens — should be capped at 3
        many = list(TOKEN_MAP.keys())[:6]
        result = translate_expression(many)
        assert len(result) <= 3

    def test_empty_input(self):
        result = translate_expression([])
        assert result == []


# ── shape_to_lumen_trigger ──

class TestShapeToLumenTrigger:
    def test_known_shapes(self):
        known_shapes = [
            "settled_presence", "rising_entropy", "falling_energy",
            "basin_transition_down", "basin_transition_up",
            "entropy_spike_recovery", "drift_dissonance",
            "void_rising", "convergence",
        ]
        for shape in known_shapes:
            trigger = shape_to_lumen_trigger(shape)
            assert trigger["should_generate"] is True
            assert "reason" in trigger
            assert trigger["token_count_hint"] > 0

    def test_unknown_shape(self):
        trigger = shape_to_lumen_trigger("nonexistent")
        assert trigger["should_generate"] is False
        assert trigger["reason"] == "unknown_shape"
        assert trigger["token_count_hint"] == 0


# ── generate_lumen_expression ──

class TestGenerateLumenExpression:
    def test_returns_required_keys(self):
        result = generate_lumen_expression(
            "settled_presence",
            {"E": 0.5, "I": 0.6, "S": 0.3, "V": 0.1},
        )
        assert "shape" in result
        assert "eisv_tokens" in result
        assert "lumen_tokens" in result
        assert "trigger" in result

    def test_shape_preserved(self):
        result = generate_lumen_expression(
            "rising_entropy",
            {"E": 0.5, "I": 0.6, "S": 0.3, "V": 0.1},
        )
        assert result["shape"] == "rising_entropy"

    def test_with_explicit_generator(self):
        gen = ExpressionGenerator(seed=42)
        result = generate_lumen_expression(
            "convergence",
            {"E": 0.5, "I": 0.6, "S": 0.3, "V": 0.1},
            generator=gen,
        )
        assert isinstance(result["eisv_tokens"], list)
        assert isinstance(result["lumen_tokens"], list)

    def test_lumen_tokens_from_vocabulary(self):
        result = generate_lumen_expression(
            "settled_presence",
            {"E": 0.5, "I": 0.6, "S": 0.3, "V": 0.1},
            generator=ExpressionGenerator(seed=42),
        )
        for token in result["lumen_tokens"]:
            assert token in LUMEN_TOKENS
