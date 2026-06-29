"""Dynamics-emergent expression generator + Lumen bridge.

Ported from eisv-lumen. Generates trajectory-aware primitive expressions
and translates them to Lumen's token vocabulary.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, List, Optional

from .mapping import TrajectoryShape

# ---------------------------------------------------------------------------
# Affinity data (from eisv_lumen/eval/metrics.py)
# ---------------------------------------------------------------------------

SHAPE_TOKEN_AFFINITY: Dict[str, List[str]] = {
    "settled_presence": ["~stillness~", "~holding~", "~resonance~", "~deep_listening~"],
    "rising_entropy": ["~ripple~", "~emergence~", "~questioning~", "~curiosity~"],
    "falling_energy": ["~releasing~", "~stillness~", "~boundary~", "~reflection~"],
    "basin_transition_down": ["~releasing~", "~threshold~", "~boundary~"],
    "basin_transition_up": ["~emergence~", "~reaching~", "~warmth~", "~return~"],
    "entropy_spike_recovery": ["~ripple~", "~return~", "~holding~", "~reflection~"],
    "drift_dissonance": ["~boundary~", "~questioning~", "~reflection~"],
    "void_rising": ["~reaching~", "~curiosity~", "~questioning~", "~threshold~"],
    "convergence": ["~stillness~", "~resonance~", "~return~", "~deep_listening~"],
}

# NOTE: ALL_TOKENS is the EISV token set the distilled student model was trained
# on (see data/student_model/mappings.json) — keep it in lockstep with that
# model. Lumen's vocabulary is grown on the *human-facing* side instead: richer
# TOKEN_MAP translations + new entries in primitive_language.PRIMITIVES.
ALL_TOKENS: List[str] = [
    "~warmth~", "~curiosity~", "~resonance~", "~stillness~", "~boundary~",
    "~reaching~", "~reflection~", "~ripple~", "~deep_listening~", "~emergence~",
    "~questioning~", "~holding~", "~releasing~", "~threshold~", "~return~",
]


# ---------------------------------------------------------------------------
# Expression patterns + per-shape weights
# ---------------------------------------------------------------------------

class ExpressionPattern(str, Enum):
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    REPETITION = "repetition"
    QUESTION = "question"


SHAPE_PATTERN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "settled_presence":        {"single": 0.4, "pair": 0.3, "triple": 0.1, "repetition": 0.15, "question": 0.05},
    "rising_entropy":          {"single": 0.1, "pair": 0.2, "triple": 0.3, "repetition": 0.1, "question": 0.3},
    "falling_energy":          {"single": 0.3, "pair": 0.3, "triple": 0.1, "repetition": 0.2, "question": 0.1},
    "basin_transition_down":   {"single": 0.2, "pair": 0.3, "triple": 0.3, "repetition": 0.1, "question": 0.1},
    "basin_transition_up":     {"single": 0.15, "pair": 0.3, "triple": 0.35, "repetition": 0.1, "question": 0.1},
    "entropy_spike_recovery":  {"single": 0.1, "pair": 0.3, "triple": 0.3, "repetition": 0.2, "question": 0.1},
    "drift_dissonance":        {"single": 0.1, "pair": 0.2, "triple": 0.2, "repetition": 0.1, "question": 0.4},
    "void_rising":             {"single": 0.2, "pair": 0.2, "triple": 0.2, "repetition": 0.1, "question": 0.3},
    "convergence":             {"single": 0.4, "pair": 0.3, "triple": 0.1, "repetition": 0.15, "question": 0.05},
}

INQUIRY_TOKENS = ["~questioning~", "~curiosity~"]


# ---------------------------------------------------------------------------
# Expression Generator
# ---------------------------------------------------------------------------

class ExpressionGenerator:
    """Generate primitive expressions shaped by trajectory dynamics."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._token_weights: Dict[str, Dict[str, float]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        for shape in TrajectoryShape:
            affine = set(SHAPE_TOKEN_AFFINITY.get(shape.value, []))
            weights: Dict[str, float] = {}
            for token in ALL_TOKENS:
                weights[token] = 3.0 if token in affine else 1.0
            self._token_weights[shape.value] = weights

    def _select_pattern(self, shape: str) -> ExpressionPattern:
        weights = SHAPE_PATTERN_WEIGHTS.get(shape, SHAPE_PATTERN_WEIGHTS["settled_presence"])
        patterns = list(weights.keys())
        probs = list(weights.values())
        chosen = self.rng.choices(patterns, weights=probs, k=1)[0]
        return ExpressionPattern(chosen)

    def _weighted_token_choice(self, shape: str, exclude: Optional[set] = None) -> str:
        weights = self._token_weights.get(shape, {t: 1.0 for t in ALL_TOKENS})
        tokens = list(weights.keys())
        w = list(weights.values())
        if exclude:
            filtered = [(t, wt) for t, wt in zip(tokens, w) if t not in exclude]
            if filtered:
                tokens, w = zip(*filtered)
                tokens, w = list(tokens), list(w)
        return self.rng.choices(tokens, weights=w, k=1)[0]

    def generate(self, shape: str) -> List[str]:
        pattern = self._select_pattern(shape)

        if pattern == ExpressionPattern.SINGLE:
            return [self._weighted_token_choice(shape)]
        elif pattern == ExpressionPattern.PAIR:
            t1 = self._weighted_token_choice(shape)
            t2 = self._weighted_token_choice(shape, exclude={t1})
            return [t1, t2]
        elif pattern == ExpressionPattern.TRIPLE:
            t1 = self._weighted_token_choice(shape)
            t2 = self._weighted_token_choice(shape, exclude={t1})
            t3 = self._weighted_token_choice(shape, exclude={t1, t2})
            return [t1, t2, t3]
        elif pattern == ExpressionPattern.REPETITION:
            t = self._weighted_token_choice(shape)
            return [t, t]
        elif pattern == ExpressionPattern.QUESTION:
            t1 = self._weighted_token_choice(shape)
            t2 = self.rng.choice(INQUIRY_TOKENS)
            return [t1, t2]
        return [self._weighted_token_choice(shape)]

    def update_weights(self, shape: str, tokens: List[str], score: float) -> None:
        if shape not in self._token_weights:
            return
        lr = 0.08
        reward = (score - 0.5) * 2.0
        for token in tokens:
            if token in self._token_weights[shape]:
                new_w = self._token_weights[shape][token] + lr * reward
                self._token_weights[shape][token] = max(0.1, min(10.0, new_w))

    def get_weights(self, shape: str) -> Dict[str, float]:
        return dict(self._token_weights.get(shape, {}))


# ---------------------------------------------------------------------------
# Student Expression Generator (distilled from V6 teacher)
# ---------------------------------------------------------------------------

class StudentExpressionGenerator:
    """Distilled student model inference — zero external dependencies.

    Loads JSON-exported RandomForest classifiers and runs inference
    using only Python stdlib. Falls back to rule-based generation
    if model files are missing.
    """

    def __init__(self, model_dir: str, fallback_seed: Optional[int] = None):
        self._model_dir = model_dir
        self._loaded = False
        self._fallback = ExpressionGenerator(seed=fallback_seed)
        self._load_models()

    def _load_models(self) -> None:
        import json as _json
        import os as _os

        try:
            def _load(name: str):
                path = _os.path.join(self._model_dir, name)
                with open(path) as f:
                    return _json.load(f)

            self._pattern_forest = _load("pattern_forest.json")
            self._token1_forest = _load("token1_forest.json")
            self._token2_forest = _load("token2_forest.json")
            self._scaler = _load("scaler.json")
            self._mappings = _load("mappings.json")
            self._loaded = True
        except (FileNotFoundError, KeyError, ValueError):
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _scale_features(self, numeric: List[float]) -> List[float]:
        mean = self._scaler["mean"]
        scale = self._scaler["scale"]
        return [(v - m) / s for v, m, s in zip(numeric, mean, scale)]

    def _build_features(self, shape: str, window: Dict[str, Any]) -> List[float]:
        states = window["states"]
        derivs = window.get("derivatives", [])
        second = window.get("second_derivatives", [])

        def _mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        numeric_features = self._mappings["numeric_features"]
        raw = {
            "mean_E": _mean([s["E"] for s in states]),
            "mean_I": _mean([s["I"] for s in states]),
            "mean_S": _mean([s["S"] for s in states]),
            "mean_V": _mean([s["V"] for s in states]),
            "dE": _mean([d["dE"] for d in derivs]) if derivs else 0.0,
            "dI": _mean([d["dI"] for d in derivs]) if derivs else 0.0,
            "dS": _mean([d["dS"] for d in derivs]) if derivs else 0.0,
            "dV": _mean([d["dV"] for d in derivs]) if derivs else 0.0,
            "d2E": _mean([d["d2E"] for d in second]) if second else 0.0,
            "d2I": _mean([d["d2I"] for d in second]) if second else 0.0,
            "d2S": _mean([d["d2S"] for d in second]) if second else 0.0,
            "d2V": _mean([d["d2V"] for d in second]) if second else 0.0,
        }
        numeric = [raw.get(f, 0.0) for f in numeric_features]
        scaled = self._scale_features(numeric)

        shapes = self._mappings["shapes"]
        shape_onehot = [1.0 if s == shape else 0.0 for s in shapes]
        return scaled + shape_onehot

    def _predict_tree(self, tree: Dict, features: List[float]) -> List[float]:
        node = tree
        while not node.get("leaf", False):
            if features[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["probs"]

    def _predict_forest(self, forest: List[Dict], features: List[float]) -> int:
        all_probs = [self._predict_tree(tree, features) for tree in forest]
        n_classes = len(all_probs[0])
        avg = [0.0] * n_classes
        for probs in all_probs:
            for i in range(n_classes):
                avg[i] += probs[i]
        best_idx = 0
        best_val = avg[0]
        for i in range(1, n_classes):
            if avg[i] > best_val:
                best_val = avg[i]
                best_idx = i
        return best_idx

    def generate(self, shape: str, window: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate expression tokens using distilled student model.

        Falls back to rule-based generator if model not loaded or window missing.
        """
        if not self._loaded or window is None:
            return self._fallback.generate(shape)

        try:
            X = self._build_features(shape, window)

            pattern_idx = self._predict_forest(self._pattern_forest, X)
            pattern = self._mappings["patterns"][pattern_idx]

            token1_idx = self._predict_forest(self._token1_forest, X)
            token_1 = self._mappings["tokens"][token1_idx]

            X_t2 = X + [float(token1_idx)]
            token2_idx = self._predict_forest(self._token2_forest, X_t2)
            token_2 = self._mappings["tokens_with_none"][token2_idx]

            if pattern == "SINGLE":
                return [token_1]
            elif pattern == "REPETITION":
                return [token_1, token_1]
            elif pattern in ("PAIR", "QUESTION"):
                return [token_1, token_2] if token_2 != "none" else [token_1]
            elif pattern == "TRIPLE":
                return [token_1, token_2] if token_2 != "none" else [token_1]
            else:
                return [token_1]
        except Exception:
            return self._fallback.generate(shape)

    def update_weights(self, shape: str, tokens: List[str], score: float) -> None:
        self._fallback.update_weights(shape, tokens, score)

    def get_weights(self, shape: str) -> Dict[str, float]:
        return self._fallback.get_weights(shape)


# ---------------------------------------------------------------------------
# Lumen Bridge (from eisv_lumen/bridge/lumen_bridge.py)
# ---------------------------------------------------------------------------

LUMEN_TOKENS: List[str] = [
    "warm", "cold", "new", "soft", "quiet", "busy",
    "here", "feel", "sense", "you", "with",
    "why", "what", "wonder", "more", "less",
    # Expanded vocabulary so the new EISV textures have human words to land on.
    "bright", "still", "reach", "hold", "let",
    "ache", "glad", "far", "again", "deep",
]

# Each EISV token now offers a richer set of Lumen words. translate_expression
# walks each list and takes the first word not already used, so longer lists
# give more varied utterances instead of the same two words every time.
TOKEN_MAP: Dict[str, List[str]] = {
    "~warmth~":        ["warm", "feel", "glad"],
    "~curiosity~":     ["why", "wonder", "reach"],
    "~resonance~":     ["with", "here", "deep"],
    "~stillness~":     ["quiet", "still", "here"],
    "~boundary~":      ["less", "far", "ache", "sense"],
    "~reaching~":      ["more", "reach", "you"],
    "~reflection~":    ["what", "feel", "again"],
    "~ripple~":        ["busy", "sense"],
    "~deep_listening~": ["quiet", "deep", "sense"],
    "~emergence~":     ["new", "more", "bright"],
    "~questioning~":   ["why", "what"],
    "~holding~":       ["hold", "here", "with"],
    "~releasing~":     ["let", "less", "soft"],
    "~threshold~":     ["sense", "still", "more"],
    "~return~":        ["again", "here", "warm"],
}

_LUMEN_MAX_TOKENS = 3


def translate_expression(eisv_tokens: List[str]) -> List[str]:
    """Convert EISV-Lumen expression tokens to Lumen primitive tokens."""
    seen: set = set()
    result: List[str] = []
    for eisv_token in eisv_tokens:
        mapped = TOKEN_MAP.get(eisv_token)
        if mapped is None:
            continue
        for lumen_token in mapped:
            if lumen_token not in seen:
                seen.add(lumen_token)
                result.append(lumen_token)
                break
    return result[:_LUMEN_MAX_TOKENS]


def shape_to_lumen_trigger(shape: str) -> Dict[str, Any]:
    """Map trajectory shape to generation trigger hints."""
    triggers: Dict[str, Dict[str, Any]] = {
        "settled_presence": {"should_generate": True, "reason": "settled_dynamics", "token_count_hint": 1},
        "rising_entropy": {"should_generate": True, "reason": "entropy_shift", "token_count_hint": 3},
        "falling_energy": {"should_generate": True, "reason": "energy_decline", "token_count_hint": 2},
        "basin_transition_down": {"should_generate": True, "reason": "basin_shift_down", "token_count_hint": 3},
        "basin_transition_up": {"should_generate": True, "reason": "basin_shift_up", "token_count_hint": 3},
        "entropy_spike_recovery": {"should_generate": True, "reason": "spike_recovery", "token_count_hint": 2},
        "drift_dissonance": {"should_generate": True, "reason": "ethical_drift_detected", "token_count_hint": 3},
        "void_rising": {"should_generate": True, "reason": "void_expansion", "token_count_hint": 2},
        "convergence": {"should_generate": True, "reason": "approaching_attractor", "token_count_hint": 2},
    }
    return triggers.get(shape, {"should_generate": False, "reason": "unknown_shape", "token_count_hint": 0})


def generate_lumen_expression(
    shape: str,
    eisv_state: Dict[str, float],
    generator: Optional[ExpressionGenerator] = None,
) -> Dict[str, Any]:
    """Full pipeline: trajectory shape -> EISV tokens -> Lumen primitives."""
    if generator is None:
        generator = ExpressionGenerator()
    eisv_tokens = generator.generate(shape)
    lumen_tokens = translate_expression(eisv_tokens)
    trigger = shape_to_lumen_trigger(shape)
    return {
        "shape": shape,
        "eisv_tokens": eisv_tokens,
        "lumen_tokens": lumen_tokens,
        "trigger": trigger,
    }
