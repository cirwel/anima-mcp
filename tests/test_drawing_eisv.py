"""
Tests for Drawing EISV thermodynamics.

Validates that the same EISV equations from governance_core produce meaningful
coherence dynamics in Lumen's drawing system.
"""

import math

from anima_mcp.display.screens import DrawingEISV, DrawingIntent, _EISV_PARAMS


class TestDrawingEISV:
    """Test the DrawingEISV dataclass."""

    def test_initialization(self):
        """Test EISV starts with correct values."""
        eisv = DrawingEISV()
        assert eisv.E == 0.4
        assert eisv.I == 0.2
        assert eisv.S == 0.5
        assert eisv.V == 0.0
        assert eisv.gesture_history == []

    def test_reset(self):
        """Test reset restores initial state."""
        eisv = DrawingEISV()
        eisv.E = 0.1
        eisv.I = 0.9
        eisv.S = 1.5
        eisv.V = 0.8
        eisv.gesture_history = ["dot", "stroke", "curve"]
        eisv.reset()
        assert eisv.E == 0.4
        assert eisv.I == 0.2
        assert eisv.S == 0.5
        assert eisv.V == 0.0
        assert eisv.gesture_history == []

    def test_coherence_at_zero_V(self):
        """Test C(V=0) = 0.5 (midpoint)."""
        eisv = DrawingEISV()
        eisv.V = 0.0
        C = eisv.coherence()
        assert abs(C - 0.5) < 0.01

    def test_coherence_positive_V(self):
        """Test C rises with positive V (high intentionality)."""
        eisv = DrawingEISV()
        eisv.V = 1.0
        C = eisv.coherence()
        assert C > 0.6

    def test_coherence_negative_V(self):
        """Test C falls with negative V (exploratory)."""
        eisv = DrawingEISV()
        eisv.V = -1.0
        C = eisv.coherence()
        assert C < 0.4

    def test_coherence_bounded(self):
        """Test coherence stays in [0, Cmax]."""
        eisv = DrawingEISV()
        for v in [-10.0, -2.0, 0.0, 2.0, 10.0]:
            eisv.V = v
            C = eisv.coherence()
            assert 0.0 <= C <= _EISV_PARAMS["Cmax"]

    def test_coherence_monotonic(self):
        """Test coherence is monotonically increasing with V."""
        eisv = DrawingEISV()
        prev_c = -1.0
        for v in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            eisv.V = v
            C = eisv.coherence()
            assert C > prev_c
            prev_c = C


class TestDrawingIntentEISV:
    """Test EISV integration in DrawingIntent."""

    def test_intent_has_eisv(self):
        """Test DrawingIntent includes EISV state."""
        intent = DrawingIntent()
        assert hasattr(intent, "eisv")
        assert isinstance(intent.eisv, DrawingEISV)

    def test_intent_reset_resets_eisv(self):
        """Test DrawingIntent.reset() also resets EISV."""
        intent = DrawingIntent()
        intent.eisv.E = 0.1
        intent.eisv.V = 0.5
        intent.eisv.gesture_history = ["dot"] * 10
        intent.reset()
        assert intent.eisv.E == 0.4
        assert intent.eisv.V == 0.0
        assert intent.eisv.gesture_history == []


class TestVProgression:
    """Test that V progresses from negative to positive over a drawing's lifetime."""

    def test_V_goes_negative_when_E_exceeds_I(self):
        """Early drawing: E high, I low → V should go negative."""
        eisv = DrawingEISV()
        p = _EISV_PARAMS
        # Simulate 100 steps with low I signal
        for _ in range(100):
            I_signal = 0.1  # Low intentionality (exploring)
            dV = p["kappa"] * (I_signal - eisv.E) - p["delta"] * eisv.V
            eisv.V = max(-2.0, min(2.0, eisv.V + dV * p["dt"]))
        assert eisv.V < 0, f"V should be negative when I < E, got {eisv.V}"

    def test_V_goes_positive_when_I_exceeds_E(self):
        """Late drawing: I high, E low → V should go positive."""
        eisv = DrawingEISV()
        eisv.E = 0.2  # Low energy (late drawing)
        p = _EISV_PARAMS
        for _ in range(100):
            I_signal = 0.7  # High intentionality (committed)
            dV = p["kappa"] * (I_signal - eisv.E) - p["delta"] * eisv.V
            eisv.V = max(-2.0, min(2.0, eisv.V + dV * p["dt"]))
        assert eisv.V > 0, f"V should be positive when I > E, got {eisv.V}"

    def test_coherence_trajectory(self):
        """Test C follows the expected early-low, late-high trajectory."""
        eisv = DrawingEISV()
        p = _EISV_PARAMS

        # Simulate early drawing (E high, I low)
        for _ in range(50):
            dV = p["kappa"] * (0.1 - 0.8) - p["delta"] * eisv.V
            eisv.V = max(-2.0, min(2.0, eisv.V + dV * p["dt"]))
        C_early = eisv.coherence()

        # Simulate late drawing (E low, I high)
        eisv.E = 0.15
        for _ in range(200):
            dV = p["kappa"] * (0.7 - eisv.E) - p["delta"] * eisv.V
            eisv.V = max(-2.0, min(2.0, eisv.V + dV * p["dt"]))
        C_late = eisv.coherence()

        assert C_late > C_early, f"Coherence should rise over drawing: early={C_early:.3f}, late={C_late:.3f}"


class TestBehavioralEntropy:
    """Test behavioral entropy (S signal) computation."""

    def test_uniform_gestures_high_entropy(self):
        """All 5 gestures equally → high entropy."""
        history = ["dot", "stroke", "curve", "cluster", "drag"] * 4
        counts: dict = {}
        for g in history:
            counts[g] = counts.get(g, 0) + 1
        total = len(history)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        normalized = entropy / 2.32
        assert normalized > 0.9, f"Uniform distribution should have high entropy: {normalized}"

    def test_single_gesture_zero_entropy(self):
        """All same gesture → zero entropy."""
        history = ["dot"] * 20
        counts: dict = {}
        for g in history:
            counts[g] = counts.get(g, 0) + 1
        total = len(history)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        assert entropy == 0.0

    def test_two_gestures_moderate_entropy(self):
        """Two gestures → moderate entropy."""
        history = ["dot"] * 10 + ["stroke"] * 10
        counts: dict = {}
        for g in history:
            counts[g] = counts.get(g, 0) + 1
        total = len(history)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        normalized = entropy / 2.32
        assert 0.3 < normalized < 0.6, f"Two gestures should have moderate entropy: {normalized}"


class TestEISVParameters:
    """Test EISV parameter configuration."""

    def test_params_exist(self):
        """Test all required parameters are present."""
        required = [
            "alpha", "beta_E", "gamma_E", "beta_I", "k", "gamma_I",
            "mu", "lambda1", "lambda2", "kappa", "delta", "C1", "Cmax", "dt"
        ]
        for key in required:
            assert key in _EISV_PARAMS, f"Missing parameter: {key}"

    def test_params_positive(self):
        """Test all parameters are positive."""
        for key, value in _EISV_PARAMS.items():
            assert value > 0, f"Parameter {key} should be positive: {value}"

    def test_params_reasonable_scale(self):
        """Test parameters are scaled for drawing (much smaller than governance defaults)."""
        # Governance defaults: alpha=0.42, beta_E=0.1, kappa=0.3, delta=0.4
        assert _EISV_PARAMS["alpha"] < 0.1, "alpha should be << governance (0.42)"
        assert _EISV_PARAMS["kappa"] < 0.1, "kappa should be << governance (0.3)"
        assert _EISV_PARAMS["delta"] < 0.1, "delta should be << governance (0.4)"


class TestEnergyModulation:
    """Test that EISV coupling modulates energy depletion correctly."""

    def test_high_intentionality_slows_depletion(self):
        """When I > E, coupling should counter depletion (positive dE)."""
        eisv = DrawingEISV()
        eisv.E = 0.3
        eisv.I = 0.6  # I > E → α(I-E) positive
        eisv.S = 0.2
        p = _EISV_PARAMS

        dE = p["alpha"] * (eisv.I - eisv.E) - p["beta_E"] * eisv.E * eisv.S + p["gamma_E"] * 0.0
        # α(0.6 - 0.3) = 0.01 * 0.3 = 0.003 (positive)
        # β_E * 0.3 * 0.2 = 0.005 * 0.06 = 0.0003 (negative)
        # Net should be positive (coupling counters depletion)
        assert dE > 0, f"High I should produce positive dE coupling: {dE}"

    def test_low_intentionality_accelerates_depletion(self):
        """When I < E, coupling should add to depletion (negative dE)."""
        eisv = DrawingEISV()
        eisv.E = 0.8
        eisv.I = 0.1  # I < E → α(I-E) negative
        eisv.S = 0.5
        p = _EISV_PARAMS

        dE = p["alpha"] * (eisv.I - eisv.E) - p["beta_E"] * eisv.E * eisv.S + p["gamma_E"] * 0.0
        # α(0.1 - 0.8) = 0.01 * (-0.7) = -0.007 (negative)
        # β_E * 0.8 * 0.5 = 0.005 * 0.4 = -0.002 (also negative)
        # Net should be negative
        assert dE < 0, f"Low I should produce negative dE coupling: {dE}"

    def test_modulation_bounded(self):
        """EISV coupling should be small relative to flat depletion (0.001)."""
        p = _EISV_PARAMS
        # Worst case: maximum parameter values
        max_dE = abs(p["alpha"] * 1.0 + p["beta_E"] * 1.0 + p["gamma_E"] * 1.0) * p["dt"]
        # Should be same order of magnitude as flat depletion (0.001), not 10x
        assert max_dE < 0.01, f"EISV coupling too large: {max_dE} (flat depletion is 0.001)"
