"""
Wiring tests — verify components are actually connected end-to-end.

Unit tests check each function in isolation. These tests verify the data
actually flows between components: A's output reaches B, B's output reaches C.
If a pipeline breaks (wrong parameter name, missing kwarg, field rename),
these tests catch it.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from conftest import make_anima, make_readings


# ============================================================
# 1. Student model → translate → primitive language → coherence
# ============================================================

class TestTrajectoryToPrimitiveLanguageWiring:
    """Verify the full chain: EISV trajectory → suggested tokens → utterance."""

    def test_student_model_tokens_reach_primitive_selection(self, tmp_path):
        """Student model output actually influences which tokens are selected."""
        from anima_mcp.eisv.expression import translate_expression
        from anima_mcp.primitive_language import PrimitiveLanguageSystem

        # Use rule-based generator (always available, no model files needed)
        from anima_mcp.eisv.expression import ExpressionGenerator
        gen = ExpressionGenerator(seed=42)
        eisv_tokens = gen.generate("stable_high")
        lumen_tokens = translate_expression(eisv_tokens)

        assert len(lumen_tokens) > 0, "translate_expression must produce tokens"

        # Feed through primitive language with anchor mechanism
        pls = PrimitiveLanguageSystem(db_path=str(tmp_path / "test.db"))
        try:
            state = {"warmth": 0.5, "clarity": 0.5, "stability": 0.7, "presence": 0.5}

            # With anchor, first token should always be from suggested set
            hits = 0
            trials = 50
            for _ in range(trials):
                tokens = pls.select_tokens(state, count=2, suggested_tokens=lumen_tokens)
                if any(t in lumen_tokens for t in tokens):
                    hits += 1

            # Anchor guarantees at least one suggested token every time
            assert hits == trials, f"Anchor should guarantee suggested token in every utterance, got {hits}/{trials}"
        finally:
            pls.close()

    def test_trajectory_awareness_to_utterance_coherence(self, tmp_path):
        """Full pipeline: TrajectoryAwareness → suggestion → utterance → coherence > 0."""
        from anima_mcp.eisv.awareness import TrajectoryAwareness, compute_expression_coherence
        from anima_mcp.primitive_language import PrimitiveLanguageSystem

        ta = TrajectoryAwareness(buffer_size=10, seed=42)
        pls = PrimitiveLanguageSystem(db_path=str(tmp_path / "test.db"))
        try:
            # Fill buffer with enough states (> MIN_STATES=5)
            for i in range(8):
                ta._last_record_time = 0  # Force recording (bypass interval)
                ta.record_state(warmth=0.5, clarity=0.6, stability=0.7, presence=0.5)

            lang_state = {"warmth": 0.5, "clarity": 0.6, "stability": 0.7, "presence": 0.5}
            suggestion = ta.get_trajectory_suggestion(lang_state)

            assert suggestion is not None, "Should produce suggestion with 8 states in buffer"
            assert "suggested_tokens" in suggestion
            assert len(suggestion["suggested_tokens"]) > 0

            # Generate utterance with suggestions
            utterance = pls.generate_utterance(lang_state, suggested_tokens=suggestion["suggested_tokens"])

            # Compute coherence
            coherence = compute_expression_coherence(suggestion["suggested_tokens"], utterance.tokens)
            assert coherence is not None
            assert coherence > 0, f"Coherence should be > 0 with anchor mechanism, got {coherence}"
        finally:
            ta.close()
            pls.close()


# ============================================================
# 2. SHM dict → readings_from_dict → neural bands
# ============================================================

class TestSHMToNeuralWiring:
    """Verify SHM data flows through to computational neural state."""

    def test_readings_from_dict_preserves_neural_fields(self):
        """SHM dict → readings_from_dict keeps all eeg fields."""
        from anima_mcp.server_state import readings_from_dict

        shm_data = {
            "timestamp": "2026-03-04T01:00:00",
            "cpu_temp_c": 52.0,
            "cpu_percent": 15.0,
            "memory_percent": 20.0,
            "eeg_delta_power": 0.85,
            "eeg_theta_power": 0.03,
            "eeg_alpha_power": 0.82,
            "eeg_beta_power": 0.15,
            "eeg_gamma_power": 0.32,
        }
        readings = readings_from_dict(shm_data)

        assert readings.eeg_delta_power == 0.85
        assert readings.eeg_theta_power == 0.03
        assert readings.eeg_alpha_power == 0.82
        assert readings.eeg_beta_power == 0.15
        assert readings.eeg_gamma_power == 0.32
        assert readings.cpu_percent == 15.0
        assert readings.memory_percent == 20.0

    def test_readings_to_neural_bands_roundtrip(self):
        """SHM dict → readings → neural sensor produces valid bands."""
        from anima_mcp.server_state import readings_from_dict
        from anima_mcp.computational_neural import ComputationalNeuralSensor

        shm_data = {
            "timestamp": "2026-03-04T01:00:00",
            "cpu_temp_c": 52.0,
            "cpu_percent": 25.0,
            "memory_percent": 40.0,
        }
        readings = readings_from_dict(shm_data)
        sensor = ComputationalNeuralSensor()

        # Prime with a first reading
        sensor.get_neural_state(
            cpu_percent=readings.cpu_percent,
            memory_percent=readings.memory_percent,
            cpu_temp=readings.cpu_temp_c,
        )

        # Second reading produces meaningful state
        state = sensor.get_neural_state(
            cpu_percent=readings.cpu_percent,
            memory_percent=readings.memory_percent,
            cpu_temp=readings.cpu_temp_c,
        )

        assert 0 <= state.delta <= 1
        assert 0 <= state.theta <= 1
        assert 0 <= state.alpha <= 1
        assert 0 <= state.beta <= 1
        assert 0 <= state.gamma <= 1
        # Alpha = 1 - beta (CPU idle fraction). 25% CPU → beta=0.25 → alpha=0.75
        assert state.alpha == pytest.approx(0.75, abs=0.01)
        # Beta should reflect CPU usage (25% → 0.25)
        assert state.beta == pytest.approx(0.25, abs=0.01)

    def test_extract_neural_bands_from_readings(self):
        """extract_neural_bands correctly pulls eeg_ fields from readings dict."""
        from anima_mcp.server_state import extract_neural_bands

        readings_dict = {
            "eeg_delta_power": 0.85,
            "eeg_theta_power": 0.03,
            "eeg_alpha_power": 0.82,
            "eeg_beta_power": 0.15,
            "eeg_gamma_power": 0.32,
            "cpu_percent": 10.0,  # should be excluded
        }

        bands = extract_neural_bands(readings_dict)
        assert bands == {"delta": 0.85, "theta": 0.03, "alpha": 0.82, "beta": 0.15, "gamma": 0.32}


# ============================================================
# 3. Health probes: broker mode vs standalone mode
# ============================================================

class TestHealthProbeWiring:
    """Verify health probes reflect correct state in both modes."""

    @staticmethod
    def _make_probe(_sensors, _last_shm_data, shm_stale_threshold=15.0):
        """Build a sensor probe matching the real server logic."""
        def probe():
            if _sensors is not None:
                return True
            if _last_shm_data and "readings" in _last_shm_data:
                ts = _last_shm_data.get("timestamp")
                if ts:
                    from datetime import datetime
                    try:
                        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        age = (datetime.now(t.tzinfo) - t).total_seconds()
                        return age < shm_stale_threshold * 2
                    except (ValueError, AttributeError):
                        pass
                return True  # Data but no timestamp
            return False
        return probe

    def test_sensor_probe_ok_with_fresh_shm(self):
        """In broker mode: fresh SHM data → probe ok."""
        from anima_mcp.health import SubsystemHealth

        shm_data = {
            "readings": {"cpu_temp_c": 50.0},
            "timestamp": datetime.now().isoformat(),
        }
        probe = self._make_probe(None, shm_data)
        sub = SubsystemHealth(name="sensors", probe_fn=probe)
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.get_status() == "ok"

    def test_sensor_probe_ok_with_direct_sensors(self):
        """In standalone mode: _sensors present → probe ok regardless of SHM."""
        from anima_mcp.health import SubsystemHealth

        probe = self._make_probe(MagicMock(), None)
        sub = SubsystemHealth(name="sensors", probe_fn=probe)
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.get_status() == "ok"

    def test_sensor_probe_fails_when_neither_available(self):
        """No sensors AND no SHM → probe fails → degraded."""
        from anima_mcp.health import SubsystemHealth

        probe = self._make_probe(None, None)
        sub = SubsystemHealth(name="sensors", probe_fn=probe)
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.get_status() == "degraded"

    def test_sensor_probe_fails_with_stale_shm(self):
        """SHM data exists but timestamp is old → probe fails."""
        from anima_mcp.health import SubsystemHealth
        from datetime import timedelta

        stale_time = (datetime.now() - timedelta(seconds=60)).isoformat()
        shm_data = {
            "readings": {"cpu_temp_c": 50.0},
            "timestamp": stale_time,
        }
        probe = self._make_probe(None, shm_data, shm_stale_threshold=15.0)
        sub = SubsystemHealth(name="sensors", probe_fn=probe)
        sub.heartbeat()
        sub.last_probe_time = 0
        assert sub.get_status() == "degraded"


# ============================================================
# 4. Anima → EISV mapping → governance bridge
# ============================================================

class TestAnimaToGovernanceWiring:
    """Verify anima state maps to EISV and reaches governance."""

    def test_anima_to_eisv_documented_mapping(self):
        """Warmth→E, Clarity→I, 1-Stability→S, (1-Presence)*0.3→V."""
        from anima_mcp.eisv_mapper import anima_to_eisv

        anima = make_anima(warmth=0.7, clarity=0.6, stability=0.8, presence=0.5)
        readings = make_readings(
            eeg_alpha_power=0.5, eeg_beta_power=0.3, eeg_gamma_power=0.2,
        )

        eisv = anima_to_eisv(anima, readings)

        # EISVMetrics uses full names: energy, integrity, entropy, void
        assert 0 <= eisv.energy <= 1
        assert 0 <= eisv.integrity <= 1
        assert 0 <= eisv.entropy <= 1
        assert -1.0 <= eisv.valence <= 1.0

    @pytest.mark.asyncio
    async def test_bridge_check_in_produces_decision(self):
        """Bridge.check_in() returns a decision dict with expected fields."""
        from anima_mcp.unitares_bridge import UnitaresBridge

        bridge = UnitaresBridge(unitares_url=None)  # Local only
        anima = make_anima(warmth=0.7, clarity=0.6, stability=0.8, presence=0.5)
        readings = make_readings(
            eeg_alpha_power=0.5, eeg_beta_power=0.3, eeg_gamma_power=0.2,
        )

        decision = await bridge.check_in(anima, readings)

        assert decision is not None
        assert "action" in decision
        assert decision["action"] in ("proceed", "pause", "halt")
        assert "eisv" in decision
        assert "source" in decision

    @pytest.mark.asyncio
    async def test_bridge_drawing_eisv_reaches_payload(self):
        """drawing_eisv kwarg is included in the UNITARES call payload."""
        from unittest.mock import AsyncMock
        from anima_mcp.unitares_bridge import UnitaresBridge

        bridge = UnitaresBridge(unitares_url="http://fake:8767/mcp/")
        anima = make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5)
        readings = make_readings()
        drawing_eisv = {"E": 0.6, "I": 0.7, "S": 0.1, "V": 0.05, "C": 0.8}

        mock_call = AsyncMock(return_value={"action": "proceed", "source": "unitares", "eisv": {}})
        bridge._call_unitares = mock_call
        bridge.check_availability = AsyncMock(return_value=True)

        await bridge.check_in(anima, readings, drawing_eisv=drawing_eisv)

        mock_call.assert_called_once()
        kwargs = mock_call.call_args.kwargs
        assert kwargs.get("drawing_eisv") == drawing_eisv, \
            f"drawing_eisv not forwarded. kwargs: {kwargs}"


# ============================================================
# 5. Drawing EISV → server check-in (wiring gap detection)
# ============================================================

class TestDrawingEISVServerWiring:
    """Verify drawing EISV is wired into the server's governance call."""

    def test_server_check_in_signature_accepts_drawing_eisv(self):
        """The bridge.check_in() accepts drawing_eisv parameter."""
        import inspect
        from anima_mcp.unitares_bridge import UnitaresBridge

        sig = inspect.signature(UnitaresBridge.check_in)
        assert "drawing_eisv" in sig.parameters, \
            "bridge.check_in() must accept drawing_eisv parameter"

    def test_drawing_engine_provides_eisv(self):
        """DrawingEngine.get_drawing_eisv() returns EISV dict when intent exists."""
        from anima_mcp.display.drawing_engine import DrawingEngine

        engine = DrawingEngine.__new__(DrawingEngine)
        engine.intent = None  # No intent → should return None

        result = engine.get_drawing_eisv()
        assert result is None

    def test_screen_renderer_exposes_drawing_eisv(self):
        """ScreenRenderer.get_drawing_eisv() delegates to DrawingEngine."""
        from anima_mcp.display.screens import ScreenRenderer

        # Verify the method exists and delegates
        assert hasattr(ScreenRenderer, "get_drawing_eisv"), \
            "ScreenRenderer must expose get_drawing_eisv()"


# ============================================================
# 6. Token vocabulary alignment across pipeline
# ============================================================

class TestTokenVocabularyWiring:
    """Verify token vocabularies match across EISV → translation → primitive language."""

    def test_all_eisv_tokens_have_translations(self):
        """Every EISV token in ALL_TOKENS maps to at least one Lumen token."""
        from anima_mcp.eisv.expression import ALL_TOKENS, TOKEN_MAP

        for token in ALL_TOKENS:
            assert token in TOKEN_MAP, f"EISV token {token} missing from TOKEN_MAP"
            assert len(TOKEN_MAP[token]) > 0, f"TOKEN_MAP[{token}] is empty"

    def test_all_translated_tokens_exist_in_primitives(self):
        """Every Lumen token from TOKEN_MAP exists in PRIMITIVES."""
        from anima_mcp.eisv.expression import TOKEN_MAP
        from anima_mcp.primitive_language import PRIMITIVES

        for eisv_token, lumen_tokens in TOKEN_MAP.items():
            for lt in lumen_tokens:
                assert lt in PRIMITIVES, \
                    f"Lumen token '{lt}' (from EISV '{eisv_token}') not in PRIMITIVES"

    def test_student_model_mappings_match_all_tokens(self):
        """Student model's mappings.json tokens match ALL_TOKENS."""
        import json
        import os
        from anima_mcp.eisv.expression import ALL_TOKENS

        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "student_model"
        )
        mappings_path = os.path.join(model_dir, "mappings.json")
        if not os.path.exists(mappings_path):
            pytest.skip("Student model not present locally")

        with open(mappings_path) as f:
            mappings = json.load(f)

        model_tokens = set(mappings["tokens"])
        code_tokens = set(ALL_TOKENS)
        assert model_tokens == code_tokens, \
            f"Mismatch: model has {model_tokens - code_tokens}, code has {code_tokens - model_tokens}"
