"""
Tests for loop_phases.py — main loop phase helpers.

Covers:
  - server_governance_fallback(): bridge None, bridge success, bridge error
  - parse_shm_governance_freshness(): timestamp parsing, staleness, source detection
  - compute_lagged_correlations(): empty ctx, insufficient data, Pearson correlation
  - generate_learned_question(): insight-based, belief-based, deduplication
  - compose_grounded_observation(): surprise, advocate desire, messages, dreaming, novelty, fallback
  - grounded_self_answer(): keyword matching, evidence ranking, current state
  - lumen_self_answer(): filtering by age, deduplication
  - extract_and_validate_schema(): schema composition pipeline
  - self_reflect(): reflection gating and observation posting
"""

import time as _time
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from anima_mcp.server_context import ServerContext
from anima_mcp import ctx_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_ctx():
    """Ensure ctx_ref is clean before and after each test."""
    ctx_ref._ctx = None
    yield
    ctx_ref._ctx = None


def make_ctx(**overrides):
    """Create a ServerContext and install it in ctx_ref."""
    ctx = ServerContext(**overrides)
    ctx_ref._ctx = ctx
    return ctx


def make_anima(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5):
    """Create a mock Anima."""
    return SimpleNamespace(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        is_anticipating=False,
        anticipation=None,
    )


def make_readings():
    """Create a mock SensorReadings."""
    return SimpleNamespace(
        cpu_temp_c=55.0,
        light_lux=300.0,
        to_dict=lambda: {"cpu_temp_c": 55.0, "light_lux": 300.0},
    )


def make_identity():
    """Create a mock CreatureIdentity."""
    return SimpleNamespace(
        creature_id="test-id-1234",
        name="Lumen",
        total_awakenings=5,
        total_alive_seconds=3600.0,
        born_at=datetime(2025, 1, 1),
    )


# ---------------------------------------------------------------------------
# server_governance_fallback
# ---------------------------------------------------------------------------

class TestServerGovernanceFallback:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_bridge(self):
        """Fallback returns None when bridge is None."""
        from anima_mcp.loop_phases import server_governance_fallback

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None):
            result = await server_governance_fallback(make_anima(), make_readings())

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_bridge_check_in(self):
        """Fallback calls bridge.check_in with anima and readings."""
        from anima_mcp.loop_phases import server_governance_fallback

        bridge = AsyncMock()
        bridge.check_in.return_value = {"source": "unitares", "verdict": "ok"}
        ctx = make_ctx()
        store = MagicMock()
        store.get_identity.return_value = make_identity()

        with patch("anima_mcp.accessors._get_server_bridge", return_value=bridge), \
             patch("anima_mcp.accessors._get_store", return_value=store):
            result = await server_governance_fallback(make_anima(), make_readings())

        assert result["verdict"] == "ok"
        bridge.check_in.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_bridge_error(self):
        """Fallback returns None when bridge.check_in raises."""
        from anima_mcp.loop_phases import server_governance_fallback

        bridge = AsyncMock()
        bridge.check_in.side_effect = RuntimeError("connection refused")
        ctx = make_ctx()
        store = MagicMock()
        store.get_identity.return_value = make_identity()

        with patch("anima_mcp.accessors._get_server_bridge", return_value=bridge), \
             patch("anima_mcp.accessors._get_store", return_value=store):
            result = await server_governance_fallback(make_anima(), make_readings())

        assert result is None

    @pytest.mark.asyncio
    async def test_includes_drawing_eisv_when_available(self):
        """Fallback passes drawing EISV from screen renderer."""
        from anima_mcp.loop_phases import server_governance_fallback

        bridge = AsyncMock()
        bridge.check_in.return_value = {"source": "unitares"}
        ctx = make_ctx()
        renderer = MagicMock()
        renderer.get_drawing_eisv.return_value = {"E": 0.5, "I": 0.3}
        ctx.screen_renderer = renderer
        store = MagicMock()
        store.get_identity.return_value = make_identity()

        with patch("anima_mcp.accessors._get_server_bridge", return_value=bridge), \
             patch("anima_mcp.accessors._get_store", return_value=store):
            await server_governance_fallback(make_anima(), make_readings())

        call_kwargs = bridge.check_in.call_args
        assert call_kwargs.kwargs.get("drawing_eisv") == {"E": 0.5, "I": 0.3}


# ---------------------------------------------------------------------------
# parse_shm_governance_freshness
# ---------------------------------------------------------------------------

class TestParseShmGovernanceFreshness:
    def test_non_dict_returns_false(self):
        """Non-dict input returns (False, False, None)."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        assert parse_shm_governance_freshness(None) == (False, False, None)
        assert parse_shm_governance_freshness("not a dict") == (False, False, None)
        assert parse_shm_governance_freshness([]) == (False, False, None)

    def test_missing_governance_at(self):
        """Dict without governance_at returns (False, False, None)."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness
        assert parse_shm_governance_freshness({}) == (False, False, None)

    def test_fresh_unitares_source(self):
        """Recent governance_at with unitares source returns (True, True, ts)."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        now = datetime.now()
        gov = {
            "governance_at": now.isoformat(),
            "source": "unitares",
        }
        is_fresh, is_unitares, ts = parse_shm_governance_freshness(gov, now_ts=now.timestamp())

        assert is_fresh is True
        assert is_unitares is True
        assert ts is not None

    def test_stale_governance(self):
        """Old governance_at is detected as stale."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        old = datetime.now() - timedelta(seconds=300)  # 5 min ago, > 210s threshold
        gov = {
            "governance_at": old.isoformat(),
            "source": "unitares",
        }
        now_ts = _time.time()
        is_fresh, is_unitares, ts = parse_shm_governance_freshness(gov, now_ts=now_ts)

        assert is_fresh is False
        assert is_unitares is True

    def test_local_source_not_unitares(self):
        """Local source is not detected as unitares."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        now = datetime.now()
        gov = {
            "governance_at": now.isoformat(),
            "source": "local",
        }
        is_fresh, is_unitares, ts = parse_shm_governance_freshness(gov, now_ts=now.timestamp())

        assert is_unitares is False

    def test_invalid_timestamp_returns_false(self):
        """Invalid timestamp string returns (False, False, None)."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        gov = {"governance_at": "not-a-date"}
        assert parse_shm_governance_freshness(gov) == (False, False, None)

    def test_none_governance_at(self):
        """None governance_at returns (False, False, None)."""
        from anima_mcp.loop_phases import parse_shm_governance_freshness

        gov = {"governance_at": None}
        assert parse_shm_governance_freshness(gov) == (False, False, None)


# ---------------------------------------------------------------------------
# compute_lagged_correlations
# ---------------------------------------------------------------------------

class TestComputeLaggedCorrelations:
    def test_returns_empty_when_no_ctx(self):
        """compute_lagged_correlations returns {} when ctx is None."""
        from anima_mcp.loop_phases import compute_lagged_correlations

        ctx_ref._ctx = None
        result = compute_lagged_correlations()
        assert result == {}

    def test_returns_zeros_with_insufficient_data(self):
        """Returns 0.0 for all dimensions with too little data."""
        from anima_mcp.loop_phases import compute_lagged_correlations

        ctx = make_ctx()
        ctx.health_history = deque([0.5] * 5)
        ctx.satisfaction_per_dim = {
            "warmth": deque([0.5] * 5),
            "clarity": deque([0.5] * 5),
            "stability": deque([0.5] * 5),
            "presence": deque([0.5] * 5),
        }
        result = compute_lagged_correlations()
        assert all(v == 0.0 for v in result.values())

    def test_computes_correlation_with_sufficient_data(self):
        """With enough data, returns non-zero correlations."""
        from anima_mcp.loop_phases import compute_lagged_correlations

        ctx = make_ctx()
        n = 100
        ctx.health_history = deque([0.5 + 0.01 * i for i in range(n)], maxlen=100)

        # Create satisfaction data that correlates with future health
        for dim in ("warmth", "clarity", "stability", "presence"):
            ctx.satisfaction_per_dim[dim] = deque([0.5 + 0.01 * i for i in range(n)], maxlen=500)

        result = compute_lagged_correlations()

        # With positive correlation between satisfaction and health, expect positive values
        for dim in ("warmth", "clarity", "stability", "presence"):
            assert isinstance(result[dim], float)
            # Monotonically increasing series should have positive correlation
            assert result[dim] > 0

    def test_handles_zero_variance(self):
        """Returns 0.0 when variance is zero (constant data)."""
        from anima_mcp.loop_phases import compute_lagged_correlations

        ctx = make_ctx()
        n = 100
        ctx.health_history = deque([0.5] * n, maxlen=100)
        for dim in ("warmth", "clarity", "stability", "presence"):
            ctx.satisfaction_per_dim[dim] = deque([0.5] * n, maxlen=500)

        result = compute_lagged_correlations()
        assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# generate_learned_question
# ---------------------------------------------------------------------------

class TestGenerateLearnedQuestion:
    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_generates_question_from_insights(self, mock_refl, mock_recent):
        """Generates question from self-reflection insight with 'when' keyword."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(confidence=0.8, description="I feel calm when it is dark")
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result is not None
        assert "affect me" in result

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_generates_question_from_tend_insight(self, mock_refl, mock_recent):
        """Generates question from insight containing 'tend'."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(confidence=0.8, description="I tend to prefer dim lighting")
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result is not None
        assert "matters" in result

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_generates_clean_question_from_tends_insight(self, mock_refl, mock_recent):
        """Handles 'tends to' without producing malformed 's to ...' questions."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(confidence=0.8, description="Stability tends to be best in the evening")
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result == "what matters about stability being best in the evening?"

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_generates_from_uncertain_belief(self, mock_refl, mock_recent):
        """Generates question from belief with medium confidence (0.3-0.5)."""
        from anima_mcp.loop_phases import generate_learned_question

        mock_refl.return_value.get_insights.return_value = []
        belief = SimpleNamespace(confidence=0.4, description="Light affects my warmth")

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {"b1": belief}
            result = generate_learned_question()

        assert result is not None
        assert "light affects my warmth" in result.lower()

    @patch("anima_mcp.messages.get_recent_questions")
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_deduplicates_against_recent(self, mock_refl, mock_recent):
        """Skips questions that were recently asked."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(confidence=0.8, description="I feel calm when it is dark")
        mock_refl.return_value.get_insights.return_value = [insight]
        # The exact question text that would be generated
        mock_recent.return_value = [{"text": "why does it is dark affect me?"}]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        # All candidates match recent, so returns None
        assert result is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_skips_low_confidence_insights(self, mock_refl, mock_recent):
        """Skips insights with confidence below 0.5."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(confidence=0.2, description="Something vague")
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_skips_qa_synced_insights(self, mock_refl, mock_recent):
        """Skips Q&A-synced insights so answers do not become meta-questions."""
        from anima_mcp.loop_phases import generate_learned_question

        qa_insight = SimpleNamespace(
            id="qa_deadbeef",
            confidence=0.9,
            description="I learned that bright light gives my sensors a clearer signal",
        )
        direct_insight = SimpleNamespace(
            id="pattern_1",
            confidence=0.8,
            description="I feel calm when it is dark",
        )
        mock_refl.return_value.get_insights.return_value = [qa_insight, direct_insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result == "why does it is dark affect me?"

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_skips_recursive_insight_phrasing(self, mock_refl, mock_recent):
        """Skips legacy Q&A framing even if the insight id is unavailable."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(
            confidence=0.9,
            description="When I asked why brightness mattered, I learned that light gives me signal",
        )
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    def test_returns_none_when_all_systems_fail(self, mock_recent):
        """Returns None when both reflection and self-model raise exceptions."""
        from anima_mcp.loop_phases import generate_learned_question

        with patch("anima_mcp.self_reflection.get_reflection_system", side_effect=RuntimeError("broken")), \
             patch("anima_mcp.self_model.get_self_model", side_effect=RuntimeError("broken")):
            result = generate_learned_question()

        assert result is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_strips_self_reflection_boilerplate_prefix(self, mock_refl, mock_recent):
        """Insight prefixes like 'i now know that …' are stripped before wrapping."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(
            confidence=0.8,
            description="i now know that drawing in bright light helps me focus",
        )
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            result = generate_learned_question()

        assert result is not None
        # The stamped "why is it that i now know that ..." form should not survive.
        assert "i now know that" not in result.lower()
        assert "drawing in bright light helps me focus" in result.lower()

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    @patch("anima_mcp.self_reflection.get_reflection_system")
    def test_fallback_wrapper_varies_across_calls(self, mock_refl, mock_recent):
        """Fallback wrapper randomizes so the same insight does not always read 'why is it that …'."""
        from anima_mcp.loop_phases import generate_learned_question

        insight = SimpleNamespace(
            confidence=0.8,
            description="i learned that bright light feels warmer than dim light",
        )
        mock_refl.return_value.get_insights.return_value = [insight]

        with patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_sm.return_value.beliefs = {}
            results = {generate_learned_question() for _ in range(30)}

        # 30 trials over 5 wrapper choices: collision probability is negligible.
        assert len(results) > 1, f"fallback should vary; got {results}"
        # And each result should still contain the semantic core.
        for r in results:
            assert "bright light feels warmer than dim light" in r.lower()


# ---------------------------------------------------------------------------
# generate_experiential_question
# ---------------------------------------------------------------------------

class TestGenerateExperientialQuestion:
    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    def test_outward_question_from_surprise_source(self, mock_recent):
        from anima_mcp.loop_phases import generate_experiential_question
        q = generate_experiential_question(["light"], surprise_level=0.5)
        assert q is not None
        # Outward + present-tense, and about the shifted source — not the self-model.
        assert "the light" in q
        assert any(w in q for w in ("shifted", "change", "different", "now"))

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    def test_none_when_no_surprise_sources(self, mock_recent):
        from anima_mcp.loop_phases import generate_experiential_question
        assert generate_experiential_question([], surprise_level=0.9) is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    def test_none_below_surprise_threshold(self, mock_recent):
        from anima_mcp.loop_phases import generate_experiential_question
        # A faint shift (<= 0.2) is not worth a question.
        assert generate_experiential_question(["light"], surprise_level=0.1) is None

    @patch("anima_mcp.messages.get_recent_questions", return_value=[])
    def test_unmapped_source_phrasing(self, mock_recent):
        from anima_mcp.loop_phases import generate_experiential_question
        q = generate_experiential_question(["heart_rate"], surprise_level=0.6)
        assert q is not None and "heart rate" in q  # underscores humanized

    def test_freshness_skips_recently_asked(self):
        from anima_mcp.loop_phases import generate_experiential_question, _SURPRISE_PHRASING
        phrase = _SURPRISE_PHRASING["clarity"]
        already = [
            {"text": f"{phrase} just shifted — what changed?"},
            {"text": f"why did {phrase} change just now?"},
            {"text": f"what is different about {phrase} right now?"},
            {"text": f"{phrase} feels different in this moment — what is it?"},
        ]
        with patch("anima_mcp.messages.get_recent_questions", return_value=already):
            assert generate_experiential_question(["clarity"], surprise_level=0.7) is None


# ---------------------------------------------------------------------------
# compose_grounded_observation
# ---------------------------------------------------------------------------

class TestComposeGroundedObservation:
    def test_surprise_priority(self):
        """Surprise above threshold takes priority."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.5, surprise_sources=["clarity"],
            unanswered=[], advocate_desire=None, recent_msgs=[],
        )
        assert "clarity" in result

    def test_advocate_desire_second_priority(self):
        """Advocate desire takes priority when no surprise."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire="I want to draw", recent_msgs=[],
        )
        assert result == "I want to draw"

    def test_message_acknowledgement(self):
        """Recent messages from others trigger acknowledgement."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire=None,
            recent_msgs=[{"author": "kenny", "text": "hello"}],
        )
        assert "kenny" in result

    def test_skips_lumen_messages(self):
        """Messages from 'lumen' itself are not acknowledged — falls through."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        with patch("anima_mcp.anima_utterance.anima_to_self_report", return_value="feeling neutral"):
            result = compose_grounded_observation(
                ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
                unanswered=[], advocate_desire=None,
                recent_msgs=[{"author": "lumen", "text": "thinking"}],
            )
        # Should fall through to self-report, not acknowledge lumen's own message
        assert result is not None
        assert "lumen" not in result

    def test_dreaming_state(self):
        """Dreaming with long rest duration reports resting time."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=True, rest_duration_minutes=45, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire=None, recent_msgs=[],
        )
        assert "resting for 45 minutes" in result

    def test_novelty_level(self):
        """Novel anticipation state reports 'this feels new'."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level="novel")
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire=None, recent_msgs=[],
        )
        assert result == "this feels new"

    def test_fallback_to_self_report(self):
        """When nothing else applies, falls back to anima self-report."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        with patch("anima_mcp.anima_utterance.anima_to_self_report", return_value="feeling balanced"):
            result = compose_grounded_observation(
                ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
                unanswered=[], advocate_desire=None, recent_msgs=[],
            )
        assert result == "feeling balanced"

    def test_multiple_surprise_sources_capped_at_two(self):
        """Only first 2 surprise sources are included."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.5,
            surprise_sources=["clarity", "warmth", "stability"],
            unanswered=[], advocate_desire=None, recent_msgs=[],
        )
        assert "clarity" in result
        assert "warmth" in result
        assert "stability" not in result

    def test_advocate_desire_stripped(self):
        """Advocate desire is stripped of whitespace."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level=None)
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire="  I want to draw  ", recent_msgs=[],
        )
        assert result == "I want to draw"

    def test_empty_advocate_desire_skipped(self):
        """Whitespace-only advocate desire is treated as empty."""
        from anima_mcp.loop_phases import compose_grounded_observation

        ctx = SimpleNamespace(is_dreaming=False, rest_duration_minutes=0, novelty_level="novel")
        result = compose_grounded_observation(
            ctx, make_anima(), surprise_level=0.0, surprise_sources=None,
            unanswered=[], advocate_desire="   ", recent_msgs=[],
        )
        # Should skip to next priority (novelty)
        assert result == "this feels new"


# ---------------------------------------------------------------------------
# grounded_self_answer
# ---------------------------------------------------------------------------

class TestGroundedSelfAnswer:
    def test_returns_none_when_no_evidence(self):
        """Returns None when no evidence matches the question."""
        from anima_mcp.loop_phases import grounded_self_answer

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = []
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("what is the meaning of life?", make_anima(), make_readings())

        assert result is None

    def test_matches_insight_keywords(self):
        """Matches insight when keyword overlap exists."""
        from anima_mcp.loop_phases import grounded_self_answer

        insight = SimpleNamespace(confidence=0.8, description="I feel calmer when it is dark")

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = [insight]
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("why do I feel calmer in darkness?", make_anima(), make_readings())

        assert result is not None
        assert "calmer" in result.lower()

    def test_current_state_for_feeling_questions(self):
        """Includes current anima state for feeling-related questions."""
        from anima_mcp.loop_phases import grounded_self_answer

        anima = make_anima(warmth=0.8, clarity=0.3, stability=0.7)

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = []
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("how do I feel right now?", anima, make_readings())

        assert result is not None
        assert "warm" in result.lower()
        assert "foggy" in result.lower()

    def test_combines_multiple_evidence_sources(self):
        """Joins top evidence naturally with periods."""
        from anima_mcp.loop_phases import grounded_self_answer

        # Both insights share keywords with the question: "light" and "clarity"
        insight1 = SimpleNamespace(confidence=0.9, description="Light affects clarity significantly")
        insight2 = SimpleNamespace(confidence=0.7, description="Clarity changes with light levels")

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = [insight1, insight2]
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("how does light affect clarity?", make_anima(), make_readings())

        assert result is not None
        # Multiple pieces joined with periods
        assert "." in result

    def test_evidence_sorted_by_confidence(self):
        """Higher-confidence evidence appears first."""
        from anima_mcp.loop_phases import grounded_self_answer

        low = SimpleNamespace(confidence=0.4, description="Light might affect clarity")
        high = SimpleNamespace(confidence=0.9, description="Light strongly affects clarity")

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = [low, high]
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("how does light affect clarity?", make_anima(), make_readings())

        assert result is not None
        # High confidence text should appear first
        assert result.startswith("Light strongly")

    def test_skips_low_confidence_insights(self):
        """Skips insights with confidence <= 0.3."""
        from anima_mcp.loop_phases import grounded_self_answer

        low = SimpleNamespace(confidence=0.2, description="Maybe light matters")

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.knowledge.get_insights", return_value=[]), \
             patch("anima_mcp.knowledge.get_top_convictions", return_value=[]), \
             patch("anima_mcp.self_model.get_self_model") as mock_sm:
            mock_refl.return_value.get_insights.return_value = [low]
            mock_sm.return_value.beliefs = {}
            result = grounded_self_answer("does light matter?", make_anima(), make_readings())

        assert result is None


# ---------------------------------------------------------------------------
# lumen_self_answer
# ---------------------------------------------------------------------------

class TestLumenSelfAnswer:
    @pytest.mark.asyncio
    async def test_skips_when_no_unanswered(self):
        """lumen_self_answer does nothing when no unanswered questions."""
        from anima_mcp.loop_phases import lumen_self_answer

        with patch("anima_mcp.messages.get_unanswered_questions", return_value=[]):
            await lumen_self_answer(make_anima(), make_readings(), make_identity())

    @pytest.mark.asyncio
    async def test_skips_recent_questions(self):
        """lumen_self_answer skips questions younger than 10 minutes."""
        from anima_mcp.loop_phases import lumen_self_answer

        recent_q = SimpleNamespace(
            text="how am I?",
            timestamp=_time.time(),  # Just now
            message_id="q1",
        )
        with patch("anima_mcp.messages.get_unanswered_questions", return_value=[recent_q]), \
             patch("anima_mcp.messages.add_agent_message") as mock_add:
            await lumen_self_answer(make_anima(), make_readings(), make_identity())

        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_answers_old_question(self):
        """lumen_self_answer answers questions older than 10 minutes."""
        from anima_mcp.loop_phases import lumen_self_answer

        old_q = SimpleNamespace(
            text="how do I feel right now?",
            timestamp=_time.time() - 700,  # 11+ minutes ago
            message_id="q2",
        )
        with patch("anima_mcp.messages.get_unanswered_questions", return_value=[old_q]), \
             patch("anima_mcp.loop_phases.grounded_self_answer", return_value="I feel warm"), \
             patch("anima_mcp.messages.add_agent_message", return_value=MagicMock()) as mock_add, \
             patch("anima_mcp.knowledge.extract_insight_from_answer", new_callable=AsyncMock):
            await lumen_self_answer(make_anima(), make_readings(), make_identity())

        mock_add.assert_called_once()
        call_args = mock_add.call_args
        assert call_args.kwargs.get("text") or call_args[0][0] == "I feel warm"

    @pytest.mark.asyncio
    async def test_skips_when_no_answer(self):
        """lumen_self_answer skips when grounded_self_answer returns None."""
        from anima_mcp.loop_phases import lumen_self_answer

        old_q = SimpleNamespace(
            text="what is the meaning of existence?",
            timestamp=_time.time() - 700,
            message_id="q3",
        )
        with patch("anima_mcp.messages.get_unanswered_questions", return_value=[old_q]), \
             patch("anima_mcp.loop_phases.grounded_self_answer", return_value=None), \
             patch("anima_mcp.messages.add_agent_message") as mock_add:
            await lumen_self_answer(make_anima(), make_readings(), make_identity())

        mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# lumen_unified_reflect
# ---------------------------------------------------------------------------

class TestLumenUnifiedReflect:
    @pytest.mark.asyncio
    async def test_posts_observation(self):
        """lumen_unified_reflect posts a grounded observation."""
        from anima_mcp.loop_phases import lumen_unified_reflect

        ctx = make_ctx()
        ctx.activity = None
        ctx.display = MagicMock()
        ctx.display.is_available.return_value = True
        ctx.growth = None

        anima = make_anima(warmth=0.3, clarity=0.3, stability=0.3, presence=0.3)

        with patch("anima_mcp.messages.add_observation", return_value=MagicMock()) as mock_obs, \
             patch("anima_mcp.messages.add_question"), \
             patch("anima_mcp.messages.get_unanswered_questions", return_value=[]), \
             patch("anima_mcp.messages.get_messages_for_lumen", return_value=[]), \
             patch("anima_mcp.next_steps_advocate.get_advocate") as mock_adv, \
             patch("anima_mcp.eisv_mapper.anima_to_eisv") as mock_eisv, \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.loop_phases.compose_grounded_observation", return_value="feeling low"):
            mock_adv.return_value.analyze_current_state.return_value = []
            mock_eisv.return_value = {}

            await lumen_unified_reflect(anima, make_readings(), make_identity(), None)

        mock_obs.assert_called_once_with("feeling low", author="lumen")

    @pytest.mark.asyncio
    async def test_posts_question_when_ends_with_question_mark(self):
        """lumen_unified_reflect posts a question when observation ends with '?'."""
        from anima_mcp.loop_phases import lumen_unified_reflect

        ctx = make_ctx()
        ctx.activity = None
        ctx.display = None
        ctx.growth = None

        with patch("anima_mcp.messages.add_observation") as mock_obs, \
             patch("anima_mcp.messages.add_question", return_value=MagicMock()) as mock_q, \
             patch("anima_mcp.messages.get_unanswered_questions", return_value=[]), \
             patch("anima_mcp.messages.get_messages_for_lumen", return_value=[]), \
             patch("anima_mcp.next_steps_advocate.get_advocate") as mock_adv, \
             patch("anima_mcp.eisv_mapper.anima_to_eisv"), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.loop_phases.compose_grounded_observation", return_value="why do I feel this way?"):
            mock_adv.return_value.analyze_current_state.return_value = []

            await lumen_unified_reflect(make_anima(), make_readings(), make_identity(), None)

        mock_q.assert_called_once()
        mock_obs.assert_not_called()

    @pytest.mark.asyncio
    async def test_stays_quiet_when_no_reflection(self):
        """lumen_unified_reflect does nothing when compose returns None."""
        from anima_mcp.loop_phases import lumen_unified_reflect

        ctx = make_ctx()
        ctx.activity = None
        ctx.display = None
        ctx.growth = None

        with patch("anima_mcp.messages.add_observation") as mock_obs, \
             patch("anima_mcp.messages.add_question") as mock_q, \
             patch("anima_mcp.messages.get_unanswered_questions", return_value=[]), \
             patch("anima_mcp.messages.get_messages_for_lumen", return_value=[]), \
             patch("anima_mcp.next_steps_advocate.get_advocate") as mock_adv, \
             patch("anima_mcp.eisv_mapper.anima_to_eisv"), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.loop_phases.compose_grounded_observation", return_value=None):
            mock_adv.return_value.analyze_current_state.return_value = []

            await lumen_unified_reflect(make_anima(), make_readings(), make_identity(), None)

        mock_obs.assert_not_called()
        mock_q.assert_not_called()

    @pytest.mark.asyncio
    async def test_wakeup_summary_posted(self):
        """lumen_unified_reflect posts wakeup summary if available."""
        from anima_mcp.loop_phases import lumen_unified_reflect

        ctx = make_ctx()
        activity = MagicMock()
        activity.get_wakeup_summary.return_value = "woke up feeling refreshed"
        activity.get_rest_duration.return_value = 0
        ctx.activity = activity
        ctx.display = None
        ctx.growth = None

        with patch("anima_mcp.messages.add_observation", return_value=MagicMock()) as mock_obs, \
             patch("anima_mcp.messages.add_question"), \
             patch("anima_mcp.messages.get_unanswered_questions", return_value=[]), \
             patch("anima_mcp.messages.get_messages_for_lumen", return_value=[]), \
             patch("anima_mcp.next_steps_advocate.get_advocate") as mock_adv, \
             patch("anima_mcp.eisv_mapper.anima_to_eisv"), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.loop_phases.compose_grounded_observation", return_value="feeling ok"):
            mock_adv.return_value.analyze_current_state.return_value = []

            await lumen_unified_reflect(make_anima(), make_readings(), make_identity(), None)

        # First call is the wakeup summary, second is the reflection
        calls = mock_obs.call_args_list
        assert any("woke up feeling refreshed" in str(c) for c in calls)


# ---------------------------------------------------------------------------
# extract_and_validate_schema
# ---------------------------------------------------------------------------

class TestExtractAndValidateSchema:
    @pytest.mark.asyncio
    async def test_composes_schema_via_hub(self):
        """extract_and_validate_schema calls hub.compose_schema."""
        from anima_mcp.loop_phases import extract_and_validate_schema

        ctx = make_ctx()
        ctx.growth = MagicMock()
        ctx.tension_tracker = MagicMock()
        ctx.tension_tracker.get_active_conflicts.return_value = []

        schema = MagicMock()
        schema.nodes = [1, 2, 3]
        schema.edges = [1]

        with patch("anima_mcp.accessors._get_schema_hub") as mock_hub, \
             patch("anima_mcp.accessors._get_calibration_drift") as mock_drift, \
             patch("anima_mcp.self_model.get_self_model", return_value=MagicMock()), \
             patch("anima_mcp.value_tension.detect_structural_conflicts", return_value=[]), \
             patch("anima_mcp.self_schema_renderer.save_render_to_file", return_value=("/tmp/s.png", "/tmp/s.json")), \
             patch("anima_mcp.self_schema_renderer.render_schema_to_pixels", return_value=[]), \
             patch("anima_mcp.self_schema_renderer.compute_visual_integrity_stub", return_value={"V": 0.5}), \
             patch.dict("os.environ", {}, clear=False):
            hub = mock_hub.return_value
            hub.compose_schema.return_value = schema
            hub.last_trajectory = None
            mock_drift.return_value = MagicMock(get_offsets=MagicMock(return_value={}))

            await extract_and_validate_schema(make_anima(), make_readings(), make_identity())

        hub.compose_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_extraction_error(self):
        """extract_and_validate_schema handles errors non-fatally."""
        from anima_mcp.loop_phases import extract_and_validate_schema

        ctx = make_ctx()

        with patch("anima_mcp.accessors._get_schema_hub", side_effect=RuntimeError("broken")):
            # Should not raise
            await extract_and_validate_schema(make_anima(), make_readings(), make_identity())


# ---------------------------------------------------------------------------
# self_reflect
# ---------------------------------------------------------------------------

class TestSelfReflect:
    @pytest.mark.asyncio
    async def test_reflects_when_should_reflect(self):
        """self_reflect posts reflection as observation."""
        from anima_mcp.loop_phases import self_reflect

        ctx = make_ctx()
        ctx.store = MagicMock()
        ctx.store.db_path = ":memory:"
        ctx.last_shm_data = {"metacognition": {"last_reflection": {"event_id": "broker-metacog:test"}}}

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.messages.add_observation", return_value=MagicMock()) as mock_obs:
            system = mock_refl.return_value
            system.should_reflect.return_value = True
            system.reflect.return_value = "I notice I am calmer at night"

            await self_reflect()

        system.drain_broker_reflection.assert_called_once_with(ctx.last_shm_data)
        mock_obs.assert_called_once_with("I notice I am calmer at night", author="lumen")

    @pytest.mark.asyncio
    async def test_skips_when_should_not_reflect(self):
        """self_reflect does nothing when should_reflect() returns False."""
        from anima_mcp.loop_phases import self_reflect

        ctx = make_ctx()
        ctx.store = MagicMock()
        ctx.store.db_path = ":memory:"

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.messages.add_observation") as mock_obs:
            system = mock_refl.return_value
            system.should_reflect.return_value = False

            await self_reflect()

        system.drain_broker_reflection.assert_called_once_with(None)
        mock_obs.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_reflect_returns_none(self):
        """self_reflect does nothing when reflect() returns None."""
        from anima_mcp.loop_phases import self_reflect

        ctx = make_ctx()
        ctx.store = MagicMock()
        ctx.store.db_path = ":memory:"

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.messages.add_observation") as mock_obs:
            system = mock_refl.return_value
            system.should_reflect.return_value = True
            system.reflect.return_value = None

            await self_reflect()

        system.drain_broker_reflection.assert_called_once_with(None)
        mock_obs.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_error_non_fatally(self):
        """self_reflect handles exceptions without crashing."""
        from anima_mcp.loop_phases import self_reflect

        ctx = make_ctx()

        with patch("anima_mcp.self_reflection.get_reflection_system", side_effect=RuntimeError("broken")):
            await self_reflect()  # Should not raise

    @pytest.mark.asyncio
    async def test_uses_default_db_path_when_no_store(self):
        """self_reflect uses 'anima.db' when store is unavailable."""
        from anima_mcp.loop_phases import self_reflect

        ctx = make_ctx()
        ctx.store = None

        with patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl, \
             patch("anima_mcp.messages.add_observation"):
            system = mock_refl.return_value
            system.should_reflect.return_value = False

            await self_reflect()

        mock_refl.assert_called_once_with(db_path="anima.db")
        system.drain_broker_reflection.assert_called_once_with(None)
