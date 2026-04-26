from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import parse_result


class TestCallerNameResolution:
    def test_get_caller_session_id_prefers_mcp_header(self):
        from anima_mcp.handlers.communication import _get_caller_session_id

        with patch(
            "anima_mcp.handlers.communication._get_request_headers",
            return_value={"mcp-session-id": "mcp-1", "x-session-id": "x-1"},
        ):
            assert _get_caller_session_id() == "mcp-1"

    def test_get_caller_session_id_falls_back_to_x_session(self):
        from anima_mcp.handlers.communication import _get_caller_session_id

        with patch(
            "anima_mcp.handlers.communication._get_request_headers",
            return_value={"x-session-id": "x-1"},
        ):
            assert _get_caller_session_id() == "x-1"

    def test_parse_caller_name_from_ua_variants(self):
        from anima_mcp.handlers.communication import _parse_caller_name_from_ua

        assert _parse_caller_name_from_ua("claude-code/2.1.56 (claude-vscode)") == "Claude Code (VSCode)"
        assert _parse_caller_name_from_ua("claude-code/2.1.56") == "Claude Code"
        assert _parse_caller_name_from_ua("cursor/0.50.1") == "Cursor"
        assert _parse_caller_name_from_ua("windsurf/0.1") == "Windsurf"
        assert _parse_caller_name_from_ua("custom-client/1.0") == "custom-client"
        assert _parse_caller_name_from_ua("") is None

    def test_resolve_caller_name_prefers_explicit_agent_name(self):
        from anima_mcp.handlers.communication import _resolve_caller_name

        assert _resolve_caller_name({"agent_name": "Kenny"}) == "Kenny"

    def test_resolve_caller_name_uses_user_agent_fallback(self):
        from anima_mcp.handlers.communication import _resolve_caller_name

        with patch("anima_mcp.handlers.communication._get_request_headers", return_value={"user-agent": "cursor/1.2.3"}):
            assert _resolve_caller_name({"agent_name": "agent"}) == "Cursor"


@pytest.mark.asyncio
class TestConfigureVoice:
    async def test_voice_unavailable_returns_error(self):
        from anima_mcp.handlers.communication import handle_configure_voice

        with patch("anima_mcp.accessors._get_voice", return_value=None):
            data = parse_result(await handle_configure_voice({"action": "status"}))
        assert "Voice system not available" in data["error"]

    async def test_status_returns_voice_state(self):
        from anima_mcp.handlers.communication import handle_configure_voice

        state = SimpleNamespace(
            is_listening=True,
            is_speaking=False,
            last_heard=SimpleNamespace(text="hello"),
        )
        voice = SimpleNamespace(is_running=True, chattiness=0.4, state=state)

        with patch("anima_mcp.accessors._get_voice", return_value=voice):
            data = parse_result(await handle_configure_voice({"action": "status"}))

        assert data["action"] == "status"
        assert data["available"] is True
        assert data["running"] is True
        assert data["is_listening"] is True
        assert data["last_heard"] == "hello"

    async def test_configure_updates_voice_settings(self):
        from anima_mcp.handlers.communication import handle_configure_voice

        low_voice = MagicMock()
        low_voice._config = SimpleNamespace(wake_word="lumen")
        voice = SimpleNamespace(
            is_running=True,
            chattiness=0.2,
            state=None,
            _voice=low_voice,
        )

        with patch("anima_mcp.accessors._get_voice", return_value=voice):
            data = parse_result(await handle_configure_voice({
                "action": "configure",
                "always_listening": True,
                "chattiness": 0.9,
                "wake_word": "anima",
            }))

        low_voice.set_always_listening.assert_called_once_with(True)
        assert data["success"] is True
        assert data["changes"]["always_listening"] is True
        assert data["changes"]["chattiness"] == 0.9
        assert data["changes"]["wake_word"] == "anima"

    async def test_unknown_action_returns_error(self):
        from anima_mcp.handlers.communication import handle_configure_voice

        voice = SimpleNamespace(is_running=True, chattiness=0.2, state=None)
        with patch("anima_mcp.accessors._get_voice", return_value=voice):
            data = parse_result(await handle_configure_voice({"action": "bad"}))

        assert "error" in data
        assert "valid_actions" in data


@pytest.mark.asyncio
class TestPrimitiveFeedback:
    async def test_resonate_success_returns_score(self):
        from anima_mcp.handlers.communication import handle_primitive_feedback

        lang = MagicMock()
        lang.record_explicit_feedback.return_value = {"score": 0.7, "token_updates": {"warm": 1}}
        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(db_path=":memory:")), \
             patch("anima_mcp.primitive_language.get_language_system", return_value=lang):
            data = parse_result(await handle_primitive_feedback({"action": "resonate"}))

        assert data["success"] is True
        assert data["action"] == "resonate"
        assert data["score"] == 0.7

    async def test_recent_returns_utterances_list(self):
        from anima_mcp.handlers.communication import handle_primitive_feedback

        lang = MagicMock()
        lang.get_recent_utterances.return_value = [{"text": "pulse", "score": 0.2}]
        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(db_path=":memory:")), \
             patch("anima_mcp.primitive_language.get_language_system", return_value=lang):
            data = parse_result(await handle_primitive_feedback({"action": "recent"}))

        assert data["action"] == "recent"
        assert data["count"] == 1

    async def test_stats_is_default_action(self):
        from anima_mcp.handlers.communication import handle_primitive_feedback

        lang = MagicMock()
        lang.get_stats.return_value = {"utterances": 42}
        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(db_path=":memory:")), \
             patch("anima_mcp.primitive_language.get_language_system", return_value=lang):
            data = parse_result(await handle_primitive_feedback({}))

        assert data["action"] == "stats"
        assert data["primitive_language_system"]["utterances"] == 42


@pytest.mark.asyncio
class TestPostMessageRespondsToMatching:
    async def test_empty_message_returns_error(self):
        from anima_mcp.handlers.communication import handle_post_message

        data = parse_result(await handle_post_message({"message": "   "}))
        assert data["error"] == "message parameter required"

    async def test_human_message_tracks_interaction_and_social_boost(self):
        from anima_mcp.handlers.communication import handle_post_message

        growth = MagicMock()
        activity = MagicMock()
        store = SimpleNamespace()
        anima = SimpleNamespace(clarity=0.77)
        boost_flag = SimpleNamespace(touch=MagicMock())

        with patch("anima_mcp.accessors._get_growth", return_value=growth), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(SimpleNamespace(), anima)), \
             patch("anima_mcp.messages.add_user_message", return_value="u1"), \
             patch("anima_mcp.handlers.communication._SOCIAL_BOOST_PATH", boost_flag):
            data = parse_result(await handle_post_message({"message": "hello lumen", "source": "human"}))

        assert data["success"] is True
        assert data["source"] == "human"
        growth.record_interaction.assert_called_once()
        activity.record_interaction.assert_called_once()

    async def test_agent_message_matches_partial_question_id(self):
        from anima_mcp.handlers.communication import handle_post_message

        question = SimpleNamespace(message_id="q_abcdef1234", msg_type="question")
        board = SimpleNamespace(_messages=[question], _load=MagicMock())
        msg = SimpleNamespace(message_id="msg_1")
        growth = MagicMock()
        growth.get_visitor_context.return_value = {"relationship": "trusted"}

        with patch("anima_mcp.accessors._get_growth", return_value=growth), \
             patch("anima_mcp.accessors._get_activity", return_value=None), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.get_board", return_value=board), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg):
            data = parse_result(await handle_post_message({
                "message": "Here is my answer",
                "source": "agent",
                "responds_to": "q_abc",
                "agent_name": "Cursor",
            }))

        assert data["success"] is True
        assert data["answered_question"] == "q_abcdef1234"
        assert "Matched partial ID" in data["note"]
        assert data["visitor_context"]["relationship"] == "trusted"

    async def test_agent_message_returns_error_for_unknown_question(self):
        from anima_mcp.handlers.communication import handle_post_message

        board = SimpleNamespace(_messages=[], _load=MagicMock())
        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=None), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.get_board", return_value=board):
            data = parse_result(await handle_post_message({
                "message": "answer",
                "source": "agent",
                "responds_to": "missing",
            }))

        assert "error" in data
        assert "not found" in data["error"]

    async def test_delivery_status_active(self):
        """delivery_status is 'delivered' when Lumen is active."""
        from anima_mcp.handlers.communication import handle_post_message
        from anima_mcp.activity_state import ActivityLevel

        activity = MagicMock()
        activity._current_level = ActivityLevel.ACTIVE
        msg = SimpleNamespace(message_id="msg_1")

        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg):
            data = parse_result(await handle_post_message({
                "message": "hello",
                "source": "agent",
                "agent_name": "TestAgent",
            }))

        assert data["success"] is True
        assert data["delivery_status"] == "delivered"

    async def test_delivery_status_dormant(self):
        """delivery_status is 'delivered_dormant' when Lumen is resting."""
        from anima_mcp.handlers.communication import handle_post_message
        from anima_mcp.activity_state import ActivityLevel

        activity = MagicMock()
        activity._current_level = ActivityLevel.RESTING
        msg = SimpleNamespace(message_id="msg_2")

        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg):
            data = parse_result(await handle_post_message({
                "message": "goodnight",
                "source": "agent",
                "agent_name": "TestAgent",
            }))

        assert data["success"] is True
        assert data["delivery_status"] == "delivered_dormant"

    async def test_delivery_status_drowsy(self):
        """delivery_status is 'delivered_drowsy' when Lumen is drowsy."""
        from anima_mcp.handlers.communication import handle_post_message
        from anima_mcp.activity_state import ActivityLevel

        activity = MagicMock()
        activity._current_level = ActivityLevel.DROWSY
        msg = SimpleNamespace(message_id="msg_3")

        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg):
            data = parse_result(await handle_post_message({
                "message": "still here?",
                "source": "agent",
                "agent_name": "TestAgent",
            }))

        assert data["success"] is True
        assert data["delivery_status"] == "delivered_drowsy"

    async def test_delivery_status_human_source_dormant(self):
        """delivery_status works for human-source messages too."""
        from anima_mcp.handlers.communication import handle_post_message
        from anima_mcp.activity_state import ActivityLevel

        activity = MagicMock()
        activity._current_level = ActivityLevel.RESTING

        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=activity), \
             patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.messages.add_user_message", return_value="u1"):
            data = parse_result(await handle_post_message({
                "message": "goodnight lumen",
                "source": "human",
            }))

        assert data["success"] is True
        assert data["delivery_status"] == "delivered_dormant"


@pytest.mark.asyncio
class TestLumenQaExtended:
    async def test_list_mode_returns_only_truly_unanswered(self):
        from anima_mcp.handlers.communication import handle_lumen_qa

        q1 = SimpleNamespace(
            message_id="q1",
            msg_type="question",
            text="What is light?",
            context=None,
            answered=False,
            state_snapshot={"mood": "curious"},
            age_str=lambda: "5m",
        )
        q2 = SimpleNamespace(
            message_id="q2",
            msg_type="question",
            text="Why am I calm at night?",
            context=None,
            answered=True,  # auto-expired but still unanswered
            state_snapshot=None,
            age_str=lambda: "15m",
        )
        q3 = SimpleNamespace(
            message_id="q3",
            msg_type="question",
            text="Will anyone answer this?",
            context=None,
            answered=False,
            state_snapshot=None,
            age_str=lambda: "1m",
        )
        a3 = SimpleNamespace(msg_type="agent", responds_to="q3")
        board = SimpleNamespace(_messages=[q1, q2, q3, a3], _load=MagicMock(), repair_orphaned_answered=lambda: 1)

        with patch("anima_mcp.messages.get_board", return_value=board):
            data = parse_result(await handle_lumen_qa({"limit": "2"}))

        assert data["action"] == "list"
        assert data["unanswered_count"] == 2
        ids = [q["id"] for q in data["questions"]]
        assert "q3" not in ids
        assert "q2" in ids

    async def test_answer_mode_prefix_match_and_insight_enrichment(self):
        from anima_mcp.handlers.communication import handle_lumen_qa

        q = SimpleNamespace(message_id="q_abcdef123", msg_type="question", text="How does light affect you?")
        board = SimpleNamespace(_messages=[q], _load=MagicMock())
        msg = SimpleNamespace(message_id="m1")
        bridge = SimpleNamespace(resolve_caller_identity=AsyncMock(return_value="Resolved Agent"))
        insight = SimpleNamespace(text="Dim light helps calm", category="environment")
        growth = SimpleNamespace(get_visitor_context=lambda name: {"name": name, "bond": "trusted"})

        with patch("anima_mcp.messages.get_board", return_value=board), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg), \
             patch("anima_mcp.handlers.communication._get_unitares_bridge", return_value=bridge), \
             patch("anima_mcp.knowledge.extract_insight_from_answer", AsyncMock(return_value=insight)), \
             patch("anima_mcp.knowledge.apply_insight", return_value={"shift": "positive"}), \
             patch("anima_mcp.accessors._get_growth", return_value=growth):
            data = parse_result(await handle_lumen_qa({
                "question_id": "q_abc",
                "answer": "I feel calmer when the room is dim.",
                "client_session_id": "sess-1",
                "agent_name": "agent",
            }))

        assert data["success"] is True
        assert data["question_id"] == "q_abcdef123"
        assert data["matched_partial_id"] == "q_abc"
        assert data["agent_name"] == "Resolved Agent"
        assert data["insight"]["behavior_effects"]["shift"] == "positive"
        assert data["visitor_context"]["bond"] == "trusted"

    async def test_answer_mode_returns_helpful_error_when_question_missing(self):
        from anima_mcp.handlers.communication import handle_lumen_qa

        board = SimpleNamespace(_messages=[], _load=MagicMock())
        with patch("anima_mcp.messages.get_board", return_value=board):
            data = parse_result(await handle_lumen_qa({"question_id": "missing", "answer": "test"}))

        assert data["success"] is False
        assert "not found" in data["error"]
        assert "recent_question_ids" in data

    async def test_answer_mode_accepts_integer_question_id(self):
        # All-numeric hex IDs (e.g. "83372556") may arrive as ints because some
        # MCP relays coerce numeric-looking strings. The handler must accept
        # either form and stringify before lookup.
        from anima_mcp.handlers.communication import handle_lumen_qa

        q = SimpleNamespace(message_id="83372556", msg_type="question", text="ok?")
        board = SimpleNamespace(_messages=[q], _load=MagicMock())
        msg = SimpleNamespace(message_id="m1")
        with patch("anima_mcp.messages.get_board", return_value=board), \
             patch("anima_mcp.messages.add_agent_message", return_value=msg), \
             patch("anima_mcp.handlers.communication._get_unitares_bridge", return_value=None), \
             patch("anima_mcp.knowledge.extract_insight_from_answer", AsyncMock(return_value=None)), \
             patch("anima_mcp.accessors._get_growth", return_value=None):
            data = parse_result(await handle_lumen_qa({
                "question_id": 83372556,
                "answer": "answered via integer id",
            }))

        assert data["success"] is True
        assert data["question_id"] == "83372556"


@pytest.mark.asyncio
class TestSayAndPrimitiveFeedbackEdges:
    async def test_say_requires_text(self):
        from anima_mcp.handlers.communication import handle_say

        data = parse_result(await handle_say({"text": ""}))
        assert data["error"] == "No text provided"

    async def test_say_audio_mode_invokes_tts_and_returns_success(self):
        from anima_mcp.handlers.communication import handle_say

        voice_inner = SimpleNamespace(say=MagicMock())
        voice = SimpleNamespace(_voice=voice_inner)
        with patch("anima_mcp.messages.add_observation", return_value=SimpleNamespace(message_id="obs1")), \
             patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(add_note=MagicMock())), \
             patch("anima_mcp.accessors._get_voice", return_value=voice), \
             patch("anima_mcp.accessors.VOICE_MODE", "audio"):
            data = parse_result(await handle_say({"text": "Hello world"}))

        voice_inner.say.assert_called_once()
        assert data["success"] is True
        assert data["mode"] == "audio"

    async def test_primitive_feedback_confused_and_no_recent_paths(self):
        from anima_mcp.handlers.communication import handle_primitive_feedback

        lang = MagicMock()
        lang.record_explicit_feedback.side_effect = [None, {"score": -0.2, "token_updates": {"noise": -1}}]
        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(db_path=":memory:")), \
             patch("anima_mcp.primitive_language.get_language_system", return_value=lang):
            no_recent = parse_result(await handle_primitive_feedback({"action": "resonate"}))
            confused = parse_result(await handle_primitive_feedback({"action": "confused"}))

        assert "error" in no_recent
        assert no_recent["error"] == "No recent utterance to give feedback on"
        assert confused["success"] is True
        assert confused["action"] == "confused"

    async def test_primitive_feedback_handles_backend_exception(self):
        from anima_mcp.handlers.communication import handle_primitive_feedback

        with patch(
            "anima_mcp.primitive_language.get_language_system",
            side_effect=RuntimeError("db unavailable"),
        ):
            data = parse_result(await handle_primitive_feedback({"action": "stats"}))

        assert "Primitive language error" in data["error"]
