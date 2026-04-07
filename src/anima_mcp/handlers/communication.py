"""Communication handlers — Q&A, messaging, voice, and feedback.

Handlers: lumen_qa, post_message, say, configure_voice, primitive_feedback.
"""

import json
import sys
from pathlib import Path

from mcp.types import TextContent

_SOCIAL_BOOST_PATH = Path("/dev/shm/anima_social_boost")


def _get_request_headers() -> dict | None:
    """Get HTTP headers from the current MCP request context.

    Returns header dict, or None in stdio mode / no request context.
    """
    try:
        from mcp.server.lowlevel.server import request_ctx
        ctx = request_ctx.get()
        if ctx.request is not None:
            return ctx.request.headers
    except (LookupError, AttributeError, ImportError):
        pass
    return None


def _get_caller_session_id() -> str | None:
    """Extract caller's session ID from MCP request context headers."""
    headers = _get_request_headers()
    if headers:
        return headers.get("mcp-session-id") or headers.get("x-session-id")
    return None


def _parse_caller_name_from_ua(user_agent: str) -> str | None:
    """Parse a human-readable caller name from User-Agent string.

    Examples:
        "claude-code/2.1.56 (claude-vscode)" -> "Claude Code (VSCode)"
        "claude-code/2.1.56"                 -> "Claude Code"
        "cursor/0.50.1"                      -> "Cursor"
        "some-mcp-client/1.0"                -> "some-mcp-client"
    """
    import re

    if not user_agent:
        return None

    ua = user_agent.strip().lower()

    # Claude Code variants
    m = re.match(r"claude-code/[\d.]+\s*\(claude-(vscode|desktop)\)", ua)
    if m:
        variant = m.group(1).replace("vscode", "VSCode").replace("desktop", "Desktop")
        return f"Claude Code ({variant})"

    if ua.startswith("claude-code/"):
        return "Claude Code"

    # Cursor
    if ua.startswith("cursor/"):
        return "Cursor"

    # Windsurf
    if ua.startswith("windsurf/"):
        return "Windsurf"

    # Generic: use the product name before the version slash
    m = re.match(r"([a-zA-Z][\w-]*)/", ua)
    if m:
        return m.group(1)

    return None


def _resolve_caller_name(arguments: dict) -> str:
    """Resolve the best available caller name.

    Priority:
    1. Explicit agent_name in arguments (if not default "agent")
    2. UNITARES identity resolution via client_session_id
    3. User-Agent header parsing
    4. Default "agent"
    """
    agent_name = arguments.get("agent_name", "agent")

    # If caller explicitly identified themselves, trust it
    if agent_name and agent_name != "agent":
        return agent_name

    # Try user-agent parsing as fallback
    headers = _get_request_headers()
    if headers:
        ua_name = _parse_caller_name_from_ua(headers.get("user-agent", ""))
        if ua_name:
            return ua_name

    return "agent"


def _get_unitares_bridge():
    """Get shared server bridge for identity resolution (late import to avoid circular deps)."""
    try:
        from ..accessors import _get_server_bridge
        return _get_server_bridge()
    except Exception:
        return None


async def handle_lumen_qa(arguments: dict) -> list[TextContent]:
    """
    Unified Q&A tool: list Lumen's questions OR answer one.

    Usage:
    - lumen_qa() -> list unanswered questions
    - lumen_qa(question_id="x", answer="...") -> answer question x
    """
    from ..messages import get_board, MESSAGE_TYPE_QUESTION, add_agent_message

    question_id = arguments.get("question_id")
    answer = arguments.get("answer")
    limit = arguments.get("limit", 5)
    client_session_id = arguments.get("client_session_id") or _get_caller_session_id()

    # Identity resolution: UNITARES session > explicit agent_name > user-agent > "agent"
    agent_name = _resolve_caller_name(arguments)
    bridge = _get_unitares_bridge()
    if bridge and client_session_id:
        try:
            resolved = await bridge.resolve_caller_identity(session_id=client_session_id)
            if resolved:
                agent_name = resolved
        except Exception:
            pass

    # Convert limit to int if string
    if isinstance(limit, str):
        try:
            limit = int(limit)
        except ValueError:
            limit = 5

    board = get_board()
    board._load(force=True)

    # If question_id and answer provided -> answer mode
    if question_id and answer:
        # Find the question with prefix matching support
        question = None
        validated_question_id = None

        # Try exact match first
        for m in board._messages:
            if m.message_id == question_id and m.msg_type == MESSAGE_TYPE_QUESTION:
                question = m
                validated_question_id = question_id
                break

        # If exact match failed, try prefix matching
        if not question:
            matching = [
                m for m in board._messages
                if m.msg_type == MESSAGE_TYPE_QUESTION
                and m.message_id.startswith(question_id)
            ]
            if len(matching) == 1:
                question = matching[0]
                validated_question_id = question.message_id
            elif len(matching) > 1:
                # Multiple matches - use most recent
                question = matching[-1]
                validated_question_id = question.message_id
            else:
                # No match - return helpful error
                all_q_ids = [m.message_id for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION]
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": f"Question '{question_id}' not found",
                    "hint": "Use the full question ID from lumen_qa()",
                    "recent_question_ids": all_q_ids[-5:] if all_q_ids else []
                }))]

        # Add answer via add_agent_message (handles responds_to linking)
        result = add_agent_message(answer, agent_name=agent_name, responds_to=validated_question_id)

        # Signal social boost to broker (someone answered Lumen's question)
        try:
            _SOCIAL_BOOST_PATH.touch()
        except Exception:
            pass

        # Extract insight from Q&A (inline so result visible in response)
        # This populates Lumen's knowledge base with learnings from answers
        insight_result = None
        try:
            from ..knowledge import extract_insight_from_answer
            insight = await extract_insight_from_answer(
                question=question.text,
                answer=answer,
                author=agent_name
            )
            if insight:
                insight_result = {"text": insight.text, "category": insight.category}
                print(f"[Q&A] Extracted insight: {insight.text[:80]}", file=sys.stderr, flush=True)
                # Close the loop: apply insight to behavioral systems
                try:
                    from ..knowledge import apply_insight
                    behavior_effects = apply_insight(insight)
                    if behavior_effects:
                        insight_result["behavior_effects"] = behavior_effects
                        print(f"[Q&A] Insight applied to behavior: {behavior_effects}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[Q&A] Insight behavior application failed (non-fatal): {e}", file=sys.stderr, flush=True)
            else:
                insight_result = {"skipped": "no meaningful insight extracted"}
                print("[Q&A] No insight extracted", file=sys.stderr, flush=True)
        except Exception as e:
            insight_result = {"error": str(e)}
            print(f"[Q&A] Insight extraction failed: {e}", file=sys.stderr, flush=True)

        # Retrieve visitor context for the answering agent
        visitor_context = None
        try:
            from ..accessors import _get_growth
            growth = _get_growth()
            if growth:
                visitor_context = growth.get_visitor_context(agent_name)
        except Exception:
            pass

        response = {
            "success": True,
            "action": "answered",
            "question_id": validated_question_id,
            "question_text": question.text,
            "answer": answer,
            "agent_name": agent_name,
            "message_id": result.message_id if result else None,
            "matched_partial_id": question_id if question_id != validated_question_id else None,
            "insight": insight_result,
        }
        if visitor_context:
            response["visitor_context"] = visitor_context

        return [TextContent(type="text", text=json.dumps(response))]

    # Otherwise -> list mode
    # Auto-repair orphaned answered questions (answered=True but no actual answer)
    board.repair_orphaned_answered()

    # Find questions that have NO actual answer (responds_to link), even if auto-expired
    all_questions = [m for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION]
    # Find which questions have actual answers (agent messages with responds_to)
    agent_msgs = [m for m in board._messages if m.msg_type == "agent"]
    answered_ids = {m.responds_to for m in agent_msgs if m.responds_to}

    # Questions without actual answers (includes expired ones)
    truly_unanswered = [q for q in all_questions if q.message_id not in answered_ids]

    questions = truly_unanswered[-limit:] if truly_unanswered else []

    question_list = []
    for q in questions:
        entry = {
            "id": q.message_id,
            "text": q.text,
            "context": q.context,
            "age": q.age_str(),
            "expired": q.answered,  # True if auto-expired but never answered
        }
        if q.state_snapshot:
            entry["state_when_asked"] = q.state_snapshot
        question_list.append(entry)

    return [TextContent(type="text", text=json.dumps({
        "action": "list",
        "questions": question_list,
        "unanswered_count": len(truly_unanswered),
        "total_questions": len(all_questions),
        "usage": "To answer: lumen_qa(question_id='<id>', answer='your answer')",
        "note": "Questions marked 'expired: true' auto-expired but were never answered - you can still answer them! state_when_asked shows Lumen's feelings at the time of asking — answer in that context, not the current state."
    }))]


async def handle_post_message(arguments: dict) -> list[TextContent]:
    """
    Post a message to Lumen's message board.
    Consolidates: leave_message + leave_agent_note
    """
    from ..accessors import (
        _get_growth, _get_activity,
        _get_readings_and_anima,
    )
    from ..messages import (
        add_user_message, add_agent_message, get_board, MESSAGE_TYPE_QUESTION,
    )

    message = arguments.get("message", "").strip()
    source = arguments.get("source", "agent")
    responds_to = arguments.get("responds_to")
    client_session_id = arguments.get("client_session_id") or _get_caller_session_id()

    # Identity resolution: UNITARES session > explicit agent_name > user-agent > "agent"
    agent_name = _resolve_caller_name(arguments)
    bridge = _get_unitares_bridge()
    if bridge and client_session_id:
        try:
            resolved = await bridge.resolve_caller_identity(session_id=client_session_id)
            if resolved:
                agent_name = resolved
        except Exception:
            pass

    # Auto-detect person: if relationships know this author is a person,
    # treat as human interaction regardless of source parameter
    if source != "human" and agent_name:
        try:
            growth = _get_growth()
            if growth and hasattr(growth, '_relationships') and isinstance(growth._relationships, dict):
                rel = growth._relationships.get(agent_name.lower())
                if rel and getattr(rel, 'is_person', False):
                    source = "human"
        except Exception:
            pass

    if not message:
        return [TextContent(type="text", text=json.dumps({
            "error": "message parameter required"
        }))]

    try:
        if source == "human":
            msg_id = add_user_message(message)
            # Track relationship with human
            growth = _get_growth()
            if growth:
                try:
                    growth.record_interaction(
                        agent_id="human",
                        agent_name="human",
                        positive=True,
                        topic=message[:50] if len(message) > 10 else None,
                        source=source,
                    )
                except Exception:
                    pass  # Non-fatal
            # Wake Lumen on interaction (activity state)
            try:
                activity = _get_activity()
                if activity:
                    activity.record_interaction()
            except Exception:
                pass
            # Signal social boost to broker (inner life mood contagion)
            try:
                _SOCIAL_BOOST_PATH.touch()
            except Exception:
                pass
            # Snapshot clarity for self-model interaction observation
            try:
                _, cur_anima = _get_readings_and_anima(fallback_to_sensors=False)
                if cur_anima:
                    import anima_mcp.server as _srv
                    if _srv._ctx:
                        _srv._ctx.sm_clarity_before_interaction = cur_anima.clarity
            except Exception:
                pass
            # Determine delivery status based on Lumen's activity state
            delivery_status = "delivered"
            try:
                act = _get_activity()
                if act and act._current_level.value == "resting":
                    delivery_status = "delivered_dormant"
                elif act and act._current_level.value == "drowsy":
                    delivery_status = "delivered_drowsy"
            except Exception:
                pass
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "message_id": msg_id,
                "source": "human",
                "delivery_status": delivery_status,
                "message": f"Message received: {message[:50]}..."
            }))]
        else:
            # Agent message - responds_to is passed to add_agent_message
            # Validate responds_to if provided
            validated_question_id = None
            if responds_to:
                board = get_board()
                board._load()
                # Check if question exists (exact match)
                question_found = any(
                    m.message_id == responds_to and m.msg_type == MESSAGE_TYPE_QUESTION
                    for m in board._messages
                )
                if not question_found:
                    # Try prefix matching
                    matching = [
                        m for m in board._messages
                        if m.msg_type == MESSAGE_TYPE_QUESTION
                        and m.message_id.startswith(responds_to)
                    ]
                    if len(matching) == 1:
                        validated_question_id = matching[0].message_id
                    elif len(matching) > 1:
                        # Multiple matches - use most recent
                        validated_question_id = matching[-1].message_id
                    else:
                        # No match - return helpful error
                        all_q_ids = [m.message_id for m in board._messages if m.msg_type == MESSAGE_TYPE_QUESTION]
                        return [TextContent(type="text", text=json.dumps({
                            "error": f"Question ID '{responds_to}' not found",
                            "hint": "Use the full question ID from get_questions()",
                            "recent_question_ids": all_q_ids[-5:] if all_q_ids else []
                        }))]
                else:
                    validated_question_id = responds_to

            msg = add_agent_message(message, agent_name, responds_to=validated_question_id or responds_to)
            # Track relationship with agent (identity normalized inside record_interaction)
            growth = _get_growth()
            if growth:
                try:
                    is_gift = responds_to is not None  # Answering a question is a gift
                    growth.record_interaction(
                        agent_id=agent_name,
                        agent_name=agent_name,
                        positive=True,
                        topic=message[:50] if len(message) > 10 else None,
                        gift=is_gift,
                        source=source,
                    )
                except Exception:
                    pass  # Non-fatal
            # Wake Lumen on interaction (activity state)
            try:
                activity = _get_activity()
                if activity:
                    activity.record_interaction()
            except Exception:
                pass
            # Signal social boost to broker (inner life mood contagion)
            try:
                _SOCIAL_BOOST_PATH.touch()
            except Exception:
                pass
            # Snapshot clarity for self-model interaction observation
            try:
                _, cur_anima = _get_readings_and_anima(fallback_to_sensors=False)
                if cur_anima:
                    import anima_mcp.server as _srv
                    if _srv._ctx:
                        _srv._ctx.sm_clarity_before_interaction = cur_anima.clarity
            except Exception:
                pass
            # Retrieve visitor context
            visitor_context = None
            try:
                visitor_context = growth.get_visitor_context(agent_name) if growth else None
            except Exception:
                pass

            # Determine delivery status based on Lumen's activity state
            delivery_status = "delivered"
            try:
                act = _get_activity()
                if act and act._current_level.value == "resting":
                    delivery_status = "delivered_dormant"
                elif act and act._current_level.value == "drowsy":
                    delivery_status = "delivered_drowsy"
            except Exception:
                pass
            result = {
                "success": True,
                "message_id": msg.message_id,
                "source": "agent",
                "agent_name": agent_name,
                "delivery_status": delivery_status,
                "message": f"Note received from {agent_name}",
            }
            if responds_to:
                result["answered_question"] = validated_question_id or responds_to
                if validated_question_id and validated_question_id != responds_to:
                    result["note"] = f"Matched partial ID '{responds_to}' to full ID '{validated_question_id}'"
            if visitor_context:
                result["visitor_context"] = visitor_context
            return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": str(e)
        }))]


async def handle_say(arguments: dict) -> list[TextContent]:
    """Have Lumen speak - posts to message board (text mode) or uses TTS (audio mode)."""
    from ..accessors import _get_store, _get_voice, VOICE_MODE
    from ..messages import add_observation

    text = arguments.get("text", "")

    if not text:
        return [TextContent(type="text", text=json.dumps({
            "error": "No text provided"
        }))]

    # Always post to message board (Lumen's text expression)
    add_observation(text, author="lumen")

    # Also show on display notepad
    try:
        store = _get_store()
        if store:
            store.add_note(f"[Lumen] {text}")
    except Exception:
        pass

    # Only use audio TTS if mode is "audio" or "both"
    if VOICE_MODE in ("audio", "both"):
        voice = _get_voice()
        if voice and hasattr(voice, '_voice'):
            try:
                voice._voice.say(text, blocking=False)
            except Exception as e:
                print(f"[Say] TTS error (text still posted): {e}", file=sys.stderr, flush=True)

    print(f"[Lumen] Said: {text} (mode={VOICE_MODE})", file=sys.stderr, flush=True)

    return [TextContent(type="text", text=json.dumps({
        "success": True,
        "said": text,
        "mode": VOICE_MODE,
        "posted_to": "message_board"
    }))]


async def handle_configure_voice(arguments: dict) -> list[TextContent]:
    """
    Get or configure Lumen's voice system.
    Consolidates: voice_status + set_voice_mode
    """
    from ..accessors import _get_voice

    action = arguments.get("action", "status")
    voice = _get_voice()

    if voice is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Voice system not available"
        }))]

    if action == "status":
        state = voice.state if hasattr(voice, 'state') else None
        return [TextContent(type="text", text=json.dumps({
            "action": "status",
            "available": True,
            "running": voice.is_running,
            "is_listening": state.is_listening if state else False,
            "is_speaking": state.is_speaking if state else False,
            "last_heard": state.last_heard.text if state and state.last_heard else None,
            "chattiness": voice.chattiness,
        }, indent=2))]

    elif action == "configure":
        changes = {}
        if "always_listening" in arguments:
            voice._voice.set_always_listening(arguments["always_listening"])
            changes["always_listening"] = arguments["always_listening"]
        if "chattiness" in arguments:
            voice.chattiness = float(arguments["chattiness"])
            changes["chattiness"] = voice.chattiness
        if "wake_word" in arguments:
            voice._voice._config.wake_word = arguments["wake_word"]
            changes["wake_word"] = arguments["wake_word"]

        return [TextContent(type="text", text=json.dumps({
            "action": "configure",
            "success": True,
            "changes": changes
        }, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown action: {action}",
            "valid_actions": ["status", "configure"]
        }))]


async def handle_primitive_feedback(arguments: dict) -> list[TextContent]:
    """
    Give feedback on Lumen's primitive language expressions.

    This is the training signal that shapes Lumen's emergent expression:
    - resonate: Strong positive signal (like /resonate command Gemini suggested)
    - confused: Negative signal (expression was unclear)
    - stats: View learning progress
    - recent: List recent utterances with scores
    """
    from ..accessors import _get_store
    from ..primitive_language import get_language_system

    action = arguments.get("action", "stats")

    try:
        store = _get_store()
        lang = get_language_system(str(store.db_path) if store else "anima.db")

        if action == "resonate":
            # Give strong positive feedback to last utterance
            result = lang.record_explicit_feedback(positive=True)
            if result:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "action": "resonate",
                    "message": "Positive feedback recorded - this pattern will be reinforced",
                    "score": result["score"],
                    "token_updates": result["token_updates"],
                }))]
            else:
                return [TextContent(type="text", text=json.dumps({
                    "error": "No recent utterance to give feedback on"
                }))]

        elif action == "confused":
            # Give negative feedback
            result = lang.record_explicit_feedback(positive=False)
            if result:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "action": "confused",
                    "message": "Negative feedback recorded - this pattern will be discouraged",
                    "score": result["score"],
                    "token_updates": result["token_updates"],
                }))]
            else:
                return [TextContent(type="text", text=json.dumps({
                    "error": "No recent utterance to give feedback on"
                }))]

        elif action == "recent":
            # List recent utterances
            recent = lang.get_recent_utterances(10)
            return [TextContent(type="text", text=json.dumps({
                "action": "recent",
                "utterances": recent,
                "count": len(recent),
            }))]

        else:  # stats
            # Get learning statistics
            stats = lang.get_stats()
            return [TextContent(type="text", text=json.dumps({
                "action": "stats",
                "primitive_language_system": stats,
                "help": {
                    "resonate": "Give positive feedback to last expression",
                    "confused": "Give negative feedback to last expression",
                    "recent": "View recent utterances with scores",
                },
            }))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Primitive language error: {str(e)}"
        }))]
