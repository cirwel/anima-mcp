"""Live server wiring for causal agency selection, execution, and learning."""

from __future__ import annotations

import logging
import random
import sys
from typing import Any, Optional

from .accessors import _get_last_shm_data, _get_metacog_monitor
from .agency import Action, ActionType, build_agency_inputs, get_action_selector
from .loop_phases import generate_experiential_question, generate_learned_question
from .server_state import SCHEMA_LOG_THROTTLE

logger = logging.getLogger("anima.server")


def _record_question_evidence(
    *,
    supports: bool,
    surprise_level: float,
    source: str,
) -> None:
    """Best-effort handoff to the broker-owned self-model."""
    try:
        from .learning_events import enqueue_self_belief_evidence

        strength = (
            max(0.05, min(1.0, surprise_level))
            if supports else min(1.0, surprise_level * 0.5)
        )
        enqueue_self_belief_evidence(
            "question_asking_tendency",
            supports=supports,
            strength=strength,
            source=source,
        )
    except Exception as exc:
        logger.debug("[SelfModel] agency question evidence enqueue error: %s", exc)


def _choose_question(
    action: Action,
    surprise_sources: list[str],
    surprise_level: float,
) -> Optional[str]:
    """Choose a grounded agency question without repeating recent prompts."""
    if surprise_sources and surprise_level > 0.2:
        question = generate_experiential_question(surprise_sources, surprise_level)
        if question:
            return question

    question = generate_learned_question()
    if question:
        return question

    if not action.motivation:
        return None

    from .messages import get_recent_questions

    motivation = action.motivation.lower().replace("curious about ", "")
    fallback_templates = [
        "what would help me feel more grounded?",
        "what does this moment have that the last one didn't?",
        "what am I feeling right now, and why?",
        "what connects all these changes?",
    ]
    if motivation.strip():
        fallback_templates.insert(0, f"why do I notice {motivation} right now?")
    recent_texts = {
        item.get("text", "").lower()
        for item in get_recent_questions(hours=24)
    }
    available = [
        prompt for prompt in fallback_templates
        if prompt.lower() not in recent_texts
    ]
    return random.choice(available) if available else None


def execute_action(
    action: Action,
    *,
    ctx: Any,
    action_selector: Any,
    anima: Any,
    readings: Any,
    prediction_error: Any,
    metacog: Any,
    surprise_level: float,
    surprise_sources: list[str],
) -> Action:
    """Execute one selected action and annotate its observable outcome."""
    action.execution_succeeded = False
    action.execution_detail = "no actuator handled the action"

    if action.action_type == ActionType.ASK_QUESTION:
        from .messages import add_question

        question = _choose_question(action, surprise_sources, surprise_level)
        if not question:
            action.execution_detail = "no grounded question available"
            if surprise_level > 0.2:
                _record_question_evidence(
                    supports=False,
                    surprise_level=surprise_level,
                    source="server:agency_question_unavailable",
                )
            print("[Agency] Skipped (no questions available)", file=sys.stderr, flush=True)
            return action

        result = add_question(
            question,
            author="lumen",
            context=f"agency: {action.action_type.value}",
        )
        if result:
            action.execution_succeeded = True
            action.execution_detail = "question posted"
            _record_question_evidence(
                supports=True,
                surprise_level=surprise_level,
                source="server:agency_question_posted",
            )
            print(f"[Agency] Asked: {question}", file=sys.stderr, flush=True)
        else:
            action.execution_detail = "message board suppressed question"
            if surprise_level > 0.2:
                _record_question_evidence(
                    supports=False,
                    surprise_level=surprise_level,
                    source="server:agency_question_suppressed",
                )

    elif action.action_type == ActionType.FOCUS_ATTENTION:
        sensor = action.parameters.get("sensor")
        if sensor and metacog and prediction_error:
            action_selector.set_attention_focus(sensor)
            metacog.record_curiosity([sensor], prediction_error)
            action.execution_succeeded = True
            action.execution_detail = f"curiosity tracking focused on {sensor}"
            print(f"[Agency] Focusing attention on: {sensor}", file=sys.stderr, flush=True)
        else:
            action.execution_detail = "focus lacked a sensor or prediction error"

    elif action.action_type == ActionType.ADJUST_SENSITIVITY:
        direction = action.parameters.get("direction", "increase")
        if direction in {"increase", "decrease"}:
            action_selector.adjust_sensitivity(direction)
            action.execution_succeeded = True
            action.execution_detail = f"sensitivity {direction} applied"
            print(f"[Agency] Adjusted sensitivity: {direction}", file=sys.stderr, flush=True)
        else:
            action.execution_detail = f"invalid sensitivity direction: {direction}"

    elif action.action_type == ActionType.LED_BRIGHTNESS:
        direction = action.parameters.get("direction")
        if direction not in {"increase", "decrease"}:
            action.execution_detail = f"invalid LED direction: {direction}"
        elif ctx.leds and ctx.leds.is_available():
            current = getattr(ctx.leds, "_brightness", 0.1)
            target = (
                min(0.12, current + 0.05)
                if direction == "increase" else max(0.02, current - 0.05)
            )
            if target != current:
                ctx.leds.set_brightness(target)
                action.execution_succeeded = True
                action.execution_detail = "LED brightness changed"
                print(
                    f"[Agency] LED brightness: {current:.2f} → {target:.2f} ({direction})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                action.execution_detail = "LED brightness already at bound"
        else:
            action.execution_detail = "LED actuator unavailable"

    elif action.action_type == ActionType.REQUEST_REFLECTION:
        if metacog:
            reflection = metacog.trigger_manual_reflection(anima, readings)
            action.execution_succeeded = reflection is not None
            action.execution_detail = (
                "reflection recorded" if reflection is not None
                else "reflection monitor returned no result"
            )

    elif action.action_type == ActionType.STAY_QUIET:
        action.execution_succeeded = True
        action.execution_detail = "quiet action observed"

    return action


def run_agency_phase(
    ctx: Any,
    anima: Any,
    readings: Any,
    prediction_error: Any,
    loop_count: int,
) -> Action:
    """Run one live agency cycle from prior-outcome learning through execution."""
    from .preferences import get_preference_system
    from .weighted_pathways import get_weighted_pathways

    db_path = str(ctx.store.db_path) if ctx and ctx.store else "anima.db"
    selector = get_action_selector(db_path=db_path)
    preferences = get_preference_system()
    pathways = get_weighted_pathways(db_path=db_path)

    surprise_level = prediction_error.surprise if prediction_error else 0.0
    surprise_sources = (
        prediction_error.surprise_sources
        if prediction_error and hasattr(prediction_error, "surprise_sources") else []
    )
    current_state = {
        "warmth": anima.warmth,
        "clarity": anima.clarity,
        "stability": anima.stability,
        "presence": anima.presence,
        "last_surprise": surprise_level,
    }

    if ctx.last_action is not None and ctx.last_state_before is not None:
        prior_action = ctx.last_action
        prior_pathway_context = ctx.last_pathway_context
        outcome = selector.record_outcome(
            action=prior_action,
            state_before=ctx.last_state_before,
            state_after=current_state,
            preference_satisfaction_before=preferences.get_overall_satisfaction(
                ctx.last_state_before
            ),
            preference_satisfaction_after=preferences.get_overall_satisfaction(
                current_state
            ),
            surprise_after=surprise_level,
        )
        # The outcome has been durably consumed by the selector. Clear it before
        # optional secondary learning so a pathway failure cannot replay it.
        ctx.last_action = None
        ctx.last_state_before = None
        ctx.last_pathway_context = None
        if prior_pathway_context:
            try:
                pathways.reinforce(
                    prior_pathway_context,
                    prior_action.action_type.value,
                    max(-1.0, min(1.0, outcome.reward)),
                )
            except Exception as exc:
                logger.warning("[Agency] Pathway reinforcement failed: %s", exc)

    conflict_rates = None
    if ctx.tension_tracker:
        conflict_rates = {
            action_type.value: rate
            for action_type in ActionType
            if (rate := ctx.tension_tracker.get_conflict_rate(action_type.value)) > 0
        }

    metacog = _get_metacog_monitor()
    self_model = None
    try:
        from .self_model import get_self_model

        self_model = get_self_model(read_only=True)
    except Exception:
        pass
    led_available = bool(ctx.leds and ctx.leds.is_available())
    inputs = build_agency_inputs(
        current_state=current_state,
        surprise_level=surprise_level,
        surprise_sources=surprise_sources,
        preferences=preferences,
        shm_data=_get_last_shm_data(),
        self_model=self_model,
        pathways=pathways,
        can_focus=metacog is not None and prediction_error is not None,
        can_reflect=metacog is not None,
        can_led=led_available,
        can_speak=False,
    )
    pathway_context = inputs.pop("pathway_context")
    inputs.pop("activity")
    action = selector.select_action(
        current_state=current_state,
        surprise_level=surprise_level,
        surprise_sources=surprise_sources,
        can_speak=False,
        conflict_rates=conflict_rates or None,
        **inputs,
    )
    try:
        execute_action(
            action,
            ctx=ctx,
            action_selector=selector,
            anima=anima,
            readings=readings,
            prediction_error=prediction_error,
            metacog=metacog,
            surprise_level=surprise_level,
            surprise_sources=surprise_sources,
        )
    except Exception as exc:
        action.execution_succeeded = False
        action.execution_detail = f"actuator error: {type(exc).__name__}: {exc}"
        if action.action_type == ActionType.ASK_QUESTION and surprise_level > 0.2:
            _record_question_evidence(
                supports=False,
                surprise_level=surprise_level,
                source="server:agency_question_actuator_error",
            )
        logger.exception("[Agency] Action execution failed")

    if loop_count % SCHEMA_LOG_THROTTLE == 0:
        stats = selector.get_action_stats()
        print(
            f"[Agency] Stats: {stats.get('action_counts', {})} "
            f"explore_rate={selector._exploration_rate:.2f}",
            file=sys.stderr,
            flush=True,
        )

    ctx.last_action = action
    ctx.last_state_before = current_state.copy()
    ctx.last_pathway_context = pathway_context
    return action
