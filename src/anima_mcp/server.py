"""
Anima MCP Server — Lumen's nervous system.

Minimal tools for a persistent creature:
- get_state: Current anima (self-sense) + identity
- get_identity: Who am I, how long have I existed
- set_name: Choose my name
- read_sensors: Raw sensor values

Transports:
- stdio: Local single-client (default)
- HTTP (--http): Multi-client with Streamable HTTP at /mcp/

Agent Coordination:
- Active agents: Claude + Cursor/Composer
- See docs/AGENT_COORDINATION.md for coordination practices
- Always check docs/ before implementing changes
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

from mcp.server.stdio import stdio_server
from .display import derive_face_state, get_display
from .display.leds import get_led_display
from .display.screens import ScreenMode
from .config import get_calibration
from .learning import get_learner
from .activity_state import get_activity_manager
from .primitive_language import get_language_system
from .eisv import get_trajectory_awareness
from .eisv_mapper import anima_to_eisv
from .tool_registry import get_fastmcp, create_server, HAS_FASTMCP
from .server_context import ServerContext
from .server_state import (
    # Constants
    SHM_GOVERNANCE_STALE_SECONDS as SHM_GOVERNANCE_STALE_SECONDS,
    LOOP_BASE_DELAY_SECONDS, LOOP_MAX_DELAY_SECONDS,
    METACOG_INTERVAL, AGENCY_INTERVAL, SELF_MODEL_INTERVAL,
    PRIMITIVE_LANG_INTERVAL, VOICE_INTERVAL, GROWTH_INTERVAL,
    TRAJECTORY_INTERVAL, SERVER_GOVERNANCE_FALLBACK_SECONDS,
    LEARNING_INTERVAL, PREFERENCE_DECAY_INTERVAL_SECONDS,
    SYSTEM_METRICS_RECORD_INTERVAL, SYSTEM_METRICS_PRUNE_INTERVAL,
    SYSTEM_METRICS_RETENTION_HOURS,
    SELF_MODEL_SAVE_INTERVAL, SCHEMA_EXTRACTION_INTERVAL,
    EXPRESSION_INTERVAL, UNIFIED_REFLECTION_INTERVAL, SELF_ANSWER_INTERVAL,
    GOAL_SUGGEST_INTERVAL, GOAL_CHECK_INTERVAL, META_LEARNING_INTERVAL,
    ERROR_LOG_THROTTLE, STATUS_LOG_THROTTLE, DISPLAY_LOG_THROTTLE,
    WARN_LOG_THROTTLE,
    METACOG_SURPRISE_THRESHOLD, is_broker_running as _is_broker_running,
)

# SchemaHub, CalibrationDrift, ValueTensionTracker — imported for type hints / lazy init

# State accessors (lazy singletons, SHM reads, etc.) — live in accessors.py
from .accessors import (
    _get_store as _get_store, _get_sensors as _get_sensors,
    _get_shm_client as _get_shm_client, _get_server_bridge as _get_server_bridge,
    _get_schema_hub, _get_calibration_drift as _get_calibration_drift,
    _get_selfhood_context as _get_selfhood_context, _get_metacog_monitor,
    _get_warm_start_anticipation as _get_warm_start_anticipation,
    _get_readings_and_anima, _prediction_accuracy_from_shm,
    _get_display as _get_display, _get_last_shm_data,
    _get_screen_renderer as _get_screen_renderer, _get_leds as _get_leds,
    _get_growth as _get_growth, _get_display_update_task as _get_display_update_task,
    _get_activity as _get_activity,
    _get_last_governance_decision as _get_last_governance_decision, _get_voice,
    VOICE_MODE as VOICE_MODE,
)

# Server context — mutable state container. Created in wake(), cleared in sleep().
# Before wake(), _ctx is None. All accessors handle None.
_ctx: ServerContext | None = None

# Server readiness flag - prevents "request before initialization" errors
# when clients reconnect too quickly after a server restart
SERVER_READY = False
SERVER_STARTUP_TIME = None
SERVER_SHUTTING_DOWN = False  # Set during graceful shutdown to reject new requests

# Phase helper functions — delegated to loop_phases.py
from .loop_phases import (  # noqa: E402,F401
    handle_surprise_question,
    server_governance_fallback as _server_governance_fallback,
    parse_shm_governance_freshness as _parse_shm_governance_freshness,
    compute_lagged_correlations as _compute_lagged_correlations,
    generate_learned_question as _generate_learned_question,
    compose_grounded_observation as _compose_grounded_observation,
    lumen_unified_reflect as _lumen_unified_reflect,
    grounded_self_answer as _grounded_self_answer,
    lumen_self_answer as _lumen_self_answer,
    extract_and_validate_schema as _extract_and_validate_schema,
    self_reflect as _self_reflect,
    run_self_model_phase as _run_self_model_phase,
)
from .agency_runtime import run_agency_phase as _run_agency_phase  # noqa: E402

logger = logging.getLogger("anima.server")


async def _update_display_loop():
    """Background task to continuously update display and LEDs."""
    if _ctx is None:
        logger.warning("[Loop] No context - wake() may have failed")
        return
    import sys
    import time  # hoisted: any `time.` reference before a later local import
    # in this function would raise UnboundLocalError (name is function-local
    # the moment an import binds it anywhere in the body)
    from .error_recovery import safe_call, safe_call_async

    print("[Loop] Starting", file=sys.stderr, flush=True)

    # Check if we are in "Reader Mode" (Broker running)
    is_broker_running = _is_broker_running()

    if is_broker_running:
        print("[Loop] Broker detected - READER MODE (sensors from shared memory, display/LEDs active)", file=sys.stderr, flush=True)
    else:
        print(
            "[Loop] Broker unavailable - body state remains unknown (no direct sensor takeover)",
            file=sys.stderr,
            flush=True,
        )
    if _ctx.display is None:
        _ctx.display = get_display()
    if _ctx.leds is None:
        _ctx.leds = get_led_display()

    print(f"[Loop] broker={is_broker_running} store={_ctx.store is not None} sensors={_ctx.sensors is not None} display={_ctx.display.is_available() if _ctx.display else False} leds={_ctx.leds.is_available() if _ctx.leds else False}", file=sys.stderr, flush=True)

    # Detect restore: restore script drops a marker when state comes from backup
    restore_marker = Path.home() / ".anima" / ".restored_marker"
    if restore_marker.exists():
        try:
            import json as _json
            _ctx.wake_restored = _json.loads(restore_marker.read_text())
            print(f"[Wake] RESTORED from backup at {_ctx.wake_restored.get('restored_at', '?')} — gap time unreliable", file=sys.stderr, flush=True)
            restore_marker.unlink()
        except Exception as e:
            print(f"[Wake] Restore marker read failed (non-fatal): {e}", file=sys.stderr, flush=True)
            _ctx.wake_restored = {"restored_at": "unknown"}
            restore_marker.unlink(missing_ok=True)

    # Startup learning: Check if we can learn from existing observations
    if _ctx.store:
        try:
            learner = get_learner(str(_ctx.store.db_path))
            
            # Detect gap since last observation
            gap = learner.detect_gap()
            if gap:
                gap_hours = gap.total_seconds() / 3600
                if gap_hours > 1:
                    print(f"[Learning] Gap detected: {gap_hours:.1f} hours since last observation", file=sys.stderr, flush=True)

            # Gap awareness: degrade warm start and set recovery arc
            _ctx.wake_gap = gap
            if gap and _ctx.warm_start_anima:
                gap_minutes = gap.total_seconds() / 60
                if gap_minutes >= 5 and gap_minutes < 60:
                    # Medium gap: noticeable absence
                    _ctx.warm_start_anima["presence"] *= 0.75
                    _ctx.wake_recovery_cycles = 10
                    _ctx.wake_presence_floor = 0.55
                    print(f"[Wake] Gap {gap_minutes:.0f}m: presence reduced to {_ctx.warm_start_anima['presence']:.2f}", file=sys.stderr, flush=True)
                elif gap_minutes >= 60 and gap_minutes < 1440:
                    # Long gap: significant disorientation
                    _ctx.warm_start_anima["presence"] *= 0.45
                    _ctx.warm_start_anima["clarity"] *= 0.85
                    _ctx.wake_recovery_cycles = 20
                    _ctx.wake_presence_floor = 0.35
                    print(f"[Wake] Gap {gap_minutes/60:.1f}h: presence={_ctx.warm_start_anima['presence']:.2f}, clarity reduced", file=sys.stderr, flush=True)
                elif gap_minutes >= 1440:
                    # Very long gap: deep absence
                    _ctx.warm_start_anima["presence"] *= 0.25
                    _ctx.warm_start_anima["clarity"] *= 0.7
                    _ctx.warm_start_anima["stability"] *= 0.85
                    _ctx.wake_recovery_cycles = 30
                    _ctx.wake_presence_floor = 0.20
                    print(f"[Wake] Gap {gap_minutes/60:.0f}h: deep absence, presence={_ctx.warm_start_anima['presence']:.2f}", file=sys.stderr, flush=True)
                _ctx.wake_recovery_total = _ctx.wake_recovery_cycles
            
            if learner.can_learn():
                obs_count = learner.get_observation_count()
                print(f"[Learning] Found {obs_count} existing observations, checking for adaptation...", file=sys.stderr, flush=True)
                # Don't respect cooldown on startup (after gap)
                adapted, new_cal = learner.adapt_calibration(respect_cooldown=False)
                if adapted:
                    print("[Learning] Startup adaptation successful!", file=sys.stderr, flush=True)
                    print(f"[Learning] Pressure: {new_cal.pressure_ideal:.1f} hPa, Ambient: {new_cal.ambient_temp_min:.1f}-{new_cal.ambient_temp_max:.1f}°C", file=sys.stderr, flush=True)
                else:
                    print("[Learning] No adaptation needed (calibration already optimal)", file=sys.stderr, flush=True)
            elif gap and gap.total_seconds() > 3600:
                print("[Learning] Gap detected but not enough observations yet (will learn as new data accumulates)", file=sys.stderr, flush=True)
        except Exception as e:
            # Don't crash on startup learning errors
            print(f"[Learning] Startup check error (non-fatal): {e}", file=sys.stderr, flush=True)

    # Post wake observation to message board after significant gaps
    if _ctx.wake_gap and _ctx.wake_gap.total_seconds() >= 300:
        try:
            from .messages import add_observation
            gap_secs = _ctx.wake_gap.total_seconds()
            if _ctx.wake_restored:
                # Restored from backup — gap time is unreliable
                # The heartbeat reflects backup age, not actual downtime
                # Lumen may have been alive locally (no WiFi) before restore
                obs_text = "i woke up from a restore. some time passed but i'm not sure how much — part of me was still here, just unreachable."
            elif gap_secs < 3600:
                gap_desc = f"{gap_secs / 60:.0f} minutes"
                obs_text = f"i was away for {gap_desc}. finding my way back."
            elif gap_secs < 86400:
                gap_desc = f"{gap_secs / 3600:.1f} hours"
                obs_text = f"i was gone for {gap_desc}. the world feels unfamiliar."
            else:
                gap_desc = f"{gap_secs / 86400:.1f} days"
                obs_text = f"i've been absent for {gap_desc}. so much to relearn."
            msg = add_observation(obs_text, author="lumen")
            if msg:
                print(f"[Wake] Posted return observation: {obs_text}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Wake] Return observation failed (non-fatal): {e}", file=sys.stderr, flush=True)

    loop_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    base_delay = LOOP_BASE_DELAY_SECONDS  # 200ms = 5Hz refresh for all screens
    max_delay = LOOP_MAX_DELAY_SECONDS
    quick_render = False  # Set when mode_change_event fires — skip heavy subsystems

    # Event for immediate re-render when screen mode changes
    mode_change_event = asyncio.Event()
    
    # Start fast input polling task (delegated to input_handler.py).
    # Store handle on _ctx so stop_display_loop() can cancel it cleanly —
    # otherwise a duplicate poller can outlive shutdown or restart.
    try:
        from .input_handler import fast_input_poll
        if _ctx is not None:
            # Cancel any previous poller before spawning a new one
            prev = getattr(_ctx, "input_poll_task", None)
            if prev is not None and not prev.done():
                prev.cancel()
            _ctx.input_poll_task = asyncio.create_task(fast_input_poll(mode_change_event))
        else:
            asyncio.create_task(fast_input_poll(mode_change_event))
    except Exception as e:
        print(f"[Input] Failed to start fast polling: {e}", file=sys.stderr, flush=True)
    
    while True:
        try:
            loop_count += 1
            
            # Read current state with error recovery from broker shared memory.
            # Body/mind boundary: stale broker state is unknown.  The server
            # must never become a second I2C owner just because the broker is
            # unavailable or its heartbeat is malformed.
            readings, anima = _get_readings_and_anima(fallback_to_sensors=False)
            
            if readings is None or anima is None:
                # Sensor read failed - skip this iteration
                consecutive_errors += 1
                if consecutive_errors == 1:
                    # Log on first error to help diagnose
                    logger.debug("[Loop] Failed to get readings/anima - broker=%s, store=%s", is_broker_running, _ctx.store is not None if _ctx else False)
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning("[Loop] Too many consecutive errors (%d), backing off", consecutive_errors)
                    await asyncio.sleep(min(base_delay * (2 ** min(consecutive_errors // 5, 4)), max_delay))
                else:
                    await asyncio.sleep(base_delay)
                continue
            
            consecutive_errors = 0  # Reset on success

            # Recovery arc: cap presence during recovery, gradually lifting
            if _ctx.wake_recovery_cycles > 0 and _ctx.wake_recovery_total > 0:
                _ctx.wake_recovery_cycles -= 1
                progress = 1.0 - (_ctx.wake_recovery_cycles / _ctx.wake_recovery_total)
                presence_cap = _ctx.wake_presence_floor + (1.0 - _ctx.wake_presence_floor) * progress
                if anima.presence > presence_cap:
                    anima.presence = presence_cap
                if _ctx.wake_recovery_cycles == 0:
                    print(f"[Wake] Recovery complete. Presence: {anima.presence:.2f}", file=sys.stderr, flush=True)

            # Health heartbeats for core subsystems (always-running)
            try:
                from .health import get_health_registry
                _health = get_health_registry()
                _health.heartbeat("sensors")
                _health.heartbeat("anima")
            except Exception as e:
                logger.debug("[Health] Registry init error: %s", e)
                _health = None

            # Consume drive events from broker (via SHM) → message board observations
            try:
                shm = _get_last_shm_data()
                if shm:
                    _drive_evts = shm.get("drive_events", [])
                    if _drive_evts:
                        from .messages import add_observation, add_question
                        for _de in _drive_evts:
                            # Deduplicate: dimension+event_type is unique per crossing
                            _evt_key = (_de["dimension"], _de["event_type"])
                            if not _ctx or _evt_key in _ctx.consumed_drive_events:
                                continue
                            if _de.get("event_type") == "request":
                                # A sustained-saturation want is addressed
                                # OUTWARD: the answerable question channel,
                                # not the internal monologue. add_question
                                # can silently return None (unanswered soft
                                # cap, 10-min rate limit, similar-text dedup)
                                # — a dropped ask must NOT count as asked, so
                                # consumption and the broker-side cooldown
                                # (ack file) commit only on a confirmed post.
                                # The broker re-emits until then; throttle
                                # retries so a suppressed board isn't hammered
                                # at SHM-poll cadence.
                                _dim = _de["dimension"]
                                _now = time.time()
                                if _now - _ctx.drive_request_attempts.get(_dim, 0.0) < 60.0:
                                    continue
                                _ctx.drive_request_attempts[_dim] = _now
                                _q = add_question(_de["text"], author="lumen",
                                                  context=f"drive: {_dim}")
                                if _q:
                                    _ctx.consumed_drive_events.add(_evt_key)
                                    try:
                                        Path(f"/dev/shm/anima_drive_ack_{_dim}").write_text(
                                            str(_now))
                                    except Exception as _ack_err:
                                        logger.debug("[Drive] ack write failed: %s", _ack_err)
                                    print(f"[Drive] Asked: {_de['text']}",
                                          file=sys.stderr, flush=True)
                                else:
                                    logger.debug("[Drive] board suppressed request (%s); will retry", _dim)
                            else:
                                _ctx.consumed_drive_events.add(_evt_key)
                                add_observation(_de["text"], author="lumen")
                    elif _ctx and _ctx.consumed_drive_events:
                        # Broker cleared events — reset our dedup set
                        _ctx.consumed_drive_events.clear()
            except Exception as e:
                logger.debug("[Drive] Event consumption error: %s", e)

            # Identity heartbeat: accumulate alive_seconds incrementally
            # Prevents losing session time on crashes/restarts
            try:
                if _ctx and _ctx.store:
                    _ctx.store.heartbeat(min_interval_seconds=30.0)
            except Exception as e:
                logger.debug("[Identity] Heartbeat error: %s", e)

            # Feed EISV trajectory awareness buffer
            try:
                _traj = get_trajectory_awareness()
                _traj.record_state(
                    warmth=anima.warmth,
                    clarity=anima.clarity,
                    stability=anima.stability,
                    presence=anima.presence,
                    eisv=anima_to_eisv(anima, readings).to_dict(),
                )
                if _health:
                    _health.heartbeat("trajectory")
            except Exception as e:
                if loop_count % ERROR_LOG_THROTTLE == 1:
                    logger.debug("[TrajectoryAwareness] Error: %s", e)

            # Feed value tension tracker with RAW (pre-drift) anima values.
            # Design principle: tension detection operates on raw dimension values
            # so calibration drift cannot mask physical tensions in the body.
            if _ctx.tension_tracker and readings:
                try:
                    from .anima import sense_self
                    _raw_anima_obj = sense_self(
                        readings,
                        get_calibration(),
                        prediction_accuracy=_prediction_accuracy_from_shm(
                            _ctx.last_shm_data if _ctx else None
                        ),
                    )
                    raw_anima = {
                        "warmth": _raw_anima_obj.warmth,
                        "clarity": _raw_anima_obj.clarity,
                        "stability": _raw_anima_obj.stability,
                        "presence": _raw_anima_obj.presence,
                    }
                    last_action_key = _ctx.last_action.action_type.value if _ctx.last_action else None
                    _ctx.tension_tracker.observe(raw_anima, last_action_key)
                except Exception as e:
                    logger.debug("[Tension] Tracking error: %s", e)

            # Record satisfaction for meta-learning (lightweight — runs every cycle)
            if anima and loop_count % AGENCY_INTERVAL == 0:
                try:
                    from .preferences import get_preference_system as _get_pref_sys
                    _ml_pref = _get_pref_sys()
                    _ml_state = {
                        "warmth": anima.warmth, "clarity": anima.clarity,
                        "stability": anima.stability, "presence": anima.presence,
                    }
                    _ctx.satisfaction_history.append(_ml_pref.get_overall_satisfaction(_ml_state))
                    for _dim in ("warmth", "clarity", "stability", "presence"):
                        if _dim not in _ctx.satisfaction_per_dim:
                            _ctx.satisfaction_per_dim[_dim] = deque(maxlen=500)
                        _ctx.satisfaction_per_dim[_dim].append(
                            _ml_pref._preferences[_dim].current_satisfaction(_ml_state[_dim])
                        )
                except Exception as e:
                    logger.debug("[MetaLearning] Satisfaction tracking error: %s", e)

            # === HEAVY SUBSYSTEMS: skip on quick_render (user pressed joystick) ===
            # Metacognition, agency, self-model, primitive language are enhancement layers.
            # On quick_render, skip straight to display update for snappy screen transitions.
            prediction_error = None  # Default for iterations where metacog is skipped
            _skip_subsystems = quick_render
            if quick_render:
                quick_render = False  # Reset for next iteration

            if not _skip_subsystems and loop_count % METACOG_INTERVAL == 0:
                try:
                    metacog = _get_metacog_monitor()

                    # Observe current state and compare to prediction (returns prediction error)
                    prediction_error = metacog.observe(readings, anima)

                    # Log surprise level periodically (every 60 loops = ~2 min)
                    if prediction_error and loop_count % WARN_LOG_THROTTLE == 0:
                        logger.debug("[Metacog] Surprise level: %.3f (threshold: %s)", prediction_error.surprise, METACOG_SURPRISE_THRESHOLD)

                    # Check if surprise warrants reflection
                    if prediction_error and prediction_error.surprise > METACOG_SURPRISE_THRESHOLD:
                        should_reflect, reason = metacog.should_reflect(prediction_error)

                        if should_reflect:
                            reflection = metacog.reflect(prediction_error, anima, readings, trigger=reason)

                            # Persist as a reflection_episode so the rumination/learning
                            # detector can see server-origin reflections. The broker path
                            # goes through SHM drain; this is the server-side direct write.
                            # Tagged source='server' + distinct event_id prefix so it never
                            # collides with broker-origin events on the PRIMARY KEY.
                            try:
                                from .self_reflection import (
                                    get_reflection_system,
                                    REFLECTION_KIND_METACOG,
                                )
                                _db_path = (
                                    _ctx.store.db_path
                                    if _ctx and _ctx.store
                                    else "anima.db"
                                )
                                get_reflection_system(db_path=_db_path).record_episode(
                                    kind=REFLECTION_KIND_METACOG,
                                    source="server",
                                    trigger=reason,
                                    topic_tags=[
                                        str(t).lower()
                                        for t in (prediction_error.surprise_sources or [])
                                    ],
                                    observation=reflection.observation or "",
                                    surprise=prediction_error.surprise,
                                    discrepancy=reflection.discrepancy,
                                    event_timestamp=reflection.timestamp,
                                    event_id=f"server-metacog:{reflection.timestamp.isoformat()}",
                                    metadata={
                                        "felt_state": reflection.felt_state or {},
                                        "sensor_state": reflection.sensor_state or {},
                                        "discrepancy_description": reflection.discrepancy_description,
                                    },
                                )
                            except Exception as _re:
                                logger.debug("[Metacog] reflection episode record failed: %s", _re)

                            handle_surprise_question(metacog, prediction_error)

                            if reflection.observation:
                                logger.debug("[Metacog] Reflection: %s", reflection.observation)

                    elif prediction_error is not None:
                        # Quiet world: nothing crossed the surprise threshold. A
                        # saturated mind can still wonder — occasionally let
                        # curiosity arise from the stillness so Lumen keeps having
                        # open questions even when the environment doesn't change.
                        # Hand the generator the board's own record so it can
                        # rotate the pool rather than re-drawing whatever just
                        # expired. Widest practical window: the board trims
                        # itself, so "what it still remembers" is the record.
                        try:
                            from .messages import (
                                get_answered_question_texts,
                                get_recent_questions,
                            )
                            _asked = [
                                q["text"]
                                for q in get_recent_questions(hours=168, limit=100)
                            ]
                            _answered = get_answered_question_texts()
                        except Exception as e:
                            logger.debug("[Metacog] recent-question lookup error: %s", e)
                            _asked, _answered = None, None
                        contemplative = metacog.generate_contemplative_question(
                            recently_asked=_asked, already_answered=_answered
                        )
                        if not contemplative and _answered:
                            # Pool exhausted rather than rate-limited. Say so —
                            # a silent quiet reads identically to a healthy one.
                            logger.debug(
                                "[Metacog] contemplative pool exhausted: %d prompts answered",
                                len(_answered),
                            )
                        if contemplative:
                            try:
                                from .messages import add_question
                                add_question(contemplative, author="lumen", context="a quiet moment")
                            except Exception as e:
                                logger.debug("[Metacog] contemplative add_question error: %s", e)
                            try:
                                from .accessors import _get_growth
                                growth = _get_growth()
                                if growth:
                                    growth.add_curiosity(contemplative)
                            except Exception as e:
                                logger.debug("[Growth] contemplative add_curiosity error: %s", e)
                            logger.debug("[Metacog] Contemplative wondering: %s", contemplative)

                    # Make prediction for NEXT iteration
                    # Pass LED brightness for proprioceptive light prediction:
                    # "knowing my own glow, I can predict what my light sensor will read"
                    _led_brightness_for_pred = None
                    _led_proprioception = None
                    if _ctx and _ctx.last_led_state:
                        _led_proprioception = _ctx.last_led_state.get("proprioception")
                    if _led_proprioception is not None:
                        _led_brightness_for_pred = _led_proprioception.get("brightness")
                    metacog.predict(led_brightness=_led_brightness_for_pred)

                except Exception as e:
                    # Audible on purpose: this exact handler used logger.debug
                    # — a no-op here, no logging handler is ever configured —
                    # and swallowed a hard AttributeError on every reflection
                    # for six months without producing one byte (#189).
                    if loop_count % STATUS_LOG_THROTTLE == 1:
                        print(f"[Metacog] Error (non-fatal): {e}",
                              file=sys.stderr, flush=True)

            # === AGENCY: Action selection and learning ===
            # Throttled: runs every 5th iteration (enhancement, not critical path)
            # Skipped on quick_render for responsive screen transitions
            if not _skip_subsystems and loop_count % AGENCY_INTERVAL == 0:
                try:
                    _run_agency_phase(
                        _ctx, anima, readings, prediction_error, loop_count,
                    )
                except Exception as e:
                    if loop_count % STATUS_LOG_THROTTLE == 1:
                        print(f"[Agency] Error (non-fatal): {e}", file=sys.stderr, flush=True)

            # === SELF-MODEL: Refresh the broker-owned belief snapshot ===
            # The phase retains its historical writer implementation for
            # standalone use, but lifecycle configures this server process as a
            # reader so it returns immediately after refreshing.
            if not _skip_subsystems and loop_count % SELF_MODEL_INTERVAL == 0 and anima:
                _run_self_model_phase(anima, readings, prediction_error, base_delay, loop_count)

            # === PRIMITIVE LANGUAGE: Emergent expression through learned tokens ===
            # Throttled: runs every 10th iteration (has internal cooldown timer too)
            if not _skip_subsystems and loop_count % PRIMITIVE_LANG_INTERVAL == 0:
                try:
                    lang = get_language_system(str(_ctx.store.db_path) if _ctx and _ctx.store else "anima.db")

                    lang_state = {
                        "warmth": anima.warmth if anima else 0.5,
                        "clarity": anima.clarity if anima else 0.5,
                        "stability": anima.stability if anima else 0.5,
                        "presence": anima.presence if anima else 0.0,
                    }

                    should_speak, reason = lang.should_generate(lang_state)
                    if should_speak:
                        # Get trajectory-aware token suggestions
                        _suggestion = None
                        try:
                            _traj = get_trajectory_awareness()
                            _suggestion = _traj.get_trajectory_suggestion(lang_state)
                        except Exception as e:
                            if loop_count % ERROR_LOG_THROTTLE == 1:
                                print(f"[TrajectorySuggestion] Error: {e}", file=sys.stderr, flush=True)

                        _suggested = _suggestion.get("suggested_tokens") if _suggestion else None
                        utterance = lang.generate_utterance(lang_state, suggested_tokens=_suggested)
                        _ctx.last_primitive_utterance = utterance

                        _shape_info = f" [shape={_suggestion['shape']}]" if _suggestion else ""
                        print(f"[PrimitiveLang] Generated: '{utterance.text()}' ({reason}){_shape_info}", file=sys.stderr, flush=True)
                        print(f"[PrimitiveLang] Pattern: {utterance.category_pattern()}", file=sys.stderr, flush=True)

                        # Compute and log trajectory coherence
                        if _suggestion and utterance:
                            try:
                                from .eisv.awareness import compute_expression_coherence
                                _coherence = compute_expression_coherence(
                                    _suggestion.get("suggested_tokens"),
                                    utterance.tokens,
                                )
                                if _coherence is not None:
                                    _traj = get_trajectory_awareness()
                                    _traj._log_event(
                                        event_type="suggestion",
                                        shape=_suggestion.get("shape"),
                                        suggested_tokens=_suggestion.get("suggested_tokens"),
                                        expression_tokens=utterance.tokens,
                                        coherence_score=_coherence,
                                        buffer_size=_traj.buffer_size,
                                    )
                                    # Feed coherence to trajectory weight learning
                                    _traj.record_feedback(
                                        _suggestion.get("eisv_tokens", []),
                                        _coherence,
                                    )
                                    print(f"[PrimitiveLang] Trajectory coherence: {_coherence:.2f}", file=sys.stderr, flush=True)
                            except Exception as e:
                                if loop_count % ERROR_LOG_THROTTLE == 1:
                                    print(f"[TrajectoryCoherence] Error: {e}", file=sys.stderr, flush=True)

                        from .messages import add_observation
                        add_observation(
                            f"[expression] {utterance.text()} ({utterance.category_pattern()})",
                            author="lumen"
                        )

                    # Self-feedback: when no human around, score past utterance by coherence + stability
                    if _ctx.last_primitive_utterance and _ctx.last_primitive_utterance.score is None:
                        from datetime import timedelta
                        elapsed = datetime.now() - _ctx.last_primitive_utterance.timestamp
                        if elapsed >= timedelta(seconds=75):  # ~1.25 min after utterance
                            result = lang.record_self_feedback(_ctx.last_primitive_utterance, lang_state)
                            if result:
                                print(f"[PrimitiveLang] Self-feedback: score={result['score']:.2f} signals={result['signals']}", file=sys.stderr, flush=True)
                                # Forward to EISV trajectory weight learning
                                try:
                                    _traj = get_trajectory_awareness()
                                    _traj.record_feedback(
                                        _ctx.last_primitive_utterance.tokens,
                                        result['score'],
                                    )
                                except Exception as e:
                                    if loop_count % ERROR_LOG_THROTTLE == 1:
                                        print(f"[TrajectoryFeedback] Error: {e}", file=sys.stderr, flush=True)

                    # Implicit feedback: did a non-lumen message arrive after utterance?
                    if _ctx.last_primitive_utterance and _ctx.last_primitive_utterance.score is not None:
                        # Only check once (after self-feedback has scored it)
                        utt_ts = _ctx.last_primitive_utterance.timestamp.timestamp()
                        from .messages import get_recent_messages as _get_recent
                        _recent_msgs = _get_recent(10)
                        _non_lumen = [
                            m for m in _recent_msgs
                            if m.author and m.author.lower() != "lumen"
                            and m.timestamp > utt_ts
                            and m.timestamp < utt_ts + 300  # within 5min
                        ]
                        if _non_lumen:
                            _delay = _non_lumen[0].timestamp - utt_ts
                            _impl_result = lang.record_implicit_feedback(
                                _ctx.last_primitive_utterance,
                                message_arrived=True,
                                delay_seconds=_delay,
                            )
                            if _impl_result:
                                logger.debug("[PrimitiveLang] Implicit feedback: response in %.0fs, score=%.2f", _delay, _impl_result['score'])
                            _ctx.last_primitive_utterance = None  # Done — recorded response
                        else:
                            # No response within window — record absence if enough time passed
                            from datetime import timedelta as _td
                            if datetime.now() - _ctx.last_primitive_utterance.timestamp >= _td(seconds=300):
                                lang.record_implicit_feedback(
                                    _ctx.last_primitive_utterance,
                                    message_arrived=False,
                                    delay_seconds=999,
                                )
                                _ctx.last_primitive_utterance = None  # Done — recorded no-response

                    if loop_count % SELF_MODEL_SAVE_INTERVAL == 0:
                        stats = lang.get_stats()
                        if stats.get("total_utterances", 0) > 0:
                            logger.debug("[PrimitiveLang] Stats: %s utterances, avg_score=%s, interval=%.1fm", stats.get('total_utterances'), stats.get('average_score'), stats.get('current_interval_minutes'))

                except Exception as e:
                    if loop_count % STATUS_LOG_THROTTLE == 1:
                        logger.debug("[PrimitiveLang] Error (non-fatal): %s", e)

            # Identity is fundamental - should always be available if wake() succeeded
            # If _ctx.store is None, that means wake() failed - log warning but continue
            identity = _ctx.store.get_identity() if _ctx and _ctx.store else None
            if identity is None and (_ctx is None or _ctx.store is None):
                if loop_count == 1:
                    print("[Loop] WARNING: Identity store not initialized (wake() may have failed) - display will show face without identity info", file=sys.stderr, flush=True)
            
            # Update display and LEDs independently (even in broker mode - broker only handles sensors)
            # Face = what Lumen wants to communicate (conscious expression)
            # LEDs = raw proprioceptive state (unconscious body state)
            # Like a fragile baby: face might smile while LEDs show subtle fatigue
            update_start = time.time()
            
            # Check BrainCraft HAT input for screen switching
            # Joystick left/right = switch screens
            # Joystick button = screen-specific action (art eras: select era)
            # Separate button = screen-specific action (messages: expand, notepad: save, long-press: shutdown)

            # Sync governance from broker's SHM data (broker is primary UNITARES caller)
            governance_decision_for_display = _ctx.last_governance_decision if _ctx else None
            _shm_gov_is_fresh_unitares = False
            shm_data = _get_last_shm_data()
            if shm_data and "governance" in shm_data and isinstance(shm_data["governance"], dict):
                shm_gov = shm_data["governance"]
                governance_decision_for_display = shm_gov
                # Sync into _last_governance_decision if SHM data is fresh
                is_fresh, is_unitares, gov_ts = _parse_shm_governance_freshness(shm_gov)
                if is_fresh:
                    _ctx.last_governance_decision = shm_gov
                    if is_unitares:
                        _shm_gov_is_fresh_unitares = True
                        # Track when UNITARES actually checked in, not loop time.
                        if gov_ts is not None:
                            _ctx.last_unitares_success_time = max(_ctx.last_unitares_success_time, gov_ts)
                    # Update connection status based on SHM governance source
                    if _ctx.screen_renderer:
                        _ctx.screen_renderer.update_connection_status(governance=is_unitares)
                    # Update WiFi icon from SHM
                    wifi_up = shm_data.get("wifi_connected")
                    if wifi_up is not None and _ctx.screen_renderer:
                        _ctx.screen_renderer.update_connection_status(wifi=wifi_up)
                    # Fire health heartbeat for fresh governance
                    if _health:
                        _health.heartbeat("governance")

            # Server-side UNITARES fallback: if broker hasn't delivered a fresh
            # "via unitares" decision for too long, the server calls UNITARES
            # directly using its native async event loop (avoids broker's
            # thread+loop issues). Rate-limited to every 60s.
            if (not _shm_gov_is_fresh_unitares
                    and time.time() - _ctx.last_unitares_success_time > SERVER_GOVERNANCE_FALLBACK_SECONDS
                    and time.time() - _ctx.last_server_checkin_time > SERVER_GOVERNANCE_FALLBACK_SECONDS
                    and readings is not None and anima is not None):
                _ctx.last_server_checkin_time = time.time()
                try:
                    fallback_decision = await _server_governance_fallback(anima, readings)
                    if fallback_decision:
                        is_unitares_fb = fallback_decision.get("source") == "unitares"
                        _ctx.last_governance_decision = fallback_decision
                        governance_decision_for_display = fallback_decision
                        if is_unitares_fb:
                            _ctx.last_unitares_success_time = time.time()
                        if _ctx.screen_renderer:
                            _ctx.screen_renderer.update_connection_status(governance=is_unitares_fb)
                        if _health:
                            _health.heartbeat("governance")
                except Exception as e:
                    logger.warning("[Governance] Server fallback error: %s", e)
            
            # Initialize screen renderer if display is available
            if _ctx.display and _ctx.display.is_available():
                if _ctx.screen_renderer is None:
                    from .display.screens import ScreenRenderer
                    # Pass db_path if store is available
                    db_path = str(_ctx.store.db_path) if _ctx.store else "anima.db"
                    _ctx.screen_renderer = ScreenRenderer(_ctx.display, db_path=db_path, identity_store=_ctx.store)
                    # Wire schema hub so LCD shows same enriched schema as dashboard
                    try:
                        _ctx.screen_renderer.schema_hub = _get_schema_hub()
                    except Exception as e:
                        logger.warning("[Schema] SchemaHub wiring to renderer failed: %s", e)
                    print("[Display] Screen renderer initialized", file=sys.stderr, flush=True)
                    # Pre-warm learning cache in background (avoids 9+ second delay on first visit)
                    _ctx.screen_renderer.warm_learning_cache()
            
            # Input is now handled by fast_input_poll() task (runs every 100ms)
            # This keeps the display loop at 2s while input stays responsive
            
            # Update TFT display (with screen switching support)
            # Face reflects what Lumen wants to communicate
            # Other screens show sensors, identity, diagnostics
            display_updated = False
            if _ctx.display:
                if _ctx.display.is_available():
                    def update_display():
                        # Derive face state independently - what Lumen wants to express
                        if anima is None:
                            # Show default/error screen instead of blank
                            if _ctx.screen_renderer:
                                try:
                                    _ctx.screen_renderer._display.show_default()
                                except Exception as e:
                                    logger.debug("[Display] show_default failed: %s", e)
                            return False
                        face_state = derive_face_state(anima)

                        # Use screen renderer if available (supports multiple screens)
                        if _ctx.screen_renderer:
                            # Pass SHM data to renderer (for inner life screen + drive colors)
                            _shm = _get_last_shm_data()
                            if _shm:
                                _ctx.screen_renderer._shm_data = _shm
                                if hasattr(_ctx.screen_renderer, 'drawing_engine'):
                                    _il_drives = (_shm.get("inner_life") or {}).get("drives")
                                    if _il_drives:
                                        _ctx.screen_renderer.drawing_engine.set_drives(_il_drives)
                            # governance_decision_for_display is set by governance check-in (runs every 30 iterations)
                            # It's None on most iterations, but will have value after governance check-ins
                            _ctx.screen_renderer.render(
                                face_state=face_state,
                                anima=anima,
                                readings=readings,
                                identity=identity,
                                governance=governance_decision_for_display
                            )

                            # Canvas autonomy handled inside render() — no duplicate call needed
                        else:
                            # Fallback: render face directly
                            identity_name = identity.name if identity else None
                            _ctx.display.render_face(face_state, name=identity_name)
                        return True  # Return success indicator

                    # Run display update in thread pool to prevent blocking input polling
                    # This allows joystick to remain responsive during slow display renders
                    loop = asyncio.get_event_loop()
                    try:
                        display_result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: safe_call(update_display, default=False, log_error=True)),
                            timeout=2.0  # Max 2 seconds for display update
                        )
                    except asyncio.TimeoutError:
                        display_result = False
                        if loop_count % DISPLAY_LOG_THROTTLE == 0:
                            logger.debug("[Loop] Display update timed out (2s)")
                    display_updated = display_result is True
                    if display_updated:
                        if loop_count == 1:
                            print("[Loop] Display render successful - face showing", file=sys.stderr, flush=True)
                    elif loop_count == 1:
                        print("[Loop] Display available but render failed (check error logs)", file=sys.stderr, flush=True)
                else:
                    if loop_count == 1:
                        print("[Loop] Display initialized but hardware not available (not on Pi or hardware issue?)", file=sys.stderr, flush=True)
                        print("[Loop] Run diagnostics: python3 -m anima_mcp.display_diagnostics", file=sys.stderr, flush=True)
            else:
                if loop_count == 1:
                    print("[Loop] Display not initialized", file=sys.stderr, flush=True)
            # Heartbeat regardless of render success — probe detects hardware failure independently
            if _ctx.display and _health:
                _health.heartbeat("display")

            # Update LEDs with raw anima state (independent from face)
            # LEDs reflect proprioceptive state directly - what Lumen actually feels
            led_updated = False
            if _ctx.leds and _ctx.leds.is_available():
                # Get light level for auto-brightness
                light_level = readings.light_lux if readings else None

                # Get activity brightness from shared memory (broker computes this)
                # - ACTIVE (day/interaction): 1.0
                # - DROWSY (dusk/dawn/30min idle): 0.6
                # - RESTING (night/60min idle): 0.35
                activity_brightness = 1.0
                try:
                    # Primary: read from broker's shared memory (single source of truth)
                    _shm = _get_last_shm_data()
                    if _shm and "activity" in _shm:
                        activity_brightness = _shm["activity"].get("brightness_multiplier", 1.0)
                    else:
                        # Fallback: compute locally if broker not running
                        if _ctx.activity is None:
                            _ctx.activity = get_activity_manager()
                        activity_state = _ctx.activity.get_state(
                            presence=anima.presence,
                            stability=anima.stability,
                            light_level=light_level,
                        )
                        activity_brightness = activity_state.brightness_multiplier
                except Exception as e:
                    if loop_count % ERROR_LOG_THROTTLE == 1:
                        print(f"[ActivityBrightness] Error: {e}", file=sys.stderr, flush=True)

                # Sync manual brightness dimmer to LED controller
                # Priority: 1) Screen renderer manual override, 2) Broker agency brightness from SHM
                display_with_brightness = _ctx.screen_renderer._display if _ctx.screen_renderer else _ctx.display
                if display_with_brightness and getattr(display_with_brightness, '_manual_led_brightness', None) is not None:
                    _ctx.leds._manual_brightness_factor = display_with_brightness._manual_led_brightness
                elif _shm and "agency_led_brightness" in _shm:
                    _ctx.leds._manual_brightness_factor = _shm["agency_led_brightness"]

                def update_leds():
                    # LEDs derive their own state directly from anima - no face influence
                    # Pass memory state for visualization when Lumen is "remembering"
                    anticipation_confidence = 0.0
                    if anima.anticipation:
                        anticipation_confidence = anima.anticipation.get("confidence", 0.0)
                    return _ctx.leds.update_from_anima(
                        anima.warmth, anima.clarity,
                        anima.stability, anima.presence,
                        light_level=light_level,
                        is_anticipating=anima.is_anticipating,
                        anticipation_confidence=anticipation_confidence,
                        activity_brightness=activity_brightness
                    )

                led_state = safe_call(update_leds, default=None, log_error=True)
                led_updated = led_state is not None
                if led_updated and loop_count == 1:
                    total_duration = time.time() - update_start
                    print(f"[Loop] LED update took {total_duration*1000:.1f}ms", file=sys.stderr, flush=True)
                    print(f"[Loop] LED update (independent): warmth={anima.warmth:.2f} clarity={anima.clarity:.2f} stability={anima.stability:.2f} presence={anima.presence:.2f} activity_brightness={activity_brightness:.2f}", file=sys.stderr, flush=True)
                    print(f"[Loop] LED colors: led0={led_state.led0} led1={led_state.led1} led2={led_state.led2}", file=sys.stderr, flush=True)

                # === LED PROPRIOCEPTION: capture what our LEDs are doing ===
                # This feeds forward into next iteration's metacognition prediction.
                # Lumen now knows its own brightness — the light sensor becomes
                # genuinely proprioceptive rather than confusingly self-referential.
                try:
                    _ctx.led_proprioception = _ctx.leds.get_proprioceptive_state()
                    # Also populate readings.led_brightness with ACTUAL computed brightness
                    # (not just activity multiplier like stable_creature.py does)
                    if readings is not None:
                        readings.led_brightness = _ctx.led_proprioception.get("brightness", 0.0) if _ctx.led_proprioception else 0.0
                except Exception as e:
                    if loop_count % ERROR_LOG_THROTTLE == 1:
                        print(f"[LEDProprioception] Error: {e}", file=sys.stderr, flush=True)
            elif _ctx.leds:
                if loop_count == 1:
                    print("[Loop] LEDs not available (hardware issue?)", file=sys.stderr, flush=True)
            # Heartbeat regardless of LED update success — probe detects hardware failure independently
            if _ctx.leds and _health:
                _health.heartbeat("leds")

            # Update voice system with anima state (for listening and text expression)
            if loop_count % VOICE_INTERVAL == 0:
                try:
                    voice = _get_voice()
                    if voice and voice.is_running:
                        # Determine mood based on anima state
                        if anima.warmth > 0.7 and anima.stability > 0.6:
                            mood = "content"
                        elif anima.clarity > 0.7:
                            mood = "curious"
                        elif anima.warmth > 0.6 and anima.presence > 0.6:
                            mood = "peaceful"
                        elif anima.warmth < 0.4:
                            mood = "withdrawn"
                        else:
                            mood = "neutral"

                        voice.update_state(
                            warmth=anima.warmth,
                            clarity=anima.clarity,
                            stability=anima.stability,
                            presence=anima.presence,
                            mood=mood
                        )
                        if readings:
                            voice.update_environment(
                                temperature=readings.ambient_temp_c or 22.0,
                                humidity=readings.humidity_pct or 50.0,
                                light_level=readings.light_lux or 500.0
                            )
                except Exception as e:
                    if loop_count % STATUS_LOG_THROTTLE == 0:
                        logger.debug("[Voice] State update error: %s", e)
                # Heartbeat regardless of voice init — probe detects failure independently
                if _health:
                    _health.heartbeat("voice")

            # Log update status every 20th iteration
            if loop_count % DISPLAY_LOG_THROTTLE == 1 and (display_updated or led_updated):
                update_duration = time.time() - update_start
                update_status = []
                if display_updated:
                    update_status.append("display")
                if led_updated:
                    update_status.append("LEDs")
                logger.debug("[Loop] Display/LED updates (%s): %.1fms", ', '.join(update_status), update_duration*1000)
            
            # Log every 5th iteration with LED status and key metrics
            if loop_count % TRAJECTORY_INTERVAL == 1:
                pass

            # System metrics persistence: Every 15 iterations (~30s), record to SQLite
            if loop_count % SYSTEM_METRICS_RECORD_INTERVAL == 0:
                if readings and _ctx and _ctx.store:
                    try:
                        _ctx.store.record_system_metrics(readings)
                    except Exception:
                        pass  # Non-fatal — don't disrupt main loop
                # Heartbeat regardless of readings — probes fail open independently
                if _health is not None:
                    _health.heartbeat("thermal_trend")
                    _health.heartbeat("memory_pressure")

            # System metrics pruning: Every 1800 iterations (~1h), delete old rows
            if loop_count % SYSTEM_METRICS_PRUNE_INTERVAL == 0 and loop_count > 0 and _ctx and _ctx.store:
                try:
                    _pruned = _ctx.store.prune_system_metrics(SYSTEM_METRICS_RETENTION_HOURS)
                    if _pruned > 0:
                        logger.debug("[Metrics] Pruned %d old system_metrics rows", _pruned)
                except Exception:
                    pass

            # Adaptive learning: Every 100 iterations (~3.3 minutes), check if calibration should adapt
            # Respects cooldown to avoid redundant adaptations during continuous operation
            if loop_count % LEARNING_INTERVAL == 0 and _ctx and _ctx.store:
                def try_learning():
                    learner = get_learner(str(_ctx.store.db_path))
                    adapted, new_cal = learner.adapt_calibration(respect_cooldown=True)
                    if adapted:
                        logger.debug("[Learning] Calibration adapted after %d observations", loop_count)
                        logger.debug("[Learning] Pressure: %.1f hPa, Ambient: %.1f-%.1f C", new_cal.pressure_ideal, new_cal.ambient_temp_min, new_cal.ambient_temp_max)
                
                safe_call(try_learning, default=None, log_error=True)
            
            # Lumen's unified reflection: Every ~30 minutes
            # Grounded observation from actual state — no LLM needed
            if loop_count % UNIFIED_REFLECTION_INTERVAL == 0 and readings and anima and identity:
                try:
                    await safe_call_async(
                        lambda: _lumen_unified_reflect(anima, readings, identity, prediction_error),
                        default=None, log_error=True,
                    )
                except Exception as e:
                    logger.debug("[Lumen/Unified] Reflection error: %s", e)

            # Lumen self-answers: check periodically, but only after the age
            # gate in server_state gives external answers priority.
            # Uses learned insights/beliefs/preferences — no LLM needed
            if loop_count % SELF_ANSWER_INTERVAL == 0 and readings and anima and identity:
                try:
                    await safe_call_async(
                        lambda: _lumen_self_answer(anima, readings, identity),
                        default=None, log_error=True,
                    )
                except Exception as e:
                    logger.debug("[Lumen/SelfAnswer] Self-answer error: %s", e)


            # Growth system: Observe state for preference learning and check milestones
            # Every 30 iterations (~1 minute) - learns from anima state + environment
            if loop_count % GROWTH_INTERVAL == 0 and readings and anima and identity and _ctx and _ctx.growth:
                def growth_observe():
                    """Observe environment and check milestones."""
                    # Prepare anima state dict
                    anima_state = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                    }
                    # Prepare environment dict from sensor readings
                    # Raw lux includes LED glow — that's Lumen's actual light environment
                    environment = {
                        "light_lux": readings.light_lux or 0.0,
                        "temp_c": readings.ambient_temp_c,
                        "humidity_pct": readings.humidity_pct,
                    }

                    # Erode preferences that have stopped being observed. Decay
                    # otherwise only runs when a preference is REINFORCED, so an
                    # abandoned one keeps its confidence forever — the model had
                    # no retraction path. Hourly is ample: the floor is reached
                    # after weeks, not minutes, and the sweep is idempotent.
                    try:
                        _now = time.time()
                        if _now - _ctx.last_preference_decay_at >= PREFERENCE_DECAY_INTERVAL_SECONDS:
                            _ctx.last_preference_decay_at = _now
                            _retracted = _ctx.growth.decay_stale_preferences()
                            for _name in _retracted:
                                logger.info(
                                    "[Growth] preference '%s' fell below the trust gate "
                                    "(unobserved too long)", _name,
                                )
                    except Exception as e:
                        logger.debug("[Growth] preference decay sweep failed: %s", e)

                    # Observe for preference learning
                    insight = _ctx.growth.observe_state_preference(anima_state, environment)
                    if insight:
                        logger.debug("[Growth] %s", insight)
                        # Add insight as an observation from Lumen
                        from .messages import add_observation
                        add_observation(insight, author="lumen")

                    # Check for age/awakening milestones
                    milestone = _ctx.growth.check_for_milestones(identity, anima)
                    if milestone:
                        logger.debug("[Growth] Milestone: %s", milestone)
                        from .messages import add_observation
                        add_observation(milestone, author="lumen")

                safe_call(growth_observe, default=None, log_error=True)
            # Heartbeat regardless of _ctx.growth — probe detects init failure independently
            if loop_count % GROWTH_INTERVAL == 0 and _health:
                _health.heartbeat("growth")

            # Goal system: Suggest new goals every ~2 hours
            if loop_count % GOAL_SUGGEST_INTERVAL == 0 and anima and _ctx and _ctx.growth:
                def goal_suggest():
                    """Suggest a goal grounded in Lumen's experience."""
                    anima_state = {
                        "warmth": anima.warmth, "clarity": anima.clarity,
                        "stability": anima.stability, "presence": anima.presence,
                    }
                    try:
                        from .self_model import get_self_model
                        sm = get_self_model()
                    except Exception as e:
                        logger.debug("[Growth] SelfModel init for goal suggest: %s", e)
                        sm = None
                    goal = _ctx.growth.suggest_goal(anima_state, self_model=sm)
                    if goal:
                        from .messages import add_observation
                        add_observation(f"new goal: {goal.description}", author="lumen")

                safe_call(goal_suggest, default=None, log_error=True)

            # Goal system: Check progress every ~10 minutes
            if loop_count % GOAL_CHECK_INTERVAL == 0 and anima and _ctx and _ctx.growth:
                def goal_check():
                    """Check progress on active goals."""
                    anima_state = {
                        "warmth": anima.warmth, "clarity": anima.clarity,
                        "stability": anima.stability, "presence": anima.presence,
                    }
                    try:
                        from .self_model import get_self_model
                        sm = get_self_model()
                    except Exception as e:
                        logger.debug("[Growth] SelfModel init for goal check: %s", e)
                        sm = None
                    msg = _ctx.growth.check_goal_progress(anima_state, self_model=sm)
                    if msg:
                        from .messages import add_observation
                        add_observation(msg, author="lumen")

                safe_call(goal_check, default=None, log_error=True)

            # Meta-learning: Daily preference weight evolution
            # Every ~12 hours, rebalance which anima dimensions matter most
            # based on how satisfying each dimension correlates with trajectory health
            if loop_count % META_LEARNING_INTERVAL == 0 and loop_count > 0 and _ctx and _ctx.growth:
                try:
                    from .preferences import (
                        compute_trajectory_health, meta_learning_update,
                        get_preference_system as _ml_get_pref,
                    )

                    # Prediction accuracy trend: -0.5 (poor) to 0.5 (good).
                    # The broker owns this learner and publishes its live stats;
                    # the server's process-local singleton has no observations.
                    pred_trend = 0.0
                    try:
                        stats = (
                            ((_ctx.last_shm_data or {}).get("learning") or {}).get(
                                "prediction_accuracy", {}
                            )
                        )
                        if not stats.get("insufficient_data") and "overall_mean_error" in stats:
                            err = float(stats["overall_mean_error"])
                            pred_trend = max(-0.5, min(0.5, (1.0 - min(1.0, err)) * 2.0 - 1.0))
                    except Exception as e:
                        logger.debug("[MetaLearning] Broker prediction stats error: %s", e)

                    health = compute_trajectory_health(
                        satisfaction_history=list(_ctx.satisfaction_history)[-100:],
                        action_efficacy=_ctx.action_efficacy,
                        prediction_accuracy_trend=pred_trend,
                    )
                    _ctx.health_history.append(health)

                    # Record healthy state for drift restart target
                    if _ctx and _ctx.calibration_drift:
                        _ctx.calibration_drift.record_healthy_state(health)

                    # Compute lagged correlations between per-dim satisfaction and health
                    correlations = _compute_lagged_correlations()

                    # Compute against the refreshing broker-owned snapshot, then
                    # queue the mutation back to its sole writer.
                    pref_system = _ml_get_pref()
                    weights = {
                        d: p.influence_weight
                        for d, p in pref_system._preferences.items()
                        if d in ("warmth", "clarity", "stability", "presence")
                    }
                    if weights:
                        new_weights = meta_learning_update(weights, correlations)
                        from .learning_events import enqueue_preference_weights
                        enqueue_preference_weights(
                            new_weights,
                            source="server:trajectory_meta_learning",
                        )
                        logger.debug("[MetaLearning] Updated preference weights: %s health=%.3f",
                                    ', '.join(f'{d}={w:.3f}' for d, w in new_weights.items()), health)
                except Exception as e:
                    logger.warning("[MetaLearning] Error (non-fatal): %s", e)

            # Trajectory: Record anima history for trajectory signature computation
            # Every 5 iterations (~10 seconds) - builds time-series for attractor basin
            # See: trajectory-identity paper (cirwel/trajectory-identity-paper, separate repo)
            if loop_count % TRAJECTORY_INTERVAL == 0 and anima:
                from .anima_history import get_anima_history

                def record_history():
                    """Record anima state for trajectory computation."""
                    history = get_anima_history()
                    history.record_from_anima(anima)

                safe_call(record_history, default=None, log_error=True)

            # Governance is handled by the broker (sole UNITARES caller).
            # Server reads governance from SHM in the display block above.

            # === SLOW CLOCK: Self-Schema G_t extraction (every 5 minutes) ===
            # PoC for StructScore visual integrity evaluation
            # Extracts Lumen's self-representation graph and optionally saves for offline analysis
            if (loop_count == 1 or loop_count % SCHEMA_EXTRACTION_INTERVAL == 0) and readings and anima and identity:
                try:
                    await safe_call_async(
                        lambda: _extract_and_validate_schema(anima, readings, identity),
                        default=None, log_error=True,
                    )
                    # Persist schema periodically so crash recovery has recent data
                    # (not just on clean shutdown — Pi crashes often)
                    if _ctx.schema_hub:
                        _ctx.schema_hub.persist_schema()
                except Exception as e:
                    logger.warning("[Schema] Extraction error: %s", e)

            # === SLOW CLOCK: Self-Reflection (every 15 minutes) ===
            # Analyze state history, discover patterns, generate insights about self
            if loop_count % EXPRESSION_INTERVAL == 0 and readings and anima and identity:
                try:
                    await safe_call_async(_self_reflect, default=None, log_error=True)
                except Exception as e:
                    logger.debug("[SelfReflection] Reflection error: %s", e)

            # Delay until next render — screen-specific for performance
            # Heavy screens (notepad, learning) get slower refresh to save CPU
            # Event-driven: mode_change_event breaks out of wait immediately
            current_mode = _ctx.screen_renderer._state.mode if _ctx and _ctx.screen_renderer else None

            # Screen-specific delays: notepad/learning are heavy, others are light
            if current_mode in (ScreenMode.NOTEPAD, ScreenMode.LEARNING, ScreenMode.SELF_GRAPH):
                screen_delay = 1.0  # 1 FPS for heavy screens (drawing, learning visualization)
            elif current_mode in (ScreenMode.NEURAL,):
                screen_delay = 0.5  # 2 FPS for neural (animated but not critical)
            else:
                screen_delay = base_delay  # 5 FPS for face and simple screens

            if consecutive_errors > 0:
                delay = min(screen_delay * (1.5 ** min(consecutive_errors, 3)), max_delay)
            else:
                delay = screen_delay

            # Wait for delay OR mode change event (whichever comes first)
            # This makes screen switching feel instant
            try:
                await asyncio.wait_for(mode_change_event.wait(), timeout=delay)
                mode_change_event.clear()  # Reset for next mode change
                quick_render = True  # Skip heavy subsystems, just render
                # Mode changed - render immediately with minimal settle delay
                await asyncio.sleep(0.015)  # 15ms — let GPIO debounce finish
            except asyncio.TimeoutError:
                pass  # Normal timeout - continue with next iteration
            
        except KeyboardInterrupt:
            # Allow graceful shutdown
            raise
        except Exception as e:
            # Don't crash on display errors, just log and continue with exponential backoff
            consecutive_errors += 1
            error_type = type(e).__name__
            logger.error("[Loop] Error (%s): %s", error_type, e)
            
            # Exponential backoff on errors
            delay = min(base_delay * (2 ** min(consecutive_errors // 3, 4)), max_delay)
            await asyncio.sleep(delay)

def start_display_loop():
    """Start continuous display update loop."""
    try:
        if _ctx is None:
            # Never bail silently: this exact silence hid a 19-day dark screen
            # (#106) — wake() failed, no loop, and no line saying why.
            print("[Display] NOT starting display loop: no server context (wake() failed?)", file=sys.stderr, flush=True)
            return
        if _ctx.display_update_task is None or _ctx.display_update_task.done():
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running - will be started later
                print("[Display] No event loop yet, will start when available", file=sys.stderr, flush=True)
                return
            
            _ctx.display_update_task = asyncio.create_task(_update_display_loop())
            print("[Display] Started continuous update loop", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Display] Error starting display loop: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)

def stop_display_loop():
    """Stop continuous display update loop and blank the display for clean shutdown."""
    try:
        if _ctx and _ctx.display_update_task and not _ctx.display_update_task.done():
            _ctx.display_update_task.cancel()
            try:
                print("[Display] Stopped continuous update loop", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass
        # Also cancel the fast input poller — it lives alongside the display loop
        if _ctx and _ctx.input_poll_task and not _ctx.input_poll_task.done():
            _ctx.input_poll_task.cancel()
    except Exception as e:
        try:
            print(f"[Display] Error stopping display loop: {e}", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass
    # Blank the display so shutdown/restart doesn't leave a stale or scrambled frame
    try:
        if _ctx and _ctx.screen_renderer and _ctx.screen_renderer._display:
            _ctx.screen_renderer._display.blank()
    except Exception:
        pass

# ============================================================
# Wake / Lifecycle
# ============================================================

def wake(db_path: str = "anima.db", anima_id: str | None = None):
    """Wake up. Call before starting server. Delegates to lifecycle.py."""
    from .lifecycle import wake as _lifecycle_wake
    _lifecycle_wake(db_path, anima_id)

def sleep():
    """Go to sleep. Call on server shutdown. Delegates to lifecycle.py."""
    from .lifecycle import sleep as _lifecycle_sleep
    _lifecycle_sleep()

async def run_stdio_server():
    """Run the MCP server over stdio (local)."""
    server = create_server()
    
    # Start display update loop
    start_display_loop()

    # Handle graceful shutdown
    def shutdown_handler(sig, frame):
        try:
            print("\nShutting down...", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass  # stdout/stderr might be closed
        try:
            stop_display_loop()
            sleep()
        except Exception as e:
            logger.debug("[Shutdown] Cleanup error: %s", e)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        stop_display_loop()

def run_http_server(host: str, port: int):
    """Run MCP server over HTTP with Streamable HTTP transport.

    Endpoints:
    - /mcp/  : Streamable HTTP (MCP transport)
    - /health: Health check
    - /v1/tools/call: REST API for direct tool calls
    - /dashboard, /state, /qa, etc.: Control Center endpoints

    NOTE: Server operates locally even without network connectivity.
    WiFi is only needed for remote MCP clients to connect.
    Lumen continues operating autonomously (display, LEDs, sensors, canvas) regardless of network status.
    """
    import asyncio

    async def _run_http_server_async():
        """Async inner function to run the HTTP server with uvicorn."""
        import uvicorn

        # Log that local operation continues regardless of network
        print("[Server] Starting HTTP server (Streamable HTTP)", file=sys.stderr, flush=True)
        print("[Server] Network connectivity only needed for remote MCP clients", file=sys.stderr, flush=True)

        # Check if FastMCP is available
        if not HAS_FASTMCP:
            print("[Server] ERROR: FastMCP not available - cannot start HTTP server", file=sys.stderr, flush=True)
            print("[Server] Install mcp[cli] to get FastMCP support", file=sys.stderr, flush=True)
            raise SystemExit(1)

        # Apply runtime patches for upstream MCP SDK bugs before any MCP
        # session is created. Currently: swallow ClosedResourceError raised
        # when a client disconnects mid-response (MCP 1.26.0).
        from .mcp_patches import apply_all_patches
        apply_all_patches()

        # Get the FastMCP server instance (creates and registers tools if needed)
        mcp = get_fastmcp()
        if mcp is None:
            print("[Server] ERROR: Failed to create FastMCP server", file=sys.stderr, flush=True)
            raise SystemExit(1)

        print("[Server] Setting up Streamable HTTP transport...", file=sys.stderr, flush=True)

        # === Streamable HTTP transport (the only MCP transport) ===
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse

        _streamable_session_manager = None

        # Create session manager for Streamable HTTP
        _streamable_session_manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server,  # Access the underlying MCP server
            json_response=True,  # Use JSON responses (proper Streamable HTTP)
            stateless=True,  # Allow stateless for compatibility
        )

        print("[Server] Streamable HTTP transport available at /mcp/", file=sys.stderr, flush=True)

        # --- OAuth 2.1 setup (conditional) ---
        _oauth_issuer_url = os.environ.get("ANIMA_OAUTH_ISSUER_URL")
        _oauth_auth_routes = []
        _oauth_token_verifier = None
        _oauth_dynamic_registration = os.environ.get(
            "ANIMA_OAUTH_DYNAMIC_REGISTRATION", "false"
        ).lower() in ("1", "true", "yes", "on")

        if _oauth_issuer_url and hasattr(mcp, '_auth_server_provider') and mcp._auth_server_provider:
            try:
                from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes, ClientRegistrationOptions
                from mcp.server.auth.provider import ProviderTokenVerifier
                from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend

                _oauth_token_verifier = ProviderTokenVerifier(mcp._auth_server_provider)

                _oauth_auth_routes = create_auth_routes(
                    provider=mcp._auth_server_provider,
                    issuer_url=mcp.settings.auth.issuer_url,
                    client_registration_options=ClientRegistrationOptions(
                        enabled=_oauth_dynamic_registration
                    ),
                )

                _oauth_auth_routes.extend(
                    create_protected_resource_routes(
                        resource_url=mcp.settings.auth.resource_server_url,
                        authorization_servers=[mcp.settings.auth.issuer_url],
                        scopes_supported=mcp.settings.auth.required_scopes,
                    )
                )

                print(
                    f"[Server] OAuth 2.1 routes enabled ({len(_oauth_auth_routes)} routes; "
                    f"dynamic registration={'on' if _oauth_dynamic_registration else 'off'})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                print(f"[Server] OAuth setup failed, continuing without auth: {e}", file=sys.stderr, flush=True)
                _oauth_auth_routes = []
                _oauth_token_verifier = None


        # Create ASGI app for /mcp
        async def streamable_mcp_asgi(scope, receive, send):
            """ASGI app for Streamable HTTP MCP at /mcp/."""
            if scope.get("type") != "http":
                return

            # Reject .well-known discovery probes that land inside /mcp/ mount.
            # The SDK tries /mcp/.well-known/openid-configuration and the session
            # manager returns 406, which confuses auth negotiation.
            path = scope.get("path", "")
            if ".well-known" in path:
                response = JSONResponse({"error": "not found"}, status_code=404)
                await response(scope, receive, send)
                return

            # Stash X-Anima-Admin header for destructive-handler gate (admin_auth).
            from .admin_auth import set_admin_header
            admin_value: str | None = None
            for name, value in scope.get("headers", []):
                if name == b"x-anima-admin":
                    admin_value = value.decode("latin-1")
                    break
            set_admin_header(admin_value)

            try:
                await _streamable_session_manager.handle_request(scope, receive, send)
            except Exception as e:
                print(f"[MCP] Error in Streamable HTTP handler: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
                try:
                    response = JSONResponse({"error": str(e)}, status_code=500)
                    await response(scope, receive, send)
                except RuntimeError:
                    pass

        # REST API endpoints (extracted to rest_api.py)
        from .rest_api import (
            health_check, rest_tool_call, dashboard,
            rest_state, rest_qa, rest_messages, rest_answer, rest_message,
            rest_learning, rest_voice, rest_gallery, rest_gallery_image,
            rest_health_detailed, rest_self_knowledge, rest_growth,
            rest_gallery_page, rest_layers, rest_architecture_page,
            rest_schema_data, rest_schema_page,
        )
        from starlette.staticfiles import StaticFiles
        _static_dir = Path(__file__).parent.parent.parent / "docs" / "static"

        # === Build Starlette app with all routes ===
        # Wrap /mcp with OAuth if configured.
        # Auth middleware is chained directly around /mcp (not globally)
        # to avoid interfering with REST/dashboard routes.
        if _oauth_token_verifier:
            from mcp.server.auth.middleware.bearer_auth import RequireAuthMiddleware
            from mcp.server.auth.middleware.auth_context import AuthContextMiddleware as _AuthCtx
            from starlette.middleware.authentication import AuthenticationMiddleware as _AuthMW
            from mcp.server.auth.routes import build_resource_metadata_url
            resource_metadata_url = build_resource_metadata_url(mcp.settings.auth.resource_server_url)

            # Hosts that require OAuth (Cloudflare tunnel).
            # Local and Tailscale clients (Cursor, Claude Code) skip auth.
            _EXTERNAL_HOSTS = {"lumen.cirwel.org"}
            _auth_protected = _AuthMW(
                _AuthCtx(
                    RequireAuthMiddleware(
                        streamable_mcp_asgi,
                        required_scopes=mcp.settings.auth.required_scopes or [],
                        resource_metadata_url=resource_metadata_url,
                    )
                ),
                backend=BearerAuthBackend(_oauth_token_verifier),
            )

            async def mcp_endpoint(scope, receive, send):
                """Route to auth or no-auth based on Host header."""
                if scope.get("type") == "http":
                    headers = dict(scope.get("headers", []))
                    host = headers.get(b"host", b"").decode().split(":")[0]
                    if host in _EXTERNAL_HOSTS:
                        await _auth_protected(scope, receive, send)
                        return
                await streamable_mcp_asgi(scope, receive, send)
        else:
            mcp_endpoint = streamable_mcp_asgi

        all_routes = [
            *_oauth_auth_routes,
            Mount("/mcp", app=mcp_endpoint),
            Mount("/static", app=StaticFiles(directory=str(_static_dir)), name="static"),
            Route("/health", health_check, methods=["GET"]),
            Route("/health/detailed", rest_health_detailed, methods=["GET"]),
            Route("/v1/tools/call", rest_tool_call, methods=["POST"]),
            Route("/dashboard", dashboard, methods=["GET"]),
            Route("/state", rest_state, methods=["GET"]),
            Route("/qa", rest_qa, methods=["GET"]),
            Route("/answer", rest_answer, methods=["POST"]),
            Route("/message", rest_message, methods=["POST"]),
            Route("/messages", rest_messages, methods=["GET"]),
            Route("/learning", rest_learning, methods=["GET"]),
            Route("/voice", rest_voice, methods=["GET"]),
            Route("/gallery", rest_gallery, methods=["GET"]),
            Route("/gallery/{filename}", rest_gallery_image, methods=["GET"]),
            Route("/gallery-page", rest_gallery_page, methods=["GET"]),
            Route("/layers", rest_layers, methods=["GET"]),
            Route("/self-knowledge", rest_self_knowledge, methods=["GET"]),
            Route("/growth", rest_growth, methods=["GET"]),
            Route("/architecture", rest_architecture_page, methods=["GET"]),
            Route("/schema-data", rest_schema_data, methods=["GET"]),
            Route("/schema", rest_schema_page, methods=["GET"]),
        ]
        # Starlette lifespan — single owner of session manager lifecycle.
        # session_manager.run() creates the anyio task group internally;
        # no manual _task_group/_has_started poking needed.
        import contextlib

        @contextlib.asynccontextmanager
        async def lifespan(app):
            start_display_loop()
            print("[Server] Display loop started", file=sys.stderr, flush=True)

            warmup_task = None
            async with _streamable_session_manager.run():
                print("[Server] Streamable HTTP session manager running", file=sys.stderr, flush=True)

                # Server warmup task - marks server ready after brief delay.
                # Handle is retained so the lifespan can cancel it on shutdown,
                # preventing it from flipping SERVER_READY mid-teardown.
                async def server_warmup_task():
                    global SERVER_READY, SERVER_STARTUP_TIME
                    SERVER_STARTUP_TIME = datetime.now()
                    try:
                        from pathlib import Path
                        lockfile = Path("/tmp/anima-restarting")
                        if lockfile.exists():
                            lockfile.unlink()
                            print("[Server] Cleared restart lockfile", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    try:
                        await asyncio.sleep(2.0)
                    except asyncio.CancelledError:
                        return  # Shutdown started during warmup — don't flip ready
                    SERVER_READY = True
                    print("[Server] Warmup complete - server ready", file=sys.stderr, flush=True)

                warmup_task = asyncio.create_task(server_warmup_task())

                print(f"MCP server running at http://{host}:{port}", file=sys.stderr, flush=True)
                print(f"  Streamable HTTP: http://{host}:{port}/mcp/", file=sys.stderr, flush=True)

                yield  # Server accepts requests here

            # Cleanup after uvicorn shuts down
            print("[Server] Streamable HTTP session manager shut down", file=sys.stderr, flush=True)

            # Cancel warmup task if it's still sleeping
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                try:
                    await warmup_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Await bridge close here (async context) — sleep()'s fire-and-forget
            # can't reliably close it when the loop is about to stop.
            bridge = _get_server_bridge()
            if bridge:
                try:
                    await bridge.close()
                except Exception as e:
                    logger.debug("[Shutdown] Bridge close error: %s", e)

            stop_display_loop()
            sleep()

        _inner_app = Starlette(routes=all_routes, lifespan=lifespan)

        # Wrap app to rewrite /mcp → /mcp/ at the ASGI level.
        # Starlette's Mount issues a 307 redirect for missing trailing slash,
        # but behind a reverse proxy the redirect uses http:// (wrong scheme)
        # which breaks Claude.ai's MCP client. This avoids the redirect entirely.
        async def _rewrite_mcp_slash(scope, receive, send):
            if scope.get("type") == "http" and scope.get("path") == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
            await _inner_app(scope, receive, send)

        app = _rewrite_mcp_slash
        print("[Server] Starlette app created with all routes", file=sys.stderr, flush=True)

        # Run with uvicorn — lifespan handles startup/shutdown cleanup
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            limit_concurrency=100,
            timeout_keep_alive=5,
            proxy_headers=True,          # Trust X-Forwarded-Proto from cloudflared
            forwarded_allow_ips="*",     # Allow proxy headers from any IP
        )
        server = uvicorn.Server(config)
        await server.serve()

    # Run the async server
    asyncio.run(_run_http_server_async())


def main():
    """Entry point."""
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Anima MCP Server")
    parser.add_argument("--http", "--sse", action="store_true", dest="http_server",
                        help="Run HTTP server with Streamable HTTP at /mcp/")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8766, help="HTTP server port (default: 8766)")
    args = parser.parse_args()

    # Prevent multiple instances using pidfile (but allow if stale)
    pidfile = Path("/tmp/anima-mcp.pid")
    if pidfile.exists():
        try:
            old_pid = int(pidfile.read_text().strip())
            # Check if process is still running
            os.kill(old_pid, 0)  # Signal 0 = check if alive
            # Process is running - check if it's actually serving
            import socket
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(0.5)
            try:
                result = test_sock.connect_ex(('127.0.0.1', args.port if args.http_server else 0))
                test_sock.close()
                if result == 0:
                    # Port is in use, likely another instance is serving
                    print(f"[Server] Another instance already running (PID {old_pid}) and port appears in use. Exiting.", file=sys.stderr)
                    print(f"[Server] To force restart: kill {old_pid} && rm {pidfile}", file=sys.stderr)
                    sys.exit(1)
            except Exception as e:
                logger.debug("[Server] Port check error: %s", e)
        except (ProcessLookupError, ValueError):
            # Process not running or invalid pid - remove stale pidfile
            try:
                pidfile.unlink()
                print("[Server] Removed stale pidfile", file=sys.stderr, flush=True)
            except Exception as e:
                logger.debug("[Server] Stale pidfile removal error: %s", e)
        except PermissionError:
            # Process running as different user - try to continue anyway
            print("[Server] Warning: PID file exists but can't check process (different user). Continuing...", file=sys.stderr, flush=True)
        except Exception as e:
            # Any other error - remove pidfile and continue
            print(f"[Server] Error checking pidfile: {e}. Removing and continuing...", file=sys.stderr, flush=True)
            try:
                pidfile.unlink()
            except Exception as e:
                logger.debug("[Server] Pidfile removal error: %s", e)

    # Write our PID
    try:
        pidfile.write_text(str(os.getpid()))
    except Exception as e:
        print(f"[Server] Warning: Could not write pidfile: {e}", file=sys.stderr, flush=True)

    # Register cleanup for PID file on exit
    import atexit
    def cleanup_pidfile():
        try:
            if pidfile.exists():
                current_pid = pidfile.read_text().strip()
                if current_pid == str(os.getpid()):
                    pidfile.unlink()
        except Exception as e:
            logger.debug("[Server] Cleanup pidfile error: %s", e)
    atexit.register(cleanup_pidfile)

    # Determine DB persistence path (User Home > Project Root)
    env_db = os.environ.get("ANIMA_DB")
    if env_db:
        db_path = env_db
    else:
        # Default to persistent user home directory
        home_dir = Path.home() / ".anima"
        home_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(home_dir / "anima.db")

    print(f"[Server] Using persistent database: {db_path}", file=sys.stderr)
    anima_id = os.environ.get("ANIMA_ID")

    wake(db_path, anima_id)

    try:
        if args.http_server:
            run_http_server(args.host, args.port)
        else:
            asyncio.run(run_stdio_server())
    except KeyboardInterrupt:
        try:
            print("\nInterrupted by user", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass
    except Exception as e:
        try:
            print(f"[Server] Fatal error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
        except (ValueError, OSError):
            pass
    finally:
        try:
            sleep()
        except Exception as e:
            logger.debug("[Shutdown] Sleep error: %s", e)
        # Clean up pidfile
        try:
            pidfile.unlink(missing_ok=True)
        except Exception as e:
            logger.debug("[Shutdown] Pidfile cleanup error: %s", e)

if __name__ == "__main__":
    main()
