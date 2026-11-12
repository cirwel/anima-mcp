"""Display loop phase helpers extracted from server.py.

Contains startup recovery/adaptation, per-tick telemetry and drive-event
consumption, quiet-world contemplation, primitive-language generation and
feedback, governance synchronization and display/LED/voice rendering, and
periodic growth/goal/meta-learning/schema/reflection scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from .activity_state import get_activity_manager
from .config import get_calibration
from .display import derive_face_state, get_display
from .display.leds import get_led_display
from .display.screens import ScreenMode
from .eisv import get_trajectory_awareness
from .eisv_mapper import anima_to_body_eisv_projection
from .error_recovery import safe_call, safe_call_async
from .learning import get_learner
from .light_attribution import gated_external_light_lux
from .server_state import (
    AGENCY_INTERVAL,
    DISPLAY_LOG_THROTTLE,
    ERROR_LOG_THROTTLE,
    EXPRESSION_INTERVAL,
    GOAL_CHECK_INTERVAL,
    GOAL_SUGGEST_INTERVAL,
    GROWTH_INTERVAL,
    LEARNING_INTERVAL,
    META_LEARNING_INTERVAL,
    PREFERENCE_DECAY_INTERVAL_SECONDS,
    SCHEMA_EXTRACTION_INTERVAL,
    SELF_ANSWER_INTERVAL,
    SELF_DERIVATION_CHECK_INTERVAL,
    SERVER_GOVERNANCE_FALLBACK_SECONDS,
    STATUS_LOG_THROTTLE,
    TRAJECTORY_INTERVAL,
    UNIFIED_REFLECTION_INTERVAL,
    VOICE_INTERVAL,
)
from .accessors import (
    _get_last_shm_data,
    _get_schema_hub,
    _get_voice,
    _prediction_accuracy_from_shm,
)

logger = logging.getLogger("anima.server")


def _srv():
    import anima_mcp.server as server_mod
    return server_mod


def run_display_loop_startup(_ctx: Any, is_broker_running: bool, mode_change_event: asyncio.Event) -> None:
    """Initialize display/LEDs, check restore markers, run startup learning, and start input polling."""
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


def run_per_tick_telemetry(_ctx: Any, readings: Any, anima: Any, loop_count: int) -> Any:
    """Run recovery arc, health heartbeats, drive events, identity/trajectory/tension, and satisfaction."""
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
            body_eisv_projection=anima_to_body_eisv_projection(
                anima, readings
            ).to_dict(),
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
                external_light_lux=gated_external_light_lux(
                    ((_ctx.last_shm_data if _ctx else None) or {}).get(
                        "light_attribution"
                    )
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
    return _health


def record_server_metacog_episode(_ctx: Any, prediction_error: Any, reflection: Any, reason: str) -> None:
    """Persist a server-origin metacognitive reflection episode."""
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


def run_quiet_world_contemplation(metacog: Any) -> None:
    """Generate and post contemplative questions during quiet periods."""
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


def capture_led_proprioception(ctx, readings) -> None:
    """Record what the LEDs are actually doing, or that it is unknown.

    The efference copy feeds the next metacognitive prediction and
    ``readings.led_brightness`` is persisted to ``system_metrics``, so both
    must be honest: with LEDs unavailable (never initialised, or frame writes
    failing) the copy is None and the brightness is None. Until 2026-11-12 a
    missing copy was written as brightness 0.0 — a measured-looking "off".
    """
    leds = ctx.leds
    prop = None
    if leds is not None and leds.is_available():
        prop = leds.get_proprioceptive_state()
    ctx.led_proprioception = prop
    if readings is not None:
        readings.led_brightness = prop.get("brightness") if prop else None


async def sync_governance_and_render_display(
    _ctx: Any, anima: Any, readings: Any, identity: Any, _health: Any, loop_count: int
) -> None:
    """Sync SHM/fallback governance and update TFT display, LEDs, and voice."""
    _parse_shm_governance_freshness = _srv()._parse_shm_governance_freshness
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
            fallback_decision = await _srv()._server_governance_fallback(anima, readings)
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
        # Auto-brightness is itself an LED action, so feeding it raw
        # VEML7700 lux would close a self-exciting sensor→LED loop. Use
        # only the capture-aligned external residual; unknown stays
        # neutral until attribution is ready.
        _shm = _get_last_shm_data() or {}
        light_level = gated_external_light_lux(
            _shm.get("light_attribution")
        )

        # Get activity brightness from shared memory (broker computes this)
        # - ACTIVE (day/interaction): 1.0
        # - DROWSY (dusk/dawn/30min idle): 0.6
        # - RESTING (night/60min idle): 0.35
        activity_brightness = 1.0
        try:
            # Primary: read from broker's shared memory (single source of truth)
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
            capture_led_proprioception(_ctx, readings)
        except Exception as e:
            if loop_count % ERROR_LOG_THROTTLE == 1:
                print(f"[LEDProprioception] Error: {e}", file=sys.stderr, flush=True)
    elif _ctx.leds:
        if loop_count == 1:
            print("[Loop] LEDs not available (hardware issue?)", file=sys.stderr, flush=True)
        # Unavailable LEDs have no brightness to report. Skipping this block
        # used to leave the last good efference copy in place, so a dead bus
        # kept predicting and persisting the brightness of its last frame.
        capture_led_proprioception(_ctx, readings)
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
                    _voice_external_light = gated_external_light_lux(
                        ((_get_last_shm_data() or {}).get(
                            "light_attribution"
                        ) or {})
                    )
                    voice.update_environment(
                        temperature=readings.ambient_temp_c or 22.0,
                        humidity=readings.humidity_pct or 50.0,
                        light_level=_voice_external_light,
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


async def run_periodic_growth_and_learning_phases(
    _ctx: Any,
    anima: Any,
    readings: Any,
    identity: Any,
    prediction_error: Any,
    _health: Any,
    loop_count: int,
) -> None:
    """Run periodic calibration adaptation, reflection, self-answer, growth, goals, meta-learning, and schema phases."""
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
                lambda: _srv()._lumen_unified_reflect(anima, readings, identity, prediction_error),
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
                lambda: _srv()._lumen_self_answer(anima, readings, identity),
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
            # Raw VEML7700 lux includes DotStar self-glow. Every claim
            # about room light shares the broker's gated residual;
            # raw lux remains untouched physical telemetry.
            _light_attribution = (
                (_get_last_shm_data() or {}).get("light_attribution") or {}
            )
            _external_light = gated_external_light_lux(
                _light_attribution
            )
            environment = {
                "external_light_lux": _external_light,
                "light_attribution_status": _light_attribution.get(
                    "status", "unavailable"
                ),
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

    # Self-derivation: Lumen re-reads its own drawing corpus weekly and
    # applies the coverage cuts and curiosity pivots it derives — the
    # loop that used to wait on an operator script nobody ran. Only
    # schedules; the scan runs off-loop (self_derivation.py).
    # Offset so the first check comes ~10 min after a restart rather
    # than an hour — a deploy-heavy week must not starve it.
    if loop_count % SELF_DERIVATION_CHECK_INTERVAL == 300:
        def self_derive_check():
            from .self_derivation import start_if_due
            start_if_due()

        safe_call(self_derive_check, default=None, log_error=True)

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
            correlations = _srv()._compute_lagged_correlations()

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
        safe_call(
            lambda: _srv()._record_anima_history(anima, _health),
            default=None,
            log_error=True,
        )

    # Governance is handled by the broker (sole UNITARES caller).
    # Server reads governance from SHM in the display block above.

    # === SLOW CLOCK: Self-Schema G_t extraction (every 5 minutes) ===
    # PoC for StructScore visual integrity evaluation
    # Extracts Lumen's self-representation graph and optionally saves for offline analysis
    if (loop_count == 1 or loop_count % SCHEMA_EXTRACTION_INTERVAL == 0) and readings and anima and identity:
        try:
            await safe_call_async(
                lambda: _srv()._extract_and_validate_schema(anima, readings, identity),
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
            await safe_call_async(_srv()._self_reflect, default=None, log_error=True)
        except Exception as e:
            logger.debug("[SelfReflection] Reflection error: %s", e)


async def wait_for_next_display_tick(
    _ctx: Any,
    mode_change_event: asyncio.Event,
    base_delay: float,
    max_delay: float,
    consecutive_errors: int,
) -> bool:
    """Wait for screen-specific refresh delay or immediate mode_change_event."""
    quick_render = False
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
    return quick_render
