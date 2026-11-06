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
from datetime import datetime
from pathlib import Path

from mcp.server.stdio import stdio_server
from .primitive_language import get_language_system
from .eisv import get_trajectory_awareness
from .state_history import maybe_record_state_history
from .tool_registry import get_fastmcp, create_server, HAS_FASTMCP
from .server_context import ServerContext
from .server_state import (  # noqa: F401
    # Constants
    SHM_GOVERNANCE_STALE_SECONDS as SHM_GOVERNANCE_STALE_SECONDS,
    SHM_STALE_THRESHOLD_SECONDS,
    LOOP_BASE_DELAY_SECONDS, LOOP_MAX_DELAY_SECONDS,
    METACOG_INTERVAL, AGENCY_INTERVAL, SELF_MODEL_INTERVAL,
    PRIMITIVE_LANG_INTERVAL, VOICE_INTERVAL, GROWTH_INTERVAL,
    TRAJECTORY_INTERVAL, SERVER_GOVERNANCE_FALLBACK_SECONDS,
    LEARNING_INTERVAL, PREFERENCE_DECAY_INTERVAL_SECONDS,
    SYSTEM_METRICS_RECORD_INTERVAL, SYSTEM_METRICS_PRUNE_INTERVAL,
    STATE_HISTORY_RECORD_SECONDS,
    SYSTEM_METRICS_RETENTION_HOURS,
    SELF_MODEL_SAVE_INTERVAL, SCHEMA_EXTRACTION_INTERVAL,
    EXPRESSION_INTERVAL, UNIFIED_REFLECTION_INTERVAL, SELF_ANSWER_INTERVAL,
    GOAL_SUGGEST_INTERVAL, GOAL_CHECK_INTERVAL, META_LEARNING_INTERVAL,
    SELF_DERIVATION_CHECK_INTERVAL,
    ERROR_LOG_THROTTLE, STATUS_LOG_THROTTLE, DISPLAY_LOG_THROTTLE,
    WARN_LOG_THROTTLE,
    METACOG_SURPRISE_THRESHOLD, is_broker_running as _is_broker_running,
)

# SchemaHub, CalibrationDrift, ValueTensionTracker — imported for type hints / lazy init

# State accessors (lazy singletons, SHM reads, etc.) — live in accessors.py
from .accessors import (  # noqa: F401
    _get_store as _get_store, _get_sensors as _get_sensors,
    _get_shm_client as _get_shm_client, _get_server_bridge as _get_server_bridge,
    _get_schema_hub, _get_calibration_drift as _get_calibration_drift,
    _get_selfhood_context as _get_selfhood_context, _get_metacog_monitor,
    _get_warm_start_anticipation as _get_warm_start_anticipation,
    _get_readings_and_anima, _prediction_accuracy_from_shm,
    _get_display as _get_display, _get_last_shm_data,
    _get_screen_renderer as _get_screen_renderer, _get_leds as _get_leds,
    _get_growth as _get_growth, _get_display_update_task as _get_display_update_task,
    _get_activity as _get_activity, _get_led_brightness as _get_led_brightness,
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


def _predict_with_led_proprioception(metacog):
    """Predict with the context's canonical, nullable LED efference copy."""
    return metacog.predict(led_brightness=_get_led_brightness())

# Phase helper functions — delegated to loop_phases.py
from .loop_phases import (  # noqa: E402,F401
    handle_surprise_question,
    handle_self_surprise,
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


def _record_anima_history(anima, health_registry=None):
    """Record live state and run the server-owned day-summary writer.

    The broker owns activity transitions, but this process owns the live
    AnimaHistory deque.  Keeping both recording and consolidation here avoids
    summarizing a broker-local snapshot that was loaded from disk only once.
    """
    from .anima_history import get_anima_history

    history = get_anima_history()
    history.record_from_anima(anima)
    summary = history.maybe_consolidate_daily()
    if summary is not None and health_registry is not None:
        # A writer heartbeat means an atomic commit completed, not merely that
        # the scheduling path ran.
        health_registry.heartbeat("day_summary_writer")
    if summary is not None:
        logger.info(
            "[AnimaHistory] Day summary committed: %d observations through %s",
            summary.n_observations,
            summary.date,
        )
    return summary


async def _update_display_loop():
    """Background task to continuously update display and LEDs."""
    if _ctx is None:
        logger.warning("[Loop] No context - wake() may have failed")
        return
    import sys
    from .display_loop import (
        record_server_metacog_episode,
        run_display_loop_startup,
        run_per_tick_telemetry,
        run_periodic_growth_and_learning_phases,
        run_quiet_world_contemplation,
        sync_governance_and_render_display,
        wait_for_next_display_tick,
    )

    print("[Loop] Starting", file=sys.stderr, flush=True)

    # Check if we are in "Reader Mode" (Broker running)
    # Read sensors from shared memory (no direct sensor takeover)
    is_broker_running = _is_broker_running()
    mode_change_event = asyncio.Event()
    run_display_loop_startup(_ctx, is_broker_running, mode_change_event)

    loop_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    base_delay = LOOP_BASE_DELAY_SECONDS  # 200ms = 5Hz refresh for all screens
    max_delay = LOOP_MAX_DELAY_SECONDS
    quick_render = False  # Set when mode_change_event fires — skip heavy subsystems

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
                    logger.debug("[Loop] Failed to get readings/anima - broker=%s, store=%s", is_broker_running, _ctx.store is not None if _ctx else False)
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning("[Loop] Too many consecutive errors (%d), backing off", consecutive_errors)
                    await asyncio.sleep(min(base_delay * (2 ** min(consecutive_errors // 5, 4)), max_delay))
                else:
                    await asyncio.sleep(base_delay)
                continue

            consecutive_errors = 0  # Reset on success
            _health = run_per_tick_telemetry(_ctx, readings, anima, loop_count)

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

                    # Self-prediction: did Lumen move between active / drowsy /
                    # resting the way it expected of itself? A stale or missing
                    # level is unknown, never "stayed the same".
                    try:
                        from .self_prediction import activity_level_from_shm
                        _now = datetime.now()
                        _self_surprise = metacog.self_forecaster.observe(
                            activity_level_from_shm(
                                _get_last_shm_data(), _now,
                                SHM_STALE_THRESHOLD_SECONDS),
                            _now)
                        if _self_surprise:
                            handle_self_surprise(_self_surprise)
                    except Exception as _se:
                        logger.debug("[SelfPrediction] error: %s", _se)

                    # Log surprise level periodically (every 60 loops = ~2 min)
                    if prediction_error and loop_count % WARN_LOG_THROTTLE == 0:
                        logger.debug("[Metacog] Surprise level: %.3f (threshold: %s)", prediction_error.surprise, METACOG_SURPRISE_THRESHOLD)

                    # Check if surprise warrants reflection
                    if prediction_error and prediction_error.surprise > METACOG_SURPRISE_THRESHOLD:
                        should_reflect, reason = metacog.should_reflect(prediction_error)

                        if should_reflect:
                            reflection = metacog.reflect(prediction_error, anima, readings, trigger=reason)
                            record_server_metacog_episode(_ctx, prediction_error, reflection, reason)
                            handle_surprise_question(metacog, prediction_error)

                            if reflection.observation:
                                logger.debug("[Metacog] Reflection: %s", reflection.observation)

                    elif prediction_error is not None:
                        run_quiet_world_contemplation(metacog)

                    # Make prediction for NEXT iteration
                    _predict_with_led_proprioception(metacog)

                except Exception as e:
                    if loop_count % STATUS_LOG_THROTTLE == 1:
                        print(f"[Metacog] Error (non-fatal): {e}",
                              file=sys.stderr, flush=True)

            # === AGENCY: Action selection and learning ===
            if not _skip_subsystems and loop_count % AGENCY_INTERVAL == 0:
                try:
                    _run_agency_phase(
                        _ctx, anima, readings, prediction_error, loop_count,
                    )
                except Exception as e:
                    if loop_count % STATUS_LOG_THROTTLE == 1:
                        print(f"[Agency] Error (non-fatal): {e}", file=sys.stderr, flush=True)

            # === SELF-MODEL: Refresh the broker-owned belief snapshot ===
            if not _skip_subsystems and loop_count % SELF_MODEL_INTERVAL == 0 and anima:
                _run_self_model_phase(anima, readings, prediction_error, base_delay, loop_count)

            # === PRIMITIVE LANGUAGE: Emergent expression through learned tokens ===
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

                        # Record suggestion recall, then explicitly apply the
                        # same score to the originating EISV token weights.
                        if _suggestion and utterance:
                            try:
                                _traj = get_trajectory_awareness()
                                _suggestion_recall = _traj.record_suggestion_recall(
                                    _suggestion.get("suggested_tokens"),
                                    utterance.tokens,
                                    shape=_suggestion.get("shape"),
                                )
                                if _suggestion_recall is not None:
                                    _traj.record_eisv_weight_feedback(
                                        _suggestion.get("eisv_tokens", []),
                                        _suggestion_recall,
                                    )
                                    print(f"[PrimitiveLang] Suggestion recall: {_suggestion_recall:.2f}", file=sys.stderr, flush=True)
                            except Exception as e:
                                if loop_count % ERROR_LOG_THROTTLE == 1:
                                    print(f"[SuggestionRecall] Error: {e}", file=sys.stderr, flush=True)

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
            identity = _ctx.store.get_identity() if _ctx and _ctx.store else None
            if identity is None and (_ctx is None or _ctx.store is None):
                if loop_count == 1:
                    print("[Loop] WARNING: Identity store not initialized (wake() may have failed) - display will show face without identity info", file=sys.stderr, flush=True)

            await sync_governance_and_render_display(_ctx, anima, readings, identity, _health, loop_count)

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

            # Durable anima + sensor history, on Lumen's own schedule. The loop is
            # the only writer: reads no longer record (state_history.py).
            maybe_record_state_history(
                _ctx, anima, readings, _get_last_shm_data(), STATE_HISTORY_RECORD_SECONDS
            )

            # System metrics pruning: Every 1800 iterations (~1h), delete old rows
            if loop_count % SYSTEM_METRICS_PRUNE_INTERVAL == 0 and loop_count > 0 and _ctx and _ctx.store:
                try:
                    _pruned = _ctx.store.prune_system_metrics(SYSTEM_METRICS_RETENTION_HOURS)
                    if _pruned > 0:
                        logger.debug("[Metrics] Pruned %d old system_metrics rows", _pruned)
                except Exception:
                    pass

            await run_periodic_growth_and_learning_phases(
                _ctx, anima, readings, identity, prediction_error, _health, loop_count
            )

            quick_render = await wait_for_next_display_tick(
                _ctx, mode_change_event, base_delay, max_delay, consecutive_errors
            )

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
            # Shared with the REST gate so one hostname list governs both
            # surfaces; they drifted apart once already.
            from .rest_api import _EXTERNAL_HOSTS
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
            # Only cloudflared, which reaches us over loopback, may speak for
            # the client. "*" put uvicorn in always_trust, where it reads
            # X-Forwarded-For[0] -- the entry the *caller* wrote, ahead of the
            # address Cloudflare appends. Naming the hop makes uvicorn scan the
            # list from the right instead, past trusted proxies, onto the
            # address Cloudflare actually observed.
            forwarded_allow_ips=os.environ.get(
                "ANIMA_FORWARDED_ALLOW_IPS", "127.0.0.1,::1"
            ),
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

    # Record which code this process is actually running, OUTSIDE the repo
    # tree (~/.anima/deployed_ref.json). Deploys reset/clean/overlay the
    # checkout, so on the Pi `git log` cannot be trusted to name the running
    # ref — this marker, written at every process start, can. Best-effort:
    # startup must never fail on marker problems (see deploy_marker.py).
    try:
        from .deploy_marker import write_startup_marker
        if not write_startup_marker():
            print("[Server] Warning: could not write deployed-ref marker",
                  file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Server] Warning: deployed-ref marker error: {e}",
              file=sys.stderr, flush=True)

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
