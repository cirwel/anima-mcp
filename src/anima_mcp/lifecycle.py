"""Server lifecycle — wake (startup) and sleep (shutdown).

Extracted from server.py to isolate the startup/shutdown logic.
Operates on the global _ctx (ServerContext) via ctx_ref (single source of truth).
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from .ctx_ref import get_ctx as _get_ctx, set_ctx as _set_ctx_ref
from .db_paths import resolve_db_path

logger = logging.getLogger("anima.server")

# Strong refs for fire-and-forget shutdown tasks. asyncio only holds a weak
# reference to a bare create_task() result, so without this the task can be
# garbage-collected mid-await before bridge.close() completes (RUF006 / Watcher
# P001). The done-callback discards the ref once the task finishes.
_SHUTDOWN_TASKS: set = set()


def _set_ctx(ctx):
    """Set _ctx in ctx_ref (canonical) and server.py (backward compat)."""
    from . import server
    _set_ctx_ref(ctx)
    server._ctx = ctx  # bridge: server.py still reads _ctx directly in ~30 places


def _register_health_probes():
    """Register subsystem health probes. Shared by the normal wake() path and
    the degraded (store-failed) path — health visibility matters MOST when the
    identity store is down, so this must not depend on the store existing.
    Store-dependent probes report *unknown* (never ok) on ctx.store is None."""
    import math
    from datetime import datetime

    from .error_recovery import note_suppressed
    from .health import get_health_registry, outbound_heartbeat_health
    from .eisv import get_trajectory_awareness
    from .anima_history import get_anima_history
    from .accessors import _get_last_shm_data
    from .server_state import (
        SHM_STALE_THRESHOLD_SECONDS, SHM_GOVERNANCE_STALE_SECONDS,
        THERMAL_RATE_THRESHOLD, MEMORY_PRESSURE_THRESHOLD,
    )

    _health = get_health_registry()

    # Environment channels the broker envelope carries (SensorReadings).
    _ENV_CHANNELS = ("ambient_temp_c", "humidity_pct", "light_lux", "pressure_hpa")

    def _sensor_probe():
        """Fresh broker envelope that actually carries environment readings.

        2026-11-12: this checked only that the envelope was fresh and had a
        ``readings`` key, so it read ok with every environment channel None --
        a live heartbeat around a dead sensor stack.  It now fails when ALL
        environment channels are absent.  One missing channel does not fail it
        (pressure is legitimately absent at times); it is named in the detail.
        """
        shm = _get_last_shm_data()
        if not shm or "readings" not in shm:
            return {"ok": False, "state": "no_envelope"}
        fresh = False
        ts = shm.get("timestamp")
        if isinstance(ts, str) and ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = (datetime.now(t.tzinfo) - t).total_seconds()
                fresh = -5 <= age < SHM_STALE_THRESHOLD_SECONDS * 2
            except (ValueError, AttributeError):
                fresh = False
        if not fresh:
            return {"ok": False, "state": "stale_envelope"}

        readings = shm.get("readings")
        present = []
        for channel in _ENV_CHANNELS:
            value = readings.get(channel) if isinstance(readings, dict) else None
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                present.append(channel)
        missing = [channel for channel in _ENV_CHANNELS if channel not in present]
        if not present:
            state = "no_environment_channels"
        elif missing:
            state = "partial"
        else:
            state = "ok"
        return {
            "ok": bool(present),
            "state": state,
            "channels": {"present": present, "missing": missing},
        }

    _health.register("sensors", probe=_sensor_probe, debounce_seconds=6.0)
    _health.register("display", probe=lambda: _get_ctx() and _get_ctx().display is not None and _get_ctx().display.is_available(), debounce_seconds=6.0)
    _health.register("leds", probe=lambda: _get_ctx() and _get_ctx().leds is not None and _get_ctx().leds.is_available(), debounce_seconds=6.0)
    _health.register("growth", probe=lambda: _get_ctx() and _get_ctx().growth is not None, stale_threshold=90.0)

    def _gov_probe():
        shm = _get_last_shm_data()
        return bool(shm and "governance" in shm and isinstance(shm.get("governance"), dict))

    _health.register("governance", probe=_gov_probe, stale_threshold=SHM_GOVERNANCE_STALE_SECONDS)
    _health.register("drawing", probe=lambda: _get_ctx() and _get_ctx().screen_renderer is not None and hasattr(_get_ctx().screen_renderer, '_canvas'), debounce_seconds=6.0)
    _health.register("trajectory", probe=lambda: get_trajectory_awareness() is not None)
    _health.register(
        "day_summary_writer",
        probe=lambda: get_anima_history().day_summary_health(),
        stale_threshold=36 * 60 * 60,
        debounce_seconds=0.0,
        optional=False,
    )
    _health.register(
        "outbound_heartbeat",
        probe=outbound_heartbeat_health,
        debounce_seconds=0.0,
        optional=False,
        probe_only=True,
    )

    def _voice_probe():
        ctx = _get_ctx()
        voice = ctx.voice_instance if ctx else None
        return bool(
            voice
            and voice.is_running
            and (voice.can_hear or voice.can_speak)
        )

    # optional=True: Lumen is text-first by design — it speaks via the message
    # board, and the audio path may simply not exist on this host. A voice that
    # never started reads "absent", not "degraded"; one that started and then
    # died still reads "degraded".
    _health.register("voice", probe=_voice_probe, debounce_seconds=6.0, optional=True)
    _health.register("anima", probe=lambda: _get_ctx() and _get_ctx().screen_renderer is not None and getattr(_get_ctx().screen_renderer, '_last_anima', None) is not None, debounce_seconds=6.0)

    # Rate-of-change probes — bridge system_metrics → health
    #
    # 2026-11-12: both probes returned True for no store, too few rows, NULL
    # values and any exception ("Fail open"), so they could only ever read ok
    # or threshold-crossed -- never "I cannot tell".  A dead metrics writer
    # was indistinguishable from a cool, roomy Pi.  They now answer with a
    # details dict: no evidence is ``ok: False, state: "unknown"``; a real
    # verdict, good or bad, carries ``evaluated: True``.  Registered
    # optional=True, so health reads ``absent`` until a probe has reached its
    # first verdict (a fresh boot cannot pin overall()) and ``degraded`` when
    # it could tell before and no longer can.  The thresholds are unchanged.
    def _unknown(reason, **details):
        return {"ok": False, "state": "unknown", "reason": reason, **details}

    def _recent_metrics(site, limit):
        """Rows from the last 5 min, or an ``unknown`` verdict explaining why not."""
        ctx = _get_ctx()
        if not ctx or not ctx.store:
            return None, _unknown("no identity store")
        try:
            return ctx.store.get_system_metrics(hours=1.0 / 12, limit=limit), None
        except Exception as e:
            note_suppressed(site, e)
            return None, _unknown(f"system_metrics unreadable: {type(e).__name__}")

    def _thermal_rate_probe():
        """Check CPU temp isn't rising dangerously fast."""
        rows, unknown = _recent_metrics("health.thermal_trend", 20)
        if unknown is not None:
            return unknown
        if len(rows) < 3:
            return _unknown("fewer than 3 system_metrics rows in the last 5 min", rows=len(rows))
        try:
            first, last = rows[0], rows[-1]
            t0 = datetime.fromisoformat(first["timestamp"])
            t1 = datetime.fromisoformat(last["timestamp"])
            minutes = (t1 - t0).total_seconds() / 60.0
            if minutes < 0.5:
                return _unknown("rows span under 30 s", rows=len(rows))
            temp0 = first.get("cpu_temp_c")
            temp1 = last.get("cpu_temp_c")
            if temp0 is None or temp1 is None:
                return _unknown("cpu_temp_c is NULL in system_metrics", rows=len(rows))
            rate = (temp1 - temp0) / minutes  # °C/min
        except Exception as e:
            note_suppressed("health.thermal_trend", e)
            return _unknown(f"system_metrics rows unusable: {type(e).__name__}")
        ok = rate <= THERMAL_RATE_THRESHOLD
        return {
            "ok": ok,
            "evaluated": True,
            "state": "ok" if ok else "rising_fast",
            "rate_c_per_min": round(rate, 2),
            "threshold_c_per_min": THERMAL_RATE_THRESHOLD,
            "rows": len(rows),
        }

    def _memory_pressure_probe():
        """Check memory isn't critically high."""
        # The newest row inside the same 5-minute window the thermal probe
        # reads.  This was a 1-minute window, which did not matter while "no
        # row" meant ok; now that it means unknown, a window of two ~30 s
        # write cadences would flip on a single skipped write.  The row's age
        # is reported so the freshness is visible rather than assumed.
        rows, unknown = _recent_metrics("health.memory_pressure", 3)
        if unknown is not None:
            return unknown
        if not rows:
            return _unknown("no system_metrics rows in the last 5 min", rows=0)
        try:
            latest = rows[-1]
            mem = latest.get("memory_percent")
            if mem is None:
                return _unknown("memory_percent is NULL in system_metrics", rows=len(rows))
            age = (datetime.now() - datetime.fromisoformat(latest["timestamp"])).total_seconds()
            ok = mem < MEMORY_PRESSURE_THRESHOLD
        except Exception as e:
            note_suppressed("health.memory_pressure", e)
            return _unknown(f"system_metrics rows unusable: {type(e).__name__}")
        return {
            "ok": ok,
            "evaluated": True,
            "state": "ok" if ok else "high",
            "memory_percent": round(float(mem), 1),
            "threshold_percent": MEMORY_PRESSURE_THRESHOLD,
            "sample_age_seconds": round(age),
        }

    _health.register("thermal_trend", probe=_thermal_rate_probe, stale_threshold=3600.0, debounce_seconds=10.0, optional=True)
    _health.register("memory_pressure", probe=_memory_pressure_probe, stale_threshold=3600.0, debounce_seconds=10.0, optional=True)

    print(f"[Wake] ✓ Health monitoring registered ({len(_health.subsystem_names())} subsystems)", file=sys.stderr, flush=True)


def wake(db_path: str = "anima.db", anima_id: str | None = None):
    """
    Wake up. Call before starting server. Safe, never crashes.

    Retries on SQLite lock errors (e.g. old process still shutting down).

    On unrecoverable identity-store failure (e.g. corrupt anima.db), enters
    DEGRADED mode: a ServerContext with store=None is kept so the display
    loop, sensors, and LEDs still run. The 2026-07 outage (#106) was 19 days
    of dark screen because this path used to discard the whole context.

    Args:
        db_path: Path to SQLite database
        anima_id: UUID from environment or database (DO NOT override - use existing identity)
    """
    # A bare "anima.db" default would bind the whole server's persistence to
    # whatever directory systemd started it in (#123).
    db_path = resolve_db_path(db_path)

    # The broker owns embodied learned-state snapshots.  Configure server-side
    # singletons as refreshing readers before any schema/display subsystem can
    # instantiate them.  Semantic mutations are handed back through the durable
    # learning-event inbox.
    from .self_model import configure_self_model
    from .preferences import configure_preference_system
    configure_self_model(read_only=True)
    configure_preference_system(read_only=True)

    import time as _time

    from .identity import IdentityStore
    from .server_context import ServerContext
    from .growth import get_growth_system
    from .eisv import get_trajectory_awareness
    from .value_tension import ValueTensionTracker
    from .accessors import (
        _get_schema_hub, _get_calibration_drift,
        _get_readings_and_anima,
    )

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            _ctx = ServerContext()
            _set_ctx(_ctx)
            _ctx.store = IdentityStore(db_path)

            # CRITICAL: Use provided anima_id OR check database for existing identity
            # DO NOT generate new UUID if identity already exists - preserves Lumen's identity
            if anima_id:
                _ctx.anima_id = anima_id
            else:
                # Check if identity exists in database
                conn = _ctx.store._connect()
                existing = conn.execute("SELECT creature_id FROM identity LIMIT 1").fetchone()
                if existing:
                    _ctx.anima_id = existing[0]
                    print(f"[Wake] Using existing identity: {_ctx.anima_id[:8]}...", file=sys.stderr, flush=True)
                else:
                    # Only generate new UUID if no identity exists (first time)
                    _ctx.anima_id = str(uuid.uuid4())
                    print(f"[Wake] Creating new identity: {_ctx.anima_id[:8]}...", file=sys.stderr, flush=True)

            if _ctx.anima_id is None:
                raise ValueError("anima_id must be set before calling wake()")
            identity = _ctx.store.wake(_ctx.anima_id)

            # Identity (name + birthdate) is fundamental to Lumen's existence
            print(f"Awake: {identity.name or '(unnamed)'}")
            print(f"  ID: {identity.creature_id[:8]}...")
            print(f"  Awakening #{identity.total_awakenings}")
            print(f"  Born: {identity.born_at.isoformat()}")
            print(f"  Total alive: {identity.total_alive_seconds:.0f}s")
            print("[Wake] ✓ Identity established - message board will be active", file=sys.stderr, flush=True)

            # Initialize growth system for learning, relationships, and goals
            try:
                _ctx.growth = get_growth_system(db_path=db_path)
                _ctx.growth.born_at = identity.born_at
                print("[Wake] ✓ Growth system initialized", file=sys.stderr, flush=True)
            except Exception as ge:
                import traceback
                print(f"[Wake] Growth system error (non-fatal): {ge}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                _ctx.growth = None

            # Register subsystems with health monitoring
            try:
                _register_health_probes()
            except Exception as he:
                print(f"[Wake] Health monitoring setup error (non-fatal): {he}", file=sys.stderr, flush=True)

            # Bootstrap trajectory awareness from state history
            # and restore last known anima state for warm start
            try:
                import os as _os
                _db_path = _os.path.join(_os.path.expanduser("~"), ".anima", "anima.db")
                _student_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "data", "student_model")
                if not _os.path.isdir(_student_dir):
                    _student_dir = None
                _traj = get_trajectory_awareness(db_path=_db_path, student_model_dir=_student_dir)
                history = _ctx.store.get_recent_state_history(limit=30)
                if history:
                    n = _traj.bootstrap_from_history(history)
                    print(f"[EISV] Bootstrapped trajectory buffer with {n} historical states", file=sys.stderr, flush=True)

                    # Warm start: use last state_history row as initial anticipation
                    last = history[-1]  # Most recent (ascending order)
                    _ctx.warm_start_anima = {
                        "warmth": last["warmth"],
                        "clarity": last["clarity"],
                        "stability": last["stability"],
                        "presence": last["presence"],
                    }
                    print(f"[Wake] Warm start from last state: w={last['warmth']:.2f} c={last['clarity']:.2f} s={last['stability']:.2f} p={last['presence']:.2f}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[EISV] Bootstrap failed (non-fatal): {e}", file=sys.stderr, flush=True)

            # Initialize SchemaHub and check for gap from previous session
            try:
                hub = _get_schema_hub()
                gap_delta = hub.on_wake()
                if gap_delta:
                    print(f"[SchemaHub] Woke after {gap_delta.duration_seconds:.0f}s gap", file=sys.stderr, flush=True)
                else:
                    print("[SchemaHub] Initialized (no previous schema found)", file=sys.stderr, flush=True)

                # Seed hub's trajectory from existing trajectory system so
                # trajectory nodes appear immediately, not after ~7 hours.
                try:
                    from .trajectory import compute_trajectory_signature
                    from .anima_history import get_anima_history
                    from .self_model import get_self_model as _get_sm
                    _hub_traj = compute_trajectory_signature(
                        growth_system=_ctx.growth,
                        self_model=_get_sm(),
                        anima_history=get_anima_history(),
                    )
                    if _hub_traj and _hub_traj.observation_count > 0:
                        hub.last_trajectory = _hub_traj
                        print(f"[SchemaHub] Seeded trajectory: {_hub_traj.observation_count} obs", file=sys.stderr, flush=True)
                except Exception as te:
                    print(f"[SchemaHub] Trajectory seed failed (non-fatal): {te}", file=sys.stderr, flush=True)

                # Seed hub with initial schema so Pi LCD and /schema-data have
                # data immediately, not after the first 20-min main loop tick.
                try:
                    from .self_model import get_self_model as _gsm_init
                    _sm_init = None
                    try:
                        _sm_init = _gsm_init()
                    except Exception as e:
                        logger.debug("[SchemaHub] SelfModel init for seed: %s", e)
                    readings_init, anima_init = _get_readings_and_anima()
                    init_schema = hub.compose_schema(
                        identity=identity,
                        anima=anima_init,
                        readings=readings_init,
                        growth_system=_ctx.growth,
                        self_model=_sm_init,
                        light_attribution=(
                            (_ctx.last_shm_data or {}).get("light_attribution")
                            if _ctx else None
                        ),
                    )
                    print(f"[SchemaHub] Seeded initial schema: {len(init_schema.nodes)}n {len(init_schema.edges)}e", file=sys.stderr, flush=True)
                except Exception as seed_e:
                    print(f"[SchemaHub] Initial seed failed (non-fatal): {seed_e}", file=sys.stderr, flush=True)
            except Exception as she:
                print(f"[SchemaHub] Init failed (non-fatal): {she}", file=sys.stderr, flush=True)

            # Initialize CalibrationDrift (load from disk or create fresh)
            try:
                drift = _get_calibration_drift()
                midpoints = drift.get_midpoints()
                any_drift = any(abs(m - 0.5) > 0.001 for m in midpoints.values())
                if any_drift:
                    print(f"[CalDrift] Loaded with drift: {', '.join(f'{k}={v:.3f}' for k, v in midpoints.items() if abs(v - 0.5) > 0.001)}", file=sys.stderr, flush=True)
                    # Apply restart decay if there was a significant gap
                    if _ctx.wake_gap and _ctx.wake_gap.total_seconds() >= 86400:  # 24h+
                        gap_hours = _ctx.wake_gap.total_seconds() / 3600
                        drift.apply_restart_decay(gap_hours)
                        print(f"[CalDrift] Applied restart decay for {gap_hours:.0f}h gap", file=sys.stderr, flush=True)
                else:
                    print("[CalDrift] Initialized (no prior drift)", file=sys.stderr, flush=True)
            except Exception as cde:
                print(f"[CalDrift] Init failed (non-fatal): {cde}", file=sys.stderr, flush=True)

            # Initialize ValueTensionTracker (transient — no persistence needed)
            _ctx.tension_tracker = ValueTensionTracker()
            print("[Tension] Initialized value tension tracker", file=sys.stderr, flush=True)

            return  # Success
        except Exception as e:
            _ctx = _get_ctx()
            is_lock_error = "database is locked" in str(e) or "database is locked" in repr(e)
            if is_lock_error and attempt < max_attempts:
                wait = attempt * 2  # 2s, 4s, 6s, 8s
                print(f"[Wake] Database locked (attempt {attempt}/{max_attempts}), retrying in {wait}s...", file=sys.stderr, flush=True)
                # Close the failed connection before retrying
                if _ctx and _ctx.store and _ctx.store._conn:
                    try:
                        _ctx.store._conn.close()
                    except Exception as e:
                        logger.debug("[Wake] Connection close error: %s", e)
                _set_ctx(None)
                _time.sleep(wait)
            else:
                print("[Wake] ❌ ERROR: Identity store failed!", file=sys.stderr, flush=True)
                print(f"[Wake] Error details: {e}", file=sys.stderr, flush=True)
                print("[Wake] Impact: Message board will NOT post, identity features unavailable", file=sys.stderr, flush=True)
                # DEGRADED mode (#106): keep a context with store=None so the
                # display loop, sensors, and LEDs still run. Discarding the
                # whole context here left the TFT dark for 19 days in Jul 2026
                # while the log promised "display will work".
                _ctx = ServerContext()
                _ctx.store = None
                _ctx.anima_id = anima_id  # env-provided id if any; DB one unavailable
                _ctx.tension_tracker = ValueTensionTracker()
                _set_ctx(_ctx)
                try:
                    _register_health_probes()
                except Exception as he:
                    print(f"[Wake] Health monitoring setup error (non-fatal): {he}", file=sys.stderr, flush=True)
                print("[Wake] ⚠ DEGRADED MODE: display/sensors/LEDs will run; identity, message board, growth, schema disabled until the store recovers (restart after repairing anima.db)", file=sys.stderr, flush=True)
                return


def sleep():
    """Go to sleep. Call on server shutdown."""
    from .accessors import _get_server_bridge

    _ctx = _get_ctx()

    # Persist metacognition (baselines, curiosity log, self-forecast). It
    # otherwise saves only every 100 observations, so a shutdown could lose
    # ~10 minutes of what Lumen learned about itself.
    if _ctx and getattr(_ctx, "metacog_monitor", None) is not None:
        try:
            _ctx.metacog_monitor.save()
        except Exception as e:
            try:
                print(f"[Sleep] Error saving metacognition: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    # Persist calibration drift state
    if _ctx and _ctx.calibration_drift:
        try:
            drift_path = Path.home() / ".anima" / "calibration_drift.json"
            drift_path.parent.mkdir(parents=True, exist_ok=True)
            _ctx.calibration_drift.save(str(drift_path))
            try:
                print("[Sleep] Calibration drift saved", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass
        except Exception as e:
            try:
                print(f"[Sleep] Error saving calibration drift: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    # Persist schema for gap recovery on next wake
    if _ctx and _ctx.schema_hub:
        try:
            if _ctx.schema_hub.persist_schema():
                try:
                    print("[Sleep] Schema persisted for gap recovery", file=sys.stderr, flush=True)
                except (ValueError, OSError):
                    pass
        except Exception as e:
            try:
                print(f"[Sleep] Error persisting schema: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    # Persist trajectory for anomaly detection on next session
    if _ctx and _ctx.growth:
        try:
            from .trajectory import compute_trajectory_signature, save_trajectory
            from .anima_history import get_anima_history
            from .self_model import get_self_model
            sig = compute_trajectory_signature(
                growth_system=_ctx.growth,
                self_model=get_self_model(),
                anima_history=get_anima_history(),
            )
            if save_trajectory(sig):
                try:
                    print("[Sleep] Trajectory persisted", file=sys.stderr, flush=True)
                except (ValueError, OSError):
                    pass
        except Exception as e:
            try:
                print(f"[Sleep] Error persisting trajectory: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    # Persist canvas state so in-progress drawings survive restart
    if _ctx and _ctx.screen_renderer:
        try:
            canvas = _ctx.screen_renderer.drawing_engine.canvas
            if canvas.pixels:
                canvas.save_to_disk()
                try:
                    print(f"[Sleep] Canvas saved ({len(canvas.pixels)}px)", file=sys.stderr, flush=True)
                except (ValueError, OSError):
                    pass
        except Exception as e:
            try:
                print(f"[Sleep] Error saving canvas: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    # Close server-side UNITARES bridge if it was used.
    # NOTE: bridge.close() is async. In HTTP mode, the lifespan awaits it
    # before calling sleep(). This fire-and-forget is a best-effort fallback
    # for non-lifespan shutdown paths (stdio signal handler, main() finally).
    bridge = _get_server_bridge()
    if bridge:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _close_task = loop.create_task(bridge.close())
                _SHUTDOWN_TASKS.add(_close_task)
                _close_task.add_done_callback(_SHUTDOWN_TASKS.discard)
            else:
                loop.run_until_complete(bridge.close())
        except Exception as e:
            logger.debug("[Sleep] Bridge close error: %s", e)

    # Close SelfReflection SQLite connection
    try:
        from .self_reflection import get_reflection_system
        get_reflection_system().close()
    except Exception as e:
        logger.debug("[Sleep] SelfReflection close error: %s", e)

    # Close PrimitiveLanguageSystem SQLite connection
    try:
        from .primitive_language import get_language_system
        get_language_system().close()
    except Exception as e:
        logger.debug("[Sleep] PrimitiveLanguageSystem close error: %s", e)

    # Close TrajectoryAwareness SQLite connection
    try:
        from .eisv import get_trajectory_awareness
        get_trajectory_awareness().close()
    except Exception as e:
        logger.debug("[Sleep] TrajectoryAwareness close error: %s", e)

    # Stop voice system if running
    if _ctx and _ctx.voice_instance:
        try:
            _ctx.voice_instance.stop()
            _ctx.voice_instance = None
        except Exception as e:
            try:
                print(f"[Sleep] Error stopping voice: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass

    if _ctx and _ctx.store:
        try:
            session_seconds = _ctx.store.sleep()
            try:
                print(f"Sleeping. Session: {session_seconds:.0f}s", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                # stdout/stderr might be closed - ignore
                pass
            # Checkpoint WAL to prevent corruption on next startup
            for attempt in range(3):
                try:
                    if _ctx.store._conn:
                        _ctx.store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        break
                except Exception as e:
                    logger.debug("[Sleep] WAL checkpoint attempt %d failed: %s", attempt + 1, e)
                    if attempt < 2:
                        import time
                        time.sleep(0.5)
            _ctx.store.close()
        except Exception as e:
            # Don't crash on shutdown errors
            try:
                print(f"[Sleep] Error during sleep: {e}", file=sys.stderr, flush=True)
            except (ValueError, OSError):
                pass
        finally:
            _set_ctx(None)
    else:
        _set_ctx(None)
