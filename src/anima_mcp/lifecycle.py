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

logger = logging.getLogger("anima.server")


def _set_ctx(ctx):
    """Set _ctx in ctx_ref (canonical) and server.py (backward compat)."""
    from . import server
    _set_ctx_ref(ctx)
    server._ctx = ctx  # bridge: server.py still reads _ctx directly in ~30 places


def wake(db_path: str = "anima.db", anima_id: str | None = None):
    """
    Wake up. Call before starting server. Safe, never crashes.

    Retries on SQLite lock errors (e.g. old process still shutting down).

    Args:
        db_path: Path to SQLite database
        anima_id: UUID from environment or database (DO NOT override - use existing identity)
    """
    import time as _time

    from .identity import IdentityStore
    from .server_context import ServerContext
    from .growth import get_growth_system
    from .eisv import get_trajectory_awareness
    from .value_tension import ValueTensionTracker
    from .accessors import (
        _get_schema_hub, _get_calibration_drift,
        _get_readings_and_anima, _get_last_shm_data,
    )
    from .server_state import (
        SHM_STALE_THRESHOLD_SECONDS, SHM_GOVERNANCE_STALE_SECONDS,
        THERMAL_RATE_THRESHOLD, MEMORY_PRESSURE_THRESHOLD,
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
                from .health import get_health_registry
                _health = get_health_registry()
                def _sensor_probe():
                    ctx = _get_ctx()
                    if ctx and ctx.sensors is not None:
                        return True
                    shm = _get_last_shm_data()
                    if shm and "readings" in shm:
                        ts = shm.get("timestamp")
                        if ts:
                            from datetime import datetime
                            try:
                                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                age = (datetime.now(t.tzinfo) - t).total_seconds()
                                return age < SHM_STALE_THRESHOLD_SECONDS * 2  # 30s grace
                            except (ValueError, AttributeError):
                                pass
                        return True  # Data exists but no timestamp — assume ok
                    return False
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
                _health.register("voice", probe=lambda: _get_ctx() and _get_ctx().voice_instance is not None, debounce_seconds=6.0)
                _health.register("anima", probe=lambda: _get_ctx() and _get_ctx().screen_renderer is not None and getattr(_get_ctx().screen_renderer, '_last_anima', None) is not None, debounce_seconds=6.0)

                # Rate-of-change probes — bridge system_metrics → health
                def _thermal_rate_probe():
                    """Check CPU temp isn't rising dangerously fast."""
                    ctx = _get_ctx()
                    if not ctx or not ctx.store:
                        return True  # No store yet — can't check
                    try:
                        rows = ctx.store.get_system_metrics(hours=1.0/12, limit=20)  # last 5 min
                        if len(rows) < 3:
                            return True  # Not enough data yet
                        from datetime import datetime
                        first, last = rows[0], rows[-1]
                        t0 = datetime.fromisoformat(first["timestamp"])
                        t1 = datetime.fromisoformat(last["timestamp"])
                        minutes = (t1 - t0).total_seconds() / 60.0
                        if minutes < 0.5:
                            return True  # Window too short
                        temp0 = first.get("cpu_temp_c")
                        temp1 = last.get("cpu_temp_c")
                        if temp0 is None or temp1 is None:
                            return True
                        rate = (temp1 - temp0) / minutes  # °C/min
                        return rate <= THERMAL_RATE_THRESHOLD
                    except Exception:
                        return True  # Fail open

                def _memory_pressure_probe():
                    """Check memory isn't critically high."""
                    ctx = _get_ctx()
                    if not ctx or not ctx.store:
                        return True
                    try:
                        rows = ctx.store.get_system_metrics(hours=1.0/60, limit=3)  # last 1 min
                        if not rows:
                            return True
                        mem = rows[-1].get("memory_percent")
                        if mem is None:
                            return True
                        return mem < MEMORY_PRESSURE_THRESHOLD
                    except Exception:
                        return True

                _health.register("thermal_trend", probe=_thermal_rate_probe, stale_threshold=3600.0, debounce_seconds=10.0)
                _health.register("memory_pressure", probe=_memory_pressure_probe, stale_threshold=3600.0, debounce_seconds=10.0)

                print(f"[Wake] ✓ Health monitoring registered ({len(_health.subsystem_names())} subsystems)", file=sys.stderr, flush=True)
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
                print("[Server] Display will work but without identity/messages", file=sys.stderr, flush=True)
                _set_ctx(None)
                return


def sleep():
    """Go to sleep. Call on server shutdown."""
    from .accessors import _get_server_bridge

    _ctx = _get_ctx()

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
                loop.create_task(bridge.close())
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
