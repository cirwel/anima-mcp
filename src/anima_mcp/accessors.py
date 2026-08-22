"""
State accessors for the anima-mcp server.

All _get_* functions live here. They provide thread-safe, None-safe access
to ServerContext subsystems with lazy initialization.

_ctx lives in ctx_ref.py (single source of truth). This module re-exports
set_ctx() for backward compatibility and reads _ctx via the ctx_ref module.
Server flags (SERVER_READY, etc.) remain in server.py.
"""

import logging
import math
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .sensors import get_sensors, SensorBackend, SensorReadings
from .anima import sense_self_with_memory, Anima
from .memory import anticipate_state
from .display import get_display, DisplayRenderer
from .shared_memory import SharedMemoryClient
from .config import get_calibration
from .server_state import (
    SHM_STALE_THRESHOLD_SECONDS,
    anima_from_dict as _anima_from_dict,
    is_broker_running as _is_broker_running,
    readings_from_dict as _readings_from_dict,
)

# SchemaHub, CalibrationDrift — imported for type hints / lazy init
from .schema_hub import SchemaHub
from .calibration_drift import CalibrationDrift

# ctx_ref is the single source of truth for _ctx
from . import ctx_ref as _cr
from .ctx_ref import set_ctx as set_ctx  # re-exported for lifecycle.py and tests

logger = logging.getLogger("anima.server")

# ============================================================
# Global State
# ============================================================

# Thread lock for lazy-init singletons (metacog, unitares bridge)
_state_lock = threading.Lock()

# VOICE_MODE controls how Lumen speaks: "text" (message board), "audio" (TTS), "both"
VOICE_MODE = os.environ.get("LUMEN_VOICE_MODE", "text")  # Default: text only


# ============================================================
# Accessors — safe, lazy, None-tolerant
# ============================================================

def _get_store():
    """Get identity store - safe, returns None if not available instead of crashing."""
    if _cr._ctx is None:
        print("[Server] Warning: Store not initialized (wake() may have failed)", file=sys.stderr)
        return None
    return _cr._ctx.store


def _get_sensors() -> SensorBackend:
    """Return the server's SHM-backed sensor facade.

    Accessors are part of the MCP/mind process, so backend selection is not
    delegated to Pi auto-detection.  This keeps the single-I2C-owner invariant
    true even if a service environment file or manual launch omits the systemd
    safeguard.
    """
    if _cr._ctx is None:
        return get_sensors(backend="shm")
    if _cr._ctx.sensors is None:
        _cr._ctx.sensors = get_sensors(backend="shm")
    return _cr._ctx.sensors


def _get_shm_client() -> SharedMemoryClient:
    """Get shared memory client for reading broker data."""
    if _cr._ctx is None:
        return SharedMemoryClient(mode="read", backend="file")
    if _cr._ctx.shm_client is None:
        _cr._ctx.shm_client = SharedMemoryClient(mode="read", backend="file")
    return _cr._ctx.shm_client


def _get_server_bridge():
    """Lazy UNITARES bridge for server-side fallback check-in.

    Only used when the broker's SHM governance is stale/local for too long.
    The server's native async event loop avoids the broker's thread+loop issues.
    """
    bridge = _cr._ctx.server_bridge if _cr._ctx else None
    if bridge is not None:
        return bridge
    unitares_url = os.environ.get("UNITARES_URL")
    if not unitares_url:
        return None
    try:
        from .unitares_bridge import UnitaresBridge
        bridge = UnitaresBridge(unitares_url=unitares_url, timeout=8.0)
        store = _get_store()
        if store:
            identity = store.get_identity()
            if identity:
                bridge.set_agent_id(identity.creature_id)
                bridge.set_session_id(f"anima-server-{identity.creature_id[:8]}")
        if _cr._ctx:
            _cr._ctx.server_bridge = bridge
        return bridge
    except Exception as e:
        logger.debug("[Governance] Bridge init failed: %s", e)
        return None


def _get_schema_hub() -> SchemaHub:
    """Get or create the SchemaHub singleton for schema composition."""
    if _cr._ctx is None:
        return SchemaHub()  # Transient when not woken
    if _cr._ctx.schema_hub is None:
        _cr._ctx.schema_hub = SchemaHub()
    return _cr._ctx.schema_hub


def _get_calibration_drift() -> CalibrationDrift:
    """Get or create the CalibrationDrift singleton."""
    if _cr._ctx is None:
        return CalibrationDrift()
    if _cr._ctx.calibration_drift is None:
        drift_path = Path.home() / ".anima" / "calibration_drift.json"
        if drift_path.exists():
            try:
                _cr._ctx.calibration_drift = CalibrationDrift.load(str(drift_path))
                print(f"[CalDrift] Loaded drift state from {drift_path}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[CalDrift] Failed to load drift (using fresh): {e}", file=sys.stderr, flush=True)
                _cr._ctx.calibration_drift = CalibrationDrift()
        else:
            _cr._ctx.calibration_drift = CalibrationDrift()
    return _cr._ctx.calibration_drift


def _get_selfhood_context() -> Dict[str, Any] | None:
    """Get read-only selfhood context for LLM narrator.

    Returns a dict with current state of drift, tensions, and preference
    weights — or None if no selfhood systems are active.  LLM output NEVER
    feeds back into drift rates, conflict thresholds, or preference weights.
    """
    context: Dict[str, Any] = {}
    drift = _get_calibration_drift() if _cr._ctx else None
    if drift:
        context["drift_offsets"] = drift.get_offsets()
    tension = _cr._ctx.tension_tracker if _cr._ctx else None
    if tension:
        active = tension.get_active_conflicts(last_n=5)
        context["active_tensions"] = [
            {"dim_a": c.dim_a, "dim_b": c.dim_b, "category": c.category}
            for c in active
        ]

    # Preference weights (read-only snapshot)
    try:
        from .preferences import get_preference_system
        pref = get_preference_system()
        if pref and hasattr(pref, '_preferences'):
            context["weight_changes"] = {
                d: p.influence_weight for d, p in pref._preferences.items()
                if d in ("warmth", "clarity", "stability", "presence")
            }
    except Exception as e:
        logger.debug("[Selfhood] Preference weight read error: %s", e)

    return context if context else None


def _get_metacog_monitor():
    """Get metacognitive monitor - Lumen's self-awareness through prediction errors."""
    if _cr._ctx is None:
        return None
    if _cr._ctx.metacog_monitor is None:
        with _state_lock:
            if _cr._ctx.metacog_monitor is None:
                from .metacognition import MetacognitiveMonitor
                _cr._ctx.metacog_monitor = MetacognitiveMonitor(
                    surprise_threshold=0.3,  # Trigger reflection at 30% surprise
                    reflection_cooldown_seconds=120.0,  # 2 min between reflections
                )
    return _cr._ctx.metacog_monitor


def _get_warm_start_anticipation():
    """Consume warm start state as a synthetic Anticipation (one-shot).

    On the first sense after wake, this blends last-known anima state
    with current sensor readings so Lumen doesn't start from scratch.
    After a gap, confidence is reduced so Lumen relies more on fresh sensors.
    """
    if not _cr._ctx or _cr._ctx.warm_start_anima is None:
        return None
    from .memory import Anticipation
    state = _cr._ctx.warm_start_anima
    _cr._ctx.warm_start_anima = None  # One-shot: only used for first sense

    # Scale confidence by staleness — longer gaps mean less trust in old state
    confidence = 0.6
    description = "warm start from last shutdown"
    if _cr._ctx.wake_gap:
        gap_hours = _cr._ctx.wake_gap.total_seconds() / 3600
        if gap_hours > 24:
            confidence = 0.1
            description = f"warm start after {gap_hours:.0f}h absence"
        elif gap_hours > 1:
            confidence = max(0.15, 0.6 - gap_hours * 0.02)
            description = f"warm start after {gap_hours:.1f}h gap"
        elif gap_hours > 1 / 12:  # > 5 minutes
            confidence = 0.4
            description = f"warm start after {gap_hours * 60:.0f}m gap"

    return Anticipation(
        warmth=state["warmth"],
        clarity=state["clarity"],
        stability=state["stability"],
        presence=state["presence"],
        confidence=confidence,
        sample_count=1,
        bucket_description=description,
    )


def _prediction_accuracy_from_shm(shm_data: Any) -> float | None:
    """Read broker-owned prediction accuracy without opening a stale singleton."""
    if not isinstance(shm_data, dict):
        return None
    learning = shm_data.get("learning")
    if not isinstance(learning, dict):
        return None
    stats = learning.get("prediction_accuracy")
    if not isinstance(stats, dict) or stats.get("insufficient_data"):
        return None
    error = stats.get("overall_mean_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)):
        return None
    error = float(error)
    if not math.isfinite(error):
        return None
    return 1.0 - max(0.0, min(1.0, error))


def _get_readings_and_anima(fallback_to_sensors: bool = False) -> tuple[SensorReadings | None, Anima | None]:
    """
    Read sensor data from shared memory (broker) or fallback to direct sensor access.

    Returns:
        Tuple of (readings, anima) or (None, None) if unavailable
    """
    # Try shared memory first (broker mode)
    # SharedMemoryClient.read() already returns envelope["data"] (the inner dict)
    shm_client = _get_shm_client()
    shm_data = shm_client.read()
    if _cr._ctx:
        _cr._ctx.last_shm_data = shm_data  # Cache for reuse within same iteration

    # Check if shared memory data is recent
    shm_stale = True
    shm_valid = False
    if shm_data:
        try:
            # Check if we have the required fields
            if "readings" in shm_data and "anima" in shm_data:
                # Check timestamp in shared memory data (broker writes "timestamp" field)
                timestamp_str = shm_data.get("timestamp")
                if isinstance(timestamp_str, str) and timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    age_seconds = (datetime.now(timestamp.tzinfo) - timestamp).total_seconds()
                    shm_stale = (
                        age_seconds > SHM_STALE_THRESHOLD_SECONDS
                        or age_seconds < -5
                    )
                    if not shm_stale:
                        shm_valid = True
        except Exception as e:
            logger.debug("[Server] Error checking shared memory timestamp: %s", e)
            # A malformed heartbeat is not evidence of a present body.  Keep
            # the reading unknown until the broker publishes a valid timestamp.
            shm_valid = False

    # Try to use shared memory if valid
    if shm_valid:
        try:
            # Reconstruct SensorReadings from shared memory
            readings = _readings_from_dict(shm_data["readings"])

            # The broker owns sensing, calibration, anticipation, and mood
            # momentum.  Its published anima is therefore authoritative.
            anima = _anima_from_dict(shm_data["anima"], readings)

            return readings, anima
        except Exception as e:
            logger.debug("[Server] Error reading from shared memory: %s", e)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fall through to sensor fallback

    # Direct access is opt-in.  This module runs in the MCP server, while the
    # broker owns /dev/i2c-1; stale or malformed SHM must become "unknown", not
    # an implicit second hardware owner.  Standalone development callers can
    # still request a mock/direct backend explicitly.
    if fallback_to_sensors:
        # Check if broker is running (for logging purposes)
        broker_running = _is_broker_running()

        # Log why we're falling back (throttled to avoid spam)
        import time as _time
        _now = _time.time()
        if not hasattr(_get_readings_and_anima, '_last_fallback_log'):
            _get_readings_and_anima._last_fallback_log = 0.0
        if _now - _get_readings_and_anima._last_fallback_log > 30.0:
            _get_readings_and_anima._last_fallback_log = _now
            if broker_running and not shm_valid:
                logger.debug("[Server] Broker running but shared memory %s - using direct sensor fallback", 'stale' if shm_stale else 'invalid/empty')
            elif not broker_running:
                logger.debug("[Server] Broker not running - using direct sensor access")

        try:
            sensors = _get_sensors()
            if sensors is None:
                logger.warning("[Server] Sensors not initialized - cannot read")
                return None, None

            readings = sensors.read()
            if readings is None:
                logger.debug("[Server] Sensor read returned None")
                return None, None

            calibration = get_calibration()

            # Use warm start anticipation (first sense after wake) or memory
            anticipation = _get_warm_start_anticipation() or anticipate_state(readings.to_dict() if readings else {})

            drift = _get_calibration_drift()
            anima = sense_self_with_memory(readings, anticipation, calibration, drift_midpoints=drift.get_midpoints())
            if anima is None:
                logger.warning("[Server] Failed to create anima from readings")
                return None, None

            return readings, anima
        except Exception as e:
            logger.warning("[Server] Error reading sensors directly: %s", e)
            import traceback
            traceback.print_exc(file=sys.stderr)

    return None, None


def _get_display() -> DisplayRenderer:
    if _cr._ctx is None:
        return get_display()
    if _cr._ctx.display is None:
        _cr._ctx.display = get_display()
    return _cr._ctx.display


def _get_last_shm_data() -> dict | None:
    """Cached per-iteration shared memory read. Used by handlers and display loop."""
    return _cr._ctx.last_shm_data if _cr._ctx else None


def _get_screen_renderer():
    return _cr._ctx.screen_renderer if _cr._ctx else None


def _get_leds():
    return _cr._ctx.leds if _cr._ctx else None


def _get_growth():
    return _cr._ctx.growth if _cr._ctx else None


def _get_display_update_task():
    return _cr._ctx.display_update_task if _cr._ctx else None


def _get_activity():
    return _cr._ctx.activity if _cr._ctx else None


def _get_led_brightness() -> float | None:
    """Lumen's own LED brightness right now, or None if unknown.

    The display loop keeps `_ctx.led_proprioception` current (server.py), but
    the SensorReadings used for state_history come from shared memory, where
    `led_brightness` is None — so it was persisted as NULL in all 255,720
    history rows.

    That matters because the VEML7700 sits beside the DotStars and lux is used
    RAW, with no glow correction (see anima.py). Lux therefore mixes room light
    with Lumen's own output, and lux feeds both clarity (weight 0.15) and the
    activity score (weight 0.15) — which in turn sets LED brightness. Without
    this field recorded alongside lux, that loop is not decomposable after the
    fact, and `SelfModel.observe_led_lux()` — which exists precisely to learn
    it — can never fire from history.
    """
    ctx = _cr._ctx
    if not ctx:
        return None
    prop = getattr(ctx, "led_proprioception", None)
    if isinstance(prop, dict):
        brightness = prop.get("brightness")
        if isinstance(brightness, (int, float)):
            return float(brightness)
    leds = getattr(ctx, "leds", None)
    if leds is not None:
        try:
            state = leds.get_proprioceptive_state()
            brightness = state.get("brightness") if isinstance(state, dict) else None
            if isinstance(brightness, (int, float)):
                return float(brightness)
        except Exception:
            return None
    return None


def _get_last_governance_decision() -> dict | None:
    """Last governance decision from broker SHM or server fallback."""
    return _cr._ctx.last_governance_decision if _cr._ctx else None


def _get_voice():
    """Get or initialize the voice instance (for listening capability)."""
    if _cr._ctx is None:
        return None
    if _cr._ctx.voice_instance is None:
        try:
            from .audio import AutonomousVoice

            voice = AutonomousVoice()

            # Autonomous voice canned phrases ("I'm curious about something", etc.)
            # are not posted to message board — primitives are more authentic.
            # Voice system still runs for listening capability.
            started = voice.start()
            # Retain the attempted instance even when no audio path is usable;
            # this prevents repeated dependency/device probes every ten loop
            # iterations and lets status report exact capabilities.
            _cr._ctx.voice_instance = voice
            if started:
                caps = voice.capabilities
                print(
                    f"[Server] Voice initialized (mode={VOICE_MODE}, "
                    f"hearing={'yes' if caps['hearing'] else 'no'}, "
                    f"speaking={'yes' if caps['speaking'] else 'no'})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[Server] Voice unavailable (mode={VOICE_MODE}, text remains active)",
                    file=sys.stderr,
                    flush=True,
                )
        except ImportError:
            print("[Server] Voice module not available (missing dependencies)", file=sys.stderr, flush=True)
            return None
        except Exception as e:
            print(f"[Server] Voice initialization failed: {e}", file=sys.stderr, flush=True)
            return None
    return _cr._ctx.voice_instance
