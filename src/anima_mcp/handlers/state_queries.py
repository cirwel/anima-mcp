"""State query handlers — read-only access to Lumen's current state.

Handlers: get_state, get_identity, read_sensors, get_health, get_calibration.
"""

import json

from mcp.types import TextContent

from ..server_state import extract_neural_bands
from ..config import ConfigManager
from ..error_recovery import note_suppressed
from ..light_attribution import gated_external_light_lux


async def handle_get_state(arguments: dict) -> list[TextContent]:
    """Get current state: anima (self-sense) + identity. Safe, never crashes."""
    # Late imports to avoid circular dependency (server.py imports us)
    from ..accessors import _get_store, _get_sensors, _get_readings_and_anima

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed",
            "suggestion": "Check server logs for initialization errors"
        }))]

    sensors = _get_sensors()

    # Read from shared memory (broker) or fallback to sensors
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Unable to read sensor data"
        }))]

    try:
        identity = store.get_identity()
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error reading identity: {e}"
        }))]

    # Clean sensor output - suppress nulls and group logically
    raw_sensors = readings.to_dict()
    sensors_clean = {
        "environment": {},
        "system": {},
        "neural": {},
    }
    # Environment sensors
    for k in ["ambient_temp_c", "humidity_pct", "light_lux", "pressure_hpa"]:
        if raw_sensors.get(k) is not None:
            sensors_clean["environment"][k] = raw_sensors[k]
    # System sensors
    for k in ["cpu_temp_c", "cpu_percent", "memory_percent", "disk_percent"]:
        if raw_sensors.get(k) is not None:
            sensors_clean["system"][k] = raw_sensors[k]
    # Neural (computational proprioception) - only the power bands
    sensors_clean["neural"] = extract_neural_bands(raw_sensors)

    result = {
        "anima": {
            "warmth": round(anima.warmth, 3),
            "clarity": round(anima.clarity, 3),
            "stability": round(anima.stability, 3),
            "presence": round(anima.presence, 3),
        },
        "mood": anima.feeling()["mood"],
        "feeling": anima.feeling(),
        "identity": {
            "name": identity.name,
            "id": identity.creature_id[:8] + "...",
            "awakenings": identity.total_awakenings,
            "age_seconds": round(identity.age_seconds()),
            "alive_seconds": round(identity.total_alive_seconds),
            "alive_ratio": round(identity.alive_ratio(), 3),
        },
        "sensors": sensors_clean,
        "is_pi": sensors.is_pi(),
    }

    # Add inner life from shared memory (temperament, drives)
    light_attribution = None
    try:
        from ..accessors import _get_last_shm_data
        shm = _get_last_shm_data()
        il = shm.get("inner_life") if shm else None
        if il:
            result["inner_life"] = {
                "temperament": il.get("temperament"),
                "drives": il.get("drives"),
                "strongest_drive": il.get("strongest_drive"),
            }
        light_attribution = shm.get("light_attribution") if shm else None
        if isinstance(light_attribution, dict):
            result["light_attribution"] = light_attribution
        clarity_attribution = (
            shm.get("clarity_attribution") if shm else None
        )
        if not isinstance(clarity_attribution, dict):
            clarity_attribution = getattr(
                anima, "clarity_attribution", None
            )
        if isinstance(clarity_attribution, dict):
            result["clarity_attribution"] = clarity_attribution
    except Exception as e:
        note_suppressed("state_queries.inner_life", e)

    # A read does not record: the main loop writes state_history on its own
    # schedule (state_history.py), so the record no longer depends on polling.
    external_light = gated_external_light_lux(light_attribution)
    if external_light is not None:
        sensors_clean["environment"]["external_light_lux"] = external_light

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_identity(arguments: dict) -> list[TextContent]:
    """Get full identity: birth, awakenings, name history. Safe, never crashes."""
    from ..accessors import _get_store

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed"
        }))]

    try:
        identity = store.get_identity()
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error reading identity: {e}"
        }))]

    result = {
        "id": identity.creature_id,
        "name": identity.name,
        "born_at": identity.born_at.isoformat(),
        "total_awakenings": identity.total_awakenings,
        "current_awakening_at": identity.current_awakening_at.isoformat() if identity.current_awakening_at else None,
        "total_alive_seconds": round(identity.total_alive_seconds),
        "age_seconds": round(identity.age_seconds()),
        "alive_ratio": round(identity.alive_ratio(), 3),
        "name_history": identity.name_history,
        "session_alive_seconds": round(store.get_session_alive_seconds()),
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_read_sensors(arguments: dict) -> list[TextContent]:
    """Read raw sensor values - returns only active sensors (nulls suppressed)."""
    from ..accessors import (
        _get_last_shm_data,
        _get_readings_and_anima,
        _get_sensors,
        _get_shm_client,
    )

    sensors = _get_sensors()

    # Read from shared memory (broker) or fallback to sensors
    readings, _ = _get_readings_and_anima()
    if readings is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Unable to read sensor data"
        }))]

    # Filter out null values for cleaner output
    raw = readings.to_dict()
    active_readings = {k: v for k, v in raw.items() if v is not None}

    result = {
        "timestamp": raw["timestamp"],
        "readings": active_readings,
        "available_sensors": sensors.available_sensors(),
        "is_pi": sensors.is_pi(),
        "source": "shared_memory" if (shm := _get_shm_client()) and shm.read() else "direct_sensors",
    }
    light_attribution = (_get_last_shm_data() or {}).get("light_attribution")
    if isinstance(light_attribution, dict):
        result["light_attribution"] = light_attribution

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_health(arguments: dict) -> list[TextContent]:
    """Get subsystem health status with heartbeat liveness and functional probes."""
    try:
        from ..health import get_health_registry
        registry = get_health_registry()
        result = {
            "overall": registry.overall(),
            "subsystems": registry.status(),
            "summary": registry.summary_line(),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def handle_get_calibration(arguments: dict) -> list[TextContent]:
    """Get current nervous system calibration."""
    config_manager = ConfigManager()
    # Force reload to get latest from disk
    config = config_manager.reload()
    calibration = config.nervous_system
    metadata = config.metadata

    result = {
        "calibration": calibration.to_dict(),
        "config_file": str(config_manager.config_path),
        "config_exists": config_manager.config_path.exists(),
        "metadata": {
            "last_updated": metadata.get("calibration_last_updated"),
            "last_updated_by": metadata.get("calibration_last_updated_by"),
            "update_count": metadata.get("calibration_update_count", 0),
            "recent_changes": metadata.get("calibration_history", [])[-5:],  # Last 5 changes
        },
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
