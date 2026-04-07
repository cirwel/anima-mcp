"""Workflow handlers — orchestration, next steps, calibration, context, visualization.

Handlers: unified_workflow, next_steps, set_calibration, get_lumen_context, learning_visualization.
"""

import json
import sys

from mcp.types import TextContent


async def handle_unified_workflow(arguments: dict) -> list[TextContent]:
    """Execute unified workflows across anima-mcp and unitares-governance. Safe, never crashes.

    Supports both original workflows and workflow templates.
    If workflow name matches a template, uses template. Otherwise uses original workflow logic.
    """
    import os
    from ..accessors import _get_store, _get_sensors
    from ..workflow_orchestrator import get_orchestrator
    from ..workflow_templates import WorkflowTemplates

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed"
        }))]

    sensors = _get_sensors()
    unitares_url = os.environ.get("UNITARES_URL")

    orchestrator = get_orchestrator(
        unitares_url=unitares_url,
        anima_store=store,
        anima_sensors=sensors
    )

    workflow = arguments.get("workflow")

    # If no workflow specified, return available options
    if not workflow:
        templates = WorkflowTemplates(orchestrator)
        template_list = templates.list_templates()
        return [TextContent(type="text", text=json.dumps({
            "available_workflows": ["check_state_and_governance", "monitor_and_govern"],
            "available_templates": [t["name"] for t in template_list],
            "usage": "Call with workflow=<name> to execute"
        }, indent=2))]

    interval = arguments.get("interval", 60.0)

    # Check if it's a template first
    templates = WorkflowTemplates(orchestrator)
    template = templates.get_template(workflow)

    if template:
        # It's a template - run it
        result_obj = await templates.run(workflow)
        result = {
            "status": result_obj.status.value,
            "summary": result_obj.summary,
            "steps": result_obj.steps,
            "errors": result_obj.errors,
            "template": workflow,
        }
    elif workflow == "check_state_and_governance":
        # Original workflow
        result = await orchestrator.workflow_check_state_and_governance()
    elif workflow == "monitor_and_govern":
        # Original workflow
        result = await orchestrator.workflow_check_state_and_governance()
        result["note"] = f"Single check performed. Use interval={interval}s for continuous monitoring."
    else:
        # Unknown - suggest alternatives
        template_list = templates.list_templates()
        result = {
            "error": f"Unknown workflow: {workflow}",
            "available_workflows": ["check_state_and_governance", "monitor_and_govern"],
            "available_templates": [t["name"] for t in template_list],
        }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_next_steps(arguments: dict) -> list[TextContent]:
    """Get proactive next steps to achieve goals. Safe, never crashes."""
    from ..accessors import _get_store, _get_display, _get_readings_and_anima
    from ..next_steps_advocate import get_advocate
    from ..eisv_mapper import anima_to_eisv

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed"
        }))]

    display = _get_display()

    # Read from shared memory (broker) or fallback to sensors
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Unable to read sensor data"
        }))]

    eisv = anima_to_eisv(anima, readings)

    # Check availability
    display_available = display.is_available()
    # BrainCraft HAT hardware (display + LEDs + sensors) is available if display is available
    # Note: No physical EEG hardware exists - neural signals come from computational proprioception
    brain_hat_hardware_available = display_available  # BrainCraft HAT = display hardware (not EEG)
    # Check UNITARES (use shared server bridge)
    unitares_connected = False
    unitares_status = "not_configured"
    try:
        from ..accessors import _get_server_bridge
        bridge = _get_server_bridge()
        if bridge is not None:
            unitares_connected = await bridge.check_availability()
            unitares_status = "connected" if unitares_connected else "unavailable"
            if unitares_connected:
                print("[Diagnostics] UNITARES connected via shared bridge", file=sys.stderr, flush=True)
            else:
                print("[Diagnostics] UNITARES URL set but unavailable", file=sys.stderr, flush=True)
        else:
            unitares_status = "not_configured"
            print("[Diagnostics] UNITARES_URL not set", file=sys.stderr, flush=True)
    except Exception as e:
        unitares_status = f"error: {str(e)}"
        print(f"[Diagnostics] UNITARES check failed: {e}", file=sys.stderr, flush=True)

    # Get advocate recommendations (with actual drives from SHM)
    from ..accessors import _get_last_shm_data
    _shm = _get_last_shm_data()
    _il = (_shm.get("inner_life") or {}) if _shm else {}
    advocate = get_advocate()
    advocate.analyze_current_state(
        anima=anima,
        readings=readings,
        eisv=eisv,
        display_available=display_available,
        brain_hat_available=brain_hat_hardware_available,
        unitares_connected=unitares_connected,
        drives=_il.get("drives"),
        strongest_drive=_il.get("strongest_drive"),
    )

    summary = advocate.get_next_steps_summary()

    # Extract next action details for easier access
    next_action = summary.get("next_action", {})

    result = {
        "summary": {
            "priority": next_action.get("priority", "unknown") if next_action else "none",
            "feeling": next_action.get("feeling", "unknown") if next_action else "none",
            "desire": next_action.get("desire", "unknown") if next_action else "none",
            "action": next_action.get("action", "unknown") if next_action else "none",
            "total_steps": summary.get("total_steps", 0),
            "critical": summary.get("critical", 0),
            "high": summary.get("high", 0),
            "medium": summary.get("medium", 0),
            "low": summary.get("low", 0),
            "all_steps": summary.get("all_steps", []),
        },
        "current_state": {
            "display_available": display_available,
            "brain_hat_hardware_available": brain_hat_hardware_available,
            "unitares_connected": unitares_connected,
            "unitares_status": unitares_status,
            "anima": {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            },
            "eisv": eisv.to_dict(),
        },
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_set_calibration(arguments: dict) -> list[TextContent]:
    """Update nervous system calibration (partial updates supported)."""
    from ..config import get_calibration, ConfigManager, NervousSystemCalibration

    calibration = get_calibration()
    config_manager = ConfigManager()

    # Allow partial updates
    updates = arguments.get("updates", {})
    if not updates:
        return [TextContent(type="text", text=json.dumps({
            "error": "updates parameter required",
            "example": {
                "updates": {
                    "ambient_temp_min": 10.0,
                    "ambient_temp_max": 30.0,
                    "pressure_ideal": 833.0
                }
            }
        }))]

    # Track who/what is updating (for metadata)
    update_source = arguments.get("source", "agent")  # "agent", "manual", "automatic"

    # Update calibration values
    cal_dict = calibration.to_dict()
    cal_dict.update(updates)

    try:
        updated_cal = NervousSystemCalibration.from_dict(cal_dict)

        # Validate
        valid, error = updated_cal.validate()
        if not valid:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Invalid calibration: {error}",
                "current": calibration.to_dict(),
            }))]

        # Update config
        config = config_manager.load()
        config.nervous_system = updated_cal

        if config_manager.save(config, update_source=update_source):
            # Force reload to get updated metadata
            updated_config = config_manager.reload()
            metadata = updated_config.metadata

            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "message": "Calibration updated",
                "calibration": updated_cal.to_dict(),
                "metadata": {
                    "last_updated": metadata.get("calibration_last_updated"),
                    "last_updated_by": metadata.get("calibration_last_updated_by"),
                    "update_count": metadata.get("calibration_update_count", 0),
                },
            }))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": "Failed to save calibration",
            }))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error updating calibration: {e}",
        }))]


async def handle_get_lumen_context(arguments: dict) -> list[TextContent]:
    """
    Get Lumen's complete current context in one call.
    Consolidates: get_state + get_identity + read_sensors
    """
    from ..accessors import _get_store, _get_sensors, _get_readings_and_anima

    store = _get_store()
    sensors = _get_sensors()

    include = arguments.get("include", ["identity", "anima", "sensors", "mood"])
    if isinstance(include, str):
        include = [include]

    result = {}

    # Always need readings/anima for most queries
    readings, anima = _get_readings_and_anima()

    if "identity" in include:
        if store is None:
            result["identity"] = {"error": "Store not initialized"}
        else:
            try:
                identity = store.get_identity()
                result["identity"] = {
                    "name": identity.name,
                    "id": identity.creature_id,
                    "born_at": identity.born_at.isoformat(),
                    "awakenings": identity.total_awakenings,
                    "age_seconds": round(identity.age_seconds()),
                    "alive_seconds": round(identity.total_alive_seconds + store.get_session_alive_seconds()),
                    "alive_ratio": round(identity.alive_ratio(), 3),
                }
            except Exception as e:
                result["identity"] = {"error": str(e)}

    if "anima" in include:
        if anima:
            result["anima"] = {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            }
        else:
            result["anima"] = {"error": "Unable to read anima state"}

    if "sensors" in include:
        if readings:
            result["sensors"] = readings.to_dict()
            result["sensors"]["is_pi"] = sensors.is_pi()
        else:
            result["sensors"] = {"error": "Unable to read sensor data"}

    if "mood" in include:
        if anima:
            result["mood"] = anima.feeling()
        else:
            result["mood"] = {"error": "Unable to determine mood"}

    # Include EISV metrics when anima is available
    if ("eisv" in include or "anima" in include) and anima and readings:
        try:
            from ..eisv_mapper import anima_to_eisv
            eisv = anima_to_eisv(anima, readings)
            result["eisv"] = eisv.to_dict()
        except Exception:
            pass  # EISV is optional enrichment

    # Record state for history if we have it (enriched with interaction context)
    if store and anima and readings:
        sensors_for_history = readings.to_dict()
        try:
            from ..messages import get_recent_messages
            from datetime import datetime
            recent = get_recent_messages(limit=10)
            now = datetime.now()
            human = [m for m in recent if getattr(m, 'msg_type', '') == 'user']
            if human:
                last_ts = max(m.timestamp for m in human)
                minutes_ago = (now.timestamp() - last_ts) / 60
                sensors_for_history["interaction_level"] = max(0.0, 1.0 - minutes_ago / 30.0)
            else:
                sensors_for_history["interaction_level"] = 0.0
        except Exception:
            pass
        store.record_state(
            anima.warmth, anima.clarity, anima.stability, anima.presence,
            sensors_for_history
        )

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_learning_visualization(arguments: dict) -> list[TextContent]:
    """Get learning visualization - shows why Lumen feels what it feels."""
    from ..accessors import _get_store, _get_readings_and_anima
    from ..learning_visualization import LearningVisualizer

    store = _get_store()
    if store is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Server not initialized - wake() failed"
        }))]

    # Get current state
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Unable to read sensor data"
        }))]

    # Create visualizer
    visualizer = LearningVisualizer(db_path=str(store.db_path))

    # Get comprehensive learning summary
    summary = visualizer.get_learning_summary(readings=readings, anima=anima)

    return [TextContent(type="text", text=json.dumps(summary, indent=2))]
