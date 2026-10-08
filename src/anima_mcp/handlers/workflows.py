"""Workflow handlers — orchestration, next steps, calibration, context, visualization.

Handlers: unified_workflow, next_steps, set_calibration, get_lumen_context, learning_visualization.
"""

import json
import sys
from typing import Any

from mcp.types import TextContent

from ..error_recovery import note_suppressed


async def handle_unified_workflow(arguments: dict) -> list[TextContent]:
    """Execute unified workflows across anima-mcp and unitares-governance. Safe, never crashes.

    Supports both original workflows and workflow templates.
    If workflow name matches a template, uses template. Otherwise uses original workflow logic.
    """
    import os
    from ..accessors import _get_store
    from ..workflow_orchestrator import get_orchestrator
    from ..workflow_templates import WorkflowTemplates

    store = _get_store()
    if store is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Server not initialized - wake() failed"}),
            )
        ]

    unitares_url = os.environ.get("UNITARES_URL")

    orchestrator = get_orchestrator(
        unitares_url=unitares_url,
        anima_store=store,
        # The MCP process never becomes a direct sensor owner.  The
        # orchestrator may accept a backend in standalone use, but production
        # workflows fail toward unknown when broker SHM is stale or absent.
        anima_sensors=None,
    )

    workflow = arguments.get("workflow")

    # If no workflow specified, return available options
    if not workflow:
        templates = WorkflowTemplates(orchestrator)
        template_list = templates.list_templates()
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "available_workflows": [
                            "check_state_and_governance",
                            "monitor_and_govern",
                        ],
                        "available_templates": [t["name"] for t in template_list],
                        "usage": "Call with workflow=<name> to execute",
                    },
                    indent=2,
                ),
            )
        ]

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
        result["note"] = (
            f"Single check performed. Use interval={interval}s for continuous monitoring."
        )
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
    from ..eisv_mapper import (
        BODY_EISV_PROJECTION_SCHEMA,
        anima_to_body_eisv_projection,
    )

    store = _get_store()
    if store is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Server not initialized - wake() failed"}),
            )
        ]

    display = _get_display()

    # Read from shared memory (broker) or fallback to sensors
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [
            TextContent(
                type="text", text=json.dumps({"error": "Unable to read sensor data"})
            )
        ]

    body_projection = anima_to_body_eisv_projection(anima, readings)

    # Check availability
    display_available = display.is_available()
    # BrainCraft HAT hardware (display + LEDs + sensors) is available if display is available
    # Note: No physical EEG hardware exists - neural signals come from computational proprioception
    brain_hat_hardware_available = (
        display_available  # BrainCraft HAT = display hardware (not EEG)
    )
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
                print(
                    "[Diagnostics] UNITARES connected via shared bridge",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "[Diagnostics] UNITARES URL set but unavailable",
                    file=sys.stderr,
                    flush=True,
                )
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
    try:
        from ..self_iteration import get_self_iteration_system

        self_iteration_attention = get_self_iteration_system().attention(limit=20)
    except Exception as e:
        self_iteration_attention = {
            "schema": "anima.self_iteration.attention.v1",
            "error": str(e),
            "items": [
                {
                    "attention_id": "si-attn-9e0a644a451afd525f5a0e2a",
                    "proposal_id": "self-iteration-ledger",
                    "candidate_id": None,
                    "stage": "attention",
                    "state": "attention_projection_failed",
                    "priority": "critical",
                    "active": True,
                    "summary": "The self-iteration attention projection failed.",
                    "next_action": "An operator must inspect the ledger and artifacts.",
                    "required_role": "operator_recovery",
                    "reference_id": "self-iteration-ledger",
                    "occurred_at": None,
                    "target_paths": [],
                    "claim_provenance": {
                        "source_epistemic_status": "unknown",
                        "request_trust_classification": "unknown",
                        "request_actor_authenticated": False,
                        "claims_verified_by_request_provenance": False,
                        "independent_verification_status": "unknown",
                        "effective_weight": 0.0,
                        "authority_granted": False,
                    },
                    "status_query": {
                        "tool": "self_iteration",
                        "arguments": {"action": "attention", "limit": 20},
                        "read_only": True,
                    },
                    "signed_approval_required": False,
                    "acknowledgement_is_approval": False,
                    "authority_granted": False,
                }
            ],
            "acknowledgement_is_approval": False,
            "authority_granted": False,
        }
    # Lumen's own intentions. Read-only on purpose: growth._goals is the
    # active-only in-memory mirror loaded at startup and _curiosities holds the
    # unexplored ones, so both are plain reads. check_goal_progress() would be
    # the tempting call here and is the wrong one — it auto-abandons stale
    # goals as a side effect, and a status tool must not change the status it
    # reports.
    goals: list = []
    curiosities: list = []
    intentions_available = False
    intentions_error = None
    try:
        from ..accessors import _get_growth

        growth = _get_growth()
        if growth is not None:
            goals = [
                goal.to_dict()
                for goal in getattr(growth, "_goals", {}).values()
                if getattr(goal.status, "value", goal.status) == "active"
            ]
            curiosities = list(getattr(growth, "_curiosities", []) or [])
            intentions_available = True
        else:
            intentions_error = "growth store not initialized"
    except Exception as e:
        intentions_error = str(e)
        print(f"[Diagnostics] Intentions unavailable: {e}", file=sys.stderr, flush=True)

    advocate = get_advocate()
    advocate.analyze_current_state(
        anima=anima,
        readings=readings,
        eisv=body_projection,
        display_available=display_available,
        brain_hat_available=brain_hat_hardware_available,
        unitares_connected=unitares_connected,
        drives=_il.get("drives"),
        strongest_drive=_il.get("strongest_drive"),
        wants=_il.get("wants"),
        self_iteration_attention=self_iteration_attention,
        goals=goals,
        curiosities=curiosities,
    )

    summary = advocate.get_next_steps_summary()

    # Extract next action details for easier access
    next_action = summary.get("next_action", {})

    result = {
        "summary": {
            # status is the field to read first: "unknown" means no analysis is
            # on record, and it is the one case where the four "none"s below do
            # not mean "nothing to report".
            "status": summary.get("status", "unknown"),
            "last_analyzed": summary.get("last_analyzed"),
            "state_checks": summary.get("state_checks", {}),
            # Without this, a growth store that failed to load would report
            # zero goals and read as "Lumen has nothing planned" — the same
            # silent-toward-healthy failure `status` exists to prevent.
            "intentions": {
                "available": intentions_available,
                "goals": len(goals),
                "curiosities": len(curiosities),
                "error": intentions_error,
            },
            "priority": next_action.get("priority", "unknown")
            if next_action
            else "none",
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
            "body_anima": {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            },
            "anima": {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            },
            "body_eisv_projection": body_projection.to_dict(),
            "eisv": body_projection.to_dict(),
            "eisv_source": "body_eisv_projection_legacy_alias",
            "state_space_provenance": {
                "body_anima": {
                    "source": "broker_published_anima",
                    "role": "physical_self_sense",
                },
                "anima": {
                    "alias_of": "body_anima",
                    "deprecated": True,
                },
                "body_eisv_projection": {
                    "schema": BODY_EISV_PROJECTION_SCHEMA,
                    "source": "anima_sensor_projection",
                    "role": "body_measurement",
                },
                "eisv": {
                    "alias_of": "body_eisv_projection",
                    "deprecated": True,
                },
            },
        },
        "self_iteration_attention": self_iteration_attention,
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
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "updates parameter required",
                        "example": {
                            "updates": {
                                "ambient_temp_min": 10.0,
                                "ambient_temp_max": 30.0,
                                "pressure_ideal": 833.0,
                            }
                        },
                    }
                ),
            )
        ]

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
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Invalid calibration: {error}",
                            "current": calibration.to_dict(),
                        }
                    ),
                )
            ]

        # Update config
        config = config_manager.load()
        config.nervous_system = updated_cal

        if config_manager.save(config, update_source=update_source):
            # Force reload to get updated metadata
            updated_config = config_manager.reload()
            metadata = updated_config.metadata

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": True,
                            "message": "Calibration updated",
                            "calibration": updated_cal.to_dict(),
                            "metadata": {
                                "last_updated": metadata.get(
                                    "calibration_last_updated"
                                ),
                                "last_updated_by": metadata.get(
                                    "calibration_last_updated_by"
                                ),
                                "update_count": metadata.get(
                                    "calibration_update_count", 0
                                ),
                            },
                        }
                    ),
                )
            ]
        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "Failed to save calibration",
                        }
                    ),
                )
            ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Error updating calibration: {e}",
                    }
                ),
            )
        ]


async def handle_get_lumen_context(arguments: dict) -> list[TextContent]:
    """
    Get Lumen's complete current context in one call.
    Consolidates: get_state + get_identity + read_sensors
    """
    from ..accessors import (
        _get_last_shm_data,
        _get_readings_and_anima,
        _get_sensors,
        _get_store,
    )

    store = _get_store()
    sensors = _get_sensors()

    include = arguments.get(
        "include", ["identity", "anima", "sensors", "mood", "attention"]
    )
    if isinstance(include, str):
        include = [include]

    result: dict[str, Any] = {}

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
                    "alive_seconds": round(identity.total_alive_seconds),
                    "alive_ratio": round(identity.alive_ratio(), 3),
                }
            except Exception as e:
                result["identity"] = {"error": str(e)}

    if "anima" in include or "body_anima" in include:
        if anima:
            body_anima = {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            }
            anima_clarity_attribution = getattr(
                anima, "clarity_attribution", None
            )
            if isinstance(anima_clarity_attribution, dict):
                body_anima["clarity_attribution"] = anima_clarity_attribution
            result["body_anima"] = body_anima
            if "anima" in include:
                result["anima"] = body_anima
        else:
            result["body_anima"] = {"error": "Unable to read anima state"}
            if "anima" in include:
                result["anima"] = result["body_anima"]

    if "sensors" in include:
        if readings:
            result["sensors"] = readings.to_dict()
            result["sensors"]["is_pi"] = sensors.is_pi()
        else:
            result["sensors"] = {"error": "Unable to read sensor data"}

    if (
        "sensors" in include
        or "light_attribution" in include
        or "clarity_attribution" in include
    ):
        shm_data = _get_last_shm_data() or {}
        light_attribution = shm_data.get("light_attribution")
        if isinstance(light_attribution, dict):
            result["light_attribution"] = light_attribution
        elif "light_attribution" in include:
            result["light_attribution"] = {
                "status": "unavailable",
                "reason": "broker_has_not_published_light_attribution",
            }
        clarity_attribution = shm_data.get("clarity_attribution")
        if not isinstance(clarity_attribution, dict) and anima is not None:
            clarity_attribution = getattr(anima, "clarity_attribution", None)
        if isinstance(clarity_attribution, dict):
            result["clarity_attribution"] = clarity_attribution
        elif "clarity_attribution" in include:
            result["clarity_attribution"] = {
                "status": "unavailable",
                "reason": "broker_has_not_published_clarity_attribution",
            }

    if "mood" in include:
        if anima:
            result["mood"] = anima.feeling()
        else:
            result["mood"] = {"error": "Unable to determine mood"}

    if "code" in include:
        try:
            from ..self_iteration import get_self_iteration_system

            overview = get_self_iteration_system().inspect()
            boundaries = overview["boundaries"]
            result["code"] = {
                "mode": overview["mode"],
                "autonomy_level": overview["autonomy_level"],
                "runtime": overview["runtime"],
                "source": overview["source"],
                "capabilities": overview["capabilities"],
                "boundary_summary": {
                    "protected_surface_count": len(boundaries["protected_surfaces"]),
                    "auto_eligible_surface_count": len(
                        boundaries["initial_auto_eligible_surfaces"]
                    ),
                    "implementation_rule": boundaries["implementation_rule"],
                },
                "ledger": overview["ledger"],
            }
        except Exception as e:
            result["code"] = {"error": str(e)}

    if "attention" in include:
        try:
            from ..self_iteration import get_self_iteration_system

            result["self_iteration_attention"] = get_self_iteration_system().attention(
                limit=20
            )
        except Exception as e:
            result["self_iteration_attention"] = {"error": str(e)}

    # Include the body projection when anima is available. Bare ``eisv`` is a
    # compatibility alias, never a claim about UNITARES's inferred state.
    if (
        {"eisv", "body_eisv_projection", "anima", "body_anima"}
        .intersection(include)
        and anima
        and readings
    ):
        try:
            from ..eisv_mapper import (
                BODY_EISV_PROJECTION_SCHEMA,
                anima_to_body_eisv_projection,
            )

            body_projection = anima_to_body_eisv_projection(anima, readings)
            result["body_eisv_projection"] = body_projection.to_dict()
            if "eisv" in include or "anima" in include:
                result["eisv"] = body_projection.to_dict()
                result["eisv_source"] = "body_eisv_projection_legacy_alias"
            result["state_space_provenance"] = {
                "body_anima": {
                    "source": "broker_published_anima",
                    "role": "physical_self_sense",
                },
                "anima": {
                    "alias_of": "body_anima",
                    "deprecated": True,
                },
                "body_eisv_projection": {
                    "schema": BODY_EISV_PROJECTION_SCHEMA,
                    "source": "anima_sensor_projection",
                    "role": "body_measurement",
                },
                "eisv": {
                    "alias_of": "body_eisv_projection",
                    "deprecated": True,
                },
            }
        except Exception as e:
            note_suppressed("workflows.eisv", e)  # optional enrichment

    # A read does not record: the main loop writes state_history (state_history.py).

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_learning_visualization(arguments: dict) -> list[TextContent]:
    """Get learning visualization - shows why Lumen feels what it feels."""
    from ..accessors import _get_store, _get_readings_and_anima
    from ..learning_visualization import LearningVisualizer

    store = _get_store()
    if store is None:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Server not initialized - wake() failed"}),
            )
        ]

    # Get current state
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [
            TextContent(
                type="text", text=json.dumps({"error": "Unable to read sensor data"})
            )
        ]

    # Create visualizer
    visualizer = LearningVisualizer(db_path=str(store.db_path))

    # Get comprehensive learning summary
    summary = visualizer.get_learning_summary(readings=readings, anima=anima)

    return [TextContent(type="text", text=json.dumps(summary, indent=2))]
