"""Display operation handlers — screen capture, face rendering, diagnostics, display management.

Handlers: capture_screen, show_face, diagnostics, manage_display.
"""

import json
from typing import Any

from mcp.types import TextContent, ImageContent


VALID_DISPLAY_ACTIONS = [
    "switch",
    "face",
    "next",
    "previous",
    "list_eras",
    "get_era",
    "set_era",
    "resonance_critique",
    "calibrate_leds",
]


def _build_resonance_critique_packet(renderer: Any) -> dict:
    """Build an advisory packet for a human/model Resonance-era critique loop.

    This helper intentionally does not capture pixels, call an LLM, switch eras,
    toggle auto-rotate, or clear the canvas. It only returns the exact tool path
    and recommendation contract that an external critic should follow.
    """
    era_info = renderer.get_current_era()
    try:
        current_screen = renderer.get_mode().value
    except Exception:
        current_screen = None

    try:
        drawing_state = renderer.get_drawing_eisv()
    except Exception:
        drawing_state = None

    return {
        "success": True,
        "action": "resonance_critique",
        "mode": "advisory",
        "manual_control_preserved": True,
        "current_era": era_info.get("current_era"),
        "current_description": era_info.get("current_description"),
        "auto_rotate": era_info.get("auto_rotate"),
        "current_screen": current_screen,
        "drawing_state": drawing_state,
        "available_eras": era_info.get("all_eras", []),
        "loop": [
            {
                "step": "capture",
                "tool": "capture_screen",
                "purpose": "Get the exact 240x240 LCD image before interpreting it.",
            },
            {
                "step": "embodied_context",
                "tool": "get_lumen_context",
                "arguments": {"include": ["state", "mood", "sensors", "identity"]},
                "purpose": "Ground the reading in current mood, room weather, and identity context.",
            },
            {
                "step": "era_context",
                "tool": "manage_display",
                "arguments": {"action": "get_era"},
                "purpose": "Confirm active era and auto-rotate status before recommending changes.",
            },
            {
                "step": "recommend",
                "tool": "external_visual_reader",
                "purpose": "Return only stay/tune/switch advice; do not mutate Lumen directly.",
            },
        ],
        "recommendation_contract": {
            "allowed_recommendations": ["stay", "tune", "switch"],
            "forbidden_side_effects": [
                "do_not_set_era",
                "do_not_toggle_auto_rotate",
                "do_not_clear_canvas",
                "do_not_infer_fixed_intent_from_marks",
            ],
            "switch_requires": "A human or operator must explicitly call manage_display(action='set_era', screen='<era>').",
            "tune_means": "Suggest palette/mark-density/context emphasis for a future change; do not change runtime state.",
        },
        "resonance_focus": [
            "sediment",
            "flow",
            "scratch",
            "memory field",
            "biological ornament",
            "intake/residue",
            "history deforming present expression",
        ],
        "visual_reading_cues": [
            "line density",
            "negative space",
            "branching or vascular structure",
            "directionality and flow",
            "scar/scratch texture",
            "whether marks feel accumulated or decorative",
        ],
        "critic_prompt": (
            "Read Lumen's current screen as a grounded visual trace. Mention the active era, "
            "visible shapes, palette, density, and room/mood context. Then recommend exactly one "
            "of stay, tune, or switch. Preserve manual era control; do not claim the drawing has a "
            "fixed intent."
        ),
    }


async def handle_capture_screen(arguments: dict) -> list[TextContent | ImageContent]:
    """
    Capture current display screen as a viewable PNG image.

    Returns the actual visual output on Lumen's 240×240 LCD display,
    allowing remote viewing of what Lumen is drawing, showing, or expressing.
    """
    from ..accessors import _get_screen_renderer

    renderer = _get_screen_renderer()
    if renderer is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Screen renderer not initialized"
        }))]

    try:
        # Access the renderer's display object to get the current image
        renderer_display = renderer._display
        if renderer_display is None or not hasattr(renderer_display, '_image'):
            return [TextContent(type="text", text=json.dumps({
                "error": "Display not available or no image cached"
            }))]

        # Get the current image from the PIL renderer
        current_image = renderer_display._image
        if current_image is None:
            return [TextContent(type="text", text=json.dumps({
                "error": "No image currently displayed"
            }))]

        # Convert PIL Image to base64-encoded PNG
        import base64
        from io import BytesIO

        buffer = BytesIO()
        current_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Get current screen/era context
        screen_mode = renderer.get_mode().value
        era_name = None
        if screen_mode == "art_eras" and hasattr(renderer, '_active_era') and renderer._active_era:
            era_name = renderer._active_era.name

        # Return image as ImageContent (viewable by agents) + metadata as TextContent
        return [
            ImageContent(type="image", data=img_base64, mimeType="image/png"),
            TextContent(type="text", text=json.dumps({
                "success": True,
                "width": current_image.width,
                "height": current_image.height,
                "screen": screen_mode,
                "era": era_name,
            }))
        ]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to capture screen: {str(e)}",
            "traceback": traceback.format_exc()
        }))]


async def handle_show_face(arguments: dict) -> list[TextContent]:
    """Show face on display (or return ASCII art if no display). Safe, never crashes."""
    from ..accessors import _get_store, _get_display, _get_readings_and_anima
    from ..display import derive_face_state, face_to_ascii

    store = _get_store()
    display = _get_display()

    # Read from shared memory (broker) or fallback to sensors
    readings, anima = _get_readings_and_anima()
    if readings is None or anima is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Unable to read sensor data"
        }))]

    if store is None:
        identity_name = None
        identity = None
    else:
        try:
            identity = store.get_identity()
            identity_name = identity.name if identity else None
        except Exception:
            identity_name = None
            identity = None
    face_state = derive_face_state(anima)

    # Try to render on hardware display
    if display.is_available():
        display.render_face(face_state, name=identity_name)
        result = {
            "rendered": True,
            "display": "hardware",
            "eyes": face_state.eyes.value,
            "mouth": face_state.mouth.value,
            "mood": anima.feeling()["mood"],
        }
    else:
        # Return ASCII art
        ascii_face = face_to_ascii(face_state)
        result = {
            "rendered": False,
            "display": "ascii",
            "face": ascii_face,
            "eyes": face_state.eyes.value,
            "mouth": face_state.mouth.value,
            "mood": anima.feeling()["mood"],
        }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_diagnostics(arguments: dict) -> list[TextContent]:
    """Get system diagnostics including LED and display status."""
    from ..accessors import _get_leds, _get_display, _get_display_update_task, _get_sensors

    sensors = _get_sensors()

    # LED diagnostics
    led_info = {}
    leds = _get_leds()
    display = _get_display()
    display_task = _get_display_update_task()
    if leds:
        led_info = leds.get_diagnostics()
    else:
        led_info = {"available": False, "reason": "not initialized"}

    # Display diagnostics
    display_info = {
        "available": display.is_available() if display else False,
        "initialized": display is not None,
    }
    if display and hasattr(display, '_init_error') and display._init_error:
            display_info["init_error"] = display._init_error

    # Update loop status
    loop_info = {
        "task_exists": display_task is not None,
        "task_done": display_task.done() if display_task else None,
        "task_cancelled": display_task.cancelled() if display_task else None,
    }

    # Drawing diagnostics
    drawing_info = None
    try:
        from ..accessors import _get_screen_renderer
        import time as _time
        renderer = _get_screen_renderer()
        if renderer and hasattr(renderer, 'drawing_engine'):
            engine = renderer.drawing_engine
            drawing_info = engine.get_drawing_eisv()
            if drawing_info:
                drawing_info["pixel_count"] = len(engine.canvas.pixels)
                drawing_info["drawing_age_s"] = round(
                    _time.time() - engine.canvas.last_clear_time, 1
                ) if engine.canvas.last_clear_time > 0 else None
    except Exception:
        pass

    result = {
        "leds": led_info,
        "display": display_info,
        "update_loop": loop_info,
        "sensors": {
            "is_pi": sensors.is_pi(),
            "available": sensors.available_sensors(),
        },
    }
    if drawing_info:
        result["drawing"] = drawing_info

    # Governance / UNITARES reachability — surfaces silent local-fallback.
    # get_health() reports governance=ok whenever decisions are produced, even
    # local-only ones, so a prolonged UNITARES outage is otherwise invisible here.
    try:
        from ..ctx_ref import get_ctx
        import time as _gt
        _c = get_ctx()
        if _c is not None:
            last_dec = _c.last_governance_decision or {}
            last_ok = _c.last_unitares_success_time or 0.0
            age = round(_gt.time() - last_ok, 1) if last_ok > 0 else None
            result["governance"] = {
                "last_decision_source": last_dec.get("source"),
                "unitares_last_success_age_s": age,
                # No successful UNITARES check-in within 5 min → running on local fallback
                "unitares_stale": (age is None) or (age > 300),
            }
            # Self-model liveness — observation_count stuck at 0 while the loop
            # runs means cross-iteration learning has silently broken (ab984f9).
            sm_last = _c.sm_last_observation_time or 0.0
            result["self_model"] = {
                "observation_count": _c.sm_observation_count,
                "last_observation_age_s": round(_gt.time() - sm_last, 1) if sm_last > 0 else None,
            }
    except Exception:
        pass

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_manage_display(arguments: dict) -> list[TextContent]:
    """
    Control Lumen's display.
    Consolidates: switch_screen + show_face
    """
    from ..accessors import _get_screen_renderer
    from ..display.screens import ScreenMode

    renderer = _get_screen_renderer()
    action = arguments.get("action")
    if not action:
        return [TextContent(type="text", text=json.dumps({
            "error": "action parameter required (switch, face, next, previous)"
        }))]

    if action == "face":
        # Delegate to show_face handler
        return await handle_show_face({})

    if not renderer:
        return [TextContent(type="text", text=json.dumps({
            "error": "Screen renderer not initialized"
        }))]

    if action == "switch":
        screen = arguments.get("screen", "").lower()
        mode_map = {
            "face": ScreenMode.FACE,
            "sensors": ScreenMode.SENSORS,
            "identity": ScreenMode.IDENTITY,
            "diagnostics": ScreenMode.DIAGNOSTICS,
            "neural": ScreenMode.NEURAL,
            "notepad": ScreenMode.NOTEPAD,
            "learning": ScreenMode.LEARNING,
            "self_graph": ScreenMode.SELF_GRAPH,
            "messages": ScreenMode.MESSAGES,
            "questions": ScreenMode.QUESTIONS,
            "visitors": ScreenMode.VISITORS,
            "art_eras": ScreenMode.ART_ERAS,
            "health": ScreenMode.HEALTH,
            "goals_beliefs": ScreenMode.GOALS_BELIEFS,
            "agency": ScreenMode.AGENCY,
        }
        if screen in mode_map:
            renderer.set_mode(mode_map[screen])
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "action": "switch",
                "screen": screen
            }))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Invalid screen: {screen}",
                "valid_screens": list(mode_map.keys())
            }))]

    elif action == "next":
        renderer.next_mode()
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "action": "next",
            "screen": renderer.get_mode().value
        }))]

    elif action == "previous":
        renderer.previous_mode()
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "action": "previous",
            "screen": renderer.get_mode().value
        }))]

    elif action == "list_eras":
        info = renderer.get_current_era()
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "action": "list_eras",
            **info,
        }))]

    elif action == "get_era":
        info = renderer.get_current_era()
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "action": "get_era",
            "current_era": info["current_era"],
            "current_description": info["current_description"],
            "auto_rotate": info["auto_rotate"],
        }))]

    elif action == "set_era":
        era_name = arguments.get("screen", "").lower()
        if not era_name:
            return [TextContent(type="text", text=json.dumps({
                "error": "screen parameter required — set it to the era name (e.g. 'geometric', 'gestural')"
            }))]
        result = renderer.set_era(era_name)
        return [TextContent(type="text", text=json.dumps({
            "action": "set_era",
            **result,
        }))]

    elif action == "resonance_critique":
        return [TextContent(
            type="text",
            text=json.dumps(_build_resonance_critique_packet(renderer), indent=2),
        )]

    elif action == "calibrate_leds":
        import asyncio
        from ..accessors import _get_leds, _get_sensors

        leds = _get_leds()
        if not leds or not leds.is_available():
            return [TextContent(type="text", text=json.dumps({
                "error": "LEDs not available"
            }))]

        sensors = _get_sensors()
        BRIGHTNESS_LEVELS = [0.0, 0.12, 0.25]
        SETTLE_SECONDS = 1.0
        SAMPLES_PER_LEVEL = 1

        original_factor = leds._manual_brightness_factor
        calibration_data = []

        try:
            for brightness in BRIGHTNESS_LEVELS:
                # Override auto-brightness pipeline
                leds._manual_brightness_factor = brightness
                # Directly set LEDs to white at desired brightness
                if leds._dots:
                    for i in range(3):
                        leds._dots[i] = (255, 255, 255)
                    hw_brightness = max(0.001, brightness) if brightness > 0 else 0.0
                    leds._dots.brightness = hw_brightness
                    leds._dots.show()

                # Wait for sensor to settle
                await asyncio.sleep(SETTLE_SECONDS)

                # Sample light sensor multiple times
                lux_readings = []
                for _ in range(SAMPLES_PER_LEVEL):
                    try:
                        readings = sensors.read()
                        if readings.light_lux is not None:
                            lux_readings.append(readings.light_lux)
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)

                avg_lux = sum(lux_readings) / len(lux_readings) if lux_readings else None
                calibration_data.append({
                    "brightness": brightness,
                    "raw_lux": round(avg_lux, 2) if avg_lux is not None else None,
                    "samples": len(lux_readings),
                })

        finally:
            # Always restore normal operation
            leds._manual_brightness_factor = original_factor

        # Linear fit: lux = slope * brightness + intercept
        fitted = None
        nonzero = [(d["brightness"], d["raw_lux"])
                    for d in calibration_data
                    if d["brightness"] > 0 and d["raw_lux"] is not None]
        if len(nonzero) >= 2:
            xs = [b for b, _ in nonzero]
            ys = [lux for _, lux in nonzero]
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            ss_xx = sum((x - mean_x) ** 2 for x in xs)
            slope = ss_xy / ss_xx if ss_xx > 0 else 0
            intercept = mean_y - slope * mean_x
            fitted = {
                "LED_LUX_QUADRATIC_linear_slope": round(slope, 1),
                "LED_LUX_QUADRATIC_linear_intercept": round(intercept, 1),
            }

        zero_reading = next((d for d in calibration_data if d["brightness"] == 0.0), None)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "action": "calibrate_leds",
            "data": calibration_data,
            "ambient_lux_at_zero_brightness": zero_reading["raw_lux"] if zero_reading else None,
            "fitted_constants": fitted,
            "current_config": {
                "LED_LUX_QUADRATIC": 1150.0,
                "model": "quadratic: glow = 1150 * brightness^2",
            },
            "note": "Compare fitted data against current_config. Update config.py if significantly different.",
        }, indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown action: {action}",
            "valid_actions": VALID_DISPLAY_ACTIONS,
        }))]
