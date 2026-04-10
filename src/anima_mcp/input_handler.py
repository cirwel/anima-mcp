"""Fast input polling — joystick and button handling for BrainCraft HAT.

Extracted from server.py's _update_display_loop to isolate the input state machine.
Runs as an async task at ~60fps, handles edge-detected button/joystick events.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

logger = logging.getLogger("anima.server")


async def fast_input_poll(mode_change_event: asyncio.Event):
    """Fast input polling — runs independently of display loop for responsive controls.

    Args:
        mode_change_event: Event to signal display refresh when user navigates.
    """
    from .input.brainhat_input import get_brainhat_input, JoystickDirection as InputDirection
    from .display.screens import ScreenMode
    from .server_state import (
        INPUT_ERROR_LOG_INTERVAL, INPUT_POLL_INTERVAL_SECONDS,
    )

    def _get_ctx():
        from .ctx_ref import get_ctx
        return get_ctx()

    brainhat = get_brainhat_input()
    _ctx = _get_ctx()
    if _ctx is None:
        return

    if not _ctx.joystick_enabled:
        try:
            brainhat.enable()
            if brainhat.is_available():
                _ctx.joystick_enabled = True
                print("[Input] BrainHat input enabled - buttons and joystick ready", file=sys.stderr, flush=True)
            else:
                if not hasattr(fast_input_poll, '_logged_unavailable'):
                    print("[Input] BrainHat hardware not available - buttons disabled (not on Pi or hardware issue)", file=sys.stderr, flush=True)
                    fast_input_poll._logged_unavailable = True
        except Exception as e:
            if not hasattr(fast_input_poll, '_logged_error'):
                print(f"[Input] Failed to enable input: {e}", file=sys.stderr, flush=True)
                fast_input_poll._logged_error = True

    while True:
        try:
            _ctx = _get_ctx()
            if _ctx is None:
                break
            renderer = _ctx.screen_renderer
            if _ctx.joystick_enabled and renderer:
                input_state = brainhat.read()
                if input_state:
                    prev_state = brainhat.get_prev_state()
                    current_mode = renderer.get_mode()

                    # Check button presses (edge detection)
                    current_dir = input_state.joystick_direction
                    if input_state.joystick_button:
                        if _ctx.joy_btn_press_start is None:
                            _ctx.joy_btn_press_start = time.time()
                            _ctx.joy_btn_help_shown = False
                        elif not _ctx.joy_btn_help_shown and (time.time() - _ctx.joy_btn_press_start) >= 1.0:
                            renderer.trigger_controls_overlay()
                            mode_change_event.set()
                            _ctx.joy_btn_help_shown = True
                            if _ctx.leds and _ctx.leds.is_available():
                                _ctx.leds.quick_flash((70, 110, 140), 70)
                    else:
                        if _ctx.joy_btn_press_start is not None:
                            hold_time = time.time() - _ctx.joy_btn_press_start
                            if hold_time < 0.8 and not _ctx.joy_btn_help_shown:
                                # Short joystick button press: cycle to next screen in group
                                current_mode = renderer.get_mode()
                                if current_mode not in (ScreenMode.MESSAGES, ScreenMode.QUESTIONS, ScreenMode.VISITORS):
                                    renderer.trigger_input_feedback("button")
                                    renderer.next_in_group()
                                    mode_change_event.set()
                                    current_mode = renderer.get_mode()
                                    print(f"[Input] btn -> {current_mode.value} (group cycle)", file=sys.stderr, flush=True)
                        _ctx.joy_btn_press_start = None
                        _ctx.joy_btn_help_shown = False

                    if prev_state:
                        prev_dir = prev_state.joystick_direction
                        qa_expanded = renderer._state.qa_expanded if renderer else False
                        qa_needs_lr = (current_mode == ScreenMode.QUESTIONS and qa_expanded)

                        if not qa_needs_lr:
                            if current_dir == InputDirection.LEFT and prev_dir != InputDirection.LEFT:
                                renderer.trigger_input_feedback("left")
                                if _ctx.leds and _ctx.leds.is_available():
                                    _ctx.leds.quick_flash((60, 60, 120), 50)
                                old_mode = renderer.get_mode()
                                renderer.navigate_left()
                                new_mode = renderer.get_mode()
                                renderer._state.last_user_action_time = time.time()
                                mode_change_event.set()
                                print(f"[Input] {old_mode.value} -> {new_mode.value} (left)", file=sys.stderr, flush=True)
                            elif current_dir == InputDirection.RIGHT and prev_dir != InputDirection.RIGHT:
                                renderer.trigger_input_feedback("right")
                                if _ctx.leds and _ctx.leds.is_available():
                                    _ctx.leds.quick_flash((60, 60, 120), 50)
                                old_mode = renderer.get_mode()
                                renderer.navigate_right()
                                new_mode = renderer.get_mode()
                                renderer._state.last_user_action_time = time.time()
                                mode_change_event.set()
                                print(f"[Input] {old_mode.value} -> {new_mode.value} (right)", file=sys.stderr, flush=True)

                    # Refresh mode after possible group navigation
                    current_mode = renderer.get_mode()

                    # Per-screen UP/DOWN (and QUESTIONS L/R) via dispatch
                    if prev_state:
                        _handle_up_down(
                            input_state.joystick_direction,
                            prev_state.joystick_direction,
                            current_mode, renderer, mode_change_event,
                            ScreenMode, InputDirection,
                        )

                    # Separate button - with long-press shutdown for mobile readiness
                    _handle_separate_button(
                        _ctx, input_state, renderer, current_mode,
                        mode_change_event,
                    )
        except Exception as e:
            # Log errors but don't spam - only log occasionally
            _ctx = _get_ctx()
            if _ctx:
                current_time = time.time()
                if current_time - _ctx.last_input_error_log > INPUT_ERROR_LOG_INTERVAL:
                    print(f"[Input] Error in input polling: {e}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    _ctx.last_input_error_log = current_time
        await asyncio.sleep(INPUT_POLL_INTERVAL_SECONDS)


# Dispatch table: ScreenMode → (up_method, down_method, triggers_mode_change_event)
# Populated lazily since ScreenMode isn't available at import time.
_SCREEN_ACTIONS: dict | None = None
_GROUP_LOCAL_SCREENS: frozenset | None = None


def _get_dispatch_tables(ScreenMode):
    """Build dispatch tables on first use (avoids import-time dependency on ScreenMode)."""
    global _SCREEN_ACTIONS, _GROUP_LOCAL_SCREENS
    if _SCREEN_ACTIONS is not None:
        return _SCREEN_ACTIONS, _GROUP_LOCAL_SCREENS
    _SCREEN_ACTIONS = {
        ScreenMode.MESSAGES: ("message_scroll_up", "message_scroll_down", False),
        ScreenMode.VISITORS: ("message_scroll_up", "message_scroll_down", False),
        ScreenMode.ART_ERAS: ("era_cursor_up", "era_cursor_down", True),
        ScreenMode.QUESTIONS: ("qa_scroll_up", "qa_scroll_down", False),
    }
    _GROUP_LOCAL_SCREENS = frozenset({
        ScreenMode.IDENTITY, ScreenMode.SENSORS, ScreenMode.DIAGNOSTICS,
        ScreenMode.HEALTH, ScreenMode.NEURAL, ScreenMode.INNER_LIFE,
        ScreenMode.LEARNING, ScreenMode.SELF_GRAPH, ScreenMode.GOALS_BELIEFS,
        ScreenMode.AGENCY, ScreenMode.NOTEPAD,
    })
    return _SCREEN_ACTIONS, _GROUP_LOCAL_SCREENS


def _handle_up_down(current_dir, prev_dir, current_mode, renderer, mode_change_event,
                    ScreenMode, InputDirection):
    """Dispatch UP/DOWN joystick input based on screen mode."""
    # FACE has special brightness handler
    if current_mode == ScreenMode.FACE:
        if current_dir == InputDirection.UP and prev_dir != InputDirection.UP:
            renderer.trigger_input_feedback("up")
            preset_name = renderer._display.brightness_up()
            preset = renderer._display.get_brightness_preset()
            display_level = min(1.0, preset["leds"] / 0.28)
            renderer.trigger_brightness_overlay(preset_name, display_level)
            mode_change_event.set()
        elif current_dir == InputDirection.DOWN and prev_dir != InputDirection.DOWN:
            renderer.trigger_input_feedback("down")
            preset_name = renderer._display.brightness_down()
            preset = renderer._display.get_brightness_preset()
            display_level = min(1.0, preset["leds"] / 0.28)
            renderer.trigger_brightness_overlay(preset_name, display_level)
            mode_change_event.set()
        return

    # QUESTIONS also handles L/R for focus navigation
    if current_mode == ScreenMode.QUESTIONS:
        if current_dir == InputDirection.LEFT and prev_dir != InputDirection.LEFT:
            renderer.trigger_input_feedback("left")
            renderer.qa_focus_next()
        elif current_dir == InputDirection.RIGHT and prev_dir != InputDirection.RIGHT:
            renderer.trigger_input_feedback("right")
            renderer.qa_focus_next()

    # Look up in dispatch table
    actions, group_screens = _get_dispatch_tables(ScreenMode)
    entry = actions.get(current_mode)
    if entry is None and current_mode in group_screens:
        entry = ("previous_in_group", "next_in_group", True)

    if entry is None:
        return

    up_method, down_method, needs_event = entry
    if current_dir == InputDirection.UP and prev_dir != InputDirection.UP:
        renderer.trigger_input_feedback("up")
        getattr(renderer, up_method)()
        if needs_event:
            mode_change_event.set()
    elif current_dir == InputDirection.DOWN and prev_dir != InputDirection.DOWN:
        renderer.trigger_input_feedback("down")
        getattr(renderer, down_method)()
        if needs_event:
            mode_change_event.set()


def _handle_separate_button(
    _ctx,
    input_state,
    renderer,
    current_mode,
    mode_change_event: asyncio.Event,
):
    """Handle the separate (side) button — short press and long-press shutdown."""
    from .display.screens import ScreenMode
    from .server_state import SHUTDOWN_LONG_PRESS_SECONDS

    if input_state.separate_button:
        if _ctx.sep_btn_press_start is None:
            _ctx.sep_btn_press_start = time.time()

        hold_duration = time.time() - _ctx.sep_btn_press_start

        # Long press (3+ seconds) = graceful shutdown
        # Signal the process to shut down via SIGTERM so uvicorn/lifespan
        # drive cleanup ordering. Do NOT call sleep()/stop_display_loop()
        # directly — that bypasses the lifespan's teardown path.
        if hold_duration >= SHUTDOWN_LONG_PRESS_SECONDS:
            print(f"[Shutdown] Long press detected ({hold_duration:.1f}s) - requesting graceful shutdown...", file=sys.stderr, flush=True)
            try:
                if current_mode == ScreenMode.NOTEPAD:
                    saved_path = renderer.canvas_save()
                    if saved_path:
                        print(f"[Shutdown] Saved drawing to {saved_path}", file=sys.stderr, flush=True)

                import os
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
                return  # Let the signal handler / lifespan drive shutdown
            except Exception as e:
                print(f"[Shutdown] Error requesting shutdown: {e}", file=sys.stderr, flush=True)
                raise SystemExit(1)
    else:
        # Button released - check if it was a short press
        if _ctx.sep_btn_press_start is not None:
            hold_duration = time.time() - _ctx.sep_btn_press_start
            was_short_press = hold_duration < SHUTDOWN_LONG_PRESS_SECONDS
            _ctx.sep_btn_press_start = None

            if was_short_press:
                renderer.trigger_input_feedback("press")
                if _ctx.leds and _ctx.leds.is_available():
                    _ctx.leds.quick_flash((80, 100, 60), 80)
                handled_short_press = False
                if current_mode == ScreenMode.MESSAGES:
                    renderer.message_toggle_expand()
                    print("[Messages] Toggled message expansion", file=sys.stderr, flush=True)
                    handled_short_press = True
                elif current_mode == ScreenMode.VISITORS:
                    renderer.message_toggle_expand()
                    print("[Visitors] Toggled message expansion", file=sys.stderr, flush=True)
                    handled_short_press = True
                elif current_mode == ScreenMode.QUESTIONS:
                    renderer.qa_toggle_expand()
                    print("[Questions] Toggled Q&A expansion", file=sys.stderr, flush=True)
                    handled_short_press = True
                elif current_mode == ScreenMode.NOTEPAD:
                    era_name = getattr(renderer, '_active_era', None)
                    era_name = getattr(era_name, 'name', '') if era_name else ''
                    if era_name == 'gestural':
                        print("[Notepad] Gestural era — no manual save", file=sys.stderr, flush=True)
                    else:
                        saved = renderer.canvas_save(manual=True)
                        if saved:
                            print(f"[Notepad] Manual save: {saved}", file=sys.stderr, flush=True)
                        else:
                            print("[Notepad] Manual save: canvas empty", file=sys.stderr, flush=True)
                    handled_short_press = True
                elif current_mode == ScreenMode.ART_ERAS:
                    result = renderer.era_select_current()
                    renderer._state.last_user_action_time = time.time()
                    mode_change_event.set()
                    print(f"[ArtEras] Button press: {result}", file=sys.stderr, flush=True)
                    handled_short_press = True
                # Universal fallback
                if not handled_short_press:
                    if current_mode != ScreenMode.FACE:
                        old_mode = current_mode
                        renderer.set_mode(ScreenMode.FACE)
                        mode_change_event.set()
                        print(f"[Input] Side button fallback: {old_mode.value} -> face", file=sys.stderr, flush=True)
                    else:
                        renderer.trigger_controls_overlay()
                        mode_change_event.set()
                        print("[Input] Side button fallback: controls overlay", file=sys.stderr, flush=True)
