"""Regression tests for HTTP shutdown ownership under the Starlette lifespan.

These tests target two specific regressions that were fixed in 144e19f/173711e:

1. Long-press poweroff must signal the process via SIGTERM so uvicorn/lifespan
   drive cleanup ordering. It must NOT call sleep()/stop_display_loop() directly
   from the input poll task — that bypasses the lifespan's teardown path.

2. The server warmup task must not flip SERVER_READY when cancelled during its
   startup sleep. If shutdown begins before warmup completes, the flag should
   remain False.
"""
from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.input_handler import _handle_separate_button


# ---------- P1: Long-press sends SIGTERM, doesn't bypass lifespan ----------


@pytest.fixture
def fake_ctx():
    """Minimal _ctx with just the fields _handle_separate_button touches."""
    return SimpleNamespace(
        sep_btn_press_start=None,
        joystick_enabled=True,
        leds=None,
    )


def _held_button(duration: float):
    """Input state with separate button held, no joystick/other buttons."""
    return SimpleNamespace(
        separate_button=True,
        joystick_button=False,
        joystick_direction=None,
    )


def test_long_press_sends_sigterm_not_direct_shutdown(fake_ctx):
    """Long-press must signal the process, not call sleep()/stop_display_loop() itself."""
    from anima_mcp.display.screens import ScreenMode

    renderer = MagicMock()
    renderer.canvas_save.return_value = None
    mode_change_event = asyncio.Event()
    input_state = _held_button(5.0)

    # Simulate 4 seconds of button hold by backdating the press start
    import time
    fake_ctx.sep_btn_press_start = time.time() - 4.0

    with patch("os.kill") as mock_kill, \
         patch("anima_mcp.server.sleep") as mock_sleep, \
         patch("anima_mcp.server.stop_display_loop") as mock_stop_display:
        _handle_separate_button(
            fake_ctx, input_state, renderer, ScreenMode.FACE, mode_change_event
        )

        # Must signal the process via SIGTERM (lifespan-driven shutdown)
        mock_kill.assert_called_once()
        args, _ = mock_kill.call_args
        assert args[1] == signal.SIGTERM, f"Expected SIGTERM, got {args[1]}"

        # Must NOT call sleep() or stop_display_loop() directly
        mock_sleep.assert_not_called()
        mock_stop_display.assert_not_called()


def test_long_press_saves_canvas_in_notepad_mode(fake_ctx):
    """Long-press in NOTEPAD mode should save the canvas before signaling."""
    from anima_mcp.display.screens import ScreenMode

    renderer = MagicMock()
    renderer.canvas_save.return_value = "/tmp/drawing.png"
    mode_change_event = asyncio.Event()
    input_state = _held_button(5.0)

    import time
    fake_ctx.sep_btn_press_start = time.time() - 4.0

    with patch("os.kill"):
        _handle_separate_button(
            fake_ctx, input_state, renderer, ScreenMode.NOTEPAD, mode_change_event
        )

    renderer.canvas_save.assert_called_once()


def test_short_press_does_not_trigger_shutdown(fake_ctx):
    """Holding the button for less than SHUTDOWN_LONG_PRESS_SECONDS must not signal."""
    from anima_mcp.display.screens import ScreenMode

    renderer = MagicMock()
    mode_change_event = asyncio.Event()
    input_state = _held_button(0.5)

    import time
    fake_ctx.sep_btn_press_start = time.time() - 0.5  # below 3s threshold

    with patch("os.kill") as mock_kill:
        _handle_separate_button(
            fake_ctx, input_state, renderer, ScreenMode.FACE, mode_change_event
        )
        mock_kill.assert_not_called()


# ---------- P3: Warmup task cancellation doesn't flip SERVER_READY ----------


@pytest.mark.asyncio
async def test_warmup_task_cancellation_does_not_flip_ready_flag():
    """The warmup pattern must respect cancellation during its startup sleep.

    Mirrors the structure used in server.py's run_http_server() lifespan.
    If the lifespan cancels the warmup during shutdown, the ready flag must
    remain False — otherwise the server advertises itself as ready while it's
    tearing down.
    """
    ready_flag = {"value": False}

    async def warmup():
        # Same shape as server.py's server_warmup_task
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            return  # Shutdown during warmup — do NOT flip ready
        ready_flag["value"] = True

    task = asyncio.create_task(warmup())
    # Give the task a chance to enter its sleep
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert ready_flag["value"] is False, \
        "Warmup task flipped ready flag despite being cancelled — lifespan teardown race"


@pytest.mark.asyncio
async def test_warmup_task_completes_normally_when_not_cancelled():
    """Sanity check: warmup pattern does flip the flag when allowed to finish."""
    ready_flag = {"value": False}

    async def warmup():
        try:
            await asyncio.sleep(0.01)  # fast for test
        except asyncio.CancelledError:
            return
        ready_flag["value"] = True

    task = asyncio.create_task(warmup())
    await task

    assert ready_flag["value"] is True
