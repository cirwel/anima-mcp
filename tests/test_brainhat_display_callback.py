"""Tests for BrainHatInput → renderer callback after display-pin pulse.

The pulse on D22/D24 only restores the pin *voltage*. If the ST7789 chip
entered reset during a prior D24 droop, releasing reset returns the chip to
sleep/display-off mode and the pulse alone cannot re-display. After every
pulse, BrainHatInput must notify a registered callback so the renderer can
verify chip state and recover if needed.
"""

import sys
import types

import pytest


@pytest.fixture
def mocked_brainhat(monkeypatch):
    """Same fixture shape as test_brainhat_input_heartbeat — keep them aligned."""
    fake_board = types.SimpleNamespace(
        D16=object(), D17=object(),
        D22=object(), D23=object(), D24=object(), D27=object(),
    )
    pins = {}

    class FakePin:
        def __init__(self, board_pin):
            self.board_pin = board_pin
            self.direction = None
            self.pull = None
            self._value = True
            self._deinited = False
            pins.setdefault(id(board_pin), []).append(self)

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, v):
            self._value = v

        def deinit(self):
            self._deinited = True

    fake_digitalio = types.SimpleNamespace(
        DigitalInOut=FakePin,
        Direction=types.SimpleNamespace(INPUT="input", OUTPUT="output"),
        Pull=types.SimpleNamespace(UP="up", DOWN="down"),
    )

    monkeypatch.setitem(sys.modules, "board", fake_board)
    monkeypatch.setitem(sys.modules, "digitalio", fake_digitalio)
    sys.modules.pop("anima_mcp.input.brainhat_input", None)
    from anima_mcp.input import brainhat_input as mod
    monkeypatch.setattr(mod, "HAS_GPIO", True)
    monkeypatch.setattr(mod, "board", fake_board)
    monkeypatch.setattr(mod, "digitalio", fake_digitalio)
    return mod


def test_registered_callback_fires_after_pulse(mocked_brainhat):
    mod = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    calls = []
    inp.set_post_pulse_callback(lambda: calls.append(True))

    inp._refresh_display_pins()
    assert calls == [True]


def test_callback_fires_after_pin_restoration(mocked_brainhat):
    """Callback must fire AFTER pins are restored to INPUT PULL_UP.

    This ordering matters: the renderer's verify_and_recover may briefly
    drive CS/DC (different pins) but the callback should never observe
    D22/D24 mid-pulse.
    """
    mod = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    observed_left = []

    def callback():
        observed_left.append(inp._joy_left)

    inp.set_post_pulse_callback(callback)
    inp._refresh_display_pins()

    assert observed_left[0] is inp._joy_left  # the restored input pin
    assert observed_left[0].direction == "input"
    assert observed_left[0].pull == "up"


def test_callback_exception_is_swallowed(mocked_brainhat, capsys):
    mod = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    def boom():
        raise RuntimeError("display module unreachable")

    inp.set_post_pulse_callback(boom)
    # Must not raise
    inp._refresh_display_pins()

    err = capsys.readouterr().err
    assert "post_pulse_callback" in err.lower() or "callback" in err.lower()

    # Pins remain valid afterwards
    assert inp._joy_left is not None
    assert inp._joy_right is not None


def test_no_callback_is_safe(mocked_brainhat):
    mod = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()
    # No callback registered; refresh must still work
    inp._refresh_display_pins()
    assert inp._joy_left is not None
    assert inp._joy_right is not None
