"""Tests for BrainHatInput display-pin heartbeat refresh."""

import sys
import types

import pytest


@pytest.fixture
def mocked_brainhat(monkeypatch):
    """Install fake board/digitalio so brainhat_input imports with HAS_GPIO=True."""
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

    return mod, fake_board, pins, FakePin


def test_refresh_pulses_d22_d24_high_then_restores_input_pullup(mocked_brainhat):
    mod, fake_board, pins, FakePin = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    initial_left = inp._joy_left
    initial_right = inp._joy_right
    assert initial_left is not None
    assert initial_right is not None
    assert initial_left.direction == "input"
    assert initial_left.pull == "up"

    inp._refresh_display_pins()

    # Original input pins were deinit'd
    assert initial_left._deinited is True
    assert initial_right._deinited is True

    # Pins were re-claimed as fresh INPUT PULL_UP objects
    assert inp._joy_left is not None and inp._joy_left is not initial_left
    assert inp._joy_right is not None and inp._joy_right is not initial_right
    assert inp._joy_left.direction == "input"
    assert inp._joy_left.pull == "up"
    assert inp._joy_right.direction == "input"
    assert inp._joy_right.pull == "up"

    # Intermediate OUTPUT HIGH pin was created and deinit'd for each of D22/D24
    d22_pins = pins[id(fake_board.D22)]
    d24_pins = pins[id(fake_board.D24)]
    # init -> refresh output pulse -> refresh re-input. So 3 pins per.
    assert len(d22_pins) >= 3
    assert len(d24_pins) >= 3
    # The middle (output) pin was driven high and released
    out_d22 = d22_pins[-2]
    out_d24 = d24_pins[-2]
    assert out_d22.direction == "output"
    assert out_d22.value is True
    assert out_d22._deinited is True
    assert out_d24.direction == "output"
    assert out_d24.value is True
    assert out_d24._deinited is True


def test_read_triggers_refresh_after_interval(mocked_brainhat, monkeypatch):
    mod, _, _, _ = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    calls = []
    monkeypatch.setattr(inp, "_refresh_display_pins", lambda: calls.append(True))

    times = iter([0.0, 5.0, 29.9, 30.1, 35.0, 60.2])

    def fake_time():
        return next(times)

    monkeypatch.setattr(mod.time, "time", fake_time)

    for _ in range(6):
        inp.read()

    # Fires at first read (0.0 - 0.0 >= 30 is False, but _last_backlight_refresh=0.0
    # and now=0.0 means delta=0, not >= 30, so no fire)... actually we want the
    # first-ever call to fire since _last_backlight_refresh starts at 0.0 and
    # threshold is 30. Verify exact semantics:
    # times: 0.0 -> 0-0=0, no fire
    #        5.0 -> 5-0=5, no fire
    #        29.9 -> 29.9-0=29.9, no fire
    #        30.1 -> 30.1-0=30.1, FIRE (set _last_backlight_refresh=30.1)
    #        35.0 -> 35.0-30.1=4.9, no fire
    #        60.2 -> 60.2-30.1=30.1, FIRE
    assert len(calls) == 2


def test_refresh_is_noop_without_gpio(monkeypatch):
    sys.modules.pop("anima_mcp.input.brainhat_input", None)
    from anima_mcp.input import brainhat_input as mod
    monkeypatch.setattr(mod, "HAS_GPIO", False)
    inp = mod.BrainHatInput()
    # Must not raise even with no pins
    inp._refresh_display_pins()


def test_refresh_failure_is_swallowed(mocked_brainhat, capsys):
    mod, _, _, FakePin = mocked_brainhat
    inp = mod.BrainHatInput()
    inp.enable()

    def boom():
        raise RuntimeError("pin busy")

    inp._joy_left.deinit = boom
    # Should not raise
    inp._refresh_display_pins()
    err = capsys.readouterr().err
    assert "Display-pin refresh failed" in err
