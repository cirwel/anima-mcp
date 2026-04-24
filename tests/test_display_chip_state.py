"""Tests for PilRenderer chip-state proprioception.

The ST7789 TFT can enter reset via a D24 droop transient, coming back up in
sleep/display-off mode. SPI writes continue to succeed at the kernel level but
the chip silently discards pixels. The renderer needs a way to detect this and
recover without a full service restart.

These tests exercise three methods on PilRenderer:
- ``wake_chip()``: idempotent SLPOUT + DISPON resend
- ``probe_chip_state()``: RDDPM (0x0A) readback
- ``verify_and_recover()``: probe; if asleep/off, wake
"""

from unittest.mock import MagicMock

import pytest

from anima_mcp.display.renderer import PilRenderer


# RDDPM (0x0A) bits per ST7789 datasheet:
#   bit 4 = SLPOUT (1 = out of sleep)
#   bit 2 = DISON  (1 = display on)
AWAKE_DISPLAYING = 0b00010100  # SLPOUT=1, DISON=1
ASLEEP = 0b00000000
AWAKE_DISPLAY_OFF = 0b00010000  # SLPOUT=1, DISON=0


@pytest.fixture
def renderer_with_fake_display():
    """Bypass _init_display by assigning a fake ST7789 handle directly."""
    r = PilRenderer.__new__(PilRenderer)
    from anima_mcp.display.renderer import DisplayConfig
    r.config = DisplayConfig()
    r._display = MagicMock()
    r._image = None
    r._init_error = None
    r._last_face_state = None
    r._last_blink_time = 0.0
    r._blink_in_progress = False
    r._blink_start_time = 0.0
    r._deferred = False
    r._cached_dimmed_image = None
    r._cached_source_id = None
    r._cached_brightness = 1.0
    r._manual_brightness = 1.0
    r._display_fail_count = 0
    r._last_reinit_attempt = 0.0
    r._cs_pin = None
    r._dc_pin = None
    return r


def test_wake_chip_sends_slpout_then_dispon(renderer_with_fake_display):
    r = renderer_with_fake_display
    r.wake_chip()
    calls = [c.args for c in r._display.write.call_args_list]
    # First command: SLPOUT (0x11). Second: DISPON (0x29). No data bytes either.
    assert calls[0][0] == 0x11
    assert calls[1][0] == 0x29
    assert len(calls) == 2


def test_wake_chip_noop_when_no_display():
    r = PilRenderer.__new__(PilRenderer)
    r._display = None
    # Must not raise
    r.wake_chip()


def test_wake_chip_swallows_write_exception(renderer_with_fake_display, capsys):
    r = renderer_with_fake_display
    r._display.write.side_effect = RuntimeError("spi busy")
    r.wake_chip()  # must not raise
    assert "wake_chip" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("raw,expected_sleep_out,expected_on", [
    (AWAKE_DISPLAYING, True, True),
    (ASLEEP, False, False),
    (AWAKE_DISPLAY_OFF, True, False),
])
def test_probe_chip_state_parses_rddpm(renderer_with_fake_display,
                                        raw, expected_sleep_out, expected_on):
    r = renderer_with_fake_display
    r._display.read.return_value = bytes([raw])
    state = r.probe_chip_state()
    assert state is not None
    assert state["sleep_out"] is expected_sleep_out
    assert state["display_on"] is expected_on
    assert state["raw"] == raw


def test_probe_chip_state_uses_last_byte_if_multibyte(renderer_with_fake_display):
    """Some SPI read paths prepend a dummy byte; use the last byte as the value."""
    r = renderer_with_fake_display
    r._display.read.return_value = bytes([0x00, AWAKE_DISPLAYING])
    state = r.probe_chip_state()
    assert state is not None
    assert state["sleep_out"] is True
    assert state["display_on"] is True


def test_probe_chip_state_returns_none_on_read_error(renderer_with_fake_display):
    r = renderer_with_fake_display
    r._display.read.side_effect = OSError("SPI read failed")
    assert r.probe_chip_state() is None


def test_probe_chip_state_returns_none_when_no_display():
    r = PilRenderer.__new__(PilRenderer)
    r._display = None
    assert r.probe_chip_state() is None


def test_verify_and_recover_wakes_asleep_chip(renderer_with_fake_display):
    r = renderer_with_fake_display
    r._display.read.return_value = bytes([ASLEEP])
    result = r.verify_and_recover()
    assert result["recovered"] is True
    # Two writes (SLPOUT + DISPON) were issued
    assert r._display.write.call_count == 2


def test_verify_and_recover_noop_when_awake(renderer_with_fake_display):
    r = renderer_with_fake_display
    r._display.read.return_value = bytes([AWAKE_DISPLAYING])
    result = r.verify_and_recover()
    assert result["recovered"] is False
    assert r._display.write.call_count == 0


def test_verify_and_recover_recovers_display_off_state(renderer_with_fake_display):
    r = renderer_with_fake_display
    r._display.read.return_value = bytes([AWAKE_DISPLAY_OFF])
    result = r.verify_and_recover()
    assert result["recovered"] is True
    assert r._display.write.call_count == 2


def test_verify_and_recover_reports_probe_failure(renderer_with_fake_display):
    r = renderer_with_fake_display
    r._display.read.side_effect = OSError("boom")
    result = r.verify_and_recover()
    assert result["probed"] is False
    # On probe failure, still attempt wake as a best-effort release valve
    assert r._display.write.call_count == 2
    assert result["recovered"] is True
