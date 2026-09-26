"""Tests for brightness simplification (Lighthouse LED plan Task 2)."""

import threading

from anima_mcp.display.leds import brightness as brightness_mod
from anima_mcp.display.leds.display import LEDDisplay
from anima_mcp.display.leds.types import LEDState


class TestRemovedFunctions:
    """Auto-brightness, pulsing-brightness, and gamma should no longer exist."""

    def test_get_auto_brightness_removed(self):
        assert not hasattr(brightness_mod, "get_auto_brightness"), (
            "get_auto_brightness should be removed"
        )

    def test_get_pulsing_brightness_removed(self):
        assert not hasattr(brightness_mod, "get_pulsing_brightness"), (
            "get_pulsing_brightness should be removed"
        )

    def test_apply_gamma_removed(self):
        assert not hasattr(brightness_mod, "apply_gamma"), (
            "apply_gamma should be removed"
        )


class TestGetPulse:
    """get_pulse should still exist and return 0-1."""

    def test_returns_float(self):
        result = brightness_mod.get_pulse()
        assert isinstance(result, float)

    def test_in_range(self):
        for _ in range(100):
            result = brightness_mod.get_pulse(pulse_cycle=4.0)
            assert 0.0 <= result <= 1.0, f"Pulse {result} out of [0, 1]"


class TestEstimateInstantaneousBrightness:
    """estimate_instantaneous_brightness should scale amplitude with brightness."""

    def test_returns_positive(self):
        result = brightness_mod.estimate_instantaneous_brightness(0.04)
        assert result > 0.0

    def test_low_brightness_small_amplitude(self):
        """At low brightness (0.04), pulse amplitude should be very small."""
        base = 0.04
        result = brightness_mod.estimate_instantaneous_brightness(base, pulse_amount=0.05)
        # Result should be close to base -- amplitude capped
        deviation = abs(result - base)
        assert deviation < 0.01, (
            f"Deviation {deviation} too large for base={base}"
        )

    def test_high_brightness_larger_amplitude(self):
        """At higher brightness, amplitude can be larger but still proportional."""
        base = 0.12
        result = brightness_mod.estimate_instantaneous_brightness(base, pulse_amount=0.05)
        assert result >= 0.008  # floor

    def test_floor_brightness(self):
        """Should never go below 0.008."""
        result = brightness_mod.estimate_instantaneous_brightness(0.001)
        assert result >= 0.008


class TestDefaultBrightnessInTypes:
    """LEDState default brightness should be 0.04."""

    def test_default_is_004(self):
        state = LEDState(led0=(0, 0, 0), led1=(0, 0, 0), led2=(0, 0, 0))
        assert state.brightness == 0.04, f"Expected 0.04, got {state.brightness}"


class _FakeDots:
    def __init__(self):
        self.brightness = 1.0
        self.colors = {}
        self.show_count = 0

    def __setitem__(self, index, color):
        self.colors[index] = color

    def show(self):
        self.show_count += 1


class TestDotStarWireBrightness:
    """Software brightness must not mutate DotStar's per-pixel header."""

    def test_frame_scales_rgb_and_leaves_library_brightness_invariant(self):
        display = LEDDisplay.__new__(LEDDisplay)
        display._dots = _FakeDots()
        display._spi_lock = threading.Lock()

        wire = display._write_frame(
            [(200, 100, 50), (255, 128, 0), (10, 20, 30)],
            0.18,
        )

        assert wire == [(36, 18, 9), (45, 23, 0), (1, 3, 5)]
        assert display._dots.colors == dict(enumerate(wire))
        assert display._dots.brightness == 1.0
        assert display._dots.show_count == 1

    def test_cached_updates_keep_converging_to_absolute_preset(self):
        display = LEDDisplay.__new__(LEDDisplay)
        display._hardware_brightness_floor = 0.008
        display._brightness_transition_speed = 0.08
        display._current_brightness = 0.18
        display._target_brightness = 0.28
        display._cached_pipeline_brightness = 0.18
        display._known_brightness = 0.18

        for _ in range(200):
            display._advance_current_brightness()

        assert display._current_brightness == 0.28
        assert display._cached_pipeline_brightness == 0.28
        assert display._known_brightness == 0.28


class TestNoLuxImport:
    """brightness.py should not import LED_LUX_PER_BRIGHTNESS or LED_LUX_AMBIENT_FLOOR."""

    def test_no_lux_constants_in_module(self):
        assert not hasattr(brightness_mod, "LED_LUX_PER_BRIGHTNESS"), (
            "LED_LUX_PER_BRIGHTNESS should not be imported in brightness.py"
        )
        assert not hasattr(brightness_mod, "LED_LUX_AMBIENT_FLOOR"), (
            "LED_LUX_AMBIENT_FLOOR should not be imported in brightness.py"
        )


class TestAgencyBrightnessFactor:
    """The agency LED action is a real, bounded dimmer under the user preset.

    It used to call set_brightness(), which moved ``_base_brightness`` -- a
    value the always-set renderer preset overrides -- so it changed nothing
    physically while reporting a change.
    """

    PRESETS = (0.28, 0.12, 0.06, 0.008)  # Full, Medium, Dim, Night

    def _display(self, preset: float, activity: float = 1.0) -> LEDDisplay:
        display = LEDDisplay(brightness=0.04)
        display._dots = _FakeDots()
        display._manual_brightness_factor = preset
        display.update_from_anima(0.5, 0.5, 0.5, 0.5, activity_brightness=activity)
        return display

    def test_decrease_physically_lowers_requested_target(self):
        display = self._display(0.12)
        assert display._target_brightness == 0.12
        result = display.adjust_agency_brightness("decrease")
        assert result["changed"] is True
        assert display._target_brightness < 0.12
        assert result["target_after"] == display._target_brightness
        # The next pipeline pass keeps the agency's choice instead of
        # reverting to the preset (the original bug).
        display.update_from_anima(0.9, 0.1, 0.9, 0.1)
        assert abs(display._target_brightness - 0.12 * 0.8) < 1e-9
        # Easing actually carries the applied brightness down to it.
        for _ in range(300):
            display._advance_current_brightness()
        assert abs(display._current_brightness - 0.12 * 0.8) < 1e-9

    def test_never_exceeds_user_preset(self):
        for preset in self.PRESETS:
            for activity in (1.0, 0.6, 0.35):
                display = self._display(preset, activity)
                ceiling = display._requested_target(agency_factor=1.0)
                for direction in ["increase"] * 5 + ["decrease"] * 7 + ["increase"] * 20:
                    display.adjust_agency_brightness(direction)
                    display.update_from_anima(0.5, 0.5, 0.5, 0.5, activity_brightness=activity)
                    assert display._target_brightness <= ceiling + 1e-12
                    assert display._target_brightness <= max(
                        preset, display._hardware_brightness_floor
                    ) + 1e-12
                    assert 0.0 < display._agency_brightness_factor <= 1.0
                # Enough increases return exactly to the preset, not past it.
                assert display._agency_brightness_factor == 1.0
                assert abs(display._target_brightness - ceiling) < 1e-12

    def test_increase_at_preset_reports_no_change(self):
        display = self._display(0.12)
        result = display.adjust_agency_brightness("increase")
        assert result["changed"] is False
        assert result["target_before"] == result["target_after"] == 0.12
        assert display._agency_brightness_factor == 1.0

    def test_decrease_at_hardware_floor_reports_no_change(self):
        display = self._display(0.008)  # Night preset sits on the floor
        result = display.adjust_agency_brightness("decrease")
        assert result["changed"] is False
        assert display._agency_brightness_factor == 1.0
        assert display._target_brightness == display._hardware_brightness_floor

    def test_agency_step_is_visible_to_light_attribution_as_target_change(self):
        from anima_mcp.light_attribution import LearnedLedLuxResidual

        display = self._display(0.06)
        before = display.get_proprioceptive_state()["target_brightness"]
        display.adjust_agency_brightness("decrease")
        after = display.get_proprioceptive_state()["target_brightness"]
        assert abs(before - after) > LearnedLedLuxResidual.TARGET_BRIGHTNESS_TOLERANCE

    def test_does_not_touch_base_brightness(self):
        display = self._display(0.06)
        display.adjust_agency_brightness("decrease")
        assert display._base_brightness == 0.04
