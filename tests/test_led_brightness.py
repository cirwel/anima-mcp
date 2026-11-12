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


# ---------------------------------------------------------------------------
# 2026-11-12 -- frame-write failures must be audible, and a dead bus must read
# as unavailable (invariant 2: fail toward unknown, never toward healthy).
# ---------------------------------------------------------------------------


class _FlakyDots:
    """DotStar stand-in whose ``show()`` can be made to fail like a dead SPI bus."""

    def __init__(self, fail=True):
        self.fail = fail
        self.attempts = 0
        self.colors = {}

    def __setitem__(self, index, color):
        self.colors[index] = color

    def fill(self, color):
        pass

    def show(self):
        self.attempts += 1
        if self.fail:
            raise OSError(5, "Input/output error")


class TestFrameWriteFailuresAreAudible:
    """The animation thread is the only DotStar writer.

    It used to wrap the write in ``except Exception: pass``: with every write
    raising OSError there were zero log lines, ``is_available()`` stayed True,
    health read ``leds: ok`` and proprioception kept reporting a brightness
    "sent to hardware" that no frame had carried.
    """

    @staticmethod
    def _display(monkeypatch, tmp_path, dots):
        import pytest

        from anima_mcp import error_recovery, light_attribution

        error_recovery.reset_suppressed_counts()
        action_path = tmp_path / "led_proprioception.json"
        monkeypatch.setenv(light_attribution.LED_PROPRIOCEPTION_ENV, str(action_path))
        display = LEDDisplay(brightness=0.04)  # off-Pi: no DotStar, no thread
        if display._dots is not None:  # pragma: no cover - only on a real Pi
            pytest.skip("real DotStar hardware present")
        display._dots = dots
        display.update_from_anima(0.5, 0.6, 0.7, 0.8)
        return display, action_path

    @staticmethod
    def _run_ticks(monkeypatch, display, count):
        """Drive the real animation loop for ``count`` iterations, no thread."""
        import time as time_module

        seen = {"n": 0}

        def stop_after(_seconds):
            seen["n"] += 1
            if seen["n"] >= count:
                display._animation_running = False

        monkeypatch.setattr(time_module, "sleep", stop_after)
        display._animation_running = True
        display._animation_loop()

    @staticmethod
    def _floor():
        return LEDDisplay.WRITE_FAILURES_BEFORE_UNAVAILABLE

    def test_every_failed_write_is_counted(self, monkeypatch, tmp_path):
        from anima_mcp.error_recovery import suppressed_counts

        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        self._run_ticks(monkeypatch, display, 5)

        assert dots.attempts == 5
        assert suppressed_counts().get("leds.write_frame") == 5

    def test_first_failure_is_logged(self, monkeypatch, tmp_path, capsys):
        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        capsys.readouterr()
        self._run_ticks(monkeypatch, display, 1)

        err = capsys.readouterr().err
        assert "[suppressed] leds.write_frame" in err
        assert "Input/output error" in err

    def test_consecutive_failures_mark_hardware_unavailable(self, monkeypatch, tmp_path):
        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        assert display.is_available() is True

        self._run_ticks(monkeypatch, display, self._floor())

        assert display.is_available() is False
        assert display.get_diagnostics()["available"] is False
        assert display.get_diagnostics()["consecutive_write_failures"] == self._floor()

    def test_one_transient_error_does_not_flip_availability(self, monkeypatch, tmp_path):
        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)

        # Failures separated by a good frame never form a run.
        for _ in range(3):
            dots.fail = True
            self._run_ticks(monkeypatch, display, self._floor() - 1)
            assert display.is_available() is True
            dots.fail = False
            self._run_ticks(monkeypatch, display, 1)
            assert display.is_available() is True
            assert display.get_diagnostics()["consecutive_write_failures"] == 0

    def test_proprioception_claims_no_brightness_it_did_not_apply(self, monkeypatch, tmp_path):
        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)

        # No frame has ever reached the wire: nothing was applied.
        assert display.get_proprioceptive_state()["brightness"] is None
        self._run_ticks(monkeypatch, display, self._floor() + 2)
        assert display.get_proprioceptive_state()["brightness"] is None

    def test_dead_bus_after_good_frames_withdraws_brightness(self, monkeypatch, tmp_path):
        dots = _FlakyDots(fail=False)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        self._run_ticks(monkeypatch, display, 2)
        applied = display.get_proprioceptive_state()["brightness"]
        assert isinstance(applied, float) and applied > 0

        dots.fail = True
        self._run_ticks(monkeypatch, display, self._floor())
        assert display.is_available() is False
        assert display.get_proprioceptive_state()["brightness"] is None

    def test_successful_write_recovers(self, monkeypatch, tmp_path, capsys):
        from anima_mcp.light_attribution import read_led_proprioception

        dots = _FlakyDots(fail=True)
        display, action_path = self._display(monkeypatch, tmp_path, dots)
        self._run_ticks(monkeypatch, display, self._floor() + 4)
        assert display.is_available() is False
        # The light-attribution action copy is never written for a frame that
        # did not land, so the broker reads it as unknown.
        assert read_led_proprioception(path=action_path, max_age_seconds=2.0) is None

        dots.fail = False
        capsys.readouterr()
        self._run_ticks(monkeypatch, display, 1)

        assert display.is_available() is True
        assert display.get_diagnostics()["consecutive_write_failures"] == 0
        assert isinstance(display.get_proprioceptive_state()["brightness"], float)
        published = read_led_proprioception(path=action_path, max_age_seconds=2.0)
        assert published is not None
        assert published["brightness"] == display.get_proprioceptive_state()["brightness"]
        assert "available again" in capsys.readouterr().err

    def test_loop_errors_outside_the_write_are_counted_too(self, monkeypatch, tmp_path):
        from anima_mcp.error_recovery import suppressed_counts

        dots = _FlakyDots(fail=False)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        display._pulse_cycle = 0.0  # ZeroDivisionError before any write

        self._run_ticks(monkeypatch, display, 2)

        assert dots.attempts == 0
        assert suppressed_counts().get("leds.animation_loop") == 2
        # Not a hardware verdict: no write was attempted, so none failed.
        assert display.is_available() is True

    def test_failed_first_write_is_not_announced_as_success(self, monkeypatch, capsys):
        from types import SimpleNamespace

        from anima_mcp import error_recovery
        from anima_mcp.display.leds import display as display_module

        error_recovery.reset_suppressed_counts()
        dots = _FlakyDots(fail=True)
        monkeypatch.setattr(display_module, "HAS_DOTSTAR", True)
        monkeypatch.setattr(display_module, "board", SimpleNamespace(D6=6, D5=5))
        monkeypatch.setattr(
            display_module,
            "adafruit_dotstar",
            SimpleNamespace(DotStar=lambda *args, **kwargs: dots),
        )
        monkeypatch.setattr(LEDDisplay, "_start_animation", lambda self: None)

        display = LEDDisplay(brightness=0.04)

        err = capsys.readouterr().err
        assert dots.attempts == 1
        assert "initialized successfully" not in err
        assert "first frame write failed" in err
        assert error_recovery.suppressed_counts().get("leds.write_frame") == 1
        # One failure is evidence, not a verdict; the animation loop decides.
        assert display.is_available() is True
        assert display.get_proprioceptive_state()["brightness"] is None

    def test_good_first_write_still_announces_success(self, monkeypatch, capsys):
        from types import SimpleNamespace

        from anima_mcp.display.leds import display as display_module

        dots = _FlakyDots(fail=False)
        monkeypatch.setattr(display_module, "HAS_DOTSTAR", True)
        monkeypatch.setattr(display_module, "board", SimpleNamespace(D6=6, D5=5))
        monkeypatch.setattr(
            display_module,
            "adafruit_dotstar",
            SimpleNamespace(DotStar=lambda *args, **kwargs: dots),
        )
        monkeypatch.setattr(LEDDisplay, "_start_animation", lambda self: None)

        display = LEDDisplay(brightness=0.04)

        assert "initialized successfully" in capsys.readouterr().err
        assert dots.colors == {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)}  # int(10 * 0.04)
        assert display.get_proprioceptive_state()["brightness"] == 0.04

    def test_health_probe_reports_dead_leds(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from anima_mcp import ctx_ref, health, lifecycle

        dots = _FlakyDots(fail=True)
        display, _ = self._display(monkeypatch, tmp_path, dots)
        self._run_ticks(monkeypatch, display, self._floor())

        registry = health.HealthRegistry()
        monkeypatch.setattr(health, "_registry", registry)
        monkeypatch.setattr(
            ctx_ref,
            "_ctx",
            SimpleNamespace(
                store=None, leds=display, display=None, growth=None,
                screen_renderer=None, voice_instance=None, last_shm_data=None,
            ),
        )
        lifecycle._register_health_probes()
        leds = registry.get_subsystem("leds")
        leds.debounce_seconds = 0.0
        registry.heartbeat("leds")  # the display loop heartbeats regardless

        assert leds.get_status() == "degraded"
