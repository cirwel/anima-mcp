"""LEDDisplay - hardware and orchestration."""

import math
import sys
import threading
import time
from typing import Optional, Tuple

from . import brightness as _brightness
from . import colors as _colors
from . import dances as _dances
from . import patterns as _patterns
from .colors import blend_colors
from .dances import Dance, DanceType, render_dance
from .types import LEDState

try:
    import board
    import adafruit_dotstar
    HAS_DOTSTAR = True
except ImportError:
    HAS_DOTSTAR = False
    board = None
    adafruit_dotstar = None


_led_display: Optional["LEDDisplay"] = None


def get_led_display() -> "LEDDisplay":
    """Get or create LED display singleton."""
    global _led_display
    if _led_display is None:
        _led_display = LEDDisplay()
    return _led_display


class LEDDisplay:
    """Controls BrainCraft HAT's 3 DotStar LEDs."""

    NUM_LEDS = 3

    def __init__(
        self,
        brightness: Optional[float] = None,
        enable_breathing: Optional[bool] = None,
        enable_patterns: Optional[bool] = None,
        expression_mode: str = "balanced",
    ):
        try:
            from ...config import get_display_config
            cfg = get_display_config()
            self._base_brightness = brightness or cfg.led_brightness
            self._enable_breathing = enable_breathing if enable_breathing is not None else cfg.breathing_enabled
            self._color_transitions_enabled = cfg.color_transitions_enabled
            self._pattern_mode = cfg.pattern_mode
            self._enable_patterns = enable_patterns if enable_patterns is not None else True
        except ImportError:
            self._base_brightness = brightness or 0.04
            self._enable_breathing = enable_breathing if enable_breathing is not None else True
            self._color_transitions_enabled = True
            self._pattern_mode = "standard"
            self._enable_patterns = True

        self._base_brightness = max(0.0, min(0.12, self._base_brightness))

        self._dots = None
        self._brightness = self._base_brightness
        self._hardware_brightness_floor = 0.008
        self._update_count = 0
        self._last_state: Optional[LEDState] = None
        self._last_applied_colors = [(0, 0, 0)] * self.NUM_LEDS
        self._last_wire_colors = [(0, 0, 0)] * self.NUM_LEDS
        self._last_colors = [None, None, None]
        self._last_light_level: Optional[float] = None
        self._last_state_values: Optional[Tuple[float, float, float, float]] = None
        self._pattern_active: Optional[str] = None
        self._pattern_start_time: float = 0.0
        self._last_anima_values: Optional[Tuple[float, float, float, float]] = None
        self._expression_mode = expression_mode
        self._known_brightness = self._base_brightness  # Proprioceptive: Lumen knows this
        self._cached_anima_state = None
        self._cached_light_level = None
        self._cached_manual_brightness = None
        self._cached_activity_brightness = None
        self._cached_pipeline_brightness = None
        self._cached_state_change_threshold = 0.05
        self._current_dance: Optional[Dance] = None
        self._dance_cooldown_until = 0.0
        self._last_dance_trigger: Optional[str] = None
        self._spontaneous_dance_chance = 0.005
        # Legacy name: the renderer supplies an absolute LED brightness preset
        # here (Full=.28, Medium=.12, Dim=.06, Night=.008), not a multiplier.
        self._manual_brightness_factor = 1.0
        # The agency learner's actuator: a relative factor in (0, 1] applied
        # AFTER the user's preset, never instead of it.  1.0 is the preset
        # itself, so the agency can dim below the user's choice and return
        # toward it, but can never make the LEDs brighter than the user chose.
        # Held as an integer count of dimming steps (factor = STEP ** steps)
        # so repeated steps cannot drift and "back at the preset" is exact.
        self._agency_dim_steps = 0
        self._last_activity_scale = 1.0
        self._current_brightness = 0.1
        self._target_brightness = self._base_brightness
        self._brightness_transition_speed = 0.08
        self._pulse_cycle = 12.0
        self._pulse_amount = 0.05
        self._flash_until = 0.0
        self._flash_color = (100, 100, 100)
        self._last_applied_brightness = self._base_brightness  # Actual brightness sent to hardware
        self._last_proprioception_publish = 0.0
        self._spi_lock = threading.Lock()
        self._animation_running = False
        self._animation_thread: Optional[threading.Thread] = None
        self._init_leds()

    def _init_leds(self):
        if not HAS_DOTSTAR:
            print("[LEDs] DotStar library not available", file=sys.stderr, flush=True)
            return
        try:
            self._dots = adafruit_dotstar.DotStar(
                board.D6, board.D5, self.NUM_LEDS,
                # Keep PixelBuf's DotStar header invariant.  Its dynamic
                # ``brightness`` setter scales the per-pixel start byte rather
                # than RGB bytes on this library version, producing malformed
                # frames.  Smooth brightness is applied explicitly to RGB in
                # _write_frame instead.
                brightness=1.0, auto_write=False
            )
            try:
                for i in range(3):
                    self._dots[i] = tuple(
                        int(channel * self._base_brightness)
                        for channel in (10, 10, 10)
                    )
                self._dots.show()
            except Exception:
                pass
            print("[LEDs] DotStar LEDs initialized successfully", file=sys.stderr, flush=True)
            self._start_animation()
        except Exception as e:
            print(f"[LEDs] Failed to initialize: {e}", file=sys.stderr, flush=True)
            self._dots = None

    def _start_animation(self):
        """Start background animation thread for smooth LED breathing."""
        if self._animation_thread is not None:
            return
        self._animation_running = True
        self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()
        print("[LEDs] Animation thread started (~5fps)", file=sys.stderr, flush=True)

    def _animation_loop(self):
        """Fast LED update loop for smooth per-LED wave + pulse."""
        while self._animation_running:
            try:
                state = self._last_state
                if self._dots and state and self._cached_pipeline_brightness is not None:
                    t = time.time()
                    # Flash override
                    if t < self._flash_until:
                        colors = [self._flash_color] * 3
                    else:
                        colors = [state.led2, state.led1, state.led0]
                    brightness = max(self._hardware_brightness_floor, self._current_brightness)
                    pulse = _brightness.get_pulse(self._pulse_cycle) * self._pulse_amount * min(1.0, max(0.15, brightness / 0.04))
                    pulse = min(pulse, max(0.005, brightness * 0.08))
                    perceptual = max(self._hardware_brightness_floor, min(0.5, brightness + pulse))
                    # RGB scaling: dim below base via RGB values (hardware brightness has a floor)
                    rgb_scale = min(1.0, brightness / self._base_brightness) if self._base_brightness > 0 else 1.0
                    with self._spi_lock:
                        applied_colors = []
                        for i, color in enumerate(colors):
                            phase = t * 2 * math.pi / self._pulse_cycle + i * math.pi * 2 / 3
                            mod = (0.92 + 0.08 * math.sin(phase)) * rgb_scale
                            applied_color = tuple(
                                max(0, min(255, int(c * mod))) for c in color
                            )
                            applied_colors.append(applied_color)
                        final_brightness = max(self._hardware_brightness_floor, perceptual)
                        self._last_wire_colors = self._write_frame(
                            applied_colors, final_brightness, lock_held=True
                        )
                    self._last_applied_colors = applied_colors
                    self._last_applied_brightness = final_brightness
                    # Publish close to the 200ms VEML integration window. The
                    # slower server display loop cannot timestamp the breathing
                    # phase precisely enough for causal system identification.
                    if t - self._last_proprioception_publish >= 0.5:
                        try:
                            from ...light_attribution import publish_led_proprioception

                            publish_led_proprioception(
                                self.get_proprioceptive_state(), captured_at=t
                            )
                            self._last_proprioception_publish = t
                        except Exception:
                            # Server-loop diagnostics retain the hardware state;
                            # a missing/stale action copy fails closed in broker.
                            pass
            except Exception:
                pass
            time.sleep(0.25)

    def _write_frame(
        self,
        applied_colors: list[Tuple[int, int, int]],
        brightness: float,
        *,
        lock_held: bool = False,
    ) -> list[Tuple[int, int, int]]:
        """Write one valid DotStar frame with brightness encoded in RGB bytes.

        ``adafruit_pixelbuf`` treats DotStar's fourth byte as a per-pixel start
        byte.  Changing ``PixelBuf.brightness`` dynamically rewrites that byte
        on the deployed library version, so it cannot be used as a smooth
        dimmer.  The controller instead keeps the header at full/valid and
        scales the actual channel bytes here.  The returned colors are exactly
        the bytes sent on the wire.
        """
        level = max(0.0, min(1.0, float(brightness)))
        wire_colors = [
            tuple(max(0, min(255, int(channel * level))) for channel in color)
            for color in applied_colors
        ]

        def write() -> None:
            for index, color in enumerate(wire_colors):
                self._dots[index] = color
            self._dots.show()

        if lock_held:
            write()
        else:
            with self._spi_lock:
                write()
        return wire_colors

    def _advance_current_brightness(self, *, immediate: bool = False) -> None:
        """Move the applied brightness toward the durable requested target."""
        target = max(
            self._hardware_brightness_floor,
            min(0.5, float(self._target_brightness)),
        )
        if immediate:
            self._current_brightness = target
        else:
            delta = target - self._current_brightness
            if abs(delta) <= 0.001:
                self._current_brightness = target
            else:
                speed = self._brightness_transition_speed
                if abs(delta) > 0.05:
                    speed = min(0.2, speed * 2)
                elif abs(delta) < 0.02:
                    speed *= 0.5
                self._current_brightness += delta * speed
        self._cached_pipeline_brightness = self._current_brightness
        self._known_brightness = self._current_brightness

    def is_available(self) -> bool:
        return self._dots is not None

    def set_brightness(self, brightness: float):
        self._base_brightness = max(0, min(0.12, brightness))
        self._brightness = self._base_brightness
        self._target_brightness = self._base_brightness

    # Multiplicative step for one agency adjustment.  Relative, so it means the
    # same thing under every preset; it is a step size, not a gate.
    AGENCY_BRIGHTNESS_STEP = 0.8

    @property
    def _agency_brightness_factor(self) -> float:
        """Relative agency dimmer in (0, 1]; 1.0 means exactly the preset."""
        return self.AGENCY_BRIGHTNESS_STEP ** getattr(self, "_agency_dim_steps", 0)

    def _requested_target(
        self,
        agency_factor: Optional[float] = None,
        activity_scale: Optional[float] = None,
    ) -> float:
        """Target brightness the pipeline requests: preset x agency x activity.

        The renderer value is an absolute preset.  A value of 1.0 is the legacy
        sentinel meaning "use the configured base".  Activity and the agency
        factor are independent multipliers in [0, 1] applied to every preset,
        so neither can push the target above the user's choice.
        """
        if self._manual_brightness_factor < 1.0:
            ceiling = self._manual_brightness_factor
        else:
            ceiling = self._base_brightness
        factor = self._agency_brightness_factor if agency_factor is None else agency_factor
        factor = max(0.0, min(1.0, float(factor)))
        scale = self._last_activity_scale if activity_scale is None else activity_scale
        scale = max(0.0, min(1.0, float(scale)))
        return max(
            self._hardware_brightness_floor,
            min(0.5, ceiling * factor * scale),
        )

    def adjust_agency_brightness(self, direction: str) -> dict:
        """Apply one agency step and report what the pipeline will now request.

        ``increase`` moves back toward the user's preset (factor capped at
        1.0); ``decrease`` dims below it.  A step that would not change the
        requested target -- already at the preset ceiling, or already at the
        hardware floor -- is not applied and is reported as unchanged, so the
        agency never learns a value for an action that did nothing physically.
        """
        if direction not in {"increase", "decrease"}:
            raise ValueError(f"invalid LED direction: {direction}")
        factor_before = self._agency_brightness_factor
        target_before = self._requested_target()
        steps_after = (
            max(0, self._agency_dim_steps - 1)
            if direction == "increase" else self._agency_dim_steps + 1
        )
        target_after = self._requested_target(
            agency_factor=self.AGENCY_BRIGHTNESS_STEP ** steps_after
        )
        changed = target_after != target_before
        if changed:
            self._agency_dim_steps = steps_after
            # Published immediately as a target change, so light attribution
            # sees an actuator step rather than a fixed-target sample.
            self._target_brightness = target_after
        return {
            "changed": changed,
            "direction": direction,
            "factor_before": factor_before,
            "factor_after": self._agency_brightness_factor,
            "target_before": target_before,
            "target_after": self._requested_target(),
            "preset_ceiling": self._requested_target(agency_factor=1.0, activity_scale=1.0),
        }

    def clear(self):
        if self._dots:
            try:
                self._dots.fill((0, 0, 0))
                self._dots.show()
                print("[LEDs] Cleared", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[LEDs] Error clearing LEDs: {e}", file=sys.stderr, flush=True)
                self._dots = None

    def set_led(self, index: int, color: Tuple[int, int, int]):
        if self._dots and 0 <= index < self.NUM_LEDS:
            try:
                self._dots[index] = color
                self._dots.show()
            except Exception as e:
                print(f"[LEDs] Error setting LED {index}: {e}", file=sys.stderr, flush=True)
                self._dots = None

    def _apply_flash(self, state: LEDState) -> LEDState:
        if time.time() < self._flash_until:
            fb = min(0.1, max(state.brightness * 2, 0.03))
            return LEDState(led0=self._flash_color, led1=self._flash_color, led2=self._flash_color, brightness=fb)
        return state

    def set_all(self, state: LEDState):
        if not self._dots:
            if self._update_count % 100 == 0:
                print("[LEDs] set_all called but LEDs unavailable", file=sys.stderr, flush=True)
            return
        state = self._apply_flash(state)
        with self._spi_lock:
            self._last_state = state
            self._update_count += 1

    def quick_flash(self, color: Tuple[int, int, int] = (100, 100, 100), duration_ms: int = 50):
        if self._dots:
            self._flash_until = time.time() + (duration_ms / 1000.0)
            self._flash_color = color

    def start_dance(self, dance_type: DanceType, duration: float = 2.0, intensity: float = 1.0) -> bool:
        now = time.time()
        if now < self._dance_cooldown_until:
            return False
        if self._current_dance and not self._current_dance.is_complete:
            return False
        self._current_dance = Dance(dance_type=dance_type, duration=duration, start_time=now, intensity=intensity)
        self._dance_cooldown_until = now + duration + 3.0
        self._last_dance_trigger = dance_type.value
        print(f"[LEDs] Starting dance: {dance_type.value}", file=sys.stderr, flush=True)
        return True

    def get_proprioceptive_state(self) -> dict:
        return {
            "brightness": self._last_applied_brightness,
            "target_brightness": self._target_brightness,
            "expression_mode": self._expression_mode,
            "is_dancing": self._current_dance is not None and not self._current_dance.is_complete,
            "is_flashing": time.time() < self._flash_until,
            "dance_type": self._current_dance.dance_type.value if self._current_dance and not self._current_dance.is_complete else None,
            "manual_dimmed": self._manual_brightness_factor < 1.0,
            "colors": [
                self._last_state.led0 if self._last_state else (0, 0, 0),
                self._last_state.led1 if self._last_state else (0, 0, 0),
                self._last_state.led2 if self._last_state else (0, 0, 0),
            ],
            # The animation loop applies activity/manual RGB scaling and a
            # per-LED breathing phase before writing.  These are the actual
            # channel values paired with ``brightness`` for optical drive.
            "applied_colors": list(self._last_applied_colors),
            "wire_colors": list(self._last_wire_colors),
        }

    def get_diagnostics(self) -> dict:
        dance_info = None
        if self._current_dance:
            dance_info = {
                "type": self._current_dance.dance_type.value,
                "progress": self._current_dance.progress,
                "remaining": self._current_dance.duration - self._current_dance.elapsed,
            }
        return {
            "available": self.is_available(),
            "has_dotstar": HAS_DOTSTAR,
            "update_count": self._update_count,
            "base_brightness": self._base_brightness,
            "current_brightness": self._current_brightness,
            "target_brightness": self._target_brightness,
            "agency_brightness_factor": self._agency_brightness_factor,
            "last_applied_brightness": self._last_applied_brightness,
            "last_wire_colors": list(self._last_wire_colors),
            "pulse_cycle": self._pulse_cycle,
            "pulse_amount": self._pulse_amount,
            "known_brightness": self._known_brightness,
            "color_transitions_enabled": self._color_transitions_enabled,
            "pattern_mode": self._pattern_mode,
            "last_light_level": self._last_light_level,
            "current_dance": dance_info,
            "last_dance": self._last_dance_trigger,
            "spontaneous_dance_chance": self._spontaneous_dance_chance,
            "last_state": {
                "led0": self._last_state.led0 if self._last_state else None,
                "led1": self._last_state.led1 if self._last_state else None,
                "led2": self._last_state.led2 if self._last_state else None,
            } if self._last_state else None,
        }

    def update_from_anima(
        self,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_level: Optional[float] = None,
        is_anticipating: bool = False,
        anticipation_confidence: float = 0.0,
        activity_brightness: float = 1.0,
    ) -> LEDState:
        update_start = time.time()
        # Always track light level for diagnostics, even on early-return paths.
        # Previously this was only set after the debounce check, so if the first
        # call had light_level=None (sensor not ready) it would stay None forever
        # because subsequent calls would early-return before reaching the setter.
        if light_level is not None:
            self._last_light_level = light_level
        state_changed = True
        if self._cached_anima_state is not None:
            lw, lc, ls, lp = self._cached_anima_state
            max_delta = max(abs(warmth - lw), abs(clarity - lc), abs(stability - ls), abs(presence - lp))
            light_changed = light_level is not None and self._cached_light_level is not None and abs(light_level - self._cached_light_level) > 50
            manual_changed = self._cached_manual_brightness is not None and abs(self._manual_brightness_factor - self._cached_manual_brightness) > 0.01
            activity_changed = (
                self._cached_activity_brightness is not None
                and abs(activity_brightness - self._cached_activity_brightness) > 0.01
            )
            if (
                max_delta < self._cached_state_change_threshold
                and not light_changed
                and not manual_changed
                and not activity_changed
            ):
                state_changed = False
                if self._last_state and self._cached_pipeline_brightness is not None:
                    self._advance_current_brightness()
                    return LEDState(
                        led0=self._last_state.led0,
                        led1=self._last_state.led1,
                        led2=self._last_state.led2,
                        brightness=self._current_brightness,
                    )

        _manual_just_changed = (
            self._cached_manual_brightness is not None
            and abs(self._manual_brightness_factor - self._cached_manual_brightness) > 0.01
        )
        self._cached_anima_state = (warmth, clarity, stability, presence)
        self._cached_light_level = light_level
        self._cached_manual_brightness = self._manual_brightness_factor
        self._cached_activity_brightness = activity_brightness

        self._last_state_values, pattern_trigger = _patterns.detect_state_change(
            warmth, clarity, stability, presence, self._last_state_values
        )
        if pattern_trigger:
            self._pattern_active = pattern_trigger
            self._pattern_start_time = time.time()

        state = _colors.derive_led_state(
            warmth, clarity, stability, presence,
            pattern_mode=self._pattern_mode,
            enable_color_mixing=self._color_transitions_enabled,
            expression_mode=self._expression_mode,
        )

        if self._pattern_active and self._enable_patterns:
            state, pattern_done = _patterns.get_pattern_colors(
                self._pattern_active, state, self._pattern_start_time
            )
            if pattern_done is None:
                self._pattern_active = None

        try:
            from ...eisv import get_trajectory_awareness
            dr, dg, db = _colors.get_shape_color_bias(get_trajectory_awareness().current_shape)
            if dr != 0 or dg != 0 or db != 0:
                state = LEDState(
                    led0=(max(0, min(255, state.led0[0] + dr)), max(0, min(255, state.led0[1] + dg)), max(0, min(255, state.led0[2] + db))),
                    led1=(max(0, min(255, state.led1[0] + dr)), max(0, min(255, state.led1[1] + dg)), max(0, min(255, state.led1[2] + db))),
                    led2=(max(0, min(255, state.led2[0] + dr)), max(0, min(255, state.led2[1] + dg)), max(0, min(255, state.led2[2] + db))),
                    brightness=state.brightness,
                )
        except Exception:
            pass

        state_change_detected = False
        if self._last_anima_values is not None:
            lw, lc, ls, lp = self._last_anima_values
            if abs(warmth - lw) > 0.15 or abs(clarity - lc) > 0.15 or abs(stability - ls) > 0.15 or abs(presence - lp) > 0.15:
                state_change_detected = True
        self._last_anima_values = (warmth, clarity, stability, presence)

        if self._color_transitions_enabled and self._last_state and self._last_colors[0] is not None:
            tf = 0.15 if state_change_detected else 0.10
            state = LEDState(
                led0=_colors.transition_color(self._last_colors[0], state.led0, tf, self._color_transitions_enabled),
                led1=_colors.transition_color(self._last_colors[1], state.led1, tf, self._color_transitions_enabled),
                led2=_colors.transition_color(self._last_colors[2], state.led2, tf, self._color_transitions_enabled),
                brightness=state.brightness,
            )
        self._last_colors = [state.led0, state.led1, state.led2]

        # Preset x agency factor x activity; see _requested_target.
        self._last_activity_scale = max(0.0, min(1.0, float(activity_brightness)))
        self._target_brightness = self._requested_target()
        self._advance_current_brightness(immediate=_manual_just_changed)
        state = LEDState(
            led0=state.led0,
            led1=state.led1,
            led2=state.led2,
            brightness=self._current_brightness,
        )

        if is_anticipating and anticipation_confidence > 0.1:
            memory_color = (255, 200, 100)
            blend_strength = min(0.25, anticipation_confidence * 0.3)
            state = LEDState(
                led0=blend_colors(state.led0, memory_color, blend_strength),
                led1=blend_colors(state.led1, memory_color, blend_strength),
                led2=blend_colors(state.led2, memory_color, blend_strength),
                brightness=state.brightness,
            )
            if anticipation_confidence > 0.5 and state.brightness >= 0.05:
                state = LEDState(led0=state.led0, led1=state.led1, led2=state.led2, brightness=min(0.12, state.brightness * (1.0 + (anticipation_confidence - 0.5) * 0.1)))

        if self._manual_brightness_factor >= 0.05:
            spontaneous = _dances.maybe_spontaneous_dance(
                warmth, clarity, stability, presence,
                self._spontaneous_dance_chance, self._dance_cooldown_until
            )
            if spontaneous:
                self.start_dance(spontaneous, duration=2.0, intensity=0.8)

        if self._current_dance:
            if self._current_dance.is_complete:
                self._current_dance = None
            else:
                state = render_dance(self._current_dance, state)
                if self._manual_brightness_factor < 0.05 and state.brightness > self._manual_brightness_factor * 1.1:
                    state = LEDState(led0=state.led0, led1=state.led1, led2=state.led2, brightness=self._manual_brightness_factor)

        try:
            self.set_all(state)
        except Exception as e:
            print(f"[LEDs] Error: {e}", file=sys.stderr, flush=True)
            if "hardware" in str(e).lower() or "io" in str(e).lower():
                self._dots = None

        if self._update_count % 10 == 1:
            features = []
            if not state_changed:
                features.append("cached")
            if self._current_dance and not self._current_dance.is_complete:
                features.append(f"dance({self._current_dance.dance_type.value})")
            fs = f" [{', '.join(features)}]" if features else ""
            print(f"[LEDs] #{self._update_count} ({1000*(time.time()-update_start):.1f}ms): w={warmth:.2f} c={clarity:.2f} s={stability:.2f} p={presence:.2f}{fs}", file=sys.stderr, flush=True)

        return state
