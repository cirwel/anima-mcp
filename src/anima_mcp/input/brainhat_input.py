"""
BrainCraft HAT Input - Joystick and Button Support

BrainCraft HAT has:
- Joystick (analog X/Y via Seesaw or GPIO)
- Button on joystick (press down)
- Separate button (GPIO pin)

This module provides unified input handling for all three.
"""

import sys
import time
from dataclasses import dataclass
from typing import Optional
from enum import Enum

# Try to import hardware libraries
try:
    import board
    import digitalio
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

try:
    from adafruit_seesaw.seesaw import Seesaw  # noqa: F401
    HAS_SEESAW = True
except ImportError:
    HAS_SEESAW = False


class JoystickDirection(Enum):
    """Joystick direction."""
    CENTER = "center"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


@dataclass
class InputState:
    """Complete input state from BrainCraft HAT."""
    # Joystick
    joystick_x: float = 0.0  # -1.0 to 1.0
    joystick_y: float = 0.0  # -1.0 to 1.0
    joystick_direction: JoystickDirection = JoystickDirection.CENTER
    joystick_button: bool = False  # Button on joystick (press down)
    
    # Separate button
    separate_button: bool = False
    
    timestamp: float = 0.0


class BrainHatInput:
    """Unified input handler for BrainCraft HAT joystick and buttons.

    BrainCraft HAT uses GPIO pins directly (not analog/Seesaw):
    - Joystick directions: D22 (left), D24 (right), D23 (up), D27 (down)
    - Joystick button: D16 (center press)
    - Separate button: D17
    """

    # Debounce settings (in seconds) - tuned for maximum responsiveness
    DEBOUNCE_TIME = 0.025  # 25ms debounce for directions (very snappy)
    BUTTON_DEBOUNCE_TIME = 0.05  # 50ms for buttons
    REPEAT_DELAY = 0.25  # 250ms before repeat starts
    REPEAT_RATE = 0.08  # 80ms between repeats (fast scrolling)

    # Display-pin heartbeat refresh. D22 (backlight) and D24 (ST7789 reset)
    # are shared with joystick left/right. Display init pulses them HIGH then
    # releases to pull-up; input reclaims as INPUT PULL_UP. If the pull-up
    # alone fails to hold a pin HIGH (MOSFET gate droop, transient low pulse,
    # EMI), the TFT backlight or controller can get stuck off until the next
    # broker restart. This heartbeat briefly re-asserts OUTPUT HIGH on both
    # pins, then restores INPUT PULL_UP — imperceptible to joystick polling.
    BACKLIGHT_REFRESH_INTERVAL = 30.0  # seconds between pulses
    BACKLIGHT_REFRESH_PULSE = 0.010  # 10ms active-high pulse

    def __init__(self):
        """Initialize input handler."""
        self._joy_left = None
        self._joy_right = None
        self._joy_up = None
        self._joy_down = None
        self._joy_button = None
        self._separate_button_pin = None
        self._available = False
        self._enabled = False
        self._deadzone = 0.15
        self._last_state: Optional[InputState] = None
        self._prev_state: Optional[InputState] = None  # Previous state for edge detection

        # Debounce state - track last stable state and when it changed
        self._last_direction = JoystickDirection.CENTER
        self._last_direction_time = 0.0
        self._last_joy_button = False
        self._last_joy_button_time = 0.0
        self._last_sep_button = False
        self._last_sep_button_time = 0.0

        # Hold/repeat state for directions
        self._direction_hold_start = 0.0
        self._last_repeat_time = 0.0

        # Last display-pin refresh (see BACKLIGHT_REFRESH_INTERVAL)
        self._last_backlight_refresh = 0.0
    
    def enable(self):
        """Explicitly enable input (call to activate)."""
        if self._enabled:
            return
        self._enabled = True
        self._init_hardware()
    
    def _init_hardware(self):
        """Initialize joystick and button hardware.

        BrainCraft HAT GPIO pin configuration:
        - Button: GPIO #17 (D17)
        - Joystick Select (Center Press): GPIO #16 (D16)
        - Joystick Left: GPIO #22 (D22) — released after display backlight init
        - Joystick Up: GPIO #23 (D23)
        - Joystick Right: GPIO #24 (D24) — released after display reset pulse
        - Joystick Down: GPIO #27 (D27)

        D22 and D24 are released by display after init — reclaim both.
        """
        if not HAS_GPIO:
            print("[BrainHatInput] GPIO library not available", file=sys.stderr, flush=True)
            return

        try:
            # D22 (left) released after display sets backlight HIGH — reclaim it
            self._joy_left = digitalio.DigitalInOut(board.D22)
            self._joy_left.direction = digitalio.Direction.INPUT
            self._joy_left.pull = digitalio.Pull.UP

            # D24 (right) released after display reset pulse — reclaim it
            self._joy_right = digitalio.DigitalInOut(board.D24)
            self._joy_right.direction = digitalio.Direction.INPUT
            self._joy_right.pull = digitalio.Pull.UP

            self._joy_up = digitalio.DigitalInOut(board.D23)
            self._joy_up.direction = digitalio.Direction.INPUT
            self._joy_up.pull = digitalio.Pull.UP

            self._joy_down = digitalio.DigitalInOut(board.D27)
            self._joy_down.direction = digitalio.Direction.INPUT
            self._joy_down.pull = digitalio.Pull.UP

            # Joystick button (center press)
            self._joy_button = digitalio.DigitalInOut(board.D16)
            self._joy_button.direction = digitalio.Direction.INPUT
            self._joy_button.pull = digitalio.Pull.UP

            # Separate button
            self._separate_button_pin = digitalio.DigitalInOut(board.D17)
            self._separate_button_pin.direction = digitalio.Direction.INPUT
            self._separate_button_pin.pull = digitalio.Pull.UP

            self._available = True
            print(f"[BrainHatInput] Initialized (GPIO, all directions - D22/D24 reclaimed) - debounce={self.DEBOUNCE_TIME*1000:.0f}ms btn={self.BUTTON_DEBOUNCE_TIME*1000:.0f}ms", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[BrainHatInput] Failed to initialize GPIO pins: {e}", file=sys.stderr, flush=True)
            self._available = False
    
    def is_available(self) -> bool:
        """Check if input hardware is available."""
        return self._enabled and self._available and (
            self._joy_up is not None or self._separate_button_pin is not None
        )

    def _refresh_display_pins(self) -> None:
        """Pulse D22/D24 OUTPUT HIGH briefly then restore INPUT PULL_UP.

        Defends against backlight droop and stuck-low conditions on pins
        shared between the display (backlight/reset) and joystick (L/R).
        See BACKLIGHT_REFRESH_INTERVAL for rationale.
        """
        if not HAS_GPIO:
            return
        if self._joy_left is None and self._joy_right is None:
            return
        try:
            if self._joy_left is not None:
                self._joy_left.deinit()
                self._joy_left = None
            if self._joy_right is not None:
                self._joy_right.deinit()
                self._joy_right = None

            d22 = digitalio.DigitalInOut(board.D22)
            d22.direction = digitalio.Direction.OUTPUT
            d22.value = True
            d24 = digitalio.DigitalInOut(board.D24)
            d24.direction = digitalio.Direction.OUTPUT
            d24.value = True
            time.sleep(self.BACKLIGHT_REFRESH_PULSE)
            d22.deinit()
            d24.deinit()

            self._joy_left = digitalio.DigitalInOut(board.D22)
            self._joy_left.direction = digitalio.Direction.INPUT
            self._joy_left.pull = digitalio.Pull.UP
            self._joy_right = digitalio.DigitalInOut(board.D24)
            self._joy_right.direction = digitalio.Direction.INPUT
            self._joy_right.pull = digitalio.Pull.UP
        except Exception as e:
            print(f"[BrainHatInput] Display-pin refresh failed: {e}", file=sys.stderr, flush=True)
    
    def read(self) -> Optional[InputState]:
        """Read current input state from GPIO pins with debouncing."""
        if not self.is_available():
            return None

        try:
            now = time.time()

            if now - self._last_backlight_refresh >= self.BACKLIGHT_REFRESH_INTERVAL:
                self._refresh_display_pins()
                self._last_backlight_refresh = now

            # Read raw GPIO states (pull-up: pressed = LOW = False)
            # D22 (left) and D24 (right) may be None if display holds them
            raw_left = not self._joy_left.value if self._joy_left else False
            raw_right = not self._joy_right.value if self._joy_right else False
            raw_up = not self._joy_up.value
            raw_down = not self._joy_down.value
            raw_joy_btn = not self._joy_button.value
            raw_sep_btn = not self._separate_button_pin.value

            # Convert to analog-like values
            joystick_x = 0.0
            joystick_y = 0.0
            if raw_left:
                joystick_x = -1.0
            elif raw_right:
                joystick_x = 1.0
            if raw_up:
                joystick_y = 1.0
            elif raw_down:
                joystick_y = -1.0

            # Determine raw direction
            raw_direction = self._get_direction(joystick_x, joystick_y)

            # === Debounce direction ===
            # Only accept direction change if it's been stable for DEBOUNCE_TIME
            if raw_direction != self._last_direction:
                if now - self._last_direction_time >= self.DEBOUNCE_TIME:
                    self._last_direction = raw_direction
                    self._last_direction_time = now
                    self._direction_hold_start = now  # Reset hold timer
                    self._last_repeat_time = 0.0
                # else: ignore - likely bounce
            else:
                # Same direction - update time for stability tracking
                self._last_direction_time = now

            # Use debounced direction
            direction = self._last_direction

            # === Debounce joystick button ===
            if raw_joy_btn != self._last_joy_button:
                if now - self._last_joy_button_time >= self.BUTTON_DEBOUNCE_TIME:
                    self._last_joy_button = raw_joy_btn
                    self._last_joy_button_time = now
            joystick_button = self._last_joy_button

            # === Debounce separate button ===
            if raw_sep_btn != self._last_sep_button:
                if now - self._last_sep_button_time >= self.BUTTON_DEBOUNCE_TIME:
                    self._last_sep_button = raw_sep_btn
                    self._last_sep_button_time = now
            separate_button = self._last_sep_button

            state = InputState(
                joystick_x=joystick_x,
                joystick_y=joystick_y,
                joystick_direction=direction,
                joystick_button=joystick_button,
                separate_button=separate_button,
                timestamp=now
            )

            # Store previous state BEFORE updating (for edge detection)
            self._prev_state = self._last_state
            self._last_state = state
            return state

        except Exception as e:
            print(f"[BrainHatInput] Read error: {e}", file=sys.stderr, flush=True)
            return None
    
    def get_prev_state(self) -> Optional[InputState]:
        """Get previous state (before last read) for edge detection."""
        return self._prev_state
    
    def _get_direction(self, x: float, y: float) -> JoystickDirection:
        """Determine joystick direction from x, y values."""
        # Use original deadzone - was too strict
        deadzone = 0.15
        
        if abs(x) < deadzone and abs(y) < deadzone:
            return JoystickDirection.CENTER
        
        abs_x = abs(x)
        abs_y = abs(y)
        
        # Prefer horizontal for screen switching, but don't be too strict
        if abs_x > abs_y * 1.2:  # Lower threshold - more forgiving
            return JoystickDirection.LEFT if x < 0 else JoystickDirection.RIGHT
        elif abs_y > abs_x * 1.2:
            return JoystickDirection.DOWN if y < 0 else JoystickDirection.UP
        else:
            # Diagonal - prefer horizontal for screen switching
            if abs_x >= abs_y:
                return JoystickDirection.LEFT if x < 0 else JoystickDirection.RIGHT
            else:
                return JoystickDirection.DOWN if y < 0 else JoystickDirection.UP
    
    def get_last_state(self) -> Optional[InputState]:
        """Get last read state (cached)."""
        return self._last_state


# Singleton instance
_brainhat_input: Optional[BrainHatInput] = None


def get_brainhat_input() -> BrainHatInput:
    """Get or create BrainHat input handler (disabled by default)."""
    global _brainhat_input
    if _brainhat_input is None:
        _brainhat_input = BrainHatInput()
    return _brainhat_input
