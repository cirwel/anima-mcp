"""
Display Renderer - Interface to BrainCraft HAT TFT.

240x240 pixel display for the creature's face.
Falls back to image generation (no display) on Mac.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import json
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .face import FaceState, EyeState, MouthState
from .design import Timing, radial_gradient_color


# Display dimensions (BrainCraft HAT)
WIDTH = 240
HEIGHT = 240

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


@dataclass
class DisplayConfig:
    """Display configuration."""
    width: int = WIDTH
    height: int = HEIGHT
    rotation: int = 180  # BrainCraft HAT default
    fps: int = 10


class DisplayRenderer(ABC):
    """Abstract display renderer."""

    @abstractmethod
    def render_face(self, state: FaceState, name: Optional[str] = None) -> None:
        """Render face state to display."""
        pass

    @abstractmethod
    def render_text(self, text: str, position: Tuple[int, int] = (10, 10)) -> None:
        """Render text to display."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the display."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if display hardware is available."""
        pass

    @abstractmethod
    def show_default(self) -> None:
        """Show minimal default screen (non-grey, non-distracting)."""
        pass


class PilRenderer(DisplayRenderer):
    """
    PIL-based renderer.

    Generates images that can be:
    - Displayed on BrainCraft HAT TFT (on Pi)
    - Saved to file (for debugging)
    - Shown in terminal as ASCII (fallback)
    """

    def __init__(self, config: Optional[DisplayConfig] = None):
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for display rendering")

        self.config = config or DisplayConfig()
        self._display = None
        self._cs_pin = None
        self._dc_pin = None
        # D22 backlight released after init — no longer held
        self._init_error: Optional[str] = None  # Last init failure reason
        self._image: Optional[Image.Image] = None
        self._last_face_state: Optional[FaceState] = None
        self._last_blink_time: float = 0.0
        self._blink_in_progress: bool = False
        self._blink_start_time: float = 0.0
        self._deferred: bool = False  # When True, _show() skips SPI push (caller must call flush())
        # Font cache (avoid loading from disk on every render)
        self._name_font: Optional[ImageFont.FreeTypeFont] = None
        # Brightness cache (avoid re-dimming unchanged images)
        self._cached_dimmed_image: Optional[Image.Image] = None
        self._cached_source_id: Optional[int] = None  # id(image) when dimmed
        self._cached_brightness: float = 1.0
        # Manual brightness control (user-adjustable via joystick on face screen)
        # Screen always stays full brightness — only LEDs dim.
        # LED brightness presets - wider spread for noticeable difference between modes
        # Perceived brightness is logarithmic; Night must be genuinely dim (bedroom-safe)
        self._brightness_presets = [
            {"name": "Full",   "display": 1.0,  "leds": 0.28, "absolute": True},   # Bright
            {"name": "Medium", "display": 1.0,  "leds": 0.12, "absolute": True},   # Moderate
            {"name": "Dim",    "display": 1.0,  "leds": 0.06, "absolute": True},   # Dim
            {"name": "Night",  "display": 1.0,  "leds": 0.008, "absolute": True},  # Minimal - barely visible, bedroom-safe
        ]
        self._brightness_index: int = 0  # Index into presets
        self._manual_brightness: float = 1.0  # Display multiplier
        # Default Medium (0.12) - manual brightness for LEDs (auto-brightness removed)
        self._manual_led_brightness: float = 0.12
        self._brightness_config_path = Path.home() / ".anima" / "display_brightness.json"
        self._load_brightness()
        self._display_fail_count: int = 0
        self._last_reinit_attempt: float = 0.0
        self._init_display()

    def _load_brightness(self):
        """Load saved brightness preset from disk. Defaults to Medium if missing."""
        try:
            if self._brightness_config_path.exists():
                data = json.loads(self._brightness_config_path.read_text())
                name = data.get("name", "Full")
                for i, preset in enumerate(self._brightness_presets):
                    if preset["name"] == name:
                        self._brightness_index = i
                        self._manual_brightness = preset["display"]
                        self._manual_led_brightness = preset["leds"]
                        print(f"[Display] Loaded brightness: {name}", file=sys.stderr, flush=True)
                        return
            # No config or not found — use Medium (index 1) so we never fall through to auto-brightness chaos
            self._brightness_index = 1
            self._manual_brightness = self._brightness_presets[1]["display"]
            self._manual_led_brightness = self._brightness_presets[1]["leds"]
        except Exception as e:
            print(f"[Display] Could not load brightness: {e}", file=sys.stderr, flush=True)

    def _save_brightness(self):
        """Save current brightness preset to disk."""
        try:
            self._brightness_config_path.parent.mkdir(parents=True, exist_ok=True)
            preset = self._brightness_presets[self._brightness_index]
            self._brightness_config_path.write_text(json.dumps({
                "name": preset["name"],
                "display": preset["display"],
                "leds": preset["leds"],
            }))
        except Exception as e:
            print(f"[Display] Could not save brightness: {e}", file=sys.stderr, flush=True)

    def brightness_up(self) -> str:
        """Cycle to next brighter preset. Returns preset name."""
        if self._brightness_index > 0:
            self._brightness_index -= 1
        preset = self._brightness_presets[self._brightness_index]
        self._manual_brightness = preset["display"]
        self._manual_led_brightness = preset["leds"]
        self._cached_dimmed_image = None  # Invalidate cache on brightness change
        self._save_brightness()
        print(f"[Display] Brightness: {preset['name']} (display={preset['display']}, leds={preset['leds']})", file=sys.stderr, flush=True)
        return preset["name"]

    def brightness_down(self) -> str:
        """Cycle to next dimmer preset. Returns preset name."""
        if self._brightness_index < len(self._brightness_presets) - 1:
            self._brightness_index += 1
        preset = self._brightness_presets[self._brightness_index]
        self._manual_brightness = preset["display"]
        self._manual_led_brightness = preset["leds"]
        self._cached_dimmed_image = None  # Invalidate cache on brightness change
        self._save_brightness()
        print(f"[Display] Brightness: {preset['name']} (display={preset['display']}, leds={preset['leds']})", file=sys.stderr, flush=True)
        return preset["name"]

    def get_brightness_preset(self) -> dict:
        """Get current brightness preset info."""
        return self._brightness_presets[self._brightness_index]

    def _get_name_font(self) -> ImageFont.FreeTypeFont:
        """Get cached font for name display."""
        if self._name_font is None:
            try:
                self._name_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                self._name_font = ImageFont.load_default()
        return self._name_font

    def _cleanup_display(self):
        """Release display GPIO pins so _init_display() can re-acquire them."""
        for pin_attr in ("_cs_pin", "_dc_pin"):
            pin = getattr(self, pin_attr, None)
            if pin is not None:
                try:
                    pin.deinit()
                except Exception:
                    pass
                setattr(self, pin_attr, None)
        self._display = None

    def _init_display(self):
        """Initialize display hardware if available."""
        # Release any previously held pins before re-init
        self._cleanup_display()

        try:
            import board
            import digitalio
            from adafruit_rgb_display import st7789

            # BrainCraft HAT pin configuration — store refs for cleanup
            # CE0=CS, D25=DC
            # D22=backlight, D24=reset — both set HIGH then released for joystick
            self._cs_pin = digitalio.DigitalInOut(board.CE0)
            self._dc_pin = digitalio.DigitalInOut(board.D25)

            # Backlight on D22 — set HIGH then release
            # BrainCraft HAT has hardware pull-up on backlight, keeps it HIGH
            _bl = digitalio.DigitalInOut(board.D22)
            _bl.direction = digitalio.Direction.OUTPUT
            _bl.value = True
            import time as _time
            _time.sleep(0.05)
            _bl.deinit()  # Release D22 — hardware pull-up holds it HIGH

            # Manual hardware reset via D24 — pulse LOW then release pin
            # ST7789 needs a reset pulse to enter a known state
            _rst = digitalio.DigitalInOut(board.D24)
            _rst.direction = digitalio.Direction.OUTPUT
            _rst.value = True
            _time.sleep(0.05)
            _rst.value = False
            _time.sleep(0.05)
            _rst.value = True
            _time.sleep(0.15)  # Wait for ST7789 to boot after reset
            _rst.deinit()  # Release D24 — not needed after reset pulse

            # SPI setup - use high speed for fast display updates
            spi = board.SPI()

            self._display = st7789.ST7789(
                spi,
                height=self.config.height,
                width=self.config.width,
                y_offset=80,
                rotation=self.config.rotation,
                cs=self._cs_pin,
                dc=self._dc_pin,
                rst=None,  # Already reset manually above
                baudrate=24000000,  # 24 MHz - max SPI speed for ST7789
            )

            print("BrainCraft HAT display initialized", file=sys.stderr, flush=True)

            # Validate readback. Some TFT carriers (including the BrainCraft
            # HAT) don't wire the display's MISO back to the Pi, so every
            # .read() returns zeros regardless of chip state. If RDDID is
            # all zeros, probe-based recovery would fire on every heartbeat
            # against phantom data. Verified empirically 2026-04-24: this
            # HAT returns 0 from RDDID/RDDST/RDDPM/RDDMADCTL/RDDCOLMOD.
            self._probe_supported = True
            try:
                rddid = self._display.read(0x04, 4)  # chip manufacturer ID
                if not rddid or all(b == 0 for b in rddid):
                    self._probe_supported = False
                    print("[Display] SPI readback not functional (RDDID=0) — probe-based chip-state recovery disabled", file=sys.stderr, flush=True)
            except Exception as e:
                self._probe_supported = False
                print(f"[Display] SPI readback raised ({e}) — probe-based chip-state recovery disabled", file=sys.stderr, flush=True)

            # Immediately blank the display to clear any ST7789 framebuffer noise,
            # then show the waking face. This prevents scrambled pixels on startup.
            try:
                self.blank()
                self._show_waking_face()
                print("Display showing waking face", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Warning: Could not show waking face: {e}", file=sys.stderr, flush=True)
                # Continue - display initialized but waking face failed
        except Exception as e:
            import traceback
            self._init_error = f"{type(e).__name__}: {e}"
            print(f"No display hardware: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._display = None
            # D22 backlight released after init — no longer held

    def _show_waking_face(self):
        """Show minimal default screen - subtle border, non-distracting. Safe, never crashes."""
        try:
            image = Image.new("RGB", (self.config.width, self.config.height), BLACK)
            draw = ImageDraw.Draw(image)
        except Exception as e:
            print(f"[Display] Error creating waking face image: {e}", file=sys.stderr)
            return

        # Subtle border - dark enough to be minimal, visible enough to confirm it's not grey
        # Dark blue-grey, minimal but clear
        border_color = (25, 35, 45)  # Dark but visible
        border_width = 2
        
        # Draw a thin border around the edge
        # Top
        draw.rectangle([0, 0, self.config.width, border_width], fill=border_color)
        # Bottom
        draw.rectangle([0, self.config.height - border_width, self.config.width, self.config.height], fill=border_color)
        # Left
        draw.rectangle([0, 0, border_width, self.config.height], fill=border_color)
        # Right
        draw.rectangle([self.config.width - border_width, 0, self.config.width, self.config.height], fill=border_color)

        try:
            self._image = image
            if self._display:
                self._display.image(self._image)
        except Exception as e:
            print(f"[Display] Error showing waking face: {e}", file=sys.stderr)
            # Don't crash - display might be temporarily unavailable

    # ST7789 proprioception — see docs/operations/DISPLAY_PROPRIOCEPTION.md
    # D24 is shared with joystick RIGHT; a droop on D24 can reset the chip
    # into sleep/display-off state. SPI writes continue to succeed silently
    # from that point, so the software never notices. These three methods
    # give the renderer a way to observe the chip and recover without a
    # service restart.
    SLPOUT_CMD = 0x11   # Sleep Out
    DISPON_CMD = 0x29   # Display On
    RDDPM_CMD = 0x0A    # Read Display Power Mode
    RDDPM_SLPOUT_BIT = 1 << 4   # 1 = awake
    RDDPM_DISON_BIT = 1 << 2    # 1 = display on

    def wake_chip(self) -> None:
        """Idempotent wake: send SLPOUT + DISPON.

        Safe to call regardless of current chip state — awake chip is
        unaffected. Used by verify_and_recover() and the brainhat heartbeat
        callback. Swallows exceptions; this is a best-effort release valve.
        """
        if not self._display:
            return
        try:
            self._display.write(self.SLPOUT_CMD, b"")
            self._display.write(self.DISPON_CMD, b"")
        except Exception as e:
            print(f"[Display] wake_chip failed: {e}", file=sys.stderr, flush=True)

    def probe_chip_state(self) -> Optional[dict]:
        """Read RDDPM (0x0A) and decode sleep + display-on bits.

        Returns ``{"sleep_out": bool, "display_on": bool, "raw": int}`` or
        None if the display handle is missing or SPI read fails.
        """
        if not self._display:
            return None
        try:
            data = self._display.read(self.RDDPM_CMD, 1)
        except Exception as e:
            print(f"[Display] probe_chip_state read failed: {e}", file=sys.stderr, flush=True)
            return None
        if not data:
            return None
        raw = data[-1]  # last byte — some SPI paths prepend a dummy
        return {
            "sleep_out": bool(raw & self.RDDPM_SLPOUT_BIT),
            "display_on": bool(raw & self.RDDPM_DISON_BIT),
            "raw": raw,
        }

    def verify_and_recover(self) -> dict:
        """Probe chip state; on a non-fully-awake reading, full re-init.

        Self-disabling: if the hardware doesn't wire MISO (see
        ``_probe_supported`` check at init), the probe would return
        phantom zeros on every read and drive recovery every heartbeat.
        On such hardware this returns ``unsupported`` and fires nothing.

        Recovery is a full ``_init_display()``, not just SLPOUT+DISPON.
        A hardware reset wipes ``COLMOD`` (pixel format) and ``MADCTL``
        (rotation / color order); resending only the wake commands leaves
        the library's state (RGB565, rotation=180) mismatched against
        the chip's post-reset defaults — pixels flow but render garbled.
        """
        if not getattr(self, "_probe_supported", True):
            return {"probed": False, "recovered": False, "reason": "readback unsupported on this hardware"}
        state = self.probe_chip_state()
        fully_awake = (
            state is not None
            and state["sleep_out"]
            and state["display_on"]
            and state["raw"] != 0  # raw=0 means all status bits cleared
        )
        if fully_awake:
            return {"probed": True, "recovered": False, "state": state}
        self._init_display()
        return {"probed": state is not None, "recovered": True, "state": state}

    def _create_canvas(self, background: Tuple[int, int, int] = BLACK) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        """Create a new canvas for drawing."""
        image = Image.new("RGB", (self.config.width, self.config.height), background)
        draw = ImageDraw.Draw(image)
        return image, draw

    def render_face(self, state: FaceState, name: Optional[str] = None) -> None:
        """Render face to display with micro-expressions and transitions. Safe, never crashes."""
        if not self._display:
            # Attempt recovery via _push_to_display (which has reinit logic)
            self._push_to_display()
            if not self._display:
                return  # Still unavailable after recovery attempt

        try:
            import time
            t0 = time.time()

            image, draw = self._create_canvas(BLACK)

            # Face background circle with tint
            center_x, center_y = self.config.width // 2, self.config.height // 2 - 20
            face_radius = 90

            # Smooth color transition for tint
            if self._last_face_state:
                # Transition tint smoothly — gentle drift, not snap
                transition_factor = Timing.FACE_TINT_FACTOR
                current_tint = state.tint
                last_tint = self._last_face_state.tint
                smooth_tint = tuple(
                    int(last_tint[i] + (current_tint[i] - last_tint[i]) * transition_factor)
                    for i in range(3)
                )
                state.tint = smooth_tint

            # Draw face background with radial gradient
            # Creates depth and warmth - the face glows from within
            self._draw_gradient_face(draw, center_x, center_y, face_radius, state.tint)

            t1 = time.time()

            # Handle blinking animation
            now = time.time()
            time_since_last_blink = now - self._last_blink_time

            # Check if should blink
            if not self._blink_in_progress and time_since_last_blink >= state.blink_frequency:
                self._blink_in_progress = True
                self._blink_start_time = now
                self._last_blink_time = now

            # Apply blink if in progress
            blink_state = state
            if self._blink_in_progress:
                elapsed = now - self._blink_start_time
                if elapsed < state.blink_duration:
                    # During blink: reduce eye openness
                    blink_progress = elapsed / state.blink_duration
                    # Smooth blink curve (ease in/out)
                    blink_curve = 0.5 - 0.5 * math.cos(blink_progress * math.pi)
                    effective_openness = state.eye_openness * (1 - blink_curve * (1 - state.blink_intensity))
                    # Create modified state for blink
                    blink_state = FaceState(
                        eyes=state.eyes,
                        mouth=state.mouth,
                        tint=state.tint,
                        eye_openness=effective_openness,
                        blinking=True,
                        eyebrow_raise=state.eyebrow_raise,
                        blink_frequency=state.blink_frequency,
                        blink_duration=state.blink_duration,
                        blink_intensity=state.blink_intensity,
                    )
                else:
                    # Blink complete
                    self._blink_in_progress = False

            # Draw eyes (with blink applied)
            self._draw_eyes(draw, blink_state, center_x, center_y)
            t2 = time.time()

            # Draw mouth
            self._draw_mouth(draw, state, center_x, center_y)
            t3 = time.time()

            # Store state for next transition
            self._last_face_state = state

            # Draw name at bottom if provided
            if name:
                font = self._get_name_font()
                bbox = draw.textbbox((0, 0), name, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text(
                    ((self.config.width - text_width) // 2, self.config.height - 30),
                    name,
                    fill=WHITE,
                    font=font
                )
            t4 = time.time()

            self._image = image
            self._show()
            t5 = time.time()

            # Log timing breakdown for slow renders
            total = t5 - t0
            if total > 0.3:  # Log renders over 300ms
                print(f"[Face] draw={int((t1-t0)*1000)}ms eyes={int((t2-t1)*1000)}ms mouth={int((t3-t2)*1000)}ms text={int((t4-t3)*1000)}ms show={int((t5-t4)*1000)}ms TOTAL={int(total*1000)}ms", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Display] Error rendering face: {e}", file=sys.stderr, flush=True)
            # Don't print full traceback in production - too verbose
            # Don't crash - display might be temporarily unavailable

    def _draw_gradient_face(self, draw: ImageDraw.ImageDraw, cx: int, cy: int,
                             radius: int, tint: Tuple[int, int, int]):
        """Draw face with radial gradient - glowing from within.

        Includes subtle breathing animation (+/- 2px) tied to wall clock.
        """
        # Breathing: subtle radius modulation (0.4 Hz = 2.5s cycle)
        import time
        breath = 2 * math.sin(time.time() * 0.4 * 2 * math.pi)
        radius = radius + int(breath)

        # Center color: brighter version of tint
        center_color = tuple(min(255, c // 6 + 20) for c in tint)

        # Edge color: darker, fades to background
        edge_color = tuple(c // 20 for c in tint)

        # Draw gradient using concentric circles (efficient for small display)
        # Use fewer rings with thicker bands for performance
        num_rings = 12  # Fewer rings = faster, still looks smooth on 240px
        for i in range(num_rings, 0, -1):
            t = i / num_rings
            r = int(radius * t + 4)  # +4 for soft edge
            color = radial_gradient_color(center_color, edge_color, 1 - t, 1.0)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # Subtle rim highlight (top edge catches light)
        rim_color = tuple(min(255, c // 4 + 30) for c in tint)
        draw.arc([cx - radius + 2, cy - radius + 2, cx + radius - 2, cy + radius - 2],
                 200, 340, fill=rim_color, width=1)

    def _draw_eyes(self, draw: ImageDraw.ImageDraw, state: FaceState, cx: int, cy: int):
        """Draw eyes based on state - fluid, expressive, and alive."""
        eye_spacing = 35
        eye_y = cy - 15

        left_x = cx - eye_spacing
        right_x = cx + eye_spacing

        eye_color = state.tint

        # Softer highlight color (warm white, not harsh)
        highlight_color = (255, 252, 245)  # Warm white

        # Apply eyebrow raise (affects eye position)
        eyebrow_offset = int(state.eyebrow_raise * 5)  # -5 to +5 pixels
        eye_y += eyebrow_offset

        # Base eye size from state, then apply fluid size factor
        if state.eyes == EyeState.WIDE:
            base_r = 18
        elif state.eyes == EyeState.NORMAL:
            base_r = 14
        elif state.eyes == EyeState.DROOPY:
            base_r = 14
        elif state.eyes == EyeState.SQUINT:
            base_r = 8  # Smaller for squint
        else:  # CLOSED
            base_r = 0

        # Apply fluid size factor (continuous variation)
        r = int(base_r * state.eye_size_factor)
        r = max(4, min(25, r))  # Clamp to reasonable range

        # Subtle asymmetry - right eye slightly different (feels more alive)
        right_r = int(r * 0.95)  # Right eye 5% smaller
        right_y_offset = 1  # Right eye 1px lower

        # Apply eye openness (continuous, not discrete)
        if state.eyes == EyeState.CLOSED:
            # Closed eyes - curved arcs (gentle, not flat)
            draw.arc([left_x - 12, eye_y - 4, left_x + 12, eye_y + 8], 0, 180, fill=eye_color, width=3)
            draw.arc([right_x - 12, eye_y - 4 + right_y_offset, right_x + 12, eye_y + 8 + right_y_offset], 0, 180, fill=eye_color, width=3)
        elif state.eyes == EyeState.SQUINT:
            # Squinting - narrow ovals, not harsh lines
            line_width = max(2, int(4 * state.eye_openness))
            h = max(2, line_width // 2)
            draw.ellipse([left_x - 12, eye_y - h, left_x + 12, eye_y + h], fill=eye_color)
            draw.ellipse([right_x - 12, eye_y - h + right_y_offset, right_x + 12, eye_y + h + right_y_offset], fill=eye_color)
        elif state.eyes == EyeState.DROOPY:
            # Droopy - soft ovals with fluid openness
            h = max(4, int(r * state.eye_openness))
            # Left eye
            draw.ellipse([left_x - r, eye_y - h, left_x + r, eye_y + h], fill=eye_color)
            # Right eye (slightly different)
            draw.ellipse([right_x - right_r, eye_y - h + right_y_offset, right_x + right_r, eye_y + h + right_y_offset], fill=eye_color)
            # Pupils (smaller for droopy)
            pr = max(2, int(4 * state.eye_openness))
            draw.ellipse([left_x - pr, eye_y - pr, left_x + pr, eye_y + pr], fill=BLACK)
            draw.ellipse([right_x - pr, eye_y - pr + right_y_offset, right_x + pr, eye_y + pr + right_y_offset], fill=BLACK)
            # Subtle highlights (make eyes alive)
            if pr > 3:
                hl_r = max(1, pr // 3)
                draw.ellipse([left_x - pr + 2, eye_y - pr + 2, left_x - pr + 2 + hl_r * 2, eye_y - pr + 2 + hl_r * 2], fill=highlight_color)
                draw.ellipse([right_x - pr + 2, eye_y - pr + 2 + right_y_offset, right_x - pr + 2 + hl_r * 2, eye_y - pr + 2 + hl_r * 2 + right_y_offset], fill=highlight_color)
        else:
            # WIDE or NORMAL - soft circles with fluid size and openness
            # Apply openness to create more variation
            actual_r = max(4, int(r * (0.7 + state.eye_openness * 0.3)))
            actual_r_right = max(4, int(right_r * (0.7 + state.eye_openness * 0.3)))

            # Draw eyes (slightly oval - more natural)
            oval_factor = 1.1  # Slightly taller than wide
            draw.ellipse([left_x - actual_r, eye_y - int(actual_r * oval_factor),
                         left_x + actual_r, eye_y + int(actual_r * oval_factor)], fill=eye_color)
            draw.ellipse([right_x - actual_r_right, eye_y - int(actual_r_right * oval_factor) + right_y_offset,
                         right_x + actual_r_right, eye_y + int(actual_r_right * oval_factor) + right_y_offset], fill=eye_color)

            # Pupils - size varies with openness
            pr = max(3, int(6 * state.eye_openness))
            pr = min(pr, actual_r - 2)  # Don't exceed eye size
            pr_right = min(pr, actual_r_right - 2)
            draw.ellipse([left_x - pr, eye_y - pr, left_x + pr, eye_y + pr], fill=BLACK)
            draw.ellipse([right_x - pr_right, eye_y - pr_right + right_y_offset, right_x + pr_right, eye_y + pr_right + right_y_offset], fill=BLACK)

            # Eye highlights - the sparkle that makes eyes alive
            # Upper-left highlight (simulates light source)
            if pr > 3 and state.eye_openness > 0.5:
                hl_r = max(1, pr // 3)
                hl_offset = pr // 2
                # Left eye highlight
                draw.ellipse([left_x - hl_offset - hl_r, eye_y - hl_offset - hl_r,
                             left_x - hl_offset + hl_r, eye_y - hl_offset + hl_r], fill=highlight_color)
                # Right eye highlight
                draw.ellipse([right_x - hl_offset - hl_r, eye_y - hl_offset - hl_r + right_y_offset,
                             right_x - hl_offset + hl_r, eye_y - hl_offset + hl_r + right_y_offset], fill=highlight_color)

    def _draw_mouth(self, draw: ImageDraw.ImageDraw, state: FaceState, cx: int, cy: int):
        """Draw mouth based on state - fluid with smile intensity."""
        mouth_y = cy + 35
        mouth_color = state.tint
        
        # Apply mouth width factor (fluid variation)
        base_width = 20
        mouth_width = int(base_width * state.mouth_width_factor)
        mouth_width = max(12, min(30, mouth_width))  # Clamp to reasonable range
        
        # Use smile_intensity for fluid expression (can blend states)
        smile_intensity = getattr(state, 'smile_intensity', 0.0)
        
        if state.mouth == MouthState.OPEN:
            # Open oval - size varies with expression intensity
            open_size = int(8 + state.expression_intensity * 4)
            draw.ellipse([cx - open_size, mouth_y - open_size, cx + open_size, mouth_y + open_size], fill=mouth_color)
        elif state.mouth == MouthState.FLAT:
            # Small flat line - very minimal
            draw.line([cx - int(mouth_width * 0.7), mouth_y, cx + int(mouth_width * 0.7), mouth_y], fill=mouth_color, width=2)
        elif smile_intensity > 0.3:
            # Positive expression - smile (can be subtle or strong)
            # Arc height varies with smile_intensity
            arc_height = int(10 + smile_intensity * 10)  # 10-20 range
            draw.arc([cx - mouth_width, mouth_y - arc_height, cx + mouth_width, mouth_y + arc_height], 
                    0, 180, fill=mouth_color, width=3)
        elif smile_intensity < -0.2:
            # Negative expression - frown
            arc_height = int(5 + abs(smile_intensity) * 10)
            draw.arc([cx - mouth_width, mouth_y - 5, cx + mouth_width, mouth_y + arc_height], 
                    180, 360, fill=mouth_color, width=3)
        elif state.mouth == MouthState.SMILE:
            # Explicit smile state
            arc_height = int(12 + smile_intensity * 8) if smile_intensity > 0 else 12
            draw.arc([cx - mouth_width, mouth_y - arc_height, cx + mouth_width, mouth_y + arc_height], 
                    0, 180, fill=mouth_color, width=3)
        elif state.mouth == MouthState.FROWN:
            # Explicit frown state
            arc_height = int(5 + abs(smile_intensity) * 10) if smile_intensity < 0 else 10
            draw.arc([cx - mouth_width, mouth_y - 5, cx + mouth_width, mouth_y + arc_height], 
                    180, 360, fill=mouth_color, width=3)
        else:
            # Neutral - but can have subtle curve from smile_intensity
            if abs(smile_intensity) > 0.1:
                # Subtle curve (blended state)
                if smile_intensity > 0:
                    # Subtle smile
                    arc_height = int(smile_intensity * 8)
                    draw.arc([cx - mouth_width, mouth_y - arc_height, cx + mouth_width, mouth_y + arc_height], 
                            0, 180, fill=mouth_color, width=2)
                else:
                    # Subtle frown
                    arc_height = int(abs(smile_intensity) * 8)
                    draw.arc([cx - mouth_width, mouth_y - 3, cx + mouth_width, mouth_y + arc_height], 
                            180, 360, fill=mouth_color, width=2)
            else:
                # Truly neutral - straight line
                draw.line([cx - mouth_width, mouth_y, cx + mouth_width, mouth_y], fill=mouth_color, width=3)

    def render_text(self, text: str, position: Tuple[int, int] = (10, 10), color: Optional[Tuple[int, int, int]] = None) -> None:
        """Render text to display. Supports multi-line text (\\n separated)."""
        image, draw = self._create_canvas(BLACK)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Handle multi-line text
        lines = text.split('\n')
        x, y = position
        line_height = 18  # Approximate line height for 14pt font
        text_color = color or WHITE
        
        for line in lines:
            if line.strip():  # Only draw non-empty lines
                draw.text((x, y), line, fill=text_color, font=font)
            y += line_height
        
        self._image = image
        self._show()
    
    def render_colored_text(self, lines_with_colors: list, position: Tuple[int, int] = (10, 10)) -> None:
        """Render multi-line text with different colors per line.
        
        Args:
            lines_with_colors: List of tuples (text, color) or just text (defaults to white)
            position: Starting position
        """
        image, draw = self._create_canvas(BLACK)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        x, y = position
        line_height = 18
        
        for item in lines_with_colors:
            if isinstance(item, tuple):
                line, color = item
            else:
                line, color = item, WHITE
            
            if line.strip():  # Only draw non-empty lines
                draw.text((x, y), line, fill=color, font=font)
            y += line_height
        
        self._image = image
        self._show()

    def clear(self) -> None:
        """Clear the display - shows minimal default instead of grey."""
        self._show_waking_face()  # Show minimal default instead of blank

    def _show(self):
        """Send image to display if available - safe, never crashes.

        When _deferred is True, the SPI push is skipped (image is still stored).
        Call flush() to push the final image after all overlays are applied.
        """
        if self._deferred:
            return  # Image stored in self._image, SPI push deferred to flush()
        self._push_to_display()

    def _push_to_display(self):
        """Actually push self._image to SPI display hardware, applying brightness.

        Uses 1.0s timeout to prevent SPI hangs from blocking render thread.
        On failure, attempts reinit after a cooldown rather than permanently giving up.
        """
        if not self._display and self._image:
            # Try to recover display if enough time has passed (30s cooldown)
            import time as _time
            now = _time.time()
            if now - self._last_reinit_attempt > 30.0:
                self._last_reinit_attempt = now
                print("[Display] Attempting display reinit ...", file=sys.stderr, flush=True)
                self._init_display()
                if self._display:
                    print("[Display] Reinit succeeded!", file=sys.stderr, flush=True)
                    self._display_fail_count = 0

        if self._display and self._image:
            try:
                from ..error_recovery import safe_call_with_timeout
                img_to_show = None
                if self._manual_brightness < 1.0:
                    # Use cached dimmed image if source and brightness unchanged
                    source_id = id(self._image)
                    if (self._cached_dimmed_image is not None and
                        self._cached_source_id == source_id and
                        self._cached_brightness == self._manual_brightness):
                        img_to_show = self._cached_dimmed_image
                    else:
                        # Cache miss - compute and cache
                        dimmed = ImageEnhance.Brightness(self._image).enhance(self._manual_brightness)
                        self._cached_dimmed_image = dimmed
                        self._cached_source_id = source_id
                        self._cached_brightness = self._manual_brightness
                        img_to_show = dimmed
                else:
                    # Full brightness - clear cache (not needed)
                    self._cached_dimmed_image = None
                    self._cached_source_id = None
                    img_to_show = self._image
                if img_to_show is not None:
                    # Wrap to return True on success (image() returns None)
                    # Use 3.0s timeout — first render after boot can be very slow
                    result = safe_call_with_timeout(
                        lambda: (self._display.image(img_to_show), True)[1],
                        timeout_seconds=3.0,
                        default=False,
                        log_error=True
                    )
                    if result is False:
                        self._display_fail_count += 1
                        print(f"[Display] SPI timeout (fail #{self._display_fail_count})", file=sys.stderr, flush=True)
                        if self._display_fail_count >= 10:
                            print("[Display] 10 consecutive failures — marking unavailable for reinit", file=sys.stderr, flush=True)
                            self._display = None
                    else:
                        self._display_fail_count = 0
            except Exception as e:
                self._display_fail_count += 1
                print(f"[Display] Hardware error during show: {e} (fail #{self._display_fail_count})", file=sys.stderr, flush=True)
                if self._display_fail_count >= 10:
                    print("[Display] Marking display as unavailable for reinit", file=sys.stderr, flush=True)
                    self._display = None

    def flush(self):
        """Push the current image to display. Call after deferred rendering is complete."""
        self._push_to_display()

    def blank(self):
        """Push a solid black frame to the display. Used for clean shutdown/startup."""
        if not self._display:
            return
        try:
            black = Image.new("RGB", (self.config.width, self.config.height), (0, 0, 0))
            self._image = black
            self._display.image(black)
        except Exception as e:
            print(f"[Display] Error blanking: {e}", file=sys.stderr, flush=True)

    def is_available(self) -> bool:
        """Check if display hardware is available."""
        return self._display is not None

    def show_default(self) -> None:
        """Show minimal default screen (non-grey, non-distracting)."""
        self._show_waking_face()

    def render_image(self, image: Image.Image) -> None:
        """Render a PIL Image directly to the display."""
        if not self._display:
            return
        try:
            self._image = image
            self._display.image(image)
        except Exception as e:
            print(f"[Display] Error rendering image: {e}", file=sys.stderr, flush=True)

    def save_image(self, path: str) -> None:
        """Save current image to file (for debugging)."""
        if self._image:
            self._image.save(path)

    def get_image(self) -> Optional[Image.Image]:
        """Get current image (for testing)."""
        return self._image


class NoopRenderer(DisplayRenderer):
    """No-op renderer when PIL not available."""

    def render_face(self, state: FaceState, name: Optional[str] = None) -> None:
        pass

    def render_text(self, text: str, position: Tuple[int, int] = (10, 10)) -> None:
        pass

    def clear(self) -> None:
        pass

    def is_available(self) -> bool:
        return False

    def show_default(self) -> None:
        """No-op for systems without display."""
        pass


def get_display(config: Optional[DisplayConfig] = None) -> DisplayRenderer:
    """Get appropriate display renderer."""
    if HAS_PIL:
        return PilRenderer(config)
    return NoopRenderer()
