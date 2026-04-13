"""
Resonance Era — Lumen's memory-field art period.

Marks deposit energy into a 48x48 memory field that decays and diffuses
over time. Color and placement are guided by field gradients: Lumen
revisits regions of accumulated memory, producing layered, resonant forms.

Pure NumPy operations — no scipy dependency.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..art_era import EraState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_SIZE = 48          # 48x48 grid, each cell = 5x5 pixels on 240x240 canvas
CELL_SIZE = 5            # 240 / 48
DECAY_RATE = 0.995       # Per-cycle multiplicative decay
CLEAR_DECAY = 0.3        # Field multiplier on canvas clear
DEPOSIT_W_WARMTH = 0.5
DEPOSIT_W_PRESENCE = 0.3
DEPOSIT_W_CLARITY = 0.2
DIFFUSION_SIGMA_MIN = 0.5
DIFFUSION_SIGMA_MAX = 2.0
GRADIENT_LOW = 0.15
GRADIENT_HIGH = 0.45
WARMTH_BIAS_DEGREES = 10.0
FIELD_HIGH_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Pure functions — memory field core
# ---------------------------------------------------------------------------


def _deposit(
    field: np.ndarray, pixel_x: int, pixel_y: int, value: float
) -> None:
    """Add *value* to the memory field cell at the given pixel position.

    Cell indices are ``pixel // CELL_SIZE``, clamped to ``[0, FIELD_SIZE-1]``.
    Operates **in-place** on *field*.
    """
    cx = min(max(pixel_x // CELL_SIZE, 0), FIELD_SIZE - 1)
    cy = min(max(pixel_y // CELL_SIZE, 0), FIELD_SIZE - 1)
    field[cy, cx] += value


def _decay(field: np.ndarray) -> None:
    """Apply multiplicative decay to the entire field **in-place**."""
    field *= DECAY_RATE


def _diffuse(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur the field using a 3x3 kernel built from *sigma*.

    Returns a **new** array (does not mutate *field*).
    If ``sigma < 0.1`` the field is returned as an unchanged copy.

    Implementation: manual 3x3 convolution with zero-padded boundaries,
    pure NumPy — no scipy.
    """
    if sigma < 0.1:
        return field.copy()

    # Build a 3x3 Gaussian kernel
    ax = np.array([-1, 0, 1], dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()  # normalise so energy is conserved

    # Zero-pad the field by 1 on each side
    padded = np.pad(field, pad_width=1, mode="constant", constant_values=0.0)

    # Convolve: slide the kernel over every cell
    h, w = field.shape
    out = np.zeros_like(field, dtype=np.float64)
    for ki in range(3):
        for kj in range(3):
            out += kernel[ki, kj] * padded[ki : ki + h, kj : kj + w]

    return out.astype(field.dtype)


def _gradient_at(
    field: np.ndarray, cx: int, cy: int
) -> Tuple[float, float, float]:
    """Finite-difference gradient at cell ``(cx, cy)``.

    Returns ``(gx, gy, magnitude)``.  Boundary cells use clamped (replicated)
    neighbours — no wrapping.
    """
    h, w = field.shape

    # Clamped neighbours
    x_lo = max(cx - 1, 0)
    x_hi = min(cx + 1, w - 1)
    y_lo = max(cy - 1, 0)
    y_hi = min(cy + 1, h - 1)

    # Central-difference style, but at edges the denominator stays 2
    # (clamped value repeats, so gradient → 0 at boundary — intentional).
    gx = (float(field[cy, x_hi]) - float(field[cy, x_lo])) / 2.0
    gy = (float(field[y_hi, cx]) - float(field[y_lo, cx])) / 2.0
    mag = math.sqrt(gx * gx + gy * gy)

    return gx, gy, mag


# ---------------------------------------------------------------------------
# Era state
# ---------------------------------------------------------------------------


@dataclass
class ResonanceState(EraState):
    """Resonance era state — carries the memory field."""

    field: np.ndarray = None  # initialized in __post_init__
    cycle_count: int = 0
    _grad_gx: float = 0.0
    _grad_gy: float = 0.0
    _grad_mag: float = 0.0
    _focus_cx: int = 24  # Current focus cell coordinates
    _focus_cy: int = 24

    def __post_init__(self):
        if self.field is None:
            self.field = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)

    def intentionality(self) -> float:
        """Gradient magnitude + gesture commitment."""
        base = 0.1
        base += min(0.4, self._grad_mag * 2.0)  # gradient contribution
        if self.gesture_remaining > 0:
            base += min(0.3, self.gesture_remaining / 20.0 * 0.3)
        return min(1.0, base)

    def gestures(self) -> List[str]:
        return ["sediment", "flow", "scratch"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalized_gradient(state: ResonanceState) -> float:
    """Gradient magnitude normalized to field max. Returns 0 if field empty."""
    field_max = state.field.max()
    if field_max < 1e-6:
        return 0.0
    gx, gy, mag = _gradient_at(state.field, state._focus_cx, state._focus_cy)
    state._grad_gx = gx
    state._grad_gy = gy
    state._grad_mag = mag
    return min(1.0, mag / field_max) if field_max > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Era class (partial — create_state + choose_gesture)
# ---------------------------------------------------------------------------


class ResonanceEra:
    """Resonance era — marks respond to emotional memory."""

    name = "resonance"
    description = "Marks respond to emotional memory: sediment, flow, and scratches"
    min_drawings = 50

    def create_state(self) -> ResonanceState:
        return ResonanceState()

    def choose_gesture(
        self,
        state: ResonanceState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Select gesture based on memory-field gradient at focus."""
        norm_grad = _normalized_gradient(state)
        if norm_grad < GRADIENT_LOW:
            state.gesture = "sediment"
        elif norm_grad < GRADIENT_HIGH:
            state.gesture = "flow"
        else:
            state.gesture = "scratch"
        state.gesture_remaining = random.randint(10, 25 + int(15 * coherence))

    @staticmethod
    def _brush(canvas, cx, cy, radius, color):
        ix, iy = int(cx), int(cy)
        if radius <= 1:
            if 0 <= ix < 240 and 0 <= iy < 240:
                canvas.draw_pixel(ix, iy, color)
            return
        for dx in range(-radius + 1, radius):
            for dy in range(-radius + 1, radius):
                if dx * dx + dy * dy < radius * radius:
                    px, py = ix + dx, iy + dy
                    if 0 <= px < 240 and 0 <= py < 240:
                        canvas.draw_pixel(px, py, color)

    def place_mark(self, state, canvas, focus_x, focus_y, direction, energy, color):
        scale = 0.5 + energy
        x = int(focus_x)
        y = int(focus_y)

        if state.gesture == "sediment":
            radius = max(1, int(1 + energy * 2))
            ox = random.randint(-1, 1)
            oy = random.randint(-1, 1)
            self._brush(canvas, x + ox, y + oy, radius, color)

        elif state.gesture == "flow":
            grad_angle = (
                math.atan2(state._grad_gy, state._grad_gx)
                if (state._grad_gx != 0 or state._grad_gy != 0)
                else direction
            )
            length = int(random.randint(3, 7) * scale)
            angle = grad_angle
            cx, cy = float(x), float(y)
            brush_r = max(1, int(energy * 2))
            for i in range(length):
                angle += random.gauss(0, 0.2)
                cx += math.cos(angle) * 1.2
                cy += math.sin(angle) * 1.2
                self._brush(canvas, cx, cy, brush_r, color)

        elif state.gesture == "scratch":
            grad_angle = math.atan2(state._grad_gy, state._grad_gx)
            cross_angle = grad_angle + math.pi / 2
            length = int(random.randint(8, 16) * scale)
            cx, cy = float(x), float(y)
            for i in range(length):
                cx += math.cos(cross_angle)
                cy += math.sin(cross_angle)
                ix, iy = int(cx), int(cy)
                if 0 <= ix < 240 and 0 <= iy < 240:
                    canvas.draw_pixel(ix, iy, color)

        # Deposit to memory field
        deposit_val = 0.5
        _deposit(state.field, int(focus_x), int(focus_y), deposit_val)

        # Update focus cell
        state._focus_cx = min(int(focus_x) // CELL_SIZE, FIELD_SIZE - 1)
        state._focus_cy = min(int(focus_y) // CELL_SIZE, FIELD_SIZE - 1)

        # Decay + diffuse
        state.cycle_count += 1
        _decay(state.field)
        sigma = (DIFFUSION_SIGMA_MIN + DIFFUSION_SIGMA_MAX) / 2
        state.field = _diffuse(state.field, sigma=sigma)

    def generate_color(
        self,
        state: ResonanceState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Generate an RGB color and hue category from anima dimensions.

        Base hue sweeps from cool blue (220deg) at warmth=0 to warm amber
        (40deg) at warmth=1.  High-field zones bias the hue further toward
        amber, and the light regime applies final shifts.
        """
        import colorsys

        # Base hue: 220° at warmth=0 (cool blue), 40° at warmth=1 (warm amber)
        hue_deg = 220.0 - warmth * 180.0

        # Field-driven warmth bias: marks in high-field zones shift toward amber
        field_max = state.field.max()
        if field_max > 1e-6:
            field_val = state.field[state._focus_cx, state._focus_cy]
            norm_field = field_val / field_max
            if norm_field > FIELD_HIGH_THRESHOLD:
                hue_deg -= (
                    WARMTH_BIAS_DEGREES
                    * (norm_field - FIELD_HIGH_THRESHOLD)
                    / (1.0 - FIELD_HIGH_THRESHOLD)
                )

        # Light regime shifts
        if light_regime == "dark":
            hue_deg += 30.0
            sat_mod = -0.1
            val_mod = -0.1
        elif light_regime == "bright":
            hue_deg -= 15.0
            sat_mod = 0.1
            val_mod = 0.05
        else:
            sat_mod = 0.0
            val_mod = 0.0

        hue_deg = hue_deg % 360.0
        hue = hue_deg / 360.0
        saturation = max(0.1, min(1.0, 0.3 + clarity * 0.6 + sat_mod))
        brightness = max(0.2, min(1.0, 0.4 + stability * 0.5 + val_mod))

        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        if hue_deg < 60 or hue_deg > 300:
            hue_category = "warm"
        elif hue_deg < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"

        return color, hue_category
