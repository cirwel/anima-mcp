"""
Resonance Era — Lumen's memory-field art period.

Marks deposit energy into a 48x48 memory field that decays and diffuses
over time. Color and placement are guided by field gradients: Lumen
revisits regions of accumulated memory, producing layered, resonant forms.

Pure NumPy operations — no scipy dependency.
"""

import math
import random
from typing import Tuple

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
