"""LED color mapping: anima state → RGB.

Warm amber/gold palette only. No blue, green, cyan, or pure white.
R >= G >= B invariant across all outputs.
"""

from typing import Optional, Tuple

from ..design import ease_smooth
from .types import LEDState


def transition_color(
    current: Optional[Tuple[int, int, int]],
    target: Tuple[int, int, int],
    factor: float,
    enabled: bool = True
) -> Tuple[int, int, int]:
    """Smooth color transition. Adaptive: faster when far, gentler when close."""
    if not enabled or current is None:
        return target
    dr, dg, db = abs(target[0] - current[0]), abs(target[1] - current[1]), abs(target[2] - current[2])
    dist = (dr + dg + db) / (3 * 255)
    dist_norm = min(1.0, dist / 0.25)
    adaptive = 0.6 + 0.4 * ease_smooth(dist_norm)
    f = factor * adaptive
    r = int(current[0] + (target[0] - current[0]) * f)
    g = int(current[1] + (target[1] - current[1]) * f)
    b = int(current[2] + (target[2] - current[2]) * f)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def blend_colors(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    ratio: float
) -> Tuple[int, int, int]:
    """Blend two RGB colors. ratio 0=color1, 1=color2."""
    ratio = max(0.0, min(1.0, ratio))
    r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
    g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
    b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
    return (r, g, b)


def _interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    ratio: float
) -> Tuple[int, int, int]:
    """Interpolate between two colors."""
    ratio = max(0.0, min(1.0, ratio))
    return tuple(int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3))


def _create_gradient_palette(
    warmth: float,
    clarity: float,
    stability: float,
    presence: float
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Create warm amber/gold gradient palette. R >= G >= B invariant.

    LED0 (warmth/energy): (180,110,40) -> (255,150,50) range
    LED1 (clarity): warm white i, int(i*0.82), int(i*0.45) -- never blue-white
    LED2 (stability/presence): (180,100,30) -> (220,150,55) range, NO green
    """
    # LED0: warmth/energy — deep amber to bright gold
    if warmth < 0.3:
        led0 = (180, 110, 40)
    elif warmth < 0.6:
        ratio = (warmth - 0.3) / 0.3
        led0 = _interpolate_color((180, 110, 40), (220, 130, 45), ratio)
    else:
        ratio = (warmth - 0.6) / 0.4
        led0 = _interpolate_color((220, 130, 45), (255, 150, 50), ratio)

    # LED1: clarity — warm white, never blue-white
    # i scales with clarity: dim amber-white at low clarity, brighter warm-white at high
    i = max(120, min(220, int(80 + clarity * 140)))
    led1 = (i, int(i * 0.82), int(i * 0.45))

    # LED2: stability/presence — warm earth tones, NO green
    combined = (stability * 0.6 + presence * 0.4)
    if combined < 0.3:
        led2 = (180, 100, 30)
    elif combined < 0.6:
        ratio = (combined - 0.3) / 0.3
        led2 = _interpolate_color((180, 100, 30), (200, 125, 42), ratio)
    else:
        ratio = (combined - 0.6) / 0.4
        led2 = _interpolate_color((200, 125, 42), (220, 150, 55), ratio)

    return (led0, led1, led2)


def get_shape_color_bias(shape) -> Tuple[int, int, int]:
    """Subtle RGB bias for trajectory shape. Returns (dr, dg, db).

    All biases constrained to warm direction (positive R, neutral/negative B).
    """
    if shape is None:
        return (0, 0, 0)
    SHAPE_BIASES = {
        "settled_presence": (8, 4, -4),
        "convergence": (4, 2, -2),
        "rising_entropy": (10, 5, -2),
        "falling_energy": (2, 0, -2),
        "basin_transition_down": (2, 0, -2),
        "basin_transition_up": (10, 4, -2),
        "entropy_spike_recovery": (6, 4, -2),
        "drift_dissonance": (0, -2, -4),
        "valence_rising": (6, 4, -2),
        "void_rising": (6, 4, -2),  # legacy persisted trajectory history
    }
    return SHAPE_BIASES.get(str(shape), (0, 0, 0))


def derive_led_state(
    warmth: float,
    clarity: float,
    stability: float,
    presence: float,
    pattern_mode: str = "standard",
    enable_color_mixing: bool = True,
    expression_mode: str = "balanced"
) -> LEDState:
    """Map anima metrics to LED colors. Warm amber/gold only.

    Only 'standard' mode is supported. Other pattern_mode values
    are silently treated as 'standard' (no alert, minimal, or expressive).
    Default brightness: 0.04 (manual control only).
    """
    # All modes map to standard — no alert, minimal, or expressive
    led0, led1, led2 = _create_gradient_palette(warmth, clarity, stability, presence)

    # Expression intensity scaling (subtle dimming only, no amplification beyond warm range)
    intensity = {"subtle": 0.6, "balanced": 1.0}.get(expression_mode, 1.0)
    if intensity != 1.0:
        led0 = tuple(max(0, min(255, int(c * intensity))) for c in led0)
        led1 = tuple(max(0, min(255, int(c * intensity))) for c in led1)
        led2 = tuple(max(0, min(255, int(c * intensity))) for c in led2)

    return LEDState(led0=led0, led1=led1, led2=led2, brightness=0.04)
