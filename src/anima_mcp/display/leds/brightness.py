"""Brightness pipeline: pulse (breathing) only.

Manual brightness control — no auto-brightness, no gamma, no pulsing-brightness.
The operator controls a dimmer; the system just breathes gently.
"""

import math
import time


def get_pulse(pulse_cycle: float = 12.0) -> float:
    """Primary + secondary breath wave. Returns 0-1."""
    t = time.time()
    primary = (1.0 + math.sin(t * 2 * math.pi / pulse_cycle)) * 0.5
    breath = (1.0 + math.sin(t * 2 * math.pi / 18.0)) * 0.5
    return primary * (0.92 + 0.08 * breath)


def estimate_instantaneous_brightness(
    base_brightness: float,
    pulse_cycle: float = 12.0,
    pulse_amount: float = 0.05,
) -> float:
    """Estimate current LED brightness including breathing pulse.

    Amplitude scales with brightness so low brightness (0.04) stays calm.
    """
    pulse = get_pulse(pulse_cycle)
    amplitude = pulse_amount * min(1.0, max(0.15, base_brightness / 0.04))
    amplitude = min(amplitude, max(0.005, base_brightness * 0.08))
    return max(0.008, base_brightness + pulse * amplitude)
