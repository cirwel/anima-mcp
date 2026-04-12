"""
Gestural Era — Lumen's second art period.
Feb 7, 2026 – present.

5 micro-primitives (dot, stroke, curve, cluster, drag).
Focus drift with direction locks (no forced orbits — circles emerge organically or not at all).
Full-palette HSV color generation.
Granular mark-making: small deliberate acts that accumulate into forms.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from ..art_era import EraState


@dataclass
class GesturalState(EraState):
    """Gestural era's per-drawing state."""

    # Direction memory — when locked, direction resists wobble (sustained lines)
    direction_locked: bool = False
    direction_lock_remaining: int = 0
    direction_commitment: float = 0.0  # Smooth signal: ramps during locks, decays after
    # Color coherence — hue anchors per gesture run, drifts slightly per mark
    _run_hue: float = -1.0  # -1 means unset; set on first color of gesture run

    def intentionality(self) -> float:
        """Proprioceptive I_signal for EISV.

        Smooth commitment replaces binary lock contribution.
        """
        intentionality_signal = 0.15
        intentionality_signal += 0.55 * self.direction_commitment
        if self.gesture_remaining > 0:
            intentionality_signal += min(0.3, self.gesture_remaining / 20.0 * 0.3)
        return min(1.0, intentionality_signal)

    def gestures(self) -> List[str]:
        return ["dot", "stroke", "curve", "cluster", "drag"]


class GesturalEra:
    """Gestural era — granular mark-making with 5 micro-primitives."""

    name = "gestural"
    description = "Granular mark-making: dots, strokes, curves, clusters, drags"

    def create_state(self) -> GesturalState:
        return GesturalState()

    def choose_gesture(
        self,
        state: GesturalState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose a new gesture type. Near-random choice, long committed runs."""
        state.gesture = random.choice(state.gestures())
        # Coherence extends runs: low C -> 15-30, high C -> 15-45
        state.gesture_remaining = random.randint(15, 30 + int(15 * coherence))
        # New gesture run gets fresh color anchor
        state._run_hue = -1.0

    def place_mark(
        self,
        state: GesturalState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place a mark using the active gesture.

        Scale breath: mark sizes grow with energy. High energy = bold, confident marks.
        Low energy = delicate, precise marks. Creates natural visual weight progression.
        """
        x = int(focus_x)
        y = int(focus_y)
        gesture = state.gesture

        # Scale breath — energy modulates mark size
        # energy 1.0 -> scale 1.5 (bold), energy 0.1 -> scale 0.6 (delicate)
        scale = 0.5 + energy

        # Brush radius: 1px at low energy, 2-3px at high energy
        brush_radius = max(1, int(energy * 3))

        if gesture == "dot":
            if energy > 0.5:
                # Multi-pixel dot cluster (cross/diamond pattern)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx * dx + dy * dy <= 1:
                            px, py = x + dx, y + dy
                            if 0 <= px < 240 and 0 <= py < 240:
                                canvas.draw_pixel(px, py, color)
            else:
                if 0 <= x < 240 and 0 <= y < 240:
                    canvas.draw_pixel(x, y, color)

        elif gesture == "stroke":
            length = int(random.randint(2, 6) * scale)
            for i in range(length):
                cx = x + math.cos(direction) * i
                cy = y + math.sin(direction) * i
                self._brush(canvas, cx, cy, brush_radius, color)

        elif gesture == "curve":
            length = int(random.randint(3, 8) * scale)
            angle = direction
            cx, cy = float(x), float(y)
            step_size = 1.0 + scale * 0.5  # bigger steps when bold
            for i in range(length):
                angle += random.gauss(0, 0.3)
                cx += math.cos(angle) * step_size
                cy += math.sin(angle) * step_size
                self._brush(canvas, cx, cy, brush_radius, color)

        elif gesture == "cluster":
            count = int(random.randint(2, 5) * scale)
            spread = int(2 * scale) + brush_radius
            for _ in range(count):
                px = x + random.randint(-spread, spread)
                py = y + random.randint(-spread, spread)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "drag":
            length = int(random.randint(8, 15) * scale)
            angle = direction + random.gauss(0, 0.1)
            for i in range(length):
                cx = x + math.cos(angle) * i
                cy = y + math.sin(angle) * i
                self._brush(canvas, cx, cy, brush_radius, color)

    @staticmethod
    def _brush(canvas, cx: float, cy: float, radius: int, color: Tuple[int, int, int]):
        """Draw a filled circle of given radius at (cx, cy)."""
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

    def _sparse_jump(self, canvas) -> Tuple[float, float]:
        """Jump biased toward sparse canvas areas. 50% sparse-biased, 50% random."""
        if canvas is not None and random.random() < 0.5:
            gx, gy = canvas.sparsest_cell()
            x = gx * 30 + random.uniform(5, 25)
            y = gy * 30 + random.uniform(5, 25)
            return max(40.0, min(200.0, x)), max(40.0, min(200.0, y))
        return random.uniform(40, 200), random.uniform(40, 200)

    def drift_focus(
        self,
        state: GesturalState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Drift the focus point — wander influenced by stability, coherence, clarity.

        clarity modulates direction wobble and jump probability:
        high clarity = steadier direction, fewer jumps (focused strokes).
        canvas: if provided, focus jumps bias toward sparse areas.
        """
        C = coherence

        # --- Direction memory: sometimes direction locks for sustained lines ---
        if state.direction_lock_remaining > 0:
            # Locked: minimal wobble (tight lines)
            direction += random.gauss(0, 0.03)
            state.direction_lock_remaining -= 1
            # Ramp up commitment smoothly (field-era style)
            state.direction_commitment = min(1.0, state.direction_commitment + 0.06)
            if state.direction_lock_remaining <= 0:
                state.direction_locked = False
        else:
            # Decay commitment gradually (not instant drop)
            state.direction_commitment *= 0.95
            # Wobble modulated by clarity: high clarity = steadier hand
            wobble = 0.1 + (1.0 - clarity) * 0.2  # 0.1 at clarity=1, 0.3 at clarity=0
            direction += random.gauss(0, wobble)

            # Lock probability: coherence + clarity (focused = more sustained lines)
            lock_prob = 0.06 * (0.5 + C) * (0.5 + clarity * 0.5)
            if random.random() < lock_prob:
                state.direction_locked = True
                state.direction_lock_remaining = random.randint(15, 40)

        # Step in current direction — organic meandering
        step = 3 + random.random() * 5
        focus_x += math.cos(direction) * step
        focus_y += math.sin(direction) * step

        # Soft bounce off edges
        margin = 20
        if focus_x < margin:
            direction = random.uniform(-math.pi / 4, math.pi / 4)
            focus_x = float(margin)
        elif focus_x > 240 - margin:
            direction = random.uniform(math.pi * 3 / 4, math.pi * 5 / 4)
            focus_x = float(240 - margin)
        if focus_y < margin:
            direction = random.uniform(math.pi / 4, math.pi * 3 / 4)
            focus_y = float(margin)
        elif focus_y > 240 - margin:
            direction = random.uniform(-math.pi * 3 / 4, -math.pi / 4)
            focus_y = float(240 - margin)

        # Focus jump — coherence and clarity reduce jumps
        jump_prob = 0.03 * (1.0 - 0.4 * C) * (1.0 - 0.4 * clarity)
        if random.random() < jump_prob:
            focus_x, focus_y = self._sparse_jump(canvas)
            direction = random.uniform(0, 2 * math.pi)
            state.direction_locked = False
            state.direction_lock_remaining = 0
            # Don't zero commitment — let it decay naturally via *= 0.95

        return focus_x, focus_y, direction

    def generate_color(
        self,
        state: GesturalState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Color with per-run coherence. Hue anchors at gesture start, drifts per mark.

        5% chance of a vibrant accent — full-sat random hue that bypasses the
        run anchor. Acts as punctuation, not palette shift.
        """
        import colorsys

        # Vibrant accent: 5% chance of a full-sat surprise at a random hue
        if random.random() < 0.05:
            accent_hue = random.random()
            rgb = colorsys.hsv_to_rgb(accent_hue, 1.0, 0.8 + random.random() * 0.2)
            color = tuple(int(c * 255) for c in rgb)
            return color, "vibrant"

        # Anchor hue at start of gesture run; drift +/-5 degrees per mark
        if state._run_hue < 0:
            state._run_hue = (warmth * 360.0 + random.random() * 180.0) % 360.0
        else:
            state._run_hue = (state._run_hue + random.gauss(0, 5.0)) % 360.0

        hue = state._run_hue

        # Light regime shifts
        if light_regime == "dark":
            hue = (hue + 30) % 360.0
            sat_mod = -0.1
            val_mod = -0.1
        elif light_regime == "bright":
            hue = (hue - 15) % 360.0
            sat_mod = 0.1
            val_mod = 0.05
        else:
            sat_mod = 0.0
            val_mod = 0.0

        saturation = max(0.1, min(1.0, 0.3 + clarity * 0.7 + random.gauss(0, 0.1) + sat_mod))
        brightness = max(0.2, min(1.0, 0.4 + stability * 0.6 + random.gauss(0, 0.1) + val_mod))

        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        color = tuple(int(c * 255) for c in rgb)

        if hue < 60 or hue > 300:
            hue_category = "warm"
        elif hue < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"
        return color, hue_category
