"""
Field Era — marks follow an invisible vector field.

Uses cheap trig to define a smooth vector field across the canvas.
Marks are elongated and aligned to the field direction. Focus follows
flow lines with occasional cross-field jumps.

Near-monochromatic palette per drawing — brightness varies by field strength.
Organic, flowing compositions that emerge from the underlying mathematics.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from ..art_era import EraState


@dataclass
class FieldState(EraState):
    """Field era's per-drawing state."""

    # Field seed — determines the vector field shape for this drawing
    field_seed_a: float = 0.0
    field_seed_b: float = 0.0

    # Base hue for near-monochromatic palette (0-360)
    base_hue: float = 0.0

    # Flow line tracking — how long we've been following the current flow
    flow_steps: int = 0
    flow_max: int = 0  # How long this flow run lasts

    def intentionality(self) -> float:
        """Following a flow line = committed."""
        intentionality_signal = 0.1
        if self.flow_steps > 0 and self.flow_max > 0:
            # Deeper into a flow line = more committed
            progress = self.flow_steps / max(self.flow_max, 1)
            intentionality_signal += 0.4 * min(1.0, progress)
        if self.gesture_remaining > 0:
            intentionality_signal += min(0.2, self.gesture_remaining / 25.0 * 0.2)
        return min(1.0, intentionality_signal)

    def gestures(self) -> List[str]:
        return ["flow_dot", "flow_dash", "flow_strand"]


class FieldEra:
    """Field era — marks aligned to an invisible vector field."""

    name = "field"
    description = "Flow-aligned marks following invisible vector fields"

    # Completion tuning: flow lines need to develop but each mark is moderate
    fatigue_rate = 0.7  # Slightly less tiring than gestural (flow is meditative)
    min_marks_for_completion = 30  # Need enough marks to reveal the field

    def create_state(self) -> FieldState:
        state = FieldState()
        state.field_seed_a = random.uniform(0, 2 * math.pi)
        state.field_seed_b = random.uniform(0, 2 * math.pi)
        state.base_hue = random.uniform(0, 360)
        state.flow_steps = 0
        state.flow_max = random.randint(30, 80)
        return state

    def _field_angle(self, state: FieldState, x: float, y: float) -> float:
        """Compute the vector field direction at (x, y).

        Cheap trig: sin/cos combinations with different frequencies
        create a smooth, visually interesting flow pattern.
        """
        a = state.field_seed_a
        b = state.field_seed_b
        # Two-frequency field for interesting topology
        angle = (
            math.sin(x / 30.0 + a) * math.cos(y / 40.0 + b)
            + 0.5 * math.sin(x / 60.0 - b) * math.cos(y / 25.0 + a)
        ) * math.pi
        return angle

    def _field_strength(self, state: FieldState, x: float, y: float) -> float:
        """Field strength at (x, y) — affects brightness and mark weight.

        Returns 0.0 to 1.0. Higher near field convergence zones.
        """
        a = state.field_seed_a
        b = state.field_seed_b
        strength = abs(
            math.cos(x / 35.0 + a) * math.sin(y / 45.0 + b)
        )
        return min(1.0, strength)

    def choose_gesture(
        self,
        state: FieldState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose flow-aligned gesture. Weighted toward dashes and strands."""
        r = random.random()
        if r < 0.25:
            state.gesture = "flow_dot"
        elif r < 0.65:
            state.gesture = "flow_dash"
        else:
            state.gesture = "flow_strand"
        # Moderate run lengths
        state.gesture_remaining = random.randint(10, 25 + int(15 * coherence))

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

    def place_mark(
        self,
        state: FieldState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place flow-aligned marks. Direction comes from the vector field."""
        x = int(focus_x)
        y = int(focus_y)
        gesture = state.gesture

        # Get field direction at this point
        field_dir = self._field_angle(state, focus_x, focus_y)

        if gesture == "flow_dot":
            # Single pixel at flow point
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)
            # Sometimes a second dot perpendicular (cross-hatch effect)
            if energy > 0.5 and random.random() < 0.3:
                perp = field_dir + math.pi / 2
                px = int(x + math.cos(perp))
                py = int(y + math.sin(perp))
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "flow_dash":
            # 3-6 pixel line along field direction
            length = random.randint(3, 6)
            for i in range(length):
                px = int(x + math.cos(field_dir) * i)
                py = int(y + math.sin(field_dir) * i)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "flow_strand":
            # 8-15 pixel curve following field
            length = random.randint(8, 15)
            cx, cy = float(x), float(y)
            for i in range(length):
                local_dir = self._field_angle(state, cx, cy)
                local_dir += random.gauss(0, 0.15)
                cx += math.cos(local_dir) * 1.2
                cy += math.sin(local_dir) * 1.2
                px, py = int(cx), int(cy)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

    def drift_focus(
        self,
        state: FieldState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Follow flow lines with occasional cross-field jumps.

        Clarity modulates line tightness and flow length: high clarity =
        longer, tighter flow lines. Low clarity = shorter, driftier lines.
        """
        C = coherence

        state.flow_steps += 1

        if state.flow_steps < state.flow_max:
            # Follow the field — step in field direction
            field_dir = self._field_angle(state, focus_x, focus_y)
            step = 2.5 + random.random() * 3.0

            # Blend field direction with current direction (circular interpolation)
            blend = 0.7  # Strong field influence
            diff = math.atan2(math.sin(field_dir - direction), math.cos(field_dir - direction))
            direction = direction + blend * diff

            focus_x += math.cos(direction) * step
            focus_y += math.sin(direction) * step

            # Perpendicular drift: tighter when focused
            perp_sigma = 0.5 + (1.0 - clarity) * 2.0  # 0.5 at clarity=1, 2.5 at clarity=0
            perp = direction + math.pi / 2
            focus_x += math.cos(perp) * random.gauss(0, perp_sigma)
            focus_y += math.sin(perp) * random.gauss(0, perp_sigma)
        else:
            # Flow exhausted — jump to new position, start new flow line
            if random.random() < 0.6:
                # Nearby jump — perpendicular offset
                perp = direction + math.pi / 2
                offset = random.gauss(0, 30)
                focus_x += math.cos(perp) * offset
                focus_y += math.sin(perp) * offset
            else:
                # Far jump — random position
                focus_x = random.uniform(30, 210)
                focus_y = random.uniform(30, 210)

            # Reset flow tracking — clarity extends flow lines (longer focus)
            state.flow_steps = 0
            state.flow_max = random.randint(25, 70 + int(30 * C) + int(20 * clarity))
            direction = self._field_angle(state, focus_x, focus_y)

        # Soft bounce off edges
        margin = 15
        if focus_x < margin:
            focus_x = float(margin)
            direction = self._field_angle(state, focus_x, focus_y)
        elif focus_x > 240 - margin:
            focus_x = float(240 - margin)
            direction = self._field_angle(state, focus_x, focus_y)
        if focus_y < margin:
            focus_y = float(margin)
            direction = self._field_angle(state, focus_x, focus_y)
        elif focus_y > 240 - margin:
            focus_y = float(240 - margin)
            direction = self._field_angle(state, focus_x, focus_y)

        return focus_x, focus_y, direction

    def generate_color(
        self,
        state: FieldState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Near-monochromatic palette — base hue with brightness from field strength.

        The base hue drifts very slowly. Saturation and brightness vary
        with field strength and anima state, creating depth through
        value contrast rather than hue contrast.
        light_regime modulates: dark → deeper/cooler tones, bright → warmer/vivid.
        """
        import colorsys

        # Very slow hue drift (0.5 degrees per mark)
        state.base_hue = (state.base_hue + random.gauss(0, 0.5)) % 360.0

        # Narrow hue range around base (±10 degrees for near-monochromatic)
        hue = (state.base_hue + random.gauss(0, 5)) % 360.0

        # Occasional accent: complementary splash (rare)
        if random.random() < 0.08:
            hue = (state.base_hue + 180 + random.gauss(0, 15)) % 360.0

        # Light regime modulation
        if light_regime == "dark":
            sat_mod = -0.05
            val_mod = -0.1
        elif light_regime == "bright":
            sat_mod = 0.08
            val_mod = 0.1
        else:
            sat_mod = 0.0
            val_mod = 0.0

        # Saturation: moderate, influenced by clarity
        saturation = max(0.2, min(0.9, 0.4 + clarity * 0.3 + random.gauss(0, 0.1) + sat_mod))

        # Brightness: varies widely for depth — field strength modulates
        brightness_base = 0.3 + stability * 0.4
        brightness = max(0.15, min(1.0, brightness_base + random.gauss(0, 0.2) + val_mod))

        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        color = tuple(int(c * 255) for c in rgb)

        if hue < 60 or hue > 300:
            hue_category = "warm"
        elif hue < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"
        return color, hue_category
