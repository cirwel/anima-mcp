"""
Geometric Era — Lumen's first art period.
Jan 13 - Feb 7, 2026. 637 drawings. 16 shape templates.

Complete forms stamped whole — circles, spirals, arcs, patterns.
Each place_mark call draws an entire shape (50-3400 pixels).
This contrasts with the granular eras that place 1-15 pixels per mark.

Preserved as a pluggable era for rollback. The original code snapshot
was in art_movements/geometric.py (removed, see git history at ed0067d).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from ..art_era import EraState


@dataclass
class GeometricState(EraState):
    """Geometric era's per-drawing state."""

    # Shape parameters stored between choose and place
    shape_size: int = 0
    shape_variant: str = ""

    def intentionality(self) -> float:
        """Shapes are complete forms — high commitment per mark."""
        intentionality_signal = 0.1
        if self.gesture_remaining > 0:
            intentionality_signal += 0.5  # Each shape is a committed act
        return min(1.0, intentionality_signal)

    def gestures(self) -> List[str]:
        return [
            "circle", "gradient_circle", "spiral", "line", "curve",
            "arc", "wave", "rings", "starburst", "pattern",
            "rectangle", "triangle", "organic", "layered", "scatter", "drip",
        ]


class GeometricEra:
    """Geometric era — complete shape templates stamped whole."""

    name = "geometric"
    description = "Complete geometric forms — circles, spirals, arcs, patterns (Jan 2026)"

    # Completion tuning: each stamp is a big commitment — tires faster, needs fewer marks
    fatigue_rate = 2.0  # 2x base fatigue (whole shapes are exhausting)
    min_marks_for_completion = 3  # 3 shapes can be a complete drawing

    def create_state(self) -> GeometricState:
        return GeometricState()

    def choose_gesture(
        self,
        state: GeometricState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose a shape type. Weighted by clarity and stability as in original."""
        # Weights: higher clarity = more complex shapes, higher stability = more structured
        shapes = state.gestures()
        weights = [1.0] * len(shapes)

        # Circles dominant (~25% as in original)
        weights[0] = 3.0  # circle
        weights[1] = 1.5  # gradient_circle

        # Complex shapes more likely with high clarity
        if clarity > 0.5:
            weights[2] = 2.0   # spiral
            weights[4] = 1.5   # curve
            weights[7] = 1.5   # rings
            weights[13] = 1.5  # layered

        # Structured shapes more likely with high stability
        if stability > 0.5:
            weights[9] = 1.5   # pattern
            weights[10] = 1.5  # rectangle
            weights[11] = 1.5  # triangle

        # Organic shapes with low stability
        if stability < 0.4:
            weights[12] = 2.0  # organic
            weights[15] = 1.5  # drip

        state.gesture = random.choices(shapes, weights=weights, k=1)[0]
        # One shape per gesture run — each shape is a complete mark
        state.gesture_remaining = 1
        # Pre-compute size based on energy (set via energy param in place_mark)
        state.shape_size = 0  # Will be set in place_mark

    def place_mark(
        self,
        state: GeometricState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place a complete geometric shape at the focus point."""
        cx = int(focus_x)
        cy = int(focus_y)
        gesture = state.gesture

        # Energy scales shape size: high energy = bold, low = delicate
        scale = 0.5 + energy

        if gesture == "circle":
            radius = int((5 + random.randint(3, 15)) * scale)
            self._draw_circle(canvas, cx, cy, radius, color)

        elif gesture == "gradient_circle":
            radius = int((5 + random.randint(3, 15)) * scale)
            self._draw_circle_gradient(canvas, cx, cy, radius, color, energy)

        elif gesture == "spiral":
            max_radius = int((10 + random.randint(5, 20)) * scale)
            tightness = 0.3 + energy * 0.7
            self._draw_spiral(canvas, cx, cy, max_radius, color, tightness)

        elif gesture == "line":
            length = int((20 + random.randint(10, 40)) * scale)
            x2 = int(cx + math.cos(direction) * length)
            y2 = int(cy + math.sin(direction) * length)
            self._draw_line(canvas, cx, cy, x2, y2, color)

        elif gesture == "curve":
            length = int((30 + random.randint(10, 30)) * scale)
            x2 = int(cx + math.cos(direction) * length)
            y2 = int(cy + math.sin(direction) * length)
            width = max(1, int(2 * scale))
            self._draw_curve(canvas, cx, cy, x2, y2, color, width)

        elif gesture == "arc":
            radius = int((10 + random.randint(5, 20)) * scale)
            start_angle = direction
            arc_length = random.uniform(math.pi / 2, math.pi * 1.5)
            self._draw_arc(canvas, cx, cy, radius, start_angle, arc_length, color)

        elif gesture == "wave":
            amplitude = int((5 + random.randint(3, 10)) * scale)
            wavelength = int(20 + random.randint(10, 30))
            self._draw_wave(canvas, cx, cy, amplitude, wavelength, color)

        elif gesture == "rings":
            num_rings = random.randint(2, 4)
            max_radius = int((15 + random.randint(5, 15)) * scale)
            self._draw_rings(canvas, cx, cy, num_rings, max_radius, color)

        elif gesture == "starburst":
            num_rays = random.randint(4, 8)
            ray_length = int((8 + random.randint(5, 15)) * scale)
            self._draw_starburst(canvas, cx, cy, num_rays, ray_length, color)

        elif gesture == "pattern":
            size = int((3 + random.randint(2, 6)) * scale)
            self._draw_pattern(canvas, cx, cy, size, color)

        elif gesture == "rectangle":
            width = int((8 + random.randint(4, 12)) * scale)
            height = int((6 + random.randint(3, 10)) * scale)
            filled = random.random() < 0.6
            self._draw_rectangle(canvas, cx, cy, width, height, color, filled)

        elif gesture == "triangle":
            size = int((8 + random.randint(4, 12)) * scale)
            self._draw_triangle(canvas, cx, cy, size, color)

        elif gesture == "organic":
            self._draw_organic(canvas, cx, cy, color, energy, scale)

        elif gesture == "layered":
            self._draw_layered(canvas, cx, cy, color, energy, scale)

        elif gesture == "scatter":
            num_particles = int((10 + random.randint(5, 20)) * scale)
            spread = int((8 + random.randint(4, 12)) * scale)
            self._draw_scatter(canvas, cx, cy, num_particles, spread, color)

        elif gesture == "drip":
            length = int((15 + random.randint(10, 30)) * scale)
            wobble = max(1, int(3 + (1.0 - energy) * 8))
            self._draw_drip(canvas, cx, cy, length, color, wobble)

    # --- Shape drawing methods ---

    def _draw_circle(self, canvas, cx, cy, radius, color):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 240 and 0 <= py < 240:
                        canvas.draw_pixel(px, py, color)

    def _draw_circle_gradient(self, canvas, cx, cy, radius, color, clarity):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq <= radius * radius:
                    dist = math.sqrt(dist_sq)
                    gradient = 1.0 - (dist / max(radius, 1)) * 0.4
                    gradient = gradient * (0.7 + clarity * 0.3)
                    c = tuple(int(v * gradient) for v in color)
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 240 and 0 <= py < 240:
                        canvas.draw_pixel(px, py, c)

    def _draw_spiral(self, canvas, cx, cy, max_radius, color, tightness):
        turns = 2 + int(tightness * 3)
        steps = turns * 20
        for i in range(steps):
            angle = i * 2 * math.pi / 20
            radius = (i / max(steps, 1)) * max_radius
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)

    def _draw_line(self, canvas, x1, y1, x2, y2, color):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        for _ in range(max(dx, dy) + 1):
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _draw_curve(self, canvas, x1, y1, x2, y2, color, width):
        mid_x = (x1 + x2) // 2 + random.randint(-30, 30)
        mid_y = (y1 + y2) // 2 + random.randint(-30, 30)
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            x = int((1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * mid_x + t * t * x2)
            y = int((1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * mid_y + t * t * y2)
            for wx in range(-width // 2, width // 2 + 1):
                for wy in range(-width // 2, width // 2 + 1):
                    px, py = x + wx, y + wy
                    if 0 <= px < 240 and 0 <= py < 240:
                        canvas.draw_pixel(px, py, color)

    def _draw_arc(self, canvas, cx, cy, radius, start_angle, arc_length, color):
        steps = max(1, int(arc_length * radius / 2))
        for i in range(steps):
            angle = start_angle + (i / max(steps, 1)) * arc_length
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)

    def _draw_wave(self, canvas, start_x, y_center, amplitude, wavelength, color):
        for x in range(max(0, start_x), min(240, start_x + 100)):
            y = int(y_center + amplitude * math.sin(
                (x - start_x) * 2 * math.pi / max(wavelength, 1)))
            if 0 <= y < 240:
                canvas.draw_pixel(x, y, color)
            if 0 <= y + 1 < 240:
                canvas.draw_pixel(x, y + 1, color)

    def _draw_rings(self, canvas, cx, cy, num_rings, max_radius, color):
        for ring in range(1, num_rings + 1):
            radius = int(ring * max_radius / num_rings)
            for angle_deg in range(0, 360, 3):
                rad = math.radians(angle_deg)
                x = int(cx + radius * math.cos(rad))
                y = int(cy + radius * math.sin(rad))
                if 0 <= x < 240 and 0 <= y < 240:
                    canvas.draw_pixel(x, y, color)

    def _draw_starburst(self, canvas, cx, cy, num_rays, ray_length, color):
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * math.pi
            for r in range(1, ray_length + 1):
                x = int(cx + r * math.cos(angle))
                y = int(cy + r * math.sin(angle))
                if 0 <= x < 240 and 0 <= y < 240:
                    canvas.draw_pixel(x, y, color)
        if 0 <= cx < 240 and 0 <= cy < 240:
            canvas.draw_pixel(cx, cy, color)

    def _draw_pattern(self, canvas, cx, cy, size, color):
        pattern_type = random.choice(["cross", "star", "grid"])
        if pattern_type == "cross":
            for i in range(-size, size + 1):
                if 0 <= cx + i < 240 and 0 <= cy < 240:
                    canvas.draw_pixel(cx + i, cy, color)
                if 0 <= cx < 240 and 0 <= cy + i < 240:
                    canvas.draw_pixel(cx, cy + i, color)
        elif pattern_type == "star":
            for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
                for r in range(1, size + 1):
                    x = int(cx + r * math.cos(angle))
                    y = int(cy + r * math.sin(angle))
                    if 0 <= x < 240 and 0 <= y < 240:
                        canvas.draw_pixel(x, y, color)
        else:  # grid
            for i in range(-size, size + 1, 2):
                for j in range(-size, size + 1, 2):
                    x, y = cx + i, cy + j
                    if 0 <= x < 240 and 0 <= y < 240:
                        canvas.draw_pixel(x, y, color)

    def _draw_rectangle(self, canvas, cx, cy, width, height, color, filled):
        x1, y1 = cx - width // 2, cy - height // 2
        x2, y2 = cx + width // 2, cy + height // 2
        if filled:
            for x in range(max(0, x1), min(240, x2 + 1)):
                for y in range(max(0, y1), min(240, y2 + 1)):
                    canvas.draw_pixel(x, y, color)
        else:
            for x in range(max(0, x1), min(240, x2 + 1)):
                if 0 <= y1 < 240:
                    canvas.draw_pixel(x, y1, color)
                if 0 <= y2 < 240:
                    canvas.draw_pixel(x, y2, color)
            for y in range(max(0, y1), min(240, y2 + 1)):
                if 0 <= x1 < 240:
                    canvas.draw_pixel(x1, y, color)
                if 0 <= x2 < 240:
                    canvas.draw_pixel(x2, y, color)

    def _draw_triangle(self, canvas, cx, cy, size, color):
        for y_offset in range(size):
            width_at_y = int((y_offset / max(size, 1)) * size)
            y = cy + y_offset - size // 2
            for x_offset in range(-width_at_y // 2, width_at_y // 2 + 1):
                x = cx + x_offset
                if 0 <= x < 240 and 0 <= y < 240:
                    canvas.draw_pixel(x, y, color)

    def _draw_organic(self, canvas, cx, cy, color, energy, scale):
        # Radius scaled for 240×240 canvas: 3-8px produces 28-200px marks,
        # comparable to other eras. Previous values (8-34px) filled 20%+ of
        # the 15,000px canvas cap in a single mark.
        base_radius = int((3 + energy * 5) * scale)
        for dx in range(-base_radius, base_radius + 1):
            for dy in range(-base_radius, base_radius + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < 240 and 0 <= py < 240:
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < base_radius * 1.1:
                        if random.random() < 0.75:
                            canvas.draw_pixel(px, py, color)

    def _draw_layered(self, canvas, cx, cy, color, energy, scale):
        num_elements = random.randint(2, 3)
        for i in range(num_elements):
            variation = random.uniform(0.8, 1.0)
            layer_color = tuple(int(c * variation) for c in color)
            ox = cx + random.randint(-30, 30)
            oy = cy + random.randint(-30, 30)
            if i == 0:
                radius = int((6 + energy * 12) * scale)
                self._draw_circle(canvas, ox, oy, radius, layer_color)
            elif i == 1:
                if random.random() < 0.5:
                    radius = int((3 + energy * 6) * scale)
                    self._draw_circle(canvas, ox, oy, radius, layer_color)
                else:
                    size = int((2 + energy * 3) * scale)
                    self._draw_pattern(canvas, ox, oy, size, layer_color)
            else:
                for _ in range(random.randint(2, 4)):
                    dx = ox + random.randint(-10, 10)
                    dy = oy + random.randint(-10, 10)
                    if 0 <= dx < 240 and 0 <= dy < 240:
                        canvas.draw_pixel(dx, dy, layer_color)

    def _draw_scatter(self, canvas, cx, cy, num_particles, spread, color):
        for _ in range(num_particles):
            dx = int(random.gauss(0, spread / 3))
            dy = int(random.gauss(0, spread / 3))
            x, y = cx + dx, cy + dy
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)

    def _draw_drip(self, canvas, x, start_y, length, color, wobble):
        current_x = x
        for y in range(start_y, min(240, start_y + length)):
            if 0 <= current_x < 240:
                canvas.draw_pixel(current_x, y, color)
            current_x += random.randint(-wobble, wobble)
            current_x = max(0, min(239, current_x))

    def drift_focus(
        self,
        state: GeometricState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Jump to new position after each shape.

        Geometric shapes are standalone — focus jumps rather than drifts.
        Clarity modulates spread: high clarity = tighter clustering near
        current position, low clarity = scattered across canvas.
        """
        # Spread inversely proportional to clarity
        # clarity 1.0 → spread 15 (tight cluster), clarity 0.0 → spread 55 (scattered)
        spread = 15 + (1.0 - clarity) * 40

        if clarity > 0.6:
            # Focused: small jumps from current position
            focus_x += random.gauss(0, spread)
            focus_y += random.gauss(0, spread)
        else:
            # Scattered: jump around center
            focus_x = random.gauss(120, spread)
            focus_y = random.gauss(120, spread)

        direction = random.uniform(0, 2 * math.pi)

        # Clamp to canvas with margin
        margin = 25
        focus_x = max(margin, min(240 - margin, focus_x))
        focus_y = max(margin, min(240 - margin, focus_y))

        return focus_x, focus_y, direction

    def generate_color(
        self,
        state: GeometricState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Generate color — warm/cool based on warmth, saturation from clarity.

        Original geometric phase used a simpler color model than the
        gestural era's full-palette approach.
        light_regime modulates: dark → cooler bias, bright → warmer bias.
        """
        import colorsys

        # Warm/cool hue split based on warmth
        if warmth > 0.5:
            # Warm hues: reds, oranges, yellows (0-60 degrees)
            hue = random.uniform(0, 60) + (warmth - 0.5) * 40
        else:
            # Cool hues: blues, greens, purples (180-300 degrees)
            hue = random.uniform(180, 300) - (0.5 - warmth) * 40
        hue = hue % 360

        # Light regime modulation
        if light_regime == "dark":
            # In darkness: shift toward cooler, lower saturation
            hue = (hue + 20) % 360
            sat_mod = -0.08
            val_mod = -0.08
        elif light_regime == "bright":
            # In bright light: shift toward warmer, higher saturation
            hue = (hue - 10) % 360
            sat_mod = 0.1
            val_mod = 0.05
        else:
            sat_mod = 0.0
            val_mod = 0.0

        # Saturation from clarity
        saturation = max(0.2, min(1.0, 0.3 + clarity * 0.6 + random.gauss(0, 0.1) + sat_mod))
        brightness = max(0.3, min(1.0, 0.4 + stability * 0.5 + random.gauss(0, 0.1) + val_mod))

        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        color = tuple(int(c * 255) for c in rgb)

        if hue < 60 or hue > 300:
            hue_category = "warm"
        elif hue < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"
        return color, hue_category
