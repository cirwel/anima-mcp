"""
Pointillist Era — pure dot accumulation, optical color mixing.

Only places single pixels. Color mixing emerges through dot density
and adjacency — complementary hues placed side-by-side create optical
blending. Focus moves in tight, controlled patterns within "density zones."

Inspired by Seurat and Signac, adapted for Lumen's 240x240 canvas.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from ..art_era import EraState


@dataclass
class PointillistState(EraState):
    """Pointillist era's per-drawing state."""

    # Color anchor — dominant hue that slowly shifts
    color_anchor_hue: float = 0.0  # 0-360 degrees, set randomly at start

    # Density zone — focus stays within a zone, building dot density
    zone_center_x: float = 120.0
    zone_center_y: float = 120.0
    zone_radius: float = 40.0
    zone_remaining: int = 0  # Marks left in current zone

    def intentionality(self) -> float:
        """Working within a density zone = committed."""
        intentionality_signal = 0.1
        if self.zone_remaining > 0:
            intentionality_signal += 0.4  # Zone work is committed
        if self.gesture_remaining > 0:
            intentionality_signal += min(0.2, self.gesture_remaining / 30.0 * 0.2)
        return min(1.0, intentionality_signal)

    def gestures(self) -> List[str]:
        return ["single", "pair", "trio"]


class PointillistEra:
    """Pointillist era — pure dot accumulation with optical color mixing."""

    name = "pointillist"
    description = "Single-pixel dots building density, optical color mixing"

    # Completion tuning: tiny dots need lots of accumulation
    fatigue_rate = 0.5  # Half base fatigue (each dot is effortless)
    min_marks_for_completion = 80  # Need density before a pointillist drawing is "done"

    def create_state(self) -> PointillistState:
        state = PointillistState()
        state.color_anchor_hue = random.uniform(0, 360)
        # Start with a random zone
        state.zone_center_x = random.uniform(50, 190)
        state.zone_center_y = random.uniform(50, 190)
        state.zone_radius = random.uniform(25, 60)
        state.zone_remaining = random.randint(40, 100)
        return state

    def choose_gesture(
        self,
        state: PointillistState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose dot placement pattern. Weighted toward singles."""
        r = random.random()
        if r < 0.6:
            state.gesture = "single"
        elif r < 0.85:
            state.gesture = "pair"
        else:
            state.gesture = "trio"
        # Shorter runs than gestural — dots are placed individually
        state.gesture_remaining = random.randint(8, 20 + int(10 * coherence))

    def place_mark(
        self,
        state: PointillistState,
        canvas,
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place single-pixel dots. No multi-pixel strokes."""
        x = int(focus_x)
        y = int(focus_y)
        gesture = state.gesture

        if gesture == "single":
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)

        elif gesture == "pair":
            # Two adjacent pixels
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)
            dx, dy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
            px, py = x + dx, y + dy
            if 0 <= px < 240 and 0 <= py < 240:
                canvas.draw_pixel(px, py, color)

        elif gesture == "trio":
            # Three pixels in an L-shape
            if 0 <= x < 240 and 0 <= y < 240:
                canvas.draw_pixel(x, y, color)
            dx1, dy1 = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
            px1, py1 = x + dx1, y + dy1
            if 0 <= px1 < 240 and 0 <= py1 < 240:
                canvas.draw_pixel(px1, py1, color)
            # L-turn: perpendicular to first step
            dx2, dy2 = -dy1, dx1  # rotate 90 degrees
            px2, py2 = px1 + dx2, py1 + dy2
            if 0 <= px2 < 240 and 0 <= py2 < 240:
                canvas.draw_pixel(px2, py2, color)

    def drift_focus(
        self,
        state: PointillistState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Tight movement within density zones, jumps between zones.

        Clarity modulates dot scatter within zones: high clarity = tighter
        dot placement, low clarity = looser/wider scatter.
        """
        C = coherence
        # Wander tightness: clarity 1.0 → 1.5px scatter, clarity 0.0 → 4.5px
        wander_sigma = 1.5 + (1.0 - clarity) * 3.0

        if state.zone_remaining > 0:
            # Stay within zone — tight Gaussian wander
            dx = state.zone_center_x - focus_x
            dy = state.zone_center_y - focus_y
            dist = math.sqrt(dx * dx + dy * dy) + 0.01

            # Drift toward center if near edge of zone
            if dist > state.zone_radius * 0.7:
                pull = 0.2 + clarity * 0.2  # Focused = stronger pull back
                focus_x += dx / dist * pull + random.gauss(0, wander_sigma)
                focus_y += dy / dist * pull + random.gauss(0, wander_sigma)
            else:
                focus_x += random.gauss(0, wander_sigma)
                focus_y += random.gauss(0, wander_sigma)

            direction = math.atan2(
                focus_y - state.zone_center_y,
                focus_x - state.zone_center_x,
            )
            state.zone_remaining -= 1
        else:
            # Zone exhausted — jump to new zone
            state.zone_center_x = random.uniform(40, 200)
            state.zone_center_y = random.uniform(40, 200)
            # Zone size: coherence and clarity both tighten zones
            max_radius = 50 - 20 * C - 10 * clarity
            state.zone_radius = random.uniform(15, max(16, max_radius))
            state.zone_remaining = random.randint(30, 80 + int(40 * C))
            focus_x = state.zone_center_x + random.gauss(0, 5)
            focus_y = state.zone_center_y + random.gauss(0, 5)
            direction = random.uniform(0, 2 * math.pi)

        # Soft bounce off edges
        margin = 15
        focus_x = max(margin, min(240 - margin, focus_x))
        focus_y = max(margin, min(240 - margin, focus_y))

        return focus_x, focus_y, direction

    def generate_color(
        self,
        state: PointillistState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Generate colors for optical mixing — complementary pairs, analogous clusters.

        The color anchor shifts slowly. Each dot is either near the anchor hue
        or its complement, creating optical mixing when dots overlap.
        light_regime modulates: dark → deeper values, bright → higher saturation.
        """
        import colorsys

        # Slowly drift the color anchor (1-3 degrees per mark)
        state.color_anchor_hue = (state.color_anchor_hue + random.gauss(0, 2.0)) % 360.0
        anchor = state.color_anchor_hue

        # Choose hue strategy
        r = random.random()
        if r < 0.45:
            # Near anchor hue (analogous range ±30 degrees)
            hue = (anchor + random.gauss(0, 15)) % 360.0
        elif r < 0.75:
            # Complementary (180 degrees away, ±20 degrees)
            hue = (anchor + 180 + random.gauss(0, 10)) % 360.0
        else:
            # Split complementary (±120 degrees from anchor)
            offset = random.choice([120, -120])
            hue = (anchor + offset + random.gauss(0, 10)) % 360.0

        # Light regime modulation
        if light_regime == "dark":
            sat_mod = -0.08
            val_mod = -0.12
        elif light_regime == "bright":
            sat_mod = 0.1
            val_mod = 0.08
        else:
            sat_mod = 0.0
            val_mod = 0.0

        # Higher saturation than gestural — dots need to be vivid for optical mixing
        saturation = max(0.4, min(1.0, 0.6 + clarity * 0.3 + random.gauss(0, 0.1) + sat_mod))
        brightness = max(0.3, min(1.0, 0.5 + stability * 0.4 + random.gauss(0, 0.1) + val_mod))

        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        color = tuple(int(c * 255) for c in rgb)

        if hue < 60 or hue > 300:
            hue_category = "warm"
        elif hue < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"
        return color, hue_category
