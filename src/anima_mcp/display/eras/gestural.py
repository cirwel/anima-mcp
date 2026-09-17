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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..art_era import (
    EraState,
    draw_distinct,
    hue_distance,
    weighted_choice,
)


@dataclass
class GesturalState(EraState):
    """Gestural era's per-drawing state."""

    # Direction memory — when locked, direction resists wobble (sustained lines)
    direction_locked: bool = False
    direction_lock_remaining: int = 0
    direction_commitment: float = 0.0  # Smooth signal: ramps during locks, decays after

    # --- Per-piece disposition (drawn once in create_state) -----------------
    # Before 2026-09-17 create_state() was a bare `GesturalState()`: every
    # piece began identical, and all variation was per-mark. Hundreds of
    # independent local draws converge on their own mean, so every gestural
    # piece came out the same texture — all five primitives in equal measure,
    # over a 180-degree hue window sampled afresh every mark.
    hue_offset: float = 0.0    # this piece's palette centre, degrees off warmth
    hue_span: float = 180.0    # width of the hue window (was the hardcoded 180)
    lead_gesture: str = ""     # the primitive this piece leans on
    lead_weight: float = 1.0   # how hard it leans (1.0 = no lean, the old behavior)

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

    def affinity(self) -> Dict[str, float]:
        """Per-gesture weights for this piece. Unmentioned gestures stay 1.0,
        so a lean never removes a primitive from the piece's vocabulary."""
        if not self.lead_gesture:
            return {}
        return {self.lead_gesture: self.lead_weight}

    def disposition(self) -> Dict[str, Any]:
        return {
            "hue_offset": round(self.hue_offset, 1),
            "hue_span": round(self.hue_span, 1),
            "lead": self.lead_gesture,
            "lead_weight": round(self.lead_weight, 2),
        }


class GesturalEra:
    """Gestural era — granular mark-making with 5 micro-primitives."""

    name = "gestural"
    description = "Granular mark-making: dots, strokes, curves, clusters, drags"

    def create_state(self, recent: Sequence[Mapping] = ()) -> GesturalState:
        """Draw this piece's global character, unlike the pieces just made.

        The lead gesture is chosen uniformly, so across the corpus the mean
        pixels-per-mark is unchanged (a drag-led piece is denser, a dot-led one
        sparser, and they are equally likely) — only the spread between pieces
        grows. That is deliberate: the derivations read ``drawing_records``,
        and a change that shifted the corpus mean would move the ground they
        stand on. ``tests/test_era_disposition.py`` measures it.
        """
        options = GesturalState().gestures()

        def make() -> dict:
            return {
                "hue_offset": random.uniform(0.0, 360.0),
                "hue_span": random.uniform(30.0, 220.0),
                "lead": random.choice(options),
                "lead_weight": random.uniform(2.0, 3.2),
            }

        def distance(candidate: dict, prior: Mapping) -> float:
            hue = hue_distance(candidate["hue_offset"], prior.get("hue_offset", 0.0))
            lead = 0.0 if candidate["lead"] == prior.get("lead") else 1.0
            return 0.5 * hue + 0.5 * lead

        d = draw_distinct(make, distance, recent)
        state = GesturalState()
        state.hue_offset = d["hue_offset"]
        state.hue_span = d["hue_span"]
        state.lead_gesture = d["lead"]
        state.lead_weight = d["lead_weight"]
        return state

    def choose_gesture(
        self,
        state: GesturalState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose a new gesture type. Leans on this piece's primitive, long committed runs.

        Run LENGTH is untouched, so the number of gesture switches per piece is
        the same as before — which matters because fatigue accrues per switch
        and `bailout_fatigue` reads fatigue. Only *which* gesture is chosen
        changes.
        """
        state.gesture = weighted_choice(state.gestures(), state.affinity())
        # Coherence extends runs: low C -> 15-30, high C -> 15-45
        state.gesture_remaining = random.randint(15, 30 + int(15 * coherence))

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
                px = int(x + math.cos(direction) * i)
                py = int(y + math.sin(direction) * i)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "curve":
            length = int(random.randint(3, 8) * scale)
            angle = direction
            cx, cy = float(x), float(y)
            step_size = 1.0 + scale * 0.5  # bigger steps when bold
            for i in range(length):
                angle += random.gauss(0, 0.3)
                cx += math.cos(angle) * step_size
                cy += math.sin(angle) * step_size
                px, py = int(cx), int(cy)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "cluster":
            count = int(random.randint(2, 5) * scale)
            spread = int(2 * scale)
            for _ in range(count):
                px = x + random.randint(-spread, spread)
                py = y + random.randint(-spread, spread)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

        elif gesture == "drag":
            length = int(random.randint(8, 15) * scale)
            angle = direction + random.gauss(0, 0.1)
            for i in range(length):
                px = int(x + math.cos(angle) * i)
                py = int(y + math.sin(angle) * i)
                if 0 <= px < 240 and 0 <= py < 240:
                    canvas.draw_pixel(px, py, color)

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
            focus_x = random.uniform(40, 200)
            focus_y = random.uniform(40, 200)
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
        """Generate a color for the current mark. Full palette, state-influenced not restricted.

        light_regime modulates palette: dark → cooler/deeper, bright → warmer/vivid.
        """
        VIBRANT_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 64, 64),
            (255, 192, 203), (255, 215, 0), (0, 191, 255), (138, 43, 226),
            (75, 0, 130), (0, 128, 128), (139, 69, 19), (34, 139, 34),
            (210, 180, 140), (255, 182, 193), (173, 216, 230), (144, 238, 144),
            (255, 255, 224), (221, 160, 221), (128, 0, 0), (0, 100, 0),
            (25, 25, 112), (128, 0, 128),
        ]

        use_vibrant = random.random() < (0.15 + presence * 0.15)
        if use_vibrant:
            color = random.choice(VIBRANT_COLORS)
            if stability < 0.5 and random.random() < 0.3:
                color = tuple(int(c * (0.6 + stability * 0.4)) for c in color)
            # Light regime: dim vibrants in dark, boost in bright
            if light_regime == "dark":
                color = tuple(int(c * 0.7) for c in color)
            elif light_regime == "bright":
                color = tuple(min(255, int(c * 1.15)) for c in color)
            return color, "vibrant"

        import colorsys

        # Warmth still sets the anchor; the piece's own offset and span decide
        # where around it this composition lives and how wide it ranges. A
        # narrow-span piece reads near-monochrome, a wide one polychrome —
        # where before every piece sampled the same 180-degree window.
        hue_base = warmth * 360.0
        half_span = state.hue_span / 2.0
        hue = (
            hue_base + state.hue_offset + random.uniform(-half_span, half_span)
        ) % 360.0

        # Light regime shifts: dark → cooler hues, lower sat; bright → warmer, higher sat
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

        saturation = max(0.1, min(1.0, 0.3 + clarity * 0.7 + (random.random() - 0.5) * 0.4 + sat_mod))
        brightness = max(0.2, min(1.0, 0.4 + stability * 0.6 + (random.random() - 0.5) * 0.3 + val_mod))
        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, brightness)
        color = tuple(int(c * 255) for c in rgb)

        if hue < 60 or hue > 300:
            hue_category = "warm"
        elif hue < 180:
            hue_category = "cool"
        else:
            hue_category = "neutral"
        return color, hue_category
