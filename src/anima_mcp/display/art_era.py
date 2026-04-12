"""
Art Era Protocol — the minimal interface for pluggable drawing styles.

Each era defines Lumen's visual character: how marks look, how colors are chosen,
how focus drifts, and how intentionality is signaled back to the EISV engine.

Eras READ coherence and energy from the engine. They do NOT modify EISV math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Protocol


@dataclass
class EraState:
    """Per-era state that persists between marks within a single drawing session.

    Subclassed by each era for its own fields (direction locks, orbits, zones, etc.).
    NOT persisted to disk — transient within a session. Energy/mark_count persist
    in the engine (CanvasState), not here.
    """

    gesture: str = "dot"
    gesture_remaining: int = 0

    def intentionality(self) -> float:
        """Proprioceptive intentionality signal [0, 1] for EISV I_signal.

        How 'committed' is the current drawing behavior? Higher = more
        sustained/structured. The EISV engine reads this to compute energy coupling.

        Base: gesture run length contributes up to 0.3.
        Subclasses add era-specific signals (locks, orbits, grids, etc.).
        """
        intentionality_signal = 0.1
        if self.gesture_remaining > 0:
            intentionality_signal += min(0.3, self.gesture_remaining / 20.0 * 0.3)
        return min(1.0, intentionality_signal)

    def gestures(self) -> List[str]:
        """Return gesture vocabulary for this era (used for entropy normalization)."""
        return ["dot"]


class ArtEra(Protocol):
    """Protocol for art era modules. Duck-typed — no inheritance required.

    Any object with these attributes and methods can serve as an era.
    """

    name: str
    description: str

    def create_state(self) -> EraState:
        """Create fresh era state for a new drawing."""
        ...

    def choose_gesture(
        self,
        state: EraState,
        clarity: float,
        stability: float,
        presence: float,
        coherence: float,
    ) -> None:
        """Choose a new gesture type. Mutates state.gesture and state.gesture_remaining."""
        ...

    def place_mark(
        self,
        state: EraState,
        canvas: object,  # CanvasState — avoids circular import
        focus_x: float,
        focus_y: float,
        direction: float,
        energy: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Place a mark at the focus point using the active gesture.

        Calls canvas.draw_pixel(x, y, color) for each pixel.
        Energy modulates mark scale (high = bold, low = delicate).
        Direction is the current heading (radians) for directional gestures.
        """
        ...

    def drift_focus(
        self,
        state: EraState,
        focus_x: float,
        focus_y: float,
        direction: float,
        stability: float,
        presence: float,
        coherence: float,
        clarity: float = 0.5,
        canvas=None,
    ) -> Tuple[float, float, float]:
        """Drift the focus point. Returns (new_focus_x, new_focus_y, new_direction).

        May mutate state (e.g., toggling locks, starting orbits).
        Must handle edge bouncing (canvas is 240x240, 20px margin).
        canvas: optional CanvasState for spatial awareness (e.g., density grid).
        clarity: higher = tighter focus, lower = more scattered.
        """
        ...

    def generate_color(
        self,
        state: EraState,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        light_regime: str = "dim",
    ) -> Tuple[Tuple[int, int, int], str]:
        """Generate a color for the current mark.

        Returns (rgb_tuple, hue_category_string).
        hue_category is one of: "warm", "cool", "neutral", "vibrant" (for mood tracker).
        light_regime: "dark" (<20 lux), "dim" (20-200), "bright" (>200) — total visual field.
        """
        ...
