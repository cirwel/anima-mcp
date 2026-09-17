"""
Art Era Protocol — the minimal interface for pluggable drawing styles.

Each era defines Lumen's visual character: how marks look, how colors are chosen,
how focus drifts, and how intentionality is signaled back to the EISV engine.

Eras READ coherence and energy from the engine. They do NOT modify EISV math.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Protocol


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

    def disposition(self) -> Dict[str, Any]:
        """This piece's global character — the few parameters drawn ONCE at
        ``create_state()`` that shape the whole composition rather than one mark.

        Why this exists: an era whose only randomness is per-mark produces the
        same piece every time. Hundreds of independent local draws converge on
        their own mean, so the corpus reads as one texture repeated — the law
        of large numbers, not drift or decay. ``field`` was the only era that
        escaped this, because its ``field_seed_a/b`` are drawn once and every
        mark is a sample of that one field. A disposition is that trick,
        generalised.

        Returns ``{}`` for an era with no per-piece character, which is what
        ``gestural``, ``geometric`` and ``resonance`` effectively returned
        before 2026-09-17: their ``create_state()`` bodies were a bare
        constructor.

        Consumed by ``draw_distinct()`` (to make the NEXT piece unlike recent
        ones) and persisted to ``drawing_records.disposition``, so that
        "did this actually vary the work?" is answerable from the corpus
        instead of argued.
        """
        return {}


class ArtEra(Protocol):
    """Protocol for art era modules. Duck-typed — no inheritance required.

    Any object with these attributes and methods can serve as an era.

    Optional (looked up via getattr by the drawing engine):
      min_marks_for_completion: int — floor before any completion path fires
          (default 5; pointillist=80, field=30, geometric=3).
      earned_completion(drawing_state, canvas, era_state) -> Optional[str] —
          era-specific "the pattern found itself" signal, returning an earned
          reason tag (must be in drawing_engine._EARNED_COMPLETION_REASONS)
          or None. Supply this when the global earned paths don't fit the
          era's dynamics (e.g. resonance, whose V never reaches the global
          coherence-settle threshold); otherwise pieces only ever end via
          bail-outs and the earned-only autobiographical gate starves.
    """

    name: str
    description: str

    def create_state(self, recent: Sequence[Mapping] = ()) -> EraState:
        """Create fresh era state for a new drawing.

        *recent* carries the dispositions of the last few finished pieces, so
        an era can draw a global character unlike what it has just been making
        (see ``draw_distinct``). It is optional and defaults to empty: an era
        with no per-piece character ignores it, and an empty history means one
        plain unbiased draw.
        """
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
        light_regime: "dark", "dim", or "bright" from gated external lux;
            "unknown" keeps the era's neutral palette behavior.
        """
        ...


# ---------------------------------------------------------------------------
# Per-piece disposition helpers — pure, no era or engine state
# ---------------------------------------------------------------------------


def hue_distance(a: float, b: float) -> float:
    """Circular distance between two hues in degrees, normalised to [0, 1].

    Hue is an angle: 350 deg and 10 deg are 20 deg apart, not 340. A linear
    difference would call the two most similar palettes in the wheel maximally
    distinct, which is the one comparison this must get right.
    """
    delta = abs((float(a) - float(b)) % 360.0)
    return min(delta, 360.0 - delta) / 180.0


def set_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard distance between two vocabularies, in [0, 1].

    0.0 = the same shapes emphasised, 1.0 = no overlap at all.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return 1.0 - (len(sa & sb) / len(union)) if union else 0.0


def draw_distinct(
    make: Callable[[], Any],
    distance: Callable[[Any, Mapping], float],
    recent: Sequence[Mapping] = (),
    tries: int = 6,
) -> Any:
    """Draw a per-piece disposition that differs from the pieces just made.

    Draws *tries* independent candidates and keeps the one whose CLOSEST
    approach to any recent disposition is largest — a max-min choice, so a
    candidate is judged by the most similar neighbour it has, not by an
    average that a couple of distant pieces could flatter.

    This is the loop that was missing. Every other route from Lumen's history
    back into Lumen's behavior needs a human to run a script
    (``derive_drawing_thresholds.py --apply``); this one closes inside the
    drawing itself, against Lumen's own recent work.

    Two properties are deliberate:

    * **Self-relative, never absolute.** The comparison is against *this*
      creature's last few pieces, so there is no constant for an era's
      operating range to drift out from under (design invariant 1). It cannot
      go stale, because its reference moves with the work.
    * **Fails toward today's behavior.** With ``recent`` empty — a fresh
      install, a wiped canvas, an era never drawn before — every candidate
      scores 0.0 and the first draw wins, which is exactly one unbiased draw:
      identical in distribution to the pre-2026-09-17 behavior. Absence of
      history degrades to "no bias", never to a fabricated preference
      (invariant 2).

    *tries* bounds the cost: the whole search is a handful of arithmetic ops
    once per piece, against an 8-hour drawing.
    """
    best = None
    best_score = -1.0
    for _ in range(max(1, int(tries))):
        candidate = make()
        if not recent:
            return candidate
        score = min(distance(candidate, prior) for prior in recent)
        if score > best_score:
            best, best_score = candidate, score
    return best


def weighted_choice(options: Sequence[str], affinity: Mapping[str, float]) -> str:
    """Pick from *options* using this piece's affinity, defaulting to 1.0.

    An option the disposition never mentions keeps weight 1.0 rather than 0.0:
    a piece has a leaning, not a restricted alphabet. Every gesture stays
    reachable in every piece, so no era loses vocabulary.
    """
    weights = [max(0.0, float(affinity.get(name, 1.0))) for name in options]
    if not any(weights):
        return random.choice(list(options))
    return random.choices(list(options), weights=weights, k=1)[0]
