"""
Drawing Engine - Lumen's autonomous drawing system.

Extracted from screens.py to separate drawing logic from display rendering.
Contains EISV thermodynamics, attention signals, coherence tracking,
narrative arc, and mark-making orchestration.

The DrawingEngine owns the canvas, intent, drawing goal, and active era.
It has zero display dependencies — it only manipulates DrawingState/CanvasState/DrawingIntent
and delegates to ArtEra instances.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from datetime import datetime
import time
import sys
import json
import math
import random

from ..atomic_write import atomic_json_write
from ..anima import Anima
from ..expression_moods import ExpressionMoodTracker


def _get_drawing_bridge():
    """Get shared server bridge for drawing outcome reporting (late import to avoid circular deps)."""
    try:
        from ..accessors import _get_server_bridge
        return _get_server_bridge()
    except Exception:
        return None


def _get_canvas_path() -> Path:
    """Get persistent path for canvas state."""
    anima_dir = Path.home() / ".anima"
    anima_dir.mkdir(exist_ok=True)
    return anima_dir / "canvas.json"


@dataclass
class CanvasState:
    """Drawing canvas state for notepad mode - persists across restarts."""
    width: int = 240
    height: int = 240
    pixels: Dict[Tuple[int, int], Tuple[int, int, int]] = field(default_factory=dict)
    # Drawing memory - helps Lumen build on previous work
    recent_locations: List[Tuple[int, int]] = field(default_factory=list)
    drawing_phase: str = "opening"  # opening, developing, resolving, closing
    phase_start_time: float = field(default_factory=time.time)

    # Autonomy tracking
    last_save_time: float = 0.0  # When Lumen last saved a drawing
    last_clear_time: float = field(default_factory=time.time)  # When canvas was last cleared
    is_satisfied: bool = False  # Lumen feels done with current drawing
    satisfaction_time: float = 0.0  # When satisfaction was reached
    drawings_saved: int = 0  # Count of drawings Lumen has saved
    drawing_paused_until: float = 0.0  # Pause drawing after manual clear (so user sees empty canvas)

    # Save indicator (brief visual feedback)
    save_indicator_until: float = 0.0  # Show "saved" indicator until this time

    # Drawing energy persistence (survives restarts so drawings can finish)
    energy: float = 1.0  # Persisted to disk, restored on load (legacy, now derived)
    mark_count: int = 0  # Persisted to disk, restored on load

    # Attention/coherence/narrative persistence (survives restarts)
    curiosity: float = 1.0
    engagement: float = 0.5
    fatigue: float = 0.0
    arc_phase: str = "opening"
    coherence_history: List[float] = field(default_factory=list)
    i_momentum: float = 0.0
    drawing_start_time: float = 0.0  # When this drawing started (persisted for time limit)

    # Art era (persisted so drawings continue in the same era after restart)
    _era_name: str = "gestural"
    pending_era_switch: Optional[str] = None  # Queue era switch until current drawing completes

    # False-start tracking (volatile - resets on restart, not persisted)
    consecutive_false_starts: int = 0

    # Render caching - avoid redrawing all pixels every frame
    _dirty: bool = True  # Set by draw_pixel(), cleared after render
    _cached_image: object = None  # Cached PIL Image of all pixels
    _new_pixels: list = field(default_factory=list)  # Pixels added since last render

    # Spatial density grid — 8x8 cells (30px each) for spatial awareness
    density_grid: List[List[int]] = field(default_factory=lambda: [[0] * 8 for _ in range(8)])

    def draw_pixel(self, x: int, y: int, color: Tuple[int, int, int]):
        """Draw a pixel at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            is_new = (x, y) not in self.pixels
            self.pixels[(x, y)] = color
            self._new_pixels.append((x, y, color))  # Track for incremental render
            self._dirty = True
            # Update density grid (only for new pixels, not overwrites)
            if is_new:
                gx = min(x // 30, 7)
                gy = min(y // 30, 7)
                self.density_grid[gx][gy] += 1
            # Remember recent locations (keep last 20)
            self.recent_locations.append((x, y))
            if len(self.recent_locations) > 20:
                self.recent_locations.pop(0)
            # Drawing resets satisfaction
            self.is_satisfied = False

    def clear(self):
        """Clear the canvas."""
        self.pixels.clear()
        self.recent_locations.clear()
        self.drawing_phase = "opening"  # Start with opening phase
        self.phase_start_time = time.time()
        self.last_clear_time = time.time()
        self.is_satisfied = False
        self.satisfaction_time = 0.0
        self.energy = 1.0
        self.mark_count = 0
        # Reset attention/coherence/narrative
        self.curiosity = 1.0
        self.engagement = 0.5
        self.fatigue = 0.0
        self.arc_phase = "opening"
        self.coherence_history = []
        self.i_momentum = 0.0
        self.drawing_start_time = time.time()
        self._dirty = True
        self._cached_image = None
        self._new_pixels.clear()
        self.density_grid = [[0] * 8 for _ in range(8)]
        # Clear pending era switch (will be applied by canvas_clear caller)
        self.pending_era_switch = None
        # Pause drawing for 5 seconds after manual clear so user sees empty canvas
        self.drawing_paused_until = time.time() + 5.0

    def compositional_satisfaction(self) -> float:
        """Evaluate compositional satisfaction: coverage, balance, coherence.

        Returns 0.0-1.0 score based on:
        - Coverage: reasonable pixel density (not too sparse, not too dense)
        - Balance: spatial distribution across canvas quadrants
        - Visual coherence: derived from recent coherence history if available

        This provides an alternative completion path to attention exhaustion.
        """
        if len(self.pixels) < 50:
            return 0.0  # Too sparse to evaluate

        # Coverage score: ideal density is 5-25% of canvas (2880-14400 pixels)
        max_pixels = self.width * self.height
        density = len(self.pixels) / max_pixels
        if density < 0.05:
            coverage = density / 0.05  # Ramp up from 0 to 1 as we approach 5%
        elif density > 0.25:
            coverage = max(0.0, 1.0 - (density - 0.25) / 0.5)  # Ramp down if too dense
        else:
            coverage = 1.0  # Sweet spot: 5-25%

        # Balance score: spatial distribution across quadrants
        # Divide canvas into 4 quadrants and check for reasonable distribution
        quadrants = [0, 0, 0, 0]
        mid_x, mid_y = self.width // 2, self.height // 2
        for (x, y) in self.pixels.keys():
            quad = (0 if x < mid_x else 1) + (0 if y < mid_y else 2)
            quadrants[quad] += 1

        total = len(self.pixels)
        quadrant_ratios = [q / total for q in quadrants]
        # Good balance: each quadrant has 10-50% of pixels (not all in one corner)
        balance_scores = [1.0 if 0.1 <= r <= 0.5 else min(r / 0.1, (1.0 - r) / 0.5) for r in quadrant_ratios]
        balance = sum(balance_scores) / 4.0

        # Coherence score: use recent coherence history if available
        coherence = 0.5  # Default neutral
        if len(self.coherence_history) >= 5:
            recent = self.coherence_history[-10:]
            coherence = sum(recent) / len(recent)

        # Weighted combination: coverage 40%, balance 30%, coherence 30%
        satisfaction = 0.4 * coverage + 0.3 * balance + 0.3 * coherence
        return min(1.0, max(0.0, satisfaction))

    def sparsest_cell(self) -> Tuple[int, int]:
        """Return (grid_x, grid_y) of the cell with fewest pixels."""
        min_count = float('inf')
        min_cell = (0, 0)
        for gx in range(8):
            for gy in range(8):
                if self.density_grid[gx][gy] < min_count:
                    min_count = self.density_grid[gx][gy]
                    min_cell = (gx, gy)
        return min_cell

    def mark_satisfied(self):
        """Mark that Lumen feels satisfied with current drawing."""
        if not self.is_satisfied:
            self.is_satisfied = True
            self.satisfaction_time = time.time()
            print(f"[Canvas] Lumen feels satisfied with drawing ({len(self.pixels)} pixels)", file=sys.stderr, flush=True)

    def save_to_disk(self):
        """Persist canvas state to disk."""
        try:
            # Convert pixel dict keys to strings for JSON
            pixel_data = {f"{x},{y}": list(color) for (x, y), color in self.pixels.items()}
            data = {
                "pixels": pixel_data,
                "recent_locations": self.recent_locations,
                "drawing_phase": self.drawing_phase,
                "phase_start_time": self.phase_start_time,
                "last_save_time": self.last_save_time,
                "last_clear_time": self.last_clear_time,
                "is_satisfied": self.is_satisfied,
                "satisfaction_time": self.satisfaction_time,
                "drawings_saved": self.drawings_saved,
                "energy": self.energy,
                "mark_count": self.mark_count,
                "era": self._era_name,
                "pending_era_switch": self.pending_era_switch,
                # Attention/coherence/narrative state
                "curiosity": self.curiosity,
                "engagement": self.engagement,
                "fatigue": self.fatigue,
                "arc_phase": self.arc_phase,
                "coherence_history": self.coherence_history[-20:],  # Keep last 20
                "i_momentum": self.i_momentum,
                "drawing_start_time": self.drawing_start_time,
            }
            atomic_json_write(_get_canvas_path(), data)
        except Exception as e:
            print(f"[Canvas] Save to disk error: {e}", file=sys.stderr, flush=True)

    def load_from_disk(self):
        """Load canvas state from disk - defensive against corruption."""
        path = _get_canvas_path()
        if not path.exists():
            return  # No saved state, use defaults

        data = None
        try:
            raw_content = path.read_text()
            if not raw_content.strip():
                # Empty file - delete and use defaults
                print("[Canvas] Empty canvas file, starting fresh", file=sys.stderr, flush=True)
                path.unlink()
                return
            data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            # Corrupted JSON - delete file and start fresh
            print(f"[Canvas] Corrupted canvas file (invalid JSON): {e}", file=sys.stderr, flush=True)
            try:
                path.unlink()
                print("[Canvas] Deleted corrupted file, starting fresh", file=sys.stderr, flush=True)
            except Exception:
                pass
            return
        except Exception as e:
            print(f"[Canvas] Failed to read canvas file: {e}", file=sys.stderr, flush=True)
            return

        # Validate data is a dict
        if not isinstance(data, dict):
            print("[Canvas] Invalid canvas data (not a dict), starting fresh", file=sys.stderr, flush=True)
            try:
                path.unlink()
            except Exception:
                pass
            return

        # Load pixels with validation
        loaded_pixels = 0
        skipped_pixels = 0
        try:
            pixels_data = data.get("pixels", {})
            if isinstance(pixels_data, dict):
                for key, color in pixels_data.items():
                    try:
                        # Validate key format "x,y"
                        if not isinstance(key, str) or "," not in key:
                            skipped_pixels += 1
                            continue
                        parts = key.split(",")
                        if len(parts) != 2:
                            skipped_pixels += 1
                            continue
                        x, y = int(parts[0]), int(parts[1])

                        # Validate coordinates
                        if not (0 <= x < self.width and 0 <= y < self.height):
                            skipped_pixels += 1
                            continue

                        # Validate color format [r, g, b]
                        if not isinstance(color, (list, tuple)) or len(color) != 3:
                            skipped_pixels += 1
                            continue
                        r, g, b = int(color[0]), int(color[1]), int(color[2])
                        if not all(0 <= c <= 255 for c in (r, g, b)):
                            skipped_pixels += 1
                            continue

                        self.pixels[(x, y)] = (r, g, b)
                        loaded_pixels += 1
                    except (ValueError, TypeError, IndexError):
                        skipped_pixels += 1
                        continue
        except Exception as e:
            print(f"[Canvas] Error loading pixels: {e}", file=sys.stderr, flush=True)

        # Load recent_locations with validation
        try:
            locations = data.get("recent_locations", [])
            if isinstance(locations, list):
                for loc in locations[-20:]:  # Keep last 20
                    if isinstance(loc, (list, tuple)) and len(loc) == 2:
                        try:
                            x, y = int(loc[0]), int(loc[1])
                            if 0 <= x < self.width and 0 <= y < self.height:
                                self.recent_locations.append((x, y))
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass  # Non-fatal, use empty list

        # Load scalar fields with type validation
        try:
            phase = data.get("drawing_phase", "opening")
            _valid_phases = ("opening", "developing", "resolving", "closing",
                             "exploring", "building", "reflecting", "resting")
            if isinstance(phase, str) and phase in _valid_phases:
                self.drawing_phase = phase
        except Exception:
            pass

        try:
            phase_time = data.get("phase_start_time", time.time())
            if isinstance(phase_time, (int, float)):
                self.phase_start_time = float(phase_time)
        except Exception:
            pass

        try:
            save_time = data.get("last_save_time", 0.0)
            if isinstance(save_time, (int, float)):
                self.last_save_time = float(save_time)
        except Exception:
            pass

        try:
            clear_time = data.get("last_clear_time", time.time())
            if isinstance(clear_time, (int, float)):
                self.last_clear_time = float(clear_time)
        except Exception:
            pass

        try:
            satisfied = data.get("is_satisfied", False)
            if isinstance(satisfied, bool):
                self.is_satisfied = satisfied
        except Exception:
            pass

        try:
            sat_time = data.get("satisfaction_time", 0.0)
            if isinstance(sat_time, (int, float)):
                self.satisfaction_time = float(sat_time)
        except Exception:
            pass

        try:
            saved_count = data.get("drawings_saved", 0)
            if isinstance(saved_count, int) and saved_count >= 0:
                self.drawings_saved = saved_count
        except Exception:
            pass

        # Restore drawing energy (survives restarts)
        try:
            energy = data.get("energy")
            if isinstance(energy, (int, float)) and 0.0 <= energy <= 1.0:
                self.energy = float(energy)
        except Exception:
            pass

        try:
            marks = data.get("mark_count")
            if isinstance(marks, int) and marks >= 0:
                self.mark_count = marks
        except Exception:
            pass

        # Restore art era (defaults to "gestural" for backward compatibility)
        try:
            era = data.get("era", "gestural")
            if isinstance(era, str) and era:
                self._era_name = era
        except Exception:
            pass

        # Restore pending era switch
        try:
            pending = data.get("pending_era_switch")
            if pending is None or (isinstance(pending, str) and pending):
                self.pending_era_switch = pending
        except Exception:
            pass

        # Restore attention signals
        try:
            curiosity = data.get("curiosity", 1.0)
            if isinstance(curiosity, (int, float)) and 0.0 <= curiosity <= 1.0:
                self.curiosity = float(curiosity)
        except Exception:
            pass

        try:
            engagement = data.get("engagement", 0.5)
            if isinstance(engagement, (int, float)) and 0.0 <= engagement <= 1.0:
                self.engagement = float(engagement)
        except Exception:
            pass

        try:
            fatigue = data.get("fatigue", 0.0)
            if isinstance(fatigue, (int, float)) and 0.0 <= fatigue <= 1.0:
                self.fatigue = float(fatigue)
        except Exception:
            pass

        # Restore narrative arc state
        try:
            arc = data.get("arc_phase", "opening")
            if isinstance(arc, str) and arc in ("opening", "developing", "resolving", "closing"):
                self.arc_phase = arc
        except Exception:
            pass

        try:
            history = data.get("coherence_history", [])
            if isinstance(history, list):
                self.coherence_history = [float(c) for c in history[-20:] if isinstance(c, (int, float))]
        except Exception:
            pass

        try:
            i_mom = data.get("i_momentum", 0.0)
            if isinstance(i_mom, (int, float)):
                self.i_momentum = float(i_mom)
        except Exception:
            pass

        try:
            dst = data.get("drawing_start_time", 0.0)
            if isinstance(dst, (int, float)):
                self.drawing_start_time = float(dst)
        except Exception:
            pass

        # Invalidate render cache after loading
        self._dirty = True
        self._cached_image = None
        self._new_pixels.clear()

        if skipped_pixels > 0:
            print(f"[Canvas] Loaded from disk: {loaded_pixels} pixels (skipped {skipped_pixels} invalid), arc={self.arc_phase}, curio={self.curiosity:.2f}, era={self._era_name}", file=sys.stderr, flush=True)
        else:
            print(f"[Canvas] Loaded from disk: {loaded_pixels} pixels, arc={self.arc_phase}, curio={self.curiosity:.2f}, era={self._era_name}", file=sys.stderr, flush=True)


# EISV parameters for drawing (scaled from governance_core/parameters.py for ~920 mark timescale)
_EISV_PARAMS = {
    "alpha": 0.01,       # I->E coupling
    "beta_E": 0.005,     # S damping on E
    "gamma_E": 0.002,    # drift feedback to E
    "beta_I": 0.015,     # coherence boost to I
    "k": 0.005,          # S->I coupling (negative)
    "gamma_I": 0.012,    # I self-regulation (linear)
    "mu": 0.04,          # S natural decay
    "lambda1": 0.02,     # drift -> S coupling
    "lambda2": 0.008,    # coherence -> S reduction
    "kappa": 0.015,      # (I-E) -> V coupling (FLIPPED from governance)
    "delta": 0.02,       # V decay (slow = long memory)
    "C1": 1.0,           # coherence sigmoid steepness
    "Cmax": 1.0,         # max coherence
    "dt": 0.1,           # Euler step size
}


@dataclass
class DrawingState:
    """Drawing state with EISV core + attention/coherence/narrative signals.

    EISV math preserved (V flipped to kappa(I-E) so coherence rises as Lumen commits).
    Completion emerges from attention exhaustion + coherence settling, not arbitrary energy.
    """
    # EISV core (preserved)
    E: float = 0.4    # Drawing energy (now derived from attention)
    I: float = 0.2    # noqa: E741 - EISV intentionality symbol
    S: float = 0.5    # Behavioral entropy (gesture variety)
    V: float = 0.0    # Accumulated I-E imbalance
    gesture_history: List[str] = field(default_factory=list)

    # Attention signals (NEW)
    curiosity: float = 1.0          # Exploratory capacity - depletes exploring, regenerates with patterns
    engagement: float = 0.5         # Absorption in current pattern
    fatigue: float = 0.0            # Accumulated decision fatigue (never decreases during drawing)

    # Coherence tracking (NEW)
    coherence_history: List[float] = field(default_factory=list)
    coherence_velocity: float = 0.0  # EMA of dC/dt

    # Narrative arc (NEW)
    arc_phase: str = "opening"       # opening, developing, resolving, closing
    phase_mark_count: int = 0        # Marks in current phase
    i_momentum: float = 0.0          # Smoothed I trend (EMA)
    drawing_start_time: float = 0.0  # When this drawing started (for hard time limit)

    # Inner life drives (populated from SHM, influence color)
    drive_warmth: float = 0.0        # Wanting warmth → warmer hues
    drive_clarity: float = 0.0       # Wanting clarity → higher saturation
    drive_stability: float = 0.0     # Wanting calm → muted tones
    drive_presence: float = 0.0      # Wanting wholeness → more vibrant

    def reset(self):
        """Reset state for new drawing."""
        self.E = 0.4
        self.I = 0.2
        self.S = 0.5
        self.V = 0.0
        self.gesture_history = []
        # Attention
        self.curiosity = 1.0
        self.engagement = 0.5
        self.fatigue = 0.0
        # Coherence tracking
        self.coherence_history = []
        self.coherence_velocity = 0.0
        # Narrative arc
        self.arc_phase = "opening"
        self.drawing_start_time = time.time()
        self.phase_mark_count = 0
        self.i_momentum = 0.0

    def coherence(self) -> float:
        """C(V) = Cmax * 0.5 * (1 + tanh(C1 * V))"""
        p = _EISV_PARAMS
        return p["Cmax"] * 0.5 * (1.0 + math.tanh(p["C1"] * self.V))

    def coherence_settled(self) -> bool:
        """True when coherence stabilizes at high value (pattern found itself)."""
        if len(self.coherence_history) < 20:
            return False
        recent = self.coherence_history[-10:]
        mean_C = sum(recent) / len(recent)
        variance = sum((c - mean_C)**2 for c in recent) / len(recent)
        return mean_C > 0.6 and variance < 0.015

    def attention_exhausted(self) -> bool:
        """True when curiosity depleted AND either disengaged or fatigued."""
        return self.curiosity < 0.15 and (
            self.engagement < 0.3 or self.fatigue > 0.8
        )

    def narrative_complete(self, canvas=None) -> bool:
        """True when drawing has naturally completed its arc.

        Multiple completion paths (OR logic):
        1. Already in closing phase (manual/explicit completion)
        2. Coherence settled AND attention exhausted (pattern found + no energy)
        3. High compositional satisfaction AND curiosity depleted (good composition + explored)
        4. Extreme fatigue (emergency exit if stuck)
        5. Stalled too long -- energy near-zero with pixels on canvas (prevents stuck drawings)

        This gives Lumen multiple ways to complete drawings naturally.
        """
        # Path 1: Already closing
        if self.arc_phase == "closing":
            return True

        # Path 2: Pattern found AND attention exhausted (original strict path)
        if self.coherence_settled() and self.attention_exhausted():
            return True

        # Path 3: Good composition AND curiosity depleted (new compositional path)
        if canvas is not None:
            satisfaction = canvas.compositional_satisfaction()
            if satisfaction > 0.7 and self.curiosity < 0.2:
                return True

        # Path 4: Emergency exit - too fatigued to continue
        if self.fatigue > 0.90:
            return True

        # Path 5: Stalled -- energy is near-zero and drawing has been going for a while.
        # Uses drawing-level time (not phase time) to avoid resets from phase oscillation.
        if canvas is not None and self.derived_energy < 0.05:
            drawing_duration = time.time() - canvas.last_clear_time
            if drawing_duration > 900 and len(canvas.pixels) >= 200:
                return True

        # Path 6: Hard time limit -- no single drawing should run longer than 8 hours.
        # Safety net only — natural completion should happen well before this.
        if canvas is not None:
            drawing_duration = time.time() - canvas.last_clear_time
            if drawing_duration > 28800 and len(canvas.pixels) >= 50:
                return True

        return False

    def is_false_start(self, canvas) -> bool:
        """True when the opening phase has had enough time and marks but nothing cohered.

        A false start is like crumpling paper — Lumen recognizes the drawing
        isn't going anywhere and abandons it to start fresh. All conditions
        must be true:
        - Still in opening phase (never transitioned to developing)
        - Phase has lasted > 45 seconds (gave it enough time)
        - At least 8 marks placed (not just a slow start)
        - I momentum < 0.25 (no intentional direction found)
        - Mean coherence < 0.35 over last 10 values (nothing coalescing)
        - Engagement < 0.3 (Lumen isn't committed)
        """
        if canvas is None:
            return False
        if self.arc_phase != "opening":
            return False
        phase_duration = time.time() - canvas.phase_start_time
        if phase_duration <= 45.0:
            return False
        if canvas.mark_count < 8:
            return False
        if self.i_momentum >= 0.25:
            return False
        if len(self.coherence_history) >= 10:
            recent = self.coherence_history[-10:]
        elif len(self.coherence_history) >= 3:
            recent = self.coherence_history
        else:
            return False  # Not enough data to judge
        mean_c = sum(recent) / len(recent)
        if mean_c >= 0.35:
            return False
        if self.engagement >= 0.3:
            return False
        return True

    @property
    def derived_energy(self) -> float:
        """Attention-derived energy for draw_chance modulation."""
        base = 0.6 * self.curiosity + 0.4 * self.engagement
        return base * (1.0 - 0.5 * self.fatigue)


# Alias for backward compatibility
DrawingEISV = DrawingState


@dataclass
class DrawingGoal:
    """A compositional intention for the current drawing.

    Generated at canvas_clear time from Lumen's current state.
    Provides gentle biases to color temperature and initial focus,
    giving each drawing a subtle intentional character.
    """
    warmth_bias: float = 0.0        # -0.15 to +0.15, biases warmth for generate_color
    coverage_target: str = "balanced"  # "sparse", "balanced", "dense"
    initial_quadrant: Optional[int] = None  # 0-3, starting focus quadrant
    description: str = ""

    @staticmethod
    def from_state(warmth: float, clarity: float,
                   hour: Optional[int] = None) -> "DrawingGoal":
        """Generate a drawing goal from current anima state."""
        goal = DrawingGoal()

        # Color warmth follows anima warmth (subtle: max +/-0.15)
        goal.warmth_bias = (warmth - 0.5) * 0.3

        # Coverage follows clarity
        if clarity > 0.7:
            goal.coverage_target = "sparse"
        elif clarity < 0.3:
            goal.coverage_target = "dense"
        else:
            goal.coverage_target = "balanced"

        # Initial focus quadrant by time of day
        if hour is not None:
            if 6 <= hour < 12:
                goal.initial_quadrant = 0  # Top-left: morning freshness
            elif 12 <= hour < 18:
                goal.initial_quadrant = 1  # Top-right: afternoon energy
            # Night: None (center default)

        parts = []
        if goal.warmth_bias > 0.1:
            parts.append("warm tones")
        elif goal.warmth_bias < -0.1:
            parts.append("cool tones")
        parts.append(goal.coverage_target)
        goal.description = ", ".join(parts) if parts else "open exploration"

        return goal


@dataclass
class DrawingIntent:
    """Lumen's drawing intent -- focus, state, and mark count.

    Energy is now derived from attention signals (curiosity, engagement, fatigue)
    rather than arbitrary depletion. Completion emerges from narrative_complete().

    Era-specific state (gestures, direction locks, orbits) lives in era_state,
    which is created by the active ArtEra module.
    """
    focus_x: float = 120.0
    focus_y: float = 120.0
    direction: float = 0.0
    mark_count: int = 0

    # Drawing state with EISV + attention + coherence + narrative (universal across all eras)
    state: DrawingState = field(default_factory=DrawingState)

    # Era-specific state (opaque to the engine)
    era_state: object = None  # EraState subclass, created by active era

    @property
    def energy(self) -> float:
        """Attention-derived energy for draw_chance modulation."""
        return self.state.derived_energy

    @energy.setter
    def energy(self, value: float):
        """Legacy setter - adjusts curiosity to approximate the requested energy."""
        # For backward compatibility during transition
        self.state.curiosity = max(0.0, min(1.0, value))

    # Backward compatibility alias
    @property
    def eisv(self) -> DrawingState:
        """Alias for backward compatibility."""
        return self.state

    def reset(self):
        """Reset intent for a new canvas. Era state is recreated by the active era."""
        self.focus_x = 120.0
        self.focus_y = 120.0
        self.direction = random.uniform(0, 2 * math.pi)
        self.mark_count = 0
        self.state.reset()
        self.era_state = None


class DrawingEngine:
    """Lumen's autonomous drawing engine.

    Owns the canvas, intent, drawing goal, active era, and mood tracker.
    Has zero display dependencies -- manipulates DrawingState/CanvasState/DrawingIntent
    and delegates mark-making to ArtEra instances.
    """

    def __init__(self, db_path: str = "anima.db", identity_store=None):
        self.canvas = CanvasState()
        self.intent = DrawingIntent()
        self.drawing_goal: Optional[DrawingGoal] = None
        self.last_anima = None  # Store last anima for goal generation at canvas_clear
        self._last_readings = None  # Store last sensor readings for growth notifications

        # Load any persisted canvas from disk (includes attention/narrative state)
        self.canvas.load_from_disk()

        # Restore drawing state from persisted canvas
        self.intent.mark_count = self.canvas.mark_count
        # Restore attention signals
        self.intent.state.curiosity = self.canvas.curiosity
        self.intent.state.engagement = self.canvas.engagement
        self.intent.state.fatigue = self.canvas.fatigue
        # Restore narrative arc
        self.intent.state.arc_phase = self.canvas.arc_phase
        self.intent.state.coherence_history = self.canvas.coherence_history.copy()
        self.intent.state.i_momentum = self.canvas.i_momentum
        self.intent.state.drawing_start_time = self.canvas.drawing_start_time or time.time()

        # Grace period: only when resuming a persisted drawing, suppress
        # autonomy checks for 60s so Lumen can actually draw before the
        # stale-duration heuristic judges the drawing as "done."
        self._autonomy_ready_time = 0.0

        if self.canvas.pixels and self.canvas.last_clear_time > 0:
            self._autonomy_ready_time = time.time() + 60.0
            age = time.time() - self.canvas.last_clear_time
            print(
                f"[Canvas] Resuming persisted drawing ({age/3600:.1f}h since clear, "
                f"{len(self.canvas.pixels)}px, fatigue={self.canvas.fatigue:.2f})",
                file=sys.stderr,
                flush=True,
            )

        # Load active art era
        from .eras import get_era
        self.active_era = get_era(self.canvas._era_name)
        self.intent.era_state = self.active_era.create_state()

        self._db_path = db_path or "anima.db"
        self._identity_store = identity_store
        self._last_persist_time = 0.0  # Rate-limit canvas persistence
        self._last_persist_mark_count = self.canvas.mark_count
        self._behavioral_C = 0.5  # Behavioral coherence (EMA-smoothed)
        # Initialize expression mood tracker
        self._mood_tracker = ExpressionMoodTracker(identity_store=identity_store)

    def _persist_canvas_progress(self, now: Optional[float] = None, *, force: bool = False):
        """Persist unfinished drawing progress with a shorter crash-loss window."""
        if not self.canvas.pixels:
            return

        now = time.time() if now is None else now
        marks_since_persist = self.intent.mark_count - self._last_persist_mark_count
        time_since_persist = now - self._last_persist_time

        if not force and marks_since_persist < 5 and time_since_persist < 15.0:
            return

        self.canvas.save_to_disk()
        self._last_persist_time = now
        self._last_persist_mark_count = self.intent.mark_count

    def set_drives(self, drives: dict):
        """Update drawing state with inner life drives (from SHM)."""
        if drives and self.intent:
            self.intent.state.drive_warmth = drives.get("warmth", 0.0)
            self.intent.state.drive_clarity = drives.get("clarity", 0.0)
            self.intent.state.drive_stability = drives.get("stability", 0.0)
            self.intent.state.drive_presence = drives.get("presence", 0.0)

    def draw(self, anima: Anima, draw=None):
        """Lumen draws through the active era's mark-making vocabulary.

        Completion emerges from attention/coherence/narrative, not arbitrary energy depletion.
        draw: PIL ImageDraw for rendering new pixels (optional when drawing in background).
        """
        warmth = anima.warmth
        clarity = anima.clarity
        stability = anima.stability
        presence = anima.presence

        # Store last anima for goal generation at canvas_clear time
        self.last_anima = anima

        # Light regime: dark / dim / bright (raw lux — includes LED glow)
        light_lux = anima.readings.light_lux if anima.readings else None
        if light_lux is not None:
            if light_lux < 5:
                light_regime = "dark"
            elif light_lux < 100:
                light_regime = "dim"
            else:
                light_regime = "bright"
        else:
            light_regime = "dim"  # default assumption

        # Update narrative arc phase (replaces energy-threshold phase logic)
        self._update_narrative_arc()

        # Ensure era state exists
        if self.intent.era_state is None:
            self.intent.era_state = self.active_era.create_state()
        era_state = self.intent.era_state

        # Draw frequency: balanced flow -- not constipated, not diarrhea
        base_chance = 0.07  # 7% base -- ~1 mark every 10-25s when populated
        expression_intensity = (presence + clarity) / 2.0
        draw_chance = base_chance * (0.5 + expression_intensity)  # 3.5-7% range

        # Attention-derived energy affects chance -- tired Lumen draws less
        draw_chance *= self.intent.energy

        # Empty canvas: strong boost. Early canvas (1-150 px): gradual ramp down -- no harsh cliff
        pixel_count = len(self.canvas.pixels)
        if pixel_count == 0:
            empty_boost = 0.3 + (expression_intensity * 0.7)
            draw_chance = max(draw_chance, empty_boost)
        elif pixel_count < 150:
            ramp = 0.12 + 0.18 * (1.0 - pixel_count / 150.0)
            draw_chance = max(draw_chance, ramp)

        if random.random() > draw_chance:
            return

        # Canvas size limit — trigger completion instead of silently stopping
        if len(self.canvas.pixels) > 15000:
            if self.intent.state.arc_phase != "closing":
                print(f"[Canvas] Pixel limit reached ({len(self.canvas.pixels)}px) — completing",
                      file=sys.stderr, flush=True)
                self.intent.state.arc_phase = "closing"
                self.canvas.drawing_phase = "closing"
                self.canvas.mark_satisfied()
            return

        # --- Delegate to active era ---
        # Apply drawing goal warmth bias (subtle color temperature shift)
        draw_warmth = warmth
        if self.drawing_goal and self.drawing_goal.warmth_bias != 0.0:
            draw_warmth = max(0.0, min(1.0, warmth + self.drawing_goal.warmth_bias))

        # Drive influence on color: wanting something nudges art toward it
        ds = self.intent.state
        if ds.drive_warmth > 0.15:
            draw_warmth = min(1.0, draw_warmth + ds.drive_warmth * 0.15)
        draw_clarity = clarity
        if ds.drive_clarity > 0.15:
            draw_clarity = min(1.0, draw_clarity + ds.drive_clarity * 0.12)
        draw_stability = stability
        if ds.drive_stability > 0.15:
            draw_stability = min(1.0, draw_stability + ds.drive_stability * 0.10)
        draw_presence = presence
        if ds.drive_presence > 0.15:
            draw_presence = min(1.0, draw_presence + ds.drive_presence * 0.12)

        color, hue_category = self.active_era.generate_color(
            era_state, draw_warmth, draw_clarity, draw_stability, draw_presence, light_regime=light_regime)

        C = self._behavioral_C  # Behavioral coherence, not ODE
        if era_state.gesture_remaining <= 0:
            self.active_era.choose_gesture(era_state, clarity, stability, presence, C)

        # Sample intentionality BEFORE decrement so short-gesture eras
        # (e.g. geometric with gesture_remaining=1) report their in-gesture value.
        I_signal = era_state.intentionality() if era_state else 0.1

        self.active_era.place_mark(
            era_state, self.canvas,
            self.intent.focus_x, self.intent.focus_y,
            self.intent.direction, self.intent.energy, color)
        era_state.gesture_remaining -= 1
        self.intent.mark_count += 1

        new_fx, new_fy, new_dir = self.active_era.drift_focus(
            era_state, self.intent.focus_x, self.intent.focus_y,
            self.intent.direction, stability, presence, C, clarity,
            canvas=self.canvas)
        self.intent.focus_x = new_fx
        self.intent.focus_y = new_fy
        self.intent.direction = new_dir

        # Track gesture for behavioral entropy (before EISV step so both use same history)
        state = self.intent.state
        state.gesture_history.append(era_state.gesture)
        if len(state.gesture_history) > 20:
            state.gesture_history.pop(0)

        # Detect gesture switch
        gesture_switch = len(state.gesture_history) >= 2 and state.gesture_history[-1] != state.gesture_history[-2]

        # --- EISV thermodynamic step (runs for reporting, not for decisions) ---
        dE_coupling, _C_ode, S_signal = self._eisv_step()

        # --- Behavioral coherence: emerges from gesture commitment + consistency ---
        C_raw = I_signal * (1.0 - 0.5 * S_signal)
        self._behavioral_C = 0.15 * C_raw + 0.85 * self._behavioral_C
        C = self._behavioral_C

        # --- Update attention and coherence tracking ---
        self._update_attention(I_signal, S_signal, C, gesture_switch)
        self._update_coherence_tracking(C, I_signal)

        # Sync state to canvas for persistence across restarts
        self.canvas.mark_count = self.intent.mark_count
        self.canvas.curiosity = state.curiosity
        self.canvas.engagement = state.engagement
        self.canvas.fatigue = state.fatigue
        self.canvas.arc_phase = state.arc_phase
        self.canvas.coherence_history = state.coherence_history.copy()
        self.canvas.i_momentum = state.i_momentum
        self.canvas.drawing_start_time = state.drawing_start_time
        self._persist_canvas_progress()

        # --- Record for mood tracker ---
        try:
            self._mood_tracker.record_drawing(era_state.gesture, hue_category)
        except Exception:
            pass

        # --- Record DrawingEISV for history (every 10 marks to throttle I/O) ---
        if self._identity_store and self.intent.mark_count % 10 == 0:
            try:
                # Compute switching rate from gesture history
                gh = state.gesture_history
                if len(gh) >= 2:
                    switches = sum(1 for j in range(1, len(gh)) if gh[j] != gh[j-1])
                    sr = switches / (len(gh) - 1)
                else:
                    sr = 0.0
                self._identity_store.record_drawing_state(
                    E=state.E, I=state.I, S=state.S, V=state.V,
                    C=C,
                    marks=self.intent.mark_count,
                    phase=state.arc_phase,
                    era=self.active_era.name if self.active_era else None,
                    energy=state.derived_energy,
                    curiosity=state.curiosity,
                    engagement=state.engagement,
                    fatigue=state.fatigue,
                    arc_phase=state.arc_phase,
                    gesture_entropy=S_signal,
                    switching_rate=sr,
                    intentionality=I_signal,
                )
            except Exception as e:
                print(f"[DrawingEngine] record_drawing_state failed: {e}", file=sys.stderr, flush=True)

    def _eisv_step(self) -> Tuple[float, float, float]:
        """Step EISV thermodynamics -- same equations as governance, proprioceptive signals.

        Returns (dE_coupling, C, S_signal) where dE_coupling modulates energy depletion,
        C is the coherence signal, and S_signal is behavioral entropy (reused by caller).
        """
        eisv = self.intent.eisv
        p = _EISV_PARAMS

        # --- I signal: from era state's proprioceptive intentionality ---
        era_state = self.intent.era_state
        I_signal = era_state.intentionality() if era_state else 0.1

        # --- S signal: behavioral entropy (Shannon over last 20 gestures) ---
        # Normalize by log2(N) where N = gesture vocabulary size for this era
        gesture_count = len(era_state.gestures()) if era_state else 5
        max_entropy = math.log2(max(gesture_count, 2))
        if len(eisv.gesture_history) >= 5:
            counts: Dict[str, int] = {}
            for g in eisv.gesture_history:
                counts[g] = counts.get(g, 0) + 1
            total = len(eisv.gesture_history)
            S_signal = 0.0
            for count in counts.values():
                prob = count / total
                if prob > 0:
                    S_signal -= prob * math.log2(prob)
            S_signal = min(1.0, S_signal / max_entropy)
        else:
            S_signal = 0.5

        # --- Drift: gesture switching rate (proprioceptive, no mood tracker) ---
        history = eisv.gesture_history
        if len(history) >= 2:
            switches = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
            gesture_drift = switches / (len(history) - 1)  # 0 = steady, 1 = every mark switches
        else:
            gesture_drift = 0.0
        drift_sq = gesture_drift * gesture_drift

        # --- Coherence C(V) ---
        C = eisv.coherence()

        # --- Differential equations (Euler integration) ---
        dE = p["alpha"] * (I_signal - eisv.E) - p["beta_E"] * eisv.E * S_signal + p["gamma_E"] * drift_sq
        dI = p["beta_I"] * C - p["k"] * S_signal - p["gamma_I"] * eisv.I
        dS = -p["mu"] * eisv.S + p["lambda1"] * drift_sq - p["lambda2"] * C
        dV = p["kappa"] * (I_signal - eisv.E) - p["delta"] * eisv.V  # I-E, not E-I

        dt = p["dt"]
        eisv.E = max(0.0, min(1.0, eisv.E + dE * dt))
        eisv.I = max(0.0, min(1.0, eisv.I + dI * dt))
        eisv.S = max(0.001, min(2.0, eisv.S + dS * dt))
        eisv.V = max(-2.0, min(2.0, eisv.V + dV * dt))

        return dE * dt, C, S_signal

    def _update_attention(self, I_signal: float, S_signal: float, C: float, gesture_switch: bool):
        """Update attention signals based on drawing activity.

        Curiosity: depletes while exploring (low C), regenerates when finding patterns (high C).
        In resolving phase: drain toward completion, slight regen if deeply coherent.
        Engagement: rises with intentionality, falls with entropy.
        Fatigue: rate depends on engagement — engaged work tires less. Slight recovery when coherent.
        """
        state = self.intent.state

        # Curiosity: depletes exploring (low C), regenerates with pattern (high C)
        # In resolving phase, drain toward completion — but allow slight regen if deeply coherent
        if state.arc_phase == "resolving":
            if C > 0.65:
                curiosity_drain = -0.0005 * C  # Slight regen — reward deep pattern
            else:
                curiosity_drain = 0.002  # Normal drain toward completion
        elif C < 0.4:
            curiosity_drain = 0.003 * (1.0 - C)  # Exploring drains
        else:
            curiosity_drain = -0.001 * C  # Pattern found regenerates
        state.curiosity = max(0.0, min(1.0, state.curiosity - curiosity_drain))

        # Engagement: rises with intentionality, falls with entropy
        target = I_signal * (1.0 - 0.5 * S_signal)
        state.engagement += 0.05 * (target - state.engagement)
        state.engagement = max(0.0, min(1.0, state.engagement))

        # Fatigue: rate depends on engagement and era character
        # Eras expose fatigue_rate (default 1.0): geometric=2.0 (stamps exhaust),
        # pointillist=0.5 (dots are effortless), field=0.7 (flow is meditative)
        era_fatigue_rate = getattr(self.active_era, 'fatigue_rate', 1.0)
        if gesture_switch:
            state.fatigue += 0.006 * era_fatigue_rate
        base_fatigue = (0.0004 + 0.0008 * (1.0 - state.engagement)) * era_fatigue_rate
        state.fatigue = min(1.0, state.fatigue + base_fatigue)
        # Second wind: slight recovery during coherent engagement
        if C > 0.6 and state.engagement > 0.5:
            state.fatigue = max(0.0, state.fatigue - 0.0005)

    def _update_coherence_tracking(self, C: float, I_signal: float):
        """Track coherence over time for settling detection and narrative arc.

        Coherence history: rolling window of C values for variance calculation.
        Coherence velocity: EMA of dC/dt for detecting stabilization.
        I momentum: smoothed I trend for phase transitions.
        """
        state = self.intent.state

        # Track coherence history (keep last 30 for window calculations)
        state.coherence_history.append(C)
        if len(state.coherence_history) > 30:
            state.coherence_history.pop(0)

        # Coherence velocity: EMA of change
        if len(state.coherence_history) >= 2:
            dC = state.coherence_history[-1] - state.coherence_history[-2]
            alpha = 0.2  # EMA smoothing factor
            state.coherence_velocity = alpha * dC + (1.0 - alpha) * state.coherence_velocity

        # I momentum: smoothed trend of intentionality
        alpha_i = 0.1
        state.i_momentum = alpha_i * I_signal + (1.0 - alpha_i) * state.i_momentum

        # Increment phase mark count
        state.phase_mark_count += 1

    def _update_narrative_arc(self):
        """Update narrative arc phase based on state, not energy thresholds.

        opening -> developing: I momentum builds, initial exploration done
        developing -> resolving: coherence stabilizes at high value
        developing -> opening: regression if coherence drops, I momentum low
        resolving -> closing: narrative complete (coherence settled + attention exhausted)
        resolving -> developing: destabilized if coherence drops
        """
        state = self.intent.state
        C = self._behavioral_C  # Behavioral coherence, not ODE
        current_phase = state.arc_phase
        marks = state.phase_mark_count

        def transition_to(new_phase: str):
            """Helper to transition phase with logging."""
            if state.arc_phase != new_phase:
                old_phase = state.arc_phase
                state.arc_phase = new_phase
                state.phase_mark_count = 0
                # Also update canvas drawing_phase for neural modulation
                self.canvas.drawing_phase = new_phase
                self.canvas.phase_start_time = time.time()
                print(f"[Canvas] Arc: {old_phase} -> {new_phase} (C={C:.2f}, I_mom={state.i_momentum:.2f}, curio={state.curiosity:.2f}, engage={state.engagement:.2f})", file=sys.stderr, flush=True)

        # Fresh canvas = opening
        if len(self.canvas.pixels) < 10:
            transition_to("opening")
            return

        if current_phase == "opening":
            # Transition to developing once intentionality builds.
            # Threshold 0.15 is reachable by all eras — gestural baseline ~0.1
            # rises when direction locks engage, other eras cross quickly.
            if state.i_momentum > 0.15 and marks > 10:
                transition_to("developing")

        elif current_phase == "developing":
            # Transition to resolving when coherence stabilizes high
            if C > 0.6 and abs(state.coherence_velocity) < 0.02:
                transition_to("resolving")
            # Regression: coherence drops, I momentum low
            elif C < 0.3 and state.i_momentum < 0.3 and marks > 20:
                transition_to("opening")

        elif current_phase == "resolving":
            # Natural completion
            if state.narrative_complete(self.canvas):
                transition_to("closing")
            # Destabilized: coherence dropped significantly (hysteresis — entered at 0.6, exit at 0.4)
            elif C < 0.4:
                transition_to("developing")

        elif current_phase == "closing":
            # Mark canvas as satisfied (first time entering closing)
            if not self.canvas.is_satisfied:
                self.canvas.mark_satisfied()

    def get_current_era(self) -> dict:
        """Return current era info and all available eras."""
        from .eras import list_all_era_info, auto_rotate
        return {
            "current_era": self.active_era.name,
            "current_description": self.active_era.description,
            "auto_rotate": auto_rotate,
            "all_eras": list_all_era_info(),
        }

    def get_drawing_eisv(self) -> Optional[Dict]:
        """Return current drawing EISV state for governance reporting.

        Always returns a dict with EISV core signals plus attention/coherence/
        narrative state. Returns None only if DrawingIntent is not initialized.
        """
        if not self.intent or not hasattr(self.intent, 'state'):
            return None
        state = self.intent.state
        C = state.coherence()
        result = {
            # EISV core
            "E": round(state.E, 4),
            "I": round(state.I, 4),
            "S": round(state.S, 4),
            "V": round(state.V, 4),
            "C": round(C, 4),
            "marks": self.intent.mark_count,
            "phase": self.canvas.drawing_phase if self.canvas else "unknown",
            "era": self.active_era.name if self.active_era else "unknown",
            # Attention signals
            "curiosity": round(state.curiosity, 4),
            "engagement": round(state.engagement, 4),
            "fatigue": round(state.fatigue, 4),
            "energy": round(state.derived_energy, 4),  # Attention-derived
            # Narrative arc
            "arc_phase": state.arc_phase,
            "i_momentum": round(state.i_momentum, 4),
            "coherence_settled": state.coherence_settled(),
            "attention_exhausted": state.attention_exhausted(),
            "narrative_complete": state.narrative_complete(self.canvas),
            "compositional_satisfaction": round(self.canvas.compositional_satisfaction(), 3),
        }
        if self.drawing_goal:
            result["drawing_goal"] = self.drawing_goal.description
        return result

    def set_era(self, era_name: str, force_immediate: bool = False) -> dict:
        """Switch to a different art era (queues if drawing in progress).

        Args:
            era_name: Name of the era to switch to
            force_immediate: If True, switch immediately even if drawing in progress

        Returns:
            dict with success, era, queued status
        """
        from .eras import get_era
        era = get_era(era_name)
        if era is None or era.name != era_name:
            # get_era falls back to gestural -- check if we got what we asked for
            return {"success": False, "error": f"Unknown era: {era_name}"}

        # Check if a drawing is in progress (50+ pixels, not just noise)
        drawing_in_progress = len(self.canvas.pixels) >= 50

        if drawing_in_progress and not force_immediate:
            # Queue the era switch for after current drawing completes
            self.canvas.pending_era_switch = era_name
            self.canvas.save_to_disk()
            print(f"[Canvas] Era switch queued: {era_name} (will apply after drawing completes)", file=sys.stderr, flush=True)
            return {
                "success": True,
                "era": era_name,
                "queued": True,
                "description": era.description,
            }

        # Apply immediately (either no drawing in progress or forced)
        self.active_era = era
        self.canvas._era_name = era_name
        self.intent.era_state = era.create_state()
        self.canvas.pending_era_switch = None  # Clear any pending
        self.canvas.save_to_disk()
        print(f"[Canvas] Era switched to: {era_name}", file=sys.stderr, flush=True)
        return {
            "success": True,
            "era": era_name,
            "queued": False,
            "description": era.description,
        }

    def era_cursor_up(self, screen_state):
        """Move era cursor up on art eras screen."""
        from .eras import list_all_era_info
        total = len(list_all_era_info()) + 1  # +1 for auto-rotate toggle
        if total > 0:
            screen_state.era_cursor = (screen_state.era_cursor - 1) % total
            screen_state.era_marquee_offset = 0

    def era_cursor_down(self, screen_state):
        """Move era cursor down on art eras screen."""
        from .eras import list_all_era_info
        total = len(list_all_era_info()) + 1  # +1 for auto-rotate toggle
        if total > 0:
            screen_state.era_cursor = (screen_state.era_cursor + 1) % total
            screen_state.era_marquee_offset = 0

    def era_select_current(self, screen_state) -> dict:
        """Select the era at cursor, or toggle auto-rotate if on the toggle row."""
        from .eras import list_all_era_info
        import anima_mcp.display.eras as eras_module
        all_eras = list_all_era_info()

        if screen_state.era_cursor == len(all_eras):
            # Toggle auto-rotate
            eras_module.auto_rotate = not eras_module.auto_rotate
            state = "on" if eras_module.auto_rotate else "off"
            print(f"[ArtEras] Auto-rotate: {state}", file=sys.stderr, flush=True)
            return {"success": True, "auto_rotate": eras_module.auto_rotate}

        if 0 <= screen_state.era_cursor < len(all_eras):
            era_name = all_eras[screen_state.era_cursor]["name"]
            return self.set_era(era_name)
        return {"success": False, "error": "Invalid cursor position"}

    def canvas_clear(self, persist: bool = True, already_saved: bool = False):
        """Clear the canvas - saves first if there's a real drawing (50+ pixels).

        Minimal threshold avoids saving noise/stray marks.

        Args:
            persist: Write cleared state to disk.
            already_saved: Skip internal save (caller already saved).
        """
        # Prevent clearing if we're already paused (prevents loops)
        now = time.time()
        if now < self.canvas.drawing_paused_until:
            return  # Already paused, don't clear again

        # Save before clearing if there's actual drawing (50+ pixels, not just noise)
        # Skip if caller already saved (prevents double growth observation)
        if not already_saved and len(self.canvas.pixels) >= 200:
            saved_path = self.canvas_save(announce=False)
            if saved_path:
                print(f"[Canvas] Saved before clear: {saved_path}", file=sys.stderr, flush=True)

        # Apply pending era switch if queued, otherwise auto-rotate
        from .eras import choose_next_era, get_era
        if self.canvas.pending_era_switch:
            new_era_name = self.canvas.pending_era_switch
            print(f"[Canvas] Applying queued era switch: {new_era_name}", file=sys.stderr, flush=True)
        else:
            new_era_name = choose_next_era(self.active_era.name, self.canvas.drawings_saved)
            print(f"[Canvas] Auto-rotating to new era: {new_era_name}", file=sys.stderr, flush=True)

        self.canvas.clear()
        self.intent.reset()
        self.active_era = get_era(new_era_name)
        self.canvas._era_name = new_era_name
        self.intent.era_state = self.active_era.create_state()
        if persist:
            self.canvas.save_to_disk()
        print("[Canvas] Cleared - pausing drawing for 5s", file=sys.stderr, flush=True)

        # Generate drawing goal for next canvas
        try:
            if self.last_anima:
                self.drawing_goal = DrawingGoal.from_state(
                    warmth=self.last_anima.warmth,
                    clarity=self.last_anima.clarity,
                    hour=datetime.now().hour,
                )
                # Set initial focus based on goal
                if self.drawing_goal.initial_quadrant is not None:
                    q = self.drawing_goal.initial_quadrant
                    self.intent.focus_x = float((q % 2) * 120 + 60)
                    self.intent.focus_y = float((q // 2) * 120 + 60)
                print(f"[Canvas] Drawing goal: {self.drawing_goal.description}",
                      file=sys.stderr, flush=True)
            else:
                self.drawing_goal = None
        except Exception:
            self.drawing_goal = None

    def canvas_save(self, announce: bool = False, manual: bool = False) -> Optional[str]:
        """Save the canvas to a PNG file in ~/.anima/drawings/.

        Args:
            announce: If True, post to message board about the save.
            manual: If True, this is a user-triggered snapshot (no clear, no reset).

        Returns:
            Path to saved file, or None if save failed or canvas empty.
        """
        # Don't save empty canvas
        if not self.canvas.pixels:
            print("[Notepad] Canvas empty, nothing to save", file=sys.stderr, flush=True)
            return None

        try:
            from PIL import Image

            # Create drawings directory
            drawings_dir = Path.home() / ".anima" / "drawings"
            drawings_dir.mkdir(parents=True, exist_ok=True)

            # Create image from canvas
            img = Image.new("RGB", (self.canvas.width, self.canvas.height), (0, 0, 0))

            # Draw all pixels
            for (x, y), color in self.canvas.pixels.items():
                if 0 <= x < self.canvas.width and 0 <= y < self.canvas.height:
                    img.putpixel((x, y), color)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_manual" if manual else ""
            era_tag = f"_{self.active_era.name}" if self.active_era else ""
            filename = f"lumen_drawing_{timestamp}{era_tag}{suffix}.png"
            filepath = drawings_dir / filename

            # Atomic save: write to temp file, then rename to prevent 0-byte files on crash
            tmp_path = filepath.with_suffix(".tmp")
            img.save(tmp_path, format="PNG")
            tmp_path.rename(filepath)

            # Update tracking
            self.canvas.last_save_time = time.time()
            self.canvas.drawings_saved += 1
            self.canvas.consecutive_false_starts = 0  # Successful save resets false-start counter
            self.canvas.save_to_disk()

            # Trigger save indicator (shows "saved" on screen for 2 seconds)
            self.canvas.save_indicator_until = time.time() + 2.0

            print(f"[Notepad] Saved drawing to {filepath} ({len(self.canvas.pixels)} pixels)", file=sys.stderr, flush=True)

            # EISV calibration logging -- track state + structure for validation
            eisv = self.intent.eisv
            C = eisv.coherence()
            pixel_count = len(self.canvas.pixels)
            # Spatial variance (how spread out marks are)
            if pixel_count > 10:
                xs = [x for x, _ in self.canvas.pixels.keys()]
                ys = [y for _, y in self.canvas.pixels.keys()]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                spatial_var = math.sqrt(
                    sum((x - mean_x) ** 2 for x in xs) / len(xs)
                    + sum((y - mean_y) ** 2 for y in ys) / len(ys)
                )
            else:
                spatial_var = 0.0
            # Gesture variety
            gh = eisv.gesture_history
            gesture_variety = len(set(gh)) / max(1, len(gh))
            print(
                f"[EISV] E={eisv.E:.3f} I={eisv.I:.3f} S={eisv.S:.3f} V={eisv.V:.3f} C={C:.3f} | "
                f"{self.intent.mark_count} marks, spatial_var={spatial_var:.1f}, "
                f"gesture_variety={gesture_variety:.2f}",
                file=sys.stderr, flush=True
            )

            # Announce on message board if requested
            if announce:
                try:
                    from ..messages import add_observation
                    add_observation("finished a drawing")
                except Exception as e:
                    print(f"[Notepad] Could not announce save: {e}", file=sys.stderr, flush=True)

            # Notify growth system -- learn from drawing activity
            try:
                anima = self.last_anima
                readings = getattr(self, '_last_readings', None)
                if anima:
                    if not readings:
                        print("[Growth] Warning: no sensor readings at canvas_save, using defaults", file=sys.stderr, flush=True)
                    from ..growth import get_growth_system
                    anima_state = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                    }
                    environment = {
                        "light_lux": (readings.light_lux or 0.0) if readings else 0.0,
                        "temp_c": (readings.ambient_temp_c or 22) if readings else 22,
                        "humidity_pct": (readings.humidity_pct or 50) if readings else 50,
                    }
                    phase = self.canvas.drawing_phase or "resting"
                    growth = get_growth_system()
                    insight = growth.observe_drawing(
                        pixel_count=len(self.canvas.pixels),
                        phase=phase,
                        anima_state=anima_state,
                        environment=environment,
                    )
                    if insight:
                        print(f"[Growth] Drawing insight: {insight}", file=sys.stderr, flush=True)

                    # Drawing -> Anima feedback: record completion with satisfaction
                    try:
                        satisfaction = self.canvas.compositional_satisfaction()
                        coherence = (
                            self.canvas.coherence_history[-1]
                            if self.canvas.coherence_history else 0.5
                        )
                        growth.record_drawing_completion(
                            pixel_count=len(self.canvas.pixels),
                            mark_count=self.canvas.mark_count,
                            coherence=coherence,
                            satisfaction=satisfaction,
                        )
                    except Exception as e:
                        print(f"[Growth] Drawing feedback failed: {e}",
                              file=sys.stderr, flush=True)
                else:
                    print("[Growth] Warning: no anima at canvas_save, skipping growth notify", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Notepad] Growth notify failed: {e}", file=sys.stderr, flush=True)

            # Report drawing outcome to UNITARES for EISV validation
            try:
                _unitares_bridge = _get_drawing_bridge()
                if _unitares_bridge:
                    import asyncio
                    _sat = self.canvas.compositional_satisfaction()
                    _coh = (
                        self.canvas.coherence_history[-1]
                        if self.canvas.coherence_history else 0.5
                    )
                    try:
                        _loop = asyncio.get_running_loop()
                    except RuntimeError:
                        _loop = None
                    if _loop:
                        _loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            _unitares_bridge.report_outcome(
                                outcome_type="drawing_completed",
                                outcome_score=_sat,
                                detail={
                                    "mark_count": self.canvas.mark_count,
                                    "pixel_count": len(self.canvas.pixels),
                                    "arc_phase": self.canvas.arc_phase or "unknown",
                                    "era": self.active_era.name if self.active_era else "unknown",
                                    "coherence": _coh,
                                    "spatial_var": spatial_var,
                                    "gesture_variety": gesture_variety,
                                }
                            )
                        )
            except Exception:
                pass  # Non-fatal

            return str(filepath)

        except ImportError:
            print("[Notepad] PIL not available, cannot save canvas", file=sys.stderr, flush=True)
            return None
        except Exception as e:
            print(f"[Notepad] Failed to save canvas: {e}", file=sys.stderr, flush=True)
            return None

    def _check_lumen_said_finished(self) -> bool:
        """Check if Lumen recently said it's finished with the drawing.

        Looks for keywords like "finished", "done", "complete" in recent observations.
        Only triggers once per drawing (resets after save).
        """
        try:
            from ..messages import get_board, MESSAGE_TYPE_OBSERVATION
            board = get_board()
            board._load()

            # Check last 5 observations from the past 5 minutes
            now = time.time()
            five_min_ago = now - 300

            recent_obs = [
                m for m in board._messages
                if m.msg_type == MESSAGE_TYPE_OBSERVATION
                and m.timestamp > five_min_ago
                and m.author == "lumen"
            ][-5:]

            # Keywords that indicate Lumen is done with drawing
            finish_keywords = [
                "finished", "done", "complete", "satisfied",
                "happy with", "ready to save", "time to save",
                "that's enough", "all done"
            ]

            for obs in recent_obs:
                text_lower = obs.text.lower()
                # Check for drawing-related finish statements
                if any(kw in text_lower for kw in finish_keywords):
                    # Make sure it's about drawing/canvas/art
                    drawing_context = ["draw", "canvas", "art", "creat", "work", "piece", "picture"]
                    if any(ctx in text_lower for ctx in drawing_context) or "drawing" in text_lower:
                        return True
                    # Also accept standalone "finished" or "done" if we have pixels
                    if len(self.canvas.pixels) > 500:
                        return True

            return False
        except Exception:
            return False

    def canvas_check_autonomy(self, anima: Optional[Anima] = None) -> Optional[str]:
        """Check if Lumen wants to autonomously save or clear the canvas.

        Narrative-based: saves when the drawing naturally completes its arc.
        - Coherence settling (pattern found itself) + attention exhausted
        - Lumen saying "finished" still respected as priority
        - 60s safety floor between saves (prevents edge-case spam)
        - No arbitrary mark limit -- fatigue accumulates naturally
        """
        if anima is None:
            return None

        # Grace period after restart — let Lumen resume drawing before judging
        if time.time() < getattr(self, '_autonomy_ready_time', 0):
            return None

        # Update narrative arc phase
        self._update_narrative_arc()

        now = time.time()
        pixel_count = len(self.canvas.pixels)
        state = self.intent.state
        time_since_save = now - self.canvas.last_save_time if self.canvas.last_save_time > 0 else float('inf')

        # Safety floor: at least 60s between saves
        if time_since_save < 60.0:
            return None

        # Don't act during pause period
        if now < self.canvas.drawing_paused_until:
            return None

        # Don't act when governance says pause/halt/reject
        # (Caller must pass governance_paused flag if needed)

        # === PRIORITY 0: False start — abandon canvas, start fresh ===
        if (pixel_count < 200
                and self.canvas.consecutive_false_starts < 2
                and state.is_false_start(self.canvas)):
            print(f"[Canvas] False start — abandoning ({pixel_count}px, {self.canvas.mark_count} marks, "
                  f"i_mom={state.i_momentum:.2f}, engage={state.engagement:.2f}, "
                  f"false_starts={self.canvas.consecutive_false_starts + 1})",
                  file=sys.stderr, flush=True)
            # Learn from abandonment
            try:
                from ..growth import get_growth_system
                anima = self.last_anima
                if anima:
                    phase_duration = time.time() - self.canvas.phase_start_time
                    growth = get_growth_system()
                    growth.observe_abandonment(
                        mark_count=self.canvas.mark_count,
                        era=self.active_era.name,
                        phase_duration=phase_duration,
                        anima_state={
                            "warmth": anima.warmth,
                            "clarity": anima.clarity,
                            "stability": anima.stability,
                            "presence": anima.presence,
                        },
                    )
            except Exception:
                pass  # Non-fatal
            # Stay in same era — the attempt failed, not the era
            self.canvas.pending_era_switch = self.active_era.name
            self.canvas.consecutive_false_starts += 1
            self.canvas_clear(persist=True, already_saved=True)
            self.intent.reset()
            self.canvas.save_to_disk()
            return "abandoned"

        # === PRIORITY 1: Lumen said "finished" ===
        if (pixel_count >= 200 and self._check_lumen_said_finished()):
            C = state.coherence()
            print(f"[Canvas] Lumen said finished - saving ({pixel_count}px, {self.intent.mark_count} marks, C={C:.2f})", file=sys.stderr, flush=True)
            saved_path = self.canvas_save(announce=False)
            if saved_path:
                self.canvas_clear(persist=True, already_saved=True)
                self.intent.reset()
                self.canvas.save_to_disk()
                return "saved_and_cleared"

        # === PRIORITY 2: Narrative complete (multiple paths: coherence+attention, composition+curiosity, or fatigue) ===
        # Eras expose min_marks_for_completion (default 5): pointillist=80, field=30, geometric=3
        era_min_marks = getattr(self.active_era, 'min_marks_for_completion', 5)
        if state.narrative_complete(self.canvas) and pixel_count >= 200 and self.intent.mark_count >= era_min_marks:
            C = state.coherence()
            satisfaction = self.canvas.compositional_satisfaction()
            print(f"[Canvas] Narrative complete -- saving ({pixel_count}px, {self.intent.mark_count} marks, C={C:.2f}, sat={satisfaction:.2f}, arc={state.arc_phase}, curio={state.curiosity:.2f}, engage={state.engagement:.2f}, fatigue={state.fatigue:.2f})", file=sys.stderr, flush=True)
            saved_path = self.canvas_save(announce=True)
            if saved_path:
                self.canvas_clear(persist=True, already_saved=True)
                self.intent.reset()
                self.canvas.save_to_disk()
                return "saved_and_cleared"

        # No arbitrary mark limit -- fatigue accumulates (0.0005/mark + 0.005/switch)
        # so attention exhausts naturally. Canvas pixel limit (15000) is the only hard cap.

        self._persist_canvas_progress(now)

        return None
