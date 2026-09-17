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
from ..db_paths import resolve_db_path


_EARNED_COMPLETION_REASONS = frozenset({
    "earned_coherence", "earned_composition",
    # Era-supplied earned path (ArtEra.earned_completion) — e.g. resonance's
    # field-settled signal. The global paths are unreachable in some eras
    # (resonance V dynamics cap C ~0.53 < the 0.6 settle threshold), which
    # starved the earned-only autobiographical gate for the whole era.
    "earned_field",
    # Lumen declaring the piece done is the most self-determined completion
    # available — more so than any threshold. Previously this path reached the
    # gate as None and passed via the fail-open below, so treating it as earned
    # preserves its behavior rather than changing it.
    "said_finished",
    # The piece stopped changing while still being worked: N consecutive
    # actively-marked trajectory samples produced under SETTLED_FRAC_OF_PEAK of
    # this piece's own peak novelty rate. Self-relative by construction — the
    # threshold is the piece's own history, not a constant an era can't reach.
    "earned_settled",
})
MIN_RECORDED_DRAWING_PIXELS = 200

# How often to record a within-piece trajectory sample, in seconds. At the
# observed 8-hour piece length this is ~96 rows per drawing — enough resolution
# to locate where a piece plateaued, cheap enough to keep for a season.
TRAJECTORY_SAMPLE_INTERVAL = 300.0

# Self-relative settled completion, derived from the first 27 cap-length pieces
# of drawing_trajectory (2026-08-02..08-11). The corpus splits three ways:
# field pieces plateau settled (marks continue, novelty ~0) at 3.4-7.7h;
# gestural/pointillist genuinely keep changing to the 8h cap (a rule that never
# fires there is correct); geometric cap-pieces freeze idle (marks stop), which
# must NOT read as settled. Hence the streak counts only ACTIVE samples: novelty
# below threshold with marks landing advances it, novelty above resets it, and
# an idle or unknowable (post-restart) sample holds it — rest is not evidence.
# Earliest possible fire on the corpus ≈ 4.4h; the 8h cap stays as backstop, so
# miscalibration degrades to prior behavior (same posture as earned_field).
SETTLED_FRAC_OF_PEAK = 0.10
SETTLED_STREAK_SAMPLES = 12      # ~1h of actively-worked samples at 300s cadence
SETTLED_MIN_AGE_SECONDS = 7200.0  # never before 2h of piece age
SETTLED_MIN_MARKS = 100
SETTLED_SMOOTH_WINDOW = 3        # rolling mean over active samples
SETTLED_MIN_SAMPLE_MARKS = 2     # <2 marks/sample = trickle, holds not advances

# Frozen detection. Cap-length `geometric` pieces stop marking entirely after
# ~1h and then sit untouched for ~7h until the 8-hour cap. The cause is a closed
# loop, not a mistuned number: _update_attention runs once per PLACED MARK, so
# fatigue, curiosity and engagement only advance when a mark lands. Rising
# fatigue lowers derived_energy, `draw_chance *= energy` has no floor, marks
# become rare — and then the very state that could end the piece stops moving
# with them. Fatigue can no longer climb to the 0.90 bailout_fatigue, energy can
# no longer fall to the 0.05 bailout_stalled, and earned_settled correctly
# refuses because an idle sample HOLDS the streak (idle is not settled). Every
# exit is driven by a quantity that only advances when marks happen.
#
# So the detector must read the one thing still observable from outside that
# loop: marks have stopped, judged against THIS piece's own peak marks-per-
# interval. Self-relative in the rate, exactly as SETTLED_FRAC_OF_PEAK is — an
# absolute "fewer than N marks" would be unreachable for a slow era and trigger
# constantly for a fast one.
#
# ⛔ There is deliberately NO mark-count floor here. SETTLED_MIN_MARKS is 100 and
# geometric pieces reach ~70 marks total, so requiring it would make this gate
# unreachable for the one era it exists to rescue — the exact defect class this
# file keeps having to fix. Investment is gated by pixels and age instead, which
# a whole-shape-stamping era clears easily.
FROZEN_FRAC_OF_PEAK = 0.10   # a sample under 10% of own peak mark rate is idle
FROZEN_STREAK_SAMPLES = 12   # ~1h of consecutive idle samples at 300s cadence

# How many finished pieces the next one is drawn to differ from. A window, not
# a gate: it reads no signal and ends nothing, so it is not a threshold in the
# sense of design invariant 1. Six is roughly two days of production at the
# observed ~3 pieces/day, which is about as far back as "the same as what it
# has been making lately" means anything.
RECENT_DISPOSITIONS = 6


def is_earned_completion_reason(reason: Optional[str]) -> bool:
    """Gate for autobiographical writes tied to drawing completion.

    Earned reasons only, and `None` — unknown provenance — does NOT qualify.

    This used to return True for None so callers predating the reason-plumbing
    kept their old behavior. That fail-open was the whole bug: two of the three
    save paths never tagged a reason, so bail-outs arrived here as None and were
    written as "Made a drawing I'm pleased with". Because resonance canvases cap
    C ~0.52 they never enter the "resolving" arc phase where the tag was
    captured, and the only remaining guard — compositional_satisfaction > 0.7 —
    reads 0.78-0.86 always. Result: 18 of 19 large completions between
    2026-07-23 and 2026-07-29, every one an 8h cap or fatigue bail-out, became
    pride memories. Axiom 8 exists to prevent exactly that, so unknown
    provenance must fail closed: no tag, no autobiographical claim.

    All production callers now supply an explicit tag (see canvas priorities 1,
    1.5 and 2 in _lumen_draw).
    """
    return reason in _EARNED_COMPLETION_REASONS


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
    auto_rotate: bool = False  # Operator-selected era policy; survives service restarts

    # False-start tracking (volatile - resets on restart, not persisted)
    consecutive_false_starts: int = 0

    # Completion-path tag set when a drawing reaches the closing phase.
    # Distinguishes earned completion from bail-out exits so downstream
    # growth/memory systems can gate autobiographical writes on honest signal.
    # See completion_reason() for the taxonomy.
    last_completion_reason: Optional[str] = None
    # Serialized DrawingGoal for the current canvas — persisted so a service
    # restart mid-piece doesn't silently drop the piece's intention (goals
    # are only generated at canvas_clear, so a lost one stayed lost).
    drawing_goal_data: Optional[dict] = None

    # The last few finished pieces' dispositions — the per-piece globals each
    # era draws at create_state() (see EraState.disposition). Read when the
    # next piece starts, so it can be drawn unlike what Lumen has just been
    # making. This is the one loop from Lumen's own history back into Lumen's
    # behavior that closes without a human running a script.
    #
    # Each entry carries its "era", and comparisons are filtered to the SAME
    # era: a gestural disposition and a field one share no keys, so comparing
    # them would fall back on defaults and manufacture a distance that means
    # nothing. Cross-era, "unlike the last piece" is already true by
    # construction.
    recent_dispositions: List[dict] = field(default_factory=list)

    # Render caching - avoid redrawing all pixels every frame
    _dirty: bool = True  # Set by draw_pixel(), cleared after render
    _cached_image: object = None  # Cached PIL Image of all pixels
    _new_pixels: list = field(default_factory=list)  # Pixels added since last render

    # Spatial density grid — 8x8 cells (30px each) for spatial awareness
    density_grid: List[List[int]] = field(default_factory=lambda: [[0] * 8 for _ in range(8)])

    # Resonance memory field persistence (optional, list-of-lists when set)
    _resonance_field: object = None
    # Settling counters that feed earned_completion. The field beside them was
    # already persisted; these were not, so every restart reset revisit_window
    # to empty and settled_streak to 0. earned_field needs 50 deposits to
    # refill the window plus 5 consecutive settled checks, so a restart cadence
    # faster than that made the earned path structurally unreachable — measured
    # live 2026-07-30: 327 marks, revisit_window_filled 0/50.
    _resonance_settling: object = None
    # Novelty-settled tracker for the era-blind earned_settled path. Rides with
    # the canvas for the same reason as _resonance_settling above: a streak that
    # resets on restart makes the earned path unreachable at real-world restart
    # cadence. Dict: last_t/last_px/last_marks (own sample baseline, independent
    # of the trajectory DB write), recent (rolling novelty window), peak, streak.
    _novelty_settling: object = None

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
        self.last_completion_reason = None
        self.drawing_goal_data = None
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
        # Decay resonance field on clear (ghost of previous drawing)
        if self._resonance_field is not None:
            try:
                import numpy as np
                field = np.array(self._resonance_field, dtype=np.float32)
                field *= 0.3  # CLEAR_DECAY
                self._resonance_field = field.tolist()
            except Exception:
                self._resonance_field = None
        # Settling is about THIS piece — a new canvas starts unsettled, even
        # though the field carries a decayed ghost of the previous one.
        self._resonance_settling = None
        self._novelty_settling = None
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

    def _rebuild_density_grid(self) -> None:
        """Recompute the 8x8 density grid from the pixels themselves.

        `density_grid` is maintained incrementally by draw_pixel() and is NOT
        persisted, while `pixels` is — so every restart restored a full canvas
        beside an empty grid. Measured live 2026-08-02: a 9,955-pixel piece
        reported occupied_cells 0 and grid_entropy 0.0.

        This is the same shape as the resonance settling bug (#116): derived
        state living beside the thing it is derived from, and only one of them
        surviving a restart. Rebuilding is preferred to persisting because the
        grid then cannot drift from the pixels it describes.

        Consumers: occupied_cells(), grid_entropy(), and sparsest_cell() — which
        resonance uses to steer focus, so an empty grid did not merely mis-report
        structure, it aimed every post-restart drift at cell (0,0).
        """
        grid = [[0] * 8 for _ in range(8)]
        for (x, y) in self.pixels:
            grid[min(x // 30, 7)][min(y // 30, 7)] += 1
        self.density_grid = grid

    def piece_uid(self) -> str:
        """Stable identifier for the current piece, for joining samples to it.

        Keyed on last_clear_time because that is already persisted, already
        unique per piece, and already survives restarts — a counter would not.
        """
        return f"p{int(self.last_clear_time)}"

    def occupied_cells(self) -> int:
        """How many of the 64 density cells have any pixel at all.

        Structural reach, as opposed to pixel count. A piece that keeps opening
        new cells is still finding territory; one whose cell count has stopped
        moving is thickening what it already has.
        """
        return sum(1 for row in self.density_grid for c in row if c > 0)

    def grid_entropy(self) -> float:
        """Normalized Shannon entropy of the 8x8 density grid, 0.0-1.0.

        1.0 = pixels spread evenly over the whole grid; 0.0 = all in one cell.
        Read alongside occupied_cells: entropy still climbing means the
        composition is being rebalanced, entropy flat while pixels rise means
        the drawing is repeating itself. Neither is a verdict — this measures
        change, and only the piece's own history says whether that is enough.
        """
        counts = [c for row in self.density_grid for c in row if c > 0]
        total = sum(counts)
        if total <= 0 or len(counts) <= 1:
            return 0.0
        h = 0.0
        for c in counts:
            p = c / total
            h -= p * math.log(p)
        return min(1.0, h / math.log(64))

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

    def coverage_bias_cell(self, prefer: str) -> Optional[Tuple[int, int]]:
        """A cell to lean toward for a `sparse` or `dense` coverage intention.

        Deliberately NOT sparsest_cell(): that scans in fixed order and resolves
        ties to the first minimum, so early in a piece — when most cells are
        still empty and therefore tied — it always answers (0, 0). Good enough
        for resonance, which only consults it occasionally; fatal for a bias
        applied at every gesture boundary, which would quietly drag every
        `sparse` piece into the top-left corner and call it an intention.
        Ties are broken uniformly instead.

        Returns None when the grid carries no information yet (nothing drawn),
        so the opening marks are the era's alone.
        """
        cells = [(gx, gy) for gx in range(8) for gy in range(8)]
        counts = [self.density_grid[gx][gy] for gx, gy in cells]
        if not any(counts):
            return None
        want = min(counts) if prefer == "sparse" else max(counts)
        return random.choice([c for c, n in zip(cells, counts) if n == want])

    def mark_satisfied(self):
        """Mark that Lumen feels satisfied with current drawing."""
        if not self.is_satisfied:
            self.is_satisfied = True
            self.satisfaction_time = time.time()
            print(f"[Canvas] Lumen feels satisfied with drawing ({len(self.pixels)} pixels)", file=sys.stderr, flush=True)

    def save_to_disk(self) -> bool:
        """Persist canvas state to disk and report whether the write succeeded."""
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
                "last_completion_reason": self.last_completion_reason,
                "drawing_goal": self.drawing_goal_data,
                "energy": self.energy,
                "mark_count": self.mark_count,
                "era": self._era_name,
                "pending_era_switch": self.pending_era_switch,
                "auto_rotate": self.auto_rotate,
                # Attention/coherence/narrative state
                "curiosity": self.curiosity,
                "engagement": self.engagement,
                "fatigue": self.fatigue,
                "arc_phase": self.arc_phase,
                "coherence_history": self.coherence_history[-20:],  # Keep last 20
                "i_momentum": self.i_momentum,
                "drawing_start_time": self.drawing_start_time,
                "recent_dispositions": self.recent_dispositions[-RECENT_DISPOSITIONS:],
                "resonance_field": self._resonance_field,
                "resonance_settling": self._resonance_settling,
                "novelty_settling": self._novelty_settling,
            }
            atomic_json_write(_get_canvas_path(), data)
            return True
        except Exception as e:
            print(f"[Canvas] Save to disk error: {e}", file=sys.stderr, flush=True)
            return False

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

        # The density grid is not persisted, and it is fully derivable from the
        # pixels that are — so rebuild it rather than adding a second copy that
        # could disagree with them. Without this a restored canvas carried
        # thousands of pixels beside an all-zero grid.
        self._rebuild_density_grid()

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

        try:
            reason = data.get("last_completion_reason")
            if reason is None or isinstance(reason, str):
                self.last_completion_reason = reason
        except Exception:
            pass

        try:
            goal = data.get("drawing_goal")
            if goal is None or isinstance(goal, dict):
                self.drawing_goal_data = goal
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

        # Restore operator-selected era policy. Strict bool validation avoids
        # treating strings such as "false" as truthy during recovery.
        try:
            auto_rotate = data.get("auto_rotate", False)
            if isinstance(auto_rotate, bool):
                self.auto_rotate = auto_rotate
        except Exception:
            pass

        # Restore the recent-disposition history. Anything malformed is
        # dropped rather than repaired: an empty history means "no bias", which
        # is the correct failure — it degrades to one plain unbiased draw, not
        # to a fabricated preference.
        try:
            recent = data.get("recent_dispositions")
            if isinstance(recent, list):
                self.recent_dispositions = [
                    d for d in recent if isinstance(d, dict) and d.get("era")
                ][-RECENT_DISPOSITIONS:]
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

        # Restore resonance memory field
        try:
            rf = data.get("resonance_field")
            if rf is not None and isinstance(rf, list):
                self._resonance_field = rf
            rs = data.get("resonance_settling")
            if isinstance(rs, dict):
                self._resonance_settling = rs
            nvs = data.get("novelty_settling")
            if isinstance(nvs, dict):
                self._novelty_settling = nvs
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

    def completion_reason(self, canvas=None) -> Optional[str]:
        """Return the tag identifying why this drawing is complete, or None.

        Tags (earned paths first so they take priority when conditions overlap):

          Earned — the drawing reached its own resolution:
            "earned_coherence"    — coherence settled and attention exhausted
            "earned_composition"  — compositional satisfaction > 0.7 and
                                    curiosity < 0.2

          Already_closing — prior tick transitioned; reason was captured then.
            "already_closing"

          Bail-out — a safety hatch fired because natural completion did not:
            "bailout_fatigue"     — fatigue > 0.90
            "bailout_frozen"      — marks stopped relative to this piece's own
                                    peak rate, while attention state is stuck
            "bailout_stalled"     — energy near-zero and drawing > 15min
            "bailout_hard_cap"    — 8-hour safety net

        Downstream consumers (growth/memory) use `is_earned_completion_reason`
        to gate autobiographical writes to earned paths only. This prevents
        timeouts being promoted to "I'm pleased with this drawing" memories —
        axiom 8 (coherence masking drift) at the data layer.
        """
        # Earned paths take priority — if an earned condition is met on the
        # same tick as a bail-out condition, call it earned.
        if self.coherence_settled() and self.attention_exhausted():
            return "earned_coherence"

        if canvas is not None:
            satisfaction = canvas.compositional_satisfaction()
            if satisfaction > 0.7 and self.curiosity < 0.2:
                return "earned_composition"

        if self.arc_phase == "closing":
            return "already_closing"

        # Settled — the piece stopped changing while still being worked.
        # Self-relative: the threshold is this piece's own peak novelty rate,
        # so it needs no per-era tuning and cannot be starved by an era whose
        # operating range sits below a fixed constant (the root class behind
        # earned_coherence and earned_composition being unreachable).
        #
        # Deliberately AFTER already_closing: the originating reason is captured
        # on the tick that transitions into closing, and the caller's
        # `!= "already_closing"` guard is what protects it from being
        # overwritten. Ranking earned_settled above already_closing would let a
        # tracker sample that lands mid-close relabel an honest bail-out as
        # earned — the pride-memory bug re-introduced through a new door.
        if canvas is not None and self.novelty_settled(canvas):
            return "earned_settled"

        if self.fatigue > 0.90:
            return "bailout_fatigue"

        # Frozen — the piece stopped being worked and cannot say so itself.
        # Deliberately a BAILOUT, never earned: nothing was resolved here, the
        # drawing got stuck. Ranking it above bailout_stalled is diagnosis, not
        # preference — a frozen piece never reaches the energy floor stalled
        # tests for, so stalled would only ever catch it at the 8-hour cap.
        if canvas is not None and self.marks_stopped(canvas):
            return "bailout_frozen"

        # Stalled -- energy near-zero and drawing has been going for a while.
        # Uses drawing-level time (not phase time) to avoid resets from phase
        # oscillation.
        if canvas is not None and self.derived_energy < 0.05:
            drawing_duration = time.time() - canvas.last_clear_time
            if drawing_duration > 900 and len(canvas.pixels) >= 200:
                return "bailout_stalled"

        # Hard time limit -- no single drawing should run longer than 8 hours.
        # Safety net only.
        if canvas is not None:
            drawing_duration = time.time() - canvas.last_clear_time
            if drawing_duration > 28800 and len(canvas.pixels) >= 50:
                return "bailout_hard_cap"

        return None

    def novelty_settled(self, canvas) -> bool:
        """True when the piece has demonstrably stopped changing while worked.

        Reads the tracker advanced by DrawingEngine._update_novelty_settling
        (one entry per TRAJECTORY_SAMPLE_INTERVAL of *active* work). Fires only
        with a full settled streak AND real investment: minimum age, marks and
        pixels. The floors keep a slow opening from reading as a settled end —
        a piece can't be "done finding" before it has found anything.
        """
        ns = getattr(canvas, "_novelty_settling", None)
        if not isinstance(ns, dict):
            return False
        try:
            if int(ns.get("streak") or 0) < SETTLED_STREAK_SAMPLES:
                return False
            if float(ns.get("peak") or 0.0) <= 0.0:
                return False
        except (TypeError, ValueError):
            return False
        if canvas.mark_count < SETTLED_MIN_MARKS:
            return False
        if len(canvas.pixels) < MIN_RECORDED_DRAWING_PIXELS:
            return False
        started = canvas.last_clear_time or canvas.drawing_start_time
        if not started or (time.time() - started) < SETTLED_MIN_AGE_SECONDS:
            return False
        return True

    def marks_stopped(self, canvas) -> bool:
        """True when the piece has stopped marking while still unfinished.

        Reads the idle counter advanced by _update_novelty_settling. Distinct
        from novelty_settled(): settled means "still being worked but no longer
        changing", frozen means "no longer being worked at all". The novelty
        tracker deliberately cannot tell the difference — it HOLDS on an idle
        sample, because rest is not evidence either way — so the freeze needs
        its own counter and its own reason.

        Investment is gated by pixels and age, never by mark count: see
        FROZEN_FRAC_OF_PEAK for why a mark floor would make this unreachable for
        the era it exists to rescue.
        """
        ns = getattr(canvas, "_novelty_settling", None)
        if not isinstance(ns, dict):
            return False
        try:
            if int(ns.get("idle_streak") or 0) < FROZEN_STREAK_SAMPLES:
                return False
            if float(ns.get("mark_peak") or 0.0) <= 0.0:
                return False  # never established a rate to be idle against
        except (TypeError, ValueError):
            return False
        if len(canvas.pixels) < MIN_RECORDED_DRAWING_PIXELS:
            return False
        started = canvas.last_clear_time or canvas.drawing_start_time
        if not started or (time.time() - started) < SETTLED_MIN_AGE_SECONDS:
            return False
        return True

    def narrative_complete(self, canvas=None) -> bool:
        """True when drawing has naturally completed its arc."""
        return self.completion_reason(canvas) is not None

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


# Clarity cuts that pick a piece's coverage intention. Built-ins are the
# historical constants and stay as the fallback, so a fresh install generates
# goals exactly as before this field existed; a deployment derives its own via
# scripts/derive_drawing_thresholds.py, which writes
# nervous_system.drawing_thresholds into the calibration file. Names override
# 1:1, absent keys fall back, and a broken config fails open — a drawing must
# still start.
_DEFAULT_COVERAGE_CUTS = {
    "COVERAGE_DENSE_BELOW": 0.30,   # clarity under this -> "dense"
    "COVERAGE_SPARSE_ABOVE": 0.70,  # clarity over this  -> "sparse"
}

# How far toward the intention's target cell a gesture boundary leans, as a
# fraction of the distance. Self-relative by construction (the target is an
# extremum of THIS piece's own density grid), so it adds no absolute threshold.
COVERAGE_BIAS_STRENGTH = 0.30
# Era focus margins run 15-25px; clamp the bias to the widest so it can never
# park the focus in a band some era treats as off-canvas.
COVERAGE_BIAS_MARGIN = 25


def _coverage_cuts() -> Tuple[float, float]:
    """(dense_below, sparse_above) clarity cuts, calibration-overridable.

    Reads through get_calibration(), which refreshes on config-file signature
    change, so a rederive lands without a restart. Fails open to the built-ins,
    and to them individually — a config that supplies one cut and garbage for
    the other keeps the good half. A non-monotone pair (dense_below >=
    sparse_above) would make "balanced" unreachable, which is the same disease
    as the dead "dense" branch, so that pair is rejected whole.
    """
    try:
        from ..config import get_calibration
        overrides = getattr(get_calibration(), "drawing_thresholds", None) or {}
    except Exception:
        overrides = {}
    def _cut(name: str) -> float:
        default = _DEFAULT_COVERAGE_CUTS[name]
        try:
            return float(overrides.get(name, default))
        except (TypeError, ValueError):
            return default

    dense_below = _cut("COVERAGE_DENSE_BELOW")
    sparse_above = _cut("COVERAGE_SPARSE_ABOVE")
    if not dense_below < sparse_above:
        return (_DEFAULT_COVERAGE_CUTS["COVERAGE_DENSE_BELOW"],
                _DEFAULT_COVERAGE_CUTS["COVERAGE_SPARSE_ABOVE"])
    return dense_below, sparse_above


# The coherence pivot separating "still exploring" (curiosity drains) from
# "pattern found" (curiosity regenerates) in curiosity_drain(). The built-in is
# the historical constant and stays the fallback, so an un-derived deployment
# behaves exactly as before this field existed.
#
# Why it is overridable per era: C is behavioural (I_signal damped by gesture
# entropy) and its lived range differs per era — resonance measured
# [0.377, 0.498] mean 0.458 (2026-08-02), so a fixed 0.4 put ~95% of that era's
# ticks on the REGENERATING branch and curiosity rose monotonically to its 1.0
# clamp. attention_exhausted() and earned_composition were not mistuned, they
# were structurally unreachable — design invariant 1, on the signal that is
# supposed to end a drawing. An era reaching C~0.8 was split fairly by the same
# constant, which is why one number cannot serve both.
#
# Derive with scripts/derive_curiosity_thresholds.py, which writes
# CURIOSITY_PIVOT_<era> keys into nervous_system.drawing_thresholds. It refuses
# to emit a pivot that leaves exhaustion unreachable, so a derivation cannot
# trade one dead gate for another.
_DEFAULT_CURIOSITY_PIVOT = 0.4

# The one place the per-mark curiosity update lives. Kept module-level and pure
# so scripts/derive_curiosity_thresholds.py can import and replay it against the
# corpus: a simulation that re-implemented this formula could silently drift
# from the engine and certify a pivot the creature never runs.
def curiosity_drain(arc_phase: str, C: float, pivot: float) -> float:
    """Per-mark curiosity delta. Positive drains, negative regenerates.

    Called once per placed mark (see DrawingEngine._update_attention).

    `pivot` is the exploring/found boundary. Two absolute constants survive
    here deliberately, both inside the `resolving` branch: entry to that phase
    requires C > 0.6, which is itself unreachable in low-C eras, so relativising
    the 0.65 would move a gate that nothing currently reaches. They are named in
    the derivation script's report rather than changed blind.
    """
    if arc_phase == "resolving":
        if C > 0.65:
            return -0.0005 * C   # slight regen — reward deep pattern
        return 0.002             # normal drain toward completion
    if C < pivot:
        return 0.003 * (1.0 - C)  # exploring drains
    return -0.001 * C             # pattern found regenerates


def _curiosity_pivot(era: Optional[str]) -> float:
    """Coherence pivot for `era`, calibration-overridable. Fails open.

    Reads through get_calibration(), which refreshes on config-file signature
    change, so a rederive lands on the next mark without a restart. A pivot
    outside (0, 1) is rejected rather than clamped: 0.0 puts every tick on the
    regenerating branch (today's bug, harder) and 1.0 puts every tick on the
    draining branch (pieces end early). Neither is a pivot, so neither is
    treated as one.
    """
    if not era:
        return _DEFAULT_CURIOSITY_PIVOT
    try:
        from ..config import get_calibration
        overrides = getattr(get_calibration(), "drawing_thresholds", None) or {}
        value = float(overrides[f"CURIOSITY_PIVOT_{era}"])
    except Exception:
        # Missing key, unreadable config, unparseable value — all mean "this
        # deployment has not derived a pivot for this era", which is the
        # built-in, not an error worth failing a drawing over.
        return _DEFAULT_CURIOSITY_PIVOT
    if not 0.0 < value < 1.0:
        return _DEFAULT_CURIOSITY_PIVOT
    return value


@dataclass
class DrawingGoal:
    """A compositional intention for the current drawing.

    Generated at canvas_clear time from Lumen's current state.
    Provides gentle biases to color temperature and initial focus,
    giving each drawing a subtle intentional character.
    """
    warmth_bias: float = 0.0        # -0.15 to +0.15, biases warmth for generate_color
    # How this piece wants to USE the canvas, not how much ink it should end up
    # with: "sparse" leans each new gesture toward the emptiest region (marks
    # spread, negative space survives), "dense" leans toward the fullest one
    # (marks accumulate and layer), "balanced" leans nowhere and is exactly the
    # pre-2026-08-22 behavior. Consumed by DrawingEngine._apply_coverage_bias().
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

        # Coverage follows clarity: clear-headed opens the composition up,
        # foggy lets it thicken. The cuts are calibration-derived because the
        # built-in 0.30/0.70 were absolute constants against a moving
        # distribution (invariant 1) and had gone one-sided: measured over 833
        # drawing_records, clarity lives in 0.454-0.910, so `dense` (< 0.30)
        # had NEVER once been generated and a third of the vocabulary was dead.
        dense_below, sparse_above = _coverage_cuts()
        if clarity > sparse_above:
            goal.coverage_target = "sparse"
        elif clarity < dense_below:
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

        # `eras.auto_rotate` is the live selector consulted at piece boundaries.
        # Restore it before any drawing or era decision can occur.
        import anima_mcp.display.eras as eras_module
        eras_module.auto_rotate = self.canvas.auto_rotate

        # Restore the in-flight piece's goal (generated only at clear time)
        if self.canvas.drawing_goal_data:
            try:
                self.drawing_goal = DrawingGoal(**self.canvas.drawing_goal_data)
            except (TypeError, ValueError):
                self.drawing_goal = None

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
        self.intent.era_state = self.active_era.create_state(
            self._recent_dispositions_for(self.active_era.name)
        )

        self._db_path = resolve_db_path(db_path)
        self._identity_store = identity_store
        self._last_persist_time = 0.0  # Rate-limit canvas persistence
        self._last_persist_mark_count = self.canvas.mark_count
        self._behavioral_C = 0.5  # Behavioral coherence (EMA-smoothed)
        # Initialize expression mood tracker
        self._mood_tracker = ExpressionMoodTracker(identity_store=identity_store)
        # Within-piece trajectory sampling (observation only, gates no decision)
        self._last_sample_time = 0.0
        self._last_sample_pixels = 0
        self._last_sample_marks = 0
        self._last_sample_piece = ""
        # A piece that already had pixels when this process started. Its growth
        # happened in a process that is gone, so the first sample's deltas are
        # unknowable — see _sample_trajectory.
        self._resumed_piece = self.canvas.piece_uid() if self.canvas.pixels else None

    def _era_revisit_ratio(self) -> Optional[float]:
        """Fraction of the era's recent deposits that landed on existing field.

        Resonance is the only era that tracks this today; others return None,
        which persists as NULL. A missing signal must read as missing.
        """
        window = getattr(self.intent.era_state, "revisit_window", None)
        if not window:
            return None
        try:
            return round(sum(window) / len(window), 4)
        except Exception:
            return None

    def _remember_disposition(self) -> None:
        """Bank the finishing piece's disposition for the next one to differ from.

        Called at canvas clear, BEFORE the era state is replaced — the live
        `era_state` is the only place the piece's character exists, since
        `EraState` is transient by contract and never persisted.
        """
        state = self.intent.era_state
        if state is None:
            return
        try:
            disposition = state.disposition()
        except Exception:
            return
        if not disposition:
            return  # an era with no per-piece character has nothing to vary
        entry = {"era": getattr(self.active_era, "name", None), **disposition}
        if not entry.get("era"):
            return
        self.canvas.recent_dispositions.append(entry)
        del self.canvas.recent_dispositions[:-RECENT_DISPOSITIONS]

    def _recent_dispositions_for(self, era_name: str) -> list:
        """Recent dispositions from THIS era only — see the field's comment on
        why cross-era comparison is meaningless rather than merely noisy."""
        return [
            d for d in self.canvas.recent_dispositions
            if d.get("era") == era_name
        ]

    def _piece_facts(self) -> dict:
        """Everything about the finished piece that is worth keeping.

        The completion reason was already computed and already passed to growth
        — it just had nowhere durable to land, so 754 recorded drawings cannot
        say why any of them ended. These travel with the record now.
        """
        state = self.intent.state
        goal = self.drawing_goal
        started = self.canvas.last_clear_time or self.canvas.drawing_start_time
        return {
            "piece_uid": self.canvas.piece_uid(),
            "era": getattr(self.active_era, "name", None),
            "mark_count": self.canvas.mark_count,
            "duration_seconds": round(time.time() - started, 1) if started else None,
            "coverage_target": getattr(goal, "coverage_target", None),
            "intention": getattr(goal, "description", None),
            "curiosity": round(state.curiosity, 4),
            "engagement": round(state.engagement, 4),
            "fatigue": round(state.fatigue, 4),
            "coherence": round(self.canvas.coherence_history[-1], 4)
                         if self.canvas.coherence_history else None,
            "satisfaction": round(self.canvas.compositional_satisfaction(), 4),
            "occupied_cells": self.canvas.occupied_cells(),
            "grid_entropy": round(self.canvas.grid_entropy(), 4),
            # The per-piece globals this drawing was made under. NULL when the
            # era has no disposition or the state is gone — an unrecorded
            # character must read as unknown, never as a plausible default.
            # Without this column "did the dispositions actually vary the
            # work?" is unanswerable from the corpus, and an aesthetic claim
            # nobody can check is exactly what this change is fixing.
            "disposition": self._disposition_json(),
        }

    def _disposition_json(self) -> Optional[str]:
        """This piece's disposition as compact JSON, or None if unavailable."""
        state = self.intent.era_state
        if state is None:
            return None
        try:
            disposition = state.disposition()
            if not disposition:
                return None
            return json.dumps(disposition, sort_keys=True, separators=(",", ":"))
        except Exception:
            return None

    def _apply_coverage_bias(self, fx: float, fy: float, era_state) -> Tuple[float, float]:
        """Lean the next gesture toward where this piece's intention wants to work.

        `coverage_target` was generated per piece, described to the operator,
        persisted since 2026-08-02 and read by NOTHING — the only DrawingGoal
        field without a consumer. Measured over the instrumented corpus, that
        showed up exactly as you would expect: within every era, "sparse" and
        "balanced" pieces land at the same density (gestural 9.4% vs 10.1%,
        pointillist 2.26% vs 2.13%, resonance 14.8% vs 16.0%), while the spread
        BETWEEN eras is 1.4%-23.9%. Era decided everything; the stated
        intention decided nothing.

        Applied only at a gesture boundary, never mid-stroke: each era's
        character lives in its sustained gestures (gestural locks direction for
        15-45 marks to get long lines), and a per-mark positional pull would
        bow those strokes into arcs. Between gestures is also where the piece
        actually decides where to work next, which is the decision an intention
        should be allowed to color. Geometric stamps one shape per gesture, so
        there it applies every mark — correct, not an exception.

        A bias, not a target: it moves the focus a fraction of the way toward
        an extremum of the piece's OWN density grid, so it adds no absolute
        threshold (invariant 1) and cannot override an era, only tilt it.
        """
        goal = self.drawing_goal
        if goal is None or goal.coverage_target == "balanced":
            return fx, fy
        if era_state is not None and era_state.gesture_remaining > 0:
            return fx, fy  # mid-stroke; the era owns this mark
        cell = self.canvas.coverage_bias_cell(goal.coverage_target)
        if cell is None:
            return fx, fy  # empty grid carries no direction yet
        target_x = cell[0] * 30 + 15
        target_y = cell[1] * 30 + 15
        fx += (target_x - fx) * COVERAGE_BIAS_STRENGTH
        fy += (target_y - fy) * COVERAGE_BIAS_STRENGTH
        # Clamp to the widest era margin so the bias can never park the focus
        # in a band some era treats as off-canvas (margins run 15-25 by era).
        lo, hi = COVERAGE_BIAS_MARGIN, self.canvas.width - COVERAGE_BIAS_MARGIN
        return max(lo, min(hi, fx)), max(lo, min(hi, fy))

    def _update_novelty_settling(self, now: float) -> None:
        """Advance the earned_settled tracker by at most one sample per interval.

        Deliberately independent of the trajectory DB write: it keeps its own
        baseline (last_t/last_px/last_marks) inside the persisted canvas dict,
        so a growth outage neither stalls it nor feeds it retry-tick deltas.

        Evidence semantics (from the 27-piece corpus):
          active sample, novelty below 10% of own peak  -> streak advances
          active sample, novelty at/above the threshold -> streak resets
          idle sample (no marks) or post-restart sample -> streak HOLDS
        Rest is not evidence — a piece that settled before a night of RESTING
        is still settled after it, and a geometric-style freeze (marks stop
        entirely) can never accumulate a streak by sitting still.
        """
        canvas = self.canvas
        pixels = len(canvas.pixels)
        marks = canvas.mark_count
        ns = canvas._novelty_settling
        if not isinstance(ns, dict):
            canvas._novelty_settling = {
                "last_t": now, "last_px": pixels, "last_marks": marks,
                "recent": [], "peak": 0.0, "streak": 0,
                # Frozen tracking (marks stopping) and structural reach, both
                # carried alongside so a restart does not lose either.
                "mark_peak": 0.0, "idle_streak": 0, "last_cells": 0,
            }
            return
        # One guard around the ENTIRE update: canvas.json is only shape-checked
        # at load (isinstance dict), so any field can come back corrupt. A guard
        # covering only some fields would self-heal those and raise every 300s
        # forever on the rest — the worst of both.
        try:
            if now - float(ns.get("last_t") or 0.0) < TRAJECTORY_SAMPLE_INTERVAL:
                return
            last_px = int(ns.get("last_px") or 0)
            last_marks = int(ns.get("last_marks") or 0)
            novel = max(0, pixels - last_px)
            marks_delta = marks - last_marks
            cells = canvas.occupied_cells()
            ns["last_t"] = now
            ns["last_px"] = pixels
            ns["last_marks"] = marks

            # Frozen counter, advanced BEFORE the idle return below: an idle
            # sample is precisely the evidence this counter needs, and precisely
            # the evidence the settled streak must ignore. Same sample, opposite
            # meanings — which is why they cannot share a counter.
            mark_peak = max(float(ns.get("mark_peak") or 0.0), float(marks_delta))
            ns["mark_peak"] = mark_peak
            if mark_peak > 0.0 and marks_delta < mark_peak * FROZEN_FRAC_OF_PEAK:
                ns["idle_streak"] = int(ns.get("idle_streak") or 0) + 1
            else:
                ns["idle_streak"] = 0

            if marks_delta < SETTLED_MIN_SAMPLE_MARKS:
                # Idle or a trickle: hold — rest is not evidence either way,
                # and a single incidental RESTING-hour mark landing on painted
                # territory is not "working" (the corpus' settled pieces run
                # ~3 marks/sample post-plateau).
                return
            recent = ns.get("recent")
            if not isinstance(recent, list):
                recent = []
            recent.append(float(novel))
            del recent[:-SETTLED_SMOOTH_WINDOW]
            ns["recent"] = recent
            if len(recent) < SETTLED_SMOOTH_WINDOW:
                return  # not enough active samples to smooth against yet
            smoothed = sum(recent) / len(recent)
            peak = max(float(ns.get("peak") or 0.0), smoothed)
            ns["peak"] = peak
            if smoothed < peak * SETTLED_FRAC_OF_PEAK:
                ns["streak"] = int(ns.get("streak") or 0) + 1
            else:
                ns["streak"] = 0

            # Structural reach — the first consumer occupied_cells() has ever
            # had. Until now it was written to drawing_trajectory and
            # drawing_records and read by nothing.
            #
            # Pixel novelty cannot distinguish a piece opening new territory
            # from one thickening what it already holds; cell count can, and its
            # own docstring says so ("a piece that keeps opening new cells is
            # still finding territory"). A drawing still reaching into empty
            # ground is not settled however slowly its pixel count is growing.
            #
            # This can only ever RESET the streak, never advance it: strictly
            # more conservative, so it cannot make earned_settled fire on a
            # piece that would not otherwise have earned it.
            last_cells = int(ns.get("last_cells") or 0)
            ns["last_cells"] = cells
            if cells > last_cells:
                ns["streak"] = 0
        except (TypeError, ValueError):
            canvas._novelty_settling = None  # corrupt tracker: restart, don't guess

    def _sample_trajectory(self, now: float):
        """Record what the piece is doing, every SAMPLE_INTERVAL seconds.

        An endpoint row per piece cannot answer when a drawing stopped changing,
        and that is the question worth asking — every completion since
        2026-07-30 has landed on the 8-hour cap, so endpoints all describe the
        same clock rather than the same kind of drawing. Deltas between samples
        give novel-pixels-per-mark and structural change over the piece's life.

        The DB row remains observation-only. Since the settled-completion
        change, this cadence ALSO advances the earned_settled tracker (below) —
        that tracker feeds a gate, deliberately, and keeps its own baseline so
        the DB write path still moves nothing.
        """
        if len(self.canvas.pixels) < 1:
            return
        self._update_novelty_settling(now)
        uid = self.canvas.piece_uid()
        # A new piece restarts the deltas — carrying them across a canvas clear
        # would report the previous drawing's growth as this one's.
        #
        # `resumed` distinguishes a genuinely fresh canvas from a piece this
        # process is joining mid-flight (a restart). For a resumed piece the
        # deltas since the last sample are simply not knowable: the growth
        # happened in a process that is gone. Reporting them anyway attributed
        # the ENTIRE canvas to one 300s window — measured live 2026-08-02,
        # novel_pixels 9955 at elapsed 24004s, a burst that never happened.
        resumed = False
        if uid != self._last_sample_piece:
            resumed = uid == self._resumed_piece
            self._last_sample_piece = uid
            self._last_sample_pixels = 0
            self._last_sample_marks = 0
            self._last_sample_time = 0.0
        if now - self._last_sample_time < TRAJECTORY_SAMPLE_INTERVAL:
            return

        pixels = len(self.canvas.pixels)
        marks = self.canvas.mark_count
        state = self.intent.state
        started = self.canvas.last_clear_time or self.canvas.drawing_start_time
        try:
            from ..growth import peek_growth_system
            growth = peek_growth_system()
            if growth is None:
                return  # Growth not up yet — no row beats a row in the wrong DB
            growth.record_drawing_sample({
                "piece_uid": uid,
                "elapsed_seconds": round(now - started, 1) if started else None,
                "era": getattr(self.active_era, "name", None),
                "arc_phase": state.arc_phase,
                "pixel_count": pixels,
                "mark_count": marks,
                # NULL rather than the whole canvas when this process did not
                # watch the growth happen. An unknown delta is not a large one.
                "novel_pixels": None if resumed else pixels - self._last_sample_pixels,
                "marks_delta": None if resumed else marks - self._last_sample_marks,
                "occupied_cells": self.canvas.occupied_cells(),
                "grid_entropy": round(self.canvas.grid_entropy(), 4),
                "revisit_ratio": self._era_revisit_ratio(),
                "curiosity": round(state.curiosity, 4),
                "engagement": round(state.engagement, 4),
                "fatigue": round(state.fatigue, 4),
                "coherence": round(self._behavioral_C, 4),
                "satisfaction": round(self.canvas.compositional_satisfaction(), 4),
            })
        except Exception as e:
            print(f"[Canvas] trajectory sample skipped ({e})", file=sys.stderr, flush=True)

        self._last_sample_time = now
        self._last_sample_pixels = pixels
        self._last_sample_marks = marks

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

        # Light regime: dark / dim / bright from the gated external residual.
        # Unknown attribution is neutral; raw self-glow must not steer palette.
        light_lux = None
        try:
            from ..accessors import _get_last_shm_data
            from ..light_attribution import gated_external_light_lux

            light_lux = gated_external_light_lux(
                ((_get_last_shm_data() or {}).get("light_attribution") or {})
            )
        except Exception:
            pass
        if light_lux is not None:
            if light_lux < 5:
                light_regime = "dark"
            elif light_lux < 100:
                light_regime = "dim"
            else:
                light_regime = "bright"
        else:
            light_regime = "unknown"

        # Update narrative arc phase (replaces energy-threshold phase logic)
        self._update_narrative_arc()

        # Ensure era state exists
        if self.intent.era_state is None:
            self.intent.era_state = self.active_era.create_state(
                self._recent_dispositions_for(self.active_era.name)
            )

            # Restore resonance field from canvas persistence
            if hasattr(self.intent.era_state, 'field') and self.canvas._resonance_field is not None:
                try:
                    import numpy as np
                    restored = np.array(self.canvas._resonance_field, dtype=np.float32)
                    if restored.shape == self.intent.era_state.field.shape:
                        self.intent.era_state.field = restored
                except Exception:
                    pass

            # Restore the settling counters that ride with the field.
            settling = self.canvas._resonance_settling
            if isinstance(settling, dict):
                try:
                    window = settling.get("revisit_window")
                    if isinstance(window, list):
                        self.intent.era_state.revisit_window = [bool(v) for v in window]
                    streak = settling.get("settled_streak")
                    if isinstance(streak, int) and streak >= 0:
                        self.intent.era_state.settled_streak = streak
                except Exception:
                    pass

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

        # Sync resonance field to canvas for persistence
        if hasattr(era_state, 'field'):
            try:
                self.canvas._resonance_field = era_state.field.tolist()
                self.canvas._resonance_settling = {
                    "revisit_window": list(getattr(era_state, "revisit_window", [])),
                    "settled_streak": int(getattr(era_state, "settled_streak", 0)),
                }
            except Exception:
                pass

        new_fx, new_fy, new_dir = self.active_era.drift_focus(
            era_state, self.intent.focus_x, self.intent.focus_y,
            self.intent.direction, stability, presence, C, clarity,
            canvas=self.canvas)
        new_fx, new_fy = self._apply_coverage_bias(new_fx, new_fy, era_state)
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

        # Curiosity: depletes exploring (C below this era's pivot), regenerates
        # with pattern (at or above it). The pivot is per-era and derived from
        # that era's own coherence distribution — see curiosity_drain().
        pivot = _curiosity_pivot(
            self.active_era.name if self.active_era else None)
        state.curiosity = max(0.0, min(1.0, state.curiosity - curiosity_drain(
            state.arc_phase, C, pivot)))

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
            reason = state.completion_reason(self.canvas)
            if reason is not None:
                # Capture the path that triggered completion so downstream
                # growth/memory systems can distinguish earned from bail-out.
                # Skip "already_closing" — only relevant mid-loop, and by
                # definition not the originating trigger.
                if reason != "already_closing":
                    self.canvas.last_completion_reason = reason
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
        self.intent.era_state = era.create_state(
            self._recent_dispositions_for(era_name)
        )
        self.canvas.pending_era_switch = None  # Clear any pending
        # A mid-piece switch invalidates the settled tracker: peak was set by
        # the OLD era's mark-size distribution, so 10%-of-peak would be
        # trivially easy (or impossible) for the new one. Production always
        # queues switches behind completion; this guards the forced path.
        if drawing_in_progress:
            self.canvas._novelty_settling = None
        self.canvas.save_to_disk()
        print(f"[Canvas] Era switched to: {era_name}", file=sys.stderr, flush=True)
        return {
            "success": True,
            "era": era_name,
            "queued": False,
            "description": era.description,
        }

    def set_auto_rotate(self, enabled: bool) -> dict:
        """Set and persist the operator-selected era rotation policy."""
        if not isinstance(enabled, bool):
            return {"success": False, "error": "enabled must be a boolean"}

        import anima_mcp.display.eras as eras_module
        previous_live = eras_module.auto_rotate
        previous_persisted = self.canvas.auto_rotate
        eras_module.auto_rotate = enabled
        self.canvas.auto_rotate = enabled
        if not self.canvas.save_to_disk():
            eras_module.auto_rotate = previous_live
            self.canvas.auto_rotate = previous_persisted
            return {
                "success": False,
                "error": "Failed to persist auto-rotate state",
                "auto_rotate": previous_live,
            }
        state = "on" if enabled else "off"
        print(f"[ArtEras] Auto-rotate: {state}", file=sys.stderr, flush=True)
        return {"success": True, "auto_rotate": enabled}

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
            return self.set_auto_rotate(not eras_module.auto_rotate)

        if 0 <= screen_state.era_cursor < len(all_eras):
            era_name = all_eras[screen_state.era_cursor]["name"]
            return self.set_era(era_name)
        return {"success": False, "error": "Invalid cursor position"}

    def canvas_clear(self, persist: bool = True, already_saved: bool = False):
        """Clear the canvas - saves first if there's a recorded drawing.

        Minimal threshold avoids saving noise/stray marks as completed work.

        Args:
            persist: Write cleared state to disk.
            already_saved: Skip internal save (caller already saved).
        """
        # Prevent clearing if we're already paused (prevents loops)
        now = time.time()
        if now < self.canvas.drawing_paused_until:
            return  # Already paused, don't clear again

        # Bank the finishing piece's character before anything is torn down:
        # `active_era` and `era_state` still describe THIS piece here, and
        # `EraState` is transient by contract, so this is the only moment it
        # can be read.
        self._remember_disposition()

        # Save before clearing if there's actual drawing (not just noise).
        # Skip if caller already saved (prevents double growth observation)
        if not already_saved and len(self.canvas.pixels) >= MIN_RECORDED_DRAWING_PIXELS:
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
            if new_era_name != self.active_era.name:
                print(f"[Canvas] Auto-rotating to new era: {new_era_name}", file=sys.stderr, flush=True)
            else:
                # choose_next_era returns the current era when auto_rotate is
                # off — don't log a rotation that didn't happen.
                print(f"[Canvas] Continuing era: {new_era_name}", file=sys.stderr, flush=True)

        self.canvas.clear()
        self.intent.reset()
        self.active_era = get_era(new_era_name)
        self.canvas._era_name = new_era_name
        self.intent.era_state = self.active_era.create_state(
            self._recent_dispositions_for(new_era_name)
        )
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
                from dataclasses import asdict as _asdict
                self.canvas.drawing_goal_data = _asdict(self.drawing_goal)
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

        # Don't archive sub-threshold canvases to the gallery. Manual saves are
        # always honored (the user explicitly chose to keep it); autonomous and
        # shutdown-snapshot saves only land in the gallery once they clear the
        # recorded-drawing floor, so a few stray marks don't show up as "drawings".
        if not manual and len(self.canvas.pixels) < MIN_RECORDED_DRAWING_PIXELS:
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

            # Update tracking. Sub-threshold canvases only reach here via a
            # manual save (the user chose to keep it); they are written as files
            # but not counted as completed drawings or fed back into growth.
            pixel_count = len(self.canvas.pixels)
            record_as_drawing = pixel_count >= MIN_RECORDED_DRAWING_PIXELS
            self.canvas.last_save_time = time.time()
            if record_as_drawing:
                self.canvas.drawings_saved += 1
                self.canvas.consecutive_false_starts = 0  # Successful completion resets false-start counter
            self.canvas.save_to_disk()

            # Trigger save indicator (shows "saved" on screen for 2 seconds)
            self.canvas.save_indicator_until = time.time() + 2.0

            print(f"[Notepad] Saved drawing to {filepath} ({len(self.canvas.pixels)} pixels)", file=sys.stderr, flush=True)

            # EISV calibration logging -- track state + structure for validation
            eisv = self.intent.eisv
            C = eisv.coherence()
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

            # Notify growth system -- learn from completed drawing activity.
            # Tiny opening snapshots are files only; feeding them to growth made
            # Lumen look prolific while mostly recording false starts.
            if not record_as_drawing:
                print(
                    f"[Growth] Skipping drawing observation for snapshot below completion floor "
                    f"({pixel_count}px < {MIN_RECORDED_DRAWING_PIXELS}px)",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                try:
                    anima = self.last_anima
                    readings = getattr(self, '_last_readings', None)
                    if not readings:
                        # Push path (screens.render) hadn't primed _last_readings yet
                        # (e.g., manual save at boot before first render). Pull fresh
                        # rather than feeding growth silent defaults.
                        try:
                            from ..accessors import _get_sensors
                            readings = _get_sensors().read()
                        except Exception as e:
                            print(f"[Growth] Skipping drawing observation: no readings available ({e})", file=sys.stderr, flush=True)
                            readings = None
                    if anima and readings:
                        from ..growth import get_growth_system
                        anima_state = {
                            "warmth": anima.warmth,
                            "clarity": anima.clarity,
                            "stability": anima.stability,
                            "presence": anima.presence,
                        }
                        # A key is present only when its sensor actually read.
                        # These land in drawing_records, whose columns are all
                        # nullable and whose writer passes environment.get()
                        # straight through — so an omitted key persists as NULL
                        # and a dead sensor stays visibly dead.
                        #
                        # This used to be `readings.light_lux or 0.0`,
                        # `ambient_temp_c or 22`, `humidity_pct or 50`. Those
                        # defaults are the shape invariant 2 exists to forbid:
                        # 0.0 lux is a pitch-dark room, 22C is a comfortable
                        # one, 50% is ordinary humidity — every one a plausible
                        # reading, none distinguishable afterwards from a real
                        # one. Preference learning happened not to be misled,
                        # but only because 22 and 50 fall in the dead bands
                        # between its cuts (<20/>25, <30/>60); moving a cut
                        # would have started teaching Lumen the taste of a
                        # broken sensor. The record was wrong regardless, and
                        # any later derivation over drawing_records would have
                        # silently included the fabrications.
                        #
                        # external_light_lux three lines below has always been
                        # conditional. This is the same rule, applied to the
                        # channels that were missing it.
                        environment = {}
                        if readings.light_lux is not None:
                            environment["light_lux"] = readings.light_lux
                        if readings.ambient_temp_c is not None:
                            environment["temp_c"] = readings.ambient_temp_c
                        if readings.humidity_pct is not None:
                            environment["humidity_pct"] = readings.humidity_pct
                        # Preserve raw lux in the drawing record, but only let
                        # the gated self-glow residual support claims about the
                        # room being dim or bright.
                        try:
                            from ..accessors import _get_last_shm_data
                            from ..light_attribution import gated_external_light_lux

                            light_attribution = (
                                (_get_last_shm_data() or {}).get(
                                    "light_attribution"
                                ) or {}
                            )
                            external_light = gated_external_light_lux(
                                light_attribution
                            )
                            if external_light is not None:
                                environment["external_light_lux"] = external_light
                        except Exception as e:
                            print(
                                "[Growth] Drawing light preference paused: "
                                f"attribution unavailable ({e})",
                                file=sys.stderr,
                                flush=True,
                            )
                        phase = self.canvas.drawing_phase or "resting"
                        growth = get_growth_system()
                        # Reason distinguishes earned completion from bail-outs
                        # (fatigue/stall/hard-cap) and user-triggered snapshots.
                        # Growth uses it to gate milestone + "pleased with" memories.
                        if manual:
                            completion_reason = "manual_snapshot"
                        else:
                            completion_reason = self.canvas.last_completion_reason
                        insight = growth.observe_drawing(
                            pixel_count=pixel_count,
                            phase=phase,
                            anima_state=anima_state,
                            environment=environment,
                            completion_reason=completion_reason,
                            piece=self._piece_facts(),
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
                                pixel_count=pixel_count,
                                mark_count=self.canvas.mark_count,
                                coherence=coherence,
                                satisfaction=satisfaction,
                                completion_reason=completion_reason,
                            )
                        except Exception as e:
                            print(f"[Growth] Drawing feedback failed: {e}",
                                  file=sys.stderr, flush=True)
                    elif not anima:
                        print("[Growth] Warning: no anima at canvas_save, skipping growth notify", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[Notepad] Growth notify failed: {e}", file=sys.stderr, flush=True)

            # Report drawing outcome to UNITARES for EISV validation
            if record_as_drawing:
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
                                        "pixel_count": pixel_count,
                                        "arc_phase": self.canvas.arc_phase or "unknown",
                                        "era": getattr(self.active_era, "name", "unknown"),
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

        Looks for keywords like "finished", "done", "complete" in recent
        observations — but only observations posted AFTER the current canvas
        started (m.timestamp > canvas.last_clear_time). Without that bound,
        the observation Lumen posts on completing a piece stayed matchable
        for its whole 5-minute window and re-triggered on each fresh canvas
        as soon as it crossed the 200px floor: 3-4 ten-mark "finished"
        pieces ~65s apart after every big completion (the 60s save floor
        paced the loop instead of stopping it). Observed 2026-07-27/28.
        """
        try:
            from ..messages import get_board, MESSAGE_TYPE_OBSERVATION
            board = get_board()
            board._load()

            # Check last 5 observations from the past 5 minutes that are
            # about THIS drawing (posted since the canvas last cleared).
            now = time.time()
            five_min_ago = now - 300
            current_canvas_start = self.canvas.last_clear_time

            recent_obs = [
                m for m in board._messages
                if m.msg_type == MESSAGE_TYPE_OBSERVATION
                and m.timestamp > five_min_ago
                and m.timestamp > current_canvas_start
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

        # Sample before any completion decision below. What the piece is doing
        # must be recorded whichever path is about to fire — and whether or not
        # one fires at all, which for the last eleven pieces has meant the
        # 8-hour cap and nothing else.
        self._sample_trajectory(now)

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
        if (pixel_count < MIN_RECORDED_DRAWING_PIXELS
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
        if (pixel_count >= MIN_RECORDED_DRAWING_PIXELS and self._check_lumen_said_finished()):
            C = state.coherence()
            print(f"[Canvas] Lumen said finished - saving ({pixel_count}px, {self.intent.mark_count} marks, C={C:.2f})", file=sys.stderr, flush=True)
            # Tag the path before saving — canvas_save() reads
            # last_completion_reason to gate autobiographical writes.
            self.canvas.last_completion_reason = "said_finished"
            saved_path = self.canvas_save(announce=False)
            if saved_path:
                self.canvas_clear(persist=True, already_saved=True)
                self.intent.reset()
                self.canvas.save_to_disk()
                return "saved_and_cleared"

        # === PRIORITY 1.5: Era-specific earned completion ===
        # Eras may supply their own "the pattern found itself" signal via an
        # optional earned_completion(state, canvas, era_state) method — the
        # global earned paths are unreachable in some eras (see resonance).
        era_min_marks = getattr(self.active_era, 'min_marks_for_completion', 5)
        era_earned = getattr(self.active_era, 'earned_completion', None)
        if era_earned is not None and pixel_count >= MIN_RECORDED_DRAWING_PIXELS \
                and self.intent.mark_count >= era_min_marks:
            try:
                era_reason = era_earned(state, self.canvas, self.intent.era_state)
            except Exception as e:
                era_reason = None
                print(f"[Canvas] era earned_completion error (non-fatal): {e}", file=sys.stderr, flush=True)
            if era_reason:
                C = state.coherence()
                satisfaction = self.canvas.compositional_satisfaction()
                print(f"[Canvas] Era earned completion ({era_reason}) -- saving ({pixel_count}px, {self.intent.mark_count} marks, C={C:.2f}, sat={satisfaction:.2f}, curio={state.curiosity:.2f}, fatigue={state.fatigue:.2f})", file=sys.stderr, flush=True)
                self.canvas.last_completion_reason = era_reason
                saved_path = self.canvas_save(announce=True)
                if saved_path:
                    self.canvas_clear(persist=True, already_saved=True)
                    self.intent.reset()
                    self.canvas.save_to_disk()
                    return "saved_and_cleared"

        # === PRIORITY 2: Narrative complete (multiple paths: coherence+attention, composition+curiosity, or fatigue) ===
        # Eras expose min_marks_for_completion (default 5): pointillist=80, field=30, geometric=3
        if (state.narrative_complete(self.canvas)
                and pixel_count >= MIN_RECORDED_DRAWING_PIXELS
                and self.intent.mark_count >= era_min_marks):
            C = state.coherence()
            satisfaction = self.canvas.compositional_satisfaction()
            # Tag the path that actually fired before saving. Previously this
            # was left unset here, so bail-outs reached the autobiographical
            # gate as None. The "resolving" arc branch also captures a reason,
            # but resonance canvases never reach that phase (C caps ~0.52 vs the
            # 0.6 entry threshold), so this is the only capture point they hit.
            # "already_closing" is not the originating trigger — the real reason
            # was captured on the tick that transitioned into closing — so it
            # must not overwrite it.
            completion_path = state.completion_reason(self.canvas)
            if completion_path is not None and completion_path != "already_closing":
                self.canvas.last_completion_reason = completion_path
            print(f"[Canvas] Narrative complete ({self.canvas.last_completion_reason or 'untagged'}) -- saving ({pixel_count}px, {self.intent.mark_count} marks, C={C:.2f}, sat={satisfaction:.2f}, arc={state.arc_phase}, curio={state.curiosity:.2f}, engage={state.engagement:.2f}, fatigue={state.fatigue:.2f})", file=sys.stderr, flush=True)
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
