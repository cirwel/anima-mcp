"""State, canvas persistence, and compositional goal structures for the drawing engine.

Contains CanvasState, DrawingState (and DrawingEISV alias), DrawingGoal,
DrawingIntent, completion/settling/frozen/coverage/curiosity constants, and
threshold-derivation helpers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import random
import sys
import time

from ..atomic_write import atomic_json_write


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


def _resolve_canvas_path() -> Path:
    """Resolve canvas path, respecting monkeypatches on drawing_engine._get_canvas_path."""
    de_mod = sys.modules.get("anima_mcp.display.drawing_engine")
    fn = getattr(de_mod, "_get_canvas_path", _get_canvas_path) if de_mod is not None else _get_canvas_path
    return fn()


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
            atomic_json_write(_resolve_canvas_path(), data)
            return True
        except Exception as e:
            print(f"[Canvas] Save to disk error: {e}", file=sys.stderr, flush=True)
            return False

    def load_from_disk(self):
        """Load canvas state from disk - defensive against corruption."""
        path = _resolve_canvas_path()
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


__all__ = [
    "COVERAGE_BIAS_MARGIN",
    "COVERAGE_BIAS_STRENGTH",
    "CanvasState",
    "DrawingEISV",
    "DrawingGoal",
    "DrawingIntent",
    "DrawingState",
    "FROZEN_FRAC_OF_PEAK",
    "FROZEN_STREAK_SAMPLES",
    "MIN_RECORDED_DRAWING_PIXELS",
    "RECENT_DISPOSITIONS",
    "SETTLED_FRAC_OF_PEAK",
    "SETTLED_MIN_AGE_SECONDS",
    "SETTLED_MIN_MARKS",
    "SETTLED_MIN_SAMPLE_MARKS",
    "SETTLED_SMOOTH_WINDOW",
    "SETTLED_STREAK_SAMPLES",
    "TRAJECTORY_SAMPLE_INTERVAL",
    "_DEFAULT_COVERAGE_CUTS",
    "_DEFAULT_CURIOSITY_PIVOT",
    "_EARNED_COMPLETION_REASONS",
    "_EISV_PARAMS",
    "_coverage_cuts",
    "_curiosity_pivot",
    "_get_canvas_path",
    "_get_drawing_bridge",
    "curiosity_drain",
    "is_earned_completion_reason",
]
