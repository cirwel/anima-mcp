"""
Drawing Engine - Lumen's autonomous drawing system.

Extracted from screens.py to separate drawing logic from display rendering.
Contains EISV thermodynamics, attention signals, coherence tracking,
narrative arc, and mark-making orchestration.

The DrawingEngine owns the canvas, intent, drawing goal, and active era.
It has zero display dependencies — it only manipulates DrawingState/CanvasState/DrawingIntent
and delegates to ArtEra instances.
"""

from typing import Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime
import time
import sys
import json
import math
import random

from ..atomic_write import atomic_json_write  # noqa: F401
from ..anima import Anima
from ..expression_moods import ExpressionMoodTracker
from ..db_paths import resolve_db_path
from .drawing_state import (  # noqa: F401
    COVERAGE_BIAS_MARGIN,
    COVERAGE_BIAS_STRENGTH,
    CanvasState,
    DrawingEISV,
    DrawingGoal,
    DrawingIntent,
    DrawingState,
    FROZEN_FRAC_OF_PEAK,
    FROZEN_STREAK_SAMPLES,
    MIN_RECORDED_DRAWING_PIXELS,
    RECENT_DISPOSITIONS,
    SETTLED_FRAC_OF_PEAK,
    SETTLED_MIN_AGE_SECONDS,
    SETTLED_MIN_MARKS,
    SETTLED_MIN_SAMPLE_MARKS,
    SETTLED_SMOOTH_WINDOW,
    SETTLED_STREAK_SAMPLES,
    TRAJECTORY_SAMPLE_INTERVAL,
    _DEFAULT_COVERAGE_CUTS,
    _DEFAULT_CURIOSITY_PIVOT,
    _EARNED_COMPLETION_REASONS,
    _EISV_PARAMS,
    _coverage_cuts,
    _curiosity_pivot,
    _get_canvas_path,
    _get_drawing_bridge,
    curiosity_drain,
    is_earned_completion_reason,
)


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
        # An era that stamps one whole form per gesture begins a new gesture on
        # every mark, whether or not the template's NAME repeats. Detecting the
        # switch by name alone made geometric's fatigue depend on how varied a
        # piece's shapes were (a switch on ~90% of stamps while every piece
        # sampled all 16 templates), so giving a piece a vocabulary of its own
        # would have stretched pieces ~25% — the shapes repeat, the name stops
        # changing, fatigue stops accruing. The cost of a stamp is the stamp.
        if getattr(self.active_era, "each_mark_is_a_gesture", False):
            gesture_switch = len(state.gesture_history) >= 2

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
