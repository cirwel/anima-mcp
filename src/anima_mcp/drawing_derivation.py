"""Derive the per-era curiosity pivot from Lumen's own coherence distribution.

The reporting half of scripts/derive_curiosity_thresholds.py (and, since
2026-09-26, of derive_drawing_thresholds.py), extracted so the running server
can serve the same report without a shell on the device. Nothing here writes
anything, and the database is opened read-only so it cannot. The acting halves
are the scripts' --apply (operator) and self_derivation.py (the creature,
weekly); both merge through merge_coverage / merge_curiosity below.

Why it lives in the package rather than only in the script: the derivation was
unrunnable on the Pi. Lumen's calibration file carries `drawing_thresholds: {}`
with `update_count: 0` — the coverage derivation shipped in 2026-08-22 and was
never once applied, so `dense` is still never generated on the live creature.
A derivation nothing can run is a derivation that does not happen.

See `curiosity_drain` in display/drawing_engine.py for the formula this replays
and the defect it exists to remove. This module imports that function rather
than restating it: a copy could drift and certify a pivot the creature never
runs.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .db_paths import resolve_db_path
from .display.drawing_engine import curiosity_drain

# One percentile contract, stated once. "Pattern found" is a RELATIVE judgement:
# a piece is finding its pattern when it is more coherent than this era's own
# typical moment. Never adjusted to make an era pass — see refusal semantics in
# `evaluate`.
PIVOT_PERCENTILE = 50

# Below this a cut encodes the last few days' weather rather than a range.
# Matched to derive_drawing_thresholds.py rather than argued separately.
MIN_SAMPLES = 500
MIN_SAMPLES_PER_ERA = 120
MIN_INTERVALS_PER_PIECE = 4

# Mirrors DrawingState.attention_exhausted() and completion_reason()'s
# earned_composition branch; the looser of the two is what "reachable" means.
COMPOSITION_FLOOR = 0.2
EXHAUSTION_FLOOR = 0.15
CURIOSITY_START = 1.0  # DrawingState.reset()

# Earliest point in a piece's own mark count at which crossing is acceptable.
MIN_CROSS_FRACTION = 0.5

# Hard bound on rows pulled into a single report. drawing_trajectory holds ~96
# rows per 8h piece on a 90-day retention (~26k rows), so this is slack rather
# than a real ceiling — it exists so a runaway table cannot stall the server.
MAX_ROWS = 200_000


def percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_samples(db_path: str, days: int = 90, max_rows: int = MAX_ROWS):
    """Ordered per-piece trajectory samples grouped by era. Read-only.

    marks_delta IS NULL marks a sample this process could not attribute (a
    restart joined the piece mid-flight). Those intervals are dropped rather
    than replayed as zero marks — an unknown delta is not an absent one.
    """
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select era, piece_uid, coherence, arc_phase, marks_delta "
            "from drawing_trajectory "
            "where era is not null and coherence is not null "
            "  and marks_delta is not null and marks_delta > 0 "
            "  and timestamp > ? "
            "order by era, piece_uid, elapsed_seconds limit ?",
            (since, max_rows)).fetchall()
    finally:
        con.close()
    by_era: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for era, uid, coherence, arc_phase, marks_delta in rows:
        by_era[era][uid].append({
            "coherence": float(coherence),
            "arc_phase": arc_phase or "opening",
            "marks_delta": int(marks_delta),
        })
    return by_era, len(rows)


def replay(intervals, pivot: float):
    """Curiosity trace for one piece under `pivot`.

    Returns (final_curiosity, marks_at_first_composition_cross_or_None, marks).

    Closed form rather than a per-mark loop, and exactly equivalent to one:
    coherence and arc_phase are constant within a trajectory interval, so
    `curiosity_drain` is constant across its marks and curiosity moves
    monotonically. A monotone walk can only reach one clamp bound, so a single
    max/min reproduces per-step clamping, and the first crossing solves
    directly. tests/test_curiosity_pivot.py checks this against a naive loop —
    the equivalence is the reason it is safe to serve from the server.
    """
    curiosity = CURIOSITY_START
    marks_seen = 0
    crossed_at: Optional[int] = None
    for iv in intervals:
        n = iv["marks_delta"]
        if n <= 0:
            continue
        delta = curiosity_drain(iv["arc_phase"], iv["coherence"], pivot)
        if delta > 0:  # draining: monotonically down, floor at 0.0
            if crossed_at is None and curiosity >= COMPOSITION_FLOOR:
                # First k with curiosity - k*delta < COMPOSITION_FLOOR.
                k = int((curiosity - COMPOSITION_FLOOR) / delta) + 1
                if k <= n:
                    crossed_at = marks_seen + k
            curiosity = max(0.0, curiosity - n * delta)
        elif delta < 0:  # regenerating: monotonically up, ceiling at 1.0
            curiosity = min(1.0, curiosity - n * delta)
        marks_seen += n
        if crossed_at is None and curiosity < COMPOSITION_FLOOR:
            crossed_at = marks_seen
    return curiosity, crossed_at, marks_seen


def evaluate(pieces, pivot: float) -> Dict[str, Any]:
    """Replay every piece in an era under `pivot`. Returns a verdict.

    Two refusals, opposite failure modes:
      unreachable — no piece crosses the composition floor. That is today's bug
                    at a different pivot: a dead gate traded for a dead gate.
      premature   — the median piece crosses before MIN_CROSS_FRACTION of its
                    marks. Pieces would end before they are worked, worse than
                    ending late: the 8h cap at least produced a finished canvas.
    """
    traces = []
    for intervals in pieces.values():
        if len(intervals) < MIN_INTERVALS_PER_PIECE:
            continue
        final, crossed_at, total_marks = replay(intervals, pivot)
        if total_marks <= 0:
            continue
        traces.append({
            "final": final, "crossed_at": crossed_at,
            "cross_fraction": (crossed_at / total_marks) if crossed_at else None,
        })
    if not traces:
        return {"ok": False, "traces": 0,
                "reason": "no piece had enough usable intervals"}
    crossings = [t for t in traces if t["crossed_at"] is not None]
    verdict: Dict[str, Any] = {
        "traces": len(traces),
        "reached_composition_floor": len(crossings),
        "reach_rate": round(len(crossings) / len(traces), 3),
        "median_final_curiosity": round(
            percentile(sorted(t["final"] for t in traces), 50) or 0.0, 4),
    }
    if not crossings:
        verdict.update(ok=False, reason=(
            f"unreachable: no replayed piece reaches curiosity < "
            f"{COMPOSITION_FLOOR} under this pivot — the gate stays dead, "
            "which is the defect this derivation exists to remove"))
        return verdict
    median_fraction = percentile(
        sorted(t["cross_fraction"] for t in crossings), 50) or 0.0
    verdict["median_cross_fraction"] = round(median_fraction, 3)
    if median_fraction < MIN_CROSS_FRACTION:
        verdict.update(ok=False, reason=(
            f"premature: the median piece crosses at {median_fraction:.2f} of "
            f"its marks (floor {MIN_CROSS_FRACTION}) — pieces would end before "
            "they are worked, which is a worse failure than ending late"))
        return verdict
    verdict.update(ok=True, reason="reachable without being premature")
    return verdict


def derive(by_era, total: int):
    """(thresholds, report) for an already-loaded corpus. Writes nothing."""
    report: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}
    if total < MIN_SAMPLES:
        return thresholds, {"_refused": (
            f"only {total} usable samples (< {MIN_SAMPLES}) — a pivot derived "
            "from a sliver would encode a mood, not a range")}
    for era in sorted(by_era):
        pieces = by_era[era]
        coherence = sorted(iv["coherence"] for ivs in pieces.values() for iv in ivs)
        entry: Dict[str, Any] = {"samples": len(coherence), "pieces": len(pieces)}
        if len(coherence) < MIN_SAMPLES_PER_ERA:
            entry.update(emitted=False, reason=(
                f"only {len(coherence)} samples (< {MIN_SAMPLES_PER_ERA}); "
                "keeping the built-in"))
            report[era] = entry
            continue
        pivot = round(percentile(coherence, PIVOT_PERCENTILE) or 0.0, 4)
        entry["coherence_range"] = [round(coherence[0], 4), round(coherence[-1], 4)]
        entry["pivot"] = pivot
        # The built-in beside the proposal: the report should show the change,
        # not only the destination. The built-in is what is running right now.
        entry["baseline_builtin"] = evaluate(pieces, 0.4)
        verdict = evaluate(pieces, pivot)
        entry["verdict"] = verdict
        entry["emitted"] = bool(verdict.get("ok"))
        if verdict.get("ok"):
            thresholds[f"CURIOSITY_PIVOT_{era}"] = pivot
        report[era] = entry
    return thresholds, report


# Environment channels recorded per drawing. Reported as distributions, never
# judged here: a threshold's health is a question about the consumer's cuts, and
# this module does not know them.
_ENV_CHANNELS = ("light_lux", "external_light_lux", "ambient_temp_c",
                 "humidity_pct")


def channel_report(db_path: Optional[str] = None, days: int = 90) -> Dict[str, Any]:
    """Percentile summary of the environment channels in drawing_records.

    Why this exists: a fixed cut is only meaningful against the range of the
    signal feeding it, and that signal can change underneath it. It did — #204
    moved clarity, drawing light_regime and activity state from raw lux to the
    gated self-glow residual, ten days after activity_state's light cuts
    (<10 / <50 / >500) were last touched. Removing self-glow compresses the top
    of the range: measured live, the same room read 696 lux raw and 226 lux
    residual. Whether `> 500` is still reachable is a question about the
    residual's distribution, and nobody could ask it without a shell.

    Reports only. It names no threshold and passes no verdict — read p95 against
    whatever cut you are checking. Read-only, same posture as derive_report:
    never raises, opens the database `mode=ro`, and returns available=false with
    a reason rather than an empty summary that would read as "no data recorded".
    """
    path = resolve_db_path(db_path)
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    out: Dict[str, Any] = {"db_path": path, "days": days}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            have = {row[1] for row in con.execute(
                "pragma table_info(drawing_records)")}
            if not have:
                return {"available": False, "db_path": path,
                        "reason": "drawing_records not present"}
            channels: Dict[str, Any] = {}
            for name in _ENV_CHANNELS:
                if name not in have:
                    channels[name] = {"available": False,
                                      "reason": "column not present"}
                    continue
                vals = sorted(
                    float(v) for (v,) in con.execute(
                        f"select {name} from drawing_records "
                        f"where {name} is not null and timestamp > ?", (since,))
                )
                if not vals:
                    # Distinct from a zero reading: nothing was recorded at all.
                    channels[name] = {"available": False, "n": 0,
                                      "reason": "no non-null rows in window"}
                    continue
                channels[name] = {
                    "available": True,
                    "n": len(vals),
                    "min": round(vals[0], 4),
                    "p05": round(percentile(vals, 5) or 0.0, 4),
                    "p50": round(percentile(vals, 50) or 0.0, 4),
                    "p95": round(percentile(vals, 95) or 0.0, 4),
                    "max": round(vals[-1], 4),
                }
        finally:
            con.close()
    except sqlite3.Error as e:
        return {"available": False, "db_path": path,
                "reason": f"database unreadable: {e}"}
    except (TypeError, ValueError) as e:
        return {"available": False, "db_path": path,
                "reason": f"unusable row data: {e}"}
    out["available"] = True
    out["channels"] = channels
    return out


def derive_report(db_path: Optional[str] = None, days: int = 90,
                  max_rows: int = MAX_ROWS) -> Dict[str, Any]:
    """Full read-only report. Never raises — diagnostics must not break.

    Fails toward *unknown*: a missing table, unreadable database or bad row
    returns `available: false` with the reason, never an empty report that
    would read as "no era qualifies".
    """
    path = resolve_db_path(db_path)
    try:
        by_era, total = load_samples(path, days=days, max_rows=max_rows)
    except sqlite3.Error as e:
        return {"available": False, "reason": f"database unreadable: {e}",
                "db_path": path}
    except (TypeError, ValueError) as e:
        return {"available": False, "reason": f"unusable row data: {e}",
                "db_path": path}
    thresholds, report = derive(by_era, total)
    return {
        "available": True,
        "db_path": path,
        "days": days,
        "usable_intervals": total,
        "truncated": total >= max_rows,
        "eras": report,
        "thresholds": thresholds,
        "apply_with": (
            "python3 scripts/derive_curiosity_thresholds.py --db "
            "~/.anima/anima.db --apply ~/.anima/anima_config.json"),
    }


# ---------------------------------------------------------------------------
# Coverage-intention cuts, and the merge rules both families share
# ---------------------------------------------------------------------------
#
# Lives here, not only in scripts/derive_drawing_thresholds.py, for the same
# reason the curiosity report does: the running server applies it now
# (self_derivation.py), and a second copy of the contract could drift from the
# script's. The script imports these; it keeps only its CLI.

# A three-word vocabulary should have three reachable words: the creature's
# own foggiest third lets a piece thicken, its clearest third opens it up.
# Tertiles, not tuned numbers. Full rationale: derive_drawing_thresholds.py.
COVERAGE_PERCENTILES = {
    "COVERAGE_DENSE_BELOW": 33,
    "COVERAGE_SPARSE_ABOVE": 67,
}
# One row per PIECE at ~3 pieces/day: 90 days holds ~270 and refuses against
# the 500 floor. Widen the window, never lower the floor (CLAUDE.md, measured
# 2026-09-19). The curiosity population is ~96 rows per piece, so 90 is enough.
COVERAGE_DAYS = 365
CURIOSITY_DAYS = 90


def derive_coverage(clarity: List[float]):
    """(thresholds, None) or (None, refusal reason). Pure; writes nothing."""
    n = len(clarity)
    if n < MIN_SAMPLES:
        return None, (f"only {n} samples (< {MIN_SAMPLES}) — a cut derived "
                      "from a sliver would encode a mood, not a range")
    vals = sorted(clarity)
    out = {name: round(percentile(vals, pct), 4)
           for name, pct in COVERAGE_PERCENTILES.items()}
    # A pinned window collapses the tertiles onto each other; emitting that
    # would starve `balanced` the way the built-in 0.30 starved `dense`.
    if not out["COVERAGE_DENSE_BELOW"] < out["COVERAGE_SPARSE_ABOVE"]:
        return None, (f"cuts not separated: {out} — clarity range is "
                      "degenerate over this window")
    return out, None


def coverage_report(db_path: Optional[str] = None,
                    days: int = COVERAGE_DAYS) -> Dict[str, Any]:
    """Read-only coverage derivation. Never raises; fails toward unknown."""
    path = resolve_db_path(db_path)
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            clarity = [float(c) for (c,) in con.execute(
                "select clarity from drawing_records "
                "where clarity is not null and timestamp > ?", (since,))]
        finally:
            con.close()
    except sqlite3.Error as e:
        return {"available": False, "db_path": path,
                "reason": f"database unreadable: {e}"}
    except (TypeError, ValueError) as e:
        return {"available": False, "db_path": path,
                "reason": f"unusable row data: {e}"}
    thresholds, refused = derive_coverage(clarity)
    out: Dict[str, Any] = {"available": True, "db_path": path, "days": days,
                           "samples": len(clarity)}
    if refused:
        out["refused"] = refused
    else:
        out["thresholds"] = thresholds
    return out


def merge_coverage(existing: Optional[Dict[str, Any]],
                   emitted: Dict[str, float]) -> Dict[str, Any]:
    """drawing_thresholds with the COVERAGE_* family replaced by `emitted`.

    Merged, never replaced whole: the curiosity derivation keeps its
    CURIOSITY_PIVOT_* keys in this same dict. The script used to overwrite it,
    so running coverage after curiosity silently reverted every pivot.
    """
    out = {k: v for k, v in (existing or {}).items()
           if not k.startswith("COVERAGE_")}
    out.update(emitted)
    return out


def merge_curiosity(existing: Optional[Dict[str, Any]],
                    emitted: Dict[str, float]) -> Dict[str, Any]:
    """drawing_thresholds with the CURIOSITY_PIVOT_* family replaced.

    Stale pivots for eras this run did not emit are dropped, so a
    since-retired or since-refused era stops being steered by a number nothing
    re-verified. COVERAGE_* and any other keys are preserved.
    """
    out = {k: v for k, v in (existing or {}).items()
           if not k.startswith("CURIOSITY_PIVOT_")}
    out.update(emitted)
    return out
