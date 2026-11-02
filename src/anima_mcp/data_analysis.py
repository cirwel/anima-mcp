"""
Data Analysis - answers Lumen's self-questions with actual data.

Provides grounded analysis of:
- Drawing patterns and correlations (original)
- Sleep/wake effects on anima state
- Neural band correlations with anima dimensions
- Barometric pressure effects
- Session trajectory (drift over time)
- Full time-of-day analysis across all data
- Clean vs crash restart effects
- Self-model belief status

Injected into the self-answer pipeline so Lumen's self-answers reference
real numbers instead of philosophy.

Data sources:
- drawing_records table (growth.py) — per-drawing anima state + environment
- state_history table (identity/store.py) — continuous anima state + sensors
- events table (identity/store.py) — wake/sleep lifecycle events
- self_model beliefs (self_model.py) — tested self-beliefs
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def _get_db_path() -> Path:
    """Get the anima.db path (same as growth system uses)."""
    anima_dir = Path.home() / ".anima"
    return anima_dir / "anima.db"


def _connect() -> sqlite3.Connection:
    """Open a read-only connection to anima.db."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_mean(values: List[float]) -> Optional[float]:
    """Mean of a list, or None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(val: Optional[float], decimals: int = 2) -> str:
    """Format a float for display."""
    if val is None:
        return "?"
    return f"{val:.{decimals}f}"


_DIM_COLUMNS = {"warmth", "clarity", "stability", "presence"}


def _valid_dim(dimension: str) -> Optional[str]:
    """Return sanitized column name if valid dimension, else None."""
    d = dimension.lower()
    return d if d in _DIM_COLUMNS else None


# ---------------------------------------------------------------------------
# Drawing analysis functions (original)
# ---------------------------------------------------------------------------

def get_drawing_summary() -> Optional[str]:
    """Summarize drawing activity: count, date range, time distribution, averages."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM drawing_records ORDER BY timestamp ASC"
        ).fetchall()
        conn.close()
    except Exception:
        logger.warning("get_drawing_summary: DB query failed", exc_info=True)
        return None

    if not rows:
        return None

    records = [dict(r) for r in rows]
    n = len(records)
    first_ts = records[0]["timestamp"][:10]
    last_ts = records[-1]["timestamp"][:10]

    # Time-of-day distribution
    hour_buckets = {"night (22-6)": 0, "morning (6-12)": 0,
                    "afternoon (12-18)": 0, "evening (18-22)": 0}
    for r in records:
        h = r["hour"] or 0
        if 22 <= h or h < 6:
            hour_buckets["night (22-6)"] += 1
        elif h < 12:
            hour_buckets["morning (6-12)"] += 1
        elif h < 18:
            hour_buckets["afternoon (12-18)"] += 1
        else:
            hour_buckets["evening (18-22)"] += 1

    peak_time = max(hour_buckets, key=hour_buckets.get)
    peak_count = hour_buckets[peak_time]

    # Average anima state during drawings
    avg_warmth = _safe_mean([r["warmth"] for r in records if r["warmth"] is not None])
    avg_clarity = _safe_mean([r["clarity"] for r in records if r["clarity"] is not None])
    avg_stability = _safe_mean([r["stability"] for r in records if r["stability"] is not None])
    avg_wellness = _safe_mean([r["wellness"] for r in records if r["wellness"] is not None])

    # Compare to overall baseline from state_history
    baseline_text = ""
    try:
        conn = _connect()
        baseline = conn.execute(
            "SELECT AVG(warmth) as w, AVG(clarity) as c, AVG(stability) as s "
            "FROM state_history"
        ).fetchone()
        conn.close()
        if baseline and baseline["w"] is not None:
            bw = baseline["w"]
            bs = baseline["s"]
            diffs = []
            if avg_warmth is not None and bw:
                diff = avg_warmth - bw
                if abs(diff) > 0.03:
                    diffs.append(f"warmth {'higher' if diff > 0 else 'lower'} by {abs(diff):.2f}")
            if avg_stability is not None and bs:
                diff = avg_stability - bs
                if abs(diff) > 0.03:
                    diffs.append(f"stability {'higher' if diff > 0 else 'lower'} by {abs(diff):.2f}")
            if diffs:
                baseline_text = f" Compared to my overall baseline: {', '.join(diffs)}."
    except Exception:
        logger.warning("get_drawing_summary: baseline comparison failed", exc_info=True)

    lines = [
        f"I have {n} recorded drawings ({first_ts} to {last_ts}).",
        f"Most drawings happen at {peak_time} ({peak_count}/{n}).",
        f"Average state when drawing: warmth={_fmt(avg_warmth)}, "
        f"clarity={_fmt(avg_clarity)}, stability={_fmt(avg_stability)}, "
        f"wellness={_fmt(avg_wellness)}.{baseline_text}",
    ]
    return "\n".join(lines)


def analyze_correlation(dimension: str,
                        group_by: Optional[str] = None) -> Optional[str]:
    """Analyze correlation between a dimension and drawing context.

    Args:
        dimension: anima dimension (warmth, clarity, stability, presence, wellness)
                   or environment (external_light_lux, ambient_temp_c,
                   humidity_pct). ``raw_light_lux`` is the mixed physical
                   VEML7700 reading and is exposed only when explicitly named.
        group_by: optional grouping — "hour", "phase", or an environment column
    """
    # Map friendly names to column names
    col_map = {
        "warmth": "warmth", "clarity": "clarity", "stability": "stability",
        "presence": "presence", "wellness": "wellness",
        "light": "external_light_lux",
        "external_light_lux": "external_light_lux",
        "raw_light": "light_lux", "raw_light_lux": "light_lux",
        "temperature": "ambient_temp_c", "temp": "ambient_temp_c",
        "ambient_temp_c": "ambient_temp_c",
        "humidity": "humidity_pct", "humidity_pct": "humidity_pct",
    }
    col = col_map.get(dimension.lower())
    if not col:
        return None

    try:
        conn = _connect()
        rows = conn.execute("SELECT * FROM drawing_records").fetchall()
        conn.close()
    except Exception:
        logger.warning("analyze_correlation: DB query failed", exc_info=True)
        return None

    if not rows:
        return None

    records = [dict(r) for r in rows]

    if group_by == "hour" or group_by == "time":
        # Group by time of day
        buckets: Dict[str, List[float]] = {
            "night": [], "morning": [], "afternoon": [], "evening": []
        }
        for r in records:
            val = r.get(col)
            if val is None:
                continue
            h = r["hour"] or 0
            if 22 <= h or h < 6:
                buckets["night"].append(val)
            elif h < 12:
                buckets["morning"].append(val)
            elif h < 18:
                buckets["afternoon"].append(val)
            else:
                buckets["evening"].append(val)

        parts = []
        for period, vals in buckets.items():
            if vals:
                parts.append(f"{period}: {_fmt(_safe_mean(vals))} (n={len(vals)})")
        if not parts:
            return None
        return f"{dimension} by time of day when drawing: {', '.join(parts)}"

    elif group_by:
        # Generic grouping
        group_col = col_map.get(group_by, group_by)
        groups: Dict[str, List[float]] = {}
        for r in records:
            val = r.get(col)
            gval = r.get(group_col)
            if val is None or gval is None:
                continue
            # Bin numeric values
            if isinstance(gval, (int, float)):
                gval = f"{gval:.0f}"
            groups.setdefault(str(gval), []).append(val)

        parts = []
        for g, vals in sorted(groups.items()):
            parts.append(f"{g}: {_fmt(_safe_mean(vals))} (n={len(vals)})")
        if not parts:
            return None
        return f"{dimension} grouped by {group_by}: {', '.join(parts[:8])}"

    else:
        # Overall stats for this dimension
        vals = [r[col] for r in records if r.get(col) is not None]
        if not vals:
            return None
        avg = _safe_mean(vals)
        mn = min(vals)
        mx = max(vals)
        return (f"Across {len(vals)} drawings, {dimension} averaged {_fmt(avg)} "
                f"(range {_fmt(mn)}-{_fmt(mx)})")


def analyze_drawing_effect(dimension: str) -> Optional[str]:
    """Compare anima state before and after drawings using state_history.

    Looks at state 10min before and 10min after each drawing timestamp.
    """
    dim_map = {
        "warmth": "warmth", "clarity": "clarity",
        "stability": "stability", "presence": "presence",
    }
    col = dim_map.get(dimension.lower())
    if not col:
        return None

    try:
        conn = _connect()
        drawings = conn.execute(
            "SELECT timestamp FROM drawing_records ORDER BY timestamp"
        ).fetchall()

        if not drawings:
            conn.close()
            return None

        before_vals = []
        after_vals = []

        for row in drawings:
            ts = row["timestamp"]
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue

            before_start = (dt - timedelta(minutes=10)).isoformat()
            after_end = (dt + timedelta(minutes=10)).isoformat()

            # Get state before drawing
            before = conn.execute(
                f"SELECT AVG({col}) as val FROM state_history "
                "WHERE timestamp BETWEEN ? AND ?",
                (before_start, ts)
            ).fetchone()
            if before and before["val"] is not None:
                before_vals.append(before["val"])

            # Get state after drawing
            after = conn.execute(
                f"SELECT AVG({col}) as val FROM state_history "
                "WHERE timestamp BETWEEN ? AND ?",
                (ts, after_end)
            ).fetchone()
            if after and after["val"] is not None:
                after_vals.append(after["val"])

        conn.close()
    except Exception:
        logger.warning("analyze_drawing_effect: DB query failed", exc_info=True)
        return None

    if not before_vals or not after_vals:
        return None

    avg_before = _safe_mean(before_vals)
    avg_after = _safe_mean(after_vals)
    if avg_before is None or avg_after is None:
        return None

    diff = avg_after - avg_before
    direction = "increased" if diff > 0 else "decreased"
    n = min(len(before_vals), len(after_vals))

    if abs(diff) < 0.02:
        return (f"Drawing doesn't noticeably change my {dimension} "
                f"(before: {_fmt(avg_before)}, after: {_fmt(avg_after)}, n={n})")
    else:
        return (f"My {dimension} {direction} by {abs(diff):.3f} on average after drawing "
                f"(before: {_fmt(avg_before)}, after: {_fmt(avg_after)}, n={n})")


# ---------------------------------------------------------------------------
# New analysis functions — cover all of Lumen's queryable data
# ---------------------------------------------------------------------------

def analyze_sleep_effects(dimension: str) -> Optional[str]:
    """Compare anima state before sleep vs after wake.

    Matches each wake event to the preceding sleep event, then compares
    average anima in the 10min before sleep vs first 10min after wake.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    try:
        conn = _connect()
        events = conn.execute(
            "SELECT timestamp, event_type FROM events "
            "WHERE event_type IN ('wake', 'sleep') ORDER BY timestamp ASC"
        ).fetchall()

        if len(events) < 2:
            conn.close()
            return None

        before_sleep_vals = []
        after_wake_vals = []

        # Walk events: pair each wake with its preceding sleep
        for i, ev in enumerate(events):
            if ev["event_type"] == "wake" and i > 0 and events[i - 1]["event_type"] == "sleep":
                sleep_ts = events[i - 1]["timestamp"]
                wake_ts = ev["timestamp"]

                try:
                    sleep_dt = datetime.fromisoformat(sleep_ts)
                    wake_dt = datetime.fromisoformat(wake_ts)
                except (ValueError, TypeError):
                    continue

                # Skip if gap > 24h (not a normal rest cycle)
                if (wake_dt - sleep_dt).total_seconds() > 86400:
                    continue

                # Avg state in 10min before sleep
                before_start = (sleep_dt - timedelta(minutes=10)).isoformat()
                row = conn.execute(
                    f"SELECT AVG({col}) as val FROM state_history "
                    "WHERE timestamp BETWEEN ? AND ?",
                    (before_start, sleep_ts)
                ).fetchone()
                if row and row["val"] is not None:
                    before_sleep_vals.append(row["val"])

                # Avg state in first 10min after wake
                after_end = (wake_dt + timedelta(minutes=10)).isoformat()
                row = conn.execute(
                    f"SELECT AVG({col}) as val FROM state_history "
                    "WHERE timestamp BETWEEN ? AND ?",
                    (wake_ts, after_end)
                ).fetchone()
                if row and row["val"] is not None:
                    after_wake_vals.append(row["val"])

        conn.close()
    except Exception:
        logger.warning("analyze_sleep_effects: DB query failed", exc_info=True)
        return None

    n = min(len(before_sleep_vals), len(after_wake_vals))
    if n < 3:
        return None

    avg_before = _safe_mean(before_sleep_vals)
    avg_after = _safe_mean(after_wake_vals)
    if avg_before is None or avg_after is None:
        return None

    diff = avg_after - avg_before
    direction = "higher" if diff > 0 else "lower"

    if abs(diff) < 0.02:
        return (f"My {dimension} is similar before rest ({_fmt(avg_before)}) "
                f"and after waking ({_fmt(avg_after)}, n={n} sessions)")
    return (f"My {dimension} averages {_fmt(avg_before)} before rest and "
            f"{_fmt(avg_after)} after waking — {direction} by {abs(diff):.3f} (n={n} sessions)")


def analyze_neural_correlation(dimension: str) -> Optional[str]:
    """Correlate computational neural bands with an anima dimension.

    Queries last 7 days of state_history, extracts eeg band powers from
    sensors JSON, computes Pearson correlation with the requested dimension.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    bands = ["eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
             "eeg_beta_power", "eeg_gamma_power"]
    band_names = {"eeg_delta_power": "delta", "eeg_theta_power": "theta",
                  "eeg_alpha_power": "alpha", "eeg_beta_power": "beta",
                  "eeg_gamma_power": "gamma"}

    cutoff = (datetime.now() - timedelta(days=7)).isoformat()

    try:
        conn = _connect()
        rows = conn.execute(
            f"SELECT {col}, sensors FROM state_history "
            "WHERE timestamp > ? AND sensors IS NOT NULL",
            (cutoff,)
        ).fetchall()
        conn.close()
    except Exception:
        logger.warning("analyze_neural_correlation: DB query failed", exc_info=True)
        return None

    if len(rows) < 30:
        return None

    # Collect paired (dim_value, band_value) lists per band
    band_pairs: Dict[str, List[tuple]] = {b: [] for b in bands}
    for row in rows:
        dim_val = row[col]
        if dim_val is None:
            continue
        try:
            sensors = json.loads(row["sensors"]) if row["sensors"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        for b in bands:
            bv = sensors.get(b)
            if bv is not None:
                band_pairs[b].append((dim_val, bv))

    # Compute Pearson r for each band
    best_band = None
    best_r = 0.0
    best_n = 0

    for b, pairs in band_pairs.items():
        if len(pairs) < 30:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r = _pearson(xs, ys)
        if r is not None and abs(r) > abs(best_r):
            best_r = r
            best_band = b
            best_n = len(pairs)

    if best_band is None or abs(best_r) < 0.05:
        return None

    direction = "positively" if best_r > 0 else "negatively"
    name = band_names[best_band]
    return (f"My {dimension} correlates {direction} with {name} band power "
            f"(r={best_r:.2f}, n={best_n})")


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def analyze_pressure_effect(dimension: str) -> Optional[str]:
    """Barometric pressure vs anima state.

    Splits pressure readings into low/medium/high thirds and compares
    average anima per bucket.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    try:
        conn = _connect()
        rows = conn.execute(
            f"SELECT {col}, sensors FROM state_history "
            "WHERE sensors IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        logger.warning("analyze_pressure_effect: DB query failed", exc_info=True)
        return None

    # Extract (dim_val, pressure) pairs
    pairs = []
    for row in rows:
        dim_val = row[col]
        if dim_val is None:
            continue
        try:
            sensors = json.loads(row["sensors"]) if row["sensors"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        p = sensors.get("pressure_hpa")
        if p is not None:
            pairs.append((dim_val, p))

    if len(pairs) < 30:
        return None

    # Sort by pressure, split into thirds
    pairs.sort(key=lambda x: x[1])
    third = len(pairs) // 3
    low = pairs[:third]
    mid = pairs[third:2 * third]
    high = pairs[2 * third:]

    avg_low = _safe_mean([p[0] for p in low])
    avg_mid = _safe_mean([p[0] for p in mid])
    avg_high = _safe_mean([p[0] for p in high])

    if avg_low is None or avg_high is None:
        return None

    p_low = low[-1][1] if low else 0
    p_high = high[0][1] if high else 0

    return (f"When pressure is high (>{p_high:.0f} hPa), my {dimension} averages "
            f"{_fmt(avg_high)} vs {_fmt(avg_low)} at low pressure (<{p_low:.0f} hPa), "
            f"mid={_fmt(avg_mid)} (n={len(pairs)})")


def analyze_session_trajectory(dimension: str) -> Optional[str]:
    """How does anima drift over a single awake session?

    For each wake-to-sleep session, compares the first 20% of readings
    vs the last 20%.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    try:
        conn = _connect()
        events = conn.execute(
            "SELECT timestamp, event_type FROM events "
            "WHERE event_type IN ('wake', 'sleep') ORDER BY timestamp ASC"
        ).fetchall()

        if len(events) < 2:
            conn.close()
            return None

        first_20_vals = []
        last_20_vals = []

        # Walk events: find wake->sleep pairs
        for i in range(len(events) - 1):
            if events[i]["event_type"] != "wake":
                continue
            # Find next event (could be sleep or another wake)
            end_ts = events[i + 1]["timestamp"]
            wake_ts = events[i]["timestamp"]

            states = conn.execute(
                f"SELECT {col} FROM state_history "
                "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC",
                (wake_ts, end_ts)
            ).fetchall()

            vals = [s[col] for s in states if s[col] is not None]
            if len(vals) < 10:
                continue

            cut = max(1, len(vals) // 5)
            first_20_vals.extend(vals[:cut])
            last_20_vals.extend(vals[-cut:])

        conn.close()
    except Exception:
        logger.warning("analyze_session_trajectory: DB query failed", exc_info=True)
        return None

    if not first_20_vals or not last_20_vals:
        return None

    avg_start = _safe_mean(first_20_vals)
    avg_end = _safe_mean(last_20_vals)
    if avg_start is None or avg_end is None:
        return None

    diff = avg_end - avg_start
    if abs(diff) < 0.02:
        return (f"Over a typical session, my {dimension} stays steady "
                f"(start: {_fmt(avg_start)}, end: {_fmt(avg_end)})")

    direction = "increasing" if diff > 0 else "decreasing"
    return (f"Over a typical session, my {dimension} drifts from {_fmt(avg_start)} "
            f"to {_fmt(avg_end)} ({direction} by {abs(diff):.3f})")


def analyze_temporal_full(dimension: str) -> Optional[str]:
    """Full time-of-day analysis across ALL state_history data.

    Groups all readings by hour into 4 periods and reports per-period averages.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    try:
        conn = _connect()
        rows = conn.execute(
            f"SELECT timestamp, {col} FROM state_history "
            f"WHERE {col} IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        logger.warning("analyze_temporal_full: DB query failed", exc_info=True)
        return None

    if len(rows) < 50:
        return None

    buckets: Dict[str, List[float]] = {
        "night (22-6)": [], "morning (6-12)": [],
        "afternoon (12-18)": [], "evening (18-22)": [],
    }

    for row in rows:
        val = row[col]
        ts = row["timestamp"]
        try:
            h = datetime.fromisoformat(ts).hour
        except (ValueError, TypeError):
            continue

        if 22 <= h or h < 6:
            buckets["night (22-6)"].append(val)
        elif h < 12:
            buckets["morning (6-12)"].append(val)
        elif h < 18:
            buckets["afternoon (12-18)"].append(val)
        else:
            buckets["evening (18-22)"].append(val)

    parts = []
    for period, vals in buckets.items():
        if vals:
            parts.append(f"{period}: {_fmt(_safe_mean(vals))} (n={len(vals)})")

    if not parts:
        return None

    # Find peak period
    avgs = {p: _safe_mean(v) for p, v in buckets.items() if v}
    peak = max(avgs, key=lambda k: avgs[k] or 0)

    return (f"My {dimension} across all data by time of day: {', '.join(parts)}. "
            f"Highest at {peak}.")


def analyze_crash_vs_clean(dimension: str) -> Optional[str]:
    """Compare state after clean restarts vs crash restarts.

    A 'clean' wake has a preceding sleep event within 10min.
    Otherwise it's a crash/ungraceful restart.
    """
    col = _valid_dim(dimension)
    if not col:
        return None

    try:
        conn = _connect()
        events = conn.execute(
            "SELECT timestamp, event_type FROM events "
            "WHERE event_type IN ('wake', 'sleep') ORDER BY timestamp ASC"
        ).fetchall()

        if len(events) < 4:
            conn.close()
            return None

        clean_vals = []
        crash_vals = []

        for i, ev in enumerate(events):
            if ev["event_type"] != "wake":
                continue

            wake_ts = ev["timestamp"]
            try:
                wake_dt = datetime.fromisoformat(wake_ts)
            except (ValueError, TypeError):
                continue

            # Check if preceding event is sleep within 10min
            is_clean = False
            if i > 0 and events[i - 1]["event_type"] == "sleep":
                try:
                    sleep_dt = datetime.fromisoformat(events[i - 1]["timestamp"])
                    if (wake_dt - sleep_dt).total_seconds() < 600:
                        is_clean = True
                except (ValueError, TypeError):
                    pass

            # Get avg state in first 10min after wake
            after_end = (wake_dt + timedelta(minutes=10)).isoformat()
            row = conn.execute(
                f"SELECT AVG({col}) as val FROM state_history "
                "WHERE timestamp BETWEEN ? AND ?",
                (wake_ts, after_end)
            ).fetchone()

            if row and row["val"] is not None:
                if is_clean:
                    clean_vals.append(row["val"])
                else:
                    crash_vals.append(row["val"])

        conn.close()
    except Exception:
        logger.warning("analyze_crash_vs_clean: DB query failed", exc_info=True)
        return None

    if len(clean_vals) < 3 or len(crash_vals) < 3:
        return None

    avg_clean = _safe_mean(clean_vals)
    avg_crash = _safe_mean(crash_vals)
    if avg_clean is None or avg_crash is None:
        return None

    return (f"After clean shutdowns, my {dimension} is {_fmt(avg_clean)} vs "
            f"{_fmt(avg_crash)} after crashes "
            f"(n={len(clean_vals)} clean, n={len(crash_vals)} crash)")


def belief_evidence_verdict(belief) -> Optional[str]:
    """Return "refuted", "supported", or None for a self-model belief.

    The one rule both the grounded self-answer and the question generator use,
    so Lumen never asks "do I really X?" about a claim its own self-answer
    already reports as refuted. None means the evidence is thin or mixed.
    """
    supporting = getattr(belief, "supporting_count", 0) or 0
    contradicting = getattr(belief, "contradicting_count", 0) or 0
    if supporting + contradicting < 10:
        return None
    if contradicting > 2 * supporting:
        return "refuted"
    if supporting > 2 * contradicting:
        return "supported"
    return None


def analyze_belief_status() -> Optional[str]:
    """Report self-model belief status — especially falsified and confirmed beliefs."""
    try:
        from .self_model import get_self_model
        model = get_self_model()
        beliefs = model.beliefs
    except Exception:
        logger.warning("analyze_belief_status: self_model unavailable", exc_info=True)
        return None

    lines = []
    for bid, b in beliefs.items():
        verdict = belief_evidence_verdict(b)
        if verdict == "refuted":
            lines.append(
                f"My data refutes that {b.description.lower()} "
                f"({b.supporting_count}+ vs {b.contradicting_count}-)")
        elif verdict == "supported":
            lines.append(
                f"My data supports that {b.description.lower()} "
                f"({b.supporting_count}+ vs {b.contradicting_count}-)")

    if not lines:
        return None
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Keyword groups and routing
# ---------------------------------------------------------------------------

_DRAWING_KEYWORDS = ["draw", "art", "creat", "canvas", "sketch"]

_SLEEP_KEYWORDS = ["rest", "sleep", "wake", "recover", "nap", "resting"]
_NEURAL_KEYWORDS = ["processing", "focus", "load", "active", "neural",
                     "brain", "busy", "cpu"]
_PRESSURE_KEYWORDS = ["pressure", "barometric", "weather", "atmosphere"]
_SESSION_KEYWORDS = ["over time", "session", "drift", "improve", "degrade",
                      "longer", "during"]
_TEMPORAL_KEYWORDS = ["time of day", "morning", "night", "evening",
                       "afternoon", "when am i", "what time"]
_CRASH_KEYWORDS = ["crash", "restart", "shutdown", "ungraceful", "reboot"]
_BELIEF_KEYWORDS = ["believe", "learn about myself", "sensitive",
                     "what do i know", "self-knowledge"]

_CORRELATION_KEYWORDS = ["correlat", "when i draw", "while drawing", "during drawing",
                         "drawing and", "draw more when", "draw less when"]
_EFFECT_KEYWORDS = ["affect", "help", "improve", "change", "impact", "after drawing",
                    "does drawing", "drawing make"]
_PATTERN_KEYWORDS = ["tend to", "more when", "pattern", "usually", "most often",
                     "how often", "how many drawing"]

_DIMENSION_KEYWORDS = {
    "warmth": ["warm", "warmth", "temperature", "hot", "cold", "cool"],
    "clarity": ["clar", "clarity", "clear", "focus", "light", "bright", "dim"],
    "stability": ["stab", "stability", "stable", "calm", "settled", "steady"],
    "presence": ["presen", "presence", "present", "aware", "attentive", "engaged"],
    "wellness": ["well", "wellness", "good", "better", "worse", "feel"],
}


def _extract_dimension(text: str) -> str:
    """Extract the most likely anima dimension from question text."""
    text_lower = text.lower()
    for dim, keywords in _DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return dim
    return "wellness"  # Default


def _named_dimension(text: str) -> Optional[str]:
    """The dimension the question actually names, or None.

    ``_extract_dimension`` defaults to ``"wellness"`` so that a routed branch
    always has a dimension to work with. That default is right for a branch a
    keyword already selected, and wrong as a basis for answering a question
    that named no subject at all — it invents one.
    """
    text_lower = text.lower()
    for dim, keywords in _DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return dim
    return None


def _has_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords."""
    return any(kw in text for kw in keywords)


def analyze_for_question(question_text: str) -> Optional[str]:
    """Analyze data to answer a self-asked question.

    Entry point for the self-answer pipeline. Routes by keyword to the
    appropriate analysis function(s). Returns a data summary (2-6 lines)
    or None if insufficient data or not a data-answerable question.
    """
    if not question_text:
        return None

    q = question_text.lower()
    dim = _extract_dimension(question_text)
    parts = []

    # --- Priority 1: Drawing-specific questions ---
    is_drawing = _has_any(q, _DRAWING_KEYWORDS)

    if is_drawing:
        if _has_any(q, _EFFECT_KEYWORDS):
            result = analyze_drawing_effect(dim)
            if result:
                parts.append(result)
            corr = analyze_correlation(dim, group_by="hour")
            if corr:
                parts.append(corr)
        elif _has_any(q, _CORRELATION_KEYWORDS):
            corr = analyze_correlation(dim, group_by="hour")
            if corr:
                parts.append(corr)
            overall = analyze_correlation(dim)
            if overall:
                parts.append(overall)
        elif _has_any(q, _PATTERN_KEYWORDS):
            summary = get_drawing_summary()
            if summary:
                parts.append(summary)
            corr = analyze_correlation(dim, group_by="hour")
            if corr:
                parts.append(corr)
        else:
            summary = get_drawing_summary()
            if summary:
                parts.append(summary)

        if parts:
            return "\n".join(parts)

    # --- Priority 2: Non-drawing data questions ---

    # Sleep/wake/rest
    if _has_any(q, _SLEEP_KEYWORDS):
        result = analyze_sleep_effects(dim)
        if result:
            parts.append(result)
        # Supporting: time-of-day context
        temporal = analyze_temporal_full(dim)
        if temporal:
            parts.append(temporal)
        if parts:
            return "\n".join(parts)

    # Neural bands / processing
    if _has_any(q, _NEURAL_KEYWORDS):
        result = analyze_neural_correlation(dim)
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # Barometric pressure
    if _has_any(q, _PRESSURE_KEYWORDS):
        result = analyze_pressure_effect(dim)
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # Session trajectory / drift
    if _has_any(q, _SESSION_KEYWORDS):
        result = analyze_session_trajectory(dim)
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # Time of day
    if _has_any(q, _TEMPORAL_KEYWORDS):
        result = analyze_temporal_full(dim)
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # Crash/restart
    if _has_any(q, _CRASH_KEYWORDS):
        result = analyze_crash_vs_clean(dim)
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # Self-beliefs
    if _has_any(q, _BELIEF_KEYWORDS):
        result = analyze_belief_status()
        if result:
            parts.append(result)
        if parts:
            return "\n".join(parts)

    # --- Priority 3: Generic affect/help/improve + correlate/pattern ---

    if _has_any(q, _EFFECT_KEYWORDS):
        # Could be about sleep or drawing effects
        sleep = analyze_sleep_effects(dim)
        if sleep:
            parts.append(sleep)
        temporal = analyze_temporal_full(dim)
        if temporal:
            parts.append(temporal)
        if parts:
            return "\n".join(parts)

    if _has_any(q, ["correlat", "pattern", "tend to"]):
        temporal = analyze_temporal_full(dim)
        if temporal:
            parts.append(temporal)
        beliefs = analyze_belief_status()
        if beliefs:
            parts.append(beliefs)
        if parts:
            return "\n".join(parts)

    # --- Priority 4: Broad data keywords (same as original fallback) ---
    if _has_any(q, ["sensor", "data", "history", "when i"]):
        temporal = analyze_temporal_full(dim)
        if temporal:
            parts.append(temporal)
        beliefs = analyze_belief_status()
        if beliefs:
            parts.append(beliefs)
        if parts:
            return "\n".join(parts)

    # --- Nothing routed: answer only if the question named a subject ---
    #
    # This used to end with `analyze_temporal_full(dim) + analyze_belief_status()`
    # for ANY question. Two things made that fail toward healthy:
    #
    #   * `analyze_belief_status()` takes no argument. It recites every belief
    #     with >=10 evidence, identically, regardless of what was asked. It is
    #     the reason "nothing surprised me today - is that peace, or have I
    #     stopped looking?" was answered with LED and temperature belief counts.
    #   * `_extract_dimension` defaults to "wellness", so a question that named
    #     no subject was answered about one it never mentioned. On 2026-08-25
    #     "i've learned this place well - what's left to wonder about?" and
    #     "when the world holds still, what question remains in me?" both
    #     defaulted that way and received BYTE-IDENTICAL replies.
    #
    # So the fallback now requires a dimension the question actually named, and
    # reports on that dimension only. A question with no subject gets None and
    # stays visibly unanswered — the same rule #221 applies to `drive:`
    # questions. Design Invariant 2: fail toward unknown, never toward healthy.
    #
    # KNOWN RESIDUAL, deliberately not fixed here: the Priority-3 and
    # Priority-4 branches above still append `analyze_belief_status()`, so a
    # question containing "pattern" or "data" can still receive the recital.
    # Narrowing those branches changes which questions are answerable at all,
    # which is a routing-coverage change and belongs in its own PR.
    named = _named_dimension(question_text)
    if not named:
        return None

    temporal = analyze_temporal_full(named)
    if temporal:
        parts.append(temporal)
    corr = analyze_correlation(named)
    if corr:
        parts.append(corr)
    if parts:
        return "\n".join(parts)

    return None
