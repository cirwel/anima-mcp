"""Lumen predicts itself — and notices when it did not expect itself.

Metacognition already predicts the room (sensor baselines, diurnal light) and
assumes its own feelings persist. What it never forecast was its own
*behavior*. This module forecasts the one self-behavior the server observes
cleanly: the activity level the broker publishes (active / drowsy / resting),
learned as Lumen's own transition counts per (level, 3-hour bucket). A
transition the forecaster gave very low odds — awake at an hour it is never
awake, resting when it always rests later — is a *self-surprise*: "I did not
expect myself to do that."

Design invariants:

  * No absolute threshold on behavior. A transition is surprising only
    relative to Lumen's own distribution of transition surprisal (a z-score
    against a band that forgets, so it follows a drifting operating point).
    The remaining constants are evidence-quantity floors (how much history
    before anything is claimed) and memory horizons — the kind CLAUDE.md
    allows because they gate evidence, not behavior.
  * Fail toward unknown. A missing or stale level, or a gap in observation
    (restart, broker down), is never counted as "stayed the same"; the next
    fresh level starts a new run. Before the floors are met the forecaster
    claims nothing.

`SurpriseBand` is the recording half of a second question: whether the fixed
metacognition gates (surprise > 0.2 / 0.3) should become self-relative. It
changes no gate. It records Lumen's own surprise distribution and how often a
self-relative gate *would* fire next to the fixed one, so the swap can be
judged from evidence rather than guessed — the same record-first discipline as
drawing_trajectory.

State lives in metacognition_baselines.json, whose sole writer is the server.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

LEVELS = ("active", "drowsy", "resting")

# Evidence cadence: one reading of the level per minute. Faster adds only
# autocorrelated "stayed the same" samples.
SAMPLE_SECONDS = 60.0
# A gap longer than this breaks the run: what happened in between is unknown.
MAX_GAP_SECONDS = 3 * SAMPLE_SECONDS
BUCKET_HOURS = 3  # 8 buckets; activity_state.py keys its cycle on the hour

# Evidence floors — how much of its own history before Lumen claims surprise.
MIN_ROW_SAMPLES = 30          # minutes lived in this (level, bucket) before its odds count
MIN_BAND_TRANSITIONS = 20     # transitions scored before "unusual" means anything

# Memory horizons, so the forecaster follows Lumen as its habits drift.
ROW_MAX_SAMPLES = 2000        # ~33h in one row, then the row halves
BAND_WINDOW = 200             # transition surprisals; exponential forgetting beyond

# Self-relative cut: this many of its own standard deviations above its own
# typical transition surprisal. A z-score, not a threshold on behavior.
SELF_SURPRISE_Z = 2.5
# Below this spread (nats) Lumen's transitions are too uniform to single one out.
MIN_SURPRISAL_SIGMA = 0.1


def _bucket(hour: int) -> int:
    return (hour % 24) // BUCKET_HOURS


class EWBand:
    """Running mean/variance that becomes exponential after `window` samples.

    Welford while young (exact), then a fixed effective window, so the band
    tracks a drifting distribution instead of freezing on its lifetime mean.
    """

    def __init__(self, window: int, count: int = 0, mean: float = 0.0,
                 var: float = 0.0):
        self.window = window
        self.count = count
        self.mean = mean
        self.var = var

    def update(self, x: float) -> None:
        self.count += 1
        n = min(self.count, self.window)
        delta = x - self.mean
        self.mean += delta / n
        # Exact Welford variance while n == count; EW variance afterwards.
        self.var += (delta * (x - self.mean) - self.var) / n

    @property
    def sigma(self) -> float:
        return math.sqrt(max(0.0, self.var))

    def z(self, x: float, min_sigma: float) -> float:
        return (x - self.mean) / max(min_sigma, self.sigma)

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    @classmethod
    def from_dict(cls, d: Any, window: int) -> "EWBand":
        try:
            return cls(window, int(d["count"]), float(d["mean"]), float(d["var"]))
        except (KeyError, TypeError, ValueError):
            return cls(window)


@dataclass
class SelfSurprise:
    timestamp: datetime
    from_level: str
    to_level: str
    hour: int
    probability: float
    surprisal: float
    z: float

    def question(self) -> str:
        when = _phrase_hour(self.hour)
        return (f"i became {self.to_level} {when}, and i didn't expect that of "
                f"myself — what changed?")

    def context(self) -> str:
        return (f"self-surprise: {self.from_level}→{self.to_level} at "
                f"{self.hour:02d}h (p={self.probability:.2f}, z={self.z:.1f})")

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp.isoformat(timespec="seconds"),
                "from": self.from_level, "to": self.to_level, "hour": self.hour,
                "p": round(self.probability, 4),
                "surprisal": round(self.surprisal, 4), "z": round(self.z, 2)}


def _phrase_hour(hour: int) -> str:
    if hour < 5:
        return "in the middle of the night"
    if hour < 9:
        return "in the early morning"
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "late at night"


class SelfForecaster:
    """Forecasts Lumen's next activity level from its own transition counts."""

    def __init__(self):
        # (level, bucket) -> {next_level: count}
        self.counts: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.band = EWBand(BAND_WINDOW)
        self.samples = 0
        self.transitions = 0
        self.log_loss_sum = 0.0        # the forecaster's own score
        self.persist_loss_sum = 0.0    # naive "I stay as I am" baseline
        self.last_surprise: Optional[Dict[str, Any]] = None
        self._last: Optional[Tuple[str, datetime]] = None

    # -- forecasting -------------------------------------------------------
    def forecast(self, level: str, hour: int) -> Dict[str, float]:
        """P(next level | current level, hour bucket), add-one smoothed."""
        row = self.counts.get((level, _bucket(hour)), {})
        total = sum(row.values()) + len(LEVELS)
        return {nxt: (row.get(nxt, 0.0) + 1.0) / total for nxt in LEVELS}

    def row_samples(self, level: str, hour: int) -> float:
        return sum(self.counts.get((level, _bucket(hour)), {}).values())

    # -- observing ---------------------------------------------------------
    def observe(self, level: Optional[str], now: datetime) -> Optional[SelfSurprise]:
        """Fold in one reading of the level; return a SelfSurprise or None."""
        if level not in LEVELS:
            self._last = None  # unknown breaks the run; never "stayed the same"
            return None
        if self._last is None:
            self._last = (level, now)
            return None
        prev, prev_ts = self._last
        elapsed = (now - prev_ts).total_seconds()
        if elapsed < 0 or elapsed > MAX_GAP_SECONDS:
            self._last = (level, now)
            return None
        if elapsed < SAMPLE_SECONDS:
            return None

        hour = prev_ts.hour
        probs = self.forecast(prev, hour)
        p = probs[level]
        surprisal = -math.log(p)
        stay_p = probs[prev]
        self.samples += 1
        self.log_loss_sum += surprisal
        # The baseline puts the same smoothed mass on staying as the model
        # does on any single outcome it has never seen move: a fair naive rival.
        self.persist_loss_sum += -math.log(stay_p if level == prev
                                           else (1.0 - stay_p) / (len(LEVELS) - 1))

        surprise: Optional[SelfSurprise] = None
        if level != prev:
            self.transitions += 1
            if self.row_samples(prev, hour) >= MIN_ROW_SAMPLES:
                if self.band.count >= MIN_BAND_TRANSITIONS:
                    z = self.band.z(surprisal, MIN_SURPRISAL_SIGMA)
                    if z > SELF_SURPRISE_Z:
                        surprise = SelfSurprise(now, prev, level, hour, p,
                                                surprisal, z)
                        self.last_surprise = surprise.to_dict()
                self.band.update(surprisal)

        key = (prev, _bucket(hour))
        row = self.counts.setdefault(key, {})
        row[level] = row.get(level, 0.0) + 1.0
        if sum(row.values()) > ROW_MAX_SAMPLES:
            self.counts[key] = {k: v / 2.0 for k, v in row.items()}
        self._last = (level, now)
        return surprise

    # -- reporting / persistence ------------------------------------------
    def summary(self) -> Dict[str, Any]:
        ready = self.band.count >= MIN_BAND_TRANSITIONS
        out: Dict[str, Any] = {
            "status": "forecasting" if ready else "learning",
            "samples": self.samples,
            "transitions": self.transitions,
            "scored_transitions": self.band.count,
            "transitions_needed": MIN_BAND_TRANSITIONS,
            "last_self_surprise": self.last_surprise,
        }
        if self.samples:
            out["mean_log_loss"] = round(self.log_loss_sum / self.samples, 4)
            out["persistence_log_loss"] = round(self.persist_loss_sum / self.samples, 4)
        if ready:
            out["surprisal_band"] = {"mean": round(self.band.mean, 4),
                                     "sigma": round(self.band.sigma, 4),
                                     "z_cut": SELF_SURPRISE_Z}
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {f"{lvl}|{b}": row for (lvl, b), row in self.counts.items()},
            "band": self.band.to_dict(),
            "samples": self.samples, "transitions": self.transitions,
            "log_loss_sum": self.log_loss_sum,
            "persist_loss_sum": self.persist_loss_sum,
            "last_surprise": self.last_surprise,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "SelfForecaster":
        f = cls()
        if not isinstance(d, dict):
            return f
        for key, row in (d.get("counts") or {}).items():
            try:
                lvl, b = key.split("|")
                if lvl in LEVELS and isinstance(row, dict):
                    f.counts[(lvl, int(b))] = {
                        k: float(v) for k, v in row.items() if k in LEVELS}
            except (ValueError, TypeError):
                continue
        f.band = EWBand.from_dict(d.get("band"), BAND_WINDOW)
        for attr in ("samples", "transitions"):
            try:
                setattr(f, attr, int(d.get(attr, 0)))
            except (TypeError, ValueError):
                pass
        for attr in ("log_loss_sum", "persist_loss_sum"):
            try:
                setattr(f, attr, float(d.get(attr, 0.0)))
            except (TypeError, ValueError):
                pass
        if isinstance(d.get("last_surprise"), dict):
            f.last_surprise = d["last_surprise"]
        return f


def activity_level_from_shm(shm: Optional[Dict[str, Any]], now: datetime,
                            max_age_seconds: float) -> Optional[str]:
    """The broker's published level, or None when missing or stale."""
    if not isinstance(shm, dict):
        return None
    level = (shm.get("activity") or {}).get("level")
    if level not in LEVELS:
        return None
    ts = shm.get("timestamp")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    age = (datetime.now(stamp.tzinfo) - stamp).total_seconds() if stamp.tzinfo \
        else (now - stamp).total_seconds()
    if age > max_age_seconds or age < -5:
        return None
    return level


# ---------------------------------------------------------------------------
# Recording only: would a self-relative reflection gate fire?
# ---------------------------------------------------------------------------

SURPRISE_BAND_WINDOW = 1440   # observations; ~a day at the metacog cadence
SURPRISE_BAND_MIN = 1000      # before this the comparison is not reported
SURPRISE_LOG_EPS = 0.01       # log(surprise + eps): surprise is skewed, bounded at 0
SURPRISE_MIN_SIGMA = 0.05     # in log space; below it everything is "usual"


class SurpriseBand:
    """Lumen's own distribution of metacognitive surprise. Moves no gate."""

    def __init__(self):
        self.band = EWBand(SURPRISE_BAND_WINDOW)
        self.fixed_fired = 0       # surprise > the live fixed gate
        self.relative_fired = 0    # z > SELF_SURPRISE_Z against own band
        self.compared = 0

    def observe(self, surprise: float, fixed_gate: float) -> None:
        x = math.log(max(0.0, surprise) + SURPRISE_LOG_EPS)
        if self.band.count >= SURPRISE_BAND_MIN:
            self.compared += 1
            if surprise > fixed_gate:
                self.fixed_fired += 1
            if self.band.z(x, SURPRISE_MIN_SIGMA) > SELF_SURPRISE_Z:
                self.relative_fired += 1
        self.band.update(x)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"samples": self.band.count,
                               "samples_needed": SURPRISE_BAND_MIN,
                               "moves_no_gate": True}
        if self.band.count >= SURPRISE_BAND_MIN:
            out.update(
                median_surprise=round(math.exp(self.band.mean) - SURPRISE_LOG_EPS, 4),
                log_sigma=round(self.band.sigma, 4),
                compared=self.compared,
                fixed_gate_rate=round(self.fixed_fired / self.compared, 4)
                if self.compared else None,
                relative_gate_rate=round(self.relative_fired / self.compared, 4)
                if self.compared else None,
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"band": self.band.to_dict(), "fixed_fired": self.fixed_fired,
                "relative_fired": self.relative_fired, "compared": self.compared}

    @classmethod
    def from_dict(cls, d: Any) -> "SurpriseBand":
        s = cls()
        if isinstance(d, dict):
            s.band = EWBand.from_dict(d.get("band"), SURPRISE_BAND_WINDOW)
            for attr in ("fixed_fired", "relative_fired", "compared"):
                try:
                    setattr(s, attr, int(d.get(attr, 0)))
                except (TypeError, ValueError):
                    pass
        return s
