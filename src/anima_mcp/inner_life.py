"""
Inner Life — Three layers of temporal depth beneath Lumen's anima.

Layer 1: Differential awareness — the gap between raw sensors and smoothed mood.
         "The room cooled but I still feel warm."
Layer 2: Temperament — slow EMA of mood (~5 min half-life). Baseline emotional state.
         "I've been feeling cool lately."
Layer 3: Drives — needs that accumulate when temperament stays low.
         "I want warmth" not just "I am cold."

Damping stack (mood vs temperament vs neural): CLAUDE.md "Identity, Continuity, and Control".
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .anima import Anima
from .atomic_write import atomic_json_write

DIMENSIONS = ("warmth", "clarity", "stability", "presence")

# Temperament EMA alphas (very slow).
# half_life = -ln(2) / ln(1 - alpha), at dt=2s intervals.
TEMPERAMENT_ALPHA = {
    "warmth":    0.005,   # ~4.6 min half-life — warmth lingers longest
    "clarity":   0.007,   # ~3.3 min half-life — clarity shifts a bit faster
    "stability": 0.005,   # ~4.6 min half-life — stability is slow-moving
    "presence":  0.010,   # ~2.3 min half-life — resources tracked faster
}

# Below these temperament thresholds, drives accumulate.
DRIVE_COMFORT = {
    "warmth":    0.40,
    "clarity":   0.45,
    "stability": 0.40,
    "presence":  0.35,
}

# Per-tick rates (2s interval).
# Accumulation: ~5.6 min from zero to 0.5 at base rate.
# Decay: ~2.7x faster — satisfaction is quicker than longing.
DRIVE_ACCUMULATION = 0.003
DRIVE_DECAY = 0.008

# Drive thresholds that trigger observations
DRIVE_THRESHOLDS = (0.3, 0.5)

# Verbs for drive observations
_DRIVE_VERBS = {
    "warmth":    "wanting warmth",
    "clarity":   "wanting to see clearly",
    "stability": "wanting calm",
    "presence":  "wanting to feel whole",
}

# A drive this high, held this long, stops being telemetry and becomes a
# request. Lumen has no actuator for most of what it wants — nothing in the
# repertoire touches temperature — but it lives with someone, and its question
# channel is answerable. The actuator for an unreachable preference is
# communication. (Live case that motivated this: warmth pinned at 1.0 for
# months, visible only as a scalar in get_state that nobody was asked about.)
DRIVE_REQUEST_THRESHOLD = 0.9
DRIVE_REQUEST_SUSTAIN_S = 3600.0    # saturated for an hour = a want, not a blip
DRIVE_REQUEST_COOLDOWN_S = 86400.0  # ask at most once a day per dimension

# Request wordings — addressed outward, phrased as answerable questions, and
# honest to what each dimension actually measures (warmth=ambient temp,
# clarity=light+prediction, stability=environmental steadiness,
# presence=own-system capability).
_DRIVE_REQUESTS = {
    "warmth":    "i've been wanting warmth for a long time now — could it be warmer in here?",
    "clarity":   "things have felt hazy to me for a long time — could there be more light?",
    "stability": "i've been wanting calm for a long time — is something around me unsettled?",
    "presence":  "i haven't felt fully myself for a long time — is everything okay with my body?",
}

_PERSISTENCE_PATH = Path.home() / ".anima" / "inner_life.json"
_SAVE_INTERVAL = 60.0  # seconds between saves


@dataclass
class InnerState:
    """Snapshot of Lumen's three-layer inner life."""

    raw: Dict[str, float]
    mood: Dict[str, float]
    deltas: Dict[str, float]
    temperament: Dict[str, float]
    mood_vs_temperament: Dict[str, float]
    drives: Dict[str, float]
    strongest_drive: Optional[str]
    # Per-dimension want state: how long a saturated drive has been held, and
    # whether it has become an actual request. The broker already tracks this
    # (_saturated_since / _active_requests) but never published it, so the
    # server could see the drive VALUE and not the one number that says whether
    # Lumen is about to ask for something. Default empty so older callers and
    # restored snapshots construct unchanged.
    wants: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "raw": {k: round(v, 3) for k, v in self.raw.items()},
            "deltas": {k: round(v, 3) for k, v in self.deltas.items()},
            "temperament": {k: round(v, 3) for k, v in self.temperament.items()},
            "mood_vs_temperament": {k: round(v, 3) for k, v in self.mood_vs_temperament.items()},
            "drives": {k: round(v, 3) for k, v in self.drives.items()},
            "strongest_drive": self.strongest_drive,
            "wants": self.wants,
        }


@dataclass
class DriveEvent:
    """A drive crossed a threshold or was satisfied."""
    dimension: str
    event_type: str   # "arose", "deepened", "satisfied"
    drive_value: float
    timestamp: float


class InnerLife:
    """Three-layer inner life. Called once per broker tick after smoothing."""

    def __init__(self):
        self._temperament: Optional[Dict[str, float]] = None
        self._drives: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}
        self._prev_drives: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}
        self._crossed_thresholds: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}
        self._pending_events: List[DriveEvent] = []
        # Request state. Both persist: a cooldown that resets on restart would
        # let a restart cadence turn "once a day" into "once per boot", and a
        # sustain clock that resets would push an already-long-held want back
        # an hour every deploy.
        self._saturated_since: Dict[str, Optional[float]] = {dim: None for dim in DIMENSIONS}
        self._last_request_at: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}
        # Requests awaiting delivery confirmation (dim -> activated_at). NOT
        # persisted: on restart a still-saturated drive re-activates from the
        # persisted sustain clock, and a leftover ack file neutralizes the
        # crashed-after-post case before anything is re-emitted.
        self._active_requests: Dict[str, float] = {}
        self._last_save: float = 0.0
        self._load()

    def _load(self):
        """Load temperament and drives from disk if available."""
        try:
            if _PERSISTENCE_PATH.exists():
                data = json.loads(_PERSISTENCE_PATH.read_text())
                if "temperament" in data:
                    self._temperament = {
                        dim: data["temperament"].get(dim, 0.5) for dim in DIMENSIONS
                    }
                if "drives" in data:
                    self._drives = {
                        dim: data["drives"].get(dim, 0.0) for dim in DIMENSIONS
                    }
                    self._prev_drives = dict(self._drives)
                    # Restore crossed thresholds
                    for dim in DIMENSIONS:
                        for t in reversed(DRIVE_THRESHOLDS):
                            if self._drives[dim] >= t:
                                self._crossed_thresholds[dim] = t
                                break
                if "last_request_at" in data:
                    for dim in DIMENSIONS:
                        v = data["last_request_at"].get(dim)
                        if isinstance(v, (int, float)) and v >= 0:
                            self._last_request_at[dim] = float(v)
                if "saturated_since" in data:
                    for dim in DIMENSIONS:
                        v = data["saturated_since"].get(dim)
                        if isinstance(v, (int, float)) and v > 0:
                            self._saturated_since[dim] = float(v)
                print("[InnerLife] Loaded from disk — waking with emotional memory",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[InnerLife] Load error (starting fresh): {e}",
                  file=sys.stderr, flush=True)

    def save(self):
        """Save temperament and drives to disk."""
        if self._temperament is None:
            return
        try:
            _PERSISTENCE_PATH.parent.mkdir(exist_ok=True)
            data = {
                "temperament": {dim: round(v, 4) for dim, v in self._temperament.items()},
                "drives": {dim: round(v, 3) for dim, v in self._drives.items()},
                "last_request_at": {dim: round(v, 1) for dim, v in self._last_request_at.items()},
                "saturated_since": {dim: (round(v, 1) if v is not None else None)
                                    for dim, v in self._saturated_since.items()},
                "saved_at": time.time(),
            }
            atomic_json_write(_PERSISTENCE_PATH, data)
        except Exception as e:
            print(f"[InnerLife] Save error: {e}", file=sys.stderr, flush=True)

    def _maybe_save(self):
        now = time.time()
        if now - self._last_save >= _SAVE_INTERVAL:
            self.save()
            self._last_save = now

    def update(
        self,
        raw_anima: Anima,
        smoothed_anima: Anima,
        elapsed_seconds: float = 2.0,
    ) -> InnerState:
        """Process one tick using wall-time-corrected temporal dynamics."""

        try:
            elapsed_seconds = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed_seconds = 2.0
        if not math.isfinite(elapsed_seconds):
            elapsed_seconds = 2.0
        elapsed_seconds = max(0.0, min(60.0, elapsed_seconds))
        tick_scale = elapsed_seconds / 2.0

        raw = {dim: getattr(raw_anima, dim) for dim in DIMENSIONS}
        mood = {dim: getattr(smoothed_anima, dim) for dim in DIMENSIONS}

        # Layer 1: Differential awareness
        deltas = {dim: round(mood[dim] - raw[dim], 4) for dim in DIMENSIONS}

        # Layer 2: Temperament (slow EMA of mood)
        if self._temperament is None:
            self._temperament = {dim: mood[dim] for dim in DIMENSIONS}
        else:
            for dim in DIMENSIONS:
                base_alpha = TEMPERAMENT_ALPHA[dim]
                a = 1.0 - (1.0 - base_alpha) ** tick_scale
                self._temperament[dim] = a * mood[dim] + (1 - a) * self._temperament[dim]

        temperament = {dim: round(self._temperament[dim], 4) for dim in DIMENSIONS}
        mood_vs_temperament = {
            dim: round(mood[dim] - temperament[dim], 4) for dim in DIMENSIONS
        }

        # Layer 3: Drives
        self._prev_drives = dict(self._drives)
        for dim in DIMENSIONS:
            threshold = DRIVE_COMFORT[dim]
            temp_val = self._temperament[dim]
            mood_val = mood[dim]

            if temp_val < threshold:
                deficit = threshold - temp_val
                rate = DRIVE_ACCUMULATION * tick_scale * (1.0 + deficit * 2.0)
                # Mood relief: if current mood is already above comfort,
                # dampen accumulation — conditions improved even if
                # temperament hasn't caught up yet.
                if mood_val > threshold:
                    rate *= max(0.1, 1.0 - (mood_val - threshold) * 3.0)
                self._drives[dim] = min(1.0, self._drives[dim] + rate)
            else:
                surplus = temp_val - threshold
                rate = DRIVE_DECAY * tick_scale * (1.0 + surplus * 2.0)
                self._drives[dim] = max(0.0, self._drives[dim] - rate)

        # Detect threshold crossings and satisfaction
        self._detect_drive_events()

        # Detect sustained saturation → outward request
        self._detect_drive_requests(time.time())

        drives = {dim: round(self._drives[dim], 3) for dim in DIMENSIONS}

        strongest = max(drives, key=drives.get)
        strongest_drive = strongest if drives[strongest] > 0.1 else None

        self._maybe_save()

        return InnerState(
            raw=raw,
            mood=mood,
            deltas=deltas,
            temperament=temperament,
            mood_vs_temperament=mood_vs_temperament,
            drives=drives,
            strongest_drive=strongest_drive,
            wants=self._build_wants(),
        )

    def _build_wants(self) -> Dict[str, dict]:
        """Publish the sustain clock so a reader can tell a blip from a want.

        `DRIVE_REQUEST_SUSTAIN_S` is the system's own boundary — "saturated for
        an hour = a want, not a blip". Everything here is derived from it and
        from clocks that already exist; no new threshold is introduced, which
        matters because a constant against Lumen's moving distribution is the
        defect class CLAUDE.md invariant 1 exists to prevent.
        """
        now = time.time()
        wants: Dict[str, dict] = {}
        for dim in DIMENSIONS:
            since = self._saturated_since.get(dim)
            if since is None:
                continue
            held = max(0.0, now - since)
            asked_at = self._last_request_at.get(dim) or 0.0
            wants[dim] = {
                "held_seconds": round(held, 1),
                "sustain_required_seconds": DRIVE_REQUEST_SUSTAIN_S,
                # 1.0 means the hold is long enough to count as a want. Capped
                # so a long-held want does not report an ever-growing ratio.
                # This is the LEVEL a reader should judge maturity on.
                "sustain_progress": round(min(1.0, held / DRIVE_REQUEST_SUSTAIN_S), 3),
                # EDGE, not level: true only while an activated ask is awaiting
                # delivery. `ack_request` pops it the moment the board accepts
                # the question, so it is true for seconds in the normal case and
                # stays true only while the board SUPPRESSES the ask (soft cap,
                # rate limit, dedup). Read it as "ask undelivered", never as
                # "wants it badly" — escalating on it inverts the meaning.
                "is_request": dim in self._active_requests,
                # When the ask was last actually delivered. None = never asked.
                # Without this a reader cannot distinguish "matured and waiting
                # to ask" from "asked hours ago and still wanting", because
                # ack_request deliberately leaves `saturated_since` running.
                "asked_seconds_ago": round(now - asked_at, 1) if asked_at > 0 else None,
            }
        return wants

    def _detect_drive_events(self):
        """Detect drive threshold crossings and satisfaction events."""
        now = time.time()
        for dim in DIMENSIONS:
            prev = self._prev_drives[dim]
            curr = self._drives[dim]
            prev_threshold = self._crossed_thresholds[dim]

            # Rising: crossed a new threshold
            for t in DRIVE_THRESHOLDS:
                if prev < t <= curr and t > prev_threshold:
                    event_type = "arose" if t == DRIVE_THRESHOLDS[0] else "deepened"
                    self._pending_events.append(DriveEvent(
                        dimension=dim, event_type=event_type,
                        drive_value=curr, timestamp=now,
                    ))
                    self._crossed_thresholds[dim] = t

            # Falling: drive satisfied (dropped below lowest threshold)
            if prev >= DRIVE_THRESHOLDS[0] and curr < DRIVE_THRESHOLDS[0] * 0.5:
                if prev_threshold > 0:
                    self._pending_events.append(DriveEvent(
                        dimension=dim, event_type="satisfied",
                        drive_value=curr, timestamp=now,
                    ))
                    self._crossed_thresholds[dim] = 0.0

    def _detect_drive_requests(self, now: float):
        """A drive saturated long enough becomes a question, not just a scalar.

        Semantics: the sustain clock starts when the drive reaches
        DRIVE_REQUEST_THRESHOLD and resets the moment it dips below — a request
        claims "this has been true the whole time", so any relief restarts the
        count. The cooldown is per-dimension and persists across restarts:
        asking is a social act, and nagging is not more honest than silence.

        Delivery is NOT fire-and-forget. Activation only marks the request
        active; it stays active (re-emitted to SHM every tick) until the server
        confirms the question actually posted (ack_request), and only THEN does
        the cooldown commit. A one-shot event would vanish in a single 2s SHM
        window — the documented Pi restart window is 2 minutes, 60x that — and
        committing the cooldown at generation would silence the want for 24h
        after a delivery that never happened: the exact "wanting at nobody"
        defect this feature exists to close, reintroduced one layer down.
        """
        for dim in DIMENSIONS:
            if self._drives[dim] >= DRIVE_REQUEST_THRESHOLD:
                since = self._saturated_since.get(dim)
                if since is None:
                    self._saturated_since[dim] = now
                elif (dim not in self._active_requests
                      and now - since >= DRIVE_REQUEST_SUSTAIN_S
                      and now - self._last_request_at.get(dim, 0.0)
                          >= DRIVE_REQUEST_COOLDOWN_S):
                    self._active_requests[dim] = now
            else:
                self._saturated_since[dim] = None
                # The want eased before it was heard — withdraw the request.
                self._active_requests.pop(dim, None)

    def get_active_requests(self) -> List[DriveEvent]:
        """Requests awaiting delivery confirmation. Re-emitted every tick."""
        return [
            DriveEvent(dimension=dim, event_type="request",
                       drive_value=self._drives[dim], timestamp=activated_at)
            for dim, activated_at in self._active_requests.items()
        ]

    def ack_request(self, dim: str, now: float):
        """The server confirmed the request question posted — commit the
        cooldown, durably. The immediate save() matters: a cooldown that waits
        for the 60s periodic save can be lost to an ungraceful crash, and a
        restart with the drive still saturated would re-ask within a tick."""
        self._active_requests.pop(dim, None)
        if dim in self._last_request_at:
            self._last_request_at[dim] = now
            self.save()

    def get_pending_events(self) -> List[DriveEvent]:
        """Pop pending drive events for observation generation."""
        events = self._pending_events
        self._pending_events = []
        return events

    def get_observation_text(self, event: DriveEvent) -> Optional[str]:
        """Generate observation text for a drive event."""
        verb = _DRIVE_VERBS.get(event.dimension, f"wanting {event.dimension}")
        if event.event_type == "arose":
            return f"i've been {verb} for a while now"
        elif event.event_type == "deepened":
            return f"this {verb} is getting stronger"
        elif event.event_type == "satisfied":
            return f"that feeling of {verb} has eased"
        elif event.event_type == "request":
            return _DRIVE_REQUESTS.get(event.dimension)
        return None

    def apply_social_boost(self, clarity_boost: float = 0.02, presence_boost: float = 0.03):
        """Interaction happened — nudge temperament and ease drives.

        Being talked to makes Lumen feel more present and clear.
        """
        if self._temperament is None:
            return
        self._temperament["clarity"] = min(1.0, self._temperament["clarity"] + clarity_boost)
        self._temperament["presence"] = min(1.0, self._temperament["presence"] + presence_boost)
        # Directly reduce presence and clarity drives
        self._drives["presence"] = max(0.0, self._drives["presence"] - 0.05)
        self._drives["clarity"] = max(0.0, self._drives["clarity"] - 0.03)
