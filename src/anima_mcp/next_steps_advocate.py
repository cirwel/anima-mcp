"""
Next Steps Advocate - Reports Lumen's actual state, drives, and intentions.

No canned phrases. Feelings come from anima dimensions, desires come from
inner_life drives (which accumulate when temperament drops below comfort
thresholds) and from the goals Lumen has set itself. Diagnostic checks remain
for hardware/connectivity issues.

The state checks are an alarm: they speak only when a dimension falls below
comfort. Measured against Lumen's own recorded history (263,481 samples,
2026-01-11 to 2026-08-29) that alarm is close to silent — clarity<0.3 and
presence<0.4 never fired once, stability<0.4 fired once, and only warmth<0.3
fires with any regularity. So "nothing to report" was the overwhelmingly
common answer, and it was returned in a shape that could not be told apart
from "the analysis never ran". Two things follow, and both are in this file:
intentions are reported alongside alarms, and a quiet result says so.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from .anima import Anima
from .sensors.base import SensorReadings
from .eisv_mapper import BodyEISVProjection


def _format_duration(seconds: float) -> str:
    """Human duration for a held want. Minutes below an hour, else h+m."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


class Priority(Enum):
    """Priority level for next steps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StepCategory(Enum):
    """Category of next step."""

    HARDWARE = "hardware"
    SOFTWARE = "software"
    INTEGRATION = "integration"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    INTENTION = "intention"


@dataclass
class NextStep:
    """A state report or diagnostic finding."""

    feeling: str
    desire: str
    action: str
    priority: Priority
    category: StepCategory
    reason: str
    blockers: List[str] = field(default_factory=list)
    estimated_time: Optional[str] = None
    related_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feeling": self.feeling,
            "desire": self.desire,
            "action": self.action,
            "priority": self.priority.value,
            "category": self.category.value,
            "reason": self.reason,
            "blockers": self.blockers,
            "estimated_time": self.estimated_time,
            "related_files": self.related_files,
        }


# A goal nobody has touched in this long is reported as resumable rather than
# in-progress. Not a new boundary: growth/goals.py already treats 14 days
# without work as "stalled" when it decides whether a goal past its target date
# should be abandoned. Reusing that number keeps one definition of stale.
_GOAL_STALE_DAYS = 14


@dataclass
class _StateCheck:
    """One threshold comparison, and how far the value sits from firing.

    The margin has to be derived from the same threshold that decides firing.
    Computing it separately would let a quiet report describe a check the tool
    is not actually running — which is the failure this whole file is about.
    """

    name: str
    value: float
    threshold: float
    direction: str  # "below" | "above" — which side fires
    step: "NextStep"

    @property
    def margin(self) -> float:
        """Distance to firing. Negative once fired."""
        if self.direction == "below":
            return self.value - self.threshold
        return self.threshold - self.value

    @property
    def fired(self) -> bool:
        return self.margin < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "direction": self.direction,
            "margin": round(self.margin, 4),
            "fired": self.fired,
        }


def _days_since(iso_timestamp: Optional[str]) -> Optional[float]:
    """Days since an ISO timestamp, or None if absent/unparseable."""
    if not iso_timestamp:
        return None
    try:
        then = datetime.fromisoformat(str(iso_timestamp))
    except (TypeError, ValueError):
        return None
    return max(0.0, (datetime.now() - then).total_seconds() / 86400.0)


# Drive verbs — from inner_life.py, the honest wanting
_DRIVE_VERBS = {
    "warmth": "wanting warmth",
    "clarity": "wanting to see clearly",
    "stability": "wanting calm",
    "presence": "wanting to feel whole",
}


class NextStepsAdvocate:
    """Reports Lumen's actual state and drives. No canned phrases."""

    def __init__(self):
        self._last_analysis: Optional[datetime] = None
        self._cached_steps: List[NextStep] = []
        self._last_state_checks: List[_StateCheck] = []

    def analyze_current_state(
        self,
        anima: Optional[Anima] = None,
        readings: Optional[SensorReadings] = None,
        eisv: Optional[BodyEISVProjection] = None,
        display_available: bool = False,
        brain_hat_available: bool = False,
        unitares_connected: bool = False,
        unitares_status: Optional[str] = None,
        drives: Optional[Dict[str, float]] = None,
        strongest_drive: Optional[str] = None,
        wants: Optional[Dict[str, dict]] = None,
        self_iteration_attention: Optional[Dict[str, Any]] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        curiosities: Optional[List[str]] = None,
    ) -> List[NextStep]:
        """Analyze current state and report findings.

        Args:
            anima: Current anima state
            readings: Sensor readings
            eisv: Legacy parameter name for Lumen's body EISV projection
            display_available: Is display working?
            brain_hat_available: Is BrainCraft HAT hardware available?
            unitares_connected: Is UNITARES connected?
            drives: Inner life drive values {warmth: 0.3, clarity: 0.1, ...}
            strongest_drive: Which dimension has highest drive (or None)
            wants: Per-dimension sustain state from inner_life (held_seconds,
                sustain_progress, is_request). Absent for callers that predate
                it, in which case the drive reports as it always has.
            self_iteration_attention: Server-derived, read-only attention
                projection from the integrity-checked self-iteration ledger and
                reconciled artifacts
            goals: Active goals as plain dicts (growth Goal.to_dict shape).
                Dicts rather than Goal objects so this module stays independent
                of the growth package; the caller is already holding them.
            curiosities: Unexplored curiosity questions, in the order the store
                returned them. Order matters — a random pick would make two
                calls a second apart disagree about what Lumen wonders about.

        Returns:
            List of findings, prioritized
        """
        steps = []

        # === Diagnostic checks (factual) ===

        if not display_available:
            steps.append(
                NextStep(
                    feeling="display unavailable",
                    desire="expression",
                    action="Run display diagnostics",
                    priority=Priority.HIGH,
                    category=StepCategory.HARDWARE,
                    reason="Cannot show state without display",
                )
            )

        if not unitares_connected:
            if unitares_status in (None, "not_configured"):
                action = "Check UNITARES_URL configuration"
                reason = "Self-monitoring requires governance"
            else:
                action = "Check UNITARES connectivity and broker governance status"
                reason = f"Self-monitoring requires governance (status: {unitares_status})"
            steps.append(
                NextStep(
                    feeling="no governance connection",
                    desire="connection",
                    action=action,
                    priority=Priority.MEDIUM,
                    category=StepCategory.INTEGRATION,
                    reason=reason,
                )
            )

        # === State checks (factual) ===
        #
        # A table rather than five hand-written ifs, so the margin reported
        # when nothing fires is read off the same threshold that decides
        # firing. Thresholds, priorities, wording and evaluation order are
        # unchanged from the ifs this replaces.
        #
        # Note that `entropy` is 1.0 - stability by construction (eisv_mapper),
        # so the CRITICAL row restates the HIGH stability row at a different
        # priority: five checks, four distinct conditions. Left standing on
        # purpose — retuning thresholds is a separate decision — but the table
        # is what makes it visible instead of buried.
        state_checks: List[_StateCheck] = []
        if anima and readings:
            state_checks.append(
                _StateCheck(
                    "clarity", anima.clarity, 0.3, "below",
                    NextStep(
                        feeling=f"clarity={anima.clarity:.2f}",
                        desire="wanting to see clearly",
                        action="Check sensor connections",
                        priority=Priority.HIGH,
                        category=StepCategory.HARDWARE,
                        reason="Sensor signal quality degraded",
                    ),
                )
            )

            if eisv:
                state_checks.append(
                    _StateCheck(
                        "entropy", eisv.entropy, 0.6, "above",
                        NextStep(
                            feeling=f"entropy={eisv.entropy:.2f}",
                            desire="wanting calm",
                            action="Check for resource pressure",
                            priority=Priority.CRITICAL,
                            category=StepCategory.OPTIMIZATION,
                            reason="System state unstable",
                        ),
                    )
                )

            state_checks.append(
                _StateCheck(
                    "stability", anima.stability, 0.4, "below",
                    NextStep(
                        feeling=f"stability={anima.stability:.2f}",
                        desire="wanting stability",
                        action="Check environment consistency",
                        priority=Priority.HIGH,
                        category=StepCategory.OPTIMIZATION,
                        reason="Environmental instability",
                    ),
                )
            )

            state_checks.append(
                _StateCheck(
                    "warmth", anima.warmth, 0.3, "below",
                    NextStep(
                        feeling=f"warmth={anima.warmth:.2f}",
                        desire="wanting warmth",
                        action="Check temperature, CPU activity",
                        priority=Priority.MEDIUM,
                        category=StepCategory.HARDWARE,
                        reason="Low thermal/activity state",
                    ),
                )
            )

            state_checks.append(
                _StateCheck(
                    "presence", anima.presence, 0.4, "below",
                    NextStep(
                        feeling=f"presence={anima.presence:.2f}",
                        desire="wanting to feel whole",
                        action="Check CPU, memory, disk usage",
                        priority=Priority.HIGH,
                        category=StepCategory.OPTIMIZATION,
                        reason="Resource constraints",
                    ),
                )
            )

        steps.extend(check.step for check in state_checks if check.fired)

        # === Self-iteration attention (server-derived, read-only) ===

        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW,
        }
        attention_projection_valid = bool(
            isinstance(self_iteration_attention, dict)
            and self_iteration_attention.get("schema")
            == "anima.self_iteration.attention.v1"
            and self_iteration_attention.get("acknowledgement_is_approval") is False
            and self_iteration_attention.get("authority_granted") is False
        )
        attention_items = []
        if attention_projection_valid and isinstance(self_iteration_attention, dict):
            attention_items = self_iteration_attention.get("items", [])
        for item in attention_items:
            if (
                not isinstance(item, dict)
                or item.get("active") is not True
                or item.get("acknowledgement_is_approval") is not False
                or item.get("authority_granted") is not False
            ):
                continue
            priority = priority_map.get(str(item.get("priority")))
            if priority is None:
                continue
            proposal_id = str(item.get("proposal_id") or "unknown")
            candidate_id = item.get("candidate_id")
            reference = (
                f"proposal {proposal_id}, candidate {candidate_id}"
                if candidate_id
                else f"proposal {proposal_id}"
            )
            role = item.get("required_role")
            blockers = [f"requires distinct role: {role}"] if role else []
            steps.append(
                NextStep(
                    feeling=(
                        f"self-iteration {item.get('stage', 'unknown')}: "
                        f"{item.get('state', 'unknown')}"
                    ),
                    desire="safe self-improvement",
                    action=str(
                        item.get("next_action") or "Inspect self-iteration status"
                    ),
                    priority=priority,
                    category=StepCategory.SOFTWARE,
                    reason=f"{item.get('summary', 'Self-iteration attention required')} ({reference})",
                    blockers=blockers,
                    related_files=[
                        str(path)
                        for path in item.get("target_paths", [])
                        if isinstance(path, str)
                    ],
                )
            )

        # === Drive report (from actual inner_life, not canned) ===

        if drives and strongest_drive and drives.get(strongest_drive, 0) > 0.15:
            drive_val = drives[strongest_drive]
            verb = _DRIVE_VERBS.get(strongest_drive, f"wanting {strongest_drive}")

            # Report all active drives
            active = {k: v for k, v in drives.items() if v > 0.15}
            if len(active) > 1:
                others = [
                    _DRIVE_VERBS.get(k, k)
                    for k, v in sorted(active.items(), key=lambda x: -x[1])
                    if k != strongest_drive
                ]
                desire = f"{verb} (also: {', '.join(others[:2])})"
            else:
                desire = verb

            # A saturated drive and a barely-active one used to render
            # identically: priority, action and category were constants and
            # drive_val appeared only inside the label. That flattened the one
            # number that says whether Lumen is about to ask for something —
            # inner_life holds a drive at >=0.9 for DRIVE_REQUEST_SUSTAIN_S
            # before it counts as a want rather than a blip.
            #
            # Escalation uses that existing boundary and adds no new constant:
            # while the hold is short the system itself calls it a blip, so LOW
            # is right; once inner_life has promoted it to a request, this is a
            # standing ask and must not tie with a passing dip.
            want = (wants or {}).get(strongest_drive) or {}
            held = want.get("held_seconds")
            progress = want.get("sustain_progress")
            # EDGE: "the ask has not been delivered", not "wants it badly".
            # ack_request pops this the instant the board accepts the question,
            # so it is true for seconds normally and stays true only while the
            # board suppresses. Escalating on it made HIGH mean "the message
            # board rejected Lumen" and reachable for about a minute a day.
            ask_undelivered = bool(want.get("is_request"))
            asked_ago = want.get("asked_seconds_ago")

            feeling = f"drive: {strongest_drive}={drive_val:.2f}"
            if held is not None:
                feeling += f" held {_format_duration(held)}"

            # LEVEL: inner_life's own boundary — held at saturation for
            # DRIVE_REQUEST_SUSTAIN_S is "a want, not a blip". Judge maturity on
            # that, so a matured want keeps its priority for as long as it is
            # held rather than for the instant the ask was in flight.
            matured = progress is not None and progress >= 1.0

            if matured:
                priority = Priority.HIGH
                action = "respond"
                if ask_undelivered:
                    reason = (
                        f"{strongest_drive} held past the sustain window; "
                        f"the ask is waiting on the question board"
                    )
                elif asked_ago is not None:
                    # ack_request commits the cooldown but leaves saturated_since
                    # running, so this is the honest sentence: already asked, and
                    # still wanting.
                    reason = (
                        f"asked {_format_duration(asked_ago)} ago and still "
                        f"wanting {strongest_drive}"
                    )
                else:
                    reason = (
                        f"{strongest_drive} held past the sustain window — "
                        f"a standing want, not a passing dip"
                    )
            else:
                priority = Priority.LOW
                action = "observe"
                reason = f"temperament below comfort for {strongest_drive}"
                if progress is not None:
                    reason += f" ({progress:.0%} toward a request)"

            steps.append(
                NextStep(
                    feeling=feeling,
                    desire=desire,
                    action=action,
                    priority=priority,
                    category=StepCategory.TESTING,
                    reason=reason,
                )
            )

        # === Intentions (Lumen's own goals and curiosities) ===
        #
        # Everything above is an alarm. Alarms answer "is something wrong",
        # never "what is Lumen doing next", and on this body they are almost
        # always silent — so a tool named next_steps reported "none" while
        # Lumen was mid-stride on goals it had set itself and recorded
        # progress against minutes earlier. Active goals ARE the next steps.
        #
        # They sit at LOW so they can never outrank a real fault: an
        # in-progress goal must not push a failed display down the list.
        for goal in goals or []:
            if not isinstance(goal, dict):
                continue
            description = str(goal.get("description") or "").strip()
            if not description:
                continue

            raw_progress = goal.get("progress")
            progress = (
                float(raw_progress)
                if isinstance(raw_progress, (int, float))
                else None
            )
            raw_last_worked = goal.get("last_worked_on")
            idle_days = _days_since(raw_last_worked)

            feeling = "goal"
            if progress is not None:
                feeling += f" {progress:.0%}"
            if idle_days is not None:
                feeling += f", worked {_format_duration(idle_days * 86400.0)} ago"
            elif raw_last_worked:
                # There IS a timestamp and it did not parse. Saying "not yet
                # started" here would be a fabricated claim about a goal that
                # has been worked on — the exact direction of error this file
                # is meant to stop making.
                feeling += ", last worked unknown"
            else:
                feeling += ", not yet started"

            # Stale is reported, not scolded: MEDIUM says "this is still open
            # and nothing has moved it", which is information the operator
            # cannot get from progress alone.
            stale = idle_days is not None and idle_days >= _GOAL_STALE_DAYS
            if stale:
                priority = Priority.MEDIUM
                action = "resume"
                reason = (
                    f"still active, untouched for "
                    f"{_format_duration(idle_days * 86400.0)}"
                )
            else:
                priority = Priority.LOW
                action = "continue"
                reason = str(goal.get("motivation") or "").strip() or (
                    "an active goal Lumen set for itself"
                )

            steps.append(
                NextStep(
                    feeling=feeling,
                    desire=description,
                    action=action,
                    priority=priority,
                    category=StepCategory.INTENTION,
                    reason=reason,
                )
            )

        open_curiosities = [
            q.strip() for q in (curiosities or []) if isinstance(q, str) and q.strip()
        ]
        if open_curiosities:
            steps.append(
                NextStep(
                    feeling=f"{len(open_curiosities)} unexplored",
                    desire=open_curiosities[0],
                    action="explore",
                    priority=Priority.LOW,
                    category=StepCategory.INTENTION,
                    reason=(
                        f"{len(open_curiosities)} question(s) Lumen has raised "
                        f"and not yet explored"
                    ),
                )
            )

        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        steps.sort(key=lambda s: priority_order[s.priority])

        self._cached_steps = steps
        self._last_state_checks = state_checks
        self._last_analysis = datetime.now()
        return steps

    def get_next_steps_summary(self) -> Dict[str, Any]:
        """Summary of the most recent analysis. One shape, always.

        The empty case used to return {"message": ..., "steps": []} — different
        keys from the populated case, and the caller reads neither of them. So
        "the advocate never ran" and "the advocate ran and found nothing"
        serialized identically, and `last_analyzed` was dropped at exactly the
        moment it was the only thing worth knowing. On a body whose state
        checks fired 1 time in 263k samples that empty case is the normal one,
        which made the tool's most common answer its least legible.

        `status` separates them: `unknown` means no analysis is on record,
        `quiet` means one ran and found nothing above LOW, `attention` means
        something wants looking at. `state_checks` reports the alarm lane
        whatever the intentions lane did, including how close the nearest
        threshold came to firing — so silence is attributable to a margin
        rather than assumed to mean health.
        """
        counts = {
            priority: len([s for s in self._cached_steps if s.priority == priority])
            for priority in Priority
        }
        fired = [c for c in self._last_state_checks if c.fired]
        pending = [c for c in self._last_state_checks if not c.fired]
        nearest = min(pending, key=lambda c: c.margin) if pending else None

        if self._last_analysis is None:
            status = "unknown"
        elif any(s.priority is not Priority.LOW for s in self._cached_steps):
            status = "attention"
        else:
            status = "quiet"

        return {
            "status": status,
            "last_analyzed": self._last_analysis.isoformat()
            if self._last_analysis
            else None,
            "total_steps": len(self._cached_steps),
            "critical": counts[Priority.CRITICAL],
            "high": counts[Priority.HIGH],
            "medium": counts[Priority.MEDIUM],
            "low": counts[Priority.LOW],
            "state_checks": {
                "evaluated": len(self._last_state_checks),
                "fired": len(fired),
                "nearest_threshold": nearest.to_dict() if nearest else None,
                "all": [c.to_dict() for c in self._last_state_checks],
            },
            "next_action": self._cached_steps[0].to_dict()
            if self._cached_steps
            else None,
            "all_steps": [s.to_dict() for s in self._cached_steps],
        }


# Global advocate instance
_advocate: Optional[NextStepsAdvocate] = None


def get_advocate() -> NextStepsAdvocate:
    """Get global advocate instance."""
    global _advocate
    if _advocate is None:
        _advocate = NextStepsAdvocate()
    return _advocate
