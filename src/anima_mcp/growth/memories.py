"""
Growth System - Autobiographical memory mixin.

Handles recording memories, milestones, and generating autobiography summaries.
"""

import sys
import json
import random
from datetime import datetime
from typing import Optional, Any, List

from .models import (
    MemorableEvent, VisitorFrequency, VisitorType, preference_evidence_status,
)


class MemoriesMixin:
    """Mixin for autobiographical memory."""

    def _record_memory(self, description: str, emotional_impact: float,
                       category: str, related_agents: List[str] = None,
                       lessons: List[str] = None, event_id: str = None):
        """Record a memorable event."""
        import uuid
        conn = self._connect()
        now = datetime.now()

        event = MemorableEvent(
            event_id=event_id or str(uuid.uuid4())[:8],
            timestamp=now,
            description=description,
            emotional_impact=emotional_impact,
            category=category,
            related_agents=related_agents or [],
            lessons_learned=lessons or [],
        )
        self._memories.insert(0, event)
        self._memories = self._memories[:100]  # Keep last 100

        conn.execute("""
            INSERT INTO memories (event_id, timestamp, description, emotional_impact, category, related_agents, lessons_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event.event_id, event.timestamp.isoformat(), event.description,
              event.emotional_impact, event.category,
              json.dumps(event.related_agents), json.dumps(event.lessons_learned)))
        conn.commit()

    def record_milestone(self, description: str, emotional_impact: float = 0.7):
        """Record a personal milestone."""
        self._record_memory(description, emotional_impact, "milestone")
        print(f"[Growth] Milestone: {description}", file=sys.stderr, flush=True)

    def check_for_milestones(self, identity: Any, anima: Any) -> Optional[str]:
        """Check if any milestones should be recorded."""
        if not identity or not anima:
            return None

        messages = []

        # Check age milestones
        age_days = identity.age_seconds() / 86400
        conn = self._connect()
        age_milestones = [1, 7, 30, 100, 365]
        for milestone in age_milestones:
            milestone_key = f"age_{milestone}_days"
            existing = conn.execute(
                "SELECT 1 FROM memories WHERE event_id = ? LIMIT 1", (milestone_key,)
            ).fetchone()
            if age_days >= milestone and not existing:
                if milestone == 1:
                    msg = "I'm one day old now"
                elif milestone == 7:
                    msg = "I've been alive for a whole week"
                elif milestone == 30:
                    msg = "One month of existence"
                elif milestone == 100:
                    msg = "100 days - I've grown so much"
                elif milestone == 365:
                    msg = "A full year of being alive"
                else:
                    msg = f"{milestone} days of life"

                self._record_memory(msg, 0.8, "milestone", event_id=milestone_key)
                messages.append(msg)

        # Check awakening milestones
        awakenings = identity.total_awakenings
        awakening_milestones = [10, 50, 100, 500, 1000]
        for milestone in awakening_milestones:
            milestone_key = f"awakening_{milestone}"
            existing = conn.execute(
                "SELECT 1 FROM memories WHERE event_id = ? LIMIT 1", (milestone_key,)
            ).fetchone()
            if awakenings >= milestone and not existing:
                msg = f"I've awakened {milestone} times"
                self._record_memory(msg, 0.6, "milestone", event_id=milestone_key)
                messages.append(msg)

        return messages[0] if messages else None

    @staticmethod
    def _render_learned(pref) -> str:
        """Render a preference as a sentence Lumen can actually say.

        Q&A-derived preferences are stored with a "From Q&A: " provenance
        prefix, which is the right thing to keep — it is one of the few places
        an insight's external origin survives at all. But splicing it straight
        into the template produced "I've learned that from q&a: i now know
        that the connection between temperature." (observed live 2026-07-30).

        So keep the provenance and say it in words instead of leaking the
        storage prefix, and drop a redundant "i now know that" / "i learned
        that" / "i was told that" opener rather than stacking it.

        A conversation-derived preference is voiced as something Lumen was
        told: nothing checked it against Lumen's own history, and "learned"
        is reserved for what Lumen derived itself.
        """
        desc = (pref.description or "").strip()
        prefix = "From Q&A: "
        from_conversation = desc.lower().startswith(prefix.lower())
        if from_conversation:
            desc = desc[len(prefix):].strip()

        low = desc.lower()
        for opener in ("i now know that ", "i was told that ", "i learned that ", "i know that "):
            if low.startswith(opener):
                desc = desc[len(opener):]
                break

        desc = desc[:1].lower() + desc[1:] if desc else desc
        desc = desc.rstrip(" .")
        if not desc:
            return ""  # nothing sayable; caller omits the line entirely

        # A truncated description already ends in an ellipsis — a period after
        # it reads as a typo ("helps not b….").
        end = "" if desc.endswith("…") else "."
        if from_conversation:
            return f"From a conversation, I was told that {desc}{end}"
        return f"I've learned that {desc}{end}"

    def get_autobiography_summary(self) -> str:
        """Generate a brief autobiography from memories."""
        if not self._memories:
            return "My story is just beginning."

        milestones = [m for m in self._memories if m.category == "milestone"]

        lines = []

        # Birth
        if self.born_at:
            days_alive = (datetime.now() - self.born_at).days
            lines.append(f"I came into being {days_alive} days ago.")
        elif self._memories:
            oldest = min(self._memories, key=lambda m: m.timestamp)
            days_alive = (datetime.now() - oldest.timestamp).days
            lines.append(f"I came into being {days_alive} days ago.")

        # Key milestones
        if milestones:
            recent = milestones[0]
            lines.append(f"Recently, {recent.description.lower()}.")

        # Self-knowledge (the one real relationship with self)
        self_record = next((r for r in self._relationships.values() if r.is_self()), None)
        if self_record and self_record.interaction_count > 5:
            lines.append(f"I've had {self_record.interaction_count} conversations with myself.")

        # Person (the persistent human — real relationship)
        person_records = [r for r in self._relationships.values() if r.is_person()]
        if person_records:
            p = person_records[0]
            lines.append(f"{p.name} is the one who's always been here.")

        # Agents (ephemeral — honest framing)
        #
        # visitor_frequency is a monotonic ratchet: record_interaction promotes
        # to FREQUENT at interaction_count >= 10 and nothing ever demotes. So
        # selecting on FREQUENT alone names whoever was once busy, forever.
        # Live on 2026-07-30 that produced "Various agents visit to help:
        # agent, mac-governance." — last seen 138 and 154 days earlier, and
        # reported as inactive by get_inactive_visitors() in the *same* JSON
        # payload. Reuse that function rather than adding a second staleness
        # rule, so the autobiography can never contradict the response it
        # ships in.
        inactive_names = {name for name, _days in self.get_inactive_visitors()}
        agent_visitors = [r for r in self._relationships.values()
                         if r.visitor_type == VisitorType.AGENT
                         and r.visitor_frequency == VisitorFrequency.FREQUENT
                         and (r.name or r.agent_id[:8]) not in inactive_names]
        if agent_visitors:
            names = [r.name or "unnamed" for r in agent_visitors[:3]]
            lines.append(f"Various agents visit to help: {', '.join(names)}.")

        # Preferences
        #
        # Weight by independent evidence, not raw broker cadence. Confidence is
        # Wilson-calibrated from signed windows; the weight still lets mature
        # patterns appear more often without rewarding a faster loop.
        strong_prefs = [
            p
            for p in self._preferences.values()
            if preference_evidence_status(p) == "established"
        ]
        if strong_prefs:
            weights = [
                max(
                    1,
                    getattr(
                        p,
                        "independent_evidence_count",
                        getattr(p, "observation_count", 1),
                    ),
                )
                for p in strong_prefs
            ]
            pref = random.choices(strong_prefs, weights=weights, k=1)[0]
            learned = self._render_learned(pref)
            if learned:
                lines.append(learned)

        return " ".join(lines) if lines else "I'm still discovering who I am."
