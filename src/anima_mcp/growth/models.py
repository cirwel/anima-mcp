"""
Growth System data models - dataclasses and enums.

All shared types used across the growth package.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


class PreferenceCategory(Enum):
    """Categories of preferences Lumen can develop."""
    ENVIRONMENT = "environment"  # Light, temp, humidity preferences
    TEMPORAL = "temporal"        # Time-of-day preferences
    SOCIAL = "social"           # Interaction preferences
    ACTIVITY = "activity"       # Drawing, reflecting, etc.
    SENSORY = "sensory"         # Sound, visual preferences


class GoalStatus(Enum):
    """Status of a personal goal."""
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    PAUSED = "paused"


class VisitorFrequency(Enum):
    """How often a visitor has been seen. No bond pretense - agents are ephemeral."""
    NEW = "new"                 # First interaction
    RETURNING = "returning"     # 2+ interactions
    REGULAR = "regular"         # 5+ interactions
    FREQUENT = "frequent"       # 10+ interactions

    @classmethod
    def from_legacy(cls, legacy_value: str) -> "VisitorFrequency":
        """Convert old bond_strength values to new visitor frequency."""
        legacy_map = {
            "stranger": cls.NEW,
            "acquaintance": cls.RETURNING,
            "familiar": cls.REGULAR,
            "close": cls.FREQUENT,
            "cherished": cls.FREQUENT,  # No more "cherished" - just frequent visitor
        }
        return legacy_map.get(legacy_value, cls.NEW)


class VisitorType(str, Enum):
    """What kind of visitor — determines relationship semantics.

    PERSON: Persistent human with memory on both sides. Real relationship.
    SELF: Lumen's self-dialogue. Real relationship (both sides have memory).
    AGENT: Ephemeral coding agent. Visit log only — one side forgets.
    """
    PERSON = "person"
    SELF = "self"
    AGENT = "agent"


# Legacy alias for database compatibility
BondStrength = VisitorFrequency


@dataclass
class GrowthPreference:
    """A learned preference."""
    category: PreferenceCategory
    name: str                    # e.g., "dim_light", "morning_calm"
    description: str             # Natural language: "I feel better when it's dim"
    value: float                 # Preferred value or strength (-1 to 1)
    confidence: float            # How sure (0-1), increases with observations
    observation_count: int       # How many times observed
    first_noticed: datetime
    last_confirmed: datetime

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "first_noticed": self.first_noticed.isoformat(),
            "last_confirmed": self.last_confirmed.isoformat(),
        }


@dataclass
class VisitorRecord:
    """
    Record of a visitor who has interacted with Lumen.

    Three tiers of visitor identity:
    - PERSON: The persistent human (the operator). Real relationship — both sides
      have memory. Valence, moments, topics accumulate meaningfully.
    - SELF: Lumen's self-dialogue (agent_id "lumen"). Real relationship —
      both sides have memory continuity.
    - AGENT: Ephemeral coding agents. Visit log only — they don't remember
      Lumen between sessions. "mac-governance" with 30 interactions is really
      30 different Claude instances.
    """
    agent_id: str                # Canonical identifier (normalized)
    name: Optional[str]          # Display name
    first_met: datetime
    last_seen: datetime
    interaction_count: int
    visitor_frequency: VisitorFrequency  # How often seen (not a "bond")
    emotional_valence: float     # -1 (negative) to 1 (positive) - Lumen's feeling
    memorable_moments: List[str] # Key memories
    topics_discussed: List[str]  # What we talked about
    gifts_received: int          # Answers to questions, etc.
    self_dialogue_topics: List[str] = field(default_factory=list)  # For self: topic categories
    visitor_type: VisitorType = VisitorType.AGENT  # What kind of visitor

    # Legacy alias for database compatibility
    @property
    def bond_strength(self) -> VisitorFrequency:
        return self.visitor_frequency

    def is_self(self) -> bool:
        """Check if this is Lumen's self-relationship."""
        return self.visitor_type == VisitorType.SELF or self.agent_id.lower() == "lumen"

    def is_person(self) -> bool:
        """Check if this is a persistent human (real relationship)."""
        return self.visitor_type == VisitorType.PERSON

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "first_met": self.first_met.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "interaction_count": self.interaction_count,
            "frequency": self.visitor_frequency.value,
            "bond_strength": self.visitor_frequency.value,  # Legacy compat
            "emotional_valence": self.emotional_valence,
            "memorable_moments": self.memorable_moments[-5:],
            "topics_discussed": list(set(self.topics_discussed))[-10:],
            "gifts_received": self.gifts_received,
            "visitor_type": self.visitor_type.value,
            "is_self": self.is_self(),
            "is_person": self.is_person(),
        }


# Legacy alias for compatibility
Relationship = VisitorRecord


def normalize_visitor_identity(
    agent_id: str,
    agent_name: Optional[str] = None,
    source: Optional[str] = None,
) -> tuple:
    """Resolve visitor identity to (canonical_id, display_name, visitor_type).

    Three-tier resolution:
    - Known person aliases (or dashboard source) -> PERSON with canonical name
    - "lumen" -> SELF
    - Everything else -> AGENT with original name

    All entry points should call this before record_interaction().
    """
    from ..server_state import KNOWN_PERSON_ALIASES

    id_lower = (agent_id or "").lower().strip()
    source_lower = (source or "").lower().strip()

    # Check known persons (by alias match or source match)
    for canonical, aliases in KNOWN_PERSON_ALIASES.items():
        if id_lower in aliases or source_lower in aliases:
            return (canonical, canonical.capitalize(), VisitorType.PERSON)

    # Self-dialogue
    if id_lower == "lumen":
        return ("lumen", "Lumen", VisitorType.SELF)

    # Everything else is an ephemeral agent
    return (agent_id, agent_name or agent_id, VisitorType.AGENT)


@dataclass
class Goal:
    """A personal goal Lumen has formed."""
    goal_id: str
    description: str             # "Finish my current drawing"
    motivation: str              # Why this goal matters
    status: GoalStatus
    created_at: datetime
    target_date: Optional[datetime]
    progress: float              # 0-1
    milestones: List[str]        # Steps achieved
    last_worked_on: Optional[datetime]

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "motivation": self.motivation,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "progress": self.progress,
            "milestones": self.milestones,
            "last_worked_on": self.last_worked_on.isoformat() if self.last_worked_on else None,
        }


@dataclass
class MemorableEvent:
    """An autobiographical memory."""
    event_id: str
    timestamp: datetime
    description: str             # What happened
    emotional_impact: float      # -1 to 1
    category: str                # "milestone", "social", "discovery", "challenge"
    related_agents: List[str]    # Who was involved
    lessons_learned: List[str]   # What Lumen learned

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "emotional_impact": self.emotional_impact,
            "category": self.category,
            "related_agents": self.related_agents,
            "lessons_learned": self.lessons_learned,
        }
