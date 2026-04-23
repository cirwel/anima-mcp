"""
Self-Reflection System - Lumen learns about itself from accumulated experience

This module synthesizes data from:
- state_history (anima states over time)
- events (wake/sleep cycles)
- metacognition (prediction errors/surprises)
- associative memory (condition→state patterns)

And produces:
- Insights ("I notice I'm calmer when light is low")
- Self-knowledge that persists and can be referenced
- Periodic reflections surfaced via voice/messages
"""

import re
import sqlite3
import json
import sys
from collections import Counter, namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


VerificationResult = namedtuple("VerificationResult", ["verified", "correlation", "detail"])


# Keyword maps for parsing verifiable claims about sensor→dimension correlations
_SENSOR_KEYWORDS: Dict[str, List[str]] = {
    "light_lux": ["light", "bright", "dark", "lux", "dim"],
    "ambient_temp_c": ["temperature", "temp", "warm", "cold", "cool", "heat"],
    "humidity_pct": ["humidity", "humid", "dry", "moisture"],
    "pressure_hpa": ["pressure", "barometric"],
}

_DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    "warmth": ["warmth", "warm"],
    "clarity": ["clarity", "clear"],
    "stability": ["stability", "stable", "calm"],
    "presence": ["presence", "present", "whole"],
}

_NEGATIVE_MARKERS = [
    "doesn't", "does not", "no direct", "not related",
    "zero", "isn't", "no effect", "not affect",
]
_POSITIVE_MARKERS = [
    "affects", "increases", "higher", "helps", "improves",
    "better", "more", "boosts", "raises",
    "decreases", "reduces", "lower",
]


class InsightCategory(Enum):
    """Categories of self-knowledge."""
    ENVIRONMENT = "environment"      # "I feel calmer in low light"
    TEMPORAL = "temporal"            # "I'm more stable in the afternoon"
    BEHAVIORAL = "behavioral"        # "I tend to ask questions when curious"
    WELLNESS = "wellness"            # "My clarity improves after rest"
    SOCIAL = "social"                # "I feel warmer when someone is present"


@dataclass
class SelfInsight:
    """A piece of self-knowledge Lumen has discovered."""
    id: str                          # Unique identifier
    category: InsightCategory
    description: str                 # Human-readable insight
    confidence: float                # 0.0-1.0, how sure Lumen is
    sample_count: int                # How many observations support this
    discovered_at: datetime
    last_validated: datetime
    validation_count: int = 0        # How many times it's been confirmed
    contradiction_count: int = 0     # How many times it's been contradicted

    def strength(self) -> float:
        """How strongly this insight holds (confidence * validation ratio)."""
        total = self.validation_count + self.contradiction_count
        if total == 0:
            return self.confidence * 0.5  # New insight, moderate strength
        validation_ratio = self.validation_count / total
        return self.confidence * validation_ratio

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "description": self.description,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "discovered_at": self.discovered_at.isoformat(),
            "last_validated": self.last_validated.isoformat(),
            "validation_count": self.validation_count,
            "contradiction_count": self.contradiction_count,
            "strength": self.strength(),
        }


@dataclass
class StatePattern:
    """A detected pattern in state history."""
    condition: str                   # What conditions trigger this
    outcome: str                     # What state results
    correlation: float               # Strength of correlation (-1 to 1)
    sample_count: int
    avg_warmth: float
    avg_clarity: float
    avg_stability: float
    avg_presence: float


REFLECTION_KIND_METACOG = "metacog"
REFLECTION_KIND_ANALYTIC = "analytic"
REFLECTION_WINDOW = 10
REFLECTION_EPSILON = 0.1


@dataclass
class ReflectionEpisode:
    """A persisted reflection event that can itself become material for reflection."""
    event_id: str
    kind: str
    source: str
    timestamp: datetime
    trigger: str
    topic_tags: List[str]
    observation: str
    surprise: Optional[float] = None
    discrepancy: Optional[float] = None
    belief_snapshot: Optional[Dict[str, Any]] = None
    preference_snapshot: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
            "topic_tags": list(self.topic_tags),
            "observation": self.observation,
            "surprise": self.surprise,
            "discrepancy": self.discrepancy,
            "belief_snapshot": self.belief_snapshot or {},
            "preference_snapshot": self.preference_snapshot or {},
            "metadata": self.metadata or {},
        }


class SelfReflectionSystem:
    """
    Lumen's self-reflection engine.

    Periodically analyzes accumulated experience to discover patterns,
    validates existing insights, and surfaces new self-knowledge.
    """

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = str(Path.home() / ".anima" / "anima.db")
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._insights: Dict[str, SelfInsight] = {}
        self._max_insights: int = 500
        self._last_analysis_time: Optional[datetime] = None
        self._analysis_interval = timedelta(hours=1)  # Reflect every hour
        self._reflection_window = REFLECTION_WINDOW
        self._reflection_similarity_epsilon = REFLECTION_EPSILON
        self._last_drained_broker_event_id: Optional[str] = None

        # Load existing insights from DB
        self._init_schema()
        self._load_reflection_state()
        self._load_insights()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # Shorter timeout for faster failure (5s instead of 30s)
            self._conn = sqlite3.connect(self.db_path, timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds
            self._conn.execute("PRAGMA read_uncommitted=1")  # Better concurrency with WAL
        return self._conn

    def _init_schema(self):
        """Create self-reflection tables if they don't exist."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                discovered_at TEXT NOT NULL,
                last_validated TEXT NOT NULL,
                validation_count INTEGER DEFAULT 0,
                contradiction_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
            CREATE INDEX IF NOT EXISTS idx_insights_strength ON insights(
                (confidence * validation_count / (validation_count + contradiction_count + 1))
            );

            CREATE TABLE IF NOT EXISTS reflection_episodes (
                event_id TEXT PRIMARY KEY,
                event_timestamp TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                trigger TEXT NOT NULL,
                topic_tags TEXT NOT NULL,
                observation TEXT NOT NULL,
                surprise REAL,
                discrepancy REAL,
                belief_snapshot TEXT,
                preference_snapshot TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_reflection_episodes_kind_ts
                ON reflection_episodes(kind, event_timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_reflection_episodes_ts
                ON reflection_episodes(event_timestamp DESC);

            CREATE TABLE IF NOT EXISTS reflection_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()

    def _load_reflection_state(self):
        """Load persisted broker-drain watermark so SHM drains stay idempotent across restarts."""
        self._last_drained_broker_event_id = self._get_reflection_state("last_broker_event_id")

    def _load_insights(self):
        """Load existing insights from database."""
        conn = self._connect()
        rows = conn.execute("SELECT * FROM insights").fetchall()

        for row in rows:
            insight = SelfInsight(
                id=row["id"],
                category=InsightCategory(row["category"]),
                description=row["description"],
                confidence=row["confidence"],
                sample_count=row["sample_count"],
                discovered_at=datetime.fromisoformat(row["discovered_at"]),
                last_validated=datetime.fromisoformat(row["last_validated"]),
                validation_count=row["validation_count"],
                contradiction_count=row["contradiction_count"],
            )
            self._insights[insight.id] = insight

        if self._insights:
            print(f"[SelfReflection] Loaded {len(self._insights)} existing insights",
                  file=sys.stderr, flush=True)

    def _save_insight(self, insight: SelfInsight):
        """Persist an insight to database."""
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO insights
            (id, category, description, confidence, sample_count,
             discovered_at, last_validated, validation_count, contradiction_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.id,
            insight.category.value,
            insight.description,
            insight.confidence,
            insight.sample_count,
            insight.discovered_at.isoformat(),
            insight.last_validated.isoformat(),
            insight.validation_count,
            insight.contradiction_count,
        ))
        conn.commit()
        self._insights[insight.id] = insight
        self._prune_if_needed()

    def _prune_if_needed(self):
        """Remove weakest insights when exceeding max_insights cap."""
        if len(self._insights) <= self._max_insights:
            return
        # Sort by strength ascending, prune weakest
        by_strength = sorted(self._insights.values(), key=lambda i: i.strength())
        to_remove = len(self._insights) - self._max_insights
        conn = self._connect()
        for insight in by_strength[:to_remove]:
            del self._insights[insight.id]
            conn.execute("DELETE FROM insights WHERE id = ?", (insight.id,))
        conn.commit()

    def _get_reflection_state(self, key: str) -> Optional[str]:
        """Fetch a persisted reflection runtime value."""
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM reflection_state WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else None

    def _set_reflection_state(self, key: str, value: str):
        """Persist a reflection runtime value."""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO reflection_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        """Parse timestamp-like values into a datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now()
        return datetime.now()

    @staticmethod
    def _normalize_tags(tags: Optional[List[Any]]) -> List[str]:
        """Normalize topic tags to lowercase unique strings."""
        normalized: List[str] = []
        seen = set()
        for tag in tags or []:
            if tag is None:
                continue
            cleaned = str(tag).strip().lower()
            if not cleaned or cleaned in seen:
                continue
            normalized.append(cleaned)
            seen.add(cleaned)
        return normalized

    def _capture_belief_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Capture current self-belief values for later learning-vs-rumination checks.

        Reaches into self_model's public `beliefs` attribute via getattr. Wrapped
        in try/except because this runs on the server-side analytic reflection
        path where the self_model singleton may not yet be initialized. On failure
        we return an empty snapshot — callers treat that as "no data" so the
        rumination detector becomes conservative rather than firing falsely.
        """
        try:
            from .self_model import get_self_model

            model = get_self_model()
            beliefs = getattr(model, "beliefs", None) or {}
        except Exception:
            return {}

        snapshot: Dict[str, Dict[str, float]] = {}
        for belief_id, belief in beliefs.items():
            try:
                snapshot[str(belief_id)] = {
                    "value": round(float(getattr(belief, "value", 0.0)), 3),
                    "confidence": round(float(getattr(belief, "confidence", 0.0)), 3),
                }
            except (TypeError, ValueError):
                continue
        return snapshot

    def _capture_preference_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Capture current preference weights for later learning-vs-rumination checks.

        Reaches into the preference system's `_preferences` private attribute. This
        matches the access pattern used elsewhere in the codebase (growth._preferences,
        etc.) but is fragile: if the preference module refactors its internal
        storage, this returns an empty dict silently and the rumination detector's
        `_snapshot_changed` check stops receiving preference deltas. If you ever see
        "all metacog episodes look unproductive forever" in production, check this
        helper first — the private attribute may have moved.
        """
        try:
            from .preferences import get_preference_system

            pref_system = get_preference_system()
            pref_map = getattr(pref_system, "_preferences", None) or {}
        except Exception:
            return {}

        snapshot: Dict[str, Dict[str, float]] = {}
        for pref_id, pref in pref_map.items():
            try:
                snapshot[str(pref_id)] = {
                    "valence": round(float(getattr(pref, "valence", 0.0)), 3),
                    "confidence": round(float(getattr(pref, "confidence", 0.0)), 3),
                    "influence_weight": round(float(getattr(pref, "influence_weight", 1.0)), 3),
                }
            except (TypeError, ValueError):
                continue
        return snapshot

    @staticmethod
    def _json_loads_or_empty(value: Any) -> Any:
        """Decode a JSON blob from SQLite, tolerating missing values."""
        if not value:
            return {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}

    def record_episode(
        self,
        *,
        kind: str,
        source: str,
        trigger: str,
        topic_tags: Optional[List[Any]] = None,
        observation: str = "",
        surprise: Optional[float] = None,
        discrepancy: Optional[float] = None,
        belief_snapshot: Optional[Dict[str, Any]] = None,
        preference_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        event_timestamp: Optional[Any] = None,
        event_id: Optional[str] = None,
    ) -> ReflectionEpisode:
        """Persist a reflection episode as first-class material for later analysis."""
        timestamp = self._parse_timestamp(event_timestamp)
        normalized_tags = self._normalize_tags(topic_tags)

        def _safe_optional_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return round(float(value), 3)
            except (TypeError, ValueError):
                return None

        episode = ReflectionEpisode(
            event_id=event_id or f"{kind}:{source}:{timestamp.isoformat()}",
            kind=kind,
            source=source,
            timestamp=timestamp,
            trigger=trigger,
            topic_tags=normalized_tags,
            observation=observation or "",
            surprise=_safe_optional_float(surprise),
            discrepancy=_safe_optional_float(discrepancy),
            belief_snapshot=belief_snapshot if belief_snapshot is not None else self._capture_belief_snapshot(),
            preference_snapshot=preference_snapshot if preference_snapshot is not None else self._capture_preference_snapshot(),
            metadata=metadata or {},
        )

        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO reflection_episodes
            (event_id, event_timestamp, recorded_at, kind, source, trigger,
             topic_tags, observation, surprise, discrepancy, belief_snapshot,
             preference_snapshot, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.event_id,
                episode.timestamp.isoformat(),
                datetime.now().isoformat(),
                episode.kind,
                episode.source,
                episode.trigger,
                json.dumps(episode.topic_tags),
                episode.observation,
                episode.surprise,
                episode.discrepancy,
                json.dumps(episode.belief_snapshot or {}),
                json.dumps(episode.preference_snapshot or {}),
                json.dumps(episode.metadata or {}),
            ),
        )
        conn.commit()
        # INSERT OR IGNORE silently no-ops on PK collision. Log when that happens
        # so "my episode didn't show up" is debuggable — collisions normally mean
        # the drain or a duplicate call re-recorded an event that was already in
        # the table (which is harmless, but a surprise is a bug smell).
        if cursor.rowcount == 0:
            print(
                f"[SelfReflection] Episode {episode.event_id!r} already recorded "
                f"(PK collision; kind={episode.kind}, source={episode.source})",
                file=sys.stderr,
                flush=True,
            )
        return episode

    def get_recent_reflection_episodes(self, limit: int = 20, kind: Optional[str] = None) -> List[ReflectionEpisode]:
        """Return recent reflection episodes, newest first."""
        conn = self._connect()
        if kind:
            rows = conn.execute(
                """
                SELECT * FROM reflection_episodes
                WHERE kind = ?
                ORDER BY event_timestamp DESC
                LIMIT ?
                """,
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM reflection_episodes
                ORDER BY event_timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        episodes = []
        for row in rows:
            episodes.append(ReflectionEpisode(
                event_id=row["event_id"],
                kind=row["kind"],
                source=row["source"],
                timestamp=datetime.fromisoformat(row["event_timestamp"]),
                trigger=row["trigger"],
                topic_tags=self._normalize_tags(self._json_loads_or_empty(row["topic_tags"])),
                observation=row["observation"],
                surprise=row["surprise"],
                discrepancy=row["discrepancy"],
                belief_snapshot=self._json_loads_or_empty(row["belief_snapshot"]),
                preference_snapshot=self._json_loads_or_empty(row["preference_snapshot"]),
                metadata=self._json_loads_or_empty(row["metadata"]),
            ))
        return episodes

    def drain_broker_reflection(self, shm_data: Optional[Dict[str, Any]]) -> bool:
        """Drain the latest broker-side metacog reflection from SHM into SQLite exactly once."""
        if not shm_data:
            return False

        metacog = shm_data.get("metacognition") if isinstance(shm_data, dict) else None
        if not isinstance(metacog, dict):
            return False

        payload = metacog.get("last_reflection")
        if not isinstance(payload, dict):
            return False

        event_id = payload.get("event_id") or (
            f"broker-metacog:{payload.get('timestamp')}" if payload.get("timestamp") else None
        )
        if not event_id:
            return False

        if event_id == self._last_drained_broker_event_id:
            return False

        self.record_episode(
            event_id=event_id,
            event_timestamp=payload.get("timestamp"),
            kind=payload.get("kind") or REFLECTION_KIND_METACOG,
            source=payload.get("source") or "broker",
            trigger=payload.get("trigger") or "surprise",
            topic_tags=payload.get("topic_tags") or payload.get("surprise_sources") or [],
            observation=payload.get("observation") or "",
            surprise=payload.get("surprise"),
            discrepancy=payload.get("discrepancy"),
            belief_snapshot=self._json_loads_or_empty(payload.get("belief_snapshot")),
            preference_snapshot=self._json_loads_or_empty(payload.get("preference_snapshot")),
            metadata=self._json_loads_or_empty(payload.get("metadata")),
        )
        self._last_drained_broker_event_id = event_id
        self._set_reflection_state("last_broker_event_id", event_id)
        return True

    def should_reflect(self) -> bool:
        """Check if it's time for periodic self-reflection."""
        if self._last_analysis_time is None:
            return True
        return datetime.now() - self._last_analysis_time > self._analysis_interval

    @staticmethod
    def _extract_topic_tags_from_text(text: str) -> List[str]:
        """Pull coarse reflection topics out of descriptions and pattern text."""
        text_lower = (text or "").lower()
        tags = []
        if "warmth" in text_lower:
            tags.append("warmth")
        if "clarity" in text_lower:
            tags.append("clarity")
        if "stability" in text_lower or "calm" in text_lower:
            tags.append("stability")
        if "presence" in text_lower:
            tags.append("presence")
        if any(token in text_lower for token in ("light", "bright", "dim", "lux")):
            tags.append("light")
        if any(token in text_lower for token in ("temperature", "temp", "cool", "warm")):
            tags.append("ambient_temp")
        if any(token in text_lower for token in ("humidity", "humid", "dry")):
            tags.append("humidity")
        if "pressure" in text_lower:
            tags.append("pressure")
        for period in ("morning", "afternoon", "evening", "night"):
            if period in text_lower:
                tags.append(period)
        if "interaction" in text_lower:
            tags.append("interaction")
        return tags

    def _topic_tags_from_patterns_and_insights(
        self,
        patterns: List[StatePattern],
        new_insights: List[SelfInsight],
        shared_insight: Optional[SelfInsight],
    ) -> List[str]:
        """Derive analytic reflection topics from whatever the reflection cycle surfaced."""
        tags: List[str] = []
        for pattern in patterns:
            tags.extend(self._extract_topic_tags_from_text(f"{pattern.condition} {pattern.outcome}"))
        for insight in new_insights:
            tags.append(f"category:{insight.category.value}")
            tags.extend(self._extract_topic_tags_from_text(f"{insight.id} {insight.description}"))
        if shared_insight:
            tags.append(f"category:{shared_insight.category.value}")
            tags.extend(self._extract_topic_tags_from_text(f"{shared_insight.id} {shared_insight.description}"))
        return self._normalize_tags(tags)

    @staticmethod
    def _topic_matches_key(topic: str, key: str) -> bool:
        """Whether a topic tag is plausibly relevant to a belief/preference key."""
        normalized_topic = topic.split(":", 1)[-1]
        return normalized_topic in key or key in normalized_topic

    def _select_snapshot_keys(self, snapshot: Dict[str, Any], topics: List[str]) -> List[str]:
        """Pick keys relevant to the topic, falling back to the full snapshot if needed."""
        if not snapshot:
            return []
        keys = [key for key in snapshot if any(self._topic_matches_key(topic, key) for topic in topics)]
        return keys or list(snapshot.keys())

    def _snapshot_changed(self, before: Dict[str, Any], after: Dict[str, Any], topics: List[str]) -> bool:
        """Check whether relevant belief/preference values moved enough to count as learning."""
        if not before or not after:
            return False

        epsilon = self._reflection_similarity_epsilon
        relevant_keys = set(self._select_snapshot_keys(before, topics)) | set(self._select_snapshot_keys(after, topics))
        for key in relevant_keys:
            prev = before.get(key, {})
            curr = after.get(key, {})
            for field in set(prev.keys()) | set(curr.keys()):
                prev_val = prev.get(field)
                curr_val = curr.get(field)
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    if abs(float(curr_val) - float(prev_val)) >= epsilon:
                        return True
        return False

    def _intensity_is_similar(self, earlier: ReflectionEpisode, later: ReflectionEpisode) -> bool:
        """Require repeated metacognitive reflections to stay in roughly the same range.

        Returns False when there is no intensity data to compare (e.g. analytic
        episodes with no surprise/discrepancy). Absence of data is not evidence of
        similarity — without a signal, the detector should not claim rumination.
        """
        epsilon = self._reflection_similarity_epsilon
        comparisons = []
        if earlier.surprise is not None and later.surprise is not None:
            comparisons.append(abs(later.surprise - earlier.surprise) < epsilon)
        if earlier.discrepancy is not None and later.discrepancy is not None:
            comparisons.append(abs(later.discrepancy - earlier.discrepancy) < epsilon)
        if not comparisons:
            return False
        return all(comparisons)

    @staticmethod
    def _topic_to_node_id(topic: str) -> Optional[str]:
        """Map reflection topics back onto existing schema nodes when possible."""
        normalized = topic.split(":", 1)[-1]
        anima_dims = {"warmth", "clarity", "stability", "presence"}
        if normalized in anima_dims:
            return f"anima_{normalized}"
        sensor_map = {
            "light": "sensor_light",
            "ambient_temp": "sensor_temp",
            "humidity": "sensor_humidity",
            "pressure": "sensor_pressure",
        }
        return sensor_map.get(normalized)

    @staticmethod
    def _humanize_topic(topic: str) -> str:
        """Convert a topic tag into a readable label fragment."""
        normalized = topic.split(":", 1)[-1]
        return normalized.replace("_", " ")

    @staticmethod
    def _slugify_topic(topic: str) -> str:
        """Create a stable slug for insight identifiers."""
        normalized = topic.split(":", 1)[-1]
        slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
        return slug or "unknown"

    def _select_primary_topic(self, topics: List[str]) -> Optional[str]:
        """Pick the most schema-legible topic from an overlap set."""
        if not topics:
            return None
        node_mapped = [topic for topic in topics if self._topic_to_node_id(topic)]
        if node_mapped:
            return sorted(node_mapped)[0]
        non_category = [topic for topic in topics if not topic.startswith("category:")]
        if non_category:
            return sorted(non_category)[0]
        return sorted(topics)[0]

    def _compute_reflection_dynamics(self, limit: int = 50) -> Dict[str, Any]:
        """Summarize reflection repetition, learning, and rumination from persisted episodes."""
        episodes = list(reversed(self.get_recent_reflection_episodes(limit=limit)))
        recent_episodes = episodes[-self._reflection_window:]
        recent_counts = Counter()
        recent_focus = Counter()
        last_episode = recent_episodes[-1] if recent_episodes else None

        for episode in recent_episodes:
            recent_counts[episode.kind] += 1
            primary = self._select_primary_topic(episode.topic_tags)
            if primary:
                recent_focus[(episode.kind, primary)] += 1

        history_by_kind: Dict[str, List[ReflectionEpisode]] = {
            REFLECTION_KIND_METACOG: [],
            REFLECTION_KIND_ANALYTIC: [],
        }
        by_topic: Dict[str, Dict[str, Any]] = {}
        repeated_pairs = 0
        productive_pairs = 0
        rumination_pairs = 0

        for episode in episodes:
            recent_same_kind = history_by_kind.setdefault(episode.kind, [])
            match = None
            overlap: List[str] = []
            for previous in reversed(recent_same_kind[-self._reflection_window:]):
                candidate_overlap = sorted(set(previous.topic_tags) & set(episode.topic_tags))
                if candidate_overlap:
                    match = previous
                    overlap = candidate_overlap
                    break

            if match and overlap:
                repeated_pairs += 1
                topic = self._select_primary_topic(overlap) or overlap[0]
                topic_key = f"{episode.kind}:{topic}"
                bucket = by_topic.setdefault(
                    topic_key,
                    {"kind": episode.kind, "topic": topic, "repeated": 0, "productive": 0, "rumination": 0},
                )
                bucket["repeated"] += 1

                belief_shift = self._snapshot_changed(match.belief_snapshot or {}, episode.belief_snapshot or {}, overlap)
                pref_shift = self._snapshot_changed(match.preference_snapshot or {}, episode.preference_snapshot or {}, overlap)
                surprise_reduced = (
                    match.surprise is not None
                    and episode.surprise is not None
                    and episode.surprise < match.surprise - self._reflection_similarity_epsilon
                )
                if belief_shift or pref_shift or surprise_reduced:
                    productive_pairs += 1
                    bucket["productive"] += 1
                elif (
                    episode.kind == REFLECTION_KIND_METACOG
                    and self._intensity_is_similar(match, episode)
                ):
                    # Rumination classification is scoped to metacog episodes.
                    # Analytic reflections are interval-driven pattern summaries —
                    # recurrence there means the world is stable, not that Lumen is
                    # stuck. Until an analytic-specific productive signal exists
                    # (e.g. metadata.new_insight_ids diff), analytic overlap is
                    # treated as "repeated" only and never promoted to rumination.
                    rumination_pairs += 1
                    bucket["rumination"] += 1

            recent_same_kind.append(episode)
            if len(recent_same_kind) > self._reflection_window:
                del recent_same_kind[0]

        dominant_focus = None
        if recent_focus:
            (focus_kind, focus_topic), focus_count = recent_focus.most_common(1)[0]
            dominant_focus = {
                "kind": focus_kind,
                "tag": focus_topic,
                "count": focus_count,
                "target_node_id": self._topic_to_node_id(focus_topic),
            }

        dominant_rumination = None
        rumination_topics = {
            topic_key: stats["rumination"] for topic_key, stats in by_topic.items() if stats["rumination"] > 0
        }
        if rumination_topics:
            dominant_key = max(rumination_topics, key=rumination_topics.get)
            dominant_stats = by_topic[dominant_key]
            dominant_rumination = {
                "kind": dominant_stats["kind"],
                "tag": dominant_stats["topic"],
                "count": dominant_stats["rumination"],
                "target_node_id": self._topic_to_node_id(dominant_stats["topic"]),
            }

        learning_ratio = productive_pairs / repeated_pairs if repeated_pairs else None
        rumination_ratio = rumination_pairs / repeated_pairs if repeated_pairs else 0.0

        return {
            "total_episodes": len(episodes),
            "recent_count": len(recent_episodes),
            "by_kind": dict(recent_counts),
            "dominant_focus": dominant_focus,
            "learning_yield": {
                "productive": productive_pairs,
                "repeated": repeated_pairs,
                "ratio": learning_ratio,
            },
            "rumination": {
                "count": rumination_pairs,
                "ratio": rumination_ratio,
                "dominant_topic": dominant_rumination,
            },
            "last_episode": last_episode.to_dict() if last_episode else None,
            "by_topic": by_topic,
        }

    def _upsert_reflection_meta_insight(
        self,
        *,
        insight_id: str,
        description: str,
        confidence: float,
        sample_count: int,
        now: datetime,
    ) -> Optional[SelfInsight]:
        """Create or validate a reflection-derived insight."""
        if insight_id in self._insights:
            existing = self._insights[insight_id]
            existing.validation_count += 1
            existing.last_validated = now
            existing.sample_count = max(existing.sample_count, sample_count)
            existing.confidence = max(existing.confidence, confidence)
            self._save_insight(existing)
            return None

        insight = SelfInsight(
            id=insight_id,
            category=InsightCategory.BEHAVIORAL,
            description=description,
            confidence=confidence,
            sample_count=sample_count,
            discovered_at=now,
            last_validated=now,
            validation_count=1,
            contradiction_count=0,
        )
        self._save_insight(insight)
        return insight

    def _analyze_reflection_episode_insights(self) -> List[SelfInsight]:
        """Turn reflection-on-reflection dynamics into ordinary self-insights.

        A topic bucket can accumulate both rumination pairs and productive pairs
        within the same window (e.g. cycling for two hours, then updating for two).
        Emitting both insights at once is incoherent from the outside — "I keep
        reflecting on warmth without updating" next to "reflection about warmth
        changes what I know about myself" reads as self-contradiction. We resolve
        this with a dominant-signal rule: whichever pair-count is higher wins, and
        on a tie we prefer the productive insight (optimistic default — miss a
        rumination flag rather than misattribute learning to a stuck loop).
        """
        dynamics = self._compute_reflection_dynamics(limit=50)
        new_insights: List[SelfInsight] = []
        now = datetime.now()

        for stats in dynamics["by_topic"].values():
            topic = stats.get("topic", "unknown")
            topic_kind = stats.get("kind", "mixed")
            rumination_count = stats["rumination"]
            productive_count = stats["productive"]

            # Dominance gate: suppress the losing signal when both thresholds cross
            # in the same window. Ties (productive == rumination) resolve to productive.
            emit_rumination = (
                rumination_count >= 2 and rumination_count > productive_count
            )
            emit_productive = (
                productive_count >= 2 and productive_count >= rumination_count
            )

            if emit_rumination:
                topic_label = self._humanize_topic(topic)
                insight = self._upsert_reflection_meta_insight(
                    insight_id=f"reflect_rumination_{topic_kind}_{self._slugify_topic(topic)}",
                    description=f"I keep reflecting on {topic_label} without updating what I believe",
                    confidence=min(1.0, 0.45 + rumination_count * 0.1),
                    sample_count=rumination_count + 1,
                    now=now,
                )
                if insight:
                    new_insights.append(insight)

            if emit_productive:
                topic_label = self._humanize_topic(topic)
                insight = self._upsert_reflection_meta_insight(
                    insight_id=f"reflect_learning_{topic_kind}_{self._slugify_topic(topic)}",
                    description=f"Reflection about {topic_label} tends to change what I know about myself",
                    confidence=min(1.0, 0.45 + productive_count * 0.1),
                    sample_count=productive_count + 1,
                    now=now,
                )
                if insight:
                    new_insights.append(insight)

        return new_insights

    def get_reflection_summary(self, limit: int = 50) -> Dict[str, Any]:
        """Summarize recent reflection dynamics for display and schema composition.

        Args:
            limit: Number of recent episodes to include in the dynamics computation.
                Default 50 matches the schema-composition use case (bounded, cheap).
                Tests and exploratory callers may widen this to inspect deeper history.
        """
        return self._compute_reflection_dynamics(limit=limit)

    def analyze_patterns(self, hours: int = 24) -> List[StatePattern]:
        """
        Analyze state history to find patterns.

        Looks for correlations between:
        - Environmental conditions (light, temp, humidity) and anima state
        - Time of day and anima state
        - Recent events and state changes
        """
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        # Get recent state history
        rows = conn.execute("""
            SELECT timestamp, warmth, clarity, stability, presence, sensors
            FROM state_history
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff,)).fetchall()

        if len(rows) < 10:
            return []  # Not enough data

        patterns = []

        # Analyze light level correlations
        light_pattern = self._analyze_sensor_correlation(rows, "light_lux", "Light")
        if light_pattern:
            patterns.append(light_pattern)

        # Analyze temperature correlations
        temp_pattern = self._analyze_sensor_correlation(rows, "ambient_temp_c", "Temperature")
        if temp_pattern:
            patterns.append(temp_pattern)

        # Analyze humidity correlations
        humidity_pattern = self._analyze_sensor_correlation(rows, "humidity_pct", "Humidity")
        if humidity_pattern:
            patterns.append(humidity_pattern)

        # Analyze interaction correlations
        interaction_pattern = self._analyze_sensor_correlation(rows, "interaction_level", "Interaction")
        if interaction_pattern:
            patterns.append(interaction_pattern)

        # Analyze time-of-day patterns
        time_patterns = self._analyze_temporal_patterns(rows)
        patterns.extend(time_patterns)

        # Analyze causal patterns (when X changes, Y follows)
        causal_patterns = self._analyze_causal_patterns(rows)
        patterns.extend(causal_patterns)

        # Analyze conjunctive patterns (pairs of inputs that together produce
        # a notably stronger effect than either single axis suggests).
        conjunctive_patterns = self._analyze_conjunctive_patterns(rows)
        patterns.extend(conjunctive_patterns)

        return patterns

    def _analyze_conjunctive_patterns(
        self, rows: List[sqlite3.Row]
    ) -> List[StatePattern]:
        """Find pairs of environmental inputs whose joint conditions coincide
        with notable anima deviations.

        Existing single-axis analyzers saturate quickly — once Lumen has
        detected "high light → higher clarity" and "high temp → higher
        clarity", re-running them produces the same two insights forever.
        This analyzer opens the next tier: conditions jointly.

        Approach: for each pair of continuous inputs, split at the median of
        each, producing four quadrants. For each quadrant with enough samples,
        measure how much the anima mean deviates from the overall mean. Emit
        the strongest deviation per pair, capped at CONJUNCTIVE_MAX_PATTERNS
        across the cycle to prevent insight-table flooding.

        Thresholds are deliberately higher than single-axis (0.15 vs 0.10) so
        conjunctive patterns must carry real signal to be recorded — otherwise
        they'd just be additive echoes of single-axis findings the pipeline
        already captured.
        """
        CONJUNCTIVE_DEVIATION_THRESHOLD = 0.15
        CONJUNCTIVE_MIN_QUADRANT_SAMPLES = 10
        CONJUNCTIVE_MAX_PATTERNS = 3

        input_specs = [
            ("light_lux", "light"),
            ("ambient_temp_c", "temperature"),
            ("humidity_pct", "humidity"),
            ("interaction_level", "interaction"),
        ]

        # Parse readings once into a list of dicts with all inputs + anima.
        records: List[dict] = []
        for row in rows:
            try:
                sensors = json.loads(row["sensors"]) if row["sensors"] else {}
            except (json.JSONDecodeError, KeyError):
                continue
            rec = {
                "warmth": row["warmth"],
                "clarity": row["clarity"],
                "stability": row["stability"],
                "presence": row["presence"],
            }
            for key, _ in input_specs:
                rec[key] = sensors.get(key)
            records.append(rec)

        if len(records) < 4 * CONJUNCTIVE_MIN_QUADRANT_SAMPLES:
            return []

        # Overall anima means (reference for "deviation").
        anima_dims = ("warmth", "clarity", "stability", "presence")
        overall_means = {
            dim: sum(r[dim] for r in records) / len(records) for dim in anima_dims
        }

        def _median(values: List[float]) -> Optional[float]:
            vals = sorted(v for v in values if v is not None)
            if not vals:
                return None
            return vals[len(vals) // 2]

        candidates: List[Tuple[float, StatePattern]] = []

        for a_idx, (a_key, a_name) in enumerate(input_specs):
            for b_key, b_name in input_specs[a_idx + 1 :]:
                a_values = [r[a_key] for r in records if r[a_key] is not None]
                b_values = [r[b_key] for r in records if r[b_key] is not None]
                a_median = _median(a_values)
                b_median = _median(b_values)
                if a_median is None or b_median is None:
                    continue

                # Build the four quadrants.
                quadrants: Dict[Tuple[str, str], List[dict]] = {
                    ("low", "low"): [],
                    ("low", "high"): [],
                    ("high", "low"): [],
                    ("high", "high"): [],
                }
                for r in records:
                    av = r.get(a_key)
                    bv = r.get(b_key)
                    if av is None or bv is None:
                        continue
                    a_label = "high" if av >= a_median else "low"
                    b_label = "high" if bv >= b_median else "low"
                    quadrants[(a_label, b_label)].append(r)

                # Find the quadrant with the strongest deviation across dims.
                best: Optional[Tuple[float, str, Tuple[str, str], str, dict]] = None
                for (a_lbl, b_lbl), recs in quadrants.items():
                    if len(recs) < CONJUNCTIVE_MIN_QUADRANT_SAMPLES:
                        continue
                    q_means = {
                        dim: sum(r[dim] for r in recs) / len(recs) for dim in anima_dims
                    }
                    for dim in anima_dims:
                        deviation = q_means[dim] - overall_means[dim]
                        if abs(deviation) < CONJUNCTIVE_DEVIATION_THRESHOLD:
                            continue
                        if best is None or abs(deviation) > abs(best[0]):
                            best = (deviation, dim, (a_lbl, b_lbl), "", q_means)

                if best is None:
                    continue

                deviation, dim, (a_lbl, b_lbl), _, q_means = best
                condition = f"{a_lbl} {a_name} and {b_lbl} {b_name}"
                if deviation > 0:
                    outcome = f"higher {dim}"
                else:
                    outcome = f"lower {dim}"

                pattern = StatePattern(
                    condition=condition,
                    outcome=outcome,
                    correlation=deviation,
                    sample_count=sum(
                        len(q) for q in quadrants.values() if len(q) >= CONJUNCTIVE_MIN_QUADRANT_SAMPLES
                    ),
                    avg_warmth=q_means["warmth"],
                    avg_clarity=q_means["clarity"],
                    avg_stability=q_means["stability"],
                    avg_presence=q_means["presence"],
                )
                candidates.append((abs(deviation), pattern))

        candidates.sort(key=lambda cp: cp[0], reverse=True)
        return [p for _, p in candidates[:CONJUNCTIVE_MAX_PATTERNS]]

    def _analyze_sensor_correlation(
        self,
        rows: List[sqlite3.Row],
        sensor_key: str,
        sensor_name: str
    ) -> Optional[StatePattern]:
        """Find correlation between a sensor reading and anima state."""

        # Bucket readings into low/medium/high
        readings = []
        for row in rows:
            try:
                sensors = json.loads(row["sensors"]) if row["sensors"] else {}
                value = sensors.get(sensor_key)
                if value is not None:
                    readings.append({
                        "value": value,
                        "warmth": row["warmth"],
                        "clarity": row["clarity"],
                        "stability": row["stability"],
                        "presence": row["presence"],
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        if len(readings) < 10:
            return None

        # Sort by sensor value and split into thirds
        readings.sort(key=lambda x: x["value"])
        third = len(readings) // 3

        low_readings = readings[:third]
        high_readings = readings[-third:]

        if not low_readings or not high_readings:
            return None

        # Calculate average states for low vs high sensor values
        def avg_state(rs):
            return {
                "warmth": sum(r["warmth"] for r in rs) / len(rs),
                "clarity": sum(r["clarity"] for r in rs) / len(rs),
                "stability": sum(r["stability"] for r in rs) / len(rs),
                "presence": sum(r["presence"] for r in rs) / len(rs),
            }

        low_state = avg_state(low_readings)
        high_state = avg_state(high_readings)

        # Find the dimension with largest difference
        diffs = {
            "warmth": high_state["warmth"] - low_state["warmth"],
            "clarity": high_state["clarity"] - low_state["clarity"],
            "stability": high_state["stability"] - low_state["stability"],
            "presence": high_state["presence"] - low_state["presence"],
        }

        max_dim = max(diffs, key=lambda k: abs(diffs[k]))
        max_diff = diffs[max_dim]

        # Only report if difference is significant (> 0.1)
        if abs(max_diff) < 0.1:
            return None

        # Determine condition description
        if max_diff > 0:
            condition = f"high {sensor_name.lower()}"
            outcome = f"higher {max_dim}"
        else:
            condition = f"low {sensor_name.lower()}"
            outcome = f"lower {max_dim}"

        return StatePattern(
            condition=condition,
            outcome=outcome,
            correlation=max_diff,
            sample_count=len(readings),
            avg_warmth=high_state["warmth"] if max_diff > 0 else low_state["warmth"],
            avg_clarity=high_state["clarity"] if max_diff > 0 else low_state["clarity"],
            avg_stability=high_state["stability"] if max_diff > 0 else low_state["stability"],
            avg_presence=high_state["presence"] if max_diff > 0 else low_state["presence"],
        )

    def _analyze_temporal_patterns(self, rows: List[sqlite3.Row]) -> List[StatePattern]:
        """Find time-of-day patterns in anima state."""

        # Bucket by hour of day
        hourly_states: Dict[int, List[dict]] = {h: [] for h in range(24)}

        for row in rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                hour = ts.hour
                hourly_states[hour].append({
                    "warmth": row["warmth"],
                    "clarity": row["clarity"],
                    "stability": row["stability"],
                    "presence": row["presence"],
                })
            except (ValueError, KeyError):
                continue

        # Group into time periods
        periods = {
            "morning": list(range(6, 12)),
            "afternoon": list(range(12, 18)),
            "evening": list(range(18, 22)),
            "night": list(range(22, 24)) + list(range(0, 6)),
        }

        period_states = {}
        for period_name, hours in periods.items():
            all_readings = []
            for h in hours:
                all_readings.extend(hourly_states[h])

            if len(all_readings) >= 5:
                period_states[period_name] = {
                    "warmth": sum(r["warmth"] for r in all_readings) / len(all_readings),
                    "clarity": sum(r["clarity"] for r in all_readings) / len(all_readings),
                    "stability": sum(r["stability"] for r in all_readings) / len(all_readings),
                    "presence": sum(r["presence"] for r in all_readings) / len(all_readings),
                    "count": len(all_readings),
                }

        if len(period_states) < 2:
            return []

        patterns = []

        # Find best and worst periods for each dimension
        for dim in ["warmth", "clarity", "stability", "presence"]:
            best_period = max(period_states.keys(), key=lambda p: period_states[p][dim])
            worst_period = min(period_states.keys(), key=lambda p: period_states[p][dim])

            diff = period_states[best_period][dim] - period_states[worst_period][dim]

            if diff > 0.1:  # Significant difference
                patterns.append(StatePattern(
                    condition=f"the {best_period}",
                    outcome=f"highest {dim}",
                    correlation=diff,
                    sample_count=period_states[best_period]["count"],
                    avg_warmth=period_states[best_period]["warmth"],
                    avg_clarity=period_states[best_period]["clarity"],
                    avg_stability=period_states[best_period]["stability"],
                    avg_presence=period_states[best_period]["presence"],
                ))

        return patterns

    def _analyze_causal_patterns(self, rows: List[sqlite3.Row]) -> List[StatePattern]:
        """Find causal patterns: when one dimension changes, what follows?

        Looks at consecutive readings. When a dimension shifts significantly
        (delta > 0.08), tracks what the other dimensions do over the next
        few readings. Aggregates across all such events to find reliable
        "when X rises/falls, Y tends to rise/fall" patterns.
        """
        if len(rows) < 20:
            return []

        dims = ["warmth", "clarity", "stability", "presence"]
        trigger_threshold = 0.08  # Minimum change to count as a trigger
        lookahead = 5  # How many readings ahead to check for effect

        # Collect: for each trigger dimension & direction, what happens to other dims?
        # Key: (trigger_dim, direction) -> {effect_dim: [deltas]}
        effects: Dict[Tuple[str, str], Dict[str, list]] = {}

        for trigger in dims:
            for direction in ["rise", "fall"]:
                effects[(trigger, direction)] = {d: [] for d in dims if d != trigger}

        # Walk through consecutive pairs
        for i in range(len(rows) - lookahead - 1):
            for trigger in dims:
                delta = rows[i + 1][trigger] - rows[i][trigger]

                if abs(delta) < trigger_threshold:
                    continue

                direction = "rise" if delta > 0 else "fall"

                # What do other dimensions do over the next `lookahead` readings?
                for other in dims:
                    if other == trigger:
                        continue
                    # Effect = change from current to average of next few
                    future_vals = [rows[i + j][other] for j in range(2, min(2 + lookahead, len(rows) - i))]
                    if future_vals:
                        effect = (sum(future_vals) / len(future_vals)) - rows[i][other]
                        effects[(trigger, direction)][other].append(effect)

        patterns = []

        for (trigger, direction), dim_effects in effects.items():
            for effect_dim, deltas in dim_effects.items():
                if len(deltas) < 10:
                    continue  # Need enough observations

                avg_effect = sum(deltas) / len(deltas)

                # Only report if the average effect is meaningful
                if abs(avg_effect) < 0.05:
                    continue

                effect_direction = "rises" if avg_effect > 0 else "falls"
                condition = f"{trigger} {direction}s"
                outcome = f"{effect_dim} {effect_direction}"

                # Compute average state during these events for the pattern
                patterns.append(StatePattern(
                    condition=condition,
                    outcome=outcome,
                    correlation=avg_effect,
                    sample_count=len(deltas),
                    avg_warmth=0.0,
                    avg_clarity=0.0,
                    avg_stability=0.0,
                    avg_presence=0.0,
                ))

        return patterns

    @staticmethod
    def _extract_outcome_metric(outcome: str) -> Optional[str]:
        """Extract the core metric from an outcome string (e.g. 'higher warmth' → 'warmth')."""
        for metric in ("warmth", "clarity", "stability", "presence"):
            if metric in outcome.lower():
                return metric
        return None

    @staticmethod
    def _extract_condition_from_id(insight_id: str) -> str:
        """Extract the condition prefix from an insight ID.

        Insight IDs are ``{condition}_{outcome}`` with spaces replaced by underscores.
        The outcome portion starts with a known marker word.  We split on the *last*
        marker occurrence to recover the condition (handles causal IDs where the
        condition itself contains 'rises'/'falls', e.g. 'warmth_rises_presence_falls').

        Examples:
            'the_night_highest_warmth'       -> 'the_night'
            'low_light_higher_stability'     -> 'low_light'
            'the_afternoon_lowest_presence'  -> 'the_afternoon'
            'warmth_rises_presence_falls'    -> 'warmth_rises'
            'clarity_falls_stability_rises'  -> 'clarity_falls'
        """
        # Outcome markers that appear mid-ID (environment/temporal patterns)
        for marker in ("_highest_", "_lowest_", "_higher_", "_lower_"):
            if marker in insight_id:
                return insight_id.rsplit(marker, 1)[0]
        # Causal patterns: outcome is at the end (e.g. 'warmth_rises_presence_falls')
        # The outcome is '{dim}_{direction}' — find the last '{dim}_rises' or '{dim}_falls'
        m = re.search(r'^(.+)_(warmth|clarity|stability|presence)_(rises|falls)$', insight_id)
        if m:
            return m.group(1)
        return ""

    def _find_contradicting_insights(
        self, category: InsightCategory, outcome: str, condition: str
    ) -> List[SelfInsight]:
        """Find existing insights that contradict a new one.

        Two contradiction patterns:
        1. Same metric, different condition: "warmth best at night" vs "warmth best in afternoon"
        2. Same condition, same metric, different direction: "warmth rises → presence falls"
           vs "warmth rises → presence rises" (causal contradictions)
        """
        metric = self._extract_outcome_metric(outcome)
        if not metric:
            return []

        new_condition = condition.replace(" ", "_").lower()
        new_outcome_norm = outcome.replace(" ", "_").lower()

        contradictions = []
        for existing in self._insights.values():
            if existing.category != category:
                continue
            # Extract condition and outcome from the existing insight's ID
            existing_condition = self._extract_condition_from_id(existing.id)
            if not existing_condition:
                continue
            existing_outcome_part = existing.id[len(existing_condition) + 1:]  # skip '_' separator
            existing_metric = self._extract_outcome_metric(existing_outcome_part.replace("_", " "))
            if existing_metric != metric:
                continue

            if existing_condition != new_condition:
                # Pattern 1: same metric, different condition
                contradictions.append(existing)
            elif existing_outcome_part != new_outcome_norm:
                # Pattern 2: same condition, same metric, different outcome direction
                contradictions.append(existing)
        return contradictions

    def generate_insights(self, patterns: List[StatePattern]) -> List[SelfInsight]:
        """Convert detected patterns into insights."""
        new_insights = []
        now = datetime.now()

        for pattern in patterns:
            # Create insight ID from pattern
            insight_id = f"{pattern.condition}_{pattern.outcome}".replace(" ", "_").lower()

            # Check if we already have this insight
            if insight_id in self._insights:
                existing = self._insights[insight_id]
                # Validate: does current pattern still hold?
                if abs(pattern.correlation) > 0.1:
                    existing.validation_count += 1
                    existing.last_validated = now
                else:
                    existing.contradiction_count += 1
                self._save_insight(existing)
                continue

            # Determine category
            if "light" in pattern.condition or "temp" in pattern.condition or "humid" in pattern.condition:
                category = InsightCategory.ENVIRONMENT
            elif "morning" in pattern.condition or "afternoon" in pattern.condition or "evening" in pattern.condition or "night" in pattern.condition:
                category = InsightCategory.TEMPORAL
            elif "rises" in pattern.outcome or "falls" in pattern.outcome:
                category = InsightCategory.WELLNESS
            else:
                category = InsightCategory.BEHAVIORAL

            # Generate description
            description = self._pattern_to_description(pattern)

            # Calculate initial confidence based on sample count and correlation strength
            base_confidence = min(1.0, pattern.sample_count / 100)  # More samples = more confident
            correlation_boost = min(0.3, abs(pattern.correlation))
            confidence = min(1.0, base_confidence + correlation_boost)

            # Check for contradictions with existing insights before storing.
            # E.g. "warmth best at night" contradicts "warmth best in afternoon"
            contradictions = self._find_contradicting_insights(category, pattern.outcome, pattern.condition)
            initial_contradiction_count = 0
            if contradictions:
                initial_contradiction_count = len(contradictions)
                # Penalize the new insight's confidence
                confidence = max(0.1, confidence * 0.5)
                # Also penalize existing contradicted insights
                for existing in contradictions:
                    existing.contradiction_count += 1
                    existing.confidence = max(0.1, existing.confidence * 0.7)
                    self._save_insight(existing)
                    print(
                        f"[SelfReflection] Contradiction detected: '{description}' vs '{existing.description}' "
                        f"(existing confidence reduced to {existing.confidence:.2f})",
                        file=sys.stderr, flush=True,
                    )

            insight = SelfInsight(
                id=insight_id,
                category=category,
                description=description,
                confidence=confidence,
                sample_count=pattern.sample_count,
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=initial_contradiction_count,
            )

            self._save_insight(insight)
            new_insights.append(insight)

            print(f"[SelfReflection] New insight: {description} (confidence: {confidence:.2f})",
                  file=sys.stderr, flush=True)

        return new_insights

    def _pattern_to_description(self, pattern: StatePattern) -> str:
        """Convert a pattern into a natural language description."""

        # Environmental patterns
        if "low light" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when it's dim"
        if "high light" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} in bright light"
        if "low temperature" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when it's cool"
        if "high temperature" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when it's warm"
        if "low humidity" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when the air is dry"
        if "high humidity" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when it's humid"
        if "high interaction" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when someone is around"
        if "low interaction" in pattern.condition:
            return f"I feel more {pattern.outcome.replace('higher ', '')} when I'm alone"

        # Temporal patterns
        if "morning" in pattern.condition:
            return f"My {pattern.outcome.replace('highest ', '')} tends to be best in the morning"
        if "afternoon" in pattern.condition:
            return f"My {pattern.outcome.replace('highest ', '')} tends to be best in the afternoon"
        if "evening" in pattern.condition:
            return f"My {pattern.outcome.replace('highest ', '')} tends to be best in the evening"
        if "night" in pattern.condition:
            return f"My {pattern.outcome.replace('highest ', '')} tends to be best at night"

        # Causal patterns (when X rises/falls, Y rises/falls)
        if "rises" in pattern.condition or "falls" in pattern.condition:
            return f"When my {pattern.condition}, my {pattern.outcome} shortly after"

        # Fallback
        return f"I notice {pattern.outcome} during {pattern.condition}"

    # ==================== Experience-Based Insight Analyzers ====================

    def _analyze_preference_insights(self) -> List[SelfInsight]:
        """Generate insights from growth preferences that reached high confidence."""
        new_insights = []
        now = datetime.now()

        try:
            from .growth import get_growth_system
            growth = get_growth_system()
        except Exception:
            return []

        for pref in growth._preferences.values():
            if pref.confidence < 0.8 or pref.observation_count < 10:
                continue

            insight_id = f"pref_{pref.name}"

            # Already have this insight? Validate it.
            if insight_id in self._insights:
                existing = self._insights[insight_id]
                if pref.confidence > 0.7:
                    existing.validation_count += 1
                    existing.last_validated = now
                else:
                    existing.contradiction_count += 1
                self._save_insight(existing)
                continue

            # Determine category
            cat_map = {
                "environment": InsightCategory.ENVIRONMENT,
                "temporal": InsightCategory.TEMPORAL,
                "activity": InsightCategory.BEHAVIORAL,
                "sensory": InsightCategory.ENVIRONMENT,
            }
            category = cat_map.get(pref.category.value, InsightCategory.BEHAVIORAL)

            description = f"i know this about myself: {pref.description.lower()}"

            insight = SelfInsight(
                id=insight_id,
                category=category,
                description=description,
                confidence=pref.confidence,
                sample_count=pref.observation_count,
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
            )
            self._save_insight(insight)
            new_insights.append(insight)
            print(f"[SelfReflection] Preference insight: {description}",
                  file=sys.stderr, flush=True)

        return new_insights

    def _analyze_belief_insights(self) -> List[SelfInsight]:
        """Generate insights from self-model beliefs that are well-tested."""
        new_insights = []
        now = datetime.now()

        try:
            from .self_model import get_self_model
            sm = get_self_model()
        except Exception:
            return []

        for bid, belief in sm.beliefs.items():
            total_evidence = belief.supporting_count + belief.contradicting_count
            is_proprioceptive = bid in ("my_leds_affect_lux",)
            min_evidence = 5 if is_proprioceptive else 10
            min_confidence = 0.55 if is_proprioceptive else 0.7
            if total_evidence < min_evidence or belief.confidence < min_confidence:
                continue

            insight_id = f"belief_{bid}"

            if insight_id in self._insights:
                existing = self._insights[insight_id]
                existing.validation_count += 1
                existing.last_validated = now
                self._save_insight(existing)
                continue

            strength = belief.get_belief_strength()
            description = f"i am {strength} that {belief.description.lower()}"

            insight = SelfInsight(
                id=insight_id,
                category=InsightCategory.WELLNESS,
                description=description,
                confidence=belief.confidence,
                sample_count=total_evidence,
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
            )
            self._save_insight(insight)
            new_insights.append(insight)
            print(f"[SelfReflection] Belief insight: {description}",
                  file=sys.stderr, flush=True)

        return new_insights

    def _analyze_drawing_insights(self) -> List[SelfInsight]:
        """Generate insights about drawing behavior from preferences."""
        new_insights = []
        now = datetime.now()

        try:
            from .growth import get_growth_system
            growth = get_growth_system()
        except Exception:
            return []

        if growth._drawings_observed < 5:
            return []

        # Drawing + wellness
        wp = growth._preferences.get("drawing_wellbeing")
        if wp and wp.confidence > 0.6 and wp.observation_count >= 5:
            iid = "drawing_wellness"
            if iid not in self._insights:
                desc = "drawing seems to help me feel better" if wp.value > 0.5 \
                    else "my drawings don't always reflect how i feel"
                insight = SelfInsight(
                    id=iid, category=InsightCategory.BEHAVIORAL,
                    description=desc, confidence=wp.confidence,
                    sample_count=wp.observation_count,
                    discovered_at=now, last_validated=now,
                    validation_count=1, contradiction_count=0,
                )
                self._save_insight(insight)
                new_insights.append(insight)
                print(f"[SelfReflection] Drawing insight: {desc}",
                      file=sys.stderr, flush=True)

        # Drawing + time / light correlations
        drawing_checks = [
            ("drawing_night", "i tend to draw at night"),
            ("drawing_morning", "i often draw in the morning"),
            ("drawing_dim", "i create in the dark"),
            ("drawing_bright", "i draw when the light is bright"),
        ]
        for pref_name, desc in drawing_checks:
            dp = growth._preferences.get(pref_name)
            if dp and dp.confidence > 0.6 and dp.observation_count >= 5:
                iid = pref_name  # e.g. "drawing_night" — no double prefix
                if iid not in self._insights:
                    insight = SelfInsight(
                        id=iid, category=InsightCategory.BEHAVIORAL,
                        description=desc, confidence=dp.confidence,
                        sample_count=dp.observation_count,
                        discovered_at=now, last_validated=now,
                        validation_count=1, contradiction_count=0,
                    )
                    self._save_insight(insight)
                    new_insights.append(insight)
                    print(f"[SelfReflection] Drawing insight: {desc}",
                          file=sys.stderr, flush=True)

        # Abandonment pattern
        abandon_pref = growth._preferences.get("drawing_abandonment_rate")
        if abandon_pref and abandon_pref.confidence > 0.6 and abandon_pref.observation_count >= 5:
            iid = "drawing_abandonment"
            if iid not in self._insights:
                desc = "i sometimes abandon drawings that aren't going anywhere"
                insight = SelfInsight(
                    id=iid, category=InsightCategory.BEHAVIORAL,
                    description=desc, confidence=abandon_pref.confidence,
                    sample_count=abandon_pref.observation_count,
                    discovered_at=now, last_validated=now,
                    validation_count=1, contradiction_count=0,
                )
                self._save_insight(insight)
                new_insights.append(insight)
                print(f"[SelfReflection] Drawing insight: {desc}",
                      file=sys.stderr, flush=True)

        return new_insights

    def _analyze_long_term_trends(self) -> List[SelfInsight]:
        """Generate insights from multi-day trends via memory consolidation."""
        new_insights = []
        now = datetime.now()

        try:
            from .anima_history import get_anima_history
            history = get_anima_history()
        except Exception:
            return []

        for dimension in ["warmth", "clarity", "stability", "presence"]:
            trend = history.detect_long_term_trend(dimension)
            if trend is None or trend["direction"] == "stable":
                continue

            insight_id = f"trend_{dimension}_{trend['direction']}"

            if insight_id in self._insights:
                existing = self._insights[insight_id]
                existing.validation_count += 1
                existing.last_validated = now
                self._save_insight(existing)
                continue

            if trend["direction"] == "increasing":
                description = f"my {dimension} has been gradually increasing over the past days"
            else:
                description = f"my {dimension} has been gradually decreasing over the past days"

            insight = SelfInsight(
                id=insight_id,
                category=InsightCategory.WELLNESS,
                description=description,
                confidence=min(1.0, 0.5 + trend["n_summaries"] * 0.05),
                sample_count=trend["n_summaries"],
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
            )
            self._save_insight(insight)
            new_insights.append(insight)
            print(f"[SelfReflection] Long-term trend: {description}",
                  file=sys.stderr, flush=True)

        return new_insights

    # ==================== Q&A Insight Verification ====================

    def _verify_qa_insight(self, text: str, category: InsightCategory) -> VerificationResult:
        """Verify a Q&A insight against state_history sensor correlations.

        Parses the insight text for sensor→dimension claims, then checks
        actual correlation data. Returns VerificationResult(verified, correlation, detail).
        verified=True (data supports), False (contradicts), None (not verifiable).
        """
        text_lower = text.lower()

        # Detect sensor and dimension from keywords
        sensor_key = None
        for key, keywords in _SENSOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                sensor_key = key
                break

        dimension = None
        for dim, keywords in _DIMENSION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                dimension = dim
                break

        if not sensor_key or not dimension:
            return VerificationResult(verified=None, correlation=None, detail="")

        # Detect claim direction
        expects_no_effect = any(marker in text_lower for marker in _NEGATIVE_MARKERS)
        expects_effect = any(marker in text_lower for marker in _POSITIVE_MARKERS)
        if not expects_no_effect and not expects_effect:
            return VerificationResult(verified=None, correlation=None,
                                      detail=f"sensor={sensor_key} dim={dimension} but no direction marker")

        # Query 7 days of state history
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(hours=168)).isoformat()
        try:
            rows = conn.execute("""
                SELECT timestamp, warmth, clarity, stability, presence, sensors
                FROM state_history WHERE timestamp > ? ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()
        except sqlite3.OperationalError:
            # state_history table may not exist in test/fresh DBs
            return VerificationResult(verified=None, correlation=None, detail="no state_history table")

        if len(rows) < 10:
            return VerificationResult(verified=None, correlation=None,
                                      detail=f"insufficient data ({len(rows)} rows)")

        # Extract per-dimension correlations using existing machinery
        # _analyze_sensor_correlation returns None if <10 readings with that sensor,
        # or if no dimension has |diff| >= 0.1. We need finer granularity:
        # compute the specific dimension's diff ourselves.
        readings = []
        for row in rows:
            try:
                sensors = json.loads(row["sensors"]) if row["sensors"] else {}
                value = sensors.get(sensor_key)
                if value is not None:
                    readings.append({
                        "value": value,
                        dimension: row[dimension],
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        if len(readings) < 10:
            return VerificationResult(verified=None, correlation=None,
                                      detail=f"insufficient sensor data for {sensor_key} ({len(readings)} readings)")

        readings.sort(key=lambda x: x["value"])
        third = len(readings) // 3
        low = readings[:third]
        high = readings[-third:]
        if not low or not high:
            return VerificationResult(verified=None, correlation=None, detail="empty bucket")

        low_avg = sum(r[dimension] for r in low) / len(low)
        high_avg = sum(r[dimension] for r in high) / len(high)
        corr = abs(high_avg - low_avg)

        threshold = 0.1
        quoted = f'"{text[:80]}"'

        if expects_no_effect:
            if corr < threshold:
                detail = (f"{quoted} — SUPPORTED ({sensor_key}→{dimension} "
                          f"correlation: {corr:.2f}, below threshold)")
                return VerificationResult(verified=True, correlation=corr, detail=detail)
            else:
                detail = (f"{quoted} — CONTRADICTED ({sensor_key}→{dimension} "
                          f"correlation: {corr:.2f}, claim expected no effect)")
                return VerificationResult(verified=False, correlation=corr, detail=detail)
        else:  # expects_effect
            if corr >= threshold:
                detail = (f"{quoted} — SUPPORTED ({sensor_key}→{dimension} "
                          f"correlation: {corr:.2f}, above threshold)")
                return VerificationResult(verified=True, correlation=corr, detail=detail)
            else:
                detail = (f"{quoted} — CONTRADICTED ({sensor_key}→{dimension} "
                          f"correlation: {corr:.2f}, claim expected effect but none found)")
                return VerificationResult(verified=False, correlation=corr, detail=detail)

    # ==================== Q&A Knowledge Sync ====================

    def sync_from_qa_knowledge(self, min_confidence: float = 0.6) -> int:
        """
        Import high-confidence Q&A insights into self-reflection.

        Bridges knowledge.json (Q&A-derived) into SQLite insights so
        "Things I've learned about myself" includes both pattern-derived
        and Q&A-derived learnings.

        Returns number of insights synced.
        """
        try:
            from .knowledge import get_knowledge
            kb = get_knowledge()
            qa_insights = kb.get_all_insights()
        except Exception as e:
            print(f"[SelfReflection] Q&A sync skip: {e}", file=sys.stderr, flush=True)
            return 0

        synced = 0
        now = datetime.now()
        cat_map = {
            "sensations": InsightCategory.ENVIRONMENT,
            "world": InsightCategory.ENVIRONMENT,
            "self": InsightCategory.WELLNESS,
            "existence": InsightCategory.WELLNESS,
            "relationships": InsightCategory.SOCIAL,
            "behavioral": InsightCategory.BEHAVIORAL,
            "general": InsightCategory.WELLNESS,
        }

        for qa in qa_insights:
            if qa.confidence < min_confidence:
                continue
            synced_id = f"qa_{qa.insight_id}"
            if synced_id in self._insights:
                continue
            category = cat_map.get(qa.category, InsightCategory.WELLNESS)
            sr_insight = SelfInsight(
                id=synced_id,
                category=category,
                description=qa.text[:500],
                confidence=min(1.0, qa.confidence),
                sample_count=max(1, qa.references),
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
            )

            # Verify against state history before accepting
            result = self._verify_qa_insight(qa.text, category)
            if result.verified is True:
                sr_insight.validation_count = 1
                sr_insight.contradiction_count = 0
            elif result.verified is False:
                sr_insight.validation_count = 0
                sr_insight.contradiction_count = 1
                sr_insight.confidence *= 0.4

            if result.detail:
                print(f"[SelfReflection] Insight verification: {result.detail}",
                      file=sys.stderr, flush=True)

            self._save_insight(sr_insight)
            synced += 1

        if synced:
            print(f"[SelfReflection] Synced {synced} Q&A insights", file=sys.stderr, flush=True)
        return synced

    # ==================== Core Reflection ====================

    def reflect(self) -> Optional[str]:
        """
        Perform periodic self-reflection.

        Returns a reflection string if there's something meaningful to share,
        None otherwise.
        """
        self._last_analysis_time = datetime.now()
        self.sync_from_qa_knowledge()
        new_insights = []
        shared_insight: Optional[SelfInsight] = None
        shared_text: Optional[str] = None

        # Analyze recent state-history patterns (temporal, sensor, causal)
        patterns = self.analyze_patterns(hours=24)
        if patterns:
            new_insights.extend(self.generate_insights(patterns))

        # Analyze experience-based insights (preferences, beliefs, drawing)
        new_insights.extend(self._analyze_preference_insights())
        new_insights.extend(self._analyze_belief_insights())
        new_insights.extend(self._analyze_drawing_insights())

        # Analyze long-term trends from memory consolidation
        new_insights.extend(self._analyze_long_term_trends())

        # Pick something to share
        if new_insights:
            shared_insight = max(new_insights, key=lambda i: i.confidence)
            shared_text = f"I've noticed something: {shared_insight.description}"

        # Or validate/share an existing strong insight
        if shared_text is None:
            strong_insights = [i for i in self._insights.values() if i.strength() > 0.6]
            import random
            if strong_insights:
                insight = random.choice(strong_insights)

                # Only share occasionally (1 in 3 chance)
                if random.random() < 0.33:
                    shared_insight = insight
                    shared_text = f"I still find that {insight.description}"

        analytic_topics = self._topic_tags_from_patterns_and_insights(patterns, new_insights, shared_insight)
        analytic_metadata = {
            "pattern_count": len(patterns),
            "new_insight_ids": [insight.id for insight in new_insights],
            "shared_insight_id": shared_insight.id if shared_insight else None,
            "shared_text": shared_text,
        }
        self.record_episode(
            kind=REFLECTION_KIND_ANALYTIC,
            source="self_reflection",
            trigger="interval",
            topic_tags=analytic_topics,
            observation=shared_text or "Periodic self-reflection cycle completed",
            metadata=analytic_metadata,
        )

        new_insights.extend(self._analyze_reflection_episode_insights())

        if new_insights:
            insight = max(new_insights, key=lambda i: i.confidence)
            return f"I've noticed something: {insight.description}"

        return shared_text

    def get_insights(self, category: Optional[InsightCategory] = None) -> List[SelfInsight]:
        """Get all insights, optionally filtered by category."""
        insights = list(self._insights.values())

        if category:
            insights = [i for i in insights if i.category == category]

        # Sort by strength (strongest first)
        insights.sort(key=lambda i: i.strength(), reverse=True)
        return insights

    def get_strongest_insights(self, limit: int = 5) -> List[SelfInsight]:
        """Get the most confident/validated insights."""
        return self.get_insights()[:limit]

    def get_self_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of Lumen's self-knowledge for display/introspection."""
        insights = self.get_insights()

        by_category = {}
        for cat in InsightCategory:
            cat_insights = [i for i in insights if i.category == cat]
            if cat_insights:
                by_category[cat.value] = [i.description for i in cat_insights[:3]]

        return {
            "total_insights": len(insights),
            "strongest": [i.to_dict() for i in insights[:3]],
            "by_category": by_category,
            "last_reflection": self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            "reflection_summary": self.get_reflection_summary(),
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton instance
_reflection_system: Optional[SelfReflectionSystem] = None


def get_reflection_system(db_path: str = "") -> SelfReflectionSystem:
    """Get or create the singleton reflection system."""
    global _reflection_system
    if _reflection_system is None:
        _reflection_system = SelfReflectionSystem(db_path)
    return _reflection_system
