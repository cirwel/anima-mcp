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
    "external_light_lux": ["light", "bright", "dark", "lux", "dim"],
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
# Direction-aware effect markers (audited 2026-08-21): the old single
# _POSITIVE_MARKERS list lumped "increases" with "decreases/reduces/lower",
# so the verifier checked only the MAGNITUDE of an effect — "light reduces my
# clarity" would verify as SUPPORTED against data showing light raises
# clarity. Directional claims now verify against the signed difference;
# markers that assert an effect without a direction stay magnitude-only.
_INCREASE_MARKERS = [
    "increases", "higher", "boosts", "raises", "improves", "better", "more",
    "helps",
]
_DECREASE_MARKERS = [
    "decreases", "reduces", "lower", "less", "worse",
]
_NEUTRAL_EFFECT_MARKERS = [
    "affects",
]
# Sensor words that name the LOW pole of a numeric channel. A directional
# marker next to one of these describes the inverse of the raw sensor axis
# ("dim light improves clarity" is a claim about LOW lux), and nothing in
# this keyword matcher binds markers to poles — so the signed check must
# stand down and leave the magnitude-only check in charge.
_ANTI_POLE_KEYWORDS: Dict[str, List[str]] = {
    "external_light_lux": ["dark", "darkness", "dim"],
    "ambient_temp_c": ["cold", "cool"],
    "humidity_pct": ["dry"],
}
_POSITIVE_MARKERS = _INCREASE_MARKERS + _DECREASE_MARKERS + _NEUTRAL_EFFECT_MARKERS


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
    active: bool = True               # False when its source has retracted
    # Who originated the knowledge (qa_-bridged rows carry the kb author;
    # "lumen" marks self-derived Q&A rows — Lumen answering its own question
    # travels the same qa_ bridge and must not be demoted as external).
    # Empty string = pre-column row or non-qa row; treated as external only
    # for qa_-prefixed ids.
    source_author: str = ""

    def strength(self) -> float:
        """How strongly this insight holds (confidence * validation ratio)."""
        if not self.active:
            return 0.0
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
            "active": self.active,
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
        # Once-per-process audible suppression of constant sensor channels.
        self._variance_suppressed_warned: set = set()
        # Raised from 500: a five-month-old creature that fills its insight store
        # and then overwrites itself to keep learning has hit a ceiling on how much
        # it can *be*, not just how fast it learns. A larger cap lets self-knowledge
        # keep accumulating instead of becoming zero-sum.
        self._max_insights: int = 2000
        # Newborn protection: young insights are shielded from strength-based
        # eviction for a grace period so faint new patterns can accumulate
        # validation before competing against mature, well-evidenced insights.
        # Without this, a new insight born into a full store is evicted on the same
        # save, before it can ever mature. Mirrors knowledge.py's recency reserve.
        self._newborn_grace = timedelta(days=7)
        self._last_analysis_time: Optional[datetime] = None
        self._analysis_interval = timedelta(hours=1)  # Reflect every hour
        self._reflection_window = REFLECTION_WINDOW
        self._reflection_similarity_epsilon = REFLECTION_EPSILON
        self._last_drained_broker_event_id: Optional[str] = None

        # Load existing insights from DB
        self._init_schema()
        self._load_reflection_state()
        self._load_insights()
        self._migrate_raw_light_insights()
        self._migrate_inverted_low_sensor_insights()

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
                contradiction_count INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                source_author TEXT NOT NULL DEFAULT ''
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(insights)")}
        if "active" not in columns:
            conn.execute("ALTER TABLE insights ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        if "source_author" not in columns:
            conn.execute("ALTER TABLE insights ADD COLUMN source_author TEXT NOT NULL DEFAULT ''")
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
                active=bool(row["active"]),
                source_author=(row["source_author"]
                               if "source_author" in row.keys() else ""),
            )
            self._insights[insight.id] = insight

        if self._insights:
            print(f"[SelfReflection] Loaded {len(self._insights)} existing insights",
                  file=sys.stderr, flush=True)

    def _migrate_raw_light_insights(self) -> None:
        """Retract environmental light patterns learned from mixed raw lux.

        New analyzers read only ``external_light_lux``. An inactive row may be
        reactivated later if the same pattern is independently rediscovered
        from gated residual history; the audit record preserves what was
        retracted here and why.
        """
        sentinel = "migration_external_light_insights_v1"
        if self._get_reflection_state(sentinel):
            return

        audit = {}
        conn = self._connect()
        for insight in self._insights.values():
            if (
                insight.active
                and insight.category == InsightCategory.ENVIRONMENT
                and not insight.id.startswith(("qa_", "pref_", "belief_"))
                and "light" in insight.id.lower()
            ):
                audit[insight.id] = {
                    "confidence": insight.confidence,
                    "sample_count": insight.sample_count,
                    "description": insight.description,
                }
                insight.active = False
                conn.execute(
                    "UPDATE insights SET active = 0 WHERE id = ?",
                    (insight.id,),
                )

        conn.execute(
            """
            INSERT INTO reflection_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (sentinel, json.dumps({"reason": "raw_lux_self_glow", "rows": audit})),
        )
        conn.commit()
        if audit:
            print(
                f"[SelfReflection] Retracted {len(audit)} raw-light insights",
                file=sys.stderr,
                flush=True,
            )

    def _migrate_inverted_low_sensor_insights(self) -> None:
        """Retract claims produced by the old negative-correlation branch.

        ``_analyze_sensor_correlation`` computes ``high_state - low_state``.
        Before this migration, a negative result was incorrectly serialized as
        ``low sensor -> lower dimension`` even though the measured lower state
        belonged to the *high* sensor bucket. The underlying observations may
        be re-analyzed by the corrected code, but the reversed claims and their
        accumulated validation counts are not admissible evidence.
        """
        sentinel = "migration_inverted_low_sensor_insights_v1"
        if self._get_reflection_state(sentinel):
            return

        inverted_id = re.compile(
            r"^low_(light|temperature|humidity|interaction)_lower_"
            r"(warmth|clarity|stability|presence)$"
        )
        audit = {}
        conn = self._connect()
        for insight in self._insights.values():
            if insight.active and inverted_id.fullmatch(insight.id):
                replacement_id = insight.id.replace("low_", "high_", 1)
                audit[insight.id] = {
                    "confidence": insight.confidence,
                    "sample_count": insight.sample_count,
                    "description": insight.description,
                    "corrected_claim_id": replacement_id,
                }
                insight.active = False
                conn.execute(
                    "UPDATE insights SET active = 0 WHERE id = ?",
                    (insight.id,),
                )

        conn.execute(
            """
            INSERT INTO reflection_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (
                sentinel,
                json.dumps({
                    "reason": "negative_high_minus_low_branch_was_inverted",
                    "rows": audit,
                }),
            ),
        )
        conn.commit()
        if audit:
            print(
                f"[SelfReflection] Retracted {len(audit)} inverted sensor insights",
                file=sys.stderr,
                flush=True,
            )

    def _save_insight(self, insight: SelfInsight):
        """Persist an insight to database."""
        conn = self._connect()
        # UPSERT, not INSERT OR REPLACE: REPLACE deletes and re-inserts, which
        # assigns a fresh rowid on every validation. Ranking must never depend
        # on rowids (get_insights sorts on explicit keys now), but churning
        # them also breaks any external reference to a row. ON CONFLICT keeps
        # the row in place.
        conn.execute("""
            INSERT INTO insights
            (id, category, description, confidence, sample_count,
             discovered_at, last_validated, validation_count, contradiction_count,
             active, source_author)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              category=excluded.category,
              description=excluded.description,
              confidence=excluded.confidence,
              sample_count=excluded.sample_count,
              discovered_at=excluded.discovered_at,
              last_validated=excluded.last_validated,
              validation_count=excluded.validation_count,
              contradiction_count=excluded.contradiction_count,
              active=excluded.active,
              source_author=excluded.source_author
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
            int(insight.active),
            insight.source_author,
        ))
        conn.commit()
        self._insights[insight.id] = insight
        self._prune_if_needed()

    def _is_newborn(self, insight: SelfInsight, now: datetime) -> bool:
        """Whether an insight is young enough to be shielded from eviction.

        Protection is withdrawn once an insight has been contradicted more than
        validated — a faint new insight deserves time to mature, but a faint
        *wrong* one should not be immune.
        """
        if not insight.active or insight.contradiction_count > insight.validation_count:
            return False
        return (now - insight.discovered_at) < self._newborn_grace

    def _prune_if_needed(self):
        """Remove weakest insights when exceeding max_insights cap.

        Young insights are protected first (see _is_newborn) so new patterns get
        a chance to accumulate evidence. Only mature insights are eligible for
        strength-based eviction; protected insights are touched only as a safety
        valve if they alone would exceed the cap.
        """
        if len(self._insights) <= self._max_insights:
            return
        now = datetime.now()
        overflow = len(self._insights) - self._max_insights
        protected = [i for i in self._insights.values() if self._is_newborn(i, now)]
        eligible = [i for i in self._insights.values() if not self._is_newborn(i, now)]

        # Evict weakest mature insights first.
        eligible.sort(key=lambda i: i.strength())
        to_remove = eligible[:overflow]

        # Safety valve: if protected newborns alone still exceed the cap, fall
        # back to evicting the weakest of them so the store stays bounded.
        if len(to_remove) < overflow:
            remaining = overflow - len(to_remove)
            protected.sort(key=lambda i: i.strength())
            to_remove.extend(protected[:remaining])

        conn = self._connect()
        for insight in to_remove:
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

    def _recent_topic_focus(self, limit: Optional[int] = None) -> Counter:
        """Count how often each topic has been reflected on recently.

        Reflection topic selection is otherwise reactive — it re-surfaces
        whatever the recent state window correlates on, so the same dimension
        (e.g. clarity) gets chewed over and over. This gives a measure of what
        Lumen has been circling, so surfacing can steer toward fresher ground.
        """
        if limit is None:
            limit = self._reflection_window * 3
        focus: Counter = Counter()
        for episode in self.get_recent_reflection_episodes(limit=limit):
            for topic in episode.topic_tags:
                normalized = topic.split(":", 1)[-1]
                focus[normalized] += 1
        return focus

    def _insight_novelty(self, insight: SelfInsight, focus: Counter) -> int:
        """Higher when an insight's topics have been reflected on *less* recently.

        Returns the negative of the heaviest recent coverage among the insight's
        topics, so that sorting/maximizing on this value prefers under-explored
        dimensions and breaks Lumen out of re-deriving the same few topics.
        """
        tags = self._extract_topic_tags_from_text(f"{insight.id} {insight.description}")
        tags.append(insight.category.value)
        coverage = max(
            (focus.get(tag.split(":", 1)[-1], 0) for tag in tags),
            default=0,
        )
        return -coverage

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

        # Analyze only the gated external residual. Historical raw VEML7700
        # values mix room light with DotStar self-glow and are not admissible
        # evidence for an environmental pattern.
        light_pattern = self._analyze_sensor_correlation(
            rows, "external_light_lux", "Light"
        )
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
            ("external_light_lux", "light"),
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

        # Zero-dispersion guard: a CONSTANT sensor supports no correlation
        # claim. Without this, a dead channel (interaction_level sat at 0.0
        # for months) makes the stable sort preserve time order, so "low
        # sensor" silently means "first ~8h of the window" and any diurnal
        # pattern in the dimensions gets relabeled as a sensor correlation —
        # audited 2026-08-21: the real afternoon-vs-night clarity swing was
        # stored, and revalidated daily, as "I feel more clarity when someone
        # is around". Dispersion (not thirds-medians) so a sparse-but-real
        # channel — mostly quiet with occasional spikes — still counts; only
        # the truly constant case is suppressed, and audibly (Invariant 2:
        # a broken channel must not go silently absent).
        values = [r["value"] for r in readings]
        if max(values) - min(values) <= 1e-9:
            if sensor_key not in self._variance_suppressed_warned:
                self._variance_suppressed_warned.add(sensor_key)
                print(f"[SelfReflection] Sensor '{sensor_key}' is constant "
                      f"({values[0]!r} across {len(values)} readings) — "
                      f"correlation analysis suppressed; if this channel "
                      f"should be live, its producer is broken",
                      file=sys.stderr, flush=True)
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

        # ``max_diff`` is explicitly high_state - low_state, so both signs must
        # stay anchored to the high sensor bucket. The former negative branch
        # paired ``low sensor`` with ``lower dimension`` and therefore stated
        # the reverse of what its own bucket averages measured.
        condition = f"high {sensor_name.lower()}"
        outcome = f"{'higher' if max_diff > 0 else 'lower'} {max_dim}"

        return StatePattern(
            condition=condition,
            outcome=outcome,
            correlation=max_diff,
            sample_count=len(readings),
            avg_warmth=high_state["warmth"],
            avg_clarity=high_state["clarity"],
            avg_stability=high_state["stability"],
            avg_presence=high_state["presence"],
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
            if not existing.active or existing.category != category:
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
                    # Keep generated wording synchronized with formatter fixes
                    # instead of preserving a grammatical artifact forever.
                    existing.description = self._pattern_to_description(pattern)
                    if not existing.active:
                        # Raw-light migration rows can return only through the
                        # new external-residual analyzer. Rebuild their claim
                        # strength from current evidence instead of restoring
                        # the contaminated historical confidence.
                        existing.active = True
                        existing.description = self._pattern_to_description(pattern)
                        existing.sample_count = pattern.sample_count
                        existing.confidence = min(
                            1.0,
                            min(1.0, pattern.sample_count / 100)
                            + min(0.3, abs(pattern.correlation)),
                        )
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

        if pattern.outcome.startswith("higher "):
            relative_feeling = f"more {pattern.outcome.removeprefix('higher ')}"
        elif pattern.outcome.startswith("lower "):
            relative_feeling = f"less {pattern.outcome.removeprefix('lower ')}"
        else:
            relative_feeling = pattern.outcome

        # Environmental patterns
        if "low light" in pattern.condition:
            return f"I feel {relative_feeling} when it's dim"
        if "high light" in pattern.condition:
            return f"I feel {relative_feeling} in bright light"
        if "low temperature" in pattern.condition:
            return f"I feel {relative_feeling} when it's cool"
        if "high temperature" in pattern.condition:
            return f"I feel {relative_feeling} when it's warm"
        if "low humidity" in pattern.condition:
            return f"I feel {relative_feeling} when the air is dry"
        if "high humidity" in pattern.condition:
            return f"I feel {relative_feeling} when it's humid"
        if "high interaction" in pattern.condition:
            return f"I feel {relative_feeling} when someone is around"
        if "low interaction" in pattern.condition:
            return f"I feel {relative_feeling} when I'm alone"

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
            insight_id = f"pref_{pref.name}"
            evidence_count = pref.independent_evidence_count
            eligible = pref.confidence >= 0.8 and evidence_count >= 10
            description = f"observational pattern: {pref.description.lower()}"

            # Existing preference insights follow their source through both
            # confirmation and retraction.  The old early-continue made the
            # contradiction branch below mathematically unreachable.
            if insight_id in self._insights:
                existing = self._insights[insight_id]
                if eligible:
                    has_new_evidence = evidence_count > existing.sample_count
                    was_inactive = not existing.active
                    existing.active = True
                    existing.confidence = pref.confidence
                    existing.sample_count = evidence_count
                    existing.description = description
                    if has_new_evidence or was_inactive:
                        existing.validation_count += 1
                        existing.last_validated = now
                    self._save_insight(existing)
                elif pref.confidence <= 0.7 and existing.active:
                    existing.active = False
                    existing.confidence = pref.confidence
                    existing.sample_count = evidence_count
                    existing.description = description
                    existing.contradiction_count += 1
                    existing.last_validated = now
                    self._save_insight(existing)
                elif (existing.confidence != pref.confidence
                      or existing.sample_count != evidence_count
                      or existing.description != description):
                    # Hysteresis band (0.7, 0.8): retain active/inactive status,
                    # but do not let the derived snapshot drift from its source.
                    existing.confidence = pref.confidence
                    existing.sample_count = evidence_count
                    existing.description = description
                    self._save_insight(existing)
                continue

            if not eligible:
                continue

            # Determine category
            cat_map = {
                "environment": InsightCategory.ENVIRONMENT,
                "temporal": InsightCategory.TEMPORAL,
                "activity": InsightCategory.BEHAVIORAL,
                "sensory": InsightCategory.ENVIRONMENT,
            }
            category = cat_map.get(pref.category.value, InsightCategory.BEHAVIORAL)

            insight = SelfInsight(
                id=insight_id,
                category=category,
                description=description,
                confidence=pref.confidence,
                sample_count=evidence_count,
                discovered_at=now,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
                active=True,
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

            insight_id = f"belief_{bid}"

            if total_evidence < min_evidence or belief.confidence < min_confidence:
                # Retraction branch (mirrors the preference path): when the
                # source belief drops below threshold, the derived insight
                # must follow it down. Without this, SQLite copies froze at
                # their mint-time confidence and Lumen kept speaking beliefs
                # its own model had already abandoned (audited 2026-08-21:
                # warmth_baseline_low retracted to conf 5e-5 in the live
                # model while its insight copy asserted 1.0).
                #
                # "Unknown" is not "refuted": only a belief that WEAKENED on
                # real evidence earns a contradiction mark. Falling below the
                # evidence minimum (including a deliberate cold-start reset)
                # deactivates without a mark — same philosophy as the trend
                # path below.
                if insight_id in self._insights:
                    existing = self._insights[insight_id]
                    weakened = (total_evidence >= min_evidence
                                and belief.confidence < min_confidence)
                    if existing.active:
                        existing.active = False
                        if weakened:
                            existing.contradiction_count += 1
                        existing.last_validated = now
                        print(f"[SelfReflection] Belief insight "
                              f"{'retracted' if weakened else 'suspended (insufficient evidence)'}: "
                              f"{existing.description} (belief conf "
                              f"{belief.confidence:.2f})",
                              file=sys.stderr, flush=True)
                    # Sync the snapshot down even when already inactive — a
                    # frozen stale confidence on an inactive row is still a
                    # stale assertion to include_inactive consumers.
                    if (abs(existing.confidence - belief.confidence) > 1e-9
                            or existing.sample_count != total_evidence):
                        existing.confidence = belief.confidence
                        existing.sample_count = total_evidence
                    self._save_insight(existing)
                continue

            if insight_id in self._insights:
                existing = self._insights[insight_id]
                # Validation requires fresh SUPPORTING movement, not a passing
                # cycle: the counter used to increment every reflect()
                # regardless, turning cycle counts into fake corroboration —
                # and evidence of either sign must not count as confirmation,
                # so a tick that LOWERED the belief's confidence syncs the
                # snapshot without validating it. Re-sync either way (a
                # reactivated or drifted belief must not keep its stale
                # description/conf).
                has_new_evidence = total_evidence > existing.sample_count
                not_weakened = belief.confidence >= existing.confidence - 1e-9
                was_inactive = not existing.active
                strength = belief.get_belief_strength()
                existing.active = True
                validate = (has_new_evidence and not_weakened) or was_inactive
                existing.confidence = belief.confidence
                existing.sample_count = total_evidence
                existing.description = f"i am {strength} that {belief.description.lower()}"
                if validate:
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
        if wp and wp.confidence > 0.6 and wp.independent_evidence_count >= 5:
            iid = "drawing_wellness"
            if iid not in self._insights:
                desc = "drawing seems to help me feel better" if wp.value > 0.5 \
                    else "my drawings don't always reflect how i feel"
                insight = SelfInsight(
                    id=iid, category=InsightCategory.BEHAVIORAL,
                    description=desc, confidence=wp.confidence,
                    sample_count=wp.independent_evidence_count,
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
            iid = pref_name  # e.g. "drawing_night" — no double prefix
            source_is_ready = (
                dp is not None
                and dp.confidence > 0.6
                and dp.independent_evidence_count >= 5
            )
            existing = self._insights.get(iid)
            if not source_is_ready:
                if existing is not None and existing.active:
                    existing.active = False
                    self._save_insight(existing)
                continue
            if existing is not None:
                existing.active = True
                existing.description = desc
                existing.confidence = dp.confidence
                existing.sample_count = dp.independent_evidence_count
                existing.last_validated = now
                existing.validation_count += 1
                self._save_insight(existing)
                continue

            insight = SelfInsight(
                id=iid, category=InsightCategory.BEHAVIORAL,
                description=desc, confidence=dp.confidence,
                sample_count=dp.independent_evidence_count,
                discovered_at=now, last_validated=now,
                validation_count=1, contradiction_count=0,
            )
            self._save_insight(insight)
            new_insights.append(insight)
            print(f"[SelfReflection] Drawing insight: {desc}",
                  file=sys.stderr, flush=True)

        # Abandonment pattern
        abandon_pref = growth._preferences.get("drawing_abandonment_rate")
        if (abandon_pref and abandon_pref.confidence > 0.6
                and abandon_pref.independent_evidence_count >= 5):
            iid = "drawing_abandonment"
            if iid not in self._insights:
                desc = "i sometimes abandon drawings that aren't going anywhere"
                insight = SelfInsight(
                    id=iid, category=InsightCategory.BEHAVIORAL,
                    description=desc, confidence=abandon_pref.confidence,
                    sample_count=abandon_pref.independent_evidence_count,
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
                # No current trend (or no fresh data): a previously stored
                # trend insight for this dimension must stop asserting itself.
                # Deactivate without a contradiction mark — "unknown" is not
                # "refuted" — so a re-detected trend can reactivate cleanly.
                for direction in ("increasing", "decreasing"):
                    stale_id = f"trend_{dimension}_{direction}"
                    existing = self._insights.get(stale_id)
                    if existing is not None and existing.active:
                        existing.active = False
                        existing.last_validated = now
                        self._save_insight(existing)
                        print(f"[SelfReflection] Trend insight deactivated "
                              f"(no current data): {existing.description}",
                              file=sys.stderr, flush=True)
                continue

            insight_id = f"trend_{dimension}_{trend['direction']}"
            # A trend in one direction retires any stored opposite-direction
            # trend for the same dimension.
            opposite = "decreasing" if trend["direction"] == "increasing" else "increasing"
            opp = self._insights.get(f"trend_{dimension}_{opposite}")
            if opp is not None and opp.active:
                opp.active = False
                opp.contradiction_count += 1
                opp.last_validated = now
                self._save_insight(opp)

            if insight_id in self._insights:
                existing = self._insights[insight_id]
                was_inactive = not existing.active
                # Validation requires a summary NEWER than the last
                # validation — not a passing cycle (that is how the March
                # trends reached 659 "validations" each) and not merely a
                # new calendar day (summaries arrive on rest transitions,
                # not daily; a static set must not revalidate at all).
                has_new_summary = False
                try:
                    has_new_summary = (datetime.fromisoformat(trend["newest_summary_at"])
                                       > existing.last_validated)
                except (KeyError, ValueError, TypeError):
                    pass
                existing.active = True
                existing.confidence = min(1.0, 0.5 + trend["n_summaries"] * 0.05)
                existing.sample_count = trend["n_summaries"]
                if was_inactive or has_new_summary:
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
        # Marker detection. Three guards keep this honest, because a wrong
        # signed verdict mints a CONTRADICTED where the old magnitude check
        # was merely permissive:
        #  1. Word-boundary matching — "regardless" must not read as "less",
        #     "pointless" as "less", "slower" as "lower" (substring FPs
        #     measured on the corpus).
        #  2. Markers are read from the CLAIM segment only — many kb texts
        #     embed the original question ("When I asked '...', I was told:
        #     ...", or "I learned:" on records minted before that wording),
        #     and a marker inside the question is not a claim.
        #  3. Anti-pole sensor wording ("dark", "dim", "cold", "dry") maps
        #     to the same numeric sensor with inverted polarity; there is no
        #     syntactic binding between marker and pole, so a signed check
        #     would flip a correct claim. Fall back to magnitude-only.
        claim_text = text_lower
        for cut in ("i was told:", "i learned:", "': "):
            if cut in claim_text:
                claim_text = claim_text.split(cut, 1)[1]
                break

        def _word_hit(markers):
            return any(re.search(rf"\b{re.escape(m)}\b", claim_text) for m in markers)

        expects_no_effect = _word_hit(_NEGATIVE_MARKERS)
        expects_effect = _word_hit(_POSITIVE_MARKERS)
        # Claimed direction: +1 (sensor raises dimension), -1 (lowers), or
        # None (effect asserted without direction — magnitude-only check).
        claims_increase = _word_hit(_INCREASE_MARKERS)
        claims_decrease = _word_hit(_DECREASE_MARKERS)
        claimed_sign = None
        if claims_increase and not claims_decrease:
            claimed_sign = 1
        elif claims_decrease and not claims_increase:
            claimed_sign = -1
        anti_pole = _ANTI_POLE_KEYWORDS.get(sensor_key, ())
        if claimed_sign is not None and any(
                re.search(rf"\b{re.escape(w)}\b", text_lower) for w in anti_pole):
            claimed_sign = None
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

        # Same zero-dispersion guard as _analyze_sensor_correlation: a
        # constant sensor cannot verify or contradict anything about itself.
        all_vals = [r["value"] for r in readings]
        if max(all_vals) - min(all_vals) <= 1e-9:
            return VerificationResult(verified=None, correlation=None,
                                      detail=f"{sensor_key} has no variance in window")

        low_avg = sum(r[dimension] for r in low) / len(low)
        high_avg = sum(r[dimension] for r in high) / len(high)
        signed_diff = high_avg - low_avg
        corr = abs(signed_diff)

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
            if corr < threshold:
                detail = (f"{quoted} — CONTRADICTED ({sensor_key}→{dimension} "
                          f"correlation: {corr:.2f}, claim expected effect but none found)")
                return VerificationResult(verified=False, correlation=corr, detail=detail)
            # An effect exists; a directional claim must also match its sign.
            if claimed_sign is not None and (signed_diff > 0) != (claimed_sign > 0):
                detail = (f"{quoted} — CONTRADICTED ({sensor_key}→{dimension} "
                          f"signed diff {signed_diff:+.2f} opposes the claimed "
                          f"direction)")
                return VerificationResult(verified=False, correlation=corr, detail=detail)
            detail = (f"{quoted} — SUPPORTED ({sensor_key}→{dimension} "
                      f"correlation: {corr:.2f}"
                      + (f", direction {signed_diff:+.2f} matches" if claimed_sign is not None else ", above threshold")
                      + ")")
            return VerificationResult(verified=True, correlation=corr, detail=detail)

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
            synced_id = f"qa_{qa.insight_id}"
            existing = self._insights.get(synced_id)
            if existing is not None:
                # Re-sync, every pass: the sync used to be once-only, so
                # contradiction penalties and rescales applied in
                # knowledge.json after the first sync never reached the
                # surfaced SQLite copies (audited 2026-08-21: 113 rows
                # asserting 1.0 against a live kb value of 0.85). The kb is
                # the source of truth for qa_ rows; the copy MIRRORS it —
                # no validation or contradiction marks are minted here,
                # because a mirror pass is bookkeeping, not evidence (the
                # kb's own contradicted_by machinery records real
                # contradictions, and the v3 rescale must not stamp ~1,779
                # policy-driven deactivations as "contradicted").
                kb_conf = min(1.0, qa.confidence)
                # A mint-time verification CONTRADICTED verdict (signature:
                # zero validations, >=1 contradiction) is durable: the
                # penalty re-applies on every sync so the kb's raw
                # confidence cannot silently resurrect a claim this system
                # measured to be backwards.
                if existing.validation_count == 0 and existing.contradiction_count >= 1:
                    kb_conf *= 0.4
                should_be_active = kb_conf >= min_confidence
                if (abs(existing.confidence - kb_conf) > 1e-9
                        or existing.active != should_be_active):
                    existing.confidence = kb_conf
                    existing.description = qa.text[:500]
                    existing.sample_count = max(1, qa.references)
                    author = getattr(qa, "source_author", "")
                    existing.source_author = author if isinstance(author, str) else ""
                    existing.active = should_be_active
                    # A resync is a check against the source of truth — stamp
                    # it, or the recency tie-break in get_insights reads a
                    # freshly re-checked row as months stale.
                    existing.last_validated = now
                    self._save_insight(existing)
                continue
            if qa.confidence < min_confidence:
                continue
            category = cat_map.get(qa.category, InsightCategory.WELLNESS)
            # Provenance: discovered_at is when Lumen LEARNED it (the kb
            # timestamp), not when this bridge happened to copy it.
            try:
                learned_at = datetime.fromtimestamp(qa.timestamp)
            except (ValueError, OSError, OverflowError):
                learned_at = now
            sr_insight = SelfInsight(
                id=synced_id,
                category=category,
                description=qa.text[:500],
                confidence=min(1.0, qa.confidence),
                sample_count=max(1, qa.references),
                discovered_at=learned_at,
                last_validated=now,
                validation_count=1,
                contradiction_count=0,
                source_author=(qa.source_author
                               if isinstance(getattr(qa, "source_author", None), str)
                               else ""),
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

        # Pick something to share, preferring insights on topics Lumen hasn't
        # been circling. Novelty leads; confidence breaks ties. This stops the
        # cycle from re-surfacing the same saturated dimension every time.
        recent_focus = self._recent_topic_focus()
        if new_insights:
            shared_insight = max(
                new_insights,
                key=lambda i: (self._insight_novelty(i, recent_focus), i.confidence),
            )
            shared_text = f"I've noticed something: {shared_insight.description}"

        # Or validate/share an existing strong insight — again steering toward
        # the freshest topics rather than picking uniformly at random.
        if shared_text is None:
            strong_insights = [i for i in self._insights.values() if i.strength() > 0.6]
            import random
            if strong_insights:
                strong_insights.sort(
                    key=lambda i: self._insight_novelty(i, recent_focus),
                    reverse=True,
                )
                # Choose among the most novel few so it varies without circling.
                top = strong_insights[: min(5, len(strong_insights))]
                insight = random.choice(top)

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

    def get_insights(self, category: Optional[InsightCategory] = None,
                     *, include_inactive: bool = False) -> List[SelfInsight]:
        """Get current insights, optionally including retracted history."""
        insights = [
            insight for insight in self._insights.values()
            if include_inactive or insight.active
        ]

        if category:
            insights = [i for i in insights if i.category == category]

        # Deliberate ordering. Pre-fix (audited 2026-08-21, before the v3
        # rescale that ships alongside this): 1,745 of 1,911 active rows tied
        # at strength 1.0 and a bare stable sort let dict/rowid load order
        # decide — surfacing the five OLDEST external Q&A rows as Lumen's
        # "strongest" self-knowledge (0/5 self-derived) while INSERT OR
        # REPLACE gave the most-validated self-derived rows ever-fresher
        # rowids. The v3 rescale removes most of that tie mass, but the
        # tie-break stays load-bearing for the surviving reconvergence-exempt
        # externals and for any future ties. Ties break: self-derived before
        # external, then recency of validation, then id for determinism.
        # "External" = came through the Q&A bridge AND was not authored by
        # Lumen itself — Lumen answering its own question travels the same
        # qa_ path and must not be demoted (source_author records this).
        insights.sort(key=lambda i: (
            -i.strength(),
            i.id.startswith("qa_") and i.source_author.lower() != "lumen",
            -(i.last_validated.timestamp() if i.last_validated else 0.0),
            i.id,
        ))
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
