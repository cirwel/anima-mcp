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

import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .self_reflection_analyzers import (  # noqa: F401
    InsightCategory,
    REFLECTION_EPSILON,
    REFLECTION_KIND_ANALYTIC,
    REFLECTION_KIND_METACOG,
    REFLECTION_WINDOW,
    ReflectionEpisode,
    SelfInsight,
    SelfReflectionAnalyzersMixin,
    StatePattern,
    VerificationResult,
    _ANTI_POLE_KEYWORDS,
    _DECREASE_MARKERS,
    _DIMENSION_KEYWORDS,
    _INCREASE_MARKERS,
    _NEGATIVE_MARKERS,
    _NEUTRAL_EFFECT_MARKERS,
    _POSITIVE_MARKERS,
    _SENSOR_KEYWORDS,
)


class SelfReflectionSystem(SelfReflectionAnalyzersMixin):
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
        # Every claim the last analyze_patterns() run measured, detected or
        # not: claim id -> (effect in the claimed direction, the analyzer's own
        # significance cut). generate_insights() judges stored pattern
        # insights against this, so a pattern that fades is contradicted
        # instead of silently skipped.
        self._pattern_measurements: Dict[str, Tuple[float, float]] = {}
        self._pattern_window = timedelta(hours=24)

        # Load existing insights from DB
        self._init_schema()
        self._load_reflection_state()
        self._load_insights()
        self._migrate_raw_light_insights()
        self._migrate_inverted_low_sensor_insights()
        self._migrate_recount_collapse()

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
                source_author TEXT NOT NULL DEFAULT '',
                last_judged TEXT
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
        if "last_judged" not in columns:
            conn.execute("ALTER TABLE insights ADD COLUMN last_judged TEXT")
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
                last_judged=(datetime.fromisoformat(row["last_judged"])
                             if "last_judged" in row.keys() and row["last_judged"]
                             else None),
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

    # Families whose counts were re-counted, not independent (2026-10-15):
    # pattern insights (hourly runs over a 24h window), reflect_* (every run
    # re-read the same 50 episodes) and drawing_* (a validation per run while
    # the preference merely stayed ready). pref_/belief_/trend_ mirror a
    # source whose own confidence can fall; qa_ rows are Q&A-derived.
    _RECOUNT_EXEMPT_PREFIXES = ("qa_", "pref_", "belief_", "trend_")

    # Weight the collapsed history keeps, in judgment units. A bounded
    # evidence floor, not a behavioral cut: it decides how many fresh
    # independent judgments it takes to move an old ratio.
    _RECOUNT_COLLAPSE_TOTAL = 4

    def _migrate_recount_collapse(self) -> None:
        """Rescale re-counted history to a small total, keeping its ratio.

        The overlapping counts cannot be de-duplicated after the fact, and
        left as they were (hundreds of validations) no future contradiction
        could move ``strength()``, which is a validation ratio. Each row's
        counts are rescaled to ``_RECOUNT_COLLAPSE_TOTAL`` with the ratio
        preserved (rounded), so strength and the newborn-protection
        comparison keep their direction: 959:0 -> 4:0 (unchanged strength),
        1:108 -> 0:4 (still refuted), 300:100 -> 3:1. Prior counts are kept
        in reflection_state as the audit.
        """
        sentinel = "migration_insight_recount_collapse_v1"
        if self._get_reflection_state(sentinel):
            return

        audit = {}
        conn = self._connect()
        for insight in self._insights.values():
            if insight.id.startswith(self._RECOUNT_EXEMPT_PREFIXES):
                continue
            v, c = insight.validation_count, insight.contradiction_count
            total = self._RECOUNT_COLLAPSE_TOTAL
            if v + c <= total:
                continue
            audit[insight.id] = {"validation_count": v, "contradiction_count": c}
            insight.validation_count = round(total * v / (v + c))
            insight.contradiction_count = total - insight.validation_count
            conn.execute(
                "UPDATE insights SET validation_count = ?, contradiction_count = ? "
                "WHERE id = ?",
                (insight.validation_count, insight.contradiction_count, insight.id),
            )

        conn.execute(
            """
            INSERT INTO reflection_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (sentinel, json.dumps({"reason": "overlapping_window_recounts",
                                   "rows": audit})),
        )
        conn.commit()
        if audit:
            print(f"[SelfReflection] Collapsed re-counted history on "
                  f"{len(audit)} insights", file=sys.stderr, flush=True)

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
             active, source_author, last_judged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
              source_author=excluded.source_author,
              last_judged=excluded.last_judged
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
            insight.last_judged.isoformat() if insight.last_judged else None,
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
        # Always judge: a run that detects nothing can still contradict
        # stored patterns it measured.
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
