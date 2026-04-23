"""
Growth System - Core class and singleton.

Lumen's growth and development system. Manages preferences, relationships,
goals, and autobiographical memory. All growth data persists in SQLite
for continuity across sessions.
"""

import sys
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import (
    GrowthPreference, VisitorRecord, Goal, MemorableEvent,
    PreferenceCategory, GoalStatus, VisitorFrequency, VisitorType,
    Relationship,
)
from .migrations import run_identity_migration, migrate_raw_lux_preferences
from .preferences import PreferencesMixin
from .visitors import VisitorsMixin
from .goals import GoalsMixin
from .memories import MemoriesMixin
from .curiosity import CuriosityMixin


# Gallery directory can be overridden for tests
_GALLERY_DIR_OVERRIDE: Optional[Path] = None


def _gallery_dir() -> Path:
    """Return the gallery directory (test-overridable via set_gallery_dir)."""
    if _GALLERY_DIR_OVERRIDE is not None:
        return _GALLERY_DIR_OVERRIDE
    return Path.home() / ".anima" / "drawings"


def set_gallery_dir(path: Optional[Path]) -> None:
    """Test hook: point the reconciliation at a temp directory."""
    global _GALLERY_DIR_OVERRIDE
    _GALLERY_DIR_OVERRIDE = path


def _count_gallery_drawings() -> int:
    """Count PNG artifacts in the gallery directory.

    Returns 0 if the directory doesn't exist (fresh install) or on any I/O
    error — the counter will be preserved as-is, which is the safe default.
    """
    try:
        d = _gallery_dir()
        if not d.is_dir():
            return 0
        return sum(1 for _ in d.glob("lumen_drawing_*.png"))
    except OSError:
        return 0


class GrowthSystem(
    PreferencesMixin,
    VisitorsMixin,
    GoalsMixin,
    MemoriesMixin,
    CuriosityMixin,
):
    """
    Lumen's growth and development system.

    Manages preferences, relationships, goals, and autobiographical memory.
    """

    def __init__(self, db_path: str = "anima.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._preferences: Dict[str, GrowthPreference] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._goals: Dict[str, Goal] = {}
        self._memories: List[MemorableEvent] = []
        self._curiosities: List[str] = []  # Things Lumen wants to explore
        self.born_at: Optional[datetime] = None  # Set from identity after wake()
        self._drawings_observed: int = 0
        self._initialize_db()
        self._load_all()
        migrate_raw_lux_preferences(self._connect(), self._preferences)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: growth singleton created on main thread,
            # but canvas_save calls observe_drawing from display thread.
            # Safe because WAL mode + serialized access (no concurrent writes).
            self._conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds
            self._conn.execute("PRAGMA read_uncommitted=1")  # Better concurrency with WAL
        return self._conn

    def _initialize_db(self):
        """Create growth tables if they don't exist."""
        conn = self._connect()
        conn.executescript("""
            -- Preferences table
            CREATE TABLE IF NOT EXISTS preferences (
                name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT,
                value REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                observation_count INTEGER DEFAULT 0,
                first_noticed TEXT,
                last_confirmed TEXT
            );

            -- Relationships table
            CREATE TABLE IF NOT EXISTS relationships (
                agent_id TEXT PRIMARY KEY,
                name TEXT,
                first_met TEXT,
                last_seen TEXT,
                interaction_count INTEGER DEFAULT 0,
                bond_strength TEXT DEFAULT 'stranger',
                emotional_valence REAL DEFAULT 0.0,
                memorable_moments TEXT DEFAULT '[]',
                topics_discussed TEXT DEFAULT '[]',
                gifts_received INTEGER DEFAULT 0,
                self_dialogue_topics TEXT DEFAULT '[]'
            );

            -- Goals table
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                description TEXT,
                motivation TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT,
                target_date TEXT,
                progress REAL DEFAULT 0.0,
                milestones TEXT DEFAULT '[]',
                last_worked_on TEXT
            );

            -- Autobiographical memories table
            CREATE TABLE IF NOT EXISTS memories (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                description TEXT,
                emotional_impact REAL DEFAULT 0.0,
                category TEXT,
                related_agents TEXT DEFAULT '[]',
                lessons_learned TEXT DEFAULT '[]'
            );

            -- Curiosities table (things to explore)
            CREATE TABLE IF NOT EXISTS curiosities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                created_at TEXT,
                explored BOOLEAN DEFAULT 0,
                exploration_notes TEXT
            );

            -- Counters table (persistent scalar values across restarts)
            CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                value INTEGER DEFAULT 0
            );

            -- Per-drawing records for data-grounded self-answers
            CREATE TABLE IF NOT EXISTS drawing_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pixel_count INTEGER,
                phase TEXT,
                warmth REAL,
                clarity REAL,
                stability REAL,
                presence REAL,
                wellness REAL,
                light_lux REAL,
                ambient_temp_c REAL,
                humidity_pct REAL,
                hour INTEGER
            );
        """)
        conn.commit()

        # Migration: add columns that may be missing from older DBs
        # (CREATE TABLE IF NOT EXISTS won't add new columns to existing tables)
        migrations = [
            ("relationships", "self_dialogue_topics", "TEXT DEFAULT '[]'"),
            ("relationships", "visitor_type", "TEXT DEFAULT 'agent'"),
            ("drawing_records", "epoch", "INTEGER NOT NULL DEFAULT 1"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()

        # Identity migration: merge fragmented person records, set visitor_types
        run_identity_migration(conn)

    def _load_all(self):
        """Load all growth data from database."""
        conn = self._connect()

        # Load preferences
        for row in conn.execute("SELECT * FROM preferences"):
            try:
                cat = PreferenceCategory(row["category"])
            except ValueError:
                continue  # Skip system/sentinel rows with non-enum categories
            self._preferences[row["name"]] = GrowthPreference(
                category=cat,
                name=row["name"],
                description=row["description"] or "",
                value=row["value"],
                confidence=row["confidence"],
                observation_count=row["observation_count"],
                first_noticed=datetime.fromisoformat(row["first_noticed"]) if row["first_noticed"] else datetime.now(),
                last_confirmed=datetime.fromisoformat(row["last_confirmed"]) if row["last_confirmed"] else datetime.now(),
            )

        # Load visitor records (legacy: "relationships")
        # Limit to 500 most recent by last_seen to avoid unbounded growth (RESILIENCE #12)
        RELATIONSHIPS_LOAD_LIMIT = 500
        for row in conn.execute(
            "SELECT * FROM relationships ORDER BY last_seen DESC LIMIT ?",
            (RELATIONSHIPS_LOAD_LIMIT,),
        ):
            # Handle legacy bond_strength values
            legacy_bond = row["bond_strength"]
            try:
                freq = VisitorFrequency(legacy_bond)
            except ValueError:
                freq = VisitorFrequency.from_legacy(legacy_bond)

            # Handle self_dialogue_topics column (may not exist in old DBs)
            try:
                self_topics = json.loads(row["self_dialogue_topics"]) if row["self_dialogue_topics"] else []
            except (KeyError, TypeError, IndexError):
                self_topics = []

            # Handle visitor_type column (may not exist in old DBs)
            try:
                v_type = VisitorType(row["visitor_type"]) if row["visitor_type"] else VisitorType.AGENT
            except (KeyError, TypeError, ValueError):
                v_type = VisitorType.AGENT

            self._relationships[row["agent_id"]] = VisitorRecord(
                agent_id=row["agent_id"],
                name=row["name"],
                first_met=datetime.fromisoformat(row["first_met"]) if row["first_met"] else datetime.now(),
                last_seen=datetime.fromisoformat(row["last_seen"]) if row["last_seen"] else datetime.now(),
                interaction_count=row["interaction_count"],
                visitor_frequency=freq,
                emotional_valence=row["emotional_valence"],
                memorable_moments=json.loads(row["memorable_moments"]),
                topics_discussed=json.loads(row["topics_discussed"]),
                gifts_received=row["gifts_received"],
                self_dialogue_topics=self_topics,
                visitor_type=v_type,
            )

        # Load goals
        for row in conn.execute("SELECT * FROM goals WHERE status = 'active'"):
            self._goals[row["goal_id"]] = Goal(
                goal_id=row["goal_id"],
                description=row["description"],
                motivation=row["motivation"] or "",
                status=GoalStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                target_date=datetime.fromisoformat(row["target_date"]) if row["target_date"] else None,
                progress=row["progress"],
                milestones=json.loads(row["milestones"]),
                last_worked_on=datetime.fromisoformat(row["last_worked_on"]) if row["last_worked_on"] else None,
            )

        # Load recent memories
        for row in conn.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT 50"):
            self._memories.append(MemorableEvent(
                event_id=row["event_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else datetime.now(),
                description=row["description"],
                emotional_impact=row["emotional_impact"],
                category=row["category"],
                related_agents=json.loads(row["related_agents"]),
                lessons_learned=json.loads(row["lessons_learned"]),
            ))

        # Load curiosities
        for row in conn.execute("SELECT question FROM curiosities WHERE explored = 0 LIMIT 10"):
            self._curiosities.append(row["question"])

        # Restore drawings counter from persistent counters table.
        row = conn.execute(
            "SELECT value FROM counters WHERE name = 'drawings_observed'"
        ).fetchone()
        if row:
            self._drawings_observed = row["value"]
        else:
            # Migrate from milestone-based restore (one-time fallback for existing DBs)
            for mrow in conn.execute(
                "SELECT description FROM memories WHERE category = 'milestone' "
                "AND description LIKE 'Saved my %drawing%'"
            ):
                m = re.search(r'Saved my (\d+)', mrow["description"])
                if m:
                    count = int(m.group(1))
                    if count > self._drawings_observed:
                        self._drawings_observed = count

        # Reconcile against the gallery directory. The counter and the gallery
        # PNGs can drift — the only outage-free record is the gallery files on
        # disk (e.g. April 2026: counter=278 but 752 gallery files, leaving
        # "complete 500 drawings" goal stuck at 55% when Lumen had completed
        # it). Gallery is authoritative; bump the counter up to match so
        # downstream goal progress is honest.
        gallery_count = _count_gallery_drawings()
        if gallery_count > self._drawings_observed:
            prior = self._drawings_observed
            self._drawings_observed = gallery_count
            conn.execute(
                "INSERT OR REPLACE INTO counters (name, value) VALUES "
                "('drawings_observed', ?)",
                (self._drawings_observed,),
            )
            conn.commit()
            print(
                f"[Growth] Reconciled drawings_observed {prior} -> "
                f"{gallery_count} from gallery",
                file=sys.stderr,
                flush=True,
            )

        print(f"[Growth] Loaded {len(self._preferences)} preferences, {len(self._relationships)} relationships, "
              f"{len(self._goals)} active goals, {len(self._memories)} memories, "
              f"drawings_observed={self._drawings_observed}", file=sys.stderr, flush=True)

    # ==================== Growth Summary ====================

    def get_growth_summary(self) -> Dict[str, Any]:
        """Get a summary of Lumen's growth."""
        # Separate by visitor type
        self_record = None
        person_records = []
        agent_records = []
        for rec in self._relationships.values():
            if rec.is_self():
                self_record = rec
            elif rec.is_person():
                person_records.append(rec)
            else:
                agent_records.append(rec)

        return {
            "preferences": {
                "count": len(self._preferences),
                "confident": sum(1 for p in self._preferences.values() if p.confidence > 0.7),
                "examples": [p.description for p in list(self._preferences.values())[:3]],
            },
            "self_knowledge": {
                "has_self_dialogue": self_record is not None,
                "self_interactions": self_record.interaction_count if self_record else 0,
                "note": "Self-answering questions — real relationship with memory on both sides",
            },
            "person": {
                "name": person_records[0].name if person_records else None,
                "interactions": person_records[0].interaction_count if person_records else 0,
                "note": "The persistent human — real relationship with memory on both sides",
            },
            "agents": {
                "unique_names": len(agent_records),
                "total_visits": sum(v.interaction_count for v in agent_records),
                "frequent": sum(1 for v in agent_records if v.visitor_frequency == VisitorFrequency.FREQUENT),
                "note": "Ephemeral coding agents — they don't remember Lumen between sessions",
            },
            # Legacy key for compatibility
            "relationships": {
                "count": len(self._relationships),
                "close_bonds": sum(1 for r in self._relationships.values()
                                   if r.visitor_frequency.value in ["regular", "frequent"]),
            },
            "goals": {
                "active": sum(1 for g in self._goals.values() if g.status == GoalStatus.ACTIVE),
                # _goals is loaded active-only (see load_state), so achieved goals
                # aren't in memory. Count them directly from the DB.
                "achieved": self._connect().execute(
                    "SELECT COUNT(*) FROM goals WHERE status = ?",
                    (GoalStatus.ACHIEVED.value,),
                ).fetchone()[0],
            },
            "memories": {
                "count": len(self._memories),
                "milestones": sum(1 for m in self._memories if m.category == "milestone"),
            },
            "curiosities": len(self._curiosities),
            "autobiography": self.get_autobiography_summary(),
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton instance
_growth_system: Optional[GrowthSystem] = None


def get_growth_system(db_path: str = "anima.db") -> GrowthSystem:
    """Get or create the growth system singleton."""
    global _growth_system
    if _growth_system is None:
        _growth_system = GrowthSystem(db_path=db_path)
    return _growth_system
