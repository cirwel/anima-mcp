"""
Weighted Pathways - Layer 1 of the Experiential Accumulation system.

Hebbian-style context-action pathway strengths. When Lumen takes an action
in a particular context and the outcome is good, the pathway between that
context and that action gets stronger. Bad outcomes weaken it. Pathways
also decay over time toward a neutral baseline.

This gives Lumen a form of experiential learning that accumulates across
sessions: "in this kind of situation, doing X tends to work well."

Context is discretized into buckets so that similar-but-not-identical
states map to the same pathways (generalization).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import math
import sqlite3
import sys
import time
from .db_paths import resolve_db_path


PATHWAY_REWARD_VERSION = 2


# ---------------------------------------------------------------------------
# Context discretization
# ---------------------------------------------------------------------------

class SurpriseBucket(Enum):
    LOW = "low"          # < 0.15
    MODERATE = "mod"     # 0.15 - 0.35
    HIGH = "hi"          # > 0.35


class SatisfactionBucket(Enum):
    UNSATISFIED = "unsat"  # < 0.35
    NEUTRAL = "neut"       # 0.35 - 0.65
    SATISFIED = "sat"      # > 0.65


class DriveBucket(Enum):
    CALM = "calm"      # < 0.2
    WANTING = "want"   # 0.2 - 0.5
    URGENT = "urg"     # > 0.5


class ActivityBucket(Enum):
    RESTING = "rest"
    DROWSY = "drow"
    ACTIVE = "act"


def discretize_surprise(value: float) -> SurpriseBucket:
    """Bucket a surprise level."""
    if value < 0.15:
        return SurpriseBucket.LOW
    elif value <= 0.35:
        return SurpriseBucket.MODERATE
    else:
        return SurpriseBucket.HIGH


def discretize_satisfaction(value: float) -> SatisfactionBucket:
    """Bucket a satisfaction level."""
    if value < 0.35:
        return SatisfactionBucket.UNSATISFIED
    elif value <= 0.65:
        return SatisfactionBucket.NEUTRAL
    else:
        return SatisfactionBucket.SATISFIED


def discretize_drive(value: float) -> DriveBucket:
    """Bucket a drive intensity."""
    if value < 0.2:
        return DriveBucket.CALM
    elif value <= 0.5:
        return DriveBucket.WANTING
    else:
        return DriveBucket.URGENT


def discretize_activity(level: str) -> ActivityBucket:
    """Bucket an activity level string."""
    level_lower = level.lower()
    if level_lower == "active":
        return ActivityBucket.ACTIVE
    elif level_lower == "drowsy":
        return ActivityBucket.DROWSY
    else:
        return ActivityBucket.RESTING


def discretize_context(
    surprise: float = 0.0,
    satisfaction: float = 0.5,
    drive: float = 0.0,
    activity: str = "active",
) -> str:
    """
    Discretize a continuous context into a string key.

    Returns a key like "low|sat|calm|act".
    """
    s = discretize_surprise(surprise)
    sat = discretize_satisfaction(satisfaction)
    d = discretize_drive(drive)
    a = discretize_activity(activity)
    return f"{s.value}|{sat.value}|{d.value}|{a.value}"


# ---------------------------------------------------------------------------
# Pathway dataclass
# ---------------------------------------------------------------------------

@dataclass
class Pathway:
    """A single context-action pathway with Hebbian-style strength."""

    context_key: str
    action_key: str
    strength: float = 0.5
    use_count: int = 0
    last_used: float = 0.0  # time.time() timestamp
    total_reward: float = 0.0

    def decay(self, now: float) -> None:
        """
        Apply temporal decay to pathway strength.

        strength *= 0.999 ^ (seconds_since_use / 3600)
        Minimum strength is 0.01.
        """
        if self.last_used <= 0:
            return
        elapsed = max(0.0, now - self.last_used)
        hours = elapsed / 3600.0
        self.strength *= 0.999 ** hours
        self.strength = max(0.01, self.strength)

    def reinforce(self, outcome_quality: float, now: float, lr_bonus: float = 0.0) -> None:
        """
        Reinforce (or weaken) this pathway based on outcome quality.

        outcome_quality is clamped to [-1, 1].
        strength += lr * quality, bounded to [0.01, 5.0].
        lr_bonus: from experiential marks (pathway_lr_bonus), scales learning rate.
        """
        quality = float(outcome_quality)
        if not math.isfinite(quality):
            raise ValueError("outcome quality must be finite")
        quality = max(-1.0, min(1.0, quality))
        try:
            bonus = float(lr_bonus)
        except (TypeError, ValueError):
            bonus = 0.0
        if not math.isfinite(bonus):
            bonus = 0.0
        lr = 0.15 * max(0.0, 1.0 + bonus)
        self.strength += lr * quality
        self.strength = max(0.01, min(5.0, self.strength))
        self.use_count += 1
        self.last_used = now
        self.total_reward += quality


# ---------------------------------------------------------------------------
# WeightedPathways — persistent pathway store
# ---------------------------------------------------------------------------

class WeightedPathways:
    """
    Manages a collection of context-action pathways with SQLite persistence.

    Pathways strengthen when actions succeed in a given context
    and weaken through decay and negative outcomes.
    """

    def __init__(self, db_path: str = "anima.db"):
        self._db_path = Path(resolve_db_path(db_path))
        self._conn: Optional[sqlite3.Connection] = None
        self._pathways: Dict[str, Pathway] = {}  # keyed by "context_key|action_key"
        self._init_db()
        self._load_all()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with automatic reconnection on failure."""
        if self._conn is None:
            self._conn = self._create_connection()
        else:
            try:
                self._conn.execute("SELECT 1")
            except (sqlite3.Error, sqlite3.OperationalError):
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = self._create_connection()
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with retry logic."""
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self._db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=10000")
                return conn
            except sqlite3.Error as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
        raise last_error or sqlite3.Error("Failed to connect to database")

    def _init_db(self):
        """Create the pathways table if it doesn't exist."""
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pathways (
                    context_key TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    use_count INTEGER DEFAULT 0,
                    last_used REAL DEFAULT 0,
                    total_reward REAL DEFAULT 0,
                    PRIMARY KEY (context_key, action_key)
                );
                CREATE INDEX IF NOT EXISTS idx_pathways_context ON pathways(context_key);
                CREATE TABLE IF NOT EXISTS pathways_state (
                    key TEXT PRIMARY KEY,
                    data TEXT
                );
            """)
            self._ensure_reward_model_version(conn)
            conn.commit()
        except Exception as e:
            print(f"[WeightedPathways] DB init error (non-fatal): {e}", file=sys.stderr, flush=True)

    def _ensure_reward_model_version(self, conn: sqlite3.Connection) -> None:
        """Neutralize rewardless legacy pathways while preserving a snapshot."""
        import json

        row = conn.execute(
            "SELECT data FROM pathways_state WHERE key = 'reward_model_version'"
        ).fetchone()
        try:
            version = int(json.loads(row["data"])) if row else None
        except (TypeError, ValueError, json.JSONDecodeError):
            version = None
        if version is not None and version >= PATHWAY_REWARD_VERSION:
            return

        legacy_rows = conn.execute(
            """SELECT context_key, action_key, strength, use_count, last_used, total_reward
               FROM pathways
               WHERE use_count > 0 AND ABS(total_reward) < 1e-12
               ORDER BY context_key, action_key"""
        ).fetchall()
        if legacy_rows:
            snapshot = [dict(item) for item in legacy_rows]
            conn.execute(
                "INSERT OR REPLACE INTO pathways_state (key, data) VALUES (?, ?)",
                ("legacy_rewardless_v1", json.dumps(snapshot, sort_keys=True)),
            )
            conn.execute(
                """UPDATE pathways
                   SET strength = 0.5, use_count = 0, last_used = 0
                   WHERE use_count > 0 AND ABS(total_reward) < 1e-12"""
            )
            print(
                f"[WeightedPathways] Preserved and neutralized {len(legacy_rows)} "
                "rewardless legacy pathways",
                file=sys.stderr,
                flush=True,
            )
        conn.execute(
            "INSERT OR REPLACE INTO pathways_state (key, data) VALUES (?, ?)",
            ("reward_model_version", json.dumps(PATHWAY_REWARD_VERSION)),
        )

    def _load_all(self):
        """Load all pathways from the database."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT context_key, action_key, strength, use_count, last_used, total_reward FROM pathways"
            ).fetchall()
            for row in rows:
                pw = Pathway(
                    context_key=row["context_key"],
                    action_key=row["action_key"],
                    strength=row["strength"],
                    use_count=row["use_count"],
                    last_used=row["last_used"],
                    total_reward=row["total_reward"],
                )
                key = f"{pw.context_key}|{pw.action_key}"
                self._pathways[key] = pw
            if self._pathways:
                print(
                    f"[WeightedPathways] Loaded {len(self._pathways)} pathways",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:
            print(f"[WeightedPathways] DB load error (non-fatal): {e}", file=sys.stderr, flush=True)

    def _persist(self, pathway: Pathway):
        """Persist a single pathway to the database."""
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO pathways
                   (context_key, action_key, strength, use_count, last_used, total_reward)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    pathway.context_key,
                    pathway.action_key,
                    pathway.strength,
                    pathway.use_count,
                    pathway.last_used,
                    pathway.total_reward,
                ),
            )
            conn.commit()
        except Exception as e:
            print(f"[WeightedPathways] DB persist error (non-fatal): {e}", file=sys.stderr, flush=True)

    def _get_or_create(self, context_key: str, action_key: str) -> Pathway:
        """Get an existing pathway or create a new one with default strength."""
        key = f"{context_key}|{action_key}"
        if key not in self._pathways:
            self._pathways[key] = Pathway(
                context_key=context_key,
                action_key=action_key,
                strength=0.5,
                use_count=0,
                last_used=0.0,
                total_reward=0.0,
            )
        return self._pathways[key]

    def get_strength(self, context_key: str, action_key: str) -> float:
        """
        Get the current strength of a context-action pathway.

        Applies decay before returning. Returns 0.5 (neutral) for unknown pathways.
        """
        pw = self._get_or_create(context_key, action_key)
        pw.decay(time.time())
        return pw.strength

    def get_all_strengths(self, context_key: str) -> Dict[str, float]:
        """
        Get all pathway strengths for a given context.

        Returns a dict mapping action_key -> strength (with decay applied).
        Only returns pathways that match the exact context key.
        """
        now = time.time()
        result: Dict[str, float] = {}
        for key, pw in self._pathways.items():
            if pw.context_key == context_key:
                pw.decay(now)
                result[pw.action_key] = pw.strength
        return result

    def reinforce(self, context_key: str, action_key: str, outcome_quality: float,
                  lr_bonus: float = 0.0) -> None:
        """
        Reinforce a context-action pathway based on outcome quality.

        outcome_quality: positive for good outcomes, negative for bad.
        lr_bonus: from experiential marks, scales learning rate.
        Persists the updated pathway.
        """
        pw = self._get_or_create(context_key, action_key)
        now = time.time()
        pw.decay(now)
        pw.reinforce(outcome_quality, now, lr_bonus=lr_bonus)
        self._persist(pw)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for shared memory / diagnostics."""
        if not self._pathways:
            return {
                "total_pathways": 0,
                "unique_contexts": 0,
                "unique_actions": 0,
                "avg_strength": 0.5,
                "total_reinforcements": 0,
            }
        contexts = set()
        actions = set()
        total_strength = 0.0
        total_uses = 0
        for pw in self._pathways.values():
            contexts.add(pw.context_key)
            actions.add(pw.action_key)
            total_strength += pw.strength
            total_uses += pw.use_count
        return {
            "total_pathways": len(self._pathways),
            "unique_contexts": len(contexts),
            "unique_actions": len(actions),
            "avg_strength": round(total_strength / len(self._pathways), 3),
            "total_reinforcements": total_uses,
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_weighted_pathways: Optional[WeightedPathways] = None


def get_weighted_pathways(db_path: str = "anima.db") -> WeightedPathways:
    """Get or create the WeightedPathways singleton."""
    global _weighted_pathways
    if _weighted_pathways is None:
        _weighted_pathways = WeightedPathways(db_path=db_path)
    return _weighted_pathways
