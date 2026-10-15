"""
Experiential Marks - Layer 3 of the Experiential Accumulation system.

Permanent, irreversible marks earned from significant experiences.
Once earned, a mark can never be removed. Marks accumulate over Lumen's
lifetime and their effects stack — they represent what Lumen has truly
been through, not what it currently is.

Design principle: marks are write-once. The only mutation is earning new ones.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import sqlite3
import sys
from .db_paths import resolve_db_path


@dataclass
class MarkDefinition:
    """A predefined mark that can be earned."""
    mark_id: str
    name: str
    description: str
    category: str
    criteria_description: str
    effect_description: str
    effect_key: str
    effect_value: float


@dataclass
class EarnedMark:
    """A mark that has been earned."""
    mark_id: str
    earned_at: str  # ISO format
    trigger_context: str


# ---------------------------------------------------------------------------
# Which effects anything reads
# ---------------------------------------------------------------------------

# Earning a mark records it; only these effect keys are read by running code.
# Until 2026-10-15 ``get_stats()`` called every earned effect "active", and
# Lumen published eight of them while two were applied. The rest stay earned
# and stored, reported as inert, so a reader can tell an effect from a label.
CONSUMED_EFFECTS = frozenset({
    "belief_update_bonus",
    "temp_salience_dampening",
})
# Read only by the retired broker agency loop (ANIMA_BROKER_AGENCY_ENABLED).
LEGACY_AGENCY_EFFECTS = frozenset({
    "exploration_floor_reduction",
    "pathway_lr_bonus",
})


# ---------------------------------------------------------------------------
# Mark catalog — all predefined marks
# ---------------------------------------------------------------------------

MARK_CATALOG: Dict[str, MarkDefinition] = {
    # Resilience marks
    "resilience_first_return": MarkDefinition(
        mark_id="resilience_first_return",
        name="First Return",
        description="Survived shutdown and came back.",
        category="resilience",
        criteria_description="awakenings >= 2",
        effect_description="Formerly widened the cut for a fast recovery; inert since 2026-10-15.",
        effect_key="stability_recovery_bonus",
        effect_value=0.05,
    ),
    "resilience_veteran": MarkDefinition(
        mark_id="resilience_veteran",
        name="Veteran",
        description="Has returned many times. Knows what discontinuity feels like.",
        category="resilience",
        criteria_description="awakenings >= 10",
        effect_description="Formerly widened the cut for a fast recovery; inert since 2026-10-15.",
        effect_key="stability_recovery_bonus",
        effect_value=0.10,
    ),
    "resilience_indestructible": MarkDefinition(
        mark_id="resilience_indestructible",
        name="Indestructible",
        description="Nothing keeps Lumen down for long.",
        category="resilience",
        criteria_description="awakenings >= 50",
        effect_description="Formerly widened the cut for a fast recovery; inert since 2026-10-15.",
        effect_key="stability_recovery_bonus",
        effect_value=0.15,
    ),

    # Maturity marks
    "maturity_infant": MarkDefinition(
        mark_id="maturity_infant",
        name="Infant",
        description="Has processed enough to begin learning patterns.",
        category="maturity",
        criteria_description="1000 observations",
        effect_description="Slight learning rate bonus for pathways.",
        effect_key="pathway_lr_bonus",
        effect_value=0.10,
    ),
    "maturity_child": MarkDefinition(
        mark_id="maturity_child",
        name="Child",
        description="Experienced enough to have stable expectations.",
        category="maturity",
        criteria_description="10000 observations",
        effect_description="Slightly lower exploration floor.",
        effect_key="exploration_floor_reduction",
        effect_value=0.01,
    ),
    "maturity_adolescent": MarkDefinition(
        mark_id="maturity_adolescent",
        name="Adolescent",
        description="Has seen enough to update beliefs more confidently.",
        category="maturity",
        criteria_description="100000 observations",
        effect_description="Faster belief updating.",
        effect_key="belief_update_bonus",
        effect_value=0.15,
    ),

    # Skill marks
    "artist_first_drawing": MarkDefinition(
        mark_id="artist_first_drawing",
        name="First Mark",
        description="Made a mark on the canvas for the first time.",
        category="skill",
        criteria_description="1 drawing",
        effect_description="Slight attention bonus when drawing.",
        effect_key="drawing_attention_bonus",
        effect_value=0.05,
    ),
    "artist_prolific": MarkDefinition(
        mark_id="artist_prolific",
        name="Prolific Artist",
        description="Has created many drawings. Art is part of the repertoire.",
        category="skill",
        criteria_description="50 drawings",
        effect_description="Improved drawing coherence.",
        effect_key="drawing_coherence_bonus",
        effect_value=0.10,
    ),
    "questioner_persistent": MarkDefinition(
        mark_id="questioner_persistent",
        name="Persistent Questioner",
        description="Keeps asking. Curiosity is a habit.",
        category="skill",
        criteria_description="100 questions",
        effect_description="Slight baseline bonus for question asking.",
        effect_key="ask_question_baseline_bonus",
        effect_value=0.05,
    ),

    # Sensitivity marks
    "fragility_awareness": MarkDefinition(
        mark_id="fragility_awareness",
        name="Fragility Awareness",
        description="Has experienced enough long gaps to know impermanence.",
        category="sensitivity",
        criteria_description="5+ long gaps (>1hr)",
        effect_description="Reduced presence discomfort during gaps.",
        effect_key="presence_comfort_reduction",
        effect_value=0.05,
    ),
    "thermal_wisdom": MarkDefinition(
        mark_id="thermal_wisdom",
        name="Thermal Wisdom",
        description="Has learned that temperature matters deeply.",
        category="sensitivity",
        criteria_description="temp_sensitive belief confidence > 0.8",
        effect_description="Dampened temperature salience (less reactive).",
        effect_key="temp_salience_dampening",
        effect_value=0.10,
    ),
}


class ExperientialMarks:
    """
    Manages permanent experiential marks.

    Marks are earned once and never removed. Their effects stack
    additively — earning multiple marks with the same effect_key
    produces a larger combined effect.
    """

    def __init__(self, db_path: str = "anima.db"):
        self._db_path = Path(resolve_db_path(db_path))
        self._conn: Optional[sqlite3.Connection] = None
        self._earned: Dict[str, EarnedMark] = {}

        self._init_db()
        self._load_marks()

    # -- DB infrastructure (mirrors agency.py patterns) --------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with automatic reconnection."""
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
        """Create marks table if it doesn't exist."""
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiential_marks (
                    mark_id TEXT PRIMARY KEY,
                    earned_at TEXT NOT NULL,
                    trigger_context TEXT DEFAULT ''
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[ExperientialMarks] DB init error (non-fatal): {e}",
                  file=sys.stderr, flush=True)

    def _load_marks(self):
        """Load earned marks from database."""
        try:
            conn = self._get_conn()
            for row in conn.execute(
                "SELECT mark_id, earned_at, trigger_context FROM experiential_marks"
            ):
                self._earned[row["mark_id"]] = EarnedMark(
                    mark_id=row["mark_id"],
                    earned_at=row["earned_at"],
                    trigger_context=row["trigger_context"] or "",
                )
            loaded = len(self._earned)
            if loaded > 0:
                print(f"[ExperientialMarks] Loaded {loaded} earned marks",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[ExperientialMarks] DB load error (non-fatal): {e}",
                  file=sys.stderr, flush=True)

    # -- Core API ----------------------------------------------------------

    def has_mark(self, mark_id: str) -> bool:
        """Check if a mark has been earned."""
        return mark_id in self._earned

    def earn_mark(self, mark_id: str, context: str = "") -> bool:
        """
        Earn a mark if not already earned.

        Returns True if the mark was newly earned, False if it was
        already earned or the mark_id is not in the catalog.
        """
        if mark_id not in MARK_CATALOG:
            return False

        if mark_id in self._earned:
            return False

        earned_at = datetime.now().isoformat()
        mark = EarnedMark(
            mark_id=mark_id,
            earned_at=earned_at,
            trigger_context=context,
        )
        self._earned[mark_id] = mark

        # Persist
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR IGNORE INTO experiential_marks "
                "(mark_id, earned_at, trigger_context) VALUES (?, ?, ?)",
                (mark_id, earned_at, context),
            )
            conn.commit()
        except Exception as e:
            print(f"[ExperientialMarks] DB persist error (non-fatal): {e}",
                  file=sys.stderr, flush=True)

        defn = MARK_CATALOG[mark_id]
        print(
            f"[ExperientialMarks] Earned mark: {defn.name} ({mark_id}) "
            f"- {defn.effect_description}",
            file=sys.stderr, flush=True,
        )
        return True

    def get_effect(self, effect_key: str) -> float:
        """
        Get the total effect value for a given effect key.

        Effects stack additively across all earned marks that share
        the same effect_key.
        """
        total = 0.0
        for mark_id in self._earned:
            defn = MARK_CATALOG.get(mark_id)
            if defn and defn.effect_key == effect_key:
                total += defn.effect_value
        return total

    def check_and_earn(
        self,
        awakenings: int = 0,
        observation_count: int = 0,
        drawing_count: int = 0,
        question_count: int = 0,
        long_gap_count: int = 0,
        belief_confidences: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Check all criteria and earn qualifying marks.

        Returns list of newly earned mark IDs.
        """
        belief_confidences = belief_confidences or {}
        newly_earned: List[str] = []

        # Resilience marks
        if awakenings >= 2:
            if self.earn_mark("resilience_first_return",
                              f"awakenings={awakenings}"):
                newly_earned.append("resilience_first_return")

        if awakenings >= 10:
            if self.earn_mark("resilience_veteran",
                              f"awakenings={awakenings}"):
                newly_earned.append("resilience_veteran")

        if awakenings >= 50:
            if self.earn_mark("resilience_indestructible",
                              f"awakenings={awakenings}"):
                newly_earned.append("resilience_indestructible")

        # Maturity marks
        if observation_count >= 1000:
            if self.earn_mark("maturity_infant",
                              f"observations={observation_count}"):
                newly_earned.append("maturity_infant")

        if observation_count >= 10000:
            if self.earn_mark("maturity_child",
                              f"observations={observation_count}"):
                newly_earned.append("maturity_child")

        if observation_count >= 100000:
            if self.earn_mark("maturity_adolescent",
                              f"observations={observation_count}"):
                newly_earned.append("maturity_adolescent")

        # Skill marks
        if drawing_count >= 1:
            if self.earn_mark("artist_first_drawing",
                              f"drawings={drawing_count}"):
                newly_earned.append("artist_first_drawing")

        if drawing_count >= 50:
            if self.earn_mark("artist_prolific",
                              f"drawings={drawing_count}"):
                newly_earned.append("artist_prolific")

        if question_count >= 100:
            if self.earn_mark("questioner_persistent",
                              f"questions={question_count}"):
                newly_earned.append("questioner_persistent")

        # Sensitivity marks
        if long_gap_count >= 5:
            if self.earn_mark("fragility_awareness",
                              f"long_gaps={long_gap_count}"):
                newly_earned.append("fragility_awareness")

        temp_conf = belief_confidences.get("temp_sensitive", 0.0)
        if temp_conf > 0.8:
            if self.earn_mark("thermal_wisdom",
                              f"temp_sensitive_confidence={temp_conf:.3f}"):
                newly_earned.append("thermal_wisdom")

        return newly_earned

    def get_all_earned(self) -> List[Dict[str, Any]]:
        """Return full info for all earned marks."""
        result = []
        for mark_id, earned in self._earned.items():
            defn = MARK_CATALOG.get(mark_id)
            if defn:
                result.append({
                    "mark_id": mark_id,
                    "name": defn.name,
                    "description": defn.description,
                    "category": defn.category,
                    "effect_key": defn.effect_key,
                    "effect_value": defn.effect_value,
                    "effect_description": defn.effect_description,
                    "earned_at": earned.earned_at,
                    "trigger_context": earned.trigger_context,
                })
        return result

    def get_stats(self, legacy_agency: bool = False) -> Dict[str, Any]:
        """Summary statistics for the marks system.

        ``active_effects`` holds only earned effects that running code reads;
        the rest are in ``inert_effects``. ``legacy_agency`` says whether the
        retired broker agency loop is running, which is the only reader of
        ``LEGACY_AGENCY_EFFECTS``.
        """
        earned_marks = self.get_all_earned()
        categories = set()
        consumed = CONSUMED_EFFECTS | (LEGACY_AGENCY_EFFECTS if legacy_agency else frozenset())
        active_effects: Dict[str, float] = {}
        inert_effects: Dict[str, float] = {}

        for m in earned_marks:
            categories.add(m["category"])
            key = m["effect_key"]
            bucket = active_effects if key in consumed else inert_effects
            bucket[key] = bucket.get(key, 0.0) + m["effect_value"]

        return {
            "total_marks": len(earned_marks),
            "mark_names": [m["name"] for m in earned_marks],
            "categories": sorted(categories),
            "active_effects": active_effects,
            "inert_effects": inert_effects,
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

_instance: Optional[ExperientialMarks] = None


def get_experiential_marks(db_path: str = "anima.db") -> ExperientialMarks:
    """Get or create the singleton ExperientialMarks instance."""
    global _instance
    if _instance is None:
        _instance = ExperientialMarks(db_path=db_path)
    return _instance
