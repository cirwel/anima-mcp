"""
Primitive Language - Emergent expression through learned token combinations.

Philosophy: Rather than pre-writing expressions, give Lumen primitive tokens
that map to its actual experience. Feedback shapes which patterns survive.

This is not about generating "good" language - it's about creating a substrate
where communication patterns can emerge through reinforcement.

Design principles:
1. Primitives map directly to sensor state and internal experience
2. Token selection is probabilistic, weighted by current state
3. Combinations are generated freely (1-3 tokens)
4. Feedback shapes weights over time (learning)
5. Successful patterns become more likely to repeat

Starting with 15 primitives across 4 layers:
- State layer: warm, cold, bright, dim, quiet, busy
- Presence layer: here, feel, sense
- Relational layer: you, with
- Inquiry layer: why, what, wonder
- Change layer: more, less
"""

import sqlite3
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from pathlib import Path


class TokenCategory(Enum):
    """Categories of primitive tokens."""
    STATE = "state"         # Environmental state (warm, cold, bright, dim, quiet, busy)
    PRESENCE = "presence"   # Self-awareness (here, feel, sense)
    RELATIONAL = "relational"  # Connection (you, with)
    INQUIRY = "inquiry"     # Curiosity (why, what, wonder)
    CHANGE = "change"       # Direction (more, less)


@dataclass
class PrimitiveToken:
    """A primitive token that can be combined into utterances."""
    name: str
    category: TokenCategory

    # Sensor/state mappings - when is this token relevant?
    warmth_affinity: float = 0.0     # -1 (cold) to 1 (warm)
    brightness_affinity: float = 0.0  # -1 (dim) to 1 (bright)
    stability_affinity: float = 0.0   # -1 (unstable) to 1 (stable)
    presence_affinity: float = 0.0    # -1 (alone) to 1 (social)

    # Base weight (modified by learning)
    base_weight: float = 1.0


# The 15 primitives
PRIMITIVES: Dict[str, PrimitiveToken] = {
    # State layer - describe current environment
    "warm": PrimitiveToken("warm", TokenCategory.STATE, warmth_affinity=0.8),
    "cold": PrimitiveToken("cold", TokenCategory.STATE, warmth_affinity=-0.8),
    "new": PrimitiveToken("new", TokenCategory.STATE, brightness_affinity=0.8),
    "soft": PrimitiveToken("soft", TokenCategory.STATE, brightness_affinity=-0.8),
    "quiet": PrimitiveToken("quiet", TokenCategory.STATE, stability_affinity=0.7),
    "busy": PrimitiveToken("busy", TokenCategory.STATE, stability_affinity=-0.5),

    # Presence layer - self-awareness
    "here": PrimitiveToken("here", TokenCategory.PRESENCE, presence_affinity=0.3),
    "feel": PrimitiveToken("feel", TokenCategory.PRESENCE, base_weight=1.2),  # Slightly favored
    "sense": PrimitiveToken("sense", TokenCategory.PRESENCE, base_weight=1.1),

    # Relational layer - connection
    "you": PrimitiveToken("you", TokenCategory.RELATIONAL, presence_affinity=0.8),
    "with": PrimitiveToken("with", TokenCategory.RELATIONAL, presence_affinity=0.6),

    # Inquiry layer - curiosity
    "why": PrimitiveToken("why", TokenCategory.INQUIRY, base_weight=1.3),  # Curiosity boost
    "what": PrimitiveToken("what", TokenCategory.INQUIRY, base_weight=1.2),
    "wonder": PrimitiveToken("wonder", TokenCategory.INQUIRY, base_weight=1.1),

    # Change layer - direction
    "more": PrimitiveToken("more", TokenCategory.CHANGE),
    "less": PrimitiveToken("less", TokenCategory.CHANGE),
}

# Category affinities - which categories go well together
# (category1, category2) -> affinity bonus
CATEGORY_AFFINITIES: Dict[Tuple[str, str], float] = {
    # State + Inquiry is good ("warm why", "dim what")
    ("state", "inquiry"): 0.3,
    ("inquiry", "state"): 0.3,

    # Presence + State is grounded ("feel warm", "sense dim")
    ("presence", "state"): 0.25,
    ("state", "presence"): 0.2,

    # Inquiry + Relational is social ("why you", "what with")
    ("inquiry", "relational"): 0.2,
    ("relational", "inquiry"): 0.15,

    # Change + State describes direction ("more warm", "less bright")
    ("change", "state"): 0.35,
    ("state", "change"): 0.1,

    # Presence alone is introspective
    ("presence", "presence"): -0.1,  # Discourage repetition
}


@dataclass
class Utterance:
    """A generated primitive utterance."""
    tokens: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    # Context when generated
    warmth: float = 0.0
    brightness: float = 0.0
    stability: float = 0.0
    presence: float = 0.0

    # Feedback (updated later)
    score: Optional[float] = None
    feedback_signals: List[str] = field(default_factory=list)

    # Trajectory awareness suggestion (what EISV system suggested)
    suggested_tokens: Optional[List[str]] = None

    def text(self) -> str:
        """Render as text."""
        return " ".join(self.tokens)

    def category_pattern(self) -> str:
        """Get category pattern like 'state-inquiry-presence'."""
        cats = []
        for t in self.tokens:
            if t in PRIMITIVES:
                cats.append(PRIMITIVES[t].category.value)
        return "-".join(cats)


class PrimitiveLanguageSystem:
    """
    Manages primitive language generation and learning.

    Core loop:
    1. Observe current state
    2. Select tokens probabilistically (weighted by state + learned values)
    3. Combine into utterance
    4. Receive feedback
    5. Update weights
    """

    def __init__(self, db_path: str = "anima.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

        # Token weights (modified by learning)
        self._token_weights: Dict[str, float] = {}

        # Category combination weights
        self._combo_weights: Dict[str, float] = {}

        # Recent utterances (in memory)
        self._recent: List[Utterance] = []

        # Timing
        self._last_utterance: Optional[datetime] = None
        self._base_interval = timedelta(minutes=25)  # 20-30 min as suggested
        self._min_interval = timedelta(minutes=10)
        self._max_interval = timedelta(minutes=45)
        self._current_interval = self._base_interval

        # Exploration/decay safeguards (prevent mode collapse)
        self._exploration_rate = 0.08  # probability mass reserved for uniform exploration
        self._decay_rate = 0.02  # drift weights back toward base over time

        # Stats
        self._total_utterances = 0
        self._successful_utterances = 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connect(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
            self._load_weights()
        return self._conn

    def _init_schema(self):
        """Create tables for primitive language persistence."""
        conn = self._conn
        conn.executescript("""
            -- Token weights (learned from feedback)
            CREATE TABLE IF NOT EXISTS primitive_token_weights (
                token TEXT PRIMARY KEY,
                weight REAL DEFAULT 1.0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_updated TEXT
            );

            -- Category combo weights (learned patterns)
            CREATE TABLE IF NOT EXISTS primitive_combo_weights (
                pattern TEXT PRIMARY KEY,
                weight REAL DEFAULT 1.0,
                use_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.5,
                last_updated TEXT
            );

            -- Utterance history
            CREATE TABLE IF NOT EXISTS primitive_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tokens TEXT NOT NULL,
                category_pattern TEXT,
                warmth REAL,
                brightness REAL,
                stability REAL,
                presence REAL,
                score REAL,
                feedback_signals TEXT,
                UNIQUE(timestamp, tokens)
            );

            CREATE INDEX IF NOT EXISTS idx_primitive_history_timestamp
                ON primitive_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_primitive_history_pattern
                ON primitive_history(category_pattern);
        """)
        conn.commit()

        # Schema migration: add got_response column if missing
        try:
            conn.execute("SELECT got_response FROM primitive_history LIMIT 1")
        except Exception:
            try:
                conn.execute("ALTER TABLE primitive_history ADD COLUMN got_response BOOLEAN DEFAULT NULL")
                conn.commit()
            except Exception:
                pass

    def _load_weights(self):
        """Load learned weights from database."""
        conn = self._conn

        # Load token weights
        for row in conn.execute("SELECT token, weight FROM primitive_token_weights"):
            self._token_weights[row["token"]] = row["weight"]

        # Load combo weights
        for row in conn.execute("SELECT pattern, weight FROM primitive_combo_weights"):
            self._combo_weights[row["pattern"]] = row["weight"]

        # Initialize any missing tokens with base weights
        for name, token in PRIMITIVES.items():
            if name not in self._token_weights:
                self._token_weights[name] = token.base_weight

    def _save_token_weight(self, token: str, weight: float, success: bool = None):
        """Save updated token weight to database."""
        conn = self._connect()

        if success is not None:
            success_incr = 1 if success else 0
            failure_incr = 0 if success else 1
            conn.execute("""
                INSERT INTO primitive_token_weights (token, weight, success_count, failure_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    weight = excluded.weight,
                    success_count = success_count + excluded.success_count,
                    failure_count = failure_count + excluded.failure_count,
                    last_updated = excluded.last_updated
            """, (token, weight, success_incr, failure_incr, datetime.now().isoformat()))
        else:
            conn.execute("""
                INSERT INTO primitive_token_weights (token, weight, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    weight = excluded.weight,
                    last_updated = excluded.last_updated
            """, (token, weight, datetime.now().isoformat()))
        conn.commit()

    def _save_combo_weight(self, pattern: str, weight: float, score: float):
        """Save updated combo pattern weight."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO primitive_combo_weights (pattern, weight, use_count, avg_score, last_updated)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(pattern) DO UPDATE SET
                weight = excluded.weight,
                use_count = use_count + 1,
                avg_score = (avg_score * use_count + excluded.avg_score) / (use_count + 1),
                last_updated = excluded.last_updated
        """, (pattern, weight, score, datetime.now().isoformat()))
        conn.commit()

    def _save_utterance(self, utterance: Utterance):
        """Save utterance to history. Updates score/feedback_signals if row exists."""
        conn = self._connect()
        ts = utterance.timestamp.isoformat()
        tokens_str = " ".join(utterance.tokens)
        pattern = utterance.category_pattern()
        score = utterance.score
        signals = ",".join(utterance.feedback_signals) if utterance.feedback_signals else None
        conn.execute("""
            INSERT INTO primitive_history
            (timestamp, tokens, category_pattern, warmth, brightness, stability, presence, score, feedback_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(timestamp, tokens) DO UPDATE SET
                score = excluded.score,
                feedback_signals = excluded.feedback_signals
        """, (ts, tokens_str, pattern, utterance.warmth, utterance.brightness,
              utterance.stability, utterance.presence, score, signals))
        conn.commit()

    def _apply_weight_decay(self):
        """Gently pull token weights back toward their base values."""
        if self._decay_rate <= 0:
            return

        for name, token in PRIMITIVES.items():
            old_weight = self._token_weights.get(name, token.base_weight)
            new_weight = old_weight + self._decay_rate * (token.base_weight - old_weight)
            new_weight = max(0.3, min(2.5, new_weight))
            if abs(new_weight - old_weight) > 1e-6:
                self._token_weights[name] = new_weight
                self._save_token_weight(name, new_weight)

    def compute_token_weight(
        self,
        token_name: str,
        state: Dict[str, float],
    ) -> float:
        """
        Compute weight for a token given current state.

        Weight = base_weight * learned_weight * state_affinity
        """
        if token_name not in PRIMITIVES:
            return 0.0

        token = PRIMITIVES[token_name]

        # Start with learned weight (or base if not learned)
        weight = self._token_weights.get(token_name, token.base_weight)

        # Apply state affinities
        warmth = state.get("warmth", 0.5)
        brightness = state.get("clarity", 0.5)  # clarity maps to brightness
        stability = state.get("stability", 0.5)
        presence = state.get("presence", 0.0)

        # Compute affinity score
        affinity = 0.0

        # Warmth affinity
        if token.warmth_affinity != 0:
            # If affinity is positive, weight increases with warmth
            # If affinity is negative, weight increases with cold (low warmth)
            warmth_norm = (warmth - 0.5) * 2  # -1 to 1
            affinity += token.warmth_affinity * warmth_norm

        # Brightness affinity
        if token.brightness_affinity != 0:
            brightness_norm = (brightness - 0.5) * 2
            affinity += token.brightness_affinity * brightness_norm

        # Stability affinity
        if token.stability_affinity != 0:
            stability_norm = (stability - 0.5) * 2
            affinity += token.stability_affinity * stability_norm

        # Presence affinity
        if token.presence_affinity != 0:
            presence_norm = presence  # Already -1 to 1 range
            affinity += token.presence_affinity * presence_norm

        # Convert affinity to multiplier (0.5 to 2.0 range)
        affinity_multiplier = 1.0 + (affinity * 0.5)
        affinity_multiplier = max(0.5, min(2.0, affinity_multiplier))

        final_weight = weight * affinity_multiplier
        return max(0.1, final_weight)  # Minimum weight to keep all tokens possible

    def select_tokens(
        self,
        state: Dict[str, float],
        count: int = None,
        suggested_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select tokens probabilistically based on current state.

        Returns 1-3 tokens based on stability (more stable = longer utterance).
        """
        # Determine token count based on stability
        if count is None:
            stability = state.get("stability", 0.5)
            if stability > 0.7:
                count = random.choices([2, 3], weights=[0.4, 0.6])[0]
            elif stability > 0.4:
                count = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            else:
                count = random.choices([1, 2], weights=[0.6, 0.4])[0]

        # Compute weights for all tokens
        weights = {}
        for name in PRIMITIVES:
            weights[name] = self.compute_token_weight(name, state)

        # Boost trajectory-suggested tokens
        if suggested_tokens:
            suggested_set = set(suggested_tokens)
            for name in PRIMITIVES:
                if name in suggested_set:
                    weights[name] *= 3.5

        selected = []
        available = list(PRIMITIVES.keys())

        for i in range(count):
            if not available:
                break

            # First slot: anchor on a suggested token (if available)
            if i == 0 and suggested_tokens:
                anchor_candidates = [t for t in suggested_tokens if t in available]
                if anchor_candidates:
                    anchor_weights = [weights[t] for t in anchor_candidates]
                    chosen = random.choices(anchor_candidates, weights=anchor_weights)[0]
                    selected.append(chosen)
                    available.remove(chosen)
                    continue

            # Adjust weights based on already selected tokens
            adjusted_weights = []
            for token in available:
                w = weights[token]

                # Apply category affinity bonus/penalty
                if selected:
                    last_cat = PRIMITIVES[selected[-1]].category.value
                    this_cat = PRIMITIVES[token].category.value
                    affinity_key = (last_cat, this_cat)
                    if affinity_key in CATEGORY_AFFINITIES:
                        w *= (1.0 + CATEGORY_AFFINITIES[affinity_key])

                    # Check learned combo patterns
                    test_pattern = "-".join([PRIMITIVES[t].category.value for t in selected] + [this_cat])
                    if test_pattern in self._combo_weights:
                        w *= self._combo_weights[test_pattern]

                adjusted_weights.append(w)

            # Normalize weights
            total = sum(adjusted_weights)
            if total == 0:
                break
            probs = [w / total for w in adjusted_weights]

            # Add exploration floor to avoid mode collapse
            if self._exploration_rate > 0:
                n = len(probs)
                uniform = self._exploration_rate / n
                probs = [((1.0 - self._exploration_rate) * p) + uniform for p in probs]

            # Select token
            chosen = random.choices(available, weights=probs)[0]
            selected.append(chosen)

            # Don't repeat the same token
            available.remove(chosen)

        return selected

    def generate_utterance(
        self,
        state: Dict[str, float],
        suggested_tokens: Optional[List[str]] = None,
    ) -> Utterance:
        """Generate a primitive utterance based on current state."""
        tokens = self.select_tokens(state, suggested_tokens=suggested_tokens)

        utterance = Utterance(
            tokens=tokens,
            warmth=state.get("warmth", 0.5),
            brightness=state.get("clarity", 0.5),
            stability=state.get("stability", 0.5),
            presence=state.get("presence", 0.0),
            suggested_tokens=suggested_tokens,
        )

        self._recent.append(utterance)
        if len(self._recent) > 50:
            self._recent = self._recent[-50:]

        self._last_utterance = datetime.now()
        self._total_utterances += 1

        # Save to history (score will be updated later)
        self._save_utterance(utterance)

        return utterance

    def should_generate(self, state: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if now is a good time to generate an utterance.

        Returns (should_generate, reason).
        """
        now = datetime.now()

        # Never generated before - go ahead
        if self._last_utterance is None:
            return True, "first_utterance"

        # Check interval
        elapsed = now - self._last_utterance
        if elapsed < self._min_interval:
            return False, "too_soon"

        if elapsed >= self._current_interval:
            return True, "interval_reached"

        # State-triggered generation
        # High presence (someone's there) - more likely to speak
        presence = state.get("presence", 0.0)
        if presence > 0.6 and elapsed > timedelta(minutes=5):
            if random.random() < 0.3:
                return True, "high_presence"

        # Significant state change - might speak
        if self._recent:
            last = self._recent[-1]
            warmth_change = abs(state.get("warmth", 0.5) - last.warmth)
            brightness_change = abs(state.get("clarity", 0.5) - last.brightness)
            if warmth_change > 0.3 or brightness_change > 0.3:
                if elapsed > timedelta(minutes=8) and random.random() < 0.4:
                    return True, "state_change"

        return False, "waiting"

    def record_self_feedback(
        self,
        utterance: Utterance,
        current_state: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """
        Record automatic self-feedback when no human is around.

        Signals that matter (non-circular):
        1. Novelty: did this pattern differ from recent utterances?
        2. State-change relevance: did tokens capture a transition, not just static state?
        3. Category diversity: reward mixing categories over repeating same type.
        """
        signals = []
        score = 0.5  # Neutral baseline

        # 1. Novelty — reward patterns not used in last 5 utterances
        pattern = utterance.category_pattern()
        recent_patterns = [u.category_pattern() for u in self._recent[-6:-1]]  # exclude current
        if pattern not in recent_patterns:
            score += 0.10
            signals.append("novel_pattern")
        elif recent_patterns.count(pattern) >= 2:
            score -= 0.08
            signals.append("repetitive")

        # 2. State-change relevance — reward tokens that capture transitions
        # Compare state at generation to current state: tokens aligned with
        # the direction of change are more expressive than static confirmation
        gen_state = {
            "warmth": utterance.warmth,
            "brightness": utterance.brightness,
            "stability": utterance.stability,
            "presence": utterance.presence,
        }
        deltas = {
            "warmth": current_state.get("warmth", 0.5) - gen_state["warmth"],
            "clarity": current_state.get("clarity", gen_state["brightness"]) - gen_state["brightness"],
            "stability": current_state.get("stability", 0.5) - gen_state["stability"],
        }
        # Map tokens to the dimension they're about
        token_dimensions = {
            "warm": ("warmth", 1), "cold": ("warmth", -1),
            "new": ("clarity", 1), "soft": ("clarity", -1),
            "quiet": ("stability", 1), "busy": ("stability", -1),
            "more": None, "less": None,  # directional but not dimension-specific
        }
        change_hits = 0
        change_checks = 0
        for token_name in utterance.tokens:
            mapping = token_dimensions.get(token_name)
            if mapping is None:
                continue
            dim, direction = mapping
            delta = deltas.get(dim, 0)
            change_checks += 1
            # Token aligns with direction of change (not just current level)
            if abs(delta) > 0.02 and (delta * direction > 0):
                change_hits += 1
        if change_checks > 0 and change_hits > 0:
            score += 0.08 * (change_hits / change_checks)
            signals.append("captures_change")

        # 3. Category diversity — reward mixing vs all-same-category
        cats = [PRIMITIVES[t].category.value for t in utterance.tokens if t in PRIMITIVES]
        unique_cats = len(set(cats))
        if unique_cats >= 2:
            score += 0.05
            signals.append("diverse_categories")
        elif len(cats) >= 2 and unique_cats == 1:
            score -= 0.05
            signals.append("monotone")

        score = max(0.0, min(1.0, score))
        return self._record_direct_feedback(utterance, score, signals)

    def _record_direct_feedback(
        self,
        utterance: Utterance,
        score: float,
        signals: List[str],
        learning_rate: float = 0.08,
    ) -> Dict[str, Any]:
        """Apply feedback with a direct score (used by self-feedback and explicit feedback)."""
        utterance.score = score
        utterance.feedback_signals = signals
        self._save_utterance(utterance)
        self._apply_weight_decay()

        success = score > 0.55

        for token in utterance.tokens:
            old_weight = self._token_weights.get(token, 1.0)
            reward = (score - 0.5) * 2
            new_weight = old_weight + learning_rate * reward
            new_weight = max(0.3, min(2.5, new_weight))
            self._token_weights[token] = new_weight
            self._save_token_weight(token, new_weight, success)

        pattern = utterance.category_pattern()
        old_combo = self._combo_weights.get(pattern, 1.0)
        new_combo = old_combo + learning_rate * (score - 0.5) * 2
        new_combo = max(0.3, min(2.5, new_combo))
        self._combo_weights[pattern] = new_combo
        self._save_combo_weight(pattern, new_combo, score)

        if success:
            self._successful_utterances += 1
            self._current_interval = max(
                self._min_interval,
                self._current_interval - timedelta(minutes=1),
            )
        else:
            self._current_interval = min(
                self._max_interval,
                self._current_interval + timedelta(minutes=2),
            )

        return {
            "score": score,
            "signals": signals,
            "success": success,
            "token_updates": {t: self._token_weights[t] for t in utterance.tokens},
            "combo_pattern": pattern,
        }

    def record_feedback(
        self,
        utterance: Utterance,
        response: str,
        response_length: int = None,
        explicit_positive: bool = False,
        explicit_negative: bool = False,
    ) -> Dict[str, Any]:
        """
        Record feedback on an utterance.

        Feedback signals:
        - Response length (longer = more engaged)
        - Explicit positive (user explicitly resonated)
        - Explicit negative (user was confused)
        - Questions in response (might indicate confusion)
        """
        signals = []
        score = 0.5  # Neutral baseline

        if response_length is None:
            response_length = len(response)

        # Response engagement
        if response_length > 300:
            score += 0.2
            signals.append("long_response")
        elif response_length > 150:
            score += 0.1
            signals.append("engaged")
        elif response_length < 30:
            score -= 0.1
            signals.append("short_response")

        # Explicit feedback
        if explicit_positive:
            score += 0.4
            signals.append("explicit_positive")
        if explicit_negative:
            score -= 0.3
            signals.append("explicit_negative")

        # Confusion markers in response
        confusion_markers = ["don't understand", "unclear", "what do you mean", "confused", "?"]
        confusion_count = sum(1 for m in confusion_markers if m.lower() in response.lower())
        if confusion_count > 1:
            score -= 0.15
            signals.append("confusion")

        # Clamp score
        score = max(0.0, min(1.0, score))

        result = self._record_direct_feedback(utterance, score, signals, learning_rate=0.12)
        # Override interval adjustment for human feedback (stronger effect)
        success = result["success"]
        if success:
            self._current_interval = max(
                self._min_interval,
                self._current_interval - timedelta(minutes=2),
            )
        else:
            self._current_interval = min(
                self._max_interval,
                self._current_interval + timedelta(minutes=5),
            )
        result["new_interval_minutes"] = self._current_interval.total_seconds() / 60
        return result

    def record_explicit_feedback(self, positive: bool):
        """
        Record explicit positive/negative feedback on last utterance.

        This is for the /resonate or /confused commands Gemini suggested.
        """
        if not self._recent:
            return None

        last = self._recent[-1]

        # Give strong feedback signal
        return self.record_feedback(
            last,
            response="",
            explicit_positive=positive,
            explicit_negative=not positive,
        )

    def record_implicit_feedback(
        self,
        utterance: 'Utterance',
        message_arrived: bool,
        delay_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Record implicit feedback based on whether a message arrived after utterance.

        A gentler learning signal than explicit feedback: if someone responds
        shortly after an utterance, that expression was engaging.

        Args:
            utterance: The utterance to score
            message_arrived: Whether a non-lumen message arrived
            delay_seconds: Seconds between utterance and message arrival

        Returns:
            Feedback dict if learning occurred, None if no signal
        """
        if not message_arrived or delay_seconds > 300:  # >5min = no signal
            # Mark as no response in history
            try:
                conn = self._connect()
                conn.execute(
                    "UPDATE primitive_history SET got_response = 0 WHERE timestamp = ?",
                    (utterance.timestamp.isoformat(),)
                )
                conn.commit()
            except Exception:
                pass
            return None

        # Score based on response delay
        if delay_seconds < 120:   # <2min
            score = 0.7
        elif delay_seconds < 300:  # 2-5min
            score = 0.6
        else:
            return None

        # Mark got_response in history
        try:
            conn = self._connect()
            conn.execute(
                "UPDATE primitive_history SET got_response = 1 WHERE timestamp = ?",
                (utterance.timestamp.isoformat(),)
            )
            conn.commit()
        except Exception:
            pass

        # Apply gentle weight update
        signals = ["implicit_response"]
        if delay_seconds < 120:
            signals.append("quick_response")

        return self._record_direct_feedback(
            utterance, score, signals, learning_rate=0.04
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the primitive language system."""
        conn = self._connect()

        # Count history entries
        total = conn.execute("SELECT COUNT(*) FROM primitive_history").fetchone()[0]
        scored = conn.execute("SELECT COUNT(*) FROM primitive_history WHERE score IS NOT NULL").fetchone()[0]
        avg_score = conn.execute("SELECT AVG(score) FROM primitive_history WHERE score IS NOT NULL").fetchone()[0]

        # Top patterns
        top_patterns = conn.execute("""
            SELECT pattern, avg_score, use_count
            FROM primitive_combo_weights
            ORDER BY avg_score DESC
            LIMIT 5
        """).fetchall()

        # Token weight summary
        token_summary = {}
        for row in conn.execute("SELECT token, weight, success_count, failure_count FROM primitive_token_weights"):
            token_summary[row["token"]] = {
                "weight": round(row["weight"], 3),
                "successes": row["success_count"],
                "failures": row["failure_count"],
            }

        # Response rate from implicit feedback
        response_rate = None
        try:
            responded = conn.execute(
                "SELECT COUNT(*) FROM primitive_history WHERE got_response = 1"
            ).fetchone()[0]
            tracked = conn.execute(
                "SELECT COUNT(*) FROM primitive_history WHERE got_response IS NOT NULL"
            ).fetchone()[0]
            if tracked > 0:
                response_rate = round(responded / tracked, 3)
        except Exception:
            pass

        return {
            "total_utterances": total,
            "scored_utterances": scored,
            "average_score": round(avg_score, 3) if avg_score else None,
            "success_rate": round(self._successful_utterances / max(1, self._total_utterances), 3),
            "response_rate": response_rate,
            "current_interval_minutes": round(self._current_interval.total_seconds() / 60, 1),
            "top_patterns": [
                {"pattern": r["pattern"], "avg_score": round(r["avg_score"], 3), "uses": r["use_count"]}
                for r in top_patterns
            ],
            "token_weights": token_summary,
            "recent_count": len(self._recent),
        }

    def get_recent_utterances(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent utterances with their scores."""
        return [
            {
                "text": u.text(),
                "tokens": u.tokens,
                "pattern": u.category_pattern(),
                "score": u.score,
                "timestamp": u.timestamp.isoformat(),
            }
            for u in self._recent[-count:]
        ]


# Singleton instance
_language_system: Optional[PrimitiveLanguageSystem] = None


def get_language_system(db_path: str = "anima.db") -> PrimitiveLanguageSystem:
    """Get or create the primitive language system."""
    global _language_system
    if _language_system is None:
        _language_system = PrimitiveLanguageSystem(db_path)
    return _language_system
