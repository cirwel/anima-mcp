"""
Agency - Actions Lumen can take and exploration behavior.

Core philosophical insight: Agency requires action repertoire.
Without the ability to do things, there's no agency, only reaction.

This module gives Lumen choices it can make:
1. Adjust LED brightness (seek or avoid stimulation)
2. Request interaction (ask a question vs stay quiet)
3. Modulate attention (focus on one sensor more than others)
4. Adjust prediction confidence (be more or less cautious)
5. Exploration vs exploitation (try new things vs stick with known)

Key principle: Actions have consequences that Lumen experiences.
This creates a closed loop: action → consequence → learning → better action.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import deque
import json
import random
import math
import sqlite3
import sys
from .db_paths import resolve_db_path


AGENCY_REWARD_VERSION = 2


class ActionType(Enum):
    """Types of actions Lumen can take."""
    # Display actions
    LED_BRIGHTNESS = "led_brightness"  # Adjust LED brightness
    FACE_EXPRESSION = "face_expression"  # Change face display

    # Communication actions
    ASK_QUESTION = "ask_question"  # Generate a curiosity question
    STAY_QUIET = "stay_quiet"  # Suppress question asking
    SPEAK = "speak"  # Use voice if available

    # Internal actions
    FOCUS_ATTENTION = "focus_attention"  # Focus on specific sensor
    ADJUST_SENSITIVITY = "adjust_sensitivity"  # Change surprise threshold
    REQUEST_REFLECTION = "request_reflection"  # Trigger metacognitive reflection

    # Exploration actions
    EXPLORE = "explore"  # Try something new/unexpected


@dataclass
class Action:
    """An action Lumen can take."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    motivation: str = ""  # Why this action was chosen
    # None preserves compatibility for historical/offline callers that did
    # not report execution. The live runtime always records True or False.
    execution_succeeded: Optional[bool] = None
    execution_detail: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ActionOutcome:
    """The observed outcome of an action."""
    action: Action
    state_before: Dict[str, float]
    state_after: Dict[str, float]

    # Outcome metrics
    preference_satisfaction_change: float = 0.0  # Did preferences become more satisfied?
    surprise_after: float = 0.0  # How surprising was the result?
    goal_achieved: bool = False  # Did the action achieve its goal?

    # For learning
    reward: float = 0.0  # Computed reward signal


def build_agency_inputs(
    current_state: Dict[str, float],
    surprise_level: float,
    surprise_sources: List[str],
    preferences: Any,
    shm_data: Optional[dict] = None,
    self_model: Any = None,
    pathways: Any = None,
    *,
    can_focus: bool = False,
    can_reflect: bool = False,
    can_adjust: bool = True,
    can_led: bool = False,
    can_speak: bool = False,
) -> Dict[str, Any]:
    """Assemble the live selector inputs and capability boundary.

    This keeps the server's wiring inspectable: every learned input is sourced
    from the same broker snapshot and every candidate action is backed by an
    actuator available in the server process.
    """
    shm_data = shm_data if isinstance(shm_data, dict) else {}
    inner_life = shm_data.get("inner_life")
    raw_drives = inner_life.get("drives") if isinstance(inner_life, dict) else None
    drives: Dict[str, float] = {}
    if isinstance(raw_drives, dict):
        for dimension, value in raw_drives.items():
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                drives[str(dimension)] = max(0.0, min(1.0, float(value)))

    activity_data = shm_data.get("activity")
    activity = (
        str(activity_data.get("level", "unknown"))
        if isinstance(activity_data, dict) else "unknown"
    )

    self_predictions = None
    if self_model is not None and surprise_sources:
        context = None
        lowered_sources = [str(source).lower() for source in surprise_sources]
        if any("light" in source or "lux" in source for source in lowered_sources):
            context = "light_change"
        elif any("temp" in source for source in lowered_sources):
            context = "temp_change"
        elif current_state.get("stability", 1.0) < 0.3:
            context = "stability_drop"
        if context:
            try:
                self_predictions = self_model.predict_own_response(context)
            except Exception:
                self_predictions = None

    pathway_context = None
    pathway_strengths = None
    if pathways is not None:
        try:
            from .weighted_pathways import discretize_context

            satisfaction = preferences.get_overall_satisfaction(current_state)
            strongest_drive = max(drives.values()) if drives else 0.0
            pathway_context = discretize_context(
                surprise=surprise_level,
                satisfaction=satisfaction,
                drive=strongest_drive,
                activity=activity,
            )
            pathway_strengths = pathways.get_all_strengths(pathway_context)
        except Exception:
            pathway_context = None
            pathway_strengths = None

    available_actions = {ActionType.STAY_QUIET, ActionType.ASK_QUESTION}
    if can_focus and surprise_sources:
        available_actions.add(ActionType.FOCUS_ATTENTION)
    if can_reflect:
        available_actions.add(ActionType.REQUEST_REFLECTION)
    if can_adjust:
        available_actions.add(ActionType.ADJUST_SENSITIVITY)
    if can_led:
        available_actions.add(ActionType.LED_BRIGHTNESS)
    if can_speak:
        available_actions.add(ActionType.SPEAK)

    return {
        "preferences": preferences,
        "self_predictions": self_predictions,
        "drives": drives or None,
        "pathway_strengths": pathway_strengths,
        "available_actions": available_actions,
        "pathway_context": pathway_context,
        "activity": activity,
    }


def _log_store_binding(db_path: Path) -> None:
    """Announce which caller pinned this process's agency store (#123).

    ``get_action_selector()`` is first-call-wins, so whichever caller fires
    first decides where a process's TD-learning persists for its whole
    lifetime. That was silent, which is how the broker and the server ended up
    learning into two different databases without anyone noticing. Naming the
    resolved absolute path and the caller puts the race in the journal instead
    of leaving it to be found by diffing two SQLite files.
    """
    caller = "unknown"
    try:
        import traceback
        for frame in reversed(traceback.extract_stack()[:-2]):
            if Path(frame.filename).name != "agency.py":
                caller = f"{Path(frame.filename).name}:{frame.lineno}"
                break
    except Exception:
        pass
    try:
        resolved = db_path.resolve()
    except OSError:
        resolved = db_path
    print(f"[Agency] action store: {resolved} (bound by {caller})",
          file=sys.stderr, flush=True)


class ActionSelector:
    """
    Selects actions based on current state, preferences, and exploration.

    Uses a simple value-based approach:
    1. Each action has expected value based on past outcomes
    2. Exploration bonus for less-tried actions
    3. Preference satisfaction drives action choice
    """

    def __init__(self, db_path: str = "anima.db"):
        self._db_path = Path(resolve_db_path(db_path))
        self._conn: Optional[sqlite3.Connection] = None

        # Action value estimates (action_type -> expected reward)
        self._action_values: Dict[str, float] = {}

        # Action counts for exploration bonus
        self._action_counts: Dict[str, int] = {}

        # Recent action outcomes for learning
        self._outcome_history: deque = deque(maxlen=100)

        # Exploration parameters
        self._exploration_rate = 0.2  # Probability of exploring
        self._exploration_decay = 0.995  # How fast exploration decreases

        # Current focus (which sensor to pay attention to)
        self._attention_focus: Optional[str] = None
        self._sensitivity_modifier: float = 1.0  # Multiplier for surprise threshold

        # Load persisted state
        _log_store_binding(self._db_path)
        self._init_db()
        self._load_state()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with automatic reconnection on failure."""
        if self._conn is None:
            self._conn = self._create_connection()
        else:
            # Test if connection is still valid
            try:
                self._conn.execute("SELECT 1")
            except (sqlite3.Error, sqlite3.OperationalError):
                # Connection lost, recreate
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
                    import time
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        raise last_error or sqlite3.Error("Failed to connect to database")

    def _init_db(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agency_values (
                    action_key TEXT PRIMARY KEY,
                    value REAL DEFAULT 0.5,
                    count INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS agency_state (
                    key TEXT PRIMARY KEY,
                    data TEXT
                );
            """)
            conn.commit()
        except Exception as e:
            print(f"[Agency] DB init error (non-fatal): {e}", file=sys.stderr, flush=True)

    def _load_state(self):
        try:
            conn = self._get_conn()
            self._ensure_reward_model_version(conn)
            for row in conn.execute("SELECT action_key, value, count FROM agency_values"):
                self._action_values[row["action_key"]] = row["value"]
                self._action_counts[row["action_key"]] = row["count"]

            row = conn.execute("SELECT data FROM agency_state WHERE key = 'exploration_rate'").fetchone()
            if row:
                self._exploration_rate = float(json.loads(row["data"]))

            loaded = len(self._action_values)
            if loaded > 0:
                print(f"[Agency] Loaded {loaded} action values, exploration_rate={self._exploration_rate:.3f}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Agency] DB load error (non-fatal): {e}", file=sys.stderr, flush=True)

    def _ensure_reward_model_version(self, conn: sqlite3.Connection) -> None:
        """Preserve and reset values learned before execution was observable."""
        row = conn.execute(
            "SELECT data FROM agency_state WHERE key = 'reward_model_version'"
        ).fetchone()
        try:
            version = int(json.loads(row["data"])) if row else None
        except (TypeError, ValueError, json.JSONDecodeError):
            version = None

        legacy_rows = conn.execute(
            "SELECT action_key, value, count FROM agency_values ORDER BY action_key"
        ).fetchall()
        if version is None and not legacy_rows:
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("reward_model_version", json.dumps(AGENCY_REWARD_VERSION)),
            )
            conn.commit()
            return

        if version is None or version < AGENCY_REWARD_VERSION:
            exploration_row = conn.execute(
                "SELECT data FROM agency_state WHERE key = 'exploration_rate'"
            ).fetchone()
            try:
                legacy_exploration_rate = (
                    json.loads(exploration_row["data"]) if exploration_row else 0.2
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                legacy_exploration_rate = 0.2
            legacy = {
                "reward_model_version": version or 1,
                "action_values": [dict(item) for item in legacy_rows],
                "exploration_rate": legacy_exploration_rate,
            }
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("legacy_values_v1", json.dumps(legacy, sort_keys=True)),
            )
            conn.execute("DELETE FROM agency_values")
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("exploration_rate", json.dumps(0.2)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("reward_model_version", json.dumps(AGENCY_REWARD_VERSION)),
            )
            conn.commit()
            self._exploration_rate = 0.2
            print(
                f"[Agency] Preserved and reset {len(legacy_rows)} legacy action values "
                f"for reward model v{AGENCY_REWARD_VERSION}",
                file=sys.stderr,
                flush=True,
            )

    def _persist_action(self, action_key: str):
        try:
            conn = self._get_conn()
            value = self._action_values.get(action_key, 0.5)
            count = self._action_counts.get(action_key, 0)
            conn.execute(
                "INSERT OR REPLACE INTO agency_values (action_key, value, count) VALUES (?, ?, ?)",
                (action_key, value, count),
            )
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("exploration_rate", json.dumps(self._exploration_rate)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO agency_state (key, data) VALUES (?, ?)",
                ("reward_model_version", json.dumps(AGENCY_REWARD_VERSION)),
            )
            conn.commit()
        except Exception as e:
            print(f"[Agency] DB persist error (non-fatal): {e}", file=sys.stderr, flush=True)

    def select_action(
        self,
        current_state: Dict[str, float],
        preferences: Optional[Any] = None,  # PreferenceSystem
        surprise_level: float = 0.0,
        surprise_sources: Optional[List[str]] = None,
        can_speak: bool = False,
        self_predictions: Optional[Dict[str, float]] = None,
        conflict_rates: Optional[Dict[str, float]] = None,
        drives: Optional[Dict[str, float]] = None,
        pathway_strengths: Optional[Dict[str, float]] = None,
        available_actions: Optional[set[ActionType]] = None,
    ) -> Action:
        """
        Select an action based on current context.

        This is where agency lives: choosing what to do based on
        state, preferences, and learned values.
        """
        surprise_sources = surprise_sources or []
        effective_surprise = max(
            0.0, min(1.0, surprise_level * self._sensitivity_modifier)
        )
        if available_actions is None:
            available_actions = {
                ActionType.STAY_QUIET,
                ActionType.ASK_QUESTION,
                ActionType.FOCUS_ATTENTION,
                ActionType.ADJUST_SENSITIVITY,
                ActionType.REQUEST_REFLECTION,
                ActionType.LED_BRIGHTNESS,
            }
            if can_speak:
                available_actions.add(ActionType.SPEAK)
        else:
            available_actions = set(available_actions)
        # EXPLORE is a policy mode, not an executable physical action.
        available_actions.discard(ActionType.EXPLORE)

        # Build candidate actions with expected values
        candidates = []

        # 1. Question asking (communication)
        if effective_surprise > 0.2:
            question_value = self._get_action_value("ask_question")
            question_value += effective_surprise * 0.5  # Higher surprise → more likely to ask
            candidates.append((
                Action(
                    ActionType.ASK_QUESTION,
                    {"surprise_sources": surprise_sources},
                    motivation=f"Curious about {', '.join(surprise_sources)}",
                ),
                question_value,
            ))
        else:
            # Low surprise - might stay quiet
            quiet_value = self._get_action_value("stay_quiet")
            candidates.append((
                Action(ActionType.STAY_QUIET, motivation="Nothing surprising"),
                quiet_value,
            ))

        # 2. Attention focus
        if surprise_sources:
            # Focus on most surprising source
            focus_value = self._get_action_value("focus_attention")
            primary_source = surprise_sources[0] if surprise_sources else "general"
            candidates.append((
                Action(
                    ActionType.FOCUS_ATTENTION,
                    {"sensor": primary_source},
                    motivation=f"Focusing on {primary_source}",
                ),
                focus_value + 0.2,  # Slight bonus for focusing when surprised
            ))

        # 3. Sensitivity adjustment
        if preferences:
            satisfaction = preferences.get_overall_satisfaction(current_state)
            if satisfaction < 0.3:
                # Low satisfaction - maybe increase sensitivity
                candidates.append((
                    Action(
                        ActionType.ADJUST_SENSITIVITY,
                        {"direction": "increase"},
                        motivation="Low satisfaction, increasing sensitivity",
                    ),
                    self._get_action_value("adjust_sensitivity") + 0.3,
                ))
            elif satisfaction > 0.7:
                # High satisfaction - maybe decrease sensitivity
                candidates.append((
                    Action(
                        ActionType.ADJUST_SENSITIVITY,
                        {"direction": "decrease"},
                        motivation="High satisfaction, relaxing sensitivity",
                    ),
                    self._get_action_value("adjust_sensitivity") + 0.2,
                ))

        # 4. Exploration vs exploitation
        if random.random() < self._exploration_rate:
            # Explore: try something less common
            explore_action = self._select_exploration_action(
                current_state,
                available_actions=available_actions,
                surprise_sources=surprise_sources,
            )
            if explore_action is not None:
                candidates.append((explore_action, 1.0))  # High value for exploration
        else:
            # Exploit: choose based on learned values
            pass  # Use the candidates we've built

        # 5. Voice action (if available)
        if can_speak and effective_surprise > 0.4:
            speak_value = self._get_action_value("speak")
            candidates.append((
                Action(
                    ActionType.SPEAK,
                    {"trigger": "surprise"},
                    motivation="High surprise, expressing vocally",
                ),
                speak_value + effective_surprise * 0.3,
            ))

        # 6. LED brightness (if unsatisfied with warmth)
        if preferences and "warmth" in current_state:
            warmth = current_state["warmth"]
            direction = preferences.get_preferred_direction("warmth", warmth)
            if abs(direction) > 0.3:
                brightness_change = "increase" if direction > 0 else "decrease"
                candidates.append((
                    Action(
                        ActionType.LED_BRIGHTNESS,
                        {"direction": brightness_change},
                        motivation="Adjusting warmth expression",
                    ),
                    self._get_action_value("led_brightness") + abs(direction) * 0.4,
                ))

        # 6b. Drive-motivated actions (from inner life)
        if drives:
            strongest_dim = max(drives, key=drives.get)
            strongest_val = drives[strongest_dim]
            if strongest_val > 0.2:
                drive_actions = {
                    "warmth": (ActionType.LED_BRIGHTNESS, {"direction": "increase"}),
                    "clarity": (ActionType.REQUEST_REFLECTION, {}),
                    "stability": (ActionType.ADJUST_SENSITIVITY, {"direction": "decrease"}),
                    "presence": (ActionType.FACE_EXPRESSION, {"trigger": "drive"}),
                }
                if strongest_dim == "presence" and can_speak:
                    drive_actions["presence"] = (ActionType.SPEAK, {"trigger": "drive"})
                action_type, params = drive_actions[strongest_dim]
                candidates.append((
                    Action(
                        action_type, params,
                        motivation=f"drive: wanting {strongest_dim} ({strongest_val:.2f})",
                    ),
                    self._get_action_value(action_type.value) + strongest_val * 0.4,
                ))

        # The runtime declares what it can actually execute. Never learn from
        # a candidate that has no effect in the current process/capability set.
        candidates = [
            (action, value)
            for action, value in candidates
            if action.action_type in available_actions
        ]

        # 7. Prediction-informed adjustments
        if self_predictions and candidates:
            try:
                pred_surprise = self_predictions.get("surprise_likelihood", 0.5)
                pred_recovery = self_predictions.get("fast_recovery", 0.5)

                for i, (action, value) in enumerate(candidates):
                    boost = 0.0
                    if action.action_type == ActionType.ASK_QUESTION and pred_surprise > 0.6:
                        # High predicted surprise → more curious
                        boost = pred_surprise * 0.3
                        action.motivation += f" (predicted surprise: {pred_surprise:.1f})"
                    elif action.action_type == ActionType.STAY_QUIET and pred_surprise < 0.3:
                        # Low predicted surprise → more comfortable staying quiet
                        boost = 0.2
                    elif action.action_type == ActionType.FOCUS_ATTENTION and pred_recovery > 0.6:
                        # High predicted recovery → more willing to explore
                        boost = pred_recovery * 0.2
                        action.motivation += " (confident in recovery)"
                    if boost > 0:
                        candidates[i] = (action, value + boost)
            except Exception:
                pass  # Predictions are advisory, never block action selection

        # Select action with highest value (with some noise for stochasticity)
        if not candidates:
            fallback = (
                ActionType.STAY_QUIET
                if ActionType.STAY_QUIET in available_actions
                else min(available_actions, key=lambda item: item.value)
                if available_actions else ActionType.STAY_QUIET
            )
            return self._build_exploration_action(
                fallback, current_state, surprise_sources,
                motivation="No scored action available",
            )

        # Apply value tension discount: actions that frequently cause conflicts
        # between anima dimensions get their expected value reduced.
        # discount = 0.9^rate — e.g. 50% conflict rate => ~5% reduction, 100% => 10%
        if conflict_rates:
            for i, (action, value) in enumerate(candidates):
                action_key = action.action_type.value
                rate = conflict_rates.get(action_key, 0.0)
                if rate > 0:
                    candidates[i] = (action, value * (0.9 ** rate))

        # Apply experiential pathway strengths
        if pathway_strengths:
            for i, (action, value) in enumerate(candidates):
                action_key = action.action_type.value
                strength = pathway_strengths.get(action_key, 0.5)
                multiplier = max(0.25, min(4.0, strength / 0.5))
                candidates[i] = (action, value * multiplier)

        # Add noise for stochasticity
        noisy_candidates = [
            (action, value + random.gauss(0, 0.1))
            for action, value in candidates
        ]

        # Sort by value and pick best
        noisy_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = noisy_candidates[0][0]

        # Track action
        action_key = selected.action_type.value
        self._action_counts[action_key] = self._action_counts.get(action_key, 0) + 1

        return selected

    def _get_action_value(self, action_key: str) -> float:
        """Get expected value for an action, including exploration bonus."""
        base_value = self._action_values.get(action_key, 0.5)  # Default neutral value

        # Exploration bonus (UCB-style) — scales with log(total)/count for all actions
        count = self._action_counts.get(action_key, 0)
        total_count = sum(self._action_counts.values()) + 1
        exploration_bonus = math.sqrt(2 * math.log(total_count) / (count + 1))

        return base_value + exploration_bonus * self._exploration_rate

    def _select_exploration_action(
        self,
        current_state: Dict[str, float],
        available_actions: Optional[set[ActionType]] = None,
        surprise_sources: Optional[List[str]] = None,
    ) -> Optional[Action]:
        """Select an action for exploration (trying something new)."""
        surprise_sources = surprise_sources or []
        all_action_types = list(available_actions or {
            ActionType.STAY_QUIET,
            ActionType.ASK_QUESTION,
            ActionType.FOCUS_ATTENTION,
            ActionType.ADJUST_SENSITIVITY,
            ActionType.REQUEST_REFLECTION,
            ActionType.LED_BRIGHTNESS,
        })
        all_action_types = [
            action_type for action_type in all_action_types
            if action_type != ActionType.EXPLORE
            and (action_type != ActionType.FOCUS_ATTENTION or surprise_sources)
        ]
        if not all_action_types:
            return None

        # Find least-tried executable action.
        action_counts = [(a, self._action_counts.get(a.value, 0)) for a in all_action_types]
        action_counts.sort(key=lambda x: x[1])

        # Pick from least-tried actions
        least_tried = action_counts[:3]
        selected_type = random.choice(least_tried)[0]

        return self._build_exploration_action(
            selected_type,
            current_state,
            surprise_sources,
            motivation="Exploring an executable action",
        )

    def _build_exploration_action(
        self,
        action_type: ActionType,
        current_state: Dict[str, float],
        surprise_sources: List[str],
        motivation: str,
    ) -> Action:
        """Build valid parameters for an action chosen through exploration."""
        parameters: Dict[str, Any]
        if action_type == ActionType.ASK_QUESTION:
            parameters = {"surprise_sources": list(surprise_sources), "exploration": True}
        elif action_type == ActionType.FOCUS_ATTENTION:
            parameters = {"sensor": surprise_sources[0], "exploration": True}
        elif action_type == ActionType.ADJUST_SENSITIVITY:
            direction = "decrease" if self._sensitivity_modifier > 1.0 else "increase"
            parameters = {"direction": direction, "exploration": True}
        elif action_type == ActionType.LED_BRIGHTNESS:
            direction = "increase" if current_state.get("warmth", 0.5) < 0.5 else "decrease"
            parameters = {"direction": direction, "exploration": True}
        elif action_type == ActionType.SPEAK:
            parameters = {"trigger": "exploration"}
        else:
            parameters = {"exploration": True} if action_type != ActionType.STAY_QUIET else {}
        return Action(action_type, parameters, motivation=motivation)

    def record_outcome(
        self,
        action: Action,
        state_before: Dict[str, float],
        state_after: Dict[str, float],
        preference_satisfaction_before: float,
        preference_satisfaction_after: float,
        surprise_after: float,
        exploration_floor_reduction: float = 0.0,
    ) -> ActionOutcome:
        """
        Record the outcome of an action for learning.

        This is the critical learning signal: did the action help?
        exploration_floor_reduction: from experiential marks, lowers the min exploration rate.
        """
        outcome = ActionOutcome(
            action=action,
            state_before=state_before,
            state_after=state_after,
            preference_satisfaction_change=preference_satisfaction_after - preference_satisfaction_before,
            surprise_after=surprise_after,
        )

        executed = action.execution_succeeded is not False
        if not executed:
            # A selected action that did not happen has no causal claim on the
            # subsequent state. Give it an explicit, bounded failure signal.
            reward = -0.1
        else:
            # Preference satisfaction is the primary state-based reward.
            reward = outcome.preference_satisfaction_change * 2.0

            # Successful engagement can earn a small intrinsic bonus. Silence
            # remains a legitimate neutral action.
            engagement_actions = {
                ActionType.ASK_QUESTION,
                ActionType.FOCUS_ATTENTION,
                ActionType.SPEAK,
                ActionType.REQUEST_REFLECTION,
            }
            if action.action_type in engagement_actions:
                reward += 0.05

            baseline_surprise = state_before.get("last_surprise")
            has_baseline = (
                not isinstance(baseline_surprise, bool)
                and isinstance(baseline_surprise, (int, float))
                and math.isfinite(float(baseline_surprise))
            )

            if action.action_type == ActionType.ASK_QUESTION:
                # Questions are rewarded only when successfully emitted from
                # genuine surprise, not simply because the next tick surprised us.
                question_surprise = float(baseline_surprise) if has_baseline else surprise_after
                if question_surprise > 0.15:
                    reward += 0.2
                    outcome.goal_achieved = True

            elif action.action_type == ActionType.FOCUS_ATTENTION:
                # Without a measured before value, a later low surprise cannot
                # be attributed to focusing.
                if has_baseline and surprise_after < float(baseline_surprise):
                    reward += 0.3
                    outcome.goal_achieved = True

        outcome.reward = reward
        self._outcome_history.append(outcome)

        # Update action value estimate (simple TD learning)
        action_key = action.action_type.value
        old_value = self._action_values.get(action_key, 0.5)
        learning_rate = 0.1
        self._action_values[action_key] = old_value + learning_rate * (reward - old_value)

        # Adjust exploration rate: decay normally, but recover when surprised
        # High surprise signals environment change — explore more to adapt.
        if surprise_after > 0.3:
            self._exploration_rate = min(1.0, self._exploration_rate + 0.02 * surprise_after)
        else:
            self._exploration_rate *= self._exploration_decay
        floor = max(0.01, 0.05 - exploration_floor_reduction)
        self._exploration_rate = max(floor, min(1.0, self._exploration_rate))

        # Persist learned values
        self._persist_action(action_key)
        return outcome

    def get_attention_focus(self) -> Optional[str]:
        """Get current attention focus (which sensor to prioritize)."""
        return self._attention_focus

    def set_attention_focus(self, sensor: Optional[str]):
        """Set attention focus."""
        self._attention_focus = sensor

    def get_sensitivity_modifier(self) -> float:
        """Get sensitivity modifier for surprise threshold."""
        return self._sensitivity_modifier

    def adjust_sensitivity(self, direction: str):
        """Adjust sensitivity modifier."""
        if direction == "increase":
            self._sensitivity_modifier = min(2.0, self._sensitivity_modifier * 1.2)
        else:
            self._sensitivity_modifier = max(0.5, self._sensitivity_modifier * 0.8)

    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about actions."""
        return {
            "action_values": {k: round(v, 3) for k, v in self._action_values.items()},
            "action_counts": self._action_counts.copy(),
            "exploration_rate": round(self._exploration_rate, 3),
            "sensitivity_modifier": round(self._sensitivity_modifier, 3),
            "attention_focus": self._attention_focus,
            "recent_outcomes": len(self._outcome_history),
            "question_feedback_count": len(getattr(self, '_question_feedback', [])),
        }

    def record_question_feedback(self, question: str, feedback: dict):
        """
        Record feedback on a question Lumen asked.

        This is how Lumen learns which question patterns work:
        - High score = question got engaged, substantive response
        - Low score = question was confusing, incomplete, malformed

        Over time, patterns that get good feedback should be favored.
        """
        if not hasattr(self, '_question_feedback'):
            self._question_feedback = []

        self._question_feedback.append({
            "timestamp": datetime.now(),
            "question": question,
            "score": feedback["score"],
            "signals": feedback["signals"],
        })

        # Keep last 100 feedback entries
        if len(self._question_feedback) > 100:
            self._question_feedback = self._question_feedback[-100:]

        # Update ASK_QUESTION action value based on feedback
        # Good feedback reinforces question-asking, bad feedback weakens it
        current_value = self._action_values.get("ask_question", 0.5)
        learning_rate = 0.15
        reward = (feedback["score"] - 0.5) * 2  # Map 0-1 score to -1 to +1
        new_value = current_value + learning_rate * reward
        new_value = max(0.1, min(0.9, new_value))  # Clamp
        self._action_values["ask_question"] = new_value

        # Persist learned values
        self._persist_action("ask_question")

    def get_question_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of question feedback for analysis."""
        if not hasattr(self, '_question_feedback') or not self._question_feedback:
            return {"count": 0}

        recent = self._question_feedback[-20:]
        scores = [f["score"] for f in recent]
        avg_score = sum(scores) / len(scores)

        # Count signal types
        signal_counts = {}
        for f in recent:
            for s in f["signals"]:
                signal_counts[s] = signal_counts.get(s, 0) + 1

        return {
            "count": len(self._question_feedback),
            "recent_avg_score": round(avg_score, 3),
            "signal_counts": signal_counts,
            "ask_question_value": round(self._action_values.get("ask_question", 0.5), 3),
        }


class ExplorationManager:
    """
    Manages exploration behavior - trying new things vs sticking with known.

    Philosophy: Curiosity is not just noticing surprise, it's seeking novelty.
    True exploration means taking actions whose outcomes are uncertain.
    """

    def __init__(self):
        self._novelty_buffer: deque = deque(maxlen=100)  # Recent novel experiences
        self._exploration_history: deque = deque(maxlen=50)  # Recent exploration attempts
        self._last_exploration: Optional[datetime] = None
        self._exploration_cooldown = timedelta(seconds=30)

    def should_explore(self, current_state: Dict[str, float], surprise_level: float) -> Tuple[bool, str]:
        """
        Determine if now is a good time to explore.

        Returns (should_explore, reason).
        """
        now = datetime.now()

        # Cooldown check
        if self._last_exploration and now - self._last_exploration < self._exploration_cooldown:
            return False, "cooldown"

        # Explore when things are stable (can afford the risk)
        stability = current_state.get("stability", 0.5)
        if stability < 0.3:
            return False, "unstable"

        # Explore when not already surprised (seeking novelty, not overwhelmed)
        if surprise_level > 0.4:
            return False, "already_surprised"

        # Explore when bored (low surprise for a while)
        recent_surprises = [n.get("surprise", 0) for n in list(self._novelty_buffer)[-10:]]
        if recent_surprises and sum(recent_surprises) / len(recent_surprises) < 0.1:
            return True, "bored"

        # Random exploration with probability
        if random.random() < 0.1:
            return True, "random"

        return False, "no_reason"

    def record_exploration(self, action: Action, outcome: ActionOutcome):
        """Record an exploration attempt and its outcome."""
        self._exploration_history.append({
            "timestamp": datetime.now(),
            "action": action.action_type.value,
            "reward": outcome.reward,
            "goal_achieved": outcome.goal_achieved,
        })
        self._last_exploration = datetime.now()

    def record_novelty(self, novelty_level: float, source: str):
        """Record a novel experience."""
        self._novelty_buffer.append({
            "timestamp": datetime.now(),
            "novelty": novelty_level,
            "source": source,
            "surprise": novelty_level,  # For backward compat
        })

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        if not self._exploration_history:
            return {"explorations": 0}

        recent = list(self._exploration_history)[-20:]
        rewards = [e["reward"] for e in recent]
        successes = [e for e in recent if e["goal_achieved"]]

        return {
            "total_explorations": len(self._exploration_history),
            "recent_explorations": len(recent),
            "average_reward": sum(rewards) / len(rewards) if rewards else 0,
            "success_rate": len(successes) / len(recent) if recent else 0,
        }


# Singleton instances
_action_selector: Optional[ActionSelector] = None
_exploration_manager: Optional[ExplorationManager] = None


def get_action_selector(db_path: str = "anima.db") -> ActionSelector:
    """Get or create the action selector.

    First call wins for the life of the process. A bare call resolves through
    ``db_paths.resolve_db_path``, so it can no longer land on a cwd-relative
    file; the broker passes ``BROKER_AGENCY_DB`` explicitly to stay off the
    server's live value table. See #123.
    """
    global _action_selector
    if _action_selector is None:
        _action_selector = ActionSelector(db_path=db_path)
    return _action_selector


def get_exploration_manager() -> ExplorationManager:
    """Get or create the exploration manager."""
    global _exploration_manager
    if _exploration_manager is None:
        _exploration_manager = ExplorationManager()
    return _exploration_manager
