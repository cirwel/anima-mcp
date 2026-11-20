"""Trajectory awareness for Lumen's primitive language.

Maintains an in-memory ring buffer of recent anima states,
computes EISV trajectory classification, and provides
suggested tokens for the primitive language system.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .mapping import (
    anima_to_body_eisv_projection,
    compute_derivatives,
    compute_trajectory_window,
    classify_trajectory,
)
from .expression import (
    ExpressionGenerator,
    StudentExpressionGenerator,
    translate_expression,
    shape_to_lumen_trigger,
)


_DRIFT_DIMS = ("warmth", "clarity", "stability")


def _anima_state(
    warmth: float,
    clarity: float,
    stability: float,
) -> Dict[str, float]:
    """Normalize the raw anima dimensions that governance uses for drift."""
    state = {
        "warmth": float(warmth),
        "clarity": float(clarity),
        "stability": float(stability),
    }
    if any(not math.isfinite(value) for value in state.values()):
        raise ValueError("anima drift dimensions must be finite")
    return {
        dimension: max(0.0, min(1.0, value))
        for dimension, value in state.items()
    }


def _ethical_drift_magnitude(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
) -> float:
    """Collapse governance's clamped 3x drift vector to max magnitude."""
    if previous is None:
        return 0.0
    components = [
        max(-0.5, min(0.5, 3.0 * (current[dim] - previous[dim])))
        for dim in _DRIFT_DIMS
    ]
    return max(abs(component) for component in components)


def _stratum_recall_bounds(suggested_count: int, actual_count: int) -> Dict[str, Any]:
    """Structural anchor floor and achievable ceiling for a recall stratum."""
    if suggested_count > 0 and actual_count > 0:
        return {
            "anchored_floor": round(1.0 / suggested_count, 4),
            "achievable_ceiling": round(
                min(actual_count, suggested_count) / suggested_count, 4
            ),
            "informative": min(actual_count, suggested_count) > 1,
        }
    return {
        "anchored_floor": None,
        "achievable_ceiling": None,
        "informative": False,
    }


class TrajectoryAwareness:
    """EISV trajectory awareness for primitive language.

    Maintains an in-memory ring buffer of recent anima states,
    computes trajectory shapes, and suggests tokens for expressions.
    """

    # Minimum states needed for meaningful classification
    MIN_STATES = 5

    # Minimum seconds between buffer recordings (subsampling)
    RECORD_INTERVAL = 2.0

    def __init__(
        self,
        buffer_size: int = 30,
        cache_seconds: float = 60.0,
        seed: Optional[int] = None,
        db_path: Optional[str] = None,
        student_model_dir: Optional[str] = None,
    ):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._cache_seconds = cache_seconds

        # Use student model if available, fall back to rule-based
        if student_model_dir is not None:
            self._generator = StudentExpressionGenerator(
                model_dir=student_model_dir,
                fallback_seed=seed,
            )
        else:
            self._generator = ExpressionGenerator(seed=seed)
        self._use_student = isinstance(self._generator, StudentExpressionGenerator)

        # Cache
        self._cached_result: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0.0
        self._cache_buffer_len: int = 0

        # Tracking
        self._last_record_time: float = 0.0
        self._current_shape: Optional[str] = None
        self._last_anima_state: Optional[Dict[str, float]] = None

        # Process-lifetime observability counters. Suggestion recall is read from
        # persisted events when a database is configured (see ``get_state``).
        self._total_generations: int = 0
        self._eisv_weight_update_count: int = 0
        self._eisv_weight_matched_token_count: int = 0
        self._eisv_weight_positive_updates: int = 0
        self._eisv_weight_negative_updates: int = 0
        self._eisv_weight_net_reward: float = 0.0
        self._eisv_weight_skipped: Dict[str, int] = {
            "not_fallback_generated": 0,
            "degenerate_stratum": 0,
            "unknown_bounds": 0,
            "floor_violation": 0,
            "no_baseline": 0,
            "zero_delta": 0,
        }
        self._free_recall_baseline: Dict[tuple[int, int], tuple[int, float]] = {}
        self._suggestion_recall_stats: Dict[
            tuple[int, int], tuple[int, float]
        ] = {}

        # Persistence
        self._db_path = db_path
        self._db_conn: Optional[sqlite3.Connection] = None
        if db_path is not None:
            self._init_db()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    def _init_db(self) -> None:
        """Create the trajectory_events table if it doesn't exist."""
        try:
            self._db_conn = sqlite3.connect(self._db_path)
            self._db_conn.execute(
                """CREATE TABLE IF NOT EXISTS trajectory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    shape TEXT,
                    eisv_state TEXT,
                    derivatives TEXT,
                    suggested_tokens TEXT,
                    expression_tokens TEXT,
                    coherence_score REAL,
                    cache_hit INTEGER DEFAULT 0,
                    buffer_size INTEGER,
                    suggestion_recall REAL,
                    suggested_count INTEGER,
                    actual_count INTEGER,
                    eisv_weight_feedback_score REAL
                )"""
            )
            # Existing databases predate suggestion-recall telemetry. Append
            # nullable columns without renaming or rewriting legacy scores.
            existing_columns = {
                row[1]
                for row in self._db_conn.execute(
                    "PRAGMA table_info(trajectory_events)"
                ).fetchall()
            }
            for column, declaration in (
                ("suggestion_recall", "REAL"),
                ("suggested_count", "INTEGER"),
                ("actual_count", "INTEGER"),
                ("eisv_weight_feedback_score", "REAL"),
            ):
                if column not in existing_columns:
                    self._db_conn.execute(
                        f"ALTER TABLE trajectory_events "
                        f"ADD COLUMN {column} {declaration}"
                    )
            self._db_conn.commit()
        except Exception:
            self._db_conn = None

    def _log_event(self, **kwargs: Any) -> None:
        """Write a row to the trajectory_events table.

        Best-effort: never raises.  All dict values are JSON-serialized.
        """
        if self._db_conn is None:
            return
        try:
            def _ser(v: Any) -> Any:
                if isinstance(v, dict) or isinstance(v, list):
                    return json.dumps(v)
                return v

            now_iso = datetime.now(timezone.utc).isoformat()
            self._db_conn.execute(
                """INSERT INTO trajectory_events
                   (timestamp, event_type, shape, eisv_state, derivatives,
                    suggested_tokens, expression_tokens, coherence_score,
                    cache_hit, buffer_size, suggestion_recall,
                    suggested_count, actual_count, eisv_weight_feedback_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now_iso,
                    kwargs.get("event_type", "unknown"),
                    kwargs.get("shape"),
                    _ser(kwargs.get("eisv_state")),
                    _ser(kwargs.get("derivatives")),
                    _ser(kwargs.get("suggested_tokens")),
                    _ser(kwargs.get("expression_tokens")),
                    kwargs.get("coherence_score"),
                    1 if kwargs.get("cache_hit") else 0,
                    kwargs.get("buffer_size", len(self._buffer)),
                    kwargs.get("suggestion_recall"),
                    kwargs.get("suggested_count"),
                    kwargs.get("actual_count"),
                    kwargs.get("eisv_weight_feedback_score"),
                ),
            )
            self._db_conn.commit()
        except Exception:
            pass

    def record_state(
        self,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        body_eisv_projection: Optional[Dict[str, float]] = None,
        eisv: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a body-projection snapshot into the trajectory buffer.

        Only records if at least RECORD_INTERVAL seconds have elapsed
        since the last recording (subsampling to avoid overfilling buffer).
        ``eisv`` is the deprecated keyword alias retained for callers written
        before the state-space provenance split.
        """
        now = time.time()
        if now - self._last_record_time < self.RECORD_INTERVAL:
            return

        if body_eisv_projection is not None and eisv is not None:
            raise ValueError(
                "pass body_eisv_projection or legacy eisv, not both"
            )
        supplied = body_eisv_projection if body_eisv_projection is not None else eisv
        snapshot = (
            dict(supplied)
            if supplied is not None
            else anima_to_body_eisv_projection(
                warmth, clarity, stability, presence
            )
        )
        for dimension in ("E", "I", "S", "V"):
            value = snapshot.get(dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"EISV {dimension} must be numeric")
            value = float(value)
            lower = -1.0 if dimension == "V" else 0.0
            if not math.isfinite(value) or not lower <= value <= 1.0:
                raise ValueError(f"EISV {dimension} outside valid range")
            snapshot[dimension] = value
        current_anima = _anima_state(warmth, clarity, stability)
        snapshot["ethical_drift"] = _ethical_drift_magnitude(
            current_anima,
            self._last_anima_state,
        )
        snapshot["t"] = now
        self._buffer.append(snapshot)
        self._last_anima_state = current_anima
        self._last_record_time = now

    def bootstrap_from_history(self, state_records: List[Dict]) -> int:
        """Pre-fill buffer from historical state_history records.

        Parameters
        ----------
        state_records:
            List of dicts with 'timestamp' (ISO string), 'warmth', 'clarity',
            'stability', 'presence' keys. Should be in chronological order.

        Returns number of records added to buffer.
        """
        from datetime import datetime, timezone

        added = 0
        for rec in state_records:
            ts_str = rec.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                t = dt.timestamp()
            except (ValueError, TypeError):
                continue

            warmth = rec.get("warmth", 0.5)
            clarity = rec.get("clarity", 0.5)
            stability = rec.get("stability", 0.5)
            current_anima = _anima_state(warmth, clarity, stability)
            body_projection = anima_to_body_eisv_projection(
                warmth=warmth,
                clarity=clarity,
                stability=stability,
                presence=rec.get("presence", 0.0),
            )
            body_projection["ethical_drift"] = _ethical_drift_magnitude(
                current_anima,
                self._last_anima_state,
            )
            body_projection["t"] = t
            self._buffer.append(body_projection)
            self._last_anima_state = current_anima
            added += 1

        if added > 0:
            self._last_record_time = time.time()
        return added

    def get_trajectory_suggestion(
        self,
        lang_state: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get trajectory-aware token suggestions.

        Returns None if insufficient data or on error.
        Otherwise returns dict with:
            shape, suggested_tokens, eisv_tokens, trigger
        """
        if len(self._buffer) < self.MIN_STATES:
            return None

        # Check cache
        now = time.time()
        if (
            self._cached_result is not None
            and (now - self._cache_time) < self._cache_seconds
            and len(self._buffer) == self._cache_buffer_len
        ):
            return self._cached_result

        try:
            states = list(self._buffer)
            window = compute_trajectory_window(states)
            shape = classify_trajectory(window)
            self._current_shape = shape.value

            if self._use_student:
                eisv_tokens = self._generator.generate(shape.value, window=window)
            else:
                eisv_tokens = self._generator.generate(shape.value)
            lumen_tokens = translate_expression(eisv_tokens)
            trigger = shape_to_lumen_trigger(shape.value)

            student_active = bool(
                self._use_student and getattr(self._generator, "is_loaded", False)
            )
            result = {
                "shape": shape.value,
                "suggested_tokens": lumen_tokens,
                "eisv_tokens": eisv_tokens,
                "trigger": trigger,
                "generator_source": "student" if student_active else "fallback",
            }

            self._cached_result = result
            self._cache_time = now
            self._cache_buffer_len = len(self._buffer)

            # Observability: count and log fresh classification
            self._total_generations += 1
            last_state = states[-1] if states else None
            eisv_snapshot = (
                {k: last_state[k] for k in ("E", "I", "S", "V")}
                if last_state
                else None
            )
            self._log_event(
                event_type="classification",
                shape=shape.value,
                eisv_state=eisv_snapshot,
                suggested_tokens=lumen_tokens,
                expression_tokens=eisv_tokens,
                buffer_size=len(self._buffer),
            )

            return result

        except Exception:
            return None

    def record_suggestion_recall(
        self,
        suggested_tokens: Optional[List[str]],
        actual_tokens: List[str],
        *,
        shape: Optional[str] = None,
    ) -> Optional[float]:
        """Persist one typed suggestion-recall observation.

        This is telemetry only. It deliberately does not mutate expression
        weights; callers that intend learning must make that separate action
        explicit through :meth:`record_eisv_weight_feedback`.
        """
        score = compute_suggestion_recall(suggested_tokens, actual_tokens)
        if score is None:
            return None

        suggested_count = len(suggested_tokens)
        actual_count = len(actual_tokens)
        stratum = (suggested_count, actual_count)
        observation_count, recall_sum = self._suggestion_recall_stats.get(
            stratum, (0, 0.0)
        )
        self._suggestion_recall_stats[stratum] = (
            observation_count + 1,
            recall_sum + score,
        )
        self._log_event(
            event_type="suggestion",
            shape=shape if shape is not None else self._current_shape,
            suggested_tokens=suggested_tokens,
            expression_tokens=actual_tokens,
            suggestion_recall=score,
            suggested_count=suggested_count,
            actual_count=actual_count,
            buffer_size=len(self._buffer),
        )
        return score

    def record_eisv_weight_feedback(
        self,
        eisv_tokens: List[str],
        score: float,
        *,
        suggested_count: Optional[int] = None,
        actual_count: Optional[int] = None,
        generator_source: Optional[str] = None,
        anchored_recall: bool = False,
    ) -> int:
        """Apply feedback to matching EISV tokens and return updated token count.

        When ``anchored_recall`` is True, guards against structural ratchets:
        skips student-generated output, skips degenerate strata where
        ``min(actual_count, suggested_count) <= 1`` (achievable ceiling ==
        anchored floor), treats scores below the anchored floor as anomalies,
        and centers non-anchor recall on its pre-update per-stratum mean.
        """
        if self._current_shape is None:
            return 0
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            if anchored_recall:
                self._eisv_weight_skipped["unknown_bounds"] += 1
            return 0
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            if anchored_recall:
                self._eisv_weight_skipped["unknown_bounds"] += 1
            return 0

        if anchored_recall:
            student_active = bool(
                self._use_student and getattr(self._generator, "is_loaded", False)
            )
            if generator_source == "student" or (
                generator_source is None and student_active
            ):
                self._eisv_weight_skipped["not_fallback_generated"] += 1
                return 0
            if (
                suggested_count is None
                or actual_count is None
                or isinstance(suggested_count, bool)
                or isinstance(actual_count, bool)
                or suggested_count <= 0
                or actual_count <= 0
            ):
                self._eisv_weight_skipped["unknown_bounds"] += 1
                return 0

            anchored_floor = 1.0 / suggested_count
            achievable_ceiling = min(actual_count, suggested_count) / suggested_count
            if min(actual_count, suggested_count) <= 1 or achievable_ceiling <= anchored_floor:
                self._eisv_weight_skipped["degenerate_stratum"] += 1
                return 0
            if numeric_score < anchored_floor - 1e-9:
                self._eisv_weight_skipped["floor_violation"] += 1
                return 0

            free_recall = max(
                0.0,
                min(
                    1.0,
                    (numeric_score - anchored_floor)
                    / (achievable_ceiling - anchored_floor),
                ),
            )
            stratum = (suggested_count, actual_count)
            prev_n, prev_mean = self._free_recall_baseline.get(stratum, (0, 0.0))
            new_n = prev_n + 1
            new_mean = prev_mean + (free_recall - prev_mean) / new_n
            self._free_recall_baseline[stratum] = (new_n, new_mean)
            if prev_n == 0:
                self._eisv_weight_skipped["no_baseline"] += 1
                return 0
            effective_reward = free_recall - prev_mean
            if abs(effective_reward) < 1e-9:
                self._eisv_weight_skipped["zero_delta"] += 1
                return 0
            logged_score = round(0.5 + 0.5 * effective_reward, 6)
        else:
            effective_reward = (numeric_score - 0.5) * 2.0
            if abs(effective_reward) < 1e-9:
                self._eisv_weight_skipped["zero_delta"] += 1
                return 0
            logged_score = numeric_score

        try:
            matched_count = self._generator.update_weights(
                self._current_shape,
                eisv_tokens,
                numeric_score,
                reward=effective_reward,
            )
        except Exception:
            return 0
        if matched_count <= 0:
            shape_weights = self._generator.get_weights(self._current_shape)
            if any(token in shape_weights for token in eisv_tokens):
                self._eisv_weight_skipped["zero_delta"] += 1
            return 0

        self._eisv_weight_update_count += 1
        self._eisv_weight_matched_token_count += matched_count
        if effective_reward > 0:
            self._eisv_weight_positive_updates += 1
        else:
            self._eisv_weight_negative_updates += 1
        self._eisv_weight_net_reward += effective_reward
        self._log_event(
            event_type="eisv_weight_update",
            shape=self._current_shape,
            expression_tokens=eisv_tokens,
            eisv_weight_feedback_score=logged_score,
            buffer_size=len(self._buffer),
        )
        return matched_count

    def record_feedback(self, tokens: List[str], score: float) -> int:
        """Deprecated compatibility wrapper for EISV weight feedback."""
        return self.record_eisv_weight_feedback(tokens, score)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of the awareness subsystem."""
        buf = list(self._buffer)
        buf_size = len(buf)
        buf_capacity = self._buffer.maxlen or 0

        # Current EISV from last buffer entry
        current_eisv: Optional[Dict[str, float]] = None
        if buf:
            last = buf[-1]
            current_eisv = {k: last[k] for k in ("E", "I", "S", "V")}

        # Derivatives from buffer
        derivatives: Optional[Dict[str, float]] = None
        if len(buf) >= 2:
            derivs = compute_derivatives(buf)
            if derivs:
                last_d = derivs[-1]
                derivatives = {k: last_d[k] for k in ("dE", "dI", "dS", "dV")}

        # Window seconds
        window_seconds: float = 0.0
        if len(buf) >= 2:
            window_seconds = buf[-1]["t"] - buf[0]["t"]

        # Cache info
        cache_shape: Optional[str] = None
        cache_age: float = 0.0
        if self._cached_result is not None:
            cache_shape = self._cached_result.get("shape")
            cache_age = time.time() - self._cache_time

        # In-memory fallback is explicitly process-scoped. With persistence,
        # the database is the source of truth so restarts do not reset recall.
        suggestion_recall_scope = "process_lifetime"
        suggestion_recall_strata = [
            {
                "suggested_count": suggested_count,
                "actual_count": actual_count,
                "observation_count": observation_count,
                "mean_suggestion_recall": recall_sum / observation_count,
                **_stratum_recall_bounds(suggested_count, actual_count),
            }
            for (suggested_count, actual_count), (
                observation_count,
                recall_sum,
            ) in sorted(self._suggestion_recall_stats.items())
        ]

        # Recent events from DB
        recent_events: List[Dict[str, Any]] = []
        shape_distribution: Dict[str, int] = {}
        if self._db_conn is not None:
            try:
                cursor = self._db_conn.execute(
                    "SELECT id, timestamp, event_type, shape, eisv_state, "
                    "derivatives, suggested_tokens, expression_tokens, "
                    "coherence_score AS legacy_mixed_score, cache_hit, "
                    "buffer_size, suggestion_recall, suggested_count, "
                    "actual_count, eisv_weight_feedback_score "
                    "FROM trajectory_events ORDER BY id DESC LIMIT 10"
                )
                cols = [d[0] for d in cursor.description]
                for row in cursor.fetchall():
                    event = dict(zip(cols, row))
                    legacy_score_kind = None
                    if event["legacy_mixed_score"] is not None:
                        if event["event_type"] == "suggestion":
                            legacy_score_kind = "suggestion_recall"
                        elif event["event_type"] == "feedback":
                            legacy_score_kind = "untyped_feedback_score"
                        else:
                            legacy_score_kind = "unknown"
                    event["legacy_score_kind"] = legacy_score_kind
                    recent_events.append(event)
                recent_events.reverse()  # chronological order

                dist_cursor = self._db_conn.execute(
                    "SELECT shape, COUNT(*) FROM trajectory_events "
                    "WHERE shape IS NOT NULL GROUP BY shape"
                )
                for shape_name, count in dist_cursor.fetchall():
                    shape_distribution[shape_name] = count

                recall_cursor = self._db_conn.execute(
                    "SELECT suggested_count, actual_count, COUNT(*), "
                    "AVG(suggestion_recall) FROM trajectory_events "
                    "WHERE event_type = 'suggestion' "
                    "AND suggestion_recall IS NOT NULL "
                    "AND suggested_count IS NOT NULL "
                    "AND actual_count IS NOT NULL "
                    "GROUP BY suggested_count, actual_count "
                    "ORDER BY suggested_count, actual_count"
                )
                suggestion_recall_strata = [
                    {
                        "suggested_count": suggested_count,
                        "actual_count": actual_count,
                        "observation_count": observation_count,
                        "mean_suggestion_recall": mean_recall,
                        **_stratum_recall_bounds(suggested_count, actual_count),
                    }
                    for (
                        suggested_count,
                        actual_count,
                        observation_count,
                        mean_recall,
                    ) in recall_cursor.fetchall()
                ]
                suggestion_recall_scope = "persisted_history"
            except Exception:
                pass

        return {
            "current_shape": self._current_shape,
            "current_eisv": current_eisv,
            "derivatives": derivatives,
            "buffer": {
                "size": buf_size,
                "capacity": buf_capacity,
                "window_seconds": window_seconds,
            },
            "cache": {
                "shape": cache_shape,
                "age_seconds": cache_age,
                "ttl_seconds": self._cache_seconds,
            },
            "expression_generator": {
                "total_generations": self._total_generations,
                "total_generations_scope": "process_lifetime",
                "suggestion_recall": {
                    "scope": suggestion_recall_scope,
                    "strata": suggestion_recall_strata,
                },
                "eisv_weight_updates": {
                    "scope": "process_lifetime",
                    "persisted": False,
                    "fallback_only": True,
                    "student_active": bool(
                        self._use_student
                        and getattr(self._generator, "is_loaded", False)
                    ),
                    "update_count": self._eisv_weight_update_count,
                    "matched_token_count": self._eisv_weight_matched_token_count,
                    "positive_updates": self._eisv_weight_positive_updates,
                    "negative_updates": self._eisv_weight_negative_updates,
                    "net_reward": round(self._eisv_weight_net_reward, 4)
                    if self._eisv_weight_update_count > 0
                    else None,
                    "skipped": dict(self._eisv_weight_skipped),
                },
            },
            "recent_events": recent_events,
            "shape_distribution": shape_distribution,
        }

    @property
    def current_shape(self) -> Optional[str]:
        """Last classified trajectory shape, or None."""
        return self._current_shape

    @property
    def buffer_size(self) -> int:
        """Number of states currently in the buffer."""
        return len(self._buffer)


# Singleton
_awareness: Optional[TrajectoryAwareness] = None


def compute_suggestion_recall(
    suggested_tokens: Optional[List[str]],
    actual_tokens: List[str],
) -> Optional[float]:
    """Return the fraction of suggested tokens present in the output."""
    if not suggested_tokens:
        return None
    overlap = set(suggested_tokens) & set(actual_tokens)
    return len(overlap) / max(len(suggested_tokens), 1)


def compute_expression_coherence(
    suggested_tokens: Optional[List[str]],
    actual_tokens: List[str],
) -> Optional[float]:
    """Deprecated compatibility alias for :func:`compute_suggestion_recall`."""
    return compute_suggestion_recall(suggested_tokens, actual_tokens)


def get_trajectory_awareness(**kwargs) -> TrajectoryAwareness:
    """Get or create the singleton TrajectoryAwareness instance."""
    global _awareness
    if _awareness is None:
        _awareness = TrajectoryAwareness(**kwargs)
    return _awareness
