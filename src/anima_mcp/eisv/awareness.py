"""Trajectory awareness for Lumen's primitive language.

Maintains an in-memory ring buffer of recent anima states,
computes EISV trajectory classification, and provides
suggested tokens for the primitive language system.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .mapping import (
    anima_to_eisv,
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

        # Observability counters
        self._total_generations: int = 0
        self._total_feedback: int = 0
        self._coherence_sum: float = 0.0

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
                    buffer_size INTEGER
                )"""
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
                    cache_hit, buffer_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    ) -> None:
        """Record an anima state snapshot into the trajectory buffer.

        Only records if at least RECORD_INTERVAL seconds have elapsed
        since the last recording (subsampling to avoid overfilling buffer).
        """
        now = time.time()
        if now - self._last_record_time < self.RECORD_INTERVAL:
            return

        eisv = anima_to_eisv(warmth, clarity, stability, presence)
        eisv["t"] = now
        self._buffer.append(eisv)
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

            eisv = anima_to_eisv(
                warmth=rec.get("warmth", 0.5),
                clarity=rec.get("clarity", 0.5),
                stability=rec.get("stability", 0.5),
                presence=rec.get("presence", 0.0),
            )
            eisv["t"] = t
            self._buffer.append(eisv)
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

            result = {
                "shape": shape.value,
                "suggested_tokens": lumen_tokens,
                "eisv_tokens": eisv_tokens,
                "trigger": trigger,
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

    def record_feedback(self, tokens: List[str], score: float) -> None:
        """Forward feedback to the expression generator's weight learning."""
        if self._current_shape is not None:
            try:
                self._generator.update_weights(self._current_shape, tokens, score)
                self._total_feedback += 1
                self._coherence_sum += score
                self._log_event(
                    event_type="feedback",
                    shape=self._current_shape,
                    expression_tokens=tokens,
                    coherence_score=score,
                    buffer_size=len(self._buffer),
                )
            except Exception:
                pass

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

        # Expression generator stats
        mean_coherence: Optional[float] = None
        if self._total_feedback > 0:
            mean_coherence = self._coherence_sum / self._total_feedback

        # Recent events from DB
        recent_events: List[Dict[str, Any]] = []
        shape_distribution: Dict[str, int] = {}
        if self._db_conn is not None:
            try:
                cursor = self._db_conn.execute(
                    "SELECT id, timestamp, event_type, shape, eisv_state, "
                    "derivatives, suggested_tokens, expression_tokens, "
                    "coherence_score, cache_hit, buffer_size "
                    "FROM trajectory_events ORDER BY id DESC LIMIT 10"
                )
                cols = [d[0] for d in cursor.description]
                for row in cursor.fetchall():
                    recent_events.append(dict(zip(cols, row)))
                recent_events.reverse()  # chronological order

                dist_cursor = self._db_conn.execute(
                    "SELECT shape, COUNT(*) FROM trajectory_events "
                    "WHERE shape IS NOT NULL GROUP BY shape"
                )
                for shape_name, count in dist_cursor.fetchall():
                    shape_distribution[shape_name] = count
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
                "feedback_count": self._total_feedback,
                "mean_coherence": mean_coherence,
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


def compute_expression_coherence(
    suggested_tokens: Optional[List[str]],
    actual_tokens: List[str],
) -> Optional[float]:
    """Compute coherence between trajectory-suggested and actually-generated tokens."""
    if not suggested_tokens:
        return None
    overlap = set(suggested_tokens) & set(actual_tokens)
    return len(overlap) / max(len(suggested_tokens), 1)


def get_trajectory_awareness(**kwargs) -> TrajectoryAwareness:
    """Get or create the singleton TrajectoryAwareness instance."""
    global _awareness
    if _awareness is None:
        _awareness = TrajectoryAwareness(**kwargs)
    return _awareness
