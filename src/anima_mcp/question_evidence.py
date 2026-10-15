"""Evidence for the ``question_asking_tendency`` self-belief.

The belief is "I tend to ask questions when surprised". The unit of evidence
is one surprise episode, judged once: did that surprise lead to a question?

Before 2026-10-15 every posting and every miss was its own event. A post at
any surprise level counted as support, while a miss counted against the belief
only above 0.2 surprise and at half strength. On Lumen that ran 111:0. The
posts that supported it were mostly unsurprised exploration, so they said
nothing about surprise.

Now each producer reports what happened in an episode, and episodes are
five-minute wall-clock buckets (the same unit the self-model uses for
sensitivity beliefs). When a note arrives in a later bucket, the earlier
bucket is judged:

- no note in it above ``MEANINGFUL_SURPRISE`` means no evidence, because
  nothing was surprising;
- otherwise it **supports** the belief if a question was posted in the bucket
  and **contradicts** it if not, with strength equal to the bucket's peak
  surprise on both sides.

The pending bucket lives in memory. A restart loses at most one episode and
records nothing for it, which errs toward unknown.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

# The existing "meaningful surprise" cut both producers already used. It is
# grandfathered, not new: it now gates both sides instead of only misses.
MEANINGFUL_SURPRISE = 0.2
EPISODE_SECONDS = 300

BELIEF_ID = "question_asking_tendency"


def _enqueue(supports: bool, strength: float, source: str) -> None:
    from .learning_events import enqueue_self_belief_evidence

    enqueue_self_belief_evidence(
        BELIEF_ID, supports=supports, strength=strength, source=source,
    )


class QuestionEpisodes:
    """Collapse question attempts into one judgment per surprise episode."""

    def __init__(
        self,
        emit: Optional[Callable[[bool, float, str], None]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._emit = emit or _enqueue
        self._clock = clock
        self._lock = threading.Lock()
        self._bucket: Optional[int] = None
        self._peak = 0.0
        self._asked = False
        self._sources: set[str] = set()

    def note(self, surprise_level: float, *, asked: bool, source: str) -> None:
        """Record one question attempt (asked) or miss at this surprise level."""
        try:
            level = float(surprise_level)
        except (TypeError, ValueError):
            return
        if level != level:  # NaN carries no episode information
            return
        bucket = int(self._clock() // EPISODE_SECONDS)
        with self._lock:
            finished = None
            if self._bucket is not None and bucket != self._bucket:
                finished = (self._peak, self._asked, sorted(self._sources))
                self._bucket = None
            if self._bucket is None:
                self._bucket = bucket
                self._peak = 0.0
                self._asked = False
                self._sources = set()
            self._peak = max(self._peak, level)
            self._asked = self._asked or asked
            self._sources.add(source)
        if finished is not None:
            self._judge(*finished)

    def _judge(self, peak: float, asked: bool, sources: list[str]) -> None:
        if peak <= MEANINGFUL_SURPRISE:
            return
        outcome = "asked" if asked else "missed"
        try:
            self._emit(asked, min(1.0, peak), f"server:episode_{outcome}:{'+'.join(sources)}")
        except Exception as exc:
            # Audible: logger.debug has no handler in this codebase.
            print(f"[SelfModel] question episode enqueue error: {exc}",
                  file=sys.stderr, flush=True)


_episodes = QuestionEpisodes()


def note_question(surprise_level: float, *, asked: bool, source: str) -> None:
    """Module-level entry point used by the agency and surprise paths."""
    _episodes.note(surprise_level, asked=asked, source=source)
