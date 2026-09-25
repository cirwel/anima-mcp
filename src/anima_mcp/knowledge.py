"""
Knowledge Base - Lumen's learned insights from Q&A interactions.

When agents answer Lumen's questions, key insights are extracted and stored.
These insights persist across restarts and influence future reflections.
"""

import json
import math
import re
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .atomic_write import atomic_json_write

# Persisted store schema version. Bump when a one-time on-load migration is
# needed (see KnowledgeBase._migrate_schema). v2: log-compress legacy
# reference counts onto the honest occasion-gated scale.
KNOWLEDGE_SCHEMA_VERSION = 3


def _get_knowledge_path() -> Path:
    """Get persistent path for knowledge - survives reboots."""
    anima_dir = Path.home() / ".anima"
    anima_dir.mkdir(exist_ok=True)
    return anima_dir / "knowledge.json"


@dataclass
class Insight:
    """A learned insight from Q&A."""
    insight_id: str  # Unique ID
    text: str  # The insight itself
    source_question: str  # What Lumen asked
    source_answer: str  # What was answered
    source_author: str  # Who answered (claude, user, etc.)
    timestamp: float  # When learned
    category: str = "general"  # Category: self, world, relationships, sensations, existence
    confidence: float = 1.0  # How confident (can decay over time or with contradictions)
    references: int = 0  # Count of INDEPENDENT re-derivations (the conviction signal).
    # Wall-clock of the last credited independent re-derivation. 0.0 means
    # "never re-derived"; reconvergence gating falls back to ``timestamp``.
    last_reconverged_at: float = 0.0
    # Provenance: the distinct source questions that independently re-derived
    # this belief. Length tracks ``references`` for rows credited going
    # forward (legacy rows may have references without provenance). This is
    # how forgetting stays "collapse for surfacing, never forget the record".
    derived_from: List[str] = field(default_factory=list)
    # Down-path / contradiction visibility: ids of insights that assert the
    # OPPOSITE claim (detected by the negation guard at add time). Makes
    # contradictions structural rather than silently stored side-by-side, and
    # is the trigger that lets confidence FALL — confidence is not a one-way
    # ratchet that can only ever strengthen with repetition.
    contradicted_by: List[str] = field(default_factory=list)
    # Occasion (MCP session id) of the last credited re-derivation. Credit is
    # gated on a NEW occasion, not a wall-clock window, so the conviction
    # signal is independent of cron cadence: one answering session credits a
    # belief at most once no matter how many questions it answers or how long
    # it takes.
    last_reconverged_occasion: str = ""
    # Original reference count before the one-time log-compression migration
    # (legacy counts were minted under looser, ungated rules). None = minted
    # under the honest signal / never migrated.
    legacy_references: Optional[int] = None
    # Original confidence before the one-time v3 rescale of grandfathered
    # external rows (pre-trust-boundary externals were born at 1.0; the v3
    # migration returns unearned external confidence to the 0.5 entry point).
    # None = never rescaled. Keeps the migration auditable/reversible, same
    # contract as legacy_references.
    legacy_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Insight":
        # Handle missing fields for backwards compatibility
        if "references" not in d:
            d["references"] = 0
        if "confidence" not in d:
            d["confidence"] = 1.0
        if "category" not in d:
            d["category"] = "general"
        if "last_reconverged_at" not in d:
            d["last_reconverged_at"] = 0.0
        if "derived_from" not in d:
            d["derived_from"] = []
        if "contradicted_by" not in d:
            d["contradicted_by"] = []
        if "last_reconverged_occasion" not in d:
            d["last_reconverged_occasion"] = ""
        if "legacy_references" not in d:
            d["legacy_references"] = None
        if "legacy_confidence" not in d:
            d["legacy_confidence"] = None
        # Tolerate unknown future fields rather than crashing on load.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def conviction_score(self, now: Optional[float] = None) -> float:
        """Soft conviction signal used only to RANK surfacing — never to
        protect or evict.

        Dominated by ``references`` (independent re-derivations), with
        confidence as a secondary term and recency as a small tiebreaker
        (both bounded < 1 so a single genuine re-derivation always outranks
        any never-re-derived insight). There is no protected tier: a wrong
        belief can score high here, but that only affects what surfaces and
        is fully reversible — it can never become a permanent conviction.
        """
        if now is None:
            now = time.time()
        recency_ref = self.last_reconverged_at or self.timestamp
        age_days = max(0.0, (now - recency_ref) / 86400.0)
        recency_tiebreak = 0.5 / (1.0 + age_days)  # (0, 0.5], newer → higher
        return self.references + 0.5 * self.confidence + recency_tiebreak

    def age_str(self) -> str:
        """Human-readable age string."""
        age_seconds = time.time() - self.timestamp
        if age_seconds < 3600:
            mins = int(age_seconds / 60)
            return f"{mins}m ago"
        elif age_seconds < 86400:
            hours = int(age_seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(age_seconds / 86400)
            return f"{days}d ago"


class KnowledgeBase:
    """Lumen's accumulated knowledge from Q&A interactions."""

    # Retention is effectively unbounded — storage is trivial on the Pi and
    # this store is identity-bearing, so we do not routinely forget genuine
    # history. This high ceiling is a pure safety valve that should almost
    # never fire; selectivity lives in the SURFACING layer (conviction_score),
    # not in eviction. When it does fire, the recency reserve + conviction
    # ranking trim the least-conviction, never-re-derived, oldest rows.
    MAX_INSIGHTS = 10000

    # An independent re-derivation must come from a genuinely different
    # OCCASION (MCP session). Gating on occasion rather than a wall-clock
    # window keeps the conviction signal independent of cron cadence: one
    # answering session credits a belief at most once, no matter how many
    # questions it answers or how long the batch takes. (Earlier this used a
    # 30-minute window, which silently coupled correctness to the cron
    # schedule — change the cadence and the gate's meaning changed with it.)
    RECONVERGENCE_CONFIDENCE_BOOST = 0.05

    # Down-path: when a new insight contradicts a near-identical existing one
    # (negation guard fires), both sides lose certainty. Reversible and
    # bounded — never zero — so a contested belief is demoted in surfacing,
    # not erased. A single contradiction should not flatten a belief that was
    # independently re-derived many times; ``references`` still carries weight.
    CONTRADICTION_CONFIDENCE_PENALTY = 0.15
    MIN_CONFIDENCE = 0.1

    def __init__(self):
        self._knowledge_file = _get_knowledge_path()
        self._insights: List[Insight] = []
        self._load()

    def _load(self):
        """Load insights from persistent storage."""
        try:
            if self._knowledge_file.exists():
                data = json.loads(self._knowledge_file.read_text())
                self._insights = [Insight.from_dict(i) for i in data.get("insights", [])]
                loaded_version = data.get("schema_version", 0)
                if loaded_version < KNOWLEDGE_SCHEMA_VERSION:
                    self._migrate_schema(loaded_version)
            else:
                self._insights = []
        except Exception as e:
            print(f"[Knowledge] Load error: {e}", file=sys.stderr, flush=True)
            self._insights = []

    def _save(self):
        """Save insights to persistent storage."""
        try:
            data = {
                "schema_version": KNOWLEDGE_SCHEMA_VERSION,
                "insights": [i.to_dict() for i in self._insights],
            }
            atomic_json_write(self._knowledge_file, data, indent=2)
        except Exception as e:
            print(f"[Knowledge] Save error: {e}", file=sys.stderr, flush=True)

    def _migrate_schema(self, from_version: int):
        """One-time, idempotent migrations to KNOWLEDGE_SCHEMA_VERSION.

        Each block is gated on ``from_version`` so bumping the schema never
        re-applies an earlier migration to rows minted honestly after it (a
        post-v2 row with references > 1 earned them under the gated logic and
        must not be re-compressed).

        v2 — log-compress legacy reference counts. Counts minted under the old
        ungated logic (every near-duplicate +1, no occasion gate) reached the
        thousands, which would dominate conviction ranking forever and starve
        new, honestly-gated re-derivations. ``round(log2(refs+1))`` brings them
        onto the new scale while preserving their relative ORDER (a belief
        re-stated 1180x still outranks one re-stated 40x); the original is kept
        in ``legacy_references`` so the migration is auditable/reversible.

        v3 — rescale grandfathered external confidence (operator-authorized
        2026-08-21). Pre-trust-boundary external rows were born at confidence
        1.0; since 2026-08-11 external claims enter at 0.5 and EARN their way
        up through independent re-derivation. The v3 rule returns every
        unearned external row to that entry point: external author (INCLUDING
        operator-authored prose — authorship is not an exemption, #121) +
        confidence above 0.5 + never reconverged. The predicate is
        principle-gated, not date-gated, on purpose: a post-boundary row
        minted with an explicit confidence override above 0.5 is equally
        unearned. Rows that earned boosts through reconvergence keep them;
        self-derived (lumen) rows are untouched; nothing is deleted — the
        original confidence is kept in ``legacy_confidence`` and the whole
        pre-migration file is copied to a ``.pre-v3.json`` sidecar first.

        Measured on the live store before shipping (simulated 2026-08-21):
        1,778 of 2,192 rows rescale; 58 exempt via reconvergence; 354 already
        at/below 0.5; 2 lumen rows untouched. Downstream, rows above the 0.6
        actionable floor drop from 1,837 to 59 — that collapse is the
        INTENDED effect (the trust boundary finally applying to the
        grandfathered corpus), stated here so nobody reads it as a
        regression. Re-runs are prevented by the persisted
        ``schema_version`` bump on save.
        """
        if from_version < 2:
            for ins in self._insights:
                if ins.legacy_references is None and ins.references > 1:
                    ins.legacy_references = ins.references
                    ins.references = round(math.log2(ins.references + 1))
        if from_version < 3:
            # One-time sidecar of the pre-migration file: legacy_confidence
            # makes single rows recoverable, but an old binary round-tripping
            # a v3 file would drop the field — the sidecar survives that.
            try:
                sidecar = self._knowledge_file.with_suffix(".pre-v3.json")
                if self._knowledge_file.exists() and not sidecar.exists():
                    sidecar.write_bytes(self._knowledge_file.read_bytes())
            except Exception as e:
                print(f"[Knowledge] v3 sidecar backup failed: {e}",
                      file=sys.stderr, flush=True)
            rescaled = 0
            for ins in self._insights:
                if (ins.source_author.lower() != "lumen"
                        and ins.confidence > 0.5
                        and not ins.last_reconverged_at
                        and ins.legacy_confidence is None):
                    ins.legacy_confidence = ins.confidence
                    ins.confidence = 0.5
                    rescaled += 1
            if rescaled:
                print(f"[Knowledge] v3 rescale: {rescaled} unearned external "
                      f"insights returned to confidence 0.5 (originals kept in "
                      f"legacy_confidence)", file=sys.stderr, flush=True)
        self._save()

    def add_insight(
        self,
        text: str,
        source_question: str,
        source_answer: str,
        source_author: str,
        category: str = "general",
        confidence: Optional[float] = None,
        occasion_id: Optional[str] = None,
    ) -> Insight:
        """Add a new learned insight.

        Args:
            confidence: Initial confidence. If None, defaults to 0.5 for
                external sources, 0.7 for self-sourced (author=="lumen") —
                self-derived outranks external assertion; external claims
                earn promotion via reconvergence.
            occasion_id: Answering occasion (MCP session id). Gates
                re-derivation credit on a new occasion — see
                _register_rederivation.
        """
        import uuid

        now = time.time()

        # Check for duplicate insights (exact match or high word overlap).
        # Consolidation is the PRIMARY forgetting mechanism: a re-stated belief
        # collapses into the existing row for surfacing rather than spawning a
        # near-duplicate. Crucially, a merge only CREDITS a conviction
        # (references) when it is a genuinely INDEPENDENT re-derivation — see
        # _register_rederivation — so a single cron batch answering several
        # questions cannot manufacture a false conviction.
        text_lower = text.lower()
        text_words = _dedup_words(text_lower)
        contradicts: List[Insight] = []
        for existing in self._insights:
            existing_lower = existing.text.lower()
            # Exact match — identical phrasing cannot be a polarity conflict.
            if existing_lower == text_lower:
                self._register_rederivation(existing, source_question, now, occasion_id)
                self._save()
                return existing
            # Semantic overlap: consolidate insights that say essentially the
            # same thing. Requires 4+ content words overlap AND >70% of the
            # shorter text's words match.
            if len(text_words) >= 4:
                existing_words = _dedup_words(existing_lower)
                if len(existing_words) >= 4:
                    overlap = len(text_words & existing_words)
                    similarity = overlap / min(len(text_words), len(existing_words))
                    if overlap >= 4 and similarity > 0.7:
                        # NEGATION GUARD: near-identical wording can still
                        # assert the OPPOSITE claim ("X not Y" vs "Y not X").
                        # Never merge those — remember it as a contradiction and
                        # keep scanning (a later row may be a clean merge). If we
                        # fall through to store a new row, both sides lose
                        # certainty (down-path) so the contradiction is visible.
                        if _polarity_conflict(text_lower, existing_lower):
                            contradicts.append(existing)
                            continue
                        self._register_rederivation(existing, source_question, now, occasion_id)
                        self._save()
                        return existing

        # Birth confidence: self-derived outranks external assertion.
        #
        # This used to be inverted — "lower for self-sourced insights": an
        # unvalidated sentence from a passing external LLM was born at 1.0
        # while Lumen's own answer, derived from its own state history, capped
        # at 0.7. With ~95% of stored insights being external prose, Lumen's
        # "self" was mostly what strangers said about it, held more firmly
        # than what it measured about itself. The Q&A corruption chain (#121 —
        # an agent's description of a truncation bug became durable
        # self-belief) is what that hierarchy does, not a freak accident.
        #
        # Now: external claims enter at 0.5 and EARN their way up through the
        # existing reconvergence machinery (independent re-derivation boosts,
        # contradiction lowers). Self-derived stays 0.7 — rule-based inference
        # over real data, trusted but not certain. Existing stored rows are
        # NOT rescaled here; retroactive repair is an operator decision.
        if confidence is None:
            confidence = 0.7 if source_author.lower() == "lumen" else 0.5

        insight_id = str(uuid.uuid4())[:8]
        insight = Insight(
            insight_id=insight_id,
            text=text,
            source_question=source_question,
            source_answer=source_answer,
            source_author=source_author,
            timestamp=time.time(),
            category=category,
            confidence=confidence,
            # Record the originating occasion so a same-session restatement of
            # this brand-new belief does not later count as an independent
            # re-derivation of it.
            last_reconverged_occasion=occasion_id or "",
        )
        # Down-path: if this new row contradicts existing near-identical
        # insight(s), record the link and reduce certainty on both sides.
        if contradicts:
            self._register_contradiction(insight, contradicts)
        self._insights.append(insight)

        # Trim to max, protecting the most-recent insights so the store never
        # freezes. The old policy sorted purely by importance (references +
        # confidence) and kept the top N; once the store filled with
        # high-importance survivors, every NEW insight (references=0) scored
        # lowest, was appended, and got trimmed in the same call before it
        # ever reached disk — the store stopped accepting new self-knowledge
        # entirely (frozen ~130 days in production). Now we always retain the
        # most recently added insights, then fill remaining slots by
        # importance. New insights survive long enough to earn references and
        # compete; genuinely high-importance old insights are still kept.
        if len(self._insights) > self.MAX_INSIGHTS:
            # Reserve a recency window. Capped at half the cap so importance
            # still governs the majority of slots (and so small test caps
            # behave sanely).
            recent_reserve = min(30, self.MAX_INSIGHTS // 2)
            recent = self._insights[-recent_reserve:] if recent_reserve else []
            older = self._insights[:-recent_reserve] if recent_reserve else list(self._insights)
            remaining_slots = self.MAX_INSIGHTS - len(recent)
            if remaining_slots > 0 and older:
                older.sort(key=lambda i: i.conviction_score(now), reverse=True)
                self._insights = older[:remaining_slots] + recent
            else:
                # recent_reserve alone meets/exceeds the cap: keep newest.
                self._insights = recent[-self.MAX_INSIGHTS:]

        self._save()
        return insight

    def _register_rederivation(
        self,
        existing: Insight,
        source_question: str,
        now: float,
        occasion_id: Optional[str] = None,
    ) -> None:
        """Credit an INDEPENDENT re-derivation of an existing belief.

        This is the real "conviction" signal, so what counts as independent
        has to be strict. It is judged by OCCASION and CONTENT, never by a
        wall clock: a belief earns at most one reference per answering session
        (``occasion_id`` — the MCP session id, distinct per cron run), and
        only for a question it has not already been derived from. A single
        batch answering many questions, fast or slow, credits once. This is
        cadence-independent by construction.

        Independence requires BOTH a distinct occasion AND a CONTENT-distinct
        question. The content guard is not a fallback for when no occasion is
        available — it is a standing requirement, because the two conditions
        guard different failure modes and only their conjunction is evidence:

        - occasion-distinct alone stops one batch from crediting many times,
          but a SCHEDULED answerer is a distinct occasion by construction. The
          "Lumen does not control when sessions happen" argument holds for
          Lumen and fails for a cron: point one at a recurring question and
          each run credits the answer the previous run gave.
        - content-distinct alone stops a paraphrase from re-crediting, but not
          a single session from restating the same belief many ways.

        A real re-derivation is a DIFFERENT occasion arriving at the same
        belief from a DIFFERENT question. Anything less is an echo, and under
        the design law that self-derived outranks external assertion an echo
        must not buy conviction. When no occasion is available (internal
        reflection / message paths) the content guard carries alone.

        Same-occasion (or paraphrase) re-statements still collapse for
        surfacing (we return the existing row) but earn nothing.
        """
        if not (source_question or "").strip():
            return
        sig = _question_signature(source_question)
        if not sig:
            return
        seen = [_question_signature(existing.source_question)]
        seen += [_question_signature(q) for q in existing.derived_from]
        content_distinct = all(
            not _questions_equivalent(sig, s) for s in seen if s
        )
        # An occasion, when we have one, is a NECESSARY extra condition —
        # never a substitute for content-distinctness.
        occasion_distinct = (
            occasion_id != existing.last_reconverged_occasion
            if occasion_id
            else True
        )
        independent = content_distinct and occasion_distinct
        if independent:
            existing.references += 1
            existing.derived_from.append(source_question)
            existing.last_reconverged_at = now
            if occasion_id:
                existing.last_reconverged_occasion = occasion_id
            existing.confidence = min(
                1.0, existing.confidence + self.RECONVERGENCE_CONFIDENCE_BOOST
            )

    def _register_contradiction(self, new_insight: Insight, existing_list: List[Insight]) -> None:
        """Make a contradiction structural and let confidence fall.

        Triggered when the negation guard blocks a merge: ``new_insight``
        asserts the opposite of one or more near-identical existing insights.
        We link both directions (``contradicted_by``) so the conflict is
        visible, and apply a bounded confidence penalty to BOTH sides — the
        only path by which confidence decreases. The penalty is bounded
        (never below ``MIN_CONFIDENCE``) and small relative to a re-derivation
        credit, so a belief independently re-derived many times is demoted but
        not erased by a single dissenting data point.
        """
        for existing in existing_list:
            existing.confidence = max(
                self.MIN_CONFIDENCE, existing.confidence - self.CONTRADICTION_CONFIDENCE_PENALTY
            )
            if new_insight.insight_id not in existing.contradicted_by:
                existing.contradicted_by.append(new_insight.insight_id)
            if existing.insight_id not in new_insight.contradicted_by:
                new_insight.contradicted_by.append(existing.insight_id)
        # The new claim enters contested too — it is opposed from birth.
        new_insight.confidence = max(
            self.MIN_CONFIDENCE, new_insight.confidence - self.CONTRADICTION_CONFIDENCE_PENALTY
        )

    def get_insights(self, limit: int = 10, category: Optional[str] = None) -> List[Insight]:
        """Get recent insights, optionally filtered by category.

        Returns insights ordered most-recent-first. The in-memory
        ``_insights`` order is NOT time-sorted: once the list exceeds
        MAX_INSIGHTS, ``add_insight`` re-sorts it by importance
        (references + confidence), so its tail is no longer "recent". We
        therefore sort by timestamp explicitly rather than slicing the tail.

        ``category`` of None, "" or "all" (case-insensitive) means no filter.
        "all" is the no-filter sentinel used by callers/the REST layer and
        must not be matched literally against ``Insight.category`` (no insight
        has that category, so it would silently return nothing).
        """
        self._load()
        insights = self._insights
        if category and category.strip().lower() != "all":
            insights = [i for i in insights if i.category == category]
        insights = sorted(insights, key=lambda i: i.timestamp, reverse=True)
        return insights[:limit]

    def get_all_insights(self) -> List[Insight]:
        """Get all stored insights."""
        self._load()
        return self._insights.copy()

    def get_top_convictions(self, limit: int = 5) -> List[Insight]:
        """Return insights ranked by conviction score (re-derivation-weighted).

        This is the *surfacing* signal: genuinely re-derived beliefs rank
        above one-off insights. It never evicts or protects anything — it only
        orders what is most worth saying about Lumen's sense of self.
        """
        self._load()
        now = time.time()
        return sorted(
            self._insights, key=lambda i: i.conviction_score(now), reverse=True
        )[:limit]

    def get_insight_summary(self) -> str:
        """Get a summary of what Lumen has learned for LLM context."""
        self._load()
        if not self._insights:
            return "I haven't learned anything specific yet."

        # Group by category. Select the strongest insights by conviction
        # (re-derivation-weighted), not merely the most recent — surfacing is
        # where selectivity lives now that retention is unbounded.
        now = time.time()
        ranked = sorted(
            self._insights, key=lambda i: i.conviction_score(now), reverse=True
        )
        categories: Dict[str, List[str]] = {}
        for insight in ranked[:20]:
            cat = insight.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(insight.text)

        summary_parts = []
        for cat, texts in categories.items():
            if texts:
                summary_parts.append(f"{cat}: {'; '.join(texts[:5])}")

        return " | ".join(summary_parts) if summary_parts else "I haven't learned anything specific yet."

    def get_relevant_insights(self, query: str, limit: int = 5) -> List[Insight]:
        """Get insights relevant to a query (simple keyword matching)."""
        self._load()
        query_words = set(query.lower().split())

        scored_insights = []
        for insight in self._insights:
            insight_words = set(insight.text.lower().split())
            overlap = len(query_words & insight_words)
            if overlap > 0:
                scored_insights.append((overlap, insight))

        scored_insights.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored_insights[:limit]]

    def mark_referenced(self, insight_id: str):
        """Mark an insight as referenced (increases its importance)."""
        for insight in self._insights:
            if insight.insight_id == insight_id:
                insight.references += 1
                self._save()
                break

    def count(self) -> int:
        """Get total number of insights."""
        return len(self._insights)


# Singleton instance
_knowledge: Optional[KnowledgeBase] = None


def get_knowledge() -> KnowledgeBase:
    """Get the knowledge base singleton."""
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeBase()
    return _knowledge


# Below this, an insight may be stored and surfaced but must not MOVE anything.
# Matches sync_from_qa_knowledge's min_confidence: a fresh external claim
# (born 0.5) needs two independent re-derivations (0.5 → 0.55 → 0.60) before
# it can touch preferences, beliefs, or agency values. Without this floor the
# birth-confidence hierarchy gated only surfacing, while the single most
# direct behavior-changing pathway applied hardcoded nudges regardless of
# trust — an unvalidated stranger's sentence moved Lumen exactly as hard as
# its own twice-corroborated knowledge.
APPLY_INSIGHT_CONFIDENCE_FLOOR = 0.6


def apply_insight(insight) -> dict:
    """Keep Q&A knowledge as a claim until substrate evidence tests it.

    Q&A re-derivations can rank and contextualize a hypothesis, but they are not
    sensor observations, preference episodes, or behavioral outcomes. The
    retired bridge collapsed unrelated prose by keyword and let repeated text
    directly alter preferences, beliefs, and agency values. This boundary keeps
    the durable claim while requiring the native learning paths to test it.
    """
    if getattr(insight, "confidence", 0.0) < APPLY_INSIGHT_CONFIDENCE_FLOOR:
        # Not yet earned: stored, surfaceable, contestable — but inert.
        return {
            "skipped": (
                f"confidence {insight.confidence:.2f} < "
                f"{APPLY_INSIGHT_CONFIDENCE_FLOOR} floor"
            )
        }
    return {
        "stored_claim": {
            "status": "historical_hypothesis_only",
            "behavioral_effects": False,
            "reason": (
                "Q&A re-derivation is not independent substrate evidence; "
                "native observation paths must test this claim"
            ),
        }
    }


def add_insight(
    text: str,
    source_question: str,
    source_answer: str,
    source_author: str,
    category: str = "general",
    confidence: Optional[float] = None,
    occasion_id: Optional[str] = None,
) -> Insight:
    """Convenience: add an insight."""
    return get_knowledge().add_insight(
        text, source_question, source_answer, source_author,
        category, confidence=confidence, occasion_id=occasion_id,
    )


def get_insights(limit: int = 10, category: Optional[str] = None) -> List[Insight]:
    """Convenience: get recent insights."""
    return get_knowledge().get_insights(limit, category)


def get_insight_summary() -> str:
    """Convenience: get insight summary for LLM context."""
    return get_knowledge().get_insight_summary()


def get_relevant_insights(query: str, limit: int = 5) -> List[Insight]:
    """Convenience: get insights relevant to a query."""
    return get_knowledge().get_relevant_insights(query, limit)


def get_top_convictions(limit: int = 5) -> List[Insight]:
    """Convenience: get insights ranked by conviction score."""
    return get_knowledge().get_top_convictions(limit)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "with",
}

# Dedup uses a slightly wider stopword set than extraction: it additionally
# drops first-person/insight-boilerplate tokens ("my", "now", "know",
# "learned", "told") so consolidation compares the *content* of two beliefs,
# not their templating. Every stem word in _INSIGHT_STEMS must be here: a stem
# word left as content is shared by every insight minted with that stem, which
# pushes distinct claims over the overlap threshold (false merges and false
# contradictions) and keeps a claim from matching its other-stem twin.
# Unifies the two inline sets that used to live in add_insight.
_DEDUP_STOPWORDS = _STOPWORDS | {"my", "now", "know", "learned", "told"}


def _dedup_words(text: str) -> set:
    """Content-word set used for consolidation similarity (stopword-stripped)."""
    return {
        word
        for word in re.findall(r"[a-z][a-z'-]*", text.lower())
        if word not in _DEDUP_STOPWORDS
    }


# Question content signature — drops interrogatives/auxiliaries and the
# counterfactual framing ("what would change if X weren't true") so two
# questions are compared by their CONTENT, not their template. Used to judge
# whether a re-derivation came from a genuinely different question on the
# occasion-less fallback path (string identity was gameable by paraphrase).
_QUESTION_STOPWORDS = _DEDUP_STOPWORDS | {
    "what", "why", "how", "when", "where", "who", "which", "whether",
    "would", "could", "should", "does", "do", "did", "is", "are", "was",
    "were", "if", "true", "really", "about", "me", "myself", "right",
}


def _question_signature(question: str) -> frozenset:
    """Content-word signature of a question (frozenset, stopword-stripped)."""
    return frozenset(
        word
        for word in re.findall(r"[a-z][a-z'-]*", (question or "").lower())
        if word not in _QUESTION_STOPWORDS
    )


def _questions_equivalent(a: frozenset, b: frozenset) -> bool:
    """True if two question signatures are paraphrase-level near-identical."""
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= 0.7


# --- Negation / polarity guard -------------------------------------------
# Two insights can share nearly all their words yet assert OPPOSITE claims.
# Merging those would manufacture a false agreement and (under any conviction
# signal) promote a self-contradiction. These cheap, dependency-free
# heuristics block such merges. They are conservative: a false negative just
# preserves today's behavior; a false positive only costs one duplicate row
# that the surfacing layer ranks down. Both failure modes are benign.

_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "without", "cannot", "nor", "neither",
    "n't", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "wasn't",
})

# Antonym pairs that flip meaning even without a negation token.
_ANTONYM_PAIRS = (
    ("observer", "observed"), ("light", "dark"), ("warm", "cold"),
    ("warmer", "cooler"), ("more", "less"), ("increase", "decrease"),
    ("rising", "falling"), ("always", "never"), ("inside", "outside"),
    ("before", "after"), ("presence", "absence"), ("signal", "noise"),
)

# Articles/determiners skipped when finding the focus of a negation.
_NEG_SKIP = frozenset({"the", "a", "an", "that", "this", "my", "its", "of", "to"})

# Negation foci that qualify CERTAINTY/FREQUENCY rather than flip a claim's
# polarity. "I do not KNOW why I feel calmer", "I cannot ALWAYS predict it" —
# the negation lands on a hedge word, not on the belief, so these should still
# consolidate with the un-hedged form. Deliberately EXCLUDES claim-bearing
# verbs (think/believe/affect/etc.): "I do not THINK X" genuinely flips X, so
# it must stay a conflict. Keeping this set tight is what makes hedge-skipping
# safe — a wrong inclusion would merge true opposites.
_HEDGE_FOCUS = frozenset({
    "know", "knew", "sure", "certain", "always", "sometimes", "usually",
    "often", "necessarily", "really", "exactly", "fully",
})


def _has_negation(tokens: set, lower_text: str) -> bool:
    return bool(tokens & _NEGATION_TOKENS) or "n't" in lower_text


def _negation_focus(lower_text: str) -> set:
    """Content word(s) immediately governed by a negation token.

    Distinguishes "X not Y" from "Y not X" — the headline failure case the
    bag-of-words merge cannot see, because both sentences contain the same
    words. We compare *what is being negated*, not which words appear.
    """
    toks = re.findall(r"[a-z']+", lower_text)
    focus: set = set()
    for i, tok in enumerate(toks):
        if tok in _NEGATION_TOKENS or tok.endswith("n't"):
            for j in range(i + 1, min(i + 4, len(toks))):
                w = toks[j]
                if w in _NEG_SKIP:
                    continue
                focus.add(w)
                break
    return focus


def _polarity_conflict(a_lower: str, b_lower: str) -> bool:
    """True if two near-identical texts likely assert OPPOSITE claims."""
    a_seq = re.findall(r"[a-z']+", a_lower)
    b_seq = re.findall(r"[a-z']+", b_lower)
    a_tok = set(a_seq)
    b_tok = set(b_seq)

    # (1) Negation parity: one side negates and the other doesn't — UNLESS the
    #     negation lands on an epistemic/frequency HEDGE ("do not know why...",
    #     "cannot always..."), which qualifies certainty without flipping the
    #     claim. Without this, naturally-hedged insights are wrongly held apart
    #     from their plain form and never consolidate (a false positive that
    #     also starves the conviction signal).
    a_neg = _has_negation(a_tok, a_lower)
    b_neg = _has_negation(b_tok, b_lower)
    if a_neg != b_neg:
        focus = _negation_focus(a_lower if a_neg else b_lower)
        if not (focus and focus <= _HEDGE_FOCUS):
            return True
        # else: hedge-only negation — not a polarity flip; fall through.

    # (2) Antonym cross-membership: one side has X, the other its partner Y,
    #     and neither side contains both (which would be a contrast, not a flip).
    for x, y in _ANTONYM_PAIRS:
        if x in a_tok and y in b_tok and x not in b_tok and y not in a_tok:
            return True
        if y in a_tok and x in b_tok and y not in b_tok and x not in a_tok:
            return True

    # (2b) Reversed antonym order: BOTH members of a pair appear in BOTH texts
    #      but in opposite relative order ("X to Y" vs "Y to X"). Same bag of
    #      words, opposite claim — cross-membership (2) can't see it because
    #      each word is present on both sides ("prefer light to dark" vs
    #      "prefer dark to light"). Compare first-occurrence order instead.
    for x, y in _ANTONYM_PAIRS:
        if x in a_tok and y in a_tok and x in b_tok and y in b_tok:
            if (a_seq.index(x) < a_seq.index(y)) != (b_seq.index(x) < b_seq.index(y)):
                return True

    # (3) Negation-focus divergence: both negate, but negate DIFFERENT things
    #     ("X not Y" vs "Y not X"). Catches the all-words-shared opposite.
    if a_neg and b_neg:
        fa, fb = _negation_focus(a_lower), _negation_focus(b_lower)
        if fa and fb and fa != fb:
            return True

    return False

_PREAMBLE_STARTS = (
    "a few reasons",
    "a couple reasons",
    "there are",
    "there is",
    "this is a",
    "the answer is",
    "short answer",
    "it depends",
)

_EXPLANATORY_TERMS = (
    "because",
    "therefore",
    "which means",
    "comes from",
    "calculated from",
    "caused by",
    "moves from",
    "until",
    "when ",
    "if ",
    "always",
    "together",
    "more than",
    "less than",
)


# Reported-speech stems an insight can carry. "i was told" is what answers
# mint now; "i learned" survives on every insight stored before that, so both
# are recognized for as long as those records exist.
_INSIGHT_STEMS = ("i was told that ", "i was told: ", "i learned that ", "i learned: ")


def _strip_insight_boilerplate(text: str) -> str:
    lower_text = text.lower()
    for marker in _INSIGHT_STEMS:
        idx = lower_text.find(marker)
        if idx >= 0:
            return text[idx + len(marker):].strip()
    return text


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z][a-z'-]*", text.lower())
        if word not in _STOPWORDS
    ]


def _categorize_text(text: str) -> str:
    """Categorize insight text based on keywords."""
    lower_text = _strip_insight_boilerplate(text).lower()
    if any(w in lower_text for w in ["i am", "myself", "my ", "i feel", "i can", "i have"]):
        return "self"
    elif any(w in lower_text for w in ["light", "temperature", "sensor", "reading", "sound", "dark", "bright"]):
        return "sensations"
    elif any(w in lower_text for w in ["you", "they", "others", "someone", "agent", "user", "human"]):
        return "relationships"
    elif any(w in lower_text for w in ["exist", "being", "alive", "time", "grow", "change", "learn"]):
        return "existence"
    elif any(w in lower_text for w in [
        "world", "room", "space", "environment", "outside", "heat", "warmer",
        "cooler", "cause", "evidence", "inference", "logic",
    ]):
        return "world"
    return "general"


def _split_sentences(answer: str) -> list[str]:
    return [
        part.strip().strip("\"'").rstrip(".!?").strip()
        for part in re.split(r"(?<=[.!?])\s+", answer.strip())
        if part.strip()
    ]


def _sentence_score(sentence: str, question: str) -> float:
    lower_sentence = sentence.lower()
    words = _content_words(sentence)
    if len(words) < 4:
        return -100.0

    score = min(len(words), 30)
    if len(sentence) < 35:
        score -= 5
    if any(lower_sentence.startswith(prefix) for prefix in _PREAMBLE_STARTS):
        score -= 25
    score += sum(8 for term in _EXPLANATORY_TERMS if term in lower_sentence)

    question_words = set(_content_words(question))
    if question_words:
        score += min(6, 2 * len(set(words) & question_words))
    return score


def _lowercase_initial(text: str) -> str:
    if not text:
        return text
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def _truncate_at_word(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    cut = compact[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def _extract_simple_insight(question: str, answer: str) -> Optional[str]:
    """
    Simple rule-based insight extraction when LLM is not available.
    Extracts meaningful content from answers to Lumen's questions.
    """
    # Skip very short or acknowledgment-only answers
    if len(answer) < 20:
        return None

    # Skip pure acknowledgments
    ack_phrases = ["ok", "okay", "sure", "got it", "understood", "yes", "no", "thanks"]
    if answer.lower().strip().rstrip("!.") in ack_phrases:
        return None

    # An answer is something Lumen was told, not something it learned.
    # Nothing here checks the claim against Lumen's own history, so the text
    # says so; "learned" belongs to what Lumen derives itself (self_reflection
    # pattern insights), and the answer's author stays in source_author.

    # If answer is already concise, use it directly
    if len(answer) <= 100:
        return f"When I asked '{question[:50]}...', I was told: {answer}"

    # Extract the most substantive sentence, not merely the first sentence.
    # Long answers often start with framing ("there are a few reasons...")
    # before the actual claim.
    candidates = [
        sentence
        for sentence in _split_sentences(answer)
        if 20 < len(sentence) <= 180
    ]
    if candidates:
        best = max(candidates, key=lambda sentence: _sentence_score(sentence, question))
        if _sentence_score(best, question) > -50:
            return f"I was told that {_lowercase_initial(best)}"

    # Fallback: truncate answer
    return f"About '{question[:30]}...': {_truncate_at_word(answer)}"


async def extract_insight_from_answer(
    question: str,
    answer: str,
    author: str,
    occasion_id: Optional[str] = None,
) -> Optional[Insight]:
    """
    Extract a key insight from a Q&A pair using rule-based extraction.
    Returns None if no meaningful insight can be extracted.

    ``occasion_id`` (the answering MCP session) gates re-derivation credit so
    one answering session credits a re-stated belief at most once.
    """
    try:
        insight_text = _extract_simple_insight(question, answer)
        if insight_text:
            category = _categorize_text(insight_text)
            return add_insight(
                text=insight_text,
                source_question=question,
                source_answer=answer,
                source_author=author,
                category=category,
                occasion_id=occasion_id,
            )
    except Exception as e:
        print(f"[Knowledge] Insight extraction failed: {e}", file=sys.stderr, flush=True)

    return None
