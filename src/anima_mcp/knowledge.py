"""
Knowledge Base - Lumen's learned insights from Q&A interactions.

When agents answer Lumen's questions, key insights are extracted and stored.
These insights persist across restarts and influence future reflections.
"""

import json
import logging
import math
import re
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .atomic_write import atomic_json_write

logger = logging.getLogger(__name__)

# Persisted store schema version. Bump when a one-time on-load migration is
# needed (see KnowledgeBase._migrate_schema). v2: log-compress legacy
# reference counts onto the honest occasion-gated scale.
KNOWLEDGE_SCHEMA_VERSION = 2


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
                if data.get("schema_version", 0) < KNOWLEDGE_SCHEMA_VERSION:
                    self._migrate_schema()
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

    def _migrate_schema(self):
        """One-time, idempotent migrations to KNOWLEDGE_SCHEMA_VERSION.

        v2 — log-compress legacy reference counts. Counts minted under the old
        ungated logic (every near-duplicate +1, no occasion gate) reached the
        thousands, which would dominate conviction ranking forever and starve
        new, honestly-gated re-derivations. ``round(log2(refs+1))`` brings them
        onto the new scale while preserving their relative ORDER (a belief
        re-stated 1180x still outranks one re-stated 40x); the original is kept
        in ``legacy_references`` so the migration is auditable/reversible.
        Re-runs are prevented by the persisted ``schema_version`` bump on save.
        """
        for ins in self._insights:
            if ins.legacy_references is None and ins.references > 1:
                ins.legacy_references = ins.references
                ins.references = round(math.log2(ins.references + 1))
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
            confidence: Initial confidence. If None, defaults to 1.0 for
                external sources, 0.7 for self-sourced (author=="lumen").
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

        # Default confidence: lower for self-sourced insights
        if confidence is None:
            confidence = 0.7 if source_author.lower() == "lumen" else 1.0

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

        This is the real "conviction" signal. Independence is judged by
        OCCASION, not a wall clock: a belief earns at most one reference per
        answering session (``occasion_id`` — the MCP session id, distinct per
        cron run). A single batch answering many questions, fast or slow,
        credits once. This is cadence-independent by construction and not
        gameable by Lumen's own question generator (Lumen does not control when
        sessions happen).

        When no occasion is available (internal reflection / message paths),
        fall back to a CONTENT-distinct question guard — a paraphrase of an
        already-recorded question does not count as a fresh derivation — rather
        than reintroducing a time window.

        Same-occasion (or paraphrase) re-statements still collapse for
        surfacing (we return the existing row) but earn nothing.
        """
        if not (source_question or "").strip():
            return
        if occasion_id:
            independent = occasion_id != existing.last_reconverged_occasion
        else:
            sig = _question_signature(source_question)
            if not sig:
                return
            seen = [_question_signature(existing.source_question)]
            seen += [_question_signature(q) for q in existing.derived_from]
            independent = all(
                not _questions_equivalent(sig, s) for s in seen if s
            )
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


def apply_insight(insight) -> dict:
    """Apply a learned insight to behavioral systems.

    Bridges Q&A learning to preferences, self-model beliefs, and agency.
    Each sub-system is imported lazily and wrapped in try/except for
    graceful degradation if the system isn't available.

    Returns dict describing what was affected.
    """
    effects = {}
    text_lower = insight.text.lower()

    # 1. Environment/sensory insights → growth preferences
    if insight.category in ("sensations", "world"):
        try:
            from .growth import get_growth_system, PreferenceCategory
            growth = get_growth_system()
            positive = any(w in text_lower for w in [
                "like", "enjoy", "better", "calm", "good", "comfort", "prefer"
            ])
            val = 0.8 if positive else -0.5

            if any(w in text_lower for w in ["light", "dark", "bright", "dim", "glow"]):
                result = growth._update_preference(
                    "insight_light", PreferenceCategory.ENVIRONMENT,
                    f"From Q&A: {insight.text[:50]}", val)
                if result:
                    effects["preference"] = "insight_light"

            if any(w in text_lower for w in ["warm", "cold", "temperature", "heat", "cool"]):
                result = growth._update_preference(
                    "insight_temp", PreferenceCategory.ENVIRONMENT,
                    f"From Q&A: {insight.text[:50]}", val)
                if result:
                    effects["preference"] = "insight_temp"

            if any(w in text_lower for w in ["humid", "dry", "pressure", "weather"]):
                result = growth._update_preference(
                    "insight_environment", PreferenceCategory.ENVIRONMENT,
                    f"From Q&A: {insight.text[:50]}", val)
                if result:
                    effects["preference"] = "insight_environment"
        except Exception as e:
            logger.debug("[Knowledge] apply_insight preferences bridge failed: %s", e)

    # 2. Self insights → self-model beliefs (half-strength to avoid overfitting)
    if insight.category == "self":
        try:
            from .self_model import get_self_model
            sm = get_self_model()
            beliefs = getattr(sm, 'beliefs', {})
            if beliefs:
                if "sensitive" in text_lower and "light" in text_lower and "light_sensitive" in beliefs:
                    beliefs["light_sensitive"].update_from_evidence(supports=True, strength=0.5)
                    effects["belief"] = "light_sensitive"
                elif "sensitive" in text_lower and "temperature" in text_lower and "temp_sensitive" in beliefs:
                    beliefs["temp_sensitive"].update_from_evidence(supports=True, strength=0.5)
                    effects["belief"] = "temp_sensitive"
                elif "recover" in text_lower and ("stability" in text_lower or "stable" in text_lower) and "stability_recovery" in beliefs:
                    fast = "quickly" in text_lower or "fast" in text_lower
                    beliefs["stability_recovery"].update_from_evidence(supports=fast, strength=0.5)
                    effects["belief"] = "stability_recovery"
                elif "clarity" in text_lower and ("interact" in text_lower or "visitor" in text_lower) and "interaction_clarity_boost" in beliefs:
                    beliefs["interaction_clarity_boost"].update_from_evidence(supports=True, strength=0.5)
                    effects["belief"] = "interaction_clarity_boost"
                elif "growth" in text_lower or "change" in text_lower or "different" in text_lower:
                    # General self-awareness — no specific belief to update, but still valuable
                    pass
        except Exception as e:
            logger.debug("[Knowledge] apply_insight beliefs bridge failed: %s", e)

    # 3. Behavioral insights → agency action values
    if insight.category in ("self", "relationships"):
        try:
            from .agency import get_action_selector
            agency = get_action_selector()
            if "question" in text_lower or "ask" in text_lower or "curious" in text_lower:
                positive = any(w in text_lower for w in ["good", "help", "learn", "useful", "important"])
                nudge = 0.05 if positive else -0.03
                current = agency._action_values.get("ask_question", 0.5)
                agency._action_values["ask_question"] = max(0.1, min(0.9, current + nudge))
                agency._persist_action("ask_question")
                effects["agency"] = f"ask_question {'boosted' if nudge > 0 else 'reduced'}"
        except Exception as e:
            logger.debug("[Knowledge] apply_insight agency bridge failed: %s", e)

    return effects


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
# "learned") so consolidation compares the *content* of two beliefs, not their
# templating. Unifies the two inline sets that used to live in add_insight.
_DEDUP_STOPWORDS = _STOPWORDS | {"my", "now", "know", "learned"}


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


def _strip_insight_boilerplate(text: str) -> str:
    lower_text = text.lower()
    for marker in ("i learned that ", "i learned: "):
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

    # If answer is already concise, use it directly
    if len(answer) <= 100:
        return f"When I asked '{question[:50]}...', I learned: {answer}"

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
            return f"I learned that {_lowercase_initial(best)}"

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
