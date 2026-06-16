"""
Knowledge Base - Lumen's learned insights from Q&A interactions.

When agents answer Lumen's questions, key insights are extracted and stored.
These insights persist across restarts and influence future reflections.
"""

import json
import re
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .atomic_write import atomic_json_write


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
    references: int = 0  # How many times this insight has been referenced

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
        return cls(**d)

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

    MAX_INSIGHTS = 100  # Keep most recent/important insights

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
            else:
                self._insights = []
        except Exception as e:
            print(f"[Knowledge] Load error: {e}", file=sys.stderr, flush=True)
            self._insights = []

    def _save(self):
        """Save insights to persistent storage."""
        try:
            data = {"insights": [i.to_dict() for i in self._insights]}
            atomic_json_write(self._knowledge_file, data, indent=2)
        except Exception as e:
            print(f"[Knowledge] Save error: {e}", file=sys.stderr, flush=True)

    def add_insight(
        self,
        text: str,
        source_question: str,
        source_answer: str,
        source_author: str,
        category: str = "general",
        confidence: Optional[float] = None,
    ) -> Insight:
        """Add a new learned insight.

        Args:
            confidence: Initial confidence. If None, defaults to 1.0 for
                external sources, 0.7 for self-sourced (author=="lumen").
        """
        import uuid

        # Check for duplicate insights (exact match or high word overlap)
        text_lower = text.lower()
        text_words = set(text_lower.split()) - {"i", "a", "the", "is", "that", "and", "or", "of", "to", "in", "my", "now", "know", "learned"}
        for existing in self._insights:
            existing_lower = existing.text.lower()
            # Exact match
            if existing_lower == text_lower:
                existing.references += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                self._save()
                return existing
            # Semantic overlap: consolidate insights that say essentially the same thing
            # Requires 4+ content words overlap AND >70% of the shorter text's words match
            if len(text_words) >= 4:
                existing_words = set(existing_lower.split()) - {"i", "a", "the", "is", "that", "and", "or", "of", "to", "in", "my", "now", "know", "learned"}
                if len(existing_words) >= 4:
                    overlap = len(text_words & existing_words)
                    similarity = overlap / min(len(text_words), len(existing_words))
                    if overlap >= 4 and similarity > 0.7:
                        existing.references += 1
                        existing.confidence = min(1.0, existing.confidence + 0.05)
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
        )
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
                older.sort(key=lambda i: i.references + i.confidence, reverse=True)
                self._insights = older[:remaining_slots] + recent
            else:
                # recent_reserve alone meets/exceeds the cap: keep newest.
                self._insights = recent[-self.MAX_INSIGHTS:]

        self._save()
        return insight

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

    def get_insight_summary(self) -> str:
        """Get a summary of what Lumen has learned for LLM context."""
        self._load()
        if not self._insights:
            return "I haven't learned anything specific yet."

        # Group by category
        categories: Dict[str, List[str]] = {}
        for insight in self._insights[-20:]:  # Last 20 insights
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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

    return effects


def add_insight(
    text: str,
    source_question: str,
    source_answer: str,
    source_author: str,
    category: str = "general",
    confidence: Optional[float] = None,
) -> Insight:
    """Convenience: add an insight."""
    return get_knowledge().add_insight(
        text, source_question, source_answer, source_author,
        category, confidence=confidence,
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


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "with",
}

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
    author: str
) -> Optional[Insight]:
    """
    Extract a key insight from a Q&A pair using rule-based extraction.
    Returns None if no meaningful insight can be extracted.
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
                category=category
            )
    except Exception as e:
        print(f"[Knowledge] Insight extraction failed: {e}", file=sys.stderr, flush=True)

    return None
