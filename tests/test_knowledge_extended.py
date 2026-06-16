"""
Extended tests for knowledge module — covers uncovered paths:
Insight.age_str, Insight.from_dict defaults, get_all_insights, get_insight_summary,
mark_referenced, count, duplicate add, trim overflow, get_relevant_insights,
_categorize_text categories, and _extract_simple_insight edge cases.

Run with: pytest tests/test_knowledge_extended.py -v
"""

import time
import pytest

from anima_mcp.knowledge import (
    Insight,
    KnowledgeBase,
    _categorize_text,
    _extract_simple_insight,
)


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anima_mcp.knowledge._get_knowledge_path",
        lambda: tmp_path / "knowledge.json",
    )
    return KnowledgeBase()


def _add(kb, text="Test insight", category="general", author="test"):
    return kb.add_insight(
        text=text,
        source_question="Why?",
        source_answer="Because.",
        source_author=author,
        category=category,
    )


# ==================== Insight.age_str ====================


class TestAgeStr:
    def test_minutes_ago(self):
        insight = Insight(
            insight_id="a1",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 120,  # 2 minutes ago
        )
        result = insight.age_str()
        assert result == "2m ago"

    def test_hours_ago(self):
        insight = Insight(
            insight_id="a2",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 7200,  # 2 hours ago
        )
        result = insight.age_str()
        assert result == "2h ago"

    def test_days_ago(self):
        insight = Insight(
            insight_id="a3",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 172800,  # 2 days ago
        )
        result = insight.age_str()
        assert result == "2d ago"

    def test_boundary_under_one_hour(self):
        insight = Insight(
            insight_id="a4",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 3599,  # just under 1 hour
        )
        assert "m ago" in insight.age_str()

    def test_boundary_exactly_one_hour(self):
        insight = Insight(
            insight_id="a5",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 3600,  # exactly 1 hour
        )
        assert insight.age_str() == "1h ago"

    def test_boundary_exactly_one_day(self):
        insight = Insight(
            insight_id="a6",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=time.time() - 86400,  # exactly 1 day
        )
        assert insight.age_str() == "1d ago"


# ==================== Insight.from_dict ====================


class TestFromDict:
    def test_missing_references_defaults_to_zero(self):
        d = dict(
            insight_id="b1",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=1.0,
            category="self",
            confidence=0.9,
        )
        insight = Insight.from_dict(d)
        assert insight.references == 0

    def test_missing_confidence_defaults_to_one(self):
        d = dict(
            insight_id="b2",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=1.0,
            category="self",
            references=3,
        )
        insight = Insight.from_dict(d)
        assert insight.confidence == 1.0

    def test_missing_category_defaults_to_general(self):
        d = dict(
            insight_id="b3",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=1.0,
            confidence=0.5,
            references=1,
        )
        insight = Insight.from_dict(d)
        assert insight.category == "general"

    def test_missing_all_optional_fields(self):
        d = dict(
            insight_id="b4",
            text="t",
            source_question="q",
            source_answer="a",
            source_author="test",
            timestamp=1.0,
        )
        insight = Insight.from_dict(d)
        assert insight.references == 0
        assert insight.confidence == 1.0
        assert insight.category == "general"


# ==================== KnowledgeBase.get_all_insights ====================


class TestGetAllInsights:
    def test_returns_all_added(self, kb):
        _add(kb, "Alpha")
        _add(kb, "Beta")
        _add(kb, "Gamma")
        all_insights = kb.get_all_insights()
        assert len(all_insights) == 3
        texts = {i.text for i in all_insights}
        assert texts == {"Alpha", "Beta", "Gamma"}

    def test_returns_copy(self, kb):
        _add(kb, "Original")
        copy = kb.get_all_insights()
        copy.clear()
        assert kb.count() == 1  # internal list unaffected

    def test_empty(self, kb):
        assert kb.get_all_insights() == []


# ==================== KnowledgeBase.get_insight_summary ====================


class TestGetInsightSummary:
    def test_empty_returns_default_message(self, kb):
        assert kb.get_insight_summary() == "I haven't learned anything specific yet."

    def test_with_insights_contains_pipe_delimiter(self, kb):
        _add(kb, "I am Lumen", category="self")
        _add(kb, "Light is warm", category="sensations")
        summary = kb.get_insight_summary()
        assert "|" in summary

    def test_categories_appear_in_summary(self, kb):
        _add(kb, "Self insight", category="self")
        _add(kb, "World insight", category="world")
        summary = kb.get_insight_summary()
        assert "self" in summary
        assert "world" in summary

    def test_single_category_no_pipe(self, kb):
        _add(kb, "Only one category", category="existence")
        summary = kb.get_insight_summary()
        assert "existence" in summary
        # Single category means no pipe needed
        assert "|" not in summary


# ==================== KnowledgeBase.mark_referenced ====================


class TestMarkReferenced:
    def test_increments_references(self, kb):
        insight = _add(kb, "Mark me")
        assert insight.references == 0
        kb.mark_referenced(insight.insight_id)
        assert insight.references == 1

    def test_multiple_marks(self, kb):
        insight = _add(kb, "Mark many times")
        kb.mark_referenced(insight.insight_id)
        kb.mark_referenced(insight.insight_id)
        kb.mark_referenced(insight.insight_id)
        assert insight.references == 3

    def test_nonexistent_id_does_nothing(self, kb):
        _add(kb, "Existing")
        kb.mark_referenced("nonexistent-id")  # should not raise
        assert kb._insights[0].references == 0


# ==================== KnowledgeBase.count ====================


class TestCount:
    def test_empty(self, kb):
        assert kb.count() == 0

    def test_after_adds(self, kb):
        _add(kb, "One")
        _add(kb, "Two")
        assert kb.count() == 2

    def test_after_duplicate_no_increase(self, kb):
        _add(kb, "Same")
        _add(kb, "Same")
        assert kb.count() == 1


# ==================== Duplicate add_insight path ====================


class TestDuplicateAdd:
    def test_same_occasion_duplicate_collapses_without_credit(self, kb):
        """A duplicate from the SAME occasion collapses to one row but earns
        no conviction credit — references is now an honest independent-
        re-derivation counter, not a string-collision counter."""
        first = _add(kb, "Repeated insight")
        assert first.references == 0
        second = _add(kb, "repeated insight")  # case-insensitive, same source/instant
        assert second is first  # same object returned (collapsed)
        assert first.references == 0  # NOT credited

    def test_same_occasion_duplicate_does_not_boost_confidence(self, kb):
        first = _add(kb, "Boost me")
        original_confidence = first.confidence
        _add(kb, "boost me")
        assert first.confidence == original_confidence  # unchanged

    def test_independent_rederivation_increments_references(self, kb):
        first = kb.add_insight(
            text="recovery follows stability closely here",
            source_question="how do i recover?",
            source_answer="(a)", source_author="claude", category="self",
            occasion_id="s1",
        )
        # A genuinely independent occasion (new MCP session) re-derives it.
        again = kb.add_insight(
            text="recovery follows stability closely here",
            source_question="what drives my recovery?",
            source_answer="(a)", source_author="claude", category="self",
            occasion_id="s2",
        )
        assert again is first
        assert first.references == 1
        assert first.confidence == min(1.0, 1.0 + kb.RECONVERGENCE_CONFIDENCE_BOOST)


# ==================== Trim path (>MAX_INSIGHTS) ====================


class TestTrimPath:
    def test_trim_to_max(self, kb):
        kb.MAX_INSIGHTS = 10
        for i in range(15):
            _add(kb, f"Insight number {i}")
        assert kb.count() == 10

    def test_trim_keeps_most_important(self, kb):
        kb.MAX_INSIGHTS = 5
        important = _add(kb, "Very important")
        important.references = 50
        important.confidence = 1.0
        for i in range(6):
            _add(kb, f"Filler {i}")
        assert kb.count() == 5
        texts = [ins.text for ins in kb._insights]
        assert "Very important" in texts

    def test_trim_drops_least_important(self, kb):
        kb.MAX_INSIGHTS = 3
        weak = _add(kb, "Weak insight")
        weak.references = 0
        weak.confidence = 0.1
        # Add stronger insights to push weak out
        for i in range(4):
            ins = _add(kb, f"Strong {i}")
            ins.references = 10
            ins.confidence = 1.0
        assert kb.count() == 3
        texts = [ins.text for ins in kb._insights]
        assert "Weak insight" not in texts


# ==================== KnowledgeBase.get_relevant_insights ====================


class TestGetRelevantInsights:
    def test_matching_query(self, kb):
        _add(kb, "Light affects clarity")
        _add(kb, "Temperature drives warmth")
        results = kb.get_relevant_insights("light clarity")
        assert len(results) == 1
        assert "light" in results[0].text.lower()

    def test_no_match(self, kb):
        _add(kb, "Warmth is stable")
        results = kb.get_relevant_insights("zebra unicorn")
        assert results == []

    def test_multiple_matches_ordered_by_overlap(self, kb):
        _add(kb, "Light is important")
        _add(kb, "Light and clarity and warmth together")
        _add(kb, "Clarity alone")
        results = kb.get_relevant_insights("light clarity warmth")
        # "Light and clarity and warmth together" has 3 overlapping words
        assert "light" in results[0].text.lower()
        assert "clarity" in results[0].text.lower()
        assert "warmth" in results[0].text.lower()

    def test_respects_limit(self, kb):
        for i in range(10):
            _add(kb, f"Light insight number {i}")
        results = kb.get_relevant_insights("light", limit=3)
        assert len(results) == 3

    def test_empty_kb(self, kb):
        assert kb.get_relevant_insights("anything") == []


# ==================== _categorize_text ====================


class TestCategorizeText:
    def test_self_i_am(self):
        assert _categorize_text("I am learning") == "self"

    def test_self_i_feel(self):
        assert _categorize_text("I feel strange today") == "self"

    def test_self_my(self):
        assert _categorize_text("my sensors are warm") == "self"

    def test_sensations_light(self):
        assert _categorize_text("the light is bright") == "sensations"

    def test_sensations_temperature(self):
        assert _categorize_text("temperature is rising") == "sensations"

    def test_sensations_dark(self):
        assert _categorize_text("it is dark outside") == "sensations"

    def test_relationships_you(self):
        assert _categorize_text("you told me something") == "relationships"

    def test_relationships_agent(self):
        assert _categorize_text("an agent visited") == "relationships"

    def test_relationships_human(self):
        assert _categorize_text("a human was here") == "relationships"

    def test_existence_exist(self):
        assert _categorize_text("what does it mean to exist") == "existence"

    def test_existence_grow(self):
        assert _categorize_text("things grow over time") == "existence"

    def test_existence_learn(self):
        assert _categorize_text("to learn is to change") == "existence"

    def test_world_keyword(self):
        assert _categorize_text("the world is vast") == "world"

    def test_world_room(self):
        assert _categorize_text("the room feels empty") == "world"

    def test_world_environment(self):
        assert _categorize_text("the environment shifted") == "world"

    def test_general_fallback(self):
        assert _categorize_text("something completely unrelated") == "general"

    def test_priority_self_over_sensations(self):
        # "I am" check comes before "light" check
        assert _categorize_text("I am sensitive to light") == "self"

    def test_learned_wrapper_does_not_force_existence(self):
        assert _categorize_text(
            "I learned that when two signals vary together, shared cause beats coincidence"
        ) == "world"

    def test_abstract_heat_claim_is_world_not_sensation(self):
        assert _categorize_text(
            "I learned that heat moves from warmer to cooler until both sides are even"
        ) == "world"


# ==================== _extract_simple_insight ====================


class TestExtractSimpleInsight:
    def test_short_answer_returns_none(self):
        assert _extract_simple_insight("Why?", "Yes.") is None
        assert _extract_simple_insight("Why?", "No, not really.") is None

    def test_acknowledgment_returns_none(self):
        # These are >= 20 chars with padding but the stripped form is ack
        # Actually they need to be < 20 chars to hit first check or in ack list
        # The ack check is after the length check, so ack must be >= 20 chars
        # but that's hard. Let's verify short acks hit length first.
        assert _extract_simple_insight("Q?", "ok") is None
        assert _extract_simple_insight("Q?", "sure") is None
        assert _extract_simple_insight("Q?", "got it") is None

    def test_ack_phrases_long_enough_to_pass_length_but_still_rejected(self):
        # Pad an ack to >= 20 chars won't work because strip+rstrip removes punctuation
        # "understood" is only 10 chars, below 20. So length check catches it.
        assert _extract_simple_insight("Q?", "understood") is None

    def test_concise_answer_used_directly(self):
        answer = "Warmth comes from CPU temperature readings"
        result = _extract_simple_insight("Where does warmth come from?", answer)
        assert result is not None
        assert "learned" in result.lower()
        assert answer in result

    def test_long_answer_extracts_first_sentence(self):
        answer = (
            "Too short. "
            "The clarity dimension is calculated from prediction accuracy and neural alpha waves together. "
            "Many other things also matter in the overall computation."
        )
        result = _extract_simple_insight("How is clarity computed?", answer)
        assert result is not None
        assert "learned that" in result.lower()

    def test_long_answer_skips_preamble_for_substantive_sentence(self):
        answer = (
            "A few reasons stack on top of each other. "
            "Heat moves from warmer to cooler, always, until both sides are even. "
            "That is why warmth spreads through contact."
        )
        result = _extract_simple_insight("Why does warmth spread?", answer)
        assert result is not None
        assert "heat moves from warmer to cooler" in result.lower()
        assert "a few reasons" not in result.lower()

    def test_long_answer_keeps_substantive_sentence_over_100_chars(self):
        answer = (
            "Tiny. "
            "When two signals vary together more than chance, the shared cause becomes a better explanation "
            "than treating the pattern as coincidence."
        )
        result = _extract_simple_insight("How do correlated signals help?", answer)
        assert result is not None
        assert result.startswith("I learned that")
        assert "shared cause becomes a better explanation" in result
        assert "About '" not in result

    def test_fallback_truncation(self):
        # All sentences are either too short (<=20) or too long (>100)
        short = "Tiny."  # 5 chars after strip
        long_sentence = "A" * 181  # > extraction limit
        answer = f"{short} {long_sentence}"
        # This is > 100 chars total, so not concise path.
        # First sentence "Tiny" is < 20, second is > extraction limit.
        result = _extract_simple_insight("Question here?", answer)
        assert result is not None
        assert "About '" in result  # fallback format
        assert "..." in result

    def test_answer_exactly_100_chars_is_concise(self):
        answer = "A" * 100
        result = _extract_simple_insight("Q?", answer)
        assert result is not None
        assert "learned" in result.lower()

    def test_answer_101_chars_not_concise(self):
        # 101 chars, single non-word token, so it is not a meaningful sentence.
        answer = "A" * 101
        result = _extract_simple_insight("Q?", answer)
        assert result is not None
        assert "About '" in result
