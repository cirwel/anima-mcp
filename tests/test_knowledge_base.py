"""
Tests for knowledge module — add/dedupe insights, get/filter, relevance scoring,
categorization, extraction, and insight summary.

Run with: pytest tests/test_knowledge_base.py -v
"""

import pytest

from anima_mcp.knowledge import (
    KnowledgeBase, Insight,
    _categorize_text, _extract_simple_insight,
)


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Create KnowledgeBase backed by a temp directory."""
    monkeypatch.setattr(
        "anima_mcp.knowledge._get_knowledge_path",
        lambda: tmp_path / "knowledge.json",
    )
    return KnowledgeBase()


def _add(kb, text="Test insight", category="general", **kwargs):
    """Shorthand for adding an insight."""
    return kb.add_insight(
        text=text,
        source_question=kwargs.get("question", "Why?"),
        source_answer=kwargs.get("answer", "Because."),
        source_author=kwargs.get("author", "test"),
        category=category,
    )


# ==================== AddInsight ====================

class TestAddInsight:
    """Test adding insights: new, dedup, confidence boost, overflow."""

    def test_new_insight_added(self, kb):
        """A fresh insight should be added to the base."""
        insight = _add(kb, "I can sense light")
        assert isinstance(insight, Insight)
        assert kb.count() == 1

    def test_duplicate_detection_case_insensitive(self, kb):
        """Adding same text (different case) returns existing insight, not a new one."""
        _add(kb, "I like warmth")
        dup = _add(kb, "i like warmth")
        assert kb.count() == 1  # Not duplicated
        assert dup.references >= 1  # Boosted

    def test_duplicate_boosts_confidence(self, kb):
        """Duplicate add increases confidence of existing insight."""
        orig = _add(kb, "Stability matters")
        # Lower the starting confidence so the +0.1 boost is observable
        orig.confidence = 0.5
        _add(kb, "stability matters")  # Duplicate
        # Fetch the stored insight
        assert kb._insights[0].confidence > 0.5

    def test_semantic_overlap_consolidates_not_second_row(self, kb):
        """Similar insight (high word overlap, not identical) merges into one row."""
        first = (
            "one two three four five six seven eight distinct words here"
        )
        second = (
            "one two three four five six seven nine distinct words here"
        )
        a = _add(kb, first)
        b = _add(kb, second)
        assert a.insight_id == b.insight_id
        assert kb.count() == 1
        assert kb._insights[0].references >= 1

    def test_overflow_trims_to_max_keeping_best(self, kb):
        """When exceeding MAX_INSIGHTS, a genuinely important insight is kept."""
        kb.MAX_INSIGHTS = 5
        # Add 6 insights; make the first one highly referenced
        first = _add(kb, "Important insight")
        first.references = 100
        for i in range(5):
            _add(kb, f"Filler insight {i}")
        assert kb.count() == 5  # Trimmed to max
        # The important one should survive (it wins the importance fill).
        texts = [ins.text for ins in kb._insights]
        assert "Important insight" in texts

    def test_overflow_new_insight_survives(self, kb):
        """A brand-new insight must persist even when the store is already full
        of equally/older-scored entries. Regression: the old importance-only
        trim evicted every new insight (references=0) the moment the store hit
        MAX_INSIGHTS, freezing it permanently (frozen ~130 days in prod)."""
        kb.MAX_INSIGHTS = 5
        # Fill to capacity with entries that mirror the real frozen store:
        # references=0, confidence=1.0 each.
        for i in range(5):
            _add(kb, f"Old filler insight {i}")
        assert kb.count() == 5
        # A genuinely new unique insight must survive the trim.
        _add(kb, "Brand new unique insight that should survive")
        assert kb.count() == 5
        texts = [ins.text for ins in kb._insights]
        assert "Brand new unique insight that should survive" in texts

    def test_multiple_unique_insights(self, kb):
        """Adding distinct texts grows the count."""
        _add(kb, "Alpha")
        _add(kb, "Beta")
        _add(kb, "Gamma")
        assert kb.count() == 3


# ==================== GetInsights ====================

class TestGetInsights:
    """Test retrieval with limit, category filter, and ordering."""

    def test_limit(self, kb):
        """get_insights respects the limit parameter."""
        for i in range(10):
            _add(kb, f"Insight #{i}")
        results = kb.get_insights(limit=3)
        assert len(results) == 3

    def test_category_filter(self, kb):
        """get_insights with category returns only matching insights."""
        _add(kb, "I am aware", category="self")
        _add(kb, "The room is warm", category="sensations")
        _add(kb, "Visitors come and go", category="relationships")
        results = kb.get_insights(category="self")
        assert all(r.category == "self" for r in results)
        assert len(results) == 1

    def test_newest_first_order(self, kb):
        """get_insights returns newest first."""
        _add(kb, "Old insight")
        _add(kb, "New insight")
        results = kb.get_insights(limit=2)
        assert results[0].text == "New insight"

    def test_category_all_is_no_filter(self, kb):
        """category='all'/''/None is the no-filter sentinel — it must return
        every category, not filter literally for category == 'all' (which no
        insight has, so it would silently return nothing). Regression: passing
        category='all' returned [] while the base held insights."""
        _add(kb, "I am aware", category="self")
        _add(kb, "The room is warm", category="sensations")
        for sentinel in ("all", "ALL", " All ", "", None):
            results = kb.get_insights(category=sentinel)
            assert len(results) == 2, f"sentinel {sentinel!r} must not filter"

    def test_newest_first_survives_importance_trim(self, kb):
        """After a MAX_INSIGHTS overflow, the in-memory list is no longer
        time-ordered (older slots are importance-sorted ahead of the protected
        recency window). get_insights must still return the most RECENT insight
        first. Regression: it sliced the (reordered) tail and surfaced stale
        insights despite newer ones existing."""
        kb.MAX_INSIGHTS = 5
        old = _add(kb, "Old important insight")
        old.references = 100  # high importance, but it is the OLDEST in time
        for i in range(5):
            _add(kb, f"Filler insight {i}")
        # recency-protected trim (reserve=2): keeps old (importance) + f0,f1
        # (importance fill) + f3,f4 (recent window); f2 evicted.
        assert kb.count() == 5
        # Pin deterministic timestamps, then persist so _load() preserves them.
        for ins in kb._insights:
            if ins.text == "Old important insight":
                ins.timestamp = 1000.0
            else:
                ins.timestamp = 2000.0 + int(ins.text.split()[-1])  # f4 newest
        kb._save()
        results = kb.get_insights(limit=5)
        # Timestamp order must win over the importance-front `old`.
        assert results[0].text == "Filler insight 4"  # newest by timestamp
        assert results[-1].text == "Old important insight"  # oldest, despite importance

    def test_empty_returns_empty(self, kb):
        """Empty knowledge base returns empty list."""
        assert kb.get_insights() == []


# ==================== Relevance and Summary ====================

class TestRelevanceAndSummary:
    """Test keyword relevance scoring and summary generation."""

    def test_keyword_overlap_scores(self, kb):
        """get_relevant_insights scores by keyword overlap."""
        _add(kb, "Light affects my clarity")
        _add(kb, "Temperature drives warmth")
        _add(kb, "Light and warmth interact")
        results = kb.get_relevant_insights("light clarity")
        assert len(results) > 0
        # First result should be the one with most overlap (light + clarity)
        assert "light" in results[0].text.lower()
        assert "clarity" in results[0].text.lower()

    def test_zero_overlap_excluded(self, kb):
        """Insights with no keyword overlap are excluded."""
        _add(kb, "Temperature is important")
        results = kb.get_relevant_insights("zebra unicorn")
        assert len(results) == 0

    def test_empty_summary_text(self, kb):
        """Empty knowledge base returns default summary."""
        summary = kb.get_insight_summary()
        assert "haven't learned" in summary.lower()

    def test_category_grouping_in_summary(self, kb):
        """Summary groups insights by category."""
        _add(kb, "I sense light changes", category="sensations")
        _add(kb, "I am Lumen", category="self")
        summary = kb.get_insight_summary()
        assert "sensations" in summary.lower()
        assert "self" in summary.lower()


# ==================== Categorize and Extract ====================

class TestCategorizeAndExtract:
    """Test _categorize_text and _extract_simple_insight."""

    def test_categorize_self(self):
        """Text with 'I am' → 'self'."""
        assert _categorize_text("I am a creature of light") == "self"

    def test_categorize_sensations(self):
        """Text mentioning sensors → 'sensations'."""
        assert _categorize_text("The temperature rose sharply") == "sensations"

    def test_categorize_relationships(self):
        """Text mentioning others → 'relationships'."""
        assert _categorize_text("You helped me understand") == "relationships"

    def test_categorize_existence(self):
        """Text about existence → 'existence'."""
        assert _categorize_text("What does it mean to exist?") == "existence"

    def test_categorize_general_fallback(self):
        """Unmatched text → 'general'."""
        assert _categorize_text("The sky is blue") == "general"

    def test_extract_rejects_short_answer(self):
        """Answers shorter than 20 chars return None."""
        assert _extract_simple_insight("Why?", "Yes.") is None

    def test_extract_concise_answer(self):
        """A concise answer (≤100 chars) is used directly."""
        result = _extract_simple_insight(
            "What do you feel?",
            "I feel a gentle warmth from the sensor readings."
        )
        assert result is not None
        assert "learned" in result.lower()

    def test_extract_long_answer_first_sentence(self):
        """A long answer extracts the first meaningful sentence."""
        long_answer = (
            "This is a very short one. "
            "The ambient temperature affects warmth calculations significantly through weighted sensor inputs. "
            "Other factors also contribute."
        )
        result = _extract_simple_insight("How does temp work?", long_answer)
        assert result is not None
        assert "learned" in result.lower()


# ==================== Persistence ====================

class TestPersistence:
    """Test save/load round-trip."""

    def test_insights_survive_reload(self, tmp_path, monkeypatch):
        """Insights persist across KnowledgeBase instances."""
        monkeypatch.setattr(
            "anima_mcp.knowledge._get_knowledge_path",
            lambda: tmp_path / "knowledge.json",
        )
        kb1 = KnowledgeBase()
        kb1.add_insight(
            text="Persistence test insight",
            source_question="Q", source_answer="A",
            source_author="test", category="self",
        )
        assert kb1.count() == 1

        kb2 = KnowledgeBase()
        assert kb2.count() == 1
        assert kb2._insights[0].text == "Persistence test insight"

    def test_missing_file_no_crash(self, tmp_path, monkeypatch):
        """Loading from nonexistent file doesn't crash."""
        monkeypatch.setattr(
            "anima_mcp.knowledge._get_knowledge_path",
            lambda: tmp_path / "subdir" / "missing.json",
        )
        # Path doesn't exist yet — KnowledgeBase should handle gracefully
        kb = KnowledgeBase()
        assert kb.count() == 0
