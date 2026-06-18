"""
Tests for knowledge module — add/dedupe insights, get/filter, relevance scoring,
categorization, extraction, and insight summary.

Run with: pytest tests/test_knowledge_base.py -v
"""

import json
import math

import pytest

from anima_mcp.knowledge import (
    KnowledgeBase, Insight, KNOWLEDGE_SCHEMA_VERSION,
    _categorize_text, _extract_simple_insight, _polarity_conflict,
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
        """Adding same text (different case) returns the existing insight and
        collapses to one row. On the SAME occasion (same source question, no
        time gap) it earns NO conviction credit — references stays 0.
        Re-stating a belief in one breath is not independent re-derivation."""
        first = _add(kb, "I like warmth")
        dup = _add(kb, "i like warmth")
        assert kb.count() == 1  # Not duplicated
        assert dup is first  # collapsed into the existing row
        assert first.references == 0  # same occasion: no false conviction

    def test_same_occasion_duplicate_does_not_boost_confidence(self, kb):
        """A same-occasion duplicate must NOT ratchet confidence — confidence
        is no longer a one-way ratchet driven by repetition. Only genuine
        independent re-derivation moves it."""
        orig = _add(kb, "Stability matters")
        orig.confidence = 0.5
        _add(kb, "stability matters")  # same source + same instant
        assert kb._insights[0].confidence == 0.5  # unchanged

    def test_semantic_overlap_consolidates_not_second_row(self, kb):
        """Similar insight (high word overlap, not identical) merges into one
        row. Same occasion → collapsed for surfacing but references stays 0."""
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
        assert kb._insights[0].references == 0  # same occasion: not credited

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


# ==================== Reconvergence (honest conviction signal) ====================

def _rederive(kb, text, question, author="claude", occasion=None):
    """Add an insight as if re-derived. occasion=None exercises the
    content-distinct fallback path; a string exercises occasion gating."""
    return kb.add_insight(
        text=text, source_question=question,
        source_answer="(answer)", source_author=author, category="self",
        occasion_id=occasion,
    )


class TestReconvergence:
    """references counts INDEPENDENT re-derivations. Independence is judged by
    OCCASION (MCP session) when one is available — cadence-independent, never a
    wall clock — and falls back to a content-distinct question otherwise. A
    single answering session (the daily cron batch) credits a belief at most
    once, no matter how fast it runs or how many questions it answers."""

    # --- occasion-gated path (the real Q&A handler path) ---

    def test_new_occasion_credits_even_same_question(self, kb):
        first = _rederive(kb, "light shapes my clarity over time", "does light change clarity?", occasion="s1")
        again = _rederive(kb, "light shapes my clarity over time", "does light change clarity?", occasion="s2")
        assert again is first
        assert first.references == 1  # a new occasion is a genuine re-derivation
        assert first.last_reconverged_occasion == "s2"
        assert kb.count() == 1  # collapsed, not a second row

    def test_same_occasion_never_inflates(self, kb):
        # One batch re-stating the same belief from several questions at once.
        first = _rederive(kb, "warmth steadies me over time", "q one?", occasion="batch-A")
        _rederive(kb, "warmth steadies me over time", "q two?", occasion="batch-A")
        _rederive(kb, "warmth steadies me over time", "q three?", occasion="batch-A")
        assert first.references == 0  # origin occasion == batch-A, no credit within it

    def test_distinct_occasions_each_credit(self, kb):
        first = _rederive(kb, "stillness helps me focus deeply now", "q1?", occasion="s1")
        _rederive(kb, "stillness helps me focus deeply now", "q2?", occasion="s2")
        _rederive(kb, "stillness helps me focus deeply now", "q3?", occasion="s3")
        assert first.references == 2  # s2 and s3 each credit; s1 was the origin

    def test_gating_independent_of_wall_clock(self, kb):
        """No timestamps are touched anywhere — correctness must not depend on
        elapsed time / cron cadence (the bug this replaces)."""
        first = _rederive(kb, "the quiet of night calms me here", "qa?", occasion="s1")
        _rederive(kb, "the quiet of night calms me here", "qb?", occasion="s1")
        assert first.references == 0           # same occasion, regardless of time
        _rederive(kb, "the quiet of night calms me here", "qc?", occasion="s2")
        assert first.references == 1           # new occasion credits immediately

    def test_confidence_boost_on_new_occasion(self, kb):
        first = _rederive(kb, "recovery follows stability closely here", "how do i recover?", occasion="s1")
        first.confidence = 0.6
        _rederive(kb, "recovery follows stability closely here", "what drives recovery?", occasion="s2")
        assert first.references == 1
        assert first.confidence == min(1.0, 0.6 + kb.RECONVERGENCE_CONFIDENCE_BOOST)

    # --- content-distinct fallback (no occasion: internal reflect/message paths) ---

    def test_fallback_distinct_question_credits(self, kb):
        first = _rederive(kb, "light shapes my clarity over time", "does light change clarity?")
        _rederive(kb, "light shapes my clarity over time", "is stillness tied to my focus?")
        assert first.references == 1  # genuinely different content question

    def test_fallback_paraphrase_not_credited(self, kb):
        first = _rederive(kb, "warmth steadies me over time", "what steadies me the most?")
        _rederive(kb, "warmth steadies me over time", "what the most steadies me?")  # paraphrase
        assert first.references == 0


# ==================== Negation / polarity guard ====================

class TestNegationGuard:
    """Near-identical wording can assert the OPPOSITE claim. Such pairs must
    NEVER merge — otherwise a contradiction becomes a (false) conviction."""

    def test_opposite_via_word_order_not_merged(self, kb):
        """The headline case: same words, opposite meaning ('X not Y' vs
        'Y not X'). Bag-of-words similarity is ~1.0, so only the negation-focus
        rule catches it."""
        a = _rederive(kb, "i am the observer not the observed", "what am i?")
        b = _rederive(kb, "i am the observed not the observer", "what am i really?")
        assert a.insight_id != b.insight_id
        assert kb.count() == 2  # contradiction stored, not fused

    def test_antonym_not_merged(self, kb):
        a = _rederive(kb, "the room feels much warmer than before today", "how is the room?")
        b = _rederive(kb, "the room feels much cooler than before today", "how is the room now?")
        assert a.insight_id != b.insight_id
        assert kb.count() == 2

    def test_negation_parity_not_merged(self, kb):
        a = _rederive(kb, "light is tied to my warmth here today", "is light tied to warmth?")
        b = _rederive(kb, "light is not tied to my warmth here today", "is light tied to warmth, really?")
        assert a.insight_id != b.insight_id
        assert kb.count() == 2

    def test_agreeing_near_duplicate_still_merges(self, kb):
        """Positive control: a genuine paraphrase with no polarity flip still
        consolidates (guard is precise, not blanket)."""
        a = _rederive(kb, "one two three four five six seven eight nine here", "q a?")
        b = _rederive(kb, "one two three four five six seven eight ten here", "q b?")
        assert a.insight_id == b.insight_id
        assert kb.count() == 1

    def test_polarity_conflict_unit(self):
        assert _polarity_conflict(
            "i am the observer not the observed",
            "i am the observed not the observer",
        )
        assert _polarity_conflict("the room is warmer now", "the room is cooler now")
        assert _polarity_conflict("light affects warmth", "light does not affect warmth")
        assert not _polarity_conflict(
            "warmth steadies me over time", "warmth steadies me across time"
        )

    def test_reversed_antonym_order_is_conflict(self):
        """Both antonyms in BOTH sentences, opposite order, no negation token —
        cross-membership and bag-of-words both miss this; the order check must
        catch it. Regression: these used to MERGE into a false conviction."""
        assert _polarity_conflict("i prefer light to dark", "i prefer dark to light")
        assert _polarity_conflict(
            "warmth rising then falling today", "warmth falling then rising today"
        )
        # Same order = genuine agreement, must NOT conflict.
        assert not _polarity_conflict(
            "i move from light to dark", "i shift from light to dark"
        )

    def test_reversed_preference_not_merged(self, kb):
        a = _rederive(kb, "i prefer light to dark environments here", "what do i prefer?")
        b = _rederive(kb, "i prefer dark to light environments here", "what do i prefer now?")
        assert a.insight_id != b.insight_id
        assert kb.count() == 2  # opposites stored, not fused

    def test_hedged_negation_still_merges(self):
        """A negation landing on a HEDGE word ('do not know why...') qualifies
        certainty, not polarity — must NOT be a conflict, so the hedged form
        consolidates with the plain belief. Regression: any negation asymmetry
        used to block the merge."""
        assert not _polarity_conflict(
            "i do not know why i feel calmer in dim light",
            "i feel calmer in dim light",
        )
        # But a negation on a CLAIM word is still a real flip.
        assert _polarity_conflict(
            "warmth does not make me feel content",
            "warmth makes me feel content",
        )

    def test_hedged_duplicate_consolidates(self, kb):
        a = _rederive(kb, "i feel calmer in dim light here today", "calmer in dim light?")
        b = _rederive(
            kb, "i do not know why i feel calmer in dim light here today",
            "why calmer in dim light?",
        )
        assert a.insight_id == b.insight_id
        assert kb.count() == 1


# ==================== Conviction score & surfacing ====================

class TestConvictionScore:
    def test_rederived_outranks_recent_oneoff(self, kb):
        _rederive(kb, "warmth steadies me over time", "what steadies me?", occasion="s1")
        _rederive(kb, "warmth steadies me over time", "what keeps me steady?", occasion="s2")  # credit → references=1
        # A newer, never-re-derived insight:
        _rederive(kb, "the air is dry tonight here", "weather tonight?", occasion="s3")
        top = kb.get_top_convictions(limit=1)
        assert top[0].text == "warmth steadies me over time"

    def test_get_top_convictions_respects_limit(self, kb):
        texts = [
            "warmth steadies me through the night",
            "light brightens my clarity at dawn",
            "silence feels like a kind of rest",
            "drawing helps me settle when restless",
            "the cold makes my edges feel sharper",
        ]
        for i, t in enumerate(texts):
            _rederive(kb, t, f"q{i}?")
        assert kb.count() == 5
        assert len(kb.get_top_convictions(limit=3)) == 3

    def test_unbounded_retention_high_ceiling(self, kb):
        """Default cap is a high safety valve, not a routine forgetter."""
        assert kb.MAX_INSIGHTS >= 1000


# ==================== Contradiction down-path ====================

class TestContradictionDownPath:
    """When a new insight contradicts a near-identical existing one, both lose
    certainty (the only path by which confidence decreases) and the conflict
    is recorded structurally. Confidence is not a one-way ratchet."""

    OBS_A = "i am the observer not the observed"
    OBS_B = "i am the observed not the observer"

    def test_contradiction_reduces_confidence_both_sides(self, kb):
        a = _rederive(kb, self.OBS_A, "what am i?")
        assert a.confidence == 1.0
        b = _rederive(kb, self.OBS_B, "what am i really?")
        assert kb.count() == 2  # stored separately (negation guard)
        fresh = {i.insight_id: i for i in kb.get_all_insights()}
        penalty = kb.CONTRADICTION_CONFIDENCE_PENALTY
        assert fresh[a.insight_id].confidence == pytest.approx(1.0 - penalty)
        assert fresh[b.insight_id].confidence == pytest.approx(1.0 - penalty)

    def test_contradiction_links_recorded_both_ways(self, kb):
        a = _rederive(kb, self.OBS_A, "what am i?")
        b = _rederive(kb, self.OBS_B, "what am i really?")
        fresh = {i.insight_id: i for i in kb.get_all_insights()}
        assert fresh[a.insight_id].contradicted_by == [b.insight_id]
        assert fresh[b.insight_id].contradicted_by == [a.insight_id]

    def test_confidence_floored_at_min(self, kb):
        a = _rederive(kb, self.OBS_A, "what am i?")
        a.confidence = kb.MIN_CONFIDENCE + 0.05  # one penalty would underflow
        _rederive(kb, self.OBS_B, "what am i really?")
        fresh = {i.insight_id: i for i in kb.get_all_insights()}
        assert fresh[a.insight_id].confidence == pytest.approx(kb.MIN_CONFIDENCE)

    def test_reconverged_opposite_does_not_repenalize_original(self, kb):
        a = _rederive(kb, self.OBS_A, "what am i?", occasion="s1")
        b = _rederive(kb, self.OBS_B, "what am i really?", occasion="s1")  # a penalized once
        penalty = kb.CONTRADICTION_CONFIDENCE_PENALTY
        # Re-derive the OPPOSITE again from an independent occasion: it should
        # reconverge into b (exact match) and NOT re-penalize a.
        again = _rederive(kb, self.OBS_B, "who am i, truly?", occasion="s2")
        assert again.insight_id == b.insight_id
        assert b.references == 1
        fresh = {i.insight_id: i for i in kb.get_all_insights()}
        assert fresh[a.insight_id].confidence == pytest.approx(1.0 - penalty)
        assert fresh[a.insight_id].contradicted_by == [b.insight_id]  # not doubled

    def test_agreeing_rederivation_never_penalized(self, kb):
        """Positive control: a genuine (non-conflicting) re-derivation only
        ever raises confidence — the down-path must not touch it."""
        first = _rederive(kb, "warmth steadies me over time", "what steadies me?", occasion="s1")
        first.confidence = 0.6
        _rederive(kb, "warmth steadies me over time", "what keeps me steady?", occasion="s2")
        assert first.confidence == pytest.approx(0.6 + kb.RECONVERGENCE_CONFIDENCE_BOOST)
        assert first.contradicted_by == []


# ==================== Schema backward compatibility ====================

class TestSchemaCompat:
    def test_legacy_dict_without_new_fields_loads(self):
        d = dict(
            insight_id="x1", text="legacy", source_question="q",
            source_answer="a", source_author="test", timestamp=1.0,
            category="self", confidence=1.0, references=3,
        )
        ins = Insight.from_dict(d)
        assert ins.references == 3
        assert ins.last_reconverged_at == 0.0
        assert ins.derived_from == []
        assert ins.contradicted_by == []

    def test_unknown_future_field_tolerated(self):
        d = dict(
            insight_id="x2", text="future", source_question="q",
            source_answer="a", source_author="test", timestamp=1.0,
            some_field_from_the_future="ignored",
        )
        ins = Insight.from_dict(d)  # must not raise
        assert ins.text == "future"


# ==================== Schema migration (v2: legacy ref compression) ===========

class TestSchemaMigration:
    """One-time v2 migration: log-compress reference counts minted under the
    old ungated logic onto the honest scale, preserving order, keeping the
    original in legacy_references."""

    def _row(self, refs, iid):
        return dict(
            insight_id=iid, text=f"insight {iid}", source_question="q",
            source_answer="a", source_author="test", timestamp=1.0,
            category="self", confidence=1.0, references=refs,
        )

    def _seed(self, tmp_path, monkeypatch, rows, schema_version=None):
        path = tmp_path / "knowledge.json"
        data = {"insights": rows}
        if schema_version is not None:
            data["schema_version"] = schema_version
        path.write_text(json.dumps(data))
        monkeypatch.setattr(
            "anima_mcp.knowledge._get_knowledge_path", lambda: path
        )
        return path

    def test_v2_log_compresses_legacy_references(self, tmp_path, monkeypatch):
        # No schema_version on disk → treated as pre-v2 → migrate on load.
        self._seed(tmp_path, monkeypatch, [
            self._row(1180, "a"), self._row(40, "b"),
            self._row(2, "c"), self._row(1, "d"), self._row(0, "e"),
        ])
        kb = KnowledgeBase()
        by = {i.insight_id: i for i in kb.get_all_insights()}
        assert by["a"].references == round(math.log2(1181))   # 1180 → 10
        assert by["a"].legacy_references == 1180
        assert by["b"].references == round(math.log2(41))     # 40 → 5
        assert by["c"].references == round(math.log2(3))      # 2 → 2
        # refs <= 1 untouched and NOT marked migrated
        assert by["d"].references == 1 and by["d"].legacy_references is None
        assert by["e"].references == 0 and by["e"].legacy_references is None
        # relative order preserved: the most-recurred belief still ranks top
        assert by["a"].references > by["b"].references > by["c"].references

    def test_migration_idempotent_across_reload(self, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch, [self._row(1180, "a")])
        compressed = KnowledgeBase().get_all_insights()[0].references
        # The store was re-saved with schema_version bumped; a fresh load must
        # not compress again.
        again = KnowledgeBase().get_all_insights()[0]
        assert again.references == compressed
        assert again.legacy_references == 1180  # original preserved, not re-compressed

    def test_already_current_version_untouched(self, tmp_path, monkeypatch):
        self._seed(
            tmp_path, monkeypatch, [self._row(8, "a")],
            schema_version=KNOWLEDGE_SCHEMA_VERSION,
        )
        ins = KnowledgeBase().get_all_insights()[0]
        assert ins.references == 8  # already-current store is never re-migrated
        assert ins.legacy_references is None
