"""Tests for messages module — MessageBoard storage, dedup, trimming, Q&A."""

import time
from unittest.mock import patch

import pytest

import anima_mcp.messages as msg_module
from anima_mcp.messages import (
    MessageBoard, Message,
    MESSAGE_TYPE_OBSERVATION, MESSAGE_TYPE_QUESTION,
    MESSAGE_TYPE_USER,
    MAX_QUESTIONS_PER_ROLLING_DAY,
    MAX_UNANSWERED_QUESTIONS_SOFT_CAP,
    questions_similar,
)


@pytest.fixture(autouse=True)
def isolated_board(tmp_path, monkeypatch):
    """Redirect file path and reset singleton for every test."""
    msgs_file = tmp_path / "messages.json"
    monkeypatch.setattr(msg_module, "_board", None)
    monkeypatch.setattr(msg_module, "_get_persistent_path", lambda: msgs_file)
    yield msgs_file
    monkeypatch.setattr(msg_module, "_board", None)


@pytest.fixture
def board(isolated_board):
    return MessageBoard()


class TestMessageBoardInit:
    def test_starts_empty(self, board):
        assert board.get_recent(100) == []

    def test_file_created_on_first_add(self, board, isolated_board):
        board.add_message("hello", MESSAGE_TYPE_USER)
        assert isolated_board.exists()


class TestAddAndRetrieve:
    def test_add_message_returns_message(self, board):
        msg = board.add_message("test text", MESSAGE_TYPE_USER, author="tester")
        assert isinstance(msg, Message)
        assert msg.text == "test text"
        assert msg.msg_type == MESSAGE_TYPE_USER
        assert msg.author == "tester"
        assert len(msg.message_id) > 0

    def test_get_recent_returns_newest_first(self, board):
        board.add_message("first", MESSAGE_TYPE_USER)
        board.add_message("second", MESSAGE_TYPE_USER)
        board.add_message("third", MESSAGE_TYPE_USER)
        recent = board.get_recent(2)
        assert len(recent) == 2
        # get_recent returns newest first
        assert recent[0].text == "third"
        assert recent[1].text == "second"

    def test_message_persists_across_reload(self, board, isolated_board):
        board.add_message("persisted", MESSAGE_TYPE_USER)
        board2 = MessageBoard()
        msgs = board2.get_recent(100)
        assert any(m.text == "persisted" for m in msgs)

    def test_add_user_message(self, board):
        msg = board.add_user_message("user says hello")
        assert msg.msg_type == MESSAGE_TYPE_USER
        assert msg.author == "user"


class TestObservationDedup:
    def test_rate_limited(self, board):
        """Two rapid observations: second should be rate-limited."""
        r1 = board.add_observation("first observation")
        r2 = board.add_observation("different observation")
        assert r1 is not None
        assert r2 is None  # Rate limited (within 5 min)

    def test_no_rate_limit_after_gap(self, board):
        """Observations spaced apart should both succeed."""
        board.add_observation("first observation")
        # Manually age the last observation
        for m in board._messages:
            if m.msg_type == MESSAGE_TYPE_OBSERVATION:
                m.timestamp -= 400  # 6+ minutes ago
        r2 = board.add_observation("second observation")
        assert r2 is not None


class TestQuestions:
    def test_add_question(self, board):
        q = board.add_question("What is light?", author="lumen", context="curiosity")
        assert q is not None
        assert q.msg_type == MESSAGE_TYPE_QUESTION
        assert not q.answered
        assert q.context == "curiosity"

    def test_question_rate_limited(self, board):
        q1 = board.add_question("First question?")
        q2 = board.add_question("Second question?")
        assert q1 is not None
        assert q2 is None  # Within minimum interval

    def test_question_backlog_soft_cap(self, board):
        """No new questions while many are already unanswered (reduces churn)."""
        base_t = time.time() - 10_000
        for i in range(MAX_UNANSWERED_QUESTIONS_SOFT_CAP):
            board._messages.append(
                Message(
                    message_id=f"pend{i}",
                    text=f"Pending {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=base_t + i * 1000,
                    author="lumen",
                    answered=False,
                )
            )
        board._save()
        q = board.add_question("Should not post while backlog full?")
        assert q is None

    def test_question_backlog_soft_cap_ignores_expired_questions(self, board):
        """Expired questions should not keep the soft cap stuck forever."""
        base_t = time.time() - msg_module.QUESTION_BUDGET_WINDOW_SECONDS - 6_000
        stale_question_ids = []
        for i in range(MAX_UNANSWERED_QUESTIONS_SOFT_CAP):
            qid = f"stale{i}"
            stale_question_ids.append(qid)
            board._messages.append(
                Message(
                    message_id=qid,
                    text=f"Stale pending {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=base_t + i * 1000,
                    author="lumen",
                    answered=False,
                )
            )
        board._save()

        q = board.add_question("Can the board recover after stale backlog?")

        assert q is not None
        assert q.text == "Can the board recover after stale backlog?"
        stale = [m for m in board._messages if m.message_id in stale_question_ids]
        assert stale
        assert all(not m.answered and m.expired_at is not None for m in stale)

    def test_rolling_daily_budget_does_not_reopen_when_questions_are_answered(self, board):
        """Answers clear backlog pressure, but they do not buy more questions."""
        base_t = time.time() - 3600
        for i in range(MAX_QUESTIONS_PER_ROLLING_DAY):
            board._messages.append(
                Message(
                    message_id=f"answered{i}",
                    text=f"Answered {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=base_t + i,
                    author="lumen",
                    answered=True,
                )
            )
        board._save()

        assert board.add_question("Would an answer reopen the budget?") is None

    def test_reserve_budget_slots_preserves_headroom_for_unreserved_callers(self, board):
        """Reserved callers yield the final daily slot to unreserved callers."""
        base_t = time.time() - 3600
        for i in range(MAX_QUESTIONS_PER_ROLLING_DAY - 1):
            board._messages.append(
                Message(
                    message_id=f"prior{i}",
                    text=f"Prior {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=base_t - (i + 1) * 1200,
                    author="lumen",
                )
            )
        board._save()

        # Contemplative-style reserved post yields the final slot.
        assert (
            board.add_question(
                "Quiet wondering?",
                context="a quiet moment",
                reserve_budget_slots=1,
            )
            is None
        )
        # Surprise/drive unreserved post can still claim the final slot.
        surprise_q = board.add_question("Why did temperature jump?", context="surprise=0.50")
        assert surprise_q is not None

    def test_max_same_context_per_day_limits_repeated_quiet_prompts(self, board):
        """A per-context daily cap holds after expiry within 24h and resets past 24h."""
        old_enough_to_expire = time.time() - (msg_module.QUESTION_EXPIRY_SECONDS + 600)
        board._messages.append(
            Message(
                message_id="quiet-old",
                text="Everything is steady?",
                msg_type=MESSAGE_TYPE_QUESTION,
                timestamp=old_enough_to_expire,
                author="lumen",
                context="a quiet moment",
            )
        )
        board._save()

        # Blocked even though the prior quiet question has already expired.
        assert (
            board.add_question(
                "What pattern have I stopped seeing?",
                context="a quiet moment",
                max_same_context_per_day=1,
            )
            is None
        )
        # Different context still posts within the remaining rolling-day budget.
        assert (
            board.add_question("Temperature jumped?", context="surprise=0.45")
            is not None
        )

        # Once the prior quiet question ages past the 24h window, the context slot reopens.
        board._messages[0].timestamp = time.time() - (msg_module.QUESTION_BUDGET_WINDOW_SECONDS + 60)
        board._messages[-1].timestamp = time.time() - (msg_module.MIN_QUESTION_INTERVAL_SECONDS + 60)
        board._save()
        reopened = board.add_question(
            "When the world holds still, what remains?",
            context="a quiet moment",
            max_same_context_per_day=1,
        )
        assert reopened is not None

    def test_mixed_order_quiet_first_and_rolling_window_recovery(self, board):
        """Quiet first + 2 priority posts fill the 3/day budget cleanly without starvation or bypass."""
        q_quiet = board.add_question(
            "Everything is steady?",
            context=msg_module.CONTEXT_QUIET_MOMENT,
            reserve_budget_slots=1,
            max_same_context_per_day=1,
        )
        assert q_quiet is not None
        q_quiet.timestamp = time.time() - 3 * 3600

        q_p1 = board.add_question("Why did light spike?", context="surprise=0.50")
        assert q_p1 is not None
        q_p1.timestamp = time.time() - 2 * 3600

        # A second quiet attempt in the same day is blocked, leaving the 3rd slot free.
        assert (
            board.add_question(
                "Room is quiet again?",
                context=msg_module.CONTEXT_QUIET_MOMENT,
                reserve_budget_slots=1,
                max_same_context_per_day=1,
            )
            is None
        )

        # Second priority post claims the 3rd daily slot.
        q_p2 = board.add_question("Temperature jumped?", context="surprise=0.45")
        assert q_p2 is not None
        q_p2.timestamp = time.time() - 3600

        # Budget of 3 is now full until the oldest question ages past 24h.
        assert board.add_question("Third priority too soon?", context="surprise=0.60") is None

        q_quiet.timestamp = time.time() - (msg_module.QUESTION_BUDGET_WINDOW_SECONDS + 60)
        q_p2.timestamp = time.time() - (msg_module.MIN_QUESTION_INTERVAL_SECONDS + 60)
        board._save()
        assert board.add_question("Third priority after slot ages out?", context="surprise=0.60") is not None

    def test_reserve_budget_slots_and_context_edge_cases(self, board, monkeypatch):
        """Reserve applies to unanswered soft cap, >=MAX blocks, None context skips cap, wrapper forwards."""
        monkeypatch.setattr(msg_module, "MAX_QUESTIONS_PER_ROLLING_DAY", 10)
        for i in range(MAX_UNANSWERED_QUESTIONS_SOFT_CAP - 1):
            board._messages.append(
                Message(
                    message_id=f"u{i}",
                    text=f"Backlog {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=time.time() - 3600 + i * 10,
                    author="lumen",
                    answered=False,
                )
            )
        board._save()

        # Reserved caller yields the last unanswered backlog slot.
        assert (
            board.add_question(
                "Quiet backlog check?",
                context=msg_module.CONTEXT_QUIET_MOMENT,
                reserve_budget_slots=1,
            )
            is None
        )
        # reserve_budget_slots >= MAX_QUESTIONS_PER_ROLLING_DAY blocks all posts cleanly.
        assert board.add_question("Over-reserved?", reserve_budget_slots=10) is None
        # max_same_context_per_day is a no-op when context=None; unreserved post succeeds.
        posted = board.add_question("Priority during backlog?", context=None, max_same_context_per_day=0)
        assert posted is not None

        # Module-level add_question wrapper forwards both keyword-only arguments.
        seen = {}

        class _StubBoard:
            def add_question(self, text, author="lumen", context=None, **kwargs):
                seen.update({"text": text, "author": author, "context": context, **kwargs})
                return "ok"

        monkeypatch.setattr(msg_module, "get_board", lambda: _StubBoard())
        assert (
            msg_module.add_question(
                "Wrapper forward?",
                context=msg_module.CONTEXT_QUIET_MOMENT,
                reserve_budget_slots=1,
                max_same_context_per_day=1,
            )
            == "ok"
        )
        assert seen == {
            "text": "Wrapper forward?",
            "author": "lumen",
            "context": msg_module.CONTEXT_QUIET_MOMENT,
            "reserve_budget_slots": 1,
            "max_same_context_per_day": 1,
        }

    def test_questions_similar_public_helper_matches_board_behavior(self, board):
        """Question similarity is available without reaching into MessageBoard internals."""
        q1 = "why is it that I now know that dim light changes my attention?"
        q2 = "why is it that I learned that dim light changes my attention?"

        assert questions_similar(q1, q2)
        assert board._questions_similar(q1, q2) == questions_similar(q1, q2)

    def test_told_stem_strips_like_learned_stem(self):
        """A reported claim ("i was told that ...") reduces to the same core
        as the legacy "i learned that ..." wording, so both wrappers are
        recognized as the same question rather than new ones."""
        q1 = "why is it that I was told that dim light changes my attention?"
        q2 = "why is it that I learned that dim light changes my attention?"
        assert questions_similar(q1, q2)

    def test_unanswered_questions(self, board):
        q = board.add_question("Unanswered question?")
        unanswered = board.get_unanswered_questions(auto_expire=False)
        assert any(m.message_id == q.message_id for m in unanswered)

    def test_answering_removes_from_unanswered(self, board):
        q = board.add_question("Will someone answer?")
        board.add_agent_message("Yes!", agent_name="helper", responds_to=q.message_id)
        unanswered = board.get_unanswered_questions(auto_expire=False)
        assert not any(m.message_id == q.message_id for m in unanswered)

    def test_lumen_self_answer_is_not_extracted_back_into_knowledge(self, board):
        """A deterministic self-answer must not become a new circular claim."""
        q = board.add_question("What pattern do I notice?")
        with patch("anima_mcp.knowledge._extract_simple_insight") as mock_extract:
            board.add_agent_message(
                "The stored observations show a pattern.",
                agent_name="lumen",
                responds_to=q.message_id,
            )

        mock_extract.assert_not_called()

    def test_answering_marks_growth_curiosity_explored(self, board, monkeypatch):
        """When a question Lumen asked gets answered, the matching growth
        curiosity should be marked explored — that's how "find an answer to: X"
        goals auto-complete."""
        calls = []

        class FakeGrowth:
            def mark_curiosity_explored(self, question, notes=None):
                calls.append((question, notes))

        fake = FakeGrowth()
        monkeypatch.setattr("anima_mcp.accessors._get_growth", lambda: fake)

        q = board.add_question("Why is it so dim?", author="lumen")
        board.add_agent_message("It's night", agent_name="human", responds_to=q.message_id)
        assert calls == [("Why is it so dim?", "It's night")]

    def test_answering_unknown_question_does_not_touch_growth(self, board, monkeypatch):
        """If the responds_to ID doesn't match any question, we don't guess at
        which curiosity to mark explored."""
        calls = []

        class FakeGrowth:
            def mark_curiosity_explored(self, question, notes=None):
                calls.append((question, notes))

        monkeypatch.setattr("anima_mcp.accessors._get_growth", lambda: FakeGrowth())

        board.add_agent_message("answer", agent_name="human", responds_to="nonexistent-id")
        assert calls == []

    def test_answering_survives_growth_unavailable(self, board, monkeypatch):
        """Growth singleton may not be initialized during boot; answering a
        question must not crash the message path."""
        monkeypatch.setattr("anima_mcp.accessors._get_growth", lambda: None)
        q = board.add_question("Does the void answer?", author="lumen")
        # Should not raise
        board.add_agent_message("Yes", agent_name="human", responds_to=q.message_id)
        unanswered = board.get_unanswered_questions(auto_expire=False)
        assert not any(m.message_id == q.message_id for m in unanswered)

    def test_real_growth_system_has_method_message_path_calls(self):
        """Guard against fake/real drift: the method messages.py invokes when a
        question is answered must actually exist on the real GrowthSystem.
        (A typo here — mark_explored vs mark_curiosity_explored — failed
        silently in production while passing against the test double.)"""
        from anima_mcp.growth.base import GrowthSystem
        assert hasattr(GrowthSystem, "mark_curiosity_explored")


class TestTrimming:
    def test_observations_trimmed(self, board):
        """Adding more than MAX_OBSERVATIONS should trim old ones."""
        for i in range(board.MAX_OBSERVATIONS + 10):
            board.add_message(f"obs {i}", MESSAGE_TYPE_OBSERVATION)
        obs = [m for m in board.get_recent(1000) if m.msg_type == MESSAGE_TYPE_OBSERVATION]
        assert len(obs) <= board.MAX_OBSERVATIONS

    def test_questions_not_affected_by_observation_overflow(self, board):
        q = board.add_message("my question", MESSAGE_TYPE_QUESTION)
        for i in range(board.MAX_OBSERVATIONS + 10):
            board.add_message(f"obs {i}", MESSAGE_TYPE_OBSERVATION)
        all_msgs = board.get_recent(1000)
        assert any(m.message_id == q.message_id for m in all_msgs)


class TestMessageSerialization:
    def test_to_dict_roundtrip(self):
        msg = Message(
            message_id="abc123",
            text="test",
            msg_type=MESSAGE_TYPE_USER,
            timestamp=time.time(),
            author="user",
            context="some context",
        )
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.text == msg.text
        assert restored.message_id == msg.message_id
        assert restored.author == msg.author
        assert restored.context == msg.context

    def test_from_dict_fills_defaults(self):
        """Old messages missing new fields should get defaults."""
        d = {"text": "old msg", "msg_type": "user", "timestamp": 1000}
        msg = Message.from_dict(d)
        assert msg.message_id  # Auto-generated
        assert msg.author is None
        assert msg.responds_to is None
        assert msg.answered is False

    def test_age_str(self):
        msg = Message(message_id="x", text="t", msg_type="user", timestamp=time.time())
        assert msg.age_str() == "now"

    def test_is_question(self):
        q = Message(message_id="x", text="?", msg_type=MESSAGE_TYPE_QUESTION, timestamp=0)
        u = Message(message_id="y", text="!", msg_type=MESSAGE_TYPE_USER, timestamp=0)
        assert q.is_question()
        assert not u.is_question()


class TestSingleton:
    def test_get_board_returns_same_instance(self, isolated_board):
        b1 = msg_module.get_board()
        b2 = msg_module.get_board()
        assert b1 is b2

    def test_convenience_add_observation(self, isolated_board):
        """Module-level add_observation should work through singleton."""
        result = msg_module.add_observation("lumen observation")
        # May return None if rate-limited, or Message otherwise
        # Just check it doesn't crash
        assert result is None or isinstance(result, Message)


class TestQAProvenanceAndTruncation:
    """Guards for the 2026-07-31 malformed-question chain.

    The retired behavioral bridge filed Q&A claims as "From Q&A: <text>". That
    marker is storage provenance, not something Lumen said, and it was surviving
    into generated questions — the generator wrapped a template around it
    instead of stripping it, producing:

        "why is it that from q&a: i now know that the connection between
         temperature?"

    which is both stacked boilerplate AND a claim cut off mid-sentence.
    """

    def test_provenance_prefix_is_stripped(self):
        core = msg_module._question_semantic_core(
            "from q&a: i now know that the connection between temperature and clarity"
        )
        assert not core.startswith("from q&a")
        # The stripper loops, so the stem underneath comes off on the next pass.
        assert not core.startswith("i now know that")
        assert core == "the connection between temperature and clarity"

    def test_provenance_stripping_is_case_insensitive_via_normalisation(self):
        # Callers normalise to lowercase before stripping; confirm the stored
        # capitalisation ("From Q&A: ") reduces to the same core.
        stored = "From Q&A: I learned that drawing in bright light helps"
        core = msg_module._question_semantic_core(stored.lower())
        assert core == "drawing in bright light helps"

    def test_looks_truncated_detects_ellipsis(self):
        assert msg_module._looks_truncated("the connection between\u2026")
        assert msg_module._looks_truncated("the connection between...")
        assert msg_module._looks_truncated("  trailing space then ellipsis \u2026  ")

    def test_looks_truncated_passes_complete_claims(self):
        assert not msg_module._looks_truncated("warmth makes me feel content")
        assert not msg_module._looks_truncated("i draw at night.")
        assert not msg_module._looks_truncated("")


class TestAnsweredQuestionTexts:
    """``get_answered_question_texts`` must mean REALLY answered.

    Expiry and answers now have separate durable fields. The answer link remains
    the authoritative compatibility check.
    """

    def test_real_answer_counts(self, board):
        q = board.add_question("does the light change how I settle?")
        board.add_agent_message("It does, measurably.", agent_name="helper",
                                responds_to=q.message_id)
        assert board.get_answered_question_texts() == {q.text}

    def test_expiry_does_not_count(self, board):
        """The distinction the whole change rests on."""
        import time
        q = board.add_question("what am I not noticing yet?")
        q.timestamp = time.time() - (msg_module.QUESTION_EXPIRY_SECONDS + 60)
        board._expire_old_questions(time.time())
        assert q.answered is False
        assert q.expired_at is not None
        assert board.get_answered_question_texts() == set()

    def test_unanswered_question_absent(self, board):
        board.add_question("is anyone there?")
        assert board.get_answered_question_texts() == set()

    def test_answer_to_another_question_does_not_leak(self, board):
        a = board.add_question("first question about the room?")
        # Clear MIN_QUESTION_INTERVAL_SECONDS so the second is accepted.
        a.timestamp -= msg_module.MIN_QUESTION_INTERVAL_SECONDS + 60
        b = board.add_question("second question about myself?")
        assert b is not None
        board.add_agent_message("re: first", agent_name="helper",
                                responds_to=a.message_id)
        texts = board.get_answered_question_texts()
        assert a.text in texts
        assert b.text not in texts

    def test_lumen_self_answer_keeps_question_open_for_external_qa(self, board, monkeypatch):
        calls = []

        class FakeGrowth:
            def mark_curiosity_explored(self, question, notes=None):
                calls.append((question, notes))

        monkeypatch.setattr("anima_mcp.accessors._get_growth", lambda: FakeGrowth())
        q = board.add_question("What would I notice if I attended to something I usually ignore?")
        board.add_agent_message(
            "Across 85 observations, stability averaged 0.562.",
            agent_name="lumen",
            responds_to=q.message_id,
        )

        assert board.get_by_id(q.message_id).answered is False
        assert any(m.message_id == q.message_id for m in board.get_unanswered_questions(auto_expire=False))
        assert board.get_answered_question_texts() == set()
        assert calls == []

    def test_repair_orphaned_answered_reopens_self_answered_questions(self, board):
        q = board.add_question("How does warmth change after midnight?")
        q.answered = True
        ans = board.add_message("My recorded warmth was 0.541.", msg_module.MESSAGE_TYPE_AGENT, author="lumen")
        ans.responds_to = q.message_id
        board._save()

        repaired = board.repair_orphaned_answered()
        assert repaired == 1
        assert board.get_by_id(q.message_id).answered is False

    def test_self_answers_do_not_evict_regular_visitors(self, board):
        base_t = time.time() - 5000
        for i in range(board.MAX_VISITORS):
            board._messages.append(
                Message(
                    message_id=f"vis{i}",
                    text=f"Visitor hello {i}",
                    msg_type=MESSAGE_TYPE_USER,
                    timestamp=base_t + i,
                    author="kenny",
                )
            )
        for i in range(board.MAX_QUESTIONS):
            qid = f"q{i}"
            board._messages.append(
                Message(
                    message_id=qid,
                    text=f"Question {i}?",
                    msg_type=MESSAGE_TYPE_QUESTION,
                    timestamp=base_t + 100 + i,
                    author="lumen",
                )
            )
            board._messages.append(
                Message(
                    message_id=f"self{i}",
                    text=f"Self answer {i}",
                    msg_type=msg_module.MESSAGE_TYPE_AGENT,
                    timestamp=base_t + 200 + i,
                    author="lumen",
                    responds_to=qid,
                )
            )
        board._trim_by_type()

        regular_visitors = [
            m for m in board._messages
            if m.msg_type == MESSAGE_TYPE_USER
        ]
        assert len(regular_visitors) == board.MAX_VISITORS

