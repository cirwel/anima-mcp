"""Extended tests for messages module — covering uncovered branches."""

import time
import pytest

from anima_mcp.messages import (
    Message,
    MessageBoard,
    MESSAGE_TYPE_OBSERVATION,
    MESSAGE_TYPE_QUESTION,
    MESSAGE_TYPE_USER,
    MESSAGE_TYPE_AGENT,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anima_mcp.messages._get_persistent_path",
        lambda: tmp_path / "messages.json",
    )
    return MessageBoard()


# ---------------------------------------------------------------------------
# 1. Message.age_str()
# ---------------------------------------------------------------------------
class TestAgeStr:
    def test_now(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 10)
        assert msg.age_str() == "now"

    def test_minutes_ago(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 300)
        assert msg.age_str() == "5m ago"

    def test_hours_ago(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 7200)
        assert msg.age_str() == "2h ago"

    def test_days_ago(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 172800)
        assert msg.age_str() == "2d ago"

    def test_boundary_59s_is_now(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 59)
        assert msg.age_str() == "now"

    def test_boundary_60s_is_1m(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 60)
        assert msg.age_str() == "1m ago"

    def test_boundary_3599s_is_minutes(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 3599)
        assert "m ago" in msg.age_str()

    def test_boundary_3600s_is_1h(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 3600)
        assert msg.age_str() == "1h ago"

    def test_boundary_86399s_is_hours(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 86399)
        assert "h ago" in msg.age_str()

    def test_boundary_86400s_is_1d(self):
        msg = Message(message_id="a", text="t", msg_type="user", timestamp=time.time() - 86400)
        assert msg.age_str() == "1d ago"


# ---------------------------------------------------------------------------
# 2. Message.from_dict() with missing fields
# ---------------------------------------------------------------------------
class TestFromDictMissingFields:
    def test_missing_message_id(self):
        d = {"text": "hello", "msg_type": "user", "timestamp": 1000.0}
        msg = Message.from_dict(d)
        assert msg.message_id  # auto-generated uuid prefix
        assert len(msg.message_id) == 8

    def test_missing_author(self):
        d = {"message_id": "abc", "text": "hi", "msg_type": "user", "timestamp": 1.0}
        msg = Message.from_dict(d)
        assert msg.author is None

    def test_missing_responds_to(self):
        d = {"message_id": "abc", "text": "hi", "msg_type": "agent", "timestamp": 1.0, "author": "bot"}
        msg = Message.from_dict(d)
        assert msg.responds_to is None

    def test_missing_answered(self):
        d = {"message_id": "abc", "text": "hi", "msg_type": "question", "timestamp": 1.0}
        msg = Message.from_dict(d)
        assert msg.answered is False

    def test_missing_context(self):
        d = {"message_id": "abc", "text": "hi", "msg_type": "question", "timestamp": 1.0}
        msg = Message.from_dict(d)
        assert msg.context is None

    def test_all_fields_missing_except_required(self):
        d = {"text": "bare", "msg_type": "observation", "timestamp": 0.0}
        msg = Message.from_dict(d)
        assert msg.message_id
        assert msg.author is None
        assert msg.responds_to is None
        assert msg.answered is False
        assert msg.context is None


# ---------------------------------------------------------------------------
# 3. MessageBoard._questions_similar()
# ---------------------------------------------------------------------------
class TestQuestionsSimilar:
    def test_exact_match(self, board):
        assert board._questions_similar("What is light?", "what is light")

    def test_exact_match_with_punctuation(self, board):
        assert board._questions_similar("What is warmth?!", "what is warmth")

    def test_high_jaccard_similar(self, board):
        # 4 shared words out of 5 total = 0.8 jaccard
        assert board._questions_similar(
            "why did the light change suddenly",
            "why did the light change quickly",
        )

    def test_low_jaccard_not_similar(self, board):
        assert not board._questions_similar(
            "what is the meaning of life",
            "how do sensors measure temperature",
        )

    def test_stem_match_what_connects(self, board):
        assert board._questions_similar(
            "what connects warmth and light",
            "what connects clarity and pressure",
        )

    def test_stem_match_why_did_so_many(self, board):
        assert board._questions_similar(
            "why did so many readings change",
            "why did so many values spike",
        )

    def test_no_similarity(self, board):
        assert not board._questions_similar(
            "hello world",
            "goodbye universe",
        )

    def test_empty_words(self, board):
        # After normalization and splitting, both become empty-ish
        assert board._questions_similar("", "")

    def test_one_empty(self, board):
        assert not board._questions_similar("what is light", "")


# ---------------------------------------------------------------------------
# 4. MessageBoard._compute_question_feedback()
# ---------------------------------------------------------------------------
class TestComputeQuestionFeedback:
    def test_base_score(self, board):
        fb = board._compute_question_feedback("question?", "short reply", None)
        assert fb["score"] == pytest.approx(0.5)
        assert fb["signals"] == []

    def test_long_response_bonus(self, board):
        response = "x" * 151
        fb = board._compute_question_feedback("q?", response, None)
        assert fb["score"] == pytest.approx(0.65)
        assert "long_response" in fb["signals"]

    def test_very_long_response_bonus(self, board):
        response = "x" * 301
        fb = board._compute_question_feedback("q?", response, None)
        # 0.5 + 0.15 (long) + 0.1 (very long) = 0.75
        assert fb["score"] == pytest.approx(0.75)
        assert "long_response" in fb["signals"]
        assert "very_long_response" in fb["signals"]

    def test_confusion_marker_penalty(self, board):
        fb = board._compute_question_feedback("q?", "I don't understand what you mean", None)
        # 0.5 - 0.2 = 0.3
        assert fb["score"] == pytest.approx(0.3)
        assert any("confusion:" in s for s in fb["signals"])

    def test_multiple_confusion_markers(self, board):
        fb = board._compute_question_feedback("q?", "unclear and broken text", None)
        # 0.5 - 0.2 - 0.2 = 0.1
        assert fb["score"] == pytest.approx(0.1)

    def test_questions_back_penalty(self, board):
        fb = board._compute_question_feedback("q?", "What? Really? Are you sure?", None)
        # 0.5 - 0.1 = 0.4 (3 question marks > 1)
        assert fb["score"] == pytest.approx(0.4)
        assert "questions_back" in fb["signals"]

    def test_single_question_mark_no_penalty(self, board):
        fb = board._compute_question_feedback("q?", "What do you think?", None)
        # Only 1 question mark, no penalty
        assert "questions_back" not in fb["signals"]

    def test_agency_context_penalty(self, board):
        fb = board._compute_question_feedback("q?", "ok", "agency_generated")
        # 0.5 - 0.1 = 0.4
        assert fb["score"] == pytest.approx(0.4)
        assert "agency_generated" in fb["signals"]

    def test_clamp_floor(self, board):
        # Stack enough penalties to go below 0
        fb = board._compute_question_feedback(
            "q?",
            "unclear broken malformed don't understand? really?",
            "agency",
        )
        assert fb["score"] >= 0.0

    def test_clamp_ceiling(self, board):
        # Very long response can't exceed 1.0
        fb = board._compute_question_feedback("q?", "x" * 500, None)
        assert fb["score"] <= 1.0

    def test_feedback_has_lengths(self, board):
        fb = board._compute_question_feedback("my question?", "my answer", None)
        assert fb["response_length"] == len("my answer")
        assert fb["question_length"] == len("my question?")


# ---------------------------------------------------------------------------
# 5. MessageBoard.get_by_id()
# ---------------------------------------------------------------------------
class TestGetById:
    def test_found(self, board):
        msg = board.add_message("findme", MESSAGE_TYPE_USER)
        found = board.get_by_id(msg.message_id)
        assert found is not None
        assert found.text == "findme"

    def test_not_found(self, board):
        assert board.get_by_id("nonexistent") is None

    def test_found_among_many(self, board):
        for i in range(10):
            board.add_message(f"msg {i}", MESSAGE_TYPE_USER)
        target = board.add_message("target", MESSAGE_TYPE_USER)
        for i in range(10):
            board.add_message(f"msg {i + 10}", MESSAGE_TYPE_USER)
        assert board.get_by_id(target.message_id).text == "target"


# ---------------------------------------------------------------------------
# 6. MessageBoard.delete_by_id()
# ---------------------------------------------------------------------------
class TestDeleteById:
    def test_delete_existing(self, board):
        msg = board.add_message("delete me", MESSAGE_TYPE_USER)
        assert board.delete_by_id(msg.message_id) is True
        assert board.get_by_id(msg.message_id) is None

    def test_delete_nonexistent(self, board):
        assert board.delete_by_id("nope") is False

    def test_delete_only_removes_target(self, board):
        m1 = board.add_message("keep", MESSAGE_TYPE_USER)
        m2 = board.add_message("remove", MESSAGE_TYPE_USER)
        board.delete_by_id(m2.message_id)
        assert board.get_by_id(m1.message_id) is not None
        assert board.get_by_id(m2.message_id) is None


# ---------------------------------------------------------------------------
# 7. MessageBoard.clear()
# ---------------------------------------------------------------------------
class TestClear:
    def test_clears_all(self, board):
        board.add_message("a", MESSAGE_TYPE_USER)
        board.add_message("b", MESSAGE_TYPE_OBSERVATION)
        board.add_message("c", MESSAGE_TYPE_QUESTION)
        board.clear()
        assert board.get_recent(100) == []

    def test_clear_persists(self, board, tmp_path, monkeypatch):
        board.add_message("x", MESSAGE_TYPE_USER)
        board.clear()
        # New board from same file should also be empty
        board2 = MessageBoard()
        assert board2.get_recent(100) == []


# ---------------------------------------------------------------------------
# 8. MessageBoard.get_recent() with msg_type filter
# ---------------------------------------------------------------------------
class TestGetRecentWithFilter:
    def test_filter_by_type(self, board):
        board.add_message("obs", MESSAGE_TYPE_OBSERVATION)
        board.add_message("usr", MESSAGE_TYPE_USER)
        board.add_message("agt", MESSAGE_TYPE_AGENT)
        recent = board.get_recent(10, msg_type=MESSAGE_TYPE_USER)
        assert len(recent) == 1
        assert recent[0].text == "usr"

    def test_filter_returns_empty_if_no_match(self, board):
        board.add_message("obs", MESSAGE_TYPE_OBSERVATION)
        recent = board.get_recent(10, msg_type=MESSAGE_TYPE_QUESTION)
        assert recent == []

    def test_filter_respects_limit_window(self, board):
        # Add 5 observations then 5 user messages
        for i in range(5):
            board.add_message(f"obs {i}", MESSAGE_TYPE_OBSERVATION)
        for i in range(5):
            board.add_message(f"usr {i}", MESSAGE_TYPE_USER)
        # limit=3 takes last 3 messages (all user), then filters
        recent = board.get_recent(3, msg_type=MESSAGE_TYPE_OBSERVATION)
        assert recent == []  # Last 3 are all user messages

    def test_no_filter_returns_all_types(self, board):
        board.add_message("obs", MESSAGE_TYPE_OBSERVATION)
        board.add_message("usr", MESSAGE_TYPE_USER)
        recent = board.get_recent(10)
        assert len(recent) == 2


# ---------------------------------------------------------------------------
# 9. MessageBoard.get_messages_for_lumen()
# ---------------------------------------------------------------------------
class TestGetMessagesForLumen:
    def test_returns_user_and_agent(self, board):
        board.add_message("from user", MESSAGE_TYPE_USER, author="user")
        board.add_message("from agent", MESSAGE_TYPE_AGENT, author="helper")
        msgs = board.get_messages_for_lumen()
        assert len(msgs) == 2

    def test_excludes_lumen_own(self, board):
        board.add_message("from user", MESSAGE_TYPE_USER, author="user")
        board.add_message("lumen obs", MESSAGE_TYPE_OBSERVATION, author="lumen")
        board.add_message("lumen agent", MESSAGE_TYPE_AGENT, author="lumen")
        msgs = board.get_messages_for_lumen()
        # Only user message; lumen's agent message excluded, observation excluded by type
        assert len(msgs) == 1
        assert msgs[0].author == "user"

    def test_excludes_observations_and_questions(self, board):
        board.add_message("obs", MESSAGE_TYPE_OBSERVATION, author="lumen")
        board.add_message("question", MESSAGE_TYPE_QUESTION, author="lumen")
        msgs = board.get_messages_for_lumen()
        assert msgs == []

    def test_since_timestamp(self, board):
        old = board.add_message("old", MESSAGE_TYPE_USER, author="user")
        old_ts = old.timestamp
        # Make the next message slightly newer
        board.add_message("new", MESSAGE_TYPE_USER, author="user")
        msgs = board.get_messages_for_lumen(since_timestamp=old_ts)
        # Only messages strictly after old_ts
        assert all(m.timestamp > old_ts for m in msgs)

    def test_limit(self, board):
        for i in range(10):
            board.add_message(f"msg {i}", MESSAGE_TYPE_USER, author="user")
        msgs = board.get_messages_for_lumen(limit=3)
        assert len(msgs) == 3


# ---------------------------------------------------------------------------
# 10. MessageBoard.repair_orphaned_answered()
# ---------------------------------------------------------------------------
class TestRepairOrphanedAnswered:
    def test_repairs_orphaned_question(self, board):
        """Question marked answered with no answer message gets repaired."""
        q = board.add_message("why?", MESSAGE_TYPE_QUESTION, author="lumen")
        # Manually mark as answered without an actual answer message
        q.answered = True
        board._save()

        repaired = board.repair_orphaned_answered()
        assert repaired == 1
        assert not board.get_by_id(q.message_id).answered

    def test_does_not_repair_properly_answered(self, board):
        """Question with a real answer should NOT be repaired."""
        q = board.add_message("why?", MESSAGE_TYPE_QUESTION, author="lumen")
        q.answered = True
        # Add an agent answer that responds_to the question
        ans = board.add_message("because!", MESSAGE_TYPE_AGENT, author="helper")
        ans.responds_to = q.message_id
        board._save()

        repaired = board.repair_orphaned_answered()
        assert repaired == 0
        assert board.get_by_id(q.message_id).answered is True

    def test_does_not_repair_expired_old_questions(self, board):
        """Old questions (>4h) are legitimately expired, not repaired."""
        q = board.add_message("old q?", MESSAGE_TYPE_QUESTION, author="lumen")
        q.answered = True
        q.timestamp = time.time() - 20000  # Well over 4 hours ago
        board._save()

        repaired = board.repair_orphaned_answered()
        assert repaired == 0

    def test_returns_zero_when_nothing_to_repair(self, board):
        board.add_message("just an obs", MESSAGE_TYPE_OBSERVATION)
        assert board.repair_orphaned_answered() == 0

    def test_repairs_multiple(self, board):
        """Multiple orphaned questions all get repaired."""
        q1 = board.add_message("q1?", MESSAGE_TYPE_QUESTION, author="lumen")
        q1.answered = True
        q2 = board.add_message("q2?", MESSAGE_TYPE_QUESTION, author="lumen")
        q2.answered = True
        board._save()

        repaired = board.repair_orphaned_answered()
        assert repaired == 2
