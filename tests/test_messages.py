"""Tests for messages module — MessageBoard storage, dedup, trimming, Q&A."""

import time
import pytest

import anima_mcp.messages as msg_module
from anima_mcp.messages import (
    MessageBoard, Message,
    MESSAGE_TYPE_OBSERVATION, MESSAGE_TYPE_QUESTION,
    MESSAGE_TYPE_USER,
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
        base_t = time.time() - 25_000
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
        assert all(m.answered for m in stale)

    def test_questions_similar_public_helper_matches_board_behavior(self, board):
        """Question similarity is available without reaching into MessageBoard internals."""
        q1 = "why is it that I now know that dim light changes my attention?"
        q2 = "why is it that I learned that dim light changes my attention?"

        assert questions_similar(q1, q2)
        assert board._questions_similar(q1, q2) == questions_similar(q1, q2)

    def test_unanswered_questions(self, board):
        q = board.add_question("Unanswered question?")
        unanswered = board.get_unanswered_questions(auto_expire=False)
        assert any(m.message_id == q.message_id for m in unanswered)

    def test_answering_removes_from_unanswered(self, board):
        q = board.add_question("Will someone answer?")
        board.add_agent_message("Yes!", agent_name="helper", responds_to=q.message_id)
        unanswered = board.get_unanswered_questions(auto_expire=False)
        assert not any(m.message_id == q.message_id for m in unanswered)

    def test_answering_marks_growth_curiosity_explored(self, board, monkeypatch):
        """When a question Lumen asked gets answered, the matching growth
        curiosity should be marked explored — that's how "find an answer to: X"
        goals auto-complete."""
        calls = []

        class FakeGrowth:
            def mark_explored(self, question, notes=None):
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
            def mark_explored(self, question, notes=None):
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
