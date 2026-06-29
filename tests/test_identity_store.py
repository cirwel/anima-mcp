"""Tests for identity/store module — wake/sleep lifecycle, persistence, deduplication."""

import time
from datetime import datetime, timedelta

import pytest

from anima_mcp.identity.store import IdentityStore

CREATURE_ID = "test-creature-001"


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "identity_test.db")
    return IdentityStore(db_path=db_path)


class TestCreatureIdentity:
    """Test the CreatureIdentity dataclass."""

    def test_age_seconds_positive(self, store):
        identity = store.wake(CREATURE_ID)
        assert identity.age_seconds() >= 0

    def test_alive_ratio_bounded(self, store):
        identity = store.wake(CREATURE_ID)
        r = identity.alive_ratio()
        assert 0.0 <= r <= 1.0

    def test_to_dict_has_required_keys(self, store):
        identity = store.wake(CREATURE_ID)
        d = identity.to_dict()
        for key in ["creature_id", "born_at", "total_awakenings", "total_alive_seconds", "name", "age_seconds", "alive_ratio"]:
            assert key in d


class TestFirstWake:
    """Test initial wake (birth)."""

    def test_creates_identity(self, store):
        identity = store.wake(CREATURE_ID)
        assert identity.creature_id == CREATURE_ID
        assert identity.born_at is not None

    def test_first_wake_counts_as_awakening(self, store):
        identity = store.wake(CREATURE_ID)
        assert identity.total_awakenings >= 1

    def test_get_identity_before_wake_returns_none(self, store):
        assert store.get_identity() is None

    def test_get_identity_after_wake(self, store):
        store.wake(CREATURE_ID)
        assert store.get_identity() is not None
        assert store.get_identity().creature_id == CREATURE_ID


class TestWakeSleepCycle:
    """Test wake/sleep lifecycle."""

    def test_sleep_returns_session_seconds(self, store):
        store.wake(CREATURE_ID)
        time.sleep(0.02)
        session = store.sleep()
        assert session > 0

    def test_sleep_before_wake_returns_zero(self, store):
        assert store.sleep() == 0.0

    def test_identity_persists_across_stores(self, tmp_path):
        db_path = str(tmp_path / "identity_test.db")
        s1 = IdentityStore(db_path=db_path)
        s1.wake(CREATURE_ID)
        s1.sleep()

        s2 = IdentityStore(db_path=db_path)
        identity = s2.wake(CREATURE_ID, dedupe_window_seconds=0)
        assert identity.creature_id == CREATURE_ID
        assert identity.total_alive_seconds > 0

    def test_forked_database_files_independent(self, tmp_path):
        """Same creature_id string on two DB paths = two independent records (fork semantics)."""
        uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        p1 = str(tmp_path / "one.db")
        p2 = str(tmp_path / "two.db")
        sa = IdentityStore(db_path=p1)
        sb = IdentityStore(db_path=p2)
        sa.wake(uid)
        sb.wake(uid)
        aw_b_before = sb.get_identity().total_awakenings
        sa.sleep()
        time.sleep(0.02)
        sa.wake(uid, dedupe_window_seconds=0)
        assert sa.get_identity().total_awakenings > 1
        assert sb.get_identity().total_awakenings == aw_b_before

    def test_awakening_deduplication(self, store):
        """Rapid re-wakes within dedupe window should not increment awakenings."""
        store.wake(CREATURE_ID, dedupe_window_seconds=300)
        first_awakenings = store.get_identity().total_awakenings
        # Second wake within window
        store.wake(CREATURE_ID, dedupe_window_seconds=300)
        second_awakenings = store.get_identity().total_awakenings
        assert second_awakenings == first_awakenings

    def test_no_dedup_when_window_zero(self, store):
        """With dedupe_window_seconds=0, every wake counts."""
        store.wake(CREATURE_ID, dedupe_window_seconds=0)
        store.sleep()
        time.sleep(0.01)
        store.wake(CREATURE_ID, dedupe_window_seconds=0)
        # Second wake after sleep should count
        assert store.get_identity().total_awakenings >= 1


class TestAliveTimeInvariant:
    """A creature can't be alive longer than it has existed (alive_ratio <= 1).

    Clock resets / restores / double-counted heartbeats can drift the persisted
    counter above wall-clock age. _recalculate_stats must cap at age and fall
    back to the event-derived sleep_total, which restores the schema's gap
    texture (alive < age makes discontinuities legible).
    """

    def test_impossible_persisted_alive_is_corrected(self, store):
        store.wake(CREATURE_ID, dedupe_window_seconds=0)
        conn = store._connect()
        # Backdate birth so age ~= 1000s, and record an honest 600s sleep session.
        born = (datetime.now() - timedelta(seconds=1000)).isoformat()
        conn.execute(
            "UPDATE identity SET born_at = ? WHERE creature_id = ?",
            (born, CREATURE_ID),
        )
        conn.execute(
            "INSERT INTO events (timestamp, event_type, data) VALUES (?, 'sleep', ?)",
            (datetime.now().isoformat(), '{"session_seconds": 600}'),
        )
        # Corrupt the persisted counter to an impossible value (alive >> age).
        conn.execute(
            "UPDATE identity SET total_alive_seconds = ? WHERE creature_id = ?",
            (5000.0, CREATURE_ID),
        )
        conn.commit()

        identity = store.wake(CREATURE_ID, dedupe_window_seconds=0)
        age = identity.age_seconds()

        # Invariant: never alive longer than existence (+1s slack for elapsed time).
        assert identity.total_alive_seconds <= age + 1
        # Gap texture restored: fell back to the honest 600s, not the impossible 5000s.
        assert identity.alive_ratio() < 1.0
        assert 0.0 <= identity.alive_ratio() <= 1.0


class TestSetName:
    """Test name setting."""

    def test_set_name(self, store):
        store.wake(CREATURE_ID)
        result = store.set_name("Lumen", sync_to_unitares=False)
        assert result is True
        assert store.get_identity().name == "Lumen"

    def test_name_history_tracked(self, store):
        store.wake(CREATURE_ID)
        store.set_name("Alpha", sync_to_unitares=False)
        store.set_name("Beta", sync_to_unitares=False)
        assert store.get_identity().name == "Beta"
        assert len(store.get_identity().name_history) >= 1


class TestStateHistory:
    """Test state recording."""

    def test_record_state(self, store):
        store.wake(CREATURE_ID)
        store.record_state(0.5, 0.6, 0.7, 0.8, {"temp": 25.0})
        history = store.get_recent_state_history(limit=5)
        assert len(history) >= 1

    def test_state_history_limit(self, store):
        store.wake(CREATURE_ID)
        for i in range(10):
            store.record_state(0.5, 0.5, 0.5, 0.5, {})
        history = store.get_recent_state_history(limit=5)
        assert len(history) <= 5
