"""Tests for growth.migrations — identity merge and raw-lux preference reset."""

import json
import sqlite3

import pytest

from anima_mcp.growth.migrations import migrate_raw_lux_preferences, run_identity_migration
from anima_mcp.growth.models import GrowthPreference, PreferenceCategory


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE preferences (
            name TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            description TEXT,
            value REAL DEFAULT 0.0,
            confidence REAL DEFAULT 0.0,
            observation_count INTEGER DEFAULT 0,
            first_noticed TEXT,
            last_confirmed TEXT
        );
        CREATE TABLE relationships (
            agent_id TEXT PRIMARY KEY,
            name TEXT,
            first_met TEXT,
            last_seen TEXT,
            interaction_count INTEGER DEFAULT 0,
            bond_strength TEXT DEFAULT 'stranger',
            emotional_valence REAL DEFAULT 0.0,
            memorable_moments TEXT DEFAULT '[]',
            topics_discussed TEXT DEFAULT '[]',
            gifts_received INTEGER DEFAULT 0,
            self_dialogue_topics TEXT DEFAULT '[]',
            visitor_type TEXT DEFAULT 'agent'
        );
        """
    )


def _insert_relationship(conn: sqlite3.Connection, agent_id: str, **kwargs) -> None:
    defaults = {
        "name": agent_id,
        "first_met": "2020-01-01T00:00:00",
        "last_seen": "2025-06-01T12:00:00",
        "interaction_count": 1,
        "bond_strength": "stranger",
        "emotional_valence": 0.5,
        "memorable_moments": "[]",
        "topics_discussed": "[]",
        "gifts_received": 0,
        "visitor_type": "agent",
    }
    defaults.update(kwargs)
    conn.execute(
        """
        INSERT INTO relationships (
            agent_id, name, first_met, last_seen, interaction_count,
            bond_strength, emotional_valence, memorable_moments,
            topics_discussed, gifts_received, self_dialogue_topics, visitor_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?)
        """,
        (
            agent_id,
            defaults["name"],
            defaults["first_met"],
            defaults["last_seen"],
            defaults["interaction_count"],
            defaults["bond_strength"],
            defaults["emotional_valence"],
            defaults["memorable_moments"],
            defaults["topics_discussed"],
            defaults["gifts_received"],
            defaults["visitor_type"],
        ),
    )


class TestRunIdentityMigration:
    @pytest.fixture(autouse=True)
    def _operator_is_kenny(self, monkeypatch):
        # The canonical operator name is env-driven (ANIMA_OPERATOR_NAME); these
        # tests exercise a deployment whose caretaker is "kenny". Pin the alias
        # map so the merge target is stable regardless of the host's env.
        from anima_mcp import server_state
        monkeypatch.setattr(
            server_state,
            "KNOWN_PERSON_ALIASES",
            {"kenny": {"kenny", "caretaker", "dashboard", "human"}},
        )

    def test_skips_when_user_version_ge_one(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        run_identity_migration(conn)
        assert conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 0

    def test_sets_lumen_visitor_type_self(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        _insert_relationship(conn, "lumen", visitor_type="agent")
        conn.commit()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0

        run_identity_migration(conn)

        row = conn.execute(
            "SELECT visitor_type FROM relationships WHERE agent_id = 'lumen'"
        ).fetchone()
        assert row["visitor_type"] == "self"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    def test_merges_kenny_aliases(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        _insert_relationship(
            conn,
            "kenny",
            interaction_count=3,
            emotional_valence=0.4,
            memorable_moments=json.dumps(["a"]),
            topics_discussed=json.dumps(["t1"]),
            gifts_received=1,
        )
        _insert_relationship(
            conn,
            "caretaker",
            interaction_count=7,
            emotional_valence=0.6,
            memorable_moments=json.dumps(["b"]),
            topics_discussed=json.dumps(["t2"]),
            gifts_received=2,
        )
        conn.commit()

        run_identity_migration(conn)

        n = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE LOWER(agent_id) IN ('kenny','caretaker')"
        ).fetchone()[0]
        assert n == 1
        merged = conn.execute(
            "SELECT agent_id, interaction_count, bond_strength, gifts_received, visitor_type "
            "FROM relationships WHERE agent_id = 'kenny'"
        ).fetchone()
        assert merged["interaction_count"] == 10
        assert merged["bond_strength"] == "frequent"
        assert merged["gifts_received"] == 3
        assert merged["visitor_type"] == "person"

    def test_merges_alias_rows_with_invalid_json_moments(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        _insert_relationship(
            conn,
            "kenny",
            memorable_moments="not-json",
            topics_discussed="also-bad",
            interaction_count=2,
        )
        _insert_relationship(conn, "human", interaction_count=1)
        conn.commit()

        run_identity_migration(conn)

        row = conn.execute(
            "SELECT interaction_count, visitor_type FROM relationships WHERE agent_id = 'kenny'"
        ).fetchone()
        assert row["interaction_count"] == 3
        assert row["visitor_type"] == "person"

    def test_idempotent_second_run_noop(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        _insert_relationship(conn, "lumen")
        conn.commit()
        run_identity_migration(conn)
        v1 = conn.execute("PRAGMA user_version").fetchone()[0]
        n1 = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        run_identity_migration(conn)
        v2 = conn.execute("PRAGMA user_version").fetchone()[0]
        n2 = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        assert v1 == v2 == 1
        assert n1 == n2


class TestMigrateRawLuxPreferences:
    def _pref(self, name: str, obs: int) -> GrowthPreference:
        now = __import__("datetime").datetime.now()
        return GrowthPreference(
            category=PreferenceCategory.ENVIRONMENT,
            name=name,
            description="test",
            value=0.9,
            confidence=0.9,
            observation_count=obs,
            first_noticed=now,
            last_confirmed=now,
        )

    def test_skips_when_sentinel_exists(self):
        conn = sqlite3.connect(":memory:")
        _schema(conn)
        conn.execute(
            """
            INSERT INTO preferences (name, category, description, value, confidence,
                observation_count, last_confirmed)
            VALUES ('_migration_raw_lux_v1', 'system', 'sentinel', 1.0, 1.0, 1, '2020-01-01')
            """
        )
        conn.commit()
        prefs = {"bright_light": self._pref("bright_light", 5000)}
        migrate_raw_lux_preferences(conn, prefs)
        assert prefs["bright_light"].observation_count == 5000

    def test_resets_high_count_tainted_preferences(self):
        conn = sqlite3.connect(":memory:")
        _schema(conn)
        conn.commit()
        prefs = {
            "bright_light": self._pref("bright_light", 2000),
            "drawing_bright": self._pref("drawing_bright", 1500),
        }
        migrate_raw_lux_preferences(conn, prefs)

        assert prefs["bright_light"].observation_count == 0
        assert prefs["bright_light"].confidence == pytest.approx(0.2)
        assert prefs["bright_light"].value == pytest.approx(0.5)
        assert prefs["drawing_bright"].observation_count == 0

        row = conn.execute(
            "SELECT name FROM preferences WHERE name = '_migration_raw_lux_v1'"
        ).fetchone()
        assert row is not None

    def test_inserts_sentinel_when_no_tainted_prefs(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _schema(conn)
        conn.commit()
        migrate_raw_lux_preferences(conn, {})
        row = conn.execute(
            "SELECT category FROM preferences WHERE name = '_migration_raw_lux_v1'"
        ).fetchone()
        assert row["category"] == "system"

    def test_does_not_reset_low_observation_tainted(self):
        conn = sqlite3.connect(":memory:")
        _schema(conn)
        conn.commit()
        prefs = {"bright_light": self._pref("bright_light", 100)}
        migrate_raw_lux_preferences(conn, prefs)
        assert prefs["bright_light"].observation_count == 100
        row = conn.execute(
            "SELECT name FROM preferences WHERE name = '_migration_raw_lux_v1'"
        ).fetchone()
        assert row is not None
