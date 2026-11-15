"""Tests for system_metrics persistence and rate-of-change probes."""

from datetime import datetime, timedelta

import pytest

from anima_mcp.identity.store import IdentityStore


class FakeReadings:
    """Minimal SensorReadings stand-in for testing."""
    def __init__(self, **kwargs):
        defaults = {
            "cpu_temp_c": 55.0,
            "cpu_percent": 23.5,
            "memory_percent": 41.2,
            "disk_percent": 68.0,
            "ambient_temp_c": 22.1,
            "humidity_pct": 35.0,
            "light_lux": 150.0,
            "pressure_hpa": 1013.25,
            "led_brightness": 0.8,
            "throttled_now": False,
            "undervoltage_now": False,
            "freq_capped_now": False,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_anima.db"
    s = IdentityStore(db_path=str(db_path))
    s._connect()  # Force schema init
    return s


class TestSystemMetricsTable:

    def test_table_exists(self, store):
        conn = store._connect()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='system_metrics'"
        ).fetchall()
        assert len(tables) == 1

    def test_table_creation_idempotent(self, store):
        """Calling _init_schema again should not fail."""
        store._init_schema()
        conn = store._connect()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='system_metrics'"
        ).fetchall()
        assert len(tables) == 1


class TestRecordSystemMetrics:

    def test_record_inserts_row(self, store):
        readings = FakeReadings()
        store.record_system_metrics(readings)

        conn = store._connect()
        rows = conn.execute("SELECT * FROM system_metrics").fetchall()
        assert len(rows) == 1

    def test_record_correct_values(self, store):
        readings = FakeReadings(cpu_temp_c=62.5, memory_percent=80.0)
        store.record_system_metrics(readings)

        conn = store._connect()
        row = conn.execute("SELECT cpu_temp_c, memory_percent FROM system_metrics").fetchone()
        assert row["cpu_temp_c"] == 62.5
        assert row["memory_percent"] == 80.0

    def test_record_bool_as_int(self, store):
        readings = FakeReadings(throttled_now=True, undervoltage_now=False)
        store.record_system_metrics(readings)

        conn = store._connect()
        row = conn.execute("SELECT throttled_now, undervoltage_now FROM system_metrics").fetchone()
        assert row["throttled_now"] == 1
        assert row["undervoltage_now"] == 0

    def test_record_none_values(self, store):
        readings = FakeReadings(cpu_temp_c=None, pressure_hpa=None, throttled_now=None)
        store.record_system_metrics(readings)

        conn = store._connect()
        row = conn.execute("SELECT cpu_temp_c, pressure_hpa, throttled_now FROM system_metrics").fetchone()
        assert row["cpu_temp_c"] is None
        assert row["pressure_hpa"] is None
        assert row["throttled_now"] is None

    def test_record_from_dict(self, store):
        d = {"cpu_temp_c": 50.0, "cpu_percent": 10.0}
        store.record_system_metrics(d)

        conn = store._connect()
        row = conn.execute("SELECT cpu_temp_c, cpu_percent FROM system_metrics").fetchone()
        assert row["cpu_temp_c"] == 50.0
        assert row["cpu_percent"] == 10.0

    def test_multiple_records(self, store):
        for i in range(5):
            store.record_system_metrics(FakeReadings(cpu_temp_c=50.0 + i))

        conn = store._connect()
        count = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
        assert count == 5


class TestGetSystemMetrics:

    def test_returns_recent_rows(self, store):
        for i in range(3):
            store.record_system_metrics(FakeReadings(cpu_temp_c=50.0 + i))

        rows = store.get_system_metrics(hours=1)
        assert len(rows) == 3
        # Should be oldest first
        assert rows[0]["cpu_temp_c"] == 50.0
        assert rows[2]["cpu_temp_c"] == 52.0

    def test_respects_limit(self, store):
        for i in range(10):
            store.record_system_metrics(FakeReadings(cpu_temp_c=50.0 + i))

        rows = store.get_system_metrics(hours=1, limit=3)
        assert len(rows) == 3

    def test_excludes_old_rows(self, store):
        # Insert a row with old timestamp
        conn = store._connect()
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute(
            "INSERT INTO system_metrics (timestamp, cpu_temp_c, epoch) VALUES (?, ?, 1)",
            (old_time, 99.0)
        )
        conn.commit()

        # Insert a recent row
        store.record_system_metrics(FakeReadings(cpu_temp_c=55.0))

        rows = store.get_system_metrics(hours=24)
        assert len(rows) == 1
        assert rows[0]["cpu_temp_c"] == 55.0

    def test_empty_when_no_data(self, store):
        rows = store.get_system_metrics(hours=1)
        assert rows == []


class TestPruneSystemMetrics:

    def test_prune_deletes_old_rows(self, store):
        conn = store._connect()
        # Insert old row
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute(
            "INSERT INTO system_metrics (timestamp, cpu_temp_c, epoch) VALUES (?, ?, 1)",
            (old_time, 99.0)
        )
        # Insert recent row
        recent_time = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO system_metrics (timestamp, cpu_temp_c, epoch) VALUES (?, ?, 1)",
            (recent_time, 55.0)
        )
        conn.commit()

        deleted = store.prune_system_metrics(max_age_hours=24.0)
        assert deleted == 1

        remaining = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
        assert remaining == 1

    def test_prune_keeps_recent_rows(self, store):
        for i in range(5):
            store.record_system_metrics(FakeReadings(cpu_temp_c=50.0 + i))

        deleted = store.prune_system_metrics(max_age_hours=24.0)
        assert deleted == 0

    def test_prune_returns_zero_when_empty(self, store):
        deleted = store.prune_system_metrics(max_age_hours=24.0)
        assert deleted == 0


def _real_probe(name, store, monkeypatch):
    """The probe lifecycle actually registers, pointed at ``store``.

    These classes used to test hand-copied replicas of the probe bodies, so
    they kept passing whatever the real probes did.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from anima_mcp import ctx_ref, health, lifecycle

    probes = {}
    registry = MagicMock()
    registry.register.side_effect = (
        lambda probe_name, probe=None, **_kwargs: probes.__setitem__(probe_name, probe)
    )
    monkeypatch.setattr(health, "get_health_registry", lambda: registry)
    monkeypatch.setattr(ctx_ref, "_ctx", SimpleNamespace(store=store))
    lifecycle._register_health_probes()
    return probes[name]


class TestThermalRateProbe:
    """Test the thermal_trend probe logic (metrics → health bridge)."""

    @pytest.fixture(autouse=True)
    def _bind(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def _make_probe(self, store):
        return _real_probe("thermal_trend", store, self._monkeypatch)

    def _insert_with_timestamps(self, store, temps, interval_seconds=30):
        """Insert metrics rows with controlled timestamps."""
        conn = store._connect()
        now = datetime.now()
        for i, temp in enumerate(temps):
            ts = (now - timedelta(seconds=interval_seconds * (len(temps) - 1 - i))).isoformat()
            conn.execute(
                "INSERT INTO system_metrics (timestamp, cpu_temp_c, epoch) VALUES (?, ?, 1)",
                (ts, temp)
            )
        conn.commit()

    def test_unknown_when_no_data(self, store):
        # 2026-11-12: was "healthy when no data". No rows is not evidence of
        # a cool CPU; the probe says it cannot tell.
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["state"] == "unknown"
        assert "evaluated" not in result

    def test_unknown_when_insufficient_rows(self, store):
        store.record_system_metrics(FakeReadings(cpu_temp_c=50.0))
        store.record_system_metrics(FakeReadings(cpu_temp_c=55.0))
        result = self._make_probe(store)()
        assert result["ok"] is False  # Only 2 rows, need 3
        assert result["state"] == "unknown"
        assert result["rows"] == 2

    def test_unknown_when_cpu_temp_is_null(self, store):
        self._insert_with_timestamps(store, [None, None, None, None, None])
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["state"] == "unknown"

    def test_healthy_with_stable_temp(self, store):
        # 5 rows over 2 minutes, stable at 55°C
        self._insert_with_timestamps(store, [55.0, 55.1, 54.9, 55.0, 55.1])
        result = self._make_probe(store)()
        assert result["ok"] is True
        assert result["evaluated"] is True

    def test_degraded_with_rapid_rise(self, store):
        # 5 rows over 2 minutes, rising 10°C/min (50 → 70 in 2 min)
        self._insert_with_timestamps(store, [50.0, 55.0, 60.0, 65.0, 70.0])
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["evaluated"] is True
        assert result["state"] == "rising_fast"
        assert result["rate_c_per_min"] == pytest.approx(10.0, abs=0.1)

    def test_healthy_with_gradual_rise(self, store):
        # 5 rows over 2 minutes, rising 2.5°C/min (50 → 55 in 2 min) — below threshold
        self._insert_with_timestamps(store, [50.0, 51.25, 52.5, 53.75, 55.0])
        assert self._make_probe(store)()["ok"] is True

    def test_healthy_with_cooling(self, store):
        # Temperature falling — always ok
        self._insert_with_timestamps(store, [70.0, 65.0, 60.0, 55.0, 50.0])
        assert self._make_probe(store)()["ok"] is True


class TestMemoryPressureProbe:
    """Test the memory_pressure probe logic."""

    @pytest.fixture(autouse=True)
    def _bind(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def _make_probe(self, store):
        return _real_probe("memory_pressure", store, self._monkeypatch)

    def test_unknown_when_no_data(self, store):
        # 2026-11-12: was "healthy when no data".
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["state"] == "unknown"
        assert "evaluated" not in result

    def test_unknown_when_memory_is_null(self, store):
        store.record_system_metrics(FakeReadings(memory_percent=None))
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["state"] == "unknown"

    def test_healthy_with_normal_memory(self, store):
        store.record_system_metrics(FakeReadings(memory_percent=45.0))
        result = self._make_probe(store)()
        assert result["ok"] is True
        assert result["evaluated"] is True
        assert result["sample_age_seconds"] <= 5

    def test_degraded_with_high_memory(self, store):
        store.record_system_metrics(FakeReadings(memory_percent=95.0))
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["evaluated"] is True
        assert result["state"] == "high"

    def test_healthy_at_boundary(self, store):
        store.record_system_metrics(FakeReadings(memory_percent=89.9))
        assert self._make_probe(store)()["ok"] is True

    def test_one_skipped_write_does_not_blind_the_probe(self, store):
        # The newest row is two minutes old: one or two ~30 s writes were
        # skipped. Still a reading, and its age is reported.
        conn = store._connect()
        ts = (datetime.now() - timedelta(seconds=120)).isoformat()
        conn.execute(
            "INSERT INTO system_metrics (timestamp, memory_percent, epoch) VALUES (?, ?, 1)",
            (ts, 45.0),
        )
        conn.commit()
        result = self._make_probe(store)()
        assert result["ok"] is True
        assert 115 <= result["sample_age_seconds"] <= 130

    def test_stale_rows_are_not_evidence(self, store):
        conn = store._connect()
        ts = (datetime.now() - timedelta(minutes=10)).isoformat()
        conn.execute(
            "INSERT INTO system_metrics (timestamp, memory_percent, epoch) VALUES (?, ?, 1)",
            (ts, 45.0),
        )
        conn.commit()
        result = self._make_probe(store)()
        assert result["ok"] is False
        assert result["state"] == "unknown"


class _Health:
    def __init__(self):
        self.beats = []

    def heartbeat(self, name):
        self.beats.append(name)


class _BrokenStore:
    def record_system_metrics(self, readings):
        raise RuntimeError("database disk image is malformed")


class TestSystemMetricsWriter:
    """The loop's writer: a failed write is counted and is not a heartbeat.

    2026-11-12: the write sat in `except Exception: pass` and the thermal /
    memory heartbeats fired "regardless", so a store rejecting every row left
    no trace and kept both subsystems looking alive.
    """

    def test_failed_write_is_counted_and_not_heartbeated(self):
        from types import SimpleNamespace

        from anima_mcp import server
        from anima_mcp.error_recovery import reset_suppressed_counts, suppressed_counts

        reset_suppressed_counts()
        health = _Health()
        ctx = SimpleNamespace(store=_BrokenStore())

        assert server._record_system_metrics(ctx, FakeReadings(), health) is False

        assert suppressed_counts().get("server.system_metrics") == 1
        assert health.beats == []
        reset_suppressed_counts()

    def test_successful_write_heartbeats_both_probes(self, store):
        from types import SimpleNamespace

        from anima_mcp import server

        health = _Health()
        ctx = SimpleNamespace(store=store)

        assert server._record_system_metrics(ctx, FakeReadings(), health) is True

        assert sorted(health.beats) == ["memory_pressure", "thermal_trend"]
        assert len(store.get_system_metrics(hours=1.0)) == 1

    @pytest.mark.parametrize("has_store, has_readings", [(False, True), (True, False)])
    def test_nothing_written_means_no_heartbeat(self, store, has_store, has_readings):
        from types import SimpleNamespace

        from anima_mcp import server

        health = _Health()
        ctx = SimpleNamespace(store=store if has_store else None)
        readings = FakeReadings() if has_readings else None

        assert server._record_system_metrics(ctx, readings, health) is False
        assert health.beats == []

    def test_loop_calls_the_writer_and_has_no_bare_swallow(self):
        import inspect

        from anima_mcp import server

        source = inspect.getsource(server._update_display_loop)
        assert "_record_system_metrics(_ctx, readings, _health)" in source
        assert "_ctx.store.record_system_metrics(readings)" not in source
        assert '_health.heartbeat("thermal_trend")' not in source
