"""
Tests for lifecycle.py — wake() and sleep() server lifecycle management.

Covers:
  - wake() basic flow: ServerContext creation, identity resolution
  - wake() SQLite retry logic (database locked)
  - wake() identity precedence (provided ID, existing DB, new UUID)
  - wake() non-fatal subsystem failures (growth, health, trajectory, schema, drift)
  - wake() gives up after max_attempts on non-lock errors
  - sleep() persistence: calibration drift, schema, trajectory, canvas
  - sleep() bridge cleanup
  - sleep() voice cleanup
  - sleep() store shutdown + WAL checkpoint
  - sleep() with None context (no-op)
  - _set_ctx() bridge to server module
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.server_context import ServerContext
from anima_mcp import ctx_ref


def _probe_ok(result):
    """A probe verdict, whether it answered with a bare bool or a details dict."""
    return bool(result.get("ok", False)) if isinstance(result, dict) else bool(result)


_ENV_CHANNELS = ("ambient_temp_c", "humidity_pct", "light_lux", "pressure_hpa")
_LIVE_ENV = {
    "ambient_temp_c": 22.4,
    "humidity_pct": 41.0,
    "light_lux": 180.0,
    "pressure_hpa": 1012.6,
}


def _fresh_envelope(**readings):
    return {
        "readings": {"cpu_temp_c": 51.0, **readings},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _register_real_probes(monkeypatch, *, store=None, shm=None):
    """Run the real registration and hand back {name: (probe, options)}."""
    from anima_mcp import accessors, health, lifecycle

    registrations = {}
    registry = MagicMock()
    registry.register.side_effect = (
        lambda name, probe=None, **kwargs: registrations.__setitem__(
            name, (probe, kwargs)
        )
    )
    state = {"shm": shm}
    monkeypatch.setattr(health, "get_health_registry", lambda: registry)
    monkeypatch.setattr(accessors, "_get_last_shm_data", lambda: state["shm"])
    monkeypatch.setattr(ctx_ref, "_ctx", SimpleNamespace(store=store))
    lifecycle._register_health_probes()
    return registrations, state


def test_sensor_health_probe_requires_a_valid_fresh_broker_heartbeat(monkeypatch):
    registrations, state = _register_real_probes(monkeypatch)
    sensor_probe = registrations["sensors"][0]

    state["shm"] = {
        "readings": dict(_LIVE_ENV),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    assert _probe_ok(sensor_probe()) is True
    for bad in (
        None,
        {"timestamp": datetime.now(timezone.utc).isoformat()},
        {"readings": dict(_LIVE_ENV)},
        {"readings": dict(_LIVE_ENV), "timestamp": "bad"},
        {
            "readings": dict(_LIVE_ENV),
            "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
        },
        {
            "readings": dict(_LIVE_ENV),
            "timestamp": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
        },
    ):
        state["shm"] = bad
        assert _probe_ok(sensor_probe()) is False


@pytest.mark.parametrize(
    "readings",
    [
        {},  # channels not even present in the envelope
        dict.fromkeys(_ENV_CHANNELS),  # present, every one None
        dict.fromkeys(_ENV_CHANNELS, float("nan")),  # not a reading either
    ],
)
def test_sensor_probe_fails_when_every_environment_channel_is_absent(
    monkeypatch, readings
):
    """A fresh envelope with no environment in it is not a working sensor.

    The probe used to check only that the envelope was fresh and had a
    ``readings`` key, so it read ok with every environment channel None.
    """
    registrations, _ = _register_real_probes(
        monkeypatch, shm=_fresh_envelope(**readings)
    )

    result = registrations["sensors"][0]()

    assert _probe_ok(result) is False
    assert result["state"] == "no_environment_channels"
    assert result["channels"] == {"present": [], "missing": list(_ENV_CHANNELS)}


@pytest.mark.parametrize("missing", _ENV_CHANNELS)
def test_sensor_probe_reports_but_tolerates_one_missing_channel(monkeypatch, missing):
    """Pressure (or any single channel) is legitimately absent at times."""
    readings = {**_LIVE_ENV, missing: None}
    registrations, _ = _register_real_probes(
        monkeypatch, shm=_fresh_envelope(**readings)
    )

    result = registrations["sensors"][0]()

    assert _probe_ok(result) is True
    assert result["channels"]["missing"] == [missing]
    assert result["channels"]["present"] == [c for c in _ENV_CHANNELS if c != missing]


def test_sensor_probe_reports_all_channels_present(monkeypatch):
    registrations, _ = _register_real_probes(
        monkeypatch, shm=_fresh_envelope(**_LIVE_ENV)
    )

    result = registrations["sensors"][0]()

    assert _probe_ok(result) is True
    assert result["channels"] == {"present": list(_ENV_CHANNELS), "missing": []}


class _MetricsStore:
    """system_metrics reader stand-in: fixed rows, or a failure."""

    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error

    def get_system_metrics(self, hours=24.0, limit=2880):
        if self.error is not None:
            raise self.error
        return list(self.rows)


def _metric_rows(count, *, cpu_temp_c=55.0, memory_percent=41.0, spacing_seconds=30):
    now = datetime.now()
    return [
        {
            "timestamp": (
                now - timedelta(seconds=spacing_seconds * (count - 1 - i))
            ).isoformat(),
            "cpu_temp_c": cpu_temp_c[i] if isinstance(cpu_temp_c, list) else cpu_temp_c,
            "memory_percent": memory_percent,
        }
        for i in range(count)
    ]


_NO_EVIDENCE_STORES = {
    "no_store": lambda: None,
    "no_rows": lambda: _MetricsStore(rows=[]),
    "null_values": lambda: _MetricsStore(
        rows=_metric_rows(5, cpu_temp_c=None, memory_percent=None)
    ),
    "unreadable": lambda: _MetricsStore(
        error=RuntimeError("database disk image is malformed")
    ),
}


@pytest.mark.parametrize("probe_name", ["thermal_trend", "memory_pressure"])
@pytest.mark.parametrize("case", sorted(_NO_EVIDENCE_STORES))
def test_metric_probes_report_unknown_without_evidence(monkeypatch, probe_name, case):
    """No store, no rows, NULL values and read errors are not "ok".

    Both probes returned True for all of these ("Fail open"), so they could
    only ever read ok or threshold-crossed -- never "I cannot tell".
    """
    registrations, _ = _register_real_probes(
        monkeypatch, store=_NO_EVIDENCE_STORES[case]()
    )

    result = registrations[probe_name][0]()

    assert _probe_ok(result) is False
    assert result["state"] == "unknown"
    assert result.get("evaluated") is not True
    assert result["reason"]


def test_thermal_probe_needs_three_rows_to_evaluate(monkeypatch):
    registrations, _ = _register_real_probes(
        monkeypatch, store=_MetricsStore(rows=_metric_rows(2))
    )

    result = registrations["thermal_trend"][0]()

    assert _probe_ok(result) is False
    assert result["state"] == "unknown"
    assert result["rows"] == 2


def test_metric_probe_read_failures_are_counted(monkeypatch):
    from anima_mcp.error_recovery import reset_suppressed_counts, suppressed_counts

    reset_suppressed_counts()
    registrations, _ = _register_real_probes(
        monkeypatch, store=_MetricsStore(error=RuntimeError("disk I/O error"))
    )

    registrations["thermal_trend"][0]()
    registrations["memory_pressure"][0]()

    counts = suppressed_counts()
    assert counts.get("health.thermal_trend") == 1
    assert counts.get("health.memory_pressure") == 1
    reset_suppressed_counts()


def test_metric_probes_mark_real_verdicts_as_evaluated(monkeypatch):
    healthy, _ = _register_real_probes(
        monkeypatch, store=_MetricsStore(rows=_metric_rows(5))
    )
    for name in ("thermal_trend", "memory_pressure"):
        result = healthy[name][0]()
        assert result["ok"] is True
        assert result["evaluated"] is True

    crossed, _ = _register_real_probes(
        monkeypatch,
        store=_MetricsStore(
            rows=_metric_rows(
                5, cpu_temp_c=[50.0, 55.0, 60.0, 65.0, 70.0], memory_percent=95.0
            )
        ),
    )
    for name in ("thermal_trend", "memory_pressure"):
        result = crossed[name][0]()
        assert result["ok"] is False
        assert result["evaluated"] is True
        assert result["state"] != "unknown"


def _real_registry_with_metric_probes(monkeypatch, store):
    """A real HealthRegistry holding only the two metric probes, as registered."""
    from anima_mcp import health

    registrations, _ = _register_real_probes(monkeypatch, store=store)
    registry = health.HealthRegistry()
    for name in ("thermal_trend", "memory_pressure"):
        probe, options = registrations[name]
        registry.register(name, probe=probe, **{**options, "debounce_seconds": 0.0})
        registry.heartbeat(name)
    return registry


def _rerun_probes(registry):
    for name in registry.subsystem_names():
        registry.get_subsystem(name).last_probe_time = 0.0


@pytest.mark.parametrize("case", sorted(_NO_EVIDENCE_STORES))
def test_metric_probes_are_absent_until_they_have_had_data(monkeypatch, case):
    """Fresh boot, degraded wake, dead store from the start: absent, not ok.

    And never ``degraded`` -- a boot with fewer than three rows must not pin
    ``overall()``, which is the saturation ``absent`` exists to prevent.
    """
    registry = _real_registry_with_metric_probes(
        monkeypatch, _NO_EVIDENCE_STORES[case]()
    )

    status = registry.status()

    assert status["thermal_trend"]["status"] == "absent"
    assert status["memory_pressure"]["status"] == "absent"
    assert registry.overall() == "ok"


def test_metric_probes_degrade_when_evidence_is_lost_after_working(monkeypatch):
    store = _MetricsStore(rows=_metric_rows(5))
    registry = _real_registry_with_metric_probes(monkeypatch, store)
    assert registry.status()["thermal_trend"]["status"] == "ok"
    assert registry.status()["memory_pressure"]["status"] == "ok"
    assert registry.overall() == "ok"

    store.error = RuntimeError("database disk image is malformed")
    _rerun_probes(registry)

    status = registry.status()
    assert status["thermal_trend"]["status"] == "degraded"
    assert status["memory_pressure"]["status"] == "degraded"
    assert status["thermal_trend"]["probe_details"]["state"] == "unknown"
    assert registry.overall() == "degraded"

    store.error = None
    _rerun_probes(registry)
    assert registry.overall() == "ok"


def test_threshold_crossed_on_first_evaluation_is_degraded_not_absent(monkeypatch):
    """`absent` means "cannot tell yet", never "told, and it was bad"."""
    registry = _real_registry_with_metric_probes(
        monkeypatch,
        _MetricsStore(
            rows=_metric_rows(
                5, cpu_temp_c=[50.0, 55.0, 60.0, 65.0, 70.0], memory_percent=95.0
            )
        ),
    )

    status = registry.status()

    assert status["thermal_trend"]["status"] == "degraded"
    assert status["memory_pressure"]["status"] == "degraded"
    assert registry.overall() == "degraded"


def test_day_summary_writer_health_probe_registration(monkeypatch):
    from anima_mcp import anima_history, health, lifecycle

    registrations = {}
    registry = MagicMock()
    registry.register.side_effect = (
        lambda name, probe, **kwargs: registrations.__setitem__(
            name, (probe, kwargs)
        )
    )
    details = {
        "ok": False,
        "last_error": "write failed",
        "last_attempt_at": "2026-08-23T12:00:00+00:00",
    }
    history = MagicMock()
    history.day_summary_health.return_value = details
    monkeypatch.setattr(health, "get_health_registry", lambda: registry)
    monkeypatch.setattr(anima_history, "get_anima_history", lambda: history)

    lifecycle._register_health_probes()

    probe, options = registrations["day_summary_writer"]
    assert probe() == details
    history.day_summary_health.assert_called_once_with()
    assert options == {
        "stale_threshold": 36 * 60 * 60,
        "debounce_seconds": 0.0,
        "optional": False,
    }


def test_outbound_heartbeat_is_registered_as_probe_only(monkeypatch):
    from anima_mcp import health, lifecycle

    registrations = {}
    registry = MagicMock()
    registry.register.side_effect = (
        lambda name, probe, **kwargs: registrations.__setitem__(
            name, (probe, kwargs)
        )
    )
    expected = {"ok": False, "state": "inert", "configured": False}
    monkeypatch.setattr(health, "get_health_registry", lambda: registry)
    monkeypatch.setattr(health, "outbound_heartbeat_health", lambda: expected)

    lifecycle._register_health_probes()

    probe, options = registrations["outbound_heartbeat"]
    assert probe() == expected
    assert options == {
        "debounce_seconds": 0.0,
        "optional": False,
        "probe_only": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_identity(creature_id=None, name="Lumen", awakenings=3, alive_seconds=1000.0):
    """Create a mock CreatureIdentity via SimpleNamespace."""
    return SimpleNamespace(
        creature_id=creature_id or str(uuid.uuid4()),
        name=name,
        total_awakenings=awakenings,
        total_alive_seconds=alive_seconds,
        born_at=datetime(2025, 1, 1),
    )


def make_mock_store(creature_id=None, db_has_identity=True):
    """Create a mock IdentityStore that returns an identity on wake()."""
    cid = creature_id or str(uuid.uuid4())
    store = MagicMock()
    store.db_path = "test.db"
    store._conn = MagicMock()

    identity = make_identity(creature_id=cid)
    store.wake.return_value = identity
    store.get_identity.return_value = identity
    store.sleep.return_value = 42.0  # session seconds

    # For the DB identity check in wake()
    conn_mock = MagicMock()
    if db_has_identity:
        conn_mock.execute.return_value.fetchone.return_value = (cid,)
    else:
        conn_mock.execute.return_value.fetchone.return_value = None
    store._connect.return_value = conn_mock

    return store


@pytest.fixture(autouse=True)
def cleanup_ctx():
    """Ensure ctx_ref is clean before and after each test."""
    ctx_ref._ctx = None
    yield
    ctx_ref._ctx = None


def _wake_patches():
    """Return dict of common patches needed for wake() to succeed.

    wake() does local imports from many submodules. We patch at the source
    so that when wake() does `from .identity import IdentityStore`, it gets
    our mock.
    """
    return {
        "store_cls": patch("anima_mcp.identity.IdentityStore"),
        "growth": patch("anima_mcp.growth.get_growth_system"),
        "traj": patch("anima_mcp.eisv.get_trajectory_awareness"),
        "schema_hub": patch("anima_mcp.accessors._get_schema_hub"),
        "cal_drift": patch("anima_mcp.accessors._get_calibration_drift"),
        "readings": patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)),
        "shm_data": patch("anima_mcp.accessors._get_last_shm_data", return_value=None),
        "health": patch("anima_mcp.lifecycle.get_health_registry", create=True),
    }


# ---------------------------------------------------------------------------
# _set_ctx
# ---------------------------------------------------------------------------

class TestSetCtx:
    def test_sets_ctx_ref_and_server_module(self):
        """_set_ctx writes to both ctx_ref and server._ctx."""
        from anima_mcp.lifecycle import _set_ctx
        from anima_mcp import server as server_mod
        ctx = ServerContext()
        _set_ctx(ctx)
        assert ctx_ref._ctx is ctx
        assert server_mod._ctx is ctx

    def test_set_ctx_to_none(self):
        """_set_ctx(None) clears both references."""
        from anima_mcp.lifecycle import _set_ctx
        from anima_mcp import server as server_mod
        ctx_ref._ctx = ServerContext()
        _set_ctx(None)
        assert ctx_ref._ctx is None
        assert server_mod._ctx is None


# ---------------------------------------------------------------------------
# wake()
# ---------------------------------------------------------------------------

class TestWake:
    def test_wake_basic_success(self):
        """wake() creates context, initializes store, and returns without error."""
        from anima_mcp.lifecycle import wake

        store = make_mock_store()
        store.get_recent_state_history.return_value = []

        with patch("anima_mcp.identity.IdentityStore", return_value=store) as MockStore, \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mock_traj, \
             patch("anima_mcp.accessors._get_schema_hub") as mock_hub, \
             patch("anima_mcp.accessors._get_calibration_drift") as mock_drift, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mock_traj.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            mock_hub.return_value = MagicMock(
                on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])),
                last_trajectory=None,
            )
            mock_drift.return_value = MagicMock(
                get_midpoints=MagicMock(return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}),
            )

            wake(db_path=":memory:", anima_id="test-id-1234-5678")

        MockStore.assert_called_once_with(":memory:")
        store.wake.assert_called_once_with("test-id-1234-5678")

    def test_wake_uses_provided_anima_id(self):
        """When anima_id is provided, it takes precedence over DB lookup."""
        from anima_mcp.lifecycle import wake

        store = make_mock_store()
        store.get_recent_state_history.return_value = []

        with patch("anima_mcp.identity.IdentityStore", return_value=store), \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            provided_id = "my-custom-id-1234"
            wake(db_path=":memory:", anima_id=provided_id)

        store.wake.assert_called_once_with(provided_id)
        # DB identity lookup NOT called when anima_id provided
        store._connect.assert_not_called()

    def test_wake_uses_existing_db_identity(self):
        """When no anima_id provided, wake() checks DB for existing identity."""
        from anima_mcp.lifecycle import wake

        existing_id = "existing-db-id-5678"
        store = make_mock_store(creature_id=existing_id, db_has_identity=True)
        store.get_recent_state_history.return_value = []

        with patch("anima_mcp.identity.IdentityStore", return_value=store), \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            wake(db_path=":memory:")

        store.wake.assert_called_once_with(existing_id)

    def test_wake_generates_new_uuid_when_no_existing_identity(self):
        """When no identity exists in DB, wake() generates a new UUID."""
        from anima_mcp.lifecycle import wake

        store = make_mock_store(db_has_identity=False)
        store.get_recent_state_history.return_value = []

        with patch("anima_mcp.identity.IdentityStore", return_value=store), \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            wake(db_path=":memory:")

        # A UUID was generated and passed to store.wake
        call_args = store.wake.call_args[0]
        generated_id = call_args[0]
        uuid.UUID(generated_id)  # Raises if invalid

    def test_wake_retries_on_db_locked(self):
        """wake() retries with backoff on 'database is locked' errors."""
        from anima_mcp.lifecycle import wake

        lock_error = sqlite3.OperationalError("database is locked")
        store_good = make_mock_store()
        store_good.get_recent_state_history.return_value = []

        call_count = 0
        def store_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise lock_error
            return store_good

        with patch("anima_mcp.identity.IdentityStore", side_effect=store_factory), \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("time.sleep") as mock_sleep:
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            wake(db_path=":memory:", anima_id="test-id")

        # Retried with escalating waits
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)  # attempt 1: 1 * 2
        mock_sleep.assert_any_call(4)  # attempt 2: 2 * 2

    def test_wake_degrades_on_non_lock_error(self):
        """wake() enters DEGRADED mode (ctx kept, store=None) on non-lock
        store errors — discarding the context left the display dark for 19
        days in Jul 2026 (#106)."""
        from anima_mcp.lifecycle import wake

        with patch("anima_mcp.identity.IdentityStore", side_effect=RuntimeError("some other error")):
            # Should not crash -- wake is "safe, never crashes"
            wake(db_path=":memory:", anima_id="test-id")

        # Context KEPT so the display loop can run; store absent
        assert ctx_ref._ctx is not None
        assert ctx_ref._ctx.store is None
        assert ctx_ref._ctx.anima_id == "test-id"
        assert ctx_ref._ctx.tension_tracker is not None

    def test_wake_degrades_after_max_attempts_on_lock(self):
        """wake() stops retrying after 5 database lock failures, then enters
        DEGRADED mode instead of discarding the context (#106)."""
        from anima_mcp.lifecycle import wake

        with patch("anima_mcp.identity.IdentityStore",
                    side_effect=sqlite3.OperationalError("database is locked")), \
             patch("time.sleep") as mock_sleep:
            wake(db_path=":memory:", anima_id="test-id")

        # 5 attempts total, 4 sleeps (no sleep on last failure)
        assert mock_sleep.call_count == 4
        assert ctx_ref._ctx is not None
        assert ctx_ref._ctx.store is None

    def test_wake_degraded_registers_health_probes(self):
        """DEGRADED mode still registers health probes — visibility matters
        most when the store is down."""
        from anima_mcp.lifecycle import wake

        with patch("anima_mcp.identity.IdentityStore", side_effect=RuntimeError("corrupt db")), \
             patch("anima_mcp.lifecycle._register_health_probes") as mock_reg:
            wake(db_path=":memory:", anima_id="test-id")

        mock_reg.assert_called_once()

    def test_wake_growth_failure_is_non_fatal(self):
        """If growth system fails, wake() continues (growth set to None)."""
        from anima_mcp.lifecycle import wake

        store = make_mock_store()
        store.get_recent_state_history.return_value = []

        with patch("anima_mcp.identity.IdentityStore", return_value=store), \
             patch("anima_mcp.growth.get_growth_system", side_effect=RuntimeError("growth broken")), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=0))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            wake(db_path=":memory:", anima_id="test-id")

        # wake completed (ctx should be set, growth is None on context)
        assert ctx_ref._ctx is not None
        assert ctx_ref._ctx.growth is None

    def test_wake_warm_start_from_state_history(self):
        """wake() extracts warm_start_anima from last state history row."""
        from anima_mcp.lifecycle import wake

        store = make_mock_store()
        history = [
            {"warmth": 0.4, "clarity": 0.5, "stability": 0.6, "presence": 0.7},
            {"warmth": 0.6, "clarity": 0.7, "stability": 0.8, "presence": 0.9},
        ]
        store.get_recent_state_history.return_value = history

        with patch("anima_mcp.identity.IdentityStore", return_value=store), \
             patch("anima_mcp.growth.get_growth_system", return_value=MagicMock(born_at=None)), \
             patch("anima_mcp.eisv.get_trajectory_awareness") as mt, \
             patch("anima_mcp.accessors._get_schema_hub") as ms, \
             patch("anima_mcp.accessors._get_calibration_drift") as md, \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            mt.return_value = MagicMock(bootstrap_from_history=MagicMock(return_value=5))
            ms.return_value = MagicMock(on_wake=MagicMock(return_value=None),
                compose_schema=MagicMock(return_value=MagicMock(nodes=[], edges=[])), last_trajectory=None)
            md.return_value = MagicMock(get_midpoints=MagicMock(
                return_value={"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5}))

            wake(db_path=":memory:", anima_id="test-id")

        assert ctx_ref._ctx.warm_start_anima == {
            "warmth": 0.6, "clarity": 0.7, "stability": 0.8, "presence": 0.9,
        }


# ---------------------------------------------------------------------------
# sleep()
# ---------------------------------------------------------------------------

class TestSleep:
    def test_sleep_with_none_context(self):
        """sleep() is safe when context is None."""
        from anima_mcp.lifecycle import sleep

        ctx_ref._ctx = None
        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        assert ctx_ref._ctx is None

    def test_sleep_persists_calibration_drift(self, tmp_path):
        """sleep() saves calibration drift to disk."""
        from anima_mcp.lifecycle import sleep

        drift = MagicMock()
        ctx = ServerContext()
        ctx.calibration_drift = drift
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"), \
             patch("anima_mcp.lifecycle.Path") as MockPath:
            MockPath.home.return_value = tmp_path
            sleep()

        drift.save.assert_called_once()

    def test_sleep_persists_schema(self):
        """sleep() calls schema_hub.persist_schema() when hub exists."""
        from anima_mcp.lifecycle import sleep

        hub = MagicMock()
        hub.persist_schema.return_value = True
        ctx = ServerContext()
        ctx.schema_hub = hub
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        hub.persist_schema.assert_called_once()

    def test_sleep_persists_trajectory_when_growth_exists(self):
        """sleep() computes and saves trajectory when growth system exists."""
        from anima_mcp.lifecycle import sleep

        ctx = ServerContext()
        ctx.growth = MagicMock()
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"), \
             patch("anima_mcp.trajectory.compute_trajectory_signature") as mock_compute, \
             patch("anima_mcp.trajectory.save_trajectory") as mock_save, \
             patch("anima_mcp.self_model.get_self_model"), \
             patch("anima_mcp.anima_history.get_anima_history"):
            mock_compute.return_value = MagicMock()
            mock_save.return_value = True
            sleep()

        mock_save.assert_called_once()

    def test_sleep_saves_canvas_when_pixels_exist(self):
        """sleep() saves drawing canvas when screen_renderer has pixels."""
        from anima_mcp.lifecycle import sleep

        canvas = MagicMock()
        canvas.pixels = [1, 2, 3]
        drawing_engine = MagicMock()
        drawing_engine.canvas = canvas
        renderer = MagicMock()
        renderer.drawing_engine = drawing_engine

        ctx = ServerContext()
        ctx.screen_renderer = renderer
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        canvas.save_to_disk.assert_called_once()

    def test_sleep_does_not_save_empty_canvas(self):
        """sleep() skips canvas save when no pixels drawn."""
        from anima_mcp.lifecycle import sleep

        canvas = MagicMock()
        canvas.pixels = []  # Empty (falsy)
        drawing_engine = MagicMock()
        drawing_engine.canvas = canvas
        renderer = MagicMock()
        renderer.drawing_engine = drawing_engine

        ctx = ServerContext()
        ctx.screen_renderer = renderer
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        canvas.save_to_disk.assert_not_called()

    def test_sleep_closes_bridge(self):
        """sleep() closes the UNITARES bridge if it exists."""
        from anima_mcp.lifecycle import sleep
        import asyncio

        bridge = MagicMock()
        ctx = ServerContext()
        ctx_ref._ctx = ctx

        loop = asyncio.new_event_loop()
        with patch("anima_mcp.accessors._get_server_bridge", return_value=bridge), \
             patch("anima_mcp.self_reflection.get_reflection_system"), \
             patch("asyncio.get_event_loop", return_value=loop):
            with patch.object(loop, 'is_running', return_value=False), \
                 patch.object(loop, 'run_until_complete'):
                sleep()

        loop.close()

    def test_sleep_stops_voice(self):
        """sleep() stops voice instance and sets it to None."""
        from anima_mcp.lifecycle import sleep

        voice = MagicMock()
        ctx = ServerContext()
        ctx.voice_instance = voice
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        voice.stop.assert_called_once()

    def test_sleep_calls_store_sleep_and_wal_checkpoint(self):
        """sleep() calls store.sleep(), WAL checkpoint, and store.close()."""
        from anima_mcp.lifecycle import sleep

        store = MagicMock()
        store.sleep.return_value = 123.0
        store._conn = MagicMock()

        ctx = ServerContext()
        ctx.store = store
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()

        store.sleep.assert_called_once()
        store._conn.execute.assert_called_with("PRAGMA wal_checkpoint(TRUNCATE)")
        store.close.assert_called_once()
        assert ctx_ref._ctx is None

    def test_sleep_handles_store_sleep_error(self):
        """sleep() handles errors during store.sleep() without crashing."""
        from anima_mcp.lifecycle import sleep

        store = MagicMock()
        store.sleep.side_effect = RuntimeError("shutdown error")
        store._conn = MagicMock()

        ctx = ServerContext()
        ctx.store = store
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"):
            sleep()  # Should not raise

        # Context still cleaned up in finally block
        assert ctx_ref._ctx is None

    def test_sleep_handles_drift_save_error(self):
        """sleep() continues when calibration drift save fails."""
        from anima_mcp.lifecycle import sleep

        drift = MagicMock()
        drift.save.side_effect = OSError("disk full")
        ctx = ServerContext()
        ctx.calibration_drift = drift
        ctx_ref._ctx = ctx

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system"), \
             patch("anima_mcp.lifecycle.Path") as MockPath:
            MockPath.home.return_value = Path("/tmp")
            sleep()  # Should not raise

    def test_sleep_closes_self_reflection(self):
        """sleep() closes the SelfReflection system."""
        from anima_mcp.lifecycle import sleep

        ctx_ref._ctx = None

        with patch("anima_mcp.accessors._get_server_bridge", return_value=None), \
             patch("anima_mcp.self_reflection.get_reflection_system") as mock_refl:
            sleep()

        mock_refl.return_value.close.assert_called_once()
