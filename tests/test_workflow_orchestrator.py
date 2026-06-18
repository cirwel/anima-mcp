"""Tests for workflow_orchestrator.py — cross-server workflow coordination."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from types import SimpleNamespace

from conftest import make_anima, make_readings

from anima_mcp.workflow_orchestrator import (
    WorkflowStep, WorkflowResult, WorkflowStatus,
    UnifiedWorkflowOrchestrator, get_orchestrator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    return UnifiedWorkflowOrchestrator()


@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.check_availability = AsyncMock(return_value=True)
    bridge.check_in = AsyncMock(return_value={"action": "proceed", "margin": 0.8})
    bridge.process_agent_update = AsyncMock(return_value={"status": "ok"})
    return bridge


@pytest.fixture
def mock_store():
    store = MagicMock()
    identity = MagicMock()
    identity.name = "Lumen"
    identity.creature_id = "abc12345-xxxx-yyyy-zzzz"
    identity.total_awakenings = 5
    identity.total_alive_seconds = 3600.0
    store.get_identity.return_value = identity
    store.get_session_alive_seconds.return_value = 120.0
    return store


# ---------------------------------------------------------------------------
# Dataclasses & enums
# ---------------------------------------------------------------------------

class TestWorkflowDataclasses:
    def test_step_depends_on_defaults_to_empty(self):
        step = WorkflowStep(name="s", server="anima", tool="get_state", arguments={})
        assert step.depends_on == []

    def test_step_explicit_depends_on(self):
        step = WorkflowStep(name="s", server="anima", tool="get_state", arguments={}, depends_on=["a"])
        assert step.depends_on == ["a"]

    def test_result_fields(self):
        result = WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            steps={"s": {"ok": True}},
            errors={},
            summary="done",
        )
        assert result.status == WorkflowStatus.SUCCESS
        assert result.summary == "done"

    def test_status_enum_values(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.SUCCESS.value == "success"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.PARTIAL.value == "partial"


# ---------------------------------------------------------------------------
# check_unitares_available
# ---------------------------------------------------------------------------

class TestCheckUnitaresAvailable:
    async def test_returns_false_when_no_bridge(self, orchestrator):
        assert await orchestrator.check_unitares_available() is False

    async def test_returns_true_when_bridge_available(self, orchestrator, mock_bridge):
        orchestrator._bridge = mock_bridge
        assert await orchestrator.check_unitares_available() is True

    async def test_returns_false_on_bridge_exception(self, orchestrator, mock_bridge):
        mock_bridge.check_availability = AsyncMock(side_effect=Exception("down"))
        orchestrator._bridge = mock_bridge
        assert await orchestrator.check_unitares_available() is False


# ---------------------------------------------------------------------------
# get_lumen_state
# ---------------------------------------------------------------------------

class TestGetLumenState:
    async def test_returns_error_when_no_store(self, orchestrator):
        result = await orchestrator.get_lumen_state()
        assert "error" in result

    async def test_returns_error_when_no_readings(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(None, None)):
            result = await orchestrator.get_lumen_state()
        assert "error" in result

    async def test_returns_full_state(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            result = await orchestrator.get_lumen_state()
        assert "anima" in result
        assert "eisv" in result
        assert "sensors" in result
        assert "identity" in result
        assert result["identity"]["name"] == "Lumen"

    async def test_identity_id_truncated(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            result = await orchestrator.get_lumen_state()
        assert result["identity"]["id"] == "abc12345..."


# ---------------------------------------------------------------------------
# check_governance
# ---------------------------------------------------------------------------

class TestCheckGovernance:
    async def test_returns_error_when_no_bridge(self, orchestrator):
        result = await orchestrator.check_governance()
        assert result == {"error": "UNITARES bridge not configured"}

    async def test_returns_error_when_no_data(self, orchestrator, mock_bridge):
        orchestrator._bridge = mock_bridge
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(None, None)):
            result = await orchestrator.check_governance()
        assert "error" in result

    async def test_returns_governance_result(self, orchestrator, mock_bridge):
        orchestrator._bridge = mock_bridge
        readings = make_readings()
        anima = make_anima()
        result = await orchestrator.check_governance(anima=anima, readings=readings)
        assert result["action"] == "proceed"

    async def test_returns_error_on_bridge_exception(self, orchestrator, mock_bridge):
        mock_bridge.check_in = AsyncMock(side_effect=Exception("timeout"))
        orchestrator._bridge = mock_bridge
        readings = make_readings()
        anima = make_anima()
        result = await orchestrator.check_governance(anima=anima, readings=readings)
        assert "error" in result
        assert result["source"] == "local"


# ---------------------------------------------------------------------------
# execute_workflow
# ---------------------------------------------------------------------------

class TestExecuteWorkflow:
    async def test_empty_steps_returns_failed(self, orchestrator):
        # 0 errors == 0 steps → FAILED branch (len(errors) == len(steps))
        result = await orchestrator.execute_workflow([])
        assert result.status == WorkflowStatus.FAILED

    async def test_single_anima_step_success(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            step = WorkflowStep(name="state", server="anima", tool="get_state", arguments={})
            result = await orchestrator.execute_workflow([step])
        assert result.status == WorkflowStatus.SUCCESS
        assert "state" in result.steps

    async def test_single_step_failure_returns_failed(self, orchestrator):
        step = WorkflowStep(name="bad", server="anima", tool="nonexistent", arguments={})
        result = await orchestrator.execute_workflow([step])
        assert result.status == WorkflowStatus.FAILED
        assert "bad" in result.errors

    async def test_partial_status_on_mixed(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            good = WorkflowStep(name="good", server="anima", tool="get_state", arguments={})
            bad = WorkflowStep(name="bad", server="anima", tool="nonexistent", arguments={})
            result = await orchestrator.execute_workflow([good, bad])
        assert result.status == WorkflowStatus.PARTIAL

    async def test_dependency_order(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        call_order = []

        original_execute = orchestrator._execute_anima_step

        async def tracking_execute(step):
            call_order.append(step.name)
            return await original_execute(step)

        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            with patch.object(orchestrator, "_execute_anima_step", side_effect=tracking_execute):
                step_a = WorkflowStep(name="a", server="anima", tool="get_state", arguments={})
                step_b = WorkflowStep(name="b", server="anima", tool="get_state", arguments={}, depends_on=["a"])
                await orchestrator.execute_workflow([step_b, step_a])

        assert call_order.index("a") < call_order.index("b")

    async def test_missing_dependency_errors(self, orchestrator):
        step = WorkflowStep(name="orphan", server="anima", tool="get_state", arguments={}, depends_on=["nonexistent"])
        result = await orchestrator.execute_workflow([step])
        assert "orphan" in result.errors
        assert "Missing dependency" in result.errors["orphan"]

    async def test_unknown_server_errors(self, orchestrator):
        step = WorkflowStep(name="alien", server="mars", tool="ping", arguments={})
        result = await orchestrator.execute_workflow([step])
        assert "alien" in result.errors

    async def test_summary_format(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            step = WorkflowStep(name="s", server="anima", tool="get_state", arguments={})
            result = await orchestrator.execute_workflow([step])
        assert "succeeded" in result.summary.lower() or "success" in result.summary.lower()


# ---------------------------------------------------------------------------
# _execute_anima_step
# ---------------------------------------------------------------------------

class TestExecuteAnimaStep:
    async def test_get_state(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            step = WorkflowStep(name="s", server="anima", tool="get_state", arguments={})
            result = await orchestrator._execute_anima_step(step)
        assert "anima" in result

    async def test_read_sensors_returns_dict(self, orchestrator):
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            step = WorkflowStep(name="s", server="anima", tool="read_sensors", arguments={})
            result = await orchestrator._execute_anima_step(step)
        assert "sensors" in result

    async def test_read_sensors_no_data(self, orchestrator):
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(None, None)):
            step = WorkflowStep(name="s", server="anima", tool="read_sensors", arguments={})
            result = await orchestrator._execute_anima_step(step)
        assert "error" in result

    async def test_get_identity_no_store(self, orchestrator):
        step = WorkflowStep(name="s", server="anima", tool="get_identity", arguments={})
        result = await orchestrator._execute_anima_step(step)
        assert "error" in result

    async def test_get_identity_with_store(self, orchestrator, mock_store):
        orchestrator._anima_store = mock_store
        step = WorkflowStep(name="s", server="anima", tool="get_identity", arguments={})
        result = await orchestrator._execute_anima_step(step)
        assert result["name"] == "Lumen"

    async def test_unknown_tool_raises(self, orchestrator):
        step = WorkflowStep(name="s", server="anima", tool="not_a_tool", arguments={})
        with pytest.raises(ValueError, match="Unknown anima tool"):
            await orchestrator._execute_anima_step(step)

    async def test_get_calibration_returns_serialized_config(self, orchestrator):
        calibration = SimpleNamespace(to_dict=lambda: {"warmth_bias": 0.1})
        step = WorkflowStep(name="s", server="anima", tool="get_calibration", arguments={})
        with patch("anima_mcp.config.get_calibration", return_value=calibration):
            result = await orchestrator._execute_anima_step(step)
        assert result["calibration"]["warmth_bias"] == 0.1


# ---------------------------------------------------------------------------
# _execute_unitares_step
# ---------------------------------------------------------------------------

class TestExecuteUnitaresStep:
    async def test_check_governance(self, orchestrator, mock_bridge):
        orchestrator._bridge = mock_bridge
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            step = WorkflowStep(
                name="s", server="unitares", tool="check_governance",
                arguments={},
            )
            result = await orchestrator._execute_unitares_step(step)
        assert result["action"] == "proceed"

    async def test_process_update_no_bridge(self, orchestrator):
        step = WorkflowStep(
            name="s", server="unitares", tool="process_agent_update", arguments={},
        )
        result = await orchestrator._execute_unitares_step(step)
        assert "error" in result

    async def test_unknown_tool_raises(self, orchestrator):
        step = WorkflowStep(name="s", server="unitares", tool="not_a_tool", arguments={})
        with pytest.raises(ValueError, match="Unknown unitares tool"):
            await orchestrator._execute_unitares_step(step)


# ---------------------------------------------------------------------------
# workflow_check_state_and_governance
# ---------------------------------------------------------------------------

class TestWorkflowCheckStateAndGovernance:
    async def test_returns_error_when_state_fails(self, orchestrator):
        result = await orchestrator.workflow_check_state_and_governance()
        assert "error" in result

    async def test_returns_combined_result(self, orchestrator, mock_store, mock_bridge):
        orchestrator._anima_store = mock_store
        orchestrator._bridge = mock_bridge
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            result = await orchestrator.workflow_check_state_and_governance()
        assert "state" in result
        assert "governance" in result


# ---------------------------------------------------------------------------
# workflow_monitor_and_govern
# ---------------------------------------------------------------------------

class TestWorkflowMonitorAndGovern:
    async def test_delegates_to_check_state(self, orchestrator, mock_store, mock_bridge):
        orchestrator._anima_store = mock_store
        orchestrator._bridge = mock_bridge
        readings = make_readings()
        anima = make_anima()
        with patch.object(orchestrator, "_get_readings_and_anima", return_value=(readings, anima)):
            result = await orchestrator.workflow_monitor_and_govern()
        assert "state" in result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetOrchestrator:
    def test_returns_same_instance(self):
        import anima_mcp.workflow_orchestrator as mod
        old = mod._orchestrator
        mod._orchestrator = None
        try:
            o1 = get_orchestrator()
            o2 = get_orchestrator()
            assert o1 is o2
        finally:
            mod._orchestrator = old


# ---------------------------------------------------------------------------
# _get_readings_and_anima
# ---------------------------------------------------------------------------

class TestGetReadingsAndAnima:
    def test_uses_shared_memory_when_present(self, orchestrator):
        readings = make_readings()
        anima = make_anima()
        shm_data = {
            "readings": {
                "timestamp": datetime.now().isoformat(),
                "cpu_temp_c": readings.cpu_temp_c,
                "ambient_temp_c": readings.ambient_temp_c,
                "humidity_pct": readings.humidity_pct,
                "light_lux": readings.light_lux,
                "cpu_percent": readings.cpu_percent,
                "memory_percent": readings.memory_percent,
                "disk_percent": readings.disk_percent,
                "pressure_hpa": readings.pressure_hpa,
            },
            "anima": {"warmth": 0.5},
        }
        shm_client = MagicMock()
        shm_client.read.return_value = shm_data

        with patch("anima_mcp.workflow_orchestrator.SharedMemoryClient", return_value=shm_client), \
             patch("anima_mcp.workflow_orchestrator.get_calibration", return_value=MagicMock()), \
             patch("anima_mcp.workflow_orchestrator.sense_self", return_value=anima):
            out_readings, out_anima = orchestrator._get_readings_and_anima()

        assert out_readings is not None
        assert out_anima is anima

    def test_falls_back_to_direct_sensors_when_shm_missing(self, orchestrator):
        readings = make_readings()
        anima = make_anima()
        sensors = MagicMock()
        sensors.read.return_value = readings
        orchestrator._anima_sensors = sensors

        shm_client = MagicMock()
        shm_client.read.return_value = None

        with patch("anima_mcp.workflow_orchestrator.SharedMemoryClient", return_value=shm_client), \
             patch("anima_mcp.workflow_orchestrator.get_calibration", return_value=MagicMock()), \
             patch("anima_mcp.workflow_orchestrator.sense_self", return_value=anima):
            out_readings, out_anima = orchestrator._get_readings_and_anima()

        assert out_readings is readings
        assert out_anima is anima

    def test_returns_none_when_both_sources_fail(self, orchestrator):
        orchestrator._anima_sensors = MagicMock()
        orchestrator._anima_sensors.read.side_effect = RuntimeError("sensor down")
        shm_client = MagicMock()
        shm_client.read.return_value = {"readings": {"timestamp": "bad-ts"}, "anima": {}}

        with patch("anima_mcp.workflow_orchestrator.SharedMemoryClient", return_value=shm_client), \
             patch("anima_mcp.workflow_orchestrator.get_calibration", side_effect=RuntimeError("bad cal")), \
             patch("anima_mcp.workflow_orchestrator.sense_self", side_effect=RuntimeError("bad")):
            out_readings, out_anima = orchestrator._get_readings_and_anima()

        assert out_readings is None
        assert out_anima is None
