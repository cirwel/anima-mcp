from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conftest import parse_result


@pytest.mark.asyncio
class TestNextStepsExtended:
    async def test_next_steps_success_with_bridge_connected(self):
        from anima_mcp.handlers.workflows import handle_next_steps

        class Bridge:
            async def check_availability(self):
                return True

        advocate = SimpleNamespace(
            analyze_current_state=lambda **kwargs: [{"priority": "high"}],
            get_next_steps_summary=lambda: {
                "next_action": {
                    "priority": "high",
                    "feeling": "calm",
                    "desire": "observe",
                    "action": "watch",
                },
                "total_steps": 3,
                "critical": 1,
                "high": 1,
                "medium": 1,
                "low": 0,
                "all_steps": ["watch", "reflect"],
            },
        )
        anima = SimpleNamespace(warmth=0.6, clarity=0.7, stability=0.8, presence=0.5)
        readings = SimpleNamespace()
        display = SimpleNamespace(is_available=lambda: True)
        eisv = SimpleNamespace(to_dict=lambda: {"E": 0.7})
        attention_system = SimpleNamespace(
            attention=lambda **_: {"items": [], "active_count": 0}
        )

        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace()), \
             patch("anima_mcp.accessors._get_sensors", return_value=SimpleNamespace()), \
             patch("anima_mcp.accessors._get_display", return_value=display), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_server_bridge", return_value=Bridge()), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.next_steps_advocate.get_advocate", return_value=advocate), \
             patch("anima_mcp.self_iteration.get_self_iteration_system", return_value=attention_system), \
             patch("anima_mcp.eisv_mapper.anima_to_body_eisv_projection", return_value=eisv):
            data = parse_result(await handle_next_steps({}))

        assert data["summary"]["priority"] == "high"
        assert data["current_state"]["unitares_connected"] is True
        assert data["current_state"]["eisv"]["E"] == 0.7
        assert data["current_state"]["body_eisv_projection"] == data["current_state"]["eisv"]
        assert data["current_state"]["eisv_source"] == "body_eisv_projection_legacy_alias"

    async def test_next_steps_bridge_exception_sets_error_status(self):
        from anima_mcp.handlers.workflows import handle_next_steps

        class Bridge:
            async def check_availability(self):
                raise RuntimeError("bridge down")

        advocate = SimpleNamespace(
            analyze_current_state=lambda **kwargs: [],
            get_next_steps_summary=lambda: {"next_action": {}, "total_steps": 0, "critical": 0, "high": 0, "medium": 0, "low": 0, "all_steps": []},
        )
        anima = SimpleNamespace(warmth=0.1, clarity=0.2, stability=0.3, presence=0.4)
        readings = SimpleNamespace()
        display = SimpleNamespace(is_available=lambda: False)
        eisv = SimpleNamespace(to_dict=lambda: {"E": 0.1})
        attention_system = SimpleNamespace(
            attention=lambda **_: {"items": [], "active_count": 0}
        )

        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace()), \
             patch("anima_mcp.accessors._get_sensors", return_value=SimpleNamespace()), \
             patch("anima_mcp.accessors._get_display", return_value=display), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
             patch("anima_mcp.accessors._get_server_bridge", return_value=Bridge()), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value=None), \
             patch("anima_mcp.next_steps_advocate.get_advocate", return_value=advocate), \
             patch("anima_mcp.self_iteration.get_self_iteration_system", return_value=attention_system), \
             patch("anima_mcp.eisv_mapper.anima_to_body_eisv_projection", return_value=eisv):
            data = parse_result(await handle_next_steps({}))

        assert data["current_state"]["unitares_connected"] is False
        assert "error:" in data["current_state"]["unitares_status"]


@pytest.mark.asyncio
class TestSetCalibrationExtended:
    async def test_set_calibration_rejects_invalid_calibration(self):
        from anima_mcp.handlers.workflows import handle_set_calibration

        calibration = SimpleNamespace(to_dict=lambda: {"ambient_temp_min": 10.0})
        updated_cal = SimpleNamespace(validate=lambda: (False, "bad bounds"), to_dict=lambda: {"ambient_temp_min": 10.0})

        with patch("anima_mcp.config.get_calibration", return_value=calibration), \
             patch("anima_mcp.config.ConfigManager", return_value=MagicMock()), \
             patch("anima_mcp.config.NervousSystemCalibration.from_dict", return_value=updated_cal):
            data = parse_result(await handle_set_calibration({"updates": {"ambient_temp_min": 99.0}}))

        assert "error" in data
        assert "Invalid calibration" in data["error"]

    async def test_set_calibration_success_includes_metadata(self):
        from anima_mcp.handlers.workflows import handle_set_calibration

        calibration = SimpleNamespace(to_dict=lambda: {"ambient_temp_min": 10.0})
        updated_cal = SimpleNamespace(
            validate=lambda: (True, None),
            to_dict=lambda: {"ambient_temp_min": 12.0},
        )
        cfg = SimpleNamespace(nervous_system=None)
        cfg_with_meta = SimpleNamespace(metadata={
            "calibration_last_updated": "2026-03-14T00:00:00",
            "calibration_last_updated_by": "agent",
            "calibration_update_count": 3,
        })
        manager = MagicMock()
        manager.load.return_value = cfg
        manager.save.return_value = True
        manager.reload.return_value = cfg_with_meta

        with patch("anima_mcp.config.get_calibration", return_value=calibration), \
             patch("anima_mcp.config.ConfigManager", return_value=manager), \
             patch("anima_mcp.config.NervousSystemCalibration.from_dict", return_value=updated_cal):
            data = parse_result(await handle_set_calibration({"updates": {"ambient_temp_min": 12.0}, "source": "agent"}))

        assert data["success"] is True
        assert data["metadata"]["update_count"] == 3

    async def test_set_calibration_save_failure(self):
        from anima_mcp.handlers.workflows import handle_set_calibration

        calibration = SimpleNamespace(to_dict=lambda: {"ambient_temp_min": 10.0})
        updated_cal = SimpleNamespace(validate=lambda: (True, None), to_dict=lambda: {"ambient_temp_min": 12.0})
        manager = MagicMock()
        manager.load.return_value = SimpleNamespace()
        manager.save.return_value = False

        with patch("anima_mcp.config.get_calibration", return_value=calibration), \
             patch("anima_mcp.config.ConfigManager", return_value=manager), \
             patch("anima_mcp.config.NervousSystemCalibration.from_dict", return_value=updated_cal):
            data = parse_result(await handle_set_calibration({"updates": {"ambient_temp_min": 12.0}}))

        assert data["error"] == "Failed to save calibration"


@pytest.mark.asyncio
class TestLumenContextExtended:
    async def test_get_lumen_context_records_interaction_level_and_eisv(self):
        from anima_mcp.handlers.workflows import handle_get_lumen_context

        class FakeReadings:
            def to_dict(self):
                return {"light_lux": 100}

        identity = SimpleNamespace(
            name="Lumen",
            creature_id="lmn-1",
            born_at=datetime(2026, 1, 1),
            total_awakenings=5,
            age_seconds=lambda: 3600,
            total_alive_seconds=1800,
            alive_ratio=lambda: 0.5,
        )
        store = SimpleNamespace(
            get_identity=lambda: identity,
            get_session_alive_seconds=lambda: 100,
            record_state=MagicMock(),
        )
        sensors = SimpleNamespace(is_pi=lambda: False)
        anima = SimpleNamespace(
            warmth=0.3,
            clarity=0.4,
            stability=0.5,
            presence=0.6,
            feeling=lambda: {"mood": "calm"},
        )
        eisv = SimpleNamespace(
            to_dict=lambda: {"E": 0.3, "I": 0.6, "S": 0.4, "V": -0.3}
        )
        # interaction_level now comes from visitor records, not from scanning
        # the message board for a msg_type nothing produces.
        growth = SimpleNamespace(interaction_level=lambda: 0.5)
        light_attribution = {
            "mode": "shadow",
            "status": "warming",
            "external_lux_residual": None,
            "used_by_clarity": False,
        }
        clarity_attribution = {
            "schema": "anima.clarity_attribution.v1",
            "status": "ready",
            "raw_value": 0.6,
            "published_value": 0.6,
        }

        with patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_sensors", return_value=sensors), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(FakeReadings(), anima)), \
             patch("anima_mcp.accessors._get_last_shm_data", return_value={
                 "light_attribution": light_attribution,
                 "clarity_attribution": clarity_attribution,
             }), \
             patch("anima_mcp.accessors._get_growth", return_value=growth), \
             patch("anima_mcp.eisv_mapper.anima_to_body_eisv_projection", return_value=eisv):
            data = parse_result(await handle_get_lumen_context({"include": ["identity", "anima", "sensors", "mood", "eisv"]}))

        assert data["identity"]["name"] == "Lumen"
        assert data["eisv"]["E"] == 0.3
        assert data["body_eisv_projection"] == data["eisv"]
        assert data["body_anima"] == data["anima"]
        assert data["state_space_provenance"]["anima"]["alias_of"] == "body_anima"
        assert data["light_attribution"] == light_attribution
        assert data["light_attribution"]["used_by_clarity"] is False
        assert data["clarity_attribution"] == clarity_attribution
        assert "mood" in data
        # A read does not record; the main loop owns state_history.
        store.record_state.assert_not_called()

    async def test_get_lumen_context_handles_identity_error(self):
        from anima_mcp.handlers.workflows import handle_get_lumen_context

        store = SimpleNamespace(get_identity=lambda: (_ for _ in ()).throw(RuntimeError("identity fail")))
        with patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_sensors", return_value=SimpleNamespace(is_pi=lambda: False)), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            data = parse_result(await handle_get_lumen_context({"include": "identity"}))

        assert "error" in data["identity"]


@pytest.mark.asyncio
class TestLearningVisualizationExtended:
    async def test_learning_visualization_success(self):
        from anima_mcp.handlers.workflows import handle_learning_visualization

        store = SimpleNamespace(db_path=":memory:")
        summary = {"dominant_pattern": "night calm"}
        visualizer = SimpleNamespace(get_learning_summary=lambda readings, anima: summary)

        with patch("anima_mcp.accessors._get_store", return_value=store), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(SimpleNamespace(), SimpleNamespace())), \
             patch("anima_mcp.learning_visualization.LearningVisualizer", return_value=visualizer):
            data = parse_result(await handle_learning_visualization({}))

        assert data["dominant_pattern"] == "night calm"

    async def test_learning_visualization_sensor_error(self):
        from anima_mcp.handlers.workflows import handle_learning_visualization

        with patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace(db_path=":memory:")), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)):
            data = parse_result(await handle_learning_visualization({}))

        assert data["error"] == "Unable to read sensor data"


@pytest.mark.asyncio
class TestNextStepsIntentions:
    """The handler must not let an unreadable growth store read as 'no plans'."""

    @staticmethod
    def _patches(growth):
        class Bridge:
            async def check_availability(self):
                return True

        anima = SimpleNamespace(warmth=0.7, clarity=0.8, stability=0.8, presence=0.7)
        display = SimpleNamespace(is_available=lambda: True)
        attention_system = SimpleNamespace(
            attention=lambda **_: {"items": [], "active_count": 0}
        )
        return [
            patch("anima_mcp.accessors._get_store", return_value=SimpleNamespace()),
            patch("anima_mcp.accessors._get_sensors", return_value=SimpleNamespace()),
            patch("anima_mcp.accessors._get_display", return_value=display),
            patch(
                "anima_mcp.accessors._get_readings_and_anima",
                return_value=(SimpleNamespace(), anima),
            ),
            patch("anima_mcp.accessors._get_server_bridge", return_value=Bridge()),
            patch("anima_mcp.accessors._get_last_shm_data", return_value=None),
            patch("anima_mcp.accessors._get_growth", return_value=growth),
            patch(
                "anima_mcp.self_iteration.get_self_iteration_system",
                return_value=attention_system,
            ),
            patch(
                "anima_mcp.eisv_mapper.anima_to_body_eisv_projection",
                return_value=SimpleNamespace(to_dict=lambda: {"E": 0.5}, entropy=0.2),
            ),
        ]

    async def _run(self, growth):
        from contextlib import ExitStack

        from anima_mcp.handlers.workflows import handle_next_steps

        with ExitStack() as stack:
            for p in self._patches(growth):
                stack.enter_context(p)
            return parse_result(await handle_next_steps({}))

    async def test_active_goal_becomes_the_next_action(self):
        goal = SimpleNamespace(
            status=SimpleNamespace(value="active"),
            to_dict=lambda: {
                "description": "complete 2000 drawings",
                "motivation": "",
                "progress": 0.779,
                "last_worked_on": datetime.now().isoformat(),
            },
        )
        growth = SimpleNamespace(_goals={"g1": goal}, _curiosities=[])
        data = await self._run(growth)

        assert data["summary"]["status"] == "quiet"
        assert data["summary"]["desire"] == "complete 2000 drawings"
        assert data["summary"]["action"] == "continue"
        assert data["summary"]["intentions"] == {
            "available": True,
            "goals": 1,
            "curiosities": 0,
            "error": None,
        }

    async def test_unreadable_growth_is_reported_not_silently_empty(self):
        data = await self._run(None)
        assert data["summary"]["intentions"]["available"] is False
        assert data["summary"]["intentions"]["error"] == "growth store not initialized"
        assert data["summary"]["intentions"]["goals"] == 0

    async def test_growth_raising_is_reported_not_swallowed(self):
        class Exploding:
            @property
            def _goals(self):
                raise RuntimeError("db locked")

        data = await self._run(Exploding())
        assert data["summary"]["intentions"]["available"] is False
        assert "db locked" in data["summary"]["intentions"]["error"]

    async def test_quiet_state_still_reports_the_nearest_margin(self):
        growth = SimpleNamespace(_goals={}, _curiosities=[])
        data = await self._run(growth)
        checks = data["summary"]["state_checks"]
        assert checks["evaluated"] == 5
        assert checks["fired"] == 0
        assert checks["nearest_threshold"]["margin"] > 0
        assert data["summary"]["status"] == "quiet"
        assert data["summary"]["last_analyzed"] is not None
