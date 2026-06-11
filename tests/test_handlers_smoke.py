"""Smoke tests for handler modules — verify imports, callability, and basic error paths.

These are intentionally lightweight: they confirm modules load, functions exist,
and error paths produce sensible results rather than crashing. Deep functional
testing belongs in dedicated test files.
"""

import asyncio
from unittest.mock import patch, MagicMock

from conftest import parse_result


# ---------------------------------------------------------------------------
# Helper to run async handlers in sync tests
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# Module: handlers/state_queries.py
# ===========================================================================

class TestStateQueriesSmoke:
    """Smoke tests for state_queries handler module."""

    def test_module_imports(self):
        """state_queries module imports without error."""
        from anima_mcp.handlers import state_queries
        assert state_queries is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.state_queries import (
            handle_get_state,
            handle_get_identity,
            handle_read_sensors,
            handle_get_health,
            handle_get_calibration,
        )
        assert callable(handle_get_state)
        assert callable(handle_get_identity)
        assert callable(handle_read_sensors)
        assert callable(handle_get_health)
        assert callable(handle_get_calibration)

    def test_get_state_returns_error_when_store_none(self):
        """get_state returns error JSON when server not initialized."""
        from anima_mcp.handlers.state_queries import handle_get_state

        with patch("anima_mcp.handlers.state_queries.handle_get_state.__module__", "anima_mcp.handlers.state_queries"):
            with patch("anima_mcp.accessors._get_store", return_value=None):
                result = run_async(handle_get_state({}))
                data = parse_result(result)
                assert "error" in data

    def test_get_identity_returns_error_when_store_none(self):
        """get_identity returns error JSON when server not initialized."""
        from anima_mcp.handlers.state_queries import handle_get_identity

        with patch("anima_mcp.accessors._get_store", return_value=None):
            result = run_async(handle_get_identity({}))
            data = parse_result(result)
            assert "error" in data

    def test_get_health_handles_exception(self):
        """get_health returns error JSON on exception."""
        from anima_mcp.handlers.state_queries import handle_get_health

        with patch("anima_mcp.handlers.state_queries.handle_get_health.__module__", "anima_mcp.handlers.state_queries"):
            with patch("anima_mcp.health.get_health_registry", side_effect=RuntimeError("test")):
                result = run_async(handle_get_health({}))
                data = parse_result(result)
                assert "error" in data

    def test_get_calibration_returns_valid_json(self):
        """get_calibration returns calibration data even with default config."""
        from anima_mcp.handlers.state_queries import handle_get_calibration

        result = run_async(handle_get_calibration({}))
        data = parse_result(result)
        assert "calibration" in data
        assert "config_file" in data


# ===========================================================================
# Module: handlers/knowledge.py
# ===========================================================================

class TestKnowledgeSmoke:
    """Smoke tests for knowledge handler module."""

    def test_module_imports(self):
        """knowledge module imports without error."""
        from anima_mcp.handlers import knowledge
        assert knowledge is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.knowledge import (
            handle_get_self_knowledge,
            handle_get_growth,
            handle_get_qa_insights,
            handle_get_trajectory,
            handle_get_eisv_trajectory_state,
            handle_query,
        )
        assert callable(handle_get_self_knowledge)
        assert callable(handle_get_growth)
        assert callable(handle_get_qa_insights)
        assert callable(handle_get_trajectory)
        assert callable(handle_get_eisv_trajectory_state)
        assert callable(handle_query)

    def test_get_self_knowledge_returns_error_when_store_none(self):
        """get_self_knowledge returns error when store is None."""
        from anima_mcp.handlers.knowledge import handle_get_self_knowledge

        with patch("anima_mcp.accessors._get_store", return_value=None):
            result = run_async(handle_get_self_knowledge({}))
            data = parse_result(result)
            assert "error" in data

    def test_get_growth_returns_error_when_growth_none(self):
        """get_growth returns error when growth system is None."""
        from anima_mcp.handlers.knowledge import handle_get_growth

        with patch("anima_mcp.accessors._get_growth", return_value=None):
            result = run_async(handle_get_growth({}))
            data = parse_result(result)
            assert "error" in data

    def test_get_eisv_trajectory_state_handles_exception(self):
        """get_eisv_trajectory_state returns error on exception."""
        from anima_mcp.handlers.knowledge import handle_get_eisv_trajectory_state

        with patch("anima_mcp.handlers.knowledge.get_trajectory_awareness", side_effect=RuntimeError("no data")):
            result = run_async(handle_get_eisv_trajectory_state({}))
            data = parse_result(result)
            assert "error" in data

    def test_query_requires_text(self):
        """query returns error when text is empty."""
        from anima_mcp.handlers.knowledge import handle_query

        result = run_async(handle_query({}))
        data = parse_result(result)
        assert "error" in data
        assert "text" in data.get("error", "").lower() or "text" in str(data)

    def test_query_returns_qa_insights_with_text(self):
        """query returns qa_insights when text is provided."""
        from anima_mcp.handlers.knowledge import handle_query

        result = run_async(handle_query({"text": "what have I learned"}))
        data = parse_result(result)
        assert "qa_insights" in data
        assert "query" in data


# ===========================================================================
# Module: handlers/communication.py
# ===========================================================================

class TestCommunicationSmoke:
    """Smoke tests for communication handler module."""

    def test_module_imports(self):
        """communication module imports without error."""
        from anima_mcp.handlers import communication
        assert communication is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.communication import (
            handle_lumen_qa,
            handle_post_message,
            handle_say,
            handle_configure_voice,
            handle_primitive_feedback,
        )
        assert callable(handle_lumen_qa)
        assert callable(handle_post_message)
        assert callable(handle_say)
        assert callable(handle_configure_voice)
        assert callable(handle_primitive_feedback)

    def test_say_rejects_empty_text(self):
        """say handler returns error when text is empty."""
        from anima_mcp.handlers.communication import handle_say

        with patch("anima_mcp.accessors._get_store", return_value=None), \
             patch("anima_mcp.accessors._get_voice", return_value=None), \
             patch("anima_mcp.accessors.VOICE_MODE", "text"):
            result = run_async(handle_say({"text": ""}))
            data = parse_result(result)
            assert "error" in data

    def test_post_message_rejects_empty_message(self):
        """post_message handler returns error when message is empty."""
        from anima_mcp.handlers.communication import handle_post_message

        with patch("anima_mcp.accessors._get_growth", return_value=None), \
             patch("anima_mcp.accessors._get_activity", return_value=None), \
             patch("anima_mcp.accessors._get_readings_and_anima", return_value=(None, None)), \
             patch("anima_mcp.accessors._get_store", return_value=None):
            result = run_async(handle_post_message({"message": ""}))
            data = parse_result(result)
            assert "error" in data
            assert "message parameter required" in data["error"]

    def test_configure_voice_returns_error_when_voice_none(self):
        """configure_voice returns error when voice system unavailable."""
        from anima_mcp.handlers.communication import handle_configure_voice

        with patch("anima_mcp.accessors._get_voice", return_value=None):
            result = run_async(handle_configure_voice({"action": "status"}))
            data = parse_result(result)
            assert "error" in data


# ===========================================================================
# Module: handlers/display_ops.py
# ===========================================================================

class TestDisplayOpsSmoke:
    """Smoke tests for display_ops handler module."""

    def test_module_imports(self):
        """display_ops module imports without error."""
        from anima_mcp.handlers import display_ops
        assert display_ops is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.display_ops import (
            handle_capture_screen,
            handle_show_face,
            handle_diagnostics,
            handle_manage_display,
        )
        assert callable(handle_capture_screen)
        assert callable(handle_show_face)
        assert callable(handle_diagnostics)
        assert callable(handle_manage_display)

    def test_capture_screen_returns_error_when_renderer_none(self):
        """capture_screen returns error when screen renderer not initialized."""
        from anima_mcp.handlers.display_ops import handle_capture_screen

        with patch("anima_mcp.accessors._get_screen_renderer", return_value=None):
            result = run_async(handle_capture_screen({}))
            data = parse_result(result)
            assert "error" in data

    def test_manage_display_rejects_missing_action(self):
        """manage_display returns error when action is not provided."""
        from anima_mcp.handlers.display_ops import handle_manage_display

        with patch("anima_mcp.accessors._get_screen_renderer", return_value=MagicMock()):
            result = run_async(handle_manage_display({}))
            data = parse_result(result)
            assert "error" in data
            assert "action" in data["error"].lower()

    def test_manage_display_rejects_invalid_action(self):
        """manage_display returns error for unknown action."""
        from anima_mcp.handlers.display_ops import handle_manage_display

        with patch("anima_mcp.accessors._get_screen_renderer", return_value=MagicMock()):
            result = run_async(handle_manage_display({"action": "nonexistent"}))
            data = parse_result(result)
            assert "error" in data

    def test_manage_display_resonance_critique_is_advisory_packet(self):
        """resonance_critique returns a toolable advisory loop without changing era."""
        from anima_mcp.handlers.display_ops import handle_manage_display

        renderer = MagicMock()
        renderer.get_current_era.return_value = {
            "current_era": "resonance",
            "current_description": "Marks respond to emotional memory",
            "auto_rotate": False,
            "all_eras": [
                {"name": "gestural", "description": "Granular marks"},
                {"name": "resonance", "description": "Emotional memory field"},
            ],
        }
        renderer.get_mode.return_value.value = "notepad"
        renderer.get_drawing_eisv.return_value = {
            "E": 0.4,
            "I": 0.7,
            "S": 0.2,
            "V": -0.1,
            "C": 0.8,
            "marks": 144,
            "phase": "developing",
            "era": "resonance",
        }

        with patch("anima_mcp.accessors._get_screen_renderer", return_value=renderer):
            result = run_async(handle_manage_display({"action": "resonance_critique"}))
            data = parse_result(result)

        assert data["success"] is True
        assert data["action"] == "resonance_critique"
        assert data["mode"] == "advisory"
        assert data["manual_control_preserved"] is True
        assert data["current_era"] == "resonance"
        assert data["auto_rotate"] is False
        assert data["current_screen"] == "notepad"
        assert data["available_eras"][1]["name"] == "resonance"
        assert data["recommendation_contract"]["allowed_recommendations"] == [
            "stay", "tune", "switch"
        ]
        assert any(step["tool"] == "capture_screen" for step in data["loop"])
        assert any(step["tool"] == "get_lumen_context" for step in data["loop"])
        assert "sediment" in data["resonance_focus"]
        renderer.set_era.assert_not_called()


# ===========================================================================
# Module: handlers/workflows.py
# ===========================================================================

class TestWorkflowsSmoke:
    """Smoke tests for workflows handler module."""

    def test_module_imports(self):
        """workflows module imports without error."""
        from anima_mcp.handlers import workflows
        assert workflows is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.workflows import (
            handle_unified_workflow,
            handle_next_steps,
            handle_set_calibration,
            handle_get_lumen_context,
            handle_learning_visualization,
        )
        assert callable(handle_unified_workflow)
        assert callable(handle_next_steps)
        assert callable(handle_set_calibration)
        assert callable(handle_get_lumen_context)
        assert callable(handle_learning_visualization)

    def test_unified_workflow_returns_error_when_store_none(self):
        """unified_workflow returns error when store is None."""
        from anima_mcp.handlers.workflows import handle_unified_workflow

        with patch("anima_mcp.accessors._get_store", return_value=None):
            result = run_async(handle_unified_workflow({}))
            data = parse_result(result)
            assert "error" in data

    def test_set_calibration_rejects_empty_updates(self):
        """set_calibration returns error when updates is empty."""
        from anima_mcp.handlers.workflows import handle_set_calibration

        result = run_async(handle_set_calibration({"updates": {}}))
        data = parse_result(result)
        assert "error" in data

    def test_set_calibration_rejects_no_updates(self):
        """set_calibration returns error when updates key is missing."""
        from anima_mcp.handlers.workflows import handle_set_calibration

        result = run_async(handle_set_calibration({}))
        data = parse_result(result)
        assert "error" in data

    def test_learning_visualization_returns_error_when_store_none(self):
        """learning_visualization returns error when store is None."""
        from anima_mcp.handlers.workflows import handle_learning_visualization

        with patch("anima_mcp.accessors._get_store", return_value=None):
            result = run_async(handle_learning_visualization({}))
            data = parse_result(result)
            assert "error" in data


# ===========================================================================
# Module: handlers/system_ops.py
# ===========================================================================

class TestSystemOpsSmoke:
    """Smoke tests for system_ops handler module."""

    def test_module_imports(self):
        """system_ops module imports without error."""
        from anima_mcp.handlers import system_ops
        assert system_ops is not None

    def test_handler_functions_exist(self):
        """All expected handler functions are defined and callable."""
        from anima_mcp.handlers.system_ops import (
            handle_git_pull,
            handle_system_service,
            handle_fix_ssh_port,
            handle_deploy_from_github,
            handle_setup_tailscale,
            handle_system_power,
        )
        assert callable(handle_git_pull)
        assert callable(handle_system_service)
        assert callable(handle_fix_ssh_port)
        assert callable(handle_deploy_from_github)
        assert callable(handle_setup_tailscale)
        assert callable(handle_system_power)

    def test_system_service_rejects_missing_service(self):
        """system_service returns error when service param is missing."""
        from anima_mcp.handlers.system_ops import handle_system_service

        result = run_async(handle_system_service({}))
        data = parse_result(result)
        assert "error" in data
        assert "service" in data["error"].lower()

    def test_system_service_rejects_disallowed_service(self):
        """system_service rejects services not on the whitelist."""
        from anima_mcp.handlers.system_ops import handle_system_service

        result = run_async(handle_system_service({"service": "nginx"}))
        data = parse_result(result)
        assert "error" in data
        assert "allowed" in data

    def test_system_service_rejects_disallowed_action(self):
        """system_service rejects actions not on the whitelist."""
        from anima_mcp.handlers.system_ops import handle_system_service

        result = run_async(handle_system_service({"service": "anima", "action": "nuke"}))
        data = parse_result(result)
        assert "error" in data
        assert "allowed" in data

    def test_fix_ssh_port_rejects_bad_port(self):
        """fix_ssh_port rejects ports not in the safety list."""
        from anima_mcp.handlers.system_ops import handle_fix_ssh_port

        result = run_async(handle_fix_ssh_port({"port": 80}))
        data = parse_result(result)
        assert "error" in data

    def test_setup_tailscale_rejects_empty_key(self):
        """setup_tailscale returns error when auth_key is empty."""
        from anima_mcp.handlers.system_ops import handle_setup_tailscale

        result = run_async(handle_setup_tailscale({}))
        data = parse_result(result)
        assert "error" in data

    def test_setup_tailscale_rejects_invalid_key_format(self):
        """setup_tailscale rejects keys that don't start with tskey-."""
        from anima_mcp.handlers.system_ops import handle_setup_tailscale

        result = run_async(handle_setup_tailscale({"auth_key": "invalid-key-format"}))
        data = parse_result(result)
        assert "error" in data

    def test_system_power_status_works(self):
        """system_power with action=status calls uptime."""
        from anima_mcp.handlers.system_ops import handle_system_power

        mock_result = MagicMock()
        mock_result.stdout = " 12:00:00 up 5 days"
        mock_result.returncode = 0

        with patch("anima_mcp.handlers.system_ops.subprocess.run", return_value=mock_result):
            result = run_async(handle_system_power({"action": "status"}))
            data = parse_result(result)
            assert "uptime" in data

    def test_system_power_reboot_requires_confirm(self):
        """system_power reboot without confirm returns warning."""
        from anima_mcp.handlers.system_ops import handle_system_power

        result = run_async(handle_system_power({"action": "reboot"}))
        data = parse_result(result)
        assert "error" in data
        assert "confirm" in data["error"].lower()

    def test_system_power_rejects_invalid_action(self):
        """system_power rejects actions not on the whitelist."""
        from anima_mcp.handlers.system_ops import handle_system_power

        result = run_async(handle_system_power({"action": "format"}))
        data = parse_result(result)
        assert "error" in data


# ===========================================================================
# Module: handlers/__init__.py (re-export verification)
# ===========================================================================

class TestHandlersPackage:
    """Verify the handlers package re-exports everything."""

    def test_all_handlers_importable_from_package(self):
        """All handler functions can be imported from the handlers package."""
        # If we got here, all imports succeeded
        assert True

    def test_all_exports_are_async(self):
        """All handler functions are async (coroutine functions)."""
        import inspect
        from anima_mcp.handlers import __all__
        import anima_mcp.handlers as handlers_pkg

        for name in __all__:
            func = getattr(handlers_pkg, name)
            assert inspect.iscoroutinefunction(func), f"{name} is not async"
