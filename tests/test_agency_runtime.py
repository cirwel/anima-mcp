"""Integration tests for the live server agency wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from anima_mcp.agency import Action, ActionType
from anima_mcp.agency_runtime import execute_action, run_agency_phase


def test_execute_focus_consumes_curiosity_and_marks_real_success():
    action = Action(ActionType.FOCUS_ATTENTION, {"sensor": "ambient_temp"})
    selector = MagicMock()
    metacog = MagicMock()
    prediction_error = SimpleNamespace(surprise=0.5, surprise_sources=["ambient_temp"])

    execute_action(
        action,
        ctx=SimpleNamespace(leds=None),
        action_selector=selector,
        anima=SimpleNamespace(),
        readings=SimpleNamespace(),
        prediction_error=prediction_error,
        metacog=metacog,
        surprise_level=0.5,
        surprise_sources=["ambient_temp"],
    )

    selector.set_attention_focus.assert_called_once_with("ambient_temp")
    metacog.record_curiosity.assert_called_once_with(["ambient_temp"], prediction_error)
    assert action.execution_succeeded is True


def test_unavailable_question_becomes_contradicting_belief_evidence():
    action = Action(ActionType.ASK_QUESTION)

    with patch("anima_mcp.agency_runtime._choose_question", return_value=None), \
         patch("anima_mcp.agency_runtime._record_question_evidence") as evidence:
        execute_action(
            action,
            ctx=SimpleNamespace(leds=None),
            action_selector=MagicMock(),
            anima=SimpleNamespace(),
            readings=SimpleNamespace(),
            prediction_error=None,
            metacog=None,
            surprise_level=0.6,
            surprise_sources=["light_lux"],
        )

    assert action.execution_succeeded is False
    evidence.assert_called_once_with(
        supports=False,
        surprise_level=0.6,
        source="server:agency_question_unavailable",
    )


def test_live_phase_wires_inputs_reinforces_prior_outcome_and_executes():
    previous = Action(ActionType.ASK_QUESTION, execution_succeeded=True)
    ctx = SimpleNamespace(
        store=SimpleNamespace(db_path="/tmp/lumen-test.db"),
        last_action=previous,
        last_state_before={
            "warmth": 0.4,
            "clarity": 0.5,
            "stability": 0.6,
            "presence": 0.7,
            "last_surprise": 0.6,
        },
        last_pathway_context="hi|neut|want|act",
        tension_tracker=None,
        leds=None,
    )
    anima = SimpleNamespace(warmth=0.5, clarity=0.6, stability=0.7, presence=0.8)
    prediction_error = SimpleNamespace(
        surprise=0.4,
        surprise_sources=["light_lux"],
    )
    selector = MagicMock()
    selected = Action(ActionType.STAY_QUIET)
    selector.select_action.return_value = selected
    selector.record_outcome.return_value = SimpleNamespace(reward=0.35)
    selector.get_action_stats.return_value = {"action_counts": {}}
    preferences = MagicMock()
    preferences.get_overall_satisfaction.return_value = 0.6
    pathways = MagicMock()
    pathways.get_all_strengths.return_value = {"ask_question": 0.7}
    pathways.reinforce.side_effect = RuntimeError("pathway store unavailable")
    self_model = MagicMock()
    self_model.predict_own_response.return_value = {"surprise_likelihood": 0.8}
    metacog = MagicMock()

    with patch("anima_mcp.agency_runtime.get_action_selector", return_value=selector), \
         patch("anima_mcp.preferences.get_preference_system", return_value=preferences), \
         patch("anima_mcp.weighted_pathways.get_weighted_pathways", return_value=pathways), \
         patch("anima_mcp.agency_runtime._get_metacog_monitor", return_value=metacog), \
         patch("anima_mcp.self_model.get_self_model", return_value=self_model), \
         patch("anima_mcp.agency_runtime._get_last_shm_data", return_value={
             "inner_life": {"drives": {"warmth": 0.3}},
             "activity": {"level": "drowsy"},
         }):
        result = run_agency_phase(
            ctx, anima, SimpleNamespace(), prediction_error, loop_count=1,
        )

    pathways.reinforce.assert_called_once_with(
        "hi|neut|want|act", "ask_question", 0.35,
    )
    kwargs = selector.select_action.call_args.kwargs
    assert kwargs["preferences"] is preferences
    assert kwargs["self_predictions"] == {"surprise_likelihood": 0.8}
    assert kwargs["drives"] == {"warmth": 0.3}
    assert kwargs["pathway_strengths"] == {"ask_question": 0.7}
    assert ActionType.EXPLORE not in kwargs["available_actions"]
    assert result.execution_succeeded is True
    assert ctx.last_action is result
    assert ctx.last_pathway_context == "hi|neut|want|drow"


def test_live_phase_records_actuator_exception_as_failed_execution():
    ctx = SimpleNamespace(
        store=SimpleNamespace(db_path="/tmp/lumen-test.db"),
        last_action=None,
        last_state_before=None,
        last_pathway_context=None,
        tension_tracker=None,
        leds=None,
    )
    anima = SimpleNamespace(warmth=0.5, clarity=0.6, stability=0.7, presence=0.8)
    selector = MagicMock()
    selected = Action(ActionType.STAY_QUIET)
    selector.select_action.return_value = selected
    selector.get_action_stats.return_value = {"action_counts": {}}
    preferences = MagicMock()
    pathways = MagicMock()

    with patch("anima_mcp.agency_runtime.get_action_selector", return_value=selector), \
         patch("anima_mcp.preferences.get_preference_system", return_value=preferences), \
         patch("anima_mcp.weighted_pathways.get_weighted_pathways", return_value=pathways), \
         patch("anima_mcp.agency_runtime._get_metacog_monitor", return_value=None), \
         patch("anima_mcp.self_model.get_self_model", return_value=None), \
         patch("anima_mcp.agency_runtime._get_last_shm_data", return_value={}), \
         patch(
             "anima_mcp.agency_runtime.execute_action",
             side_effect=RuntimeError("LED bus fault"),
         ):
        result = run_agency_phase(
            ctx, anima, SimpleNamespace(), prediction_error=None, loop_count=1,
        )

    assert result is selected
    assert result.execution_succeeded is False
    assert result.execution_detail == "actuator error: RuntimeError: LED bus fault"
    assert ctx.last_action is result
