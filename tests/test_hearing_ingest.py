"""
G1 — BLOCKING regression test for the hearing wire's anti-RLHF guarantee.

The acoustic channel must change Lumen's world-model (salience + learned
baseline) WITHOUT ever touching the reinforcement path. Specifically:

  - A loud/spiking sound level produces a NONZERO salience bump on the
    "sound_level" dimension of the experiential filter.
  - That bump is computed in hearing_ingest and fed to experiential_filter
    DIRECTLY — it never flows through metacognition, so "sound_level" /
    "voice_activity" can never land in pred_error.surprise_sources (the
    aggregate that drives preferences.record_event("disruption", -0.2)).
  - No new code path calls preferences.record_event.
  - When hearing_available is False, the baseline is FROZEN: observe() is not
    called.

Without this test the anti-RLHF guarantee is prose. See
docs/proposals/hearing-wire.md (risk G1).
"""

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from anima_mcp import hearing_ingest
from anima_mcp.hearing_ingest import ingest_acoustic
from anima_mcp.adaptive_prediction import AdaptivePredictionModel
from anima_mcp.experiential_filter import (
    ExperientialFilter,
    DIMENSIONS,
    SOURCE_TO_DIM,
)
from anima_mcp.metacognition import MetacognitiveMonitor
from anima_mcp.anima import sense_self
from anima_mcp.sensors.base import SensorReadings


FIXED_TIME = datetime(2026, 6, 29, 12, 0, 0)


@pytest.fixture
def adaptive_model(tmp_path):
    return AdaptivePredictionModel(persistence_path=tmp_path / "patterns.json")


@pytest.fixture
def exp_filter(tmp_path):
    return ExperientialFilter(persistence_path=str(tmp_path / "ef.json"))


def _prime_quiet_baseline(adaptive_model, exp_filter, level=500.0, n=8):
    """Teach the model a quiet-room baseline at FIXED_TIME's pattern key."""
    for _ in range(n):
        ingest_acoustic(
            level,
            hearing_available=True,
            adaptive_model=adaptive_model,
            exp_filter=exp_filter,
            current_time=FIXED_TIME,
            current_light=300.0,
            current_temp=22.0,
        )


def test_new_dimensions_registered():
    """sound_level / voice_activity are real, mapped dimensions."""
    assert "sound_level" in DIMENSIONS
    assert "voice_activity" in DIMENSIONS
    assert SOURCE_TO_DIM["sound_level"] == "sound_level"
    assert SOURCE_TO_DIM["voice_activity"] == "voice_activity"


def test_loud_spike_bumps_salience(adaptive_model, exp_filter):
    """A loud spike against a learned quiet baseline raises sound_level salience."""
    _prime_quiet_baseline(adaptive_model, exp_filter)
    before = exp_filter.get_salience("sound_level")

    result = ingest_acoustic(
        9000.0,  # loud spike, far above the ~500 learned baseline
        hearing_available=True,
        adaptive_model=adaptive_model,
        exp_filter=exp_filter,
        current_time=FIXED_TIME,
        current_light=300.0,
        current_temp=22.0,
    )
    after = exp_filter.get_salience("sound_level")

    assert result["surprise"] > 0.0, "spike should produce nonzero surprise"
    assert after > before, "salience must rise (nonzero bump) on the spike"


def test_sound_never_enters_metacognition_surprise_sources():
    """Even with a populated sound_level, metacog never emits a sound source.

    This guards the channel separation: metacognition is the only producer of
    pred_error.surprise_sources, and the punishment path keys off that. Sound
    must be structurally invisible to it.
    """
    metacog = MetacognitiveMonitor()
    readings = SensorReadings(
        timestamp=FIXED_TIME,
        cpu_temp_c=55.0,
        ambient_temp_c=25.0,
        humidity_pct=40.0,
        light_lux=300.0,
        pressure_hpa=1013.0,
        cpu_percent=10.0,
        memory_percent=30.0,
        disk_percent=50.0,
        hearing_available=True,
        sound_level=99999.0,  # absurdly loud — must not leak anywhere
    )
    anima = sense_self(readings)

    # Observe twice so a prediction/error actually forms.
    metacog.observe(readings, anima)
    pred_error = metacog.observe(readings, anima)

    assert "sound_level" not in pred_error.surprise_sources
    assert "voice_activity" not in pred_error.surprise_sources


def test_no_record_event_reference_in_module():
    """The router must not reach the reinforcement path at all.

    AST guard (ignores docstrings/comments, which legitimately name these
    symbols in prose): hearing_ingest must contain no executable call to
    ``record_event`` and no reference to ``preferences`` or ``metacog`` /
    ``metacognition`` in real code. If a future 'natural' commit wires sound
    through the punishment path, this fails.
    """
    import ast

    tree = ast.parse(inspect.getsource(hearing_ingest))
    forbidden_names = {"record_event", "preferences", "metacog", "metacognition"}

    offenders = []
    for node in ast.walk(tree):
        # Attribute access like x.record_event / preferences.record_event
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            offenders.append(node.attr)
        # Bare name references / imports of forbidden modules
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            offenders.append(node.id)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            if any(f in mod for f in forbidden_names) or any(
                any(f in n for f in forbidden_names) for n in names
            ):
                offenders.append(mod or ",".join(names))

    assert not offenders, f"hearing_ingest reaches forbidden symbols: {offenders}"


def test_frozen_when_hearing_unavailable():
    """hearing_available=False ⇒ no predict, no observe, no salience update."""
    adaptive_model = MagicMock()
    exp_filter = MagicMock()

    result = ingest_acoustic(
        9000.0,
        hearing_available=False,
        adaptive_model=adaptive_model,
        exp_filter=exp_filter,
        current_time=FIXED_TIME,
        current_light=300.0,
        current_temp=22.0,
    )

    assert result["frozen"] is True
    assert result["observed"] is False
    adaptive_model.observe.assert_not_called()
    adaptive_model.predict.assert_not_called()
    exp_filter.update_from_surprise.assert_not_called()


def test_active_path_observes_baseline():
    """hearing_available=True ⇒ observe() IS called (baseline learns)."""
    adaptive_model = MagicMock()
    adaptive_model.predict.return_value = (500.0, 0.8)
    exp_filter = MagicMock()

    ingest_acoustic(
        9000.0,
        hearing_available=True,
        adaptive_model=adaptive_model,
        exp_filter=exp_filter,
        current_time=FIXED_TIME,
        current_light=300.0,
        current_temp=22.0,
    )

    adaptive_model.observe.assert_called_once()
    observed_obs = adaptive_model.observe.call_args.args[0]
    assert observed_obs == {"sound_level": 9000.0}
    # Salience fed directly, only for the sound_level dimension.
    exp_filter.update_from_surprise.assert_called_once()
    sources = exp_filter.update_from_surprise.call_args.args[0]
    assert sources == ["sound_level"]


def test_cold_start_is_neutral():
    """With no learned pattern, a first sound is low/neutral surprise."""
    adaptive_model = MagicMock()
    adaptive_model.predict.return_value = (None, 0.0)  # cold start
    exp_filter = MagicMock()

    result = ingest_acoustic(
        9000.0,
        hearing_available=True,
        adaptive_model=adaptive_model,
        exp_filter=exp_filter,
        current_time=FIXED_TIME,
        current_light=300.0,
        current_temp=22.0,
    )

    assert result["surprise"] == 0.0
    # Still learns the baseline even on cold start.
    adaptive_model.observe.assert_called_once()
