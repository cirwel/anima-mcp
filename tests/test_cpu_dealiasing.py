"""CPU% must reach the E/I pair through exactly one door.

Alpha = 1 - beta by construction (computational_neural.py): both bands are the
same CPU reading. Before 2026-08-14, warmth carried (beta+gamma)/2 at 0.20 and
clarity carried alpha at a 0.27 share — so one variable suppressed E and
inflated I simultaneously, and V = E - I double-counted it. #141/#166 removed
the alias from the EISV mappers; this pins the removal at the SOURCE, which the
mapper-level tests cannot see (they receive warmth/clarity as opaque scalars —
the Elixir regression test swept alpha while holding clarity constant, so it
was green against the live bug).

The one sanctioned door is the mapper's explicit neural_energy term in E.

Backtested on 14d of Lumen's history before flipping: warmth 0.453->0.535,
clarity 0.672->0.576, V -0.320->-0.166, spurious below-comfort warmth episodes
15.9%->2.0%, |V|>0.30 59.2%->10.3%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_mcp.anima import _sense_clarity, _sense_warmth
from anima_mcp.config import NervousSystemCalibration
from anima_mcp.sensors.base import SensorReadings


CAL = NervousSystemCalibration()


def _readings(beta: float) -> SensorReadings:
    """Fixed physical world; only the CPU-derived bands vary."""
    return SensorReadings(
        timestamp=1_000_000.0,
        cpu_temp_c=55.0,
        ambient_temp_c=26.0,
        light_lux=100.0,
        cpu_percent=beta * 100,
        memory_percent=40.0,
        eeg_alpha_power=1.0 - beta,
        eeg_beta_power=beta,
        eeg_gamma_power=0.2,
    )


BETAS = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]


def test_warmth_is_invariant_to_cpu_load():
    """Warmth is thermal state — its own docstring has said so all along."""
    values = {_sense_warmth(_readings(b), CAL) for b in BETAS}
    assert len(values) == 1, f"warmth moved with CPU load: {sorted(values)}"


def test_clarity_is_invariant_to_cpu_load():
    """'Alpha = relaxed awareness' was an idle Pi inflating I."""
    values = {_sense_clarity(_readings(b), CAL, prediction_accuracy=0.7)
              for b in BETAS}
    assert len(values) == 1, f"clarity moved with CPU load: {sorted(values)}"


def test_weight_tables_keep_the_neural_key_at_zero():
    """Present-at-zero is load-bearing: consumers fall back via .get(), and a
    config file predating the key must not resurrect the alias."""
    assert CAL.warmth_weights["neural"] == 0.0
    assert CAL.clarity_weights["neural"] == 0.0
    assert sum(CAL.warmth_weights.values()) == pytest.approx(1.0)
    assert sum(CAL.clarity_weights.values()) == pytest.approx(1.0)


def test_legacy_config_without_neural_key_does_not_resurrect_the_alias():
    """The .get() fallbacks used to be 0.20/0.30 — an old file missing the key
    would have silently re-aliased CPU% into both dimensions."""
    legacy = NervousSystemCalibration(
        warmth_weights={"cpu_temp": 0.4375, "ambient_temp": 0.5625},
        clarity_weights={"prediction_accuracy": 0.625,
                         "sensor_coverage": 0.1875, "world_light": 0.1875},
    )
    w = {_sense_warmth(_readings(b), legacy) for b in BETAS}
    c = {_sense_clarity(_readings(b), legacy, prediction_accuracy=0.7)
         for b in BETAS}
    assert len(w) == 1 and len(c) == 1


def test_cpu_still_reaches_E_through_the_mapper_and_only_the_mapper():
    """De-aliasing must not accidentally blind E to activation entirely."""
    from anima_mcp.anima import Anima
    from anima_mcp.eisv_mapper import anima_to_eisv

    anima = Anima(warmth=0.5, clarity=0.7, stability=0.8, presence=0.7,
                  readings=_readings(0.1))
    lo = anima_to_eisv(anima, _readings(0.1))
    hi = anima_to_eisv(anima, _readings(0.9))
    # E moves with load (the sanctioned door)...
    assert hi.energy > lo.energy
    # ...I does not (alpha reaches it nowhere).
    assert hi.integrity == lo.integrity


# ---------------------------------------------------------------------------
# Presence: resource headroom, each resource read once (#175)
# ---------------------------------------------------------------------------


def _readings_gamma(gamma: float, cpu: float = 10.0) -> SensorReadings:
    return SensorReadings(
        timestamp=1_000_000.0,
        cpu_temp_c=55.0, ambient_temp_c=26.0, light_lux=100.0,
        cpu_percent=cpu, memory_percent=40.0, disk_percent=30.0,
        eeg_alpha_power=0.9, eeg_beta_power=0.1, eeg_gamma_power=gamma,
    )


def test_presence_is_invariant_to_gamma():
    """Gamma is mostly cpu_percent renamed (r=+0.743 over 14d); CPU already
    has its own door into presence. Two doors made an effective ~0.40 share."""
    from anima_mcp.anima import _sense_presence
    values = {_sense_presence(_readings_gamma(g), CAL)
              for g in (0.0, 0.2, 0.5, 0.9)}
    assert len(values) == 1, f"presence moved with gamma: {sorted(values)}"


def test_cpu_still_moves_presence_through_its_own_door():
    from anima_mcp.anima import _sense_presence
    idle = _sense_presence(_readings_gamma(0.2, cpu=5.0), CAL)
    busy = _sense_presence(_readings_gamma(0.2, cpu=95.0), CAL)
    assert idle > busy


def test_stability_fallback_matches_the_dict_default():
    """dict said neural: 0.1, the consumer's .get fallback said 0.2 — a config
    missing the key silently doubled the weight (#175).

    theta/delta are pinned so the banded branch is taken. Before #175's
    fallback fix, omitting them made ``_sense_stability`` re-derive the bands
    from the ``get_computational_neural_state`` SINGLETON (which appends to
    ``_cpu_history`` and advances ``dt`` on every call), so two calls were not
    two evaluations of a pure function: this test once failed in CI at 0.583
    vs 0.601 and passed on rerun. Absent bands now read as full uncertainty
    without sampling (tests/test_neural_fallback.py). The subject here is the
    WEIGHT fallback; pinning the neural term keeps it isolated."""
    from anima_mcp.anima import _sense_stability

    def _grounded():
        r = _readings_gamma(0.2)
        r.eeg_theta_power = 0.5
        r.eeg_delta_power = 0.5
        return r

    with_key = _sense_stability(_grounded(), CAL)
    keyless = NervousSystemCalibration(
        stability_weights={k: v for k, v in CAL.stability_weights.items()
                           if k != "neural"})
    without_key = _sense_stability(_grounded(), keyless)
    assert with_key == pytest.approx(without_key, abs=1e-9)
