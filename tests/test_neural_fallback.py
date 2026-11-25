"""Neural-band fallbacks in the sense functions (#175).

``get_computational_neural_state`` is a stateful singleton: every call appends
to its CPU/temperature history and advances its EMA and dt clock. The legacy
neural terms in warmth, clarity and presence carry weight 0.0 by default
(de-aliased from CPU%, #173/#175), yet each still called it whenever the
readings lacked the bands, so a term that contributed nothing could move the
next real delta/theta/gamma and could raise out of ``sense_self``. Stability's
neural term (weight 0.1) re-derived missing theta/delta from the same
singleton, which in the broker is the source whose failure left them None.

On the live path (bands always present; the residual tracker reports
valid_fraction 1.0 for theta/delta over 200k ticks) none of this runs, so no
live value moves. These tests pin the degraded path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anima_mcp.anima as anima_mod
from anima_mcp.anima import (
    _sense_clarity,
    _sense_presence,
    _sense_stability,
    _sense_warmth,
)
from anima_mcp.computational_neural import ComputationalNeuralState
from anima_mcp.config import NervousSystemCalibration
from anima_mcp.sensors.base import SensorReadings


CAL = NervousSystemCalibration()


def _physical(**bands) -> SensorReadings:
    return SensorReadings(
        timestamp=1_000_000.0,
        cpu_temp_c=55.0,
        ambient_temp_c=26.0,
        humidity_pct=30.0,
        light_lux=100.0,
        pressure_hpa=820.0,
        cpu_percent=10.0,
        memory_percent=20.0,
        disk_percent=60.0,
        **bands,
    )


@pytest.fixture
def singleton_forbidden(monkeypatch):
    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("computational neural source unavailable")

    monkeypatch.setattr(anima_mod, "get_computational_neural_state", _boom)
    return calls


def _clarity(readings, cal=CAL, attribution=None):
    return _sense_clarity(
        readings,
        cal,
        0.8,
        frozen_channel_count=0,
        external_light_lux=100.0,
        attribution=attribution,
    )


def test_zero_weight_terms_do_not_sample_the_singleton(singleton_forbidden):
    """No bands, default weights: warmth/clarity/presence must not call the
    singleton (it would raise here) and must equal the banded values, since
    their neural terms carry weight 0.0."""
    bare = _physical()
    banded = _physical(
        eeg_alpha_power=0.9, eeg_beta_power=0.1, eeg_gamma_power=0.7,
        eeg_theta_power=0.02, eeg_delta_power=0.9,
    )
    assert _sense_warmth(bare, CAL) == _sense_warmth(banded, CAL)
    assert _clarity(bare) == _clarity(banded)
    assert _sense_presence(bare, CAL) == _sense_presence(banded, CAL)
    assert singleton_forbidden == []


def test_clarity_attribution_still_names_the_unsampled_alpha_term(
    singleton_forbidden,
):
    attribution: dict = {}
    value = _clarity(_physical(), attribution=attribution)
    alpha = attribution["components"]["computational_alpha"]
    assert alpha["available"] is False
    assert alpha["source"] == "zero_weight_not_sampled"
    assert alpha["contribution"] == 0.0
    total = sum(c["contribution"] for c in attribution["components"].values())
    assert total == pytest.approx(value, abs=0.002)


def test_stability_reads_absent_theta_delta_as_full_uncertainty(
    singleton_forbidden,
):
    """Absent theta/delta are unknown, not re-derived: the neural term reads
    at full instability (groundedness 0), the missing-sensors convention."""
    absent = _sense_stability(_physical(), CAL, frozen_channel_count=0)
    zero_grounded = _sense_stability(
        _physical(eeg_theta_power=0.0, eeg_delta_power=0.0), CAL, frozen_channel_count=0
    )
    live_like = _sense_stability(
        _physical(eeg_theta_power=0.02, eeg_delta_power=0.89), CAL, frozen_channel_count=0
    )
    assert singleton_forbidden == []
    assert absent == zero_grounded
    # Fails toward unknown, never toward healthy (CLAUDE.md invariant 2).
    assert absent < live_like


def test_stability_banded_path_is_unchanged():
    """The live formula: 1 - weighted mean instability, neural term
    (1 - (theta+delta)/2) at weight 0.1. Hand-computed for one reading."""
    r = _physical(eeg_theta_power=0.02, eeg_delta_power=0.89)
    w = CAL.stability_weights
    humidity = min(1, abs(30.0 - CAL.humidity_ideal) / max(1, CAL.humidity_ideal))
    pressure = min(1, abs(820.0 - CAL.pressure_ideal) / 20.0)
    instability = (
        humidity * w["humidity_dev"]
        + 0.20 * w["memory"]
        + 0.0 * w["missing_sensors"]
        + pressure * w["pressure_dev"]
        + (1 - (0.02 + 0.89) / 2) * w["neural"]
    )
    expected = round(1 - instability / sum(w.values()), 3)
    assert _sense_stability(r, CAL, frozen_channel_count=0) == expected


def test_nonzero_override_keeps_the_fallback(monkeypatch):
    """A calibration that deliberately re-weights the neural presence term
    still gets the old fallback; only zero-weight terms stop sampling."""
    stub = ComputationalNeuralState(
        delta=0.9, theta=0.0, alpha=0.9, beta=0.1, gamma=1.0
    )
    calls = []

    def _stub(*args, **kwargs):
        calls.append(kwargs)
        return stub

    monkeypatch.setattr(anima_mod, "get_computational_neural_state", _stub)
    weighted = NervousSystemCalibration(
        presence_weights={"disk": 0.3, "memory": 0.3, "cpu": 0.2, "neural": 0.2}
    )
    with_fallback = _sense_presence(_physical(), weighted)
    banded = _sense_presence(_physical(eeg_gamma_power=1.0), weighted)
    assert calls, "non-zero neural weight should still consult the fallback"
    assert with_fallback == banded
