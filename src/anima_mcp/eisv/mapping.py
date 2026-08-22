"""EISV mapping, derivative computation, and trajectory shape classification.

Ported from eisv-lumen (https://github.com/CIRWEL/eisv-lumen) for use in
Lumen's live system. Pure Python, no heavy dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from ..eisv_mapper import anima_components_to_eisv


# ---------------------------------------------------------------------------
# EISV Mapping
# ---------------------------------------------------------------------------

def anima_to_eisv(
    warmth: float, clarity: float, stability: float, presence: float,
) -> Dict[str, float]:
    """Map Anima state scalars to EISV coordinates.

    Without neural readings, E = warmth. I = clarity, S = 1-stability,
    and V = E-I (signed Valence). This is the same mapping used by governance.
    """
    return anima_components_to_eisv(
        warmth, clarity, stability, presence,
    ).to_dict()


# ---------------------------------------------------------------------------
# Derivative computation
# ---------------------------------------------------------------------------

EISV_DIMS = ["E", "I", "S", "V"]


def compute_derivatives(states: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Finite-difference first derivatives from EISV state snapshots.

    Input: list of dicts with keys 't', 'E', 'I', 'S', 'V'.
    Returns: list of dicts with keys 't', 'dE', 'dI', 'dS', 'dV'.
    """
    results: List[Dict[str, float]] = []
    prev = states[0]
    for i in range(1, len(states)):
        curr = states[i]
        dt = curr["t"] - prev["t"]
        if dt == 0.0:
            prev = curr
            continue
        entry: Dict[str, float] = {"t": curr["t"]}
        for dim in EISV_DIMS:
            entry[f"d{dim}"] = (curr[dim] - prev[dim]) / dt
        results.append(entry)
        prev = curr
    return results


def compute_second_derivatives(
    derivatives: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Second derivatives from a first-derivative series."""
    results: List[Dict[str, float]] = []
    for i in range(1, len(derivatives)):
        prev = derivatives[i - 1]
        curr = derivatives[i]
        dt = curr["t"] - prev["t"]
        if dt == 0.0:
            continue
        entry: Dict[str, float] = {"t": curr["t"]}
        for dim in EISV_DIMS:
            entry[f"d2{dim}"] = (curr[f"d{dim}"] - prev[f"d{dim}"]) / dt
        results.append(entry)
    return results


def compute_trajectory_window(
    states: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Build a complete trajectory window from state snapshots."""
    derivatives = compute_derivatives(states)
    second_derivatives = compute_second_derivatives(derivatives)
    return {
        "states": states,
        "derivatives": derivatives,
        "second_derivatives": second_derivatives,
    }


# ---------------------------------------------------------------------------
# Trajectory Shape Classifier
# ---------------------------------------------------------------------------

_HIGH_BASIN_E = 0.6
_DERIV_THRESHOLD = 0.05
_BASIN_JUMP = 0.2


class TrajectoryShape(str, Enum):
    SETTLED_PRESENCE = "settled_presence"
    RISING_ENTROPY = "rising_entropy"
    FALLING_ENERGY = "falling_energy"
    BASIN_TRANSITION_DOWN = "basin_transition_down"
    BASIN_TRANSITION_UP = "basin_transition_up"
    ENTROPY_SPIKE_RECOVERY = "entropy_spike_recovery"
    DRIFT_DISSONANCE = "drift_dissonance"
    VALENCE_RISING = "valence_rising"
    VOID_RISING = "valence_rising"  # Deprecated source-compatible alias
    CONVERGENCE = "convergence"


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def classify_trajectory(window: Dict[str, Any]) -> TrajectoryShape:
    """Classify a trajectory window into one of 9 dynamical shape classes."""
    states = window["states"]
    derivs = window["derivatives"]
    second = window["second_derivatives"]

    e_vals = [s["E"] for s in states]
    s_vals = [s["S"] for s in states]

    e_range = max(e_vals) - min(e_vals)
    s_range = max(s_vals) - min(s_vals)

    de_vals = [d["dE"] for d in derivs] if derivs else [0.0]
    ds_vals = [d["dS"] for d in derivs] if derivs else [0.0]
    dv_vals = [d["dV"] for d in derivs] if derivs else [0.0]

    mean_de = _mean(de_vals)
    mean_ds = _mean(ds_vals)
    mean_dv = _mean(dv_vals)

    # 1. Basin transition down
    if e_range >= _BASIN_JUMP and mean_de < 0 and e_vals[0] > e_vals[-1]:
        return TrajectoryShape.BASIN_TRANSITION_DOWN

    # 2. Basin transition up
    if e_range >= _BASIN_JUMP and mean_de > 0 and e_vals[-1] > e_vals[0]:
        return TrajectoryShape.BASIN_TRANSITION_UP

    # 3. Entropy spike recovery
    if s_range >= _BASIN_JUMP:
        max_s_idx = s_vals.index(max(s_vals))
        if 0 < max_s_idx < len(s_vals) - 1:
            return TrajectoryShape.ENTROPY_SPIKE_RECOVERY

    # 4. Drift dissonance
    drift_vals = [s.get("ethical_drift", 0.0) for s in states]
    if max(drift_vals) > 0.3:
        return TrajectoryShape.DRIFT_DISSONANCE

    # 5. Valence rising (Energy gaining on Integrity)
    if mean_dv > _DERIV_THRESHOLD:
        return TrajectoryShape.VALENCE_RISING

    # 6. Rising entropy
    if mean_ds > _DERIV_THRESHOLD:
        return TrajectoryShape.RISING_ENTROPY

    # 7. Falling energy
    if mean_de < -_DERIV_THRESHOLD:
        return TrajectoryShape.FALLING_ENERGY

    # 8. Convergence
    if derivs and second:
        all_derivs_small = all(
            abs(d[k]) < _DERIV_THRESHOLD
            for d in derivs
            for k in ("dE", "dI", "dS", "dV")
        )
        all_second_small = all(
            abs(d[k]) < _DERIV_THRESHOLD
            for d in second
            for k in ("d2E", "d2I", "d2S", "d2V")
        )
        has_dynamics = any(
            abs(d[k]) > 1e-9
            for d in derivs
            for k in ("dE", "dI", "dS", "dV")
        )
        if all_derivs_small and all_second_small and has_dynamics:
            return TrajectoryShape.CONVERGENCE

    # 9. Default
    return TrajectoryShape.SETTLED_PRESENCE
