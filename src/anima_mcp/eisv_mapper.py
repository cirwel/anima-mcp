"""
Map anima state (physical + neural) to EISV metrics for UNITARES governance.

Creates bridge between anima-mcp creature and unitares-governance system.

This module implements the mapping from anima proprioception (warmth, clarity,
stability, presence) to EISV metrics (Energy, Integrity, Entropy, Valence) used
by UNITARES governance framework.

V is Valence — the signed E-I imbalance shared with governance (positive =
running hot, E>I; negative = running careful, I>E), NOT the older "Void"
(inverse-presence) reading. Reported as telemetry; the mapping is an
instantaneous readout (no accumulator/decay) so it does not damp.
"""

from dataclasses import dataclass
from typing import Optional
from .anima import Anima
from .sensors.base import SensorReadings


@dataclass
class EISVMetrics:
    """EISV metrics compatible with UNITARES governance."""
    
    energy: float      # E: 0-1, activation level
    integrity: float   # I: 0-1, information quality
    entropy: float     # S: 0-1, disorder/chaos
    valence: float     # V: -1..1, signed E-I imbalance (+hot / -careful)

    def to_dict(self) -> dict:
        """Convert to dictionary for MCP/JSON serialization."""
        return {
            "E": self.energy,
            "I": self.integrity,
            "S": self.entropy,
            "V": self.valence,
        }

    def __repr__(self) -> str:
        return f"EISV(E={self.energy:.2f}, I={self.integrity:.2f}, S={self.entropy:.2f}, V={self.valence:+.2f})"


def anima_to_eisv(
    anima: Anima,
    readings: SensorReadings,
    neural_weight: float = 0.3,
    physical_weight: float = 0.7
) -> EISVMetrics:
    """
    Map anima state to EISV metrics.
    
    Mapping strategy:
    - Energy (E): Warmth + Beta/Gamma power (activation)
    - Integrity (I): Clarity + Alpha power (awareness)
    - Entropy (S): Inverse of Stability (chaos)
    - Valence (V): Signed E-I imbalance (+running hot / -running careful)
    
    Args:
        anima: Anima state (warmth, clarity, stability, presence)
        readings: Sensor readings (physical + neural)
        neural_weight: Weight for neural signals (0-1)
        physical_weight: Weight for physical signals (0-1)
                      Should sum to 1.0 with neural_weight
    
    Returns:
        EISVMetrics with values in [0, 1] range
    """
    # Normalize weights
    total_weight = neural_weight + physical_weight
    if total_weight > 0:
        nw = neural_weight / total_weight
        pw = physical_weight / total_weight
    else:
        nw = 0.0
        pw = 1.0

    # Check if readings have neural signals
    has_neural = (
        getattr(readings, 'eeg_beta_power', None) is not None
        or getattr(readings, 'eeg_alpha_power', None) is not None
    )

    # Energy (E): Warmth + Beta/Gamma power (activation)
    E = anima.warmth
    if has_neural:
        beta = getattr(readings, 'eeg_beta_power', None) or 0
        gamma = getattr(readings, 'eeg_gamma_power', None) or 0
        neural_energy = beta * 0.6 + gamma * 0.4
        E = pw * anima.warmth + nw * neural_energy

    # Integrity (I): Clarity + Alpha power (awareness)
    integrity = anima.clarity
    if has_neural and getattr(readings, 'eeg_alpha_power', None) is not None:
        integrity = pw * anima.clarity + nw * readings.eeg_alpha_power
    
    # Entropy (S): Inverse of Stability (high stability = low entropy)
    # Stability incorporates Theta/Delta (deep stability)
    S = 1.0 - anima.stability
    
    # Clamp E/I/S to [0, 1]
    E = max(0.0, min(1.0, E))
    integrity = max(0.0, min(1.0, integrity))
    S = max(0.0, min(1.0, S))

    # Valence (V): signed E-I imbalance, matching governance Valence semantics
    # (positive = running hot, E>I; negative = running careful, I>E). Governance V
    # is a differential accumulator dV/dt = κ(E-I) - δV; this is its instantaneous
    # readout — no accumulator/decay, so the telemetry does not damp. The older
    # (1-presence) "Void" mapping only ever reported the positive half and was not
    # comparable to other agents' V.
    V = max(-1.0, min(1.0, E - integrity))

    return EISVMetrics(
        energy=E,
        integrity=integrity,
        entropy=S,
        valence=V,
    )


def estimate_complexity(
    anima: Anima,
    readings: Optional[SensorReadings] = None
) -> float:
    """
    Estimate task complexity from anima state and system load.

    Complexity increases with:
    - Low clarity (uncertainty)
    - Low stability (chaos)
    - High CPU/memory load (system strain)
    - High neural beta/gamma power (active processing)

    Args:
        anima: Anima state
        readings: Optional sensor readings (for system load + neural complexity)

    Returns:
        Complexity estimate in [0, 1] range
    """
    # Base complexity from anima state
    # Clarity: uncertainty increases complexity. Stability: entropy (= 1-stability) increases complexity.
    clarity_complexity = (1.0 - anima.clarity) * 0.25
    stability_complexity = (1.0 - anima.stability) * 0.35  # entropy = inverse stability

    complexity = clarity_complexity + stability_complexity

    # System load adds up to +0.15 on top (can push past 1.0, clamped below)
    if readings is not None:
        cpu = getattr(readings, 'cpu_percent', None)
        mem = getattr(readings, 'memory_percent', None)
        if cpu is not None:
            complexity += (cpu / 100.0) * 0.10
        if mem is not None:
            complexity += (mem / 100.0) * 0.05

    return max(0.0, min(1.0, complexity))


def generate_status_text(
    anima: Anima,
    readings: Optional[SensorReadings] = None,
    eisv: Optional[EISVMetrics] = None,
    experiential_summary: Optional[dict] = None,
) -> str:
    """
    Generate human-readable status text for governance system.

    Args:
        anima: Anima state
        readings: Optional sensor readings
        eisv: Optional EISV metrics (will compute if not provided)
        experiential_summary: Optional dict with marks/filter/pathway stats

    Returns:
        Status text string
    """
    feeling = anima.feeling()
    mood = feeling.get("mood", "neutral")

    # Build status text
    status_parts = [
        f"Anima state: {mood}",
        f"Warmth: {anima.warmth:.2f}",
        f"Clarity: {anima.clarity:.2f}",
        f"Stability: {anima.stability:.2f}",
        f"Presence: {anima.presence:.2f}",
    ]

    # Add neural info if available
    if readings:
        alpha = getattr(readings, 'eeg_alpha_power', None)
        beta = getattr(readings, 'eeg_beta_power', None)
        gamma = getattr(readings, 'eeg_gamma_power', None)
        if any(v is not None for v in [alpha, beta, gamma]):
            neural_parts = ["Neural:"]
            if alpha is not None:
                neural_parts.append(f"Alpha={alpha:.2f}")
            if beta is not None:
                neural_parts.append(f"Beta={beta:.2f}")
            if gamma is not None:
                neural_parts.append(f"Gamma={gamma:.2f}")
            status_parts.append(" ".join(neural_parts))

    # Add EISV if provided
    if eisv:
        status_parts.append(f"EISV: E={eisv.energy:.2f}, I={eisv.integrity:.2f}, S={eisv.entropy:.2f}, V={eisv.valence:+.2f}")

    # Add experiential accumulation summary
    if experiential_summary:
        exp_parts = []
        marks = experiential_summary.get("marks", {})
        if marks.get("total_marks", 0) > 0:
            exp_parts.append(f"{marks['total_marks']} marks")
        filt = experiential_summary.get("filter", {})
        biased = filt.get("biased_count", 0)
        if biased > 0:
            exp_parts.append(f"{biased} attention biases")
        pw = experiential_summary.get("pathways", {})
        if pw.get("total_pathways", 0) > 0:
            exp_parts.append(f"{pw['total_pathways']} pathways (avg {pw.get('avg_strength', 0.5):.2f})")
        if exp_parts:
            status_parts.append("Experience: " + ", ".join(exp_parts))

    return ". ".join(status_parts) + "."


# Convenience function for common use case
def compute_eisv_from_readings(
    readings: SensorReadings,
    neural_weight: float = 0.3,
    physical_weight: float = 0.7
) -> EISVMetrics:
    """
    Compute EISV metrics directly from sensor readings.
    
    Convenience function that creates anima state and maps to EISV in one call.
    
    Args:
        readings: Sensor readings (physical + neural)
        neural_weight: Weight for neural signals
        physical_weight: Weight for physical signals
    
    Returns:
        EISVMetrics
    """
    from .anima import sense_self
    
    anima = sense_self(readings)
    return anima_to_eisv(anima, readings, neural_weight, physical_weight)


def compute_ethical_drift(
    current_anima: Anima,
    prev_anima: Optional[Anima],
    current_readings: Optional[SensorReadings] = None,
    prev_readings: Optional[SensorReadings] = None,
) -> list:
    """
    Compute ethical drift (Δη) from changes in anima state between check-ins.

    Maps real proprioceptive changes to the 3-dimensional ethical drift vector
    that drives UNITARES governance dynamics:

    - Δη[0]: Emotional drift — change in warmth (engagement/withdrawal)
    - Δη[1]: Epistemic drift — change in clarity (certainty/confusion)
    - Δη[2]: Behavioral drift — change in stability (order/chaos)

    Without this signal, governance EISV stays at equilibrium and never moves.
    This is the primary driver of governance dynamics.

    Args:
        current_anima: Current anima state
        prev_anima: Previous anima state (None on first check-in)
        current_readings: Current sensor readings (optional, for environmental context)
        prev_readings: Previous sensor readings (optional)

    Returns:
        3-element list [Δη₀, Δη₁, Δη₂] representing ethical drift
    """
    if prev_anima is None:
        return [0.0, 0.0, 0.0]

    # Raw deltas (positive = increasing, negative = decreasing)
    d_warmth = current_anima.warmth - prev_anima.warmth
    d_clarity = current_anima.clarity - prev_anima.clarity
    d_stability = current_anima.stability - prev_anima.stability

    # Scale factors — anima changes are typically small (0.01-0.05 per interval).
    # UNITARES dynamics expects drift in roughly [-0.3, 0.3] to produce visible effects.
    # Scale by 3x to make real sensor changes produce meaningful governance response.
    scale = 3.0

    # Environmental amplification: large temperature/light changes increase drift signal
    env_amplifier = 1.0
    if current_readings and prev_readings:
        # Temperature change amplifies behavioral drift
        curr_temp = getattr(current_readings, 'ambient_temp_c', None) or getattr(current_readings, 'cpu_temp_c', None)
        prev_temp = getattr(prev_readings, 'ambient_temp_c', None) or getattr(prev_readings, 'cpu_temp_c', None)
        if curr_temp is not None and prev_temp is not None:
            temp_change = abs(curr_temp - prev_temp)
            if temp_change > 2.0:  # >2°C change is significant
                env_amplifier = 1.0 + min(temp_change / 10.0, 1.0)  # Up to 2x

        # Light change amplifies emotional drift (raw lux — includes LED glow)
        curr_light = getattr(current_readings, 'light_lux', None)
        prev_light = getattr(prev_readings, 'light_lux', None)
        if curr_light is not None and prev_light is not None:
            if prev_light > 1.0:  # Need meaningful baseline to compute ratio
                light_ratio = abs(curr_light - prev_light) / prev_light
                if light_ratio > 0.3:  # >30% change in light
                    env_amplifier = max(env_amplifier, 1.0 + min(light_ratio, 1.0))

    drift = [
        d_warmth * scale * env_amplifier,    # Emotional drift
        d_clarity * scale,                     # Epistemic drift (less env-dependent)
        d_stability * scale * env_amplifier,   # Behavioral drift
    ]

    # Clamp to reasonable range [-0.5, 0.5] — prevent extreme signals
    drift = [max(-0.5, min(0.5, d)) for d in drift]

    return drift


def compute_confidence(
    anima: Anima,
    readings: Optional[SensorReadings] = None,
    prev_anima: Optional[Anima] = None,
) -> float:
    """
    Compute agent confidence from anima state and stability.

    Higher confidence when:
    - Clarity is high (knows what it's seeing)
    - Stability is high (consistent over time)
    - Not in rapid transition (small delta from previous)

    Args:
        anima: Current anima state
        readings: Optional sensor readings
        prev_anima: Optional previous state (for transition detection)

    Returns:
        Confidence in [0.0, 1.0]
    """
    # Base: clarity is the primary confidence signal
    confidence = anima.clarity * 0.5 + anima.stability * 0.3 + anima.presence * 0.2

    # Penalize rapid transitions (low confidence when changing fast)
    if prev_anima is not None:
        total_delta = (
            abs(anima.warmth - prev_anima.warmth) +
            abs(anima.clarity - prev_anima.clarity) +
            abs(anima.stability - prev_anima.stability)
        )
        # If total change > 0.15, reduce confidence proportionally
        if total_delta > 0.15:
            transition_penalty = min(total_delta - 0.15, 0.3)  # Max 0.3 penalty
            confidence -= transition_penalty

    return max(0.05, min(1.0, confidence))
