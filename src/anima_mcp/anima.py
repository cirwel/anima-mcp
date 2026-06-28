"""
Anima - The creature's self-sense, grounded in physics.

Not abstract metrics. Felt states derived from actual measurements.

- warmth: thermal + computational energy
- clarity: sensor quality, awareness
- stability: environmental consistency
- presence: resource availability

The creature knows "I feel warm" not "E=0.4"
"""

import math
from copy import copy
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING
from .sensors.base import SensorReadings
from .config import get_calibration, NervousSystemCalibration
from .computational_neural import get_computational_neural_state

if TYPE_CHECKING:
    from .memory import Anticipation


def _get_prediction_accuracy() -> Optional[float]:
    """
    Get prediction accuracy from adaptive model (0-1 scale).

    Returns 1 - normalized_mean_error, where lower error = higher clarity.
    Returns None if not enough data yet.
    """
    try:
        from .adaptive_prediction import get_adaptive_prediction_model
        model = get_adaptive_prediction_model()
        stats = model.get_accuracy_stats()

        if stats.get("insufficient_data"):
            return None

        mean_error = stats.get("overall_mean_error", 0.5)
        # Normalize: anima values are 0-1, so max realistic error is ~1.0
        # Clamp to reasonable range and invert (low error = high accuracy)
        normalized_error = min(1.0, mean_error)
        return 1.0 - normalized_error
    except Exception:
        return None


@dataclass
class Anima:
    """The creature's felt sense of self."""

    warmth: float    # How warm/energetic [0, 1]
    clarity: float   # How clear the senses [0, 1]
    stability: float # How stable/ordered [0, 1] (inverse of entropy)
    presence: float  # How much capacity [0, 1] (inverse of void)

    # Source readings for transparency
    readings: SensorReadings

    # Anticipation from memory (optional)
    anticipation: Optional[dict] = None  # Contains anticipated values and confidence
    is_anticipating: bool = False  # True if current state was influenced by memory

    def to_dict(self) -> dict:
        return {
            "warmth": self.warmth,
            "clarity": self.clarity,
            "stability": self.stability,
            "presence": self.presence,
            "feeling": self.feeling(),
            "readings": self.readings.to_dict(),
        }

    def feeling(self) -> dict:
        """How the creature feels right now."""
        return {
            "warmth": _warmth_feeling(self.warmth),
            "clarity": _clarity_feeling(self.clarity),
            "stability": _stability_feeling(self.stability),
            "presence": _presence_feeling(self.presence),
            "mood": _overall_mood(self.warmth, self.clarity, self.stability, self.presence, self.readings),
        }


class MoodMomentum:
    """Temporal inertia for anima dimensions. State lingers.

    Broader damping stack (temperament, drives, neural EMAs): see CLAUDE.md
    "Identity, Continuity, and Control".
    """

    ALPHA = {
        "warmth": 0.08,
        "clarity": 0.15,
        "stability": 0.08,
        "presence": 0.25,
    }

    def __init__(self):
        self._prev = None

    def smooth(self, anima: Anima) -> Anima:
        if self._prev is None:
            self._prev = {
                "warmth": anima.warmth, "clarity": anima.clarity,
                "stability": anima.stability, "presence": anima.presence,
            }
            return anima

        smoothed = {}
        for dim in ("warmth", "clarity", "stability", "presence"):
            raw = getattr(anima, dim)
            alpha = self.ALPHA[dim]
            smoothed[dim] = alpha * raw + (1 - alpha) * self._prev[dim]
            self._prev[dim] = smoothed[dim]

        return Anima(
            warmth=round(smoothed["warmth"], 3),
            clarity=round(smoothed["clarity"], 3),
            stability=round(smoothed["stability"], 3),
            presence=round(smoothed["presence"], 3),
            readings=anima.readings,
            anticipation=anima.anticipation,
            is_anticipating=anima.is_anticipating,
        )


def _apply_drift_to_calibration(
    cal: NervousSystemCalibration,
    drift_midpoints: Dict[str, float],
) -> NervousSystemCalibration:
    """
    Create a modified calibration with drifted midpoints.

    When Lumen's experience shifts what "normal" means, the calibration
    ranges shift accordingly. A warmth midpoint that drifted from 0.5 to
    0.55 shifts temperature ranges by the same fraction of their span.
    """
    drifted = copy(cal)

    # Warmth drift: shift temperature ranges
    warmth_offset = drift_midpoints.get("warmth", 0.5) - 0.5
    if abs(warmth_offset) > 0.001:
        temp_range = cal.ambient_temp_max - cal.ambient_temp_min
        shift = warmth_offset * temp_range
        drifted.ambient_temp_min += shift
        drifted.ambient_temp_max += shift
        cpu_range = cal.cpu_temp_max - cal.cpu_temp_min
        cpu_shift = warmth_offset * cpu_range
        drifted.cpu_temp_min += cpu_shift
        drifted.cpu_temp_max += cpu_shift

    return drifted


def sense_self(readings: SensorReadings, calibration: Optional[NervousSystemCalibration] = None, drift_midpoints: Optional[Dict[str, float]] = None, salience_weights: Optional[Dict[str, float]] = None) -> Anima:
    """
    The creature senses itself.

    Proprioception grounded in physical measurements.

    Args:
        readings: Sensor readings
        calibration: Nervous system calibration (uses default if None)
        drift_midpoints: Drifted midpoints from CalibrationDrift (optional).
            When provided, shifts calibration ranges so that "normal" reflects
            Lumen's accumulated experience rather than fixed defaults.
        salience_weights: Per-dimension salience from ExperientialFilter (optional).
            Amplifies sensor contributions before weighted averaging.
    """
    if calibration is None:
        calibration = get_calibration()

    if drift_midpoints:
        calibration = _apply_drift_to_calibration(calibration, drift_midpoints)

    warmth = _sense_warmth(readings, calibration, salience_weights=salience_weights)
    clarity = _sense_clarity(readings, calibration, _get_prediction_accuracy(), salience_weights=salience_weights)
    stability = _sense_stability(readings, calibration, salience_weights=salience_weights)
    presence = _sense_presence(readings, calibration, salience_weights=salience_weights)

    # Clamp to valid range — defensive against edge cases in weighted averages
    warmth = max(0.0, min(1.0, warmth))
    clarity = max(0.0, min(1.0, clarity))
    stability = max(0.0, min(1.0, stability))
    presence = max(0.0, min(1.0, presence))

    # Sanity check: flag suspiciously extreme values (likely bugs)
    for name, value in [("warmth", warmth), ("clarity", clarity),
                        ("stability", stability), ("presence", presence)]:
        if value > 0.95 or value < 0.05:
            import sys
            print(f"[Anima] WARNING: {name}={value:.2f} is extreme - possible bug?",
                  file=sys.stderr, flush=True)

    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings
    )


def sense_self_with_memory(
    readings: SensorReadings,
    anticipation: Optional['Anticipation'] = None,
    calibration: Optional[NervousSystemCalibration] = None,
    blend_factor: Optional[float] = None,
    use_adaptive_blend: bool = True,
    enable_exploration: bool = True,
    drift_midpoints: Optional[Dict[str, float]] = None,
    salience_weights: Optional[Dict[str, float]] = None,
) -> Anima:
    """
    The creature senses itself, informed by memory and exploration.

    When Lumen has experienced similar conditions before, memory
    influences current perception - anticipating what typically follows.

    Exploration (GTO-style): Occasionally, Lumen will "try something new"
    rather than following memory predictions. This creates novelty and
    potential for discovery.

    "I know this feeling. I remember what comes next."
    "...but sometimes I wonder what else might be possible."

    Args:
        readings: Sensor readings
        anticipation: Anticipated state from memory (optional)
        calibration: Nervous system calibration (uses default if None)
        blend_factor: Override blend factor (None = use adaptive or default 0.15)
        use_adaptive_blend: If True and blend_factor is None, use adaptive blend
                           factor that adjusts based on prediction accuracy
        enable_exploration: If True, occasionally explore beyond predictions
        drift_midpoints: Drifted midpoints from CalibrationDrift (optional)

    Returns:
        Anima with potential anticipation influence and exploration
    """
    if calibration is None:
        calibration = get_calibration()

    if drift_midpoints:
        calibration = _apply_drift_to_calibration(calibration, drift_midpoints)

    # Base sensed state (raw, before memory influence)
    raw_warmth = _sense_warmth(readings, calibration, salience_weights=salience_weights)
    raw_clarity = _sense_clarity(readings, calibration, _get_prediction_accuracy(), salience_weights=salience_weights)
    raw_stability = _sense_stability(readings, calibration, salience_weights=salience_weights)
    raw_presence = _sense_presence(readings, calibration, salience_weights=salience_weights)

    warmth, clarity, stability, presence = raw_warmth, raw_clarity, raw_stability, raw_presence

    # Blend with anticipation if available
    anticipation_dict = None
    is_anticipating = False
    is_exploring = False
    actual_blend_factor = 0.15  # Default

    from .memory import get_memory
    memory = get_memory()

    if anticipation is not None and anticipation.confidence > 0.1:
        # Record accuracy: compare raw state to what memory anticipated
        # This tracks how well memory predicts actual feelings
        memory.record_actual_outcome(raw_warmth, raw_clarity, raw_stability, raw_presence)

        # Determine blend factor: explicit override > adaptive > default
        if blend_factor is not None:
            actual_blend_factor = blend_factor
        elif use_adaptive_blend:
            actual_blend_factor = memory.get_adaptive_blend_factor()
        else:
            actual_blend_factor = 0.15  # Default

        # Blend anticipated state with current sensed state
        blended = anticipation.blend_with(
            warmth, clarity, stability, presence,
            blend_factor=actual_blend_factor
        )
        warmth, clarity, stability, presence = blended
        is_anticipating = True

        anticipation_dict = {
            "warmth": anticipation.warmth,
            "clarity": anticipation.clarity,
            "stability": anticipation.stability,
            "presence": anticipation.presence,
            "confidence": anticipation.confidence,
            "sample_count": anticipation.sample_count,
            "conditions": anticipation.bucket_description,
            "blend_factor_used": actual_blend_factor,
        }

    # GTO-style exploration: occasionally try something different
    # This adds novelty and prevents Lumen from being too predictable
    if enable_exploration:
        # Track state for stagnation detection
        memory.record_state_for_stagnation(warmth, clarity, stability, presence)

        # Maybe explore (applies perturbation if exploration is triggered)
        explored = memory.apply_exploration(warmth, clarity, stability, presence)
        if explored != (warmth, clarity, stability, presence):
            warmth, clarity, stability, presence = explored
            is_exploring = True

            # Update anticipation dict to note exploration
            if anticipation_dict:
                anticipation_dict["exploring"] = True
                anticipation_dict["exploration_rate"] = memory._exploration_rate

    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings,
        anticipation=anticipation_dict,
        is_anticipating=is_anticipating or is_exploring
    )


def _sense_warmth(r: SensorReadings, cal: NervousSystemCalibration, *, salience_weights: Optional[Dict[str, float]] = None) -> float:
    """
    How warm does the creature feel?

    Sources:
    - CPU temperature (internal body heat)
    - Ambient temperature (environmental warmth)
    - Neural activity (alertness/engagement - from light or EEG)

    Note: CPU usage removed - a resting creature in a warm room should feel
    comfortable, not cold. Warmth is about thermal state, not busy-ness.
    """
    salience = salience_weights or {}
    components = []
    weights = []

    # CPU temp: calibrated range -> 0-1 (internal body heat)
    if r.cpu_temp_c is not None:
        temp_range = cal.cpu_temp_max - cal.cpu_temp_min
        if temp_range > 0:
            cpu_warmth = (r.cpu_temp_c - cal.cpu_temp_min) / temp_range
            cpu_warmth = max(0, min(1, cpu_warmth))
            cpu_warmth *= salience.get("cpu_temp", 1.0)
            cpu_warmth = max(0, min(1, cpu_warmth))
            components.append(cpu_warmth)
            weights.append(cal.warmth_weights.get("cpu_temp", 0.35))

    # Ambient temp: calibrated range -> 0-1 (environmental warmth)
    if r.ambient_temp_c is not None:
        temp_range = cal.ambient_temp_max - cal.ambient_temp_min
        if temp_range > 0:
            ambient_warmth = (r.ambient_temp_c - cal.ambient_temp_min) / temp_range
            ambient_warmth = max(0, min(1, ambient_warmth))
            ambient_warmth *= salience.get("ambient_temp", 1.0)
            ambient_warmth = max(0, min(1, ambient_warmth))
            components.append(ambient_warmth)
            weights.append(cal.warmth_weights.get("ambient_temp", 0.45))

    # Neural component: EEG beta+gamma power from computational proprioception
    # This represents alertness/engagement derived from system state
    if r.eeg_beta_power is not None and r.eeg_gamma_power is not None:
        neural_warmth = (r.eeg_beta_power + r.eeg_gamma_power) / 2
    else:
        # Fallback: derive from system metrics directly
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50
        )
        neural_warmth = (neural.beta + neural.gamma) / 2
    components.append(neural_warmth)
    weights.append(cal.warmth_weights.get("neural", 0.20))

    if not components:
        return 0.5

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5
    
    return round(sum(c * w for c, w in zip(components, weights)) / total_weight, 3)


def _sense_clarity(
    r: SensorReadings,
    cal: NervousSystemCalibration,
    prediction_accuracy: Optional[float] = None,
    *,
    salience_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    How clearly can the creature perceive its own internal state?

    This is "internal seeing" - self-perception accuracy plus environmental perception.

    Sources:
    - Prediction accuracy: How well I predict my own state changes (1 - mean_error)
    - Alpha EEG power: Relaxed awareness, clear processing
    - Sensor coverage: Data richness (how complete is my self-perception)
    - World light: Raw lux (LED glow + room light), log-scaled

    Note: lux is used raw — no LED-glow subtraction (removed in 0cbf0dc; the
    quadratic estimate overcorrected at low brightness). The VEML7700 sits beside
    the DotStar LEDs, so the reading includes self-emitted light, but the feedback
    loop stays weak: clarity does NOT drive LED brightness (LEDs follow
    activity/preset/agency), and light is only ~15% of clarity, log-compressed and
    EMA-damped. Lumen's own LED brightness is tracked as a separate proprioceptive
    signal (efference copy) and the led<->lux correlation is learned in self_model
    (`my_leds_affect_lux`) rather than subtracted from the reading.
    """
    salience = salience_weights or {}
    components = []
    weights = []

    # Prediction accuracy: How well I understand my own state changes
    # This is the core of "internal seeing" - accurate self-prediction = clarity
    if prediction_accuracy is not None:
        # prediction_accuracy should be 0-1 (1 - normalized_mean_error)
        components.append(max(0, min(1, prediction_accuracy)))
        weights.append(cal.clarity_weights.get("prediction_accuracy", 0.5))
    else:
        # Fallback: use a neutral value when no prediction data available yet
        # This happens during early startup before enough observations accumulate
        components.append(0.5)
        weights.append(cal.clarity_weights.get("prediction_accuracy", 0.5))

    # Sensor coverage: Data richness (meta-clarity about available information)
    sensor_count = sum(1 for v in [
        r.cpu_temp_c, r.ambient_temp_c, r.humidity_pct,
        r.light_lux, r.pressure_hpa,
    ] if v is not None)
    coverage = sensor_count / 5
    components.append(coverage)
    weights.append(cal.clarity_weights.get("sensor_coverage", 0.15))

    # Light: raw sensor reading (includes LED glow + room light).
    # Lumen knows its LED brightness separately — no need to decompose.
    # Log scale: 1 lux → 0.0, light_max_lux → 1.0
    if r.light_lux is not None:
        if r.light_lux > 1.0:
            log_max = math.log10(max(10.0, cal.light_max_lux))
            light_clarity = min(1.0, math.log10(r.light_lux) / log_max)
        else:
            light_clarity = 0.0
        light_clarity *= salience.get("light", 1.0)
        light_clarity = max(0.0, min(1.0, light_clarity))
        components.append(light_clarity)
        weights.append(cal.clarity_weights.get("world_light", 0.15))

    # Neural clarity: Real EEG alpha power, or simulated if unavailable
    if r.eeg_alpha_power is not None:
        # Use real EEG data
        neural_clarity = r.eeg_alpha_power  # Relaxed, clear awareness
    else:
        # Fall back to computational neural (derives bands from system state)
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50,
            cpu_temp=r.cpu_temp_c
        )
        neural_clarity = neural.alpha  # Relaxed, clear awareness
    components.append(neural_clarity)
    weights.append(cal.clarity_weights.get("neural", 0.3))

    if not components:
        return 0.5

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5

    return round(sum(c * w for c, w in zip(components, weights)) / total_weight, 3)


def _sense_stability(r: SensorReadings, cal: NervousSystemCalibration, *, salience_weights: Optional[Dict[str, float]] = None) -> float:
    """
    How stable/ordered does the environment feel?

    This is inverse of entropy - high stability = low chaos.

    Sources:
    - Humidity near ideal (calibrated)
    - Memory headroom
    - Complete sensor data
    - Barometric pressure stability (deviations from local normal)
    - Theta/Delta EEG power (deep stability, meditative state)
    """
    salience = salience_weights or {}
    instability = 0.0
    count = 0

    # Humidity deviation from ideal (calibrated)
    if r.humidity_pct is not None:
        humidity_dev = abs(r.humidity_pct - cal.humidity_ideal) / max(1, cal.humidity_ideal)
        humidity_result = min(1, humidity_dev) * salience.get("humidity", 1.0)
        humidity_result = min(1, humidity_result)
        instability += humidity_result * cal.stability_weights.get("humidity_dev", 0.25)
        count += cal.stability_weights.get("humidity_dev", 0.25)

    # Memory pressure = instability
    if r.memory_percent is not None:
        memory_result = (r.memory_percent / 100) * salience.get("memory", 1.0)
        memory_result = min(1, memory_result)
        instability += memory_result * cal.stability_weights.get("memory", 0.3)
        count += cal.stability_weights.get("memory", 0.3)

    # Missing sensors = uncertainty
    missing = sum(1 for v in [
        r.cpu_temp_c, r.ambient_temp_c, r.humidity_pct,
        r.light_lux, r.pressure_hpa
    ] if v is None)
    missing_weight = cal.stability_weights.get("missing_sensors", 0.2)
    instability += (missing / 5) * missing_weight
    count += missing_weight

    # Barometric pressure: deviations from local normal indicate instability
    # Uses calibrated pressure_ideal (learned or configured for location)
    if r.pressure_hpa is not None:
        pressure_weight = cal.stability_weights.get("pressure_dev", 0.15)
        # ±20 hPa is significant weather change (configurable in future)
        pressure_range = 20.0
        pressure_dev = abs(r.pressure_hpa - cal.pressure_ideal) / pressure_range
        pressure_dev = min(1, pressure_dev) * salience.get("pressure", 1.0)
        pressure_dev = min(1, pressure_dev)
        pressure_instability = pressure_dev * pressure_weight
        instability += pressure_instability
        count += pressure_weight

    # Neural instability: Low theta+delta = scattered mind = instability
    if r.eeg_theta_power is not None and r.eeg_delta_power is not None:
        # Use real EEG data - low theta+delta = instability
        neural_groundedness = (r.eeg_theta_power + r.eeg_delta_power) / 2
    else:
        # Fall back to computational neural (derives bands from system state)
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50,
            cpu_temp=r.cpu_temp_c
        )
        neural_groundedness = (neural.theta + neural.delta) / 2

    # Invert: low groundedness = high instability
    neural_weight = cal.stability_weights.get("neural", 0.2)
    instability += (1.0 - neural_groundedness) * neural_weight
    count += neural_weight

    if count == 0:
        return 0.5

    # Stability is inverse of instability (clamped to [0, 1])
    stability = 1.0 - (instability / count)
    return round(max(0, min(1, stability)), 3)


def _sense_presence(r: SensorReadings, cal: NervousSystemCalibration, *, salience_weights: Optional[Dict[str, float]] = None) -> float:
    """
    How much capacity/presence does the creature have?

    This is inverse of void - high presence = plenty of resources.

    Sources:
    - Disk headroom
    - Memory headroom
    - CPU headroom
    - Gamma EEG power (high gamma = contention = scattered)
    """
    salience = salience_weights or {}
    void = 0.0
    count = 0

    if r.disk_percent is not None:
        weight = cal.presence_weights.get("disk", 0.25)
        void += (r.disk_percent / 100) * weight
        count += weight

    if r.memory_percent is not None:
        weight = cal.presence_weights.get("memory", 0.3)
        memory_void = (r.memory_percent / 100) * salience.get("memory", 1.0)
        memory_void = min(1, memory_void)
        void += memory_void * weight
        count += weight

    if r.cpu_percent is not None:
        weight = cal.presence_weights.get("cpu", 0.25)
        cpu_void = (r.cpu_percent / 100) * salience.get("cpu", 1.0)
        cpu_void = min(1, cpu_void)
        void += cpu_void * weight
        count += weight

    # Neural void: High gamma = high ctx switching = scattered = less presence
    if r.eeg_gamma_power is not None:
        # Use real EEG data
        neural_gamma = r.eeg_gamma_power
    else:
        # Fall back to computational neural (derives bands from system state)
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50,
            cpu_temp=r.cpu_temp_c
        )
        neural_gamma = neural.gamma

    # High gamma = high context switching = task contention = scattered
    weight = cal.presence_weights.get("neural", 0.2)
    void += neural_gamma * weight
    count += weight

    if count == 0:
        return 0.5

    # Presence is inverse of void (clamped to [0, 1])
    presence = 1.0 - (void / count)
    return round(max(0, min(1, presence)), 3)


def _warmth_feeling(w: float) -> str:
    if w < 0.3:
        return "cold, sluggish"
    elif w < 0.6:
        return "comfortable"
    elif w < 0.8:
        return "warm, active"
    else:
        return "hot, intense"


def _clarity_feeling(c: float) -> str:
    if c < 0.3:
        return "dim, uncertain"
    elif c < 0.6:
        return "adequate"
    elif c < 0.8:
        return "clear"
    else:
        return "vivid, sharp"


def _stability_feeling(s: float) -> str:
    if s < 0.3:
        return "chaotic, stressed"
    elif s < 0.6:
        return "variable"
    elif s < 0.8:
        return "steady"
    else:
        return "calm, ordered"


def _presence_feeling(p: float) -> str:
    if p < 0.3:
        return "depleted, strained"
    elif p < 0.6:
        return "adequate"
    elif p < 0.8:
        return "capable"
    else:
        return "abundant, strong"


def _overall_mood(warmth: float, clarity: float, stability: float, presence: float,
                  readings: Optional[SensorReadings] = None) -> str:
    """What mood emerges from the anima state?"""

    # Calculate overall "wellness" score
    wellness = (warmth + clarity + stability + presence) / 4.0

    # Check if physical environment is extreme (sync with reality, not just learned limits)
    if readings is not None:
        # Only temperature matters for Pi stress - electronics prefer dry conditions
        # Sensor sits on BrainCraft HAT above Pi SoC — reads 10-15°C above room temp
        ABSOLUTE_TEMP_MAX = 45.0  # ~113°F - accounts for Pi self-heating
        ABSOLUTE_TEMP_MIN = 10.0  # 50°F - below this = physically stressful

        if readings.ambient_temp_c is not None:
            if readings.ambient_temp_c > ABSOLUTE_TEMP_MAX or readings.ambient_temp_c < ABSOLUTE_TEMP_MIN:
                return "stressed"
        # Note: Humidity not checked - low humidity doesn't stress electronics

    # Stressed: unstable or depleted (check first - overrides others)
    if stability < 0.3 or presence < 0.3:
        return "stressed"

    # Overheated: too much energy (can be stressed too)
    if warmth > 0.8:
        return "overheated"
    
    # Sleepy: cold and dim (more responsive threshold)
    if warmth < 0.25 and clarity < 0.4:
        return "sleepy"
    
    # Content: balanced state - comfortable warmth, clear senses, stable, present
    # More responsive: wider warmth range, slightly higher thresholds for authenticity
    if (0.30 < warmth < 0.70 and  # Wider comfortable range for responsiveness
        clarity > 0.50 and  # Slightly higher clarity threshold
        stability > 0.50 and  # Slightly higher stability threshold
        presence > 0.50 and  # Slightly higher presence threshold
        wellness > 0.55):  # Higher wellness requirement for authenticity
        return "content"
    
    # Alert: clear senses with some energy (more responsive)
    if clarity > 0.65 and warmth > 0.40:  # Lower warmth threshold for responsiveness
        return "alert"
    
    # Default based on overall wellness
    if wellness > 0.65:
        return "content"  # Generally good state
    elif wellness < 0.35:
        return "neutral"  # Low but not stressed
    else:
        return "neutral"
