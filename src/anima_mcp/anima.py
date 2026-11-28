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
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING
from .sensors.base import SensorReadings
from .config import get_calibration, NervousSystemCalibration
from .computational_neural import get_computational_neural_state

if TYPE_CHECKING:
    from .memory import Anticipation


_PREDICTION_ACCURACY_UNSET = object()
_EXTERNAL_LIGHT_UNSET = object()
CLARITY_ATTRIBUTION_SCHEMA = "anima.clarity_attribution.v1"

# Emit an extreme-state diagnostic only when a dimension crosses a band.  The
# old per-sample print turned a persistent, inspectable state into hundreds of
# identical log lines and obscured actual failures.
_extreme_diagnostic_bands: Dict[str, str] = {}


def _emit_extreme_diagnostic(name: str, value: float) -> None:
    band = "high" if value > 0.95 else "low" if value < 0.05 else None
    previous = _extreme_diagnostic_bands.get(name)
    if band is None:
        _extreme_diagnostic_bands.pop(name, None)
        return
    if band == previous:
        return
    _extreme_diagnostic_bands[name] = band
    import sys
    print(
        f"[Anima] NOTICE: {name}={value:.2f} entered the {band} diagnostic band",
        file=sys.stderr,
        flush=True,
    )


def _reset_extreme_diagnostics() -> None:
    """Clear process-local diagnostic edge state (tests and deliberate reset)."""
    _extreme_diagnostic_bands.clear()


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


def _resolve_prediction_accuracy(value: Any) -> Optional[float]:
    """Use process-local prediction state only when no owner value was supplied."""
    if value is _PREDICTION_ACCURACY_UNSET:
        return _get_prediction_accuracy()
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))


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
    clarity_attribution: Optional[dict] = None

    def to_dict(self) -> dict:
        data = {
            "warmth": self.warmth,
            "clarity": self.clarity,
            "stability": self.stability,
            "presence": self.presence,
            "feeling": self.feeling(),
            "readings": self.readings.to_dict(),
        }
        if self.clarity_attribution is not None:
            data["clarity_attribution"] = deepcopy(self.clarity_attribution)
        return data

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

    def smooth(self, anima: Anima, elapsed_seconds: float = 2.0) -> Anima:
        if self._prev is None:
            self._prev = {
                "warmth": anima.warmth, "clarity": anima.clarity,
                "stability": anima.stability, "presence": anima.presence,
            }
            return anima

        try:
            elapsed_seconds = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed_seconds = 2.0
        if not math.isfinite(elapsed_seconds):
            elapsed_seconds = 2.0
        elapsed_seconds = max(0.0, min(60.0, elapsed_seconds))
        tick_scale = elapsed_seconds / 2.0

        smoothed = {}
        for dim in ("warmth", "clarity", "stability", "presence"):
            raw = getattr(anima, dim)
            base_alpha = self.ALPHA[dim]
            alpha = 1.0 - (1.0 - base_alpha) ** tick_scale
            smoothed[dim] = alpha * raw + (1 - alpha) * self._prev[dim]
            self._prev[dim] = smoothed[dim]

        published = {
            dimension: round(value, 3)
            for dimension, value in smoothed.items()
        }
        clarity_attribution = deepcopy(anima.clarity_attribution)
        if clarity_attribution is not None:
            clarity_attribution["raw_value"] = anima.clarity
            clarity_attribution["published_value"] = published["clarity"]
            clarity_attribution["temporal_filter"] = {
                "kind": "elapsed_time_ema",
                "reference_interval_seconds": 2.0,
                "base_alpha": self.ALPHA["clarity"],
                "elapsed_seconds": round(elapsed_seconds, 6),
                "effective_alpha": round(
                    1.0 - (1.0 - self.ALPHA["clarity"]) ** tick_scale,
                    6,
                ),
            }

        return Anima(
            warmth=published["warmth"],
            clarity=published["clarity"],
            stability=published["stability"],
            presence=published["presence"],
            readings=anima.readings,
            anticipation=anima.anticipation,
            is_anticipating=anima.is_anticipating,
            clarity_attribution=clarity_attribution,
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


def sense_self(
    readings: SensorReadings,
    calibration: Optional[NervousSystemCalibration] = None,
    drift_midpoints: Optional[Dict[str, float]] = None,
    salience_weights: Optional[Dict[str, float]] = None,
    prediction_accuracy: Any = _PREDICTION_ACCURACY_UNSET,
    external_light_lux: Any = _EXTERNAL_LIGHT_UNSET,
) -> Anima:
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
        prediction_accuracy: Authoritative 0-1 accuracy supplied by another
            process.  Omitting it uses the process-local adaptive model; passing
            ``None`` deliberately represents insufficient owner data.
        external_light_lux: Gated external-light residual from the body broker.
            Passing ``None`` means attribution is not ready, so light is omitted
            from clarity. Omitting the argument preserves raw-lux compatibility
            for callers that do not yet carry attribution provenance.
    """
    if calibration is None:
        calibration = get_calibration()

    if drift_midpoints:
        calibration = _apply_drift_to_calibration(calibration, drift_midpoints)

    frozen_channel_count = _frozen_channel_count(readings)
    warmth = _sense_warmth(readings, calibration, salience_weights=salience_weights)
    clarity_attribution: Dict[str, Any] = {}
    clarity = _sense_clarity(
        readings,
        calibration,
        _resolve_prediction_accuracy(prediction_accuracy),
        salience_weights=salience_weights,
        frozen_channel_count=frozen_channel_count,
        external_light_lux=external_light_lux,
        attribution=clarity_attribution,
    )
    stability = _sense_stability(
        readings,
        calibration,
        salience_weights=salience_weights,
        frozen_channel_count=frozen_channel_count,
    )
    presence = _sense_presence(readings, calibration, salience_weights=salience_weights)

    # Clamp to valid range — defensive against edge cases in weighted averages
    warmth = max(0.0, min(1.0, warmth))
    clarity = max(0.0, min(1.0, clarity))
    stability = max(0.0, min(1.0, stability))
    presence = max(0.0, min(1.0, presence))

    clarity_attribution["raw_value"] = clarity
    clarity_attribution["published_value"] = clarity
    clarity_attribution["temporal_filter"] = {"kind": "none"}

    # Sanity diagnostics are edge-triggered. Persistent values remain visible
    # through state telemetry without flooding the journal.
    for name, value in [("warmth", warmth), ("clarity", clarity),
                        ("stability", stability), ("presence", presence)]:
        _emit_extreme_diagnostic(name, value)

    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings,
        clarity_attribution=clarity_attribution,
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
    prediction_accuracy: Any = _PREDICTION_ACCURACY_UNSET,
    external_light_lux: Any = _EXTERNAL_LIGHT_UNSET,
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
        prediction_accuracy: Authoritative 0-1 accuracy supplied by another
            process.  Omitting it uses the process-local adaptive model; passing
            ``None`` deliberately represents insufficient owner data.
        external_light_lux: Gated external-light residual. Explicit ``None``
            means unknown and removes the light contribution from clarity.

    Returns:
        Anima with potential anticipation influence and exploration
    """
    if calibration is None:
        calibration = get_calibration()

    if drift_midpoints:
        calibration = _apply_drift_to_calibration(calibration, drift_midpoints)

    # Base sensed state (raw, before memory influence). Advance channel
    # liveness once for this physical sample, not once per derived dimension.
    frozen_channel_count = _frozen_channel_count(readings)
    raw_warmth = _sense_warmth(readings, calibration, salience_weights=salience_weights)
    clarity_attribution: Dict[str, Any] = {}
    raw_clarity = _sense_clarity(
        readings,
        calibration,
        _resolve_prediction_accuracy(prediction_accuracy),
        salience_weights=salience_weights,
        frozen_channel_count=frozen_channel_count,
        external_light_lux=external_light_lux,
        attribution=clarity_attribution,
    )
    raw_stability = _sense_stability(
        readings,
        calibration,
        salience_weights=salience_weights,
        frozen_channel_count=frozen_channel_count,
    )
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

    clarity_attribution["raw_value"] = raw_clarity
    clarity_attribution["published_value"] = clarity
    clarity_attribution["perceptual_modulation"] = {
        "anticipation_applied": is_anticipating,
        "exploration_applied": is_exploring,
        "anticipation_blend_factor": (
            actual_blend_factor if is_anticipating else None
        ),
    }

    return Anima(
        warmth=warmth,
        clarity=clarity,
        stability=stability,
        presence=presence,
        readings=readings,
        anticipation=anticipation_dict,
        is_anticipating=is_anticipating or is_exploring,
        clarity_attribution=clarity_attribution,
    )


# _neural_fallback_note (#175): the legacy neural terms in warmth, clarity and
# presence carry weight 0.0 by default (de-aliased from CPU%, #173/#175), yet
# each still called get_computational_neural_state() whenever the readings
# lacked the bands. That sensor is a stateful singleton: every call appends to
# its CPU/temperature history and advances its EMA and dt clock, so a term that
# contributed nothing still moved the delta/theta/gamma the next real reading
# reports, and could raise out of sense_self(). A zero-weight term now samples
# nothing. A non-zero override keeps the old fallback unchanged.


def _sense_warmth(r: SensorReadings, cal: NervousSystemCalibration, *, salience_weights: Optional[Dict[str, float]] = None) -> float:
    """
    How warm does the creature feel?

    Sources:
    - CPU temperature (internal body heat)
    - Ambient temperature (environmental warmth)
    - Optional computational activity view (disabled by default calibration)

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

    # Legacy-named beta/gamma computational views. This contribution is
    # disabled by default calibration to avoid counting CPU activity twice.
    # Fallback 0.0, not the historical 0.20: a config file predating the
    # "neural" key must not silently resurrect the CPU% alias (config.py
    # warmth_weights explains the de-alias).
    neural_weight = cal.warmth_weights.get("neural", 0.0)
    if r.eeg_beta_power is not None and r.eeg_gamma_power is not None:
        components.append((r.eeg_beta_power + r.eeg_gamma_power) / 2)
        weights.append(neural_weight)
    elif neural_weight > 0:
        # Fallback: derive from system metrics directly. Only for a term that
        # counts — see _neural_fallback_note.
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50
        )
        components.append((neural.beta + neural.gamma) / 2)
        weights.append(neural_weight)

    if not components:
        return 0.5

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5
    
    return round(sum(c * w for c, w in zip(components, weights)) / total_weight, 3)


# Per-channel liveness. A channel is "frozen" when it has returned the exact
# same float on every sample for this many consecutive readings. Deliberately
# generous: real sensors jitter in their low bits constantly, so byte-identical
# repetition over minutes means the reader is stuck, not that the world is calm.
# The BMP280 has repeated one value for over a week.
_FROZEN_REPEAT_THRESHOLD = 60
_CHANNEL_NAMES = ("cpu_temp_c", "ambient_temp_c", "humidity_pct", "light_lux", "pressure_hpa")
_last_channel_value: Dict[str, float] = {}
_channel_repeat_count: Dict[str, int] = {}
_last_liveness_sample_key: object = None
_last_frozen_channel_count = 0


def _frozen_channel_count(r: SensorReadings) -> int:
    """How many present channels have stopped changing at all.

    Counts byte-identical repetition, not low variance. A quiet room still
    moves the low bits of a real ADC; a channel that returns a bit-for-bit
    identical float 60 times running is not reading the world.

    This is the only per-channel staleness check in the stack — every other
    freshness gate operates on the envelope timestamp, which stays current
    even when an individual sensor has been dead for days.
    """
    global _last_liveness_sample_key, _last_frozen_channel_count

    timestamp = getattr(r, "timestamp", None)
    sample_key = ("timestamp", timestamp) if timestamp is not None else ("object", id(r))
    if sample_key == _last_liveness_sample_key:
        return _last_frozen_channel_count

    frozen = 0
    for name in _CHANNEL_NAMES:
        value = getattr(r, name, None)
        if value is None:
            _channel_repeat_count[name] = 0
            _last_channel_value.pop(name, None)
            continue
        if _last_channel_value.get(name) == value:
            _channel_repeat_count[name] = _channel_repeat_count.get(name, 0) + 1
        else:
            _last_channel_value[name] = value
            _channel_repeat_count[name] = 0
        if _channel_repeat_count[name] >= _FROZEN_REPEAT_THRESHOLD:
            frozen += 1
    _last_liveness_sample_key = sample_key
    _last_frozen_channel_count = frozen
    return frozen


def _reset_channel_liveness() -> None:
    """Clear the liveness tracker (tests, and any deliberate re-baseline)."""
    global _last_liveness_sample_key, _last_frozen_channel_count
    _last_channel_value.clear()
    _channel_repeat_count.clear()
    _last_liveness_sample_key = None
    _last_frozen_channel_count = 0


def _sense_clarity(
    r: SensorReadings,
    cal: NervousSystemCalibration,
    prediction_accuracy: Optional[float] = None,
    *,
    salience_weights: Optional[Dict[str, float]] = None,
    frozen_channel_count: Optional[int] = None,
    external_light_lux: Any = _EXTERNAL_LIGHT_UNSET,
    attribution: Optional[Dict[str, Any]] = None,
) -> float:
    """
    How clearly can the creature perceive its own internal state?

    This is "internal seeing" - self-perception accuracy plus environmental perception.

    Sources:
    - Prediction accuracy: How well I predict my own state changes (1 - mean_error)
    - Optional alpha computational view (exactly 1-beta; disabled by default)
    - Sensor coverage: Data richness (how complete is my self-perception)
    - External light: Learned raw-lux minus self-glow residual, log-scaled

    The physical VEML7700 reading remains raw telemetry. The body broker passes
    the residual only after its attribution evidence gate; explicit ``None``
    means environmental light is unknown and this component is omitted. The
    private sentinel preserves compatibility for callers that have not yet
    adopted the provenance-bearing argument, but the live body path never uses
    that fallback.
    """
    salience = salience_weights or {}
    components: list[float] = []
    weights: list[float] = []
    details: Dict[str, Dict[str, Any]] = {}

    def add_component(
        name: str,
        value: float,
        weight: float,
        **metadata: Any,
    ) -> None:
        numeric_value = max(0.0, min(1.0, float(value)))
        numeric_weight = max(0.0, float(weight))
        components.append(numeric_value)
        weights.append(numeric_weight)
        details[name] = {
            "available": True,
            "value": round(numeric_value, 6),
            "configured_weight": round(numeric_weight, 6),
            **metadata,
        }

    # Prediction accuracy: How well I understand my own state changes
    # This is the core of "internal seeing" - accurate self-prediction = clarity
    if prediction_accuracy is not None:
        # prediction_accuracy should be 0-1 (1 - normalized_mean_error)
        add_component(
            "prediction_accuracy",
            prediction_accuracy,
            cal.clarity_weights.get("prediction_accuracy", 0.5),
            source="adaptive_prediction_error",
            fallback=False,
        )
    else:
        # Fallback: use a neutral value when no prediction data available yet
        # This happens during early startup before enough observations accumulate
        add_component(
            "prediction_accuracy",
            0.5,
            cal.clarity_weights.get("prediction_accuracy", 0.5),
            source="neutral_fallback",
            fallback=True,
        )

    # Sensor coverage: Data richness (meta-clarity about available information)
    #
    # Counts channels that are actually INFORMING, not merely present. A sensor
    # that returns the same byte-identical value forever is not telling Lumen
    # anything, and counting it as coverage means clarity cannot notice it died.
    # The BMP280 has returned 682.5015433175248 unchanged since the moment Lumen
    # woke from the July blackout, and nothing anywhere in the stack could see
    # it: coverage held at exactly 1.0 for 30 days with sd 0.00000.
    present_sensor_count = sum(
        1 for v in [
            r.cpu_temp_c, r.ambient_temp_c, r.humidity_pct,
            r.light_lux, r.pressure_hpa,
        ] if v is not None
    )
    frozen_count = (
        _frozen_channel_count(r)
        if frozen_channel_count is None else frozen_channel_count
    )
    sensor_count = present_sensor_count - frozen_count
    coverage = max(0.0, sensor_count) / 5
    add_component(
        "sensor_coverage",
        coverage,
        cal.clarity_weights.get("sensor_coverage", 0.15),
        source="informing_physical_channels",
        present_channels=present_sensor_count,
        frozen_channels=frozen_count,
        informing_channels=max(0, sensor_count),
        total_channels=5,
    )

    # Light: gated external residual. Legacy callers that omit attribution keep
    # their former raw behavior; the live broker passes either a residual or
    # explicit None, so self-glow never enters environmental clarity there.
    # Log scale: 1 lux → 0.0, light_max_lux → 1.0
    clarity_light = (
        r.light_lux
        if external_light_lux is _EXTERNAL_LIGHT_UNSET
        else external_light_lux
    )
    if (
        isinstance(clarity_light, (int, float))
        and not isinstance(clarity_light, bool)
        and math.isfinite(float(clarity_light))
    ):
        clarity_light = max(0.0, float(clarity_light))
        if clarity_light > 1.0:
            log_max = math.log10(max(10.0, cal.light_max_lux))
            light_clarity = min(1.0, math.log10(clarity_light) / log_max)
        else:
            light_clarity = 0.0
        # Salience is an attention priority, not sensor gain. Multiplying this
        # positive-only component by attention created a feedback loop: high
        # clarity caused dissatisfaction, dissatisfaction raised light
        # salience, and salience made clarity still higher. Preserve the
        # physical/log-scaled measurement and report attention independently.
        add_component(
            "external_light",
            light_clarity,
            cal.clarity_weights.get("world_light", 0.15),
            source=(
                "raw_lux_legacy_compatibility"
                if external_light_lux is _EXTERNAL_LIGHT_UNSET
                else "gated_external_lux_residual"
            ),
            input_lux=round(clarity_light, 6),
            light_max_lux=round(float(cal.light_max_lux), 6),
        )
    else:
        details["external_light"] = {
            "available": False,
            "value": None,
            "configured_weight": round(
                max(0.0, float(cal.clarity_weights.get("world_light", 0.15))),
                6,
            ),
            "source": "gated_external_lux_residual_unavailable",
            "input_lux": None,
        }

    # Alpha computational view in a legacy EEG-named field.
    # Fallback 0.0, not the historical 0.3 — same reasoning as warmth's.
    alpha_weight = cal.clarity_weights.get("neural", 0.0)
    if r.eeg_alpha_power is not None:
        add_component(
            "computational_alpha",
            r.eeg_alpha_power,
            alpha_weight,
            source="sensor_readings.eeg_alpha_power",
        )
    elif alpha_weight > 0:
        # Fall back to computational neural (derives bands from system state).
        # Only for a term that counts — see _neural_fallback_note.
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50,
            cpu_temp=r.cpu_temp_c
        )
        add_component(
            "computational_alpha",
            neural.alpha,
            alpha_weight,
            source="computational_alpha_fallback",
        )
    else:
        details["computational_alpha"] = {
            "available": False,
            "value": None,
            "configured_weight": 0.0,
            "source": "zero_weight_not_sampled",
        }

    total_weight = sum(weights)
    value = (
        0.5
        if not components or total_weight == 0
        else sum(c * w for c, w in zip(components, weights)) / total_weight
    )

    if attribution is not None:
        for detail in details.values():
            configured_weight = detail.get("configured_weight", 0.0)
            if detail.get("available") and total_weight > 0:
                normalized_weight = configured_weight / total_weight
                detail["normalized_weight"] = round(normalized_weight, 6)
                detail["contribution"] = round(
                    float(detail["value"]) * normalized_weight,
                    6,
                )
            else:
                detail["normalized_weight"] = 0.0
                detail["contribution"] = 0.0

        try:
            light_attention = float(salience.get("light", 1.0))
        except (TypeError, ValueError):
            light_attention = 1.0
        if not math.isfinite(light_attention):
            light_attention = 1.0
        attribution.update({
            "schema": CLARITY_ATTRIBUTION_SCHEMA,
            "status": "ready",
            "authority": "sense_self_sensor_fusion",
            "aggregation": "weighted_mean_of_available_components",
            "measurement_policy": "attention_reported_not_applied_as_sensor_gain",
            "raw_value": round(value, 3),
            "published_value": round(value, 3),
            "total_available_weight": round(total_weight, 6),
            "components": details,
            "attention": {
                "light_salience": round(light_attention, 6),
                "source": (
                    "experiential_filter"
                    if salience_weights is not None
                    else "neutral_default"
                ),
                "role": "attention_priority",
                "used_in_measurement": False,
            },
        })

    return round(value, 3)


def _sense_stability(
    r: SensorReadings,
    cal: NervousSystemCalibration,
    *,
    salience_weights: Optional[Dict[str, float]] = None,
    frozen_channel_count: Optional[int] = None,
) -> float:
    """
    How stable/ordered does the environment feel?

    This is inverse of entropy - high stability = low chaos.

    Sources:
    - Humidity near ideal (calibrated)
    - Memory headroom
    - Complete sensor data
    - Barometric pressure stability (deviations from local normal)
    - Theta/delta computational I/O and CPU/thermal-stability views
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

    # Missing sensors = uncertainty. A frozen channel counts as missing: it is
    # present in the payload and contributing nothing, which is strictly worse
    # than absent because it looks like information. See _frozen_channel_count.
    missing = sum(1 for v in [
        r.cpu_temp_c, r.ambient_temp_c, r.humidity_pct,
        r.light_lux, r.pressure_hpa
    ] if v is None) + (
        _frozen_channel_count(r)
        if frozen_channel_count is None else frozen_channel_count
    )
    missing = min(5, missing)
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

    # Computational dynamics heuristic: low I/O-integration + CPU/thermal
    # stability view contributes instability under its explicit 0.1 weight.
    # Fallback matches the dict default (was 0.2 vs the dict's 0.1 — a config
    # missing the key silently DOUBLED the weight; #175 trap class).
    neural_weight = cal.stability_weights.get("neural", 0.1)
    if r.eeg_theta_power is not None and r.eeg_delta_power is not None:
        # Invert: low groundedness = high instability
        neural_groundedness = (r.eeg_theta_power + r.eeg_delta_power) / 2
        neural_instability = 1.0 - neural_groundedness
    else:
        # Absent theta/delta are UNKNOWN, read at full uncertainty — the same
        # convention as the missing-sensors term above (CLAUDE.md invariant 2).
        # This used to re-derive the bands from the computational-neural
        # singleton: in the broker that is the very source whose failure left
        # them None (sensors/pi.py), it coerced a missing cpu_percent to an idle
        # 0, and it appended a phantom sample to the shared CPU history that
        # the next real reading's delta is computed from (#175).
        neural_instability = 1.0
    instability += neural_instability * neural_weight
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
    - Optional gamma switch/interrupt view (disabled by default calibration)
    """
    salience = salience_weights or {}
    void = 0.0
    count = 0

    if r.disk_percent is not None:
        weight = cal.presence_weights.get("disk", 0.3125)
        void += (r.disk_percent / 100) * weight
        count += weight

    if r.memory_percent is not None:
        weight = cal.presence_weights.get("memory", 0.375)
        memory_void = (r.memory_percent / 100) * salience.get("memory", 1.0)
        memory_void = min(1, memory_void)
        void += memory_void * weight
        count += weight

    if r.cpu_percent is not None:
        weight = cal.presence_weights.get("cpu", 0.3125)
        cpu_void = (r.cpu_percent / 100) * salience.get("cpu", 1.0)
        cpu_void = min(1, cpu_void)
        void += cpu_void * weight
        count += weight

    # Neural void: High gamma = high ctx switching = scattered = less presence
    # Weight 0.0 by design and fallback 0.0 deliberately (#175): gamma is
    # mostly cpu_percent renamed (r=+0.743), which already has its own door
    # above — and a config predating the key must not resurrect the alias.
    weight = cal.presence_weights.get("neural", 0.0)
    if r.eeg_gamma_power is not None:
        void += r.eeg_gamma_power * weight
        count += weight
    elif weight > 0:
        # Fall back to computational neural (derives bands from system state).
        # Only for a term that counts — see _neural_fallback_note.
        neural = get_computational_neural_state(
            cpu_percent=r.cpu_percent or 0,
            memory_percent=r.memory_percent or 50,
            cpu_temp=r.cpu_temp_c
        )
        void += neural.gamma * weight
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
