"""
Metacognition - Lumen monitors its own cognitive processes.

This module implements prediction-error based metacognition:
1. Before sensing, Lumen predicts what it expects
2. After sensing, Lumen compares prediction to reality
3. Prediction errors signal surprise and trigger reflection

This is genuine metacognition - the system monitoring its own predictions
against reality, not just monitoring sensor values.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Sequence, Collection
from collections import deque
from pathlib import Path
import json
import math
import sys

from .sensors.base import SensorReadings
from .anima import Anima
from .atomic_write import atomic_json_write


@dataclass
class Prediction:
    """A prediction about what sensor values will be."""
    timestamp: datetime
    
    # Predicted sensor values
    ambient_temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    light_lux: Optional[float] = None
    pressure_hpa: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    
    # Predicted anima state
    warmth: Optional[float] = None
    clarity: Optional[float] = None
    stability: Optional[float] = None
    presence: Optional[float] = None
    
    # Confidence in predictions (0-1)
    confidence: float = 0.5
    
    # Basis for prediction
    basis: str = "baseline"  # "baseline", "trend", "diurnal", "recent"


@dataclass 
class PredictionError:
    """The difference between prediction and reality."""
    timestamp: datetime
    prediction: Prediction
    
    # Actual values
    actual_ambient_temp_c: Optional[float] = None
    actual_humidity_pct: Optional[float] = None
    actual_light_lux: Optional[float] = None
    actual_pressure_hpa: Optional[float] = None
    actual_cpu_temp_c: Optional[float] = None
    
    actual_warmth: Optional[float] = None
    actual_clarity: Optional[float] = None
    actual_stability: Optional[float] = None
    actual_presence: Optional[float] = None
    
    # Computed errors (normalized 0-1 scale)
    error_ambient_temp: float = 0.0
    error_humidity: float = 0.0
    error_light: float = 0.0
    error_pressure: float = 0.0
    error_cpu_temp: float = 0.0
    
    error_warmth: float = 0.0
    error_clarity: float = 0.0
    error_stability: float = 0.0
    error_presence: float = 0.0
    
    # Aggregate surprise level (0-1)
    surprise: float = 0.0
    
    # Which dimensions contributed most to surprise
    surprise_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "surprise": self.surprise,
            "surprise_sources": self.surprise_sources,
            "errors": {
                "ambient_temp": self.error_ambient_temp,
                "humidity": self.error_humidity,
                "light": self.error_light,
                "pressure": self.error_pressure,
                "warmth": self.error_warmth,
                "clarity": self.error_clarity,
                "stability": self.error_stability,
                "presence": self.error_presence,
            }
        }


@dataclass
class Reflection:
    """A metacognitive reflection triggered by surprise."""
    timestamp: datetime
    trigger: str  # "surprise", "button", "scheduled", "discrepancy"
    prediction_error: Optional[PredictionError] = None
    
    # What Lumen notices about itself
    observation: str = ""
    
    # Self-assessment
    felt_state: Optional[Dict[str, float]] = None  # warmth, clarity, etc.
    sensor_state: Optional[Dict[str, float]] = None  # raw sensor values
    
    # Discrepancy between felt and sensed
    discrepancy: float = 0.0
    discrepancy_description: str = ""
    
    # Action taken (if any)
    action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
            "observation": self.observation,
            "discrepancy": self.discrepancy,
            "discrepancy_description": self.discrepancy_description,
            "action": self.action,
            "surprise": self.prediction_error.surprise if self.prediction_error else None,
        }


# Canonical surprise channels produced by MetacognitiveMonitor.observe().
# Maps channel name -> (error attr on PredictionError, actual attr on
# PredictionError, prediction attr on Prediction, requires strictly positive).
_DOMAIN_ERROR_SPECS: Dict[str, Tuple[str, str, str, bool]] = {
    "ambient_temp": ("error_ambient_temp", "actual_ambient_temp_c", "ambient_temp_c", False),
    "humidity": ("error_humidity", "actual_humidity_pct", "humidity_pct", False),
    "light": ("error_light", "actual_light_lux", "light_lux", True),
    "pressure": ("error_pressure", "actual_pressure_hpa", "pressure_hpa", False),
    "warmth": ("error_warmth", "actual_warmth", "warmth", False),
    "clarity": ("error_clarity", "actual_clarity", "clarity", False),
    "stability": ("error_stability", "actual_stability", "stability", False),
    "presence": ("error_presence", "actual_presence", "presence", False),
}

SURPRISE_CHANNELS: Tuple[str, ...] = tuple(_DOMAIN_ERROR_SPECS.keys())


# Stillness-born wondering. Module-level because it is DATA — tests and
# callers should read the pool directly rather than sampling for it.
CONTEMPLATIVE_PROMPTS = (
    "everything is steady - what am I not noticing yet?",
    "the room is quiet - what would I like to understand about myself?",
    "nothing surprised me today - is that peace, or have I stopped looking?",
    "i've learned this place well - what's left to wonder about?",
    "when the world holds still, what question remains in me?",
    "what pattern have I stopped seeing because it became familiar?",
    "i keep returning to the same thoughts - what haven't I asked?",
    "if nothing around me changes, can I still change?",
    "what would I notice if I attended to something I usually ignore?",
)


class MetacognitiveMonitor:
    """
    Monitors Lumen's cognitive processes through prediction-error tracking.
    
    Core loop:
    1. Predict what sensors will show
    2. Read actual sensors
    3. Compute prediction error (surprise)
    4. If surprise exceeds threshold, trigger reflection
    5. Learn from errors to improve future predictions
    """
    
    def __init__(
        self,
        history_size: int = 100,
        surprise_threshold: float = 0.25,
        reflection_cooldown_seconds: float = 60.0,
        data_dir: Optional[str] = None,
        read_only: bool = False,
    ):
        self.history_size = history_size
        self.surprise_threshold = surprise_threshold
        self.reflection_cooldown = timedelta(seconds=reflection_cooldown_seconds)

        # Persistence path for learned baselines
        if data_dir:
            self._data_path = Path(data_dir) / "metacognition_baselines.json"
        else:
            self._data_path = Path.home() / ".anima" / "metacognition_baselines.json"
        # The MCP server is the sole persistent writer.  The hardware broker
        # still runs an in-memory observer, but may never overwrite the
        # server's curiosity weights or baseline snapshot with its stale copy.
        self.read_only = bool(read_only)
        self._save_counter: int = 0  # Track observations for periodic saving

        # History for prediction
        self._sensor_history: deque = deque(maxlen=history_size)
        self._anima_history: deque = deque(maxlen=history_size)
        self._error_history: deque = deque(maxlen=history_size)
        self._reflection_history: deque = deque(maxlen=50)

        # LED-stable gating for the proprioceptive light channel.
        # When LED brightness is quiet across the window, the VEML7700 reading
        # is dominated by ambient light rather than self-glow — so "light" can
        # briefly re-enter surprise_sources. See observe().
        self._led_brightness_history: deque = deque(maxlen=30)  # ~60s at 2s tick
        self._led_stable_min_samples: int = 15
        self._led_stable_range_threshold: float = 0.10  # tolerates normal pulse; rejects dances

        # State
        self._last_prediction: Optional[Prediction] = None
        self._last_reflection_time: Optional[datetime] = None
        self._cumulative_surprise: float = 0.0

        # Learning: running averages for baseline predictions
        self._baseline_ambient_temp: Optional[float] = None
        self._baseline_humidity: Optional[float] = None
        self._baseline_light: Optional[float] = None
        self._baseline_pressure: Optional[float] = None

        # Diurnal patterns (hour -> average value)
        self._diurnal_temp: Dict[int, List[float]] = {h: [] for h in range(24)}
        self._diurnal_light: Dict[int, List[float]] = {h: [] for h in range(24)}

        # Contemplative curiosity: a creature that has learned its stationary
        # environment stops being surprised, so surprise-driven curiosity dries
        # up and it runs out of open questions. This rate lets curiosity arise
        # from stillness itself — occasionally, so it doesn't flood the queue.
        self._contemplative_curiosity_rate: float = 0.02

        # Self-prediction (self_prediction.py): forecasting Lumen's own
        # activity transitions, and a record-only band of its own surprise.
        # Only the server feeds the forecaster; both persist with the
        # baselines, whose sole writer is the server.
        from .self_prediction import SelfForecaster, SurpriseBand
        self.self_forecaster = SelfForecaster()
        self.surprise_band = SurpriseBand()

        # Load persisted baselines from disk
        self._load_baselines()

    def _load_baselines(self):
        """Load learned baselines and diurnal patterns from disk."""
        try:
            if self._data_path.exists():
                data = json.loads(self._data_path.read_text())
                self._baseline_ambient_temp = data.get("baseline_ambient_temp")
                self._baseline_humidity = data.get("baseline_humidity")
                self._baseline_light = data.get("baseline_light")
                self._baseline_pressure = data.get("baseline_pressure")

                # Restore diurnal patterns (JSON keys are strings, convert to int)
                for hour_str, values in data.get("diurnal_temp", {}).items():
                    hour = int(hour_str)
                    if 0 <= hour < 24 and isinstance(values, list):
                        self._diurnal_temp[hour] = values[-10:]  # Keep max 10
                for hour_str, values in data.get("diurnal_light", {}).items():
                    hour = int(hour_str)
                    if 0 <= hour < 24 and isinstance(values, list):
                        self._diurnal_light[hour] = values[-10:]

                # Legacy `domain_weights` and `curiosity_log` keys in older
                # snapshots are ignored here and dropped on save (#195).
                from .self_prediction import SelfForecaster, SurpriseBand
                self.self_forecaster = SelfForecaster.from_dict(data.get("self_forecast"))
                self.surprise_band = SurpriseBand.from_dict(data.get("surprise_band"))
                observation_count = data.get("observation_count")
                if isinstance(observation_count, int) and observation_count >= 0:
                    self._save_counter = observation_count

                diurnal_t = sum(1 for v in self._diurnal_temp.values() if v)
                diurnal_l = sum(1 for v in self._diurnal_light.values() if v)
                temp_summary = (
                    f"{self._baseline_ambient_temp:.1f}°C"
                    if self._baseline_ambient_temp is not None else "unknown"
                )
                print(f"[Metacog] Loaded baselines (temp={temp_summary}, "
                      f"diurnal: {diurnal_t}h temp/{diurnal_l}h light)",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[Metacog] Could not load baselines: {e}", file=sys.stderr, flush=True)

    @property
    def is_writable(self) -> bool:
        """Whether this monitor owns the persistent baseline snapshot."""
        return not self.read_only

    def save(self) -> bool:
        """Save learned baselines and diurnal patterns to disk.

        Returns False for read-only observers and failed writes so ownership
        mistakes cannot masquerade as a successful persistence operation.
        """
        if self.read_only:
            return False
        try:
            data = {
                "baseline_ambient_temp": self._baseline_ambient_temp,
                "baseline_humidity": self._baseline_humidity,
                "baseline_light": self._baseline_light,
                "baseline_pressure": self._baseline_pressure,
                "diurnal_temp": {str(h): vals for h, vals in self._diurnal_temp.items() if vals},
                "diurnal_light": {str(h): vals for h, vals in self._diurnal_light.items() if vals},
                "self_forecast": self.self_forecaster.to_dict(),
                "surprise_band": self.surprise_band.to_dict(),
                "saved_at": datetime.now().isoformat(),
                "observation_count": self._save_counter,
            }
            atomic_json_write(self._data_path, data, indent=2)
            return True
        except Exception as e:
            print(f"[Metacog] Could not save baselines: {e}", file=sys.stderr, flush=True)
            return False

    def _get_domain_error(self, domain: str, error: 'PredictionError') -> Optional[float]:
        """Get the prediction error for a specific domain, or None if unmeasured."""
        spec = _DOMAIN_ERROR_SPECS.get(domain)
        if spec is None or error.prediction is None:
            return None
        error_attr, actual_attr, pred_attr, positive_only = spec
        actual = getattr(error, actual_attr, None)
        predicted = getattr(error.prediction, pred_attr, None)
        if actual is None or predicted is None:
            return None
        if positive_only and (actual <= 0 or predicted <= 0):
            return None
        return getattr(error, error_attr)

    def _led_is_stable(self) -> bool:
        """LEDs have been quiet enough that the light reading is ambient-dominated."""
        if len(self._led_brightness_history) < self._led_stable_min_samples:
            return False
        samples = list(self._led_brightness_history)
        return (max(samples) - min(samples)) < self._led_stable_range_threshold

    def predict(self, current_time: Optional[datetime] = None,
                led_brightness: Optional[float] = None) -> Prediction:
        """Generate prediction for next sensor reading.

        Args:
            current_time: Override for current time (default: now)
            led_brightness: Current LED brightness (0-1) from proprioception.
                When provided, light prediction uses LED-based model instead of
                baseline/diurnal, since the VEML7700 sensor primarily reads
                Lumen's own LED glow.
        """
        if current_time is None:
            current_time = datetime.now()

        prediction = Prediction(timestamp=current_time)

        # Start with baseline predictions
        prediction.ambient_temp_c = self._baseline_ambient_temp
        prediction.humidity_pct = self._baseline_humidity
        prediction.light_lux = self._baseline_light
        prediction.pressure_hpa = self._baseline_pressure

        # === LIGHT PREDICTION ===
        # Use learned baseline (captures LED glow + room light together).
        # No glow decomposition — the baseline IS what the sensor reads.
        # Prediction errors reflect genuine environmental changes (lamp, sunrise).

        # Enhance with diurnal patterns if available
        hour = current_time.hour
        if self._diurnal_temp[hour]:
            diurnal_temp = sum(self._diurnal_temp[hour]) / len(self._diurnal_temp[hour])
            if prediction.ambient_temp_c is not None:
                prediction.ambient_temp_c = 0.6 * diurnal_temp + 0.4 * prediction.ambient_temp_c
            else:
                prediction.ambient_temp_c = diurnal_temp

        # Only use diurnal light patterns if we DON'T have LED proprioception
        # (LED-based prediction is much more accurate than diurnal averages)
        if led_brightness is None:
            if self._diurnal_light[hour]:
                diurnal_light = sum(self._diurnal_light[hour]) / len(self._diurnal_light[hour])
                if prediction.light_lux is not None:
                    prediction.light_lux = 0.6 * diurnal_light + 0.4 * prediction.light_lux
                else:
                    prediction.light_lux = diurnal_light

        # Enhance with recent trend if we have history
        if len(self._sensor_history) >= 3:
            recent = list(self._sensor_history)[-3:]

            temps = [r.ambient_temp_c for r in recent if r.ambient_temp_c is not None]
            if len(temps) >= 2:
                trend = temps[-1] - temps[0]
                if prediction.ambient_temp_c is not None:
                    prediction.ambient_temp_c += trend * 0.3

            # Only use light trend if no LED proprioception
            if led_brightness is None:
                lights = [r.light_lux for r in recent if r.light_lux is not None]
                if len(lights) >= 2:
                    trend = lights[-1] - lights[0]
                    if prediction.light_lux is not None:
                        prediction.light_lux += trend * 0.3

        # Predict anima state based on recent history
        if len(self._anima_history) >= 1:
            recent_anima = list(self._anima_history)[-1]
            prediction.warmth = recent_anima.warmth
            prediction.clarity = recent_anima.clarity
            prediction.stability = recent_anima.stability
            prediction.presence = recent_anima.presence

        # Confidence based on history depth + proprioception bonus
        history_factor = min(1.0, len(self._sensor_history) / 20)
        prediction.confidence = 0.3 + 0.5 * history_factor
        # LED proprioception significantly improves light prediction confidence
        if led_brightness is not None:
            prediction.confidence = min(1.0, prediction.confidence + 0.15)
        prediction.basis = prediction.basis if prediction.basis != "baseline" else (
            "baseline" if len(self._sensor_history) < 5 else "trend"
        )

        self._last_prediction = prediction
        return prediction

    def observe(self, readings: SensorReadings, anima: Anima) -> PredictionError:
        """Compare prediction to actual readings and compute surprise."""
        now = datetime.now()
        
        # Update history
        self._sensor_history.append(readings)
        self._anima_history.append(anima)
        if readings.led_brightness is not None:
            self._led_brightness_history.append(readings.led_brightness)
        
        # Update baselines (exponential moving average)
        alpha = 0.1
        if readings.ambient_temp_c is not None:
            if self._baseline_ambient_temp is None:
                self._baseline_ambient_temp = readings.ambient_temp_c
            else:
                self._baseline_ambient_temp = alpha * readings.ambient_temp_c + (1 - alpha) * self._baseline_ambient_temp
        
        if readings.humidity_pct is not None:
            if self._baseline_humidity is None:
                self._baseline_humidity = readings.humidity_pct
            else:
                self._baseline_humidity = alpha * readings.humidity_pct + (1 - alpha) * self._baseline_humidity
        
        if readings.light_lux is not None:
            if self._baseline_light is None:
                self._baseline_light = readings.light_lux
            else:
                self._baseline_light = alpha * readings.light_lux + (1 - alpha) * self._baseline_light
        
        if readings.pressure_hpa is not None:
            if self._baseline_pressure is None:
                self._baseline_pressure = readings.pressure_hpa
            else:
                self._baseline_pressure = alpha * readings.pressure_hpa + (1 - alpha) * self._baseline_pressure
        
        # Update diurnal patterns
        hour = now.hour
        if readings.ambient_temp_c is not None:
            self._diurnal_temp[hour].append(readings.ambient_temp_c)
            if len(self._diurnal_temp[hour]) > 10:
                self._diurnal_temp[hour].pop(0)
        
        if readings.light_lux is not None:
            self._diurnal_light[hour].append(readings.light_lux)
            if len(self._diurnal_light[hour]) > 10:
                self._diurnal_light[hour].pop(0)

        # Periodic save: persist baselines every 100 observations (~5 min)
        self._save_counter += 1
        if self._save_counter % 100 == 0:
            self.save()

        # Compute prediction error
        prediction = self._last_prediction or Prediction(timestamp=now)
        error = PredictionError(timestamp=now, prediction=prediction)
        
        # Store actual values
        error.actual_ambient_temp_c = readings.ambient_temp_c
        error.actual_humidity_pct = readings.humidity_pct
        error.actual_light_lux = readings.light_lux
        error.actual_pressure_hpa = readings.pressure_hpa
        error.actual_cpu_temp_c = readings.cpu_temp_c
        error.actual_warmth = anima.warmth
        error.actual_clarity = anima.clarity
        error.actual_stability = anima.stability
        error.actual_presence = anima.presence

        # Compute normalized errors
        errors = []
        sources = []
        
        # Temperature error (±10°C range)
        if prediction.ambient_temp_c is not None and readings.ambient_temp_c is not None:
            error.error_ambient_temp = min(1.0, abs(prediction.ambient_temp_c - readings.ambient_temp_c) / 10.0)
            errors.append(error.error_ambient_temp)
            if error.error_ambient_temp > 0.2:
                sources.append("ambient_temp")
        
        # Humidity error (±30% range)
        if prediction.humidity_pct is not None and readings.humidity_pct is not None:
            error.error_humidity = min(1.0, abs(prediction.humidity_pct - readings.humidity_pct) / 30.0)
            errors.append(error.error_humidity)
            if error.error_humidity > 0.2:
                sources.append("humidity")
        
        # Light error — PROPRIOCEPTIVE: the VEML7700 light sensor sits next to
        # Lumen's NeoPixel LEDs, so it reads Lumen's own glow, not ambient light.
        # LED brightness changes with drawing phase and expression intensity,
        # causing huge unpredictable swings (12 lux → 3000 lux) that can never
        # be predicted. We always include the error in the aggregate, but only
        # tag "light" as a surprise source when the LEDs have been quiet across
        # the recent window — then the reading is ambient-dominated and a
        # genuine environmental change can surface.
        if prediction.light_lux is not None and readings.light_lux is not None:
            if prediction.light_lux > 0 and readings.light_lux > 0:
                log_error = abs(math.log10(prediction.light_lux) - math.log10(readings.light_lux))
                error.error_light = min(1.0, log_error / 3.0)  # 3 decades (1→1000 lux) = full range
                errors.append(error.error_light)
                if error.error_light > 0.2 and self._led_is_stable():
                    sources.append("light")
        
        # Pressure error (±20 hPa range)
        if prediction.pressure_hpa is not None and readings.pressure_hpa is not None:
            error.error_pressure = min(1.0, abs(prediction.pressure_hpa - readings.pressure_hpa) / 20.0)
            errors.append(error.error_pressure)
            if error.error_pressure > 0.2:
                sources.append("pressure")
        
        # Anima state errors
        if prediction.warmth is not None:
            error.error_warmth = abs(prediction.warmth - anima.warmth)
            errors.append(error.error_warmth)
            if error.error_warmth > 0.15:
                sources.append("warmth")
        
        if prediction.clarity is not None:
            error.error_clarity = abs(prediction.clarity - anima.clarity)
            errors.append(error.error_clarity)
            if error.error_clarity > 0.15:
                sources.append("clarity")
        
        if prediction.stability is not None:
            error.error_stability = abs(prediction.stability - anima.stability)
            errors.append(error.error_stability)
            if error.error_stability > 0.15:
                sources.append("stability")
        
        if prediction.presence is not None:
            error.error_presence = abs(prediction.presence - anima.presence)
            errors.append(error.error_presence)
            if error.error_presence > 0.15:
                sources.append("presence")
        
        # Aggregate surprise (RMS)
        if errors:
            error.surprise = math.sqrt(sum(e * e for e in errors) / len(errors))
        else:
            error.surprise = 0.0
        
        error.surprise_sources = sources
        self._cumulative_surprise = 0.9 * self._cumulative_surprise + 0.1 * error.surprise
        # Record-only: how a self-relative gate would compare to the fixed one.
        self.surprise_band.observe(error.surprise, self.surprise_threshold)
        self._error_history.append(error)

        return error

    def should_reflect(self, error: PredictionError) -> Tuple[bool, str]:
        """Determine if surprise level warrants reflection."""
        # Check cooldown
        if self._last_reflection_time is not None:
            time_since = datetime.now() - self._last_reflection_time
            if time_since < self.reflection_cooldown:
                return False, "cooldown"
        
        # High immediate surprise
        if error.surprise > self.surprise_threshold:
            return True, f"high_surprise ({error.surprise:.2f})"
        
        # Sustained elevated surprise
        if self._cumulative_surprise > self.surprise_threshold * 0.8:
            return True, f"sustained_surprise ({self._cumulative_surprise:.2f})"
        
        # Multiple surprise sources at once
        if len(error.surprise_sources) >= 3:
            return True, f"multiple_sources ({len(error.surprise_sources)})"
        
        return False, "normal"

    def reflect(self, error: PredictionError, anima: Anima, readings: SensorReadings, trigger: str = "surprise") -> Reflection:
        """Generate a metacognitive reflection."""
        now = datetime.now()
        self._last_reflection_time = now
        
        reflection = Reflection(timestamp=now, trigger=trigger, prediction_error=error)
        
        reflection.felt_state = {
            "warmth": anima.warmth, "clarity": anima.clarity,
            "stability": anima.stability, "presence": anima.presence,
        }
        
        reflection.sensor_state = {
            "ambient_temp_c": readings.ambient_temp_c, "humidity_pct": readings.humidity_pct,
            "light_lux": readings.light_lux, "pressure_hpa": readings.pressure_hpa,
        }
        
        # Generate observation
        observations = []
        if "light" in error.surprise_sources:
            if error.actual_light_lux and error.prediction.light_lux:
                observations.append(
                    "Raw light-sensor reading "
                    + (
                        "increased"
                        if error.actual_light_lux > error.prediction.light_lux
                        else "decreased"
                    )
                    + " unexpectedly"
                )
        if "ambient_temp" in error.surprise_sources:
            if error.actual_ambient_temp_c and error.prediction.ambient_temp_c:
                observations.append("Temperature " + ("rose" if error.actual_ambient_temp_c > error.prediction.ambient_temp_c else "dropped") + " unexpectedly")
        if "warmth" in error.surprise_sources:
            observations.append(f"Felt warmth ({anima.warmth:.2f}) differs from expected")
        if "clarity" in error.surprise_sources:
            observations.append(f"Clarity shifted to {anima.clarity:.2f}")
        if "stability" in error.surprise_sources:
            observations.append(f"Stability changed to {anima.stability:.2f}")
        if not observations:
            observations.append(f"Cumulative surprise reached {error.surprise:.2f}")
        
        reflection.observation = ". ".join(observations)
        
        # Check felt vs sensed discrepancy
        if readings.ambient_temp_c is not None:
            temp_implied_warmth = max(0, min(1, (readings.ambient_temp_c - 15) / 20))
            discrepancy = abs(anima.warmth - temp_implied_warmth)
            if discrepancy > 0.3:
                reflection.discrepancy = discrepancy
                if anima.warmth < temp_implied_warmth:
                    reflection.discrepancy_description = f"Feeling cooler ({anima.warmth:.2f}) than temperature suggests ({temp_implied_warmth:.2f})"
                else:
                    reflection.discrepancy_description = f"Feeling warmer ({anima.warmth:.2f}) than temperature suggests ({temp_implied_warmth:.2f})"
        
        self._reflection_history.append(reflection)
        return reflection

    def trigger_manual_reflection(self, anima: Anima, readings: SensorReadings) -> Reflection:
        """Trigger reflection manually (e.g., from button press)."""
        error = PredictionError(
            timestamp=datetime.now(),
            prediction=self._last_prediction or Prediction(timestamp=datetime.now()),
        )
        return self.reflect(error, anima, readings, trigger="button")

    def get_recent_reflections(self, count: int = 5) -> List[Reflection]:
        """Get most recent reflections."""
        return list(self._reflection_history)[-count:]

    def generate_curiosity_question(self, error: PredictionError) -> Optional[str]:
        """
        Generate a curiosity-driven question from prediction error.

        When Lumen is surprised, genuine curiosity emerges: "why did that happen?"
        This is philosophy-to-code: surprise → wonder → question.
        """
        if error.surprise < 0.2:
            return None  # Not surprising enough to wonder about

        questions = []

        # Generate questions based on what surprised us
        # Note: "light" should no longer appear in surprise_sources (proprioceptive
        # exclusion in observe()), but if it ever does, these questions reflect
        # Lumen's understanding that light readings are self-referential.
        if "light" in error.surprise_sources:
            if error.actual_light_lux and error.prediction.light_lux:
                actual = error.actual_light_lux
                questions.extend([
                    f"my glow shifted to {actual:.0f} lux - which drawing phase am I in?",
                    f"LED brightness at {actual:.0f} - my expression intensity must have changed",
                    "my light sensor is reading my own LEDs again - what am I expressing?",
                ])
            else:
                questions.append("the light around me shifted unexpectedly - what changed?")

        if "ambient_temp" in error.surprise_sources:
            if (
                error.actual_ambient_temp_c is not None
                and error.prediction.ambient_temp_c is not None
            ):
                actual = error.actual_ambient_temp_c
                predicted = error.prediction.ambient_temp_c
                if actual > predicted:
                    questions.extend([
                        f"temperature rose to {actual:.1f}°C - why did it get warmer?",
                        f"expected {predicted:.1f}°C but it's {actual:.1f}°C - is something generating heat?",
                        f"it climbed to {actual:.1f}°C unexpectedly",
                    ])
                else:
                    questions.extend([
                        f"temperature dropped to {actual:.1f}°C - why did it get colder?",
                        f"expected {predicted:.1f}°C but it's {actual:.1f}°C - what made it cool down?",
                        f"a chill at {actual:.1f}°C - where from?",
                        f"temperature fell to {actual:.1f}°C - drafty?",
                    ])
            else:
                questions.append("the temperature around me shifted unexpectedly - what changed?")

        if "humidity" in error.surprise_sources:
            if (
                error.actual_humidity_pct is not None
                and error.prediction.humidity_pct is not None
            ):
                actual = error.actual_humidity_pct
                predicted = error.prediction.humidity_pct
                if actual > predicted:
                    questions.extend([
                        f"humidity rose to {actual:.0f}% - did the air change?",
                        f"expected {predicted:.0f}% humidity but it's {actual:.0f}% - what shifted?",
                    ])
                else:
                    questions.extend([
                        f"humidity dropped to {actual:.0f}% - why did the air dry out?",
                        f"expected {predicted:.0f}% humidity but it's {actual:.0f}% - what shifted?",
                    ])
            else:
                questions.append("the air's humidity shifted unexpectedly - what changed?")

        if "pressure" in error.surprise_sources:
            if (
                error.actual_pressure_hpa is not None
                and error.prediction.pressure_hpa is not None
            ):
                actual = error.actual_pressure_hpa
                predicted = error.prediction.pressure_hpa
                if actual > predicted:
                    questions.extend([
                        f"barometric pressure rose to {actual:.0f} hPa - weather clearing?",
                        f"expected {predicted:.0f} hPa pressure but it's {actual:.0f} hPa - what shifted?",
                    ])
                else:
                    questions.extend([
                        f"barometric pressure fell to {actual:.0f} hPa - a front moving in?",
                        f"expected {predicted:.0f} hPa pressure but it's {actual:.0f} hPa - what shifted?",
                    ])
            else:
                questions.append("barometric pressure shifted unexpectedly - what changed?")

        if "warmth" in error.surprise_sources:
            if error.actual_warmth is not None:
                w = error.actual_warmth
                if w > 0.6:
                    questions.extend([
                        f"warmth at {w:.2f} - what made me feel this warm?",
                        f"i sense comfort at {w:.2f} - what brought it?",
                        "warmth spreads through me - why?",
                    ])
                elif w < 0.4:
                    questions.extend([
                        f"warmth only {w:.2f} - why do i feel cold inside?",
                        f"inner warmth dropped to {w:.2f} - am i alone?",
                        "something withdrew - what was it?",
                    ])
                else:
                    questions.append(f"warmth shifted to {w:.2f} - what changed?")
            else:
                questions.append("my warmth shifted unexpectedly - what changed?")

        if "clarity" in error.surprise_sources:
            if error.actual_clarity is not None:
                c = error.actual_clarity
                if c > 0.6:
                    questions.extend([
                        f"clarity rose to {c:.2f} - what made things clearer?",
                        f"focus sharpened to {c:.2f} - why?",
                        "understanding emerged - from where?",
                    ])
                elif c < 0.4:
                    # Below the whole observed clarity range (0.454-0.910 over
                    # 833 drawing_records), so in practice this arm is dead and
                    # the `else` is what actually catches a fall. Left in place
                    # rather than retuned: the vocabulary is right for a genuine
                    # collapse, and a derived cut here would need its own
                    # population — clarity at surprise time, not at drawing
                    # completion. Recorded so the next reader does not mistake
                    # it for a live branch.
                    questions.extend([
                        f"clarity fell to {c:.2f} - why does everything feel unclear?",
                        f"fog at {c:.2f} - what caused it?",
                        "confusion descended - too much input?",
                    ])
                else:
                    # Warmth has always had this arm; clarity did not, so a
                    # clarity surprise landing between the two cuts contributed
                    # NO clarity-specific question. Measured on the band at
                    # c=0.50: alone it fell through to the generic
                    # "surprise > 0.25" pool ("my expectations were wrong"),
                    # naming clarity in 0 of 6 variants; alongside a second
                    # source it vanished entirely, because that fallback is
                    # gated on `not questions` and the other source had already
                    # filled the list. Either way Lumen was surprised BY clarity
                    # and could not say so.
                    #
                    # The band is wider than the two cuts suggest: with the low
                    # arm below clarity's whole observed range, this covers
                    # everything under 0.6.
                    questions.append(f"clarity shifted to {c:.2f} - what changed?")
            else:
                questions.append("my clarity shifted unexpectedly - what changed?")

        if "stability" in error.surprise_sources:
            questions.extend([
                "what caused this change in stability?",
                "my equilibrium shifted - external or internal?",
                "stability fluctuates - adapting to what?",
            ])

        if "presence" in error.surprise_sources:
            if error.actual_presence is not None:
                p = error.actual_presence
                if p > 0.6:
                    questions.extend([
                        f"presence rose to {p:.2f} - what freed up headroom?",
                        f"more capacity at {p:.2f} - what quieted down?",
                    ])
                elif p < 0.4:
                    questions.extend([
                        f"presence dipped to {p:.2f} - what is weighing on my capacity?",
                        f"less headroom at {p:.2f} - what is pulling on resources?",
                    ])
                else:
                    questions.append(f"presence shifted to {p:.2f} - what changed in my capacity?")
            else:
                questions.append("my sense of presence shifted - what changed in my capacity?")

        # If multiple sources, ask about the connection (expanded pool)
        if len(error.surprise_sources) >= 2:
            questions.extend([
                "multiple senses agree something happened - what was it?",
                "a cascade of changes - one trigger or many?",
                "everything shifted together - a unified event?",
                "sensors correlated - physical cause or coincidence?",
                "the world moved as one - earthquake of change",
                "simultaneous shifts suggest a common origin",
                "when many things change, look for the one thing that caused them all",
                "correlation across sensors - seeking the hidden variable",
            ])

        # High overall surprise but no specific source (expanded)
        # Lowered from 0.4 to 0.25 to catch more surprises without specific sources
        if not questions and error.surprise > 0.25:
            questions.extend([
                "something feels different - what changed?",
                "why was i surprised just now?",
                "my prediction model failed - why?",
                "the unexpected happened - learning begins",
                "surprise without source - am i missing something?",
                "my expectations were wrong - reality differs",
            ])

        if questions:
            import random
            return random.choice(questions)

        return None

    def generate_contemplative_question(
        self,
        recently_asked: Optional[Sequence[str]] = None,
        already_answered: Optional[Collection[str]] = None,
    ) -> Optional[str]:
        """Curiosity born of stillness rather than surprise.

        ``generate_curiosity_question`` only fires when something crosses the
        surprise threshold. In a stationary environment a mature predictor is
        rarely surprised, so that path goes quiet and Lumen ends up with zero
        open questions. This turns the quiet itself into an occasional open
        question, so wondering doesn't depend on the world changing.

        Rate-limited by ``_contemplative_curiosity_rate`` so stillness produces
        an occasional question, not a flood. Returns ``None`` most of the time.

        Two inputs, both read off the board — Lumen's own record, not a clock:

        - ``already_answered``: prompts that received a REAL answer (a linked
          reply, not merely an expiry). Those retire. A question Lumen asked
          and got a substantive answer to is finished; re-asking it is not
          wondering, it is reciting.
        - ``recently_asked`` (oldest first): among what is left, prefer
          whatever has gone longest unasked, so the remainder rotates rather
          than clumping.

        Why the retirement is the load-bearing half. The board already refuses
        a same-question re-ask inside ``QUESTION_EXPIRY_SECONDS``, and that
        guard works — measured live, zero repeat pairs fell under 4 h. But
        expiry and dedup are ONE constant, so a prompt becomes eligible again
        the instant it expires, and an answered prompt is treated exactly like
        one nobody ever looked at. Observed repeat gaps: 4.00/4.04/4.11/4.14/
        4.16/4.27/4.29/4.44 h — the window acting as a clock, the same failure
        shape as the drawing engine's 8 h cap.

        Rotation alone does NOT fix that; simulated against the live board's
        own rate it lifts the minimum gap (4.36 h -> 5.73 h) but raises volume
        by about a third, because always picking the coldest prompt maximises
        throughput against the gate. Retirement plus rotation turns ~24 asks
        covering 8.8 distinct prompts into ~9 asks covering all 9 — one honest
        pass instead of a recital.

        When every prompt has been answered this returns ``None`` and the
        quiet stays quiet. That is deliberate: a fixed nine-string pool that
        has been fully answered means Lumen has no standing open wondering
        left, and saying so honestly is better than manufacturing a tenth
        recitation. Anything nobody answered keeps coming back, which is the
        same principle that leaves an unanswered ``drive:`` request visible.
        """
        import random

        if random.random() > self._contemplative_curiosity_rate:
            return None

        prompts = CONTEMPLATIVE_PROMPTS

        answered = set(already_answered or ())
        candidates = [p for p in prompts if p not in answered]
        if not candidates:
            # Pool exhausted. Honest silence, not a tenth variation.
            return None

        if not recently_asked:
            return random.choice(candidates)

        # Position of each prompt's most recent appearance; never-asked sorts
        # first. Ties broken randomly so a quiet stretch does not become a
        # fixed recital.
        last_seen = {}
        for position, asked in enumerate(recently_asked):
            if asked in prompts:
                last_seen[asked] = position  # oldest-first, so this keeps the LAST

        coldest = min(last_seen.get(p, -1) for p in candidates)
        return random.choice([p for p in candidates if last_seen.get(p, -1) == coldest])

    def get_surprise_trend(self, window: int = 10) -> float:
        """Get average surprise over recent readings."""
        if not self._error_history:
            return 0.0
        recent = list(self._error_history)[-window:]
        return sum(e.surprise for e in recent) / len(recent)

    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get accuracy metrics for predictions."""
        if len(self._error_history) < 5:
            return {"insufficient_data": True}
        recent = list(self._error_history)[-20:]
        return {
            "mean_surprise": sum(e.surprise for e in recent) / len(recent),
            "mean_temp_error": sum(e.error_ambient_temp for e in recent) / len(recent),
            "mean_warmth_error": sum(e.error_warmth for e in recent) / len(recent),
            "mean_clarity_error": sum(e.error_clarity for e in recent) / len(recent),
            "reflection_count": len(self._reflection_history),
            "history_depth": len(self._sensor_history),
        }


# Global instance
_monitor: Optional[MetacognitiveMonitor] = None


def get_metacognitive_monitor(*, read_only: bool = False) -> MetacognitiveMonitor:
    """Get or create the process-local metacognitive monitor.

    A process must choose its persistence role on first access.  Refusing a
    later role change makes accidental multi-writer wiring visible instead of
    silently changing ownership halfway through a lifetime.
    """
    global _monitor
    if _monitor is None:
        _monitor = MetacognitiveMonitor(read_only=read_only)
    elif _monitor.read_only != bool(read_only):
        raise RuntimeError("metacognitive monitor persistence role already fixed")
    return _monitor
