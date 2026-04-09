"""
Adaptive Prediction - Learns temporal patterns to improve predictions.

Unlike the simple EMA in metacognition, this actually learns from prediction errors.
If Lumen keeps getting surprised by the same pattern (e.g., lights dimming at 6pm),
it should stop being surprised.

Key insight: Learning happens when predictions fail, not when they succeed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict, deque
import math
import json
from pathlib import Path

from .atomic_write import atomic_json_write


@dataclass
class PatternFeatures:
    """Features extracted for pattern matching."""
    hour: int  # 0-23
    minute_bucket: int  # 0-5 (10-min buckets)
    day_of_week: int  # 0-6
    is_weekend: bool

    # Recent history (last 3 values)
    recent_trend: float  # -1 to 1, direction of change
    recent_variance: float  # 0-1, how much recent values varied

    # Contextual
    light_level: str  # "dark", "dim", "bright", "very_bright"
    temp_zone: str  # "cold", "cool", "comfortable", "warm", "hot"

    def to_key(self) -> str:
        """Convert to hashable key for pattern lookup."""
        return f"{self.hour}:{self.minute_bucket}:{self.day_of_week}:{self.light_level}:{self.temp_zone}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hour": self.hour,
            "minute_bucket": self.minute_bucket,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "recent_trend": self.recent_trend,
            "recent_variance": self.recent_variance,
            "light_level": self.light_level,
            "temp_zone": self.temp_zone,
        }


@dataclass
class LearnedPattern:
    """A learned pattern for predicting a specific variable."""
    pattern_key: str
    variable: str  # "light", "temp", "humidity", "warmth", "clarity", "stability"

    # Learned statistics
    mean: float = 0.0
    variance: float = 0.0
    sample_count: int = 0

    # Confidence based on consistency
    confidence: float = 0.0

    # When this pattern typically occurs
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def update(self, value: float, learning_rate: float = 0.1):
        """Update pattern with new observation."""
        self.sample_count += 1
        now = datetime.now()

        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now

        # Incremental mean and variance (Welford's algorithm)
        # Guard against numerical instability with epsilon
        delta = value - self.mean
        self.mean += delta / max(self.sample_count, 1)
        delta2 = value - self.mean

        if self.sample_count > 1:
            # Welford's online variance: ensure non-negative
            new_variance = ((self.sample_count - 1) * self.variance + delta * delta2) / self.sample_count
            self.variance = max(0.0, new_variance)  # Clamp to prevent negative from float errors
        else:
            self.variance = 0.0

        # Confidence increases with samples, decreases with variance
        base_confidence = min(1.0, self.sample_count / 10)
        variance_penalty = min(0.5, max(0.0, self.variance) / 2)
        self.confidence = base_confidence * (1 - variance_penalty)


class AdaptivePredictionModel:
    """
    Learns patterns from experience to make better predictions.

    The key differences from the existing metacognition:
    1. Predictions improve from errors (not just EMA smoothing)
    2. Learns time-of-day patterns
    3. Learns context-dependent patterns (e.g., "after it gets dark, temperature drops")
    4. Tracks confidence and reduces surprise for known patterns
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or Path.home() / ".anima" / "patterns.json"

        # Learned patterns: variable -> pattern_key -> LearnedPattern
        self._patterns: Dict[str, Dict[str, LearnedPattern]] = defaultdict(dict)

        # Recent history for feature extraction (deque for O(1) append/pop)
        self._history: deque = deque(maxlen=50)

        # Prediction accuracy tracking (deque for O(1) append/pop)
        self._recent_errors: deque = deque(maxlen=100)  # (variable, error)

        # Load persisted patterns
        self._load_patterns()

    def _load_patterns(self):
        """Load learned patterns from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    for variable, patterns in data.get("patterns", {}).items():
                        for key, p in patterns.items():
                            self._patterns[variable][key] = LearnedPattern(
                                pattern_key=key,
                                variable=variable,
                                mean=p.get("mean", 0),
                                variance=p.get("variance", 0),
                                sample_count=p.get("sample_count", 0),
                                confidence=p.get("confidence", 0),
                            )
            except Exception as e:
                print(f"[AdaptivePrediction] Could not load patterns: {e}")

    def _prune_patterns(self, max_total: int = 5000):
        """Evict low-value patterns when dict grows too large."""
        total = sum(len(p) for p in self._patterns.values())
        if total <= max_total:
            return

        all_patterns = [
            (var, key, pattern)
            for var, patterns in self._patterns.items()
            for key, pattern in patterns.items()
        ]
        # Evict weakest: lowest sample count, then lowest confidence
        all_patterns.sort(key=lambda x: (x[2].sample_count, x[2].confidence))

        to_remove = total - max_total
        for var, key, _ in all_patterns[:to_remove]:
            del self._patterns[var][key]

    def _save_patterns(self):
        """Save learned patterns to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "patterns": {
                    variable: {
                        key: {
                            "mean": p.mean,
                            "variance": p.variance,
                            "sample_count": p.sample_count,
                            "confidence": p.confidence,
                        }
                        for key, p in patterns.items()
                    }
                    for variable, patterns in self._patterns.items()
                },
                "last_saved": datetime.now().isoformat(),
            }
            atomic_json_write(self.persistence_path, data, indent=2)
        except Exception as e:
            print(f"[AdaptivePrediction] Could not save patterns: {e}")

    def _extract_features(
        self,
        current_time: datetime,
        recent_values: Optional[Dict[str, List[float]]] = None,
        current_light: Optional[float] = None,
        current_temp: Optional[float] = None,
    ) -> PatternFeatures:
        """Extract features for pattern matching."""
        # Time features
        hour = current_time.hour
        minute_bucket = current_time.minute // 10
        day_of_week = current_time.weekday()
        is_weekend = day_of_week >= 5

        # Recent trend and variance
        recent_trend = 0.0
        recent_variance = 0.0
        if recent_values:
            # Use first available variable for trend
            for values in recent_values.values():
                if len(values) >= 2:
                    recent_trend = (values[-1] - values[0]) / max(abs(values[0]), 1)
                    recent_trend = max(-1, min(1, recent_trend))
                    if len(values) >= 3:
                        mean = sum(values) / len(values)
                        recent_variance = sum((v - mean) ** 2 for v in values) / len(values)
                        recent_variance = min(1.0, recent_variance)
                    break

        # Light level categorization
        if current_light is not None:
            if current_light < 10:
                light_level = "dark"
            elif current_light < 100:
                light_level = "dim"
            elif current_light < 1000:
                light_level = "bright"
            else:
                light_level = "very_bright"
        else:
            light_level = "unknown"

        # Temperature zone
        if current_temp is not None:
            if current_temp < 15:
                temp_zone = "cold"
            elif current_temp < 20:
                temp_zone = "cool"
            elif current_temp < 25:
                temp_zone = "comfortable"
            elif current_temp < 30:
                temp_zone = "warm"
            else:
                temp_zone = "hot"
        else:
            temp_zone = "unknown"

        return PatternFeatures(
            hour=hour,
            minute_bucket=minute_bucket,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            recent_trend=recent_trend,
            recent_variance=recent_variance,
            light_level=light_level,
            temp_zone=temp_zone,
        )

    def predict(
        self,
        variable: str,
        current_time: Optional[datetime] = None,
        recent_values: Optional[List[float]] = None,
        current_light: Optional[float] = None,
        current_temp: Optional[float] = None,
        fallback: Optional[float] = None,
    ) -> Tuple[Optional[float], float]:
        """
        Predict a value using learned patterns.

        Returns:
            (predicted_value, confidence)
        """
        if current_time is None:
            current_time = datetime.now()

        features = self._extract_features(
            current_time,
            {variable: recent_values} if recent_values else None,
            current_light,
            current_temp,
        )

        # Look for matching pattern
        pattern_key = features.to_key()

        if variable in self._patterns and pattern_key in self._patterns[variable]:
            pattern = self._patterns[variable][pattern_key]
            if pattern.sample_count >= 3:  # Need minimum samples
                return pattern.mean, pattern.confidence

        # Try less specific patterns (just hour + day_type)
        if variable in self._patterns:
            for key, pattern in self._patterns[variable].items():
                if key.startswith(f"{features.hour}:") and pattern.sample_count >= 5:
                    # Weight by similarity
                    return pattern.mean, pattern.confidence * 0.7

        # Fallback: use recent values if available
        if recent_values and len(recent_values) >= 1:
            # Simple trend extrapolation
            if len(recent_values) >= 2:
                trend = recent_values[-1] - recent_values[-2]
                return recent_values[-1] + trend * 0.3, 0.3
            return recent_values[-1], 0.4

        # Final fallback
        if fallback is not None:
            return fallback, 0.1

        return None, 0.0

    def observe(
        self,
        observations: Dict[str, float],
        current_time: Optional[datetime] = None,
        current_light: Optional[float] = None,
        current_temp: Optional[float] = None,
    ):
        """
        Observe actual values and learn from them.

        This is where learning happens. Each observation updates the patterns
        for the current context.
        """
        if current_time is None:
            current_time = datetime.now()

        # Get recent values for feature extraction
        recent_values = {}
        for variable in observations:
            recent = list(self._history)[-5:]  # deque doesn't support slicing
            values = [h.get(variable) for h in recent if variable in h]
            values = [v for v in values if v is not None]
            if values:
                recent_values[variable] = values

        features = self._extract_features(
            current_time,
            recent_values,
            current_light or observations.get("light"),
            current_temp or observations.get("ambient_temp"),
        )

        pattern_key = features.to_key()

        # Update patterns for each observed variable
        for variable, value in observations.items():
            if value is None:
                continue

            if pattern_key not in self._patterns[variable]:
                self._patterns[variable][pattern_key] = LearnedPattern(
                    pattern_key=pattern_key,
                    variable=variable,
                )

            self._patterns[variable][pattern_key].update(value)

        # Store in history (deque auto-manages size)
        history_entry = {**observations, "timestamp": current_time.isoformat()}
        self._history.append(history_entry)

        # Periodically prune and save patterns
        if len(self._history) % 10 == 0:
            self._prune_patterns()
            self._save_patterns()

    def record_prediction_error(self, variable: str, predicted: float, actual: float):
        """Record prediction error for tracking accuracy improvement.

        Normalizes error by variable scale so light (0-1000 lux) and warmth (0-1)
        are comparable in aggregate stats.
        """
        error = abs(predicted - actual)
        # Normalize to 0-1 scale (same logic as get_surprising_deviation)
        if variable in ("warmth", "clarity", "stability", "presence"):
            normalized = error  # Already 0-1
        elif variable == "light":
            if predicted > 0 and actual > 0:
                normalized = abs(math.log10(actual) - math.log10(predicted)) / 3.0
            else:
                normalized = 1.0
        elif "temp" in variable:
            normalized = error / 10.0  # 10°C range
        elif "humid" in variable:
            normalized = error / 30.0  # 30% range
        else:
            normalized = error
        self._recent_errors.append((variable, min(1.0, normalized)))

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get statistics on prediction accuracy."""
        if not self._recent_errors:
            return {"insufficient_data": True}

        by_variable = defaultdict(list)
        for variable, error in self._recent_errors:
            by_variable[variable].append(error)

        stats = {
            "total_errors": len(self._recent_errors),
            "overall_mean_error": sum(e for _, e in self._recent_errors) / len(self._recent_errors),
        }

        for variable, errors in by_variable.items():
            stats[f"{variable}_mean_error"] = sum(errors) / len(errors)
            stats[f"{variable}_sample_count"] = len(errors)

        # Count learned patterns
        total_patterns = sum(len(patterns) for patterns in self._patterns.values())
        high_confidence_patterns = sum(
            1 for patterns in self._patterns.values()
            for p in patterns.values()
            if p.confidence > 0.6
        )

        stats["total_patterns"] = total_patterns
        stats["high_confidence_patterns"] = high_confidence_patterns

        return stats

    def get_surprising_deviation(
        self,
        variable: str,
        actual: float,
        current_time: Optional[datetime] = None,
        current_light: Optional[float] = None,
        current_temp: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """
        Determine how surprising a value is given learned patterns.

        Returns:
            (deviation from expected, is_actually_surprising)

        Key insight: If we've learned this pattern, high deviation shouldn't
        be surprising. If we haven't, even small deviations are surprising.
        """
        predicted, confidence = self.predict(
            variable, current_time, None, current_light, current_temp
        )

        if predicted is None:
            # No prediction available - any change is potentially surprising
            return 0.5, True

        deviation = abs(actual - predicted)

        # Normalize deviation based on variable type
        if variable in ("warmth", "clarity", "stability", "presence"):
            normalized_deviation = deviation  # Already 0-1 scale
        elif variable == "light":
            # Log scale for light — 3 decades (1→1000 lux) = full range
            if predicted > 0 and actual > 0:
                normalized_deviation = abs(math.log10(actual) - math.log10(predicted)) / 3.0
            else:
                normalized_deviation = 1.0
        elif variable in ("ambient_temp", "temperature"):
            normalized_deviation = deviation / 10  # 10°C range
        elif variable == "humidity":
            normalized_deviation = deviation / 30  # 30% range
        else:
            normalized_deviation = deviation

        normalized_deviation = min(1.0, normalized_deviation)

        # Key logic: High confidence patterns reduce surprise
        # If we've seen this pattern many times with low variance,
        # deviations are expected within the learned variance
        if confidence > 0.5:
            # Check if deviation is within learned variance
            pattern_key = self._extract_features(
                current_time or datetime.now(),
                None, current_light, current_temp
            ).to_key()

            if variable in self._patterns and pattern_key in self._patterns[variable]:
                pattern = self._patterns[variable][pattern_key]
                std_dev = math.sqrt(pattern.variance) if pattern.variance > 0 else 0.1

                # Within 2 standard deviations is not surprising
                if deviation < 2 * std_dev:
                    return normalized_deviation, False

        # Otherwise, surprisingness depends on deviation
        return normalized_deviation, normalized_deviation > 0.2


# Singleton instance
_adaptive_model: Optional[AdaptivePredictionModel] = None


def get_adaptive_prediction_model() -> AdaptivePredictionModel:
    """Get or create the adaptive prediction model."""
    global _adaptive_model
    if _adaptive_model is None:
        _adaptive_model = AdaptivePredictionModel()
    return _adaptive_model
