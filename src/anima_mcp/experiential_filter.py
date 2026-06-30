"""
Experiential Filter — Layer 2 of Experiential Accumulation.

Salience weights per sensor dimension that drift based on experience.
Dimensions that consistently surprise or dissatisfy the creature become
more salient — Lumen pays more attention to them.

This is not cognitive. It is pre-attentional filtering: the creature's
nervous system learns which signals matter most.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .atomic_write import atomic_json_write


# Sensor dimensions tracked
DIMENSIONS = [
    "ambient_temp", "cpu_temp", "light",
    "humidity", "pressure", "memory", "cpu",
    # Acoustic channel (hearing wire, Stage 1) — salience only, no content.
    # Surprise on these is fed DIRECTLY from hearing_ingest, never through
    # metacognition's aggregate (so it never reaches the punishment path).
    "sound_level", "voice_activity",
]

# Map surprise source names to sensor dimensions
SOURCE_TO_DIM: Dict[str, str] = {
    "light": "light",
    "light_lux": "light",
    "ambient_temp": "ambient_temp",
    "temperature": "ambient_temp",
    "humidity": "humidity",
    "humidity_pct": "humidity",
    "pressure": "pressure",
    "cpu_temp": "cpu_temp",
    "memory": "memory",
    "cpu": "cpu",
    # Acoustic channel
    "sound_level": "sound_level",
    "voice_activity": "voice_activity",
}

# Map preference dimensions to sensor dimensions
PREF_TO_SENSOR: Dict[str, str] = {
    "warmth": "ambient_temp",
    "clarity": "light",
    "stability": "humidity",
    "presence": "cpu",
}

# Bounds
SALIENCE_MAX = 2.0
SALIENCE_MIN = 0.5

# Save interval (seconds)
SAVE_INTERVAL = 120


@dataclass
class SalienceWeight:
    """Salience weight for a single sensor dimension."""

    dimension: str
    weight: float = 1.0
    surprise_accumulator: float = 0.0
    dissatisfaction_ticks: int = 0

    def amplify_from_surprise(self, surprise: float) -> None:
        """Increase salience from surprise on this dimension."""
        self.weight += 0.02 * surprise
        self.weight = min(SALIENCE_MAX, self.weight)

    def amplify_from_dissatisfaction(self) -> None:
        """Slight salience increase from ongoing dissatisfaction."""
        self.dissatisfaction_ticks += 1
        self.weight += 0.001
        self.weight = min(SALIENCE_MAX, self.weight)

    def decay_toward_neutral(self) -> None:
        """Decay weight toward 1.0 (neutral)."""
        if self.weight > 1.0:
            excess = self.weight - 1.0
            excess *= 0.9995
            self.weight = 1.0 + excess
        elif self.weight < 1.0:
            deficit = 1.0 - self.weight
            deficit *= 0.9995
            self.weight = 1.0 - deficit
        # Enforce minimum bound
        self.weight = max(SALIENCE_MIN, self.weight)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "weight": self.weight,
            "surprise_accumulator": self.surprise_accumulator,
            "dissatisfaction_ticks": self.dissatisfaction_ticks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SalienceWeight":
        return cls(
            dimension=data["dimension"],
            weight=data.get("weight", 1.0),
            surprise_accumulator=data.get("surprise_accumulator", 0.0),
            dissatisfaction_ticks=data.get("dissatisfaction_ticks", 0),
        )


class ExperientialFilter:
    """
    Pre-attentional salience filter for sensor dimensions.

    Dimensions that consistently surprise or dissatisfy the creature
    gain salience — their signals are amplified in anima computation.
    Over time, salience decays back to neutral.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        if persistence_path is None:
            self._path = Path.home() / ".anima" / "experiential_filter.json"
        else:
            self._path = Path(persistence_path)

        self._weights: Dict[str, SalienceWeight] = {}
        self._last_save_time: float = time.time()

        # Try to load from disk
        self._load()

        # Ensure all dimensions exist
        for dim in DIMENSIONS:
            if dim not in self._weights:
                self._weights[dim] = SalienceWeight(dimension=dim)

    def _load(self) -> None:
        """Load salience weights from disk."""
        try:
            if self._path.exists():
                with open(self._path) as f:
                    data = json.load(f)
                for item in data.get("weights", []):
                    sw = SalienceWeight.from_dict(item)
                    self._weights[sw.dimension] = sw
        except (json.JSONDecodeError, OSError, KeyError):
            # Corrupt or missing file — start fresh
            pass

    def save(self) -> None:
        """Explicitly save to disk."""
        data = {
            "weights": [w.to_dict() for w in self._weights.values()],
            "saved_at": time.time(),
        }
        atomic_json_write(self._path, data, indent=2)
        self._last_save_time = time.time()

    def _maybe_save(self) -> None:
        """Save if enough time has passed since last save."""
        if time.time() - self._last_save_time >= SAVE_INTERVAL:
            self.save()

    def get_salience(self, dimension: str) -> float:
        """Get salience weight for a dimension. 1.0 = neutral."""
        if dimension in self._weights:
            return self._weights[dimension].weight
        return 1.0

    def get_all_saliences(self) -> Dict[str, float]:
        """Get all salience weights as a dict."""
        return {dim: w.weight for dim, w in self._weights.items()}

    def update_from_surprise(
        self,
        surprise_sources: List[str],
        surprise_level: float,
        temp_dampening: float = 0.0,
    ) -> None:
        """
        Amplify salience for dimensions that caused surprise.

        Args:
            surprise_sources: Source names (e.g. "light", "humidity_pct")
            surprise_level: How surprising (0-1 scale)
            temp_dampening: from experiential marks (temp_salience_dampening),
                reduces temperature surprise amplification.
        """
        for source in surprise_sources:
            dim = SOURCE_TO_DIM.get(source)
            if dim and dim in self._weights:
                effective = surprise_level
                if dim in ("ambient_temp", "cpu_temp") and temp_dampening > 0:
                    effective *= (1.0 - temp_dampening)
                self._weights[dim].amplify_from_surprise(effective)

    def update_from_dissatisfaction(self, most_unsatisfied: str) -> None:
        """
        Amplify salience for the sensor dimension linked to dissatisfaction.

        Args:
            most_unsatisfied: Preference dimension name (e.g. "warmth", "clarity")
        """
        dim = PREF_TO_SENSOR.get(most_unsatisfied)
        if dim and dim in self._weights:
            self._weights[dim].amplify_from_dissatisfaction()

    def tick(self) -> None:
        """Decay all weights toward neutral and maybe save."""
        for w in self._weights.values():
            w.decay_toward_neutral()
        self._maybe_save()

    def get_stats(self) -> dict:
        """Summary statistics for the filter."""
        saliences = self.get_all_saliences()
        biased = {
            dim: weight
            for dim, weight in saliences.items()
            if abs(weight - 1.0) > 0.01
        }
        return {
            "dimensions": len(self._weights),
            "biased_count": len(biased),
            "biased_dimensions": biased,
            "mean_salience": (
                sum(saliences.values()) / len(saliences) if saliences else 1.0
            ),
        }


# Singleton
_instance: Optional[ExperientialFilter] = None


def get_experiential_filter(
    persistence_path: Optional[str] = None,
) -> ExperientialFilter:
    """Get (or create) the singleton ExperientialFilter."""
    global _instance
    if _instance is None:
        _instance = ExperientialFilter(persistence_path=persistence_path)
    return _instance
