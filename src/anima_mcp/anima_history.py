"""
Anima History - Track anima state over time for trajectory computation.

This module enables computing attractor basins and other trajectory invariants
by maintaining a time-series of anima state observations.

Part of the Trajectory Identity framework.
See: trajectory-identity paper (cirwel/trajectory-identity-paper, separate repo)
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sys

from .atomic_write import atomic_json_write

# Numpy is optional - graceful fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class DaySummary:
    """Consolidated summary of one active period."""
    date: str                              # ISO date string
    attractor_center: List[float]          # [warmth, clarity, stability, presence]
    attractor_variance: List[float]        # variance per dimension
    n_observations: int
    time_span_hours: float
    notable_perturbations: int             # count of perturbations detected
    dimension_trends: Dict[str, float]     # per-dim mean for this period

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "center": self.attractor_center,
            "variance": self.attractor_variance,
            "n_obs": self.n_observations,
            "hours": round(self.time_span_hours, 2),
            "perturbations": self.notable_perturbations,
            "trends": self.dimension_trends,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DaySummary':
        return cls(
            date=data["date"],
            attractor_center=data["center"],
            attractor_variance=data["variance"],
            n_observations=data["n_obs"],
            time_span_hours=data["hours"],
            notable_perturbations=data["perturbations"],
            dimension_trends=data["trends"],
        )


@dataclass
class AnimaSnapshot:
    """A single anima state observation."""
    timestamp: datetime
    warmth: float
    clarity: float
    stability: float
    presence: float

    def to_vector(self) -> List[float]:
        """Convert to list (or numpy array if available)."""
        return [self.warmth, self.clarity, self.stability, self.presence]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "t": self.timestamp.isoformat(),
            "w": round(self.warmth, 4),
            "c": round(self.clarity, 4),
            "s": round(self.stability, 4),
            "p": round(self.presence, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnimaSnapshot':
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["t"]),
            warmth=data["w"],
            clarity=data["c"],
            stability=data["s"],
            presence=data["p"],
        )


class AnimaHistory:
    """
    Track anima state history for trajectory computation.

    Implements a sliding window of observations with periodic persistence.
    This is the foundation for computing attractor basins and other
    trajectory invariants.

    Usage:
        history = get_anima_history()
        history.record(warmth=0.5, clarity=0.6, stability=0.7, presence=0.8)
        basin = history.get_attractor_basin(window=100)
    """

    def __init__(
        self,
        max_size: int = 2000,  # ~30 min at 1Hz
        persistence_path: Optional[Path] = None,
        auto_save_interval: int = 100,  # Save every N records
    ):
        self.max_size = max_size
        self.persistence_path = persistence_path or Path.home() / ".anima" / "anima_history.json"
        self.auto_save_interval = auto_save_interval
        self._history: deque = deque(maxlen=max_size)
        self._records_since_save = 0
        self._load()

    def record(
        self,
        warmth: float,
        clarity: float,
        stability: float,
        presence: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a new anima state observation.

        Args:
            warmth: Warmth dimension [0, 1]
            clarity: Clarity dimension [0, 1]
            stability: Stability dimension [0, 1]
            presence: Presence dimension [0, 1]
            timestamp: Optional timestamp (defaults to now)
        """
        self._history.append(AnimaSnapshot(
            timestamp=timestamp or datetime.now(),
            warmth=warmth,
            clarity=clarity,
            stability=stability,
            presence=presence,
        ))

        self._records_since_save += 1
        if self._records_since_save >= self.auto_save_interval:
            self._save()
            self._records_since_save = 0

    def record_from_anima(self, anima) -> None:
        """
        Record from an AnimaState object.

        Args:
            anima: AnimaState with warmth, clarity, stability, presence
        """
        self.record(
            warmth=getattr(anima, 'warmth', 0.5),
            clarity=getattr(anima, 'clarity', 0.5),
            stability=getattr(anima, 'stability', 0.5),
            presence=getattr(anima, 'presence', 0.5),
        )

    def get_attractor_basin(self, window: int = 100) -> Optional[Dict[str, Any]]:
        """
        Compute attractor basin from recent history.

        The attractor basin characterizes where the agent "lives" in state space:
        - center (μ): The equilibrium point the agent returns to
        - covariance (Σ): The shape of the region the agent occupies
        - eigenvalues: Principal axes of variability

        Args:
            window: Number of recent observations to use

        Returns:
            Dictionary with center, covariance, and metadata, or None if insufficient data
        """
        if len(self._history) < 10:
            return None

        recent = list(self._history)[-window:]

        if HAS_NUMPY:
            matrix = np.array([s.to_vector() for s in recent])
            center = np.mean(matrix, axis=0)
            covariance = np.cov(matrix.T)

            # Handle edge case of constant values
            if np.any(np.isnan(covariance)):
                covariance = np.eye(4) * 0.001

            # Regularization: add epsilon to diagonal to prevent singularity
            # This ensures det(covariance) > 0 for Bhattacharyya computation
            epsilon = 1e-6
            covariance = covariance + np.eye(4) * epsilon

            # Compute eigenvalues for principal component analysis
            try:
                eigenvalues = np.linalg.eigvalsh(covariance)
            except np.linalg.LinAlgError:
                eigenvalues = [0.001] * 4

            return {
                "center": center.tolist(),
                "covariance": covariance.tolist(),
                "eigenvalues": sorted(eigenvalues.tolist(), reverse=True),
                "n_observations": len(recent),
                "time_span_seconds": (recent[-1].timestamp - recent[0].timestamp).total_seconds(),
                "dimensions": ["warmth", "clarity", "stability", "presence"],
            }
        else:
            # Fallback without numpy - basic statistics only
            n = len(recent)
            center = [
                sum(s.warmth for s in recent) / n,
                sum(s.clarity for s in recent) / n,
                sum(s.stability for s in recent) / n,
                sum(s.presence for s in recent) / n,
            ]

            # Compute variance (diagonal of covariance) only
            # Add epsilon regularization for consistency with numpy path
            epsilon = 1e-6
            variance = [
                sum((s.warmth - center[0])**2 for s in recent) / n + epsilon,
                sum((s.clarity - center[1])**2 for s in recent) / n + epsilon,
                sum((s.stability - center[2])**2 for s in recent) / n + epsilon,
                sum((s.presence - center[3])**2 for s in recent) / n + epsilon,
            ]

            return {
                "center": center,
                "variance": variance,  # Only variance, not full covariance
                "n_observations": len(recent),
                "time_span_seconds": (recent[-1].timestamp - recent[0].timestamp).total_seconds(),
                "dimensions": ["warmth", "clarity", "stability", "presence"],
                "_note": "Full covariance requires numpy",
            }

    def get_recent_trajectory(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get the most recent N observations as a trajectory.

        Useful for visualization and debugging.

        Args:
            n: Number of recent observations to return

        Returns:
            List of observation dictionaries
        """
        recent = list(self._history)[-n:]
        return [s.to_dict() for s in recent]

    def get_dimension_stats(self, dimension: str, window: int = 100) -> Optional[Dict[str, float]]:
        """
        Get statistics for a single dimension.

        Args:
            dimension: One of 'warmth', 'clarity', 'stability', 'presence'
            window: Number of recent observations to use

        Returns:
            Dictionary with mean, std, min, max, or None if insufficient data
        """
        if len(self._history) < 5:
            return None

        recent = list(self._history)[-window:]
        values = [getattr(s, dimension) for s in recent]

        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val)**2 for v in values) / len(values)
        std_val = variance ** 0.5

        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
        }

    def detect_perturbation(self, threshold: float = 0.15) -> Optional[Dict[str, Any]]:
        """
        Detect if a recent perturbation occurred.

        A perturbation is a sudden change in state that moves the agent
        away from its attractor center.

        Args:
            threshold: Minimum distance from center to count as perturbation

        Returns:
            Dictionary with perturbation info, or None if no perturbation detected
        """
        if len(self._history) < 20:
            return None

        basin = self.get_attractor_basin(window=50)
        if not basin:
            return None

        center = basin["center"]
        recent = list(self._history)[-5:]

        for snapshot in recent:
            current = snapshot.to_vector()
            # Euclidean distance from center
            distance = sum((c - v)**2 for c, v in zip(center, current)) ** 0.5

            if distance > threshold:
                return {
                    "detected": True,
                    "distance": round(distance, 4),
                    "timestamp": snapshot.timestamp.isoformat(),
                    "state": snapshot.to_dict(),
                    "center": center,
                }

        return {"detected": False, "distance": 0.0}

    def compute_void_integral(self, window: int = 100) -> Optional[Dict[str, Any]]:
        """
        Compute the Anima Void Integral V_anima(t).

        From paper Section 4.1:
        V_anima(t) = ∫ ||a(τ) - μ_a|| dτ

        This is the cumulative deviation from equilibrium - a governance
        trigger for UNITARES to check on agent wellbeing.

        Args:
            window: Number of recent observations to integrate over

        Returns:
            Dictionary with void integral value and metadata
        """
        if len(self._history) < 20:
            return None

        basin = self.get_attractor_basin(window=window)
        if not basin:
            return None

        center = basin["center"]
        recent = list(self._history)[-window:]

        # Compute integral as sum of distances (discrete approximation)
        total_deviation = 0.0
        deviations = []

        for i, snapshot in enumerate(recent):
            current = snapshot.to_vector()
            distance = sum((c - v)**2 for c, v in zip(center, current)) ** 0.5
            deviations.append(distance)
            total_deviation += distance

        # Time span for rate calculation
        if len(recent) >= 2:
            time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
            if time_span > 0:
                rate = total_deviation / time_span
            else:
                rate = 0.0
        else:
            time_span = 0.0
            rate = 0.0

        # Average deviation (normalized void)
        avg_deviation = total_deviation / len(recent) if recent else 0.0

        return {
            "void_integral": round(total_deviation, 4),
            "avg_deviation": round(avg_deviation, 4),
            "rate": round(rate, 6),  # Deviation per second
            "max_deviation": round(max(deviations), 4) if deviations else 0.0,
            "n_observations": len(recent),
            "time_span_seconds": round(time_span, 2),
            "center": [round(c, 4) for c in center],
        }

    # === Memory Consolidation ===

    def consolidate(self) -> Optional[DaySummary]:
        """
        Consolidate current buffer into a DaySummary.

        Compresses the rolling buffer into a single summary that captures
        the essential character of this active period. Requires ≥100
        observations to produce a meaningful summary.

        Returns:
            DaySummary if enough data, None otherwise
        """
        if len(self._history) < 100:
            return None

        observations = list(self._history)
        n = len(observations)

        # Compute center (mean per dimension)
        center = [
            sum(s.warmth for s in observations) / n,
            sum(s.clarity for s in observations) / n,
            sum(s.stability for s in observations) / n,
            sum(s.presence for s in observations) / n,
        ]

        # Compute variance per dimension
        variance = [
            sum((s.warmth - center[0])**2 for s in observations) / n,
            sum((s.clarity - center[1])**2 for s in observations) / n,
            sum((s.stability - center[2])**2 for s in observations) / n,
            sum((s.presence - center[3])**2 for s in observations) / n,
        ]

        # Time span
        time_span_hours = (
            observations[-1].timestamp - observations[0].timestamp
        ).total_seconds() / 3600.0

        # Count perturbations (distance from center > 0.15)
        perturbation_count = 0
        for s in observations:
            dist = sum((c - v)**2 for c, v in zip(
                center, s.to_vector()
            )) ** 0.5
            if dist > 0.15:
                perturbation_count += 1

        # Dimension trends (just the means, labeled)
        dim_names = ["warmth", "clarity", "stability", "presence"]
        trends = {name: round(center[i], 4) for i, name in enumerate(dim_names)}

        summary = DaySummary(
            date=datetime.now().isoformat(),
            attractor_center=[round(c, 4) for c in center],
            attractor_variance=[round(v, 6) for v in variance],
            n_observations=n,
            time_span_hours=time_span_hours,
            notable_perturbations=perturbation_count,
            dimension_trends=trends,
        )

        # Persist to day summaries file
        self._save_day_summary(summary)

        return summary

    def get_day_summaries(self, limit: int = 30) -> List[DaySummary]:
        """
        Load persisted day summaries.

        Args:
            limit: Maximum number of summaries to return (most recent first)

        Returns:
            List of DaySummary objects, newest first
        """
        summaries_path = self._get_summaries_path()
        if not summaries_path.exists():
            return []

        try:
            with open(summaries_path, 'r') as f:
                data = json.load(f)
            summaries = [DaySummary.from_dict(d) for d in data.get("summaries", [])]
            # Return newest first, limited
            return list(reversed(summaries[-limit:]))
        except Exception as e:
            print(f"[AnimaHistory] Could not load day summaries: {e}", file=sys.stderr)
            return []

    def detect_long_term_trend(
        self, dimension: str, window_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Detect long-term trend in a dimension across day summaries.

        Uses simple linear regression over day summary centers to find
        whether a dimension is trending up, down, or stable.

        Args:
            dimension: One of 'warmth', 'clarity', 'stability', 'presence'
            window_days: Number of recent summaries to analyze

        Returns:
            Dict with trend info, or None if insufficient data (<3 summaries)
        """
        summaries = self.get_day_summaries(limit=window_days)

        if len(summaries) < 3:
            return None

        dim_idx = ["warmth", "clarity", "stability", "presence"].index(dimension)

        # Extract values (summaries are newest-first, reverse for chronological)
        values = [s.attractor_center[dim_idx] for s in reversed(summaries)]
        n = len(values)

        # Simple linear regression: y = mx + b
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean)**2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Determine direction
        if abs(slope) < 0.005:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "dimension": dimension,
            "trend": round(slope, 6),
            "direction": direction,
            "n_summaries": n,
            "recent_value": round(values[-1], 4),
            "oldest_value": round(values[0], 4),
        }

    def _get_summaries_path(self) -> Path:
        """Get path for day summaries persistence."""
        return self.persistence_path.parent / "day_summaries.json"

    def _save_day_summary(self, summary: DaySummary):
        """Append a day summary to persistent storage, keeping max 30."""
        summaries_path = self._get_summaries_path()

        existing = []
        if summaries_path.exists():
            try:
                with open(summaries_path, 'r') as f:
                    data = json.load(f)
                existing = data.get("summaries", [])
            except Exception:
                existing = []

        existing.append(summary.to_dict())

        # Keep only last 30
        existing = existing[-30:]

        try:
            atomic_json_write(summaries_path, {"summaries": existing, "version": "1.0"})
        except Exception as e:
            print(f"[AnimaHistory] Could not save day summary: {e}", file=sys.stderr)

    def __len__(self) -> int:
        return len(self._history)

    def _save(self):
        """Persist history to disk."""
        try:
            # Only save last 500 for disk efficiency
            recent = list(self._history)[-500:]
            data = {
                "observations": [s.to_dict() for s in recent],
                "saved_at": datetime.now().isoformat(),
                "version": "1.0",
            }
            atomic_json_write(self.persistence_path, data)
        except Exception as e:
            print(f"[AnimaHistory] Could not save: {e}", file=sys.stderr)

    def _load(self):
        """Load history from disk."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            for obs in data.get("observations", []):
                self._history.append(AnimaSnapshot.from_dict(obs))

            print(f"[AnimaHistory] Loaded {len(self._history)} observations", file=sys.stderr)
        except Exception as e:
            print(f"[AnimaHistory] Could not load: {e}", file=sys.stderr)

    def save(self):
        """Explicitly save the history."""
        self._save()

    def clear(self):
        """Clear all history (use with caution)."""
        self._history.clear()


# === Singleton Pattern ===

_history: Optional[AnimaHistory] = None


def get_anima_history() -> AnimaHistory:
    """Get or create the global AnimaHistory instance."""
    global _history
    if _history is None:
        _history = AnimaHistory()
    return _history


def reset_anima_history():
    """Reset the global AnimaHistory (mainly for testing)."""
    global _history
    _history = None
