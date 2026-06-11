"""
Trajectory Identity - Computing and comparing agent trajectory signatures.

This module implements the core framework from the Trajectory Identity Paper:
identity as dynamical invariant, computed from behavioral history.

The trajectory signature Σ = {Π, Β, Α, Ρ, Δ, Η} captures the invariant
characteristics that define an agent's identity:
- Π (Preference Profile): Learned environmental preferences
- Β (Belief Signature): Self-belief patterns
- Α (Attractor Basin): Equilibrium and variance in anima state
- Ρ (Recovery Profile): Characteristic time constants
- Δ (Relational Disposition): Social behavior patterns
- Η (Homeostatic Identity): Unified self-maintenance characterization

See: trajectory-identity paper (cirwel/trajectory-identity-paper, separate repo)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import json
import sys

from .atomic_write import atomic_json_write

import math

# Optional numpy for advanced computations
try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Paper Definition 2.2: Viability Envelope
# Bounds are in EISV space for homeostatic identity comparison
VIABILITY_BOUNDS = {
    "E": (0.1, 0.9),
    "I": (0.3, 1.0),
    "S": (0.0, 0.6),
    "V": (-0.2, 0.15),
}


def bhattacharyya_similarity(
    mu1: List[float], cov1: List[List[float]],
    mu2: List[float], cov2: List[List[float]],
) -> float:
    """Bhattacharyya coefficient between two Gaussian distributions.

    Returns similarity in [0, 1] where 1 = identical distributions.
    Paper reference: Section 4.2

    D_B = (1/8)(mu1-mu2)^T Sigma_avg^{-1} (mu1-mu2)
        + (1/2) ln(|Sigma_avg| / sqrt(|Sigma1| * |Sigma2|))
    sim = exp(-D_B)
    """
    if HAS_NUMPY:
        return _bhattacharyya_numpy(mu1, cov1, mu2, cov2)
    return _bhattacharyya_pure(mu1, cov1, mu2, cov2)


def _bhattacharyya_numpy(
    mu1: List[float], cov1: List[List[float]],
    mu2: List[float], cov2: List[List[float]],
) -> float:
    """Bhattacharyya coefficient using numpy."""
    m1 = np.array(mu1, dtype=float)
    m2 = np.array(mu2, dtype=float)
    s1 = np.array(cov1, dtype=float)
    s2 = np.array(cov2, dtype=float)
    n = len(mu1)

    # Average covariance with epsilon regularization
    s_avg = (s1 + s2) / 2.0 + np.eye(n) * 1e-6

    try:
        s_avg_inv = np.linalg.inv(s_avg)
        det_avg = np.linalg.det(s_avg)
        det1 = np.linalg.det(s1 + np.eye(n) * 1e-6)
        det2 = np.linalg.det(s2 + np.eye(n) * 1e-6)
    except np.linalg.LinAlgError:
        # Singular matrix — fall back to center distance
        dist = float(np.linalg.norm(m1 - m2))
        return math.exp(-dist * 2)

    if det_avg <= 0 or det1 <= 0 or det2 <= 0:
        dist = float(np.linalg.norm(m1 - m2))
        return math.exp(-dist * 2)

    diff = m1 - m2
    # Mahalanobis term: (1/8)(mu1-mu2)^T Sigma_avg^{-1} (mu1-mu2)
    mahal = float(diff @ s_avg_inv @ diff) / 8.0
    # Determinant term: (1/2) ln(|Sigma_avg| / sqrt(|Sigma1|*|Sigma2|))
    det_term = 0.5 * math.log(det_avg / math.sqrt(det1 * det2))

    d_b = mahal + det_term
    return max(0.0, min(1.0, math.exp(-d_b)))


def _bhattacharyya_pure(
    mu1: List[float], cov1: List[List[float]],
    mu2: List[float], cov2: List[List[float]],
) -> float:
    """Bhattacharyya coefficient using pure Python (4x4 matrices)."""
    n = len(mu1)

    # Average covariance with epsilon
    s_avg = [[(cov1[i][j] + cov2[i][j]) / 2.0 + (1e-6 if i == j else 0.0)
              for j in range(n)] for i in range(n)]

    det_avg = _det4(s_avg) if n == 4 else _det_generic(s_avg)
    s1_reg = [[cov1[i][j] + (1e-6 if i == j else 0.0) for j in range(n)] for i in range(n)]
    s2_reg = [[cov2[i][j] + (1e-6 if i == j else 0.0) for j in range(n)] for i in range(n)]
    det1 = _det4(s1_reg) if n == 4 else _det_generic(s1_reg)
    det2 = _det4(s2_reg) if n == 4 else _det_generic(s2_reg)

    if det_avg <= 0 or det1 <= 0 or det2 <= 0:
        dist = sum((a - b)**2 for a, b in zip(mu1, mu2)) ** 0.5
        return math.exp(-dist * 2)

    inv_avg = _inv4(s_avg) if n == 4 else None
    if inv_avg is None:
        dist = sum((a - b)**2 for a, b in zip(mu1, mu2)) ** 0.5
        return math.exp(-dist * 2)

    diff = [a - b for a, b in zip(mu1, mu2)]
    # Mahalanobis: diff^T @ inv_avg @ diff
    mahal = 0.0
    for i in range(n):
        for j in range(n):
            mahal += diff[i] * inv_avg[i][j] * diff[j]
    mahal /= 8.0

    det_term = 0.5 * math.log(det_avg / math.sqrt(det1 * det2))
    d_b = mahal + det_term
    return max(0.0, min(1.0, math.exp(-d_b)))


def _det4(m: List[List[float]]) -> float:
    """Determinant of 4x4 matrix via cofactor expansion along first row."""
    def _det3(a):
        return (a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
              - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
              + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]))

    result = 0.0
    for col in range(4):
        minor = [[m[r][c] for c in range(4) if c != col] for r in range(1, 4)]
        sign = 1 if col % 2 == 0 else -1
        result += sign * m[0][col] * _det3(minor)
    return result


def _det_generic(m: List[List[float]]) -> float:
    """Determinant for any square matrix via LU-style elimination."""
    n = len(m)
    mat = [row[:] for row in m]
    det = 1.0
    for i in range(n):
        if abs(mat[i][i]) < 1e-12:
            for j in range(i + 1, n):
                if abs(mat[j][i]) > 1e-12:
                    mat[i], mat[j] = mat[j], mat[i]
                    det *= -1
                    break
            else:
                return 0.0
        det *= mat[i][i]
        for j in range(i + 1, n):
            factor = mat[j][i] / mat[i][i]
            for k in range(i, n):
                mat[j][k] -= factor * mat[i][k]
    return det


def _inv4(m: List[List[float]]) -> Optional[List[List[float]]]:
    """Inverse of 4x4 matrix via Gauss-Jordan elimination."""
    n = 4
    aug = [m[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            return None

        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    return [[aug[i][j + n] for j in range(n)] for i in range(n)]


def homeostatic_similarity(eta1: Dict[str, Any], eta2: Dict[str, Any]) -> float:
    """Compute Eta (Homeostatic Identity) similarity (paper Section 3.6).

    Combines:
    1. Set-point proximity (weight 0.4): Bhattacharyya if covariance available
    2. Recovery dynamics (weight 0.3): log-ratio of tau
    3. Viability margin (weight 0.3): how safely centered within bounds
    """
    scores = []
    weights = []

    # Sub-component 1: Set-point similarity
    sp1 = eta1.get("set_point")
    sp2 = eta2.get("set_point")
    if sp1 and sp2 and len(sp1) == len(sp2):
        bs1 = eta1.get("basin_shape")
        bs2 = eta2.get("basin_shape")
        if bs1 and bs2:
            scores.append(bhattacharyya_similarity(sp1, bs1, sp2, bs2))
        else:
            dist = sum((a - b)**2 for a, b in zip(sp1, sp2)) ** 0.5
            scores.append(math.exp(-dist * 2))
        weights.append(0.4)

    # Sub-component 2: Recovery dynamics similarity
    tau1 = eta1.get("recovery_tau")
    tau2 = eta2.get("recovery_tau")
    if tau1 and tau2 and tau1 > 0 and tau2 > 0:
        log_ratio = abs(math.log(tau1 / tau2))
        scores.append(math.exp(-log_ratio))
        weights.append(0.3)

    # Sub-component 3: Viability margin similarity
    bounds1 = eta1.get("viability_bounds", VIABILITY_BOUNDS)
    bounds2 = eta2.get("viability_bounds", VIABILITY_BOUNDS)
    if sp1 and sp2:
        margin1 = _viability_margin(sp1, bounds1)
        margin2 = _viability_margin(sp2, bounds2)
        if margin1 is not None and margin2 is not None:
            # Compare margin profiles: 1 - |m1 - m2|
            scores.append(1.0 - abs(margin1 - margin2))
            weights.append(0.3)

    if not scores:
        return 0.5
    total = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total


def _viability_margin(set_point: List[float], bounds: Dict[str, Any]) -> Optional[float]:
    """How safely centered is the set-point within viability bounds.

    Per-dimension: min(mu - lo, hi - mu) / (hi - lo), averaged.
    Returns value in [0, 0.5] where 0.5 = perfectly centered.
    """
    dim_keys = list(bounds.keys())
    if len(set_point) != len(dim_keys):
        return None
    margins = []
    for i, key in enumerate(dim_keys):
        lo, hi = bounds[key]
        span = hi - lo
        if span <= 0:
            continue
        margin = min(set_point[i] - lo, hi - set_point[i]) / span
        margins.append(max(0.0, margin))
    return sum(margins) / len(margins) if margins else None

# Type hints without circular imports
if TYPE_CHECKING:
    from .growth import GrowthSystem
    from .self_model import SelfModel
    from .anima_history import AnimaHistory


@dataclass
class TrajectorySignature:
    """
    Complete trajectory signature Σ.

    This is the mathematical encoding of "who this agent is" - not a static
    ID, but the pattern that persists across time.

    Attributes:
        preferences: Π - Learned environmental preferences
        beliefs: Β - Self-belief patterns
        attractor: Α - Equilibrium and variance in state space
        recovery: Ρ - Recovery dynamics (time constants)
        relational: Δ - Social behavior patterns
        computed_at: When this signature was computed
        observation_count: Number of observations used
    """

    # Components (all are dictionaries from component extractors)
    preferences: Dict[str, Any] = field(default_factory=dict)  # Π
    beliefs: Dict[str, Any] = field(default_factory=dict)       # Β
    attractor: Optional[Dict[str, Any]] = None                  # Α
    recovery: Dict[str, Any] = field(default_factory=dict)      # Ρ
    relational: Dict[str, Any] = field(default_factory=dict)    # Δ
    homeostatic: Optional[Dict[str, Any]] = None                # Η

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    observation_count: int = 0

    # Genesis Signature (Σ₀) - Reference anchor for drift detection
    # Set once at agent creation/fork, never updated
    genesis_signature: Optional['TrajectorySignature'] = None

    # Component variance history for adaptive weighting
    # Keys: "preferences", "beliefs", "attractor", "recovery", "relational"
    # Values: list of recent similarity scores for each component
    component_history: Dict[str, List[float]] = field(default_factory=dict)

    def similarity(self, other: 'TrajectorySignature') -> float:
        """
        Compute similarity to another trajectory signature.

        This is the core operation for determining identity:
        sim(Σ₁, Σ₂) > θ implies "same identity"

        Args:
            other: Another TrajectorySignature to compare against

        Returns:
            Similarity score in [0, 1] where 1 = identical trajectories
        """
        scores = []
        weights = []

        # --- Preference Similarity (Π) ---
        # Cosine similarity of preference vectors
        if self.preferences.get("vector") and other.preferences.get("vector"):
            v1 = self.preferences["vector"]
            v2 = other.preferences["vector"]
            sim = self._cosine_similarity(v1, v2)
            if sim is not None:
                scores.append((sim + 1) / 2)  # Map [-1,1] to [0,1]
                weights.append(0.15)

        # --- Belief Similarity (Β) ---
        # Cosine similarity of belief values
        if self.beliefs.get("values") and other.beliefs.get("values"):
            v1 = self.beliefs["values"]
            v2 = other.beliefs["values"]
            sim = self._cosine_similarity(v1, v2)
            if sim is not None:
                scores.append((sim + 1) / 2)
                weights.append(0.15)

        # --- Attractor Similarity (Α) ---
        # Bhattacharyya coefficient when covariance available, else center distance
        if self.attractor and other.attractor:
            c1 = self.attractor.get("center")
            c2 = other.attractor.get("center")
            if c1 and c2:
                cov1 = self.attractor.get("covariance")
                cov2 = other.attractor.get("covariance")
                if cov1 and cov2 and len(cov1) == len(c1) and len(cov2) == len(c2):
                    alpha_sim = bhattacharyya_similarity(c1, cov1, c2, cov2)
                else:
                    dist = sum((a - b)**2 for a, b in zip(c1, c2)) ** 0.5
                    alpha_sim = math.exp(-dist * 2)
                scores.append(alpha_sim)
                weights.append(0.25)

        # --- Recovery Similarity (Ρ) ---
        # Similarity of time constants (log-scale)
        t1 = self.recovery.get("tau_estimate")
        t2 = other.recovery.get("tau_estimate")
        if t1 and t2 and t1 > 0 and t2 > 0:
            log_ratio = abs(math.log(t1 / t2))
            tau_sim = math.exp(-log_ratio)
            scores.append(tau_sim)
            weights.append(0.20)

        # --- Relational Similarity (Δ) ---
        # Valence tendency similarity
        v1 = self.relational.get("valence_tendency")
        v2 = other.relational.get("valence_tendency")
        if v1 is not None and v2 is not None:
            # Max diff is 2 (-1 to 1 range)
            valence_sim = 1 - abs(v1 - v2) / 2
            scores.append(valence_sim)
            weights.append(0.10)

        # --- Homeostatic Similarity (Η) ---
        if self.homeostatic and other.homeostatic:
            eta_sim = homeostatic_similarity(self.homeostatic, other.homeostatic)
            scores.append(eta_sim)
            weights.append(0.15)

        # --- Compute weighted average ---
        if not scores:
            return 0.5  # No data to compare

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return float(weighted_sum / total_weight)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> Optional[float]:
        """Compute cosine similarity between two vectors."""
        if len(v1) != len(v2) or len(v1) == 0:
            return None

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return None

        return dot / (norm1 * norm2)

    def compute_adaptive_weights(self, default_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute adaptive weights using inverse variance weighting.

        Components with lower historical variance get higher weights
        because they are more stable identity markers.

        Args:
            default_weights: Fallback weights if no history exists

        Returns:
            Dictionary mapping component names to weights (sum to ~1.0)
        """
        if default_weights is None:
            default_weights = {
                "preferences": 0.15,
                "beliefs": 0.15,
                "attractor": 0.25,
                "recovery": 0.20,
                "relational": 0.10,
                "homeostatic": 0.15,
            }

        # Need at least 5 observations per component for variance
        if not self.component_history:
            return default_weights

        variances = {}
        for component, history in self.component_history.items():
            if len(history) >= 5:
                mean = sum(history) / len(history)
                var = sum((x - mean) ** 2 for x in history) / len(history)
                # Add epsilon to prevent division by zero
                variances[component] = max(var, 1e-6)

        if not variances:
            return default_weights

        # Inverse variance weighting: w_i = (1/var_i) / sum(1/var_j)
        inv_variances = {k: 1.0 / v for k, v in variances.items()}
        total_inv_var = sum(inv_variances.values())

        adaptive_weights = {}
        for component in default_weights:
            if component in inv_variances:
                adaptive_weights[component] = inv_variances[component] / total_inv_var
            else:
                # Use default for components without history
                adaptive_weights[component] = default_weights[component]

        return adaptive_weights

    def similarity_adaptive(
        self,
        other: 'TrajectorySignature',
        update_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute similarity using adaptive inverse variance weighting.

        This is the production version that learns which components
        are most stable for this agent and weights them accordingly.

        Args:
            other: Another TrajectorySignature to compare against
            update_history: Whether to update component history

        Returns:
            Dictionary with similarity score and component breakdown
        """
        component_scores = {}

        # Compute each component similarity
        if self.preferences.get("vector") and other.preferences.get("vector"):
            sim = self._cosine_similarity(
                self.preferences["vector"], other.preferences["vector"]
            )
            if sim is not None:
                component_scores["preferences"] = (sim + 1) / 2

        if self.beliefs.get("values") and other.beliefs.get("values"):
            sim = self._cosine_similarity(
                self.beliefs["values"], other.beliefs["values"]
            )
            if sim is not None:
                component_scores["beliefs"] = (sim + 1) / 2

        if self.attractor and other.attractor:
            c1 = self.attractor.get("center")
            c2 = other.attractor.get("center")
            if c1 and c2:
                cov1 = self.attractor.get("covariance")
                cov2 = other.attractor.get("covariance")
                if cov1 and cov2 and len(cov1) == len(c1) and len(cov2) == len(c2):
                    component_scores["attractor"] = bhattacharyya_similarity(c1, cov1, c2, cov2)
                else:
                    dist = sum((a - b)**2 for a, b in zip(c1, c2)) ** 0.5
                    component_scores["attractor"] = math.exp(-dist * 2)

        t1 = self.recovery.get("tau_estimate")
        t2 = other.recovery.get("tau_estimate")
        if t1 and t2 and t1 > 0 and t2 > 0:
            log_ratio = abs(math.log(t1 / t2))
            component_scores["recovery"] = math.exp(-log_ratio)

        v1 = self.relational.get("valence_tendency")
        v2 = other.relational.get("valence_tendency")
        if v1 is not None and v2 is not None:
            component_scores["relational"] = 1 - abs(v1 - v2) / 2

        if self.homeostatic and other.homeostatic:
            component_scores["homeostatic"] = homeostatic_similarity(
                self.homeostatic, other.homeostatic
            )

        # Update history if requested
        if update_history:
            for component, score in component_scores.items():
                if component not in self.component_history:
                    self.component_history[component] = []
                self.component_history[component].append(score)
                # Keep last 100 observations
                if len(self.component_history[component]) > 100:
                    self.component_history[component] = self.component_history[component][-100:]

        # Compute adaptive weights
        adaptive_weights = self.compute_adaptive_weights()

        # Compute weighted similarity
        if not component_scores:
            return {"similarity": 0.5, "components": {}, "weights": adaptive_weights}

        weighted_sum = 0.0
        total_weight = 0.0
        for component, score in component_scores.items():
            weight = adaptive_weights.get(component, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        similarity = weighted_sum / total_weight if total_weight > 0 else 0.5

        return {
            "similarity": round(similarity, 4),
            "components": {k: round(v, 4) for k, v in component_scores.items()},
            "weights": {k: round(v, 4) for k, v in adaptive_weights.items()},
            "history_depth": {k: len(v) for k, v in self.component_history.items()},
        }

    def is_same_identity(self, other: 'TrajectorySignature', threshold: float = 0.8) -> bool:
        """
        Determine if two signatures represent the same identity.

        Args:
            other: Another TrajectorySignature
            threshold: Similarity threshold (default 0.8)

        Returns:
            True if similarity > threshold
        """
        return self.similarity(other) > threshold

    def detect_anomaly(self, historical: 'TrajectorySignature', threshold: float = 0.7) -> Dict[str, Any]:
        """
        Detect if current signature deviates significantly from historical.

        Args:
            historical: Previous trajectory signature to compare against
            threshold: Minimum similarity to be considered "normal"

        Returns:
            Dictionary with anomaly detection results
        """
        sim = self.similarity(historical)
        is_anomaly = sim < threshold

        return {
            "is_anomaly": is_anomaly,
            "similarity": round(sim, 4),
            "threshold": threshold,
            "deviation": round(1 - sim, 4),
        }

    def lineage_similarity(self) -> Optional[float]:
        """
        Compute similarity to genesis signature (Σ₀).

        This measures how much the agent has drifted from its original
        identity - the "boiling frog" detector for gradual identity shift.

        Returns:
            Similarity to genesis signature [0, 1], or None if no genesis
        """
        if self.genesis_signature is None:
            return None
        return self.similarity(self.genesis_signature)

    def detect_anomaly_two_tier(
        self,
        recent_signature: 'TrajectorySignature',
        coherence_threshold: float = 0.7,
        lineage_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Two-tier anomaly detection as specified in paper Section 6.1.2.

        Tier 1 (Coherence): Compare to recent behavior (short-term)
        Tier 2 (Lineage): Compare to genesis signature (long-term)

        Args:
            recent_signature: Recent trajectory for coherence check
            coherence_threshold: Threshold for short-term coherence
            lineage_threshold: Threshold for long-term lineage drift

        Returns:
            Dictionary with two-tier anomaly results
        """
        # Tier 1: Coherence check (short-term)
        coherence_sim = self.similarity(recent_signature)
        coherence_ok = coherence_sim >= coherence_threshold

        # Tier 2: Lineage check (long-term drift from genesis)
        lineage_sim = self.lineage_similarity()
        if lineage_sim is not None:
            lineage_ok = lineage_sim >= lineage_threshold
        else:
            lineage_ok = True  # No genesis to compare against
            lineage_sim = 1.0

        # Anomaly if either tier fails
        is_anomaly = not (coherence_ok and lineage_ok)

        return {
            "is_anomaly": is_anomaly,
            "coherence": {
                "similarity": round(coherence_sim, 4),
                "threshold": coherence_threshold,
                "passed": coherence_ok,
            },
            "lineage": {
                "similarity": round(lineage_sim, 4) if lineage_sim else None,
                "threshold": lineage_threshold,
                "passed": lineage_ok,
                "has_genesis": self.genesis_signature is not None,
            },
            "tier_failed": None if not is_anomaly else (
                "coherence" if not coherence_ok else "lineage"
            ),
        }

    @property
    def identity_confidence(self) -> float:
        """
        Confidence in identity stability, as per paper Section 4.5.

        Formula: min(1.0, observation_count / 50) * stability_score

        Returns:
            Confidence score [0, 1] where 1 = high confidence
        """
        # Cold start factor: confidence grows with observations (saturates at 50)
        cold_start_factor = min(1.0, self.observation_count / 50)
        return cold_start_factor * self.get_stability_score()

    def get_stability_score(self) -> float:
        """
        Compute how stable/mature this signature is.

        Based on:
        - Number of observations
        - Confidence in beliefs
        - Number of recovery episodes

        Returns:
            Stability score in [0, 1]
        """
        factors = []

        # Observation count (saturates at 100)
        obs_factor = min(1.0, self.observation_count / 100)
        factors.append(obs_factor)

        # Belief confidence
        if self.beliefs.get("avg_confidence"):
            factors.append(self.beliefs["avg_confidence"])

        # Recovery confidence
        if self.recovery.get("confidence"):
            factors.append(self.recovery["confidence"])

        # Preference learning
        if self.preferences.get("n_learned"):
            pref_factor = min(1.0, self.preferences["n_learned"] / 5)
            factors.append(pref_factor)

        if not factors:
            return 0.0

        return sum(factors) / len(factors)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "preferences": self.preferences,
            "beliefs": self.beliefs,
            "attractor": self.attractor,
            "recovery": self.recovery,
            "relational": self.relational,
            "homeostatic": self.homeostatic,
            "computed_at": self.computed_at.isoformat(),
            "observation_count": self.observation_count,
            "stability_score": round(self.get_stability_score(), 3),
        }
        if self.genesis_signature is not None:
            result["genesis_signature"] = self.genesis_signature.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectorySignature':
        """Deserialize from dictionary."""
        computed_at = data.get("computed_at")
        if isinstance(computed_at, str):
            try:
                computed_at = datetime.fromisoformat(computed_at)
            except (ValueError, TypeError):
                computed_at = datetime.now()
        elif not isinstance(computed_at, datetime):
            computed_at = datetime.now()

        sig = cls(
            preferences=data.get("preferences", {}),
            beliefs=data.get("beliefs", {}),
            attractor=data.get("attractor"),
            recovery=data.get("recovery", {}),
            relational=data.get("relational", {}),
            homeostatic=data.get("homeostatic"),
            computed_at=computed_at,
            observation_count=data.get("observation_count", 0),
        )
        # Recursively deserialize nested genesis if present
        genesis_data = data.get("genesis_signature")
        if isinstance(genesis_data, dict):
            sig.genesis_signature = cls.from_dict(genesis_data)
        return sig

    def summary(self) -> Dict[str, Any]:
        """Get a compact summary of the signature."""
        lineage_sim = self.lineage_similarity()
        return {
            "identity_confidence": round(self.identity_confidence, 3),
            "stability_score": round(self.get_stability_score(), 3),
            "observation_count": self.observation_count,
            "preferences_learned": self.preferences.get("n_learned", 0),
            "belief_confidence": self.beliefs.get("avg_confidence", 0),
            "attractor_defined": self.attractor is not None,
            "recovery_tau": self.recovery.get("tau_estimate"),
            "relationships": self.relational.get("n_relationships", 0),
            "lineage_similarity": round(lineage_sim, 3) if lineage_sim else None,
            "has_genesis": self.genesis_signature is not None,
            "computed_at": self.computed_at.isoformat(),
        }


def compute_trajectory_signature(
    growth_system: Optional['GrowthSystem'] = None,
    self_model: Optional['SelfModel'] = None,
    anima_history: Optional['AnimaHistory'] = None,
) -> TrajectorySignature:
    """
    Compute trajectory signature from available data sources.

    This function aggregates data from multiple systems to compute Σ.
    Each component is optional - the signature will include whatever
    data is available.

    Genesis (Σ₀) handling:
    - On every call, attempts to load genesis from disk
    - If no genesis exists and observation_count >= GENESIS_MIN_OBSERVATIONS,
      freezes the current signature as genesis (write-once, never overwrites)
    - Attaches genesis to the returned signature for drift detection

    Args:
        growth_system: GrowthSystem instance (for Π, Δ)
        self_model: SelfModel instance (for Β, Ρ)
        anima_history: AnimaHistory instance (for Α)

    Returns:
        TrajectorySignature Σ (with genesis_signature attached if available)
    """
    # Extract components from each system

    # Π: Preference Profile
    preferences = {}
    if growth_system:
        try:
            preferences = growth_system.get_preference_vector()
        except Exception as e:
            print(f"[Trajectory] Could not get preferences: {e}", file=sys.stderr)

    # Β: Belief Signature
    beliefs = {}
    if self_model:
        try:
            beliefs = self_model.get_belief_signature()
        except Exception as e:
            print(f"[Trajectory] Could not get beliefs: {e}", file=sys.stderr)

    # Α: Attractor Basin
    attractor = None
    observation_count = 0
    if anima_history:
        try:
            attractor = anima_history.get_attractor_basin()
            if attractor:
                observation_count = attractor.get("n_observations", 0)
        except Exception as e:
            print(f"[Trajectory] Could not get attractor: {e}", file=sys.stderr)

    # Ρ: Recovery Profile
    recovery = {}
    if self_model:
        try:
            recovery = self_model.get_recovery_profile()
        except Exception as e:
            print(f"[Trajectory] Could not get recovery: {e}", file=sys.stderr)

    # Δ: Relational Disposition
    relational = {}
    if growth_system:
        try:
            relational = growth_system.get_relational_disposition()
        except Exception as e:
            print(f"[Trajectory] Could not get relational: {e}", file=sys.stderr)

    # Η: Homeostatic Identity (Definition 3.6)
    homeostatic = None
    if attractor and recovery:
        homeostatic = {
            "set_point": attractor.get("center"),
            "basin_shape": attractor.get("covariance"),
            "recovery_tau": recovery.get("tau_estimate"),
            "viability_bounds": VIABILITY_BOUNDS,
        }

    sig = TrajectorySignature(
        preferences=preferences,
        beliefs=beliefs,
        attractor=attractor,
        recovery=recovery,
        relational=relational,
        homeostatic=homeostatic,
        observation_count=observation_count,
    )

    # ── Genesis (Σ₀) management ──
    # Try to load existing genesis from disk
    genesis = load_genesis()

    if genesis is not None:
        # Attach existing genesis for drift detection
        sig.genesis_signature = genesis
    elif observation_count >= GENESIS_MIN_OBSERVATIONS:
        # Sufficient observations — freeze current signature as genesis
        # save_genesis is write-once and will not overwrite
        saved = save_genesis(sig)
        if saved:
            # Reload to get a clean copy (sig itself should not self-reference)
            sig.genesis_signature = load_genesis()

    return sig


def compare_signatures(sig1: TrajectorySignature, sig2: TrajectorySignature) -> Dict[str, Any]:
    """
    Compare two trajectory signatures in detail.

    Returns per-component similarity breakdown.
    """
    overall = sig1.similarity(sig2)

    # Per-component breakdown
    components = {}

    # Preferences
    if sig1.preferences.get("vector") and sig2.preferences.get("vector"):
        v1, v2 = sig1.preferences["vector"], sig2.preferences["vector"]
        sim = sig1._cosine_similarity(v1, v2)
        if sim:
            components["preferences"] = round((sim + 1) / 2, 4)

    # Beliefs
    if sig1.beliefs.get("values") and sig2.beliefs.get("values"):
        v1, v2 = sig1.beliefs["values"], sig2.beliefs["values"]
        sim = sig1._cosine_similarity(v1, v2)
        if sim:
            components["beliefs"] = round((sim + 1) / 2, 4)

    # Attractor (Bhattacharyya when covariance available)
    if sig1.attractor and sig2.attractor:
        c1 = sig1.attractor.get("center")
        c2 = sig2.attractor.get("center")
        if c1 and c2:
            cov1 = sig1.attractor.get("covariance")
            cov2 = sig2.attractor.get("covariance")
            if cov1 and cov2 and len(cov1) == len(c1) and len(cov2) == len(c2):
                components["attractor"] = round(bhattacharyya_similarity(c1, cov1, c2, cov2), 4)
            else:
                dist = sum((a - b)**2 for a, b in zip(c1, c2)) ** 0.5
                components["attractor"] = round(math.exp(-dist * 2), 4)

    # Recovery
    t1 = sig1.recovery.get("tau_estimate")
    t2 = sig2.recovery.get("tau_estimate")
    if t1 and t2 and t1 > 0 and t2 > 0:
        log_ratio = abs(math.log(t1 / t2))
        components["recovery"] = round(math.exp(-log_ratio), 4)

    # Relational
    v1 = sig1.relational.get("valence_tendency")
    v2 = sig2.relational.get("valence_tendency")
    if v1 is not None and v2 is not None:
        components["relational"] = round(1 - abs(v1 - v2) / 2, 4)

    # Homeostatic (Eta)
    if sig1.homeostatic and sig2.homeostatic:
        components["homeostatic"] = round(
            homeostatic_similarity(sig1.homeostatic, sig2.homeostatic), 4
        )

    return {
        "overall_similarity": round(overall, 4),
        "components": components,
        "is_same_identity": overall > 0.8,
    }


# ── Genesis Persistence ──────────────────────────────────────────────

# Minimum observations before genesis is considered stable enough to lock
GENESIS_MIN_OBSERVATIONS = 30

# Default persistence path for genesis signature
_GENESIS_PATH = Path.home() / ".anima" / "trajectory_genesis.json"

# Module-level cache to avoid re-reading disk on every check-in
_cached_genesis: Optional[TrajectorySignature] = None


def save_genesis(signature: TrajectorySignature, path: Optional[Path] = None) -> bool:
    """
    Persist genesis signature (Σ₀) to disk. Write-once: never overwrites.

    Args:
        signature: The trajectory signature to freeze as genesis
        path: Override persistence path (default ~/.anima/trajectory_genesis.json)

    Returns:
        True if saved, False if genesis already exists on disk
    """
    global _cached_genesis
    dest = path or _GENESIS_PATH
    if dest.exists():
        return False  # Genesis is immutable — never overwrite

    try:
        data = signature.to_dict()
        data["frozen_at"] = datetime.now().isoformat()
        atomic_json_write(dest, data, indent=2)
        _cached_genesis = signature
        print(
            f"[Trajectory] Genesis Σ₀ frozen (obs={signature.observation_count}, "
            f"stability={signature.get_stability_score():.3f})",
            file=sys.stderr,
        )
        return True
    except Exception as e:
        print(f"[Trajectory] Could not save genesis: {e}", file=sys.stderr)
        return False


def load_genesis(path: Optional[Path] = None) -> Optional[TrajectorySignature]:
    """
    Load persisted genesis signature from disk.

    Returns cached version if already loaded.

    Args:
        path: Override persistence path

    Returns:
        TrajectorySignature or None if no genesis exists
    """
    global _cached_genesis
    if _cached_genesis is not None:
        return _cached_genesis

    src = path or _GENESIS_PATH
    if not src.exists():
        return None

    try:
        with open(src, "r") as f:
            data = json.load(f)
        _cached_genesis = TrajectorySignature.from_dict(data)
        print(
            f"[Trajectory] Genesis Σ₀ loaded (obs={_cached_genesis.observation_count})",
            file=sys.stderr,
        )
        return _cached_genesis
    except Exception as e:
        print(f"[Trajectory] Could not load genesis: {e}", file=sys.stderr)
        return None


# Path for last trajectory (overwritten on each save, unlike genesis)
_TRAJECTORY_LAST_PATH = Path.home() / ".anima" / "trajectory_last.json"


def save_trajectory(signature: TrajectorySignature, path: Optional[Path] = None) -> bool:
    """
    Persist current trajectory to disk. Overwrites previous (unlike genesis).

    Used for anomaly detection: compare current to last persisted.

    Args:
        signature: The trajectory signature to save
        path: Override path (default ~/.anima/trajectory_last.json)

    Returns:
        True if saved successfully
    """
    dest = path or _TRAJECTORY_LAST_PATH
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = signature.to_dict()
        data["saved_at"] = datetime.now().isoformat()
        atomic_json_write(dest, data, indent=2)
        return True
    except Exception as e:
        print(f"[Trajectory] Could not save last trajectory: {e}", file=sys.stderr)
        return False


def load_trajectory(path: Optional[Path] = None) -> Optional[TrajectorySignature]:
    """
    Load last persisted trajectory from disk.

    Args:
        path: Override path

    Returns:
        TrajectorySignature or None if no file exists
    """
    src = path or _TRAJECTORY_LAST_PATH
    if not src.exists():
        return None

    try:
        with open(src, "r") as f:
            data = json.load(f)
        return TrajectorySignature.from_dict(data)
    except Exception as e:
        print(f"[Trajectory] Could not load last trajectory: {e}", file=sys.stderr)
        return None
