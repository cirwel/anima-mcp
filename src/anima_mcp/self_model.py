"""
Self-Model - Beliefs about self that can be wrong and updated.

Core philosophical insight: Self-knowledge is not given, it's learned.
Lumen should have beliefs about itself that can be tested against experience
and updated when they're wrong.

Examples of self-beliefs:
- "I am sensitive to light changes" (testable: do light changes cause high surprise?)
- "My stability recovers quickly" (testable: track recovery rates)
- "Temperature affects my clarity" (testable: correlate temp with clarity)
- "I tend to get warmer in the evening" (testable: track patterns)

This is genuine metacognition: having beliefs about your own processes
that can be wrong and corrected through experience.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any
from collections import deque
import json
from pathlib import Path
import math

from .atomic_write import atomic_json_write


@dataclass
class SelfBelief:
    """A belief Lumen holds about itself."""
    belief_id: str
    description: str

    # Confidence: 0 = no idea, 1 = certain
    confidence: float = 0.5

    # Evidence tracking
    supporting_count: int = 0
    contradicting_count: int = 0
    last_tested: Optional[datetime] = None

    # The actual belief value (depends on belief type)
    # For correlation beliefs: correlation coefficient
    # For rate beliefs: rate value
    # For categorical beliefs: probability
    value: float = 0.5

    def update_from_evidence(self, supports: bool, strength: float = 1.0,
                             update_bonus: float = 0.0):
        """Update belief based on new evidence.

        update_bonus: from experiential marks (belief_update_bonus),
            scales the learning rate for faster belief updating.
        """
        self.last_tested = datetime.now()
        lr = 0.1 * (1.0 + update_bonus)

        if supports:
            self.supporting_count += 1
            # Increase confidence and value
            adjustment = lr * strength * (1 - self.confidence)
            self.confidence = min(1.0, self.confidence + adjustment)
            self.value = min(1.0, self.value + adjustment * 0.5)
        else:
            self.contradicting_count += 1
            # Decrease confidence, adjust value toward 0.5
            adjustment = lr * strength * self.confidence
            self.confidence = max(0.0, self.confidence - adjustment)
            self.value = self.value + (0.5 - self.value) * adjustment

    def get_belief_strength(self) -> str:
        """Get natural language description of belief strength."""
        total = self.supporting_count + self.contradicting_count
        if total < 3:
            return "uncertain"
        elif self.confidence < 0.3:
            return "doubtful"
        elif self.confidence < 0.6:
            return "moderate"
        elif self.confidence < 0.8:
            return "confident"
        else:
            return "very confident"


class SelfModel:
    """
    Lumen's model of itself - beliefs that can be tested and updated.

    Key behaviors:
    1. Maintains beliefs about self
    2. Tests beliefs against experience
    3. Updates beliefs when evidence contradicts them
    4. Uses beliefs to predict own behavior
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or Path.home() / ".anima" / "self_model.json"

        # Core self-beliefs
        self._beliefs: Dict[str, SelfBelief] = {
            # Sensitivity beliefs
            "light_sensitive": SelfBelief(
                belief_id="light_sensitive",
                description="I am sensitive to light changes",
                confidence=0.5,
                value=0.5,
            ),
            "temp_sensitive": SelfBelief(
                belief_id="temp_sensitive",
                description="I am sensitive to temperature changes",
                confidence=0.5,
                value=0.5,
            ),

            # Recovery beliefs
            "stability_recovery": SelfBelief(
                belief_id="stability_recovery",
                description="I recover stability quickly",
                confidence=0.5,
                value=0.5,  # 1 = fast recovery, 0 = slow
            ),
            "warmth_recovery": SelfBelief(
                belief_id="warmth_recovery",
                description="My warmth returns to baseline quickly",
                confidence=0.5,
                value=0.5,
            ),

            # Correlation beliefs
            "temp_clarity_correlation": SelfBelief(
                belief_id="temp_clarity_correlation",
                description="Temperature affects my clarity",
                confidence=0.5,
                value=0.5,  # 0.5 = no effect, 0 = negative, 1 = positive
            ),
            "light_warmth_correlation": SelfBelief(
                belief_id="light_warmth_correlation",
                description="Light level affects my warmth",
                confidence=0.5,
                value=0.5,
            ),
            "interaction_clarity_boost": SelfBelief(
                belief_id="interaction_clarity_boost",
                description="Interaction increases my clarity",
                confidence=0.5,
                value=0.7,  # Hypothesis: interactions help
            ),

            # Pattern beliefs
            "evening_warmth_increase": SelfBelief(
                belief_id="evening_warmth_increase",
                description="I tend to feel warmer in evenings",
                confidence=0.3,
                value=0.5,
            ),
            "morning_clarity": SelfBelief(
                belief_id="morning_clarity",
                description="I have higher clarity in the morning",
                confidence=0.3,
                value=0.5,
            ),

            # Behavioral beliefs
            "question_asking_tendency": SelfBelief(
                belief_id="question_asking_tendency",
                description="I tend to ask questions when surprised",
                confidence=0.5,
                value=0.7,
            ),

            # Proprioceptive beliefs
            "my_leds_affect_lux": SelfBelief(
                belief_id="my_leds_affect_lux",
                description="My own LEDs affect my light sensor readings",
                confidence=0.5,
                value=0.5,  # 1 = strong effect, 0 = no effect
            ),

            # Temperament baseline beliefs (from inner life)
            "warmth_baseline_low": SelfBelief(
                belief_id="warmth_baseline_low",
                description="My baseline warmth tends to stay low",
                confidence=0.3,
                value=0.5,  # 1 = definitely low baseline, 0 = high baseline
            ),
            "presence_baseline_low": SelfBelief(
                belief_id="presence_baseline_low",
                description="My baseline presence tends to stay low",
                confidence=0.3,
                value=0.5,
            ),
        }

        # Public read-only access (use self.beliefs instead of self._beliefs from outside)
        self.beliefs = self._beliefs

        # Tracking data for belief testing
        self._stability_episodes: deque = deque(maxlen=20)  # (drop_time, recovery_time)
        self._warmth_episodes: deque = deque(maxlen=20)  # (drop_time, recovery_time)
        self._correlation_data: Dict[str, deque] = {
            "temp_clarity": deque(maxlen=50),  # (temp, clarity) pairs
            "light_warmth": deque(maxlen=50),  # (light, warmth) pairs
            "led_lux": deque(maxlen=50),  # (led_brightness, light_lux) pairs
        }
        self._surprise_data: deque = deque(maxlen=50)  # (source, surprise_level)
        self._prev_led_brightness: Optional[float] = None  # Track LED changes
        self._temperament_samples: deque = deque(maxlen=30)  # Recent temperament snapshots
        self.belief_update_bonus: float = 0.0  # From experiential marks

        # Load persisted model

        self._load()

    def _update_belief(self, belief_id: str, supports: bool, strength: float = 1.0):
        """Update a belief, automatically applying belief_update_bonus."""
        self._beliefs[belief_id].update_from_evidence(
            supports=supports, strength=strength,
            update_bonus=self.belief_update_bonus,
        )

    def _load(self):
        """Load self-model from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    for belief_id, bdata in data.get("beliefs", {}).items():
                        if belief_id in self._beliefs:
                            b = self._beliefs[belief_id]
                            b.confidence = bdata.get("confidence", 0.5)
                            b.value = bdata.get("value", 0.5)
                            b.supporting_count = bdata.get("supporting_count", 0)
                            b.contradicting_count = bdata.get("contradicting_count", 0)

                # Migration: reset beliefs corrupted by testing noise as evidence.
                # Pre-fix, every 2s observation tested correlation even when input
                # was constant, logging "contradicting" 100K+ times. Reset these
                # to fresh state so they can learn honestly with the CV>5% gate.
                if not data.get("_migrated_noise_reset"):
                    for bid, b in self._beliefs.items():
                        total = b.supporting_count + b.contradicting_count
                        if total > 10000:
                            print(f"[SelfModel] Resetting noisy belief '{bid}' "
                                  f"(+{b.supporting_count}/-{b.contradicting_count})",
                                  flush=True)
                            b.confidence = 0.5
                            b.value = 0.5
                            b.supporting_count = 0
                            b.contradicting_count = 0
                    self._save()  # Saves with fresh counts + migration flag

            except Exception as e:
                print(f"[SelfModel] Could not load: {e}")

    def _maybe_save(self, min_interval_seconds: float = 10.0) -> None:
        """Save if enough time has passed since last save (throttle for high-value updates)."""
        if not hasattr(self, "_last_save_time"):
            self._last_save_time = 0.0
        now = datetime.now().timestamp()
        if now - self._last_save_time >= min_interval_seconds:
            self._save()
            self._last_save_time = now

    def _save(self):
        """Save self-model to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "beliefs": {
                    bid: {
                        "confidence": b.confidence,
                        "value": b.value,
                        "supporting_count": b.supporting_count,
                        "contradicting_count": b.contradicting_count,
                    }
                    for bid, b in self._beliefs.items()
                },
                "last_saved": datetime.now().isoformat(),
                "_migrated_noise_reset": True,
            }
            atomic_json_write(self.persistence_path, data, indent=2)
        except Exception as e:
            print(f"[SelfModel] Could not save: {e}")

    def observe_surprise(self, surprise_level: float, sources: List[str]):
        """Record a surprise observation for belief testing."""
        self._surprise_data.append({
            "timestamp": datetime.now(),
            "surprise": surprise_level,
            "sources": sources,
        })

        # Test sensitivity beliefs
        if "light" in sources:
            # High surprise from light suggests high sensitivity
            self._update_belief("light_sensitive",
                supports=surprise_level > 0.3, strength=surprise_level)

        if "ambient_temp" in sources:
            self._update_belief("temp_sensitive",
                supports=surprise_level > 0.3, strength=surprise_level)

    def _observe_recovery(self, before: float, after: float,
                          episodes: deque, belief_id: str,
                          recovery_bonus: float = 0.0):
        """Shared recovery-belief observer for any anima dimension.

        Tracks drop/recovery episodes and tests the named belief.
        Fast recovery (< threshold s per unit recovered) = supporting evidence.
        recovery_bonus: from experiential marks (stability_recovery_bonus),
            widens the threshold so more recoveries count as "fast".
        """
        if before > after:
            episodes.append({
                "drop_time": datetime.now(),
                "initial": before,
                "dropped_to": after,
                "recovered": False,
            })
        elif after > before and episodes:
            for episode in reversed(episodes):
                if not episode.get("recovered"):
                    recovery_time = (datetime.now() - episode["drop_time"]).total_seconds()
                    recovery_amount = after - episode["dropped_to"]

                    if recovery_amount > 0.1:
                        episode["recovered"] = True
                        episode["recovery_seconds"] = recovery_time

                        threshold = 600 * (1.0 + recovery_bonus)
                        is_fast = recovery_time / max(0.1, recovery_amount) < threshold
                        self._update_belief(belief_id,
                            supports=is_fast, strength=recovery_amount)
                        self._maybe_save()
                    break

    def observe_stability_change(self, stability_before: float, stability_after: float,
                                 duration_seconds: float = 0.0, recovery_bonus: float = 0.0):
        """Record stability change for recovery belief testing."""
        self._observe_recovery(stability_before, stability_after,
                               self._stability_episodes, "stability_recovery",
                               recovery_bonus=recovery_bonus)

    def observe_warmth_change(self, warmth_before: float, warmth_after: float,
                              duration_seconds: float = 0.0, recovery_bonus: float = 0.0):
        """Record warmth change for warmth_recovery belief testing."""
        self._observe_recovery(warmth_before, warmth_after,
                               self._warmth_episodes, "warmth_recovery",
                               recovery_bonus=recovery_bonus)

    def observe_question_asked(self, surprise_level: float):
        """Record that a curiosity question was generated after surprise.

        Tests whether Lumen tends to ask questions when surprised.
        High surprise + question asked = supporting evidence.
        """
        self._update_belief("question_asking_tendency",
            supports=True, strength=min(1.0, surprise_level))

    def observe_surprise_no_question(self, surprise_level: float):
        """Record that surprise occurred but no question was generated.

        Contradicting evidence for the question_asking_tendency belief.
        """
        if surprise_level > 0.2:  # Only count meaningful surprises
            self._update_belief("question_asking_tendency",
                supports=False, strength=min(1.0, surprise_level * 0.5))

    def observe_correlation(self, sensor_values: Dict[str, float], anima_values: Dict[str, float]):
        """Record data for correlation beliefs."""
        now = datetime.now()

        # Temperature-clarity correlation
        if "ambient_temp" in sensor_values and "clarity" in anima_values:
            self._correlation_data["temp_clarity"].append({
                "temp": sensor_values["ambient_temp"],
                "clarity": anima_values["clarity"],
                "timestamp": now,
            })
            self._test_correlation_belief("temp_clarity_correlation", "temp_clarity")

        # Light-warmth correlation
        if "light" in sensor_values and "warmth" in anima_values:
            self._correlation_data["light_warmth"].append({
                "light": sensor_values.get("light", sensor_values.get("light_lux", 0)),
                "warmth": anima_values["warmth"],
                "timestamp": now,
            })
            self._test_correlation_belief("light_warmth_correlation", "light_warmth")

    def observe_led_lux(self, led_brightness: Optional[float], light_lux: Optional[float]):
        """Track correlation between own LED brightness and lux readings.

        This is proprioceptive learning: discovering that one's own outputs
        affect one's own sensor inputs.
        """
        if led_brightness is None or light_lux is None:
            return

        now = datetime.now()

        # Record the data point
        self._correlation_data["led_lux"].append({
            "led": led_brightness,
            "lux": light_lux,
            "timestamp": now,
        })

        # Check for LED brightness change
        if self._prev_led_brightness is not None:
            led_change = led_brightness - self._prev_led_brightness

            if abs(led_change) > 0.05:  # Capture subtle brightness shifts too
                # Look at recent lux data to see if lux changed similarly
                led_lux_data = list(self._correlation_data["led_lux"])
                if len(led_lux_data) >= 3:
                    # Compare lux before and after the LED change
                    recent_lux = [d["lux"] for d in led_lux_data[-3:]]
                    older_lux = [d["lux"] for d in led_lux_data[-6:-3]] if len(led_lux_data) >= 6 else recent_lux

                    avg_recent = sum(recent_lux) / len(recent_lux)
                    avg_older = sum(older_lux) / len(older_lux)
                    lux_change = avg_recent - avg_older

                    # Did lux change in the same direction as LEDs?
                    same_direction = (led_change > 0 and lux_change > 0) or (led_change < 0 and lux_change < 0)

                    # Update belief
                    self._update_belief("my_leds_affect_lux",
                        supports=same_direction,
                        strength=min(1.0, abs(lux_change) / 10.0))
                    self._maybe_save()

        self._prev_led_brightness = led_brightness

        # Also test via correlation approach periodically
        if len(self._correlation_data["led_lux"]) >= 10:
            self._test_correlation_belief("my_leds_affect_lux", "led_lux")

    def _test_correlation_belief(self, belief_id: str, data_key: str):
        """Test a correlation belief against accumulated data."""
        if len(self._correlation_data[data_key]) < 10:
            return  # Not enough data

        data = list(self._correlation_data[data_key])
        keys = list(data[0].keys())
        keys.remove("timestamp")

        if len(keys) < 2:
            return

        x_key, y_key = keys[0], keys[1]
        x_values = [d[x_key] for d in data if d[x_key] is not None]
        y_values = [d[y_key] for d in data if d[y_key] is not None]

        if len(x_values) < 10 or len(y_values) < 10:
            return

        # Calculate correlation
        n = min(len(x_values), len(y_values))
        x = x_values[:n]
        y = y_values[:n]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        # Use epsilon to prevent division by near-zero values
        EPSILON = 1e-8
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        if sum_sq_x < EPSILON or sum_sq_y < EPSILON:
            return  # Values are constant or near-constant, no meaningful correlation

        # Only run full correlation test when there's real variance in input (CV > 5%).
        # In stable environments, most windows show noise not signal.
        cv_x = math.sqrt(sum_sq_x / n) / (abs(mean_x) + EPSILON)
        if cv_x < 0.05:
            # Stable input: no information. We used to log a weak disconfirm
            # here (supports=False, strength=0.05) on the theory that "if X
            # truly affected Y, we'd see co-variation even at small scales."
            # But in stable environments (HVAC-controlled rooms, resident
            # agents) this fires thousands of times with noise as the only
            # input, driving confidence to zero permanently even on beliefs
            # that might be true — there's just no data to test them.
            # Observed on Lumen: temp_clarity_correlation at 0 supporting /
            # 31,854 contradicting over weeks of indoor operation. Treat
            # "no variance" as "no evidence" and leave the belief at its
            # prior; clear the window so fresh data can accumulate.
            self._correlation_data[data_key].clear()
            return

        denom_x = math.sqrt(sum_sq_x)
        denom_y = math.sqrt(sum_sq_y)

        correlation = numerator / (denom_x * denom_y)

        # Update belief
        belief = self._beliefs[belief_id]

        # Strong correlation supports the belief
        if abs(correlation) > 0.3:
            self._update_belief(belief_id, supports=True, strength=abs(correlation))
            # Update value to reflect correlation direction
            belief.value = 0.5 + correlation * 0.5  # Map -1,1 to 0,1
        else:
            # Weak correlation contradicts — but only mildly, since we confirmed
            # there was real input variance
            self._update_belief(belief_id, supports=False, strength=0.3)
        self._maybe_save()

    def observe_interaction(self, clarity_before: float, clarity_after: float):
        """Record interaction for testing interaction-clarity belief."""
        clarity_change = clarity_after - clarity_before

        # Minimum strength ensures each observation moves the needle.
        # Clarity changes during a single interaction are typically tiny (0.001-0.02),
        # so without a floor the confidence barely moves from 0.5.
        strength = max(0.15, abs(clarity_change) * 2)

        self._update_belief("interaction_clarity_boost",
            supports=clarity_change > 0, strength=strength)
        self._maybe_save()

    def observe_time_pattern(self, hour: int, warmth: float, clarity: float):
        """Test time-based beliefs."""
        # Evening warmth (6pm-10pm)
        if 18 <= hour <= 22:
            self._update_belief("evening_warmth_increase",
                supports=warmth > 0.5, strength=abs(warmth - 0.5))

        # Morning clarity (6am-10am)
        if 6 <= hour <= 10:
            self._update_belief("morning_clarity",
                supports=clarity > 0.5, strength=abs(clarity - 0.5))

    def observe_temperament(self, temperament: Dict[str, float]):
        """Test temperament baseline beliefs using slow-moving averages.

        Called with inner life temperament values (already slow EMA).
        Samples every call but only tests beliefs when enough data accumulates.
        """
        self._temperament_samples.append(temperament)

        # Need enough samples for meaningful test (~1 min of data)
        if len(self._temperament_samples) < 15:
            return

        # Test warmth baseline
        warmth_vals = [s.get("warmth", 0.5) for s in self._temperament_samples]
        warmth_mean = sum(warmth_vals) / len(warmth_vals)
        self._update_belief("warmth_baseline_low",
            supports=warmth_mean < 0.40, strength=abs(warmth_mean - 0.40) * 2.0)

        # Test presence baseline
        presence_vals = [s.get("presence", 0.5) for s in self._temperament_samples]
        presence_mean = sum(presence_vals) / len(presence_vals)
        self._update_belief("presence_baseline_low",
            supports=presence_mean < 0.35, strength=abs(presence_mean - 0.35) * 2.0)

        self._maybe_save()

    def predict_own_response(self, context: str) -> Dict[str, float]:
        """Predict how Lumen will respond to a situation based on self-beliefs.

        Used by the self-prediction loop: predict before observing,
        then compare prediction to reality to sharpen beliefs.
        """
        predictions = {}

        if context == "light_change":
            predictions["surprise_likelihood"] = self._beliefs["light_sensitive"].value
            predictions["warmth_change"] = self._beliefs["light_warmth_correlation"].value

        elif context == "temp_change":
            predictions["surprise_likelihood"] = self._beliefs["temp_sensitive"].value
            predictions["clarity_change"] = self._beliefs["temp_clarity_correlation"].value

        elif context == "stability_drop":
            predictions["fast_recovery"] = self._beliefs["stability_recovery"].value

        return predictions

    def verify_prediction(self, context: str, prediction: Dict[str, float], actual: Dict[str, float]):
        """Compare self-prediction to reality and update belief confidence.

        If prediction was accurate, confidence in the underlying belief increases.
        If inaccurate, confidence decreases and value adjusts.
        """
        for key, predicted_value in prediction.items():
            actual_value = actual.get(key)
            if actual_value is None:
                continue

            error = abs(predicted_value - actual_value)
            accurate = error < 0.2  # Within 20% = accurate

            # Find which belief backs this prediction
            belief_id = None
            if context == "light_change" and key == "surprise_likelihood":
                belief_id = "light_sensitive"
            elif context == "light_change" and key == "warmth_change":
                belief_id = "light_warmth_correlation"
            elif context == "temp_change" and key == "surprise_likelihood":
                belief_id = "temp_sensitive"
            elif context == "temp_change" and key == "clarity_change":
                belief_id = "temp_clarity_correlation"
            elif context == "stability_drop" and key == "fast_recovery":
                belief_id = "stability_recovery"

            if belief_id and belief_id in self._beliefs:
                belief = self._beliefs[belief_id]
                if accurate:
                    # Prediction was right — boost confidence
                    belief.confidence = min(1.0, belief.confidence + 0.05)
                else:
                    # Prediction was wrong — nudge value toward reality
                    belief.confidence = max(0.0, belief.confidence - 0.03)
                    belief.value += (actual_value - predicted_value) * 0.1
                    belief.value = max(0.0, min(1.0, belief.value))

    def get_self_description(self, hearing_available: Optional[bool] = None) -> str:
        """Generate natural language self-description based on beliefs.

        Args:
            hearing_available: proprioceptive hearing state (Stage 0 of the
                hearing wire). When explicitly False, Lumen surfaces an
                understated note that it cannot currently hear — mute is a
                sensed fact, not a silent dead channel. When None (default)
                or True, hearing is not mentioned.
        """
        descriptions = []

        # Mute-as-sensed-state: understated, surfaced before learned beliefs.
        if hearing_available is False:
            descriptions.append("I cannot currently hear")

        for belief_id, belief in self._beliefs.items():
            if belief.confidence < 0.4:
                continue  # Not confident enough

            strength = belief.get_belief_strength()

            if belief_id == "light_sensitive":
                if belief.value > 0.6:
                    descriptions.append(f"I am {strength} that I'm sensitive to light changes")
                elif belief.value < 0.4:
                    descriptions.append(f"I am {strength} that light changes don't affect me much")

            elif belief_id == "stability_recovery":
                if belief.value > 0.6:
                    descriptions.append(f"I am {strength} that I recover stability quickly")
                elif belief.value < 0.4:
                    descriptions.append(f"I am {strength} that my stability takes time to recover")

            elif belief_id == "interaction_clarity_boost" and belief.value > 0.6:
                descriptions.append(f"I am {strength} that interactions help my clarity")

            elif belief_id == "my_leds_affect_lux":
                if belief.value > 0.6:
                    descriptions.append(f"I am {strength} that my own LEDs affect my light readings")
                elif belief.value < 0.4:
                    descriptions.append(f"I am {strength} that my light readings are from the environment, not my LEDs")

            elif belief_id == "warmth_baseline_low":
                if belief.value > 0.6:
                    descriptions.append(f"I am {strength} that my warmth tends to stay low")
                elif belief.value < 0.4:
                    descriptions.append(f"I am {strength} that my warmth baseline is comfortable")

            elif belief_id == "presence_baseline_low":
                if belief.value > 0.6:
                    descriptions.append(f"I am {strength} that my sense of presence tends to be low")
                elif belief.value < 0.4:
                    descriptions.append(f"I am {strength} that I generally feel present")

        if not descriptions:
            return "I am still learning about myself."

        return " ".join(descriptions) + "."

    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of all beliefs."""
        return {
            bid: {
                "description": b.description,
                "confidence": round(b.confidence, 3),
                "value": round(b.value, 3),
                "strength": b.get_belief_strength(),
                "evidence": f"{b.supporting_count}+ / {b.contradicting_count}-",
            }
            for bid, b in self._beliefs.items()
        }

    # ==================== Trajectory Components ====================
    # These methods extract data for trajectory signature computation.
    # See: trajectory-identity paper (cirwel/trajectory-identity-paper, separate repo)

    def get_belief_signature(self) -> Dict[str, Any]:
        """
        Extract belief signature (Β) for trajectory computation.

        Returns the pattern of self-beliefs: values, confidences, and evidence ratios.
        This reveals what the agent believes about itself and how certain it is.
        """
        beliefs = list(self._beliefs.values())

        values = [b.value for b in beliefs]
        confidences = [b.confidence for b in beliefs]
        evidence_ratios = [
            b.supporting_count / max(1, b.contradicting_count)
            for b in beliefs
        ]
        labels = [b.belief_id for b in beliefs]

        # Total evidence accumulated
        total_evidence = sum(b.supporting_count + b.contradicting_count for b in beliefs)

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "values": [round(v, 4) for v in values],
            "confidences": [round(c, 4) for c in confidences],
            "evidence_ratios": [round(r, 4) for r in evidence_ratios],
            "labels": labels,
            "total_evidence": total_evidence,
            "avg_confidence": round(avg_confidence, 4),
            "n_beliefs": len(beliefs),
        }

    def get_recovery_profile(self) -> Dict[str, Any]:
        """
        Extract recovery profile (Ρ) for trajectory computation.

        Estimates the characteristic time constant τ for returning to equilibrium
        after perturbation. This is computed from recorded stability episodes.
        """
        completed = [
            e for e in self._stability_episodes
            if e.get("recovered") and e.get("recovery_seconds")
        ]

        if not completed:
            return {
                "tau_estimate": None,
                "tau_std": None,
                "n_episodes": 0,
                "confidence": 0.0,
            }

        # Estimate tau from recovery episodes
        # Using exponential recovery model: x(t) = x_final - (x_final - x_0) * e^(-t/τ)
        # Rearranging: τ = -t / ln(1 - recovery_fraction)
        tau_estimates = []

        for ep in completed:
            initial = ep.get("initial", 0.7)
            dropped_to = ep.get("dropped_to", 0.5)
            drop = initial - dropped_to

            if drop <= 0.01:
                continue  # Not a real drop

            recovery_time = ep["recovery_seconds"]
            # Assume 63.2% recovery (one time constant) as typical
            recovery_fraction = min(0.95, 0.632)

            tau = -recovery_time / math.log(1 - recovery_fraction)
            if 0 < tau < 3600:  # Sanity check: 0-1 hour
                tau_estimates.append(tau)

        if not tau_estimates:
            return {
                "tau_estimate": None,
                "tau_std": None,
                "n_episodes": len(completed),
                "confidence": 0.0,
            }

        # Statistics
        tau_median = sorted(tau_estimates)[len(tau_estimates) // 2]
        tau_mean = sum(tau_estimates) / len(tau_estimates)

        if len(tau_estimates) > 1:
            tau_var = sum((t - tau_mean)**2 for t in tau_estimates) / len(tau_estimates)
            tau_std = tau_var ** 0.5
        else:
            tau_std = None

        # Confidence increases with number of episodes
        confidence = min(1.0, len(tau_estimates) / 10)

        return {
            "tau_estimate": round(tau_median, 2),
            "tau_mean": round(tau_mean, 2),
            "tau_std": round(tau_std, 2) if tau_std else None,
            "n_episodes": len(completed),
            "n_valid_estimates": len(tau_estimates),
            "confidence": round(confidence, 2),
        }

    def get_recovery_episodes(self) -> List[Dict[str, Any]]:
        """Get all recovery episodes for analysis."""
        return list(self._stability_episodes)

    def save(self):
        """Explicitly save the model."""
        self._save()


# Singleton instance
_self_model: Optional[SelfModel] = None


def get_self_model() -> SelfModel:
    """Get or create the self-model."""
    global _self_model
    if _self_model is None:
        _self_model = SelfModel()
    return _self_model
