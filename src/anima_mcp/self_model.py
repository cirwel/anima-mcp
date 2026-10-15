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
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from collections import deque
import json
from pathlib import Path
import math
from statistics import NormalDist, fmean, pstdev

from .atomic_write import atomic_json_write
from .light_attribution import LearnedLedLuxResidual


# The broker owns the live causal calibration while the MCP/server process is a
# refreshing reader.  Checkpoint often enough to bound crash loss and reader
# fallback staleness without writing the Pi's storage on every two-second tick.
LIGHT_ATTRIBUTION_CHECKPOINT_SECONDS = 300.0

# Correlation beliefs are observational and sampled on the creature's fast
# loop, so a ten-sample window is both noisy and badly overconfident. Build a
# full, non-overlapping observation window, tolerate some missing pairs, and
# require a conservative two-sided Fisher-z result before treating an effect
# as present. The effect-size floor keeps statistically detectable trivia from
# becoming self-knowledge as sample sizes grow.
CORRELATION_WINDOW_SIZE = 50
CORRELATION_MIN_COMPLETE_PAIRS = 30
CORRELATION_MIN_CV = 0.05
CORRELATION_SIGNIFICANCE_ALPHA = 0.01
CORRELATION_MIN_EFFECT = 0.3
_CORRELATION_Z_CRITICAL = NormalDist().inv_cdf(
    1.0 - CORRELATION_SIGNIFICANCE_ALPHA / 2.0
)


# Time-of-day beliefs claim "higher in this window than the rest of my day".
# Until 2026-10-14 they were tested against fixed cuts (clarity > 0.5 in the
# morning, warmth > 0.5 in the evening), so with clarity living around 0.7 the
# morning belief collected unbroken support without ever comparing mornings
# with anything. Each window is now judged once, after it ends, against the hourly
# means of the 24 hours leading up to its close, outside the window.
# belief_id -> (evidence bucket key, anima dimension, first hour, last hour),
# hours inclusive.
TIME_PATTERN_WINDOWS = {
    "morning_clarity": ("time:morning_clarity", "clarity", 6, 10),
    "evening_warmth_increase": ("time:evening_warmth", "warmth", 18, 22),
}
# Evidence-quantity floors, not behavioral cuts: an hour with no samples is
# unknown, and a window judged from one or two hours describes a moment.
TIME_PATTERN_MIN_WINDOW_HOURS = 3
TIME_PATTERN_MIN_REFERENCE_HOURS = 8
# Hourly means older than this are pruned; it covers the latest window's
# reference day plus slack for a broker that was down when the window closed.
TIME_PATTERN_RETAIN_HOURS = 48


def _correlation_support_threshold(sample_count: int) -> float:
    """Return the minimum |r| that is meaningful and significant for ``n``.

    Fisher's z transform gives an accurate dependency-free approximation for
    the configured sample range (30-50 pairs). A perfect correlation remains
    admissible for the degenerate small-n branch, though runtime callers gate
    those windows before reaching this helper.
    """
    if sample_count <= 3:
        return 1.0
    significance_threshold = math.tanh(
        _CORRELATION_Z_CRITICAL / math.sqrt(sample_count - 3)
    )
    return max(CORRELATION_MIN_EFFECT, significance_threshold)


def _learning_multiplier(update_bonus: float) -> float:
    """Return a finite, non-negative multiplier for experiential learning."""
    try:
        bonus = float(update_bonus)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(bonus):
        return 1.0
    return max(0.0, 1.0 + bonus)


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
        strength = float(strength)
        if not math.isfinite(strength) or strength <= 0.0:
            return
        strength = min(1.0, strength)

        self.last_tested = datetime.now()
        bonus_multiplier = _learning_multiplier(update_bonus)
        lr = 0.1 * bonus_multiplier

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

    def update_correlation(
        self,
        correlation: float,
        sample_count: int,
        update_bonus: float = 0.0,
    ) -> None:
        """Aggregate one signed, sample-sized window without overwriting history.

        Correlation magnitude is evidence that an effect exists; its sign is
        represented by ``value`` (0.5 neutral, 0 negative, 1 positive). A new
        window therefore updates confidence and moves value toward its signed
        target with an evidence-weighted EMA. Effect support is gated by an
        n-aware significance threshold rather than a fixed |r| cutoff.
        """
        correlation = float(correlation)
        if not math.isfinite(correlation) or sample_count <= 3:
            return
        correlation = max(-1.0, min(1.0, correlation))
        magnitude = abs(correlation)
        self.last_tested = datetime.now()
        bonus_multiplier = _learning_multiplier(update_bonus)
        sample_weight = min(1.0, sample_count / CORRELATION_WINDOW_SIZE)

        if magnitude >= _correlation_support_threshold(sample_count):
            self.supporting_count += 1
            confidence_lr = min(
                1.0,
                0.1 * bonus_multiplier * magnitude * sample_weight,
            )
            self.confidence = min(
                1.0, self.confidence + confidence_lr * (1.0 - self.confidence)
            )
            target = 0.5 + correlation * 0.5
            value_lr = min(
                0.5,
                0.25 * bonus_multiplier * magnitude * sample_weight,
            )
            self.value += value_lr * (target - self.value)
        else:
            self.contradicting_count += 1
            confidence_lr = min(1.0, 0.03 * bonus_multiplier * sample_weight)
            self.confidence = max(
                0.0, self.confidence - confidence_lr * self.confidence
            )
            neutral_lr = min(
                0.5,
                0.1 * bonus_multiplier * (1.0 - magnitude) * sample_weight,
            )
            self.value += neutral_lr * (0.5 - self.value)

        self.value = max(0.0, min(1.0, self.value))

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

    # These beliefs were historically tested on every 2-second loop tick.  A
    # one-time cold start is more honest than preserving millions of correlated
    # pseudo-observations after switching to episode/time buckets.
    _EPISODE_MIGRATION_DEFAULTS = {
        "light_sensitive": (0.5, 0.5),
        "temp_sensitive": (0.5, 0.5),
        "temp_clarity_correlation": (0.5, 0.5),
        "light_warmth_correlation": (0.5, 0.5),
        "my_leds_affect_lux": (0.5, 0.5),
        "evening_warmth_increase": (0.3, 0.5),
        "morning_clarity": (0.3, 0.5),
        "warmth_baseline_low": (0.3, 0.5),
        "presence_baseline_low": (0.3, 0.5),
    }

    # 2026-08-21 audit: these four beliefs had evidence channels that did not
    # deliver in the deployed topology. The v3 migration below preserves and
    # cold-starts their historical artifacts once; current runtime wiring now
    # feeds recovery observations in the broker and communication evidence via
    # the durable learning inbox. Their prior
    # stored values are artifacts of the retired 2s-tick era (counts
    # byte-identical across every snapshot since 8-11/12); recomputation from
    # state_history contradicts the recovery pair at every observable
    # timescale. A frozen near-saturated value is worse than an honest prior —
    # so each belief cold-starts to ITS OWN constructor prior (the 0.7
    # hypothesis seeds for interaction/questions are deliberate design priors,
    # not learned values, and survive the reset as priors).
    _DEAD_CHANNEL_RESET_V3 = {
        "warmth_recovery": (0.5, 0.5),
        "stability_recovery": (0.5, 0.5),
        "interaction_clarity_boost": (0.5, 0.7),
        "question_asking_tendency": (0.5, 0.7),
    }

    # Raw VEML7700 lux mixes room light with DotStar self-glow. Historical
    # light→warmth evidence therefore cannot support an environmental belief.
    # V4 cold-starts that one channel; the broker now supplies only the gated
    # learned external-light residual to this correlation test.
    _LIGHT_ATTRIBUTION_RESET_V4 = {
        "light_warmth_correlation": (0.5, 0.5),
    }

    # Clarity's live equation changed from mixed raw lux to a gated external
    # residual, and its CPU-derived alpha term is now explicitly disabled by
    # calibration. Beliefs whose outcome variable is clarity cannot carry
    # evidence across that semantic boundary as if the measurement were the
    # same. Preserve the old state in an audit field and relearn them.
    _CLARITY_SEMANTICS_RESET_V5 = {
        "temp_clarity_correlation": (0.5, 0.5),
        "interaction_clarity_boost": (0.5, 0.7),
        "morning_clarity": (0.3, 0.5),
    }

    # The explanatory LED→lux belief was still learning from raw, closed-loop
    # LED/room-light correlation after the quantitative residual moved to an
    # instrumented breathing-pulse model. That lets activity-driven logical
    # brightness changes reinforce the narrative belief without identifying a
    # causal sensor response. V6 retires those counts; only a ready causal
    # attribution model may now add evidence to this belief.
    _LED_CAUSAL_EVIDENCE_RESET_V6 = {
        "my_leds_affect_lux": (0.5, 0.5),
    }

    # The time-of-day beliefs were tested against fixed cuts rather than
    # against the rest of Lumen's day (see TIME_PATTERN_WINDOWS). Counts from
    # that test answer a different question, so V7 retires them.
    _TIME_PATTERN_RELATIVE_RESET_V7 = {
        "morning_clarity": (0.3, 0.5),
        "evening_warmth_increase": (0.3, 0.5),
    }

    def __init__(self, persistence_path: Optional[Path] = None,
                 read_only: bool = False):
        self.persistence_path = persistence_path or Path.home() / ".anima" / "self_model.json"
        self.read_only = read_only
        self._loaded_mtime_ns = 0
        self._evidence_buckets: Dict[str, str] = {}
        self._applied_event_ids: List[str] = []
        # "YYYY-MM-DDTHH" -> {dimension: [sum, count]}, for time-of-day beliefs.
        self._time_pattern_hours: Dict[str, Dict[str, List[float]]] = {}

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
            "temp_clarity": deque(maxlen=CORRELATION_WINDOW_SIZE),
            "light_warmth": deque(maxlen=CORRELATION_WINDOW_SIZE),
            # Retained only so legacy/private diagnostics fail closed and can
            # clear stale samples. Runtime evidence no longer enters here.
            "led_lux": deque(maxlen=CORRELATION_WINDOW_SIZE),
        }
        self._surprise_data: deque = deque(maxlen=50)  # (source, surprise_level)
        self._temperament_samples: deque = deque(maxlen=30)  # Recent temperament snapshots
        self.belief_update_bonus: float = 0.0  # From experiential marks
        # Calibration is the evidence authority for the explanatory belief.
        # The belief remains a hypothesis until this causal model is ready.
        self._light_attribution_model = LearnedLedLuxResidual()

        # Load persisted model

        self._load()

    def _update_belief(self, belief_id: str, supports: bool, strength: float = 1.0):
        """Update a belief, automatically applying belief_update_bonus."""
        if self.read_only:
            return
        self._beliefs[belief_id].update_from_evidence(
            supports=supports, strength=strength,
            update_bonus=self.belief_update_bonus,
        )

    @property
    def is_writable(self) -> bool:
        return not self.read_only

    def apply_evidence(self, belief_id: str, *, supports: bool,
                       strength: float = 1.0) -> None:
        """Apply one explicit evidence episode from the durable event inbox."""
        if belief_id not in self._beliefs:
            raise ValueError(f"unknown self-belief: {belief_id}")
        self._update_belief(belief_id, supports=supports, strength=strength)

    def _claim_evidence_bucket(self, key: str, bucket: str) -> bool:
        """Claim one evidence opportunity per semantic time bucket."""
        if self.read_only or self._evidence_buckets.get(key) == bucket:
            return False
        self._evidence_buckets[key] = bucket
        return True

    def _apply_persisted_data(self, data: dict) -> None:
        for belief_id, bdata in data.get("beliefs", {}).items():
            if belief_id in self._beliefs:
                b = self._beliefs[belief_id]
                b.confidence = bdata.get("confidence", 0.5)
                b.value = bdata.get("value", 0.5)
                b.supporting_count = bdata.get("supporting_count", 0)
                b.contradicting_count = bdata.get("contradicting_count", 0)
        buckets = data.get("evidence_buckets", {})
        if isinstance(buckets, dict):
            self._evidence_buckets = {
                str(key): str(value) for key, value in buckets.items()
            }
        event_ids = data.get("applied_event_ids", [])
        if isinstance(event_ids, list):
            self._applied_event_ids = [str(value) for value in event_ids[-2000:]]
        self._light_attribution_model.load_dict(data.get("light_attribution_model"))
        # Retain the dead-channel audit trail across load/save cycles.
        self._dead_channel_audit = data.get("_migrated_dead_channel_reset_v3", None)
        self._light_channel_audit = data.get(
            "_migrated_light_attribution_reset_v4", None
        )
        self._clarity_semantics_audit = data.get(
            "_migrated_clarity_semantics_reset_v5", None
        )
        self._led_causal_evidence_audit = data.get(
            "_migrated_led_causal_evidence_reset_v6", None
        )
        self._time_pattern_audit = data.get(
            "_migrated_time_pattern_relative_reset_v7", None
        )
        hours = data.get("time_pattern_hours", {})
        self._time_pattern_hours = {}
        if isinstance(hours, dict):
            for key, dims in hours.items():
                if not isinstance(dims, dict):
                    continue
                kept = {}
                for dim, pair in dims.items():
                    try:
                        total, count = float(pair[0]), int(pair[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if math.isfinite(total) and count > 0:
                        kept[str(dim)] = [total, count]
                if kept:
                    self._time_pattern_hours[str(key)] = kept

    def _load(self):
        """Load self-model from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                self._apply_persisted_data(data)

                # Migration: reset beliefs corrupted by testing noise as evidence.
                # Pre-fix, every 2s observation tested correlation even when input
                # was constant, logging "contradicting" 100K+ times. Reset these
                # to fresh state so they can learn honestly with the CV>5% gate.
                migrated = False
                if not self.read_only and not data.get("_migrated_noise_reset"):
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
                    migrated = True

                if not self.read_only and not data.get("_migrated_episode_evidence_v2"):
                    for bid, (confidence, value) in self._EPISODE_MIGRATION_DEFAULTS.items():
                        belief = self._beliefs[bid]
                        belief.confidence = confidence
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    self._evidence_buckets.clear()
                    migrated = True

                if not self.read_only and not data.get("_migrated_dead_channel_reset_v3"):
                    # The pre-reset state is stashed INSIDE the flag (a truthy
                    # dict) so the reset stays auditable from the file itself,
                    # like the kb migration's legacy_confidence.
                    audit = {}
                    for bid, (conf, value) in self._DEAD_CHANNEL_RESET_V3.items():
                        belief = self._beliefs.get(bid)
                        if belief is None:
                            continue
                        audit[bid] = {
                            "confidence": belief.confidence,
                            "value": belief.value,
                            "supporting_count": belief.supporting_count,
                            "contradicting_count": belief.contradicting_count,
                        }
                        print(f"[SelfModel] Cold-starting dead-channel belief '{bid}' "
                              f"(+{belief.supporting_count}/-{belief.contradicting_count}, "
                              f"v={belief.value:.3f}, c={belief.confidence:.3f})",
                              flush=True)
                        belief.confidence = conf
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    self._dead_channel_audit = audit or True
                    migrated = True

                if not self.read_only and not data.get(
                    "_migrated_light_attribution_reset_v4"
                ):
                    audit = {}
                    for bid, (conf, value) in self._LIGHT_ATTRIBUTION_RESET_V4.items():
                        belief = self._beliefs.get(bid)
                        if belief is None:
                            continue
                        audit[bid] = {
                            "confidence": belief.confidence,
                            "value": belief.value,
                            "supporting_count": belief.supporting_count,
                            "contradicting_count": belief.contradicting_count,
                        }
                        belief.confidence = conf
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    self._correlation_data["light_warmth"].clear()
                    self._light_channel_audit = audit or True
                    migrated = True

                if not self.read_only and not data.get(
                    "_migrated_clarity_semantics_reset_v5"
                ):
                    audit = {}
                    for bid, (conf, value) in self._CLARITY_SEMANTICS_RESET_V5.items():
                        belief = self._beliefs.get(bid)
                        if belief is None:
                            continue
                        audit[bid] = {
                            "confidence": belief.confidence,
                            "value": belief.value,
                            "supporting_count": belief.supporting_count,
                            "contradicting_count": belief.contradicting_count,
                        }
                        belief.confidence = conf
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    self._correlation_data["temp_clarity"].clear()
                    self._evidence_buckets.pop(
                        "correlation:temp_clarity_correlation", None
                    )
                    self._evidence_buckets.pop("time:morning_clarity", None)
                    self._clarity_semantics_audit = audit or True
                    migrated = True

                if not self.read_only and not data.get(
                    "_migrated_led_causal_evidence_reset_v6"
                ):
                    audit = {}
                    for bid, (conf, value) in (
                        self._LED_CAUSAL_EVIDENCE_RESET_V6.items()
                    ):
                        belief = self._beliefs.get(bid)
                        if belief is None:
                            continue
                        audit[bid] = {
                            "confidence": belief.confidence,
                            "value": belief.value,
                            "supporting_count": belief.supporting_count,
                            "contradicting_count": belief.contradicting_count,
                            "retired_evidence": "raw_closed_loop_led_lux_correlation",
                        }
                        belief.confidence = conf
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    self._correlation_data["led_lux"].clear()
                    self._evidence_buckets.pop("led_lux:change", None)
                    self._evidence_buckets.pop(
                        "correlation:my_leds_affect_lux", None
                    )
                    self._led_causal_evidence_audit = audit or True
                    migrated = True

                if not self.read_only and not data.get(
                    "_migrated_time_pattern_relative_reset_v7"
                ):
                    audit = {}
                    for bid, (conf, value) in (
                        self._TIME_PATTERN_RELATIVE_RESET_V7.items()
                    ):
                        belief = self._beliefs.get(bid)
                        if belief is None:
                            continue
                        audit[bid] = {
                            "confidence": belief.confidence,
                            "value": belief.value,
                            "supporting_count": belief.supporting_count,
                            "contradicting_count": belief.contradicting_count,
                            "retired_evidence": "absolute_threshold_in_window",
                        }
                        belief.confidence = conf
                        belief.value = value
                        belief.supporting_count = 0
                        belief.contradicting_count = 0
                    for key, *_ in TIME_PATTERN_WINDOWS.values():
                        self._evidence_buckets.pop(key, None)
                    self._time_pattern_audit = audit or True
                    migrated = True

                if migrated:
                    self._save()

                self._loaded_mtime_ns = self.persistence_path.stat().st_mtime_ns

            except Exception as e:
                print(f"[SelfModel] Could not load: {e}")

    def refresh_if_changed(self, *, force: bool = False) -> bool:
        """Refresh a reader from the broker-owned snapshot when it changes."""
        if not self.persistence_path.exists():
            return False
        try:
            mtime_ns = self.persistence_path.stat().st_mtime_ns
            if not force and mtime_ns == self._loaded_mtime_ns:
                return False
            data = json.loads(self.persistence_path.read_text())
            self._apply_persisted_data(data)
            self._loaded_mtime_ns = mtime_ns
            return True
        except Exception as e:
            print(f"[SelfModel] Could not refresh: {e}")
            return False

    def _maybe_save(self, min_interval_seconds: float = 10.0) -> None:
        """Save if enough time has passed since last save (throttle for high-value updates)."""
        if not hasattr(self, "_last_save_time"):
            self._last_save_time = 0.0
        now = datetime.now().timestamp()
        if now - self._last_save_time >= min_interval_seconds:
            if self._save():
                self._last_save_time = now

    def _save(self) -> bool:
        """Save self-model to disk."""
        if self.read_only:
            return False
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
                "evidence_buckets": self._evidence_buckets,
                "applied_event_ids": self._applied_event_ids[-2000:],
                "time_pattern_hours": self._time_pattern_hours,
                "light_attribution_model": self._light_attribution_model.to_dict(),
                "_migrated_noise_reset": True,
                "_migrated_episode_evidence_v2": True,
                # Truthy dict carrying the pre-reset state (audit trail), or
                # True when there was nothing to reset. Preserved across loads.
                "_migrated_dead_channel_reset_v3":
                    getattr(self, "_dead_channel_audit", None) or True,
                "_migrated_light_attribution_reset_v4":
                    getattr(self, "_light_channel_audit", None) or True,
                "_migrated_clarity_semantics_reset_v5":
                    getattr(self, "_clarity_semantics_audit", None) or True,
                "_migrated_led_causal_evidence_reset_v6":
                    getattr(self, "_led_causal_evidence_audit", None) or True,
                "_migrated_time_pattern_relative_reset_v7":
                    getattr(self, "_time_pattern_audit", None) or True,
            }
            atomic_json_write(self.persistence_path, data, indent=2)
            try:
                self._loaded_mtime_ns = self.persistence_path.stat().st_mtime_ns
            except OSError:
                # The atomic write committed; force a future reader refresh if
                # metadata lookup is transiently unavailable.
                self._loaded_mtime_ns = 0
            return True
        except Exception as e:
            print(f"[SelfModel] Could not save: {e}")
            return False

    def has_applied_event(self, event_id: str) -> bool:
        """Whether a durable inbox event was committed with this snapshot."""
        return event_id in self._applied_event_ids

    def mark_applied_event(self, event_id: str) -> None:
        """Record an inbox receipt in the same atomic save as its mutation."""
        if event_id not in self._applied_event_ids:
            self._applied_event_ids.append(event_id)
            del self._applied_event_ids[:-2000]

    def forget_applied_event(self, event_id: str) -> None:
        """Roll back an uncommitted receipt after a failed snapshot save."""
        if event_id in self._applied_event_ids:
            self._applied_event_ids.remove(event_id)

    def observe_surprise(self, surprise_level: float, sources: List[str]):
        """Record a surprise observation for belief testing."""
        now = datetime.now()
        self._surprise_data.append({
            "timestamp": now,
            "surprise": surprise_level,
            "sources": sources,
        })

        # A sustained surprise is one episode, not fresh evidence every two
        # seconds.  Five-minute source buckets retain responsiveness without
        # making loop cadence determine certainty.
        bucket = str(int(now.timestamp() // 300))
        if "light" in sources and self._claim_evidence_bucket("surprise:light", bucket):
            # High surprise from light suggests high sensitivity
            self._update_belief("light_sensitive",
                supports=surprise_level > 0.3, strength=surprise_level)

        if ("ambient_temp" in sources
                and self._claim_evidence_bucket("surprise:ambient_temp", bucket)):
            self._update_belief("temp_sensitive",
                supports=surprise_level > 0.3, strength=surprise_level)

    def _observe_recovery(
        self,
        before: float,
        after: float,
        episodes: deque,
        belief_id: str,
        duration_seconds: float = 0.0,
        recovery_bonus: float = 0.0,
    ):
        """Shared recovery-belief observer for any anima dimension.

        Tracks drop/recovery episodes and tests the named belief.
        Fast recovery (< threshold s per unit recovered) = supporting evidence.
        recovery_bonus: from experiential marks (stability_recovery_bonus),
            widens the threshold so more recoveries count as "fast".
        """
        try:
            elapsed = float(duration_seconds)
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed):
            elapsed = 0.0
        elapsed = max(0.0, elapsed)

        active = next(
            (episode for episode in reversed(episodes) if not episode.get("recovered")),
            None,
        )
        if active is not None:
            active["elapsed_seconds"] = active.get("elapsed_seconds", 0.0) + elapsed

        if before > after:
            if active is None:
                episodes.append({
                    "drop_time": datetime.now(),
                    "initial": before,
                    "dropped_to": after,
                    "elapsed_seconds": 0.0,
                    "recovered": False,
                })
            elif after < active["dropped_to"]:
                # A gradual decline is one episode. Recovery starts at the
                # deepest observed trough, not at every small downward step.
                active["dropped_to"] = after
                active["drop_time"] = datetime.now()
                active["elapsed_seconds"] = 0.0
            return

        if after <= before or active is None:
            return

        recovery_amount = after - active["dropped_to"]
        if recovery_amount <= 0.1:
            return

        recovery_time = active.get("elapsed_seconds", 0.0)
        active["recovered"] = True
        active["recovery_seconds"] = recovery_time

        threshold = 600 * _learning_multiplier(recovery_bonus)
        is_fast = recovery_time / max(0.1, recovery_amount) < threshold
        self._update_belief(
            belief_id,
            supports=is_fast,
            strength=recovery_amount,
        )
        self._maybe_save()

    def observe_stability_change(self, stability_before: float, stability_after: float,
                                 duration_seconds: float = 0.0, recovery_bonus: float = 0.0):
        """Record stability change for recovery belief testing."""
        self._observe_recovery(stability_before, stability_after,
                               self._stability_episodes, "stability_recovery",
                               duration_seconds=duration_seconds,
                               recovery_bonus=recovery_bonus)

    def observe_warmth_change(self, warmth_before: float, warmth_after: float,
                              duration_seconds: float = 0.0, recovery_bonus: float = 0.0):
        """Record warmth change for warmth_recovery belief testing."""
        self._observe_recovery(warmth_before, warmth_after,
                               self._warmth_episodes, "warmth_recovery",
                               duration_seconds=duration_seconds,
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

    def observe_led_lux(
        self,
        led_brightness: Optional[float],
        light_lux: Optional[float],
        *,
        led_proprioception: Optional[Dict[str, Any]] = None,
        brightness_source: str = "broker_brightness_estimate",
        observed_at: Optional[float] = None,
        observation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the causal LED/lux residual and its explanatory belief.

        This is proprioceptive learning: discovering that one's own outputs
        affect one's own sensor inputs. Raw LED/lux correlation is deliberately
        not evidence here: activity can drive both channels. Only a ready
        internally instrumented breathing-pulse model may reinforce the
        explanatory belief.
        """
        if isinstance(led_proprioception, dict):
            led_state = dict(led_proprioception)
        else:
            led_state = {
                "brightness": led_brightness,
                "source": brightness_source,
            }
        attribution = self._light_attribution_model.observe(
            light_lux,
            led_state,
            observed_at=observed_at,
            observation_id=observation_id,
            learn=not self.read_only,
        )

        self._observe_ready_led_attribution_belief(
            attribution,
        )
        self._maybe_save(
            min_interval_seconds=LIGHT_ATTRIBUTION_CHECKPOINT_SECONDS
        )

        return attribution

    def _observe_ready_led_attribution_belief(
        self,
        attribution: Dict[str, Any],
    ) -> None:
        """Score one belief episode per hour containing fresh causal evidence.

        Until 2026-10-14 this could only ever support: a ready model credited
        the belief and every other state returned, so the tally could not lose
        and an N:0 record said nothing. Now the instrument may also contradict
        it: with full coverage (enough balanced breathing differences over a
        wide enough drive span) and a non-positive equal-direction slope, the
        breathing pulse was measured and the sensor did not follow it.
        Strength mirrors the model's sign score from the other side, so a mixed
        sign record (half or more positive) is unknown and counts as nothing.
        Every other unready state is still unknown, not evidence either way.
        """
        model = attribution.get("model", {}) if isinstance(attribution, dict) else {}
        if not isinstance(model, dict):
            return
        slope = model.get("slope_lux_per_drive")
        try:
            slope = float(slope)
        except (TypeError, ValueError):
            return
        if not math.isfinite(slope):
            return
        if model.get("ready"):
            if slope <= 0.0:
                return
            supports = True
            try:
                strength = float(model.get("confidence"))
            except (TypeError, ValueError):
                return
        elif model.get("coverage_ready") and slope <= 0.0:
            supports = False
            try:
                positive_fraction = float(model.get("positive_fraction"))
            except (TypeError, ValueError):
                return
            strength = (0.5 - positive_fraction) / 0.5
        else:
            return
        if not math.isfinite(strength) or strength <= 0.0:
            return
        timestamp = model.get("latest_transition_at_unix")
        try:
            timestamp = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp = None
        if timestamp is None or not math.isfinite(timestamp):
            return
        hour_bucket = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H")
        if not self._claim_evidence_bucket(
            "led_lux:causal_ready_model", hour_bucket
        ):
            return
        self._update_belief(
            "my_leds_affect_lux",
            supports=supports,
            strength=min(1.0, strength),
        )
        self._maybe_save()

    def get_light_attribution_model_stats(self) -> Dict[str, Any]:
        """Return the causal model state that governs LED/lux claims."""
        return dict(self._light_attribution_model.model_stats())

    def get_light_attribution(
        self,
        light_lux: Optional[float],
        *,
        led_proprioception: Optional[Dict[str, Any]] = None,
        led_brightness: Optional[float] = None,
        brightness_source: str = "broker_brightness_estimate",
    ) -> Dict[str, Any]:
        """Describe current gated attribution without adding evidence."""
        if isinstance(led_proprioception, dict):
            led_state = dict(led_proprioception)
        else:
            led_state = {
                "brightness": led_brightness,
                "source": brightness_source,
            }
        return self._light_attribution_model.attribute(light_lux, led_state)

    def _test_correlation_belief(self, belief_id: str, data_key: str):
        """Test a correlation belief against accumulated data."""
        if belief_id == "my_leds_affect_lux" or data_key == "led_lux":
            # Fail closed if an old/private caller tries to reopen the retired
            # closed-loop path. LED→lux evidence is owned exclusively by the
            # instrumented breathing-pulse attribution model above.
            data = self._correlation_data.get(data_key)
            if data is not None:
                data.clear()
            return
        if len(self._correlation_data[data_key]) < CORRELATION_WINDOW_SIZE:
            return  # Build one full, non-overlapping observation window.

        data = list(self._correlation_data[data_key])
        keys = list(data[0].keys())
        keys.remove("timestamp")

        if len(keys) < 2:
            return

        x_key, y_key = keys[0], keys[1]
        # Filter as PAIRS, not as two independent series.
        #
        # The previous form built x_values and y_values with separate
        # comprehensions, each dropping its own Nones, then truncated both to
        # min(len) and zipped positionally. Whenever either channel had a gap
        # the two lists desynchronised, so x[i] was correlated against a y[i]
        # recorded at a DIFFERENT timestamp. The result was not noisier — it
        # was a correlation between misaligned series, which can land anywhere
        # including a confident wrong sign, and nothing downstream could tell.
        # A sensor that reads None intermittently is the normal case here, not
        # an edge case.
        pairs = [(d[x_key], d[y_key]) for d in data
                 if d[x_key] is not None and d[y_key] is not None]

        if len(pairs) < CORRELATION_MIN_COMPLETE_PAIRS:
            return

        # Calculate correlation
        n = len(pairs)
        x = [a for a, _ in pairs]
        y = [b for _, b in pairs]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        # Use epsilon to prevent division by near-zero values
        EPSILON = 1e-8
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        if sum_sq_x < EPSILON or sum_sq_y < EPSILON:
            # Constant channels carry no evidence. Discard the completed
            # window so the fast loop does not retest an overlapping buffer on
            # every subsequent sample.
            self._correlation_data[data_key].clear()
            return

        # Only run a correlation test when both channels vary meaningfully.
        # Quantization noise in either series can otherwise look perfectly
        # correlated despite carrying no useful evidence.
        cv_x = math.sqrt(sum_sq_x / n) / (abs(mean_x) + EPSILON)
        cv_y = math.sqrt(sum_sq_y / n) / (abs(mean_y) + EPSILON)
        if min(cv_x, cv_y) < CORRELATION_MIN_CV:
            # Stable channels are no evidence either way. Clear the window so
            # fresh variation can accumulate without eroding the prior.
            self._correlation_data[data_key].clear()
            return

        denom_x = math.sqrt(sum_sq_x)
        denom_y = math.sqrt(sum_sq_y)

        correlation = numerator / (denom_x * denom_y)

        # Sliding windows overlap almost completely from tick to tick.  Credit
        # at most one informative window per clock hour, then clear the window
        # so the next test is built from new observations.
        bucket = datetime.now().strftime("%Y-%m-%dT%H")
        if not self._claim_evidence_bucket(f"correlation:{belief_id}", bucket):
            self._correlation_data[data_key].clear()
            return

        self._beliefs[belief_id].update_correlation(
            correlation,
            sample_count=n,
            update_bonus=self.belief_update_bonus,
        )
        self._correlation_data[data_key].clear()
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

    def observe_time_pattern(self, hour: int, warmth: float, clarity: float,
                             *, now: Optional[datetime] = None):
        """Accumulate hourly means and test time-of-day beliefs against them.

        Each belief claims its window runs higher than the rest of Lumen's day,
        so evidence is relative: once the window has closed, its mean of hourly
        means is compared with the hourly means of the 24 hours leading up to
        its close, outside the window. Strength is that difference in units of
        the reference hours' own spread. Too few hours on either side, or a
        flat reference, is unknown and scores nothing.
        """
        when = (now or datetime.now()).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        bucket = self._time_pattern_hours.setdefault(
            when.strftime("%Y-%m-%dT%H"), {})
        for dim, value in (("warmth", warmth), ("clarity", clarity)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            pair = bucket.setdefault(dim, [0.0, 0])
            pair[0] += value
            pair[1] += 1
        oldest = (when - timedelta(hours=TIME_PATTERN_RETAIN_HOURS)).strftime(
            "%Y-%m-%dT%H")
        for key in [k for k in self._time_pattern_hours if k < oldest]:
            del self._time_pattern_hours[key]

        for belief_id, (key, dim, start, end) in TIME_PATTERN_WINDOWS.items():
            self._test_time_window(belief_id, key, dim, start, end, when)

    def _hourly_mean(self, key: str, dim: str) -> Optional[float]:
        pair = self._time_pattern_hours.get(key, {}).get(dim)
        if not pair or pair[1] <= 0:
            return None
        return pair[0] / pair[1]

    def _test_time_window(self, belief_id: str, evidence_key: str, dim: str,
                          start: int, end: int, when: datetime) -> None:
        """Judge the most recent closed window once (see observe_time_pattern)."""
        # The window closes at the top of the hour after `end`. Before that
        # today, the latest closed window is yesterday's.
        close = when.replace(hour=end) + timedelta(hours=1)
        if close > when:
            close -= timedelta(days=1)
        window_day = (close - timedelta(hours=1)).date().isoformat()
        if self.read_only or self._evidence_buckets.get(evidence_key) == window_day:
            return
        window, reference = [], []
        for offset in range(1, 25):
            hour_at = close - timedelta(hours=offset)
            mean = self._hourly_mean(hour_at.strftime("%Y-%m-%dT%H"), dim)
            if mean is None:
                continue
            (window if start <= hour_at.hour <= end else reference).append(mean)
        if (len(window) < TIME_PATTERN_MIN_WINDOW_HOURS
                or len(reference) < TIME_PATTERN_MIN_REFERENCE_HOURS):
            return
        spread = pstdev(reference)
        if not math.isfinite(spread) or spread <= 1e-9:
            return
        z = (fmean(window) - fmean(reference)) / spread
        if not math.isfinite(z) or z == 0.0:
            return
        self._claim_evidence_bucket(evidence_key, window_day)
        self._update_belief(belief_id, supports=z > 0.0,
                            strength=min(1.0, abs(z) / 2.0))

    def observe_temperament(self, temperament: Dict[str, float]):
        """Test temperament baseline beliefs using slow-moving averages.

        Called with inner life temperament values (already slow EMA).
        Samples every call but only tests beliefs when enough data accumulates.
        """
        self._temperament_samples.append(temperament)

        # Need enough samples for meaningful test (~1 min of data)
        if len(self._temperament_samples) < 15:
            return

        hour_bucket = datetime.now().strftime("%Y-%m-%dT%H")
        if not self._claim_evidence_bucket("temperament:baseline", hour_bucket):
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

        self._temperament_samples.clear()
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
        if self.read_only:
            return

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
                    descriptions.append(
                        f"I am {strength} that I'm sensitive to changes in my raw light sensor"
                    )
                elif belief.value < 0.4:
                    descriptions.append(
                        f"I am {strength} that raw light-sensor changes don't affect me much"
                    )

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

    def save(self) -> bool:
        """Explicitly save the model."""
        return self._save()


# Singleton instance
_self_model: Optional[SelfModel] = None
_self_model_read_only_default = False


def configure_self_model(*, read_only: bool) -> None:
    """Set this process's self-model role before lifecycle initialization."""
    global _self_model_read_only_default
    _self_model_read_only_default = bool(read_only)
    if _self_model is not None:
        _self_model.read_only = bool(read_only)
        if read_only:
            _self_model.refresh_if_changed(force=True)


def get_self_model(*, read_only: Optional[bool] = None) -> SelfModel:
    """Get or create the process-local writer or refreshing reader."""
    global _self_model
    desired = _self_model_read_only_default if read_only is None else bool(read_only)
    if _self_model is None:
        _self_model = SelfModel(read_only=desired)
    elif _self_model.read_only != desired and read_only is not None:
        raise RuntimeError("self-model role cannot change after initialization")
    if _self_model.read_only:
        _self_model.refresh_if_changed()
    return _self_model
