"""
Server State — constants, helpers, and shared state for the anima-mcp server.

Extracted from server.py to reduce monolith size.
Constants are pure values used across handlers and the main loop.
Helper functions are stateless utilities for data transformation.
"""

import os
import math
import subprocess
from copy import deepcopy
from datetime import datetime

from .sensors import SensorReadings

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

SHM_STALE_THRESHOLD_SECONDS = 15.0  # Broker writes SHM every ~7s; threshold must exceed that
INPUT_ERROR_LOG_INTERVAL = 5.0     # Minimum seconds between input error log messages

# === Loop timing constants ===
LOOP_BASE_DELAY_SECONDS = 0.2
LOOP_MAX_DELAY_SECONDS = 30.0
INPUT_POLL_INTERVAL_SECONDS = 0.016
SHUTDOWN_LONG_PRESS_SECONDS = 3.0

# === Subsystem intervals (loop iterations, ~2s each) ===
METACOG_INTERVAL = 3
AGENCY_INTERVAL = 5
SELF_MODEL_INTERVAL = 5
PRIMITIVE_LANG_INTERVAL = 10
VOICE_INTERVAL = 10
GROWTH_INTERVAL = 30
TRAJECTORY_INTERVAL = 5
SHM_GOVERNANCE_STALE_SECONDS = 210.0  # Broker checks in every ~180s; stale after 210s
SERVER_GOVERNANCE_FALLBACK_SECONDS = 240.0  # Server calls UNITARES if broker hasn't for this long
SYSTEM_METRICS_RECORD_INTERVAL = 15   # ~30s — persist system metrics to SQLite
SYSTEM_METRICS_PRUNE_INTERVAL = 1800  # ~1h — delete metrics older than retention
SYSTEM_METRICS_RETENTION_HOURS = 24.0
THERMAL_RATE_THRESHOLD = 5.0          # °C/min — CPU temp rise rate before concern
MEMORY_PRESSURE_THRESHOLD = 90.0      # % — memory usage before concern
# Preference decay sweep. Decay otherwise only runs when a preference is
# REINFORCED, so an unobserved one never erodes. Hourly is ample — the floor
# takes weeks to reach — and the sweep is idempotent, so cadence only affects
# how promptly a retraction shows up, never the value it converges to.
PREFERENCE_DECAY_INTERVAL_SECONDS = 3600.0
LEARNING_INTERVAL = 100
SELF_MODEL_SAVE_INTERVAL = 300
SCHEMA_EXTRACTION_INTERVAL = 600
EXPRESSION_INTERVAL = 900
UNIFIED_REFLECTION_INTERVAL = 900  # ~30 min — single unified voice
SELF_ANSWER_INTERVAL = 1800
GOAL_SUGGEST_INTERVAL = 3600   # ~2 hours — suggest new goals
GOAL_CHECK_INTERVAL = 300      # ~10 minutes — check goal progress
META_LEARNING_INTERVAL = 21600  # iterations — ~daily at ~2s/iter
SELF_DERIVATION_CHECK_INTERVAL = 1800  # ~1h — is Lumen's weekly self-derivation due? (self_derivation.py)

# === Identity resolution ===
# Maps canonical person name → set of aliases (case-insensitive matching)
# The canonical operator name is deployment-specific, so it comes from the
# environment (`ANIMA_OPERATOR_NAME`) with a generic default. A fresh clone
# resolves the human to "operator"; a specific deployment sets the env var to
# its caretaker's name.
# NOTE: an existing deployment with person history keyed under a prior canonical
# name MUST set ANIMA_OPERATOR_NAME to that name, or the growth/relationship
# record will be created fresh under "operator" instead of matching history.
OPERATOR_NAME = (os.environ.get("ANIMA_OPERATOR_NAME") or "operator").strip().lower() or "operator"

# Aliases that resolve to the operator. This used to read
# `{OPERATOR_NAME, "caretaker", "dashboard", "human"}`, and the comment above it
# claimed "only dashboard source reliably identifies the human". It does not.
#
# The dashboard is a web surface, not a person. Anything that can reach the
# endpoint can post through it — and things do: an agent answering a question
# via the dashboard was recorded as the operator, as a PERSON, because
# `normalize_visitor_identity` matched on the CHANNEL and that match overrode
# the author the caller actually supplied. The generic role words were the same
# mistake in a different shape: "human" is a self-declaration anyone can type.
#
# What remains is the operator's own name — still only a name claim, but a
# deliberate one, not an inference from which door someone came through.
KNOWN_PERSON_ALIASES = {
    OPERATOR_NAME: {OPERATOR_NAME},
}

# Recorded when a caller does not say who it is. It is a real visitor with no
# established identity — not the operator, and not asserted to be a person.
ANONYMOUS_VISITOR_ID = "anonymous"

# === Error/status logging throttle intervals ===
ERROR_LOG_THROTTLE = 300       # ~10 minutes between repeated error logs
STATUS_LOG_THROTTLE = 100      # ~3.3 minutes between status logs
DISPLAY_LOG_THROTTLE = 20      # ~40 seconds between display status logs
WARN_LOG_THROTTLE = 60         # ~2 minutes between warning logs
SCHEMA_LOG_THROTTLE = 120      # ~4 minutes between schema status logs
SELF_DIALOGUE_LOG_THROTTLE = 150  # ~5 minutes between self-dialogue status logs

# === Thresholds ===
METACOG_SURPRISE_THRESHOLD = 0.2
PRIMITIVE_SELF_FEEDBACK_DELAY_SECONDS = 75.0
# 2h — the window a human answerer gets before Lumen answers itself.
#
# Was 21_600 (6h), which outlived QUESTION_EXPIRY_SECONDS (14_400, 4h), so
# every question was stamped ``expired`` before Lumen could answer it: 6 of 6
# on 2026-08-24/25, each answered at ~6.0h. The stamp is still surfaced by the
# ``lumen_qa`` MCP ledger and by ``/qa``, where a status that fires on
# literally every question tells a reader nothing. (The dashboard's renderItem
# keys only on ``q.answer`` and ignores ``status`` entirely — a separate gap,
# not fixed here.)
#
# The window was originally sized for the external Q&A cron, which was
# unloaded on 2026-08-24. It is NOT dead time: ``scripts/message_server.py``
# still serves ``POST /answer`` into ``handle_lumen_qa``, so an operator can
# still reply. 2h is that human window, deliberately a plain constant rather
# than a fraction of QUESTION_EXPIRY_SECONDS — that constant already carries
# the expiry stamp, the dedup window and the re-ask gate, and coupling a
# fourth policy to it would mean tuning the recital cadence silently moved
# this gate too. The ordering that actually matters is asserted in
# tests/test_server_state.py, not encoded as arithmetic here.
SELF_ANSWER_MIN_QUESTION_AGE_SECONDS = 7_200
DISPLAY_UPDATE_TIMEOUT_SECONDS = 2.0
MODE_CHANGE_SETTLE_SECONDS = 0.015
HEAVY_SCREEN_DELAY_SECONDS = 1.0
NEURAL_SCREEN_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Pure helper functions (no global state dependencies)
# ---------------------------------------------------------------------------

def is_broker_running() -> bool:
    """Check if the installed stable-creature entry point is running."""
    try:
        result = subprocess.run(
            # Production executes the ``anima-creature`` console script; its
            # command line never contains ``stable_creature.py``.  Keep the
            # module form for development/manual launches.
            ['pgrep', '-f', 'anima-creature|anima_mcp.stable_creature'],
            capture_output=True, timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def extract_neural_bands(readings) -> dict:
    """Extract normalized computational views from legacy band fields."""
    if not readings:
        return {}
    raw = readings.to_dict() if hasattr(readings, 'to_dict') else (readings if isinstance(readings, dict) else {})
    return {
        k.replace("eeg_", "").replace("_power", ""): round(v, 3)
        for k, v in raw.items()
        if k.startswith("eeg_") and k.endswith("_power") and v is not None
    }


def readings_from_dict(data: dict) -> SensorReadings:
    """Reconstruct SensorReadings from dictionary."""
    # Parse timestamp
    timestamp_str = data.get("timestamp", "")
    if isinstance(timestamp_str, str):
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    light_observed_at = None
    light_observed_str = data.get("light_observed_at")
    if isinstance(light_observed_str, str):
        try:
            light_observed_at = datetime.fromisoformat(
                light_observed_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            light_observed_at = None

    return SensorReadings(
        timestamp=timestamp,
        cpu_temp_c=data.get("cpu_temp_c"),
        ambient_temp_c=data.get("ambient_temp_c"),
        humidity_pct=data.get("humidity_pct"),
        light_lux=data.get("light_lux"),
        light_observed_at=light_observed_at,
        light_observed_precision_seconds=data.get(
            "light_observed_precision_seconds"
        ),
        cpu_percent=data.get("cpu_percent"),
        memory_percent=data.get("memory_percent"),
        disk_percent=data.get("disk_percent"),
        power_watts=data.get("power_watts"),
        hearing_available=data.get("hearing_available", False),
        sound_level=data.get("sound_level"),
        throttle_bits=data.get("throttle_bits"),
        undervoltage_now=data.get("undervoltage_now"),
        throttled_now=data.get("throttled_now"),
        freq_capped_now=data.get("freq_capped_now"),
        undervoltage_occurred=data.get("undervoltage_occurred"),
        led_brightness=data.get("led_brightness"),
        pressure_hpa=data.get("pressure_hpa"),
        pressure_temp_c=data.get("pressure_temp_c"),
        # EEG raw channels
        eeg_tp9=data.get("eeg_tp9"),
        eeg_af7=data.get("eeg_af7"),
        eeg_af8=data.get("eeg_af8"),
        eeg_tp10=data.get("eeg_tp10"),
        eeg_aux1=data.get("eeg_aux1"),
        eeg_aux2=data.get("eeg_aux2"),
        eeg_aux3=data.get("eeg_aux3"),
        eeg_aux4=data.get("eeg_aux4"),
        # Normalized computational views in legacy EEG-named fields
        eeg_delta_power=data.get("eeg_delta_power"),
        eeg_theta_power=data.get("eeg_theta_power"),
        eeg_alpha_power=data.get("eeg_alpha_power"),
        eeg_beta_power=data.get("eeg_beta_power"),
        eeg_gamma_power=data.get("eeg_gamma_power"),
    )


def anima_from_dict(data: dict, readings: SensorReadings):
    """Reconstruct the broker-owned Anima state without sensing it again.

    The broker publishes both the readings and the already-smoothed anima that
    it derived from them.  Consumers must use that published state as the
    authoritative body state; recomputing here creates a second creature with
    different momentum, anticipation, and calibration history.
    """
    from .anima import Anima

    if not isinstance(data, dict):
        raise ValueError("shared anima state must be an object")

    values = {}
    for dimension in ("warmth", "clarity", "stability", "presence"):
        value = data.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"shared anima {dimension} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"shared anima {dimension} must be finite and in [0, 1]")
        values[dimension] = value

    clarity_attribution = data.get("clarity_attribution")
    if clarity_attribution is not None and not isinstance(
        clarity_attribution, dict
    ):
        raise ValueError("shared anima clarity_attribution must be an object")

    return Anima(
        readings=readings,
        clarity_attribution=deepcopy(clarity_attribution),
        **values,
    )
