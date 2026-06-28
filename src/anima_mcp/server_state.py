"""
Server State — constants, helpers, and shared state for the anima-mcp server.

Extracted from server.py to reduce monolith size.
Constants are pure values used across handlers and the main loop.
Helper functions are stateless utilities for data transformation.
"""

import os
import subprocess
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
LEARNING_INTERVAL = 100
SELF_MODEL_SAVE_INTERVAL = 300
SCHEMA_EXTRACTION_INTERVAL = 600
EXPRESSION_INTERVAL = 900
UNIFIED_REFLECTION_INTERVAL = 900  # ~30 min — single unified voice
SELF_ANSWER_INTERVAL = 1800
GOAL_SUGGEST_INTERVAL = 3600   # ~2 hours — suggest new goals
GOAL_CHECK_INTERVAL = 300      # ~10 minutes — check goal progress
META_LEARNING_INTERVAL = 21600  # iterations — ~daily at ~2s/iter

# === Identity resolution ===
# Maps canonical person name → set of aliases (case-insensitive matching)
# All dashboard interactions also resolve to the first person by default.
# Note: "cirwel" excluded — agents often inherit this macOS username.
# Only dashboard source reliably identifies the human.
#
# The canonical operator name is deployment-specific, so it comes from the
# environment (`ANIMA_OPERATOR_NAME`) with a generic default. A fresh clone
# resolves the human to "operator"; a specific deployment sets the env var to
# its caretaker's name (e.g. ANIMA_OPERATOR_NAME=Kenny). The role aliases
# ("caretaker", "dashboard", "human") always resolve to that canonical person.
# NOTE: an existing deployment with person history keyed under a prior canonical
# name MUST set ANIMA_OPERATOR_NAME to that name, or the growth/relationship
# record will be created fresh under "operator" instead of matching history.
OPERATOR_NAME = (os.environ.get("ANIMA_OPERATOR_NAME") or "operator").strip().lower() or "operator"
KNOWN_PERSON_ALIASES = {
    OPERATOR_NAME: {OPERATOR_NAME, "caretaker", "dashboard", "human"},
}

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
SELF_ANSWER_MIN_QUESTION_AGE_SECONDS = 21_600  # 6h — give external Q&A automations a real window
DISPLAY_UPDATE_TIMEOUT_SECONDS = 2.0
MODE_CHANGE_SETTLE_SECONDS = 0.015
HEAVY_SCREEN_DELAY_SECONDS = 1.0
NEURAL_SCREEN_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Pure helper functions (no global state dependencies)
# ---------------------------------------------------------------------------

def is_broker_running() -> bool:
    """Check if the stable_creature broker process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'stable_creature.py'],
            capture_output=True, timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def extract_neural_bands(readings) -> dict:
    """Extract neural band powers from sensor readings."""
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

    return SensorReadings(
        timestamp=timestamp,
        cpu_temp_c=data.get("cpu_temp_c"),
        ambient_temp_c=data.get("ambient_temp_c"),
        humidity_pct=data.get("humidity_pct"),
        light_lux=data.get("light_lux"),
        cpu_percent=data.get("cpu_percent"),
        memory_percent=data.get("memory_percent"),
        disk_percent=data.get("disk_percent"),
        power_watts=data.get("power_watts"),
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
        # EEG frequency band powers
        eeg_delta_power=data.get("eeg_delta_power"),
        eeg_theta_power=data.get("eeg_theta_power"),
        eeg_alpha_power=data.get("eeg_alpha_power"),
        eeg_beta_power=data.get("eeg_beta_power"),
        eeg_gamma_power=data.get("eeg_gamma_power"),
    )
