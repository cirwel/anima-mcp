"""
Server context — mutable state container for the anima-mcp server.

Replaces scattered module-level globals with a single context object.
Created in wake(), cleared in sleep(). Passed to extracted modules (main_loop, etc.).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .activity_state import ActivityManager
    from .agency import Action
    from .calibration_drift import CalibrationDrift
    from .display import DisplayRenderer
    from .display.leds import LEDDisplay
    from .display.screens import ScreenRenderer
    from .growth import GrowthSystem
    from .identity import IdentityStore
    from .primitive_language import Utterance
    from .schema_hub import SchemaHub
    from .sensors import SensorBackend
    from .value_tension import ValueTensionTracker


@dataclass
class ServerContext:
    """Mutable container for all server state. Created in wake(), cleared in sleep()."""

    # Core subsystems (set in wake or lazy-init)
    store: "IdentityStore | None" = None
    sensors: "SensorBackend | None" = None
    display: "DisplayRenderer | None" = None
    screen_renderer: "ScreenRenderer | None" = None
    leds: "LEDDisplay | None" = None
    shm_client: Any = None  # SharedMemoryClient
    growth: "GrowthSystem | None" = None
    activity: "ActivityManager | None" = None
    schema_hub: "SchemaHub | None" = None
    calibration_drift: "CalibrationDrift | None" = None
    tension_tracker: "ValueTensionTracker | None" = None

    # Identity
    anima_id: str | None = None

    # Lazy singletons (initialized on first use)
    server_bridge: Any = None  # UnitaresBridge
    metacog_monitor: Any = None  # MetacognitiveMonitor
    voice_instance: Any = None  # AutonomousVoice

    # Display loop
    display_update_task: Any = None  # asyncio.Task
    input_poll_task: Any = None  # asyncio.Task

    # Input / joystick
    joystick_enabled: bool = False
    sep_btn_press_start: float | None = None
    joy_btn_press_start: float | None = None
    joy_btn_help_shown: bool = False
    last_input_error_log: float = 0.0

    # Governance
    last_governance_decision: dict[str, Any] | None = None
    last_server_checkin_time: float = 0.0
    last_unitares_success_time: float = 0.0

    # Per-iteration cache (updated by _get_readings_and_anima)
    last_shm_data: dict | None = None
    consumed_drive_events: set = field(default_factory=set)

    # Meta-learning / trajectory
    satisfaction_history: deque = field(default_factory=lambda: deque(maxlen=500))
    satisfaction_per_dim: dict[str, deque] = field(default_factory=dict)
    health_history: deque = field(default_factory=lambda: deque(maxlen=100))
    action_efficacy: float = 0.5

    # Agency
    last_action: "Action | None" = None
    last_state_before: dict[str, float] | None = None

    # Primitive language
    last_primitive_utterance: "Utterance | None" = None

    # Self-model cross-iteration
    sm_prev_stability: float | None = None
    sm_prev_warmth: float | None = None
    sm_pending_prediction: dict | None = None
    sm_clarity_before_interaction: float | None = None

    # LED proprioception
    led_proprioception: dict | None = None

    # Warm start / gap
    warm_start_anima: dict | None = None
    wake_gap: timedelta | None = None
    wake_restored: dict | None = None
    wake_recovery_cycles: int = 0
    wake_recovery_total: int = 0
    wake_presence_floor: float = 0.3
