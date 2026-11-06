"""
Stable Anima Creature Script

Continuous loop that:
1. Reads sensors with robust error handling (retries for I2C)
2. Updates anima state (proprioception)
3. Renders ASCII face based on state
4. Integrates with UNITARES governance bridge if available

Designed to run continuously on the Pi.

✅ BODY BROKER MODE ✅
─────────────────────────────────────────────────────────────
This script acts as Lumen's body-state broker.

HOW IT WORKS:
- In direct mode this script owns the environmental I2C sensors
- In deployed shadow mode, anima-broker-ex owns those sensors and this script
  consumes its fresh readings without opening I2C
- Reads sensors every 2 seconds
- Writes data to shared memory (/dev/shm or Redis)
- The MCP server (anima --http) reads from shared memory

YOU CAN NOW RUN BOTH:
  - stable_creature.py (hardware broker - this script)
  - anima --http (MCP server - reads from shared memory)

BENEFITS:
- No I2C conflicts
- Creature stays alive while MCP server restarts
- Fast MCP responses (reads memory, not hardware)
- Automatic coordination via shared memory

See docs/operations/BROKER_ARCHITECTURE.md for details.
─────────────────────────────────────────────────────────────
"""

import time
import signal
import sys
import asyncio
import concurrent.futures
import threading

# Force UTF-8 for stdout/stderr (prevents crash in systemd service)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass # If reconfigure fails (e.g. older python), we might be stuck

from .anima import MoodMomentum
from .inner_life import InnerLife
# NOTE: Broker does NOT import or init LEDDisplay — server owns LED hardware.
# Agency LED brightness is communicated via shared memory.
from .display.face import derive_face_state, face_to_ascii, EyeState
# NOTE: LEDs are handled by MCP server, not broker (prevents I2C conflicts)
from .db_paths import BROKER_AGENCY_DB
from .light_attribution import read_led_proprioception
from .metacognition import get_metacognitive_monitor


# Enhanced learning systems (optional - for genuine agency)
# Each module imported independently so one failure doesn't disable all 8.
_LEARNING_MODULES: dict = {}

def _has_module(name: str) -> bool:
    return _LEARNING_MODULES.get(name, False)


try:
    from .adaptive_prediction import get_adaptive_prediction_model
    _LEARNING_MODULES["adaptive_prediction"] = True
except ImportError:
    get_adaptive_prediction_model = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["adaptive_prediction"] = False

try:
    from .preferences import get_preference_system
    _LEARNING_MODULES["preferences"] = True
except ImportError:
    get_preference_system = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["preferences"] = False

try:
    from .self_model import get_self_model
    _LEARNING_MODULES["self_model"] = True
except ImportError:
    get_self_model = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["self_model"] = False

try:
    from .agency import get_action_selector, get_exploration_manager, ActionType
    _LEARNING_MODULES["agency"] = True
except ImportError:
    get_action_selector = None  # type: ignore[assignment,misc]
    get_exploration_manager = None  # type: ignore[assignment,misc]
    ActionType = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["agency"] = False

try:
    from .memory_retrieval import get_memory_retriever, retrieve_relevant_memories
    _LEARNING_MODULES["memory_retrieval"] = True
except ImportError:
    get_memory_retriever = None  # type: ignore[assignment,misc]
    retrieve_relevant_memories = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["memory_retrieval"] = False

try:
    from .weighted_pathways import get_weighted_pathways, discretize_context
    _LEARNING_MODULES["weighted_pathways"] = True
except ImportError:
    get_weighted_pathways = None  # type: ignore[assignment,misc]
    discretize_context = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["weighted_pathways"] = False

try:
    from .experiential_filter import get_experiential_filter
    _LEARNING_MODULES["experiential_filter"] = True
except ImportError:
    get_experiential_filter = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["experiential_filter"] = False

try:
    from .experiential_marks import get_experiential_marks
    _LEARNING_MODULES["experiential_marks"] = True
except ImportError:
    get_experiential_marks = None  # type: ignore[assignment,misc]
    _LEARNING_MODULES["experiential_marks"] = False

ENHANCED_LEARNING_AVAILABLE = any(_LEARNING_MODULES.values())
if not ENHANCED_LEARNING_AVAILABLE:
    print("[StableCreature] Enhanced learning not available (all modules failed)")
elif not all(_LEARNING_MODULES.values()):
    failed = [k for k, v in _LEARNING_MODULES.items() if not v]
    print(f"[StableCreature] Enhanced learning partial: missing {', '.join(failed)}")

# Activity state (sleep/wake cycle)
try:
    from .activity_state import get_activity_manager, ActivityLevel
    ACTIVITY_STATE_AVAILABLE = True
except ImportError:
    ACTIVITY_STATE_AVAILABLE = False
    print("[StableCreature] Activity state not available")

# Voice support (optional - graceful degradation if not available)
try:
    from .audio import AutonomousVoice  # noqa: F401
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("[StableCreature] Voice module not available (missing dependencies)")

# Config
UPDATE_INTERVAL = 2.0  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 0.5


running = True

def signal_handler(sig, frame):
    global running
    print("\n[StableCreature] Shutdown signal received. Closing gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)



from .creature_runtime import (  # noqa: E402,F401
    CreatureRuntimeMixin,
    _apply_belief_update_bonus,
    _build_broker_reflection_event,
    _build_preference_state,
    _direct_i2c_initialization_failed,
    _load_persistent_identity,
    _pair_led_action_to_light_capture,
    _run_coroutine_with_timeout,
    _snapshot_preferences,
    _snapshot_self_beliefs,
    _stdout_is_terminal,
    broker_agency_enabled,
    broker_voice_enabled,
)


class StableCreature(CreatureRuntimeMixin):
    """Runtime coordinator for Lumen's body-state broker."""


def run_creature():
    print("[StableCreature] Starting up...")
    runtime = StableCreature()
    legacy_broker_agency = broker_agency_enabled()
    terminal_face = _stdout_is_terminal()

    (
        identity,
        db_path,
        sensors,
        bridge,
        gov_shadow_path,
        gov_shadow_stale_s,
        shm_client,
        voice,
    ) = runtime.init_identity_and_sensors()

    _mood_momentum = MoodMomentum()
    _inner_life = InnerLife()
    inner_state = None

    # Persistent event loop for async calls (governance, cognitive, memory).
    # A single loop runs in a dedicated daemon thread so aiohttp sessions
    # are reused across calls instead of being recreated per invocation.
    _bg_loop = asyncio.new_event_loop()

    def _bg_loop_thread():
        asyncio.set_event_loop(_bg_loop)
        _bg_loop.run_forever()

    _bg_thread = threading.Thread(target=_bg_loop_thread, daemon=True, name="creature-async")
    _bg_thread.start()

    # Background thread executor for non-blocking governance/memory calls.
    _bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="creature-bg")

    def _run_async_in_background(coro, timeout=10.0):
        """Run an async coroutine on the persistent event loop with timeout."""
        return _run_coroutine_with_timeout(coro, _bg_loop, timeout)

    _governance_future = None
    _memory_future = None

    # Initialize Metacognition Monitor
    metacog = get_metacognitive_monitor(read_only=True)
    print("[StableCreature] Metacognition active (read-only observer; mind owns persistence)")

    (
        adaptive_model,
        preferences,
        self_model,
        action_selector,
        exploration_mgr,
        memory_retriever,
    ) = runtime.init_enhanced_learning(legacy_broker_agency)
    exp_marks = None

    # Initialize Experiential Accumulation (Layer 1-3)
    pathways = None
    exp_filter = None
    if ENHANCED_LEARNING_AVAILABLE:
        try:
            if _has_module("weighted_pathways") and legacy_broker_agency:
                pathways = get_weighted_pathways(db_path=str(BROKER_AGENCY_DB))
                print("[StableCreature] Weighted pathways active - decisions shaped by experience")
            if _has_module("experiential_filter"):
                exp_filter = get_experiential_filter()
                print("[StableCreature] Experiential filter active - perception colored by history")
            if _has_module("experiential_marks"):
                exp_marks = get_experiential_marks(db_path=db_path)
                _apply_belief_update_bonus(self_model, exp_marks)
                print("[StableCreature] Experiential marks active - significant events leave permanent traces")
        except Exception as e:
            print(f"[StableCreature] Experiential accumulation init error: {e}")

    # Initialize Activity State (sleep/wake cycle)
    activity_manager = None
    if ACTIVITY_STATE_AVAILABLE:
        try:
            activity_manager = get_activity_manager()
            print("[StableCreature] Activity state active - Lumen has sleep/wake cycles")
        except Exception as e:
            print(f"[StableCreature] Activity state init error: {e}")

    last_decision = None
    last_decision_checked_at = None
    first_check_in = True
    _last_reflection_event = None
    GOVERNANCE_INTERVAL = runtime.parse_governance_interval()
    last_governance_time = 0
    last_action = None
    _last_pw_ctx = None
    last_state_for_action = None
    last_self_model_state = None
    last_self_model_observed_at = time.monotonic()
    last_learning_save = time.time()
    last_body_update_at = time.monotonic()
    _adaptive_context_light = None
    _adaptive_context_temp = None
    readings = None
    last_pattern_apply = 0
    last_learning_event_drain = 0.0
    last_learning_event_failure_log = 0.0
    _preset_led_brightness = runtime.load_led_brightness_preset()
    _prev_led_brightness = _preset_led_brightness
    _agency_led_brightness = 1.0
    _latest_led_proprioception = None
    _paired_led_proprioception = None
    light_attribution = None

    try:
        while running:
            light_attribution = None
            last_learning_event_drain, last_learning_event_failure_log = runtime.drain_inbox_if_due(
                self_model, preferences, last_learning_event_drain, last_learning_event_failure_log
            )

            # 0. Metacognition: Generate prediction BEFORE sensing
            _latest_led_proprioception = read_led_proprioception(
                max_age_seconds=2.0
            )
            if _latest_led_proprioception is not None:
                _led_br = _latest_led_proprioception["brightness"]
            else:
                _led_br = (
                    readings.led_brightness
                    if readings and readings.led_brightness is not None
                    else None
                )
            prediction = metacog.predict(led_brightness=_led_br)

            # 1. Robust Sensor Read
            readings = runtime.read_sensors_with_retry(sensors)
            if not readings:
                print("[StableCreature] Failed to read sensors after retries. Skipping loop.")
                time.sleep(UPDATE_INTERVAL)
                continue

            (
                _paired_led_proprioception,
                _observed_led_proprioception,
                _led_brightness_source,
                light_attribution,
                _external_light_for_environment,
            ) = runtime.compute_led_and_light_attribution(
                readings, _latest_led_proprioception, _prev_led_brightness, self_model
            )

            # 2. Update Anima State and Inner Life
            (
                anima,
                inner_state,
                _drive_events,
                body_projection,
                last_body_update_at,
            ) = runtime.update_body_anima_and_inner_life(
                readings,
                exp_filter,
                _external_light_for_environment,
                _mood_momentum,
                _inner_life,
                last_body_update_at,
            )

            # 2a-ii. Activity State: Determine wakefulness level
            activity_state = None
            if activity_manager:
                activity_state = activity_manager.get_state(
                    presence=anima.presence,
                    stability=anima.stability,
                    light_level=_external_light_for_environment,
                )
                _base_br = _preset_led_brightness
                _prev_led_brightness = _base_br * _agency_led_brightness * activity_state.brightness_multiplier
                if _observed_led_proprioception is None:
                    readings.led_brightness = _prev_led_brightness
                if activity_manager.should_skip_update():
                    time.sleep(UPDATE_INTERVAL)
                    continue

            # 2b-pre. Collect results from background futures (non-blocking)
            if _memory_future is not None and _memory_future.done():
                try:
                    mem_result = _memory_future.result()
                    if mem_result:
                        relevant_memories = mem_result
                        print(f"[Memory] Retrieved: {len(relevant_memories)} relevant memories", file=sys.stderr, flush=True)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    print(f"[Learning] Memory retrieval error: {e}", file=sys.stderr, flush=True)
                _memory_future = None

            # 2b. Metacognition: Compare prediction to reality
            pred_error, should_reflect, _last_reflection_event = runtime.evaluate_metacognition_and_filter(
                metacog,
                readings,
                anima,
                exp_filter,
                exp_marks,
                preferences,
                self_model,
                _last_reflection_event,
            )

            # 2b-i..iii. Adaptive Prediction, Self-Model, Preferences
            (
                _adaptive_context_light,
                _adaptive_context_temp,
                last_self_model_state,
                last_self_model_observed_at,
            ) = runtime.run_adaptive_and_self_model_learning(
                adaptive_model,
                self_model,
                preferences,
                readings,
                anima,
                pred_error,
                inner_state,
                _external_light_for_environment,
                _adaptive_context_light,
                _adaptive_context_temp,
                last_self_model_state,
                last_self_model_observed_at,
            )

            # 2b-iv. Memory Retrieval: Let past inform present (background thread)
            relevant_memories = []
            if memory_retriever and should_reflect and _memory_future is None:
                _mem_sources = list(pred_error.surprise_sources or [])
                _mem_warmth = anima.warmth
                _mem_clarity = anima.clarity
                _mem_stability = anima.stability

                def _do_memory():
                    return _run_async_in_background(
                        retrieve_relevant_memories(
                            surprise_sources=_mem_sources,
                            warmth=_mem_warmth,
                            clarity=_mem_clarity,
                            stability=_mem_stability,
                            limit=2
                        ),
                        timeout=2.0
                    )

                _memory_future = _bg_executor.submit(_do_memory)

            # 2b-v..vii. Legacy Broker Agency
            (
                _agency_led_brightness,
                last_action,
                last_state_for_action,
                _last_pw_ctx,
            ) = runtime.run_legacy_broker_agency(
                action_selector,
                exploration_mgr,
                preferences,
                self_model,
                pathways,
                exp_marks,
                voice,
                anima,
                pred_error,
                inner_state,
                activity_state,
                _agency_led_brightness,
                last_action,
                last_state_for_action,
                _last_pw_ctx,
            )

            # 2c. Update Voice
            runtime.update_broker_voice(voice, anima, readings, _external_light_for_environment)

            # 3. Governance Check-in & Experiential Stats
            (
                _governance_future,
                first_check_in,
                last_governance_time,
                last_decision,
                last_decision_checked_at,
                _exp_state,
            ) = runtime.process_governance_and_collect_exp(
                bridge,
                gov_shadow_path,
                activity_manager,
                pathways,
                exp_filter,
                exp_marks,
                action_selector,
                anima,
                readings,
                identity,
                light_attribution,
                _bg_executor,
                _run_async_in_background,
                _governance_future,
                first_check_in,
                last_governance_time,
                last_decision,
                last_decision_checked_at,
                GOVERNANCE_INTERVAL,
            )

            # 3b. Write to Shared Memory (Broker)
            runtime.build_and_write_shm_snapshot(
                shm_client,
                anima,
                readings,
                inner_state,
                _drive_events,
                body_projection,
                identity,
                pred_error,
                metacog,
                prediction,
                _observed_led_proprioception,
                _led_brightness_source,
                light_attribution,
                _last_reflection_event,
                gov_shadow_path,
                gov_shadow_stale_s,
                last_decision,
                last_decision_checked_at,
                activity_state,
                preferences,
                self_model,
                action_selector,
                adaptive_model,
                _exp_state,
                _agency_led_brightness,
            )

            # 4. Render Face (terminal only). Under systemd stdout is the
            # journal, and this 8-line frame every tick was ~81% of all
            # journal lines on Lumen, cutting retention to ~32 hours.
            if terminal_face:
                face_state = derive_face_state(anima)

                # Modify face based on activity state (sleeping/drowsy)
                if activity_state and activity_state.level == ActivityLevel.RESTING:
                    face_state.eyes = EyeState.CLOSED
                    face_state.eye_openness = 0.0
                elif activity_state and activity_state.level == ActivityLevel.DROWSY:
                    face_state.eyes = EyeState.DROOPY
                    face_state.eye_openness = 0.4

                ascii_face = face_to_ascii(face_state)

                # Clear screen (terminal) - use ANSI codes to prevent flicker
                print("\033[2J\033[H", end="")

                # Print identity and mood
                print(f"Name: {identity.name or 'Anima'} | Mood: {anima.feeling()['mood']}")
                print(f"W: {anima.warmth:.2f} | C: {anima.clarity:.2f} | S: {anima.stability:.2f} | P: {anima.presence:.2f}")

                # Print face
                print(ascii_face)

                # Print governance if available
                if last_decision:
                    action = last_decision.get("action", "UNKNOWN")
                    reason = last_decision.get("reason", "")
                    print(f"Governance: {action.upper()} - {reason}")

                # Print metacognition if surprise is notable
                if pred_error.surprise > 0.1:
                    sources = ", ".join(pred_error.surprise_sources) if pred_error.surprise_sources else "general"
                    print(f"Surprise: {pred_error.surprise:.0%} ({sources})")

            # Periodic learning save: Save learning state every 5 minutes to survive crashes
            if ENHANCED_LEARNING_AVAILABLE and time.time() - last_learning_save > 300:
                try:
                    if adaptive_model:
                        adaptive_model._save_patterns()
                    if preferences:
                        preferences._save()
                    if self_model:
                        self_model.save()
                    # Layer 3: Check experiential marks (piggyback on 5-min save)
                    if exp_marks and identity:
                        try:
                            _obs_count, _belief_confs, _drawing_count, _question_count = (
                                runtime.gather_mark_check_inputs(identity, self_model, db_path)
                            )
                            _new_marks = exp_marks.check_and_earn(
                                awakenings=identity.total_awakenings,
                                observation_count=_obs_count,
                                drawing_count=_drawing_count,
                                question_count=_question_count,
                                long_gap_count=max(0, identity.total_awakenings - 1),
                                belief_confidences=_belief_confs,
                            )
                            if _new_marks:
                                print(f"[Experiential] Earned marks: {_new_marks}", file=sys.stderr, flush=True)
                                _apply_belief_update_bonus(self_model, exp_marks)
                        except Exception as e:
                            print(f"[Experiential] Mark check error: {e}", file=sys.stderr, flush=True)

                    # Layer 2: Periodic filter save
                    if exp_filter:
                        try:
                            exp_filter.save()
                        except Exception as e:
                            print(f"[Experiential] Filter save error: {e}", file=sys.stderr, flush=True)

                    last_learning_save = time.time()
                except Exception as e:
                    print(f"[StableCreature] Learning save error: {e}", file=sys.stderr, flush=True)

            # Apply learned patterns to activity schedule (~once per hour)
            if ACTIVITY_STATE_AVAILABLE and activity_manager and ENHANCED_LEARNING_AVAILABLE:
                if time.time() - last_pattern_apply > 3600:
                    try:
                        adjustments = activity_manager.apply_learned_patterns(
                            adaptive_model=adaptive_model if ENHANCED_LEARNING_AVAILABLE else None,
                            self_model=self_model if ENHANCED_LEARNING_AVAILABLE else None,
                        )
                        if adjustments:
                            print(f"[Activity] Applied {len(adjustments)} learned pattern adjustments",
                                  file=sys.stderr, flush=True)
                        last_pattern_apply = time.time()
                    except Exception as e:
                        print(f"[Activity] Pattern apply error (non-fatal): {e}", file=sys.stderr, flush=True)
                        last_pattern_apply = time.time()

            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown_creature_resources(
            adaptive_model,
            preferences,
            self_model,
            _inner_life,
            exp_filter,
            pathways,
            exp_marks,
            voice,
            bridge,
            _bg_loop,
            _bg_thread,
            _bg_executor,
            shm_client,
        )

def main():
    """Entry point for pyproject.toml scripts."""
    run_creature()


if __name__ == "__main__":
    main()
