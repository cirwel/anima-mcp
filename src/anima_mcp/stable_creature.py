"""
Stable Anima Creature Script

Continuous loop that:
1. Reads sensors with robust error handling (retries for I2C)
2. Updates anima state (proprioception)
3. Renders ASCII face based on state
4. Integrates with UNITARES governance bridge if available

Designed to run continuously on the Pi.

✅ HARDWARE BROKER MODE ✅
─────────────────────────────────────────────────────────────
This script acts as the HARDWARE BROKER for Lumen's sensors.

HOW IT WORKS:
- This script owns I2C sensors exclusively (no conflicts)
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

import json
import time
import os
import signal
import sys
import asyncio
import concurrent.futures
import threading
import psutil
from datetime import datetime
from pathlib import Path

# Force UTF-8 for stdout/stderr (prevents crash in systemd service)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass # If reconfigure fails (e.g. older python), we might be stuck

from .sensors import get_sensors
from .anima import sense_self, MoodMomentum
from .inner_life import InnerLife
from .display.leds.brightness import estimate_instantaneous_brightness
# NOTE: Broker does NOT import or init LEDDisplay — server owns LED hardware.
# Agency LED brightness is communicated via shared memory.
from .display.face import derive_face_state, face_to_ascii, EyeState
# NOTE: LEDs are handled by MCP server, not broker (prevents I2C conflicts)
from .identity import IdentityStore
from .unitares_bridge import UnitaresBridge
from .shared_memory import SharedMemoryClient
from .eisv_mapper import anima_to_eisv
from .metacognition import get_metacognitive_monitor


# Enhanced learning systems (optional - for genuine agency)
# Each module imported independently so one failure doesn't disable all 8.
_LEARNING_MODULES: dict = {}

def _has_module(name: str) -> bool:
    return _LEARNING_MODULES.get(name, False)


def _snapshot_self_beliefs(self_model) -> dict:
    """Capture lightweight self-belief state for reflection episode persistence."""
    if not self_model:
        return {}

    beliefs = getattr(self_model, "beliefs", None) or {}
    snapshot = {}
    for belief_id, belief in beliefs.items():
        try:
            snapshot[str(belief_id)] = {
                "value": round(float(getattr(belief, "value", 0.0)), 3),
                "confidence": round(float(getattr(belief, "confidence", 0.0)), 3),
            }
        except (TypeError, ValueError):
            continue
    return snapshot


def _snapshot_preferences(preferences) -> dict:
    """Capture lightweight preference weights for reflection episode persistence."""
    if not preferences:
        return {}

    pref_map = getattr(preferences, "_preferences", None) or {}
    snapshot = {}
    for pref_id, pref in pref_map.items():
        try:
            snapshot[str(pref_id)] = {
                "valence": round(float(getattr(pref, "valence", 0.0)), 3),
                "confidence": round(float(getattr(pref, "confidence", 0.0)), 3),
                "influence_weight": round(float(getattr(pref, "influence_weight", 1.0)), 3),
            }
        except (TypeError, ValueError):
            continue
    return snapshot


def _build_broker_reflection_event(reflection, error, self_model=None, preferences=None, reflect_reason: str = "") -> dict:
    """Serialize a metacognitive reflection for broker->server SHM propagation."""
    event_ts = reflection.timestamp.isoformat()
    topic_tags = [str(tag).lower() for tag in (error.surprise_sources or []) if tag]

    return {
        "event_id": f"broker-metacog:{event_ts}",
        "timestamp": event_ts,
        "kind": "metacog",
        "source": "broker",
        "trigger": reflection.trigger,
        "topic_tags": topic_tags,
        "observation": reflection.observation,
        "surprise": round(float(error.surprise), 3),
        "discrepancy": round(float(reflection.discrepancy), 3),
        "discrepancy_description": reflection.discrepancy_description,
        "belief_snapshot": _snapshot_self_beliefs(self_model),
        "preference_snapshot": _snapshot_preferences(preferences),
        "metadata": {
            "reflect_reason": reflect_reason,
            "surprise_sources": topic_tags,
            "felt_state": reflection.felt_state or {},
            "sensor_state": reflection.sensor_state or {},
        },
    }

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
    from .audio import AutonomousVoice
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("[StableCreature] Voice module not available (missing dependencies)")

# Config
UPDATE_INTERVAL = 2.0  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 0.5

# Global shutdown flag
running = True

def signal_handler(sig, frame):
    global running
    print("\n[StableCreature] Shutdown signal received. Closing gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_creature():
    print("[StableCreature] Starting up...")
    
    # Initialize components with error handling
    identity = None
    store = None
    try:
        # Determine DB persistence path (User Home > Project Root)
        env_db = os.environ.get("ANIMA_DB")
        if env_db:
            db_path = env_db
        else:
            # Default to persistent user home directory
            home_dir = Path.home() / ".anima"
            home_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(home_dir / "anima.db")

        store = IdentityStore(db_path)
        print(f"[StableCreature] Identity persistence: {db_path}")

        # Identity preservation: check database first, then env var, then generate new
        # This ensures Lumen's identity persists even if config is missing
        anima_id = os.environ.get("ANIMA_ID")
        if not anima_id:
            # Check if identity already exists in database
            conn = store._connect()
            try:
                existing = conn.execute("SELECT creature_id FROM identity LIMIT 1").fetchone()
                if existing:
                    anima_id = existing[0]
                    print(f"[StableCreature] Using existing identity: {anima_id[:8]}...")
                else:
                    # Only generate new UUID if truly first boot
                    import uuid
                    anima_id = str(uuid.uuid4())
                    print(f"[StableCreature] Creating new identity: {anima_id[:8]}...")
            finally:
                conn.close()

        identity = store.wake(anima_id)
    except Exception as e:
        print(f"[StableCreature] WARNING: Identity store failed ({e}) - using fallback identity")
        print("[StableCreature] Broker will continue (sensors -> shared memory). Server can repair DB.")
        import uuid
        from .identity import CreatureIdentity
        anima_id = os.environ.get("ANIMA_ID") or str(uuid.uuid4())
        now = datetime.now()
        identity = CreatureIdentity(
            creature_id=anima_id,
            born_at=now,
            total_awakenings=0,
            total_alive_seconds=0.0,
            name="Lumen",
            name_history=[],
            current_awakening_at=now,
            last_heartbeat_at=None,
            metadata={},
        )
        store = None  # No DB connection when using fallback
    
    # Initialize sensors - allow graceful degradation if hardware unavailable
    try:
        sensors = get_sensors()
        # Check if sensors initialized (at least I2C should be available)
        if hasattr(sensors, '_i2c') and sensors._i2c is None:
            print("[StableCreature] WARNING: I2C initialization failed - hardware may be disconnected")
            print("[StableCreature] Continuing with degraded sensor access (CPU-only readings)")
    except Exception as e:
        print(f"[StableCreature] CRITICAL: Sensor initialization failed: {e}")
        print("[StableCreature] Hardware may be disconnected. Exiting to prevent restart loop.")
        print("[StableCreature] Wait 30 seconds, then check hardware connections before restarting.")
        time.sleep(30)  # Give hardware time to stabilize
        sys.exit(1)
    
    # NOTE: LEDs are NOT initialized here - they're handled by the MCP server
    # This prevents I2C conflicts between broker and MCP server
    
    unitares_url = os.environ.get("UNITARES_URL")
    bridge = UnitaresBridge(unitares_url=unitares_url) if unitares_url else None
    
    # Initialize Shared Memory (Broker Mode)
    # Using file backend for maximum stability (Redis caused hangs)
    try:
        shm_client = SharedMemoryClient(mode="write", backend="file")
        shm_client.clear()  # Remove stale/corrupted data from previous run
        print(f"[StableCreature] Shared Memory active using backend: {shm_client.backend}")
        if shm_client.backend == "file":
            print(f"[StableCreature] File path: {shm_client.filepath}")
    except Exception as e:
        print(f"[StableCreature] CRITICAL: Shared memory initialization failed: {e}")
        print("[StableCreature] Exiting to prevent restart loop.")
        sys.exit(1)
    
    if bridge:
        try:
            bridge.set_agent_id(identity.creature_id)
            bridge.set_session_id(f"anima-{identity.creature_id[:8]}")
            print(f"[StableCreature] UNITARES bridge active: {unitares_url}")
        except Exception as e:
            print(f"[StableCreature] WARNING: UNITARES bridge setup failed: {e}")
            bridge = None  # Continue without governance

    # Initialize Voice (optional - Lumen's ability to hear and speak)
    voice = None
    if VOICE_AVAILABLE and os.environ.get("ANIMA_VOICE_ENABLED", "true").lower() == "true":
        try:
            voice = AutonomousVoice()
            voice.start()
            print("[StableCreature] Voice active - Lumen can hear and speak")
        except Exception as e:
            print(f"[StableCreature] WARNING: Voice initialization failed: {e}")
            voice = None

    print(f"[StableCreature] Creature '{identity.name or '(unnamed)'}' is alive.")
    print("[StableCreature] Entering main loop...")

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
    # Single worker prevents concurrency issues; the main loop submits work
    # and checks results on the next iteration instead of blocking.
    _bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="creature-bg")

    def _run_async_in_background(coro, timeout=10.0):
        """Run an async coroutine on the persistent event loop with timeout.

        Uses asyncio.run_coroutine_threadsafe to schedule onto _bg_loop,
        so the bridge's aiohttp session stays on a single consistent loop.
        """
        future = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
        return future.result(timeout=timeout)

    # Track background futures so the main loop can skip if still running
    _governance_future = None   # type: Optional[concurrent.futures.Future]
    _memory_future = None       # type: Optional[concurrent.futures.Future]

    # Initialize Metacognition Monitor
    metacog = get_metacognitive_monitor()
    print("[StableCreature] Metacognition active - Lumen monitors its own predictions")

    # Initialize Enhanced Learning Systems (genuine agency)
    adaptive_model = None
    preferences = None
    self_model = None
    action_selector = None
    exploration_mgr = None
    memory_retriever = None
    exp_marks = None

    if ENHANCED_LEARNING_AVAILABLE:
        try:
            if _has_module("adaptive_prediction"):
                adaptive_model = get_adaptive_prediction_model()
                print("[StableCreature] Adaptive prediction active - Lumen learns from experience")

            if _has_module("preferences"):
                preferences = get_preference_system()
                print("[StableCreature] Preferences active - Lumen develops values")

            if _has_module("self_model"):
                self_model = get_self_model()
                if exp_marks:
                    self_model.belief_update_bonus = exp_marks.get_effect("belief_update_bonus")
                print("[StableCreature] Self-model active - Lumen has beliefs about itself")

            if _has_module("agency"):
                action_selector = get_action_selector()
                exploration_mgr = get_exploration_manager()
                print("[StableCreature] Agency active - Lumen can choose actions")

            if _has_module("memory_retrieval"):
                memory_retriever = get_memory_retriever()
                print("[StableCreature] Memory retrieval active - past informs present")
        except Exception as e:
            print(f"[StableCreature] Enhanced learning init error: {e}")

    # Initialize Experiential Accumulation (Layer 1-3)
    pathways = None
    exp_filter = None
    if ENHANCED_LEARNING_AVAILABLE:
        try:
            if _has_module("weighted_pathways"):
                pathways = get_weighted_pathways(db_path=db_path)
                print("[StableCreature] Weighted pathways active - decisions shaped by experience")
            if _has_module("experiential_filter"):
                exp_filter = get_experiential_filter()
                print("[StableCreature] Experiential filter active - perception colored by history")
            if _has_module("experiential_marks"):
                exp_marks = get_experiential_marks(db_path=db_path)
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
    first_check_in = True  # Track first governance check to sync identity
    # Most recent metacognitive reflection serialized for SHM propagation. The
    # server-side reflection_episodes drain picks it up each tick. Latest-record
    # semantics: overwritten on each new reflection; the server dedupes on event_id.
    _last_reflection_event = None
    # Rate limit governance check-ins so they carry meaningful signal.
    # Override with ANIMA_GOVERNANCE_INTERVAL_SECONDS if needed.
    DEFAULT_GOVERNANCE_INTERVAL = 180.0
    MIN_GOVERNANCE_INTERVAL = 30.0
    _interval_env = os.environ.get("ANIMA_GOVERNANCE_INTERVAL_SECONDS")
    _interval_raw = None
    try:
        _interval_raw = float(_interval_env) if _interval_env is not None else None
        GOVERNANCE_INTERVAL = (
            _interval_raw if _interval_raw is not None else DEFAULT_GOVERNANCE_INTERVAL
        )
    except (TypeError, ValueError):
        print(
            f"[StableCreature] Invalid ANIMA_GOVERNANCE_INTERVAL_SECONDS='{_interval_env}', "
            f"using default {DEFAULT_GOVERNANCE_INTERVAL:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        GOVERNANCE_INTERVAL = DEFAULT_GOVERNANCE_INTERVAL
    GOVERNANCE_INTERVAL = max(MIN_GOVERNANCE_INTERVAL, GOVERNANCE_INTERVAL)
    if _interval_raw is not None and GOVERNANCE_INTERVAL != _interval_raw:
        print(
            f"[StableCreature] Governance interval clamped to {GOVERNANCE_INTERVAL:.0f}s "
            f"(minimum {MIN_GOVERNANCE_INTERVAL:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    last_governance_time = 0
    last_action = None  # Track last action for outcome recording
    _last_pw_ctx = None  # Track pathway context for outcome reinforcement
    last_state_for_action = None  # State before action for learning
    last_learning_save = time.time()  # Track periodic learning saves
    readings = None  # Initialize before loop (first iteration has no prior readings)
    last_pattern_apply = 0  # Track periodic learned pattern application
    # LED brightness tracking (broker doesn't own LED hardware — server does)
    # Read actual brightness preset from disk (renderer saves to ~/.anima/display_brightness.json).
    # Falls back to Medium preset (0.12) — NOT the old 0.04 config default.
    _brightness_preset_path = Path.home() / ".anima" / "display_brightness.json"
    _preset_led_brightness = 0.12  # Medium preset default
    try:
        if _brightness_preset_path.exists():
            _br_data = json.loads(_brightness_preset_path.read_text())
            _preset_led_brightness = _br_data.get("leds", 0.12)
            print(f"[StableCreature] LED brightness from preset: {_preset_led_brightness}", file=sys.stderr, flush=True)
    except Exception:
        pass
    _prev_led_brightness = _preset_led_brightness  # Estimate for proprioception
    _agency_led_brightness = 1.0  # Agency-desired manual brightness factor [0.05, 1.0]

    try:
        while running:
            # 0. Metacognition: Generate prediction BEFORE sensing
            # Pass LED brightness for proprioceptive light prediction if available
            _led_br = readings.led_brightness if readings and readings.led_brightness is not None else None
            prediction = metacog.predict(led_brightness=_led_br)
            
            # 1. Robust Sensor Read
            readings = None
            for attempt in range(MAX_RETRIES):
                try:
                    readings = sensors.read()
                    break
                except Exception as e:
                    print(f"[StableCreature] Sensor read error (attempt {attempt+1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
            
            if not readings:
                print("[StableCreature] Failed to read sensors after retries. Skipping loop.")
                time.sleep(UPDATE_INTERVAL)
                continue

            # 1b. LED Proprioception: track brightness for awareness (not correction)
            _instantaneous_led = estimate_instantaneous_brightness(_prev_led_brightness)
            readings.led_brightness = _instantaneous_led

            # 2. Update Anima State (now has correct led_brightness for correction)
            # Layer 2: Apply experiential filter — perception colored by accumulated experience
            _salience = exp_filter.get_all_saliences() if exp_filter else None
            raw_anima = sense_self(readings, salience_weights=_salience)
            anima = _mood_momentum.smooth(raw_anima)
            inner_state = _inner_life.update(raw_anima, anima)

            # 2-i-a. Check for social boost signal (server writes on interaction)
            _boost_path = Path("/dev/shm/anima_social_boost")
            if _boost_path.exists():
                try:
                    _boost_path.unlink()
                    _inner_life.apply_social_boost()
                except Exception:
                    pass

            # 2-i. Collect drive events for server to consume via SHM
            _drive_events = []
            for ev in _inner_life.get_pending_events():
                obs_text = _inner_life.get_observation_text(ev)
                if obs_text:
                    _drive_events.append({
                        "text": obs_text,
                        "dimension": ev.dimension,
                        "event_type": ev.event_type,
                        "drive_value": round(ev.drive_value, 3),
                    })

            # 2a. Calculate UNITARES EISV metrics
            eisv = anima_to_eisv(anima, readings)

            # 2a-ii. Activity State: Determine wakefulness level
            activity_state = None
            if activity_manager:
                activity_state = activity_manager.get_state(
                    presence=anima.presence,
                    stability=anima.stability,
                    light_level=readings.light_lux,
                )
                # Update LED brightness estimate for next cycle
                # Preset brightness × agency dimmer × activity multiplier
                _base_br = _preset_led_brightness
                _prev_led_brightness = _base_br * _agency_led_brightness * activity_state.brightness_multiplier
                readings.led_brightness = _prev_led_brightness
                # Skip some updates when resting/drowsy (power saving)
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
            pred_error = metacog.observe(readings, anima)

            # Layer 2: Update experiential filter from surprise and dissatisfaction
            if exp_filter:
                if pred_error.surprise > 0.1:
                    _temp_damp = exp_marks.get_effect("temp_salience_dampening") if exp_marks else 0.0
                    exp_filter.update_from_surprise(
                        pred_error.surprise_sources or [], pred_error.surprise,
                        temp_dampening=_temp_damp)
                if preferences:
                    try:
                        _most_unsat = preferences.get_most_unsatisfied({
                            "warmth": anima.warmth, "clarity": anima.clarity,
                            "stability": anima.stability, "presence": anima.presence,
                        })
                        _unsat_dim = _most_unsat[0] if isinstance(_most_unsat, tuple) else _most_unsat
                        if _unsat_dim and _unsat_dim != "none":
                            exp_filter.update_from_dissatisfaction(_unsat_dim)
                    except Exception:
                        pass
                exp_filter.tick()

            # Check if surprise warrants reflection
            should_reflect, reflect_reason = metacog.should_reflect(pred_error)
            if should_reflect:
                reflection = metacog.reflect(pred_error, anima, readings)
                print(f"[Metacog] REFLECTION ({reflect_reason}): {reflection.observation}", file=sys.stderr, flush=True)
                if reflection.discrepancy_description:
                    print(f"[Metacog] DISCREPANCY: {reflection.discrepancy_description}", file=sys.stderr, flush=True)
                _last_reflection_event = _build_broker_reflection_event(
                    reflection,
                    pred_error,
                    self_model=self_model,
                    preferences=preferences,
                    reflect_reason=reflect_reason,
                )

            # ==================== ENHANCED LEARNING INTEGRATION ====================

            # 2b-i. Adaptive Prediction: Learn from what just happened
            if adaptive_model:
                try:
                    observations = {
                        "light": readings.light_lux,
                        "ambient_temp": readings.ambient_temp_c,
                        "humidity": readings.humidity_pct,
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                    }
                    # Remove None values
                    observations = {k: v for k, v in observations.items() if v is not None}
                    adaptive_model.observe(
                        observations,
                        current_light=readings.light_lux,
                        current_temp=readings.ambient_temp_c
                    )
                except Exception as e:
                    print(f"[Learning] Adaptive prediction error: {e}", file=sys.stderr, flush=True)

            # 2b-ii. Update Self-Model with observations
            if self_model:
                try:
                    # Track surprise for self-beliefs
                    self_model.observe_surprise(pred_error.surprise, pred_error.surprise_sources)

                    # Track stability changes
                    if last_state_for_action:
                        prev_stability = last_state_for_action.get("stability", anima.stability)
                        _recovery_bonus = exp_marks.get_effect("stability_recovery_bonus") if exp_marks else 0.0
                        self_model.observe_stability_change(prev_stability, anima.stability, UPDATE_INTERVAL, recovery_bonus=_recovery_bonus)

                    # Track correlations (raw lux includes LED glow — that's fine,
                    # Lumen's LEDs are part of its environment)
                    sensor_vals = {
                        "ambient_temp": readings.ambient_temp_c,
                        "light": readings.light_lux,
                    }
                    anima_vals = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                    }
                    self_model.observe_correlation(sensor_vals, anima_vals)

                    # Track LED-lux proprioception (own outputs affecting inputs)
                    self_model.observe_led_lux(readings.led_brightness, readings.light_lux)

                    # Track time patterns
                    hour = datetime.now().hour
                    self_model.observe_time_pattern(hour, anima.warmth, anima.clarity)

                    # Track temperament baseline (from inner life)
                    if inner_state and inner_state.temperament:
                        self_model.observe_temperament(inner_state.temperament)
                except Exception as e:
                    print(f"[Learning] Self-model error: {e}", file=sys.stderr, flush=True)

            # 2b-iii. Update Preferences from experience
            if preferences:
                try:
                    current_state = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                    }
                    preferences.record_state(current_state)

                    # Record events that shape preferences
                    if pred_error.surprise > 0.4:
                        # High surprise is mildly negative (prefer predictability)
                        preferences.record_event("disruption", -0.2, current_state)
                    elif pred_error.surprise < 0.1 and anima.stability > 0.6:
                        # Low surprise + high stability is positive
                        preferences.record_event("calm", 0.3, current_state)
                except Exception as e:
                    print(f"[Learning] Preference error: {e}", file=sys.stderr, flush=True)

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

            # 2b-v. Action Selection: Choose what to do based on state and preferences
            selected_action = None
            action_predictions = None
            action_pred_context = None
            if action_selector and preferences:
                try:
                    current_state = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                        "last_surprise": pred_error.surprise,
                    }

                    # Get self-model predictions to inform action selection
                    if self_model and pred_error.surprise_sources:
                        try:
                            sources = pred_error.surprise_sources
                            if any("light" in s or "lux" in s for s in sources):
                                action_pred_context = "light_change"
                            elif any("temp" in s for s in sources):
                                action_pred_context = "temp_change"
                            elif anima.stability < 0.3:
                                action_pred_context = "stability_drop"
                            if action_pred_context:
                                action_predictions = self_model.predict_own_response(action_pred_context)
                        except Exception:
                            pass

                    # Layer 1: Get pathway strengths for current context
                    _pathway_strengths = None
                    _pw_ctx = None
                    if pathways:
                        try:
                            _strongest_drive = max(inner_state.drives.values()) if inner_state and inner_state.drives else 0.0
                            _activity_str = activity_state.level.value if activity_state else "active"
                            _satisfaction = preferences.get_overall_satisfaction(current_state)
                            _pw_ctx = discretize_context(
                                surprise=pred_error.surprise,
                                satisfaction=_satisfaction,
                                drive=_strongest_drive,
                                activity=_activity_str,
                            )
                            _pathway_strengths = pathways.get_all_strengths(_pw_ctx)
                        except Exception:
                            pass

                    selected_action = action_selector.select_action(
                        current_state,
                        preferences=preferences,
                        surprise_level=pred_error.surprise,
                        surprise_sources=pred_error.surprise_sources,
                        can_speak=voice is not None,
                        self_predictions=action_predictions,
                        drives=inner_state.drives if inner_state else None,
                        pathway_strengths=_pathway_strengths,
                    )

                    # Execute action effects
                    if selected_action.action_type == ActionType.FOCUS_ATTENTION:
                        sensor = selected_action.parameters.get("sensor")
                        action_selector.set_attention_focus(sensor)
                        print(f"[Agency] Focusing attention on: {sensor}", file=sys.stderr, flush=True)

                    elif selected_action.action_type == ActionType.ADJUST_SENSITIVITY:
                        direction = selected_action.parameters.get("direction", "increase")
                        action_selector.adjust_sensitivity(direction)
                        print(f"[Agency] Sensitivity {direction}d", file=sys.stderr, flush=True)

                    elif selected_action.action_type == ActionType.ASK_QUESTION:
                        # Generate question from metacognition (existing functionality)
                        pass  # Question generation already happens via metacognition

                    elif selected_action.action_type == ActionType.LED_BRIGHTNESS:
                        direction = selected_action.parameters.get("direction", "increase")
                        current = _agency_led_brightness
                        if direction == "increase" and current < 1.0:
                            _agency_led_brightness = min(1.0, current * 1.2)
                            print(f"[Agency] LED brightness increase: {current:.2f} → {_agency_led_brightness:.2f}", file=sys.stderr, flush=True)
                        elif direction == "decrease" and current > 0.05:
                            _agency_led_brightness = max(0.05, current * 0.8)
                            print(f"[Agency] LED brightness decrease: {current:.2f} → {_agency_led_brightness:.2f}", file=sys.stderr, flush=True)

                    # Record state for action outcome learning
                    satisfaction_before = preferences.get_overall_satisfaction(current_state)
                    last_state_for_action = {**current_state, "satisfaction": satisfaction_before}
                    last_action = selected_action
                    _last_pw_ctx = _pw_ctx  # capture context for pathway reinforcement

                except Exception as e:
                    print(f"[Agency] Action selection error: {e}", file=sys.stderr, flush=True)

            # 2b-vi. Record action outcomes (from previous iteration)
            if action_selector and last_action and last_state_for_action and preferences:
                try:
                    current_state = {
                        "warmth": anima.warmth,
                        "clarity": anima.clarity,
                        "stability": anima.stability,
                        "presence": anima.presence,
                    }
                    satisfaction_after = preferences.get_overall_satisfaction(current_state)

                    _floor_reduction = exp_marks.get_effect("exploration_floor_reduction") if exp_marks else 0.0
                    action_selector.record_outcome(
                        last_action,
                        last_state_for_action,
                        current_state,
                        last_state_for_action.get("satisfaction", 0.5),
                        satisfaction_after,
                        pred_error.surprise,
                        exploration_floor_reduction=_floor_reduction,
                    )

                    # Layer 1: Reinforce pathway from outcome (using context from when action was selected)
                    if pathways and _last_pw_ctx:
                        try:
                            _outcome_quality = satisfaction_after - last_state_for_action.get("satisfaction", 0.5)
                            _lr_bonus = exp_marks.get_effect("pathway_lr_bonus") if exp_marks else 0.0
                            pathways.reinforce(_last_pw_ctx, last_action.action_type.value, _outcome_quality, lr_bonus=_lr_bonus)
                        except Exception:
                            pass

                    # Verify self-model predictions against actual outcomes
                    if self_model and action_predictions and action_pred_context:
                        try:
                            actual = {
                                "surprise_likelihood": pred_error.surprise,
                            }
                            if "warmth_change" in action_predictions:
                                actual["warmth_change"] = anima.warmth - last_state_for_action.get("warmth", 0.5)
                            if "clarity_change" in action_predictions:
                                actual["clarity_change"] = anima.clarity - last_state_for_action.get("clarity", 0.5)
                            if "fast_recovery" in action_predictions:
                                # Recovery speed: how much stability rebounded since last observation
                                # Clamp to [0,1] — 0 means no recovery, 1 means full rebound
                                prev_stab = last_state_for_action.get("stability", 0.5)
                                actual["fast_recovery"] = max(0.0, min(1.0, anima.stability - prev_stab + 0.5))
                            self_model.verify_prediction(action_pred_context, action_predictions, actual)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Agency] Outcome recording error: {e}", file=sys.stderr, flush=True)

            # 2b-vii. Exploration check
            if exploration_mgr:
                try:
                    should_explore, explore_reason = exploration_mgr.should_explore(
                        {"stability": anima.stability, "clarity": anima.clarity},
                        pred_error.surprise
                    )
                    if should_explore:
                        exploration_mgr.record_novelty(pred_error.surprise, explore_reason)
                        print(f"[Agency] Exploration triggered: {explore_reason}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[Agency] Exploration error: {e}", file=sys.stderr, flush=True)

            # ==================== END ENHANCED LEARNING ====================

            # 2c. Update Voice with anima state (influences when/how Lumen speaks)
            if voice:
                try:
                    feeling = anima.feeling()
                    voice.update_state(
                        warmth=anima.warmth,
                        clarity=anima.clarity,
                        stability=anima.stability,
                        presence=anima.presence,
                        mood=feeling.get("mood", "neutral")
                    )
                    voice.update_environment(
                        temperature=readings.ambient_temp_c or readings.cpu_temp_c or 22.0,
                        humidity=readings.humidity_pct or 50.0,
                        light_level=readings.light_lux or 500.0
                    )
                except Exception as e:
                    print(f"[StableCreature] Voice update error: {e}", file=sys.stderr, flush=True)

            # 3. Governance Check-in (if bridge available) - runs in background thread
            # Rate limited to every GOVERNANCE_INTERVAL seconds to avoid overwhelming server
            # First check-in always happens (to sync identity), then rate limited
            current_time = time.time()
            should_check_governance = first_check_in or (current_time - last_governance_time >= GOVERNANCE_INTERVAL)

            # Collect results from previous governance future (non-blocking)
            if _governance_future is not None and _governance_future.done():
                try:
                    gov_result = _governance_future.result()
                    if gov_result is not None:
                        last_decision = gov_result["decision"]
                        last_decision_checked_at = gov_result.get("checked_at")
                        last_governance_time = gov_result["time"]
                        if gov_result.get("first"):
                            first_check_in = False
                        if activity_manager and last_decision and last_decision.get("action") == "wait_for_input":
                            activity_manager.record_interaction()
                except asyncio.TimeoutError:
                    last_governance_time = current_time
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    last_governance_time = current_time
                    if "Connection refused" not in str(e) and "Cannot connect" not in str(e):
                        print(f"[StableCreature] Governance error (non-fatal): {e}", file=sys.stderr, flush=True)
                _governance_future = None

            # Collect experiential stats once per tick (reused by governance + shm)
            _exp_state = {}
            if pathways:
                try:
                    _exp_state["pathways"] = pathways.get_stats()
                except Exception:
                    pass
            if exp_filter:
                try:
                    _exp_state["filter"] = exp_filter.get_stats()
                except Exception:
                    pass
            if exp_marks:
                try:
                    _exp_state["marks"] = exp_marks.get_stats()
                except Exception:
                    pass

            # Submit new governance check-in if due and no background task running
            if bridge and should_check_governance and _governance_future is None:
                # Capture current values for the closure (avoid stale references)
                _gov_anima = anima
                _gov_readings = readings
                _gov_identity = identity
                _gov_first = first_check_in
                _gov_time = current_time

                _gov_exp = _exp_state  # capture for closure

                def _do_governance():
                    # check_in internally calls check_availability via circuit breaker,
                    # so no need for a separate check_availability() call.
                    decision = _run_async_in_background(
                        bridge.check_in(
                            _gov_anima, _gov_readings,
                            identity=_gov_identity,
                            is_first_check_in=_gov_first,
                            experiential_summary=_gov_exp or None,
                        ),
                        timeout=15.0  # budget: availability (3+3s) + check-in (3s) + headroom
                    )
                    return {
                        "decision": decision,
                        "time": _gov_time,
                        "first": _gov_first,
                        "checked_at": datetime.now().isoformat(),
                    }

                _governance_future = _bg_executor.submit(_do_governance)
                last_governance_time = current_time  # Prevent re-submit while running

            # 3b. Write to Shared Memory (Broker) - includes governance and metacognition
            shm_data = {
                "timestamp": datetime.now().isoformat(),
                "readings": readings.to_dict(),
                "anima": anima.to_dict(),
                "inner_life": inner_state.to_dict() if inner_state else {},
                "drive_events": _drive_events,
                "eisv": eisv.to_dict(),
                "identity": {
                    "creature_id": identity.creature_id,
                    "name": identity.name,
                    "awakenings": identity.total_awakenings
                },
                "metacognition": {
                    "surprise": pred_error.surprise,
                    "surprise_sources": pred_error.surprise_sources,
                    "cumulative_surprise": metacog._cumulative_surprise,
                    "prediction_confidence": prediction.confidence,
                },
            }
            if _last_reflection_event:
                shm_data["metacognition"]["last_reflection"] = _last_reflection_event
            if last_decision:
                shm_data["governance"] = {
                    **last_decision,
                    "governance_at": last_decision_checked_at or datetime.now().isoformat(),
                }

            # Add activity state if available
            if activity_state:
                shm_data["activity"] = {
                    "level": activity_state.level.value,
                    "brightness_multiplier": activity_state.brightness_multiplier,
                    "reason": activity_state.reason,
                }

            # Add enhanced learning state if available
            if ENHANCED_LEARNING_AVAILABLE:
                learning_state = {}
                if preferences:
                    try:
                        learning_state["preferences"] = {
                            "satisfaction": preferences.get_overall_satisfaction({
                                "warmth": anima.warmth, "clarity": anima.clarity,
                                "stability": anima.stability, "presence": anima.presence
                            }),
                            "most_unsatisfied": preferences.get_most_unsatisfied({
                                "warmth": anima.warmth, "clarity": anima.clarity,
                                "stability": anima.stability, "presence": anima.presence
                            }),
                        }
                    except Exception:
                        pass
                if self_model:
                    try:
                        learning_state["self_beliefs"] = self_model.get_belief_summary()
                    except Exception:
                        pass
                if action_selector:
                    try:
                        learning_state["agency"] = action_selector.get_action_stats()
                    except Exception:
                        pass
                if adaptive_model:
                    try:
                        learning_state["prediction_accuracy"] = adaptive_model.get_accuracy_stats()
                    except Exception:
                        pass
                if learning_state:
                    shm_data["learning"] = learning_state

            # Experiential accumulation state (collected once per tick above)
            if _exp_state:
                shm_data["experiential"] = _exp_state

            # WiFi status from kernel (no subprocess)
            try:
                net_stats = psutil.net_if_stats()
                wlan = net_stats.get("wlan0")
                shm_data["wifi_connected"] = bool(wlan and wlan.isup)
            except Exception:
                shm_data["wifi_connected"] = False

            # Agency LED brightness: broker's desired manual brightness for server to apply
            if _agency_led_brightness != 1.0:
                shm_data["agency_led_brightness"] = _agency_led_brightness

            shm_client.write(shm_data)

            # 4. Render Face
            face_state = derive_face_state(anima)

            # Modify face based on activity state (sleeping/drowsy)
            if activity_state and activity_state.level == ActivityLevel.RESTING:
                # Eyes closed when resting
                face_state.eyes = EyeState.CLOSED
                face_state.eye_openness = 0.0
            elif activity_state and activity_state.level == ActivityLevel.DROWSY:
                # Droopy eyes when drowsy
                face_state.eyes = EyeState.DROOPY
                face_state.eye_openness = 0.4

            ascii_face = face_to_ascii(face_state)
            
            # Clear screen (terminal) - use ANSI codes to prevent flicker
            # \033[2J = clear screen, \033[H = move cursor to top-left
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
            
            # DB writes removed: server owns identity DB (Option 1 - no contention).
            # Broker only writes to shared memory; server does record_state/heartbeat.

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
                            # Approximate observation count from total alive time
                            _obs_count = int(identity.total_alive_seconds / UPDATE_INTERVAL) if identity.total_alive_seconds else 0
                            _belief_confs = {}
                            if self_model:
                                try:
                                    _beliefs = self_model.get_belief_summary()
                                    _belief_confs = {
                                        k: v.get("confidence", 0.0)
                                        for k, v in _beliefs.items()
                                        if isinstance(v, dict)
                                    }
                                except Exception:
                                    pass
                            # Read drawing count from DB (server writes to counters table)
                            _drawing_count = 0
                            try:
                                import sqlite3 as _sql
                                _c = _sql.connect(db_path, timeout=2.0)
                                _row = _c.execute("SELECT value FROM counters WHERE name = 'drawings_observed'").fetchone()
                                if _row:
                                    _drawing_count = _row[0]
                                _c.close()
                            except Exception:
                                pass
                            # Read question count from messages JSON (server writes)
                            _question_count = 0
                            try:
                                import json as _json
                                _msg_path = Path.home() / ".anima" / "messages.json"
                                if _msg_path.exists():
                                    _msgs = _json.loads(_msg_path.read_text()).get("messages", [])
                                    _question_count = sum(1 for m in _msgs if m.get("msg_type") == "question")
                            except Exception:
                                pass
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
        # Clean shutdown
        print("[StableCreature] Shutting down...")

        # Save enhanced learning state
        if ENHANCED_LEARNING_AVAILABLE:
            try:
                if adaptive_model:
                    adaptive_model._save_patterns()
                    print("[StableCreature] Saved adaptive prediction patterns")
                if preferences:
                    preferences._save()
                    print("[StableCreature] Saved preferences")
                if self_model:
                    self_model.save()
                    print("[StableCreature] Saved self-model")
                if _inner_life:
                    _inner_life.save()
                    print("[StableCreature] Saved inner life (temperament + drives)")
            except Exception as e:
                print(f"[StableCreature] Error saving learning state: {e}")

        # Save experiential accumulation state
        if exp_filter:
            try:
                exp_filter.save()
                print("[StableCreature] Saved experiential filter")
            except Exception:
                pass
        if pathways:
            try:
                pathways.close()
                print("[StableCreature] Closed pathway DB")
            except Exception:
                pass
        if exp_marks:
            try:
                exp_marks.close()
                print("[StableCreature] Closed experiential marks DB")
            except Exception:
                pass

        if voice:
            try:
                voice.stop()
            except Exception:
                pass

        # Persist store state first (synchronous SQLite, no event loop needed)
        if store:
            store.sleep()
            store.close()

        # Close UNITARES bridge session while event loop is still running
        if bridge:
            try:
                asyncio.run_coroutine_threadsafe(bridge.close(), _bg_loop).result(timeout=3)
            except Exception:
                pass

        # Shut down background executor after bridge is closed (avoids deadlock)
        try:
            _bg_executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python <3.9 doesn't have cancel_futures
            _bg_executor.shutdown(wait=False)
        except Exception:
            pass

        # Stop the loop and wait for its thread to actually exit before
        # closing. call_soon_threadsafe(stop) is asynchronous, so closing on
        # the next line races a still-running loop and raises
        # "RuntimeError: Cannot close a running event loop".
        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
        try:
            _bg_thread.join(timeout=3)
        except Exception:
            pass
        if not _bg_loop.is_running():
            try:
                _bg_loop.close()
            except Exception:
                pass
        shm_client.clear()  # Clean up shared memory
        print("[StableCreature] Stopped.")

def main():
    """Entry point for pyproject.toml scripts."""
    run_creature()


if __name__ == "__main__":
    main()
