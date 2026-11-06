"""Creature runtime mixin and helper functions extracted from stable_creature.py.

Contains subsystem initialization, sensor read retries, LED/light attribution,
body anima & inner-life updates, metacognition/experiential filter updates,
adaptive prediction & self-model/preference learning, legacy broker agency,
governance check-in scheduling, SHM snapshot construction, and clean shutdown.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from .anima import sense_self
from .db_paths import BROKER_AGENCY_DB, resolve_db_path
from .display.leds.brightness import estimate_instantaneous_brightness
from .eisv_mapper import BODY_EISV_PROJECTION_SCHEMA, anima_to_body_eisv_projection
from .governance_passthrough import (
    passthrough_path_from_env,
    read_shadow_governance,
    stale_seconds_from_env,
)
from .identity import CreatureIdentity
from .learning_events import drain_learning_events
from .light_attribution import gated_external_light_lux, read_led_proprioception
from .sensors import get_sensors
from .shared_memory import SharedMemoryClient
from .unitares_bridge import UnitaresBridge


def _sc():
    return sys.modules.get("anima_mcp.stable_creature")


def _resolve_read_led_proprioception():
    sc = _sc()
    return getattr(sc, "read_led_proprioception", read_led_proprioception) if sc is not None else read_led_proprioception


def _build_preference_state(anima, readings, *, external_light_lux) -> dict:
    """Build normalized preference occupancy from felt and environmental state."""
    state = {
        "warmth": anima.warmth,
        "clarity": anima.clarity,
        "stability": anima.stability,
        "presence": anima.presence,
    }

    # Raw VEML7700 lux includes DotStar self-glow. Only the attribution-gated
    # external residual is allowed to teach an environmental preference.
    if external_light_lux is not None:
        try:
            light = float(external_light_lux)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(light):
                state["light"] = max(0.0, min(1.0, light / 1000.0))

    ambient_temp_c = getattr(readings, "ambient_temp_c", None)
    if ambient_temp_c is not None:
        try:
            temperature = float(ambient_temp_c)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(temperature):
                state["temperature"] = max(
                    0.0,
                    min(1.0, (temperature - 15.0) / 25.0),
                )

    return state


def _run_coroutine_with_timeout(coro, loop, timeout: float = 10.0):
    """Run on the broker loop and cancel the task if the caller times out."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        # Let the loop deliver cancellation before a shutdown or replacement
        # task can strand the coroutine as pending.
        drained = threading.Event()
        try:
            loop.call_soon_threadsafe(drained.set)
            drained.wait(timeout=1.0)
        except RuntimeError:
            pass
        raise


def _stdout_is_terminal() -> bool:
    """Whether the live ASCII face has a terminal to draw on."""
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _apply_belief_update_bonus(self_model, exp_marks) -> None:
    """Hand the self-model the learning-rate bonus its earned marks grant.

    Until 2026-10-15 this ran before ``exp_marks`` was created, so it always
    saw None: Lumen earned "Adolescent" (+0.15) on 2026-03-18 and never got
    it. It now runs after both exist and again whenever a mark is earned.
    """
    if self_model is None or exp_marks is None:
        return
    self_model.belief_update_bonus = exp_marks.get_effect("belief_update_bonus")


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


def _pair_led_action_to_light_capture(readings):
    """Return the efference copy nearest the physical light capture window."""
    observed_at = getattr(readings, "light_observed_at", None)
    if observed_at is None:
        return None
    try:
        observed_unix = observed_at.timestamp()
        precision = getattr(readings, "light_observed_precision_seconds", None)
        if isinstance(precision, (int, float)):
            observed_unix += max(0.0, float(precision)) / 2.0
        return _resolve_read_led_proprioception()(
            max_age_seconds=2.0,
            at_time=observed_unix,
            max_alignment_error_seconds=0.75,
        )
    except (OSError, OverflowError, ValueError):
        return None


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


def broker_agency_enabled() -> bool:
    """Whether the retired broker TD loop is explicitly enabled for rollback."""
    return os.environ.get("ANIMA_BROKER_AGENCY_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def broker_voice_enabled() -> bool:
    """Whether this body broker should own audio devices.

    Standalone ``anima-creature`` keeps its historical voice default. The
    deployed systemd unit disables it explicitly because the MCP mind service
    is the sole voice owner; two processes probing the same microphone and
    speaker is not a capability.
    """
    configured = os.environ.get("ANIMA_BROKER_VOICE_ENABLED")
    if configured is None:
        configured = os.environ.get("ANIMA_VOICE_ENABLED", "true")
    return configured.strip().lower() in {"1", "true", "yes", "on"}



def _load_persistent_identity():
    """Load the canonical identity without claiming an awakening.

    The MCP server owns the identity lifecycle and its SQLite writes. The body
    broker only needs a snapshot for SHM metadata and experiential thresholds;
    calling ``IdentityStore.wake()`` here made both services race through the
    same write transaction at startup.
    """
    import sqlite3

    db_path = Path(resolve_db_path())
    print(f"[StableCreature] Identity persistence: {db_path}")

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        anima_id = os.environ.get("ANIMA_ID")
        if anima_id:
            row = conn.execute(
                "SELECT * FROM identity WHERE creature_id = ?", (anima_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM identity LIMIT 1").fetchone()

        if row is None:
            raise LookupError("canonical identity has not been created yet")

        anima_id = row["creature_id"]
        print(f"[StableCreature] Using existing identity: {anima_id[:8]}...")

        row_keys = set(row.keys())
        last_heartbeat_at = None
        raw_heartbeat = row["last_heartbeat_at"] if "last_heartbeat_at" in row_keys else None
        if raw_heartbeat:
            try:
                last_heartbeat_at = datetime.fromisoformat(raw_heartbeat)
            except (TypeError, ValueError):
                pass

        return CreatureIdentity(
            creature_id=anima_id,
            born_at=datetime.fromisoformat(row["born_at"]),
            total_awakenings=int(row["total_awakenings"] or 0),
            total_alive_seconds=float(row["total_alive_seconds"] or 0.0),
            name=row["name"],
            name_history=json.loads(row["name_history"] or "[]"),
            current_awakening_at=None,
            last_heartbeat_at=last_heartbeat_at,
            metadata=json.loads(row["metadata"] or "{}"),
        )
    finally:
        conn.close()


def _direct_i2c_initialization_failed(sensors) -> bool:
    """Whether a direct-I2C backend unexpectedly failed to open its bus.

    ``PiSensors`` deliberately keeps ``_i2c`` unset when it consumes the
    Elixir broker's shadow file. That is the healthy deployed ownership mode,
    not a hardware failure.
    """
    return (
        getattr(sensors, "_i2c", object()) is None
        and not getattr(sensors, "_env_shadow_path", None)
    )



class CreatureRuntimeMixin:
    """Phase methods for StableCreature hardware broker loop."""

    def init_identity_and_sensors(self):
        # Initialize components with error handling
        identity = None
        db_path = str(resolve_db_path())
        try:
            identity = _load_persistent_identity()
        except Exception as e:
            print(f"[StableCreature] WARNING: Identity store failed ({e}) - using fallback identity")
            print("[StableCreature] Broker will continue (sensors -> shared memory). Server can repair DB.")
            import uuid
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

        # Initialize sensors - allow graceful degradation if hardware unavailable
        try:
            sensors = get_sensors()
            # Direct mode requires I2C. Shadow mode deliberately leaves it unopened
            # because anima-broker-ex is the sole environmental-sensor owner.
            if _direct_i2c_initialization_failed(sensors):
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

        # Phase-2 Elixir broker cutover (Option A, addendum 2026-07-01): when set,
        # the Elixir broker owns the UNITARES governance client and this broker
        # copies the governance slice from the shadow envelope instead of making
        # its own check-ins. The bridge object is kept constructed so unsetting
        # the flag and restarting is a complete rollback.
        gov_shadow_path = passthrough_path_from_env()
        gov_shadow_stale_s = stale_seconds_from_env()
        if gov_shadow_path:
            print(
                f"[StableCreature] governance from SHM shadow: {gov_shadow_path} "
                f"(own UNITARES check-ins disabled; slices older than "
                f"{gov_shadow_stale_s:.0f}s ignored)"
            )

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
        if getattr(_sc(), 'VOICE_AVAILABLE', False) and broker_voice_enabled():
            try:
                candidate = _sc().AutonomousVoice()
                if candidate.start():
                    voice = candidate
                    caps = voice.capabilities
                    print(
                        "[StableCreature] Voice active "
                        f"(hearing={'yes' if caps['hearing'] else 'no'}, "
                        f"speaking={'yes' if caps['speaking'] else 'no'})"
                    )
                else:
                    print("[StableCreature] Voice unavailable; body broker continuing silently")
            except Exception as e:
                print(f"[StableCreature] WARNING: Voice initialization failed: {e}")
                voice = None

        print(f"[StableCreature] Creature '{identity.name or '(unnamed)'}' is alive.")
        print("[StableCreature] Entering main loop...")
        return identity, db_path, sensors, bridge, gov_shadow_path, gov_shadow_stale_s, shm_client, voice

    def init_enhanced_learning(self, legacy_broker_agency: bool):
        # Initialize Enhanced Learning Systems (genuine agency)
        adaptive_model = None
        preferences = None
        self_model = None
        action_selector = None
        exploration_mgr = None
        memory_retriever = None

        sc = _sc()
        if getattr(sc, 'ENHANCED_LEARNING_AVAILABLE', False):
            try:
                if sc._has_module("adaptive_prediction"):
                    adaptive_model = sc.get_adaptive_prediction_model()
                    print("[StableCreature] Adaptive prediction active - Lumen learns from experience")

                if sc._has_module("preferences"):
                    preferences = sc.get_preference_system(read_only=False)
                    print("[StableCreature] Preferences active - Lumen develops values")

                if sc._has_module("self_model"):
                    self_model = sc.get_self_model(read_only=False)
                    print("[StableCreature] Self-model active - Lumen has beliefs about itself")

                if sc._has_module("agency") and legacy_broker_agency:
                    # Deliberately NOT db_path. A rollback must remain isolated
                    # from the server's authoritative action table; merging it
                    # would restore the second-writer defect (#123).
                    action_selector = sc.get_action_selector(db_path=str(BROKER_AGENCY_DB))
                    exploration_mgr = sc.get_exploration_manager()
                    print("[StableCreature] Legacy broker agency enabled by operator")
                elif sc._has_module("agency"):
                    print("[StableCreature] Broker agency retired - server is the sole action learner")

                if sc._has_module("memory_retrieval"):
                    memory_retriever = sc.get_memory_retriever()
                    print("[StableCreature] Memory retrieval active - past informs present")
            except Exception as e:
                print(f"[StableCreature] Enhanced learning init error: {e}")
        return adaptive_model, preferences, self_model, action_selector, exploration_mgr, memory_retriever

    def parse_governance_interval(self) -> float:
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
        return GOVERNANCE_INTERVAL

    def load_led_brightness_preset(self) -> float:
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
        return _preset_led_brightness

    def drain_inbox_if_due(
        self, self_model: Any, preferences: Any, last_learning_event_drain: float, last_learning_event_failure_log: float
    ) -> tuple[float, float]:
        # Apply durable semantic learning events from the server.  This is
        # the only cross-process mutation seam for broker-owned JSON state.
        if time.time() - last_learning_event_drain >= 5.0:
            try:
                drained = drain_learning_events(
                    self_model=self_model,
                    preferences=preferences,
                )
                now = time.time()
                should_report_failure = (
                    drained["failed"]
                    and now - last_learning_event_failure_log >= 300.0
                )
                if drained["processed"] or drained["rejected"] or should_report_failure:
                    print(f"[LearningInbox] {drained}", file=sys.stderr, flush=True)
                if should_report_failure:
                    last_learning_event_failure_log = now
            except Exception as e:
                print(f"[LearningInbox] drain error: {e}", file=sys.stderr, flush=True)
            last_learning_event_drain = time.time()
        return last_learning_event_drain, last_learning_event_failure_log

    def read_sensors_with_retry(self, sensors: Any) -> Any:
        max_retries = getattr(_sc(), 'MAX_RETRIES', 3)
        retry_delay = getattr(_sc(), 'RETRY_DELAY', 0.5)
        readings = None
        for attempt in range(max_retries):
            try:
                readings = sensors.read()
                break
            except Exception as e:
                print(f"[StableCreature] Sensor read error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        return readings

    def compute_led_and_light_attribution(
        self, readings: Any, _latest_led_proprioception: Any, _prev_led_brightness: float, self_model: Any
    ) -> tuple[Any, Any, str, Any, Any]:
        light_attribution = None
        _paired_led_proprioception = _pair_led_action_to_light_capture(readings)

        # 1b. LED proprioception.  The actual action copy is authoritative;
        # the old preset reconstruction remains an explicitly labelled
        # fallback and is never allowed to train the residual model.
        _observed_led_proprioception = (
            _paired_led_proprioception or _latest_led_proprioception
        )
        if _observed_led_proprioception is not None:
            readings.led_brightness = _observed_led_proprioception["brightness"]
            _led_brightness_source = "led_hardware_controller"
        else:
            readings.led_brightness = estimate_instantaneous_brightness(
                _prev_led_brightness
            )
            _led_brightness_source = "broker_brightness_estimate"

        # Close the efference-copy loop before deriving felt state. The
        # physical sample remains raw; only a capture-aligned, statistically
        # ready residual may enter environmental consumers below.
        _external_light_for_environment = None
        if self_model:
            try:
                _light_capture_unix = (
                    _paired_led_proprioception.get(
                        "paired_to_light_observed_at_unix"
                    )
                    if _paired_led_proprioception is not None
                    else None
                )
                light_attribution = self_model.observe_led_lux(
                    readings.led_brightness,
                    readings.light_lux,
                    led_proprioception=_paired_led_proprioception,
                    brightness_source=_led_brightness_source,
                    observed_at=_light_capture_unix,
                    observation_id=(
                        readings.light_observed_at.isoformat()
                        if readings.light_observed_at is not None
                        else None
                    ),
                )
                _external_light_for_environment = gated_external_light_lux(
                    light_attribution
                )
            except Exception as e:
                print(
                    f"[LightAttribution] Observation error: {e}",
                    file=sys.stderr,
                    flush=True,
                )
        return (
            _paired_led_proprioception,
            _observed_led_proprioception,
            _led_brightness_source,
            light_attribution,
            _external_light_for_environment,
        )

    def update_body_anima_and_inner_life(
        self,
        readings: Any,
        exp_filter: Any,
        _external_light_for_environment: Any,
        _mood_momentum: Any,
        _inner_life: Any,
        last_body_update_at: float,
    ) -> tuple[Any, Any, list[dict[str, Any]], Any, float]:
        # 2. Update Anima State with provenance-gated environmental light.
        # Layer 2 salience remains visible as attention provenance. Light
        # salience does not amplify the clarity measurement.
        _body_now = time.monotonic()
        _body_elapsed = max(0.0, _body_now - last_body_update_at)
        last_body_update_at = _body_now
        _salience = exp_filter.get_all_saliences() if exp_filter else None
        raw_anima = sense_self(
            readings,
            salience_weights=_salience,
            external_light_lux=_external_light_for_environment,
        )
        if raw_anima.clarity_attribution is not None:
            raw_anima.clarity_attribution["authority"] = (
                "broker_sensor_fusion"
            )
        anima = _mood_momentum.smooth(
            raw_anima, elapsed_seconds=_body_elapsed,
        )
        inner_state = _inner_life.update(
            raw_anima, anima, elapsed_seconds=_body_elapsed,
        )

        # 2-i-a. Check for social boost signal (server writes on interaction)
        _boost_path = Path("/dev/shm/anima_social_boost")
        if _boost_path.exists():
            try:
                _boost_path.unlink()
                _inner_life.apply_social_boost()
            except Exception:
                pass

        # 2-i-b. Drive-request delivery acks: the server writes an ack file
        # once a request question ACTUALLY posted to the board; only then
        # does the cooldown commit (inner_life.ack_request). Until acked,
        # the request re-emits every tick — a one-shot event would vanish
        # in a single 2s SHM window whenever the server is down or
        # restarting (documented restart window: 2 minutes).
        for _dim in ("warmth", "clarity", "stability", "presence"):
            _ack_path = Path(f"/dev/shm/anima_drive_ack_{_dim}")
            if _ack_path.exists():
                try:
                    _ack_path.unlink()
                    # Stale ack for an inactive dim (crashed after post,
                    # cooldown saved) is simply cleaned up; an active one
                    # commits the cooldown before this tick re-emits.
                    _inner_life.ack_request(_dim, time.time())
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
        # Active requests ride along every tick until acked.
        for ev in _inner_life.get_active_requests():
            obs_text = _inner_life.get_observation_text(ev)
            if obs_text:
                _drive_events.append({
                    "text": obs_text,
                    "dimension": ev.dimension,
                    "event_type": ev.event_type,
                    "drive_value": round(ev.drive_value, 3),
                })

        # 2a. Project body state into EISV-shaped telemetry. UNITARES may
        # consume this measurement, but it is not UNITARES's own state.
        body_projection = anima_to_body_eisv_projection(anima, readings)
        return anima, inner_state, _drive_events, body_projection, last_body_update_at

    def evaluate_metacognition_and_filter(
        self,
        metacog: Any,
        readings: Any,
        anima: Any,
        exp_filter: Any,
        exp_marks: Any,
        preferences: Any,
        self_model: Any,
        _last_reflection_event: Any,
    ) -> tuple[Any, bool, Any]:
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
                    _unsat_satisfaction = (
                        _most_unsat[1]
                        if isinstance(_most_unsat, tuple) and len(_most_unsat) > 1
                        else None
                    )
                    if _unsat_dim and _unsat_dim != "none":
                        exp_filter.update_from_dissatisfaction(
                            _unsat_dim,
                            _unsat_satisfaction,
                        )
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
        return pred_error, should_reflect, _last_reflection_event

    def run_adaptive_and_self_model_learning(
        self,
        adaptive_model: Any,
        self_model: Any,
        preferences: Any,
        readings: Any,
        anima: Any,
        pred_error: Any,
        inner_state: Any,
        _external_light_for_environment: Any,
        _adaptive_context_light: Any,
        _adaptive_context_temp: Any,
        last_self_model_state: Any,
        last_self_model_observed_at: float,
    ) -> tuple[Any, Any, Any, float]:
        # ==================== ENHANCED LEARNING INTEGRATION ====================

        # 2b-i. Adaptive Prediction: Learn from what just happened
        if adaptive_model:
            try:
                observations = {
                    # Accuracy is a clarity input, so self-induced DotStar
                    # changes must not count as environmental prediction
                    # failures. The raw sensor has its own explicitly
                    # proprioceptive metacognition channel; this learner sees
                    # external lux only after the attribution gate opens.
                    "external_light_lux": _external_light_for_environment,
                    "ambient_temp": readings.ambient_temp_c,
                    "humidity": readings.humidity_pct,
                    "warmth": anima.warmth,
                    "clarity": anima.clarity,
                    "stability": anima.stability,
                    "presence": anima.presence,
                }
                # Remove None values
                observations = {k: v for k, v in observations.items() if v is not None}

                # Close the prediction loop BEFORE folding the new values in,
                # so the model is scored on a genuine forecast rather than on
                # a value it has already seen.
                #
                # Nothing did this. record_prediction_error() has never had a
                # production caller in the repo's history, so _recent_errors
                # stayed permanently empty, get_accuracy_stats() always
                # returned insufficient_data, and clarity's LARGEST component
                # — prediction_accuracy, weight 0.50 of 1.10, 45% of the
                # dimension — fell back to the literal 0.5 on every one of
                # 255,973 samples. The docstring calls it "the core of
                # internal seeing"; it was a placeholder.
                for _var, _actual in observations.items():
                    try:
                        _predicted, _ = adaptive_model.predict(
                            _var,
                            current_light=_adaptive_context_light,
                            current_temp=_adaptive_context_temp,
                        )
                        if _predicted is not None:
                            adaptive_model.record_prediction_error(
                                _var, _predicted, _actual
                            )
                    except Exception:
                        pass  # Scoring must never break observation

                adaptive_model.observe(
                    observations,
                    current_light=_adaptive_context_light,
                    current_temp=_adaptive_context_temp,
                )
                _adaptive_context_light = _external_light_for_environment
                _adaptive_context_temp = readings.ambient_temp_c
            except Exception as e:
                print(f"[Learning] Adaptive prediction error: {e}", file=sys.stderr, flush=True)

        # 2b-ii. Update Self-Model with observations
        if self_model:
            try:
                # Track surprise for self-beliefs
                self_model.observe_surprise(pred_error.surprise, pred_error.surprise_sources)

                # Track body recovery on every sample so elapsed time and
                # gradual drop/recovery episodes remain observable.
                # This used to depend on last_state_for_action, which only
                # exists when retired broker agency is explicitly enabled.
                _self_model_now = time.monotonic()
                if last_self_model_state is not None:
                    _self_model_dt = max(
                        0.0, _self_model_now - last_self_model_observed_at
                    )
                    _prev_stability = last_self_model_state["stability"]
                    _prev_warmth = last_self_model_state["warmth"]
                    self_model.observe_stability_change(
                        _prev_stability,
                        anima.stability,
                        _self_model_dt,
                    )
                    self_model.observe_warmth_change(
                        _prev_warmth,
                        anima.warmth,
                        _self_model_dt,
                    )
                last_self_model_state = {
                    "stability": anima.stability,
                    "warmth": anima.warmth,
                }
                last_self_model_observed_at = _self_model_now

                # Temperature is physically direct. VEML7700 lux is not:
                # it mixes room light with DotStar self-glow, so it must not
                # enter the environmental light→warmth belief here.
                sensor_vals = {
                    "ambient_temp": readings.ambient_temp_c,
                }
                anima_vals = {
                    "warmth": anima.warmth,
                    "clarity": anima.clarity,
                    "stability": anima.stability,
                }
                self_model.observe_correlation(sensor_vals, anima_vals)

                # The v4 channel contract promotes only a statistically
                # ready external-light residual into the environmental
                # correlation belief. The explanatory LED→lux belief and
                # quantitative residual keep their separate evidence gates.
                _external_lux = _external_light_for_environment
                if _external_lux is not None:
                    self_model.observe_correlation(
                        {"light": _external_lux},
                        {"warmth": anima.warmth},
                    )

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
                current_state = _build_preference_state(
                    anima,
                    readings,
                    external_light_lux=_external_light_for_environment,
                )
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
        return _adaptive_context_light, _adaptive_context_temp, last_self_model_state, last_self_model_observed_at

    def run_legacy_broker_agency(
        self,
        action_selector: Any,
        exploration_mgr: Any,
        preferences: Any,
        self_model: Any,
        pathways: Any,
        exp_marks: Any,
        voice: Any,
        anima: Any,
        pred_error: Any,
        inner_state: Any,
        activity_state: Any,
        _agency_led_brightness: float,
        last_action: Any,
        last_state_for_action: Any,
        _last_pw_ctx: Any,
    ) -> tuple[float, Any, Any, Any]:
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
                        _pw_ctx = _sc().discretize_context(
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
                if selected_action.action_type == _sc().ActionType.FOCUS_ATTENTION:
                    sensor = selected_action.parameters.get("sensor")
                    action_selector.set_attention_focus(sensor)
                    print(f"[Agency] Focusing attention on: {sensor}", file=sys.stderr, flush=True)

                elif selected_action.action_type == _sc().ActionType.ADJUST_SENSITIVITY:
                    direction = selected_action.parameters.get("direction", "increase")
                    action_selector.adjust_sensitivity(direction)
                    print(f"[Agency] Sensitivity {direction}d", file=sys.stderr, flush=True)

                elif selected_action.action_type == _sc().ActionType.ASK_QUESTION:
                    # Generate question from metacognition (existing functionality)
                    pass  # Question generation already happens via metacognition

                elif selected_action.action_type == _sc().ActionType.LED_BRIGHTNESS:
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
        return _agency_led_brightness, last_action, last_state_for_action, _last_pw_ctx

    def update_broker_voice(
        self, voice: Any, anima: Any, readings: Any, _external_light_for_environment: Any
    ) -> None:
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
                    light_level=_external_light_for_environment,
                )
            except Exception as e:
                print(f"[StableCreature] Voice update error: {e}", file=sys.stderr, flush=True)

    def process_governance_and_collect_exp(
        self,
        bridge: Any,
        gov_shadow_path: Any,
        activity_manager: Any,
        pathways: Any,
        exp_filter: Any,
        exp_marks: Any,
        action_selector: Any,
        anima: Any,
        readings: Any,
        identity: Any,
        light_attribution: Any,
        _bg_executor: Any,
        _run_async_in_background: Any,
        _governance_future: Any,
        first_check_in: bool,
        last_governance_time: float,
        last_decision: Any,
        last_decision_checked_at: Any,
        GOVERNANCE_INTERVAL: float,
    ) -> tuple[Any, bool, float, Any, Any, dict[str, Any]]:
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
                _exp_state["marks"] = exp_marks.get_stats(
                    legacy_agency=action_selector is not None)
            except Exception:
                pass

        # Submit new governance check-in if due and no background task running
        # (never under SHM passthrough — the Elixir broker owns the client)
        if bridge and gov_shadow_path is None and should_check_governance and _governance_future is None:
            # Capture current values for the closure (avoid stale references)
            _gov_anima = anima
            _gov_readings = readings
            _gov_identity = identity
            _gov_first = first_check_in
            _gov_time = current_time
            _gov_light_attribution = light_attribution

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
                        light_attribution=_gov_light_attribution,
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
        return _governance_future, first_check_in, last_governance_time, last_decision, last_decision_checked_at, _exp_state

    def build_and_write_shm_snapshot(
        self,
        shm_client: Any,
        anima: Any,
        readings: Any,
        inner_state: Any,
        _drive_events: Any,
        body_projection: Any,
        identity: Any,
        pred_error: Any,
        metacog: Any,
        prediction: Any,
        _observed_led_proprioception: Any,
        _led_brightness_source: str,
        light_attribution: Any,
        _last_reflection_event: Any,
        gov_shadow_path: Any,
        gov_shadow_stale_s: float,
        last_decision: Any,
        last_decision_checked_at: Any,
        activity_state: Any,
        preferences: Any,
        self_model: Any,
        action_selector: Any,
        adaptive_model: Any,
        _exp_state: dict[str, Any],
        _agency_led_brightness: float,
    ) -> None:
        # 3b. Write to Shared Memory (Broker) - includes governance and metacognition
        body_anima = anima.to_dict()
        body_vector = body_projection.to_dict()
        shm_data = {
            "timestamp": datetime.now().isoformat(),
            "readings": readings.to_dict(),
            "body_anima": body_anima,
            "anima": body_anima,
            "inner_life": inner_state.to_dict() if inner_state else {},
            "drive_events": _drive_events,
            "body_eisv_projection": body_vector,
            "eisv": body_vector,
            "eisv_source": "body_eisv_projection_legacy_alias",
            "state_space_provenance": {
                "body_anima": {
                    "source": "broker_sensor_fusion",
                    "role": "physical_self_sense",
                },
                "anima": {
                    "alias_of": "body_anima",
                    "deprecated": True,
                },
                "body_eisv_projection": {
                    "schema": BODY_EISV_PROJECTION_SCHEMA,
                    "source": "anima_sensor_projection",
                    "role": "body_measurement",
                },
                "eisv": {
                    "alias_of": "body_eisv_projection",
                    "deprecated": True,
                },
            },
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
            "led_proprioception": (
                _observed_led_proprioception
                if _observed_led_proprioception is not None
                else {
                    "source": _led_brightness_source,
                    "brightness": readings.led_brightness,
                    "optical_drive": None,
                    "fresh": False,
                }
            ),
        }
        if light_attribution is not None:
            shm_data["light_attribution"] = light_attribution
        if anima.clarity_attribution is not None:
            shm_data["clarity_attribution"] = anima.clarity_attribution
        if _last_reflection_event:
            shm_data["metacognition"]["last_reflection"] = _last_reflection_event
        if gov_shadow_path is not None:
            # Phase-2 passthrough: the slice carries its own governance_at;
            # stale/missing yields nothing this tick and the MCP server's
            # 240s fallback covers — same contract as bridge-down.
            _shadow_gov = read_shadow_governance(gov_shadow_path, gov_shadow_stale_s)
            if _shadow_gov:
                shm_data["governance"] = _shadow_gov
        elif last_decision:
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
        if getattr(_sc(), 'ENHANCED_LEARNING_AVAILABLE', False):
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

    def gather_mark_check_inputs(self, identity: Any, self_model: Any, db_path: str) -> tuple[int, dict[str, float], int, int]:
        # Approximate observation count from total alive time
        _obs_count = int(identity.total_alive_seconds / getattr(_sc(), 'UPDATE_INTERVAL', 2.0)) if identity.total_alive_seconds else 0
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
        return _obs_count, _belief_confs, _drawing_count, _question_count

    def shutdown_creature_resources(
        self,
        adaptive_model: Any,
        preferences: Any,
        self_model: Any,
        _inner_life: Any,
        exp_filter: Any,
        pathways: Any,
        exp_marks: Any,
        voice: Any,
        bridge: Any,
        _bg_loop: Any,
        _bg_thread: Any,
        _bg_executor: Any,
        shm_client: Any,
    ) -> None:
        # Clean shutdown
        print("[StableCreature] Shutting down...")

        # Save enhanced learning state
        if getattr(_sc(), 'ENHANCED_LEARNING_AVAILABLE', False):
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
