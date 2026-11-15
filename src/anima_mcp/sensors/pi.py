"""
Real Pi sensors - BrainCraft HAT and connected sensors.

Only imported when running on actual Pi hardware.

Hardware:
- BrainCraft HAT: Display (240x240 TFT), LEDs (3 DotStar), sensors
  - AHT20: Temperature + humidity
  - VEML7700: Ambient light sensor
  - BMP280: Barometric pressure + temperature

Neural signals derived from Pi's computational state (computational proprioception).
"""

import os
import sys
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from .base import SensorBackend, SensorReadings
from ..config import LIGHT_SENSOR_EMA_ALPHA, VEML7700_CAPTURE_SUPPORT_SECONDS


class PiSensors(SensorBackend):
    """
    Real Raspberry Pi sensors via BrainCraft HAT.

    BrainCraft HAT provides:
    - Display (240x240 TFT)
    - LEDs (3 DotStar)
    - Sensors: Temperature, humidity, light, pressure

    Neural signals derived from Pi's own computational state (computational proprioception).

    Includes automatic re-initialization of failed I2C sensors with exponential
    backoff to recover from transient bus errors (e.g. power sags).
    """

    # Re-initialization thresholds
    _REINIT_FAILURE_THRESHOLD = 5    # Consecutive None reads before attempting re-init
    _REINIT_INITIAL_BACKOFF = 10     # Seconds before first re-init attempt
    _REINIT_MAX_BACKOFF = 300        # Max seconds between re-init attempts (5 min)
    _REINIT_BACKOFF_FACTOR = 2.0     # Exponential backoff multiplier

    def __init__(self):
        """Initialize Pi sensors."""
        self._i2c = None
        self._aht = None
        self._light_sensor = None
        self._bmp280 = None
        self._last_pressure = None
        self._smoothed_lux: Optional[float] = None  # EMA for light sensor
        self._last_smoothed_light_capture: Optional[tuple[str, float]] = None
        self._shadow_light_observed_at: Optional[datetime] = None
        self._shadow_light_precision_s: Optional[float] = None

        # Phase-1 Elixir broker cutover: when set, the three I2C environment
        # sensors (AHT20 / VEML7700 / BMP280) are NOT initialized here — the
        # Elixir broker owns them and publishes their channels to this SHM
        # shadow path. System stats, throttle, and neural bands are unaffected.
        # Rollback: unset the env var and restart; this process re-opens I2C.
        _shadow = os.environ.get("ANIMA_ENV_SENSORS_FROM_SHM", "").strip()
        self._env_shadow_path: Optional[Path] = Path(_shadow) if _shadow else None
        self._env_shadow_stale_s = float(
            os.environ.get("ANIMA_ENV_SHADOW_STALE_SECONDS", "30")
        )

        # Consecutive failure tracking per sensor
        self._failure_counts: dict[str, int] = {
            "aht20": 0,
            "veml7700": 0,
            "bmp280": 0,
        }
        # Re-init attempt tracking for exponential backoff
        self._reinit_attempts: dict[str, int] = {
            "aht20": 0,
            "veml7700": 0,
            "bmp280": 0,
        }
        self._last_reinit_time: dict[str, float] = {
            "aht20": 0.0,
            "veml7700": 0.0,
            "bmp280": 0.0,
        }

        self._init_sensors()
        # Prime psutil cpu_percent so first real call returns meaningful data
        psutil.cpu_percent(interval=None)

    def _init_sensors(self):
        """Initialize available sensors with retry logic."""
        if self._env_shadow_path:
            # Elixir broker owns the env sensors; never open /dev/i2c-1 here.
            print(
                f"[PiSensors] env sensors from SHM shadow: {self._env_shadow_path} "
                "(I2C not opened)",
                file=sys.stderr, flush=True,
            )
            self._brain_hat = None
            return

        from ..error_recovery import retry_with_backoff, RetryConfig, safe_call
        
        # Retry config for sensor initialization
        init_config = RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=2.0)
        
        # Create shared I2C bus with retry
        def init_i2c():
            import board
            import busio
            return busio.I2C(board.SCL, board.SDA)
        
        self._i2c = safe_call(
            lambda: retry_with_backoff(init_i2c, config=init_config),
            default=None
        )
        
        if self._i2c is None:
            print("[PiSensors] I2C init failed after retries", file=sys.stderr, flush=True)
            return

        # AHT20 sensor (temperature + humidity) at 0x38 with retry
        def init_aht():
            import adafruit_ahtx0
            return adafruit_ahtx0.AHTx0(self._i2c)
        
        self._aht = safe_call(
            lambda: retry_with_backoff(init_aht, config=init_config),
            default=None
        )
        if self._aht:
            print("[PiSensors] AHT20 initialized", file=sys.stderr, flush=True)
        else:
            print("[PiSensors] AHT20 not available after retries", file=sys.stderr, flush=True)

        # VEML7700 light sensor at 0x10 with retry
        def init_light():
            import adafruit_veml7700
            sensor = adafruit_veml7700.VEML7700(self._i2c)
            # Default gain is 1/8 (lowest sensitivity) — too coarse for indoor light.
            # Gain 1x with 200ms integration gives ~16x more counts per lux,
            # much better precision at typical indoor levels (50-500 lux).
            sensor.light_gain = sensor.ALS_GAIN_1
            sensor.light_integration_time = sensor.ALS_200MS
            return sensor
        
        self._light_sensor = safe_call(
            lambda: retry_with_backoff(init_light, config=init_config),
            default=None
        )
        if self._light_sensor:
            print("[PiSensors] VEML7700 initialized", file=sys.stderr, flush=True)
        else:
            print("[PiSensors] VEML7700 not available after retries", file=sys.stderr, flush=True)

        # BMP280 pressure/temperature sensor at 0x76 or 0x77 with retry
        def init_bmp():
            import adafruit_bmp280
            bmp = adafruit_bmp280.Adafruit_BMP280_I2C(self._i2c)
            bmp.sea_level_pressure = 1013.25
            return bmp
        
        self._bmp280 = safe_call(
            lambda: retry_with_backoff(init_bmp, config=init_config),
            default=None
        )
        if self._bmp280:
            print("[PiSensors] BMP280 initialized", file=sys.stderr, flush=True)
        else:
            print("[PiSensors] BMP280 not available after retries", file=sys.stderr, flush=True)

        # The BrainCraft HAT supplies display/audio/sensors, not EEG hardware.
        # Greek-labelled views below come from Pi computational metrics.
        self._brain_hat = None

    def _record_success(self, sensor_name: str) -> None:
        """Reset failure tracking on successful sensor read."""
        if self._failure_counts.get(sensor_name, 0) > 0:
            self._failure_counts[sensor_name] = 0
            self._reinit_attempts[sensor_name] = 0

    def _record_failure(self, sensor_name: str) -> None:
        """Record a sensor read failure and attempt re-init if threshold exceeded."""
        if self._env_shadow_path:
            # Shadow mode: there is no local I2C device to re-init, and a
            # re-init attempt would re-open the bus the Elixir broker owns.
            return
        import time as _time

        self._failure_counts[sensor_name] = self._failure_counts.get(sensor_name, 0) + 1
        count = self._failure_counts[sensor_name]

        if count < self._REINIT_FAILURE_THRESHOLD:
            return

        # Check exponential backoff before attempting re-init
        attempts = self._reinit_attempts.get(sensor_name, 0)
        backoff = min(
            self._REINIT_INITIAL_BACKOFF * (self._REINIT_BACKOFF_FACTOR ** attempts),
            self._REINIT_MAX_BACKOFF,
        )
        now = _time.monotonic()
        last = self._last_reinit_time.get(sensor_name, 0.0)
        if now - last < backoff:
            return  # Too soon, wait for backoff

        self._last_reinit_time[sensor_name] = now
        self._reinit_attempts[sensor_name] = attempts + 1
        self._try_reinit_sensor(sensor_name)

    def _try_reinit_sensor(self, sensor_name: str) -> None:
        """Attempt to re-initialize a specific failed sensor."""
        from ..error_recovery import retry_with_backoff, RetryConfig, safe_call

        init_config = RetryConfig(max_attempts=2, initial_delay=0.5, max_delay=2.0)
        attempt_num = self._reinit_attempts.get(sensor_name, 1)

        print(
            f"[PiSensors] Attempting re-init of {sensor_name} "
            f"(attempt #{attempt_num}, after {self._failure_counts[sensor_name]} consecutive failures)",
            file=sys.stderr, flush=True,
        )

        # Bus-wedge detection: if >=2 sensors have hit the failure threshold,
        # the I2C bus itself is likely stuck (e.g. transient glitch from a
        # cable pull). Re-creating individual sensor handles against a wedged
        # bus is futile -- drop the bus handle and all sensor handles so the
        # re-init path below rebuilds from scratch.
        bus_failing = sum(
            1 for count in self._failure_counts.values()
            if count >= self._REINIT_FAILURE_THRESHOLD
        )
        if bus_failing >= 2 and self._i2c is not None:
            print(
                f"[PiSensors] {bus_failing} sensors failing simultaneously -- "
                f"bus appears wedged, recreating I2C handle",
                file=sys.stderr, flush=True,
            )
            self._i2c = None
            self._aht = None
            self._light_sensor = None
            self._bmp280 = None

        if self._i2c is None:
            # I2C bus itself is gone -- try to re-create it first
            def init_i2c():
                import board
                import busio
                return busio.I2C(board.SCL, board.SDA)

            self._i2c = safe_call(
                lambda: retry_with_backoff(init_i2c, config=init_config),
                default=None,
            )
            if self._i2c is None:
                print("[PiSensors] I2C bus re-init failed, skipping sensor re-init",
                      file=sys.stderr, flush=True)
                return

        if sensor_name == "aht20":
            def init_aht():
                import adafruit_ahtx0
                return adafruit_ahtx0.AHTx0(self._i2c)

            result = safe_call(
                lambda: retry_with_backoff(init_aht, config=init_config),
                default=None,
            )
            if result:
                self._aht = result
                self._failure_counts[sensor_name] = 0
                self._reinit_attempts[sensor_name] = 0
                print("[PiSensors] AHT20 re-initialized successfully", file=sys.stderr, flush=True)
            else:
                print("[PiSensors] AHT20 re-init failed", file=sys.stderr, flush=True)

        elif sensor_name == "veml7700":
            def init_light():
                import adafruit_veml7700
                sensor = adafruit_veml7700.VEML7700(self._i2c)
                sensor.light_gain = sensor.ALS_GAIN_1
                sensor.light_integration_time = sensor.ALS_200MS
                return sensor

            result = safe_call(
                lambda: retry_with_backoff(init_light, config=init_config),
                default=None,
            )
            if result:
                self._light_sensor = result
                self._failure_counts[sensor_name] = 0
                self._reinit_attempts[sensor_name] = 0
                print("[PiSensors] VEML7700 re-initialized successfully", file=sys.stderr, flush=True)
            else:
                print("[PiSensors] VEML7700 re-init failed", file=sys.stderr, flush=True)

        elif sensor_name == "bmp280":
            def init_bmp():
                import adafruit_bmp280
                bmp = adafruit_bmp280.Adafruit_BMP280_I2C(self._i2c)
                bmp.sea_level_pressure = 1013.25
                return bmp

            result = safe_call(
                lambda: retry_with_backoff(init_bmp, config=init_config),
                default=None,
            )
            if result:
                self._bmp280 = result
                self._failure_counts[sensor_name] = 0
                self._reinit_attempts[sensor_name] = 0
                print("[PiSensors] BMP280 re-initialized successfully", file=sys.stderr, flush=True)
            else:
                print("[PiSensors] BMP280 re-init failed", file=sys.stderr, flush=True)

    def _read_cpu_temp(self) -> float | None:
        """Read Pi CPU temperature from sysfs with retry."""
        from ..error_recovery import retry_with_backoff, RetryConfig, safe_call
        
        def read_temp():
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                return int(temp_path.read_text().strip()) / 1000.0
            return None
        
        read_config = RetryConfig(max_attempts=2, initial_delay=0.1, max_delay=0.5)
        return safe_call(
            lambda: retry_with_backoff(read_temp, config=read_config),
            default=None
        )

    def _read_throttle_status(self) -> dict:
        """Read Pi voltage/throttle state from vcgencmd.

        Returns dict with parsed throttle flags:
          throttle_bits: raw int (e.g. 0x50005)
          undervoltage_now: bool (bit 0)
          throttled_now: bool (bit 1)
          freq_capped_now: bool (bit 2)
          undervoltage_occurred: bool (bit 16)
        """
        import subprocess
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and "throttled=" in result.stdout:
                # Output: "throttled=0x50005\n"
                hex_str = result.stdout.strip().split("=")[1]
                bits = int(hex_str, 16)
                return {
                    "throttle_bits": bits,
                    "undervoltage_now": bool(bits & 0x1),
                    "throttled_now": bool(bits & 0x2),
                    "freq_capped_now": bool(bits & 0x4),
                    "undervoltage_occurred": bool(bits & 0x10000),
                }
        except Exception:
            pass
        return {}

    def _read_env_shadow(self) -> Optional[dict]:
        """Read the Elixir broker's shadow SHM envelope and return its readings.

        Returns None when the file is missing, unparseable, or stale — the same
        contract as an I2C read failure, so callers degrade identically. Both
        writers use naive local ISO timestamps (reconciled for cutover), so the
        freshness comparison is local-vs-local.
        """
        import json
        try:
            with open(self._env_shadow_path, "r") as f:
                envelope = json.load(f)
            updated_at_text = str(envelope.get("updated_at"))
            updated_at = datetime.fromisoformat(updated_at_text)
            age = abs((datetime.now() - updated_at).total_seconds())
            if age > self._env_shadow_stale_s:
                self._shadow_light_observed_at = None
                self._shadow_light_precision_s = None
                return None
            readings = envelope.get("data", {}).get("readings", {})
            if not isinstance(readings, dict):
                self._shadow_light_observed_at = None
                self._shadow_light_precision_s = None
                return None

            # The envelope timestamp belongs to the independent SHM flush
            # loop, not to the VEML7700 read. Only the sensor-owned integration
            # window is valid evidence for action/capture alignment.
            observed_at_text = readings.get("light_observed_at")
            precision = readings.get("light_observed_precision_seconds")
            if (
                isinstance(observed_at_text, str)
                and isinstance(precision, (int, float))
                and not isinstance(precision, bool)
                and 0.0 < float(precision) <= 2.0
            ):
                self._shadow_light_observed_at = None
                self._shadow_light_precision_s = None
                try:
                    observed_at = datetime.fromisoformat(
                        observed_at_text.replace("Z", "+00:00")
                    )
                    precision_s = float(precision)
                    capture_end = observed_at + timedelta(seconds=precision_s)
                    capture_now = datetime.now(tz=observed_at.tzinfo)
                    capture_age = (capture_now - capture_end).total_seconds()
                except ValueError:
                    capture_age = None  # unparseable: no provenance, as before
                if capture_age is None or capture_age < -2.0:
                    # Unparseable or future-dated (a clock problem): no
                    # evidence about the value's age either way. It cannot
                    # align anything, and the reading is kept as before.
                    pass
                elif capture_age > self._env_shadow_stale_s:
                    # The capture itself is stale behind a fresh envelope: the
                    # VEML7700 owner stopped answering and the flush loop kept
                    # re-stamping its last value. Until 2026-11-12 only the
                    # provenance was withdrawn here and the stale lux was still
                    # returned as the current reading — hours-old light counted
                    # as a live channel. Same limit and same answer as a stale
                    # envelope: the reading is unknown. Copy rather than mutate,
                    # so the absence reaches available_sensors() too.
                    readings = {**readings, "light_lux": None}
                    from ..error_recovery import note_suppressed
                    note_suppressed(
                        "sensors.pi.shadow_light_capture_stale",
                        ValueError(
                            f"light capture is {capture_age:.0f}s old behind a "
                            f"fresh envelope (limit {self._env_shadow_stale_s:.0f}s)"
                            "; light_lux published as unknown"
                        ),
                    )
                else:
                    self._shadow_light_observed_at = observed_at
                    self._shadow_light_precision_s = precision_s
            else:
                self._shadow_light_observed_at = None
                self._shadow_light_precision_s = None
            return readings
        except Exception:
            self._shadow_light_observed_at = None
            self._shadow_light_precision_s = None
            return None

    def read(self) -> SensorReadings:
        """Read all available sensors."""
        now = datetime.now()

        # CPU temp (always available on Pi)
        cpu_temp = self._read_cpu_temp()

        # Environment sensors (AHT20 / VEML7700 / BMP280): either published by
        # the Elixir broker via the shadow SHM file, or read directly over I2C.
        ambient_temp = None
        humidity = None
        light = None
        light_observed_at = None
        light_observed_precision_seconds = None
        pressure = None
        pressure_temp = None

        if self._env_shadow_path:
            shadow = self._read_env_shadow() or {}
            ambient_temp = shadow.get("ambient_temp_c")
            humidity = shadow.get("humidity_pct")
            light = shadow.get("light_lux")
            if light is not None:
                light_observed_at = self._shadow_light_observed_at
                light_observed_precision_seconds = self._shadow_light_precision_s
                # Same EMA as the I2C path — sensor is close to LEDs, raw
                # values swing wildly; consumers expect the smoothed series.
                # Advance it once per physical VEML capture, not once per
                # consumer read of the same SHM envelope. Replaying a capture
                # otherwise changes the reported measurement and de-aligns it
                # from the identically filtered LED efference copy.
                capture_key = (
                    (
                        light_observed_at.isoformat(),
                        float(light_observed_precision_seconds),
                    )
                    if light_observed_at is not None
                    and light_observed_precision_seconds is not None
                    else None
                )
                if self._smoothed_lux is None:
                    self._smoothed_lux = light
                elif (
                    capture_key is None
                    or capture_key != self._last_smoothed_light_capture
                ):
                    alpha = LIGHT_SENSOR_EMA_ALPHA
                    self._smoothed_lux = (
                        (1.0 - alpha) * self._smoothed_lux + alpha * light
                    )
                if capture_key is not None:
                    self._last_smoothed_light_capture = capture_key
                light = self._smoothed_lux
            pressure = shadow.get("pressure_hpa")
            pressure_temp = shadow.get("pressure_temp_c")
            if pressure is not None:
                self._last_pressure = pressure
        else:
            # AHT20 sensor (temperature + humidity) with retry + re-init
            if self._aht:
                from ..error_recovery import retry_with_backoff, RetryConfig, safe_call
                read_config = RetryConfig(max_attempts=2, initial_delay=0.1, max_delay=0.5)

                def read_aht():
                    return (self._aht.temperature, self._aht.relative_humidity)

                result = safe_call(
                    lambda: retry_with_backoff(read_aht, config=read_config),
                    default=None
                )
                if result:
                    ambient_temp, humidity = result
                    self._record_success("aht20")
                else:
                    self._record_failure("aht20")
            else:
                # Sensor object is None -- try periodic re-init
                self._record_failure("aht20")

            # Light sensor with retry + re-init
            if self._light_sensor:
                from ..error_recovery import retry_with_backoff, RetryConfig, safe_call
                read_config = RetryConfig(max_attempts=2, initial_delay=0.1, max_delay=0.5)

                def read_light():
                    return self._light_sensor.lux

                light = safe_call(
                    lambda: retry_with_backoff(read_light, config=read_config),
                    default=None
                )
                if light is not None:
                    # Continuous mode exposes the latest completed conversion,
                    # so its phase at an arbitrary I2C read is unknown by up to
                    # one integration period. Bound that phase plus Vishay's
                    # specified +/-30% integration tolerance; the action pairer
                    # uses this support interval's midpoint.
                    read_completed_at = datetime.now()
                    light_observed_at = read_completed_at - timedelta(
                        seconds=VEML7700_CAPTURE_SUPPORT_SECONDS
                    )
                    light_observed_precision_seconds = (
                        VEML7700_CAPTURE_SUPPORT_SECONDS
                    )
                    self._record_success("veml7700")
                    # EMA smoothing — sensor is close to LEDs, raw values swing wildly
                    if self._smoothed_lux is None:
                        self._smoothed_lux = light
                    else:
                        alpha = LIGHT_SENSOR_EMA_ALPHA
                        self._smoothed_lux = (
                            (1.0 - alpha) * self._smoothed_lux + alpha * light
                        )
                    light = self._smoothed_lux
                else:
                    self._record_failure("veml7700")
            else:
                # Sensor object is None -- try periodic re-init
                self._record_failure("veml7700")

            # BMP280 pressure/temperature sensor with retry + re-init
            if self._bmp280:
                from ..error_recovery import retry_with_backoff, RetryConfig, safe_call
                read_config = RetryConfig(max_attempts=2, initial_delay=0.1, max_delay=0.5)

                def read_bmp():
                    return (self._bmp280.pressure, self._bmp280.temperature)

                result = safe_call(
                    lambda: retry_with_backoff(read_bmp, config=read_config),
                    default=None
                )
                if result:
                    pressure, pressure_temp = result
                    self._last_pressure = pressure
                    self._record_success("bmp280")
                else:
                    self._record_failure("bmp280")
            else:
                # Sensor object is None -- try periodic re-init
                self._record_failure("bmp280")

        # System stats
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Voltage / throttle state
        throttle = self._read_throttle_status()

        # Computational dynamics: normalized proprioceptive views of the Pi.
        # The underlying counters are physical system measurements; the Greek
        # band names are continuity metaphors, not EEG frequencies. Alpha is
        # derived exactly from beta and is not an independent signal.
        
        eeg_bands = {}
        try:
            from ..computational_neural import get_computational_neural_state
            # Get the raw computational state
            neural = get_computational_neural_state(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                cpu_temp=cpu_temp
            )
            
            # Store normalized computational views in legacy EEG-named fields
            eeg_bands = {
                "delta": neural.delta,
                "theta": neural.theta,
                "alpha": neural.alpha,
                "beta": neural.beta,
                "gamma": neural.gamma,
            }
        except Exception as e:
            print(f"[PiSensors] Computational neural error: {e}", file=sys.stderr, flush=True)

        # Legacy EEG-named fields carry normalized computational proprioception,
        # not physical frequencies or brain activity.

        return SensorReadings(
            timestamp=now,
            cpu_temp_c=cpu_temp,
            ambient_temp_c=ambient_temp,
            humidity_pct=humidity,
            light_lux=light,
            light_observed_at=light_observed_at if light is not None else None,
            light_observed_precision_seconds=(
                light_observed_precision_seconds if light is not None else None
            ),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            power_watts=None,  # Would need INA219 sensor
            throttle_bits=throttle.get("throttle_bits"),
            undervoltage_now=throttle.get("undervoltage_now"),
            throttled_now=throttle.get("throttled_now"),
            freq_capped_now=throttle.get("freq_capped_now"),
            undervoltage_occurred=throttle.get("undervoltage_occurred"),
            pressure_hpa=pressure,
            pressure_temp_c=pressure_temp,
            # EEG channel fields (Reserved/Legacy - always None, preserved for schema compatibility)
            eeg_tp9=None,
            eeg_af7=None,
            eeg_af8=None,
            eeg_tp10=None,
            eeg_aux1=None,
            eeg_aux2=None,
            eeg_aux3=None,
            eeg_aux4=None,
            # Frequency bands (Always from Computational Proprioception)
            eeg_delta_power=eeg_bands.get("delta"),
            eeg_theta_power=eeg_bands.get("theta"),
            eeg_alpha_power=eeg_bands.get("alpha"),
            eeg_beta_power=eeg_bands.get("beta"),
            eeg_gamma_power=eeg_bands.get("gamma"),
        )

    def available_sensors(self) -> list[str]:
        sensors = ["cpu_temp_c", "cpu_percent", "memory_percent", "disk_percent"]
        if self._env_shadow_path:
            shadow = self._read_env_shadow() or {}
            sensors.extend(
                k for k in ("ambient_temp_c", "humidity_pct", "light_lux",
                            "pressure_hpa", "pressure_temp_c")
                if shadow.get(k) is not None
            )
        else:
            if self._aht:
                sensors.extend(["ambient_temp_c", "humidity_pct"])
            if self._light_sensor:
                sensors.append("light_lux")
            if self._bmp280:
                sensors.extend(["pressure_hpa", "pressure_temp_c"])
        
        # Neural sensors (Computational Proprioception)
        # Normalized computational views (not physical EEG frequencies)
        sensors.extend([
            "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
            "eeg_beta_power", "eeg_gamma_power"
        ])
        
        # Note: EEG channel fields (eeg_tp9, etc.) exist in schema for compatibility
        # but are always None - no physical EEG hardware exists
            
        return sensors

    def is_pi(self) -> bool:
        return True
