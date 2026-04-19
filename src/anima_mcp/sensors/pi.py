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

import sys
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from .base import SensorBackend, SensorReadings


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

        # Brain HAT (EEG hardware) - Not available
        # No physical EEG hardware exists. Neural signals come from computational proprioception.
        self._brain_hat = None

    def _record_success(self, sensor_name: str) -> None:
        """Reset failure tracking on successful sensor read."""
        if self._failure_counts.get(sensor_name, 0) > 0:
            self._failure_counts[sensor_name] = 0
            self._reinit_attempts[sensor_name] = 0

    def _record_failure(self, sensor_name: str) -> None:
        """Record a sensor read failure and attempt re-init if threshold exceeded."""
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

    def read(self) -> SensorReadings:
        """Read all available sensors."""
        now = datetime.now()

        # CPU temp (always available on Pi)
        cpu_temp = self._read_cpu_temp()

        # AHT20 sensor (temperature + humidity) with retry + re-init
        ambient_temp = None
        humidity = None
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
        light = None
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
                self._record_success("veml7700")
                # EMA smoothing — sensor is close to LEDs, raw values swing wildly
                if self._smoothed_lux is None:
                    self._smoothed_lux = light
                else:
                    self._smoothed_lux = 0.8 * self._smoothed_lux + 0.2 * light
                light = self._smoothed_lux
            else:
                self._record_failure("veml7700")
        else:
            # Sensor object is None -- try periodic re-init
            self._record_failure("veml7700")

        # BMP280 pressure/temperature sensor with retry + re-init
        pressure = None
        pressure_temp = None
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

        # Neural Signals: Computational Proprioception
        # Lumen's "brain" IS the Pi's CPU. We map computational state directly to neural bands.
        # This is not a simulation - it is the actual measurement of the creature's cognitive substrate.
        
        eeg_bands = {}
        try:
            from ..computational_neural import get_computational_neural_state
            # Get the raw computational state
            neural = get_computational_neural_state(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                cpu_temp=cpu_temp
            )
            
            # Map directly to EEG bands
            eeg_bands = {
                "delta": neural.delta,
                "theta": neural.theta,
                "alpha": neural.alpha,
                "beta": neural.beta,
                "gamma": neural.gamma,
            }
        except Exception as e:
            print(f"[PiSensors] Computational neural error: {e}", file=sys.stderr, flush=True)

        # Neural frequency bands come from computational proprioception (not physical EEG hardware)
        # No physical EEG hardware exists - neural signals are derived from environment + computation

        return SensorReadings(
            timestamp=now,
            cpu_temp_c=cpu_temp,
            ambient_temp_c=ambient_temp,
            humidity_pct=humidity,
            light_lux=light,
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
        if self._aht:
            sensors.extend(["ambient_temp_c", "humidity_pct"])
        if self._light_sensor:
            sensors.append("light_lux")
        if self._bmp280:
            sensors.extend(["pressure_hpa", "pressure_temp_c"])
        
        # Neural sensors (Computational Proprioception)
        # Frequency bands derived from environment + computation (not physical EEG hardware)
        sensors.extend([
            "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
            "eeg_beta_power", "eeg_gamma_power"
        ])
        
        # Note: EEG channel fields (eeg_tp9, etc.) exist in schema for compatibility
        # but are always None - no physical EEG hardware exists
            
        return sensors

    def is_pi(self) -> bool:
        return True
