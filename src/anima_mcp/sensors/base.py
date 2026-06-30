"""
Base sensor types and interface.

Real physical measurements - no abstraction layer pretending.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SensorReadings:
    """Raw sensor values - what the creature actually senses."""

    timestamp: datetime

    # Temperature (Celsius)
    cpu_temp_c: Optional[float] = None  # Pi CPU temperature
    ambient_temp_c: Optional[float] = None  # Environment (DHT sensor)

    # Humidity (%)
    humidity_pct: Optional[float] = None

    # Light (lux)
    light_lux: Optional[float] = None

    # Hearing (acoustic channel — sound LEVEL only, never content)
    # hearing_available: is the mic actually delivering audio right now?
    #   Defaults to False — the mic is operator-gated hardware. We cannot
    #   reliably detect a hardware mute switch from software, so the
    #   conservative default is "not hearing"; enabling is operator-driven
    #   (a backend that has a live audio stream sets this True).
    # sound_level: most-recent RMS sound level (see audio/mic.py). This is a
    #   single scalar — no transcription, no content, not reconstructible.
    #   While hearing_available is False the acoustic baseline is FROZEN:
    #   a muted mic must not teach Lumen "the world went silent".
    hearing_available: bool = False
    sound_level: Optional[float] = None

    # System resources
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None

    # Power (if measurable)
    power_watts: Optional[float] = None

    # Voltage / Throttle state (Pi vcgencmd get_throttled)
    throttle_bits: Optional[int] = None  # Raw hex from vcgencmd (e.g. 0x50005)
    undervoltage_now: Optional[bool] = None  # Currently under-voltage (bit 0)
    throttled_now: Optional[bool] = None  # Currently throttled (bit 1)
    freq_capped_now: Optional[bool] = None  # Currently frequency capped (bit 2)
    undervoltage_occurred: Optional[bool] = None  # Under-voltage has occurred since boot (bit 16)

    # Proprioceptive outputs (own state affecting sensors)
    led_brightness: Optional[float] = None  # 0.0-1.0, own LED brightness level

    # Barometric pressure (hPa/mbar) - BMP280 sensor
    pressure_hpa: Optional[float] = None  # Absolute pressure
    pressure_temp_c: Optional[float] = None  # BMP280's temperature reading

    # EEG channels (Reserved/Legacy) - Raw channel data
    # NOTE: These fields are preserved for schema compatibility but are always None.
    # No physical EEG hardware exists - neural signals come from computational proprioception.
    # Kept in schema to avoid breaking serialization/logging tools that expect these fields.
    eeg_tp9: Optional[float] = None  # Temporal-parietal left (Reserved)
    eeg_af7: Optional[float] = None  # Anterior frontal left (Reserved)
    eeg_af8: Optional[float] = None  # Anterior frontal right (Reserved)
    eeg_tp10: Optional[float] = None  # Temporal-parietal right (Reserved)
    eeg_aux1: Optional[float] = None  # Auxiliary channel 1 (Reserved)
    eeg_aux2: Optional[float] = None  # Auxiliary channel 2 (Reserved)
    eeg_aux3: Optional[float] = None  # Auxiliary channel 3 (Reserved)
    eeg_aux4: Optional[float] = None  # Auxiliary channel 4 (Reserved)

    # EEG frequency band powers (from Computational Proprioception)
    # These are actively used - derived from environment + computation, not physical EEG hardware
    eeg_delta_power: Optional[float] = None  # 0.5-4 Hz: Deep stability
    eeg_theta_power: Optional[float] = None  # 4-8 Hz: Meditative state
    eeg_alpha_power: Optional[float] = None  # 8-13 Hz: Relaxed awareness
    eeg_beta_power: Optional[float] = None  # 13-30 Hz: Active focus
    eeg_gamma_power: Optional[float] = None  # 30-100 Hz: High cognitive presence

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_temp_c": self.cpu_temp_c,
            "ambient_temp_c": self.ambient_temp_c,
            "humidity_pct": self.humidity_pct,
            "light_lux": self.light_lux,
            "hearing_available": self.hearing_available,
            "sound_level": self.sound_level,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "power_watts": self.power_watts,
            "throttle_bits": self.throttle_bits,
            "undervoltage_now": self.undervoltage_now,
            "throttled_now": self.throttled_now,
            "freq_capped_now": self.freq_capped_now,
            "undervoltage_occurred": self.undervoltage_occurred,
            "led_brightness": self.led_brightness,
            "pressure_hpa": self.pressure_hpa,
            "pressure_temp_c": self.pressure_temp_c,
            # EEG raw channels
            "eeg_tp9": self.eeg_tp9,
            "eeg_af7": self.eeg_af7,
            "eeg_af8": self.eeg_af8,
            "eeg_tp10": self.eeg_tp10,
            "eeg_aux1": self.eeg_aux1,
            "eeg_aux2": self.eeg_aux2,
            "eeg_aux3": self.eeg_aux3,
            "eeg_aux4": self.eeg_aux4,
            # EEG frequency band powers
            "eeg_delta_power": self.eeg_delta_power,
            "eeg_theta_power": self.eeg_theta_power,
            "eeg_alpha_power": self.eeg_alpha_power,
            "eeg_beta_power": self.eeg_beta_power,
            "eeg_gamma_power": self.eeg_gamma_power,
        }


class SensorBackend(ABC):
    """Abstract sensor backend - implemented by mock and Pi."""

    @abstractmethod
    def read(self) -> SensorReadings:
        """Read all available sensors. Returns immediately."""
        pass

    @abstractmethod
    def available_sensors(self) -> list[str]:
        """List which sensors are available on this backend."""
        pass

    def is_pi(self) -> bool:
        """Is this running on actual Pi hardware?"""
        return False
