"""SHM-backed sensor backend — reads broker-published readings from
/dev/shm/anima_state.json instead of touching I2C.

Architectural role: per CLAUDE.md the broker (anima-broker.service) owns
/dev/i2c-1 and writes readings to shared memory; the server
(anima.service) reads that shared memory. This backend exists so the
server's `_get_sensors()` accessor returns something that satisfies the
SensorBackend contract without opening the I2C bus.

Why this matters: when the server lazily instantiated PiSensors() (via
accessors._get_sensors()) it opened /dev/i2c-1 in the server process.
Both broker and server then held the same I2C device fd, and concurrent
operation could leave specific chips in unresponsive states. Observed
2026-04-26 → 2026-04-28: BMP280 went silent at 0x77 for ~42 hours.
Restarting only anima-broker did not recover it (fresh busio.I2C handle
on the broker side, but the server still held its stale fd). Restarting
both services did. Routing the server through SHM eliminates the
contention class.
"""

from typing import Optional

from .base import SensorBackend, SensorReadings


# Keys in the SHM "readings" dict that correspond to real sensor channels.
# Used by available_sensors() to report what the broker has actually
# published (matches PiSensors.available_sensors() semantics: only list
# channels that are currently producing non-None values).
_SENSOR_KEYS = (
    "cpu_temp_c",
    "ambient_temp_c",
    "humidity_pct",
    "light_lux",
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "power_watts",
    "pressure_hpa",
    "pressure_temp_c",
    "eeg_delta_power",
    "eeg_theta_power",
    "eeg_alpha_power",
    "eeg_beta_power",
    "eeg_gamma_power",
)


class SHMSensors(SensorBackend):
    """Sensor backend that reads from the broker's shared-memory state.

    Returns None from read() / empty list from available_sensors() when
    SHM is empty or unparseable — callers must handle this just as they
    do for PiSensors.read() returning None on transient sensor failure.
    """

    def __init__(self) -> None:
        # Lazy import: shared_memory pulls in atomic-write helpers we don't
        # want to load at module-import time on the broker side.
        from ..shared_memory import SharedMemoryClient
        self._shm = SharedMemoryClient(mode="read", backend="file")

    def read(self) -> Optional[SensorReadings]:
        # Lazy import to avoid an import cycle: server_state imports from
        # the sensors package, so we can't import readings_from_dict at
        # module load time.
        from ..server_state import readings_from_dict
        data = self._shm.read()
        if not data or "readings" not in data:
            return None
        try:
            return readings_from_dict(data["readings"])
        except Exception:
            return None

    def available_sensors(self) -> list[str]:
        data = self._shm.read()
        if not data or "readings" not in data:
            return []
        readings = data["readings"]
        return [k for k in _SENSOR_KEYS if readings.get(k) is not None]

    def is_pi(self) -> bool:
        # Proxying Pi-side data — return True so downstream code that gates
        # on is_pi() (e.g. real-vs-mock heuristics) treats SHM data as
        # authoritative substrate readings.
        return True
