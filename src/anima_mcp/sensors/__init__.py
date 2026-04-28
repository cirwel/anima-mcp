"""Sensor abstraction - mock on Mac, real on Pi, SHM-backed in server process."""

import os
from pathlib import Path
from .base import SensorReadings, SensorBackend
from .mock import MockSensors


def _is_raspberry_pi() -> bool:
    """Detect if running on actual Raspberry Pi hardware."""
    # Check for Pi-specific paths
    if Path("/sys/class/thermal/thermal_zone0/temp").exists():
        # Check /proc/cpuinfo for Pi
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            return "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo
        except Exception:
            pass
    return False


# Only import Pi sensors if actually on Pi
PiSensors = None
if _is_raspberry_pi():
    try:
        from .pi import PiSensors
    except ImportError:
        pass

DEFAULT_BACKEND = "pi" if PiSensors else "mock"


def get_sensors(backend: str = "auto") -> SensorBackend:
    """Get sensor backend.

    Resolution order when backend == "auto":
      1. ANIMA_SENSORS_BACKEND env var (one of "shm" / "pi" / "mock") —
         set by systemd unit files. anima.service sets "shm" so the
         server never opens /dev/i2c-1; broker (anima-broker.service)
         leaves it unset and gets the auto-Pi backend.
      2. Pi auto-detect — PiSensors on Pi hardware, else MockSensors.

    Explicit backend= argument always wins over the env var.

    Why the env-var path matters: pre-fix, the server's lazy
    `_get_sensors()` instantiated PiSensors() and opened /dev/i2c-1
    alongside the broker. Concurrent ownership left BMP280 silent at
    0x77 for ~42h on 2026-04-26 → 2026-04-28; only restarting BOTH
    services recovered it. Routing the server through SHM eliminates
    the contention class.
    """
    if backend == "auto":
        env_choice = os.environ.get("ANIMA_SENSORS_BACKEND", "").strip().lower()
        if env_choice in ("shm", "pi", "mock"):
            backend = env_choice
        else:
            backend = DEFAULT_BACKEND

    if backend == "shm":
        from .shm import SHMSensors
        return SHMSensors()
    if backend == "pi" and PiSensors:
        return PiSensors()
    return MockSensors()


__all__ = ["SensorReadings", "SensorBackend", "MockSensors", "get_sensors"]
