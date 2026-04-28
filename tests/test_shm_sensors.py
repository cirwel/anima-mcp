"""Tests for SHMSensors — the SHM-backed sensor backend that the server
process uses instead of PiSensors. Pre-fix, the server lazily
instantiated PiSensors and opened /dev/i2c-1 alongside the broker;
concurrent ownership left BMP280 silent at 0x77 for ~42h on 2026-04-26.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from anima_mcp.sensors import get_sensors
from anima_mcp.sensors.base import SensorBackend
from anima_mcp.sensors.mock import MockSensors
from anima_mcp.sensors.shm import SHMSensors


# ---------------------------------------------------------------------------
# get_sensors() factory dispatch — env var honored, never falls through to
# PiSensors when ANIMA_SENSORS_BACKEND=shm
# ---------------------------------------------------------------------------


class TestGetSensorsEnvDispatch:
    def test_shm_env_returns_shm_backend_no_i2c(self, monkeypatch):
        """ANIMA_SENSORS_BACKEND=shm must yield SHMSensors regardless of
        whether the host is a Pi. The server process MUST never open
        /dev/i2c-1 — this is the bug class the env var prevents."""
        monkeypatch.setenv("ANIMA_SENSORS_BACKEND", "shm")
        backend = get_sensors()
        assert isinstance(backend, SHMSensors)

    def test_mock_env_returns_mock_backend(self, monkeypatch):
        monkeypatch.setenv("ANIMA_SENSORS_BACKEND", "mock")
        backend = get_sensors()
        assert isinstance(backend, MockSensors)

    def test_unset_env_falls_through_to_default(self, monkeypatch):
        """No env var → default (Pi auto-detect; mock off-Pi). On the
        test runner this means MockSensors."""
        monkeypatch.delenv("ANIMA_SENSORS_BACKEND", raising=False)
        backend = get_sensors()
        # On macOS / non-Pi CI: should be Mock. On a Pi: would be PiSensors.
        # Either way, it MUST NOT be SHMSensors when the env var is unset.
        assert not isinstance(backend, SHMSensors)

    def test_explicit_arg_overrides_env(self, monkeypatch):
        """get_sensors(backend='mock') must win over ANIMA_SENSORS_BACKEND=shm —
        keeps tests deterministic."""
        monkeypatch.setenv("ANIMA_SENSORS_BACKEND", "shm")
        backend = get_sensors(backend="mock")
        assert isinstance(backend, MockSensors)

    def test_unknown_env_value_falls_through(self, monkeypatch):
        """Garbage env value should not pick SHMSensors."""
        monkeypatch.setenv("ANIMA_SENSORS_BACKEND", "garbage-not-a-real-backend")
        backend = get_sensors()
        assert not isinstance(backend, SHMSensors)

    def test_env_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ANIMA_SENSORS_BACKEND", "SHM")
        backend = get_sensors()
        assert isinstance(backend, SHMSensors)


# ---------------------------------------------------------------------------
# SHMSensors.read() / available_sensors() — wraps SHM reads as SensorReadings
# ---------------------------------------------------------------------------


def _make_shm_data(readings_dict):
    """Build the SHM payload shape that SharedMemoryClient.read() returns
    (the inner 'data' dict; the envelope is unwrapped by the client)."""
    return {
        "readings": readings_dict,
        "anima": {"warmth": 0.5, "clarity": 0.5, "stability": 0.5, "presence": 0.5},
    }


class TestSHMSensorsRead:
    def test_read_returns_readings_with_pressure(self):
        """The Lumen-shaped happy path: pressure_hpa is in SHM, read()
        returns it on the SensorReadings object."""
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = _make_shm_data({
            "timestamp": "2026-04-28T02:08:02.262686",
            "cpu_temp_c": 65.2,
            "ambient_temp_c": 29.3,
            "humidity_pct": 23.2,
            "light_lux": 502.7,
            "cpu_percent": 8.2,
            "memory_percent": 19.4,
            "disk_percent": 45.4,
            "pressure_hpa": 820.2,
            "pressure_temp_c": 26.0,
            "eeg_delta_power": 0.95,
        })
        readings = sensors.read()
        assert readings is not None
        assert readings.pressure_hpa == 820.2
        assert readings.cpu_temp_c == 65.2
        assert readings.eeg_delta_power == 0.95

    def test_read_handles_empty_shm(self):
        """When SHM is empty (broker not running yet), read() returns None
        — callers handle this same as PiSensors.read() failing."""
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = {}
        assert sensors.read() is None

    def test_read_handles_missing_readings_key(self):
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = {"anima": {}}  # has anima, no readings
        assert sensors.read() is None

    def test_read_handles_none_shm(self):
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = None
        assert sensors.read() is None


class TestSHMSensorsAvailable:
    def test_available_reflects_present_keys(self):
        """available_sensors() lists only the SHM keys that are non-None.
        Mirrors PiSensors.available_sensors() semantics: the report
        tracks which channels are actually producing values right now."""
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = _make_shm_data({
            "cpu_temp_c": 65.0,
            "humidity_pct": 23.0,
            "pressure_hpa": 820.0,
            "light_lux": None,        # sensor temporarily failed
            "ambient_temp_c": None,
        })
        avail = sensors.available_sensors()
        assert "cpu_temp_c" in avail
        assert "humidity_pct" in avail
        assert "pressure_hpa" in avail
        assert "light_lux" not in avail
        assert "ambient_temp_c" not in avail

    def test_available_excludes_pressure_when_bmp280_dead(self):
        """The exact failure scenario this fix exists for: BMP280 dead,
        pressure_hpa is None in SHM. available_sensors() must omit it
        (matches the public Discord alert behavior)."""
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = _make_shm_data({
            "cpu_temp_c": 65.0,
            "ambient_temp_c": 29.0,
            "humidity_pct": 23.0,
            "light_lux": 500.0,
            "pressure_hpa": None,
            "pressure_temp_c": None,
        })
        avail = sensors.available_sensors()
        assert "pressure_hpa" not in avail
        assert "ambient_temp_c" in avail  # other sensors still reported

    def test_available_empty_when_shm_empty(self):
        sensors = SHMSensors()
        sensors._shm = MagicMock()
        sensors._shm.read.return_value = {}
        assert sensors.available_sensors() == []


class TestSHMSensorsIsPi:
    def test_is_pi_true_proxying_pi_data(self):
        """SHMSensors proxies Pi-side data, so callers gating on is_pi()
        (real-vs-mock checks) treat SHM data as authoritative substrate."""
        assert SHMSensors().is_pi() is True


# ---------------------------------------------------------------------------
# Anti-regression: SHMSensors must not import or instantiate any I2C code
# ---------------------------------------------------------------------------


class TestSHMSensorsNeverTouchesI2C:
    def test_construction_does_not_import_busio(self):
        """SHMSensors.__init__ must NOT trigger an `import busio` that
        opens /dev/i2c-1 as a side effect on Pi hardware. We check this
        by verifying the module dependency graph at import time."""
        # If busio were imported transitively, sys.modules would have it.
        # We can't fully prove the negative here (busio could be loaded
        # elsewhere), but we can check the SHM module's direct imports.
        import anima_mcp.sensors.shm as shm_mod
        source = open(shm_mod.__file__).read()
        assert "import busio" not in source, (
            "shm.py must not import busio — that would open /dev/i2c-1 "
            "on Pi at import time, exactly the bug this module is "
            "designed to avoid"
        )
        assert "import board" not in source, (
            "shm.py must not import board — same I2C contamination concern"
        )
