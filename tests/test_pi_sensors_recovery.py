"""Tests for PiSensors bus-wedge recovery.

PiSensors.py is Pi-only (imports board/busio). Stub those in sys.modules so
we can exercise the recovery logic on a dev machine.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stub_board_and_busio(monkeypatch):
    """Install minimal board/busio/adafruit_* stubs so pi.py can be imported."""
    stubs = {}
    for name in (
        "board", "busio",
        "adafruit_ahtx0", "adafruit_veml7700", "adafruit_bmp280",
    ):
        mod = types.ModuleType(name)
        stubs[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)

    stubs["board"].SCL = object()
    stubs["board"].SDA = object()
    stubs["busio"].I2C = lambda scl, sda: MagicMock(name=f"I2C_bus_{id(scl)}")

    # Sensor class stubs return working mocks on call
    stubs["adafruit_ahtx0"].AHTx0 = lambda i2c: MagicMock(name="aht")

    class _VemlStub:
        ALS_GAIN_1 = 0
        ALS_200MS = 1
        def __init__(self, i2c):
            self.light_gain = self.ALS_GAIN_1
            self.light_integration_time = self.ALS_200MS
    stubs["adafruit_veml7700"].VEML7700 = _VemlStub

    class _BmpStub:
        def __init__(self, i2c):
            self.sea_level_pressure = 1013.25
    stubs["adafruit_bmp280"].Adafruit_BMP280_I2C = _BmpStub

    yield stubs


def test_wedged_bus_is_recreated_when_multiple_sensors_fail(stub_board_and_busio):
    """When >=2 sensors have hit the failure threshold, _try_reinit_sensor
    should treat the bus itself as wedged and drop self._i2c so it gets
    re-created on the next init path."""
    from anima_mcp.sensors.pi import PiSensors

    sensors = PiSensors()
    original_bus = sensors._i2c
    assert original_bus is not None, "precondition: init succeeded"

    # Simulate bus wedge: two sensors have hit the failure threshold.
    sensors._failure_counts["aht20"] = PiSensors._REINIT_FAILURE_THRESHOLD
    sensors._failure_counts["veml7700"] = PiSensors._REINIT_FAILURE_THRESHOLD

    sensors._try_reinit_sensor("aht20")

    # Bus should have been re-created (a different object now) since the
    # wedge-detector dropped the stale handle.
    assert sensors._i2c is not None, "bus should have been re-created"
    assert sensors._i2c is not original_bus, "bus should be a fresh handle after wedge recovery"


def test_single_sensor_failure_does_not_reset_bus(stub_board_and_busio):
    """A single failing sensor is not enough evidence the bus is wedged.
    Keep the existing bus handle and just re-init that one sensor."""
    from anima_mcp.sensors.pi import PiSensors

    sensors = PiSensors()
    original_bus = sensors._i2c
    assert original_bus is not None

    # Only one sensor is failing.
    sensors._failure_counts["bmp280"] = PiSensors._REINIT_FAILURE_THRESHOLD

    sensors._try_reinit_sensor("bmp280")

    # Bus handle should be preserved -- no wedge evidence.
    assert sensors._i2c is original_bus, "bus should NOT be recreated when only one sensor is failing"
