"""PiSensors shadow-env mode (Phase-1 Elixir broker cutover).

With ANIMA_ENV_SENSORS_FROM_SHM set, PiSensors never opens I2C — the Elixir
broker owns the AHT20/VEML7700/BMP280 and publishes their channels to a shadow
SHM file. That also makes PiSensors constructible off-Pi, so these tests run
on any host.
"""
import json
from datetime import datetime, timedelta

import pytest

from anima_mcp.sensors.pi import PiSensors


ENV = {
    "ambient_temp_c": 24.5,
    "humidity_pct": 23.1,
    "light_lux": 60.0,
    "pressure_hpa": 819.9,
    "pressure_temp_c": 29.8,
}


@pytest.fixture
def shadow_file(tmp_path, monkeypatch):
    """Returns a writer(readings, age_seconds=0) that (re)writes the shadow
    envelope in the Elixir broker's format and points the env var at it."""
    path = tmp_path / "anima_state.shadow.json"
    monkeypatch.setenv("ANIMA_ENV_SENSORS_FROM_SHM", str(path))

    def write(readings, *, age_seconds=0.0):
        envelope = {
            "updated_at": (
                datetime.now() - timedelta(seconds=age_seconds)
            ).isoformat(),
            "pid": "test",
            "data": {"readings": readings},
        }
        path.write_text(json.dumps(envelope))
        return path

    return write


def test_shadow_mode_never_opens_i2c(shadow_file):
    shadow_file(ENV)
    s = PiSensors()
    assert s._i2c is None
    assert s._aht is None
    assert s._light_sensor is None
    assert s._bmp280 is None


def test_shadow_mode_reads_env_channels(shadow_file):
    shadow_file(ENV)
    s = PiSensors()
    r = s.read()
    assert r.ambient_temp_c == 24.5
    assert r.humidity_pct == 23.1
    assert r.light_lux == 60.0  # first read seeds the EMA
    assert r.pressure_hpa == 819.9
    assert r.pressure_temp_c == 29.8


def test_shadow_lux_ema_matches_i2c_path(shadow_file):
    shadow_file(ENV)
    s = PiSensors()
    s.read()
    shadow_file({**ENV, "light_lux": 100.0})
    r = s.read()
    assert r.light_lux == pytest.approx(0.8 * 60.0 + 0.2 * 100.0)


def test_stale_shadow_degrades_to_none(shadow_file):
    shadow_file(ENV, age_seconds=120.0)
    s = PiSensors()
    r = s.read()
    assert r.ambient_temp_c is None
    assert r.humidity_pct is None
    assert r.light_lux is None
    assert r.pressure_hpa is None
    assert r.pressure_temp_c is None


def test_missing_shadow_file_degrades_to_none(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ANIMA_ENV_SENSORS_FROM_SHM", str(tmp_path / "does_not_exist.json")
    )
    s = PiSensors()
    r = s.read()
    assert r.ambient_temp_c is None
    assert r.humidity_pct is None


def test_shadow_mode_never_attempts_reinit(shadow_file):
    """_record_failure must be a no-op: a re-init would re-open the I2C bus
    the Elixir broker now owns (the contention class Phase 1 eliminates)."""
    shadow_file(ENV)
    s = PiSensors()
    s._record_failure("aht20")
    assert s._failure_counts["aht20"] == 0
    assert s._reinit_attempts["aht20"] == 0


def test_available_sensors_reports_shadow_channels(shadow_file):
    shadow_file(ENV)
    s = PiSensors()
    avail = s.available_sensors()
    for key in ENV:
        assert key in avail


def test_available_sensors_omits_stale_shadow_channels(shadow_file):
    shadow_file(ENV, age_seconds=120.0)
    s = PiSensors()
    avail = s.available_sensors()
    for key in ENV:
        assert key not in avail
