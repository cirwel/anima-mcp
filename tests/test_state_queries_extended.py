from unittest.mock import MagicMock, patch

import pytest

from conftest import make_anima, make_readings, parse_result


@pytest.mark.asyncio
async def test_get_state_happy_path_cleans_sensors_records_state_and_injects_inner_life():
    from anima_mcp.handlers.state_queries import handle_get_state

    readings = make_readings(
        ambient_temp_c=22.5,
        humidity_pct=None,  # should be suppressed
        light_lux=123.0,
        pressure_hpa=1012.3,
        cpu_temp_c=55.0,
        cpu_percent=None,  # should be suppressed
        eeg_delta_power=0.9,  # should appear in neural bands
    )
    anima = make_anima(warmth=0.3333333, clarity=0.6666666, stability=0.123456, presence=0.9999)

    identity = MagicMock()
    identity.name = "Lumen"
    identity.creature_id = "abc12345-xxxx-yyyy-zzzz"
    identity.total_awakenings = 7
    identity.total_alive_seconds = 100.0
    identity.age_seconds.return_value = 200.0
    identity.alive_ratio.return_value = 0.5

    store = MagicMock()
    store.get_identity.return_value = identity
    store.get_session_alive_seconds.return_value = 25.0

    sensors_backend = MagicMock()
    sensors_backend.is_pi.return_value = False

    with patch("anima_mcp.accessors._get_store", return_value=store), \
         patch("anima_mcp.accessors._get_sensors", return_value=sensors_backend), \
         patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, anima)), \
         patch("anima_mcp.accessors._get_last_shm_data", return_value={"inner_life": {"temperament": "calm", "drives": {}, "strongest_drive": "curiosity"}}):
        result = await handle_get_state({})

    data = parse_result(result)
    assert data["identity"]["name"] == "Lumen"
    assert data["identity"]["id"] == "abc12345..."
    assert data["is_pi"] is False

    # Rounding behavior
    assert data["anima"]["warmth"] == 0.333
    assert data["anima"]["clarity"] == 0.667
    assert data["anima"]["stability"] == 0.123
    assert data["anima"]["presence"] == 1.0

    # Sensor cleaning (suppresses nulls, groups)
    env = data["sensors"]["environment"]
    assert "ambient_temp_c" in env
    assert "humidity_pct" not in env
    assert env["light_lux"] == 123.0

    sys = data["sensors"]["system"]
    assert "cpu_temp_c" in sys
    assert "cpu_percent" not in sys

    neural = data["sensors"]["neural"]
    assert "delta" in neural

    # Inner life injected when present in SHM
    assert data["inner_life"]["temperament"] == "calm"
    assert data["inner_life"]["strongest_drive"] == "curiosity"

    # A read does not record; the main loop owns state_history.
    assert not store.record_state.called


@pytest.mark.asyncio
async def test_read_sensors_sets_source_shared_memory_vs_direct():
    from anima_mcp.handlers.state_queries import handle_read_sensors

    readings = make_readings(ambient_temp_c=22.0, humidity_pct=None)
    sensors_backend = MagicMock()
    sensors_backend.available_sensors.return_value = ["mock"]
    sensors_backend.is_pi.return_value = False

    # Case 1: shared memory present
    shm_client = MagicMock()
    shm_client.read.return_value = {"readings": {"timestamp": "x"}, "anima": {"warmth": 0.5}}

    with patch("anima_mcp.accessors._get_sensors", return_value=sensors_backend), \
         patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, None)), \
         patch("anima_mcp.accessors._get_shm_client", return_value=shm_client):
        data = parse_result(await handle_read_sensors({}))
        assert data["source"] == "shared_memory"
        assert "humidity_pct" not in data["readings"]

    # Case 2: no shared memory client
    with patch("anima_mcp.accessors._get_sensors", return_value=sensors_backend), \
         patch("anima_mcp.accessors._get_readings_and_anima", return_value=(readings, None)), \
         patch("anima_mcp.accessors._get_shm_client", return_value=None):
        data = parse_result(await handle_read_sensors({}))
        assert data["source"] == "direct_sensors"

