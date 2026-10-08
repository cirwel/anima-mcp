"""state_history has one writer, on Lumen's own schedule.

Until 2026-10 the table was written only as a side effect of the `get_state`
and `get_lumen_context` read handlers, so the record existed only while a
client polled; when the last poller stopped (2026-08-31) the history stopped
with it. These tests pin the replacement: the payload the loop stores, absent
values staying absent, and the loop block itself.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from anima_mcp import server, server_state
from anima_mcp.state_history import (
    history_sensors,
    maybe_record_state_history,
    record_state_history,
)


class FakeReadings:
    def __init__(self, **values):
        self.values = values

    def to_dict(self):
        return dict(self.values)


ANIMA = SimpleNamespace(warmth=0.3, clarity=0.4, stability=0.5, presence=0.6)


def test_records_anima_and_the_enriched_sensor_payload():
    store = SimpleNamespace(record_state=MagicMock())
    attribution = {"status": "ready", "external_lux_residual": 12.0}
    with patch("anima_mcp.accessors._get_growth",
               return_value=SimpleNamespace(interaction_level=lambda: 0.5)), \
         patch("anima_mcp.state_history.gated_external_light_lux", return_value=12.0):
        record_state_history(store, ANIMA, FakeReadings(light_lux=40.0, led_brightness=0.1),
                             attribution)
    args = store.record_state.call_args[0]
    assert args[:4] == (0.3, 0.4, 0.5, 0.6)
    assert args[4] == {
        "light_lux": 40.0,
        "led_brightness": 0.1,
        "external_light_lux": 12.0,
        "light_attribution_status": "ready",
        "interaction_level": 0.5,
    }


def test_unknown_values_stay_absent_not_defaulted():
    """No person on record means interaction_level is absent, not 0.0; no
    attribution means the residual is None and the status says unavailable."""
    with patch("anima_mcp.accessors._get_growth",
               return_value=SimpleNamespace(interaction_level=lambda: None)), \
         patch("anima_mcp.accessors._get_led_brightness", return_value=None):
        sensors = history_sensors(FakeReadings(light_lux=40.0))
    assert "interaction_level" not in sensors
    assert sensors["external_light_lux"] is None
    assert sensors["light_attribution_status"] == "unavailable"


def test_a_failed_enrichment_is_counted_and_the_row_still_written():
    store = SimpleNamespace(record_state=MagicMock())
    with patch("anima_mcp.accessors._get_growth", side_effect=RuntimeError("down")), \
         patch("anima_mcp.state_history.note_suppressed") as noted:
        record_state_history(store, ANIMA, FakeReadings(light_lux=1.0, led_brightness=0.2))
    assert store.record_state.called
    assert noted.call_args[0][0] == "state_history.interaction_level"


def test_the_broker_capture_aligned_led_brightness_wins():
    """The server loop rewrites readings.led_brightness with its own applied
    value before recording; the broker snapshot's capture-aligned value is the
    one lux can be decomposed against."""
    with patch("anima_mcp.accessors._get_growth", return_value=None):
        sensors = history_sensors(FakeReadings(light_lux=40.0, led_brightness=0.0),
                                  None, {"led_brightness": 0.37})
    assert sensors["led_brightness"] == 0.37


def test_an_absent_broker_value_falls_back_to_proprioception_not_zero():
    with patch("anima_mcp.accessors._get_growth", return_value=None), \
         patch("anima_mcp.accessors._get_led_brightness", return_value=0.12):
        sensors = history_sensors(FakeReadings(light_lux=40.0, led_brightness=0.0),
                                  None, {"led_brightness": None})
    assert sensors["led_brightness"] == 0.12


def _ctx(last=0.0):
    return SimpleNamespace(store=SimpleNamespace(record_state=MagicMock()),
                           last_state_history_at=last)


def test_the_gate_writes_when_due_and_not_before():
    ctx = _ctx(last=1000.0)
    with patch("anima_mcp.accessors._get_growth", return_value=None):
        assert not maybe_record_state_history(ctx, ANIMA, FakeReadings(), None, 240.0, now=1239.0)
        assert not ctx.store.record_state.called
        assert maybe_record_state_history(ctx, ANIMA, FakeReadings(), None, 240.0, now=1240.0)
    assert ctx.store.record_state.call_count == 1
    assert ctx.last_state_history_at == 1240.0


def test_the_gate_passes_shm_attribution_and_readings_through():
    ctx = _ctx()
    shm = {"light_attribution": {"status": "warming"}, "readings": {"led_brightness": 0.5}}
    with patch("anima_mcp.accessors._get_growth", return_value=None):
        maybe_record_state_history(ctx, ANIMA, FakeReadings(), shm, 240.0, now=500.0)
    sensors = ctx.store.record_state.call_args[0][4]
    assert sensors["light_attribution_status"] == "warming"
    assert sensors["led_brightness"] == 0.5


def test_the_gate_needs_a_store_readings_and_anima():
    for ctx, anima, readings in (
        (SimpleNamespace(store=None, last_state_history_at=0.0), ANIMA, FakeReadings()),
        (_ctx(), None, FakeReadings()),
        (_ctx(), ANIMA, None),
        (None, ANIMA, FakeReadings()),
    ):
        assert not maybe_record_state_history(ctx, anima, readings, None, 240.0, now=10_000.0)


def test_a_failing_write_is_counted_and_waits_a_full_interval():
    ctx = _ctx()
    ctx.store.record_state.side_effect = RuntimeError("disk")
    with patch("anima_mcp.accessors._get_growth", return_value=None), \
         patch("anima_mcp.state_history.note_suppressed") as noted:
        assert not maybe_record_state_history(ctx, ANIMA, FakeReadings(), None, 240.0, now=900.0)
        assert not maybe_record_state_history(ctx, ANIMA, FakeReadings(), None, 240.0, now=901.0)
    assert ctx.store.record_state.call_count == 1  # the second tick did not retry
    assert noted.call_args[0][0] == "server.state_history"


def test_cadence_leaves_room_for_one_missed_write_in_alive_time():
    """identity/store.py counts gaps up to 600 s as lived time; one missed
    write at this cadence must still fall inside that."""
    assert 2 * server_state.STATE_HISTORY_RECORD_SECONDS < 600.0


def test_the_main_loop_calls_the_gate():
    source = inspect.getsource(server)
    block = source[source.index("# Durable anima + sensor history"):]
    block = block[: block.index("# System metrics pruning")]
    assert "maybe_record_state_history(" in block
    assert "STATE_HISTORY_RECORD_SECONDS" in block
