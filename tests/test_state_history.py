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
from anima_mcp.state_history import history_sensors, record_state_history


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


def test_cadence_is_a_recording_interval_not_a_behavior_gate():
    assert server_state.STATE_HISTORY_RECORD_SECONDS == 300.0


def test_the_main_loop_writes_on_its_own_schedule():
    """The loop block: time-gated on STATE_HISTORY_RECORD_SECONDS, stamps the
    attempt before writing (so a failing write cannot retry every tick), and
    reports a failure through note_suppressed rather than a bare pass."""
    source = inspect.getsource(server)
    block = source[source.index("# Durable anima + sensor history"):]
    block = block[: block.index("# System metrics pruning")]
    assert "STATE_HISTORY_RECORD_SECONDS" in block
    assert block.index("last_state_history_at = time.time()") < block.index("record_state_history(")
    assert 'note_suppressed("server.state_history", e)' in block
