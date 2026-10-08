"""Durable anima + sensor history (the `state_history` table).

The server's main loop is the only writer, on its own schedule
(`STATE_HISTORY_RECORD_SECONDS`). Until 2026-10 the rows were written as a
side effect of two read handlers (`get_state`, `get_lumen_context`), so the
history existed only while some client polled: from April it was one
resident's sync every ~300 s, and when that poller stopped at the 2026-08-31
power outage, state_history stopped too (5 rows in the following five weeks)
while every other subsystem kept running. A read no longer writes; the
cadence no longer depends on who is asking.

Each optional enrichment may fail without stopping the record, but the gap is
counted through `note_suppressed`, never swallowed: a field that quietly stops
appearing is indistinguishable from one that had nothing to report.
"""

from __future__ import annotations

from typing import Any

from .error_recovery import note_suppressed
from .light_attribution import gated_external_light_lux


def history_sensors(readings, light_attribution: Any = None) -> dict:
    """The sensor payload stored with each state_history row."""
    sensors = readings.to_dict()
    sensors["external_light_lux"] = gated_external_light_lux(light_attribution)
    sensors["light_attribution_status"] = (
        light_attribution.get("status")
        if isinstance(light_attribution, dict)
        else "unavailable"
    )
    # New broker snapshots carry capture-aligned LED brightness. Fill from live
    # proprioception only for older/partial SHM payloads so history remains
    # decomposable without relabelling raw lux.
    if sensors.get("led_brightness") is None:
        try:
            from .accessors import _get_led_brightness

            sensors["led_brightness"] = _get_led_brightness()
        except Exception as e:
            note_suppressed("state_history.led_brightness", e)
    try:
        from .accessors import _get_growth

        growth = _get_growth()
        if growth is not None:
            level = growth.interaction_level()
            if level is not None:
                sensors["interaction_level"] = level
    except Exception as e:
        note_suppressed("state_history.interaction_level", e)
    return sensors


def record_state_history(store, anima, readings, light_attribution: Any = None) -> None:
    """Write one state_history row for the current anima and readings."""
    store.record_state(
        anima.warmth,
        anima.clarity,
        anima.stability,
        anima.presence,
        history_sensors(readings, light_attribution),
    )
