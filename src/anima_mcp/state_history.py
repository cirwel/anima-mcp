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

import time
from typing import Any

from .error_recovery import note_suppressed
from .light_attribution import gated_external_light_lux


def history_sensors(readings, light_attribution: Any = None,
                    shm_readings: dict | None = None) -> dict:
    """The sensor payload stored with each state_history row.

    `shm_readings` is the broker snapshot's `readings`. Its `led_brightness`
    is capture-aligned to the light sensor and wins: the server loop rewrites
    `readings.led_brightness` with its own applied brightness (0.0 when
    unknown) before this runs, which would make lux undecomposable.
    """
    sensors = readings.to_dict()
    if isinstance(shm_readings, dict):
        sensors["led_brightness"] = shm_readings.get("led_brightness")
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


def record_state_history(store, anima, readings, light_attribution: Any = None,
                         shm_readings: dict | None = None) -> None:
    """Write one state_history row for the current anima and readings."""
    store.record_state(
        anima.warmth,
        anima.clarity,
        anima.stability,
        anima.presence,
        history_sensors(readings, light_attribution, shm_readings),
    )


def maybe_record_state_history(ctx, anima, readings, shm: dict | None,
                               interval_seconds: float,
                               now: float | None = None) -> bool:
    """The main loop's writer: record if `interval_seconds` have passed.

    The attempt is stamped before writing, so a failing write waits a full
    interval instead of retrying every tick; the failure is counted through
    `note_suppressed`. Returns True when a row was written.
    """
    if not (readings and anima and ctx and ctx.store):
        return False
    now = time.time() if now is None else now
    if now - ctx.last_state_history_at < interval_seconds:
        return False
    ctx.last_state_history_at = now
    try:
        record_state_history(
            ctx.store, anima, readings,
            shm.get("light_attribution") if shm else None,
            shm.get("readings") if shm else None,
        )
        return True
    except Exception as e:
        note_suppressed("server.state_history", e)
        return False
