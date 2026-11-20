"""Throttled error prints must be reachable from the block that owns them.

The subsystem blocks in the main loop run only when ``loop_count`` is a
multiple of their interval (agency 5, self-model 5, primitive language 10).
Their error prints were throttled with ``loop_count % THROTTLE == 1``. A
multiple of 5 or 10 never leaves remainder 1 modulo 100 or 300, so those
errors were never printed at all.
"""

import re
from pathlib import Path

import pytest

from anima_mcp import server_state

SRC = Path(server_state.__file__).parent
INTERVALS = {
    "METACOG_INTERVAL": server_state.METACOG_INTERVAL,
    "AGENCY_INTERVAL": server_state.AGENCY_INTERVAL,
    "SELF_MODEL_INTERVAL": server_state.SELF_MODEL_INTERVAL,
    "PRIMITIVE_LANG_INTERVAL": server_state.PRIMITIVE_LANG_INTERVAL,
}
THROTTLES = {
    "ERROR_LOG_THROTTLE": server_state.ERROR_LOG_THROTTLE,
    "STATUS_LOG_THROTTLE": server_state.STATUS_LOG_THROTTLE,
}


@pytest.mark.parametrize("interval", INTERVALS.values(), ids=INTERVALS.keys())
@pytest.mark.parametrize("throttle", THROTTLES.values(), ids=THROTTLES.keys())
def test_remainder_zero_is_reachable_inside_every_interval_block(interval, throttle):
    reachable = [
        n for n in range(1, interval * throttle + 1)
        if n % interval == 0 and n % throttle == 0
    ]
    assert reachable


def test_remainder_one_was_unreachable_for_agency_and_primitive_language():
    for interval in (server_state.AGENCY_INTERVAL, server_state.PRIMITIVE_LANG_INTERVAL):
        assert not any(
            n % interval == 0 and n % server_state.STATUS_LOG_THROTTLE == 1
            for n in range(1, 10_000)
        )


@pytest.mark.parametrize("module", ["server.py", "loop_phases.py"])
def test_no_error_print_is_throttled_on_remainder_one(module):
    assert not re.findall(r"_LOG_THROTTLE == 1\b", (SRC / module).read_text())
