# anima_broker (Elixir) — Phase 0

The Elixir/OTP rewrite of Lumen's hardware broker, built as a **strangler
migration** alongside the Python services. See the design doc:
`docs/plans/2026-06-29-elixir-broker-migration.md`.

**Phase 0 scope:** prove the stack and the interop seam with near-zero risk.
No hardware, no governance yet — just the supervision skeleton and an
`Shm.Writer` that emits an envelope byte-compatible with what the Python MCP
server's `SharedMemoryReader` already reads.

## What's here

```
lib/anima_broker/
  application.ex          # OTP app: one_for_one supervision tree
  state/store.ex          # holds the `data` payload (the SHM body)
  sensors/supervisor.ex   # empty stub (Phase 1 adds I2C readers)
  shm/writer.ex           # atomic temp-write + rename of the envelope
  tick.ex                 # tick loop (Process.send_after); flushes each tick
```

## The interop contract

`Shm.Writer` writes:

```json
{"updated_at": "<iso8601 naive>", "pid": <int>, "data": { ... }}
```

…with **atomic temp-write + `rename()`** (matching `anima_mcp/shared_memory.py`).
Single-writer atomic rename means the Python reader never sees a torn file,
even before the advisory `flock` is replicated (deferred — see the design doc).

## Safety: shadow path

Phase 0 writes to **`/dev/shm/anima_state.shadow.json`** by default, NOT the
live `/dev/shm/anima_state.json`. The Python broker stays the sole writer of
the live file. Cutover happens in Phase 1.

## Build / test / run

```bash
# from anima_broker/
mix deps.get
mix test                      # ExUnit (autostart disabled in :test)
mix run --no-halt             # run the tree locally (writes the shadow file)

# release for the Pi:
MIX_ENV=prod mix release
_build/prod/rel/anima_broker/bin/anima_broker start
```

> Note: this skeleton was authored without a local BEAM to run it against, so
> `mix test` should be run on a machine with Elixir installed before relying on
> it. The tests assert the envelope contract and atomic-write behavior.

## Cross-language acceptance check (manual)

Phase 0 is "done" when the Python reader parses the Elixir-written file:

```bash
# 1. run the Elixir broker so it writes the shadow file
#    (on a host without /dev/shm, e.g. macOS, ANIMA_SHM_PATH=/tmp/... mix run --no-halt)
mix run --no-halt &

# 2. from the Python repo root, point the reader at the shadow file.
#    The class is SharedMemoryClient (read mode); .read() returns the `data` body.
python3 -c "from pathlib import Path; \
from anima_mcp.shared_memory import SharedMemoryClient; \
import pprint; pprint.pp(SharedMemoryClient(mode='read', filepath=Path('/dev/shm/anima_state.shadow.json')).read())"
```

(Class/arg names per `anima_mcp/shared_memory.py` — `src/` must be importable,
e.g. run from the repo root with the package installed or `PYTHONPATH=src`.)

## Configuration

| Env var | Default | Meaning |
|---------|---------|---------|
| `ANIMA_SHM_PATH` | `/dev/shm/anima_state.shadow.json` | where the envelope is written |
| `ANIMA_TICK_MS`  | `2000` | tick interval (ms) |
| `ANIMA_I2C_BUS`  | _(unset)_ | I2C bus name (e.g. `i2c-1` on the Pi). Unset ⇒ no sensors start; broker still boots and writes the envelope. |

## Phase 1 — I2C sensors (implemented)

`Circuits.I2C`-backed readers, one GenServer per device, each merging its slice
into `State.Store`:

| Sensor | Addr | Reading keys | Config |
|--------|------|--------------|--------|
| `Sensors.VEML7700` | 0x10 | `light_lux` | gain 1x, 200ms (matches Python; `0.0288` lx/count) |
| `Sensors.AHT20`    | 0x38 | `ambient_temp_c`, `humidity_pct` | trigger + 20-bit decode |
| `Sensors.BMP280`   | 0x76 | `pressure_hpa`, `pressure_temp_c` | datasheet float compensation |

Drivers depend on the `AnimaBroker.Hardware.I2C` behaviour, not on `Circuits.I2C`
directly, so the conversion/compensation math is unit-tested with a fake bus
(`test/support/fake_i2c.ex`) — no hardware needed. BMP280's signed/unsigned
calibration mix and the datasheet Appendix-A reference vector are covered in
`test/anima_broker/sensors/bmp280_test.exs`.

`Shm.Writer` now emits **local** `updated_at` (`NaiveDateTime.local_now/0`),
matching Python's naive `datetime.now()` so server freshness checks don't skew —
the prerequisite for live cutover.

### Next: cutover (not in this PR)

Soak alongside the Python broker (~24h, cross-check `read_sensors`), then flip
`ANIMA_SHM_PATH` to the live `/dev/shm/anima_state.json` and stop the Python
broker's sensor ownership. Reversible at every step.

### Next: Phase 2

Move UNITARES governance onto OTP (kills the `sync+ThreadPoolExecutor+aiohttp`
bug). Comes after Phase 1 cutover.
