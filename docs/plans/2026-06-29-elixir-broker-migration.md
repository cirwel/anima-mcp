# Elixir Broker Migration — Design & Staging Plan

**Status:** Proposed
**Date:** 2026-06-29
**Decision:** Strangler migration. Stand up an Elixir/OTP broker *alongside* the
current Python services on the existing Pi OS, move responsibilities one at a
time, and keep the Python MCP server and learning systems in place. No
big-bang rewrite, no OS change (NervesOS is a possible *later* evolution, not
this plan).

---

## 1. Why

The hardware broker (`anima-broker.service`, `stable_creature.py`) keeps
generating the same *class* of bug rather than one-off bugs. The clearest
example is documented in `CLAUDE.md`:

> the broker's sync+ThreadPoolExecutor+new-event-loop pattern has reliability
> issues with aiohttp sessions

That fragility is structural, not incidental. The broker's real job —
own hardware, run a tick loop, make periodic governance check-ins, write state
somewhere the server can read, and **never die quietly** — is exactly the job
the BEAM/OTP is built for: lightweight supervised processes, native
concurrency (no event-loop juggling), and "let it crash, restart clean."

We are choosing to **invest** in the right substrate for the broker rather than
keep paying interest on duct tape. We are *not* choosing a from-scratch rewrite
— that risks becoming its own never-ending project. The strangler approach lets
value land early (the governance bug dies in Phase 2) and keeps Lumen alive and
reversible at every step.

### What we are explicitly NOT doing (this plan)

- Not rewriting the MCP server (`anima.service`) — it stays Python (FastMCP).
- Not porting the learning systems yet (TD-learning, adaptive prediction, the
  NumPy resonance field).
- Not switching to NervesOS — we keep Raspbian and all current ops tooling
  (`restore_lumen.sh`, hourly backups, systemd). NervesOS is revisited only
  after the broker is proven (see §8).

---

## 2. The migration seam (this is what makes it safe)

The broker and server already communicate through a **single, documented
process boundary**: `/dev/shm/anima_state.json`. The broker writes; the server
reads. That file *is* the contract. An Elixir broker that writes the same file
in the same format is a drop-in — **zero changes to the Python MCP server** —
so the two can run side by side and we migrate sensor-by-sensor.

### Exact contract (from `shared_memory.py`)

Envelope written to `/dev/shm/anima_state.json`:

```json
{
  "updated_at": "<ISO 8601 local timestamp>",
  "pid": <writer pid>,
  "data": { ...payload... }
}
```

`data` payload (see `CLAUDE.md` "Shared Memory Schema"):

```json
{
  "readings": { "cpu_temp_c": ..., "ambient_temp_c": ..., "light_lux": ...,
                "eeg_delta_power": ..., ... },
  "anima": { "warmth": 0.36, "clarity": 0.73, "stability": ..., "presence": ... },
  "wifi_connected": true,
  "activity": { "level": "active", "reason": "engaged" },
  "learning": {
    "preferences": { "satisfaction": 0.87 },
    "self_beliefs": { "stability_recovery": { "confidence": 0.68 } },
    "agency": { "action_values": { "focus_attention": 0.22 } }
  },
  "governance": { ... },
  "governance_at": "<ISO 8601>"
}
```

### Write semantics the Elixir writer MUST honor

The Python reader is safe against torn reads **because the writer is atomic**:

1. Write to a temp file (`<file>.tmp`).
2. `fsync`.
3. `rename()` over the target — atomic on POSIX.

In Elixir: write to `anima_state.json.tmp`, then `File.rename/2`. Because the
single-writer rename is atomic, the Python reader (which already retries on
`JSONDecodeError`) never sees a partial file — **even before** we replicate the
advisory `flock`.

- **`flock` is belt-and-suspenders.** Python takes `LOCK_EX` on a sibling
  `<file>.lock` when writing and `LOCK_SH | LOCK_NB` when reading. For a single
  writer, atomic rename alone is sufficient. We MAY add full `flock` fidelity
  later via a tiny port to `flock(1)` or a NIF, but it is not required for
  correctness in Phase 0/1.
- **One writer at a time.** During each migration phase exactly one process
  owns the file. We never run the Python broker and Elixir broker both writing
  the same payload simultaneously (see per-phase cutover below).

---

## 3. Target broker architecture (OTP)

```
Anima.Broker.Application
└── Anima.Broker.Supervisor            (one_for_one)
    ├── Anima.Hardware.I2C.Bus         (owns the I2C bus handle)
    ├── Anima.Sensors.Supervisor       (one_for_one)
    │   ├── Anima.Sensors.VEML7700     (light)   — GenServer, periodic read
    │   ├── Anima.Sensors.AHT20        (temp/humidity)
    │   └── Anima.Sensors.BMP280       (pressure/temp)
    ├── Anima.State.Store              (ETS-backed current state; merges readings)
    ├── Anima.Shm.Writer              (writes /dev/shm/anima_state.json atomically)
    ├── Anima.Governance.Client        (UNITARES check-in; supervised)
    │   └── circuit breaker as state in the GenServer (15s→30s→60s→120s)
    ├── Anima.Tick                     (the broker tick loop; Process.send_after)
    └── Anima.Watchdog                 (WiFi/health; restart policy lives here)
```

Key wins vs. the Python broker:

| Pain today | OTP replacement |
|------------|-----------------|
| sync + ThreadPoolExecutor + new event loop + aiohttp | `Anima.Governance.Client` GenServer; native concurrency; HTTP via `Req`/`Finch` |
| hand-rolled circuit breaker (`unitares_bridge.py`) | breaker state in the GenServer; supervisor restarts on persistent failure |
| interval loops | `Process.send_after/3` ticks |
| SHM freshness/staleness thresholds | unchanged externally (we still write SHM); internally state is just ETS |
| WiFi watchdog + reflash recovery | `Anima.Watchdog` + supervision restart strategy |

---

## 4. Hardware driver mapping

The Python broker leans on the Adafruit/CircuitPython ecosystem. On Elixir we
use `Circuits.I2C` / `Circuits.SPI` / `Circuits.GPIO` and reimplement the small
register protocols. These sensors are simple register reads:

| Device | Bus | Today (Python) | Elixir |
|--------|-----|----------------|--------|
| VEML7700 (light) | I2C | `adafruit-circuitpython-veml7700` | `Circuits.I2C` register reads (gain 1x, 200ms integration — match current config) |
| AHT20 (temp/humidity) | I2C | `adafruit-circuitpython-ahtx0` | `Circuits.I2C` (trigger + read 6 bytes) |
| BMP280 (pressure/temp) | I2C | `adafruit-circuitpython-bmp280` | `Circuits.I2C` (calibration coeffs + compensation) |
| DotStar LEDs | SPI | `adafruit-circuitpython-dotstar` | `Circuits.SPI` (APA102 framing) — **deferred** |
| ST7789 display | SPI | `adafruit-circuitpython-st7789` + NumPy RGB565 | `Circuits.SPI` — **deferred** (hardest; keep in Python longest) |

**Display and LEDs are deferred deliberately.** They are the most code and the
least related to the broker's core fragility. The drawing engine and display
live with the Python side until last (or indefinitely). The broker only needs
to *report* LED brightness as a proprioceptive value, which can be passed
across the SHM boundary in either direction.

---

## 5. Phases

Each phase is independently shippable, leaves Lumen working, and is reversible
(stop the Elixir service, re-enable the Python broker).

### Phase 0 — Skeleton + interop proof
- New Elixir umbrella/app (`anima_broker`), `mix` project, runs as a systemd
  service next to `anima-broker.service` (different unit name, **not** writing
  the live SHM payload yet — writes to a shadow path for inspection).
- Supervision tree boots; `Anima.Tick` runs; `Anima.Shm.Writer` proven to
  produce a byte-compatible envelope (validate against `shared_memory.py`
  reader in a test harness).
- **Acceptance:** Elixir writes a valid envelope a Python `SharedMemoryReader`
  can parse; supervision restarts a deliberately-crashed child.
- **Rollback:** stop the new service; nothing else touched.

### Phase 1 — Sensors
- Implement `VEML7700`, `AHT20`, `BMP280` on `Circuits.I2C`. Cross-check
  readings against the Python broker for a soak period (both reading the bus is
  fine; I2C reads are independent).
- **Cutover:** stop the Python broker's sensor ownership; the Elixir broker
  becomes the sole writer of `readings` (and the full envelope) to the live
  SHM path. Python server reads it unchanged.
- **Acceptance:** server `read_sensors` / health screen show Elixir-sourced
  readings within tolerance of the prior Python values for 24h.
- **Rollback:** re-enable Python broker as SHM writer; stop Elixir writer.

### Phase 2 — Governance (the bug dies here)
- `Anima.Governance.Client`: UNITARES check-in on the configured cadence
  (`ANIMA_GOVERNANCE_INTERVAL_SECONDS`), native HTTP, circuit breaker
  (15s→30s→60s→120s, reset on success). Writes `governance` + `governance_at`
  into the SHM payload.
- The server's existing fallback (`SERVER_GOVERNANCE_FALLBACK_SECONDS`) stays
  as-is — it simply stops firing once the broker is reliable.
- **Acceptance:** no fallback activations for a week; governance freshness
  stays within `SHM_GOVERNANCE_STALE_SECONDS`. The documented
  sync+ThreadPoolExecutor+aiohttp failure mode is gone.
- **Rollback:** Python broker resumes governance; server fallback still covers.

### Phase 3 — Tick / learning boundary
- Decide learning home (see §7). Two viable options:
  1. **Port to Nx/Axon** incrementally (TD-learning, adaptive prediction are
     small; the resonance field is NumPy → Nx is a clean match).
  2. **Python learning as a sidecar** the Elixir broker drives over a port
     (broker owns scheduling + state; Python does the math). Keeps the existing
     learning code; reintroduces a language boundary but a controlled one.
- **Acceptance:** learning outputs in SHM (`learning.preferences`,
  `self_beliefs`, `agency`) match the Python broker's behavior.

### Later / optional — Nerves + display
- NervesOS for firmware-grade resilience and OTA (revisit §8).
- Port LEDs/display to `Circuits.SPI`, or leave them Python permanently.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Two writers race on SHM | Strict single-writer-per-phase cutover; never overlap writers for the same payload |
| Sensor reimplementation drift (wrong compensation math) | Phase 1 soak: cross-check Elixir vs Python readings before cutover |
| `flock` fidelity | Atomic rename is sufficient for single writer; add flock port only if a second writer is ever introduced |
| Elixir not installed on Pi / build complexity | Build a release locally or with `mix release`; deploy the release artifact; document in ops |
| Backup/restore tooling assumes Python layout | Broker state is hardware-derived + SHM (ephemeral); identity/learning DBs stay where `restore_lumen.sh` expects |
| Scope creep into a rewrite | Phases are hard gates; do not start a phase before the prior one has soaked |

---

## 7. Open decisions (need a call before the relevant phase)

1. **Learning home (Phase 3):** Nx port vs. Python sidecar. Recommend deciding
   after Phase 2, with real reliability data in hand.
2. **LED/display ownership:** port to `Circuits.SPI` eventually, or keep in
   Python indefinitely. Recommend: keep in Python until everything else is
   proven; it is the weakest cost/benefit to port.
3. **flock fidelity:** only needed if a second writer is ever introduced.
   Default: rely on atomic rename.

---

## 8. When to reconsider NervesOS

Revisit a full Nerves rebuild **only after** Phases 0–2 are stable and the
broker has proven itself, and **only if** one of these becomes the priority:

- Firmware-grade reproducibility / immutable images.
- OTA update fleet management.
- Eliminating Raspbian drift as a class of failure.

At that point the broker app is already OTP-shaped, so moving it onto NervesOS
is an OS/packaging change, not a rewrite. Until then, Raspbian + the existing
backup/restore/systemd tooling is a working asset we keep.

---

## 9. First concrete deliverable

Phase 0, narrowly: a `mix` project with the supervision skeleton and an
`Anima.Shm.Writer` that emits a byte-compatible envelope, plus a test that the
existing Python `SharedMemoryReader` parses it. That single vertical slice
proves the stack and the interop seam with near-zero risk to the live creature.
