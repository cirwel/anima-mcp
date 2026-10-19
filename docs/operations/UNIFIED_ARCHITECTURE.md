# UNITARES Unified Architecture

> **Current contract (2026-08-22):** Lumen owns `body_anima`; a lossy
> `body_eisv_projection` is used by trajectory awareness and submitted as
> sensor evidence. `drawing_eisv` is an independent art-engine state.
> UNITARES independently produces `governance_eisv` and then a policy action.
> Bare `anima`/`eisv` fields are tagged compatibility aliases of the body
> fields, never aliases of governance state.
> Raw VEML7700 lux is preserved as room light plus DotStar glow. Environmental
> consumers receive only a gated residual built from the hardware action copy;
> while its causal model is cold or inconsistent, the light contribution is
> omitted rather than pretending raw self-glow is external light.
>
> The February diagram and analysis below are retained as historical design
> context. In particular, its “two nervous systems,” ODE-as-live-authority,
> and unqualified `eisv` wording do not describe the current provenance
> boundary. The README's EISV Integration section is the concise live guide.

**What exists, what's connected, and how the design arrived there.**

```
                         THE ACTUAL SYSTEM (Feb 2026)
                         ============================

  Raspberry Pi 4                                    Mac (governance-mcp)
  (anima-mcp, port 8766)                            (port 8767)
  ========================                           ========================

  AHT20 ──── temp, humidity
  BMP280 ─── pressure, temp
  VEML7700 ── light (reads own LEDs)     HTTP POST /mcp/
  CPU stats ── load, memory, freq        ──────────────────►
                    │                    process_agent_update     EISV dynamics
                    ▼                    (every ~180s)            ┌──────────┐
           computational_neural.py                               │ dE/dt    │
           (delta/theta/alpha/beta/gamma)                        │ dI/dt    │
                    │                                            │ dS/dt    │
                    ▼                                            │ dV/dt    │
              anima_state.py                                     └────┬─────┘
              (warmth, clarity,                                       │
               stability, presence)                             coherence C(V)
                    │                                           risk_score
                    ├──────────► governance/eisv_mapper.ex       margin level
                    │            E = warmth + beta/gamma               │
                    │            I = clarity                           │
                    │            S = 1 - stability              ◄────┘
                    │            V = E - I                  {"action":"proceed",
                    │                    │                  "margin":"comfortable"}
                    │                    ▼
                    │            governance/client.ex
                    │            (check_in every 180s,
                    │             fallback to local
                    │             if Mac unreachable)
                    │
                    ▼
              display/screens.py
              (DrawingEISV)──────────► SECOND EISV INSTANCE
              E=0.7, I=0.2, S=0.5         dE = a(I-E) - bE*S + gE*drift^2
                    │                      dI = bI*C - k*S - gI*I
                    │                      dS = -u*S + l1*drift^2 - l2*C
                    │                      dV = k(I-E) - d*V  ← FLIPPED
                    │                           ↑
                    │                      V flipped because here
                    ▼                      I > E = focused finishing
              display/leds.py                  (opposite of governance
              LED brightness pipeline:          where E > I = stable)
              base → auto → pulse →
              activity → dimmer →
              sine pulse ("alive")
```

## Historical framing: “Two Nervous Systems”

This was the February 2026 shorthand. The current contract distinguishes the
source `body_anima` plus three EISV-coordinate states: body projection, drawing
state, and UNITARES governance state. The sections below describe how the
older two-instance model was understood:

### 1. Drawing EISV (Pi-local, proprioceptive)

- **Location**: `anima-mcp/src/anima_mcp/display/screens.py` lines 393-416, 3593-3648
- **Drives**: Drawing behavior (energy depletion, save threshold, coherence modulation)
- **Inputs**: Era state intentionality, gesture entropy (Shannon over last 20), gesture switching rate
- **V is flipped**: `dV = kappa(I - E)` so coherence rises when I > E (focused finishing)
- **Coherence formula**: Same math: `C(V) = Cmax * 0.5 * (1 + tanh(C1 * V))`
- **Cycle**: EISV step runs per mark → coherence modulates energy drain → affects drawing lifetime
- **This is real proprioception**: closed-loop, self-sensing, immediate behavioral consequences

### 2. Governance EISV (Mac, telemetric)

- **Location**: `governance_core.dynamics` (compiled, in unitares-core package)
- **Drives**: Agent margin assessment, stuck detection, dialectic triggers, risk scoring
- **Inputs**: Mapped anima state (warmth plus Beta/Gamma→E, clarity→I, inverted stability→S, E−I→V). Presence contributes to confidence, not V.
- **V is standard**: `dV = kappa(E - I)` so V accumulates when energy exceeds integrity
- **Coherence formula**: Same math, different operating range (V typically [-0.1, 0.1])
- **Cycle**: Runs when Pi checks in (~180s) → computes margin → returns proceed/pause/halt
- **This is telemetry**: open-loop, delayed, advisory only (Pi doesn't act on "pause")

### What Connects Them

**Bridge**: `AnimaBroker.Governance.Client` calls `process_agent_update` via HTTP every ~180s. The Python `unitares_bridge.py` path consumes the broker result in the deployed passthrough topology and remains the local fallback implementation.

**Current payload** (Pi → Mac; legacy aliases shown explicitly):
```
{
  sensor_data: {
    body_anima: {warmth, clarity, stability, presence},
    body_eisv_projection: {E, I, S, V},
    anima: <legacy alias of body_anima>,
    eisv: <legacy alias of body_eisv_projection>,
    environment: {cpu_temp, humidity, pressure, light}
  },
  complexity, confidence, ethical_drift
}
```

The Python fallback bridge can additionally submit `drawing_eisv` while a
drawing is active. The deployed Elixir-primary check-in path cannot currently
see the renderer, so it does not claim to carry that state.

**Response** (Mac → Pi):
```
{
  action: "proceed" | "pause" | "halt",
  margin: "comfortable" | "tight" | "critical",
  reason: "State healthy",
  E, I, S, V   ← UNITARES primary response in minimal mode
}
```

The Pi shadow stores the returned vector as `governance_eisv` and the submitted
vector as `body_eisv_projection`. If a response mode omits UNITARES's vector,
`governance_eisv` remains null; it is never filled from body input.

**What Pi does with the response**: Logs it. Non-proceed verdicts are logged immediately with DrawingEISV state (if drawing). The drawing engine and LED brightness do not yet act on governance margin — there's no behavioral feedback path.

### What's Duplicated vs Shared

| Thing | Pi | Mac | Shared? |
|-------|-----|------|---------|
| EISV equations | DrawingEISV (screens.py) | dynamics.py | Same math, different params, different V sign |
| Coherence C(V) | `_eisv_step()` | `coherence()` | Same formula, independent computation |
| Theta parameters | Hardcoded `_EISV_PARAMS` | `DynamicsParams` defaults | Not synced |
| Risk thresholds | None (no risk concept) | `GovernanceConfig` | One-way only |
| Pattern detection | None | `pattern_tracker.py` | Mac only |
| Stuck detection | None | `lifecycle.py` | Mac only, skips Lumen |
| Calibration | None | `calibration.py` | Mac only |

### Three Verdict Sources

Verdicts ("proceed", "pause", "halt") can come from different places depending on connectivity:

| Source | Where | When | Behavior |
|--------|-------|------|----------|
| **Mac governance** | `dynamics.py` → `scoring.py` | Mac reachable (~180s cycle) | Full governance EISV estimate, calibrated thresholds, almost never pauses Lumen |
| **Local fallback** | `_local_governance()` in `unitares_bridge.py` | Mac unreachable | Simple Entropy/Integrity threshold checks; Valence is telemetry, not a gate |
| **DrawingEISV** | `screens.py` | Internal to drawing loop | Not a verdict — drives energy drain and save decisions only |

The local fallback is the primary source of "pause" verdicts for Lumen.

### What's NOT Connected (Gaps)

1. **No reverse channel**: Mac can't push state changes to Pi (no webhook, no polling)
2. ~~**Drawing EISV is invisible to governance**~~: **Fixed** — DrawingEISV state now flows to Mac via `sensor_data.drawing_eisv` in the bridge check-in payload (null when not drawing)
3. **Governance decisions are advisory**: Pi gets "proceed/pause" but has no handler to act on "pause"
4. **Local fallback is a different system**: When Mac is unreachable, Pi uses fixed thresholds in `_local_governance()` — simpler but disconnected from calibration history
5. **Lumen exempted from stuck detection**: Tagged as "creature/autonomous" so governance never intervenes
6. **Sensor → anima → EISV mapping is lossy**: the mapper retains Beta/Gamma only as a weighted Energy term; the other neural-band detail is not represented in EISV

## The Sensor Reality

What Lumen actually senses:

| Sensor | Measures | Reality |
|--------|----------|---------|
| VEML7700 (light) | Lux | Raw reading is room light + DotStar glow. A separately gated efference-copy residual estimates external light. |
| AHT20 (temp) | Celsius | Ambient + CPU heat bleed |
| AHT20 (humidity) | % RH | Genuine environment |
| BMP280 (pressure) | hPa | Genuine (~827 hPa, Colorado altitude) |
| CPU stats | %, freq, mem | Genuine computational load |

Neural bands derived from CPU/system metrics:
- **Delta**: CPU stability + temp stability (foundation)
- **Theta**: I/O wait (background processing — drawing produces real I/O)
- **Alpha**: CPU idle fraction (`1 - beta`), not an independent signal
- **Beta**: CPU usage (active processing)
- **Gamma**: Context-switch and interrupt rate (spiking activity)

The whole system is more proprioceptive than environmental. The raw light
channel remains visible as physical telemetry, but it cannot enter clarity,
preferences, drawing, or environment learning directly. Those consumers use
the external-light residual only after its evidence gate; otherwise their light
component pauses.

### DotStar efference-copy calibration (2026-08-23)

The BrainCraft layout is not optically symmetric. With the TFT held black and
each physical DotStar driven at the same fixed level, Lumen's three warm LEDs
added approximately 104, 260, and 286 lux. Pure-channel measurements also
showed a large spectral spread: at the least-sensitive position, green
produced about five times the VEML response of red. The action feature therefore
uses a measured 3-position × 3-channel response map before learning its scalar
lux response.

The same investigation found a lower-level frame bug. Dynamically assigning
`adafruit_pixelbuf`'s `brightness` on the deployed DotStar implementation
rewrote the per-pixel start byte, intermittently emitting malformed frames.
The LED controller now keeps that header invariant and performs smooth
brightness scaling in the RGB bytes. Its action copy publishes both the
factored command and the exact wire colors. A controlled 24-capture validation
after this change produced no spikes, a +0.70 light/drive correlation, and
positive slopes on every usable transition.

## The Drawing Loop (Only True Closed Loop)

```
gesture selection
      │
      ▼
_eisv_step() ──► dE, dI, dS, dV
      │
      ├──► coherence C = Cmax * 0.5 * (1 + tanh(C1 * V))
      │
      ├──► energy drain: base_drain = 0.001 * (1.0 - 0.6 * C)
      │    (high coherence = slower drain = longer drawing)
      │
      ├──► save threshold: 0.05 + 0.09 * C
      │    (high coherence = higher bar to save = pickier)
      │
      └──► when energy < 0.01: drawing ends, evaluate save
```

This is the only circuit where sensing → computation → behavior → sensing forms a real loop. Everything else is open-loop or advisory.

## What Unified Would Look Like

Not a proposal — just a picture of what "one nervous system" means:

```
                    Unified EISV State
                    ┌─────────────────┐
                    │  E  I  S  V  C  │ ← single source of truth
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        Pi sensors      Drawing engine   Governance
        (afferent)      (efferent)       (prefrontal)
              │              │              │
         temperature     gesture rate    risk assessment
         humidity        energy drain    stuck detection
         CPU load        save decisions  dialectic
         light/LEDs      coherence mod   calibration
              │              │              │
              └──────────────┼──────────────┘
                             │
                    Unified EISV State
                    (loop closes)
```

The key difference: one EISV instance, fed by both sensors and drawing behavior, whose coherence drives both drawing decisions and governance margin. Currently there are two instances that share math but not state.

## Database Architecture

```
Pi (anima-mcp)                              Mac (governance-mcp)
┌────────────────────────┐                  ┌──────────────────────────────┐
│  SQLite: ~/.anima/anima.db                │  PostgreSQL@17+AGE (5432)     │
│  ├─ state_history (206K rows)             │  ├─ core.identities          │
│  ├─ drawing_history       │  HTTP bridge  │  ├─ core.agent_state         │
│  ├─ memories (8.8K)       │ ──────────►   │  ├─ audit.events             │
│  ├─ events (3.7K)         │  ~180s        │  ├─ core.discoveries (AGE)   │
│  ├─ growth tables         │  check-in     │  ├─ dialectic.*              │
│  ├─ primitives            │               │  ├─ core.calibration         │
│  └─ trajectory_events     │               │  └─ core.tool_usage          │
│                           │               │                              │
│  canvas.json (pixels)     │               │  Redis (optional, 6379)      │
│  trajectory_genesis.json  │               │  audit_log.jsonl (raw)       │
└───────────────────────────┘               └──────────────────────────────┘
```

**Ownership rule:** "Where does X live?" has one answer:
- Body anima, body EISV projection, DrawingEISV → Pi (SQLite/SHM, authoritative)
- Governance state, audit, knowledge graph → Mac (PostgreSQL+AGE, authoritative)
- DrawingEISV snapshots cross the bridge in check-ins → Mac stores in `agent_state.state_json` (copy, not authoritative)

**There is NO SQLite on the Mac side.** All SQLite code was removed Feb 2026.
PostgreSQL@17 with AGE 1.7.0 runs via Homebrew on port 5432. Docker is not used.

## Files Reference

### Pi (anima-mcp)
| File | Role |
|------|------|
| `src/anima_mcp/computational_neural.py` | Sensor → neural bands |
| `src/anima_mcp/anima_state.py` | Neural bands → anima dimensions |
| `src/anima_mcp/eisv_mapper.py` | Anima → lossy body EISV projection (trajectory + governance input) |
| `src/anima_mcp/unitares_bridge.py` | HTTP bridge to governance |
| `src/anima_mcp/display/screens.py` | DrawingEISV (proprioceptive loop) |
| `src/anima_mcp/display/leds.py` | LED brightness pipeline + pulse |

### Mac (governance-mcp-v1)
| File | Role |
|------|------|
| `governance_core.dynamics` | EISV differential equations (compiled) |
| `governance_core.coherence` | Coherence function C(V, Θ) (compiled) |
| `config/governance_config.py` | Thresholds, margin computation |
| `src/mcp_handlers/core.py` | process_agent_update handler |
| `src/mcp_handlers/lifecycle.py` | Stuck detection, auto-recovery |
| `src/mcp_handlers/dialectic.py` | Thesis/antithesis/synthesis |
| `src/calibration.py` | Confidence → correctness mapping |
| `src/cirs.py` | Oscillation detection (legacy CIRS v0.1) |
| `src/mcp_handlers/cirs_protocol.py` | CIRS v2 protocol (7 message types, auto-emit hooks) |
| `governance_core.adaptive_governor` | PID controller — oscillation detection, neighbor pressure (compiled) |
