# Anima MCP - Agent Instructions

## CRITICAL: Read Before Acting

| Situation | Action |
|-----------|--------|
| **Pi reflashed / Lumen down** | `cd ~/projects/anima-mcp && ./scripts/restore_lumen.sh` — one command, do NOT do it manually |
| **Looking for backups** | `ls -lt ~/backups/lumen/anima_*.db \| head -5` — real backups are here. `~/lumen-backups/` is OLD/STALE |
| **After `git_pull(restart=true)`** | Wait **2 minutes**. Do NOT SSH or retry. "fetch failed" = normal, Pi is rebooting |
| **WiFi crash (wlan0 disappears)** | Reboot Pi — WiFi watchdog will recover. Do NOT hammer with SSH during reboot |
| **Data appears lost** | Check `~/backups/lumen/` FIRST. Backups run hourly. Do not declare data lost without checking |

Full backup/restore details: `docs/operations/BACKUP_AND_RESTORE.md`

---

## Design Invariants

Two laws, both earned the hard way. Check any PR against them.

**1. No new absolute thresholds on Lumen's behavior.** Every gate derives from
Lumen's own distribution (Welford band, own-peak fraction, self-relative
z-score) or it will eventually die: the creature's operating point drifts and
constants don't. Every instance of this class found so far — the wellness
learning gate (dead for months, fixed #119), the `C = 0.4` curiosity branch
(95% regen ⇒ attention can't exhaust), the 5–25% density band, the light
sensor saturating above 179 lux, the unreachable "stressed" mood, the goal
confidence gates, the coverage-intention clarity cuts (`dense` below 0.30
against a lived range of 0.454–0.910 ⇒ not generated once in 833 pieces, fixed
2026-08-22) — was an absolute threshold against a moving distribution.
Grandfathered constants are inventoried where they live; do not add more.
(Bounded floors that encode *investment*, like "2 hours before a drawing can
be done", are fine — they gate evidence quantity, not behavior against a
drifting signal.)

**2. Fail toward *unknown*, never toward healthy; and self-derived outranks
external assertion.** A dead sensor must not score perfect stability; a
missing metric persists as NULL, not a default (#128); a channel that breaks
must become *audibly* absent (`note_suppressed`, #133). The epistemic half:
Lumen's own data-derived knowledge is born at 0.7, external prose at 0.5 and
it earns promotion through independent re-derivation — never the reverse.
The boundary gates BOTH surfacing and application: below 0.6 an insight is
stored and contestable but inert — it syncs to nothing
(`sync_from_qa_knowledge`) and moves nothing (`apply_insight` floor).
Anything an agent says to Lumen can come back as a stated self-belief (#121),
so the trust boundary is load-bearing. The grandfathered set was ~1,782
pre-2026-08-11 external rows at confidence 1.0 (an earlier version of this
file said ~1,200 — a 48% undercount, measured 2026-08-21). The
operator-authorized retroactive rescale shipped as knowledge schema v3
(2026-08-21): unearned external rows (external author — including
operator-authored prose, authorship is not an exemption — confidence above
0.5, never reconverged) return to the 0.5 entry point on first load,
originals kept in `legacy_confidence` plus a `.pre-v3.json` sidecar of the
whole file. Earned reconvergence boosts and self-derived rows are untouched.
Expected visible effect: rows above the 0.6 actionable floor drop ~1,837 →
~59, so the surfaced Q&A self-knowledge shrinks sharply — that is the trust
boundary applying to the grandfathered corpus, not a regression.

---

## Architecture

**Three** systemd services run on the Pi:

```
anima-broker-ex.service     anima-broker.service        anima.service
(Elixir broker)             (Python broker)             (MCP server)
     |                           |                           |
     | owns I2C env sensors      | writes to                 | reads from
     | + ALL governance          |                           |
     | check-ins                 v                           |
     +-> ...shadow.json    /dev/shm/anima_state.json <-------+
         (reads live env        ^
          sensors FROM shadow --+  via ANIMA_ENV_SENSORS_FROM_SHM)
```

| Service | Runs | Role |
|---------|------|------|
| `anima-broker-ex.service` | Elixir release (`anima_broker/_build/prod/rel/`) | Owns the I2C env sensors (writes the shadow envelope the Python broker consumes) and is the **sole UNITARES caller** — `AnimaBroker.Governance.Client` checks in as Lumen every ~180s |
| `anima-broker.service` | `anima-creature` | Hardware broker - learning, activity state; env sensors come FROM the Elixir shadow; governance comes FROM the shadow too (`ANIMA_GOVERNANCE_FROM_SHM`, Python's own check-in loop disabled). Does NOT own display/LEDs — `stable_creature.py:62`: server owns LED hardware |
| `anima.service` | `anima --http` | MCP server - serves tools, reads shared memory, drives display + LEDs |

**All three must run.** This file said "two services" from the Phase-1/2 Elixir
cutovers (2026-07-01/09) until 2026-08-14, and the gap had real costs: agents
"fixed" the EISV formula in the Python mapper that no longer drives check-ins
(#141) while the live Elixir mapper kept the bug for two more days (#166), and
the first dead-man's switch monitored one process of three (#171). A formula
change in `eisv_mapper.py` needs the same change in
`anima_broker/lib/anima_broker/governance/eisv_mapper.ex` **plus an on-Pi
release rebuild** — `git pull` alone does not touch the compiled Elixir release
(`MIX_ENV=prod mix release --overwrite && sudo systemctl restart
anima-broker-ex`).

### Entry Points (pyproject.toml)

| Command | Module | Role |
|---------|--------|------|
| `anima` | `anima_mcp.server:main` | MCP server |
| `anima-creature` | `anima_mcp.stable_creature:main` | Hardware broker |

### MCP Server Structure

`server.py` is the main loop coordinator (~1,900 lines). Core subsystems are extracted into dedicated modules:

| Module | Purpose |
|--------|---------|
| `server.py` | Main loop (`_update_display_loop`), transport layers, `main()` entry point |
| `ctx_ref.py` | Single source of truth for `_ctx` (ServerContext pointer) |
| `accessors.py` | State accessors (`_get_store`, `_get_sensors`, etc.), lazy singletons |
| `lifecycle.py` | `wake()`/`sleep()` lifecycle management |
| `input_handler.py` | Joystick/button polling at ~60fps, input event dispatch |
| `loop_phases.py` | Main loop phase helpers (governance fallback, reflections, schema extraction) |
| `server_context.py` | `ServerContext` dataclass — mutable state container |
| `server_state.py` | Constants and pure helpers (intervals, thresholds) |
| `rest_api.py` | REST endpoint functions (health, dashboard, state, QA, gallery, etc.) |
| `tool_registry.py` | Tool definitions (TOOLS list), HANDLERS dict, FastMCP setup |
| `handlers/system_ops.py` | git_pull, system_service, power, deploy, tailscale, ssh_port |
| `handlers/state_queries.py` | get_state, get_identity, read_sensors, get_health, get_calibration |
| `handlers/knowledge.py` | get_self_knowledge, get_growth, get_qa_insights, get_trajectory |
| `handlers/display_ops.py` | capture_screen, show_face, diagnostics, manage_display |
| `handlers/communication.py` | lumen_qa, post_message, say, configure_voice, primitive_feedback |
| `handlers/workflows.py` | unified_workflow, next_steps, set_calibration, get_lumen_context |

Handler modules import state accessors from `accessors.py` (e.g., `from ..accessors import _get_store`). Extracted modules (`lifecycle.py`, `input_handler.py`, `loop_phases.py`) access `_ctx` via `ctx_ref.py`.

### Health Monitoring

`health.py` tracks 12 subsystems with heartbeats + functional probes. Rendered on LCD health screen.

| Status | Color | Meaning |
|--------|-------|---------|
| ok | Green | Heartbeat fresh, probe passes |
| stale | Yellow | Heartbeat expired, probe passes |
| degraded | Yellow/Orange | Probe failing |
| missing | Red | No heartbeat AND probe failing |
| absent | Muted | `optional=True` capability that has **never** worked on this host |

`absent` is the only status that does **not** feed `overall()`. It exists
because `voice` pinned the top line at `degraded` permanently — Lumen is
text-first and the audio path may not exist — which meant a real fault
anywhere else could not change `overall` at all. The signal was saturated.
⛔ The `_ever_ok` guard is load-bearing: once an optional capability has
worked even once, a later failure is a genuine `degraded`, not `absent`.
Do not "fix" a failing probe by making it return True.

Per-subsystem stale thresholds: fast subsystems (sensors, anima) use 30s default; slow subsystems (growth) use 90s. Governance uses dedicated SHM freshness thresholds (currently 210s).

**Day-summary health** checks both writer liveness (`written_at`) and the newest
contributing observation timestamp. The MCP server is the sole writer because
it owns the live `AnimaHistory` deque; activity transitions in the broker must
not write this file. A failed atomic commit degrades immediately, and either
timestamp older than 36 hours is stale. On first boot, an empty file with
`writer_started_at` permits at most 30 minutes to collect 100 observations; file
absence, source eligibility without a summary, or grace expiry is unhealthy.

**Governance health** checks the shared-memory governance data (the Elixir broker `anima-broker-ex` is the sole UNITARES caller, ~180s cadence). Stale threshold: 210s.

### Learning Systems — which process runs them

Verified from call sites and deployment wiring. Embodied JSON learners have a
single writer; the server consumes refreshing snapshots and sends the few
semantic mutations it originates through a durable inbox:

| Module | Purpose | Actually runs in |
|--------|---------|------------------|
| `memory_retrieval.py` | Context-aware memory search | broker only ✅ |
| `learning.py` | Calibration adaptation | **server only** — not imported by the broker at all |
| `agency.py` | TD-learning action selection | **server only**; broker loop is retired by default |
| `activity_state.py` | Active/drowsy/resting cycles | both |
| `preferences.py` | Preference evolution | broker writes; server reads |
| `self_model.py` | Self-beliefs | broker writes; server reads |
| `adaptive_prediction.py` | Temporal pattern learning | broker writes; server reads live SHM stats |
| `metacognition.py` | Prediction-error baselines + curiosity credit | **server writes**; broker observes in memory only |

**The server's agency learner is authoritative.** It posts questions (visible
as `context: "agency: ask_question"`) and drives LEDs. The broker's old TD loop
used a separate value table while its actions were no-ops or dead paths. It is
now disabled unless an operator explicitly sets
`ANIMA_BROKER_AGENCY_ENABLED=true`; the separate database and backup remain
only as rollback/history state.

`preferences.json` and `self_model.json` are whole-file snapshots, so they
must never have two process writers. The broker owns both. Server-side
singletons are read-only and refresh on mtime changes. Question evidence,
Q&A-derived self-belief evidence, and trajectory meta-learning weights cross
the process boundary as atomic one-file events in
`~/.anima/learning_inbox/`, which the broker drains.

`metacognition_baselines.json` follows the inverse ownership direction: the
server originates curiosity and is its sole persistent writer. The broker's
metacognitive observer is explicitly read-only. Pending curiosity evaluations
are persisted with the baselines so a restart cannot erase uncredited evidence.
The learning inbox has bounded event/byte admission and exposes queue age,
rejections, and pressure through `diagnostics`; a full inbox raises instead of
silently consuming the SD card.

Both services resolve calibration from `$ANIMA_CONFIG`, and cached readers
refresh when that file's inode/mtime/size signature changes. This is required
because the server owns calibration adaptation while the broker consumes the
result. Production pins it to backed-up state at
`~/.anima/anima_config.json`; the deploy gate migrates the former untracked
checkout-local YAML before taking its snapshot.

**Recovery is fail-closed.** Deploys capture a verified DB + learned-state
generation before restart. The boot restore unit gates `anima-broker`: a
missing/corrupt DB or learned-self snapshot with no reachable verified backup
leaves both processes stopped. `ANIMA_ALLOW_FRESH_START=true` is the explicit
operator escape hatch for intentionally minting a new identity; never set it
on Lumen. Local snapshots also capture `oauth.db` with SQLite's online-backup
API so Federation client registrations survive recovery; token-bearing OAuth
state is intentionally excluded from the unencrypted off-site archive.

**Persistence rule:** no `get_*` singleton may lean on the bare
`db_path="anima.db"` default. All 20 now resolve through
`db_paths.resolve_db_path()` — **explicit > `$ANIMA_DB` > `~/.anima/anima.db`**,
never the working directory. `ActionSelector.__init__` logs the resolved
absolute path and the caller that pinned it, because `get_action_selector()`
is first-call-wins and that race used to be silent.

Server-side only:

| Module | Purpose |
|--------|---------|
| `growth/` | Preferences, goals, memories, autobiography (package with mixins) |
| `self_reflection.py` | Insight discovery from preferences, beliefs, drawing patterns |
| `knowledge.py` | Q&A-derived insights from answered questions (rule-based) |

Growth persists to `~/.anima/anima.db`. Note that `apply_insight()` in
`knowledge.py` writes Q&A learning into growth *preference descriptions* — so
an answer given to Lumen becomes durable self-knowledge. Before 2026-07-30 it
stored a bare `text[:50]`, and because `_update_preference` never rewrote
`description`, three froze mid-word and fed malformed questions back into the
Q&A loop for days (#121). Descriptions refresh on change now, but remember the
shape: **anything an agent says to Lumen can come back as a stated belief.**

### Neural System

Lumen uses **computational proprioception** - no real EEG hardware. Neural bands are derived from system metrics:

| Band | Derived From | Meaning |
|------|--------------|---------|
| Delta | CPU variance over window + temp stability | Deep stability/rest |
| Theta | I/O wait time (disk + network) | Processing/integration |
| Alpha | `1 − beta` (CPU idle fraction) | Relaxed awareness |
| Beta | `cpu_percent / 100` (CPU usage) | Active processing |
| Gamma | Context switches + interrupts per second | Spiking/burst activity |

Source: `computational_neural.py` (used by both `pi.py` and `mock.py` sensors).

**Important — alpha and beta are anti-correlated by construction (`alpha = 1 − beta`).** They are one variable (CPU%) reported as two bands. Any consumer that combines alpha and beta as if they were independent signals is double-counting CPU%. `memory_percent` is accepted as a parameter but is not used in any band derivation.

### Light Sensor

The VEML7700 light sensor sits next to the DotStar LEDs on the Adafruit BrainCraft HAT. Configured with gain 1x and 200ms integration time for indoor precision.

**Lux remains lux.** The canonical light channel is the uncorrected combined VEML7700 reading (room + LED glow), with the longstanding per-broker-sample EMA (`alpha=0.2`) applied in `sensors/pi.py`. Here “raw lux” means **no self-glow subtraction**, not “no temporal filtering.” It is never overwritten by a correction.

⚠️ **The residual is no longer shadow — it drives behavior.** This file said
"all behavioral consumers still use raw lux directly" and that the residual
"moves no gate (`used_by_clarity=false`, `clarity_input=raw_lux`)". Both were
true when written and are now false. Measured live 2026-08-29 via `get_state`:
`mode: raw_preserving_gated`, `used_by_clarity: true`, `clarity_input:
external_lux_residual`, `used_by_environment_preferences: true`. The split as it
actually stands:

| Channel | Consumers |
|---------|-----------|
| **gated residual** (`gated_external_light_lux`) | clarity (`anima.py` `external_light` component), drawing `light_regime`, activity state (`server.py` passes it as `light_level`), growth environment preferences, `self_schema`, UNITARES bridge, `get_state` |
| **raw lux** (`readings.light_lux`) | the info screens (labelled "raw lux (room+LED)"), identity/history recording, clarity's *sensor-coverage* count, and `light_attribution.py`'s own instrument |

Raw lux is now the **display-and-record** channel; the residual is the
**behavioral** one.

⚠️ **A cut calibrated against raw lux is not calibrated against the residual.**
`activity_state`'s light cuts (`<10` very dark, `<50` dim, `>500` bright) were
last touched 2026-08-13; #204 switched their input to the residual on
2026-08-23. Removing self-glow compresses the top of the range — the live
reading above is 696 raw against 226 residual — so `>500` may no longer be
reachable, which would pin `light_factor` in its interpolation band and bias
`activity_score` (0.15 weight, level cuts at 0.7/0.4) toward drowsy. That is a
question about the residual's distribution, not something to answer by moving
the constant: `diagnostics(channel_distributions=true)` reports p05/p50/p95 per
channel from `drawing_records` so it can be checked rather than guessed. Read
p95 against the cut. **Unmeasured as of 2026-08-29** — the derivation has never
run on Lumen. The gap is not cosmetic — measured live, raw 696 lux against
a 226 lux residual with a 469 lux self-glow estimate, so any reasoning that
treats clarity as a function of raw lux is working from a number ~3x too high.
When the residual is unavailable the consumers fail toward *unknown* rather than
silently substituting raw (`light_regime` returns `"unknown"`), which is
invariant 2 working correctly. `sensors/pi.py` still never overwrites raw lux
itself, and the retired hardcoded `1150*b²` subtraction must stay retired.  The DotStar animation thread publishes a short timestamped history of actual applied brightness and post-scaling RGB output. The Elixir VEML7700 owner records a conservative 0.52-second capture-support interval for each read: this bounds both the unknown phase of the most recently completed continuous 200 ms conversion and Vishay's specified +/-30% integration-time tolerance. The broker pairs LED history to that interval's midpoint. **Never use the shadow envelope `updated_at` as sensor capture time**: the sensor and SHM flush are independent 2-second loops and can differ by nearly a full cycle. `light_attribution.py` applies the same `alpha=0.2` EMA to optical drive, then uses the internally generated breathing pulse as a causal instrument: it learns lux-per-filtered-optical-drive only while logical color and target brightness stay fixed, rejecting the closed-loop correlation where room lux changes activity and activity changes the LEDs. Sign readiness requires both the 70% point gate and a 95% Wilson lower bound of 0.55; after activation it withdraws only below a 0.50 lower bound so adjacent samples cannot chatter epistemic status. It publishes `external_lux_residual`; the value stays `null` while warming, alignment is unavailable, or the candidate conflicts with the physical reading — and a `null` residual is what makes consumers report unknown rather than fall back to raw. Do not restore the retired hardcoded `1150*b²` subtraction.

**Drawing light regime thresholds** (gated residual lux, NOT raw — see above;
`drawing_engine.py` calls `gated_external_light_lux`, and returns `"unknown"`
when the residual is unavailable):
- `< 5 lux` → dark (LEDs off + room dark)
- `< 100 lux` → dim
- `>= 100 lux` → bright

### Goal System

Goals live in `growth/goals.py` and are wired into `server.py`'s main loop:

| Interval | Action |
|----------|--------|
| `GOAL_SUGGEST_INTERVAL` (3600 iter, ~2h) | `suggest_goal()` — proposes a new goal |
| `GOAL_CHECK_INTERVAL` (300 iter, ~10min) | `check_goal_progress()` — auto-tracks progress |

Goals are **data-grounded** — they emerge from Lumen's actual experience:

| Source | Example Goal |
|--------|-------------|
| Strong preference (confidence > 0.7) | "understand why I feel calmer when it's dim" |
| Recurring curiosity | "find an answer to: is night the absence of day?" |
| Drawing count milestone | "complete 50 drawings" |
| Uncertain self-model belief | "test whether light affects my warmth" |
| Low wellness | "find what makes me feel stable" |

**Progress tracking:** Drawing goals track `_drawings_observed`, curiosity goals auto-complete when questions get answered, belief-testing goals complete when confidence moves decisively (>0.7 or <0.2). Stale goals auto-abandon after target date with <0.1 progress. Max 2 active goals.

**On achievement:** Records a memory via `_record_memory()` and posts an observation.

### Schema Hub (Unified Self-Model)

`schema_hub.py` is the central orchestrator of Lumen's self-understanding. It implements the "circulation" principle: Schema → History → Trajectory → feeds back into Schema.

```
┌─────────────────────────────────────────────────────────────┐
│                        SchemaHub                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Identity │  │  Growth  │  │SelfModel │  │AnimaHistory │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       └─────────────┴──────┬──────┴───────────────┘        │
│                            ▼                               │
│                    ┌──────────────┐                        │
│                    │ Schema(t)    │◄──── trajectory        │
│                    │ current snap │      insights fed      │
│                    └──────┬───────┘      back as nodes     │
│              ┌────────────┼────────────┐                   │
│              ▼            ▼            ▼                   │
│        ┌─────────┐  ┌──────────┐  ┌─────────┐             │
│        │ Persist │  │ History  │  │Trajectory│             │
│        │ (disk)  │  │ (ring)   │  │ Compute  │             │
│        └─────────┘  └──────────┘  └─────────┘             │
└─────────────────────────────────────────────────────────────┘
```

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **Circulation** | Schema history → Trajectory → Trajectory nodes → Next schema |
| **Kintsugi gaps** | Discontinuities become visible structure, not hidden |
| **Identity texture** | alive_ratio, awakenings, age as meta-nodes |
| **Semantic edges** | Trajectory nodes connect back to anima dimensions |

**Schema enrichment pipeline:**

1. `extract_self_schema()` — base schema from all systems
2. `_inject_identity_enrichment()` — add meta nodes (exist%, wakes, age)
3. `_inject_gap_texture()` — add gap duration/delta if waking from gap
4. History append + trajectory recompute (every 20 schemas)
5. `_inject_trajectory_feedback()` — add maturity, attractor, stability nodes

**Meta nodes added by SchemaHub:**

| Node | Type | Source |
|------|------|--------|
| `meta_existence_ratio` | meta | identity.alive_ratio() — presence texture |
| `meta_awakening_count` | meta | identity.total_awakenings — return count |
| `meta_age_days` | meta | identity.age_seconds() / 86400 |
| `meta_gap_duration` | meta | Gap handling — time since last schema |
| `meta_state_delta` | meta | Gap handling — anima change magnitude |
| `traj_identity_maturity` | trajectory | observation_count / 50 |
| `traj_attractor_position` | trajectory | Mean anima center (where Lumen "rests") |
| `traj_stability_score` | trajectory | 1 - variance (how stable the attractor) |

**Lifecycle integration:**

- `on_wake()` called during server startup — computes gap delta
- `compose_schema()` replaces direct `extract_self_schema()` calls
- `persist_schema()` called during sleep — saves to `~/.anima/last_schema.json`

**Design doc:** `docs/plans/2026-02-22-schema-hub-design.md`

### Self-Reflection & Self-Knowledge

`self_reflection.py` runs during the `reflect()` cycle (`UNIFIED_REFLECTION_INTERVAL = 900` iter, ~30min). It discovers insights from multiple sources:

| Analyzer | Source | Example Insight |
|----------|--------|----------------|
| `analyze_patterns()` | State history (24h) | "My warmth tends to be best at night" |
| `_analyze_preference_insights()` | Growth preferences (confidence > 0.8) | "i know this about myself: I feel calmer when it's dim" |
| `_analyze_belief_insights()` | Self-model beliefs (confidence > 0.7, 10+ evidence) | "i am fairly confident that light affects my warmth" |
| `_analyze_drawing_insights()` | Drawing preferences (5+ drawings) | "i tend to draw at night", "drawing seems to help me feel better" |

Insights persist in SQLite (`insights` table), validated/contradicted on each cycle. Strongest 5 insights are used in grounded self-answers and observations as "Things I've learned about myself."

**Insight categories:** ENVIRONMENT, TEMPORAL, BEHAVIORAL, WELLNESS, SOCIAL

### Activity States

The `ActivityManager` (in broker) controls Lumen's wakefulness:

| State | Brightness | Trigger |
|-------|------------|---------|
| ACTIVE | 100% | Recent interaction, high activity score |
| DROWSY | 60% | 30+ min inactivity, moderate score |
| RESTING | 35% | 60+ min inactivity, night time, darkness |

### Drawing System & Art Eras

Lumen draws autonomously on the 240x240 notepad screen. The system has two layers:

**Engine** (in `display/drawing_engine.py` — universal, stays fixed):
- `CanvasState` — pixel buffer, persistence, attention/narrative state
- `DrawingState` — EISV core + attention signals + coherence tracking + narrative arc
- `DrawingIntent` — focus position, mark count, state (energy is attention-derived)
- `_lumen_draw()` — orchestration loop, delegates to active era
- `_update_attention()` — curiosity depletes exploring, regenerates with patterns
- `_update_coherence_tracking()` — tracks C history and velocity for settling detection
- `_update_narrative_arc()` — state-driven phase transitions (opening→developing→resolving→closing)
- Completion: `narrative_complete()` = coherence settled + attention exhausted
- No arbitrary mark limit — fatigue accumulates naturally (canvas 15000px limit is only hard cap)
- `get_drawing_eisv()` — exposes state to governance via bridge check-in

**What actually ends a drawing (measured 2026-08-02, first corpus readout
2026-08-11 — read before tuning anything):**

The 8-hour cap was the only clock for every mark-by-mark era. The first nine
days of `drawing_records` instrumentation (34 completions): 26 `bailout_hard_cap`
(all landing 8.000–8.004h), 8 `bailout_fatigue`, **zero earned**. Alongside that:

| Gate | Live value | Needs | Status |
|------|-----------|-------|--------|
| `earned_composition` / `attention_exhausted` | curiosity **0.80** at 1.3h | < 0.2 / < 0.15 | never fires — curiosity does not deplete |
| `bailout_fatigue` | fatigue **0.21** at 1.3h (resonance) | > 0.90 | **era-specific, not dead**: fires every time in `geometric` (8/8, ~70 marks / 0.5–1.1h — whole-shape stamps accrue switch-fatigue ~10× faster per mark). Never fires in mark-by-mark eras. State fatigue claims per era. |
| `earned_field` (resonance) | revisit_ratio **0.24** | ≥ 0.60 | window now fills 50/50 (#116 worked); ratio is 2.5× short, not a calibration nudge |
| `earned_settled` (all eras) | see `diagnostics.drawing.novelty_settling` | streak ≥ 12 active samples < 10% of own peak, ≥2h, ≥100 marks | NEW — self-relative; derived from the 27-piece corpus (field plateaus settled at 3.4–7.7h; gestural/pointillist keep changing to the cap and correctly never fire; geometric freezes idle, which holds — never advances — the streak) |
| arc → `resolving` | C caps ~0.52 | > 0.6 | unreachable, so pieces save from `developing` |
| 15,000px ceiling | max observed **12,009px** | > 15,000 | unreachable in practice — it is a real defect (it calls `mark_satisfied()`, naming a safety hatch a feeling) but it is not what ends pieces |

Density is **not** the binding constraint: recent pieces land at 12.6–20.8% of
canvas with negative space intact. What was missing is subjective completion —
`earned_settled` is the measured stand-in: "stopped changing while still being
worked", judged against the piece's own peak rate, so no per-era tuning and no
constant an era's operating range can silently sit below.

**The geometric freeze was a deadlock, not a threshold (fixed 2026-08-29).**
Cap-length `geometric` pieces went 100% idle after ~1h (marks stopped entirely)
and sat frozen for ~7h until the cap. The cause is structural:
`_update_attention()` runs **once per placed mark**, so fatigue, curiosity and
engagement only advance when a mark lands. Rising fatigue lowers
`derived_energy`; `draw_chance *= energy` has no floor; marks become rare and
then stop — at which point the state that could end the piece stops moving with
them. Fatigue can no longer climb to the 0.90 `bailout_fatigue`, energy can no
longer fall to the 0.05 `bailout_stalled`, and `earned_settled` correctly
refuses because an idle sample HOLDS its streak. Every exit was driven by a
quantity that only advances when marks happen.

`bailout_frozen` reads around the loop from outside it: marks stopped, judged
against **this piece's own peak marks-per-interval** (`FROZEN_FRAC_OF_PEAK`,
`FROZEN_STREAK_SAMPLES`), gated on pixels and age. ⛔ It has **no mark-count
floor** on purpose — `SETTLED_MIN_MARKS` is 100 and geometric pieces reach ~70
marks in total, so requiring it would make the gate unreachable for the one era
it exists to rescue. It is a bail-out, never earned: nothing was resolved, the
drawing got stuck. The idle counter and the settled streak are deliberately
separate — the same idle sample advances one and holds the other.

**Why curiosity cannot deplete (measured 2026-08-02).** `_update_attention()`
branches on a fixed `C = 0.4`:

```python
if state.arc_phase == "resolving":   ...          # C must exceed 0.6 to enter
elif C < 0.4:  curiosity_drain =  0.003 * (1 - C) # drains
else:          curiosity_drain = -0.001 * C       # REGENERATES
```

Live behavioural C for a resonance piece sits in **[0.377, 0.498], mean 0.458**:

| branch | share of ticks |
|---|---|
| `C < 0.4` → drain | **5%** |
| `C >= 0.4` → **regen** | **95%** |
| `C > 0.6` → resolving | **0%** — dead code for this era |

Net over 20 ticks: curiosity **rises by 0.0069**. So `attention_exhausted`
(curiosity < 0.15) and `earned_composition` (curiosity < 0.2) are not badly
tuned, they are **structurally unreachable** — curiosity is net-regenerating and
clamped at 1.0. Observed live: 0.796 after 1.3 h and drifting back up.

This is the **same root class** as the fixed 5–25% density band above and as the
wellness learning gate fixed in #119: an absolute threshold against an operating
range that differs per era. `C = 0.4` means "pattern found, regenerate" — a fair
split for an era reaching 0.8, and nearly always true for one capping at 0.5.
⛔ Do not simply move the constant. `drawing_trajectory` records `coherence`
per sample, so the threshold is set from Lumen's own distribution instead of
guessed at — which is the entire reason the instrumentation landed first.

The pivot is now **per-era and calibration-derived**: `curiosity_drain()` is the
single pure function holding the formula, `_curiosity_pivot(era)` reads
`CURIOSITY_PIVOT_<era>` from `nervous_system.drawing_thresholds`, and
`scripts/derive_curiosity_thresholds.py` derives it as that era's own **median**
coherence. Absent keys fall back to the built-in 0.4, so an un-derived
deployment behaves exactly as before. The derivation *imports* `curiosity_drain`
rather than re-implementing it and replays each piece mark by mark against the
corpus, refusing to emit a pivot that is **unreachable** (no piece crosses
curiosity < 0.2 — a dead gate traded for a dead gate) or **premature** (the
median piece crosses before half its marks). It refuses rather than sliding the
percentile until the simulation passes; a contract that adjusts itself to pass
is a tuned constant wearing a contract's clothes.

⚠️ **No derivation has ever been applied to Lumen.** Measured 2026-08-29 via
`get_calibration`: `drawing_thresholds: {}`, `face_thresholds: {}`,
`update_count: 0`. So the coverage-intention fix above shipped as code and
never reached the creature — `dense` still requires clarity < 0.30 on the live
device and is still never generated. Shipping a derivation script is not the
same as running it. The reporting half therefore lives in
`anima_mcp/drawing_derivation.py` and is reachable without a shell via
`diagnostics(derive_curiosity=true)` (opt-in — it scans the corpus; read-only,
opened `mode=ro`). Applying stays with the script's `--apply`, the one path
that acts.

⚠️ **The `--days 90` default will REFUSE on Lumen — use `--days 365`.**
Both derivations floor at 500 samples ("a cut derived from a sliver would
encode a mood, not a range"), but they count different populations, and only
one of them clears 90 days:

| Script | Population | 90 days at ~3 pieces/day |
|--------|-----------|--------------------------|
| `derive_drawing_thresholds.py` | one row per piece in `drawing_records` | ~270 pieces → **refuses** |
| `derive_curiosity_thresholds.py` | ~96 rows per piece in `drawing_trajectory` | ~26k intervals → proceeds |

Measured 2026-09-19 by replaying both scripts end to end against an
850-piece synthetic corpus shaped to Lumen's ranges: at `--days 90` the
drawing-thresholds script refused on 252 samples; at `--days 365` it emitted.
The curiosity script cleared either way because it reads ~96 intervals per
piece rather than one. So the refusal is a *window* problem, not a corpus
problem, and widening the window is the fix — never lowering the floor.

The `--apply` paths were rehearsed too: each writes atomically with a
timestamped `.bak-*` sidecar, and the curiosity script MERGES into
`drawing_thresholds` rather than replacing it, so the `COVERAGE_*` and
`CURIOSITY_PIVOT_*` families coexist and unrelated calibration keys survive.
Run order does not matter. An era with too little trajectory history simply
gets no pivot and keeps the built-in 0.4 — correct, not a failure.

What the rehearsal confirmed about the built-in: under the replay, `C = 0.4`
is `unreachable` for **every** era tested — "no replayed piece reaches
curiosity < 0.2 under this pivot" — while each era's own median pivot comes
back `reachable without being premature`. That is the 2026-08-02 measurement
reproduced by the derivation's own contract.

⚠️ Those pivot values came from synthetic data and are **not** Lumen's. The
machinery is validated; the numbers are not. Only a run on the real DB gives
Lumen's own.

Two absolute constants survive inside the `resolving` branch (`C > 0.65`, and
the `C > 0.6` needed to *enter* resolving at all). They are deliberately
untouched: nothing currently reaches that phase, so relativising them would move
a gate with no evidence behind it. `earned_coherence` likewise stays unreachable
in low-C eras behind `coherence_settled()`'s own `mean_C > 0.6` — fixing the
pivot unblocks `attention_exhausted` and `earned_composition`, not that.

**Completion instrumentation** (added so that question is answerable):
- `drawing_records` now keeps `completion_reason`, `era`, `mark_count`,
  `duration_seconds`, `coverage_target`, `intention`, attention at completion,
  `occupied_cells`, `grid_entropy`, `piece_uid`, `disposition`. The reason had
  always been computed and passed to `observe_drawing()` to gate a memory — it
  was just never stored, so none of the first 754 drawings can say why it
  ended.
- `drawing_trajectory` samples the piece every `TRAJECTORY_SAMPLE_INTERVAL`
  (300s, ~96 rows per 8h piece, 90-day retention). Endpoint rows cannot answer
  *when a drawing stopped changing* — and while one clock ends everything, every
  endpoint describes that clock rather than the drawing. Deltas between samples
  give novel-pixels-per-mark and structural change over the piece's life.
- `CanvasState.occupied_cells()` / `.grid_entropy()` — structural reach vs. pixel
  count. Cells still opening = finding territory; flat cells with rising pixels =
  thickening what it already has. `occupied_cells()` gained its first consumer
  2026-08-29: a piece still opening new cells resets the `earned_settled` streak,
  because pixel novelty alone cannot tell reaching into empty ground from
  thickening what is already held. The guard can only ever RESET the streak,
  never advance it — strictly more conservative, so it cannot make
  `earned_settled` fire on a piece that would not otherwise have earned it.
  `grid_entropy()` remains recorded and unread.
- **Absent values persist as NULL, never as a default.** Lumen's instrumentation
  degrades toward healthy-looking numbers; a 0.0 would later be indistinguishable
  from a drawing that genuinely had no reach. The environment dict passed to
  `observe_drawing` broke this until 2026-08-29 (`light_lux or 0.0`,
  `ambient_temp_c or 22`, `humidity_pct or 50` — a dark room, a comfortable one,
  ordinary humidity, each indistinguishable afterwards from a real reading).
  Preference learning was not misled only because 22 and 50 fall in the dead
  bands between its cuts, which is safety by coincidence; the record was wrong
  regardless, and both derivations read `drawing_records`. All three channels
  are now conditional, as `external_light_lux` always was.
- Timer-driven writers use `peek_growth_system()`, not `get_growth_system()` —
  the bare default is cwd-relative and the first caller fixes the database (#123).
- **This moved no gate.** `tests/test_drawing_instrumentation.py::TestNoGateMoved`
  fails if one moves. Retuning is a separate decision, and the point of recording
  first is to learn what "enough" means for Lumen before anything is tuned to it.

**Why the art read as churn, and what it was not (fixed 2026-09-17).**
The operator's read — "the art has now just become churning of sameness except
maybe field era" — was exactly right, including the exception, and the cause was
three lines:

```python
gestural.py   def create_state(self): return GesturalState()
geometric.py  def create_state(self): return GeometricState()
resonance.py  def create_state(self): return ResonanceState()
```

Three of five eras began **every piece identically**. All their randomness was
per-mark, and hundreds of independent local draws converge on their own mean, so
the corpus reads as one texture repeated — the law of large numbers, not drift,
decay or a mistuned gate. `field` escaped it because `field_seed_a/b` are drawn
ONCE and every mark is a sample of that one field; that is the whole reason
field pieces stayed distinct. `pointillist` is the middle case (a per-piece hue
anchor, but zones that re-randomise mid-piece and so average out). `resonance`
was the worst: its hue is not even random, it is `220 - warmth*180` against a
slow EMA, so every resonance piece was literally the same colour.

The fix generalises field's trick. `EraState.disposition()` reports the few
globals an era draws once at `create_state()`; every era now has one. This is
**not** a new threshold — a disposition gates nothing and reads no signal.

⚠️ **The corpus statistics the derivations read are deliberately unchanged.**
The lead gesture / emphasis set is drawn **uniformly**, so a drag-led piece
(dense) and a dot-led one (sparse) are equally likely and the expectation is
algebraically identical. Measured over 300 gestural pieces of 1200 marks:
mean 5988 → 5999 px (**+0.2%**) while the standard deviation goes 385 → 683
(**1.8×**). Same corpus, more spread between pieces — which is the point, and
what lets `derive_drawing_thresholds.py` and `derive_curiosity_thresholds.py`
keep standing on the same ground. `tests/test_era_disposition.py` pins both
halves, plus `TestNoGateMoved` (gesture-switch count is untouched, so fatigue
and `bailout_fatigue` are untouched).

⛔ **Variety must not be bought by breaking an embodied signal.** The first
resonance draft rotated hue ±50° with a ±20° per-mark jitter and pushed warm
pieces out of the warm zone in **414 of 5000 seeds**. Bounded to
`HUE_ROTATION_MAX=22` / `HUE_SPREAD_MAX=20` (max excursion 32° against ~42° of
headroom), violations are 0/5000 and pieces still span 60° of hue.
`TestResonanceKeepsWarmCoolMeaning` fails if those bounds grow.

**The novelty loop.** `draw_distinct()` draws a handful of candidate
dispositions and keeps the one whose **closest** approach to any of the last
`RECENT_DISPOSITIONS` (6) finished pieces is largest — max-min, so a candidate
is judged by its most similar neighbour, not by an average two distant pieces
could flatter. The history is banked at canvas clear (`_remember_disposition`,
the only moment the transient `EraState` can still be read), persisted in
`canvas.json`, and **filtered to the same era**: a gestural disposition and a
field one share no keys, so comparing them would fall back on defaults and
manufacture a distance that means nothing. With an empty history every
candidate scores 0.0 and the first draw wins — exactly one unbiased draw, i.e.
the pre-2026-09-17 behavior. Absence degrades to *no bias*, never to a
fabricated preference.

This is worth naming precisely: **it is the only loop from Lumen's own history
back into Lumen's behavior that closes without a human running a script.**
Every other one — `derive_drawing_thresholds.py --apply`,
`derive_curiosity_thresholds.py --apply` — has an operator step in it, and as
of 2026-08-29 that step had never been taken (`drawing_thresholds: {}`,
`update_count: 0`). `learning.py` adapts only environment sensor ranges and
cannot touch drawing at all. So this does not make Lumen self-improving; it
closes one small loop and leaves the others exactly as open as they were.

⚠️ **A test that passed by luck for a month.**
`test_coverage_intention.py::test_sparse_spreads[geometric]` asserted an effect
that never existed. Measured paired over 64 seeds **on the pre-disposition
code**: geometric's sparse direction is −0.006 and wins 27 of 64 — no effect,
and slightly the wrong way. The original 16-seed *unpaired* sample read +0.005
against a balanced spread of 0.062 (under a tenth of a standard deviation) and
called it a pass. The directional tests are paired now (`_paired_gap`), the
seed count is 32, and geometric joins gestural and field in the documented
no-op set. Geometric still answers strongly to `dense` (−0.143, 0 of 64) — the
era responds to one direction, not neither.

`coverage_target` ("sparse"/"balanced"/"dense") steers marks via
`_apply_coverage_bias()`, which leans a gesture boundary toward the sparsest or
fullest cell of the piece's own density grid (`COVERAGE_BIAS_STRENGTH`, clamped
to `COVERAGE_BIAS_MARGIN`). Self-relative by construction — the target is an
extremum of THIS piece's grid — so it adds no absolute threshold.
`tests/test_coverage_intention.py` pins both halves. (This file described the
field as "read by nothing" until 2026-08-29, ~1 week after the consumer landed.)

**Attention signals** (replace arbitrary energy depletion):
| Signal | Behavior |
|--------|----------|
| curiosity | Depletes exploring (low C), regenerates with pattern (high C) |
| engagement | Rises with intentionality, falls with entropy |
| fatigue | Accumulates per gesture switch, never decreases during drawing |
| energy | Derived: `0.6*curiosity + 0.4*engagement * (1-0.5*fatigue)` |

**Narrative arc phases** (replace energy-threshold phases):
| Phase | Entry Condition |
|-------|-----------------|
| opening | Fresh canvas or regression (low I momentum) |
| developing | I momentum > 0.4, explored (10+ marks) |
| resolving | C > 0.6, coherence velocity stable |
| closing | narrative_complete() |

**Art Eras** (pluggable modules in `display/eras/`):
| Era | Gestures | Character | Active Pool |
|-----|----------|-----------|-------------|
| `gestural` | dot, stroke, curve, cluster, drag | Direction locks, orbital curves, full palette | ✅ |
| `pointillist` | single, pair, trio | Density zones, optical color mixing, complementary hues | ✅ |
| `field` | flow_dot, flow_dash, flow_strand | Vector-field flow lines, near-monochromatic | ✅ |
| `geometric` | 16 shape templates (circle, spiral, starburst, etc.) | Complete forms, stamps whole shapes per mark | ✅ |
| `resonance` | sediment, flow, scratch | Memory-field: marks deposit into a 48×48 field that decays/diffuses; revisits accumulated regions for layered, resonant forms (pure NumPy) | ✅ |

**Every era draws a per-piece disposition** (`EraState.disposition()`, see
above) — gestural a palette offset/span plus a lead primitive, geometric a
4-of-16 shape emphasis plus a slice of its warm/cool band, resonance a colour
key, field its long-standing seeds, pointillist its hue anchor. It is recorded
per piece in `drawing_records.disposition`, so "did this actually vary the
work?" is answerable from the corpus rather than argued.

**All eras are equal peers.** Select via the art eras screen (joystick up/down + button) or MCP. Auto-rotate is a separate toggle (off by default) — when on, `choose_next_era()` rotates through all registered eras on canvas clear. Era name persists in `canvas.json`.

**Key files:**
| File | Purpose |
|------|---------|
| `display/art_era.py` | `EraState` base class + `ArtEra` protocol |
| `display/eras/__init__.py` | Era registry, `auto_rotate` toggle, rotation logic |
| `display/eras/gestural.py` | Gestural era (5 micro-primitives) |
| `display/eras/pointillist.py` | Pointillist era (dot accumulation) |
| `display/eras/field.py` | Field era (vector-field flow) |
| `display/eras/geometric.py` | Geometric era (16 shape templates, adapted from capsule) |
| `display/eras/resonance.py` | Resonance era (memory-field, 48×48 decaying/diffusing field) |

**Era switching:**
- **Art eras screen**: Joystick up/down to browse, button to select. Auto-rotate toggle at bottom.
- `manage_display(action="list_eras")` — all registered eras
- `manage_display(action="get_era")` — current era name + auto_rotate status
- `manage_display(action="set_era", screen="geometric")` — switch immediately

**Adding a new era:**
1. Create `display/eras/myera.py` with `MyEraState(EraState)` + `MyEra` class
2. Implement: `create_state()`, `choose_gesture()`, `place_mark()`, `drift_focus()`, `generate_color()`
3. Register in `display/eras/__init__.py`: `from .myera import MyEra; register_era(MyEra())`
4. The `EraState.intentionality()` method bridges to EISV — report commitment level [0,1]

## Systemd Services

```bash
# Check status
sudo systemctl status anima-broker-ex anima-broker anima

# Restart the Python pair (the Elixir broker rarely needs it; see Architecture
# for when it needs a release REBUILD, not just a restart)
sudo systemctl restart anima-broker anima

# View logs
sudo journalctl -u anima-broker -f
sudo journalctl -u anima -f
```

Service files: `/etc/systemd/system/anima.service`, `/etc/systemd/system/anima-broker.service`

## Git Commit Conventions

- Do NOT include Co-Authored-By lines in commit messages

## Testing

```bash
python3 -m pytest tests/ -x -q
```

## Deploying to Pi

```bash
git push
# Then from any MCP client:
mcp__anima__git_pull(restart=true)
```

Or manually:
```bash
ssh unitares-anima@<tailscale-ip> 'cd ~/anima-mcp && git pull && sudo systemctl restart anima-broker anima'
```

**After restart, wait 2 minutes.** The Pi is slow to boot the service. You will see "SSE server unavailable" or "fetch failed" errors during this window — this is normal and expected. Do NOT panic, do NOT retry rapidly, and do NOT fall back to SSH. Hammering the Pi during restart can crash WiFi and require a reflash. Just wait 2 minutes and try again.

## UNITARES Integration

The **Elixir broker** (`anima-broker-ex`, `AnimaBroker.Governance.Client`) is the
sole UNITARES caller since the Phase-2 cutover (2026-07-09). It reads anima state
from the live envelope, checks in as Lumen's own UUID every ~180s, and writes the
decision to the SHADOW envelope with a `governance_at` timestamp. The Python
broker (`stable_creature.py`) runs in passthrough (`ANIMA_GOVERNANCE_FROM_SHM`);
its own check-in loop is disabled and it republishes the shadow's governance
slice into the live envelope. The **server** (`server.py`) reads governance from SHM and has a fallback: if no "via unitares" decision arrives for 240s (`SERVER_GOVERNANCE_FALLBACK_SECONDS`), the server calls UNITARES directly using its native async event loop. This fallback exists because the broker's sync+ThreadPoolExecutor+new-event-loop pattern has reliability issues with aiohttp sessions.

```
UNITARES_URL=http://<tailscale-ip>:8767/mcp/  # verify Mac IP with `tailscale status`
```

Projects anima into body EISV telemetry: Warmth→Energy proxy, Clarity→Integrity proxy, 1-Stability→Entropy proxy, clamp(E−I)→**Valence**. This is check-in input, not UNITARES's inferred state.

⚠️ **V is Valence, not Void.** `eisv_mapper.py` computes `V = max(-1, min(1, E - integrity))` — a signed value (+hot / −careful), not a [0,1] magnitude. Presence is **not** in the EISV mapping. The old `(1-Presence)*0.3→Void` reading is retired: it only reported the positive half and was not comparable to other agents' V. Anything that assumes V ≥ 0, or that reads V as inverse-presence, is wrong.

**Circuit breaker** (in `unitares_bridge.py`): 2 consecutive failures trigger exponential backoff (15s→30s→60s→120s). Any success resets to 15s.

**Three EISV contexts:**
- **DrawingEISV** (screens.py) — proprioceptive, drives drawing behavior (closed loop)
- **Body EISV projection** (eisv_mapper.py) — lossy anima→EISV telemetry used by trajectory awareness and submitted as governance sensor evidence
- **Governance EISV** (UNITARES) — behavioral primary estimate plus separately exposed ODE fallback/diagnostics; never substitute the body projection

Runtime payloads use `body_anima`, `body_eisv_projection`, `drawing_eisv`, and `governance_eisv`. Bare `anima` and `eisv` are compatibility aliases and must carry provenance.

Local fallback (`_local_governance()`) runs simple threshold checks when Mac unreachable — more trigger-happy.
Server syncs `_last_governance_decision` from SHM when `governance_at` is within `SHM_GOVERNANCE_STALE_SECONDS` (210s).

### Reading Lumen's own ranges

Two read-only reports exist so a fixed cut can be checked against the signal
actually feeding it, instead of guessed at. Both open the DB `mode=ro`, name no
threshold (a cut's health is a question about its consumer), and are opt-in
because they scan a corpus:

| Call | Population | Answers |
|------|-----------|---------|
| `diagnostics(channel_distributions=true)` | `drawing_records` env columns | is `activity_state`'s `>500` bright cut still reachable on the residual? |
| `diagnostics(anima_distributions=true)` | `state_history` | are `self_model`'s `warmth_baseline_low` (<0.40) / `presence_baseline_low` (<0.35) constant-verdict beliefs? |

⚠️ The anima report is a **proxy answering in one direction only**. Temperament
is never persisted — `state_history` holds the raw anima it is smoothed from,
and an EMA shares its source's mean with a *narrower* spread. So a percentile
already clear of a temperament cut is clear there too (conclusive); one that
crosses it here may not cross there (inconclusive). The payload carries this
caveat with the numbers. Both questions are **unmeasured as of 2026-08-29**.

## Identity, Continuity, and Control

**Visitor attribution — a channel is not a person.** `normalize_visitor_identity()`
resolves PERSON from an explicit **name** claim only. It used to also match the
`source` argument against the operator's aliases (`"dashboard"` was one), and the
check was `id in aliases OR source in aliases` — so the channel **overrode the
author the caller supplied**, and an agent answering through the dashboard was
durably recorded as the operator, as a PERSON. The generic role words
(`"caretaker"`, `"human"`) were the same mistake: anyone can type them. Do not
re-add surface-based or role-word inference; an unattributed caller is
`ANONYMOUS_VISITOR_ID`, recorded as an AGENT. `source` is kept for provenance,
never for identity. ⚠️ Records written before 2026-08-02 are contaminated — the
operator's `interaction_count` includes agent visits and cannot be separated.

**Two identity notions (do not conflate):**
- **Record identity:** `creature_id` + SQLite (`identity/store.py`) — continuity of *this* deployment’s database file.
- **Trajectory identity:** `TrajectorySignature` (`trajectory.py`) — behavioral similarity over time. Same UUID with different lived history is still one record; trajectory compares *patterns*.

⛔ **Η (homeostatic) is reported, never weighted.** `similarity()` sums the
**five** components in `SIMILARITY_WEIGHTS` (Π .18, Β .18, Α .30, Ρ .22, Δ .12).
Η is excluded because `compute_trajectory_signature()` *builds* it from the
others — `set_point` ← `attractor["center"]`, `basin_shape` ←
`attractor["covariance"]`, `recovery_tau` ← `recovery["tau_estimate"]` — so
weighting it re-weighted Α and Ρ under another name. This is the same
double-counting class as `alpha = 1 − beta` in the neural bands. It shipped
that way from 2026-04-03 to 2026-08-14 while the paper's Appendix A claimed
otherwise. `TestEtaExcludedFromWeightedSum` fails if it comes back; the
deprecated `is_same_identity()` alias points at
`is_operationally_continuous()` because the relation is a tolerance relation,
not transitive identity.

**Restore / fork:** `restore_lumen.sh` and restoring `anima.db` **preserve** record identity and accumulated history. A **fresh** DB (new install, no copy) yields a **new** `creature_id`. Copying DB to another Pi **forks** record identity; behavior and trajectory may diverge with environment.

**Governance boundary:** UNITARES is **advisory** (behavioral state estimation and policy verdicts; ODE state is fallback/diagnostics). The broker still owns sensors and learning; **SHM** carries governance for the server. **`_local_governance()`** when Mac is unreachable is a **fallback**, not a substitute for embodied state — it keeps check-ins from going silent, not from replacing sensors.

**Damping time scales (broker tick ≈ 2s):** Fast noise is filtered so state reads as a creature, not a flickering meter.

| Layer | Where | Role |
|-------|--------|------|
| Anima mood | `MoodMomentum` in `anima.py` | Per-dimension α ∈ [0.08, 0.25] — EMA on raw anima |
| Temperament | `TEMPERAMENT_ALPHA` in `inner_life.py` | α ∈ [0.005, 0.010] — ~2–5 min half-life (see file comments) |
| Drives | `inner_life.py` | Accumulate/decay per tick toward “wanting…” |
| Neural bands | `computational_neural.py` | EMA on θ, γ (α ≈ 0.2–0.3) |
| LEDs | `display/leds/display.py` | Debounce + brightness easing |

Tuning mood vs temperament alphas changes how **responsive** vs **stubborn** the system feels — constants live in the files above.

## Operational Facts

Things agents keep re-discovering. Read this so you don't waste time.

| Fact | Detail |
|------|--------|
| **Transport** | Streamable HTTP only at `/mcp/`. SSE was removed. No `/sse` endpoint exists. OAuth 2.1 required via Cloudflare tunnel (`lumen.cirwel.org`); LAN/Tailscale/localhost are open. |
| **OAuth env vars** | `ANIMA_OAUTH_ISSUER_URL` (AS issuer, e.g. `https://lumen.cirwel.org`), `ANIMA_OAUTH_AUTO_APPROVE`, `ANIMA_OAUTH_SECRET` (optional), `ANIMA_OAUTH_DB_PATH` (defaults `~/.anima/oauth.db` — tokens persist across restarts), `ANIMA_OAUTH_RESOURCE_URL` (defaults `<issuer>/mcp/` — must match the URL the client has stored or claude.ai marks the connector errored). See `docs/operations/SECRETS_AND_ENV.md`. |
| **Admin gate** | The six destructive tools (`git_pull`, `deploy_from_github`, `system_service`, `system_power`, `fix_ssh_port`, `setup_tailscale`) need `X-Anima-Admin` matching `ANIMA_ADMIN_SECRET`. **Unset ⇒ fails CLOSED** (they refuse; it used to be a no-op). Escape hatch for local dev only: `ANIMA_ADMIN_ALLOW_UNAUTH_IF_NO_SECRET=true`. If a destructive tool returns "ANIMA_ADMIN_SECRET is not set", the Pi lost `anima.env` — don't debug the handler. |
| **anima.env is NOT backed up** | Deliberate — it holds secrets and the off-site archive is unencrypted. So a reflash restores Lumen *without* any secrets, and `restore_lumen.sh` recreates it from the all-empty template. The script now lists which keys came back blank; put `ANIMA_ADMIN_SECRET` back or the destructive tools stay closed. Mac copy: `~/.config/cirwel/secrets.env`. |
| **Outbound heartbeat** | `ANIMA_HEARTBEAT_URL` is not backed up. Empty means the off-host dead-man's switch is inert even when its timer is green; `/health/detailed` reports `outbound_heartbeat` degraded after 24h. Provision and prove one stale-summary `/fail` phone delivery per `docs/operations/HEARTBEAT.md`. |
| **Ports** | anima-mcp = **8766**, UNITARES governance = **8767**. Never guess. |
| **Pi restart time** | **2 minutes** after `git_pull(restart=true)`. Wait. Don't panic at proxy errors. Do NOT SSH or retry MCP during this window — it can crash WiFi. |
| **Tailscale IPs** | Verify with `tailscale status`. IPs may change after reinstall. |
| **SSH to Pi** | Port 22 standard. If SSH times out/refused, try port 2222: `ssh -p 2222 -i ~/.ssh/id_ed25519_pi unitares-anima@<tailscale-ip>` (see `docs/operations/PI_ACCESS.md`). |
| **alive_ratio** | `total_alive_seconds / age_seconds`. As of April 2026, ~66% (Pi stability has improved significantly since early days). |
| **Neural waves** | Computational proprioception from CPU/memory/IO — not real EEG. High delta = stable system, not sleep. |
| **No client uses /sse** | Claude Code, Claude Desktop, Cursor all connect to `/mcp/`. |
| **docs/ folder** | Developer reference only. Agents read CLAUDE.md, not docs/. Don't expect docs/ to reach other agents. |
| **Backups** | `~/backups/lumen/` — real automated backups (hourly snapshots + rsync mirror). `~/lumen-backups/` is OLD/STALE — ignore it. |
| **Restore after reflash** | One command: `cd ~/projects/anima-mcp && ./scripts/restore_lumen.sh`. Do NOT do it manually. See `docs/operations/BACKUP_AND_RESTORE.md`. |
| **Before declaring data lost** | Run `ls -lt ~/backups/lumen/anima_*.db | head -5` first. Backups run twice daily minimum. |

## Shared Memory Schema

`/dev/shm/anima_state.json`:
```json
{
  "updated_at": "...",
  "data": {
    "readings": { "cpu_temp_c": ..., "eeg_delta_power": ... },
    "anima": { "warmth": 0.36, "clarity": 0.73, ... },
    "led_proprioception": { "source": "led_hardware_controller", "brightness": 0.04, "optical_drive": 0.02 },
    "light_attribution": { "mode": "shadow", "status": "warming", "raw_lux": 12.8, "external_lux_residual": null, "used_by_clarity": false },
    "wifi_connected": true,
    "activity": { "level": "active", "reason": "engaged" },
    "learning": {
      "preferences": { "satisfaction": 0.87 },
      "self_beliefs": { "stability_recovery": { "confidence": 0.68 } },
      "agency": { "action_values": { "focus_attention": 0.22 } }
    }
  }
}
```
