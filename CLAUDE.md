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

## Architecture

Two systemd services run on the Pi:

```
anima-broker.service        anima.service
(hardware broker)           (MCP server)
     |                           |
     | writes to                 | reads from
     +---> /dev/shm/anima_state.json <--+
```

| Service | Command | Role |
|---------|---------|------|
| `anima-broker.service` | `anima-creature` | Hardware broker - owns I2C, runs learning |
| `anima.service` | `anima --http` | MCP server - serves tools, reads shared memory |

**Both must run for full functionality.** The broker writes sensor data and learning state to shared memory; the server reads it.

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

`health.py` tracks 11 subsystems with heartbeats + functional probes. Rendered on LCD health screen.

| Status | Color | Meaning |
|--------|-------|---------|
| ok | Green | Heartbeat fresh, probe passes |
| stale | Yellow | Heartbeat expired, probe passes |
| degraded | Yellow/Orange | Probe failing |
| missing | Red | No heartbeat AND probe failing |

Per-subsystem stale thresholds: fast subsystems (sensors, anima) use 30s default; slow subsystems (growth) use 90s. Governance uses dedicated SHM freshness thresholds (currently 210s).

**Governance health** checks broker's shared memory governance data (broker is sole UNITARES caller, default every 180s via `ANIMA_GOVERNANCE_INTERVAL_SECONDS`). Stale threshold: 210s.

### Learning Systems (run in broker only)

These modules run in `stable_creature.py`, not in `server.py`:

| Module | Purpose |
|--------|---------|
| `adaptive_prediction.py` | Temporal pattern learning |
| `memory_retrieval.py` | Context-aware memory search |
| `agency.py` | TD-learning action selection |
| `preferences.py` | Preference evolution |
| `self_model.py` | Self-beliefs (sensitivity, recovery, correlations) |
| `activity_state.py` | Active/drowsy/resting cycles |
| `learning.py` | Calibration adaptation |

These modules also run in `server.py` (not broker-only):

| Module | Purpose |
|--------|---------|
| `growth/` | Preferences, goals, memories, autobiography (package with mixins) |
| `self_reflection.py` | Insight discovery from preferences, beliefs, drawing patterns |
| `knowledge.py` | Q&A-derived insights from answered questions (rule-based) |

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

**Lux = lux.** Raw sensor reading used everywhere — no glow correction. The sensor reads LED glow + room light together. Lumen knows its LED brightness separately as a proprioceptive signal.

All consumers use raw lux directly: clarity, activity state, growth preferences, drawing light_regime, ethical drift, self-model correlations. LED brightness is tracked as a separate known value, not decomposed from the lux reading.

**Drawing light regime thresholds** (raw lux):
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
sudo systemctl status anima-broker anima

# Restart both
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

The **broker** (`stable_creature.py`) is the primary UNITARES caller. It checks in on a configurable cadence (`ANIMA_GOVERNANCE_INTERVAL_SECONDS`, default 180s, minimum 30s) and writes the governance decision to shared memory with a `governance_at` timestamp. The **server** (`server.py`) reads governance from SHM and has a fallback: if no "via unitares" decision arrives for 240s (`SERVER_GOVERNANCE_FALLBACK_SECONDS`), the server calls UNITARES directly using its native async event loop. This fallback exists because the broker's sync+ThreadPoolExecutor+new-event-loop pattern has reliability issues with aiohttp sessions.

```
UNITARES_URL=http://<tailscale-ip>:8767/mcp/  # verify Mac IP with `tailscale status`
```

Maps anima to EISV: Warmth→Energy, Clarity→Integrity, 1-Stability→Entropy, (1-Presence)*0.3→Void

**Circuit breaker** (in `unitares_bridge.py`): 2 consecutive failures trigger exponential backoff (15s→30s→60s→120s). Any success resets to 15s.

**Three EISV contexts:**
- **DrawingEISV** (screens.py) — proprioceptive, drives drawing behavior (closed loop)
- **Mapped EISV** (eisv_mapper.py) — anima→EISV for governance reporting
- **Governance EISV** (Mac, dynamics.py) — full thermodynamics (open loop, advisory)

Local fallback (`_local_governance()`) runs simple threshold checks when Mac unreachable — more trigger-happy.
Server syncs `_last_governance_decision` from SHM when `governance_at` is within `SHM_GOVERNANCE_STALE_SECONDS` (210s).

## Identity, Continuity, and Control

**Two identity notions (do not conflate):**
- **Record identity:** `creature_id` + SQLite (`identity/store.py`) — continuity of *this* deployment’s database file.
- **Trajectory identity:** `TrajectorySignature` (`trajectory.py`) — behavioral similarity over time. Same UUID with different lived history is still one record; trajectory compares *patterns*.

**Restore / fork:** `restore_lumen.sh` and restoring `anima.db` **preserve** record identity and accumulated history. A **fresh** DB (new install, no copy) yields a **new** `creature_id`. Copying DB to another Pi **forks** record identity; behavior and trajectory may diverge with environment.

**Governance boundary:** UNITARES is **advisory** (thermodynamic check-in, verdicts). The broker still owns sensors and learning; **SHM** carries governance for the server. **`_local_governance()`** when Mac is unreachable is a **fallback**, not a substitute for embodied state — it keeps check-ins from going silent, not from replacing sensors.

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
| **OAuth env vars** | `ANIMA_OAUTH_ISSUER_URL`, `ANIMA_OAUTH_AUTO_APPROVE`, `ANIMA_OAUTH_SECRET` (optional). Tokens in-memory, reset on restart. See `docs/operations/SECRETS_AND_ENV.md`. |
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
