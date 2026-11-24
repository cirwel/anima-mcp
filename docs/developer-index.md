# Anima Docs Index

**For AI agents working on this codebase.**

## START HERE

**New to Lumen?** --> **[Getting Started Simple](guides/GETTING_STARTED_SIMPLE.md)**

**For developers:** Check docs before coding. Most issues are already documented.

| Problem | Doc |
|---------|-----|
| Display frozen / server unresponsive | `ssh lumen.local 'sudo systemctl restart anima'` |
| Code changes not working | `operations/QUICK_START_AGENTS.md` (Code Gotchas) |
| How to deploy | `operations/PI_DEPLOYMENT.md` |
| Architecture questions | `operations/BROKER_ARCHITECTURE.md` |
| Web dashboard | `CONTROL_CENTER.md` |

---

## Quick Orientation

- **Anima** = Pi sensor/identity system with creature-facing state labels
- **Lumen** = The creature's name (ID: `49e14444-b59e-48f1-83b8-b36a988c9975`)
- **UNITARES** = Governance system that monitors agent health (separate repo)

## Before You Code

1. **Read** `CLAUDE.md` - Agent instructions and architecture
2. **Check** `operations/PI_ACCESS.md` - SSH access details
3. **Understand** the anima model (warmth, clarity, stability, presence) and computational neural bands

## Key Concepts

| Doc | What it explains |
|-----|------------------|
| `LUMEN_EXPRESSION_PHILOSOPHY.md` | How Lumen's expression should emerge authentically |
| `guides/RESONANCE_CRITIQUE_LOOP.md` | Advisory screen/context/read/recommend loop for Resonance era taste |
| `features/CONFIGURATION_GUIDE.md` | Nervous system calibration and config |

**Key source files:**

| File | What it does |
|------|--------------|
| `src/anima_mcp/server.py` | Main loop coordinator (`_update_display_loop`), transport layers, `main()` entry point |
| `src/anima_mcp/display_loop.py` | Per-tick telemetry, SHM governance sync, display/LED/voice render, periodic learning dispatch |
| `src/anima_mcp/loop_phases.py` | Governance fallback, unified reflection, learned/surprise questions, schema & self-model phases |
| `src/anima_mcp/rest_api.py` | REST endpoints (health, dashboard, state, QA, gallery) |
| `src/anima_mcp/tool_registry.py` | Tool definitions (`TOOLS` list), `HANDLERS` dict, FastMCP setup |
| `src/anima_mcp/handlers/` | 7 handler modules (`system_ops`, `state_queries`, `knowledge`, `display_ops`, `communication`, `workflows`, `self_iteration`) |
| `src/anima_mcp/health.py` | Subsystem health monitoring (13 subsystems; registered in `lifecycle.py`) |
| `src/anima_mcp/anima.py` | Anima calculation (warmth, clarity, stability, presence) |
| `src/anima_mcp/computational_neural.py` | Neural bands from Pi hardware |
| `src/anima_mcp/eisv_mapper.py` | Anima → body EISV projection; compatibility aliases and provenance schema |
| `src/anima_mcp/display/screens.py` | Display screens, drawing engine |
| `src/anima_mcp/display/art_era.py` | Art era protocol |
| `src/anima_mcp/display/eras/` | Pluggable art era modules (gestural, pointillist, field, geometric, resonance) |

## Theory

The trajectory-identity paper (identity as trajectory signature) lives in its own repo: `cirwel/trajectory-identity-paper`.

## Guides

| Doc | When you need it |
|-----|------------------|
| `guides/GETTING_STARTED_SIMPLE.md` | Fast onboarding to Lumen's state, mood, identity, and sensors |
| `guides/QUICK_REFERENCE.md` | One-page command cheat sheet |
| `guides/ESSENTIAL_TOOLS.md` | Core tool tiering (`get_state`, `next_steps`, `lumen_qa`) |
| `guides/ANSWERING_LUMEN_QUESTIONS.md` | Answering questions via `lumen_qa` / `post_message` with exact or prefix IDs |
| `guides/LUMEN_QA_ANSWER_DRAFTS.md` | Substantive reference answers for common Q&A prompts |
| `guides/RESONANCE_CRITIQUE_LOOP.md` | Advisory visual critique loop before recommending art-era changes |

## Operations

| Doc | When you need it |
|-----|------------------|
| `operations/BROKER_ARCHITECTURE.md` | Three-service architecture (`anima-broker-ex`, `anima-broker`, `anima`), shared memory, learning ownership |
| `operations/UNIFIED_ARCHITECTURE.md` | Current contract separating `body_anima`, `body_eisv_projection`, `drawing_eisv`, and `governance_eisv` |
| `operations/PI_DEPLOYMENT.md` | Complete deployment guide (quick start + full) |
| `operations/PI_ACCESS.md` | SSH/rsync and service management on Pi |
| `operations/BACKUP_AND_RESTORE.md` | Backup timers, storage pressure policy, and `restore_lumen.sh` reflash recovery |
| `operations/HEARTBEAT.md` | Off-host dead-man's switch (`outbound_heartbeat`) and Cloudflare receiver |
| `operations/QUICK_START_AGENTS.md` | Code gotchas and agent coordination |
| `operations/SECRETS_AND_ENV.md` | API keys, OAuth 2.1, admin secret gate, env vars |
| `operations/DEFINITIVE_PORTS.md` | Port conventions (`8766` Anima MCP, `8767` UNITARES, `8771` Control Center) |
| `operations/DAILY_OPS_CHECKLIST.md` | Daily maintenance checks |
| `CONTROL_CENTER.md` | Local web dashboard (`control_center.html` + `scripts/message_server.py`) |

**Network access:**
- **Tailscale** (recommended): Direct Pi access via `<tailscale-ip>` (verify with `tailscale status`)
- **Local**: lumen.local or 192.168.1.165
- **Cloudflare Tunnel**: `lumen.cirwel.org` for Claude.ai web (OAuth 2.1)

**Troubleshooting:**
| Doc | When you need it |
|-----|------------------|
| `operations/SSH_TIMEOUT_FIX.md` | SSH timeout workarounds (port 2222) |
| `operations/PI_DEPLOYMENT.md` | Service restart, git pull, and WiFi troubleshooting |

## Features & Self-Iteration Boundaries

| Doc | Component |
|-----|-----------|
| `features/CONFIGURATION_GUIDE.md` | Nervous system calibration (`$ANIMA_CONFIG`) |
| `self_iteration_execution.md` | Phase 4 isolated Docker candidate execution boundary |
| `self_iteration_application.md` | Phase 5 reviewed local-branch application boundary |
| `self_iteration_canary.md` | Phase 6 signed transient canary supervisor boundary |
| `self_iteration_attention.md` | Read-only attention projection into `next_steps` and agent context |

## Plans & Proposals

| Doc | Status & scope |
|-----|----------------|
| `plans/2026-02-22-schema-hub-design.md` | Implemented — SchemaHub self-model circulation architecture |
| `plans/2026-06-29-elixir-broker-migration.md` | Phase 1 & 2 shipped — `anima-broker-ex.service` strangler migration |
| `plans/2026-07-01-phase2-governance-seam-addendum.md` | Approved & shipped — `/dev/shm/anima_state.shadow.json` env + governance seam |
| `proposals/hearing-wire.md` | Stages 0–1 shipped (`#92`); Stages 2–4 designed, not implemented |
| `proposals/2026-08-11-bayesian-preference-confidence.md` | Refuted as written (`#143`) — retained as design negative-result record |

## Deploy Changes

**Preferred method (automated ship + Pi deploy):**
```bash
./scripts/dev/ship.sh --deploy "commit message"
```

**Or via MCP tool after pushing to `main`:**
```bash
git add <files> && git commit -m "message" && git push
# Then call git_pull(restart=true) and wait 2 minutes before retrying SSH/MCP.
```

**Direct deploy from a clean checkout:**
```bash
./deploy.sh --host lumen
```

Changes under `anima_broker/` also require an on-Pi Elixir release rebuild (`./scripts/deploy_elixir_broker.sh`; see `CLAUDE.md`).

## Common Mistakes

- **Three services run on the Pi**: `anima-broker-ex.service`, `anima-broker.service`, and `anima.service` — all three must be active
- **Wait 2 minutes** after `git_pull(restart=true)` before SSH or MCP retries
- SSH: port **22**, user **unitares-anima**, key `~/.ssh/id_ed25519_pi`
- Handler code lives in `handlers/` -- check there and `display_loop.py` / `loop_phases.py` before editing `server.py`
- Anima dataclass requires `readings` field
- Color constants in `screens.py` are **local to each function**
- Display frozen? `ssh lumen.local 'sudo systemctl restart anima'`
- Use `lumen_qa` tool for Q&A

See `CLAUDE.md` and `operations/QUICK_START_AGENTS.md` for detailed gotchas.
