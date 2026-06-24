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
| `src/anima_mcp/server.py` | Main loop, lifecycle, REST API |
| `src/anima_mcp/tool_registry.py` | Tool definitions, HANDLERS dict, FastMCP setup |
| `src/anima_mcp/handlers/` | 6 handler modules (system, state, knowledge, display, comms, workflows) |
| `src/anima_mcp/health.py` | Subsystem health monitoring (9 subsystems) |
| `src/anima_mcp/anima.py` | Anima calculation (warmth, clarity, stability, presence) |
| `src/anima_mcp/computational_neural.py` | Neural bands from Pi hardware |
| `src/anima_mcp/eisv_mapper.py` | EISV mapping for UNITARES governance |
| `src/anima_mcp/display/screens.py` | Display screens, drawing engine |
| `src/anima_mcp/display/art_era.py` | Art era protocol |
| `src/anima_mcp/display/eras/` | Pluggable art era modules (gestural, pointillist, field, geometric, resonance) |

## Theory

The trajectory-identity paper (identity as trajectory signature) lives in its own repo: `cirwel/trajectory-identity-paper`. The Lumen EISV art paper outline is archived at `docs/archive/lumen_eisv_art_paper.md`.

## Operations

| Doc | When you need it |
|-----|------------------|
| `operations/BROKER_ARCHITECTURE.md` | Body/mind separation, shared memory, learning systems |
| `operations/PI_DEPLOYMENT.md` | Complete deployment guide (quick start + full) |
| `operations/PI_ACCESS.md` | SSH/rsync to Pi |
| `operations/BACKUP_AND_RESTORE.md` | Backup, restore, and full reflash recovery |
| `operations/QUICK_START_AGENTS.md` | Code gotchas and agent coordination |
| `operations/SECRETS_AND_ENV.md` | API keys, OAuth, env vars |
| `operations/DEFINITIVE_PORTS.md` | Port conventions |
| `operations/DEPLOY_WITHOUT_SSH.md` | HTTP deploy when SSH unavailable |
| `operations/DAILY_OPS_CHECKLIST.md` | Daily maintenance tasks |
| `CONTROL_CENTER.md` | Web dashboard docs |

**Network access:**
- **Tailscale** (recommended): Direct Pi access via `<tailscale-ip>` (verify with `tailscale status`)
- **Local**: lumen.local or 192.168.1.165
- **Cloudflare Tunnel**: `lumen.cirwel.org` for Claude.ai web (OAuth 2.1)

**Troubleshooting:**
| Doc | When you need it |
|-----|------------------|
| `operations/SSH_TIMEOUT_FIX.md` | SSH timeout workarounds (port 2222) |
| `operations/FIX_GIT_PULL_ON_PI.md` | Git pull failures |

## Features

| Doc | Component |
|-----|-----------|
| `features/CONFIGURATION_GUIDE.md` | Nervous system calibration |
| `features/UNIFIED_WORKFLOWS.md` | Workflow system documentation |

## Plans & Roadmap

| Doc | What it covers |
|-----|----------------|
| `plans/FURTHER_STEPS.md` | Active roadmap: what's done, what's next |
| `plans/2026-02-22-schema-hub-design.md` | SchemaHub architecture (implemented, still referenced) |
| `plans/2026-02-23-computational-selfhood-*.md` | Calibration drift, value tension (implemented) |

## Archive

Historical docs in `archive/` -- completed plans, resolved analyses. Kept for reference.

## Deploy Changes

**Preferred method (via git + MCP tool):**
```bash
git add <files> && git commit -m "message" && git push
mcp__anima__git_pull(restart=true)
```

**Alternative (direct rsync):**
```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_pi" \
  --exclude='.venv' --exclude='*.db' --exclude='__pycache__' --exclude='.git' \
  /Users/cirwel/projects/anima-mcp/ \
  unitares-anima@lumen.local:/home/unitares-anima/anima-mcp/

ssh lumen.local 'sudo systemctl restart anima.service'
```

## Common Mistakes

- SSH: port **22**, user **unitares-anima**, key `~/.ssh/id_ed25519_pi`
- Handler code lives in `handlers/` -- check there before editing `server.py`
- Anima dataclass requires `readings` field
- Color constants in `screens.py` are **local to each function**
- Display frozen? `ssh lumen.local 'sudo systemctl restart anima'`
- Use `lumen_qa` tool for Q&A

See `operations/QUICK_START_AGENTS.md` for detailed gotchas.
