# Quick Start for Agents

**For Claude, Composer, and other AI agents working on anima-mcp**

---

## 🛑 STOP - Read Docs First

**Before writing ANY code, check the docs.** Most problems have already been solved.

1. **`docs/developer-index.md`** - Start here, find everything
2. **This file's "Code Gotchas" section** - Learn from past agent pain
3. **`docs/operations/PI_DEPLOYMENT.md`** - Service restart and troubleshooting runbook

**Rule: If you're about to try something complex, check docs first. The simple solution is probably documented.**

---

## Before You Start

### 1. Read These First
- ✅ `CLAUDE.md` - **Agent instructions and architecture**
- ✅ `docs/developer-index.md` - Find all docs
- ✅ `README.md` - Project overview

### 2. Check Current Work
- ✅ Recent git commits / file timestamps
- ✅ Knowledge graph (if available)

### 3. Understand the System
- ✅ `docs/features/CONFIGURATION_GUIDE.md` - Config system
- ✅ `docs/operations/BROKER_ARCHITECTURE.md` - Body/mind separation

---

## Governance Rhythm (Daily Use)

Use this lightweight loop so governance data stays comparable across sessions.

### Session Start

1. Run `onboard` (or `bind_session` if resuming with a known continuity token)
2. Run `get_governance_metrics` once for baseline

### During Work (Milestone-Based Check-Ins)

Call `process_agent_update` after meaningful chunks (not every tool call), typically every 10-25 minutes of real work or after a clear task boundary.

Use this format:
- `response_text`: "Did X, verified Y, next is Z."
- `task_type`: `convergent`, `divergent`, or `mixed`
- `complexity`:
  - `0.2-0.4` routine or local edits
  - `0.5-0.7` multi-file/debug/integration work
  - `0.8-1.0` architecture or high-uncertainty work
- `confidence`:
  - `0.8-0.95` verified with tests/runtime checks
  - `0.6-0.79` partially verified
  - `<0.6` hypothesis/exploration phase

### Intervention Rules

- `verdict=proceed` + no tight margin -> continue
- `margin=tight` -> reduce step size, verify sooner, check in after next milestone
- `verdict=guide` -> adapt approach immediately and log what changed
- `verdict=pause/reject` -> stop implementation, request dialectic or human review

### Session End

Send one final `process_agent_update` summarizing completed work, what remains, and confidence in the current state.

### Copy-Paste Check-In Examples

Use these as templates when calling `process_agent_update`:

```json
{
  "response_text": "Implemented local bug fix in sensor parsing, added regression test, next step is deployment validation.",
  "task_type": "convergent",
  "complexity": 0.35,
  "confidence": 0.9
}
```

```json
{
  "response_text": "Investigated intermittent governance timeout, identified two likely causes, tests partially complete, next is targeted runtime verification.",
  "task_type": "mixed",
  "complexity": 0.65,
  "confidence": 0.7
}
```

```json
{
  "response_text": "Drafted architecture change for broker/server fallback coordination; implementation paused pending dialectic or human approval.",
  "task_type": "design",
  "complexity": 0.9,
  "confidence": 0.58
}
```

---

## Common Tasks

### Adding a Feature

1. **Check docs** - Does it exist? Should it?
2. **Check code** - Similar features? Patterns?
3. **Check running services** - Two systemd services: `anima` (MCP server) and `anima-broker` (hardware)
4. **Implement** - Follow existing patterns
5. **Update docs** - Document new feature
6. **Test** - `python3 -m pytest tests/ -x -q`

### Fixing a Bug

1. **Reproduce** - Understand the issue
2. **Check logs**:
   ```bash
   ssh -i ~/.ssh/id_ed25519_pi unitares-anima@<tailscale-ip> \
     "sudo journalctl -u anima -n 50"
   ```
3. **Check docs** - Known issues? Workarounds?
4. **Fix** - Minimal change, maximum impact
5. **Test** - `python3 -m pytest tests/ -x -q`
6. **Deploy** - `git push && mcp__anima__git_pull(restart=true)`

### Changing Config

1. **Read `docs/features/CONFIGURATION_GUIDE.md`**
2. **Check `anima_config.yaml.example`**
3. **Understand impact** - What uses this config?
4. **Change** - Update config.py if needed
5. **Update docs** - Document change
6. **Test** - Verify behavior

---

## Code Patterns

### Error Handling
```python
from .error_recovery import safe_call, retry_with_backoff

# Use safe_call for non-critical operations
result = safe_call(lambda: risky_operation(), default=None)

# Use retry_with_backoff for transient failures
result = retry_with_backoff(lambda: operation(), config=RetryConfig())
```

### Configuration
```python
from .config import get_calibration, get_display_config

# Get current config
cal = get_calibration()
display = get_display_config()

# Use config values
if temp < cal.ambient_temp_min:
    # ...
```

### Logging
```python
import sys

# Use stderr for operational logs
print(f"[Component] Message", file=sys.stderr, flush=True)
```

---

## Documentation Standards

### When Creating Docs

```markdown
# Title

**Created:** {today_full()}  
**Last Updated:** {today_full()}  
**Status:** Active

---

## Overview

Brief description...

## Details

Content...

## Related

- **`OTHER_DOC.md`** - Related docs
```

### When Updating Code

- Add docstrings to new functions
- Update existing docstrings if behavior changes
- Add comments for complex logic
- Update README if adding tools/features

---

## Testing Checklist

Before considering work "done":

- [ ] Code compiles (`python3 -m py_compile`)
- [ ] Imports work (`python3 -c 'import ...'`)
- [ ] No obvious errors
- [ ] Docs updated
- [ ] Docs updated if needed

---

## Common Pitfalls

### ❌ Don't Skip Docs
- Always check `docs/` before implementing
- Update docs with code changes
- Don't assume behavior

### ❌ Don't Break Patterns
- Follow existing code style
- Use existing abstractions
- Don't reinvent wheels

### ❌ Don't Work in Isolation
- Check recent git history
- Leave notes for other agents via knowledge graph
- Coordinate on shared files

---

## 🚨 Code Gotchas (Learn from Our Pain)

These are specific issues that have bitten agents before:

### EISV Keys Are Single Letters

**Wrong:**
```python
eisv.get("energy")  # Returns None!
eisv.get("integrity")  # Returns None!
```

**Correct:**
```python
eisv.get("E")  # Energy
eisv.get("I")  # Integrity
eisv.get("S")  # Entropy (Stability)
eisv.get("V")  # Valence: signed E-I imbalance
```

### Color Constants Are Local to Functions in screens.py

When editing `src/anima_mcp/display/screens.py`, color constants like `CYAN`, `WHITE`, `LIGHT_CYAN` are defined **inside each function**, not globally. If you:
1. Change a color constant name
2. Grep only part of the function
3. Miss some references

You'll get `NameError: name 'GRAY' is not defined` at runtime.

**Fix:** After changing colors, grep the ENTIRE function for the old color name.

### Display Frozen? Just Restart the Service

Don't try:
- ❌ Different ports (8766, 8767)
- ❌ localhost binding
- ❌ Writing display-only scripts
- ❌ Complex debugging

**Just do:**
```bash
ssh -i ~/.ssh/id_ed25519_pi unitares-anima@<tailscale-ip> \
  "sudo systemctl restart anima"
```

The systemd service handles everything. See `docs/operations/PI_DEPLOYMENT.md`.

### HTTP Server Crashes (SSE Handler)

If you see `TypeError: 'NoneType' object is not callable` in Starlette:
- SSE handlers in `server.py` must return `Response(status_code=200)`
- Even though actual responses go through ASGI, Starlette expects a return value
- Add try/except to handlers to prevent crashes from malformed requests

### Governance Data Flow (Broker → Shared Memory → MCP Server)

The **broker** (`stable_creature.py`) calls UNITARES and writes governance to shared memory.
The **MCP server** (`server.py`) reads governance FROM shared memory - it doesn't call UNITARES directly.

If governance shows "waiting..." or null on diagnostics:
1. Check broker is running: `sudo systemctl status anima-broker`
2. Check shared memory has governance: `cat /dev/shm/anima_state.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('governance'))"`
3. If broker has governance but display doesn't, the MCP server isn't reading shared memory correctly

### Identity Table Auto-Correction (Events Table is Source of Truth)

**Fixed:** The identity table now auto-corrects on every wake using the events table as source of truth.

**How it works:**
- `wake()` method recalculates `total_awakenings` and `total_alive_seconds` from events table
- Updates identity table with correct values automatically
- No manual patching needed - just restart the service

**If you see identity drift:**
- Don't manually patch the identity table
- Just restart the service - it will auto-correct
- The events table is the source of truth, identity table is a cache

**Code location:** `src/anima_mcp/identity/store.py` - `_recalculate_stats()` method

---

## Quick Reference

### Key Files
- `src/anima_mcp/server.py` - Main server
- `src/anima_mcp/anima.py` - Core anima logic
- `src/anima_mcp/config.py` - Configuration
- `src/anima_mcp/learning.py` - Adaptive learning
- `docs/` - All documentation

### Key Tools
- `get_state` - Current anima state
- `unified_workflow` - Cross-server workflows
- `diagnostics` - System health
- `get_calibration` - Current config

---

**Remember: Check docs first, coordinate with other agents, update docs after changes.**
