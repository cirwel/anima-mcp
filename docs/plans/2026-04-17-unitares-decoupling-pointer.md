# Unitares / Anima Decoupling — Pointer

**Canonical plan:** `unitares/docs/specs/2026-04-17-lumen-decoupling-design.md`

This file exists so that work done inside `anima-mcp` finds the cross-repo plan.

## What changes on Lumen's side

Only one Phase A task touches this repo:

### A1 — Include `sensor_eisv` in governance check-in payload

**File:** `src/anima_mcp/unitares_bridge.py:410-418`

`update_arguments` currently sends `sensor_data` but not `sensor_eisv`. Unitares needs `sensor_eisv = {E, I, S, V}` as a top-level key so its generic spring-coupling path in `governance_monitor.py:395-405` activates without unitares having to import a Lumen-specific buffer.

**Source of the value:** `eisv_mapper.py` already computes anima→EISV for governance reporting. Reuse that output.

**Shape required by unitares:**
- `E, I` ∈ [0, 1]
- `S` ∈ [0.001, 1.0]
- `V` ∈ [-1, 1]

Unitares clamps on read, so slight over/under is non-fatal, but match the ranges for a clean signal.

**Test to add here:** `test_unitares_bridge_includes_sensor_eisv` — bridge payload contains `sensor_eisv` with the expected shape and ranges.

**Rollout constraint:** Unitares-side changes (confirming `process_agent_update` routes `sensor_eisv` into `agent_state`) land *before* this push. Behavioral fallback in unitares covers the gap if the order slips.

## What does not change here

- Nothing about hardware, sensors, neural bands, display, or brain-hat logic — those stay Lumen-specific and are *not* leaks.
- The broker-as-primary-UNITARES-caller pattern stays (documented in `CLAUDE.md` → "UNITARES Integration").
- Lumen's public MCP tool surface on `anima-mcp` is untouched.

## See also

- `unitares/docs/specs/2026-04-17-lumen-decoupling-design.md` — full Phase A / Phase B design.
