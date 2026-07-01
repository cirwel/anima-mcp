# Phase-2 governance seam — addendum to the Elixir broker migration plan

**Status:** APPROVED by operator 2026-07-01 (see Decisions at the end).
Addendum to `2026-06-29-elixir-broker-migration.md`, motivated by what
Phase-1 cutover execution (2026-07-01, PR #93) taught us about the seam.
Phase-2 build is gated on ~1 week of clean Phase-1 operation (≈ 2026-07-08).

## What Phase 1 actually established (differs from the plan's assumption)

The plan's Phase-1 cutover line said the Elixir broker "becomes the sole writer
of `readings` (and the full envelope) to the live SHM path." That was not
executable: at cutover time the Elixir broker produced **5 reading keys**,
while the live envelope carries **~32 reading keys and 13 data sections**
(`eisv`, `governance`, `identity`, `inner_life`, `learning`, `metacognition`,
`experiential`, `drive_events`, …) — all computed by the Python broker and
scheduled for Phases 2–3. Stopping the Python broker would have degraded the
creature, not swapped a sensor driver.

The seam we shipped instead (PR #93):

```
Elixir broker ──writes──> /dev/shm/anima_state.shadow.json   (sole writer)
                                   │
                    Python broker reads env channels
              (ANIMA_ENV_SENSORS_FROM_SHM, 30s staleness guard)
                                   │
Python broker ──writes──> /dev/shm/anima_state.json          (sole writer)
                                   │
                     Python MCP server reads (unchanged)
```

Single-writer-per-file is preserved; ownership migrates **section by section
through the shadow file**, with the Python broker acting as the merging
envelope writer until it has nothing left to merge.

## Phase-2 question: where does the governance client move, and how does its
output reach the live envelope?

The plan has the Elixir `Anima.Governance.Client` writing `governance` +
`governance_at` "into the SHM payload" — which assumed Elixir owned the live
envelope. It does not. Options:

### Option A — extend the shadow-consumption seam (recommended)

Elixir gains the governance client (native HTTP, circuit breaker
15s→30s→60s→120s per the plan) and writes `governance` + `governance_at` into
the **shadow** envelope. The Python broker passthrough gains one more flag
(e.g. `ANIMA_GOVERNANCE_FROM_SHM`), copying those two fields into the live
envelope with the same staleness contract, and stops making its own UNITARES
calls when the flag is set.

- Pros: same proven seam as Phase 1; independently flaggable and reversible;
  the server's existing fallback (`SERVER_GOVERNANCE_FALLBACK_SECONDS`) keeps
  covering gaps exactly as today; the sync+ThreadPoolExecutor+aiohttp failure
  mode dies in the Python broker without touching the server.
- Cons: governance data crosses two files before the server sees it (adds up
  to one broker tick of latency, ~2s — well inside the 210s freshness
  threshold); the passthrough flag list grows (acceptable at n=2, revisit at
  n=3).

### Option B — Elixir takes over the live envelope at Phase 2

Bring `anima`, `learning`, `activity`, etc. into scope so Elixir can own the
full live file, per the original plan.

- Cons: pulls Phase-3 (learning/tick, the NumPy resonance field) into Phase 2's
  blast radius — exactly the scope-creep §6 warns about. Rejected for now.

### Option C — governance client moves to the MCP server instead

The server already has a native-async fallback client; make it primary.

- Cons: abandons the migration direction (governance was the Phase-2 payoff on
  BEAM: supervised client + circuit breaker as GenServer state); couples
  check-in cadence to the server process, which restarts far more often than
  the broker. Rejected.

## Identity constraints (must hold under any option)

Lumen's governance identity is bound to its check-in path. Whatever process
carries the client must preserve:

- the existing agent UUID and check-in cadence (`ANIMA_GOVERNANCE_INTERVAL_SECONDS`),
  no re-onboarding, no `force_new` — the Elixir client presents the same
  identity material the Python bridge presents today (see
  `unitares_bridge.py` for the tool-name contract: `sync_state` /
  `record_result` / `identity` aliases, not the dropped raw twins);
- the strict-identity write requirements (client_session_id echo) — the BEAM
  client must implement the same session-binding handshake before the flag
  flips, and must be soak-verified in shadow (write governance to shadow while
  Python still owns live check-ins; diff the two decision streams) before
  cutover, mirroring the Phase-1 pattern;
- the server-side fallback stays untouched as the safety net in all options.

## Acceptance / gates (Option A)

1. Shadow soak: Elixir client checks in against UNITARES using Lumen's
   identity **read-only-shadowed** (or against a scratch identity if dual
   check-ins would pollute the trajectory — decide with operator; a scratch
   identity avoids double-counting Lumen's cadence during soak).
2. Cutover = set `ANIMA_GOVERNANCE_FROM_SHM` + unset the Python bridge's
   check-in loop; verify no `SERVER_GOVERNANCE_FALLBACK` activations for a
   week (plan's Phase-2 acceptance).
3. Rollback = unset the flag; Python bridge resumes; server fallback covers
   any gap.

## Decisions (operator approved, 2026-07-01)

1. **Option A confirmed** — Elixir governance client writes `governance` +
   `governance_at` into the shadow envelope; Python passthroughs under a flag;
   server fallback untouched.
2. **Scratch identity for the soak** — the Elixir client soaks under its own
   scratch governance identity (no double-counting of Lumen's cadence or
   trajectory pollution); a short final pre-cutover window exercises the real
   identity handshake (Lumen's UUID + session echo) to prove the path.
3. **Phase 2 waits for Phase-1 stability** — build starts no earlier than
   ~2026-07-08, contingent on a clean week from `anima-broker-ex.service`
   (no crash-restarts, env channels fresh, issue #86's 24h acceptance passed).
