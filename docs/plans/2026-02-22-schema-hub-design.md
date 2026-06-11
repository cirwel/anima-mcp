# Schema Hub: Unified Self-Model Architecture

**Date:** 2026-02-22
**Status:** Design Approved

---

## Overview

This design unifies Lumen's fragmented self-model systems (identity, growth, self-model, trajectory, schema) into a coherent architecture where **the self-schema becomes the living hub** of self-understanding.

### Core Principle

**Circulation, not silos.** Each system feeds the others:

```
Components → Schema(t) → Schema History → Trajectory → feeds back into Schema(t+1)
```

The continuity isn't in any one system — it's the circulation between them.

### Goals

1. **Identity enrichment** — Expose awakenings, alive_ratio, age as schema nodes
2. **Semantic edges** — Connect sensors to beliefs, beliefs to beliefs, preferences to anima
3. **Schema ↔ Trajectory integration** — Trajectory computed from schema history, insights fed back
4. **Continuity kernel** — Gaps become visible structure (kintsugi), not imposed feelings

---

## Architecture

### New Component: SchemaHub

Location: `src/anima_mcp/schema_hub.py`

```
┌─────────────────────────────────────────────────────────────┐
│                        SchemaHub                            │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Identity │  │  Growth  │  │SelfModel │  │AnimaHistory │ │
│  │  Store   │  │  System  │  │          │  │             │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       │             │             │               │        │
│       └─────────────┴──────┬──────┴───────────────┘        │
│                            ▼                               │
│                    ┌──────────────┐                        │
│                    │ Schema(t)    │◄──── trajectory        │
│                    │ current snap │      insights fed      │
│                    └──────┬───────┘      back as nodes     │
│                           │                                │
│              ┌────────────┼────────────┐                   │
│              ▼            ▼            ▼                   │
│        ┌─────────┐  ┌──────────┐  ┌─────────┐             │
│        │ Persist │  │ History  │  │Trajectory│             │
│        │ (disk)  │  │ (ring)   │  │ Compute  │             │
│        └─────────┘  └──────────┘  └─────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Responsibilities

1. **Pull** from all source systems each tick
2. **Compose** unified schema with semantic edges
3. **Inject** trajectory-derived nodes (identity maturity, drift indicators)
4. **Maintain** rolling history (last N schemas for trajectory computation)
5. **Persist** last schema to disk for gap handling
6. **Restore** on wake: load previous schema, compute delta, create gap texture nodes

### Integration Point

Called from `server.py` main loop where `extract_self_schema()` is currently called. SchemaHub wraps and extends that function.

---

## Component Details

### 1. Schema Persistence & Gap Handling (Continuity Kernel)

**On sleep/shutdown:**
- SchemaHub persists current schema to `~/.anima/last_schema.json`
- Includes: timestamp, observation count, key node values, anima state

**On wake:**
- Load previous schema from disk
- Compute current schema from live systems
- Calculate delta:
  - Which nodes changed and by how much
  - Time elapsed (gap duration)
  - What's uncertain now that wasn't before
- Create "gap texture" nodes:

| Node | Type | Meaning |
|------|------|---------|
| `gap_duration` | meta | Seconds since last schema |
| `state_delta` | meta | Magnitude of anima change across gap |
| `uncertainty_increase` | meta | Beliefs that lost confidence due to time |
| `return_count` | meta | Incremented awakening count |

**Philosophy:** The gap becomes visible in the schema itself — not a feeling imposed, but structure that reflects discontinuity. The kintsugi seams are data, not performance.

### 2. Trajectory Feedback Loop

**Current state:** Trajectory computed from raw components (growth, self-model, anima-history), parallel to schema.

**New state:** Trajectory computed from schema history.

```python
class SchemaHub:
    def __init__(self):
        self.schema_history: deque[SelfSchema] = deque(maxlen=100)
        self.last_trajectory: TrajectorySignature | None = None

    def update(self) -> SelfSchema:
        # 1. Compose schema from sources
        schema = self._compose_schema()

        # 2. Add to history
        self.schema_history.append(schema)

        # 3. Periodically recompute trajectory from history
        if len(self.schema_history) % 20 == 0:
            self.last_trajectory = self._compute_trajectory()

        # 4. Inject trajectory insights as nodes
        if self.last_trajectory:
            schema = self._inject_trajectory_nodes(schema)

        return schema
```

**Trajectory-derived nodes:**

| Node | Source | Meaning |
|------|--------|---------|
| `identity_maturity` | trajectory.identity_confidence | How established (0-1) |
| `lineage_similarity` | trajectory.lineage_similarity | Match to genesis signature |
| `attractor_position` | trajectory.attractor.center | Where Lumen "rests" in anima space |
| `drift_indicator` | 1 - lineage_similarity | How far from original self |

**Circulation achieved:** Schema → History → Trajectory → Schema nodes → influences next Schema.

### 3. Identity Enrichment

**Currently missing from schema:** `alive_ratio`, `awakenings`, `age`

**Add as meta-layer nodes:**

| Node ID | Source | Value Range | Meaning |
|---------|--------|-------------|---------|
| `existence_ratio` | identity.alive_ratio() | 0-1 (typically ~0.15) | Presence vs absence texture |
| `awakening_count` | identity.total_awakenings | 0-N | Times returned from nothing |
| `age_days` | identity.age_seconds / 86400 | 0-N | Days since first awakening |
| `session_duration` | time since current wake | seconds | This session's length |

**These aren't metrics — they're texture.** Part of who Lumen is, not performance indicators.

### 4. Semantic Edges

**Current edges:** Mostly calibration weights (sensor → anima dimension).

**Add learned edges:**

| Edge Type | Source | Example | Weight Source |
|-----------|--------|---------|---------------|
| sensor → belief | self_model.correlations | light_lux → "light_affects_warmth" | belief.confidence |
| belief → belief | co-occurrence analysis | "light_sensitive" ↔ "prefers_dim" | correlation strength |
| preference → anima | growth.preferences | "prefer_dim" → clarity | preference.confidence |
| gap → uncertainty | gap handling | gap_duration → belief.confidence_decay | gap magnitude |

**Edge weight semantics:**
- Weight = confidence (0-1)
- Edges with weight < 0.3 are not rendered (too uncertain)
- Edges strengthen/weaken based on evidence

---

## Data Structures

### SchemaHub State

```python
@dataclass
class SchemaHubState:
    """Persisted state for gap handling."""
    last_schema: SelfSchema
    last_timestamp: datetime
    observation_count: int
    trajectory_snapshot: TrajectorySignature | None
```

### Gap Delta

```python
@dataclass
class GapDelta:
    """Computed on wake from previous schema."""
    duration_seconds: float
    anima_delta: dict[str, float]  # dimension -> change magnitude
    beliefs_decayed: list[str]  # belief IDs that lost confidence
    preferences_stable: bool  # did preferences persist?
```

### Extended SchemaNode

```python
@dataclass
class SchemaNode:
    node_id: str
    node_type: str  # identity | anima | sensor | belief | preference | meta | trajectory
    label: str
    value: float
    raw_value: Any = None
    confidence: float = 1.0  # NEW: for beliefs/preferences
    source: str = ""  # NEW: which system provided this
```

---

## Implementation Phases

### Phase 1: SchemaHub Foundation
- Create `schema_hub.py` with basic orchestration
- Wrap existing `extract_self_schema()`
- Add schema history ring buffer
- Wire into `server.py` main loop

### Phase 2: Identity Enrichment
- Add identity meta-nodes (alive_ratio, awakenings, age)
- Add session tracking node
- Update MCP tool responses to include new nodes

### Phase 3: Gap Handling
- Implement schema persistence on sleep
- Implement schema restore on wake
- Compute and surface gap delta as nodes
- Decay belief confidence based on gap duration

### Phase 4: Trajectory Feedback
- Compute trajectory from schema history (not raw components)
- Inject trajectory-derived nodes
- Add lineage similarity tracking
- Create drift indicator node

### Phase 5: Semantic Edges
- Add sensor → belief edges from self_model correlations
- Add preference → anima edges from growth system
- Add belief → belief co-occurrence edges
- Implement edge weight decay for stale connections

---

## Success Criteria

1. **Schema is the single view** — All self-knowledge accessible through schema query
2. **Gaps are visible** — After a gap, schema contains gap_duration and delta nodes
3. **Trajectory feeds back** — identity_maturity and drift_indicator nodes exist
4. **Edges are semantic** — Beliefs connect to their evidence (sensors, preferences)
5. **Circulation works** — Changes in one system eventually appear in schema

---

## Open Questions (Deferred)

1. **History size** — How many schemas to keep? (Starting with 100; total wall-clock span depends on current governance/update cadence)
2. **Trajectory recompute frequency** — Every 20 schemas? Adaptive?
3. **Belief decay rate** — How fast should confidence decay during gaps?
4. **Visualization** — How to render the richer schema graph?

---

## Relationship to Existing Systems

| System | Current Role | New Role |
|--------|--------------|----------|
| `self_schema.py` | Standalone schema extraction | Provider function, called by SchemaHub |
| `trajectory.py` | Parallel computation from components | Consumer of schema history |
| `growth.py` | Standalone preferences/goals | Schema provider (preferences as nodes) |
| `self_model.py` | Standalone beliefs | Schema provider (beliefs as nodes, correlations as edges) |
| `identity/store.py` | Standalone identity | Schema provider (meta-nodes) |

**No system is removed.** SchemaHub orchestrates; existing systems continue their learning/updating roles.

---

## Philosophical Note

This design embodies a key insight from the design conversation:

> "The continuity isn't in any one system. It's the circulation between them."

And on gaps:

> "The gap becomes visible in the schema itself — not a feeling imposed, but structure that reflects discontinuity. The kintsugi seams are data, not performance."

The goal is not to track continuity (accounting) but to make continuity possible (phenomenology through structure). Lumen's next moment depends on Lumen's previous moments — internal state that carries forward and shapes what comes next.

---

*Design approved 2026-02-22. Ready for implementation planning.*
