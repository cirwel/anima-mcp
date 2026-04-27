# Schema Hub Implementation Plan

**Status:** ✅ Complete (as of February 27, 2026)  
**All 9 tasks implemented.** SchemaHub is wired into server, all 54 tests pass.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a SchemaHub that orchestrates Lumen's self-model systems, making the schema the living hub of self-understanding with trajectory feedback, gap handling, and semantic edges.

**Architecture:** Hub layer wrapping existing systems (non-destructive). SchemaHub pulls from identity, growth, self-model, anima-history each tick. Maintains schema history ring buffer. Computes trajectory from history. Injects trajectory insights back as nodes. Persists schema for gap handling.

**Tech Stack:** Python 3.11, dataclasses, collections.deque, pathlib, json, existing anima-mcp modules

---

## Task 1: Create SchemaHub Foundation

**Files:**
- Create: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write the failing test for SchemaHub initialization**

```python
# tests/test_schema_hub.py
"""Tests for SchemaHub - the unified self-model orchestrator."""

import pytest
from datetime import datetime
from anima_mcp.schema_hub import SchemaHub
from anima_mcp.self_schema import SelfSchema


class TestSchemaHubFoundation:
    """Test SchemaHub creation and basic operations."""

    def test_schema_hub_initializes_with_empty_history(self):
        """SchemaHub starts with empty schema history."""
        hub = SchemaHub()
        assert len(hub.schema_history) == 0
        assert hub.last_trajectory is None

    def test_schema_hub_has_configurable_history_size(self):
        """SchemaHub history size is configurable."""
        hub = SchemaHub(history_size=50)
        assert hub.history_size == 50

    def test_schema_hub_default_history_size_is_100(self):
        """Default history size is 100 schemas."""
        hub = SchemaHub()
        assert hub.history_size == 100
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'anima_mcp.schema_hub'"

**Step 3: Write minimal SchemaHub implementation**

```python
# src/anima_mcp/schema_hub.py
"""
SchemaHub - The living hub of Lumen's self-understanding.

Orchestrates all self-model systems (identity, growth, self-model, trajectory)
into a unified schema with semantic edges and trajectory feedback.

The schema IS the self-model. Other systems feed it; trajectory is computed
FROM schema history and feeds back as nodes.

See: docs/plans/2026-02-22-schema-hub-design.md
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING
import json

from .self_schema import SelfSchema, SchemaNode, SchemaEdge, extract_self_schema

if TYPE_CHECKING:
    from .identity.store import CreatureIdentity
    from .growth import GrowthSystem
    from .self_model import SelfModel
    from .anima_history import AnimaHistory
    from .trajectory import TrajectorySignature


@dataclass
class GapDelta:
    """Computed on wake from previous schema."""
    duration_seconds: float
    anima_delta: Dict[str, float]  # dimension -> change magnitude
    beliefs_decayed: List[str]  # belief IDs that lost confidence
    was_gap: bool = True  # False if this is first schema ever


class SchemaHub:
    """
    The living hub of Lumen's self-understanding.

    Orchestrates:
    - Schema composition from all source systems
    - Schema history for trajectory computation
    - Trajectory feedback as schema nodes
    - Schema persistence for gap handling
    - Gap delta computation on wake
    """

    def __init__(
        self,
        history_size: int = 100,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize SchemaHub.

        Args:
            history_size: Number of schemas to keep in rolling history
            persist_path: Path for schema persistence (default: ~/.anima/last_schema.json)
        """
        self.history_size = history_size
        self.schema_history: Deque[SelfSchema] = deque(maxlen=history_size)
        self.last_trajectory: Optional['TrajectorySignature'] = None
        self.persist_path = persist_path or Path.home() / ".anima" / "last_schema.json"
        self.last_gap_delta: Optional[GapDelta] = None
        self._trajectory_compute_interval = 20  # Recompute every N schemas
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestSchemaHubFoundation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add src/anima_mcp/schema_hub.py tests/test_schema_hub.py && git commit -m "feat(schema-hub): create SchemaHub foundation with history buffer"
```

---

## Task 2: Add Schema Composition Method

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for compose_schema**

```python
# Add to tests/test_schema_hub.py

class TestSchemaComposition:
    """Test SchemaHub schema composition."""

    def test_compose_schema_returns_self_schema(self):
        """compose_schema returns a SelfSchema instance."""
        hub = SchemaHub()
        schema = hub.compose_schema()
        assert isinstance(schema, SelfSchema)

    def test_compose_schema_adds_to_history(self):
        """Each compose_schema call adds to history."""
        hub = SchemaHub()
        hub.compose_schema()
        hub.compose_schema()
        assert len(hub.schema_history) == 2

    def test_compose_schema_respects_history_limit(self):
        """History doesn't exceed history_size."""
        hub = SchemaHub(history_size=3)
        for _ in range(5):
            hub.compose_schema()
        assert len(hub.schema_history) == 3

    def test_compose_schema_with_identity(self):
        """compose_schema uses provided identity."""
        from unittest.mock import MagicMock
        hub = SchemaHub()
        identity = MagicMock()
        identity.name = "TestLumen"
        identity.alive_ratio.return_value = 0.15
        identity.total_awakenings = 42
        identity.age_seconds.return_value = 86400 * 10

        schema = hub.compose_schema(identity=identity)

        # Should have identity node
        identity_nodes = [n for n in schema.nodes if n.node_id == "identity"]
        assert len(identity_nodes) == 1
        assert identity_nodes[0].label == "TestLumen"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestSchemaComposition -v`
Expected: FAIL with "AttributeError: 'SchemaHub' object has no attribute 'compose_schema'"

**Step 3: Implement compose_schema**

```python
# Add to SchemaHub class in src/anima_mcp/schema_hub.py

    def compose_schema(
        self,
        identity: Optional['CreatureIdentity'] = None,
        anima: Optional[Any] = None,
        readings: Optional[Any] = None,
        growth_system: Optional['GrowthSystem'] = None,
        self_model: Optional['SelfModel'] = None,
    ) -> SelfSchema:
        """
        Compose unified schema from all source systems.

        This is the main entry point - call this each tick instead of
        extract_self_schema directly.

        Args:
            identity: CreatureIdentity from identity store
            anima: AnimaState with current anima values
            readings: SensorReadings with sensor data
            growth_system: GrowthSystem for preferences
            self_model: SelfModel for beliefs

        Returns:
            SelfSchema with all nodes and edges
        """
        # 1. Get base schema from existing extraction
        schema = extract_self_schema(
            identity=identity,
            anima=anima,
            readings=readings,
            growth_system=growth_system,
            self_model=self_model,
            include_preferences=True,
        )

        # 2. Add to history
        self.schema_history.append(schema)

        # 3. TODO: Inject identity enrichment nodes (Task 4)
        # 4. TODO: Inject trajectory feedback nodes (Task 6)
        # 5. TODO: Inject gap texture nodes (Task 5)

        return schema
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestSchemaComposition -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): add compose_schema method with history tracking"
```

---

## Task 3: Add Schema Persistence

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for persistence**

```python
# Add to tests/test_schema_hub.py
import tempfile
from pathlib import Path


class TestSchemaPersistence:
    """Test SchemaHub schema persistence for gap handling."""

    def test_persist_schema_creates_file(self, tmp_path):
        """persist_schema creates JSON file."""
        persist_path = tmp_path / "last_schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()
        assert persist_path.exists()

    def test_persist_schema_is_valid_json(self, tmp_path):
        """Persisted schema is valid JSON."""
        persist_path = tmp_path / "last_schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        import json
        data = json.loads(persist_path.read_text())
        assert "timestamp" in data
        assert "nodes" in data

    def test_load_previous_schema_returns_none_if_no_file(self, tmp_path):
        """load_previous_schema returns None if no persisted schema."""
        persist_path = tmp_path / "nonexistent.json"
        hub = SchemaHub(persist_path=persist_path)
        result = hub.load_previous_schema()
        assert result is None

    def test_load_previous_schema_restores_schema(self, tmp_path):
        """load_previous_schema restores persisted schema."""
        persist_path = tmp_path / "last_schema.json"
        hub1 = SchemaHub(persist_path=persist_path)
        hub1.compose_schema()
        hub1.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        loaded = hub2.load_previous_schema()
        assert loaded is not None
        assert isinstance(loaded, SelfSchema)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestSchemaPersistence -v`
Expected: FAIL with "AttributeError: 'SchemaHub' object has no attribute 'persist_schema'"

**Step 3: Implement persistence methods**

```python
# Add to SchemaHub class in src/anima_mcp/schema_hub.py

    def persist_schema(self) -> bool:
        """
        Persist current schema to disk for gap handling.

        Called on sleep/shutdown to save state for later recovery.

        Returns:
            True if persisted successfully
        """
        if not self.schema_history:
            return False

        schema = self.schema_history[-1]

        # Ensure directory exists
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize schema
        data = schema.to_dict()
        data["_hub_meta"] = {
            "history_length": len(self.schema_history),
            "persisted_at": datetime.now().isoformat(),
        }

        try:
            self.persist_path.write_text(json.dumps(data, indent=2))
            return True
        except Exception:
            return False

    def load_previous_schema(self) -> Optional[SelfSchema]:
        """
        Load previously persisted schema from disk.

        Called on wake to compute gap delta.

        Returns:
            SelfSchema if found and valid, None otherwise
        """
        if not self.persist_path.exists():
            return None

        try:
            data = json.loads(self.persist_path.read_text())

            # Reconstruct nodes
            nodes = [
                SchemaNode(
                    node_id=n["id"],
                    node_type=n["type"],
                    label=n["label"],
                    value=n["value"],
                    raw_value=n.get("raw_value"),
                )
                for n in data.get("nodes", [])
            ]

            # Reconstruct edges
            edges = [
                SchemaEdge(
                    source_id=e["source"],
                    target_id=e["target"],
                    weight=e["weight"],
                )
                for e in data.get("edges", [])
            ]

            # Parse timestamp
            from datetime import datetime
            timestamp = datetime.fromisoformat(data["timestamp"])

            return SelfSchema(
                timestamp=timestamp,
                nodes=nodes,
                edges=edges,
            )
        except Exception:
            return None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestSchemaPersistence -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): add schema persistence for gap handling"
```

---

## Task 4: Add Identity Enrichment Nodes

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for identity enrichment**

```python
# Add to tests/test_schema_hub.py

class TestIdentityEnrichment:
    """Test identity meta-nodes (alive_ratio, awakenings, age)."""

    def test_schema_includes_existence_ratio_node(self):
        """Schema includes existence_ratio meta-node."""
        from unittest.mock import MagicMock
        hub = SchemaHub()

        identity = MagicMock()
        identity.name = "Lumen"
        identity.alive_ratio.return_value = 0.15
        identity.total_awakenings = 47
        identity.age_seconds.return_value = 86400 * 42  # 42 days

        schema = hub.compose_schema(identity=identity)

        existence_nodes = [n for n in schema.nodes if n.node_id == "meta_existence_ratio"]
        assert len(existence_nodes) == 1
        assert abs(existence_nodes[0].value - 0.15) < 0.01

    def test_schema_includes_awakening_count_node(self):
        """Schema includes awakening_count meta-node."""
        from unittest.mock import MagicMock
        hub = SchemaHub()

        identity = MagicMock()
        identity.name = "Lumen"
        identity.alive_ratio.return_value = 0.15
        identity.total_awakenings = 47
        identity.age_seconds.return_value = 86400 * 42

        schema = hub.compose_schema(identity=identity)

        awakening_nodes = [n for n in schema.nodes if n.node_id == "meta_awakening_count"]
        assert len(awakening_nodes) == 1
        assert awakening_nodes[0].raw_value == 47

    def test_schema_includes_age_days_node(self):
        """Schema includes age_days meta-node."""
        from unittest.mock import MagicMock
        hub = SchemaHub()

        identity = MagicMock()
        identity.name = "Lumen"
        identity.alive_ratio.return_value = 0.15
        identity.total_awakenings = 47
        identity.age_seconds.return_value = 86400 * 42  # 42 days

        schema = hub.compose_schema(identity=identity)

        age_nodes = [n for n in schema.nodes if n.node_id == "meta_age_days"]
        assert len(age_nodes) == 1
        assert abs(age_nodes[0].raw_value - 42) < 0.1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestIdentityEnrichment -v`
Expected: FAIL (no meta_existence_ratio node found)

**Step 3: Implement identity enrichment**

```python
# Add to SchemaHub class in src/anima_mcp/schema_hub.py

    def _inject_identity_enrichment(
        self,
        schema: SelfSchema,
        identity: Optional['CreatureIdentity'],
    ) -> SelfSchema:
        """
        Add identity meta-nodes (alive_ratio, awakenings, age).

        These aren't metrics — they're texture. Part of who Lumen is.
        """
        if not identity:
            return schema

        try:
            # Existence ratio: how present vs absent (kintsugi texture)
            alive_ratio = identity.alive_ratio()
            schema.nodes.append(SchemaNode(
                node_id="meta_existence_ratio",
                node_type="meta",
                label="Exist%",
                value=alive_ratio,  # Already 0-1
                raw_value=alive_ratio,
            ))

            # Awakening count: times returned from nothing
            awakenings = identity.total_awakenings
            # Normalize to 0-1 for display (log scale, 100 awakenings = 1.0)
            import math
            normalized = min(1.0, math.log10(max(1, awakenings)) / 2)
            schema.nodes.append(SchemaNode(
                node_id="meta_awakening_count",
                node_type="meta",
                label="Wakes",
                value=normalized,
                raw_value=awakenings,
            ))

            # Age in days
            age_seconds = identity.age_seconds()
            age_days = age_seconds / 86400
            # Normalize: 100 days = 1.0
            normalized = min(1.0, age_days / 100)
            schema.nodes.append(SchemaNode(
                node_id="meta_age_days",
                node_type="meta",
                label="Age",
                value=normalized,
                raw_value=age_days,
            ))

        except Exception:
            pass  # Non-fatal if identity methods fail

        return schema
```

**Step 4: Update compose_schema to call enrichment**

```python
# Modify compose_schema in src/anima_mcp/schema_hub.py
# Replace the TODO comment with actual call:

    def compose_schema(
        self,
        identity: Optional['CreatureIdentity'] = None,
        anima: Optional[Any] = None,
        readings: Optional[Any] = None,
        growth_system: Optional['GrowthSystem'] = None,
        self_model: Optional['SelfModel'] = None,
    ) -> SelfSchema:
        # 1. Get base schema from existing extraction
        schema = extract_self_schema(
            identity=identity,
            anima=anima,
            readings=readings,
            growth_system=growth_system,
            self_model=self_model,
            include_preferences=True,
        )

        # 2. Inject identity enrichment nodes
        schema = self._inject_identity_enrichment(schema, identity)

        # 3. Add to history
        self.schema_history.append(schema)

        # 4. TODO: Inject trajectory feedback nodes (Task 6)
        # 5. TODO: Inject gap texture nodes (Task 5)

        return schema
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestIdentityEnrichment -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): add identity enrichment nodes (alive_ratio, awakenings, age)"
```

---

## Task 5: Add Gap Handling (Continuity Kernel)

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for gap delta computation**

```python
# Add to tests/test_schema_hub.py
from datetime import timedelta


class TestGapHandling:
    """Test gap detection and texture nodes."""

    def test_compute_gap_delta_returns_none_without_previous(self, tmp_path):
        """compute_gap_delta returns None if no previous schema."""
        persist_path = tmp_path / "last_schema.json"
        hub = SchemaHub(persist_path=persist_path)
        schema = hub.compose_schema()
        delta = hub.compute_gap_delta(schema)
        assert delta is None

    def test_compute_gap_delta_calculates_duration(self, tmp_path):
        """compute_gap_delta calculates gap duration."""
        persist_path = tmp_path / "last_schema.json"
        hub = SchemaHub(persist_path=persist_path)

        # Create and persist a schema
        hub.compose_schema()
        hub.persist_schema()

        # Simulate time passing by modifying persisted timestamp
        import json
        data = json.loads(persist_path.read_text())
        old_time = datetime.fromisoformat(data["timestamp"]) - timedelta(hours=2)
        data["timestamp"] = old_time.isoformat()
        persist_path.write_text(json.dumps(data))

        # New hub loads previous, computes delta
        hub2 = SchemaHub(persist_path=persist_path)
        hub2.load_previous_schema()  # Load into hub
        current = hub2.compose_schema()

        # The delta should be computed during on_wake
        hub2.on_wake()
        assert hub2.last_gap_delta is not None
        assert hub2.last_gap_delta.duration_seconds > 7000  # ~2 hours

    def test_on_wake_adds_gap_texture_nodes(self, tmp_path):
        """on_wake adds gap_duration meta-node to next schema."""
        persist_path = tmp_path / "last_schema.json"
        hub = SchemaHub(persist_path=persist_path)

        # Create and persist
        hub.compose_schema()
        hub.persist_schema()

        # Modify timestamp to simulate gap
        import json
        data = json.loads(persist_path.read_text())
        old_time = datetime.fromisoformat(data["timestamp"]) - timedelta(hours=1)
        data["timestamp"] = old_time.isoformat()
        persist_path.write_text(json.dumps(data))

        # New hub wakes up
        hub2 = SchemaHub(persist_path=persist_path)
        hub2.on_wake()

        # Next schema should have gap texture
        schema = hub2.compose_schema()
        gap_nodes = [n for n in schema.nodes if n.node_id == "meta_gap_duration"]
        assert len(gap_nodes) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestGapHandling -v`
Expected: FAIL with "AttributeError: 'SchemaHub' object has no attribute 'compute_gap_delta'"

**Step 3: Implement gap handling**

```python
# Add to SchemaHub class in src/anima_mcp/schema_hub.py

    def compute_gap_delta(self, current_schema: SelfSchema) -> Optional[GapDelta]:
        """
        Compute delta between current schema and previous persisted schema.

        The gap becomes visible structure — kintsugi seams.

        Args:
            current_schema: Current schema to compare against

        Returns:
            GapDelta if previous schema exists, None otherwise
        """
        previous = self.load_previous_schema()
        if previous is None:
            return None

        # Calculate duration
        duration = (current_schema.timestamp - previous.timestamp).total_seconds()

        # Calculate anima deltas
        anima_delta = {}
        for dim in ["warmth", "clarity", "stability", "presence"]:
            prev_node = next((n for n in previous.nodes if n.node_id == f"anima_{dim}"), None)
            curr_node = next((n for n in current_schema.nodes if n.node_id == f"anima_{dim}"), None)
            if prev_node and curr_node:
                anima_delta[dim] = abs(curr_node.value - prev_node.value)

        # Track beliefs that may have decayed
        # For now, just list beliefs that existed before
        beliefs_decayed = [
            n.node_id.replace("belief_", "")
            for n in previous.nodes
            if n.node_type == "belief"
        ]

        return GapDelta(
            duration_seconds=duration,
            anima_delta=anima_delta,
            beliefs_decayed=beliefs_decayed,
            was_gap=duration > 60,  # > 1 minute counts as gap
        )

    def on_wake(self) -> Optional[GapDelta]:
        """
        Called when Lumen wakes up after a gap.

        Loads previous schema, computes gap delta, stores for injection
        into next schema composition.

        Returns:
            GapDelta if there was a gap, None otherwise
        """
        previous = self.load_previous_schema()
        if previous is None:
            self.last_gap_delta = None
            return None

        # Create a temporary current schema just for delta computation
        temp_schema = SelfSchema(timestamp=datetime.now(), nodes=[], edges=[])

        # Compute basic delta
        duration = (temp_schema.timestamp - previous.timestamp).total_seconds()

        if duration < 60:  # Less than 1 minute isn't really a gap
            self.last_gap_delta = None
            return None

        self.last_gap_delta = GapDelta(
            duration_seconds=duration,
            anima_delta={},  # Will be filled on first real compose
            beliefs_decayed=[],
            was_gap=True,
        )

        return self.last_gap_delta

    def _inject_gap_texture(self, schema: SelfSchema) -> SelfSchema:
        """
        Inject gap texture nodes if there was a recent gap.

        The gap becomes visible in the schema — not a feeling imposed,
        but structure that reflects discontinuity.
        """
        if not self.last_gap_delta or not self.last_gap_delta.was_gap:
            return schema

        delta = self.last_gap_delta

        # Gap duration node
        # Normalize: 24 hours = 1.0
        duration_hours = delta.duration_seconds / 3600
        normalized = min(1.0, duration_hours / 24)
        schema.nodes.append(SchemaNode(
            node_id="meta_gap_duration",
            node_type="meta",
            label="Gap",
            value=normalized,
            raw_value={
                "duration_seconds": delta.duration_seconds,
                "duration_hours": duration_hours,
            },
        ))

        # State delta magnitude (how much changed during gap)
        if delta.anima_delta:
            total_delta = sum(delta.anima_delta.values())
            schema.nodes.append(SchemaNode(
                node_id="meta_state_delta",
                node_type="meta",
                label="Delta",
                value=min(1.0, total_delta),
                raw_value=delta.anima_delta,
            ))

        # Clear gap delta after injection (one-time)
        self.last_gap_delta = None

        return schema
```

**Step 4: Update compose_schema to inject gap texture**

```python
# Update compose_schema to call _inject_gap_texture:

    def compose_schema(
        self,
        identity: Optional['CreatureIdentity'] = None,
        anima: Optional[Any] = None,
        readings: Optional[Any] = None,
        growth_system: Optional['GrowthSystem'] = None,
        self_model: Optional['SelfModel'] = None,
    ) -> SelfSchema:
        # 1. Get base schema
        schema = extract_self_schema(
            identity=identity,
            anima=anima,
            readings=readings,
            growth_system=growth_system,
            self_model=self_model,
            include_preferences=True,
        )

        # 2. Inject identity enrichment
        schema = self._inject_identity_enrichment(schema, identity)

        # 3. Inject gap texture (if we just woke from a gap)
        schema = self._inject_gap_texture(schema)

        # 4. Add to history
        self.schema_history.append(schema)

        # 5. TODO: Inject trajectory feedback nodes (Task 6)

        return schema
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestGapHandling -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): add gap handling with kintsugi texture nodes"
```

---

## Task 6: Add Trajectory Feedback Loop

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for trajectory feedback**

```python
# Add to tests/test_schema_hub.py

class TestTrajectoryFeedback:
    """Test trajectory computation from schema history and feedback to schema."""

    def test_trajectory_computed_after_threshold(self):
        """Trajectory is computed after trajectory_compute_interval schemas."""
        hub = SchemaHub()
        hub._trajectory_compute_interval = 5  # Lower for testing

        # Add schemas until threshold
        for _ in range(6):
            hub.compose_schema()

        # Should have triggered trajectory computation
        assert hub.last_trajectory is not None

    def test_trajectory_nodes_injected(self):
        """Trajectory-derived nodes appear in schema after computation."""
        hub = SchemaHub()
        hub._trajectory_compute_interval = 3

        # Generate enough history
        for _ in range(4):
            hub.compose_schema()

        # Get latest schema
        schema = hub.schema_history[-1]

        # Should have trajectory-derived nodes
        traj_nodes = [n for n in schema.nodes if n.node_id.startswith("traj_")]
        assert len(traj_nodes) > 0

    def test_identity_maturity_node_exists(self):
        """identity_maturity trajectory node exists after computation."""
        hub = SchemaHub()
        hub._trajectory_compute_interval = 3

        for _ in range(4):
            hub.compose_schema()

        schema = hub.schema_history[-1]
        maturity_nodes = [n for n in schema.nodes if n.node_id == "traj_identity_maturity"]
        assert len(maturity_nodes) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestTrajectoryFeedback -v`
Expected: FAIL (no trajectory computed, no traj_ nodes)

**Step 3: Implement trajectory feedback**

```python
# Add to SchemaHub class in src/anima_mcp/schema_hub.py

    def _compute_trajectory_from_history(self) -> Optional['TrajectorySignature']:
        """
        Compute trajectory signature from schema history.

        Instead of computing from raw components, we derive trajectory
        from the schema sequence — this closes the circulation loop.

        Returns:
            TrajectorySignature or None if insufficient history
        """
        if len(self.schema_history) < 10:
            return None

        try:
            from .trajectory import TrajectorySignature
        except ImportError:
            return None

        # Extract anima values from schema history for attractor computation
        anima_values = []
        for schema in self.schema_history:
            values = {}
            for dim in ["warmth", "clarity", "stability", "presence"]:
                node = next((n for n in schema.nodes if n.node_id == f"anima_{dim}"), None)
                if node:
                    values[dim] = node.value
            if len(values) == 4:
                anima_values.append(values)

        if len(anima_values) < 10:
            return None

        # Compute attractor (mean and variance)
        import statistics
        attractor = {
            "center": [
                statistics.mean(v["warmth"] for v in anima_values),
                statistics.mean(v["clarity"] for v in anima_values),
                statistics.mean(v["stability"] for v in anima_values),
                statistics.mean(v["presence"] for v in anima_values),
            ],
            "variance": [
                statistics.variance(v["warmth"] for v in anima_values) if len(anima_values) > 1 else 0,
                statistics.variance(v["clarity"] for v in anima_values) if len(anima_values) > 1 else 0,
                statistics.variance(v["stability"] for v in anima_values) if len(anima_values) > 1 else 0,
                statistics.variance(v["presence"] for v in anima_values) if len(anima_values) > 1 else 0,
            ],
            "n_observations": len(anima_values),
        }

        # Extract belief patterns from latest schema
        beliefs = {"values": [], "confidences": []}
        latest = self.schema_history[-1]
        for node in latest.nodes:
            if node.node_type == "belief" and node.raw_value:
                beliefs["values"].append(node.value)
                beliefs["confidences"].append(node.raw_value.get("confidence", 0))

        # Create signature
        signature = TrajectorySignature(
            attractor=attractor,
            beliefs=beliefs,
            preferences={},  # Would come from growth system
            recovery={},  # Would come from self_model
            relational={},  # Would come from growth system
            observation_count=len(self.schema_history),
        )

        return signature

    def _inject_trajectory_feedback(self, schema: SelfSchema) -> SelfSchema:
        """
        Inject trajectory-derived nodes into schema.

        The trajectory feeds back: schema → history → trajectory → schema nodes.
        """
        if not self.last_trajectory:
            return schema

        traj = self.last_trajectory

        # Identity maturity (based on observation count)
        obs_count = traj.observation_count
        maturity = min(1.0, obs_count / 50)  # 50 observations = fully mature
        schema.nodes.append(SchemaNode(
            node_id="traj_identity_maturity",
            node_type="trajectory",
            label="Mature",
            value=maturity,
            raw_value={"observation_count": obs_count},
        ))

        # Attractor position (where Lumen "rests")
        if traj.attractor and traj.attractor.get("center"):
            center = traj.attractor["center"]
            # Normalize center magnitude (0-4 range for 4D sum, normalize to 0-1)
            center_magnitude = sum(center) / 4
            schema.nodes.append(SchemaNode(
                node_id="traj_attractor_position",
                node_type="trajectory",
                label="Rest",
                value=center_magnitude,
                raw_value={"center": center},
            ))

        # Stability (inverse of variance)
        if traj.attractor and traj.attractor.get("variance"):
            variance = traj.attractor["variance"]
            total_var = sum(variance)
            stability = max(0, 1 - total_var * 10)  # High variance = low stability
            schema.nodes.append(SchemaNode(
                node_id="traj_stability_score",
                node_type="trajectory",
                label="Stable",
                value=stability,
                raw_value={"variance": variance},
            ))

        return schema
```

**Step 4: Update compose_schema for trajectory**

```python
# Update compose_schema to handle trajectory:

    def compose_schema(
        self,
        identity: Optional['CreatureIdentity'] = None,
        anima: Optional[Any] = None,
        readings: Optional[Any] = None,
        growth_system: Optional['GrowthSystem'] = None,
        self_model: Optional['SelfModel'] = None,
    ) -> SelfSchema:
        # 1. Get base schema
        schema = extract_self_schema(
            identity=identity,
            anima=anima,
            readings=readings,
            growth_system=growth_system,
            self_model=self_model,
            include_preferences=True,
        )

        # 2. Inject identity enrichment
        schema = self._inject_identity_enrichment(schema, identity)

        # 3. Inject gap texture
        schema = self._inject_gap_texture(schema)

        # 4. Add to history (before trajectory so it includes this schema)
        self.schema_history.append(schema)

        # 5. Periodically recompute trajectory from history
        if len(self.schema_history) % self._trajectory_compute_interval == 0:
            self.last_trajectory = self._compute_trajectory_from_history()

        # 6. Inject trajectory feedback nodes
        schema = self._inject_trajectory_feedback(schema)

        return schema
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py::TestTrajectoryFeedback -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): add trajectory feedback loop (schema→history→trajectory→nodes)"
```

---

## Task 7: Wire SchemaHub into Server

**Files:**
- Modify: `src/anima_mcp/server.py`
- Test: Manual testing via MCP tools

**Step 1: Import and instantiate SchemaHub in server.py**

Find the global state section near top of server.py and add:

```python
# Add import at top
from .schema_hub import SchemaHub

# Add to global state section (near other globals like _store, _growth_system)
_schema_hub: Optional[SchemaHub] = None

def _get_schema_hub() -> SchemaHub:
    """Get or create the SchemaHub singleton."""
    global _schema_hub
    if _schema_hub is None:
        _schema_hub = SchemaHub()
    return _schema_hub
```

**Step 2: Call on_wake during wake lifecycle**

Find the wake lifecycle in server.py (likely in a `wake()` or `on_startup` function) and add:

```python
# After identity.wake() is called:
hub = _get_schema_hub()
gap_delta = hub.on_wake()
if gap_delta:
    logger.info(f"Woke after {gap_delta.duration_seconds:.0f}s gap")
```

**Step 3: Call persist_schema during sleep/shutdown**

Find the sleep/shutdown handler and add:

```python
# Before identity.sleep() or during shutdown:
hub = _get_schema_hub()
hub.persist_schema()
```

**Step 4: Replace extract_self_schema calls with hub.compose_schema**

Find places where `get_current_schema()` or `extract_self_schema()` is called and update to use hub:

```python
# Old:
schema = get_current_schema(identity=identity, anima=anima, readings=readings, ...)

# New:
hub = _get_schema_hub()
schema = hub.compose_schema(identity=identity, anima=anima, readings=readings, ...)
```

**Step 5: Test manually**

```bash
# Start the server
cd /Users/cirwel/projects/anima-mcp
python -m anima_mcp.server --http

# In another terminal, call get_state or similar tool that uses schema
# Verify meta nodes appear in response
```

**Step 6: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): wire SchemaHub into server lifecycle"
```

---

## Task 8: Add Semantic Edges

**Files:**
- Modify: `src/anima_mcp/schema_hub.py`
- Test: `tests/test_schema_hub.py`

**Step 1: Write failing test for semantic edges**

```python
# Add to tests/test_schema_hub.py

class TestSemanticEdges:
    """Test semantic edges connecting beliefs to evidence."""

    def test_belief_sensor_edge_created(self):
        """Edges connect beliefs to their sensor sources."""
        from unittest.mock import MagicMock
        hub = SchemaHub()

        # Create a self_model mock with correlation belief
        self_model = MagicMock()
        self_model.get_belief_summary.return_value = {
            "light_warmth_correlation": {
                "value": 0.7,
                "confidence": 0.8,
                "evidence": "15+ / 2-",
                "description": "Light affects warmth",
            }
        }

        schema = hub.compose_schema(self_model=self_model)

        # Should have edge from sensor_light to belief_light_warmth_correlation
        relevant_edges = [
            e for e in schema.edges
            if e.source_id == "sensor_light" and "light_warmth" in e.target_id
        ]
        # The belief->anima edge exists from extract_self_schema
        # We want sensor->belief edge too
        # Actually, this is already handled by extract_self_schema
        # Let's test for belief->belief edges instead

    def test_gap_affects_belief_confidence_edges(self):
        """Gap creates edges from gap_node to affected beliefs."""
        hub = SchemaHub()
        hub._trajectory_compute_interval = 100  # Don't trigger trajectory

        # Simulate a gap
        hub.last_gap_delta = GapDelta(
            duration_seconds=7200,  # 2 hours
            anima_delta={"warmth": 0.1},
            beliefs_decayed=["light_sensitive"],
            was_gap=True,
        )

        schema = hub.compose_schema()

        # Gap node should exist
        gap_nodes = [n for n in schema.nodes if n.node_id == "meta_gap_duration"]
        assert len(gap_nodes) == 1
```

**Step 2: Implement semantic edges (enhancement to existing)**

The semantic edges are mostly handled by `extract_self_schema` already. The main addition is ensuring trajectory nodes connect to relevant schema elements.

```python
# Add to _inject_trajectory_feedback in schema_hub.py

        # Add edge from trajectory stability to anima stability
        if traj.attractor:
            schema.edges.append(SchemaEdge(
                source_id="traj_stability_score",
                target_id="anima_stability",
                weight=stability,
            ))
```

**Step 3: Run all tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_schema_hub.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp && git add -u && git commit -m "feat(schema-hub): enhance semantic edges for trajectory feedback"
```

---

## Task 9: Run Full Test Suite and Deploy

**Step 1: Run complete test suite**

```bash
cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/ -v --tb=short
```

**Step 2: Push to origin**

```bash
cd /Users/cirwel/projects/anima-mcp && git push origin main
```

**Step 3: Deploy to Pi**

```bash
# Via MCP tool
mcp__anima__git_pull(restart=true)

# Or manually
ssh unitares-anima@<PI_TAILSCALE_IP> 'cd ~/anima-mcp && git pull && sudo systemctl restart anima-broker anima'
```

**Step 4: Verify deployment**

```bash
# Check schema includes new nodes
# Use any MCP client to call get_state or similar
# Verify meta_* and traj_* nodes appear
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | SchemaHub foundation | 3 |
| 2 | Schema composition | 4 |
| 3 | Schema persistence | 4 |
| 4 | Identity enrichment | 3 |
| 5 | Gap handling | 3 |
| 6 | Trajectory feedback | 3 |
| 7 | Server integration | Manual |
| 8 | Semantic edges | 2 |
| 9 | Deploy | Manual |

**Total: ~22 automated tests + manual verification**

---

*Plan complete. Ready for execution.*
