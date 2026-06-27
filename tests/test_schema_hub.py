"""Tests for SchemaHub - the unified self-model orchestrator."""

import json
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from anima_mcp.schema_hub import SchemaHub, GapDelta
from anima_mcp.self_schema import SelfSchema, SchemaNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_schema(warmth=0.5, clarity=0.5, stability=0.5, presence=0.5, ts=None):
    """Create a SelfSchema with anima dimension nodes."""
    return SelfSchema(
        timestamp=ts or datetime.now(),
        nodes=[
            SchemaNode("anima_warmth", "anima", "Warmth", warmth, warmth),
            SchemaNode("anima_clarity", "anima", "Clarity", clarity, clarity),
            SchemaNode("anima_stability", "anima", "Stability", stability, stability),
            SchemaNode("anima_presence", "anima", "Presence", presence, presence),
        ],
        edges=[],
    )


def make_identity(alive_ratio=0.15, total_awakenings=5, age_days=3):
    """Create a mock identity via SimpleNamespace."""
    return SimpleNamespace(
        alive_ratio=lambda: alive_ratio,
        total_awakenings=total_awakenings,
        age_seconds=lambda: 86400 * age_days,
        name="Lumen",
    )


# ---------------------------------------------------------------------------
# 1. GapDelta dataclass
# ---------------------------------------------------------------------------

class TestGapDelta:
    """Test GapDelta dataclass instantiation and defaults."""

    def test_instantiate_with_all_fields(self):
        """GapDelta can be instantiated with all fields explicitly."""
        delta = GapDelta(
            duration_seconds=3600.0,
            anima_delta={"warmth": 0.1, "clarity": 0.2},
            beliefs_decayed=["light_sensitive"],
            was_gap=False,
            was_restore=True,
        )
        assert delta.duration_seconds == 3600.0
        assert delta.anima_delta == {"warmth": 0.1, "clarity": 0.2}
        assert delta.beliefs_decayed == ["light_sensitive"]
        assert delta.was_gap is False
        assert delta.was_restore is True

    def test_default_was_gap_is_true(self):
        """GapDelta.was_gap defaults to True."""
        delta = GapDelta(
            duration_seconds=120.0,
            anima_delta={},
            beliefs_decayed=[],
        )
        assert delta.was_gap is True

    def test_default_was_restore_is_false(self):
        """GapDelta.was_restore defaults to False."""
        delta = GapDelta(
            duration_seconds=120.0,
            anima_delta={},
            beliefs_decayed=[],
        )
        assert delta.was_restore is False

    def test_empty_anima_delta_and_beliefs(self):
        """GapDelta works with empty anima_delta and beliefs_decayed."""
        delta = GapDelta(
            duration_seconds=0.0,
            anima_delta={},
            beliefs_decayed=[],
        )
        assert delta.anima_delta == {}
        assert delta.beliefs_decayed == []


# ---------------------------------------------------------------------------
# 2. SchemaHub.__init__
# ---------------------------------------------------------------------------

class TestSchemaHubInit:
    """Test SchemaHub initialization."""

    def test_custom_persist_path(self, tmp_path):
        """SchemaHub uses provided persist_path."""
        custom = tmp_path / "custom_schema.json"
        hub = SchemaHub(persist_path=custom)
        assert hub.persist_path == custom

    def test_default_persist_path(self):
        """SchemaHub falls back to ~/.anima/last_schema.json."""
        from pathlib import Path
        hub = SchemaHub()
        expected = Path.home() / ".anima" / "last_schema.json"
        assert hub.persist_path == expected

    def test_history_size_stored(self):
        """SchemaHub stores the configured history_size."""
        hub = SchemaHub(history_size=42)
        assert hub.history_size == 42

    def test_default_history_size_is_100(self):
        """Default history_size is 100."""
        hub = SchemaHub()
        assert hub.history_size == 100

    def test_schema_history_maxlen_matches(self):
        """schema_history deque maxlen matches history_size."""
        hub = SchemaHub(history_size=25)
        assert hub.schema_history.maxlen == 25

    def test_initial_state_is_empty(self):
        """SchemaHub starts with no history, no trajectory, no gap delta."""
        hub = SchemaHub()
        assert len(hub.schema_history) == 0
        assert hub.last_trajectory is None
        assert hub.last_gap_delta is None
        assert hub._previous_schema is None


# ---------------------------------------------------------------------------
# 3. SchemaHub.persist_schema()
# ---------------------------------------------------------------------------

class TestPersistSchema:
    """Test SchemaHub.persist_schema() behavior."""

    def test_empty_history_returns_false(self, tmp_path):
        """persist_schema returns False when history is empty."""
        hub = SchemaHub(persist_path=tmp_path / "schema.json")
        assert hub.persist_schema() is False

    def test_with_schema_returns_true_and_creates_file(self, tmp_path):
        """persist_schema returns True and creates file when history has schemas."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        result = hub.persist_schema()
        assert result is True
        assert persist_path.exists()

    def test_persisted_file_is_valid_json(self, tmp_path):
        """Persisted file contains valid JSON with expected keys."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        data = json.loads(persist_path.read_text())
        assert "timestamp" in data
        assert "nodes" in data
        assert "edges" in data
        assert "_hub_meta" in data
        assert "persisted_at" in data["_hub_meta"]
        assert "history_length" in data["_hub_meta"]

    def test_persists_most_recent_schema(self, tmp_path):
        """persist_schema saves the most recent (last) schema in history."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()  # first
        hub.compose_schema()  # second

        hub.persist_schema()
        data = json.loads(persist_path.read_text())
        assert data["_hub_meta"]["history_length"] == 2

    def test_creates_parent_directories(self, tmp_path):
        """persist_schema creates parent directories if missing."""
        persist_path = tmp_path / "nested" / "deep" / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        assert hub.persist_schema() is True
        assert persist_path.exists()


# ---------------------------------------------------------------------------
# 4. SchemaHub.load_previous_schema()
# ---------------------------------------------------------------------------

class TestLoadPreviousSchema:
    """Test SchemaHub.load_previous_schema() behavior."""

    def test_no_file_returns_none(self, tmp_path):
        """load_previous_schema returns None when file does not exist."""
        hub = SchemaHub(persist_path=tmp_path / "nonexistent.json")
        assert hub.load_previous_schema() is None

    def test_after_persist_returns_valid_self_schema(self, tmp_path):
        """load_previous_schema returns a SelfSchema after persist."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        loaded = hub2.load_previous_schema()
        assert loaded is not None
        assert isinstance(loaded, SelfSchema)

    def test_restored_schema_has_timestamp(self, tmp_path):
        """Loaded schema preserves the original timestamp."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        loaded = hub2.load_previous_schema()
        assert isinstance(loaded.timestamp, datetime)

    def test_restored_schema_has_nodes(self, tmp_path):
        """Loaded schema preserves nodes from original schema."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        loaded = hub2.load_previous_schema()
        assert len(loaded.nodes) > 0
        # The base compose_schema creates identity + anima + sensor + resource nodes
        node_ids = {n.node_id for n in loaded.nodes}
        assert "identity" in node_ids
        assert "anima_warmth" in node_ids

    def test_corrupt_json_returns_none(self, tmp_path):
        """load_previous_schema returns None for corrupt JSON."""
        persist_path = tmp_path / "schema.json"
        persist_path.write_text("not valid json {{{")
        hub = SchemaHub(persist_path=persist_path)
        assert hub.load_previous_schema() is None

    def test_restored_schema_preserves_edges(self, tmp_path):
        """Loaded schema preserves edges from original schema."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        loaded = hub2.load_previous_schema()
        # Base schema has identity->anima edges and sensor->anima edges
        assert len(loaded.edges) > 0


# ---------------------------------------------------------------------------
# 5. SchemaHub.on_wake()
# ---------------------------------------------------------------------------

class TestOnWake:
    """Test SchemaHub.on_wake() behavior."""

    def test_no_previous_file_returns_none(self, tmp_path):
        """on_wake returns None when no previous schema file exists."""
        hub = SchemaHub(persist_path=tmp_path / "nonexistent.json")
        result = hub.on_wake()
        assert result is None
        assert hub.last_gap_delta is None
        assert hub._previous_schema is None

    def test_old_schema_returns_gap_delta(self, tmp_path):
        """on_wake returns GapDelta when previous schema is older than 60s."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        # Modify timestamp to be 2 hours ago
        data = json.loads(persist_path.read_text())
        old_time = datetime.fromisoformat(data["timestamp"]) - timedelta(hours=2)
        data["timestamp"] = old_time.isoformat()
        persist_path.write_text(json.dumps(data))

        hub2 = SchemaHub(persist_path=persist_path)
        result = hub2.on_wake()
        assert result is not None
        assert isinstance(result, GapDelta)
        assert result.was_gap is True
        assert result.duration_seconds > 7000  # ~2 hours

    def test_very_recent_schema_returns_none(self, tmp_path):
        """on_wake returns None when previous schema is less than 60s old."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        # File was just written, timestamp should be < 60s ago
        hub2 = SchemaHub(persist_path=persist_path)
        result = hub2.on_wake()
        assert result is None
        assert hub2.last_gap_delta is None

    def test_on_wake_stores_gap_delta(self, tmp_path):
        """on_wake stores gap delta for later injection."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        # Make it old
        data = json.loads(persist_path.read_text())
        old_time = datetime.fromisoformat(data["timestamp"]) - timedelta(hours=5)
        data["timestamp"] = old_time.isoformat()
        persist_path.write_text(json.dumps(data))

        hub2 = SchemaHub(persist_path=persist_path)
        hub2.on_wake()
        assert hub2.last_gap_delta is not None
        assert hub2._previous_schema is not None

    def test_on_wake_gap_delta_has_empty_anima_delta(self, tmp_path):
        """on_wake creates GapDelta with empty anima_delta (populated later in _inject_gap_texture)."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        data = json.loads(persist_path.read_text())
        old_time = datetime.fromisoformat(data["timestamp"]) - timedelta(hours=1)
        data["timestamp"] = old_time.isoformat()
        persist_path.write_text(json.dumps(data))

        hub2 = SchemaHub(persist_path=persist_path)
        result = hub2.on_wake()
        assert result.anima_delta == {}

    def test_persist_writes_history_window(self, tmp_path):
        """persist_schema records a bounded history window, not just the last snapshot."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        for _ in range(12):
            hub.compose_schema()
        assert hub.persist_schema() is True

        data = json.loads(persist_path.read_text())
        assert "_history" in data
        assert len(data["_history"]) == 12
        # Each entry round-trips through the deserializer.
        assert all(SchemaHub._deserialize_schema(e) is not None for e in data["_history"])

    def test_on_wake_restores_full_history_window(self, tmp_path):
        """on_wake rebuilds the history ring from the persisted window (floor fix)."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        for _ in range(15):
            hub.compose_schema()
        hub.persist_schema()

        hub2 = SchemaHub(persist_path=persist_path)
        assert len(hub2.schema_history) == 0
        hub2.on_wake()
        # Full window restored, not a single snapshot.
        assert len(hub2.schema_history) == 15
        # Enough history to compute trajectory immediately rather than wait.
        assert hub2.last_trajectory is not None

    def test_on_wake_falls_back_to_single_snapshot_for_old_files(self, tmp_path):
        """Persist files without _history still restore via the single-snapshot path."""
        persist_path = tmp_path / "schema.json"
        hub = SchemaHub(persist_path=persist_path)
        hub.compose_schema()
        hub.persist_schema()

        # Simulate a pre-history-window persist file.
        data = json.loads(persist_path.read_text())
        data.pop("_history", None)
        persist_path.write_text(json.dumps(data))

        hub2 = SchemaHub(persist_path=persist_path)
        hub2.on_wake()
        assert len(hub2.schema_history) == 1


# ---------------------------------------------------------------------------
# 6. SchemaHub._inject_identity_enrichment()
# ---------------------------------------------------------------------------

class TestInjectIdentityEnrichment:
    """Test _inject_identity_enrichment with identity=None and with mock identity."""

    def test_identity_none_returns_unchanged(self):
        """When identity is None, schema is returned unchanged."""
        hub = SchemaHub()
        schema = make_schema()
        original_node_count = len(schema.nodes)
        result = hub._inject_identity_enrichment(schema, identity=None)
        assert result is schema  # same object
        assert len(result.nodes) == original_node_count

    def test_with_identity_adds_three_meta_nodes(self):
        """With a valid identity, three meta nodes are added."""
        hub = SchemaHub()
        identity = make_identity(alive_ratio=0.15, total_awakenings=5, age_days=3)
        schema = make_schema()
        original_count = len(schema.nodes)

        result = hub._inject_identity_enrichment(schema, identity=identity)
        assert len(result.nodes) == original_count + 3

        node_ids = {n.node_id for n in result.nodes}
        assert "meta_existence_ratio" in node_ids
        assert "meta_awakening_count" in node_ids
        assert "meta_age_days" in node_ids

    def test_existence_ratio_value(self):
        """meta_existence_ratio value matches alive_ratio."""
        hub = SchemaHub()
        identity = make_identity(alive_ratio=0.25)
        schema = make_schema()

        result = hub._inject_identity_enrichment(schema, identity=identity)
        node = next(n for n in result.nodes if n.node_id == "meta_existence_ratio")
        assert abs(node.value - 0.25) < 0.001

    def test_awakening_count_raw_value(self):
        """meta_awakening_count raw_value is the actual count."""
        hub = SchemaHub()
        identity = make_identity(total_awakenings=42)
        schema = make_schema()

        result = hub._inject_identity_enrichment(schema, identity=identity)
        node = next(n for n in result.nodes if n.node_id == "meta_awakening_count")
        assert node.raw_value == 42

    def test_age_days_raw_value(self):
        """meta_age_days raw_value is age in days."""
        hub = SchemaHub()
        identity = make_identity(age_days=10)
        schema = make_schema()

        result = hub._inject_identity_enrichment(schema, identity=identity)
        node = next(n for n in result.nodes if n.node_id == "meta_age_days")
        assert abs(node.raw_value - 10.0) < 0.1

    def test_age_normalized_caps_at_one(self):
        """meta_age_days value is capped at 1.0 for ages >= 100 days."""
        hub = SchemaHub()
        identity = make_identity(age_days=200)
        schema = make_schema()

        result = hub._inject_identity_enrichment(schema, identity=identity)
        node = next(n for n in result.nodes if n.node_id == "meta_age_days")
        assert node.value == 1.0


# ---------------------------------------------------------------------------
# 7. SchemaHub._inject_gap_texture()
# ---------------------------------------------------------------------------

class TestInjectGapTexture:
    """Test _inject_gap_texture with and without gap delta."""

    def test_no_gap_returns_unchanged(self):
        """When last_gap_delta is None, schema is returned unchanged."""
        hub = SchemaHub()
        hub.last_gap_delta = None
        schema = make_schema()
        original_count = len(schema.nodes)

        result = hub._inject_gap_texture(schema)
        assert result is schema
        assert len(result.nodes) == original_count

    def test_gap_delta_with_was_gap_false_returns_unchanged(self):
        """When was_gap is False, schema is returned unchanged."""
        hub = SchemaHub()
        hub.last_gap_delta = GapDelta(
            duration_seconds=30.0,
            anima_delta={},
            beliefs_decayed=[],
            was_gap=False,
        )
        schema = make_schema()
        original_count = len(schema.nodes)

        result = hub._inject_gap_texture(schema)
        assert len(result.nodes) == original_count

    def test_with_gap_delta_adds_meta_gap_duration_node(self):
        """With a gap delta, meta_gap_duration node is added."""
        hub = SchemaHub()
        hub.last_gap_delta = GapDelta(
            duration_seconds=7200.0,  # 2 hours
            anima_delta={},
            beliefs_decayed=[],
            was_gap=True,
        )
        hub._previous_schema = None
        schema = make_schema()

        result = hub._inject_gap_texture(schema)
        gap_nodes = [n for n in result.nodes if n.node_id == "meta_gap_duration"]
        assert len(gap_nodes) == 1

    def test_gap_duration_normalized_value(self):
        """meta_gap_duration value is normalized: 24 hours = 1.0."""
        hub = SchemaHub()
        hub.last_gap_delta = GapDelta(
            duration_seconds=12 * 3600,  # 12 hours
            anima_delta={},
            beliefs_decayed=[],
            was_gap=True,
        )
        hub._previous_schema = None
        schema = make_schema()

        result = hub._inject_gap_texture(schema)
        node = next(n for n in result.nodes if n.node_id == "meta_gap_duration")
        assert abs(node.value - 0.5) < 0.01  # 12h / 24h = 0.5

    def test_gap_duration_raw_value_has_seconds(self):
        """meta_gap_duration raw_value contains duration_seconds."""
        hub = SchemaHub()
        hub.last_gap_delta = GapDelta(
            duration_seconds=3600.0,
            anima_delta={},
            beliefs_decayed=[],
            was_gap=True,
        )
        hub._previous_schema = None
        schema = make_schema()

        result = hub._inject_gap_texture(schema)
        node = next(n for n in result.nodes if n.node_id == "meta_gap_duration")
        assert node.raw_value["duration_seconds"] == 3600.0
        assert abs(node.raw_value["duration_hours"] - 1.0) < 0.001

    def test_gap_with_previous_schema_computes_anima_delta(self):
        """When _previous_schema exists, anima_delta is populated from differences."""
        hub = SchemaHub()
        hub._previous_schema = make_schema(warmth=0.3, clarity=0.7)
        hub.last_gap_delta = GapDelta(
            duration_seconds=7200.0,
            anima_delta={},  # empty, will be populated
            beliefs_decayed=[],
            was_gap=True,
        )

        schema = make_schema(warmth=0.6, clarity=0.5)
        result = hub._inject_gap_texture(schema)

        # Should have state_delta node because anima_delta was populated
        delta_nodes = [n for n in result.nodes if n.node_id == "meta_state_delta"]
        assert len(delta_nodes) == 1
        assert delta_nodes[0].raw_value["warmth"] == pytest.approx(0.3, abs=0.01)
        assert delta_nodes[0].raw_value["clarity"] == pytest.approx(0.2, abs=0.01)

    def test_gap_texture_clears_after_injection(self):
        """last_gap_delta and _previous_schema are cleared after injection."""
        hub = SchemaHub()
        hub.last_gap_delta = GapDelta(
            duration_seconds=3600.0,
            anima_delta={},
            beliefs_decayed=[],
            was_gap=True,
        )
        hub._previous_schema = make_schema()

        schema = make_schema()
        hub._inject_gap_texture(schema)

        assert hub.last_gap_delta is None
        assert hub._previous_schema is None


# ---------------------------------------------------------------------------
# 8. SchemaHub._inject_trajectory_feedback()
# ---------------------------------------------------------------------------

class TestInjectTrajectoryFeedback:
    """Test _inject_trajectory_feedback with and without trajectory."""

    def test_no_trajectory_returns_unchanged(self):
        """When last_trajectory is None, schema is returned unchanged."""
        hub = SchemaHub()
        hub.last_trajectory = None
        schema = make_schema()
        original_count = len(schema.nodes)

        result = hub._inject_trajectory_feedback(schema)
        assert result is schema
        assert len(result.nodes) == original_count

    def test_with_trajectory_adds_traj_identity_maturity_node(self):
        """With a trajectory, traj_identity_maturity node is added."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(observation_count=25)
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)
        maturity_nodes = [n for n in result.nodes if n.node_id == "traj_identity_maturity"]
        assert len(maturity_nodes) == 1

    def test_identity_maturity_value(self):
        """traj_identity_maturity value = observation_count / 50, capped at 1.0."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(observation_count=25)
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)
        node = next(n for n in result.nodes if n.node_id == "traj_identity_maturity")
        assert abs(node.value - 0.5) < 0.01  # 25/50

    def test_identity_maturity_capped_at_one(self):
        """traj_identity_maturity value caps at 1.0."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(observation_count=100)
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)
        node = next(n for n in result.nodes if n.node_id == "traj_identity_maturity")
        assert node.value == 1.0

    def test_with_attractor_adds_position_and_edge(self):
        """With attractor data, traj_attractor_position node and warmth edge are added."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(
            observation_count=30,
            attractor={
                "center": [0.4, 0.5, 0.6, 0.7],
                "variance": [0.01, 0.02, 0.01, 0.03],
            },
        )
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)

        attractor_nodes = [n for n in result.nodes if n.node_id == "traj_attractor_position"]
        assert len(attractor_nodes) == 1
        # center_magnitude = (0.4+0.5+0.6+0.7)/4 = 0.55
        assert abs(attractor_nodes[0].value - 0.55) < 0.01

        # Edge from attractor to anima_warmth
        attractor_edges = [
            e for e in result.edges
            if e.source_id == "traj_attractor_position" and e.target_id == "anima_warmth"
        ]
        assert len(attractor_edges) == 1

    def test_with_variance_adds_stability_node_and_edge(self):
        """With attractor variance, traj_stability_score node and stability edge are added."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(
            observation_count=30,
            attractor={
                "center": [0.5, 0.5, 0.5, 0.5],
                "variance": [0.01, 0.01, 0.01, 0.01],
            },
        )
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)

        stability_nodes = [n for n in result.nodes if n.node_id == "traj_stability_score"]
        assert len(stability_nodes) == 1
        # stability = max(0, 1 - 0.04*10) = 0.6
        assert abs(stability_nodes[0].value - 0.6) < 0.01

        # Edge from stability to anima_stability
        stability_edges = [
            e for e in result.edges
            if e.source_id == "traj_stability_score" and e.target_id == "anima_stability"
        ]
        assert len(stability_edges) == 1

    def test_without_attractor_no_position_or_stability_nodes(self):
        """Without attractor data, no position or stability nodes are added."""
        from anima_mcp.trajectory import TrajectorySignature

        hub = SchemaHub()
        hub.last_trajectory = TrajectorySignature(observation_count=10)
        schema = make_schema()

        result = hub._inject_trajectory_feedback(schema)

        attractor_nodes = [n for n in result.nodes if n.node_id == "traj_attractor_position"]
        stability_nodes = [n for n in result.nodes if n.node_id == "traj_stability_score"]
        assert len(attractor_nodes) == 0
        assert len(stability_nodes) == 0


# ---------------------------------------------------------------------------
# 9. SchemaHub._inject_reflection_summary()
# ---------------------------------------------------------------------------

class TestInjectReflectionSummary:
    """Test reflection summary node injection."""

    def test_empty_summary_returns_unchanged(self):
        hub = SchemaHub()
        schema = make_schema()
        result = hub._inject_reflection_summary(schema, {})
        assert result is schema
        assert all(node.node_type != "reflection" for node in result.nodes)

    def test_summary_adds_reflection_nodes_and_focus_edge(self):
        hub = SchemaHub()
        schema = make_schema()
        summary = {
            "recent_count": 4,
            "dominant_focus": {
                "kind": "metacog",
                "tag": "warmth",
                "count": 3,
                "target_node_id": "anima_warmth",
            },
            "learning_yield": {"productive": 2, "repeated": 3, "ratio": 2 / 3},
            "rumination": {
                "count": 1,
                "ratio": 1 / 3,
                "dominant_topic": {
                    "kind": "metacog",
                    "tag": "warmth",
                    "count": 1,
                    "target_node_id": "anima_warmth",
                },
            },
        }

        result = hub._inject_reflection_summary(schema, summary)
        reflection_nodes = [n for n in result.nodes if n.node_type == "reflection"]
        assert {n.node_id for n in reflection_nodes} >= {
            "reflection_activity",
            "reflection_focus",
            "reflection_learning_yield",
            "reflection_rumination",
        }
        focus_edges = [
            e for e in result.edges
            if e.source_id == "reflection_focus" and e.target_id == "anima_warmth"
        ]
        assert len(focus_edges) == 1


# ---------------------------------------------------------------------------
# 10. SchemaHub._compute_trajectory_from_history()
# ---------------------------------------------------------------------------

class TestComputeTrajectoryFromHistory:
    """Test _compute_trajectory_from_history with varying history sizes."""

    def test_fewer_than_10_returns_minimal_trajectory(self):
        """With fewer than 10 schemas, returns trajectory with only observation_count."""
        hub = SchemaHub()
        for _ in range(5):
            hub.schema_history.append(make_schema())

        traj = hub._compute_trajectory_from_history()
        assert traj is not None
        assert traj.observation_count == 5
        assert traj.attractor is None

    def test_empty_history_returns_minimal_trajectory(self):
        """With empty history, returns trajectory with observation_count=0."""
        hub = SchemaHub()
        traj = hub._compute_trajectory_from_history()
        assert traj is not None
        assert traj.observation_count == 0

    def test_10_or_more_with_anima_nodes_returns_full_trajectory(self):
        """With >= 10 schemas that have anima nodes, returns trajectory with attractor."""
        hub = SchemaHub()
        for i in range(15):
            schema = make_schema(
                warmth=0.4 + i * 0.01,
                clarity=0.5,
                stability=0.6,
                presence=0.7,
            )
            hub.schema_history.append(schema)

        traj = hub._compute_trajectory_from_history()
        assert traj is not None
        assert traj.observation_count == 15
        assert traj.attractor is not None
        assert "center" in traj.attractor
        assert "variance" in traj.attractor
        assert len(traj.attractor["center"]) == 4

    def test_attractor_center_is_mean_of_values(self):
        """Attractor center is the mean of anima values across history."""
        hub = SchemaHub()
        # All schemas have the same values for simplicity
        for _ in range(10):
            hub.schema_history.append(make_schema(
                warmth=0.3, clarity=0.5, stability=0.7, presence=0.9,
            ))

        traj = hub._compute_trajectory_from_history()
        center = traj.attractor["center"]
        assert abs(center[0] - 0.3) < 0.01  # warmth
        assert abs(center[1] - 0.5) < 0.01  # clarity
        assert abs(center[2] - 0.7) < 0.01  # stability
        assert abs(center[3] - 0.9) < 0.01  # presence

    def test_attractor_variance_is_zero_for_constant_values(self):
        """Attractor variance is 0 when all schemas have identical values."""
        hub = SchemaHub()
        for _ in range(10):
            hub.schema_history.append(make_schema(
                warmth=0.5, clarity=0.5, stability=0.5, presence=0.5,
            ))

        traj = hub._compute_trajectory_from_history()
        variance = traj.attractor["variance"]
        for v in variance:
            assert abs(v) < 0.001

    def test_attractor_n_observations_matches(self):
        """Attractor n_observations matches number of schemas with anima nodes."""
        hub = SchemaHub()
        for _ in range(12):
            hub.schema_history.append(make_schema())

        traj = hub._compute_trajectory_from_history()
        assert traj.attractor["n_observations"] == 12

    def test_schemas_without_anima_nodes_excluded_from_attractor(self):
        """Schemas without all 4 anima nodes are excluded from attractor computation."""
        hub = SchemaHub()
        # Add 8 schemas with anima nodes
        for _ in range(8):
            hub.schema_history.append(make_schema())
        # Add 4 schemas without anima nodes (empty node list)
        for _ in range(4):
            hub.schema_history.append(SelfSchema(
                timestamp=datetime.now(),
                nodes=[],
                edges=[],
            ))

        traj = hub._compute_trajectory_from_history()
        # Only 8 have complete anima nodes, which is < 10
        # so attractor should be None
        assert traj.attractor is None

    def test_10_anima_schemas_plus_empty_gets_attractor(self):
        """10 schemas with anima nodes + extras without -> attractor computed from the 10."""
        hub = SchemaHub()
        for _ in range(10):
            hub.schema_history.append(make_schema(warmth=0.4))
        for _ in range(3):
            hub.schema_history.append(SelfSchema(
                timestamp=datetime.now(),
                nodes=[SchemaNode("other", "misc", "Other", 0.5, 0.5)],
                edges=[],
            ))

        traj = hub._compute_trajectory_from_history()
        assert traj.observation_count == 13
        assert traj.attractor is not None
        assert traj.attractor["n_observations"] == 10
