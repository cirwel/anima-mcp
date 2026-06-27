"""
SchemaHub - The living hub of Lumen's self-understanding.

Orchestrates all self-model systems (identity, growth, self-model, trajectory)
into a unified schema with semantic edges and trajectory feedback.

The schema IS the self-model. Other systems feed it; trajectory is computed
FROM schema history and feeds back as nodes.

See: docs/plans/2026-02-22-schema-hub-design.md
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING
import json

from .atomic_write import atomic_json_write
from .self_schema import SelfSchema, SchemaNode, SchemaEdge, extract_self_schema

# Bounded window of schema history persisted to disk so trajectory structure
# survives a restart, not just the single last snapshot. Without this the
# schema graph collapses to its base "floor" on reboot until 10+ fresh
# compositions rebuild the attractor. See persist_schema() / on_wake().
HISTORY_PERSIST_LIMIT = 50

if TYPE_CHECKING:
    from .identity.store import CreatureIdentity
    from .growth import GrowthSystem
    from .self_model import SelfModel
    from .trajectory import TrajectorySignature


@dataclass
class GapDelta:
    """Computed on wake from previous schema."""
    duration_seconds: float
    anima_delta: Dict[str, float]  # dimension -> change magnitude
    beliefs_decayed: List[str]  # belief IDs that lost confidence
    was_gap: bool = True  # False if this is first schema ever
    was_restore: bool = False  # True if state was restored from backup (gap time unreliable)


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
        self._previous_schema: Optional[SelfSchema] = None
        self._trajectory_compute_interval = 20  # Recompute every N schemas
        self._schemas_since_trajectory = 0

    def compose_schema(
        self,
        identity: Optional['CreatureIdentity'] = None,
        anima: Optional[Any] = None,
        readings: Optional[Any] = None,
        growth_system: Optional['GrowthSystem'] = None,
        self_model: Optional['SelfModel'] = None,
        drift_offsets: Optional[Dict[str, float]] = None,
        tension_conflicts: Optional[list] = None,
        reflection_summary: Optional[Dict[str, Any]] = None,
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

        # 2. Inject identity enrichment nodes
        schema = self._inject_identity_enrichment(schema, identity)

        # 3. Inject gap texture (if we just woke from a gap)
        schema = self._inject_gap_texture(schema)

        # 4. Add to history (before trajectory so it includes this schema)
        self.schema_history.append(schema)

        # 5. Periodically recompute trajectory from history
        self._schemas_since_trajectory += 1
        if self._schemas_since_trajectory >= self._trajectory_compute_interval:
            self.last_trajectory = self._compute_trajectory_from_history()
            self._schemas_since_trajectory = 0

        # 6. Inject trajectory feedback nodes
        schema = self._inject_trajectory_feedback(schema)

        # 7. Inject calibration drift nodes (if drift is active)
        schema = self._inject_drift_offsets(schema, drift_offsets)

        # 8. Inject value tension nodes (structural + transient conflicts)
        schema = self._inject_tension_nodes(schema, tension_conflicts)

        # 9. Inject experiential accumulation nodes
        schema = self._inject_experiential_accumulation(schema)

        # 10. Inject bounded reflection summary nodes
        schema = self._inject_reflection_summary(schema, reflection_summary)

        return schema

    def _inject_identity_enrichment(
        self,
        schema: SelfSchema,
        identity: Optional['CreatureIdentity'],
    ) -> SelfSchema:
        """
        Add identity meta-nodes (alive_ratio, awakenings, age).

        These aren't metrics - they're texture. Part of who Lumen is.
        """
        if not identity:
            return schema

        try:
            import math

            # Existence ratio: how present vs absent (kintsugi texture)
            alive_ratio = identity.alive_ratio()
            schema.nodes.append(SchemaNode(
                node_id="meta_existence_ratio",
                node_type="meta",
                label=f"Alive {alive_ratio:.0%}",
                value=alive_ratio,  # Already 0-1
                raw_value=alive_ratio,
            ))

            # Awakening count: times returned from nothing
            awakenings = identity.total_awakenings
            # Normalize to 0-1 for display (log scale, 100 awakenings = 1.0)
            normalized = min(1.0, math.log10(max(1, awakenings)) / 2)
            schema.nodes.append(SchemaNode(
                node_id="meta_awakening_count",
                node_type="meta",
                label=f"{awakenings} Wakes",
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
                label=f"Age {age_days:.0f}d",
                value=normalized,
                raw_value=age_days,
            ))

        except Exception:
            pass  # Non-fatal if identity methods fail

        return schema

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

        # Persist trajectory so it survives restarts
        if self.last_trajectory:
            try:
                traj = self.last_trajectory
                data["_trajectory"] = {
                    "observation_count": traj.observation_count,
                    "attractor": traj.attractor,
                    "beliefs": traj.beliefs if hasattr(traj, 'beliefs') else {},
                }
            except Exception:
                pass

        # Persist a bounded window of history (not just the last snapshot) so
        # trajectory structure can be rebuilt immediately on wake instead of
        # decaying to the base "floor" graph until enough fresh compositions
        # accumulate.
        try:
            window = list(self.schema_history)[-HISTORY_PERSIST_LIMIT:]
            data["_history"] = [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "nodes": [n.to_dict() for n in s.nodes],
                    "edges": [e.to_dict() for e in s.edges],
                }
                for s in window
            ]
        except Exception:
            pass

        try:
            atomic_json_write(self.persist_path, data, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def _deserialize_schema(data: Dict[str, Any]) -> Optional[SelfSchema]:
        """Reconstruct a SelfSchema from a persisted dict (timestamp/nodes/edges)."""
        try:
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
            edges = [
                SchemaEdge(
                    source_id=e["source"],
                    target_id=e["target"],
                    weight=e["weight"],
                )
                for e in data.get("edges", [])
            ]
            timestamp = datetime.fromisoformat(data["timestamp"])
            return SelfSchema(timestamp=timestamp, nodes=nodes, edges=edges)
        except Exception:
            return None

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
        except Exception:
            return None

        return self._deserialize_schema(data)

    def compute_gap_delta(self, current_schema: SelfSchema) -> Optional[GapDelta]:
        """
        Compute delta between current schema and previous persisted schema.

        The gap becomes visible structure - kintsugi seams.
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
        into next schema composition. Seeds history so trajectory nodes
        don't have to wait for 10+ fresh compositions.
        """
        previous = self.load_previous_schema()
        if previous is None:
            self.last_gap_delta = None
            self._previous_schema = None
            return None

        try:
            data = json.loads(self.persist_path.read_text())
        except Exception:
            data = {}

        # Restore the persisted history window so trajectory structure is
        # available immediately on wake, instead of collapsing to the base
        # "floor" graph until 10+ fresh compositions rebuild the attractor.
        restored_history = False
        if not self.schema_history:
            for entry in data.get("_history") or []:
                restored = self._deserialize_schema(entry)
                if restored is not None:
                    self.schema_history.append(restored)
                    restored_history = True
            # Fall back to the single last snapshot for older persist files
            # that predate history-window persistence.
            if not self.schema_history:
                self.schema_history.append(previous)

        # Recompute trajectory from the restored window so maturity/attractor/
        # stability nodes appear on the first composition after wake.
        if restored_history and len(self.schema_history) >= 10:
            self.last_trajectory = self._compute_trajectory_from_history()
            self._schemas_since_trajectory = 0

        # Fall back to the persisted trajectory summary if we could not
        # recompute from history (sparse window / older persist file).
        if self.last_trajectory is None:
            traj_data = data.get("_trajectory")
            if traj_data:
                try:
                    from .trajectory import TrajectorySignature
                    self.last_trajectory = TrajectorySignature(
                        observation_count=traj_data.get("observation_count", 0),
                        attractor=traj_data.get("attractor"),
                        beliefs=traj_data.get("beliefs", {}),
                    )
                except Exception:
                    pass

        self._previous_schema = previous  # Store for anima_delta computation

        # Create a temporary current schema just for delta computation
        temp_schema = SelfSchema(timestamp=datetime.now(), nodes=[], edges=[])

        # Compute basic delta
        duration = (temp_schema.timestamp - previous.timestamp).total_seconds()

        if duration < 60:  # Less than 1 minute isn't really a gap
            self.last_gap_delta = None
            self._previous_schema = None
            return None

        self.last_gap_delta = GapDelta(
            duration_seconds=duration,
            anima_delta={},  # Will be populated in _inject_gap_texture
            beliefs_decayed=[],
            was_gap=True,
        )

        return self.last_gap_delta

    def _inject_gap_texture(self, schema: SelfSchema) -> SelfSchema:
        """
        Inject gap texture nodes if there was a recent gap.

        The gap becomes visible in the schema - not a feeling imposed,
        but structure that reflects discontinuity.
        """
        if not self.last_gap_delta or not self.last_gap_delta.was_gap:
            return schema

        delta = self.last_gap_delta

        # Populate anima_delta now that we have current schema
        if self._previous_schema and not delta.anima_delta:
            for dim in ["warmth", "clarity", "stability", "presence"]:
                prev_node = next((n for n in self._previous_schema.nodes if n.node_id == f"anima_{dim}"), None)
                curr_node = next((n for n in schema.nodes if n.node_id == f"anima_{dim}"), None)
                if prev_node and curr_node:
                    delta.anima_delta[dim] = abs(curr_node.value - prev_node.value)

        # Gap duration node
        # Normalize: 24 hours = 1.0
        duration_hours = delta.duration_seconds / 3600
        normalized = min(1.0, duration_hours / 24)
        schema.nodes.append(SchemaNode(
            node_id="meta_gap_duration",
            node_type="meta",
            label=f"Gap {duration_hours:.1f}h",
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
                label=f"Δ State {total_delta:.2f}",
                value=min(1.0, total_delta),
                raw_value=delta.anima_delta,
            ))

        # Clear after injection
        self.last_gap_delta = None
        self._previous_schema = None

        return schema

    def _compute_trajectory_from_history(self) -> Optional['TrajectorySignature']:
        """
        Compute trajectory signature from schema history.

        Instead of computing from raw components, we derive trajectory
        from the schema sequence - this closes the circulation loop.
        """
        if len(self.schema_history) < 10:
            # Not enough history, but still return a minimal trajectory
            # based on observation count for identity_maturity
            try:
                from .trajectory import TrajectorySignature
            except ImportError:
                return None

            return TrajectorySignature(
                observation_count=len(self.schema_history),
            )

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

        attractor = None
        if len(anima_values) >= 10:
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
                beliefs["confidences"].append(node.raw_value.get("confidence", 0) if isinstance(node.raw_value, dict) else 0)

        # Create signature
        signature = TrajectorySignature(
            attractor=attractor,
            beliefs=beliefs,
            preferences={},
            recovery={},
            relational={},
            observation_count=len(self.schema_history),
        )

        return signature

    def _inject_trajectory_feedback(self, schema: SelfSchema) -> SelfSchema:
        """
        Inject trajectory-derived nodes into schema.

        The trajectory feeds back: schema -> history -> trajectory -> schema nodes.
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
            label=f"Maturity ({obs_count} obs)",
            value=maturity,
            raw_value={"observation_count": obs_count},
        ))

        # Attractor position (where Lumen "rests")
        if traj.attractor and traj.attractor.get("center"):
            center = traj.attractor["center"]
            center_magnitude = sum(center) / 4
            schema.nodes.append(SchemaNode(
                node_id="traj_attractor_position",
                node_type="trajectory",
                label=f"Attractor {center_magnitude:.2f}",
                value=center_magnitude,
                raw_value={"center": center},
            ))

            # Add edge from attractor to primary anima dimension (warmth)
            schema.edges.append(SchemaEdge(
                source_id="traj_attractor_position",
                target_id="anima_warmth",
                weight=center_magnitude,
            ))

        # Stability (inverse of variance)
        if traj.attractor and traj.attractor.get("variance"):
            variance = traj.attractor["variance"]
            total_var = sum(variance)
            stability = max(0, 1 - total_var * 10)
            schema.nodes.append(SchemaNode(
                node_id="traj_stability_score",
                node_type="trajectory",
                label=f"Traj Stability {stability:.2f}",
                value=stability,
                raw_value={"variance": variance},
            ))

            # Add edge from trajectory stability to anima stability
            schema.edges.append(SchemaEdge(
                source_id="traj_stability_score",
                target_id="anima_stability",
                weight=stability,
            ))

        return schema

    def _inject_drift_offsets(
        self,
        schema: SelfSchema,
        drift_offsets: Optional[Dict[str, float]],
    ) -> SelfSchema:
        """
        Inject calibration drift offset nodes into the schema.

        Drift offsets show how Lumen's sense of "normal" has shifted from
        hardware defaults. Positive = experienced higher-than-default,
        negative = experienced lower-than-default.
        """
        if not drift_offsets:
            return schema

        # Only inject if at least one dimension has meaningful drift
        if all(abs(v) < 0.001 for v in drift_offsets.values()):
            return schema

        for dim_name, offset in drift_offsets.items():
            if abs(offset) < 0.001:
                continue

            # Normalize offset to 0-1 range for display
            # Max possible offset is ~0.1 (10% of 0.5 default), so scale by 5
            normalized = max(0.0, min(1.0, 0.5 + offset * 5))

            schema.nodes.append(SchemaNode(
                node_id=f"drift_{dim_name}",
                node_type="drift",
                label=f"Drift {dim_name}",
                value=normalized,
                raw_value={"offset": offset, "dimension": dim_name},
            ))

            # Connect drift to its anima dimension
            schema.edges.append(SchemaEdge(
                source_id=f"drift_{dim_name}",
                target_id=f"anima_{dim_name}",
                weight=abs(offset) * 5,  # Stronger edge for larger drift
            ))

        return schema

    def _inject_tension_nodes(
        self,
        schema: SelfSchema,
        tension_conflicts: Optional[list],
    ) -> SelfSchema:
        """
        Inject value tension nodes into the schema.

        Tensions show where improving one dimension necessarily worsens another.
        Three categories: structural (permanent), environmental (transient),
        volitional (action-caused).
        """
        if not tension_conflicts:
            return schema

        # De-duplicate: keep latest conflict per (category, dim_a, dim_b)
        seen: dict = {}
        for conflict in tension_conflicts:
            key = (conflict.category, conflict.dim_a, conflict.dim_b)
            seen[key] = conflict

        for (category, dim_a, dim_b), conflict in seen.items():
            node_id = f"tension_{category}_{dim_a}_{dim_b}"

            # Compute normalized value based on category
            if category == "structural":
                value = 0.5  # always present, constant
                label = f"{dim_a} ↔ {dim_b}"
            elif category == "environmental":
                value = max(0.0, min(1.0, conflict.duration / 10.0))
                label = f"Env: {dim_a} ↔ {dim_b}"
            elif category == "volitional":
                value = max(0.0, min(1.0, abs(conflict.grad_a - conflict.grad_b)))
                label = f"Vol: {dim_a} ↔ {dim_b}"
            else:
                continue

            schema.nodes.append(SchemaNode(
                node_id=node_id,
                node_type="tension",
                label=label,
                value=value,
                raw_value={
                    "category": category,
                    "dim_a": dim_a,
                    "dim_b": dim_b,
                    "duration": conflict.duration,
                    "action_type": conflict.action_type,
                },
            ))

            # Two edges per conflict: tension -> each dimension
            # Negative weight signifies opposing force
            edge_weight = -value * 0.5
            schema.edges.append(SchemaEdge(
                source_id=node_id,
                target_id=f"anima_{dim_a}",
                weight=edge_weight,
            ))
            schema.edges.append(SchemaEdge(
                source_id=node_id,
                target_id=f"anima_{dim_b}",
                weight=edge_weight,
            ))

        return schema

    def _inject_experiential_accumulation(
        self,
        schema: SelfSchema,
    ) -> SelfSchema:
        """
        Inject experiential accumulation nodes: marks, filter bias, pathway density.

        These represent how experience has shaped the creature — not current state,
        but accumulated structure from living.
        """
        try:
            from .experiential_marks import get_experiential_marks
            marks = get_experiential_marks()
            stats = marks.get_stats()
            if stats["total_marks"] > 0:
                for mark_info in marks.get_all_earned():
                    schema.nodes.append(SchemaNode(
                        node_id=f"mark_{mark_info['mark_id']}",
                        node_type="mark",
                        label=mark_info["name"],
                        value=mark_info["effect_value"],
                        raw_value={
                            "category": mark_info["category"],
                            "effect_key": mark_info["effect_key"],
                            "earned_at": mark_info["earned_at"],
                        },
                    ))
        except Exception:
            pass

        try:
            from .experiential_filter import get_experiential_filter
            ef = get_experiential_filter()
            ef_stats = ef.get_stats()
            if ef_stats["biased_count"] > 0:
                schema.nodes.append(SchemaNode(
                    node_id="experiential_filter_bias",
                    node_type="experiential",
                    label="Attention bias",
                    value=ef_stats["mean_salience"],
                    raw_value=ef_stats["biased_dimensions"],
                ))
        except Exception:
            pass

        try:
            from .weighted_pathways import get_weighted_pathways
            pw = get_weighted_pathways()
            pw_stats = pw.get_stats()
            if pw_stats["total_pathways"] > 0:
                schema.nodes.append(SchemaNode(
                    node_id="experiential_pathway_density",
                    node_type="experiential",
                    label="Pathway density",
                    value=pw_stats["avg_strength"],
                    raw_value={
                        "total": pw_stats["total_pathways"],
                        "contexts": pw_stats["unique_contexts"],
                        "reinforcements": pw_stats["total_reinforcements"],
                    },
                ))
        except Exception:
            pass

        return schema

    @staticmethod
    def _reflection_topic_to_node_id(topic: Optional[str]) -> Optional[str]:
        """Map reflection topics back onto existing schema nodes where possible."""
        if not topic:
            return None
        normalized = topic.split(":", 1)[-1]
        if normalized in {"warmth", "clarity", "stability", "presence"}:
            return f"anima_{normalized}"
        sensor_map = {
            "light": "sensor_light",
            "ambient_temp": "sensor_temp",
            "humidity": "sensor_humidity",
            "pressure": "sensor_pressure",
        }
        return sensor_map.get(normalized)

    @staticmethod
    def _reflection_topic_label(topic: str) -> str:
        """Humanize a reflection topic for node labels."""
        return topic.split(":", 1)[-1].replace("_", " ")

    def _inject_reflection_summary(
        self,
        schema: SelfSchema,
        reflection_summary: Optional[Dict[str, Any]],
    ) -> SelfSchema:
        """Inject bounded reflection summary nodes, not raw reflection episodes."""
        if not reflection_summary or not reflection_summary.get("recent_count"):
            return schema

        recent_count = int(reflection_summary.get("recent_count", 0))
        schema.nodes.append(SchemaNode(
            node_id="reflection_activity",
            node_type="reflection",
            label=f"Reflect {recent_count}",
            value=min(1.0, recent_count / 10.0),
            raw_value=reflection_summary,
        ))

        dominant_focus = reflection_summary.get("dominant_focus") or {}
        focus_tag = dominant_focus.get("tag")
        if focus_tag:
            focus_node_id = "reflection_focus"
            schema.nodes.append(SchemaNode(
                node_id=focus_node_id,
                node_type="reflection",
                label=f"Focus {self._reflection_topic_label(focus_tag)}",
                value=min(1.0, dominant_focus.get("count", 1) / 5.0),
                raw_value=dominant_focus,
            ))
            target_id = dominant_focus.get("target_node_id") or self._reflection_topic_to_node_id(focus_tag)
            if target_id:
                schema.edges.append(SchemaEdge(
                    source_id=focus_node_id,
                    target_id=target_id,
                    weight=max(0.2, min(1.0, dominant_focus.get("count", 1) / 5.0)),
                ))

        learning_yield = reflection_summary.get("learning_yield") or {}
        learning_ratio = learning_yield.get("ratio")
        if learning_ratio is not None:
            schema.nodes.append(SchemaNode(
                node_id="reflection_learning_yield",
                node_type="reflection",
                label=f"Learn {learning_ratio:.0%}",
                value=max(0.0, min(1.0, learning_ratio)),
                raw_value=learning_yield,
            ))

        rumination = reflection_summary.get("rumination") or {}
        rumination_count = int(rumination.get("count", 0))
        if rumination_count > 0:
            rumination_node = rumination.get("dominant_topic") or {}
            rumination_tag = rumination_node.get("tag")
            schema.nodes.append(SchemaNode(
                node_id="reflection_rumination",
                node_type="reflection",
                label=f"Ruminate {rumination.get('ratio', 0.0):.0%}",
                value=max(0.0, min(1.0, float(rumination.get("ratio", 0.0)))),
                raw_value=rumination,
            ))
            target_id = (
                rumination_node.get("target_node_id")
                if isinstance(rumination_node, dict)
                else None
            ) or self._reflection_topic_to_node_id(rumination_tag)
            if target_id:
                schema.edges.append(SchemaEdge(
                    source_id="reflection_rumination",
                    target_id=target_id,
                    weight=max(0.2, min(1.0, float(rumination.get("ratio", 0.0)))),
                ))

        return schema
