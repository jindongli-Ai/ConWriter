"""Symbolic state graph and graph-structured inference over StoryState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ConWriter.utils.types import StoryState


@dataclass(slots=True)
class SymbolicNode:
    """One entity node in the symbolic state graph."""

    node_id: str
    label: str = "entity"
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolicEdge:
    """One typed edge in the symbolic state graph."""

    source: str
    target: str
    edge_type: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SymbolicInferenceResult:
    """Output of graph-level symbolic inference."""

    inferred_facts: List[str] = field(default_factory=list)
    propagated_constraints: List[str] = field(default_factory=list)
    conflict_candidates: List[str] = field(default_factory=list)


class SymbolicStateGraph:
    """State graph with entity nodes, relation edges, and temporal edges."""

    def __init__(self, state: StoryState):
        self.state = state
        self.entity_nodes: Dict[str, SymbolicNode] = {}
        self.relation_edges: List[SymbolicEdge] = []
        self.temporal_edges: List[SymbolicEdge] = []
        self._build_graph()

    def infer(self) -> SymbolicInferenceResult:
        """Infer facts/constraints/conflicts by propagating over graph structure."""
        inferred_facts: List[str] = []
        propagated_constraints: List[str] = []
        conflict_candidates: List[str] = []

        for node in self.entity_nodes.values():
            status = node.attributes.get("status", "unknown")
            location = node.attributes.get("location", "unknown")
            inferred_facts.append(f"entity_state:{node.node_id}:{status}@{location}")
            if status.lower() in {"dead", "removed"}:
                propagated_constraints.append(f"forbid_actor:{node.node_id}")
            if location.lower() in {"unknown", "unspecified"}:
                propagated_constraints.append(f"require_location_resolution:{node.node_id}")

        relation_map: Dict[Tuple[str, str], str] = {}
        for edge in self.relation_edges:
            relation_value = edge.attributes.get("relation", "")
            inferred_facts.append(
                f"relation:{edge.source}->{edge.target}:{relation_value or edge.edge_type}"
            )
            relation_map[(edge.source, edge.target)] = relation_value
            reverse = relation_map.get((edge.target, edge.source))
            if reverse and reverse != relation_value:
                conflict_candidates.append(
                    f"relation_conflict:{edge.source}<->{edge.target}:{relation_value}!={reverse}"
                )
            lowered = relation_value.lower()
            if lowered in {"hostile", "enemy", "betrayed"}:
                propagated_constraints.append(f"avoid_joint_action:{edge.source}:{edge.target}")

        # Temporal propagation from recent event chain edges.
        for edge in self.temporal_edges:
            inferred_facts.append(f"temporal:{edge.source}->{edge.target}:{edge.edge_type}")
        if self.temporal_edges:
            propagated_constraints.append("maintain_temporal_monotonicity")

        last_order = int(self.state.timeline.last_event_order)
        if last_order >= 0:
            propagated_constraints.append(f"next_event_order_min:{last_order + 1}")

        # State-level conflict checks over graph nodes.
        for node in self.entity_nodes.values():
            status = node.attributes.get("status", "").lower()
            location = node.attributes.get("location", "").lower()
            if status in {"dead", "removed"} and location not in {"", "unknown", "graveyard"}:
                conflict_candidates.append(
                    f"state_conflict:{node.node_id}:inactive_but_located:{location}"
                )

        return SymbolicInferenceResult(
            inferred_facts=sorted(set(item for item in inferred_facts if item)),
            propagated_constraints=sorted(set(item for item in propagated_constraints if item)),
            conflict_candidates=sorted(set(item for item in conflict_candidates if item)),
        )

    def _build_graph(self) -> None:
        for entity_id, snapshot in self.state.character_states.items():
            self.entity_nodes[entity_id] = SymbolicNode(
                node_id=entity_id,
                label="entity",
                attributes={
                    "name": snapshot.name,
                    "status": snapshot.status,
                    "location": snapshot.location,
                    "last_event_order": str(snapshot.last_event_order),
                },
            )

        for src, rels in self.state.relations.items():
            for tgt, rel in rels.items():
                self.relation_edges.append(
                    SymbolicEdge(
                        source=src,
                        target=tgt,
                        edge_type="relation",
                        attributes={"relation": str(rel)},
                    )
                )

        events = list(self.state.timeline.recent_event_ids)
        for idx in range(1, len(events)):
            self.temporal_edges.append(
                SymbolicEdge(
                    source=events[idx - 1],
                    target=events[idx],
                    edge_type="before",
                )
            )
