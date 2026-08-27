"""Graph helper for relation-centric memory summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class MemoryGraph:
    """Simple adjacency graph to track entity relations over time."""

    edges: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_relations(cls, relations: Dict[str, Dict[str, str]]) -> "MemoryGraph":
        """Create graph object from nested relation mapping."""
        copied: Dict[str, Dict[str, str]] = {}
        for source, targets in relations.items():
            copied[source] = dict(targets)
        return cls(edges=copied)

    def add_relation(self, source: str, target: str, relation: str) -> None:
        """Add or overwrite a directed relation edge."""
        if source not in self.edges:
            self.edges[source] = {}
        self.edges[source][target] = relation

    def merge(self, relation_updates: Dict[str, Dict[str, str]]) -> None:
        """Merge nested relation updates into graph."""
        for source, targets in relation_updates.items():
            if source not in self.edges:
                self.edges[source] = {}
            for target, relation in targets.items():
                self.edges[source][target] = str(relation)

    def relation_count(self) -> int:
        """Return total number of directed edges."""
        return sum(len(v) for v in self.edges.values())

    def as_summary(self) -> Dict[str, object]:
        """Return compact relation-graph summary."""
        nodes = set(self.edges.keys())
        for targets in self.edges.values():
            nodes.update(targets.keys())
        return {
            "nodes": sorted(nodes),
            "edge_count": self.relation_count(),
            "edges": self.edges,
        }

    def neighbors(self, node: str) -> List[str]:
        """Return outgoing neighbors for one node."""
        return list(self.edges.get(node, {}).keys())
