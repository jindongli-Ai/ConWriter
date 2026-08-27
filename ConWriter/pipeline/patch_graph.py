"""Patch dependency graph for dependency-aware local repair planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from ConWriter.utils.types import ConstraintViolation, SceneExtraction


@dataclass(slots=True)
class PatchDependencyGraph:
    """Dependency graph over sentence nodes for coordinated patch planning."""

    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    relation_reasons: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        violations: Sequence[ConstraintViolation],
        extraction: SceneExtraction,
    ) -> "PatchDependencyGraph":
        graph = cls()
        sentence_ids = [u.sentence_id for u in extraction.sentences]
        for sid in sentence_ids:
            graph.nodes.add(sid)
            graph.edges.setdefault(sid, set())

        for i, lhs in enumerate(violations):
            lhs_sents = cls._violation_sentence_ids(lhs)
            lhs_entities, lhs_events, lhs_rels, lhs_temporal = cls._violation_signature(lhs)
            for rhs in violations[i + 1 :]:
                rhs_sents = cls._violation_sentence_ids(rhs)
                rhs_entities, rhs_events, rhs_rels, rhs_temporal = cls._violation_signature(rhs)
                reasons: List[str] = []
                if lhs_entities.intersection(rhs_entities):
                    reasons.append("entity_overlap")
                if lhs_events.intersection(rhs_events):
                    reasons.append("event_overlap")
                if lhs_rels.intersection(rhs_rels):
                    reasons.append("relation_overlap")
                if lhs_temporal and rhs_temporal:
                    reasons.append("temporal_overlap")
                if not reasons:
                    continue
                for sid1 in lhs_sents:
                    for sid2 in rhs_sents:
                        if sid1 == sid2:
                            continue
                        graph._add_edge(sid1, sid2, reasons)

        return graph

    def connected_components(self) -> List[Set[str]]:
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for node in sorted(self.nodes):
            if node in visited:
                continue
            stack = [node]
            comp: Set[str] = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for nxt in sorted(self.edges.get(cur, set())):
                    if nxt not in visited:
                        stack.append(nxt)
            components.append(comp)
        return components

    def suggest_joint_target_sets(
        self,
        base_targets: Sequence[str],
        max_targets: int,
    ) -> List[List[str]]:
        base = [sid for sid in base_targets if sid in self.nodes]
        if not base:
            return []
        comps = self.connected_components()
        suggestions: List[List[str]] = []
        for comp in comps:
            hit = sorted(set(base).intersection(comp))
            if not hit:
                continue
            expanded = sorted(comp)[: max_targets]
            if expanded:
                suggestions.append(expanded)
            if hit:
                suggestions.append(hit[:max_targets])
        dedup: List[List[str]] = []
        seen: Set[str] = set()
        for row in suggestions:
            key = "|".join(sorted(set(row)))
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(sorted(set(row))[:max_targets])
        return dedup

    def _add_edge(self, sid1: str, sid2: str, reasons: Sequence[str]) -> None:
        self.edges.setdefault(sid1, set()).add(sid2)
        self.edges.setdefault(sid2, set()).add(sid1)
        key = tuple(sorted((sid1, sid2)))
        row = self.relation_reasons.setdefault(key, [])
        for reason in reasons:
            if reason not in row:
                row.append(reason)

    @staticmethod
    def _violation_sentence_ids(violation: ConstraintViolation) -> Set[str]:
        ids: Set[str] = set()
        for anchor in violation.anchors:
            ids.update(anchor.sentence_ids)
        return ids

    @staticmethod
    def _violation_signature(violation: ConstraintViolation) -> Tuple[Set[str], Set[str], Set[str], bool]:
        entities: Set[str] = set()
        events: Set[str] = set()
        relations: Set[str] = set()
        temporal = False
        for anchor in violation.anchors:
            entities.update(anchor.related_entity_ids)
            events.update(anchor.related_event_ids)
            relations.update(anchor.related_relation_ids)
            temporal = temporal or bool(anchor.temporal_conflicts)
        return entities, events, relations, temporal

