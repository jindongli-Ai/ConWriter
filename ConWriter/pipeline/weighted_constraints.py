"""Weighted-constraint formalization utilities.

Centralizes tier/weight assignment so violations and constraints are optimized
as first-class weighted targets, not only counted by severity.
"""

from __future__ import annotations

from typing import Iterable, List

from ConWriter.utils.types import ConstraintViolation, TransitionConstraint, WeightedConstraintItem


def weight_violation(violation: ConstraintViolation) -> ConstraintViolation:
    """Assign tier/weight/cost fields in-place and return the same object."""
    message = (violation.message or "").lower()
    rule = (violation.rule_type or "").lower()
    facet = (violation.facet or "").lower()
    source = "default"
    tier = 3
    weight = 1.0
    is_hard = violation.severity.lower() == "error"

    if (
        "dead" in message
        or "life-state" in message
        or "identity" in message
        or "hard required" in message
        or "missing required events" in message
        or "core temporal" in message
        or "event order regression" in message
        or ("operator_validity" in rule and "precondition" in message)
    ):
        tier = 1
        weight = 8.0
        is_hard = True
        source = "life_state_or_core_temporal_or_required_event"
    elif (
        "relation" in message
        or "important entity-state" in message
        or ("character_consistency" in rule and violation.severity.lower() == "error")
        or ("timeline_plot" in facet and violation.severity.lower() == "error")
    ):
        tier = 2
        weight = 4.0
        source = "relation_or_entity_state_or_plan_critical"
    elif (
        "style" in facet
        or "descriptive" in message
        or violation.severity.lower() == "warning"
    ):
        tier = 3
        weight = 1.5
        source = "descriptive_or_soft_preference"
    if any(anchor.confidence_score < 0.35 for anchor in violation.anchors):
        tier = max(tier, 4)
        weight = min(weight, 0.6)
        source = "low_confidence_heuristic"
    if violation.needs_replan or violation.fatal:
        tier = 1
        weight = max(weight, 9.0)
        is_hard = True
        source = "fatal_or_replan_required"

    violation.constraint_tier = int(tier)
    violation.constraint_weight = float(weight)
    violation.is_hard = bool(is_hard)
    violation.weighted_priority_source = source
    violation.weighted_cost = float(weight) * (1.25 if is_hard else 1.0)
    for anchor in violation.anchors:
        anchor.constraint_tier = violation.constraint_tier
        anchor.constraint_weight = violation.constraint_weight
    return violation


def weight_transition_constraint(constraint: TransitionConstraint) -> TransitionConstraint:
    """Assign weight fields for structured constraints (in-place)."""
    predicate = (constraint.predicate or "").lower()
    facet = (constraint.facet or "").lower()
    tier = 3
    weight = 1.0
    source = "state_reasoning"
    is_hard = constraint.severity.lower() == "error"

    if predicate in {
        "min_next_event_order",
        "character_must_be_introduced_or_acted",
        "character_cannot_be_actor",
    }:
        tier = 1
        weight = 7.0
        is_hard = True
        source = "core_temporal_or_identity"
    elif predicate in {"dependency_scene_should_be_reflected", "graph_propagated_constraint"}:
        tier = 2
        weight = 4.0
        source = "plan_critical_dependency"
    elif facet == "narrative_style":
        tier = 4
        weight = 0.8
        source = "style_preference"

    constraint.constraint_tier = int(tier)
    constraint.constraint_weight = float(weight)
    constraint.is_hard = bool(is_hard)
    constraint.weighted_priority_source = source
    return constraint


def weighted_violation_score(violations: Iterable[ConstraintViolation]) -> float:
    """Weighted sum objective over remaining violations."""
    total = 0.0
    for item in violations:
        total += float(item.weighted_cost if item.weighted_cost > 0 else item.constraint_weight)
    return float(total)


def build_weighted_tiered_constraints(
    required: Iterable[str],
    must_keep: Iterable[str],
    forbidden: Iterable[str],
    inferred: Iterable[str],
    propagated: Iterable[str],
    high_conf_violations: Iterable[str],
    deferred_constraints: Iterable[str],
) -> List[WeightedConstraintItem]:
    """Build normalized tiered weighted constraints for generation conditioning."""
    out: List[WeightedConstraintItem] = []
    for text in list(required) + list(must_keep):
        txt = str(text).strip()
        if txt:
            out.append(WeightedConstraintItem(text=txt, weight=8.0, tier=1, is_hard=True, source="required"))
    for text in forbidden:
        txt = str(text).strip()
        if txt:
            out.append(WeightedConstraintItem(text=f"FORBID: {txt}", weight=8.0, tier=1, is_hard=True, source="forbidden"))
    for text in high_conf_violations:
        txt = str(text).strip()
        if txt:
            out.append(WeightedConstraintItem(text=txt, weight=5.0, tier=2, is_hard=False, source="high_conf_violation"))
    for text in list(inferred) + list(propagated):
        txt = str(text).strip()
        if txt:
            out.append(WeightedConstraintItem(text=txt, weight=2.5, tier=2, is_hard=False, source="inferred"))
    for text in deferred_constraints:
        txt = str(text).strip()
        if txt:
            out.append(WeightedConstraintItem(text=txt, weight=0.8, tier=3, is_hard=False, source="deferred"))
    # Stable deterministic ordering.
    out.sort(key=lambda x: (-x.weight, x.tier, x.text))
    dedup = {}
    for item in out:
        dedup.setdefault(item.text, item)
    return list(dedup.values())

