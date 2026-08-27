"""Policy adapters for ablation variants.

No method-core algorithm changes are introduced here. Policies only switch
runtime behavior of existing components to support reproducible ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ConWriter.config.schema import ConWriterConfig
from ConWriter.pipeline.weighted_constraints import weight_violation
from ConWriter.utils.types import ConstraintViolation


@dataclass(slots=True)
class RepairGranularityPolicy:
    """Resolve sentence/paragraph patch budgets for one repair strategy."""

    name: str = "full_patch_pipeline"

    def resolve_patch_budgets(
        self,
        max_sentence_patch_rounds: int,
        max_paragraph_patch_rounds: int,
    ) -> Tuple[int, int]:
        return max(0, int(max_sentence_patch_rounds)), max(0, int(max_paragraph_patch_rounds))


@dataclass(slots=True)
class SceneRewriteRepairer:
    """Baseline: bypass sentence/paragraph patch and prefer scene rewrite."""

    enabled: bool = True
    name: str = "scene_rewrite_only"

    def resolve_patch_budgets(
        self,
        max_sentence_patch_rounds: int,
        max_paragraph_patch_rounds: int,
    ) -> Tuple[int, int]:
        _ = max_sentence_patch_rounds
        _ = max_paragraph_patch_rounds
        return 0, 0


@dataclass(slots=True)
class GreedyPatchPlanner:
    """Baseline: greedy one-step patch selection without future penalty."""

    enabled: bool = True
    name: str = "greedy_patch_selection"

    def configure(self, planner: object, config: ConWriterConfig) -> None:
        planner.lookahead_top_k = 1
        planner.lambda_future_conflict = 0.0
        _ = config


@dataclass(slots=True)
class GlobalAwarePatchPlanner:
    """Full method: lookahead + global future-conflict objective."""

    enabled: bool = True
    name: str = "global_aware_patch_selection"

    def configure(self, planner: object, config: ConWriterConfig) -> None:
        planner.lookahead_top_k = max(2, int(config.generation_controls.patch_lookahead_top_k))
        planner.lambda_future_conflict = 3.2


@dataclass(slots=True)
class LookaheadPatchPlanner:
    """Ablation: lookahead objective without future-conflict penalty."""

    enabled: bool = True
    name: str = "lookahead_patch_selection"

    def configure(self, planner: object, config: ConWriterConfig) -> None:
        planner.lookahead_top_k = max(2, int(config.generation_controls.patch_lookahead_top_k))
        planner.lambda_future_conflict = 0.0


@dataclass(slots=True)
class UnweightedConstraintPolicy:
    """Baseline: flatten all violations to uniform unit cost."""

    enabled: bool = True
    name: str = "unweighted_constraints"

    def project(self, violations: Sequence[ConstraintViolation]) -> List[ConstraintViolation]:
        normalized: List[ConstraintViolation] = []
        for item in violations:
            item.constraint_weight = 1.0
            item.constraint_tier = 3
            item.is_hard = item.severity.lower() == "error"
            item.weighted_cost = 1.0
            item.weighted_priority_source = "uniform_unweighted"
            for anchor in item.anchors:
                anchor.constraint_weight = 1.0
                anchor.constraint_tier = 3
            normalized.append(item)
        return normalized


@dataclass(slots=True)
class WeightedConstraintPolicy:
    """Default: preserve weighted fields as first-class optimization target."""

    enabled: bool = True
    name: str = "weighted_constraints"

    def project(self, violations: Sequence[ConstraintViolation]) -> List[ConstraintViolation]:
        return [weight_violation(v) for v in list(violations)]


@dataclass(slots=True)
class NoReplanPolicy:
    """Baseline: disable local replan stage."""

    enabled: bool = True
    name: str = "no_replan"

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def weighted(self) -> bool:
        return False


@dataclass(slots=True)
class WeightedReplanPolicy:
    """Default: enable local replan preserving high-weight constraints."""

    enabled: bool = True
    name: str = "local_replan_weighted"

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def weighted(self) -> bool:
        return True


@dataclass(slots=True)
class UnweightedReplanPolicy:
    """Ablation: enable replan but without weighted future constraints."""

    enabled: bool = True
    name: str = "local_replan_unweighted"

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def weighted(self) -> bool:
        return False


@dataclass(slots=True)
class VariantPolicyBundle:
    """Resolved policy objects for one variant run."""

    repair: RepairGranularityPolicy | SceneRewriteRepairer
    patch_planner: GreedyPatchPlanner | LookaheadPatchPlanner | GlobalAwarePatchPlanner
    constraints: UnweightedConstraintPolicy | WeightedConstraintPolicy
    replan: NoReplanPolicy | UnweightedReplanPolicy | WeightedReplanPolicy


def build_variant_policy_bundle(flags: Dict[str, bool]) -> VariantPolicyBundle:
    """Resolve standardized ablation policies from unified flags."""
    repair_layer = str(flags.get("repair_granularity", "full_patch_pipeline"))
    optimization_layer = str(flags.get("optimization_layer", "global_aware_patch_selection"))
    constraint_layer = str(flags.get("constraint_layer", "weighted_constraints"))
    replan_layer = str(flags.get("replan_layer", "local_replan_weighted"))

    if repair_layer == "scene_rewrite_only":
        repair = SceneRewriteRepairer()
    else:
        repair = RepairGranularityPolicy(name=repair_layer)

    if optimization_layer == "greedy_patch_selection":
        patch_planner = GreedyPatchPlanner()
    elif optimization_layer == "lookahead_patch_selection":
        patch_planner = LookaheadPatchPlanner()
    else:
        patch_planner = GlobalAwarePatchPlanner()

    constraints = (
        UnweightedConstraintPolicy()
        if constraint_layer == "unweighted_constraints"
        else WeightedConstraintPolicy()
    )

    if replan_layer == "no_replan":
        replan = NoReplanPolicy()
    elif replan_layer == "local_replan_unweighted":
        replan = UnweightedReplanPolicy()
    else:
        replan = WeightedReplanPolicy()

    return VariantPolicyBundle(
        repair=repair,
        patch_planner=patch_planner,
        constraints=constraints,
        replan=replan,
    )
