"""Unified experiment variant spec and policy helpers.

This module freezes method-core behavior and exposes controlled switches for
ablation and diagnosis experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ConWriter.config.schema import VariantConfig


_GENERATION_LAYER = {
    "plain_generate_only",
    "plan_only",
    "plan_plus_memory",
    "plan_memory_patch",
}
_REPAIR_LAYER = {
    "scene_rewrite_only",
    "sentence_patch_only",
    "sentence_plus_paragraph_patch",
    "full_patch_pipeline",
}
_OPT_LAYER = {
    "greedy_patch_selection",
    "lookahead_patch_selection",
    "global_aware_patch_selection",
}
_CONSTRAINT_LAYER = {
    "unweighted_constraints",
    "weighted_constraints",
    "weighted_constraints_no_future_penalty",
}
_REPLAN_LAYER = {
    "no_replan",
    "local_replan_unweighted",
    "local_replan_weighted",
}
_PRESERVATION_LAYER = {"no_preservation_check", "preservation_check_on"}
_GROUNDING_LAYER = {"basic_anchor_grounding", "confidence_aware_grounding"}
_ENTROPY_LAYER = {
    "entropy_monitor_off",
    "entropy_monitor_on",
    "entropy_monitor_no_scope_escalation",
    "entropy_monitor_no_replan_trigger",
    "entropy_monitor_validation_only",
}


@dataclass(slots=True)
class ExperimentVariantSpec:
    """Normalized variant switches for one run."""

    variant_name: str = "full_default"
    generation_layer: str = "plan_memory_patch"
    repair_granularity: str = "full_patch_pipeline"
    optimization_layer: str = "global_aware_patch_selection"
    constraint_layer: str = "weighted_constraints"
    replan_layer: str = "local_replan_weighted"
    preservation_layer: str = "preservation_check_on"
    grounding_layer: str = "confidence_aware_grounding"
    entropy_layer: str = "entropy_monitor_off"

    @classmethod
    def from_config(cls, cfg: VariantConfig) -> "ExperimentVariantSpec":
        spec = cls(
            variant_name=cfg.variant_name,
            generation_layer=cfg.generation_layer,
            repair_granularity=cfg.repair_granularity,
            optimization_layer=cfg.optimization_layer,
            constraint_layer=cfg.constraint_layer,
            replan_layer=cfg.replan_layer,
            preservation_layer=cfg.preservation_layer,
            grounding_layer=cfg.grounding_layer,
            entropy_layer=cfg.entropy_layer,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.generation_layer not in _GENERATION_LAYER:
            raise ValueError(f"Unknown generation_layer={self.generation_layer}")
        if self.repair_granularity not in _REPAIR_LAYER:
            raise ValueError(f"Unknown repair_granularity={self.repair_granularity}")
        if self.optimization_layer not in _OPT_LAYER:
            raise ValueError(f"Unknown optimization_layer={self.optimization_layer}")
        if self.constraint_layer not in _CONSTRAINT_LAYER:
            raise ValueError(f"Unknown constraint_layer={self.constraint_layer}")
        if self.replan_layer not in _REPLAN_LAYER:
            raise ValueError(f"Unknown replan_layer={self.replan_layer}")
        if self.preservation_layer not in _PRESERVATION_LAYER:
            raise ValueError(f"Unknown preservation_layer={self.preservation_layer}")
        if self.grounding_layer not in _GROUNDING_LAYER:
            raise ValueError(f"Unknown grounding_layer={self.grounding_layer}")
        if self.entropy_layer not in _ENTROPY_LAYER:
            raise ValueError(f"Unknown entropy_layer={self.entropy_layer}")

    def flags(self) -> Dict[str, bool]:
        """Derived behavior flags used by pipeline without core method changes."""
        return {
            "generation_layer": self.generation_layer,
            "repair_granularity": self.repair_granularity,
            "optimization_layer": self.optimization_layer,
            "constraint_layer": self.constraint_layer,
            "replan_layer": self.replan_layer,
            "preservation_layer": self.preservation_layer,
            "grounding_layer": self.grounding_layer,
            "entropy_layer": self.entropy_layer,
            "use_patch_pipeline": self.repair_granularity in {
                "sentence_patch_only",
                "sentence_plus_paragraph_patch",
                "full_patch_pipeline",
            },
            "scene_rewrite_only": self.repair_granularity == "scene_rewrite_only",
            "sentence_patch_only": self.repair_granularity == "sentence_patch_only",
            "paragraph_patch_enabled": self.repair_granularity in {
                "sentence_plus_paragraph_patch",
                "full_patch_pipeline",
            },
            "global_aware_enabled": self.optimization_layer == "global_aware_patch_selection",
            "lookahead_enabled": self.optimization_layer in {
                "lookahead_patch_selection",
                "global_aware_patch_selection",
            },
            "weighted_constraints_enabled": self.constraint_layer != "unweighted_constraints",
            "future_penalty_enabled": self.constraint_layer == "weighted_constraints",
            "replan_enabled": self.replan_layer != "no_replan",
            "weighted_replan": self.replan_layer == "local_replan_weighted",
            "preservation_check": self.preservation_layer == "preservation_check_on",
            "confidence_aware_grounding": self.grounding_layer == "confidence_aware_grounding",
            "entropy_monitor_enabled": self.entropy_layer != "entropy_monitor_off",
            "entropy_validation_only": self.entropy_layer == "entropy_monitor_validation_only",
            "entropy_scope_escalation_enabled": self.entropy_layer in {
                "entropy_monitor_on",
                "entropy_monitor_no_replan_trigger",
            },
            "entropy_replan_trigger_enabled": self.entropy_layer in {
                "entropy_monitor_on",
                "entropy_monitor_no_scope_escalation",
            },
            "plan_memory_patch_mode": self.generation_layer == "plan_memory_patch",
            "plain_generate_only_mode": self.generation_layer == "plain_generate_only",
        }
