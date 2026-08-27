"""Diagnostics record builders for ablation-ready experiment analysis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

from ConWriter.utils.types import ConstraintViolation, EntropyRiskProfile, PatchPlan


def build_scene_diagnostic_record(
    *,
    prompt_id: str,
    scene_id: str,
    variant_name: str,
    accepted: bool,
    violations: Iterable[ConstraintViolation],
    patch_rounds: int,
    paragraph_patch_used: bool,
    scene_regen_used: bool,
    scene_regen_rounds: int,
    scene_patch_attempted_targeted: bool = False,
    scene_patch_exhausted_targeted: bool = False,
    local_replan_triggered: bool,
    future_conflict_penalty: float,
    unchanged_ratio: float,
    preservation_failures: int,
    oscillation_detected: bool,
    final_repair_scope: str,
    final_objective_breakdown: Dict[str, float],
    patch_plan: PatchPlan | None = None,
    entropy_profile: EntropyRiskProfile | None = None,
    entropy_triggered_validation: bool = False,
    entropy_triggered_patch_escalation: bool = False,
    entropy_triggered_replan: bool = False,
    overlap_high_entropy_violation: float = 0.0,
    entropy_validation_mode: str = "standard",
    entropy_validation_budget: int = 1,
    delta_uncertainty: float = 0.0,
    sentence_uncertainty_variance: float = 0.0,
    round_uncertainty_trend: float = 0.0,
    joint_action_events: List[Dict[str, object]] | None = None,
    patch_execution_records: List[Dict[str, object]] | None = None,
    memory_binding_mode: str = "normal_binding",
    generation_control_mode: str = "normal_generation",
    memory_binding_decision_reasons: List[str] | None = None,
    strengthened_memory_blocks: List[str] | None = None,
    strengthened_constraints: List[str] | None = None,
    rewrite_memory_binding_mode: str = "normal_binding",
    rewrite_generation_control_mode: str = "normal_generation",
    generation_control_context: Dict[str, object] | None = None,
    rewrite_control_context: Dict[str, object] | None = None,
    dynamic_memory_update_status: Dict[str, object] | None = None,
    retrieved_experience_count: int = 0,
    retrieved_experience_items: List[Dict[str, object]] | None = None,
    desired_target_length: int = 0,
    requested_target_length: int = 0,
    length_compensation_factor: float = 1.0,
    length_progress_ratio: float = 0.0,
    scene_word_count_first_pass: int = 0,
    scene_too_short_guard_triggered: bool = False,
    scene_too_short_guard_threshold: int = 50,
    under_generation_warning_triggered: bool = False,
    premature_closure_warning_triggered: bool = False,
    entropy_validation_only_zero_influence: bool = True,
) -> Dict[str, object]:
    items = list(violations)
    facet_breakdown: Dict[str, int] = {}
    tier_breakdown: Dict[str, int] = {}
    for row in items:
        facet = row.facet or "unscoped"
        tier = str(int(row.constraint_tier))
        facet_breakdown[facet] = facet_breakdown.get(facet, 0) + 1
        tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1

    entropy_profile = entropy_profile or EntropyRiskProfile()
    return {
        "prompt_id": prompt_id,
        "scene_id": scene_id,
        "variant_name": variant_name,
        "accepted": bool(accepted),
        "rejected": not bool(accepted),
        "number_of_violations": len(items),
        "weighted_violation_score": float(sum(v.weighted_cost for v in items)),
        "violation_facet_breakdown": facet_breakdown,
        "violation_tier_breakdown": tier_breakdown,
        "patch_rounds": int(patch_rounds),
        "paragraph_patch_used": bool(paragraph_patch_used),
        "scene_regenerate_used": bool(scene_regen_used),
        "scene_regenerate_rounds": int(max(0, scene_regen_rounds)),
        "scene_patch_attempted_targeted": bool(scene_patch_attempted_targeted),
        "scene_patch_exhausted_targeted": bool(scene_patch_exhausted_targeted),
        "local_replan_triggered": bool(local_replan_triggered),
        "future_conflict_penalty": float(future_conflict_penalty),
        "unchanged_ratio": float(unchanged_ratio),
        "preservation_failures": int(preservation_failures),
        "oscillation_detected": bool(oscillation_detected),
        "final_repair_scope": final_repair_scope,
        "final_objective_breakdown": dict(final_objective_breakdown),
        "patch_plan": asdict(patch_plan) if patch_plan is not None else None,
        "patch_target_hits_violation_context": bool(
            getattr(patch_plan, "patch_target_hits_violation_context", False) if patch_plan is not None else False
        ),
        "patch_target_hits_entropy_context": bool(
            getattr(patch_plan, "patch_target_hits_entropy_context", False) if patch_plan is not None else False
        ),
        "patch_target_joint_alignment_score": float(
            getattr(patch_plan, "patch_target_joint_alignment_score", 0.0) if patch_plan is not None else 0.0
        ),
        "patch_alignment_score_breakdown": dict(
            getattr(patch_plan, "patch_alignment_score_breakdown", {}) if patch_plan is not None else {}
        ),
        "patch_rewrites_key_conflict_span": bool(
            getattr(patch_plan, "patch_rewrites_key_conflict_span", False) if patch_plan is not None else False
        ),
        "patch_changes_symbolic_state_proxy": bool(
            getattr(patch_plan, "patch_changes_symbolic_state_proxy", False) if patch_plan is not None else False
        ),
        "patch_reduces_transition_violations": bool(
            getattr(patch_plan, "patch_reduces_transition_violations", False) if patch_plan is not None else False
        ),
        "patch_reduces_constraint_violations": bool(
            getattr(patch_plan, "patch_reduces_constraint_violations", False) if patch_plan is not None else False
        ),
        "patch_reduces_uncertainty": bool(
            getattr(patch_plan, "patch_reduces_uncertainty", False) if patch_plan is not None else False
        ),
        "patch_effectiveness_label": str(
            getattr(patch_plan, "patch_effectiveness_label", "unknown") if patch_plan is not None else "unknown"
        ),
        "patch_no_gain_reason": str(
            getattr(patch_plan, "patch_no_gain_reason", "") if patch_plan is not None else ""
        ),
        "rewrite_conflict_type": str(
            getattr(patch_plan, "rewrite_conflict_type", "unknown") if patch_plan is not None else "unknown"
        ),
        "rewrite_target_scope": str(
            getattr(patch_plan, "rewrite_target_scope", "sentence") if patch_plan is not None else "sentence"
        ),
        "rewrite_hits_required_state_change": bool(
            getattr(patch_plan, "rewrite_hits_required_state_change", False) if patch_plan is not None else False
        ),
        "rewrite_removes_conflicting_state": bool(
            getattr(patch_plan, "rewrite_removes_conflicting_state", False) if patch_plan is not None else False
        ),
        "rewrite_preserves_non_conflict_content": bool(
            getattr(patch_plan, "rewrite_preserves_non_conflict_content", True) if patch_plan is not None else True
        ),
        "patch_execution_id": str(
            getattr(patch_plan, "patch_execution_id", "") if patch_plan is not None else ""
        ),
        "patch_execution_status": str(
            getattr(patch_plan, "patch_execution_status", "not_applied") if patch_plan is not None else "not_applied"
        ),
        "patch_execution_round": int(
            getattr(patch_plan, "patch_execution_round", 0) if patch_plan is not None else 0
        ),
        "patch_execution_scope": str(
            getattr(patch_plan, "patch_execution_scope", "") if patch_plan is not None else ""
        ),
        "patch_execution_applied": bool(
            getattr(patch_plan, "patch_execution_applied", False) if patch_plan is not None else False
        ),
        "patch_execution_skipped_reason": str(
            getattr(patch_plan, "patch_execution_skipped_reason", "") if patch_plan is not None else ""
        ),
        "rewrite_targets_execution_spec_conflict": bool(
            getattr(patch_plan, "rewrite_targets_execution_spec_conflict", False) if patch_plan is not None else False
        ),
        "rewrite_targets_required_state_change": bool(
            getattr(patch_plan, "rewrite_targets_required_state_change", False) if patch_plan is not None else False
        ),
        "rewrite_targets_transition_conflict": bool(
            getattr(patch_plan, "rewrite_targets_transition_conflict", False) if patch_plan is not None else False
        ),
        "rewrite_targets_operator_post_state_conflict": bool(
            getattr(patch_plan, "rewrite_targets_operator_post_state_conflict", False) if patch_plan is not None else False
        ),
        "rewrite_memory_binding_mode": str(
            getattr(patch_plan, "rewrite_memory_binding_mode", "normal_binding")
            if patch_plan is not None
            else "normal_binding"
        ),
        "rewrite_generation_control_mode": str(
            getattr(patch_plan, "rewrite_generation_control_mode", "normal_generation")
            if patch_plan is not None
            else "normal_generation"
        ),
        "rewrite_generation_control_context": dict(
            getattr(patch_plan, "rewrite_generation_control_context", {}) if patch_plan is not None else {}
        ),
        "rewrite_binding_decision_reasons": list(
            getattr(patch_plan, "rewrite_binding_decision_reasons", []) if patch_plan is not None else []
        ),
        "rewrite_strengthened_memory_blocks": list(
            getattr(patch_plan, "rewrite_strengthened_memory_blocks", []) if patch_plan is not None else []
        ),
        "rewrite_strengthened_constraints": list(
            getattr(patch_plan, "rewrite_strengthened_constraints", []) if patch_plan is not None else []
        ),
        "rewrite_realizes_required_state_change": bool(
            getattr(patch_plan, "rewrite_realizes_required_state_change", False) if patch_plan is not None else False
        ),
        "rewrite_removes_forbidden_state": bool(
            getattr(patch_plan, "rewrite_removes_forbidden_state", False) if patch_plan is not None else False
        ),
        "rewrite_restores_transition_coherence_proxy": bool(
            getattr(patch_plan, "rewrite_restores_transition_coherence_proxy", False) if patch_plan is not None else False
        ),
        "rewrite_realizes_operator_post_state": bool(
            getattr(patch_plan, "rewrite_realizes_operator_post_state", False) if patch_plan is not None else False
        ),
        "state_realization_match_type": str(
            getattr(patch_plan, "state_realization_match_type", "no_match")
            if patch_plan is not None
            else "no_match"
        ),
        "forbidden_state_removal_match_type": str(
            getattr(patch_plan, "forbidden_state_removal_match_type", "no_match")
            if patch_plan is not None
            else "no_match"
        ),
        "operator_post_state_match_type": str(
            getattr(patch_plan, "operator_post_state_match_type", "no_match")
            if patch_plan is not None
            else "no_match"
        ),
        "canonical_required_states": list(
            getattr(patch_plan, "canonical_required_states", []) if patch_plan is not None else []
        ),
        "canonical_forbidden_states": list(
            getattr(patch_plan, "canonical_forbidden_states", []) if patch_plan is not None else []
        ),
        "canonical_operator_post_states": list(
            getattr(patch_plan, "canonical_operator_post_states", []) if patch_plan is not None else []
        ),
        "grounded_alias_matches": list(
            getattr(patch_plan, "grounded_alias_matches", []) if patch_plan is not None else []
        ),
        "first_pass_required_state_checklist": list(
            getattr(patch_plan, "first_pass_required_state_checklist", []) if patch_plan is not None else []
        ),
        "first_pass_forbidden_state_checklist": list(
            getattr(patch_plan, "first_pass_forbidden_state_checklist", []) if patch_plan is not None else []
        ),
        "first_pass_operator_post_state_checklist": list(
            getattr(patch_plan, "first_pass_operator_post_state_checklist", []) if patch_plan is not None else []
        ),
        "retry_required_state_checklist": list(
            getattr(patch_plan, "retry_required_state_checklist", []) if patch_plan is not None else []
        ),
        "retry_forbidden_state_checklist": list(
            getattr(patch_plan, "retry_forbidden_state_checklist", []) if patch_plan is not None else []
        ),
        "retry_operator_post_state_checklist": list(
            getattr(patch_plan, "retry_operator_post_state_checklist", []) if patch_plan is not None else []
        ),
        "first_pass_checklist_completion_rate": float(
            getattr(patch_plan, "first_pass_checklist_completion_rate", 0.0) if patch_plan is not None else 0.0
        ),
        "retry_checklist_completion_rate": float(
            getattr(patch_plan, "retry_checklist_completion_rate", 0.0) if patch_plan is not None else 0.0
        ),
        "checklist_items_fixed_by_retry": list(
            getattr(patch_plan, "checklist_items_fixed_by_retry", []) if patch_plan is not None else []
        ),
        "checklist_items_still_unresolved": list(
            getattr(patch_plan, "checklist_items_still_unresolved", []) if patch_plan is not None else []
        ),
        "retry_preserved_satisfied_items": bool(
            getattr(patch_plan, "retry_preserved_satisfied_items", True) if patch_plan is not None else True
        ),
        "retry_slot_priority_order": list(
            getattr(patch_plan, "retry_slot_priority_order", []) if patch_plan is not None else []
        ),
        "retry_slot_priority_unresolved": dict(
            getattr(patch_plan, "retry_slot_priority_unresolved", {}) if patch_plan is not None else {}
        ),
        "slot_priority_fix_progress": dict(
            getattr(patch_plan, "slot_priority_fix_progress", {}) if patch_plan is not None else {}
        ),
        "slot_priority_preserve_result": bool(
            getattr(patch_plan, "slot_priority_preserve_result", True) if patch_plan is not None else True
        ),
        "priority_step_where_failure_remains": str(
            getattr(patch_plan, "priority_step_where_failure_remains", "") if patch_plan is not None else ""
        ),
        "slot_type_rebroken_after_retry": str(
            getattr(patch_plan, "slot_type_rebroken_after_retry", "") if patch_plan is not None else ""
        ),
        "step_aware_preservation_guard_enabled": bool(
            getattr(patch_plan, "step_aware_preservation_guard_enabled", False) if patch_plan is not None else False
        ),
        "protected_items_snapshot": dict(
            getattr(patch_plan, "protected_items_snapshot", {}) if patch_plan is not None else {}
        ),
        "protected_items_preserved_after_retry": bool(
            getattr(patch_plan, "protected_items_preserved_after_retry", True) if patch_plan is not None else True
        ),
        "protected_items_broken_after_retry": list(
            getattr(patch_plan, "protected_items_broken_after_retry", []) if patch_plan is not None else []
        ),
        "forbidden_reintroduced_after_step": bool(
            getattr(patch_plan, "forbidden_reintroduced_after_step", False) if patch_plan is not None else False
        ),
        "operator_post_state_weakened_after_step": bool(
            getattr(patch_plan, "operator_post_state_weakened_after_step", False) if patch_plan is not None else False
        ),
        "required_state_regressed_after_step": bool(
            getattr(patch_plan, "required_state_regressed_after_step", False) if patch_plan is not None else False
        ),
        "scope_expansion_triggered": bool(
            getattr(patch_plan, "scope_expansion_triggered", False) if patch_plan is not None else False
        ),
        "original_scope": str(
            getattr(patch_plan, "original_scope", "") if patch_plan is not None else ""
        ),
        "expanded_scope": str(
            getattr(patch_plan, "expanded_scope", "") if patch_plan is not None else ""
        ),
        "expanded_target_sentence_ids": list(
            getattr(patch_plan, "expanded_target_sentence_ids", []) if patch_plan is not None else []
        ),
        "expanded_local_window": dict(
            getattr(patch_plan, "expanded_local_window", {}) if patch_plan is not None else {}
        ),
        "unresolved_slot_count_before": int(
            getattr(patch_plan, "unresolved_slot_count_before", 0) if patch_plan is not None else 0
        ),
        "unresolved_slot_count_after": int(
            getattr(patch_plan, "unresolved_slot_count_after", 0) if patch_plan is not None else 0
        ),
        "checklist_items_fixed_by_expansion": list(
            getattr(patch_plan, "checklist_items_fixed_by_expansion", []) if patch_plan is not None else []
        ),
        "still_unresolved_after_expansion": list(
            getattr(patch_plan, "still_unresolved_after_expansion", []) if patch_plan is not None else []
        ),
        "expansion_preserved_satisfied_items": bool(
            getattr(patch_plan, "expansion_preserved_satisfied_items", True) if patch_plan is not None else True
        ),
        "scope_expansion_effective": bool(
            getattr(patch_plan, "scope_expansion_effective", False) if patch_plan is not None else False
        ),
        "patch_retry_attempted": bool(
            getattr(patch_plan, "patch_retry_attempted", False) if patch_plan is not None else False
        ),
        "patch_retry_scope": str(
            getattr(patch_plan, "patch_retry_scope", "") if patch_plan is not None else ""
        ),
        "patch_retry_reason": str(
            getattr(patch_plan, "patch_retry_reason", "") if patch_plan is not None else ""
        ),
        "patch_retry_conflict_type": str(
            getattr(patch_plan, "patch_retry_conflict_type", "unknown") if patch_plan is not None else "unknown"
        ),
        "patch_retry_effective": bool(
            getattr(patch_plan, "patch_retry_effective", False) if patch_plan is not None else False
        ),
        "patch_retry_realizes_required_state_change": bool(
            getattr(patch_plan, "patch_retry_realizes_required_state_change", False) if patch_plan is not None else False
        ),
        "patch_retry_removes_forbidden_state": bool(
            getattr(patch_plan, "patch_retry_removes_forbidden_state", False) if patch_plan is not None else False
        ),
        "patch_retry_restores_transition_coherence_proxy": bool(
            getattr(patch_plan, "patch_retry_restores_transition_coherence_proxy", False) if patch_plan is not None else False
        ),
        "first_pass_effective": bool(
            getattr(patch_plan, "patch_first_pass_effective", False) if patch_plan is not None else False
        ),
        "retry_effective": bool(
            getattr(patch_plan, "patch_retry_effective", False) if patch_plan is not None else False
        ),
        "still_ineffective_after_retry": bool(
            getattr(patch_plan, "patch_still_ineffective_after_retry", False) if patch_plan is not None else False
        ),
        "patch_before_transition_violation_count": int(
            getattr(patch_plan, "patch_before_transition_violation_count", 0) if patch_plan is not None else 0
        ),
        "patch_after_transition_violation_count": int(
            getattr(patch_plan, "patch_after_transition_violation_count", 0) if patch_plan is not None else 0
        ),
        "patch_before_constraint_violation_count": int(
            getattr(patch_plan, "patch_before_constraint_violation_count", 0) if patch_plan is not None else 0
        ),
        "patch_after_constraint_violation_count": int(
            getattr(patch_plan, "patch_after_constraint_violation_count", 0) if patch_plan is not None else 0
        ),
        "patch_before_symbolic_state_proxy": float(
            getattr(patch_plan, "patch_before_symbolic_state_proxy", 0.0) if patch_plan is not None else 0.0
        ),
        "patch_after_symbolic_state_proxy": float(
            getattr(patch_plan, "patch_after_symbolic_state_proxy", 0.0) if patch_plan is not None else 0.0
        ),
        "entropy_mean": float(entropy_profile.scene_entropy_mean),
        "uncertainty_source_type": str(entropy_profile.source_type),
        "uncertainty_is_proxy_signal": bool(entropy_profile.is_proxy_signal),
        "uncertainty_mode_used": str(entropy_profile.uncertainty_mode),
        "uncertainty_available": bool(entropy_profile.uncertainty_available),
        "uncertainty_truncated": bool(entropy_profile.uncertainty_truncated),
        "scene_uncertainty_mean": float(entropy_profile.scene_uncertainty_mean),
        "scene_uncertainty_peak": float(entropy_profile.scene_uncertainty_peak),
        "constraint_conditioned_uncertainty": dict(entropy_profile.constraint_conditioned_uncertainty),
        "critical_constraint_uncertainty_peak": float(entropy_profile.critical_constraint_uncertainty_peak),
        "critical_constraint_uncertainty_mean": float(entropy_profile.critical_constraint_uncertainty_mean),
        "local_constraint_uncertainty": dict(entropy_profile.local_constraint_uncertainty),
        "local_vs_sentence_uncertainty_gap": dict(entropy_profile.local_vs_sentence_uncertainty_gap),
        "delta_uncertainty": float(delta_uncertainty),
        "sentence_uncertainty_variance": float(sentence_uncertainty_variance),
        "round_uncertainty_trend": float(round_uncertainty_trend),
        "entropy_spike_count": int(len(entropy_profile.entropy_spike_indices)),
        "high_risk_sentence_count": int(len(entropy_profile.high_risk_sentence_ids)),
        "entropy_risk_tier": str(entropy_profile.final_risk_tier),
        "entropy_triggered_validation": bool(entropy_triggered_validation),
        "entropy_validation_mode": str(entropy_validation_mode),
        "entropy_validation_budget": int(max(1, entropy_validation_budget)),
        "entropy_triggered_patch_escalation": bool(entropy_triggered_patch_escalation),
        "entropy_triggered_replan": bool(entropy_triggered_replan),
        "entropy_validation_only_zero_influence": bool(entropy_validation_only_zero_influence),
        "uncertainty_control_score": float(entropy_profile.uncertainty_control_score),
        "symbolic_pressure_score": float(entropy_profile.symbolic_pressure_score),
        "memory_volatility_score": float(entropy_profile.memory_volatility_score),
        "uncertainty_contribution": float(entropy_profile.uncertainty_contribution),
        "symbolic_contribution": float(entropy_profile.symbolic_contribution),
        "memory_contribution": float(entropy_profile.memory_contribution),
        "joint_risk_score": float(entropy_profile.joint_risk_score),
        "joint_action_selector": str(entropy_profile.joint_action_selector),
        "joint_weight_template_used": str(entropy_profile.joint_weight_template_used),
        "joint_local_failure_signal": float(entropy_profile.joint_local_failure_signal),
        "joint_persistent_risk_steps": int(entropy_profile.joint_persistent_risk_steps),
        "joint_patch_failure_proxy_score": float(entropy_profile.joint_patch_failure_proxy_score),
        "joint_validation_gate_passed": bool(entropy_profile.joint_validation_gate_passed),
        "joint_patch_gate_passed": bool(entropy_profile.joint_patch_gate_passed),
        "joint_replan_gate_passed": bool(entropy_profile.joint_replan_gate_passed),
        "joint_action_events": list(joint_action_events or []),
        "patch_execution_records": list(patch_execution_records or []),
        "memory_binding_mode": str(memory_binding_mode or "normal_binding"),
        "generation_control_mode": str(generation_control_mode or "normal_generation"),
        "memory_binding_decision_reasons": list(memory_binding_decision_reasons or []),
        "strengthened_memory_blocks": list(strengthened_memory_blocks or []),
        "strengthened_constraints": list(strengthened_constraints or []),
        "rewrite_memory_binding_mode": str(rewrite_memory_binding_mode or "normal_binding"),
        "rewrite_generation_control_mode": str(rewrite_generation_control_mode or "normal_generation"),
        "generation_control_context": dict(generation_control_context or {}),
        "rewrite_control_context": dict(rewrite_control_context or {}),
        "dynamic_memory_update_status": dict(dynamic_memory_update_status or {}),
        "retrieved_experience_count": int(retrieved_experience_count),
        "retrieved_experience_items": list(retrieved_experience_items or []),
        "desired_target_length": int(desired_target_length),
        "requested_target_length": int(requested_target_length),
        "length_compensation_factor": float(length_compensation_factor),
        "length_progress_ratio": float(length_progress_ratio),
        "scene_word_count_first_pass": int(scene_word_count_first_pass),
        "scene_too_short_guard_triggered": bool(scene_too_short_guard_triggered),
        "scene_too_short_guard_threshold": int(scene_too_short_guard_threshold),
        "under_generation_warning_triggered": bool(under_generation_warning_triggered),
        "premature_closure_warning_triggered": bool(premature_closure_warning_triggered),
        "symbolic_assurance_output": {
            "number_of_violations": int(len(items)),
            "violation_facet_breakdown": dict(facet_breakdown),
            "violation_tier_breakdown": dict(tier_breakdown),
            "weighted_violation_score": float(sum(v.weighted_cost for v in items)),
            "final_judgment_owner": "symbolic_validator_checker",
        },
        "uncertainty_warning_output": {
            "uncertainty_mode_used": str(entropy_profile.uncertainty_mode),
            "scene_uncertainty_mean": float(entropy_profile.scene_uncertainty_mean),
            "critical_constraint_uncertainty_peak": float(entropy_profile.critical_constraint_uncertainty_peak),
            "joint_risk_score": float(entropy_profile.joint_risk_score),
            "warning_tier": str(entropy_profile.final_risk_tier),
            "warning_triggered_validation": bool(entropy_triggered_validation),
            "warning_triggered_patch_escalation": bool(entropy_triggered_patch_escalation),
            "warning_triggered_replan": bool(entropy_triggered_replan),
            "warning_changed_downstream_control": bool(
                (str(memory_binding_mode or "normal_binding") != "normal_binding")
                or (str(generation_control_mode or "normal_generation") != "normal_generation")
                or bool(entropy_triggered_patch_escalation)
                or bool(entropy_triggered_replan)
            ),
            "is_final_judge": False,
        },
        "overlap_between_high_entropy_and_actual_violations": float(overlap_high_entropy_violation),
        "entropy_final_risk_score": float(entropy_profile.final_risk_score),
        "entropy_suggested_action": str(entropy_profile.suggested_action),
    }


def _final_action_event(scene_record: Dict[str, object]) -> Dict[str, object]:
    events = scene_record.get("joint_action_events", [])
    if isinstance(events, list):
        for item in reversed(events):
            if isinstance(item, dict):
                return dict(item)
    selector = str(scene_record.get("joint_action_selector", "do_nothing"))
    return {
        "selected_action": selector,
        "action_requested": bool(selector != "do_nothing"),
        "action_executed": bool(selector != "do_nothing"),
        "action_blocked_count": 0,
        "action_blocked_reasons": [],
        "before_transition_violation_count": int(scene_record.get("number_of_violations", 0)),
        "after_transition_violation_count": int(scene_record.get("number_of_violations", 0)),
        "before_violation_count": int(scene_record.get("number_of_violations", 0)),
        "after_violation_count": int(scene_record.get("number_of_violations", 0)),
        "before_scene_uncertainty": float(scene_record.get("scene_uncertainty_mean", 0.0)),
        "after_scene_uncertainty": float(scene_record.get("scene_uncertainty_mean", 0.0)),
        "improved_transition_violations": False,
        "improved_violations": False,
        "improved_uncertainty": False,
        "joint_risk_score": float(scene_record.get("joint_risk_score", 0.0)),
    }


def _compute_delayed_metrics(
    records: List[Dict[str, object]],
    *,
    min_uncertainty_drop: float,
    min_joint_risk_drop: float,
) -> Dict[str, object]:
    action_names = ["do_nothing", "validation_boost", "patch", "patch_plus_escalation", "replan"]
    per_action: Dict[str, Dict[str, float]] = {}
    detail_rows: List[Dict[str, object]] = []
    for action in action_names:
        per_action[action] = {
            "trigger_count": 0.0,
            "immediate_conversion_rate": 0.0,
            "delayed_improve_rate_1": 0.0,
            "delayed_improve_rate_2": 0.0,
            "delayed_improve_violations_rate_1": 0.0,
            "delayed_improve_violations_rate_2": 0.0,
            "delayed_improve_uncertainty_rate_1": 0.0,
            "delayed_improve_uncertainty_rate_2": 0.0,
            "delayed_false_trigger_rate": 0.0,
            "avg_joint_risk_before": 0.0,
            "avg_joint_risk_after_1": 0.0,
            "avg_joint_risk_after_2": 0.0,
            "avg_violations_before": 0.0,
            "avg_violations_after_1": 0.0,
            "avg_violations_after_2": 0.0,
        }

    def _future_row(idx: int, offset: int) -> Dict[str, object] | None:
        j = idx + offset
        if j < 0 or j >= len(records):
            return None
        return records[j]

    rolling: Dict[str, Dict[str, float]] = {
        a: {
            "trigger_count": 0.0,
            "executed_count": 0.0,
            "eligible_1": 0.0,
            "eligible_2": 0.0,
            "improve_1": 0.0,
            "improve_2": 0.0,
            "improve_v_1": 0.0,
            "improve_v_2": 0.0,
            "improve_u_1": 0.0,
            "improve_u_2": 0.0,
            "false_delayed": 0.0,
            "sum_joint_before": 0.0,
            "sum_joint_after_1": 0.0,
            "sum_joint_after_2": 0.0,
            "sum_v_before": 0.0,
            "sum_v_after_1": 0.0,
            "sum_v_after_2": 0.0,
        }
        for a in action_names
    }

    for idx, row in enumerate(records):
        ev = _final_action_event(row)
        action = str(ev.get("selected_action", "do_nothing"))
        if action not in rolling:
            action = "do_nothing"
        current_joint = float(row.get("joint_risk_score", ev.get("joint_risk_score", 0.0)))
        current_unc = float(row.get("scene_uncertainty_mean", ev.get("after_scene_uncertainty", 0.0)))
        current_v = int(row.get("number_of_violations", ev.get("after_violation_count", 0)))
        requested = bool(ev.get("action_requested", False))
        executed = bool(ev.get("action_executed", False))
        if requested:
            rolling[action]["trigger_count"] += 1.0
        if requested and executed:
            rolling[action]["executed_count"] += 1.0

        future1 = _future_row(idx, 1)
        future2 = _future_row(idx, 2)
        improve1_any = False
        improve2_any = False
        improve1_v = False
        improve1_u = False
        improve2_v = False
        improve2_u = False
        if future1 is not None:
            rolling[action]["eligible_1"] += 1.0
            v1 = int(future1.get("number_of_violations", 0))
            unc1 = float(future1.get("scene_uncertainty_mean", 0.0))
            joint1 = float(future1.get("joint_risk_score", 0.0))
            improve1_v = v1 < current_v
            improve1_u = (current_unc - unc1) >= float(min_uncertainty_drop)
            improve1_joint = (current_joint - joint1) >= float(min_joint_risk_drop)
            improve1_any = bool(improve1_v or improve1_u or improve1_joint)
            rolling[action]["sum_joint_after_1"] += joint1
            rolling[action]["sum_v_after_1"] += float(v1)
            if improve1_any:
                rolling[action]["improve_1"] += 1.0
            if improve1_v:
                rolling[action]["improve_v_1"] += 1.0
            if improve1_u:
                rolling[action]["improve_u_1"] += 1.0
        if future2 is not None:
            rolling[action]["eligible_2"] += 1.0
            v2 = int(future2.get("number_of_violations", 0))
            unc2 = float(future2.get("scene_uncertainty_mean", 0.0))
            joint2 = float(future2.get("joint_risk_score", 0.0))
            improve2_v = v2 < current_v
            improve2_u = (current_unc - unc2) >= float(min_uncertainty_drop)
            improve2_joint = (current_joint - joint2) >= float(min_joint_risk_drop)
            improve2_any = bool(improve2_v or improve2_u or improve2_joint)
            rolling[action]["sum_joint_after_2"] += joint2
            rolling[action]["sum_v_after_2"] += float(v2)
            if improve2_any:
                rolling[action]["improve_2"] += 1.0
            if improve2_v:
                rolling[action]["improve_v_2"] += 1.0
            if improve2_u:
                rolling[action]["improve_u_2"] += 1.0

        if requested and (future1 is not None or future2 is not None):
            if (not improve1_any) and (not improve2_any):
                rolling[action]["false_delayed"] += 1.0
        rolling[action]["sum_joint_before"] += current_joint
        rolling[action]["sum_v_before"] += float(current_v)

        detail_rows.append(
            {
                "scene_index": idx,
                "scene_id": str(row.get("scene_id", "")),
                "selected_action": action,
                "action_requested": requested,
                "action_executed": executed,
                "joint_risk_before": current_joint,
                "violations_before": current_v,
                "gain_after_1_scene": bool(improve1_any),
                "gain_after_2_scenes": bool(improve2_any),
                "delayed_improve_violations_1": bool(improve1_v),
                "delayed_improve_violations_2": bool(improve2_v),
                "delayed_improve_uncertainty_1": bool(improve1_u),
                "delayed_improve_uncertainty_2": bool(improve2_u),
            }
        )

    for action in action_names:
        stat = rolling[action]
        trigger_count = stat["trigger_count"]
        eligible_1 = stat["eligible_1"]
        eligible_2 = stat["eligible_2"]
        per_action[action] = {
            "trigger_count": float(trigger_count),
            "immediate_conversion_rate": float(stat["executed_count"] / max(1.0, trigger_count)),
            "delayed_improve_rate_1": float(stat["improve_1"] / max(1.0, eligible_1)),
            "delayed_improve_rate_2": float(stat["improve_2"] / max(1.0, eligible_2)),
            "delayed_improve_violations_rate_1": float(stat["improve_v_1"] / max(1.0, eligible_1)),
            "delayed_improve_violations_rate_2": float(stat["improve_v_2"] / max(1.0, eligible_2)),
            "delayed_improve_uncertainty_rate_1": float(stat["improve_u_1"] / max(1.0, eligible_1)),
            "delayed_improve_uncertainty_rate_2": float(stat["improve_u_2"] / max(1.0, eligible_2)),
            "delayed_false_trigger_rate": float(stat["false_delayed"] / max(1.0, trigger_count)),
            "avg_joint_risk_before": float(stat["sum_joint_before"] / max(1.0, len(records))),
            "avg_joint_risk_after_1": float(stat["sum_joint_after_1"] / max(1.0, eligible_1)),
            "avg_joint_risk_after_2": float(stat["sum_joint_after_2"] / max(1.0, eligible_2)),
            "avg_violations_before": float(stat["sum_v_before"] / max(1.0, len(records))),
            "avg_violations_after_1": float(stat["sum_v_after_1"] / max(1.0, eligible_1)),
            "avg_violations_after_2": float(stat["sum_v_after_2"] / max(1.0, eligible_2)),
        }

    return {"delayed_gain_by_action": per_action, "delayed_action_event_rows": detail_rows}


def aggregate_story_diagnostics(
    records: List[Dict[str, object]],
    *,
    delayed_gain_min_uncertainty_drop: float = 0.03,
    delayed_gain_min_joint_risk_drop: float = 0.04,
) -> Dict[str, object]:
    conflict_types = [
        "transition_conflict",
        "execution_spec_conflict",
        "operator_post_state_conflict",
        "constraint_conflict",
        "mixed_conflict",
        "unknown",
    ]
    def _safe_conflict_type(value: object) -> str:
        token = str(value or "unknown")
        return token if token in conflict_types else "unknown"

    def _as_bool(value: object) -> bool:
        return bool(value)

    executed_patch_rows: List[Dict[str, object]] = []
    for record in records:
        patch_exec_items = record.get("patch_execution_records", [])
        if not isinstance(patch_exec_items, list):
            continue
        for item in patch_exec_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) != "executed":
                continue
            executed_patch_rows.append(item)
    unresolved_state_items_all: List[Dict[str, object]] = []
    for row in executed_patch_rows:
        for item in list(row.get("checklist_items_still_unresolved", []) or []):
            if isinstance(item, dict):
                unresolved_state_items_all.append(dict(item))
    retry_attempted_count = sum(1 for row in executed_patch_rows if bool(row.get("patch_retry_attempted", False)))
    first_pass_effective_count = sum(1 for row in executed_patch_rows if bool(row.get("first_pass_effective", False)))
    retry_effective_count = sum(1 for row in executed_patch_rows if bool(row.get("retry_effective", False)))
    still_ineffective_after_retry_count = sum(
        1 for row in executed_patch_rows if bool(row.get("still_ineffective_after_retry", False))
    )
    expansion_triggered_count = sum(1 for row in executed_patch_rows if bool(row.get("scope_expansion_triggered", False)))
    expansion_effective_count = sum(1 for row in executed_patch_rows if bool(row.get("scope_expansion_effective", False)))
    expansion_rows = [row for row in executed_patch_rows if bool(row.get("scope_expansion_triggered", False))]
    non_expansion_rows = [row for row in executed_patch_rows if not bool(row.get("scope_expansion_triggered", False))]
    overall_order_distribution: Dict[str, float] = {}
    overall_failure_step_distribution: Dict[str, float] = {}
    overall_rebroken_distribution: Dict[str, float] = {}
    overall_rebroken_after_retry_distribution: Dict[str, float] = {}
    for row in executed_patch_rows:
        order_key = "|".join([str(x) for x in list(row.get("retry_slot_priority_order", []) or [])]) or "none"
        overall_order_distribution[order_key] = overall_order_distribution.get(order_key, 0.0) + 1.0
        failure_key = str(row.get("priority_step_where_failure_remains", "") or "none")
        overall_failure_step_distribution[failure_key] = overall_failure_step_distribution.get(failure_key, 0.0) + 1.0
        rebroken_key = str(row.get("slot_type_rebroken_after_retry", "") or "none")
        overall_rebroken_distribution[rebroken_key] = overall_rebroken_distribution.get(rebroken_key, 0.0) + 1.0
        for item in list(row.get("protected_items_broken_after_retry", []) or []):
            if not isinstance(item, dict):
                continue
            token = str(item.get("state_type", "")).strip() or "unknown"
            overall_rebroken_after_retry_distribution[token] = overall_rebroken_after_retry_distribution.get(token, 0.0) + 1.0
    binding_modes = ["normal_binding", "reinforced_binding", "strict_binding"]
    generation_modes = [
        "normal_generation",
        "constrained_generation",
        "strict_state_realization_generation",
    ]

    executed_num = len(executed_patch_rows)
    executed_den = max(1, executed_num)
    by_conflict: Dict[str, Dict[str, object]] = {}
    no_gain_by_conflict: Dict[str, Dict[str, float]] = {}
    for conflict_type in conflict_types:
        rows = [x for x in executed_patch_rows if _safe_conflict_type(x.get("rewrite_conflict_type", "unknown")) == conflict_type]
        n = len(rows)
        d = max(1, n)
        if n == 0:
            by_conflict[conflict_type] = {
                "executed_count": 0,
                "patch_target_hits_violation_context_rate": 0.0,
                "rewrite_hits_required_state_change_rate": 0.0,
                "rewrite_removes_conflicting_state_rate": 0.0,
                "rewrite_realizes_required_state_change_rate": 0.0,
                "rewrite_removes_forbidden_state_rate": 0.0,
                "rewrite_restores_transition_coherence_proxy_rate": 0.0,
                "patch_effective_rate": 0.0,
                "patch_partial_rate": 0.0,
                "cosmetic_only_rate": 0.0,
                "ineffective_rate": 0.0,
                "before_transition_violations_mean": 0.0,
                "after_transition_violations_mean": 0.0,
                "before_constraint_violations_mean": 0.0,
                "after_constraint_violations_mean": 0.0,
                "before_uncertainty_mean": 0.0,
                "after_uncertainty_mean": 0.0,
                "before_symbolic_state_proxy_mean": 0.0,
                "after_symbolic_state_proxy_mean": 0.0,
                "first_pass_checklist_completion_rate": 0.0,
                "retry_checklist_completion_rate": 0.0,
                "checklist_items_fixed_by_retry_mean": 0.0,
                "checklist_items_still_unresolved_mean": 0.0,
                "retry_preserved_satisfied_items_rate": 0.0,
                "scope_expansion_trigger_rate": 0.0,
                "scope_expansion_effective_rate": 0.0,
                "retry_slot_priority_order_distribution": {},
                "slot_priority_fix_progress_mean": {
                    "forbidden_removed_count": 0.0,
                    "operator_post_state_realized_count": 0.0,
                    "required_state_realized_count": 0.0,
                },
                "slot_priority_preserve_result_rate": 0.0,
                "priority_step_where_failure_remains_distribution": {},
                "slot_type_rebroken_after_retry_distribution": {},
                "step_aware_preservation_guard_enabled_rate": 0.0,
                "step_aware_preservation_success_rate": 0.0,
                "forbidden_reintroduced_after_step_rate": 0.0,
                "operator_post_state_weakened_after_step_rate": 0.0,
                "required_state_regressed_after_step_rate": 0.0,
                "rebroken_after_retry_distribution": {},
            }
            no_gain_by_conflict[conflict_type] = {}
            continue
        reason_count: Dict[str, float] = {}
        for row in rows:
            reason = str(row.get("patch_no_gain_reason", "")).strip() or "unknown"
            reason_count[reason] = reason_count.get(reason, 0.0) + 1.0
        no_gain_by_conflict[conflict_type] = {
            key: float(value / d) for key, value in sorted(reason_count.items(), key=lambda kv: (-kv[1], kv[0]))
        }
        order_distribution: Dict[str, float] = {}
        failure_step_distribution: Dict[str, float] = {}
        rebroken_distribution: Dict[str, float] = {}
        rebroken_after_retry_distribution: Dict[str, float] = {}
        for row in rows:
            order_key = "|".join([str(x) for x in list(row.get("retry_slot_priority_order", []) or [])]) or "none"
            order_distribution[order_key] = order_distribution.get(order_key, 0.0) + 1.0
            failure_step_key = str(row.get("priority_step_where_failure_remains", "") or "none")
            failure_step_distribution[failure_step_key] = failure_step_distribution.get(failure_step_key, 0.0) + 1.0
            rebroken_key = str(row.get("slot_type_rebroken_after_retry", "") or "none")
            rebroken_distribution[rebroken_key] = rebroken_distribution.get(rebroken_key, 0.0) + 1.0
            for item in list(row.get("protected_items_broken_after_retry", []) or []):
                if not isinstance(item, dict):
                    continue
                token = str(item.get("state_type", "")).strip() or "unknown"
                rebroken_after_retry_distribution[token] = rebroken_after_retry_distribution.get(token, 0.0) + 1.0
        by_conflict[conflict_type] = {
            "executed_count": n,
            "patch_target_hits_violation_context_rate": float(
                sum(1 for row in rows if _as_bool(row.get("patch_target_hits_violation_context", False))) / d
            ),
            "rewrite_hits_required_state_change_rate": float(
                sum(1 for row in rows if _as_bool(row.get("rewrite_hits_required_state_change", False))) / d
            ),
            "rewrite_removes_conflicting_state_rate": float(
                sum(1 for row in rows if _as_bool(row.get("rewrite_removes_conflicting_state", False))) / d
            ),
            "rewrite_realizes_required_state_change_rate": float(
                sum(1 for row in rows if _as_bool(row.get("rewrite_realizes_required_state_change", False))) / d
            ),
            "rewrite_removes_forbidden_state_rate": float(
                sum(1 for row in rows if _as_bool(row.get("rewrite_removes_forbidden_state", False))) / d
            ),
            "rewrite_restores_transition_coherence_proxy_rate": float(
                sum(1 for row in rows if _as_bool(row.get("rewrite_restores_transition_coherence_proxy", False))) / d
            ),
            "patch_effective_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "effective") / d
            ),
            "patch_partial_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "partial") / d
            ),
            "first_pass_effective_rate": float(
                sum(1 for row in rows if bool(row.get("first_pass_effective", False))) / d
            ),
            "retry_effective_rate": float(
                sum(1 for row in rows if bool(row.get("retry_effective", False))) / d
            ),
            "still_ineffective_after_retry_rate": float(
                sum(1 for row in rows if bool(row.get("still_ineffective_after_retry", False))) / d
            ),
            "cosmetic_only_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "cosmetic_only") / d
            ),
            "ineffective_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "ineffective") / d
            ),
            "before_transition_violations_mean": float(
                sum(float(row.get("patch_before_transition_violation_count", 0)) for row in rows) / d
            ),
            "after_transition_violations_mean": float(
                sum(float(row.get("patch_after_transition_violation_count", 0)) for row in rows) / d
            ),
            "before_constraint_violations_mean": float(
                sum(float(row.get("patch_before_constraint_violation_count", 0)) for row in rows) / d
            ),
            "after_constraint_violations_mean": float(
                sum(float(row.get("patch_after_constraint_violation_count", 0)) for row in rows) / d
            ),
            "before_uncertainty_mean": float(
                sum(float(row.get("patch_before_uncertainty", 0.0)) for row in rows) / d
            ),
            "after_uncertainty_mean": float(
                sum(float(row.get("patch_after_uncertainty", 0.0)) for row in rows) / d
            ),
            "before_symbolic_state_proxy_mean": float(
                sum(float(row.get("patch_before_symbolic_state_proxy", 0.0)) for row in rows) / d
            ),
            "after_symbolic_state_proxy_mean": float(
                sum(float(row.get("patch_after_symbolic_state_proxy", 0.0)) for row in rows) / d
            ),
            "first_pass_checklist_completion_rate": float(
                sum(float(row.get("first_pass_checklist_completion_rate", 0.0) or 0.0) for row in rows) / d
            ),
            "retry_checklist_completion_rate": float(
                sum(float(row.get("retry_checklist_completion_rate", 0.0) or 0.0) for row in rows) / d
            ),
            "checklist_items_fixed_by_retry_mean": float(
                sum(len(list(row.get("checklist_items_fixed_by_retry", []) or [])) for row in rows) / d
            ),
            "checklist_items_still_unresolved_mean": float(
                sum(len(list(row.get("checklist_items_still_unresolved", []) or [])) for row in rows) / d
            ),
            "retry_preserved_satisfied_items_rate": float(
                sum(1 for row in rows if bool(row.get("retry_preserved_satisfied_items", True))) / d
            ),
            "scope_expansion_trigger_rate": float(
                sum(1 for row in rows if bool(row.get("scope_expansion_triggered", False))) / d
            ),
            "scope_expansion_effective_rate": float(
                sum(1 for row in rows if bool(row.get("scope_expansion_effective", False))) / d
            ),
            "retry_attempt_rate": float(
                sum(1 for row in rows if bool(row.get("patch_retry_attempted", False))) / d
            ),
            "retry_slot_priority_order_distribution": {
                k: float(v / d) for k, v in sorted(order_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "slot_priority_fix_progress_mean": {
                "forbidden_removed_count": float(
                    sum(float((row.get("slot_priority_fix_progress", {}) or {}).get("forbidden_removed_count", 0)) for row in rows)
                    / d
                ),
                "operator_post_state_realized_count": float(
                    sum(float((row.get("slot_priority_fix_progress", {}) or {}).get("operator_post_state_realized_count", 0)) for row in rows)
                    / d
                ),
                "required_state_realized_count": float(
                    sum(float((row.get("slot_priority_fix_progress", {}) or {}).get("required_state_realized_count", 0)) for row in rows)
                    / d
                ),
            },
            "slot_priority_preserve_result_rate": float(
                sum(1 for row in rows if bool(row.get("slot_priority_preserve_result", True))) / d
            ),
            "priority_step_where_failure_remains_distribution": {
                k: float(v / d) for k, v in sorted(failure_step_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "slot_type_rebroken_after_retry_distribution": {
                k: float(v / d) for k, v in sorted(rebroken_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "step_aware_preservation_guard_enabled_rate": float(
                sum(1 for row in rows if bool(row.get("step_aware_preservation_guard_enabled", False))) / d
            ),
            "step_aware_preservation_success_rate": float(
                sum(1 for row in rows if bool(row.get("protected_items_preserved_after_retry", True))) / d
            ),
            "forbidden_reintroduced_after_step_rate": float(
                sum(1 for row in rows if bool(row.get("forbidden_reintroduced_after_step", False))) / d
            ),
            "operator_post_state_weakened_after_step_rate": float(
                sum(1 for row in rows if bool(row.get("operator_post_state_weakened_after_step", False))) / d
            ),
            "required_state_regressed_after_step_rate": float(
                sum(1 for row in rows if bool(row.get("required_state_regressed_after_step", False))) / d
            ),
            "rebroken_after_retry_distribution": {
                k: float(v / d)
                for k, v in sorted(rebroken_after_retry_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
            },
        }

    num_records = len(records)
    num = max(1, num_records)
    total_violations = sum(int(item.get("number_of_violations", 0)) for item in records)
    total_weighted = sum(float(item.get("weighted_violation_score", 0.0)) for item in records)
    total_patch_rounds = sum(int(item.get("patch_rounds", 0)) for item in records)
    total_unchanged = sum(float(item.get("unchanged_ratio", 0.0)) for item in records)
    total_future = sum(float(item.get("future_conflict_penalty", 0.0)) for item in records)
    replan_count = sum(1 for item in records if bool(item.get("local_replan_triggered", False)))
    reject_count = sum(1 for item in records if not bool(item.get("accepted", True)))
    patch_success = sum(1 for item in records if bool(item.get("accepted", False)))
    total_regen_rounds = sum(int(item.get("scene_regenerate_rounds", 0)) for item in records)
    total_entropy_mean = sum(float(item.get("entropy_mean", 0.0)) for item in records)
    total_critical_peak = sum(float(item.get("critical_constraint_uncertainty_peak", 0.0)) for item in records)
    total_critical_mean = sum(float(item.get("critical_constraint_uncertainty_mean", 0.0)) for item in records)
    total_delta_uncertainty = sum(float(item.get("delta_uncertainty", 0.0)) for item in records)
    total_sentence_variance = sum(float(item.get("sentence_uncertainty_variance", 0.0)) for item in records)
    total_round_trend = sum(float(item.get("round_uncertainty_trend", 0.0)) for item in records)
    total_joint_risk = sum(float(item.get("joint_risk_score", 0.0)) for item in records)
    total_uncertainty_contribution = sum(float(item.get("uncertainty_contribution", 0.0)) for item in records)
    total_symbolic_contribution = sum(float(item.get("symbolic_contribution", 0.0)) for item in records)
    total_memory_contribution = sum(float(item.get("memory_contribution", 0.0)) for item in records)
    total_spike_count = sum(int(item.get("entropy_spike_count", 0)) for item in records)
    total_patch_alignment = sum(float(item.get("patch_target_joint_alignment_score", 0.0)) for item in records)
    patch_hit_violation_count = sum(
        1 for item in records if bool(item.get("patch_target_hits_violation_context", False))
    )
    patch_effective_count = sum(1 for item in records if str(item.get("patch_effectiveness_label", "")) == "effective")
    high_risk_scene_count = sum(1 for item in records if str(item.get("entropy_risk_tier", "")) == "high_risk")
    model_logprob_count = sum(1 for item in records if str(item.get("uncertainty_source_type", "")) == "model_logprob")
    text_proxy_count = sum(1 for item in records if str(item.get("uncertainty_source_type", "")) == "text_proxy")
    trigger_to_action_count = sum(
        1
        for item in records
        if (
            bool(item.get("entropy_triggered_validation", False))
            or bool(item.get("entropy_triggered_patch_escalation", False))
            or bool(item.get("entropy_triggered_replan", False))
        )
        and (
            bool(item.get("entropy_triggered_patch_escalation", False))
            or bool(item.get("entropy_triggered_replan", False))
            or str(item.get("entropy_validation_mode", "standard")) == "escalated"
        )
    )
    entropy_trigger_count = sum(
        1
        for item in records
        if bool(item.get("entropy_triggered_validation", False))
        or bool(item.get("entropy_triggered_patch_escalation", False))
        or bool(item.get("entropy_triggered_replan", False))
    )
    overlap_total = sum(float(item.get("overlap_between_high_entropy_and_actual_violations", 0.0)) for item in records)
    entropy_patch_success_count = sum(
        1
        for item in records
        if bool(item.get("accepted", False))
        and (
            bool(item.get("entropy_triggered_patch_escalation", False))
            or bool(item.get("entropy_triggered_replan", False))
        )
    )
    validation_only_zero_influence_rate = sum(
        1 for item in records if bool(item.get("entropy_validation_only_zero_influence", True))
    ) / num
    binding_mode_rate = {
        mode: float(sum(1 for item in records if str(item.get("memory_binding_mode", "")) == mode) / num)
        for mode in binding_modes
    }
    generation_control_mode_rate = {
        mode: float(sum(1 for item in records if str(item.get("generation_control_mode", "")) == mode) / num)
        for mode in generation_modes
    }
    binding_reason_count: Dict[str, float] = {}
    for item in records:
        reasons = item.get("memory_binding_decision_reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            key = str(reason).strip() or "unknown"
            binding_reason_count[key] = binding_reason_count.get(key, 0.0) + 1.0
    binding_reason_distribution = {
        key: float(value / num) for key, value in sorted(binding_reason_count.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    warning_changed_control_rate = float(
        sum(
            1
            for item in records
            if bool(
                ((item.get("uncertainty_warning_output", {}) or {}).get(
                    "warning_changed_downstream_control",
                    False,
                ))
            )
        )
        / num
    )
    final_judge_misuse_rate = float(
        sum(
            1
            for item in records
            if bool(((item.get("uncertainty_warning_output", {}) or {}).get("is_final_judge", False)))
        )
        / num
    )
    executed_by_generation_control_mode: Dict[str, Dict[str, float]] = {}
    for mode in generation_modes:
        rows = [x for x in executed_patch_rows if str(x.get("rewrite_generation_control_mode", "")) == mode]
        d = max(1, len(rows))
        executed_by_generation_control_mode[mode] = {
            "executed_count": float(len(rows)),
            "patch_effective_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "effective") / d
            ),
            "patch_partial_rate": float(
                sum(1 for row in rows if str(row.get("patch_effectiveness_label", "")) == "partial") / d
            ),
            "retry_attempt_rate": float(sum(1 for row in rows if bool(row.get("patch_retry_attempted", False))) / d),
            "retry_effective_rate": float(sum(1 for row in rows if bool(row.get("retry_effective", False))) / d),
            "still_ineffective_after_retry_rate": float(
                sum(1 for row in rows if bool(row.get("still_ineffective_after_retry", False))) / d
            ),
        }
    joint_action_replan_rate = sum(1 for item in records if str(item.get("joint_action_selector", "")) == "replan") / num
    joint_action_patch_plus_rate = (
        sum(1 for item in records if str(item.get("joint_action_selector", "")) == "patch_plus_escalation") / num
    )
    joint_action_types = ["do_nothing", "validation_boost", "patch", "patch_plus_escalation", "replan"]
    trigger_quality_by_action: Dict[str, Dict[str, float]] = {}
    all_events: List[Dict[str, object]] = []
    for item in records:
        events = item.get("joint_action_events", [])
        if isinstance(events, list):
            all_events.extend([x for x in events if isinstance(x, dict)])
    for action_name in joint_action_types:
        events = [e for e in all_events if str(e.get("selected_action", "")) == action_name]
        requested = sum(1 for e in events if bool(e.get("action_requested", False)))
        executed = sum(1 for e in events if bool(e.get("action_executed", False)))
        blocked = sum(int(e.get("action_blocked_count", 0)) for e in events)
        improve_t = sum(1 for e in events if bool(e.get("improved_transition_violations", False)))
        improve_v = sum(1 for e in events if bool(e.get("improved_violations", False)))
        improve_u = sum(1 for e in events if bool(e.get("improved_uncertainty", False)))
        no_gain = sum(
            1
            for e in events
            if not bool(e.get("improved_transition_violations", False))
            and not bool(e.get("improved_violations", False))
            and not bool(e.get("improved_uncertainty", False))
        )
        trigger_quality_by_action[action_name] = {
            "trigger_rate": float(len(events) / max(1, len(all_events))),
            "trigger_count": float(len(events)),
            "requested_count": float(requested),
            "executed_count": float(executed),
            "blocked_count": float(blocked),
            "conversion_rate": float(executed / max(1, requested)),
            "improve_transition_violations_rate": float(improve_t / max(1, len(events))),
            "improve_violations_rate": float(improve_v / max(1, len(events))),
            "improve_uncertainty_rate": float(improve_u / max(1, len(events))),
            "false_or_empty_trigger_rate": float(no_gain / max(1, len(events))),
        }
    missed_high_violation_rate = 0.0
    dn_events = [e for e in all_events if str(e.get("selected_action", "")) == "do_nothing"]
    if dn_events:
        missed_high_violation_rate = float(
            sum(1 for e in dn_events if int(e.get("after_violation_count", 0)) >= 3) / max(1, len(dn_events))
        )
    delayed_payload = _compute_delayed_metrics(
        records,
        min_uncertainty_drop=float(delayed_gain_min_uncertainty_drop),
        min_joint_risk_drop=float(delayed_gain_min_joint_risk_drop),
    )
    high_weight_count = 0
    low_weight_count = 0
    for item in records:
        tier = item.get("violation_tier_breakdown", {})
        if isinstance(tier, dict):
            high_weight_count += int(tier.get("1", 0))
            high_weight_count += int(tier.get("2", 0))
            low_weight_count += int(tier.get("4", 0))
        patch_plan = item.get("patch_plan")
        if isinstance(patch_plan, dict):
            deferred = patch_plan.get("deferred_low_confidence_violations")
            if isinstance(deferred, list):
                low_weight_count += len(deferred)
    prompt_id = str(records[0].get("prompt_id", "")) if records else ""
    variant_name = str(records[0].get("variant_name", "")) if records else ""
    return {
        "prompt_id": prompt_id,
        "variant_name": variant_name,
        "num_scenes": num_records,
        "total_violations": total_violations,
        "total_weighted_violations": total_weighted,
        "patch_success_rate": patch_success / num,
        "average_patch_rounds": total_patch_rounds / num,
        "average_unchanged_ratio": total_unchanged / num,
        "replan_rate": replan_count / num,
        "rejection_rate": reject_count / num,
        "average_future_penalty": total_future / num,
        "average_scene_regenerate_rounds": total_regen_rounds / num,
        "high_weight_constraint_violation_count": high_weight_count,
        "low_weight_deferred_violation_count": low_weight_count,
        "average_entropy_mean": total_entropy_mean / num,
        "average_critical_constraint_uncertainty_peak": total_critical_peak / num,
        "average_critical_constraint_uncertainty_mean": total_critical_mean / num,
        "average_delta_uncertainty": total_delta_uncertainty / num,
        "average_sentence_uncertainty_variance": total_sentence_variance / num,
        "average_round_uncertainty_trend": total_round_trend / num,
        "average_joint_risk_score": total_joint_risk / num,
        "average_uncertainty_contribution": total_uncertainty_contribution / num,
        "average_symbolic_contribution": total_symbolic_contribution / num,
        "average_memory_contribution": total_memory_contribution / num,
        "model_logprob_uncertainty_rate": model_logprob_count / num,
        "text_proxy_uncertainty_rate": text_proxy_count / num,
        "average_spike_count": total_spike_count / num,
        "average_patch_target_joint_alignment_score": total_patch_alignment / num,
        "patch_target_hits_violation_context_rate": patch_hit_violation_count / num,
        "patch_effective_rate": patch_effective_count / num,
        "high_risk_scene_rate": high_risk_scene_count / num,
        "entropy_trigger_rate": entropy_trigger_count / num,
        "entropy_trigger_to_action_conversion_rate": trigger_to_action_count / max(1, entropy_trigger_count),
        "entropy_trigger_precision": overlap_total / num,
        "entropy_assisted_patch_success_rate": entropy_patch_success_count / num,
        "entropy_validation_only_zero_influence_rate": validation_only_zero_influence_rate,
        "memory_binding_mode_rate": binding_mode_rate,
        "generation_control_mode_rate": generation_control_mode_rate,
        "memory_binding_reason_distribution": binding_reason_distribution,
        "uncertainty_warning_changed_control_rate": warning_changed_control_rate,
        "uncertainty_warning_final_judge_misuse_rate": final_judge_misuse_rate,
        "joint_action_replan_rate": joint_action_replan_rate,
        "joint_action_patch_plus_escalation_rate": joint_action_patch_plus_rate,
        "trigger_quality_by_action": trigger_quality_by_action,
        "delayed_gain_by_action": delayed_payload["delayed_gain_by_action"],
        "delayed_action_event_rows": delayed_payload["delayed_action_event_rows"],
        "missed_high_violation_after_do_nothing_rate": missed_high_violation_rate,
        "executed_patch_count": executed_num,
        "executed_patch_attribution": {
            "overall": {
                "patch_target_hits_violation_context_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("patch_target_hits_violation_context", False)))
                    / executed_den
                ),
                "rewrite_hits_required_state_change_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("rewrite_hits_required_state_change", False)))
                    / executed_den
                ),
                "rewrite_removes_conflicting_state_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("rewrite_removes_conflicting_state", False)))
                    / executed_den
                ),
                "rewrite_realizes_required_state_change_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("rewrite_realizes_required_state_change", False)))
                    / executed_den
                ),
                "rewrite_removes_forbidden_state_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("rewrite_removes_forbidden_state", False)))
                    / executed_den
                ),
                "rewrite_restores_transition_coherence_proxy_rate": float(
                    sum(1 for row in executed_patch_rows if _as_bool(row.get("rewrite_restores_transition_coherence_proxy", False)))
                    / executed_den
                ),
                "patch_effective_rate": float(
                    sum(1 for row in executed_patch_rows if str(row.get("patch_effectiveness_label", "")) == "effective")
                    / executed_den
                ),
                "patch_partial_rate": float(
                    sum(1 for row in executed_patch_rows if str(row.get("patch_effectiveness_label", "")) == "partial")
                    / executed_den
                ),
                "first_pass_effective_rate": float(first_pass_effective_count / executed_den),
                "retry_attempt_rate": float(retry_attempted_count / executed_den),
                "retry_effective_rate": float(retry_effective_count / executed_den),
                "still_ineffective_after_retry_rate": float(still_ineffective_after_retry_count / executed_den),
                "first_pass_checklist_completion_rate": float(
                    sum(float(row.get("first_pass_checklist_completion_rate", 0.0) or 0.0) for row in executed_patch_rows)
                    / executed_den
                ),
                "retry_checklist_completion_rate": float(
                    sum(float(row.get("retry_checklist_completion_rate", 0.0) or 0.0) for row in executed_patch_rows)
                    / executed_den
                ),
                "checklist_items_fixed_by_retry_mean": float(
                    sum(len(list(row.get("checklist_items_fixed_by_retry", []) or [])) for row in executed_patch_rows)
                    / executed_den
                ),
                "checklist_items_still_unresolved_mean": float(
                    sum(len(list(row.get("checklist_items_still_unresolved", []) or [])) for row in executed_patch_rows)
                    / executed_den
                ),
                "retry_preserved_satisfied_items_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("retry_preserved_satisfied_items", True)))
                    / executed_den
                ),
                "scope_expansion_trigger_rate": float(expansion_triggered_count / executed_den),
                "scope_expansion_effective_rate": float(expansion_effective_count / max(1, expansion_triggered_count)),
                "retry_slot_priority_order_distribution": {
                    k: float(v / executed_den) for k, v in sorted(overall_order_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "slot_priority_fix_progress_mean": {
                    "forbidden_removed_count": float(
                        sum(
                            float((row.get("slot_priority_fix_progress", {}) or {}).get("forbidden_removed_count", 0))
                            for row in executed_patch_rows
                        )
                        / executed_den
                    ),
                    "operator_post_state_realized_count": float(
                        sum(
                            float((row.get("slot_priority_fix_progress", {}) or {}).get("operator_post_state_realized_count", 0))
                            for row in executed_patch_rows
                        )
                        / executed_den
                    ),
                    "required_state_realized_count": float(
                        sum(
                            float((row.get("slot_priority_fix_progress", {}) or {}).get("required_state_realized_count", 0))
                            for row in executed_patch_rows
                        )
                        / executed_den
                    ),
                },
                "slot_priority_preserve_result_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("slot_priority_preserve_result", True)))
                    / executed_den
                ),
                "priority_step_where_failure_remains_distribution": {
                    k: float(v / executed_den)
                    for k, v in sorted(overall_failure_step_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "slot_type_rebroken_after_retry_distribution": {
                    k: float(v / executed_den)
                    for k, v in sorted(overall_rebroken_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "step_aware_preservation_guard_enabled_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("step_aware_preservation_guard_enabled", False)))
                    / executed_den
                ),
                "step_aware_preservation_success_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("protected_items_preserved_after_retry", True)))
                    / executed_den
                ),
                "forbidden_reintroduced_after_step_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("forbidden_reintroduced_after_step", False)))
                    / executed_den
                ),
                "operator_post_state_weakened_after_step_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("operator_post_state_weakened_after_step", False)))
                    / executed_den
                ),
                "required_state_regressed_after_step_rate": float(
                    sum(1 for row in executed_patch_rows if bool(row.get("required_state_regressed_after_step", False)))
                    / executed_den
                ),
                "rebroken_after_retry_distribution": {
                    k: float(v / executed_den)
                    for k, v in sorted(
                        overall_rebroken_after_retry_distribution.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                },
                "scope_expansion_vs_non_expansion": {
                    "expansion_checklist_completion_rate": float(
                        sum(float(r.get("retry_checklist_completion_rate", 0.0) or 0.0) for r in expansion_rows)
                        / max(1, len(expansion_rows))
                    ),
                    "non_expansion_checklist_completion_rate": float(
                        sum(float(r.get("retry_checklist_completion_rate", 0.0) or 0.0) for r in non_expansion_rows)
                        / max(1, len(non_expansion_rows))
                    ),
                    "expansion_required_forbidden_combo_fix_rate": float(
                        sum(
                            1
                            for r in expansion_rows
                            if any(str(x.get("state_type", "")) == "required" for x in list(r.get("checklist_items_fixed_by_expansion", []) or []))
                            and any(str(x.get("state_type", "")) == "forbidden" for x in list(r.get("checklist_items_fixed_by_expansion", []) or []))
                        )
                        / max(1, len(expansion_rows))
                    ),
                    "expansion_required_operator_combo_fix_rate": float(
                        sum(
                            1
                            for r in expansion_rows
                            if any(str(x.get("state_type", "")) == "required" for x in list(r.get("checklist_items_fixed_by_expansion", []) or []))
                            and any(str(x.get("state_type", "")) == "operator_post_state" for x in list(r.get("checklist_items_fixed_by_expansion", []) or []))
                        )
                        / max(1, len(expansion_rows))
                    ),
                },
                "cosmetic_only_rate": float(
                    sum(1 for row in executed_patch_rows if str(row.get("patch_effectiveness_label", "")) == "cosmetic_only")
                    / executed_den
                ),
                "ineffective_rate": float(
                    sum(1 for row in executed_patch_rows if str(row.get("patch_effectiveness_label", "")) == "ineffective")
                    / executed_den
                ),
                "executed_patch_checklist_summary": {
                    "first_pass_required_state_checklist": [
                        row.get("first_pass_required_state_checklist", []) for row in executed_patch_rows
                    ],
                    "first_pass_forbidden_state_checklist": [
                        row.get("first_pass_forbidden_state_checklist", []) for row in executed_patch_rows
                    ],
                    "first_pass_operator_post_state_checklist": [
                        row.get("first_pass_operator_post_state_checklist", []) for row in executed_patch_rows
                    ],
                    "retry_required_state_checklist": [
                        row.get("retry_required_state_checklist", []) for row in executed_patch_rows
                    ],
                    "retry_forbidden_state_checklist": [
                        row.get("retry_forbidden_state_checklist", []) for row in executed_patch_rows
                    ],
                    "retry_operator_post_state_checklist": [
                        row.get("retry_operator_post_state_checklist", []) for row in executed_patch_rows
                    ],
                    "checklist_items_fixed_by_retry": [
                        row.get("checklist_items_fixed_by_retry", []) for row in executed_patch_rows
                    ],
                    "still_unresolved_state_items": [
                        row.get("checklist_items_still_unresolved", []) for row in executed_patch_rows
                    ],
                },
            },
            "by_conflict_type": by_conflict,
            "no_gain_reason_distribution_by_conflict_type": no_gain_by_conflict,
            "by_generation_control_mode": executed_by_generation_control_mode,
            "still_unresolved_state_items": unresolved_state_items_all,
        },
        "framework_dual_summary": {
            "dual_memory": {
                "memory_blocks_strengthened_rate": float(
                    sum(1 for item in records if bool(item.get("strengthened_memory_blocks", []))) / num
                ),
                "static_memory_strengthened_rate": float(
                    sum(
                        1
                        for item in records
                        if "static_world_invariants" in list(item.get("strengthened_memory_blocks", []) or [])
                    )
                    / num
                ),
                "dynamic_memory_strengthened_rate": float(
                    sum(
                        1
                        for item in records
                        if "dynamic_pending_constraints" in list(item.get("strengthened_memory_blocks", []) or [])
                    )
                    / num
                ),
                "memory_binding_mode_rate": dict(binding_mode_rate),
            },
            "dual_generation_control": {
                "generation_control_mode_rate": dict(generation_control_mode_rate),
                "reasoning_constraints_present_rate": float(
                    sum(
                        1
                        for item in records
                        if bool(
                            ((item.get("generation_control_context", {}) or {}).get("reasoning_constraints", {}))
                        )
                    )
                    / num
                ),
                "uncertainty_guided_control_present_rate": float(
                    sum(
                        1
                        for item in records
                        if bool(
                            ((item.get("generation_control_context", {}) or {}).get("uncertainty_guided_control", {}))
                        )
                    )
                    / num
                ),
            },
            "dual_consistency_assurance": {
                "symbolic_violation_mean": float(total_violations / num),
                "uncertainty_warning_changed_control_rate": warning_changed_control_rate,
                "uncertainty_warning_final_judge_misuse_rate": final_judge_misuse_rate,
            },
            "repair_loop": {
                "executed_patch_count": int(executed_num),
                "retry_attempt_rate": float(retry_attempted_count / executed_den),
                "retry_effective_rate": float(retry_effective_count / executed_den),
                "still_ineffective_after_retry_rate": float(still_ineffective_after_retry_count / executed_den),
                "dynamic_memory_update_applied_rate": float(
                    sum(
                        1
                        for item in records
                        if bool(
                            ((item.get("dynamic_memory_update_status", {}) or {}).get(
                                "dynamic_memory_update_applied",
                                False,
                            ))
                        )
                    )
                    / num
                ),
            },
        },
    }
