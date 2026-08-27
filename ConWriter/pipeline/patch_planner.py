"""Optimization-guided patch planner with weighted global-aware objective."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ConWriter.pipeline.patch_graph import PatchDependencyGraph
from ConWriter.pipeline.weighted_constraints import weight_violation, weighted_violation_score
from ConWriter.reasoning.future_conflict_estimator import FutureConflictEstimator
from ConWriter.reasoning.scene_alignment import find_sentence_ids_for_tokens, split_scene_into_units
from ConWriter.utils.types import (
    ConsistencyReport,
    ConstraintViolation,
    DynamicMemory,
    PatchPlan,
    SceneExtraction,
    ScenePlan,
    SentencePatch,
    StaticMemory,
    WeightedConstraintItem,
)


@dataclass(slots=True)
class _SimState:
    text: str
    changed_sentence_ids: List[str]
    edit_cost: float
    preservation_penalty: float
    stability_score: float


class PatchPlanner:
    """Build and optimize structured patch trajectory from localized violations."""

    def __init__(
        self,
        max_patch_targets_per_round: int = 3,
        allow_neighbor_adjustment: bool = True,
        lookahead_top_k: int = 3,
        low_confidence_threshold: float = 0.45,
    ):
        self.max_patch_targets_per_round = max(1, int(max_patch_targets_per_round))
        self.allow_neighbor_adjustment = bool(allow_neighbor_adjustment)
        self.lookahead_top_k = max(1, int(lookahead_top_k))
        self.low_confidence_threshold = float(max(0.0, min(1.0, low_confidence_threshold)))
        # Objective:
        # lambda_1 * weighted_remaining + lambda_2 * edit + lambda_3 * preserve
        # + lambda_4 * plan_deviation + lambda_5 * predicted_future_conflict_penalty.
        self.lambda_remaining = 1.0
        self.lambda_edit = 0.8
        self.lambda_preservation = 2.0
        self.lambda_plan_deviation = 2.2
        self.lambda_future_conflict = 3.2
        self.future_estimator = FutureConflictEstimator()

    def build_patch_plan(
        self,
        scene_plan: ScenePlan,
        extraction: SceneExtraction,
        report: ConsistencyReport,
        violations: Sequence[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        round_idx: int,
        future_scenes: Sequence[ScenePlan] | None = None,
        weighted_future_constraints: Sequence[dict] | None = None,
        entropy_risk_prior: Dict[str, float] | None = None,
        entropy_linked_sentence_ids: Sequence[str] | None = None,
        violation_context: Dict[str, object] | None = None,
    ) -> PatchPlan:
        sentence_ids = [unit.sentence_id for unit in extraction.sentences]
        if not sentence_ids:
            sentence_ids = [f"{scene_plan.scene_id}_sent_000"]
        violations = [weight_violation(v) for v in list(violations)]
        weighted_future_items = self._normalize_weighted_future_constraints(
            weighted_future_constraints or []
        )
        conf_by_violation = {self._violation_id(v): self._violation_confidence(v) for v in violations}
        high_conf = [
            v for v in violations
            if (conf_by_violation[self._violation_id(v)] >= self.low_confidence_threshold) or v.severity.lower() == "error"
        ]
        low_conf = [v for v in violations if v not in high_conf]

        active_violations = high_conf if high_conf else list(violations)
        base_targets = self._collect_violation_targets(
            scene_plan=scene_plan,
            extraction=extraction,
            violations=active_violations,
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
        )
        dependency_graph = PatchDependencyGraph.build(active_violations, extraction)
        candidates = self._candidate_target_sets(
            sentence_ids=sentence_ids,
            base_targets=base_targets,
            violations=active_violations,
            dependency_graph=dependency_graph,
        )
        if not candidates:
            candidates = [[sentence_ids[-1]]]

        ranked = self._rank_candidates_prior(
            candidates,
            active_violations,
            extraction,
            conf_by_violation,
            entropy_risk_prior=entropy_risk_prior,
            entropy_linked_sentence_ids=entropy_linked_sentence_ids,
            violation_context=violation_context,
        )
        top_candidates = [targets for _, targets, _ in ranked[: self.lookahead_top_k]]
        top_alignment = ranked[0][2] if ranked else {}

        best = None
        for cid, targets in enumerate(top_candidates):
            step1 = self._build_step_patch_plan(
                scene_plan=scene_plan,
                violations=active_violations,
                targets=targets,
                round_idx=round_idx,
                step_idx=1,
                sentence_ids=sentence_ids,
            )
            trajectory = self.simulate_two_step_patch(
                scene_text=extraction.scene_text,
                step1=step1,
                scene_plan=scene_plan,
                extraction=extraction,
                active_violations=active_violations,
                candidate_id=cid,
                round_idx=round_idx,
                sentence_ids=sentence_ids,
                dependency_graph=dependency_graph,
                dynamic_memory=dynamic_memory,
                future_scenes=future_scenes or [],
                weighted_future_constraints=weighted_future_items,
            )
            if best is None or trajectory["objective"] < best["objective"]:
                best = trajectory

        if best is None:
            # defensive fallback
            fallback_patch = SentencePatch(
                patch_id=f"patch_{scene_plan.scene_id}_{round_idx:02d}_fallback",
                op_type="replace",
                target_sentence_ids=[sentence_ids[-1]],
                rationale="Fallback patch due to missing optimization trajectory.",
                constraints_to_satisfy=list(scene_plan.required_constraints),
            )
            return PatchPlan(
                plan_id=f"patch_plan_{scene_plan.scene_id}_{round_idx:02d}",
                target_sentence_ids=[sentence_ids[-1]],
                protected_sentence_ids=sentence_ids[:-1],
                patch_sequence=[fallback_patch],
                fallback_level="scene",
                expected_fixed_violations=[self._violation_id(v) for v in violations],
                needs_replan=bool(getattr(report, "needs_replan", False)),
                candidate_score=-100.0,
                expected_violation_reduction=0.0,
                expected_preservation_cost=1.0,
                score_breakdown={"objective": 100.0},
                chosen_rationale="Fallback path selected.",
                global_objective_breakdown={"objective": 100.0},
                violation_context_sentence_ids=[],
                violation_context_constraint_ids=[],
                entropy_context_sentence_ids=list(dict.fromkeys(list(entropy_linked_sentence_ids or []))),
            )

        needs_replan = bool(getattr(report, "needs_replan", False) or any(v.needs_replan for v in violations))
        fatal = any(v.fatal for v in violations)
        final_targets = sorted(set(best["targets"]))
        protected = sorted(set(sentence_ids) - set(final_targets))
        total = max(1, len(active_violations))
        remaining = int(best["remaining_count"])
        expected_reduction = float(max(0, total - remaining) / total)
        preservation_cost = float(best["preservation_penalty"])
        fallback_level = self._fallback_level(active_violations, len(final_targets), needs_replan, fatal)
        deferred_ids = [self._violation_id(v) for v in low_conf]

        return PatchPlan(
            plan_id=f"patch_plan_{scene_plan.scene_id}_{round_idx:02d}",
            target_sentence_ids=final_targets,
            protected_sentence_ids=protected,
            patch_sequence=list(best["patch_sequence"]),
            fallback_level=fallback_level,
            requires_neighbor_adjustment=self.allow_neighbor_adjustment and len(final_targets) > 1,
            expected_fixed_violations=[self._violation_id(v) for v in active_violations],
            needs_replan=needs_replan,
            candidate_score=float(-best["objective"]),
            expected_violation_reduction=expected_reduction,
            expected_preservation_cost=preservation_cost,
            score_breakdown=dict(best["score_breakdown"]),
            chosen_rationale=str(best["rationale"]),
            trajectory_length=int(best["trajectory_length"]),
            deferred_low_confidence_violations=deferred_ids,
            future_conflict_penalty=float(best.get("future_conflict_penalty", 0.0)),
            weighted_remaining_violation_score=float(best.get("weighted_remaining_violation_score", 0.0)),
            critical_constraints_preserved=list(best.get("critical_constraints_preserved", [])),
            deferred_constraints=list(best.get("deferred_constraints", [])),
            global_objective_breakdown=dict(best.get("global_objective_breakdown", {})),
            impacted_future_scene_ids=list(best.get("impacted_future_scene_ids", [])),
            critical_future_constraints_at_risk=list(best.get("critical_future_constraints_at_risk", [])),
            violation_context_sentence_ids=list(top_alignment.get("violation_context_sentence_ids", [])),
            violation_context_constraint_ids=list(top_alignment.get("violation_context_constraint_ids", [])),
            entropy_context_sentence_ids=list(top_alignment.get("entropy_context_sentence_ids", [])),
            patch_target_hits_violation_context=bool(top_alignment.get("hits_violation_context", False)),
            patch_target_hits_entropy_context=bool(top_alignment.get("hits_entropy_context", False)),
            patch_target_joint_alignment_score=float(top_alignment.get("joint_alignment_score", 0.0)),
            patch_alignment_score_breakdown=dict(top_alignment.get("score_breakdown", {})),
        )

    def simulate_patch_step(
        self,
        scene_text: str,
        patch_plan: PatchPlan,
    ) -> _SimState:
        units = split_scene_into_units(scene_text, "sim_scene")
        by_id = {u.sentence_id: idx for idx, u in enumerate(units)}
        # local simulated IDs are synthetic, so map by order for compatible planning IDs.
        ordered_text = [u.text for u in units]
        changed: List[str] = []
        edit_cost = 0.0
        for patch in patch_plan.patch_sequence:
            for sid in patch.target_sentence_ids:
                idx = self._resolve_index_for_sim_id(sid, by_id, ordered_text)
                if idx is None or idx >= len(ordered_text):
                    continue
                before = ordered_text[idx]
                after = self._simulate_sentence_edit(before, patch)
                if after != before:
                    changed.append(sid)
                    edit_cost += self._token_diff_cost(before, after)
                ordered_text[idx] = after
        simulated_text = " ".join(t for t in ordered_text if t.strip()).strip()
        preservation_penalty = float(
            len([sid for sid in changed if sid in set(patch_plan.protected_sentence_ids)])
        )
        stability = self._stability_score(scene_text, simulated_text)
        return _SimState(
            text=simulated_text,
            changed_sentence_ids=sorted(set(changed)),
            edit_cost=edit_cost,
            preservation_penalty=preservation_penalty,
            stability_score=stability,
        )

    def simulate_two_step_patch(
        self,
        scene_text: str,
        step1: PatchPlan,
        scene_plan: ScenePlan,
        extraction: SceneExtraction,
        active_violations: Sequence[ConstraintViolation],
        candidate_id: int,
        round_idx: int,
        sentence_ids: Sequence[str],
        dependency_graph: PatchDependencyGraph,
        dynamic_memory: DynamicMemory,
        future_scenes: Sequence[ScenePlan],
        weighted_future_constraints: Sequence[WeightedConstraintItem],
    ) -> Dict[str, object]:
        sim1 = self.simulate_patch_step(scene_text, step1)
        remaining_after_1 = self._revalidate_simulation(sim1.text, active_violations, scene_plan)
        remaining_violations = remaining_after_1["remaining_violations"]
        # Build step-2 candidate from unresolved high-confidence violations.
        step2_candidates = self._candidate_target_sets(
            sentence_ids=sentence_ids,
            base_targets=self._collect_targets_from_violations(remaining_violations),
            violations=remaining_violations,
            dependency_graph=dependency_graph,
        )
        step2_best = None
        if step2_candidates:
            for targets in step2_candidates[: self.lookahead_top_k]:
                step2 = self._build_step_patch_plan(
                    scene_plan=scene_plan,
                    violations=remaining_violations,
                    targets=targets,
                    round_idx=round_idx,
                    step_idx=2,
                    sentence_ids=sentence_ids,
                )
                sim2 = self.simulate_patch_step(sim1.text, step2)
                remaining_after_2 = self._revalidate_simulation(sim2.text, remaining_violations, scene_plan)
                weighted_remaining = weighted_violation_score(remaining_after_2["remaining_violations"])
                future_estimate = self.future_estimator.estimate(
                    patched_scene_text=sim2.text,
                    patched_dynamic_memory=dynamic_memory,
                    future_scenes=future_scenes,
                    active_weighted_constraints=remaining_after_2["remaining_violations"],
                    weighted_future_constraints=weighted_future_constraints,
                )
                objective = self._objective(
                    weighted_remaining_violations=float(weighted_remaining),
                    edit_cost=(sim1.edit_cost + sim2.edit_cost),
                    preservation_penalty=(sim1.preservation_penalty + sim2.preservation_penalty),
                    plan_deviation=float(remaining_after_2["plan_deviation"]),
                    future_conflict_penalty=future_estimate.predicted_future_conflict_penalty,
                )
                # penalize low stability trajectories.
                stability_penalty = (1.0 - min(sim1.stability_score, sim2.stability_score)) * 2.0
                objective += stability_penalty
                critical_preserved = self._collect_preserved_critical_constraints(
                    original=active_violations,
                    remaining=remaining_after_2["remaining_violations"],
                )
                deferred_constraints = self._collect_deferred_constraints(remaining_after_2["remaining_violations"])
                candidate = {
                    "objective": objective,
                    "targets": sorted(set(step1.target_sentence_ids + step2.target_sentence_ids)),
                    "patch_sequence": list(step1.patch_sequence) + list(step2.patch_sequence),
                    "remaining_count": len(remaining_after_2["remaining_violations"]),
                    "preservation_penalty": sim1.preservation_penalty + sim2.preservation_penalty,
                    "future_conflict_penalty": future_estimate.predicted_future_conflict_penalty,
                    "weighted_remaining_violation_score": weighted_remaining,
                    "critical_constraints_preserved": critical_preserved,
                    "deferred_constraints": deferred_constraints,
                    "impacted_future_scene_ids": future_estimate.impacted_future_scene_ids,
                    "critical_future_constraints_at_risk": future_estimate.critical_future_constraints_at_risk,
                    "trajectory_length": 2,
                    "score_breakdown": {
                        "remaining_violations": float(len(remaining_after_2["remaining_violations"])),
                        "weighted_remaining_violations": float(weighted_remaining),
                        "edit_cost": float(sim1.edit_cost + sim2.edit_cost),
                        "preservation_penalty": float(sim1.preservation_penalty + sim2.preservation_penalty),
                        "plan_deviation": float(remaining_after_2["plan_deviation"]),
                        "future_conflict_penalty": float(future_estimate.predicted_future_conflict_penalty),
                        "stability_penalty": float(stability_penalty),
                        "objective": float(objective),
                    },
                    "global_objective_breakdown": {
                        "weighted_remaining_violations": float(weighted_remaining),
                        "edit_cost": float(sim1.edit_cost + sim2.edit_cost),
                        "preservation_penalty": float(sim1.preservation_penalty + sim2.preservation_penalty),
                        "plan_deviation": float(remaining_after_2["plan_deviation"]),
                        "future_conflict_penalty": float(future_estimate.predicted_future_conflict_penalty),
                    },
                    "rationale": (
                        f"candidate#{candidate_id}: two-step lookahead selected "
                        f"{len(step1.target_sentence_ids)}+{len(step2.target_sentence_ids)} target(s)."
                    ),
                }
                if step2_best is None or candidate["objective"] < step2_best["objective"]:
                    step2_best = candidate

        if step2_best is not None:
            return step2_best

        objective = self._objective(
            weighted_remaining_violations=float(weighted_violation_score(remaining_after_1["remaining_violations"])),
            edit_cost=sim1.edit_cost,
            preservation_penalty=sim1.preservation_penalty,
            plan_deviation=float(remaining_after_1["plan_deviation"]),
            future_conflict_penalty=self.future_estimator.estimate(
                patched_scene_text=sim1.text,
                patched_dynamic_memory=dynamic_memory,
                future_scenes=future_scenes,
            active_weighted_constraints=remaining_after_1["remaining_violations"],
            weighted_future_constraints=weighted_future_constraints,
        ).predicted_future_conflict_penalty,
        )
        stability_penalty = (1.0 - sim1.stability_score) * 2.0
        objective += stability_penalty
        future_estimate_1 = self.future_estimator.estimate(
            patched_scene_text=sim1.text,
            patched_dynamic_memory=dynamic_memory,
            future_scenes=future_scenes,
            active_weighted_constraints=remaining_after_1["remaining_violations"],
            weighted_future_constraints=weighted_future_constraints,
        )
        weighted_remaining_1 = weighted_violation_score(remaining_after_1["remaining_violations"])
        return {
            "objective": objective,
            "targets": list(step1.target_sentence_ids),
            "patch_sequence": list(step1.patch_sequence),
            "remaining_count": len(remaining_after_1["remaining_violations"]),
            "preservation_penalty": sim1.preservation_penalty,
            "future_conflict_penalty": future_estimate_1.predicted_future_conflict_penalty,
            "weighted_remaining_violation_score": weighted_remaining_1,
            "critical_constraints_preserved": self._collect_preserved_critical_constraints(
                original=active_violations,
                remaining=remaining_after_1["remaining_violations"],
            ),
            "deferred_constraints": self._collect_deferred_constraints(remaining_after_1["remaining_violations"]),
            "impacted_future_scene_ids": future_estimate_1.impacted_future_scene_ids,
            "critical_future_constraints_at_risk": future_estimate_1.critical_future_constraints_at_risk,
            "trajectory_length": 1,
            "score_breakdown": {
                "remaining_violations": float(len(remaining_after_1["remaining_violations"])),
                "weighted_remaining_violations": float(weighted_remaining_1),
                "edit_cost": float(sim1.edit_cost),
                "preservation_penalty": float(sim1.preservation_penalty),
                "plan_deviation": float(remaining_after_1["plan_deviation"]),
                "future_conflict_penalty": float(future_estimate_1.predicted_future_conflict_penalty),
                "stability_penalty": float(stability_penalty),
                "objective": float(objective),
            },
            "global_objective_breakdown": {
                "weighted_remaining_violations": float(weighted_remaining_1),
                "edit_cost": float(sim1.edit_cost),
                "preservation_penalty": float(sim1.preservation_penalty),
                "plan_deviation": float(remaining_after_1["plan_deviation"]),
                "future_conflict_penalty": float(future_estimate_1.predicted_future_conflict_penalty),
            },
            "rationale": f"candidate#{candidate_id}: one-step patch trajectory retained.",
        }

    def _objective(
        self,
        weighted_remaining_violations: float | None = None,
        edit_cost: float = 0.0,
        preservation_penalty: float = 0.0,
        plan_deviation: float = 0.0,
        future_conflict_penalty: float = 0.0,
        remaining_violations: int | None = None,
    ) -> float:
        if weighted_remaining_violations is None:
            weighted_remaining_violations = float(remaining_violations or 0)
        return (
            self.lambda_remaining * float(weighted_remaining_violations)
            + self.lambda_edit * float(edit_cost)
            + self.lambda_preservation * float(preservation_penalty)
            + self.lambda_plan_deviation * float(plan_deviation)
            + self.lambda_future_conflict * float(future_conflict_penalty)
        )

    def _rank_candidates_prior(
        self,
        candidates: Sequence[Sequence[str]],
        violations: Sequence[ConstraintViolation],
        extraction: SceneExtraction,
        conf_by_violation: Dict[str, float],
        entropy_risk_prior: Dict[str, float] | None = None,
        entropy_linked_sentence_ids: Sequence[str] | None = None,
        violation_context: Dict[str, object] | None = None,
    ) -> List[Tuple[float, List[str], Dict[str, object]]]:
        ranked: List[Tuple[float, List[str], Dict[str, object]]] = []
        entropy_risk_prior = entropy_risk_prior or {}
        linked = set(entropy_linked_sentence_ids or [])
        context = self._normalize_violation_context(
            violations=violations,
            extraction=extraction,
            violation_context=violation_context,
            entropy_linked_sentence_ids=entropy_linked_sentence_ids,
        )
        scope_penalty = self._patch_scope_penalty(context)
        for candidate in candidates:
            coverage = self._covered_violations(candidate, violations, extraction)
            conf_gain = 0.0
            for violation in violations:
                vid = self._violation_id(violation)
                anchored = self._collect_targets_from_violations([violation])
                if set(candidate).intersection(set(anchored)):
                    conf_gain += conf_by_violation.get(vid, 0.5)
            entropy_gain = 0.0
            # Entropy is only a weak prior and only active when linked to constraint/violation context.
            if linked:
                for sid in set(candidate):
                    if sid in linked:
                        entropy_gain += float(entropy_risk_prior.get(sid, 0.0))
                entropy_gain *= 0.35
            alignment = self._candidate_alignment_features(
                candidate=candidate,
                violations=violations,
                entropy_risk_prior=entropy_risk_prior,
                context=context,
            )
            score = (
                (1.6 * coverage)
                + (1.0 * conf_gain)
                + float(alignment.get("joint_alignment_score", 0.0))
                + (0.25 * entropy_gain)
                - (0.65 * len(set(candidate)))
                - scope_penalty
            )
            ranked.append((float(score), list(candidate), alignment))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _normalize_violation_context(
        self,
        *,
        violations: Sequence[ConstraintViolation],
        extraction: SceneExtraction,
        violation_context: Dict[str, object] | None,
        entropy_linked_sentence_ids: Sequence[str] | None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "violation_sentence_ids": set(self._collect_targets_from_violations(violations)),
            "violation_constraint_ids": set(),
            "critical_constraint_ids": set(),
            "conflict_tokens": set(),
            "entropy_sentence_ids": set(entropy_linked_sentence_ids or []),
            "memory_instability": 0.0,
            "recent_no_gain_alignment_mean": 0.0,
        }
        for item in violations:
            vid = self._violation_id(item)
            context["violation_constraint_ids"].add(vid)
            if bool(item.is_hard) or int(item.constraint_tier) <= 2:
                context["critical_constraint_ids"].add(vid)
            context["conflict_tokens"].update(
                [str(tok).strip().lower() for tok in list(item.related_ids)[:8] if str(tok).strip()]
            )
        if isinstance(violation_context, dict):
            for sid in violation_context.get("violation_sentence_ids", []) or []:
                if str(sid).strip():
                    context["violation_sentence_ids"].add(str(sid).strip())
            for cid in violation_context.get("violation_constraint_ids", []) or []:
                if str(cid).strip():
                    context["violation_constraint_ids"].add(str(cid).strip())
            for cid in violation_context.get("critical_constraint_ids", []) or []:
                if str(cid).strip():
                    context["critical_constraint_ids"].add(str(cid).strip())
            for tok in violation_context.get("conflict_tokens", []) or []:
                if str(tok).strip():
                    context["conflict_tokens"].add(str(tok).strip().lower())
            context["memory_instability"] = float(
                max(0.0, min(1.0, float(violation_context.get("memory_instability", 0.0) or 0.0)))
            )
            context["recent_no_gain_alignment_mean"] = float(
                max(0.0, min(1.0, float(violation_context.get("recent_no_gain_alignment_mean", 0.0) or 0.0)))
            )

        if not context["violation_sentence_ids"] and extraction.sentences:
            context["violation_sentence_ids"].add(extraction.sentences[-1].sentence_id)
        return context

    def _candidate_alignment_features(
        self,
        *,
        candidate: Sequence[str],
        violations: Sequence[ConstraintViolation],
        entropy_risk_prior: Dict[str, float],
        context: Dict[str, Any],
    ) -> Dict[str, object]:
        cand = set(candidate)
        vio_sentences = set(context.get("violation_sentence_ids", set()))
        entropy_sentences = set(context.get("entropy_sentence_ids", set()))
        violation_context_alignment = float(len(cand.intersection(vio_sentences)) / max(1, len(cand)))
        entropy_alignment = 0.0
        entropy_vals = [float(entropy_risk_prior.get(sid, 0.0)) for sid in cand if sid in entropy_sentences]
        if entropy_vals:
            entropy_alignment = float(sum(entropy_vals) / max(1, len(entropy_vals)))
        symbolic_criticality = self._candidate_symbolic_criticality(candidate=candidate, violations=violations)
        memory_instability = float(max(0.0, min(1.0, float(context.get("memory_instability", 0.0)))))
        historical_no_gain_penalty = float(max(0.0, min(1.0, float(context.get("recent_no_gain_alignment_mean", 0.0)))))
        joint = (
            0.50 * violation_context_alignment
            + 0.22 * entropy_alignment
            + 0.20 * symbolic_criticality
            + 0.08 * memory_instability
            - 0.10 * historical_no_gain_penalty
        )
        joint = float(max(0.0, min(1.0, joint)))
        return {
            "joint_alignment_score": joint,
            "hits_violation_context": bool(cand.intersection(vio_sentences)),
            "hits_entropy_context": bool(cand.intersection(entropy_sentences)),
            "violation_context_sentence_ids": sorted(vio_sentences),
            "violation_context_constraint_ids": sorted(set(context.get("violation_constraint_ids", set()))),
            "entropy_context_sentence_ids": sorted(entropy_sentences),
            "score_breakdown": {
                "violation_context_alignment": float(violation_context_alignment),
                "entropy_alignment": float(entropy_alignment),
                "symbolic_criticality": float(symbolic_criticality),
                "memory_instability": float(memory_instability),
                "historical_no_gain_penalty": float(historical_no_gain_penalty),
                "joint_alignment_score": float(joint),
            },
        }

    def _candidate_symbolic_criticality(
        self,
        *,
        candidate: Sequence[str],
        violations: Sequence[ConstraintViolation],
    ) -> float:
        cand = set(candidate)
        if not cand:
            return 0.0
        total = 0.0
        hit = 0.0
        for item in violations:
            weight = 1.0
            if bool(item.is_hard) or int(item.constraint_tier) <= 2:
                weight += 1.0
            if float(item.constraint_weight) >= 6.0:
                weight += 0.5
            total += weight
            anchored = set()
            for anchor in item.anchors:
                anchored.update(anchor.sentence_ids)
            if anchored and cand.intersection(anchored):
                hit += weight
        if total <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, hit / total)))

    def _patch_scope_penalty(self, context: Dict[str, Any]) -> float:
        critical = len(set(context.get("critical_constraint_ids", set())))
        # Keep scope escalation cautious when no clear critical symbolic pressure.
        if critical <= 0:
            return 0.1
        if critical <= 2:
            return 0.05
        return 0.0

    def _collect_violation_targets(
        self,
        scene_plan: ScenePlan,
        extraction: SceneExtraction,
        violations: Sequence[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
    ) -> List[str]:
        targets: List[str] = []
        for violation in violations:
            local = self._targets_for_violation(
                violation=violation,
                extraction=extraction,
                scene_plan=scene_plan,
                static_memory=static_memory,
                dynamic_memory=dynamic_memory,
            )
            for sent_id in local:
                if sent_id not in targets:
                    targets.append(sent_id)
            if len(targets) >= self.max_patch_targets_per_round:
                break
        return targets[: self.max_patch_targets_per_round]

    def _collect_targets_from_violations(self, violations: Sequence[ConstraintViolation]) -> List[str]:
        targets: List[str] = []
        for violation in violations:
            for anchor in violation.anchors:
                for sid in anchor.sentence_ids:
                    if sid not in targets:
                        targets.append(sid)
        return targets

    def _targets_for_violation(
        self,
        violation: ConstraintViolation,
        extraction: SceneExtraction,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
    ) -> List[str]:
        sentence_ids: List[str] = []
        for anchor in violation.anchors:
            sentence_ids.extend(anchor.sentence_ids)
        if sentence_ids:
            return sorted(set(sentence_ids))

        related_tokens: List[str] = list(violation.related_ids)
        related_tokens.extend([scene_plan.scene_id, scene_plan.objective, violation.message, violation.repair_hint])
        related_tokens.extend(scene_plan.required_constraints[:2])
        for entity_id in static_memory.characterization.character_profiles.keys():
            if entity_id in violation.related_ids:
                profile = static_memory.characterization.character_profiles[entity_id]
                related_tokens.extend([entity_id, profile.canonical_name])
        for event in dynamic_memory.timeline_plot.event_timeline[-3:]:
            if event.event_id in violation.related_ids:
                related_tokens.extend([event.event_id, event.description])
        return find_sentence_ids_for_tokens(extraction.sentences, related_tokens)

    def _candidate_target_sets(
        self,
        sentence_ids: Sequence[str],
        base_targets: Sequence[str],
        violations: Sequence[ConstraintViolation],
        dependency_graph: PatchDependencyGraph,
    ) -> List[List[str]]:
        if not sentence_ids:
            return []
        candidates: List[List[str]] = []
        if base_targets:
            candidates.append(list(base_targets[: self.max_patch_targets_per_round]))
        candidates.extend([[sid] for sid in base_targets[: self.max_patch_targets_per_round]])
        candidates.extend(
            dependency_graph.suggest_joint_target_sets(base_targets, self.max_patch_targets_per_round)
        )
        if self.allow_neighbor_adjustment:
            sid_to_idx = {sid: idx for idx, sid in enumerate(sentence_ids)}
            for sid in base_targets[: self.max_patch_targets_per_round]:
                idx = sid_to_idx.get(sid)
                if idx is None:
                    continue
                window = [sentence_ids[idx]]
                if idx > 0:
                    window.insert(0, sentence_ids[idx - 1])
                if idx + 1 < len(sentence_ids):
                    window.append(sentence_ids[idx + 1])
                candidates.append(window[: self.max_patch_targets_per_round])

        if any(v.repair_scope == "paragraph" for v in violations) and base_targets:
            sid_to_idx = {sid: idx for idx, sid in enumerate(sentence_ids)}
            valid_idxs = [sid_to_idx[sid] for sid in base_targets if sid in sid_to_idx]
            if valid_idxs:
                start = max(0, min(valid_idxs) - 1)
                end = min(len(sentence_ids), max(valid_idxs) + 2)
                candidates.append(list(sentence_ids[start:end]))

        dedup: List[List[str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            trimmed = sorted(set(candidate))[: self.max_patch_targets_per_round]
            if not trimmed:
                continue
            key = "|".join(trimmed)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(trimmed)
        return dedup

    def _build_step_patch_plan(
        self,
        scene_plan: ScenePlan,
        violations: Sequence[ConstraintViolation],
        targets: Sequence[str],
        round_idx: int,
        step_idx: int,
        sentence_ids: Sequence[str],
    ) -> PatchPlan:
        target_set = set(targets)
        patches: List[SentencePatch] = []
        for idx, violation in enumerate(violations):
            vtargets: List[str] = []
            anchored_ids: List[str] = []
            for anchor in violation.anchors:
                for sid in anchor.sentence_ids:
                    anchored_ids.append(sid)
                    if sid in target_set and sid not in vtargets:
                        vtargets.append(sid)
            if not vtargets and (not anchored_ids) and targets:
                vtargets = [targets[0]]
            if not vtargets:
                continue
            op = self._op_type_for_violation(violation)
            constraints = list(scene_plan.required_constraints)
            constraints.extend(scene_plan.must_keep_constraints)
            constraints.extend([violation.message, violation.repair_hint])
            patches.append(
                SentencePatch(
                    patch_id=f"patch_{scene_plan.scene_id}_{round_idx:02d}_{step_idx}_{idx:03d}",
                    op_type=op,
                    target_sentence_ids=vtargets[:2],
                    rationale=f"{violation.rule_type}: {violation.message}",
                    linked_violation_ids=[self._violation_id(violation)],
                    constraints_to_satisfy=[x for x in constraints if str(x).strip()],
                )
            )
        if not patches and targets:
            patches.append(
                SentencePatch(
                    patch_id=f"patch_{scene_plan.scene_id}_{round_idx:02d}_{step_idx}_fallback",
                    op_type="replace",
                    target_sentence_ids=[targets[0]],
                    rationale="fallback_step_patch",
                    constraints_to_satisfy=list(scene_plan.required_constraints),
                )
            )
        return PatchPlan(
            plan_id=f"patch_plan_{scene_plan.scene_id}_{round_idx:02d}_step{step_idx}",
            target_sentence_ids=list(sorted(set(targets))),
            protected_sentence_ids=[sid for sid in sentence_ids if sid not in set(targets)],
            patch_sequence=patches,
            fallback_level="paragraph" if len(set(targets)) > 1 else "scene",
            expected_fixed_violations=[self._violation_id(v) for v in violations],
        )

    def _revalidate_simulation(
        self,
        scene_text: str,
        violations: Sequence[ConstraintViolation],
        scene_plan: ScenePlan,
    ) -> Dict[str, object]:
        remaining: List[ConstraintViolation] = []
        for violation in violations:
            if not self._is_violation_resolved(violation, scene_text):
                remaining.append(violation)
        plan_deviation = 0.0
        lowered = scene_text.lower()
        for token in scene_plan.required_constraints[:3]:
            if token and str(token).strip().lower() not in lowered:
                plan_deviation += 1.0
        for token in scene_plan.forbidden_constraints[:3]:
            if token and str(token).strip().lower() in lowered:
                plan_deviation += 1.0
        return {"remaining_violations": remaining, "plan_deviation": plan_deviation}

    def _is_violation_resolved(self, violation: ConstraintViolation, scene_text: str) -> bool:
        message = violation.message.lower()
        lowered = scene_text.lower()
        req_match = re.search(r"required constraint:\s*'([^']+)'", message)
        if req_match:
            token = req_match.group(1).strip().lower()
            return bool(token and token in lowered)
        forbid_match = re.search(r"forbidden constraint:\s*'([^']+)'", message)
        if forbid_match:
            token = forbid_match.group(1).strip().lower()
            return not bool(token and token in lowered)
        missing_events = re.search(r"missing required events:\s*(\[[^\]]+\])", message)
        if missing_events:
            text = missing_events.group(1)
            tokens = [t.strip(" '\"").lower() for t in text.strip("[]").split(",") if t.strip()]
            return all(token in lowered for token in tokens if token)
        if "event order regresses" in message:
            return any(tok in lowered for tok in ("then", "after", "later", "finally"))
        if "name drift" in message:
            return any(tok in lowered for tok in violation.related_ids if tok.startswith("char_"))
        # Fallback: treat as unresolved if repair hint keywords absent.
        repair_tokens = [t for t in re.findall(r"[a-zA-Z_]{4,}", violation.repair_hint.lower()) if t]
        if repair_tokens and not any(tok in lowered for tok in repair_tokens[:2]):
            return False
        return True

    def _simulate_sentence_edit(self, sentence: str, patch: SentencePatch) -> str:
        text = sentence or ""
        op = patch.op_type.lower()
        if op == "delete":
            return ""
        forbidden_tokens = self._extract_forbidden_tokens(patch.constraints_to_satisfy + [patch.rationale])
        required_tokens = self._extract_required_tokens(patch.constraints_to_satisfy + [patch.rationale])
        for token in forbidden_tokens:
            text = re.sub(rf"\b{re.escape(token)}\b", "", text, flags=re.IGNORECASE)
        if op in {"temporal_fix", "paraphrase"} and "then" not in text.lower():
            text = f"{text.rstrip()} Then"
        if required_tokens:
            for token in required_tokens[:2]:
                if token.lower() not in text.lower():
                    text = f"{text.rstrip()} {token}"
        text = re.sub(r"\s+", " ", text).strip()
        if text and not text.endswith((".", "!", "?")):
            text = f"{text}."
        if not text:
            text = "The scene remains locally consistent."
        return text

    def _extract_forbidden_tokens(self, lines: Sequence[str]) -> List[str]:
        tokens: List[str] = []
        for line in lines:
            text = str(line or "")
            for match in re.findall(r"forbidden(?: constraint)?:\s*'([^']+)'", text, flags=re.IGNORECASE):
                tokens.append(match.strip().lower())
            if "avoid " in text.lower():
                tail = text.lower().split("avoid ", 1)[1]
                tokens.extend([t.strip() for t in re.split(r"[,.]", tail)[:1] if t.strip()])
        return [t for t in tokens if t]

    def _extract_required_tokens(self, lines: Sequence[str]) -> List[str]:
        tokens: List[str] = []
        for line in lines:
            text = str(line or "")
            for match in re.findall(r"required(?: constraint)?:\s*'([^']+)'", text, flags=re.IGNORECASE):
                tokens.append(match.strip())
            if "must " in text.lower() and "'" not in text:
                tail = text.split("must ", 1)[1] if "must " in text else ""
                if tail.strip():
                    tokens.append(tail.strip().split(".")[0][:80])
        return [t for t in tokens if t]

    def _token_diff_cost(self, before: str, after: str) -> float:
        b = before.split()
        a = after.split()
        return float(abs(len(a) - len(b)) + len(set(a).symmetric_difference(set(b))) * 0.15 + 1.0)

    def _stability_score(self, original: str, patched: str) -> float:
        o = set(re.findall(r"[a-zA-Z0-9_]+", original.lower()))
        p = set(re.findall(r"[a-zA-Z0-9_]+", patched.lower()))
        if not o and not p:
            return 1.0
        if not o or not p:
            return 0.0
        jac = len(o.intersection(p)) / max(1, len(o.union(p)))
        return float(max(0.0, min(1.0, jac)))

    def _resolve_index_for_sim_id(
        self,
        sentence_id: str,
        by_id: Dict[str, int],
        ordered_text: Sequence[str],
    ) -> int | None:
        if sentence_id in by_id:
            return by_id[sentence_id]
        m = re.search(r"_(\d{3})$", sentence_id)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(ordered_text):
                return idx
        return None

    def _covered_violations(
        self,
        targets: Sequence[str],
        violations: Sequence[ConstraintViolation],
        extraction: SceneExtraction,
    ) -> int:
        target_set = set(targets)
        if not target_set:
            return 0
        covered = 0
        for violation in violations:
            anchored: List[str] = []
            for anchor in violation.anchors:
                anchored.extend(anchor.sentence_ids)
            if anchored and target_set.intersection(set(anchored)):
                covered += 1
                continue
            if not anchored:
                local = self._fallback_violation_target(violation, extraction)
                if target_set.intersection(set(local)):
                    covered += 1
        return covered

    def _fallback_violation_target(
        self,
        violation: ConstraintViolation,
        extraction: SceneExtraction,
    ) -> List[str]:
        tokens = [violation.message] + list(violation.related_ids)
        return find_sentence_ids_for_tokens(extraction.sentences, tokens)

    def _fallback_level(
        self,
        violations: Sequence[ConstraintViolation],
        num_targets: int,
        needs_replan: bool,
        fatal: bool,
    ) -> str:
        if needs_replan:
            return "plan"
        if fatal:
            return "scene"
        if any(v.repair_scope == "paragraph" for v in violations) or num_targets > 2:
            return "paragraph"
        return "scene"

    def _op_type_for_violation(self, violation: ConstraintViolation) -> str:
        rule = violation.rule_type.lower()
        message = violation.message.lower()
        if "timeline" in rule or "event order" in message or "temporal" in message:
            return "temporal_fix"
        if "character" in rule or "name drift" in message:
            return "attribute_fix"
        if "operator" in rule and "missing" in message:
            return "insert_after"
        if "forbidden" in message:
            if rule == "world_rule_consistency":
                return "replace"
            return "delete"
        if violation.repair_scope == "paragraph":
            return "paraphrase"
        return "replace"

    def _violation_id(self, violation: ConstraintViolation) -> str:
        rel = ",".join(sorted(set(str(item) for item in violation.related_ids)))[:60]
        return f"{violation.rule_type}:{rel or 'none'}"

    def _collect_preserved_critical_constraints(
        self,
        original: Sequence[ConstraintViolation],
        remaining: Sequence[ConstraintViolation],
    ) -> List[str]:
        original_critical = {
            self._violation_id(v)
            for v in original
            if v.constraint_tier == 1 or v.is_hard or v.constraint_weight >= 6.0
        }
        remaining_critical = {
            self._violation_id(v)
            for v in remaining
            if v.constraint_tier == 1 or v.is_hard or v.constraint_weight >= 6.0
        }
        return sorted(original_critical - remaining_critical)

    def _collect_deferred_constraints(self, remaining: Sequence[ConstraintViolation]) -> List[str]:
        return sorted(
            self._violation_id(v)
            for v in remaining
            if v.constraint_tier >= 3 or v.constraint_weight <= 1.5
        )

    def _violation_confidence(self, violation: ConstraintViolation) -> float:
        if not violation.anchors:
            return 0.4
        scores = []
        for anchor in violation.anchors:
            score = float(anchor.confidence_score if hasattr(anchor, "confidence_score") else anchor.grounding_confidence)
            scores.append(max(0.0, min(1.0, score)))
        return max(scores) if scores else 0.4

    def _normalize_weighted_future_constraints(
        self,
        payload: Sequence[dict],
    ) -> List[WeightedConstraintItem]:
        items: List[WeightedConstraintItem] = []
        for row in payload:
            if isinstance(row, WeightedConstraintItem):
                items.append(row)
                continue
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            items.append(
                WeightedConstraintItem(
                    text=text,
                    weight=float(row.get("weight", 1.0)),
                    tier=int(row.get("tier", 3)),
                    is_hard=bool(row.get("is_hard", False)),
                    source=str(row.get("source", "weighted_future")),
                )
            )
        return items
