"""Incremental scene-by-scene writer with patch-first repair and local replan."""

from __future__ import annotations

from dataclasses import asdict
import logging
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ConWriter.config.schema import ConWriterConfig
from ConWriter.experiment.diagnostics import (
    aggregate_story_diagnostics,
    build_scene_diagnostic_record,
)
from ConWriter.experiment.policies import build_variant_policy_bundle
from ConWriter.experiment.variant_spec import ExperimentVariantSpec
from ConWriter.pipeline.weighted_constraints import build_weighted_tiered_constraints
from ConWriter.memory.dynamic_memory import DynamicMemoryManager
from ConWriter.memory.static_memory import StaticMemoryBuilder
from ConWriter.pipeline.local_repair import LocalRepairer
from ConWriter.pipeline.outputs import build_output_record
from ConWriter.pipeline.patch_planner import PatchPlanner
from ConWriter.pipeline.repair_metrics import RepairMetricsTracker, RepairOscillationDetector
from ConWriter.pipeline.state import initialize_generation_state
from ConWriter.pipeline.trace_writer import TraceWriter
from ConWriter.planner.local_replanner import LocalReplanner
from ConWriter.planner.story_planner import StoryPlanner
from ConWriter.reasoning.entropy_monitor import EntropyRiskMonitor
from ConWriter.reasoning.length_control import LengthController
from ConWriter.reasoning.model_experience_bank import ModelExperienceBank
from ConWriter.reasoning.scene_extractor import SceneExtractor
from ConWriter.reasoning.scene_generator import SceneGenerator
from ConWriter.reasoning.state_reasoning import StateReasoner
from ConWriter.symbolic.constraint_checker import ConstraintChecker
from ConWriter.symbolic.transition_validator import TransitionValidator
from ConWriter.utils.state_grounding import (
    build_state_grounding_bundle,
    evaluate_state_realization_with_grounding,
)
from ConWriter.utils.types import (
    ConWriterOutputRecord,
    ConWriterPromptSample,
    ConsistencyReport,
    ConstraintViolation,
    EntropyRiskProfile,
    GenerationState,
    LocalReplanResult,
    ScenePlan,
    StoryChunk,
    SymbolicFinding,
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class IncrementalWriter:
    """Main controller for constrained incremental generation."""

    def __init__(self, config: ConWriterConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("ConWriter.incremental_writer")

        self.static_builder = StaticMemoryBuilder()
        self.dynamic_manager = DynamicMemoryManager(
            max_history_entries=config.memory.max_history_entries
        )
        self.planner = StoryPlanner(config.incremental, config.planning)
        self.local_replanner = LocalReplanner(
            window_scenes=config.generation_controls.local_replan_window_scenes
        )
        self.scene_generator = SceneGenerator(config.llm, logger=self.logger)
        self.entropy_monitor = EntropyRiskMonitor(config.entropy_monitor)
        self.experience_bank = ModelExperienceBank(config.experience_bank, logger=self.logger)
        self.length_controller = LengthController(config.length_control, logger=self.logger)
        self.scene_extractor = SceneExtractor(config.llm, logger=self.logger)
        self.constraint_checker = ConstraintChecker()
        self.state_reasoner = StateReasoner()
        self.transition_validator = TransitionValidator()
        self.local_repair = LocalRepairer(config.llm, logger=self.logger)
        self.patch_planner = PatchPlanner(
            max_patch_targets_per_round=config.generation_controls.max_patch_targets_per_round,
            allow_neighbor_adjustment=config.generation_controls.allow_neighbor_adjustment,
            lookahead_top_k=config.generation_controls.patch_lookahead_top_k,
            low_confidence_threshold=config.generation_controls.low_confidence_anchor_threshold,
        )
        self.trace_writer = TraceWriter(config.trace)

    def run_single(self, sample: ConWriterPromptSample) -> Tuple[GenerationState, ConWriterOutputRecord]:
        """Run one full incremental generation pass for one prompt."""
        variant = ExperimentVariantSpec.from_config(self.config.variant)
        flags = variant.flags()
        policies = build_variant_policy_bundle(flags)
        generation_mode = str(flags["generation_layer"])
        diagnostics_records: List[Dict[str, object]] = []
        prev_scene_entropy_mean = 0.0
        self._configure_variant_runtime(flags, policies)
        resolved_model_name = (
            self.config.llm.model
            if self.config.llm.enabled and self.config.llm.model
            else self.config.chunk_generation.model_name
        )
        retrieved_experience_items: List[Dict[str, object]] = []
        retrieved_experience_seen: set[str] = set()
        under_generation_warning_triggered = False
        premature_closure_warning_triggered = False
        min_scene_words_guard = 50

        static_memory = self.static_builder.build(sample)
        story_plan = self.planner.build_plan(sample, static_memory)
        dynamic_memory = self.dynamic_manager.initialize(
            static_memory,
            seed_characters=self.config.memory.seed_characters_into_dynamic,
        )
        state = initialize_generation_state(static_memory, dynamic_memory)
        state.story_plan = story_plan
        self.trace_writer.start_story(
            experiment_name=self.config.experiment_name,
            prompt_id=sample.prompt_id,
            plan=story_plan,
            static_memory=static_memory,
        )

        scenes: List[ScenePlan] = list(story_plan.iter_scenes())
        desired_target_length = self.length_controller.estimate_desired_target_length(
            planning_target_words=int(self.config.planning.target_story_words),
            scenes=scenes,
            fallback_scene_target=int(self.config.incremental.scene_target_words),
        )
        length_targets = self.length_controller.compute_targets(
            desired_target_length=int(desired_target_length),
            model_id=resolved_model_name,
        )
        desired_target_length = int(length_targets.get("desired_target_length", desired_target_length) or desired_target_length)
        requested_target_length = int(
            length_targets.get("requested_target_length", desired_target_length) or desired_target_length
        )
        length_compensation_factor = float(length_targets.get("length_compensation_factor", 1.0) or 1.0)
        max_sentence_patch_rounds = max(0, int(self.config.generation_controls.max_sentence_patch_rounds))
        max_paragraph_patch_rounds = max(0, int(self.config.generation_controls.max_paragraph_patch_rounds))
        max_regen_rounds = max(0, int(self.config.generation_controls.max_scene_regen_rounds))
        max_patch_rounds_per_scene = max(1, int(self.config.generation_controls.max_patch_rounds_per_scene))
        max_local_replans = max(0, int(self.config.generation_controls.max_local_replans_per_story))
        reject_on_violation = bool(self.config.consistency.reject_on_violation)
        prefer_sentence_repair_first = bool(
            self.config.generation_controls.prefer_sentence_repair_first
        )
        regen_only_on_fatal = bool(self.config.generation_controls.regen_only_on_fatal)
        if not flags["use_patch_pipeline"]:
            max_sentence_patch_rounds = 0
            max_paragraph_patch_rounds = 0
        if flags["scene_rewrite_only"]:
            max_sentence_patch_rounds = 0
            max_paragraph_patch_rounds = 0
        if flags["sentence_patch_only"]:
            max_paragraph_patch_rounds = 0
        if not flags["paragraph_patch_enabled"]:
            max_paragraph_patch_rounds = 0
        if flags["plain_generate_only_mode"]:
            max_sentence_patch_rounds = 0
            max_paragraph_patch_rounds = 0
        max_sentence_patch_rounds, max_paragraph_patch_rounds = policies.repair.resolve_patch_budgets(
            max_sentence_patch_rounds=max_sentence_patch_rounds,
            max_paragraph_patch_rounds=max_paragraph_patch_rounds,
        )

        metrics = RepairMetricsTracker() if self.config.generation_controls.enable_repair_metrics else None
        local_replans_used = 0
        scene_cursor = 0

        def _record_retrieved_experience(stage: str, scene_id: str, items: Sequence[Dict[str, object]]) -> None:
            for item in list(items):
                payload = {
                    "stage": str(stage),
                    "scene_id": str(scene_id),
                    "model_id": str(item.get("model_id", "")),
                    "task_type": str(item.get("task_type", "")),
                    "conflict_type": str(item.get("conflict_type", "")),
                    "failure_pattern": str(item.get("failure_pattern", "")),
                    "prevention_guidance": str(item.get("prevention_guidance", "")),
                    "count": int(item.get("count", 1) or 1),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
                key = (
                    f"{payload['stage']}|{payload['scene_id']}|{payload['failure_pattern']}|"
                    f"{payload['prevention_guidance']}"
                )
                if key in retrieved_experience_seen:
                    continue
                retrieved_experience_seen.add(key)
                retrieved_experience_items.append(payload)

        while scene_cursor < len(scenes):
            scene = scenes[scene_cursor]
            if metrics is not None:
                metrics.start_scene(scene.scene_id)

            state.current_step = scene.scene_index
            state.dynamic_memory.current_chapter_id = scene.chapter_id
            state.dynamic_memory.current_scene_id = scene.scene_id
            state_t = self.dynamic_manager.get_state(state.dynamic_memory)
            reasoning = self.state_reasoner.reason(state_t, scene)
            state.dynamic_memory.inferred_constraints = list(reasoning.inferred_constraints)

            pre_violations: List[ConstraintViolation] = []
            if flags["generation_layer"] != "plain_generate_only":
                pre_violations = self.constraint_checker.precheck_scene(
                    static_memory=state.static_memory,
                    dynamic_memory=state.dynamic_memory,
                    scene_plan=scene,
                )
                pre_violations.extend(self.transition_validator.precheck_state(reasoning))
            if metrics is not None:
                metrics.record_validation(scene.scene_id, pre_violations)

            prepared_constraints = self.constraint_checker.build_scene_constraints(
                static_memory=state.static_memory,
                dynamic_memory=state.dynamic_memory,
                scene_plan=scene,
            )
            initial_weighted_constraints = build_weighted_tiered_constraints(
                required=prepared_constraints.get("required", []),
                must_keep=prepared_constraints.get("must_keep", []),
                forbidden=prepared_constraints.get("forbidden", []),
                inferred=[],
                propagated=[],
                high_conf_violations=[],
                deferred_constraints=[],
            )
            prepared_constraints.update(
                {
                    "state_summary": asdict(reasoning.state),
                    "allowed_transitions": list(reasoning.allowed_transitions),
                    "forbidden_transitions": list(reasoning.forbidden_transitions),
                    "required_state_changes": list(reasoning.required_state_changes),
                    "candidate_operators": [asdict(item) for item in reasoning.candidate_operators],
                    "forbidden_operators": list(reasoning.forbidden_operators),
                    "selected_operator": (
                        asdict(reasoning.selected_operator) if reasoning.selected_operator is not None else {}
                    ),
                    "execution_spec": asdict(reasoning.execution_spec),
                    "propagated_constraints": list(reasoning.propagated_constraints),
                    "conflict_candidates": list(reasoning.conflict_candidates),
                    "inferred_constraints": self.state_reasoner.constraints_to_prompt_lines(
                        reasoning.inferred_constraints
                    ),
                    "weighted_constraints_tiered": [
                        {"text": item.text, "weight": item.weight, "tier": item.tier, "is_hard": item.is_hard}
                        for item in initial_weighted_constraints
                    ],
                    "generation_mode": generation_mode,
                }
            )
            if generation_mode == "plain_generate_only":
                prepared_constraints["required"] = []
                prepared_constraints["forbidden"] = []
                prepared_constraints["must_keep"] = []
                prepared_constraints["inferred_constraints"] = []
                prepared_constraints["propagated_constraints"] = []
                prepared_constraints["hard_constraints"] = []
                prepared_constraints["soft_constraints"] = []
                prepared_constraints["weighted_constraints_tiered"] = []
            if not flags["weighted_constraints_enabled"]:
                prepared_constraints["weighted_constraints_tiered"] = []
                prepared_constraints["deferred_constraints"] = []
            if not flags["confidence_aware_grounding"]:
                prepared_constraints["high_confidence_violations"] = []

            pre_errors = [v for v in pre_violations if v.severity.lower() == "error"]
            should_block_precheck = bool(pre_errors) or (
                bool(pre_violations) and self.config.incremental.strict_precheck
            )
            if should_block_precheck:
                report = self._build_report_from_violations(pre_violations, scene.scene_id)
                state.last_report = report
                state.dynamic_memory.consistency_reports.append(report)
                state.dynamic_memory.revision_records.append(
                    {
                        "scene_id": scene.scene_id,
                        "round": 0,
                        "reason": [v.message for v in pre_violations],
                        "stage": "precheck",
                    }
                )
                if metrics is not None:
                    metrics.finalize_scene(scene.scene_id, accepted=False)
                self.logger.warning("Precheck failed for %s: %s", scene.scene_id, report.messages[:2])
                if self.config.incremental.stop_on_failed_scene:
                    break
                scene_cursor += 1
                continue

            accepted = False
            final_report: ConsistencyReport | None = None
            final_delta = None
            final_action = None
            had_revision = False
            last_violations: List[ConstraintViolation] = []
            scene_text = ""
            plan_deviation_detected = False
            failed_reason = ""
            oscillation_detector = RepairOscillationDetector(
                window=self.config.generation_controls.oscillation_window,
                threshold=self.config.generation_controls.oscillation_threshold,
            )
            pending_preservation_violations: List[ConstraintViolation] = []
            last_patch_scope = ""
            scene_regen_rounds = 0
            scene_preservation_failures = 0
            scene_oscillation_detected = False
            last_unchanged_ratio = 0.0
            last_executed_patch_plan = None
            last_patch_plan = None
            entropy_profile = EntropyRiskProfile()
            entropy_triggered_validation = False
            entropy_triggered_patch_escalation = False
            entropy_triggered_replan = False
            validation_mode = "standard"
            validation_budget = 1
            scene_uncertainty_history: List[float] = []
            scene_joint_action_events: List[Dict[str, object]] = []
            scene_patch_execution_records: List[Dict[str, object]] = []
            scene_memory_binding_mode = "normal_binding"
            scene_generation_control_mode = "normal_generation"
            scene_binding_decision_reasons: List[str] = []
            scene_strengthened_memory_blocks: List[str] = []
            scene_strengthened_constraints: List[str] = []
            last_rewrite_control_context: Dict[str, object] = {}
            last_generation_control_context: Dict[str, object] = {}
            scene_retrieved_experience_items: List[Dict[str, object]] = []
            scene_length_progress_ratio = 0.0
            scene_under_generation_warning = False
            scene_premature_closure_warning = False
            scene_patch_exhausted_targeted = False
            scene_patch_attempted_targeted = False
            scene_word_count_first_pass = 0
            scene_too_short_guard_triggered = False

            for regen_round in range(max_regen_rounds + 1):
                if (
                    regen_round > 0
                    and prefer_sentence_repair_first
                    and regen_only_on_fatal
                    and (not any(v.fatal for v in last_violations))
                ):
                    failed_reason = failed_reason or "scene_regen_blocked_non_fatal"
                    break
                regen_constraints = dict(prepared_constraints)
                if regen_round > 0 and last_violations:
                    scene_regen_rounds += 1
                    high_conf_lines = self._collect_high_conf_violation_lines(last_violations)
                    if high_conf_lines:
                        regen_constraints["high_confidence_violations"] = high_conf_lines
                        regen_constraints["hard_constraints"] = list(
                            dict.fromkeys(
                                list(regen_constraints.get("required", []))
                                + list(regen_constraints.get("must_keep", []))
                                + high_conf_lines
                            )
                        )
                        regen_constraints["soft_constraints"] = list(
                            dict.fromkeys(
                                list(regen_constraints.get("inferred_constraints", []))
                                + list(regen_constraints.get("propagated_constraints", []))
                            )
                        )
                if entropy_profile.linked_constraint_ids and entropy_profile.final_risk_tier == "high_risk":
                    regen_constraints["entropy_critical_constraints"] = list(entropy_profile.linked_constraint_ids[:6])
                scene_experience_items = self.experience_bank.retrieve_top_k(
                    model_id=resolved_model_name,
                    task_type=sample.task_type,
                    generation_stage="scene_generation",
                    conflict_type="",
                    top_k=int(self.config.experience_bank.max_retrieved_scene_items),
                    exclude_sample_id=sample.prompt_id,
                )
                scene_experience_cautions = self.experience_bank.format_prompt_cautions(
                    scene_experience_items,
                    max_items=int(self.config.experience_bank.max_retrieved_scene_items),
                )
                if scene_experience_cautions:
                    regen_constraints["model_experience_cautions"] = list(scene_experience_cautions)
                if scene_experience_items and regen_round == 0:
                    scene_retrieved_experience_items = list(scene_experience_items)
                    _record_retrieved_experience("scene_generation", scene.scene_id, scene_experience_items)

                story_so_far_text = "\n\n".join(chunk.text for chunk in state.story_chunks if chunk.accepted)
                latest_scene_hint = state.story_chunks[-1].text if state.story_chunks else ""
                unresolved_threads = list(state.dynamic_memory.timeline_plot.active_plot_threads[-6:]) + list(
                    state.dynamic_memory.timeline_plot.unresolved_foreshadowing[-4:]
                )
                length_status = self.length_controller.monitor_progress(
                    current_word_count=self.length_controller.count_words(story_so_far_text),
                    requested_target_length=requested_target_length,
                    scene_index=int(scene_cursor),
                    total_scenes=int(len(scenes)),
                    latest_scene_text=latest_scene_hint,
                    unresolved_threads=unresolved_threads,
                )
                scene_length_progress_ratio = float(length_status.get("progress_ratio", 0.0) or 0.0)
                scene_under_generation_warning = bool(
                    length_status.get("under_generation_warning_triggered", False)
                )
                scene_premature_closure_warning = bool(
                    length_status.get("premature_closure_warning_triggered", False)
                )
                under_generation_warning_triggered = under_generation_warning_triggered or scene_under_generation_warning
                premature_closure_warning_triggered = (
                    premature_closure_warning_triggered or scene_premature_closure_warning
                )
                length_expansion_guidance = list(length_status.get("expansion_guidance", []) or [])
                length_control_guidance = self.length_controller.build_generation_guidance(
                    requested_target_length=requested_target_length,
                    progress_ratio=scene_length_progress_ratio,
                    expansion_guidance=length_expansion_guidance,
                )
                regen_constraints["length_desired_target_words"] = int(desired_target_length)
                regen_constraints["length_requested_target_words"] = int(requested_target_length)
                regen_constraints["length_compensation_factor"] = float(length_compensation_factor)
                regen_constraints["length_progress_ratio"] = float(scene_length_progress_ratio)
                regen_constraints["length_control_guidance"] = list(length_control_guidance)
                regen_constraints["length_expansion_guidance"] = list(length_expansion_guidance)
                generation_control_context = self._build_generation_control_context(
                    stage="scene_generation",
                    scene_plan=scene,
                    static_memory=state.static_memory,
                    dynamic_memory=state.dynamic_memory,
                    prepared_constraints=regen_constraints,
                    entropy_profile=entropy_profile,
                    violations=(last_violations if (regen_round > 0 and last_violations) else pre_violations),
                    flags=flags,
                )
                if length_expansion_guidance:
                    merged_guidance = list(
                        dict.fromkeys(
                            list(generation_control_context.get("control_guidance", []))
                            + list(length_expansion_guidance)
                        )
                    )[: int(max(1, self.config.length_control.max_guidance_lines + 2))]
                    generation_control_context["control_guidance"] = merged_guidance
                    reasons = list(generation_control_context.get("decision_reasons", []))
                    if "length_progress_warning" not in reasons:
                        reasons.append("length_progress_warning")
                    if scene_premature_closure_warning and "premature_closure_warning" not in reasons:
                        reasons.append("premature_closure_warning")
                    generation_control_context["decision_reasons"] = reasons
                generation_control_context["length_control"] = {
                    "desired_target_length": int(desired_target_length),
                    "requested_target_length": int(requested_target_length),
                    "length_compensation_factor": float(length_compensation_factor),
                    "progress_ratio": float(scene_length_progress_ratio),
                    "under_generation_warning_triggered": bool(scene_under_generation_warning),
                    "premature_closure_warning_triggered": bool(scene_premature_closure_warning),
                }
                regen_constraints = self._apply_binding_policy_to_constraints(
                    prepared_constraints=regen_constraints,
                    control_context=generation_control_context,
                )
                scene_memory_binding_mode = str(generation_control_context.get("memory_binding_mode", "normal_binding"))
                scene_generation_control_mode = str(
                    generation_control_context.get("generation_control_mode", "normal_generation")
                )
                scene_binding_decision_reasons = list(generation_control_context.get("decision_reasons", []))
                scene_strengthened_memory_blocks = list(generation_control_context.get("strengthened_memory_blocks", []))
                scene_strengthened_constraints = list(generation_control_context.get("strengthened_constraints", []))
                last_generation_control_context = dict(generation_control_context)
                if regen_round > 0 and metrics is not None:
                    metrics.record_scene_regen(scene.scene_id)
                self.trace_writer.write_scene_memory_before(scene.scene_id, state.dynamic_memory)
                draft = self.scene_generator.generate_scene(
                    scene_plan=scene,
                    static_memory=state.static_memory,
                    dynamic_memory=state.dynamic_memory,
                    recent_chunks=state.story_chunks,
                    prepared_constraints=regen_constraints,
                    attempt=regen_round,
                    trace_prompt_callback=lambda prompt_text, sid=scene.scene_id: self.trace_writer.write_scene_prompt(
                        sid,
                        prompt_text,
                    ),
                )
                scene_text = draft.text
                scene_word_count = self.length_controller.count_words(scene_text)
                if regen_round == 0:
                    scene_word_count_first_pass = int(scene_word_count)
                if scene_word_count < int(min_scene_words_guard):
                    scene_too_short_guard_triggered = True
                    failed_reason = "scene_too_short_or_empty"
                    last_violations = [
                        ConstraintViolation(
                            rule_type="scene_generation",
                            message=(
                                f"scene generation too short: {scene_word_count} words "
                                f"(threshold={int(min_scene_words_guard)})"
                            ),
                            severity="error",
                            fatal=True,
                            needs_replan=False,
                            patchable=False,
                        )
                    ]
                    state.dynamic_memory.revision_records.append(
                        {
                            "scene_id": scene.scene_id,
                            "round": int(regen_round),
                            "stage": "scene_generation_guard",
                            "reason": [
                                f"empty_or_too_short_scene:{scene_word_count}<"
                                f"{int(min_scene_words_guard)}"
                            ],
                            "scene_word_count": int(scene_word_count),
                            "threshold": int(min_scene_words_guard),
                        }
                    )
                    if metrics is not None:
                        metrics.record_validation(scene.scene_id, last_violations)
                    self.logger.warning(
                        "Scene %s generation too short (%s words < %s), retrying regen round.",
                        scene.scene_id,
                        scene_word_count,
                        int(min_scene_words_guard),
                    )
                    continue
                sentence_patch_round = 0
                paragraph_patch_round = 0
                local_patch_round = 0
                while True:
                    extraction = self.scene_extractor.extract_scene(
                        scene_plan=scene,
                        scene_text=scene_text,
                        state=state,
                    )
                    self.trace_writer.write_scene_extraction(scene.scene_id, extraction)
                    entropy_profile = self._compute_entropy_profile(
                        scene_plan=scene,
                        extraction=extraction,
                        prepared_constraints=prepared_constraints,
                        previous_entropy_mean=prev_scene_entropy_mean,
                        flags=flags,
                        generation_metadata=dict(getattr(draft, "metadata", {}) or {}),
                        generation_tokens=list(getattr(draft, "tokens", []) or []),
                        generation_token_logprobs=list(getattr(draft, "token_logprobs", []) or []),
                        generation_top_logprobs=list(getattr(draft, "top_logprobs", []) or []),
                        round_uncertainty_history=scene_uncertainty_history,
                    )
                    entropy_triggered_validation = entropy_triggered_validation or bool(
                        entropy_profile.triggered_validation
                    )
                    entropy_triggered_patch_escalation = entropy_triggered_patch_escalation or bool(
                        entropy_profile.triggered_patch_escalation
                    )
                    entropy_triggered_replan = entropy_triggered_replan or bool(
                        entropy_profile.triggered_replan
                    )
                    validation_mode = str(entropy_profile.triggered_validation_mode or "standard")
                    validation_budget = int(max(1, entropy_profile.triggered_validation_budget))
                    transition_validation_runs = int(max(1, validation_budget))
                    if validation_mode != "escalated":
                        transition_validation_runs = 1
                    delta = extraction.to_memory_delta()
                    delta.chunk_id = scene.scene_id
                    state.proposed_deltas.append(delta)
                    state.last_delta = delta

                    action = self.state_reasoner.derive_action(
                        state=state_t,
                        scene_plan=scene,
                        scene_text=scene_text,
                        extraction=extraction,
                        selected_operator=reasoning.selected_operator,
                    )
                    transition_ok = True
                    transition_violations: List[ConstraintViolation] = []
                    for _ in range(transition_validation_runs):
                        transition_ok, transition_violations = self.transition_validator.validate(
                            state=state_t,
                            action=action,
                            constraints=reasoning.inferred_constraints,
                            forbidden_transitions=reasoning.forbidden_transitions,
                            selected_operator=reasoning.selected_operator,
                            sentence_units=extraction.sentences,
                        )
                        if (not transition_ok) or transition_violations:
                            break

                    candidate_memory = self.dynamic_manager.preview_update(
                        state.dynamic_memory,
                        delta,
                    )
                    post_report, post_violations = self.constraint_checker.check_scene(
                        static_memory=state.static_memory,
                        candidate_memory=candidate_memory,
                        scene_plan=scene,
                        delta=delta,
                        scene_text=scene_text,
                        scene_extraction=extraction,
                    )

                    violations = pre_violations + transition_violations + post_violations + pending_preservation_violations
                    violations = self._project_constraint_layer_violations(
                        violations=violations,
                        policy=policies.constraints,
                    )
                    joint_selector = self._select_joint_uncertainty_action(
                        entropy_profile=entropy_profile,
                        prepared_constraints=prepared_constraints,
                        transition_violations=transition_violations,
                        all_violations=violations,
                        delta=delta,
                        state=state,
                        flags=flags,
                        recent_action_events=scene_joint_action_events,
                        current_violation_count=int(len(violations)),
                        current_transition_violation_count=int(len(transition_violations)),
                        local_patch_round=int(local_patch_round),
                    )
                    entropy_profile.joint_action_selector = joint_selector
                    before_transition_violation_count = int(len(transition_violations))
                    before_violation_count = int(len(violations))
                    before_scene_uncertainty = float(entropy_profile.scene_uncertainty_mean)
                    action_requested = joint_selector != "do_nothing"
                    action_executed = action_requested
                    action_blocked_reasons: List[str] = []
                    if joint_selector in {"validation_boost", "patch", "patch_plus_escalation", "replan"}:
                        entropy_profile.triggered_validation = True
                    if joint_selector in {"patch_plus_escalation", "replan"}:
                        entropy_profile.triggered_patch = True
                        entropy_profile.triggered_patch_escalation = True
                    if joint_selector == "replan":
                        entropy_profile.triggered_replan = True
                    validation_mode = (
                        "escalated"
                        if bool(entropy_profile.triggered_validation)
                        else str(entropy_profile.triggered_validation_mode or "standard")
                    )
                    validation_budget = int(max(1, entropy_profile.triggered_validation_budget))
                    if validation_mode == "escalated":
                        validation_budget = max(
                            validation_budget,
                            int(max(1, 1 + int(self.config.entropy_monitor.validation_escalation_extra_checks))),
                        )
                    extra_runs_needed = int(max(0, validation_budget - transition_validation_runs))
                    for _ in range(extra_runs_needed):
                        if (not transition_ok) or transition_violations:
                            break
                        transition_validation_runs += 1
                        transition_ok, transition_violations = self.transition_validator.validate(
                            state=state_t,
                            action=action,
                            constraints=reasoning.inferred_constraints,
                            forbidden_transitions=reasoning.forbidden_transitions,
                            selected_operator=reasoning.selected_operator,
                            sentence_units=extraction.sentences,
                        )
                    transition_validation_runs = int(max(1, validation_budget))
                    if validation_mode != "escalated":
                        transition_validation_runs = 1
                    self.trace_writer.write_scene_transition(
                        scene.scene_id,
                        {
                            "state_t": asdict(state_t),
                            "reasoning": {
                                "allowed_transitions": list(reasoning.allowed_transitions),
                                "forbidden_transitions": list(reasoning.forbidden_transitions),
                                "required_state_changes": list(reasoning.required_state_changes),
                                "candidate_operators": [asdict(item) for item in reasoning.candidate_operators],
                                "forbidden_operators": list(reasoning.forbidden_operators),
                                "selected_operator": (
                                    asdict(reasoning.selected_operator)
                                    if reasoning.selected_operator is not None
                                    else {}
                                ),
                                "execution_spec": asdict(reasoning.execution_spec),
                                "propagated_constraints": list(reasoning.propagated_constraints),
                                "conflict_candidates": list(reasoning.conflict_candidates),
                                "inferred_constraints": [
                                    asdict(item) for item in reasoning.inferred_constraints
                                ],
                                "notes": list(reasoning.notes),
                            },
                            "action": asdict(action),
                            "transition_valid": transition_ok,
                            "transition_violations": [asdict(v) for v in transition_violations],
                            "validation_mode": validation_mode,
                            "validation_budget": validation_budget,
                            "transition_validation_runs": transition_validation_runs,
                        },
                    )
                    violations = pre_violations + transition_violations + post_violations + pending_preservation_violations
                    violations = self._project_constraint_layer_violations(
                        violations=violations,
                        policy=policies.constraints,
                    )
                    entropy_triggered_validation = entropy_triggered_validation or bool(
                        entropy_profile.triggered_validation
                    )
                    entropy_triggered_patch_escalation = entropy_triggered_patch_escalation or bool(
                        entropy_profile.triggered_patch_escalation
                    )
                    entropy_triggered_replan = entropy_triggered_replan or bool(
                        entropy_profile.triggered_replan
                    )
                    pending_preservation_violations = []
                    if generation_mode == "plain_generate_only":
                        violations = []
                    if joint_selector == "patch":
                        entropy_profile.linked_sentence_ids = list(
                            dict.fromkeys(
                                list(entropy_profile.linked_sentence_ids)
                                + list(entropy_profile.high_risk_sentence_ids)
                            )
                        )
                    if joint_selector in {"patch", "patch_plus_escalation", "replan"} and bool(
                        flags.get("entropy_validation_only", False)
                    ):
                        action_executed = False
                        action_blocked_reasons.append("validation_only_mode")
                    if joint_selector in {"patch", "patch_plus_escalation"} and (
                        flags["generation_layer"] in {"plain_generate_only", "plan_only", "plan_plus_memory"}
                    ):
                        action_executed = False
                        action_blocked_reasons.append("patch_disabled_by_generation_layer")
                    if joint_selector == "replan" and not self.config.generation_controls.enable_replan_hook:
                        action_executed = False
                        action_blocked_reasons.append("replan_hook_disabled")
                    if joint_selector == "patch_plus_escalation" and paragraph_patch_round >= max_paragraph_patch_rounds:
                        action_blocked_reasons.append("paragraph_budget_exhausted")
                    if joint_selector == "replan" and local_patch_round < int(
                        self.config.entropy_monitor.replan_min_patch_rounds
                    ):
                        action_blocked_reasons.append("replan_min_round_not_met")
                    after_transition_violation_count = int(len(transition_violations))
                    after_violation_count = int(len(violations))
                    # lightweight realized uncertainty proxy after current control action
                    after_scene_uncertainty = float(entropy_profile.scene_uncertainty_mean)
                    scene_joint_action_events.append(
                        {
                            "round_idx": int(local_patch_round),
                            "selected_action": str(joint_selector),
                            "action_requested": bool(action_requested),
                            "action_executed": bool(action_executed),
                            "action_blocked_count": int(len(action_blocked_reasons)),
                            "action_blocked_reasons": list(action_blocked_reasons),
                            "before_transition_violation_count": before_transition_violation_count,
                            "after_transition_violation_count": after_transition_violation_count,
                            "before_violation_count": before_violation_count,
                            "after_violation_count": after_violation_count,
                            "violation_delta": int(after_violation_count - before_violation_count),
                            "before_scene_uncertainty": before_scene_uncertainty,
                            "after_scene_uncertainty": after_scene_uncertainty,
                            "uncertainty_delta_after_action": float(after_scene_uncertainty - before_scene_uncertainty),
                            "improved_transition_violations": bool(
                                after_transition_violation_count < before_transition_violation_count
                            ),
                            "improved_violations": bool(after_violation_count < before_violation_count),
                            "improved_uncertainty": bool(after_scene_uncertainty < before_scene_uncertainty),
                            "uncertainty_control_score": float(entropy_profile.uncertainty_control_score),
                            "symbolic_pressure_score": float(entropy_profile.symbolic_pressure_score),
                            "memory_volatility_score": float(entropy_profile.memory_volatility_score),
                            "uncertainty_contribution": float(entropy_profile.uncertainty_contribution),
                            "symbolic_contribution": float(entropy_profile.symbolic_contribution),
                            "memory_contribution": float(entropy_profile.memory_contribution),
                            "joint_risk_score": float(entropy_profile.joint_risk_score),
                            "joint_local_failure_signal": float(entropy_profile.joint_local_failure_signal),
                            "joint_persistent_risk_steps": int(entropy_profile.joint_persistent_risk_steps),
                            "joint_patch_failure_proxy_score": float(entropy_profile.joint_patch_failure_proxy_score),
                            "joint_validation_signal_u": float(entropy_profile.joint_validation_signal_u),
                            "joint_validation_pre_gate_score": float(entropy_profile.joint_validation_pre_gate_score),
                            "joint_patch_pre_gate_score": float(entropy_profile.joint_patch_pre_gate_score),
                            "joint_replan_pre_gate_score": float(entropy_profile.joint_replan_pre_gate_score),
                            "joint_validation_gate_passed": bool(entropy_profile.joint_validation_gate_passed),
                            "joint_validation_threshold_reached": bool(
                                entropy_profile.joint_validation_threshold_reached
                            ),
                            "joint_validation_dual_signal_satisfied": bool(
                                entropy_profile.joint_validation_dual_signal_satisfied
                            ),
                            "joint_validation_symbolic_ok": bool(entropy_profile.joint_validation_symbolic_ok),
                            "joint_validation_uncertainty_ok": bool(entropy_profile.joint_validation_uncertainty_ok),
                            "joint_validation_low_violation_guard_blocked": bool(
                                entropy_profile.joint_validation_low_violation_guard_blocked
                            ),
                            "joint_validation_fail_reasons": list(entropy_profile.joint_validation_fail_reasons),
                            "joint_patch_gate_passed": bool(entropy_profile.joint_patch_gate_passed),
                            "joint_patch_threshold_reached": bool(entropy_profile.joint_patch_threshold_reached),
                            "joint_patch_symbolic_ok": bool(entropy_profile.joint_patch_symbolic_ok),
                            "joint_patch_local_failure_ok": bool(entropy_profile.joint_patch_local_failure_ok),
                            "joint_patch_fail_reasons": list(entropy_profile.joint_patch_fail_reasons),
                            "joint_replan_gate_passed": bool(entropy_profile.joint_replan_gate_passed),
                            "joint_replan_threshold_reached": bool(entropy_profile.joint_replan_threshold_reached),
                            "joint_replan_persistence_ok": bool(entropy_profile.joint_replan_persistence_ok),
                            "joint_replan_patch_failure_ok": bool(entropy_profile.joint_replan_patch_failure_ok),
                            "joint_replan_requires_patch_failure_blocked": bool(
                                entropy_profile.joint_replan_requires_patch_failure_blocked
                            ),
                            "joint_replan_fail_reasons": list(entropy_profile.joint_replan_fail_reasons),
                            "joint_weight_template_used": str(entropy_profile.joint_weight_template_used),
                            "patch_target_joint_alignment_score": _safe_float(
                                getattr(last_executed_patch_plan, "patch_target_joint_alignment_score", 0.0)
                                if last_executed_patch_plan is not None
                                else 0.0
                            ),
                        }
                    )
                    report = (
                        post_report
                        if (not pre_violations and not transition_violations and not violations)
                        else self._build_report_from_violations(violations, scene.scene_id)
                    )
                    if not violations and post_report.is_consistent:
                        report = post_report
                    report.neural_findings = list(post_report.neural_findings or [])
                    report.canonical_entity_table = dict(post_report.canonical_entity_table or {})
                    report.dual_consistency_decision = str(post_report.dual_consistency_decision or "symbolic_only")
                    report.dual_consistency_summary = dict(post_report.dual_consistency_summary or {})
                    self.trace_writer.write_scene_violations(scene.scene_id, violations, report)
                    if metrics is not None:
                        metrics.record_validation(scene.scene_id, violations)

                    final_report = report
                    final_delta = delta
                    final_action = action
                    last_violations = violations
                    if report.is_consistent:
                        accepted = True
                        if metrics is not None and last_patch_scope:
                            metrics.record_patch_success(scene.scene_id, last_patch_scope)
                        break

                    fatal_violations = bool(any(v.fatal for v in violations))
                    if (
                        report.needs_replan
                        and self.config.generation_controls.enable_replan_hook
                        and (not prefer_sentence_repair_first)
                        and ((not regen_only_on_fatal) or fatal_violations)
                    ):
                        plan_deviation_detected = True
                        failed_reason = "needs_replan"
                        if metrics is not None:
                            metrics.record_needs_replan(scene.scene_id)
                        state.dynamic_memory.revision_records.append(
                            {
                                "scene_id": scene.scene_id,
                                "round": local_patch_round,
                                "reason": report.messages[:3],
                                "stage": "plan_deviation_detected",
                            }
                        )
                        break

                    if local_patch_round >= max_patch_rounds_per_scene:
                        failed_reason = "patch_budget_exhausted"
                        scene_patch_exhausted_targeted = True
                        break

                    if flags["generation_layer"] in {"plain_generate_only", "plan_only", "plan_plus_memory"}:
                        failed_reason = "patch_disabled_by_generation_layer"
                        break

                    scope, force_replan = self._select_adaptive_repair_scope(
                        violations=violations,
                        local_patch_round=local_patch_round,
                        sentence_patch_round=sentence_patch_round,
                        paragraph_patch_round=paragraph_patch_round,
                        max_sentence_patch_rounds=max_sentence_patch_rounds,
                        max_paragraph_patch_rounds=max_paragraph_patch_rounds,
                    )
                    if (
                        scope == "sentence"
                        and (
                            (entropy_profile.triggered_patch_escalation and bool(entropy_profile.linked_constraint_ids))
                            or entropy_profile.joint_action_selector == "patch_plus_escalation"
                        )
                        and paragraph_patch_round < max_paragraph_patch_rounds
                    ):
                        if not prefer_sentence_repair_first:
                            scope = "paragraph"
                    if (
                        (
                            (entropy_profile.triggered_replan and bool(entropy_profile.linked_constraint_ids))
                            or entropy_profile.joint_action_selector == "replan"
                        )
                        and local_patch_round >= int(self.config.entropy_monitor.replan_min_patch_rounds)
                        and self.config.generation_controls.enable_replan_hook
                    ):
                        force_replan = True
                    if prefer_sentence_repair_first:
                        force_replan = bool(
                            force_replan
                            and (fatal_violations or scene_patch_exhausted_targeted)
                        )
                    if regen_only_on_fatal and (not fatal_violations):
                        force_replan = False
                    if force_replan and self.config.generation_controls.enable_replan_hook:
                        failed_reason = "adaptive_replan_triggered"
                        plan_deviation_detected = True
                        if metrics is not None:
                            metrics.record_needs_replan(scene.scene_id)
                        state.dynamic_memory.revision_records.append(
                            {
                                "scene_id": scene.scene_id,
                                "round": local_patch_round,
                                "stage": "adaptive_replan_triggered",
                                "reason": [v.message for v in violations[:3]],
                            }
                        )
                        break
                    if not scope:
                        failed_reason = "patch_scope_exhausted"
                        scene_patch_exhausted_targeted = True
                        break
                    if scope == "sentence":
                        sentence_patch_round += 1
                    elif scope == "paragraph":
                        paragraph_patch_round += 1
                    scene_patch_attempted_targeted = True

                    entropy_risk_prior, entropy_linked_sentence_ids = self._entropy_patch_prior_inputs(
                        entropy_profile=entropy_profile,
                        flags=flags,
                    )
                    violation_context = self._build_violation_context_payload(
                        violations=violations,
                        prepared_constraints=prepared_constraints,
                        entropy_profile=entropy_profile,
                        delta=delta,
                        recent_action_events=scene_joint_action_events,
                    )
                    patch_plan = self.patch_planner.build_patch_plan(
                        scene_plan=scene,
                        extraction=extraction,
                        report=report,
                        violations=violations,
                        static_memory=state.static_memory,
                        dynamic_memory=state.dynamic_memory,
                        future_scenes=(
                            scenes[scene_cursor + 1 : scene_cursor + 3]
                            if flags["global_aware_enabled"]
                            else []
                        ),
                        weighted_future_constraints=prepared_constraints.get("weighted_constraints_tiered", []),
                        round_idx=local_patch_round,
                        entropy_risk_prior=entropy_risk_prior,
                        entropy_linked_sentence_ids=entropy_linked_sentence_ids,
                        violation_context=violation_context,
                    )
                    if not flags["lookahead_enabled"]:
                        patch_plan.trajectory_length = 1
                    if not flags["future_penalty_enabled"]:
                        patch_plan.future_conflict_penalty = 0.0
                        patch_plan.global_objective_breakdown["future_conflict_penalty"] = 0.0
                    last_patch_plan = patch_plan
                    setattr(patch_plan, "patch_execution_status", "planned")
                    setattr(patch_plan, "patch_execution_applied", False)
                    setattr(patch_plan, "patch_execution_round", int(local_patch_round + 1))
                    setattr(patch_plan, "patch_execution_scope", str(scope))
                    setattr(
                        patch_plan,
                        "patch_execution_id",
                        f"{scene.scene_id}:round{int(local_patch_round + 1)}:{str(scope)}:{patch_plan.plan_id}",
                    )
                    report.patch_plan = patch_plan
                    self.trace_writer.write_scene_patch_plan(scene.scene_id, patch_plan)
                    prepared_constraints["deferred_constraints"] = list(patch_plan.deferred_constraints)
                    prepared_constraints["critical_constraints_preserved"] = list(
                        patch_plan.critical_constraints_preserved
                    )

                    oscillation = oscillation_detector.observe(
                        scene.scene_id,
                        patch_plan.target_sentence_ids,
                        violations,
                    )
                    if oscillation["detected"]:
                        scene_oscillation_detected = True
                        failed_reason = "repair_oscillation"
                        if metrics is not None:
                            metrics.record_oscillation(scene.scene_id)
                        state.dynamic_memory.revision_records.append(
                            {
                                "scene_id": scene.scene_id,
                                "round": local_patch_round,
                                "stage": "oscillation_detected",
                                "oscillation_signature": oscillation["signature"],
                                "repeats": oscillation["repeats"],
                            }
                        )
                        if scope == "sentence" and paragraph_patch_round < max_paragraph_patch_rounds:
                            sentence_patch_round = max_sentence_patch_rounds
                            scope = "paragraph"
                        elif (
                            self.config.generation_controls.enable_replan_hook
                            and (not prefer_sentence_repair_first)
                            and ((not regen_only_on_fatal) or fatal_violations)
                        ):
                            setattr(patch_plan, "patch_execution_status", "skipped")
                            setattr(patch_plan, "patch_execution_applied", False)
                            setattr(patch_plan, "patch_execution_skipped_reason", "repair_oscillation")
                            setattr(patch_plan, "patch_effectiveness_label", "not_applied")
                            setattr(patch_plan, "patch_no_gain_reason", "repair_oscillation")
                            scene_patch_execution_records.append(
                                {
                                    "patch_execution_id": str(getattr(patch_plan, "patch_execution_id", "")),
                                    "patch_plan_id": str(getattr(patch_plan, "plan_id", "")),
                                    "round": int(getattr(patch_plan, "patch_execution_round", 0) or 0),
                                    "scope": str(getattr(patch_plan, "patch_execution_scope", "")),
                                    "status": "skipped",
                                    "applied": False,
                                    "skipped_reason": "repair_oscillation",
                                }
                            )
                            plan_deviation_detected = True
                            break

                    if (
                        patch_plan.needs_replan
                        and self.config.generation_controls.enable_replan_hook
                        and (not prefer_sentence_repair_first)
                        and ((not regen_only_on_fatal) or fatal_violations)
                    ):
                        setattr(patch_plan, "patch_execution_status", "skipped")
                        setattr(patch_plan, "patch_execution_applied", False)
                        setattr(patch_plan, "patch_execution_skipped_reason", "patch_plan_needs_replan")
                        setattr(patch_plan, "patch_effectiveness_label", "not_applied")
                        setattr(patch_plan, "patch_no_gain_reason", "patch_plan_needs_replan")
                        scene_patch_execution_records.append(
                            {
                                "patch_execution_id": str(getattr(patch_plan, "patch_execution_id", "")),
                                "patch_plan_id": str(getattr(patch_plan, "plan_id", "")),
                                "round": int(getattr(patch_plan, "patch_execution_round", 0) or 0),
                                "scope": str(getattr(patch_plan, "patch_execution_scope", "")),
                                "status": "skipped",
                                "applied": False,
                                "skipped_reason": "patch_plan_needs_replan",
                            }
                        )
                        report.needs_replan = True
                        plan_deviation_detected = True
                        failed_reason = "patch_plan_needs_replan"
                        if metrics is not None:
                            metrics.record_needs_replan(scene.scene_id)
                        break

                    local_patch_round += 1
                    patch_before_transition_violations = int(len(transition_violations))
                    patch_before_total_violations = int(len(violations))
                    patch_before_constraint_violations = int(max(0, len(violations) - len(transition_violations)))
                    patch_before_uncertainty = float(entropy_profile.scene_uncertainty_mean)
                    rewrite_control_context = self._build_generation_control_context(
                        stage="local_rewrite",
                        scene_plan=scene,
                        static_memory=state.static_memory,
                        dynamic_memory=state.dynamic_memory,
                        prepared_constraints=prepared_constraints,
                        entropy_profile=entropy_profile,
                        violations=violations,
                        flags=flags,
                    )
                    rewrite_control_context["allow_scene_rewrite_fallback"] = bool(
                        scene_patch_exhausted_targeted and (fatal_violations or (not regen_only_on_fatal))
                    )
                    last_rewrite_control_context = dict(rewrite_control_context)
                    repaired = self.local_repair.repair_with_patch_plan(
                        scene_plan=scene,
                        scene_text=scene_text,
                        extraction=extraction,
                        patch_plan=patch_plan,
                        violations=violations,
                        static_memory=state.static_memory,
                        dynamic_memory=state.dynamic_memory,
                        attempt=local_patch_round,
                        scope=scope,
                        generation_control_context=rewrite_control_context,
                    )
                    scene_text = repaired.text
                    self.trace_writer.write_scene_patch(scene.scene_id, local_patch_round, repaired.metadata)
                    self.trace_writer.write_scene_repair(
                        scene.scene_id,
                        local_patch_round,
                        scene_text,
                    )
                    if metrics is not None:
                        metrics.record_patch_round(scene.scene_id, scope, repaired.metadata)
                    had_revision = True
                    first_pass_scope = str(repaired.metadata.get("repair_scope", scope) or scope)
                    last_patch_scope = first_pass_scope
                    last_unchanged_ratio = float(repaired.metadata.get("unchanged_ratio", 0.0) or 0.0)
                    post_patch_extraction = self.scene_extractor.extract_scene(
                        scene_plan=scene,
                        scene_text=scene_text,
                        state=state,
                    )
                    post_patch_delta = post_patch_extraction.to_memory_delta()
                    post_patch_candidate_memory = self.dynamic_manager.preview_update(
                        state.dynamic_memory,
                        post_patch_delta,
                    )
                    _, post_patch_constraint_violations = self.constraint_checker.check_scene(
                        static_memory=state.static_memory,
                        candidate_memory=post_patch_candidate_memory,
                        scene_plan=scene,
                        delta=post_patch_delta,
                        scene_text=scene_text,
                        scene_extraction=post_patch_extraction,
                    )
                    post_patch_transition_ok, post_patch_transition_violations = self.transition_validator.validate(
                        state=state_t,
                        action=action,
                        constraints=reasoning.inferred_constraints,
                        forbidden_transitions=reasoning.forbidden_transitions,
                        selected_operator=reasoning.selected_operator,
                        sentence_units=post_patch_extraction.sentences,
                    )
                    _ = post_patch_transition_ok
                    post_patch_total_violations = int(
                        len(post_patch_constraint_violations) + len(post_patch_transition_violations)
                    )
                    post_patch_scene_entropy = self._compute_entropy_profile(
                        scene_plan=scene,
                        extraction=post_patch_extraction,
                        prepared_constraints=prepared_constraints,
                        previous_entropy_mean=prev_scene_entropy_mean,
                        flags=flags,
                        generation_metadata={"uncertainty_source": "none", "uncertainty_available": False},
                        generation_tokens=[],
                        generation_token_logprobs=[],
                        generation_top_logprobs=[],
                        round_uncertainty_history=None,
                    ).scene_uncertainty_mean
                    self._populate_patch_effectiveness(
                        patch_plan=patch_plan,
                        repaired_metadata=dict(repaired.metadata or {}),
                        before_transition_violations=patch_before_transition_violations,
                        after_transition_violations=int(len(post_patch_transition_violations)),
                        before_total_violations=patch_before_total_violations,
                        after_total_violations=post_patch_total_violations,
                        before_constraint_violations=patch_before_constraint_violations,
                        after_constraint_violations=int(len(post_patch_constraint_violations)),
                        before_uncertainty=patch_before_uncertainty,
                        after_uncertainty=float(post_patch_scene_entropy),
                        rewrite_metadata=dict(repaired.metadata or {}),
                    )
                    first_pass_effectiveness_label = str(getattr(patch_plan, "patch_effectiveness_label", "unknown"))
                    first_pass_post_check = self._state_realization_post_check(
                        patch_plan=patch_plan,
                        rewrite_metadata=dict(repaired.metadata or {}),
                        rewritten_scene_text=scene_text,
                        post_patch_transition_violations=list(post_patch_transition_violations),
                        before_transition_violations=patch_before_transition_violations,
                        after_transition_violations=int(len(post_patch_transition_violations)),
                    )
                    retry_attempted = False
                    retry_effective = False
                    retry_scope = str(scope)
                    retry_reason = ""
                    retry_post_check: Dict[str, object] = {}
                    retry_slot_priority_order: List[str] = []
                    retry_slot_priority_unresolved: Dict[str, List[Dict[str, object]]] = {}
                    slot_priority_fix_progress: Dict[str, int] = {
                        "forbidden_removed_count": 0,
                        "operator_post_state_realized_count": 0,
                        "required_state_realized_count": 0,
                    }
                    slot_priority_preserve_result = True
                    priority_step_where_failure_remains = ""
                    slot_type_rebroken_after_retry = ""
                    step_aware_preservation_guard_enabled = False
                    protected_items_snapshot: Dict[str, List[Dict[str, object]]] = {}
                    protected_items_broken_after_retry: List[Dict[str, object]] = []
                    protected_items_preserved_after_retry = True
                    forbidden_reintroduced_after_step = False
                    operator_post_state_weakened_after_step = False
                    required_state_regressed_after_step = False
                    effective_rewrite_metadata = dict(repaired.metadata or {})
                    if bool(first_pass_post_check.get("state_realization_post_check_retry_eligible", False)):
                        retry_attempted = True
                        retry_reason = str(first_pass_post_check.get("state_realization_post_check_retry_reason", ""))
                        retry_context = {
                            "retry_reason": retry_reason or "state_realization_post_check_failed",
                            "failed_checks": list(first_pass_post_check.get("state_realization_post_check_failed_checks", [])),
                            "missing_required_states": list(first_pass_post_check.get("missing_required_states", [])),
                            "remaining_forbidden_states": list(first_pass_post_check.get("remaining_forbidden_states", [])),
                            "first_pass_required_state_checklist": list(
                                first_pass_post_check.get("required_state_checklist", [])
                            ),
                            "first_pass_forbidden_state_checklist": list(
                                first_pass_post_check.get("forbidden_state_checklist", [])
                            ),
                            "first_pass_operator_post_state_checklist": list(
                                first_pass_post_check.get("operator_post_state_checklist", [])
                            ),
                            "unresolved_required_states": list(
                                first_pass_post_check.get("unresolved_required_states", [])
                            ),
                            "unresolved_forbidden_states": list(
                                first_pass_post_check.get("unresolved_forbidden_states", [])
                            ),
                            "unresolved_operator_post_states": list(
                                first_pass_post_check.get("unresolved_operator_post_states", [])
                            ),
                            "still_unresolved_state_items": list(
                                first_pass_post_check.get("still_unresolved_state_items", [])
                            ),
                            "state_realization_match_type": str(
                                first_pass_post_check.get("state_realization_match_type", "no_match")
                            ),
                            "forbidden_state_removal_match_type": str(
                                first_pass_post_check.get("forbidden_state_removal_match_type", "no_match")
                            ),
                            "operator_post_state_match_type": str(
                                first_pass_post_check.get("operator_post_state_match_type", "no_match")
                            ),
                            "canonical_required_states": list(
                                first_pass_post_check.get("canonical_required_states", [])
                            ),
                            "canonical_forbidden_states": list(
                                first_pass_post_check.get("canonical_forbidden_states", [])
                            ),
                            "canonical_operator_post_states": list(
                                first_pass_post_check.get("canonical_operator_post_states", [])
                            ),
                            "transition_grounded_cues": list(
                                rewrite_control_context.get("transition_grounded_cues", [])
                                if isinstance(rewrite_control_context, dict)
                                else []
                            ),
                        }
                        retry_conflict_type = str(
                            first_pass_post_check.get(
                                "conflict_type",
                                getattr(patch_plan, "rewrite_conflict_type", "unknown"),
                            )
                        )
                        retry_experience_items = self.experience_bank.retrieve_top_k(
                            model_id=resolved_model_name,
                            task_type=sample.task_type,
                            generation_stage="retry_rewrite",
                            conflict_type=retry_conflict_type,
                            top_k=int(self.config.experience_bank.max_retrieved_retry_items),
                            exclude_sample_id=sample.prompt_id,
                        )
                        retry_experience_cautions = self.experience_bank.format_prompt_cautions(
                            retry_experience_items,
                            max_items=int(self.config.experience_bank.max_retrieved_retry_items),
                        )
                        if retry_experience_cautions:
                            retry_context["model_experience_cautions"] = list(retry_experience_cautions)
                        if retry_experience_items:
                            _record_retrieved_experience("retry_rewrite", scene.scene_id, retry_experience_items)
                            scene_retrieved_experience_items = list(scene_retrieved_experience_items) + list(
                                retry_experience_items
                            )
                        retry_repaired = self.local_repair.repair_with_patch_plan(
                            scene_plan=scene,
                            scene_text=scene_text,
                            extraction=post_patch_extraction,
                            patch_plan=patch_plan,
                            violations=violations,
                            static_memory=state.static_memory,
                            dynamic_memory=state.dynamic_memory,
                            attempt=local_patch_round,
                            scope=first_pass_scope,
                            retry_context=retry_context,
                            generation_control_context=rewrite_control_context,
                        )
                        retry_scope = str(retry_repaired.metadata.get("repair_scope", scope))
                        scene_text = retry_repaired.text
                        self.trace_writer.write_scene_patch(scene.scene_id, local_patch_round, retry_repaired.metadata)
                        self.trace_writer.write_scene_repair(
                            scene.scene_id,
                            local_patch_round,
                            scene_text,
                        )
                        retry_post_patch_extraction = self.scene_extractor.extract_scene(
                            scene_plan=scene,
                            scene_text=scene_text,
                            state=state,
                        )
                        retry_post_patch_delta = retry_post_patch_extraction.to_memory_delta()
                        retry_post_patch_candidate_memory = self.dynamic_manager.preview_update(
                            state.dynamic_memory,
                            retry_post_patch_delta,
                        )
                        _, retry_post_patch_constraint_violations = self.constraint_checker.check_scene(
                            static_memory=state.static_memory,
                            candidate_memory=retry_post_patch_candidate_memory,
                            scene_plan=scene,
                            delta=retry_post_patch_delta,
                            scene_text=scene_text,
                            scene_extraction=retry_post_patch_extraction,
                        )
                        _, retry_post_patch_transition_violations = self.transition_validator.validate(
                            state=state_t,
                            action=action,
                            constraints=reasoning.inferred_constraints,
                            forbidden_transitions=reasoning.forbidden_transitions,
                            selected_operator=reasoning.selected_operator,
                            sentence_units=retry_post_patch_extraction.sentences,
                        )
                        retry_post_patch_total_violations = int(
                            len(retry_post_patch_constraint_violations) + len(retry_post_patch_transition_violations)
                        )
                        retry_post_patch_scene_entropy = self._compute_entropy_profile(
                            scene_plan=scene,
                            extraction=retry_post_patch_extraction,
                            prepared_constraints=prepared_constraints,
                            previous_entropy_mean=prev_scene_entropy_mean,
                            flags=flags,
                            generation_metadata={"uncertainty_source": "none", "uncertainty_available": False},
                            generation_tokens=[],
                            generation_token_logprobs=[],
                            generation_top_logprobs=[],
                            round_uncertainty_history=None,
                        ).scene_uncertainty_mean
                        self._populate_patch_effectiveness(
                            patch_plan=patch_plan,
                            repaired_metadata=dict(retry_repaired.metadata or {}),
                            before_transition_violations=patch_before_transition_violations,
                            after_transition_violations=int(len(retry_post_patch_transition_violations)),
                            before_total_violations=patch_before_total_violations,
                            after_total_violations=retry_post_patch_total_violations,
                            before_constraint_violations=patch_before_constraint_violations,
                            after_constraint_violations=int(len(retry_post_patch_constraint_violations)),
                            before_uncertainty=patch_before_uncertainty,
                            after_uncertainty=float(retry_post_patch_scene_entropy),
                            rewrite_metadata=dict(retry_repaired.metadata or {}),
                        )
                        retry_post_check = self._state_realization_post_check(
                            patch_plan=patch_plan,
                            rewrite_metadata=dict(retry_repaired.metadata or {}),
                            rewritten_scene_text=scene_text,
                            post_patch_transition_violations=list(retry_post_patch_transition_violations),
                            before_transition_violations=patch_before_transition_violations,
                            after_transition_violations=int(len(retry_post_patch_transition_violations)),
                        )
                        retry_effective = str(getattr(patch_plan, "patch_effectiveness_label", "")) in {"effective", "partial"}
                        last_unchanged_ratio = float(
                            retry_repaired.metadata.get("unchanged_ratio", last_unchanged_ratio) or last_unchanged_ratio
                        )
                        effective_rewrite_metadata = dict(retry_repaired.metadata or {})
                        retry_slot_priority_order = list(
                            (effective_rewrite_metadata.get("rewrite_retry_context", {}) or {}).get(
                                "retry_slot_priority_order",
                                [],
                            )
                        )
                        retry_slot_priority_unresolved = dict(
                            (effective_rewrite_metadata.get("rewrite_retry_context", {}) or {}).get(
                                "retry_slot_priority_unresolved",
                                {},
                            )
                        )
                        step_aware_preservation_guard_enabled = bool(
                            (effective_rewrite_metadata.get("rewrite_retry_context", {}) or {}).get(
                                "step_aware_preservation_guard_enabled",
                                False,
                            )
                        )
                        protected_items_snapshot = dict(
                            (effective_rewrite_metadata.get("rewrite_retry_context", {}) or {}).get(
                                "protected_items_snapshot",
                                {},
                            )
                        )
                    setattr(patch_plan, "patch_retry_attempted", bool(retry_attempted))
                    setattr(patch_plan, "patch_retry_scope", str(retry_scope if retry_attempted else ""))
                    setattr(patch_plan, "patch_retry_reason", str(retry_reason if retry_attempted else ""))
                    setattr(
                        patch_plan,
                        "patch_retry_conflict_type",
                        str(first_pass_post_check.get("conflict_type", getattr(patch_plan, "rewrite_conflict_type", "unknown"))),
                    )
                    setattr(patch_plan, "patch_retry_effective", bool(retry_effective if retry_attempted else False))
                    setattr(
                        patch_plan,
                        "patch_retry_realizes_required_state_change",
                        bool(retry_post_check.get("rewrite_realizes_required_state_change", False) if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "patch_retry_removes_forbidden_state",
                        bool(retry_post_check.get("rewrite_removes_forbidden_state", False) if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "patch_retry_restores_transition_coherence_proxy",
                        bool(retry_post_check.get("rewrite_restores_transition_coherence_proxy", False) if retry_attempted else False),
                    )
                    setattr(patch_plan, "patch_first_pass_effective", first_pass_effectiveness_label in {"effective", "partial"})
                    setattr(
                        patch_plan,
                        "patch_still_ineffective_after_retry",
                        bool(retry_attempted and (not retry_effective)),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_realizes_operator_post_state",
                        bool(
                            retry_post_check.get("rewrite_realizes_operator_post_state", False)
                            if retry_attempted
                            else first_pass_post_check.get("rewrite_realizes_operator_post_state", False)
                        ),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_realizes_required_state_change",
                        bool(
                            retry_post_check.get("rewrite_realizes_required_state_change", False)
                            if retry_attempted
                            else first_pass_post_check.get("rewrite_realizes_required_state_change", False)
                        ),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_removes_forbidden_state",
                        bool(
                            retry_post_check.get("rewrite_removes_forbidden_state", False)
                            if retry_attempted
                            else first_pass_post_check.get("rewrite_removes_forbidden_state", False)
                        ),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_restores_transition_coherence_proxy",
                        bool(
                            retry_post_check.get("rewrite_restores_transition_coherence_proxy", False)
                            if retry_attempted
                            else first_pass_post_check.get("rewrite_restores_transition_coherence_proxy", False)
                        ),
                    )
                    final_post_check = retry_post_check if retry_attempted else first_pass_post_check
                    first_required_checklist = list(first_pass_post_check.get("required_state_checklist", []))
                    first_forbidden_checklist = list(first_pass_post_check.get("forbidden_state_checklist", []))
                    first_operator_checklist = list(first_pass_post_check.get("operator_post_state_checklist", []))
                    final_required_checklist = list(final_post_check.get("required_state_checklist", []))
                    final_forbidden_checklist = list(final_post_check.get("forbidden_state_checklist", []))
                    final_operator_checklist = list(final_post_check.get("operator_post_state_checklist", []))

                    def _checklist_completion(items: Sequence[Dict[str, object]], *, done_status: str) -> float:
                        n = len(items)
                        if n == 0:
                            return 1.0
                        done = sum(1 for item in items if str(item.get("status", "")) == done_status)
                        return float(done / max(1, n))

                    def _index_by_state(items: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
                        out: Dict[str, Dict[str, object]] = {}
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            canonical = str(item.get("canonical_state", "")).strip()
                            if canonical:
                                out[canonical] = dict(item)
                        return out

                    first_required_idx = _index_by_state(first_required_checklist)
                    first_forbidden_idx = _index_by_state(first_forbidden_checklist)
                    first_operator_idx = _index_by_state(first_operator_checklist)
                    final_required_idx = _index_by_state(final_required_checklist)
                    final_forbidden_idx = _index_by_state(final_forbidden_checklist)
                    final_operator_idx = _index_by_state(final_operator_checklist)

                    checklist_items_fixed_by_retry: List[Dict[str, object]] = []
                    checklist_items_still_unresolved: List[Dict[str, object]] = []
                    retry_preserved_satisfied_items = True

                    for canonical, first_item in first_required_idx.items():
                        first_status = str(first_item.get("status", "unsatisfied"))
                        final_status = str(final_required_idx.get(canonical, {}).get("status", "unsatisfied"))
                        if first_status in {"unsatisfied", "uncertain"} and final_status == "satisfied":
                            checklist_items_fixed_by_retry.append(
                                {"state_type": "required", "canonical_state": canonical}
                            )
                            slot_priority_fix_progress["required_state_realized_count"] = int(
                                slot_priority_fix_progress.get("required_state_realized_count", 0)
                            ) + 1
                        if final_status in {"unsatisfied", "uncertain"}:
                            checklist_items_still_unresolved.append(
                                {"state_type": "required", "canonical_state": canonical}
                            )
                        if first_status == "satisfied" and final_status != "satisfied":
                            retry_preserved_satisfied_items = False
                            slot_type_rebroken_after_retry = slot_type_rebroken_after_retry or "required"

                    for canonical, first_item in first_forbidden_idx.items():
                        first_status = str(first_item.get("status", "still_present"))
                        final_status = str(final_forbidden_idx.get(canonical, {}).get("status", "still_present"))
                        if first_status in {"still_present", "uncertain"} and final_status == "removed":
                            checklist_items_fixed_by_retry.append(
                                {"state_type": "forbidden", "canonical_state": canonical}
                            )
                            slot_priority_fix_progress["forbidden_removed_count"] = int(
                                slot_priority_fix_progress.get("forbidden_removed_count", 0)
                            ) + 1
                        if final_status in {"still_present", "uncertain"}:
                            checklist_items_still_unresolved.append(
                                {"state_type": "forbidden", "canonical_state": canonical}
                            )
                        if first_status == "removed" and final_status != "removed":
                            retry_preserved_satisfied_items = False
                            slot_type_rebroken_after_retry = slot_type_rebroken_after_retry or "forbidden"

                    for canonical, first_item in first_operator_idx.items():
                        first_status = str(first_item.get("status", "unrealized"))
                        final_status = str(final_operator_idx.get(canonical, {}).get("status", "unrealized"))
                        if first_status in {"unrealized", "uncertain"} and final_status == "realized":
                            checklist_items_fixed_by_retry.append(
                                {"state_type": "operator_post_state", "canonical_state": canonical}
                            )
                            slot_priority_fix_progress["operator_post_state_realized_count"] = int(
                                slot_priority_fix_progress.get("operator_post_state_realized_count", 0)
                            ) + 1
                        if final_status in {"unrealized", "uncertain"}:
                            checklist_items_still_unresolved.append(
                                {"state_type": "operator_post_state", "canonical_state": canonical}
                            )
                        if first_status == "realized" and final_status != "realized":
                            retry_preserved_satisfied_items = False
                            slot_type_rebroken_after_retry = slot_type_rebroken_after_retry or "operator_post_state"

                    slot_priority_preserve_result = bool(retry_preserved_satisfied_items)
                    unresolved_types_after = {str(item.get("state_type", "")) for item in checklist_items_still_unresolved}
                    forbidden_reintroduced_after_step = bool("forbidden" in unresolved_types_after)
                    operator_post_state_weakened_after_step = bool("operator_post_state" in unresolved_types_after)
                    required_state_regressed_after_step = bool("required" in unresolved_types_after)
                    if step_aware_preservation_guard_enabled:
                        if forbidden_reintroduced_after_step:
                            protected_items_broken_after_retry.append({"state_type": "forbidden"})
                        if operator_post_state_weakened_after_step:
                            protected_items_broken_after_retry.append({"state_type": "operator_post_state"})
                        if required_state_regressed_after_step:
                            protected_items_broken_after_retry.append({"state_type": "required"})
                    protected_items_preserved_after_retry = bool(not protected_items_broken_after_retry)
                    for step in list(retry_slot_priority_order or ["forbidden", "operator_post_state", "required"]):
                        if step in unresolved_types_after:
                            priority_step_where_failure_remains = str(step)
                            break

                    first_required_completion_rate = _checklist_completion(
                        first_required_checklist, done_status="satisfied"
                    )
                    first_forbidden_completion_rate = _checklist_completion(
                        first_forbidden_checklist, done_status="removed"
                    )
                    first_operator_completion_rate = _checklist_completion(
                        first_operator_checklist, done_status="realized"
                    )
                    retry_required_completion_rate = _checklist_completion(
                        final_required_checklist, done_status="satisfied"
                    )
                    retry_forbidden_completion_rate = _checklist_completion(
                        final_forbidden_checklist, done_status="removed"
                    )
                    retry_operator_completion_rate = _checklist_completion(
                        final_operator_checklist, done_status="realized"
                    )
                    first_pass_checklist_completion_rate = float(
                        first_pass_post_check.get(
                            "checklist_completion_rate",
                            (first_required_completion_rate + first_forbidden_completion_rate + first_operator_completion_rate) / 3.0,
                        )
                    )
                    retry_checklist_completion_rate = float(
                        final_post_check.get(
                            "checklist_completion_rate",
                            (retry_required_completion_rate + retry_forbidden_completion_rate + retry_operator_completion_rate) / 3.0,
                        )
                    )
                    original_scope = str(effective_rewrite_metadata.get("original_scope", first_pass_scope))
                    expanded_scope = str(effective_rewrite_metadata.get("expanded_scope", first_pass_scope))
                    expanded_target_sentence_ids = list(
                        effective_rewrite_metadata.get("expanded_target_sentence_ids", patch_plan.target_sentence_ids)
                    )
                    expanded_local_window = dict(effective_rewrite_metadata.get("expanded_local_window", {}))
                    scope_expansion_triggered = bool(
                        effective_rewrite_metadata.get("scope_expansion_triggered", False)
                    )
                    unresolved_slot_count_before = int(
                        effective_rewrite_metadata.get(
                            "unresolved_slot_count_before",
                            len(list(first_pass_post_check.get("still_unresolved_state_items", []))),
                        )
                        or 0
                    )
                    unresolved_slot_count_after = int(
                        len(list(final_post_check.get("still_unresolved_state_items", [])))
                    )
                    checklist_items_fixed_by_expansion = list(checklist_items_fixed_by_retry)
                    still_unresolved_after_expansion = list(final_post_check.get("still_unresolved_state_items", []))
                    expansion_preserved_satisfied_items = bool(retry_preserved_satisfied_items)
                    scope_expansion_effective = bool(
                        scope_expansion_triggered
                        and (
                            (unresolved_slot_count_after < unresolved_slot_count_before)
                            or bool(checklist_items_fixed_by_expansion)
                        )
                    )

                    setattr(
                        patch_plan,
                        "state_realization_match_type",
                        str(final_post_check.get("state_realization_match_type", "no_match")),
                    )
                    setattr(
                        patch_plan,
                        "forbidden_state_removal_match_type",
                        str(final_post_check.get("forbidden_state_removal_match_type", "no_match")),
                    )
                    setattr(
                        patch_plan,
                        "operator_post_state_match_type",
                        str(final_post_check.get("operator_post_state_match_type", "no_match")),
                    )
                    setattr(
                        patch_plan,
                        "canonical_required_states",
                        list(final_post_check.get("canonical_required_states", [])),
                    )
                    setattr(
                        patch_plan,
                        "canonical_forbidden_states",
                        list(final_post_check.get("canonical_forbidden_states", [])),
                    )
                    setattr(
                        patch_plan,
                        "canonical_operator_post_states",
                        list(final_post_check.get("canonical_operator_post_states", [])),
                    )
                    setattr(
                        patch_plan,
                        "grounded_alias_matches",
                        list(final_post_check.get("grounded_alias_matches", [])),
                    )
                    setattr(
                        patch_plan,
                        "first_pass_required_state_checklist",
                        list(first_required_checklist),
                    )
                    setattr(
                        patch_plan,
                        "first_pass_forbidden_state_checklist",
                        list(first_forbidden_checklist),
                    )
                    setattr(
                        patch_plan,
                        "first_pass_operator_post_state_checklist",
                        list(first_operator_checklist),
                    )
                    setattr(
                        patch_plan,
                        "retry_required_state_checklist",
                        list(final_required_checklist),
                    )
                    setattr(
                        patch_plan,
                        "retry_forbidden_state_checklist",
                        list(final_forbidden_checklist),
                    )
                    setattr(
                        patch_plan,
                        "retry_operator_post_state_checklist",
                        list(final_operator_checklist),
                    )
                    setattr(
                        patch_plan,
                        "first_pass_checklist_completion_rate",
                        float(first_pass_checklist_completion_rate),
                    )
                    setattr(
                        patch_plan,
                        "retry_checklist_completion_rate",
                        float(retry_checklist_completion_rate),
                    )
                    setattr(
                        patch_plan,
                        "checklist_items_fixed_by_retry",
                        list(checklist_items_fixed_by_retry if retry_attempted else []),
                    )
                    setattr(
                        patch_plan,
                        "checklist_items_still_unresolved",
                        list(
                            checklist_items_still_unresolved
                            if retry_attempted
                            else list(final_post_check.get("still_unresolved_state_items", []))
                        ),
                    )
                    setattr(
                        patch_plan,
                        "retry_preserved_satisfied_items",
                        bool(retry_preserved_satisfied_items if retry_attempted else True),
                    )
                    setattr(
                        patch_plan,
                        "retry_slot_priority_order",
                        list(
                            retry_slot_priority_order
                            if retry_attempted
                            else []
                        ),
                    )
                    setattr(
                        patch_plan,
                        "retry_slot_priority_unresolved",
                        dict(retry_slot_priority_unresolved if retry_attempted else {}),
                    )
                    setattr(
                        patch_plan,
                        "slot_priority_fix_progress",
                        dict(slot_priority_fix_progress if retry_attempted else {}),
                    )
                    setattr(
                        patch_plan,
                        "slot_priority_preserve_result",
                        bool(slot_priority_preserve_result if retry_attempted else True),
                    )
                    setattr(
                        patch_plan,
                        "priority_step_where_failure_remains",
                        str(priority_step_where_failure_remains if retry_attempted else ""),
                    )
                    setattr(
                        patch_plan,
                        "slot_type_rebroken_after_retry",
                        str(slot_type_rebroken_after_retry if retry_attempted else ""),
                    )
                    setattr(
                        patch_plan,
                        "step_aware_preservation_guard_enabled",
                        bool(step_aware_preservation_guard_enabled if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "protected_items_snapshot",
                        dict(protected_items_snapshot if retry_attempted else {}),
                    )
                    setattr(
                        patch_plan,
                        "protected_items_preserved_after_retry",
                        bool(protected_items_preserved_after_retry if retry_attempted else True),
                    )
                    setattr(
                        patch_plan,
                        "protected_items_broken_after_retry",
                        list(protected_items_broken_after_retry if retry_attempted else []),
                    )
                    setattr(
                        patch_plan,
                        "forbidden_reintroduced_after_step",
                        bool(forbidden_reintroduced_after_step if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "operator_post_state_weakened_after_step",
                        bool(operator_post_state_weakened_after_step if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "required_state_regressed_after_step",
                        bool(required_state_regressed_after_step if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "scope_expansion_triggered",
                        bool(scope_expansion_triggered if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "original_scope",
                        str(original_scope if retry_attempted else first_pass_scope),
                    )
                    setattr(
                        patch_plan,
                        "expanded_scope",
                        str(expanded_scope if retry_attempted else first_pass_scope),
                    )
                    setattr(
                        patch_plan,
                        "expanded_target_sentence_ids",
                        list(expanded_target_sentence_ids if retry_attempted else patch_plan.target_sentence_ids),
                    )
                    setattr(
                        patch_plan,
                        "expanded_local_window",
                        dict(expanded_local_window if retry_attempted else {}),
                    )
                    setattr(
                        patch_plan,
                        "unresolved_slot_count_before",
                        int(unresolved_slot_count_before if retry_attempted else 0),
                    )
                    setattr(
                        patch_plan,
                        "unresolved_slot_count_after",
                        int(unresolved_slot_count_after if retry_attempted else 0),
                    )
                    setattr(
                        patch_plan,
                        "checklist_items_fixed_by_expansion",
                        list(checklist_items_fixed_by_expansion if retry_attempted else []),
                    )
                    setattr(
                        patch_plan,
                        "still_unresolved_after_expansion",
                        list(still_unresolved_after_expansion if retry_attempted else []),
                    )
                    setattr(
                        patch_plan,
                        "expansion_preserved_satisfied_items",
                        bool(expansion_preserved_satisfied_items if retry_attempted else True),
                    )
                    setattr(
                        patch_plan,
                        "scope_expansion_effective",
                        bool(scope_expansion_effective if retry_attempted else False),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_memory_binding_mode",
                        str(effective_rewrite_metadata.get("rewrite_memory_binding_mode", scene_memory_binding_mode)),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_generation_control_mode",
                        str(
                            effective_rewrite_metadata.get(
                                "rewrite_generation_control_mode",
                                scene_generation_control_mode,
                            )
                        ),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_binding_decision_reasons",
                        list(effective_rewrite_metadata.get("rewrite_binding_decision_reasons", [])),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_strengthened_memory_blocks",
                        list(effective_rewrite_metadata.get("rewrite_strengthened_memory_blocks", [])),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_strengthened_constraints",
                        list(effective_rewrite_metadata.get("rewrite_strengthened_constraints", [])),
                    )
                    setattr(
                        patch_plan,
                        "rewrite_generation_control_context",
                        dict(effective_rewrite_metadata.get("rewrite_generation_control_context", {})),
                    )
                    setattr(patch_plan, "patch_execution_status", "executed")
                    setattr(patch_plan, "patch_execution_applied", True)
                    setattr(patch_plan, "patch_execution_skipped_reason", "")
                    scene_patch_execution_records.append(
                        {
                            "patch_execution_id": str(getattr(patch_plan, "patch_execution_id", "")),
                            "patch_plan_id": str(getattr(patch_plan, "plan_id", "")),
                            "round": int(getattr(patch_plan, "patch_execution_round", 0) or 0),
                            "scope": str(getattr(patch_plan, "patch_execution_scope", "")),
                            "status": "executed",
                            "applied": True,
                            "patch_effectiveness_label": str(getattr(patch_plan, "patch_effectiveness_label", "unknown")),
                            "patch_no_gain_reason": str(getattr(patch_plan, "patch_no_gain_reason", "")),
                            "rewrite_conflict_type": str(getattr(patch_plan, "rewrite_conflict_type", "unknown")),
                            "rewrite_target_scope": str(getattr(patch_plan, "rewrite_target_scope", "sentence")),
                            "patch_target_hits_violation_context": bool(
                                getattr(patch_plan, "patch_target_hits_violation_context", False)
                            ),
                            "rewrite_hits_required_state_change": bool(
                                getattr(patch_plan, "rewrite_hits_required_state_change", False)
                            ),
                            "rewrite_removes_conflicting_state": bool(
                                getattr(patch_plan, "rewrite_removes_conflicting_state", False)
                            ),
                            "rewrite_realizes_required_state_change": bool(
                                getattr(patch_plan, "rewrite_realizes_required_state_change", False)
                            ),
                            "rewrite_removes_forbidden_state": bool(
                                getattr(patch_plan, "rewrite_removes_forbidden_state", False)
                            ),
                            "rewrite_restores_transition_coherence_proxy": bool(
                                getattr(patch_plan, "rewrite_restores_transition_coherence_proxy", False)
                            ),
                            "rewrite_realizes_operator_post_state": bool(
                                getattr(patch_plan, "rewrite_realizes_operator_post_state", False)
                            ),
                            "state_realization_match_type": str(
                                getattr(patch_plan, "state_realization_match_type", "no_match")
                            ),
                            "forbidden_state_removal_match_type": str(
                                getattr(patch_plan, "forbidden_state_removal_match_type", "no_match")
                            ),
                            "operator_post_state_match_type": str(
                                getattr(patch_plan, "operator_post_state_match_type", "no_match")
                            ),
                            "canonical_required_states": list(
                                getattr(patch_plan, "canonical_required_states", [])
                            ),
                            "canonical_forbidden_states": list(
                                getattr(patch_plan, "canonical_forbidden_states", [])
                            ),
                            "canonical_operator_post_states": list(
                                getattr(patch_plan, "canonical_operator_post_states", [])
                            ),
                            "grounded_alias_matches": list(getattr(patch_plan, "grounded_alias_matches", [])),
                            "first_pass_required_state_checklist": list(
                                getattr(patch_plan, "first_pass_required_state_checklist", [])
                            ),
                            "first_pass_forbidden_state_checklist": list(
                                getattr(patch_plan, "first_pass_forbidden_state_checklist", [])
                            ),
                            "first_pass_operator_post_state_checklist": list(
                                getattr(patch_plan, "first_pass_operator_post_state_checklist", [])
                            ),
                            "retry_required_state_checklist": list(
                                getattr(patch_plan, "retry_required_state_checklist", [])
                            ),
                            "retry_forbidden_state_checklist": list(
                                getattr(patch_plan, "retry_forbidden_state_checklist", [])
                            ),
                            "retry_operator_post_state_checklist": list(
                                getattr(patch_plan, "retry_operator_post_state_checklist", [])
                            ),
                            "first_pass_checklist_completion_rate": float(
                                getattr(patch_plan, "first_pass_checklist_completion_rate", 0.0)
                            ),
                            "retry_checklist_completion_rate": float(
                                getattr(patch_plan, "retry_checklist_completion_rate", 0.0)
                            ),
                            "checklist_items_fixed_by_retry": list(
                                getattr(patch_plan, "checklist_items_fixed_by_retry", [])
                            ),
                            "checklist_items_still_unresolved": list(
                                getattr(patch_plan, "checklist_items_still_unresolved", [])
                            ),
                            "retry_preserved_satisfied_items": bool(
                                getattr(patch_plan, "retry_preserved_satisfied_items", True)
                            ),
                            "retry_slot_priority_order": list(
                                getattr(patch_plan, "retry_slot_priority_order", [])
                            ),
                            "retry_slot_priority_unresolved": dict(
                                getattr(patch_plan, "retry_slot_priority_unresolved", {})
                            ),
                            "slot_priority_fix_progress": dict(
                                getattr(patch_plan, "slot_priority_fix_progress", {})
                            ),
                            "slot_priority_preserve_result": bool(
                                getattr(patch_plan, "slot_priority_preserve_result", True)
                            ),
                            "priority_step_where_failure_remains": str(
                                getattr(patch_plan, "priority_step_where_failure_remains", "")
                            ),
                            "slot_type_rebroken_after_retry": str(
                                getattr(patch_plan, "slot_type_rebroken_after_retry", "")
                            ),
                            "step_aware_preservation_guard_enabled": bool(
                                getattr(patch_plan, "step_aware_preservation_guard_enabled", False)
                            ),
                            "protected_items_snapshot": dict(
                                getattr(patch_plan, "protected_items_snapshot", {})
                            ),
                            "protected_items_preserved_after_retry": bool(
                                getattr(patch_plan, "protected_items_preserved_after_retry", True)
                            ),
                            "protected_items_broken_after_retry": list(
                                getattr(patch_plan, "protected_items_broken_after_retry", [])
                            ),
                            "forbidden_reintroduced_after_step": bool(
                                getattr(patch_plan, "forbidden_reintroduced_after_step", False)
                            ),
                            "operator_post_state_weakened_after_step": bool(
                                getattr(patch_plan, "operator_post_state_weakened_after_step", False)
                            ),
                            "required_state_regressed_after_step": bool(
                                getattr(patch_plan, "required_state_regressed_after_step", False)
                            ),
                            "scope_expansion_triggered": bool(
                                getattr(patch_plan, "scope_expansion_triggered", False)
                            ),
                            "original_scope": str(getattr(patch_plan, "original_scope", "")),
                            "expanded_scope": str(getattr(patch_plan, "expanded_scope", "")),
                            "expanded_target_sentence_ids": list(
                                getattr(patch_plan, "expanded_target_sentence_ids", [])
                            ),
                            "expanded_local_window": dict(getattr(patch_plan, "expanded_local_window", {})),
                            "unresolved_slot_count_before": int(
                                getattr(patch_plan, "unresolved_slot_count_before", 0) or 0
                            ),
                            "unresolved_slot_count_after": int(
                                getattr(patch_plan, "unresolved_slot_count_after", 0) or 0
                            ),
                            "checklist_items_fixed_by_expansion": list(
                                getattr(patch_plan, "checklist_items_fixed_by_expansion", [])
                            ),
                            "still_unresolved_after_expansion": list(
                                getattr(patch_plan, "still_unresolved_after_expansion", [])
                            ),
                            "expansion_preserved_satisfied_items": bool(
                                getattr(patch_plan, "expansion_preserved_satisfied_items", True)
                            ),
                            "scope_expansion_effective": bool(
                                getattr(patch_plan, "scope_expansion_effective", False)
                            ),
                            "rewrite_targets_execution_spec_conflict": bool(
                                getattr(patch_plan, "rewrite_targets_execution_spec_conflict", False)
                            ),
                            "rewrite_targets_required_state_change": bool(
                                getattr(patch_plan, "rewrite_targets_required_state_change", False)
                            ),
                            "rewrite_targets_transition_conflict": bool(
                                getattr(patch_plan, "rewrite_targets_transition_conflict", False)
                            ),
                            "rewrite_targets_operator_post_state_conflict": bool(
                                getattr(patch_plan, "rewrite_targets_operator_post_state_conflict", False)
                            ),
                            "changed_sentence_count": int(
                                effective_rewrite_metadata.get("changed_sentence_count", 0) or 0
                            ),
                            "changed_target_sentence_count": int(
                                effective_rewrite_metadata.get("changed_target_sentence_count", 0) or 0
                            ),
                            "changed_non_target_sentence_count": int(
                                effective_rewrite_metadata.get("changed_non_target_sentence_count", 0) or 0
                            ),
                            "changed_sentence_ids": list(
                                effective_rewrite_metadata.get("changed_sentence_ids", [])
                            ),
                            "changed_non_target_sentence_ids": list(
                                effective_rewrite_metadata.get("changed_non_target_sentence_ids", [])
                            ),
                            "target_sentence_touched_ratio": float(
                                effective_rewrite_metadata.get("target_sentence_touched_ratio", 0.0) or 0.0
                            ),
                            "violation_context_touched": bool(
                                effective_rewrite_metadata.get("violation_context_touched", False)
                            ),
                            "rewrite_memory_binding_mode": str(
                                getattr(patch_plan, "rewrite_memory_binding_mode", "normal_binding")
                            ),
                            "rewrite_generation_control_mode": str(
                                getattr(patch_plan, "rewrite_generation_control_mode", "normal_generation")
                            ),
                            "rewrite_binding_decision_reasons": list(
                                getattr(patch_plan, "rewrite_binding_decision_reasons", [])
                            ),
                            "rewrite_strengthened_memory_blocks": list(
                                getattr(patch_plan, "rewrite_strengthened_memory_blocks", [])
                            ),
                            "rewrite_strengthened_constraints": list(
                                getattr(patch_plan, "rewrite_strengthened_constraints", [])
                            ),
                            "rewrite_generation_control_context": dict(
                                effective_rewrite_metadata.get("rewrite_generation_control_context", {})
                            ),
                            "patch_before_transition_violation_count": int(
                                getattr(patch_plan, "patch_before_transition_violation_count", 0) or 0
                            ),
                            "patch_after_transition_violation_count": int(
                                getattr(patch_plan, "patch_after_transition_violation_count", 0) or 0
                            ),
                            "patch_before_constraint_violation_count": int(
                                getattr(patch_plan, "patch_before_constraint_violation_count", 0) or 0
                            ),
                            "patch_after_constraint_violation_count": int(
                                getattr(patch_plan, "patch_after_constraint_violation_count", 0) or 0
                            ),
                            "patch_before_violation_count": int(
                                getattr(patch_plan, "patch_before_violation_count", 0) or 0
                            ),
                            "patch_after_violation_count": int(
                                getattr(patch_plan, "patch_after_violation_count", 0) or 0
                            ),
                            "patch_before_uncertainty": float(
                                getattr(patch_plan, "patch_before_uncertainty", 0.0) or 0.0
                            ),
                            "patch_after_uncertainty": float(
                                getattr(patch_plan, "patch_after_uncertainty", 0.0) or 0.0
                            ),
                            "patch_before_symbolic_state_proxy": float(
                                getattr(patch_plan, "patch_before_symbolic_state_proxy", 0.0) or 0.0
                            ),
                            "patch_after_symbolic_state_proxy": float(
                                getattr(patch_plan, "patch_after_symbolic_state_proxy", 0.0) or 0.0
                            ),
                            "first_pass_effective": bool(first_pass_effectiveness_label in {"effective", "partial"}),
                            "retry_effective": bool(retry_effective if retry_attempted else False),
                            "still_ineffective_after_retry": bool(retry_attempted and (not retry_effective)),
                            "patch_retry_attempted": bool(retry_attempted),
                            "patch_retry_scope": str(retry_scope if retry_attempted else ""),
                            "patch_retry_reason": str(retry_reason if retry_attempted else ""),
                            "patch_retry_conflict_type": str(
                                first_pass_post_check.get(
                                    "conflict_type",
                                    getattr(patch_plan, "rewrite_conflict_type", "unknown"),
                                )
                            ),
                            "patch_retry_effective": bool(retry_effective if retry_attempted else False),
                            "patch_retry_realizes_required_state_change": bool(
                                retry_post_check.get("rewrite_realizes_required_state_change", False)
                                if retry_attempted
                                else False
                            ),
                            "patch_retry_removes_forbidden_state": bool(
                                retry_post_check.get("rewrite_removes_forbidden_state", False)
                                if retry_attempted
                                else False
                            ),
                            "patch_retry_restores_transition_coherence_proxy": bool(
                                retry_post_check.get("rewrite_restores_transition_coherence_proxy", False)
                                if retry_attempted
                                else False
                            ),
                            "first_pass_failed_checks": list(
                                first_pass_post_check.get("state_realization_post_check_failed_checks", [])
                            ),
                            "retry_failed_checks": list(
                                retry_post_check.get("state_realization_post_check_failed_checks", [])
                                if retry_attempted
                                else []
                            ),
                        }
                    )
                    last_executed_patch_plan = patch_plan
                    state.dynamic_memory.revision_records.append(
                        {
                            "scene_id": scene.scene_id,
                            "round": local_patch_round,
                            "reason": [v.message for v in violations[:3]],
                            "stage": f"{scope}_patch",
                            "patch_plan_id": patch_plan.plan_id,
                            "patch_score": patch_plan.candidate_score,
                            "patch_expected_violation_reduction": patch_plan.expected_violation_reduction,
                            "unchanged_ratio": repaired.metadata.get("unchanged_ratio"),
                            "target_sentence_ids": repaired.metadata.get("target_sentence_ids", []),
                            "changed_sentence_count": repaired.metadata.get("changed_sentence_count", 0),
                            "changed_non_target_sentence_count": repaired.metadata.get(
                                "changed_non_target_sentence_count", 0
                            ),
                            "changed_non_target_sentence_ids": repaired.metadata.get(
                                "changed_non_target_sentence_ids", []
                            ),
                            "violation_context_touched": repaired.metadata.get("violation_context_touched", False),
                        }
                    )

                    if self.config.generation_controls.repair_preserve_unchanged and flags["preservation_check"]:
                        pending_preservation_violations = self._build_preservation_violations(
                            scene=scene,
                            metadata=repaired.metadata,
                        )
                        scene_preservation_failures += len(pending_preservation_violations)
                        if pending_preservation_violations and scope == "sentence":
                            sentence_patch_round = max_sentence_patch_rounds

                if accepted or plan_deviation_detected:
                    break

            if final_report is None or final_delta is None or final_action is None:
                raise RuntimeError(f"Incremental loop failed for scene {scene.scene_id}")
            if last_executed_patch_plan is None and last_patch_plan is not None:
                if str(getattr(last_patch_plan, "patch_execution_status", "planned")) == "planned":
                    setattr(last_patch_plan, "patch_execution_status", "skipped")
                    setattr(last_patch_plan, "patch_execution_applied", False)
                    setattr(last_patch_plan, "patch_execution_skipped_reason", "not_executed")
                    setattr(last_patch_plan, "patch_effectiveness_label", "not_applied")
                    if not str(getattr(last_patch_plan, "patch_no_gain_reason", "")).strip():
                        setattr(last_patch_plan, "patch_no_gain_reason", "not_executed")
                    scene_patch_execution_records.append(
                        {
                            "patch_execution_id": str(getattr(last_patch_plan, "patch_execution_id", "")),
                            "patch_plan_id": str(getattr(last_patch_plan, "plan_id", "")),
                            "round": int(getattr(last_patch_plan, "patch_execution_round", 0) or 0),
                            "scope": str(getattr(last_patch_plan, "patch_execution_scope", "")),
                            "status": "skipped",
                            "applied": False,
                            "skipped_reason": "not_executed",
                        }
                    )

            replan_result = LocalReplanResult(replan_id=f"local_replan_{scene.scene_id}", triggered=False, applied=False)
            should_try_replan = (
                (not accepted)
                and self.config.generation_controls.enable_replan_hook
                and bool(policies.replan.is_enabled)
                and (local_replans_used < max_local_replans)
                and (
                    plan_deviation_detected
                    or any(v.needs_replan or v.fatal for v in last_violations)
                    or failed_reason in {"patch_budget_exhausted", "repair_oscillation", "patch_scope_exhausted"}
                )
                and ((not regen_only_on_fatal) or any(v.fatal for v in last_violations))
                and (
                    (not prefer_sentence_repair_first)
                    or scene_patch_exhausted_targeted
                    or any(v.fatal for v in last_violations)
                )
            )
            if should_try_replan:
                replan_result = self.local_replanner.replan(
                    current_scene=scene,
                    all_scenes=scenes,
                    current_scene_idx=scene_cursor,
                    dynamic_memory=state.dynamic_memory,
                    static_memory=state.static_memory,
                    violations=last_violations,
                    weighted_future_constraints=list(
                        prepared_constraints.get("critical_constraints_preserved", [])
                    )
                    if bool(policies.replan.weighted)
                    else [],
                    failure_reason=failed_reason or "scene_rejected",
                )
                self.trace_writer.write_scene_replan(scene.scene_id, replan_result)
                if metrics is not None:
                    metrics.record_replan_result(scene.scene_id, replan_result)
                if replan_result.applied:
                    local_replans_used += 1
                    replacement_map = {item.scene_id: item for item in replan_result.revised_scenes}
                    for offset, revised in enumerate(replan_result.revised_scenes, start=1):
                        if scene_cursor + offset < len(scenes):
                            scenes[scene_cursor + offset] = revised
                    story_plan.replace_scenes(replacement_map)
                    state.dynamic_memory.revision_records.append(
                        {
                            "scene_id": scene.scene_id,
                            "stage": "local_replan_applied",
                            "replan_id": replan_result.replan_id,
                            "changed_scene_ids": replan_result.changed_scene_ids,
                            "impact_summary": replan_result.impact_summary,
                        }
                    )

            # In relaxed mode, avoid all-scene rejection loops that can return empty story.
            if (
                (not accepted)
                and (not reject_on_violation)
                and isinstance(scene_text, str)
                and scene_text.strip()
                and (not scene_too_short_guard_triggered)
            ):
                accepted = True
                state.dynamic_memory.revision_records.append(
                    {
                        "scene_id": scene.scene_id,
                        "stage": "degraded_acceptance",
                        "reason": "reject_on_violation_disabled",
                        "message": "accepted with unresolved violations to avoid empty-story failure loop",
                    }
                )

            chunk = StoryChunk(
                chunk_id=scene.scene_id,
                text=scene_text,
                position=scene.scene_index,
                accepted=False,
                revised=had_revision,
                planner_goal=scene.objective,
            )
            if accepted:
                chunk.accepted = True
                if flags["generation_layer"] == "plan_only":
                    pass
                else:
                    state.dynamic_memory = self.dynamic_manager.apply_update(
                        state.dynamic_memory,
                        final_delta,
                        report=final_report,
                        action=final_action,
                        inferred_constraints=reasoning.inferred_constraints,
                    )
            else:
                chunk.accepted = False
                if flags["generation_layer"] == "plan_only":
                    pass
                else:
                    state.dynamic_memory = self.dynamic_manager.reject_update(
                        state.dynamic_memory,
                        final_delta,
                        report=final_report,
                        reason=(
                            "scene_rejected_due_plan_deviation"
                            if plan_deviation_detected
                            else "scene_rejected_after_local_repair"
                        ),
                        action=final_action,
                        inferred_constraints=reasoning.inferred_constraints,
                    )

            state.story_chunks.append(chunk)
            state.last_report = final_report
            prev_scene_entropy_mean = float(entropy_profile.scene_entropy_mean)
            accepted_story_text = "\n\n".join(x.text for x in state.story_chunks if x.accepted)
            current_story_words = self.length_controller.count_words(accepted_story_text)
            scene_length_progress_ratio = float(
                current_story_words / max(1, int(requested_target_length))
            )
            unresolved_threads_after_scene = list(state.dynamic_memory.timeline_plot.active_plot_threads[-6:]) + list(
                state.dynamic_memory.timeline_plot.unresolved_foreshadowing[-4:]
            )
            scene_premature_closure_warning = bool(
                self.length_controller.detect_premature_closure(
                    latest_scene_text=scene_text,
                    unresolved_threads=unresolved_threads_after_scene,
                )
                and scene_length_progress_ratio < 0.95
            )
            scene_under_generation_warning = bool(
                scene_length_progress_ratio < float(self.config.length_control.progress_warning_ratio)
                and scene_cursor >= max(1, int(len(scenes) // 2))
            )
            under_generation_warning_triggered = under_generation_warning_triggered or scene_under_generation_warning
            premature_closure_warning_triggered = (
                premature_closure_warning_triggered or scene_premature_closure_warning
            )
            self.trace_writer.write_scene_memory_after(scene.scene_id, state.dynamic_memory)
            if metrics is not None:
                metrics.finalize_scene(scene.scene_id, accepted=accepted)
            diag_row = build_scene_diagnostic_record(
                prompt_id=sample.prompt_id,
                scene_id=scene.scene_id,
                variant_name=variant.variant_name,
                accepted=accepted,
                violations=last_violations,
                patch_rounds=int(local_patch_round),
                paragraph_patch_used=bool(paragraph_patch_round > 0),
                scene_regen_used=bool(scene_regen_rounds > 0),
                scene_regen_rounds=int(scene_regen_rounds),
                scene_patch_attempted_targeted=bool(scene_patch_attempted_targeted),
                scene_patch_exhausted_targeted=bool(scene_patch_exhausted_targeted),
                local_replan_triggered=bool(replan_result.applied),
                future_conflict_penalty=float(
                    (final_report.patch_plan.future_conflict_penalty if final_report and final_report.patch_plan else 0.0)
                ),
                unchanged_ratio=float(last_unchanged_ratio),
                preservation_failures=int(scene_preservation_failures),
                oscillation_detected=bool(scene_oscillation_detected),
                final_repair_scope=(last_patch_scope or "none"),
                final_objective_breakdown=(
                    dict(final_report.patch_plan.global_objective_breakdown)
                    if final_report and final_report.patch_plan
                    else {}
                ),
                patch_plan=(
                    last_executed_patch_plan
                    if (last_executed_patch_plan is not None)
                    else (
                        final_report.patch_plan
                        if (final_report and final_report.patch_plan is not None)
                        else last_patch_plan
                    )
                ),
                entropy_profile=entropy_profile,
                entropy_triggered_validation=entropy_triggered_validation,
                entropy_triggered_patch_escalation=entropy_triggered_patch_escalation,
                entropy_triggered_replan=entropy_triggered_replan,
                overlap_high_entropy_violation=self._entropy_violation_overlap(
                    entropy_profile=entropy_profile,
                    violations=last_violations,
                ),
                entropy_validation_mode=validation_mode,
                entropy_validation_budget=int(validation_budget),
                delta_uncertainty=float(entropy_profile.delta_uncertainty),
                sentence_uncertainty_variance=float(entropy_profile.sentence_uncertainty_variance),
                round_uncertainty_trend=float(entropy_profile.round_uncertainty_trend),
                joint_action_events=list(scene_joint_action_events),
                patch_execution_records=list(scene_patch_execution_records),
                memory_binding_mode=scene_memory_binding_mode,
                generation_control_mode=scene_generation_control_mode,
                memory_binding_decision_reasons=list(scene_binding_decision_reasons),
                strengthened_memory_blocks=list(scene_strengthened_memory_blocks),
                strengthened_constraints=list(scene_strengthened_constraints),
                rewrite_memory_binding_mode=str(
                    last_rewrite_control_context.get("memory_binding_mode", "normal_binding")
                ),
                rewrite_generation_control_mode=str(
                    last_rewrite_control_context.get("generation_control_mode", "normal_generation")
                ),
                generation_control_context=dict(last_generation_control_context),
                rewrite_control_context=dict(last_rewrite_control_context),
                dynamic_memory_update_status={
                    "accepted": bool(accepted),
                    "dynamic_memory_update_applied": bool(accepted),
                    "rejected_update_applied": bool(not accepted),
                },
                retrieved_experience_count=int(len(scene_retrieved_experience_items)),
                retrieved_experience_items=list(scene_retrieved_experience_items[:3]),
                desired_target_length=int(desired_target_length),
                requested_target_length=int(requested_target_length),
                length_compensation_factor=float(length_compensation_factor),
                length_progress_ratio=float(scene_length_progress_ratio),
                scene_word_count_first_pass=int(scene_word_count_first_pass),
                scene_too_short_guard_triggered=bool(scene_too_short_guard_triggered),
                scene_too_short_guard_threshold=int(min_scene_words_guard),
                under_generation_warning_triggered=bool(scene_under_generation_warning),
                premature_closure_warning_triggered=bool(scene_premature_closure_warning),
                entropy_validation_only_zero_influence=(
                    bool(flags.get("entropy_validation_only", False))
                    and (not bool(entropy_profile.constraint_sensitive_risk_scores))
                    and (not bool(entropy_profile.linked_sentence_ids))
                    and (not bool(entropy_profile.linked_constraint_ids))
                    and (not bool(entropy_triggered_patch_escalation))
                    and (not bool(entropy_triggered_replan))
                ),
            )
            diagnostics_records.append(diag_row)

            if (not accepted) and self.config.incremental.stop_on_failed_scene and not replan_result.applied:
                self.logger.warning(
                    "Scene %s rejected after repairs. stopping=true reason=%s",
                    scene.scene_id,
                    last_violations[0].message if last_violations else (failed_reason or "unknown"),
                )
                break

            scene_cursor += 1

        state.is_finished = True
        output_record = build_output_record(
            state=state,
            prompt_id=sample.prompt_id,
            prompt_text=sample.prompt,
            language=sample.language,
            task_type=sample.task_type,
            model_name=resolved_model_name,
        )
        output_record.metadata["num_scenes_planned"] = len(scenes)
        output_record.metadata["num_scenes_accepted"] = len([c for c in state.story_chunks if c.accepted])
        output_record.metadata["incremental_mode"] = True
        output_record.metadata["reject_on_violation"] = bool(reject_on_violation)
        output_record.metadata["patch_based_repair"] = True
        output_record.metadata["max_sentence_patch_rounds"] = max_sentence_patch_rounds
        output_record.metadata["max_paragraph_patch_rounds"] = max_paragraph_patch_rounds
        output_record.metadata["local_replans_used"] = local_replans_used
        output_record.metadata["variant_name"] = variant.variant_name
        actual_generated_length = self.length_controller.count_words(output_record.generated_story)
        length_completion_ratio = float(actual_generated_length / max(1, int(requested_target_length)))
        final_under_generation_warning = bool(
            length_completion_ratio < float(self.config.length_control.under_generation_completion_ratio)
        )
        unresolved_threads_final = list(state.dynamic_memory.timeline_plot.active_plot_threads[-6:]) + list(
            state.dynamic_memory.timeline_plot.unresolved_foreshadowing[-4:]
        )
        final_premature_closure_warning = bool(
            self.length_controller.detect_premature_closure(
                latest_scene_text=output_record.generated_story,
                unresolved_threads=unresolved_threads_final,
            )
            and length_completion_ratio < 0.95
        )
        under_generation_warning_triggered = under_generation_warning_triggered or final_under_generation_warning
        premature_closure_warning_triggered = (
            premature_closure_warning_triggered or final_premature_closure_warning
        )
        _ = self.length_controller.update_online_factor(
            model_id=resolved_model_name,
            desired_target_length=int(desired_target_length),
            actual_generated_length=int(actual_generated_length),
        )
        experience_bank_update_count = self.experience_bank.update_from_sample(
            model_id=resolved_model_name,
            sample_id=sample.prompt_id,
            task_type=sample.task_type,
            scene_records=diagnostics_records,
            length_feedback={
                "under_generation_warning_triggered": bool(under_generation_warning_triggered),
                "premature_closure_warning_triggered": bool(premature_closure_warning_triggered),
            },
        )
        most_common_failure_patterns_by_model = self.experience_bank.most_common_failure_patterns_by_model(top_k=5)
        output_record.metadata["desired_target_length"] = int(desired_target_length)
        output_record.metadata["requested_target_length"] = int(requested_target_length)
        output_record.metadata["length_compensation_factor"] = float(length_compensation_factor)
        output_record.metadata["actual_generated_length"] = int(actual_generated_length)
        output_record.metadata["length_completion_ratio"] = float(length_completion_ratio)
        output_record.metadata["under_generation_warning_triggered"] = bool(under_generation_warning_triggered)
        output_record.metadata["premature_closure_warning_triggered"] = bool(
            premature_closure_warning_triggered
        )
        output_record.metadata["retrieved_experience_items"] = list(retrieved_experience_items[:16])
        output_record.metadata["retrieved_experience_count"] = int(len(retrieved_experience_items))
        output_record.metadata["experience_bank_update_count"] = int(experience_bank_update_count)
        output_record.metadata["most_common_failure_patterns_by_model"] = dict(
            most_common_failure_patterns_by_model
        )
        if metrics is not None:
            summary = metrics.summarize()
            output_record.metadata["repair_metrics"] = summary
            self.trace_writer.write_metrics_summary(summary)
        diagnostics_summary = aggregate_story_diagnostics(
            diagnostics_records,
            delayed_gain_min_uncertainty_drop=float(
                self.config.entropy_monitor.delayed_gain_min_uncertainty_drop
            ),
            delayed_gain_min_joint_risk_drop=float(
                self.config.entropy_monitor.delayed_gain_min_joint_risk_drop
            ),
        )
        diagnostics_summary["desired_target_length"] = int(desired_target_length)
        diagnostics_summary["requested_target_length"] = int(requested_target_length)
        diagnostics_summary["length_compensation_factor"] = float(length_compensation_factor)
        diagnostics_summary["actual_generated_length"] = int(actual_generated_length)
        diagnostics_summary["length_completion_ratio"] = float(length_completion_ratio)
        diagnostics_summary["under_generation_warning_triggered"] = bool(under_generation_warning_triggered)
        diagnostics_summary["premature_closure_warning_triggered"] = bool(
            premature_closure_warning_triggered
        )
        diagnostics_summary["retrieved_experience_count"] = int(len(retrieved_experience_items))
        diagnostics_summary["experience_bank_update_count"] = int(experience_bank_update_count)
        diagnostics_summary["most_common_failure_patterns_by_model"] = dict(
            most_common_failure_patterns_by_model
        )
        output_record.metadata["diagnostic_summary"] = diagnostics_summary
        output_record.metadata["diagnostic_scene_records"] = diagnostics_records
        self._write_diagnostics_artifacts(
            prompt_id=sample.prompt_id,
            variant_name=variant.variant_name,
            scene_records=diagnostics_records,
            summary=diagnostics_summary,
        )
        self.trace_writer.write_final_story(output_record.generated_story)
        return state, output_record

    def _configure_variant_runtime(self, flags: Dict[str, bool], policies: object) -> None:
        policies.patch_planner.configure(self.patch_planner, self.config)
        if not flags["future_penalty_enabled"]:
            self.patch_planner.lambda_future_conflict = 0.0
        if flags["grounding_layer"] == "basic_anchor_grounding":
            self.patch_planner.low_confidence_threshold = 0.0
        else:
            self.patch_planner.low_confidence_threshold = float(
                self.config.generation_controls.low_confidence_anchor_threshold
            )
        # ERM runtime follows variant layer as primary switch.
        self.config.entropy_monitor.enable_entropy_monitor = bool(flags.get("entropy_monitor_enabled", False))
        template = str(self.config.entropy_monitor.joint_weight_template or "balanced").strip().lower()
        if template in {"uncertainty_heavy", "uncertainty_heavy_calibrated"}:
            self.config.entropy_monitor.joint_uncertainty_weight = 0.65
            self.config.entropy_monitor.joint_symbolic_weight = 0.25
            self.config.entropy_monitor.joint_memory_weight = 0.10
        elif template in {"symbolic_heavy", "symbolic_heavy_calibrated"}:
            self.config.entropy_monitor.joint_uncertainty_weight = 0.30
            self.config.entropy_monitor.joint_symbolic_weight = 0.55
            self.config.entropy_monitor.joint_memory_weight = 0.15
        elif template in {"memory_heavy", "memory_heavy_calibrated"}:
            self.config.entropy_monitor.joint_uncertainty_weight = 0.30
            self.config.entropy_monitor.joint_symbolic_weight = 0.30
            self.config.entropy_monitor.joint_memory_weight = 0.40
        else:
            self.config.entropy_monitor.joint_uncertainty_weight = 0.45
            self.config.entropy_monitor.joint_symbolic_weight = 0.40
            self.config.entropy_monitor.joint_memory_weight = 0.15
        if template in {
            "balanced_calibrated",
            "uncertainty_heavy_calibrated",
            "symbolic_heavy_calibrated",
            "memory_heavy_calibrated",
        }:
            self.config.entropy_monitor.joint_validation_threshold = 0.44
            self.config.entropy_monitor.joint_validation_min_symbolic_pressure = 0.28
            self.config.entropy_monitor.joint_validation_min_uncertainty_trend = 0.10
            self.config.entropy_monitor.joint_validation_max_low_violation_guard = 1
            self.config.entropy_monitor.joint_validation_require_dual_signal = True
            self.config.entropy_monitor.joint_patch_threshold = 0.47
            self.config.entropy_monitor.joint_patch_min_symbolic_pressure = 0.30
            self.config.entropy_monitor.joint_patch_min_local_failure_signal = 0.20
            self.config.entropy_monitor.joint_patch_escalation_threshold = 0.62
            self.config.entropy_monitor.joint_replan_threshold = 0.76
            self.config.entropy_monitor.joint_replan_min_persistent_risk_steps = 2
            self.config.entropy_monitor.joint_replan_requires_patch_failure = True

    def _project_constraint_layer_violations(
        self,
        violations: Sequence[ConstraintViolation],
        policy: object,
    ) -> List[ConstraintViolation]:
        return list(policy.project(violations))

    def _compute_entropy_profile(
        self,
        *,
        scene_plan: ScenePlan,
        extraction: object,
        prepared_constraints: Dict[str, object],
        previous_entropy_mean: float,
        flags: Dict[str, bool],
        generation_metadata: Dict[str, object] | None = None,
        generation_tokens: List[str] | None = None,
        generation_token_logprobs: List[float] | None = None,
        generation_top_logprobs: List[Dict[str, float]] | None = None,
        round_uncertainty_history: List[float] | None = None,
    ) -> EntropyRiskProfile:
        if not bool(flags.get("entropy_monitor_enabled", False)):
            return EntropyRiskProfile()
        if not bool(self.config.entropy_monitor.enable_entropy_monitor):
            return EntropyRiskProfile()
        weighted_payload = prepared_constraints.get("weighted_constraints_tiered", [])
        weighted_constraints = self.patch_planner._normalize_weighted_future_constraints(weighted_payload)
        profile = self.entropy_monitor.analyze_scene(
            scene_plan=scene_plan,
            scene_text=str(getattr(extraction, "scene_text", "")),
            sentences=list(getattr(extraction, "sentences", []) or []),
            weighted_constraints=weighted_constraints,
            previous_entropy_mean=float(previous_entropy_mean),
            force_validation=bool(
                self.config.entropy_monitor.entropy_high_risk_escalate_validation
                and not bool(flags.get("plain_generate_only_mode", False))
            ),
            allow_scope_escalation=bool(
                self.config.entropy_monitor.entropy_high_risk_escalate_patch_scope
                and bool(flags.get("entropy_scope_escalation_enabled", False))
                and not bool(flags.get("entropy_validation_only", False))
            ),
            allow_replan_trigger=bool(
                self.config.entropy_monitor.entropy_high_risk_replan_trigger
                and bool(flags.get("entropy_replan_trigger_enabled", False))
                and not bool(flags.get("entropy_validation_only", False))
            ),
            tokens=list(generation_tokens or []),
            token_logprobs=list(generation_token_logprobs or []),
            top_logprobs=list(generation_top_logprobs or []),
        )
        if isinstance(round_uncertainty_history, list):
            current_uncertainty = float(profile.scene_uncertainty_mean)
            if round_uncertainty_history:
                profile.delta_uncertainty = float(current_uncertainty - float(round_uncertainty_history[-1]))
                profile.round_uncertainty_trend = float(current_uncertainty - float(round_uncertainty_history[0]))
            else:
                profile.delta_uncertainty = 0.0
                profile.round_uncertainty_trend = 0.0
            round_uncertainty_history.append(current_uncertainty)

        delta_up = profile.delta_uncertainty >= float(
            max(0.0, self.config.entropy_monitor.delta_uncertainty_uptrend_threshold)
        )
        trend_up = profile.round_uncertainty_trend >= float(
            max(0.0, self.config.entropy_monitor.round_uncertainty_trend_threshold)
        )
        variance_high = profile.sentence_uncertainty_variance >= float(
            max(0.0, self.config.entropy_monitor.sentence_uncertainty_variance_threshold)
        )
        if delta_up or trend_up:
            profile.triggered_validation = True
        if (
            delta_up
            and variance_high
            and bool(self.config.entropy_monitor.entropy_high_risk_escalate_patch_scope)
            and bool(flags.get("entropy_scope_escalation_enabled", False))
        ):
            profile.triggered_patch = True
            profile.triggered_patch_escalation = True
        if (
            trend_up
            and bool(self.config.entropy_monitor.entropy_high_risk_replan_trigger)
            and bool(flags.get("entropy_replan_trigger_enabled", False))
        ):
            profile.triggered_replan = True
        if bool(flags.get("entropy_validation_only", False)):
            profile.triggered_patch = False
            profile.triggered_patch_escalation = False
            profile.triggered_replan = False
            profile.constraint_sensitive_risk_scores = {}
            profile.linked_sentence_ids = []
            profile.linked_constraint_ids = []
        if profile.triggered_validation:
            profile.triggered_validation_mode = "escalated"
            profile.triggered_validation_budget = int(
                max(1, 1 + int(self.config.entropy_monitor.validation_escalation_extra_checks))
            )
        else:
            profile.triggered_validation_mode = "standard"
            profile.triggered_validation_budget = 1
        if isinstance(generation_metadata, dict):
            if (not profile.uncertainty_available) and bool(generation_metadata.get("uncertainty_available", False)):
                profile.uncertainty_available = True
            source = str(generation_metadata.get("uncertainty_source", "")).strip()
            if (
                source
                and source != "none"
                and bool(generation_metadata.get("uncertainty_available", False))
                and profile.source_type == "none"
            ):
                profile.source_type = source
        return profile

    def _entropy_patch_prior_inputs(
        self,
        *,
        entropy_profile: EntropyRiskProfile,
        flags: Dict[str, bool],
    ) -> Tuple[Dict[str, float], List[str]]:
        if bool(flags.get("entropy_validation_only", False)):
            return {}, []
        return (
            dict(entropy_profile.constraint_sensitive_risk_scores),
            list(entropy_profile.linked_sentence_ids),
        )

    def _build_violation_context_payload(
        self,
        *,
        violations: Sequence[ConstraintViolation],
        prepared_constraints: Dict[str, object],
        entropy_profile: EntropyRiskProfile,
        delta: object,
        recent_action_events: Sequence[Dict[str, object]] | None = None,
    ) -> Dict[str, object]:
        sentence_ids: List[str] = []
        constraint_ids: List[str] = []
        critical_ids: List[str] = []
        conflict_tokens: List[str] = []
        for item in violations:
            vid = self.patch_planner._violation_id(item)
            constraint_ids.append(vid)
            if bool(item.is_hard) or int(item.constraint_tier) <= 2:
                critical_ids.append(vid)
            for anchor in item.anchors:
                for sid in anchor.sentence_ids:
                    if sid not in sentence_ids:
                        sentence_ids.append(sid)
            for tok in item.related_ids:
                token = str(tok).strip().lower()
                if token and token not in conflict_tokens:
                    conflict_tokens.append(token)
        conflict_candidates = prepared_constraints.get("conflict_candidates", [])
        if isinstance(conflict_candidates, list):
            for token in conflict_candidates[:12]:
                t = str(token).strip().lower()
                if t and t not in conflict_tokens:
                    conflict_tokens.append(t)
        memory_activity = (
            len(getattr(delta, "new_events", []) or [])
            + len(getattr(delta, "updated_entities", []) or [])
            + len(getattr(delta, "new_relations", {}) or {})
            + len(getattr(delta, "new_facts", {}) or {})
        )
        recent_no_gain_alignment: List[float] = []
        if recent_action_events:
            for event in list(recent_action_events)[-4:]:
                if not isinstance(event, dict):
                    continue
                if str(event.get("selected_action", "")) not in {"patch", "patch_plus_escalation"}:
                    continue
                if bool(event.get("improved_violations", False)) or bool(event.get("improved_transition_violations", False)):
                    continue
                try:
                    recent_no_gain_alignment.append(float(event.get("patch_target_joint_alignment_score", 0.0)))
                except (TypeError, ValueError):
                    continue
        return {
            "violation_sentence_ids": sentence_ids,
            "violation_constraint_ids": list(dict.fromkeys(constraint_ids)),
            "critical_constraint_ids": list(dict.fromkeys(critical_ids)),
            "conflict_tokens": conflict_tokens,
            "memory_instability": float(max(0.0, min(1.0, memory_activity / 10.0))),
            "entropy_linked_sentence_ids": list(dict.fromkeys(entropy_profile.linked_sentence_ids)),
            "recent_no_gain_alignment_mean": (
                float(sum(recent_no_gain_alignment) / max(1, len(recent_no_gain_alignment)))
                if recent_no_gain_alignment
                else 0.0
            ),
        }

    def _build_generation_control_context(
        self,
        *,
        stage: str,
        scene_plan: ScenePlan,
        static_memory: object,
        dynamic_memory: object,
        prepared_constraints: Dict[str, object],
        entropy_profile: EntropyRiskProfile,
        violations: Sequence[ConstraintViolation],
        flags: Dict[str, bool],
    ) -> Dict[str, object]:
        control = self._risk_adaptive_control_context(
            scene_plan=scene_plan,
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
            prepared_constraints=prepared_constraints,
            entropy_profile=entropy_profile,
            violations=violations,
            flags=flags,
        )
        execution_spec = prepared_constraints.get("execution_spec", {})
        if not isinstance(execution_spec, dict):
            execution_spec = {}
        selected_operator = prepared_constraints.get("selected_operator", {})
        if not isinstance(selected_operator, dict):
            selected_operator = {}
        reasoning_constraints = {
            "required_constraints": list(prepared_constraints.get("required", []) or []),
            "must_keep_constraints": list(prepared_constraints.get("must_keep", []) or []),
            "forbidden_constraints": list(prepared_constraints.get("forbidden", []) or []),
            "hard_constraints": list(prepared_constraints.get("hard_constraints", []) or []),
            "soft_constraints": list(prepared_constraints.get("soft_constraints", []) or []),
            "required_state_changes": list(prepared_constraints.get("required_state_changes", []) or []),
            "forbidden_state_changes": list(
                prepared_constraints.get("forbidden_state_changes", scene_plan.forbidden_state_changes) or []
            ),
            "allowed_transitions": list(prepared_constraints.get("allowed_transitions", []) or []),
            "forbidden_transitions": list(prepared_constraints.get("forbidden_transitions", []) or []),
            "execution_spec_required_entities": list(execution_spec.get("required_entities", []) or []),
            "execution_spec_required_events": list(execution_spec.get("required_events", []) or []),
            "execution_spec_required_state_changes": list(execution_spec.get("required_state_changes", []) or []),
            "execution_spec_forbidden_patterns": list(execution_spec.get("forbidden_patterns", []) or []),
            "operator_required_effects": list(selected_operator.get("required_effects", []) or []),
            "operator_postconditions": list(selected_operator.get("postconditions", []) or []),
        }
        dual_memory = {
            "static_memory_blocks_used": list(control.get("static_memory_reinforcement", [])),
            "dynamic_memory_blocks_used": list(control.get("dynamic_memory_reinforcement", [])),
            "static_memory_blocks_strengthened": list(control.get("static_memory_reinforcement", [])),
            "dynamic_memory_blocks_strengthened": list(control.get("dynamic_memory_reinforcement", [])),
        }
        assurance = {
            "symbolic_violation_count": int(len(list(violations))),
            "symbolic_transition_or_execution_conflict": bool(
                self._critical_conflict_presence(violations=violations, prepared_constraints=prepared_constraints)
            ),
            "uncertainty_warning_enabled": bool(flags.get("entropy_monitor_enabled", False))
            and bool(self.config.entropy_monitor.enable_entropy_monitor),
            "uncertainty_warning_tier": str(entropy_profile.final_risk_tier or "low_risk"),
            "uncertainty_warning_score": float(entropy_profile.joint_risk_score),
            "uncertainty_warning_drives_mode_change": bool(control.get("memory_binding_mode", "normal_binding") != "normal_binding"),
            "final_judgment_owner": "symbolic_validator_checker",
        }
        return {
            "stage": str(stage),
            "reasoning_constraints": reasoning_constraints,
            "uncertainty_guided_control": {
                "memory_binding_mode": str(control.get("memory_binding_mode", "normal_binding")),
                "generation_control_mode": str(control.get("generation_control_mode", "normal_generation")),
                "decision_reasons": list(control.get("decision_reasons", [])),
                "strengthened_memory_blocks": list(control.get("strengthened_memory_blocks", [])),
                "strengthened_constraints": list(control.get("strengthened_constraints", [])),
                "required_state_reminders": list(control.get("required_state_reminders", [])),
                "forbidden_state_reminders": list(control.get("forbidden_state_reminders", [])),
                "canonical_state_phrasing": list(control.get("canonical_state_phrasing", [])),
                "control_guidance": list(control.get("control_guidance", [])),
                "critical_constraints_frontloaded": bool(control.get("critical_constraints_frontloaded", False)),
            },
            "dual_memory": dual_memory,
            "assurance_boundary": assurance,
            # Keep flat keys for existing consumers.
            **dict(control),
            "generation_control_context_version": "dual_v1",
        }

    def _risk_adaptive_control_context(
        self,
        *,
        scene_plan: ScenePlan,
        static_memory: object,
        dynamic_memory: object,
        prepared_constraints: Dict[str, object],
        entropy_profile: EntropyRiskProfile,
        violations: Sequence[ConstraintViolation],
        flags: Dict[str, bool],
    ) -> Dict[str, object]:
        def _dedup(values: Sequence[object], limit: int) -> List[str]:
            out: List[str] = []
            for value in values:
                token = str(value or "").strip()
                if not token or token in out:
                    continue
                out.append(token)
                if len(out) >= limit:
                    break
            return out

        context: Dict[str, object] = {
            "memory_binding_mode": "normal_binding",
            "generation_control_mode": "normal_generation",
            "decision_reasons": [],
            "strengthened_memory_blocks": [],
            "strengthened_constraints": [],
            "required_state_reminders": [],
            "forbidden_state_reminders": [],
            "canonical_state_phrasing": [],
            "control_guidance": [],
            "critical_constraints_frontloaded": False,
            "static_memory_reinforcement": [],
            "dynamic_memory_reinforcement": [],
        }
        entropy_enabled = bool(flags.get("entropy_monitor_enabled", False)) and bool(
            self.config.entropy_monitor.enable_entropy_monitor
        )
        if not entropy_enabled:
            context["decision_reasons"] = ["entropy_monitor_disabled"]
            return context

        local_peak = 0.0
        if entropy_profile.local_constraint_uncertainty:
            local_peak = max(float(v) for v in entropy_profile.local_constraint_uncertainty.values())
        uncertainty_signal = float(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        float(entropy_profile.scene_uncertainty_mean),
                        float(entropy_profile.critical_constraint_uncertainty_peak),
                        float(local_peak),
                        float(max(0.0, entropy_profile.delta_uncertainty)),
                        float(max(0.0, entropy_profile.round_uncertainty_trend)),
                    ),
                ),
            )
        )
        symbolic_pressure = self._binding_symbolic_pressure_signal(
            violations=violations,
            prepared_constraints=prepared_constraints,
        )
        memory_volatility = self._memory_history_volatility_signal(dynamic_memory=dynamic_memory)
        has_critical_conflict = self._critical_conflict_presence(
            violations=violations,
            prepared_constraints=prepared_constraints,
        )

        is_strict = bool(
            (uncertainty_signal >= 0.62 and (symbolic_pressure >= 0.45 or has_critical_conflict))
            or (has_critical_conflict and symbolic_pressure >= 0.55 and memory_volatility >= 0.45)
        )
        is_reinforced = bool(
            is_strict
            or uncertainty_signal >= 0.38
            or symbolic_pressure >= 0.34
            or memory_volatility >= 0.44
            or has_critical_conflict
        )
        if is_strict:
            memory_binding_mode = "strict_binding"
            generation_control_mode = "strict_state_realization_generation"
        elif is_reinforced:
            memory_binding_mode = "reinforced_binding"
            generation_control_mode = "constrained_generation"
        else:
            memory_binding_mode = "normal_binding"
            generation_control_mode = "normal_generation"

        execution_spec = prepared_constraints.get("execution_spec", {})
        if not isinstance(execution_spec, dict):
            execution_spec = {}
        selected_operator = prepared_constraints.get("selected_operator", {})
        if not isinstance(selected_operator, dict):
            selected_operator = {}
        required_state_changes = _dedup(
            list(scene_plan.expected_state_changes)
            + list(prepared_constraints.get("required_state_changes", []) or [])
            + list(execution_spec.get("required_state_changes", []) or [])
            + list(selected_operator.get("required_effects", []) or [])
            + list(selected_operator.get("postconditions", []) or []),
            limit=16,
        )
        forbidden_state_changes = _dedup(
            list(scene_plan.forbidden_state_changes)
            + list(execution_spec.get("forbidden_patterns", []) or [])
            + list(prepared_constraints.get("forbidden_transitions", []) or []),
            limit=16,
        )
        critical_constraints = _dedup(
            list(prepared_constraints.get("hard_constraints", []) or [])
            + list(prepared_constraints.get("required", []) or [])
            + list(prepared_constraints.get("must_keep", []) or [])
            + list(prepared_constraints.get("entropy_critical_constraints", []) or []),
            limit=20,
        )
        canonical_state_phrasing = [self._normalize_state_phrase(token) for token in required_state_changes[:8]]
        static_lines = _dedup(
            list(getattr(static_memory.world_setting, "world_invariants", []) or [])
            + list(getattr(static_memory.timeline_plot, "required_plot_points", []) or []),
            limit=(12 if memory_binding_mode == "strict_binding" else 6),
        )
        dynamic_lines = _dedup(
            list(getattr(dynamic_memory.timeline_plot, "pending_constraints", []) or [])
            + list(getattr(dynamic_memory.timeline_plot, "active_goals", []) or [])
            + list(getattr(dynamic_memory.timeline_plot, "active_plot_threads", []) or []),
            limit=(12 if memory_binding_mode == "strict_binding" else 6),
        )

        decision_reasons: List[str] = []
        if uncertainty_signal >= 0.62:
            decision_reasons.append("scene_uncertainty_high")
        elif uncertainty_signal >= 0.38:
            decision_reasons.append("scene_uncertainty_medium")
        if float(entropy_profile.critical_constraint_uncertainty_peak) >= 0.56:
            decision_reasons.append("critical_constraint_uncertainty_peak_high")
        if symbolic_pressure >= 0.55:
            decision_reasons.append("symbolic_pressure_high")
        elif symbolic_pressure >= 0.34:
            decision_reasons.append("symbolic_pressure_medium")
        if memory_volatility >= 0.58:
            decision_reasons.append("memory_volatility_high")
        elif memory_volatility >= 0.44:
            decision_reasons.append("memory_volatility_medium")
        if has_critical_conflict:
            decision_reasons.append("execution_transition_operator_conflict_present")
        if not decision_reasons:
            decision_reasons.append("low_risk_profile")

        strengthened_blocks: List[str] = []
        strengthened_constraints: List[str] = []
        control_guidance: List[str] = []
        if memory_binding_mode == "reinforced_binding":
            strengthened_blocks = [
                "static_world_invariants",
                "dynamic_pending_constraints",
                "required_state_change_reminders",
            ]
            strengthened_constraints = _dedup(critical_constraints + required_state_changes[:4], limit=14)
            control_guidance = [
                "Preserve global invariants and critical constraints throughout scene generation.",
                "Realize required state changes when the scene transition implies them.",
                "Avoid parallel contradictory states in the same local context.",
            ]
        elif memory_binding_mode == "strict_binding":
            strengthened_blocks = [
                "static_world_invariants",
                "dynamic_pending_constraints",
                "execution_spec_required_state_changes",
                "operator_post_state_requirements",
                "canonical_state_phrasing_reminders",
            ]
            strengthened_constraints = _dedup(
                critical_constraints + required_state_changes + forbidden_state_changes,
                limit=18,
            )
            control_guidance = [
                "Preserve global invariants and hard constraints with zero contradiction.",
                "Explicitly textualize every required state change and operator-required post-state.",
                "Explicitly remove forbidden/conflicting states and do not keep both pre- and post-state.",
                "Maintain coherent transition order (causal and temporal continuity).",
            ]

        context.update(
            {
                "memory_binding_mode": memory_binding_mode,
                "generation_control_mode": generation_control_mode,
                "decision_reasons": list(decision_reasons),
                "strengthened_memory_blocks": list(strengthened_blocks),
                "strengthened_constraints": list(strengthened_constraints),
                "required_state_reminders": list(required_state_changes),
                "forbidden_state_reminders": list(forbidden_state_changes),
                "canonical_state_phrasing": list(canonical_state_phrasing),
                "control_guidance": list(control_guidance),
                "critical_constraints_frontloaded": bool(memory_binding_mode != "normal_binding"),
                "static_memory_reinforcement": list(static_lines),
                "dynamic_memory_reinforcement": list(dynamic_lines),
            }
        )
        return context

    def _apply_binding_policy_to_constraints(
        self,
        *,
        prepared_constraints: Dict[str, object],
        control_context: Dict[str, object],
    ) -> Dict[str, object]:
        out = dict(prepared_constraints)
        control_payload = dict(control_context)
        if isinstance(control_payload.get("uncertainty_guided_control"), dict):
            nested = dict(control_payload.get("uncertainty_guided_control", {}))
            for key in [
                "memory_binding_mode",
                "generation_control_mode",
                "decision_reasons",
                "strengthened_memory_blocks",
                "strengthened_constraints",
                "required_state_reminders",
                "forbidden_state_reminders",
                "canonical_state_phrasing",
                "control_guidance",
                "critical_constraints_frontloaded",
            ]:
                if key in nested and key not in control_payload:
                    control_payload[key] = nested.get(key)
        if isinstance(control_payload.get("dual_memory"), dict):
            mem = dict(control_payload.get("dual_memory", {}))
            if "static_memory_reinforcement" not in control_payload:
                control_payload["static_memory_reinforcement"] = list(
                    mem.get("static_memory_blocks_strengthened", mem.get("static_memory_blocks_used", []))
                )
            if "dynamic_memory_reinforcement" not in control_payload:
                control_payload["dynamic_memory_reinforcement"] = list(
                    mem.get("dynamic_memory_blocks_strengthened", mem.get("dynamic_memory_blocks_used", []))
                )
        memory_binding_mode = str(control_payload.get("memory_binding_mode", "normal_binding"))
        generation_control_mode = str(control_payload.get("generation_control_mode", "normal_generation"))

        def _dedup(values: Sequence[object], limit: int = 32) -> List[str]:
            result: List[str] = []
            for value in values:
                token = str(value or "").strip()
                if not token or token in result:
                    continue
                result.append(token)
                if len(result) >= limit:
                    break
            return result

        out["memory_binding_mode"] = memory_binding_mode
        out["generation_control_mode"] = generation_control_mode
        out["memory_binding_decision_reasons"] = list(control_payload.get("decision_reasons", []))
        out["strengthened_memory_blocks"] = list(control_payload.get("strengthened_memory_blocks", []))
        out["strengthened_constraints"] = list(control_payload.get("strengthened_constraints", []))
        out["required_state_reminders"] = list(control_payload.get("required_state_reminders", []))
        out["forbidden_state_reminders"] = list(control_payload.get("forbidden_state_reminders", []))
        out["canonical_state_phrasing"] = list(control_payload.get("canonical_state_phrasing", []))
        out["risk_control_guidance"] = list(control_payload.get("control_guidance", []))
        out["critical_constraints_frontloaded"] = bool(control_payload.get("critical_constraints_frontloaded", False))
        out["static_memory_reinforcement"] = list(control_payload.get("static_memory_reinforcement", []))
        out["dynamic_memory_reinforcement"] = list(control_payload.get("dynamic_memory_reinforcement", []))
        out["generation_control_context"] = dict(control_context)
        if memory_binding_mode != "normal_binding":
            out["hard_constraints"] = _dedup(
                list(control_payload.get("strengthened_constraints", []))
                + list(out.get("hard_constraints", []) or []),
                limit=24,
            )
            out["required"] = _dedup(
                list(out.get("required", []) or [])
                + list(control_payload.get("required_state_reminders", []))[:6],
                limit=18,
            )
            out["must_keep"] = _dedup(
                list(out.get("must_keep", []) or [])
                + list(control_payload.get("required_state_reminders", []))[:4],
                limit=18,
            )
            out["forbidden"] = _dedup(
                list(out.get("forbidden", []) or [])
                + list(control_payload.get("forbidden_state_reminders", []))[:6],
                limit=18,
            )
        return out

    def _binding_symbolic_pressure_signal(
        self,
        *,
        violations: Sequence[ConstraintViolation],
        prepared_constraints: Dict[str, object],
    ) -> float:
        critical_v = 0.0
        transition_related = 0.0
        for item in violations:
            if bool(item.is_hard) or int(item.constraint_tier) <= 2 or str(item.severity).lower() == "error":
                critical_v += 1.0
            token = f"{str(item.rule_type).lower()} {str(item.message).lower()}"
            if ("transition" in token) or ("execution spec" in token) or ("post-state" in token):
                transition_related += 1.0
        execution_spec = prepared_constraints.get("execution_spec", {})
        if not isinstance(execution_spec, dict):
            execution_spec = {}
        required_state_pressure = float(
            len(list(prepared_constraints.get("required_state_changes", []) or []))
            + len(list(execution_spec.get("required_state_changes", []) or []))
        )
        hard_constraints_pressure = float(len(list(prepared_constraints.get("hard_constraints", []) or [])))
        score = (
            min(1.0, critical_v / 4.0) * 0.38
            + min(1.0, transition_related / 3.0) * 0.34
            + min(1.0, required_state_pressure / 6.0) * 0.18
            + min(1.0, hard_constraints_pressure / 10.0) * 0.10
        )
        return float(max(0.0, min(1.0, score)))

    def _critical_conflict_presence(
        self,
        *,
        violations: Sequence[ConstraintViolation],
        prepared_constraints: Dict[str, object],
    ) -> bool:
        for item in violations:
            token = f"{str(item.rule_type).lower()} {str(item.message).lower()}"
            if (
                ("transition" in token)
                or ("execution spec" in token)
                or ("operator" in token and "post" in token)
                or ("required state changes" in token)
            ):
                return True
        if list(prepared_constraints.get("required_state_changes", []) or []):
            return True
        selected_operator = prepared_constraints.get("selected_operator", {})
        if isinstance(selected_operator, dict):
            if list(selected_operator.get("required_effects", []) or []):
                return True
            if list(selected_operator.get("postconditions", []) or []):
                return True
        execution_spec = prepared_constraints.get("execution_spec", {})
        if isinstance(execution_spec, dict):
            if list(execution_spec.get("required_state_changes", []) or []):
                return True
        return False

    def _memory_history_volatility_signal(self, *, dynamic_memory: object) -> float:
        revision_count = len(list(getattr(dynamic_memory, "revision_records", []) or [])[-6:])
        rejected_count = len(list(getattr(dynamic_memory, "rejected_deltas", []) or [])[-4:])
        pending_constraints = len(list(getattr(dynamic_memory.timeline_plot, "pending_constraints", []) or [])[-8:])
        active_threads = len(list(getattr(dynamic_memory.timeline_plot, "active_plot_threads", []) or [])[-6:])
        score = float(
            revision_count
            + (1.4 * rejected_count)
            + (0.35 * pending_constraints)
            + (0.25 * active_threads)
        )
        return float(max(0.0, min(1.0, score / 12.0)))

    def _normalize_state_phrase(self, token: object) -> str:
        return str(token or "").replace("_", " ").strip()

    def _build_rewrite_state_grounding_bundle(self, rewrite_metadata: Dict[str, object]) -> Dict[str, object]:
        required_states = list(
            rewrite_metadata.get(
                "rewrite_canonical_required_states",
                rewrite_metadata.get("rewrite_required_state_changes", []),
            )
            or []
        )
        forbidden_states = list(
            rewrite_metadata.get(
                "rewrite_canonical_forbidden_states",
                rewrite_metadata.get("rewrite_forbidden_state_changes", []),
            )
            or []
        )
        operator_states = list(
            rewrite_metadata.get(
                "rewrite_canonical_operator_post_states",
                rewrite_metadata.get("rewrite_operator_required_post_states", []),
            )
            or []
        )
        transition_hints = list(
            rewrite_metadata.get(
                "rewrite_transition_target_state_hints",
                rewrite_metadata.get("rewrite_required_state_changes", []),
            )
            or []
        )
        bundle = build_state_grounding_bundle(
            required_states=required_states,
            forbidden_states=forbidden_states,
            operator_post_states=operator_states,
            transition_target_states=transition_hints,
        )
        # Prefer existing rewrite payload groundings when available.
        if list(rewrite_metadata.get("rewrite_required_state_groundings", []) or []):
            bundle["required_state_groundings"] = list(rewrite_metadata.get("rewrite_required_state_groundings", []))
            bundle["canonical_required_states"] = [
                str(item.get("canonical", ""))
                for item in list(bundle.get("required_state_groundings", []))
                if isinstance(item, dict)
            ]
        if list(rewrite_metadata.get("rewrite_forbidden_state_groundings", []) or []):
            bundle["forbidden_state_groundings"] = list(rewrite_metadata.get("rewrite_forbidden_state_groundings", []))
            bundle["canonical_forbidden_states"] = [
                str(item.get("canonical", ""))
                for item in list(bundle.get("forbidden_state_groundings", []))
                if isinstance(item, dict)
            ]
        if list(rewrite_metadata.get("rewrite_operator_post_state_groundings", []) or []):
            bundle["operator_post_state_groundings"] = list(
                rewrite_metadata.get("rewrite_operator_post_state_groundings", [])
            )
            bundle["canonical_operator_post_states"] = [
                str(item.get("canonical", ""))
                for item in list(bundle.get("operator_post_state_groundings", []))
                if isinstance(item, dict)
            ]
        if list(rewrite_metadata.get("rewrite_transition_target_state_groundings", []) or []):
            bundle["transition_target_state_groundings"] = list(
                rewrite_metadata.get("rewrite_transition_target_state_groundings", [])
            )
            bundle["canonical_transition_target_states"] = [
                str(item.get("canonical", ""))
                for item in list(bundle.get("transition_target_state_groundings", []))
                if isinstance(item, dict)
            ]
        if list(rewrite_metadata.get("rewrite_transition_grounded_cues", []) or []):
            bundle["transition_grounded_cues"] = list(rewrite_metadata.get("rewrite_transition_grounded_cues", []))
        return bundle

    def _state_realization_post_check(
        self,
        *,
        patch_plan: object,
        rewrite_metadata: Dict[str, object],
        rewritten_scene_text: str,
        post_patch_transition_violations: Sequence[ConstraintViolation],
        before_transition_violations: int,
        after_transition_violations: int,
    ) -> Dict[str, object]:
        lower_text = str(rewritten_scene_text or "").lower()
        grounding_bundle = self._build_rewrite_state_grounding_bundle(rewrite_metadata)
        state_eval = evaluate_state_realization_with_grounding(
            rewritten_text=rewritten_scene_text,
            required_groundings=list(grounding_bundle.get("required_state_groundings", [])),
            forbidden_groundings=list(grounding_bundle.get("forbidden_state_groundings", [])),
            operator_groundings=list(grounding_bundle.get("operator_post_state_groundings", [])),
        )
        required_states = [str(x) for x in list(grounding_bundle.get("canonical_required_states", [])) if str(x).strip()]
        forbidden_states = [str(x) for x in list(grounding_bundle.get("canonical_forbidden_states", [])) if str(x).strip()]
        operator_required = [
            str(x) for x in list(grounding_bundle.get("canonical_operator_post_states", [])) if str(x).strip()
        ]
        conflict_type = str(getattr(patch_plan, "rewrite_conflict_type", rewrite_metadata.get("rewrite_conflict_type", "unknown")))
        targets_critical_conflict = conflict_type in {
            "execution_spec_conflict",
            "transition_conflict",
            "operator_post_state_conflict",
            "mixed_conflict",
        }
        hits_context = bool(getattr(patch_plan, "patch_target_hits_violation_context", False))
        realizes_required = bool(state_eval.get("rewrite_realizes_required_state_change", False))
        removes_forbidden = bool(state_eval.get("rewrite_removes_forbidden_state", False))
        realizes_operator_post = bool(state_eval.get("rewrite_realizes_operator_post_state", False))
        transition_cues = [str(x).strip().lower() for x in list(grounding_bundle.get("transition_grounded_cues", []))]
        transition_cue = any(
            f" {cue} " in f" {lower_text} " for cue in transition_cues if cue
        )
        transition_proxy = bool(
            int(after_transition_violations) < int(before_transition_violations)
            or (
                conflict_type in {"transition_conflict", "mixed_conflict"}
                and transition_cue
                and realizes_required
                and removes_forbidden
            )
        )
        failed_checks: List[str] = []
        missing_required_states: List[str] = list(state_eval.get("missing_required_states", []))
        remaining_forbidden_states: List[str] = list(state_eval.get("remaining_forbidden_states", []))
        if required_states and not realizes_required:
            failed_checks.append("required_state_not_realized")
        if forbidden_states and not removes_forbidden:
            failed_checks.append("forbidden_state_not_removed")
        if operator_required and not realizes_operator_post:
            failed_checks.append("operator_post_state_not_realized")
        if conflict_type in {"transition_conflict", "mixed_conflict"} and not transition_proxy:
            failed_checks.append("transition_coherence_not_restored")
        eligible_retry = bool(
            hits_context
            and targets_critical_conflict
            and bool(failed_checks)
        )
        retry_reason = "state_realization_post_check_failed" if failed_checks else ""
        if "required_state_not_realized" in failed_checks:
            retry_reason = "required_state_not_realized"
        elif "operator_post_state_not_realized" in failed_checks:
            retry_reason = "operator_post_state_not_realized"
        elif "forbidden_state_not_removed" in failed_checks:
            retry_reason = "forbidden_state_not_removed"
        elif "transition_coherence_not_restored" in failed_checks:
            retry_reason = "transition_coherence_not_restored"
        return {
            "rewrite_realizes_required_state_change": bool(realizes_required),
            "rewrite_removes_forbidden_state": bool(removes_forbidden),
            "rewrite_restores_transition_coherence_proxy": bool(transition_proxy),
            "rewrite_realizes_operator_post_state": bool(realizes_operator_post),
            "state_realization_post_check_failed_checks": list(failed_checks),
            "state_realization_post_check_retry_eligible": bool(eligible_retry),
            "state_realization_post_check_retry_reason": str(retry_reason),
            "missing_required_states": list(missing_required_states),
            "remaining_forbidden_states": list(remaining_forbidden_states),
            "state_realization_match_type": str(state_eval.get("state_realization_match_type", "no_match")),
            "forbidden_state_removal_match_type": str(
                state_eval.get("forbidden_state_removal_match_type", "no_match")
            ),
            "operator_post_state_match_type": str(state_eval.get("operator_post_state_match_type", "no_match")),
            "canonical_required_states": list(grounding_bundle.get("canonical_required_states", [])),
            "canonical_forbidden_states": list(grounding_bundle.get("canonical_forbidden_states", [])),
            "canonical_operator_post_states": list(grounding_bundle.get("canonical_operator_post_states", [])),
            "grounded_alias_matches": list(state_eval.get("grounded_alias_matches", [])),
            "conflict_type": conflict_type,
            "hits_violation_context": bool(hits_context),
        }

    def _populate_patch_effectiveness(
        self,
        *,
        patch_plan: object,
        repaired_metadata: Dict[str, object],
        before_transition_violations: int,
        after_transition_violations: int,
        before_total_violations: int,
        after_total_violations: int,
        before_constraint_violations: int,
        after_constraint_violations: int,
        before_uncertainty: float,
        after_uncertainty: float,
        rewrite_metadata: Dict[str, object] | None = None,
    ) -> None:
        target_ids = set(str(x) for x in list(getattr(patch_plan, "target_sentence_ids", []) or []))
        violation_context_ids = set(
            str(x) for x in list(getattr(patch_plan, "violation_context_sentence_ids", []) or [])
        )
        entropy_context_ids = set(
            str(x) for x in list(getattr(patch_plan, "entropy_context_sentence_ids", []) or [])
        )
        rewrite_ids: set[str] = set()
        for item in list(repaired_metadata.get("applied_patches", []) or []):
            if not isinstance(item, dict):
                continue
            for sid in list(item.get("target_sentence_ids", []) or []):
                if str(sid).strip():
                    rewrite_ids.add(str(sid).strip())
        if not rewrite_ids:
            rewrite_ids = set(str(x) for x in list(repaired_metadata.get("target_sentence_ids", []) or []))

        hits_violation_context = bool(target_ids.intersection(violation_context_ids))
        hits_entropy_context = bool(target_ids.intersection(entropy_context_ids))
        rewrites_key_span = bool(rewrite_ids.intersection(violation_context_ids)) if violation_context_ids else False
        reduces_transition = int(after_transition_violations) < int(before_transition_violations)
        reduces_constraint = int(after_constraint_violations) < int(before_constraint_violations)
        reduces_total = int(after_total_violations) < int(before_total_violations)
        reduces_uncertainty = float(after_uncertainty) < float(before_uncertainty)
        symbolic_proxy_changed = bool(reduces_transition or reduces_constraint or reduces_total)
        before_symbolic_state_proxy = float(max(0, int(before_transition_violations) + int(before_constraint_violations)))
        after_symbolic_state_proxy = float(max(0, int(after_transition_violations) + int(after_constraint_violations)))

        setattr(patch_plan, "patch_target_hits_violation_context", bool(hits_violation_context))
        setattr(patch_plan, "patch_target_hits_entropy_context", bool(hits_entropy_context))
        setattr(patch_plan, "patch_rewrites_key_conflict_span", bool(rewrites_key_span))
        setattr(patch_plan, "patch_changes_symbolic_state_proxy", bool(symbolic_proxy_changed))
        setattr(patch_plan, "patch_reduces_transition_violations", bool(reduces_transition))
        setattr(patch_plan, "patch_reduces_constraint_violations", bool(reduces_constraint))
        setattr(patch_plan, "patch_reduces_uncertainty", bool(reduces_uncertainty))
        setattr(patch_plan, "patch_before_transition_violation_count", int(before_transition_violations))
        setattr(patch_plan, "patch_after_transition_violation_count", int(after_transition_violations))
        setattr(patch_plan, "patch_before_constraint_violation_count", int(before_constraint_violations))
        setattr(patch_plan, "patch_after_constraint_violation_count", int(after_constraint_violations))
        setattr(patch_plan, "patch_before_violation_count", int(before_total_violations))
        setattr(patch_plan, "patch_after_violation_count", int(after_total_violations))
        setattr(patch_plan, "patch_before_symbolic_state_proxy", float(before_symbolic_state_proxy))
        setattr(patch_plan, "patch_after_symbolic_state_proxy", float(after_symbolic_state_proxy))
        setattr(patch_plan, "patch_before_uncertainty", float(before_uncertainty))
        setattr(patch_plan, "patch_after_uncertainty", float(after_uncertainty))
        rewrite_metadata = rewrite_metadata or {}
        rewrite_conflict_type_raw = str(rewrite_metadata.get("rewrite_conflict_type", "unknown") or "unknown")
        rewrite_conflict_type = rewrite_conflict_type_raw if rewrite_conflict_type_raw in {
            "transition_conflict",
            "execution_spec_conflict",
            "operator_post_state_conflict",
            "constraint_conflict",
            "mixed_conflict",
        } else "unknown"
        rewrite_target_scope = str(rewrite_metadata.get("rewrite_target_scope", "sentence") or "sentence")
        required_state_changes = [str(x) for x in list(rewrite_metadata.get("rewrite_required_state_changes", []) or [])]
        forbidden_state_changes = [str(x) for x in list(rewrite_metadata.get("rewrite_forbidden_state_changes", []) or [])]
        operator_required_post_states = [
            str(x) for x in list(rewrite_metadata.get("rewrite_operator_required_post_states", []) or [])
        ]
        grounding_bundle = self._build_rewrite_state_grounding_bundle(rewrite_metadata)
        rewritten_text_blob = " ".join(
            str(item.get("rewritten_text", ""))
            for item in list(rewrite_metadata.get("applied_patches", []) or [])
            if isinstance(item, dict)
        )
        if not rewritten_text_blob:
            rewritten_text_blob = str(rewrite_metadata.get("rewritten_text", "") or "")
        lower_blob = rewritten_text_blob.lower()
        state_eval = evaluate_state_realization_with_grounding(
            rewritten_text=rewritten_text_blob,
            required_groundings=list(grounding_bundle.get("required_state_groundings", [])),
            forbidden_groundings=list(grounding_bundle.get("forbidden_state_groundings", [])),
            operator_groundings=list(grounding_bundle.get("operator_post_state_groundings", [])),
        )
        required_tokens = [str(x) for x in list(grounding_bundle.get("canonical_required_states", [])) if str(x).strip()]
        forbidden_tokens = [str(x) for x in list(grounding_bundle.get("canonical_forbidden_states", [])) if str(x).strip()]
        operator_post_tokens = [
            str(x) for x in list(grounding_bundle.get("canonical_operator_post_states", [])) if str(x).strip()
        ]
        hits_required_state_change = bool(
            required_tokens
            and str(state_eval.get("state_realization_match_type", "no_match")) != "no_match"
        )
        removes_conflicting_state = bool(
            forbidden_tokens
            and bool(state_eval.get("rewrite_removes_forbidden_state", False))
        )
        realizes_operator_required_post_state = bool(
            operator_post_tokens and bool(state_eval.get("rewrite_realizes_operator_post_state", False))
        )
        realizes_required_state_change = bool(
            bool(state_eval.get("rewrite_realizes_required_state_change", False))
            or realizes_operator_required_post_state
        )
        removes_forbidden_state = bool(state_eval.get("rewrite_removes_forbidden_state", False))
        preserves_non_conflict_content = bool(
            float(rewrite_metadata.get("unchanged_ratio", 0.0) or 0.0) >= 0.55
            and bool(rewrite_metadata.get("protected_integrity_pass", True))
        )
        transition_msgs = [str(x) for x in list(rewrite_metadata.get("rewrite_transition_violations", []) or [])]
        execution_msgs = [str(x) for x in list(rewrite_metadata.get("rewrite_execution_spec_violations", []) or [])]
        transition_guidance = [str(x) for x in list(rewrite_metadata.get("rewrite_transition_coherence_guidance", []) or [])]
        transition_cues = [str(x).strip().lower() for x in list(grounding_bundle.get("transition_grounded_cues", []))]
        has_transition_cue = any(
            f" {cue} " in f" {lower_blob} " for cue in transition_cues if cue
        ) or lower_blob.startswith("after ")
        rewrite_targets_execution_spec_conflict = bool(
            bool(rewrite_metadata.get("rewrite_targets_execution_spec_conflict", False))
            or execution_msgs
            or rewrite_conflict_type in {"execution_spec_conflict", "mixed_conflict"}
        )
        rewrite_targets_required_state_change = bool(
            bool(rewrite_metadata.get("rewrite_targets_required_state_change", False))
            or required_tokens
            or operator_post_tokens
        )
        rewrite_targets_transition_conflict = bool(
            bool(rewrite_metadata.get("rewrite_targets_transition_conflict", False))
            or transition_msgs
            or rewrite_conflict_type in {"transition_conflict", "mixed_conflict"}
        )
        rewrite_targets_operator_post_state_conflict = bool(
            bool(rewrite_metadata.get("rewrite_targets_operator_post_state_conflict", False))
            or operator_post_tokens
            or rewrite_conflict_type in {"operator_post_state_conflict", "mixed_conflict"}
        )
        state_realization_match_type = str(state_eval.get("state_realization_match_type", "no_match"))
        forbidden_state_removal_match_type = str(state_eval.get("forbidden_state_removal_match_type", "no_match"))
        operator_post_state_match_type = str(state_eval.get("operator_post_state_match_type", "no_match"))
        rewrite_restores_transition_coherence_proxy = bool(
            reduces_transition
            or (
                rewrite_targets_transition_conflict
                and has_transition_cue
                and realizes_required_state_change
                and removes_forbidden_state
            )
        )
        alignment_score = _safe_float(getattr(patch_plan, "patch_target_joint_alignment_score", 0.0))
        if reduces_total or reduces_transition or reduces_constraint:
            effectiveness_label = "effective"
            no_gain_reason = ""
        elif (
            realizes_required_state_change
            and removes_forbidden_state
            and (rewrites_key_span or reduces_uncertainty or rewrite_restores_transition_coherence_proxy)
        ):
            effectiveness_label = "partial"
            no_gain_reason = "partial_conflict_resolution"
        elif (not rewrites_key_span) or float(repaired_metadata.get("unchanged_ratio", 0.0) or 0.0) >= 0.98:
            effectiveness_label = "cosmetic_only"
            no_gain_reason = "cosmetic_only_rewrite"
        else:
            effectiveness_label = "ineffective"
            no_gain_reason = "unknown"
        if effectiveness_label in {"ineffective", "cosmetic_only"}:
            if rewrite_targets_required_state_change and not realizes_required_state_change:
                no_gain_reason = "no_required_state_realization"
            elif forbidden_tokens and not removes_forbidden_state:
                no_gain_reason = "conflicting_state_not_removed"
            elif (
                hits_violation_context
                and rewrite_target_scope in {"sentence", "multi_sentence"}
                and (rewrite_targets_transition_conflict or rewrite_targets_execution_spec_conflict or rewrite_targets_operator_post_state_conflict)
            ):
                no_gain_reason = "target_hit_but_scope_too_small"
            elif effectiveness_label == "cosmetic_only":
                no_gain_reason = "cosmetic_only_rewrite"
            elif alignment_score < 0.25:
                no_gain_reason = "unknown"
        if effectiveness_label == "partial":
            if (not realizes_required_state_change) and rewrite_targets_required_state_change:
                no_gain_reason = "no_required_state_realization"
            elif forbidden_tokens and not removes_forbidden_state:
                no_gain_reason = "conflicting_state_not_removed"
        setattr(patch_plan, "patch_effectiveness_label", str(effectiveness_label))
        setattr(patch_plan, "patch_no_gain_reason", str(no_gain_reason))
        setattr(patch_plan, "rewrite_conflict_type", rewrite_conflict_type)
        setattr(patch_plan, "rewrite_target_scope", rewrite_target_scope)
        setattr(patch_plan, "rewrite_hits_required_state_change", bool(hits_required_state_change))
        setattr(patch_plan, "rewrite_removes_conflicting_state", bool(removes_conflicting_state))
        setattr(patch_plan, "rewrite_preserves_non_conflict_content", bool(preserves_non_conflict_content))
        setattr(patch_plan, "rewrite_targets_execution_spec_conflict", bool(rewrite_targets_execution_spec_conflict))
        setattr(patch_plan, "rewrite_targets_required_state_change", bool(rewrite_targets_required_state_change))
        setattr(patch_plan, "rewrite_targets_transition_conflict", bool(rewrite_targets_transition_conflict))
        setattr(
            patch_plan,
            "rewrite_targets_operator_post_state_conflict",
            bool(rewrite_targets_operator_post_state_conflict),
        )
        setattr(patch_plan, "rewrite_operator_required_post_states", list(operator_required_post_states))
        setattr(patch_plan, "rewrite_realizes_required_state_change", bool(realizes_required_state_change))
        setattr(patch_plan, "rewrite_removes_forbidden_state", bool(removes_forbidden_state))
        setattr(
            patch_plan,
            "rewrite_restores_transition_coherence_proxy",
            bool(rewrite_restores_transition_coherence_proxy),
        )
        setattr(
            patch_plan,
            "canonical_required_states",
            list(grounding_bundle.get("canonical_required_states", [])),
        )
        setattr(
            patch_plan,
            "canonical_forbidden_states",
            list(grounding_bundle.get("canonical_forbidden_states", [])),
        )
        setattr(
            patch_plan,
            "canonical_operator_post_states",
            list(grounding_bundle.get("canonical_operator_post_states", [])),
        )
        setattr(
            patch_plan,
            "grounded_alias_matches",
            list(state_eval.get("grounded_alias_matches", [])),
        )
        setattr(patch_plan, "state_realization_match_type", str(state_realization_match_type))
        setattr(
            patch_plan,
            "forbidden_state_removal_match_type",
            str(forbidden_state_removal_match_type),
        )
        setattr(
            patch_plan,
            "operator_post_state_match_type",
            str(operator_post_state_match_type),
        )

    def _select_joint_uncertainty_action(
        self,
        *,
        entropy_profile: EntropyRiskProfile,
        prepared_constraints: Dict[str, object],
        transition_violations: Sequence[ConstraintViolation],
        all_violations: Sequence[ConstraintViolation],
        delta: object,
        state: GenerationState,
        flags: Dict[str, bool],
        recent_action_events: Sequence[Dict[str, object]] | None = None,
        current_violation_count: int = 0,
        current_transition_violation_count: int = 0,
        local_patch_round: int = 0,
    ) -> str:
        if not bool(self.config.entropy_monitor.enable_entropy_monitor):
            return "do_nothing"
        if bool(flags.get("entropy_validation_only", False)):
            return "validation_boost" if entropy_profile.triggered_validation else "do_nothing"
        weight_template = str(self.config.entropy_monitor.joint_weight_template or "balanced").strip().lower()
        wu = float(self.config.entropy_monitor.joint_uncertainty_weight)
        ws = float(self.config.entropy_monitor.joint_symbolic_weight)
        wm = float(self.config.entropy_monitor.joint_memory_weight)
        entropy_profile.joint_weight_template_used = weight_template
        u = self._uncertainty_control_signal(entropy_profile)
        s = self._symbolic_pressure_signal(
            entropy_profile=entropy_profile,
            prepared_constraints=prepared_constraints,
            transition_violations=transition_violations,
            all_violations=all_violations,
        )
        m = self._memory_volatility_signal(delta=delta, state=state)
        entropy_profile.uncertainty_control_score = float(u)
        entropy_profile.symbolic_pressure_score = float(s)
        entropy_profile.memory_volatility_score = float(m)
        uncertainty_contribution = float(wu * float(u))
        symbolic_contribution = float(ws * float(s))
        memory_contribution = float(wm * float(m))
        entropy_profile.uncertainty_contribution = uncertainty_contribution
        entropy_profile.symbolic_contribution = symbolic_contribution
        entropy_profile.memory_contribution = memory_contribution
        joint = uncertainty_contribution + symbolic_contribution + memory_contribution
        joint = float(max(0.0, min(1.0, joint)))
        entropy_profile.joint_risk_score = joint
        local_failure_signal = self._joint_local_failure_signal(
            entropy_profile=entropy_profile,
            all_violations=all_violations,
            transition_violations=transition_violations,
            recent_action_events=recent_action_events,
            local_patch_round=int(local_patch_round),
        )
        persistent_steps = self._joint_persistent_high_risk_steps(
            current_joint_risk=joint,
            recent_action_events=recent_action_events,
            replan_threshold=float(self.config.entropy_monitor.joint_replan_threshold),
        )
        patch_failure_proxy = self._joint_patch_failure_proxy_score(
            current_violation_count=int(current_violation_count),
            current_transition_violation_count=int(current_transition_violation_count),
            recent_action_events=recent_action_events,
        )
        entropy_profile.joint_local_failure_signal = float(local_failure_signal)
        entropy_profile.joint_persistent_risk_steps = int(persistent_steps)
        entropy_profile.joint_patch_failure_proxy_score = float(patch_failure_proxy)

        validation_signal_u = float(
            max(0.0, max(float(entropy_profile.delta_uncertainty), float(entropy_profile.round_uncertainty_trend)))
        )
        entropy_profile.joint_validation_signal_u = float(validation_signal_u)
        entropy_profile.joint_validation_pre_gate_score = float(joint)
        entropy_profile.joint_patch_pre_gate_score = float(joint)
        entropy_profile.joint_replan_pre_gate_score = float(joint)
        validation_symbolic_ok = s >= float(self.config.entropy_monitor.joint_validation_min_symbolic_pressure)
        validation_uncertainty_ok = validation_signal_u >= float(
            self.config.entropy_monitor.joint_validation_min_uncertainty_trend
        )
        validation_threshold_reached = joint >= float(self.config.entropy_monitor.joint_validation_threshold)
        low_violation_guard = (
            int(current_violation_count) <= int(self.config.entropy_monitor.joint_validation_max_low_violation_guard)
            and int(current_transition_violation_count) == 0
            and joint < float(self.config.entropy_monitor.joint_patch_threshold)
        )
        dual_signal_satisfied = bool(validation_symbolic_ok and validation_uncertainty_ok)
        if bool(self.config.entropy_monitor.joint_validation_require_dual_signal):
            validation_gate = bool(validation_threshold_reached and dual_signal_satisfied and (not low_violation_guard))
        else:
            validation_gate = bool(
                validation_threshold_reached
                and (validation_symbolic_ok or validation_uncertainty_ok)
                and (not low_violation_guard)
            )
        validation_fail_reasons: List[str] = []
        if not validation_threshold_reached:
            validation_fail_reasons.append("threshold_not_reached")
        if bool(self.config.entropy_monitor.joint_validation_require_dual_signal):
            if not dual_signal_satisfied:
                validation_fail_reasons.append("dual_signal_failed")
        if not validation_symbolic_ok:
            validation_fail_reasons.append("symbolic_pressure_too_low")
        if not validation_uncertainty_ok:
            validation_fail_reasons.append("uncertainty_trend_too_low")
        if low_violation_guard:
            validation_fail_reasons.append("low_violation_guard_blocked")
        entropy_profile.joint_validation_gate_passed = bool(validation_gate)
        entropy_profile.joint_validation_threshold_reached = bool(validation_threshold_reached)
        entropy_profile.joint_validation_dual_signal_satisfied = bool(dual_signal_satisfied)
        entropy_profile.joint_validation_symbolic_ok = bool(validation_symbolic_ok)
        entropy_profile.joint_validation_uncertainty_ok = bool(validation_uncertainty_ok)
        entropy_profile.joint_validation_low_violation_guard_blocked = bool(low_violation_guard)
        entropy_profile.joint_validation_fail_reasons = list(dict.fromkeys(validation_fail_reasons))

        patch_threshold_reached = joint >= float(self.config.entropy_monitor.joint_patch_threshold)
        patch_symbolic_ok = s >= float(self.config.entropy_monitor.joint_patch_min_symbolic_pressure)
        patch_local_failure_ok = local_failure_signal >= float(
            self.config.entropy_monitor.joint_patch_min_local_failure_signal
        )
        patch_gate = bool(
            patch_threshold_reached
            and patch_symbolic_ok
            and patch_local_failure_ok
        )
        patch_fail_reasons: List[str] = []
        if not patch_threshold_reached:
            patch_fail_reasons.append("threshold_not_reached")
        if not patch_symbolic_ok:
            patch_fail_reasons.append("symbolic_pressure_too_low")
        if not patch_local_failure_ok:
            patch_fail_reasons.append("local_failure_signal_too_low")
        entropy_profile.joint_patch_gate_passed = bool(patch_gate)
        entropy_profile.joint_patch_threshold_reached = bool(patch_threshold_reached)
        entropy_profile.joint_patch_symbolic_ok = bool(patch_symbolic_ok)
        entropy_profile.joint_patch_local_failure_ok = bool(patch_local_failure_ok)
        entropy_profile.joint_patch_fail_reasons = list(dict.fromkeys(patch_fail_reasons))

        replan_threshold_reached = joint >= float(self.config.entropy_monitor.joint_replan_threshold)
        replan_persistence_ok = persistent_steps >= int(
            self.config.entropy_monitor.joint_replan_min_persistent_risk_steps
        )
        replan_patch_failure_ok = patch_failure_proxy > 0.0
        replan_requires_patch_failure = bool(self.config.entropy_monitor.joint_replan_requires_patch_failure)
        requires_patch_failure_blocked = bool(replan_requires_patch_failure and (not replan_patch_failure_ok))
        replan_gate = bool(
            replan_threshold_reached
            and replan_persistence_ok
        )
        if replan_requires_patch_failure:
            replan_gate = bool(replan_gate and replan_patch_failure_ok)
        replan_fail_reasons: List[str] = []
        if not replan_threshold_reached:
            replan_fail_reasons.append("threshold_not_reached")
        if not replan_persistence_ok:
            replan_fail_reasons.append("persistent_risk_steps_too_low")
        if replan_requires_patch_failure and not replan_patch_failure_ok:
            replan_fail_reasons.append("patch_failure_proxy_missing")
            replan_fail_reasons.append("blocked_by_requires_patch_failure")
        entropy_profile.joint_replan_gate_passed = bool(replan_gate)
        entropy_profile.joint_replan_threshold_reached = bool(replan_threshold_reached)
        entropy_profile.joint_replan_persistence_ok = bool(replan_persistence_ok)
        entropy_profile.joint_replan_patch_failure_ok = bool(replan_patch_failure_ok)
        entropy_profile.joint_replan_requires_patch_failure_blocked = bool(requires_patch_failure_blocked)
        entropy_profile.joint_replan_fail_reasons = list(dict.fromkeys(replan_fail_reasons))

        if replan_threshold_reached and replan_gate:
            return "replan"
        if joint >= float(self.config.entropy_monitor.joint_patch_escalation_threshold) and patch_gate:
            return "patch_plus_escalation"
        if patch_threshold_reached and patch_gate:
            return "patch"
        if validation_threshold_reached and validation_gate:
            return "validation_boost"
        return "do_nothing"

    def _joint_local_failure_signal(
        self,
        *,
        entropy_profile: EntropyRiskProfile,
        all_violations: Sequence[ConstraintViolation],
        transition_violations: Sequence[ConstraintViolation],
        recent_action_events: Sequence[Dict[str, object]] | None,
        local_patch_round: int,
    ) -> float:
        variance = max(0.0, float(entropy_profile.sentence_uncertainty_variance))
        delta_unc = max(0.0, float(entropy_profile.delta_uncertainty))
        trans_pressure = min(1.0, float(len(list(transition_violations)) / 3.0))
        hard_pressure = min(
            1.0,
            float(len([v for v in all_violations if bool(v.is_hard) or int(v.constraint_tier) <= 2]) / 3.0),
        )
        no_improve_recent = 0.0
        if recent_action_events:
            lookback = [e for e in list(recent_action_events)[-2:] if isinstance(e, dict)]
            if lookback:
                stagnant = 0
                for event in lookback:
                    if bool(event.get("action_requested", False)) and (
                        (not bool(event.get("improved_violations", False)))
                        and (not bool(event.get("improved_transition_violations", False)))
                    ):
                        stagnant += 1
                no_improve_recent = min(1.0, float(stagnant / max(1, len(lookback))))
        round_pressure = min(1.0, float(max(0, local_patch_round)) / 3.0)
        score = 0.3 * min(1.0, variance * 4.0) + 0.25 * min(1.0, delta_unc * 3.0) + 0.2 * trans_pressure
        score += 0.1 * hard_pressure + 0.1 * no_improve_recent + 0.05 * round_pressure
        return float(max(0.0, min(1.0, score)))

    def _joint_persistent_high_risk_steps(
        self,
        *,
        current_joint_risk: float,
        recent_action_events: Sequence[Dict[str, object]] | None,
        replan_threshold: float,
    ) -> int:
        count = 0
        if current_joint_risk >= float(replan_threshold):
            count += 1
        if recent_action_events:
            for event in reversed(list(recent_action_events)):
                if not isinstance(event, dict):
                    continue
                event_risk = float(event.get("joint_risk_score", 0.0))
                if event_risk >= float(replan_threshold):
                    count += 1
                    continue
                break
        return int(count)

    def _joint_patch_failure_proxy_score(
        self,
        *,
        current_violation_count: int,
        current_transition_violation_count: int,
        recent_action_events: Sequence[Dict[str, object]] | None,
    ) -> float:
        if not recent_action_events:
            return 0.0
        latest = None
        for event in reversed(list(recent_action_events)):
            if isinstance(event, dict) and str(event.get("selected_action", "")) != "do_nothing":
                latest = event
                break
        if latest is None:
            return 0.0
        previous_v = int(latest.get("after_violation_count", latest.get("before_violation_count", 0)))
        previous_tv = int(
            latest.get("after_transition_violation_count", latest.get("before_transition_violation_count", 0))
        )
        failed_improve = (
            int(current_violation_count) >= previous_v
            and int(current_transition_violation_count) >= previous_tv
        )
        blocked = bool(int(latest.get("action_blocked_count", 0)) > 0)
        base = 0.0
        if failed_improve:
            base += 0.7
        if blocked:
            base += 0.3
        return float(max(0.0, min(1.0, base)))

    def _uncertainty_control_signal(self, entropy_profile: EntropyRiskProfile) -> float:
        vals = [
            max(0.0, float(entropy_profile.scene_uncertainty_mean)),
            max(0.0, float(entropy_profile.delta_uncertainty)),
            max(0.0, float(entropy_profile.round_uncertainty_trend)),
            max(0.0, float(entropy_profile.critical_constraint_uncertainty_peak)),
        ]
        if entropy_profile.local_constraint_uncertainty:
            local_peak = max(float(v) for v in entropy_profile.local_constraint_uncertainty.values())
            vals.append(max(0.0, local_peak))
        return float(max(0.0, min(1.0, sum(vals) / max(1, len(vals)))))

    def _symbolic_pressure_signal(
        self,
        *,
        entropy_profile: EntropyRiskProfile,
        prepared_constraints: Dict[str, object],
        transition_violations: Sequence[ConstraintViolation],
        all_violations: Sequence[ConstraintViolation],
    ) -> float:
        weighted_items = prepared_constraints.get("weighted_constraints_tiered", [])
        critical_weight_count = 0
        if isinstance(weighted_items, list):
            for item in weighted_items:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("is_hard", False)) or int(item.get("tier", 9)) <= 2:
                    critical_weight_count += 1
        criticality = float(min(1.0, critical_weight_count / 6.0))
        conflict_candidates = prepared_constraints.get("conflict_candidates", [])
        conflict_pressure = float(min(1.0, len(conflict_candidates) / 6.0)) if isinstance(conflict_candidates, list) else 0.0
        transition_pressure = float(min(1.0, len(list(transition_violations)) / 3.0))
        hard_violation_pressure = float(
            min(
                1.0,
                len([v for v in all_violations if bool(v.is_hard) or int(v.constraint_tier) <= 2]) / 3.0,
            )
        )
        linked_pressure = float(min(1.0, len(entropy_profile.linked_constraint_ids) / 8.0))
        score = (
            0.28 * criticality
            + 0.2 * conflict_pressure
            + 0.24 * transition_pressure
            + 0.18 * hard_violation_pressure
            + 0.1 * linked_pressure
        )
        return float(max(0.0, min(1.0, score)))

    def _memory_volatility_signal(self, *, delta: object, state: GenerationState) -> float:
        new_events = len(getattr(delta, "new_events", []) or [])
        updated_entities = len(getattr(delta, "updated_entities", []) or [])
        relation_updates = len(getattr(delta, "new_relations", {}) or {})
        fact_updates = len(getattr(delta, "new_facts", {}) or {})
        recent_revisions = min(6, len(getattr(state.dynamic_memory, "revision_records", [])[-6:]))
        activity = float(new_events + updated_entities + relation_updates + fact_updates + recent_revisions)
        return float(max(0.0, min(1.0, activity / 14.0)))

    def _entropy_violation_overlap(
        self,
        *,
        entropy_profile: EntropyRiskProfile,
        violations: Sequence[ConstraintViolation],
    ) -> float:
        high = set(entropy_profile.high_risk_sentence_ids)
        if not high:
            return 0.0
        vio_sentences: set[str] = set()
        for item in violations:
            for anchor in item.anchors:
                for sid in anchor.sentence_ids:
                    vio_sentences.add(sid)
        if not vio_sentences:
            return 0.0
        return float(len(high.intersection(vio_sentences)) / max(1, len(high)))

    def _write_diagnostics_artifacts(
        self,
        prompt_id: str,
        variant_name: str,
        scene_records: List[Dict[str, object]],
        summary: Dict[str, object],
    ) -> None:
        cfg = self.config.diagnostics
        root = Path(cfg.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        base = f"{variant_name}_{prompt_id}"
        if cfg.export_scene_jsonl:
            path = root / f"{base}_scenes.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in scene_records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if cfg.export_story_json:
            path = root / f"{base}_summary.json"
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if cfg.export_summary_csv:
            path = root / f"{base}_summary.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)

    def _build_preservation_violations(
        self,
        scene: ScenePlan,
        metadata: Dict[str, object],
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        protected_pass = metadata.get("protected_integrity_pass")
        if protected_pass is False:
            regressions = metadata.get("protected_regressions", [])
            message = "Protected sentence integrity regression detected after patch."
            if regressions:
                message = f"{message} regressions={str(regressions)[:200]}"
            violations.append(
                ConstraintViolation(
                    rule_type="preservation_regression",
                    message=message,
                    severity="error",
                    facet="factual_detail",
                    related_ids=[scene.scene_id],
                    repair_hint="Patch target sentences without changing protected sentence facts/entities.",
                    repair_scope="paragraph",
                    patchable=True,
                )
            )
        spillover = metadata.get("spillover_count")
        if isinstance(spillover, int) and spillover > 0:
            violations.append(
                ConstraintViolation(
                    rule_type="preservation_regression",
                    message=f"Patch spillover detected in neighboring sentences: {spillover}",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[scene.scene_id],
                    repair_hint="Adjust local neighboring sentences to restore coreference/causal continuity.",
                    repair_scope="paragraph",
                    patchable=True,
                )
            )
        stability = metadata.get("stability_score")
        if isinstance(stability, (int, float)) and float(stability) < float(
            self.config.generation_controls.min_stability_score
        ):
            violations.append(
                ConstraintViolation(
                    rule_type="preservation_regression",
                    message=(
                        "Patch stability too low: "
                        f"{float(stability):.3f} < {float(self.config.generation_controls.min_stability_score):.3f}"
                    ),
                    severity="warning",
                    facet="narrative_style",
                    related_ids=[scene.scene_id],
                    repair_hint="Keep patch semantically close to pre-patch scene while fixing violations.",
                    repair_scope="paragraph",
                    patchable=True,
                )
            )
        return violations

    def _collect_high_conf_violation_lines(self, violations: Sequence[ConstraintViolation]) -> List[str]:
        lines: List[str] = []
        for violation in violations:
            conf = 0.0
            if violation.anchors:
                conf = max(
                    float(getattr(anchor, "confidence_score", getattr(anchor, "grounding_confidence", 0.0)))
                    for anchor in violation.anchors
                )
            elif violation.severity.lower() == "error":
                conf = 0.8
            if conf < self.config.generation_controls.low_confidence_anchor_threshold:
                continue
            lines.append(f"[{violation.rule_type}/{violation.severity}] {violation.message}")
        return lines[:8]

    def _select_adaptive_repair_scope(
        self,
        violations: Sequence[ConstraintViolation],
        local_patch_round: int,
        sentence_patch_round: int,
        paragraph_patch_round: int,
        max_sentence_patch_rounds: int,
        max_paragraph_patch_rounds: int,
    ) -> Tuple[str, bool]:
        if not self.config.generation_controls.enable_facet_adaptive_repair:
            if sentence_patch_round < max_sentence_patch_rounds:
                return "sentence", False
            if paragraph_patch_round < max_paragraph_patch_rounds:
                return "paragraph", False
            return "", False

        severe_fatal = any(v.fatal for v in violations)
        hard_replan = any(v.needs_replan for v in violations)
        if severe_fatal or hard_replan:
            return "", True

        errors = [v for v in violations if v.severity.lower() == "error"]
        error_count = len(errors)
        facets = {v.facet for v in violations if v.facet}
        hard_facets = {"timeline_plot", "world_setting"}
        detail_facets = {"characterization", "factual_detail", "narrative_style"}
        has_hard_facet = bool(facets.intersection(hard_facets))
        only_detail_facets = bool(facets) and facets.issubset(detail_facets)
        repeated_round = local_patch_round >= int(self.config.generation_controls.adaptive_replan_round_threshold)
        too_many_errors = error_count >= int(self.config.generation_controls.adaptive_replan_error_threshold)
        prefer_sentence_repair_first = bool(self.config.generation_controls.prefer_sentence_repair_first)

        if prefer_sentence_repair_first:
            if sentence_patch_round < max_sentence_patch_rounds:
                return "sentence", False
            if paragraph_patch_round < max_paragraph_patch_rounds:
                return "paragraph", False
            if too_many_errors and repeated_round and has_hard_facet:
                return "", True
            return "", False

        if too_many_errors and repeated_round and has_hard_facet:
            return "", True
        if too_many_errors and paragraph_patch_round < max_paragraph_patch_rounds:
            return "paragraph", False
        if has_hard_facet and paragraph_patch_round < max_paragraph_patch_rounds:
            return "paragraph", False
        if only_detail_facets and sentence_patch_round < max_sentence_patch_rounds:
            return "sentence", False
        if sentence_patch_round < max_sentence_patch_rounds:
            return "sentence", False
        if paragraph_patch_round < max_paragraph_patch_rounds:
            return "paragraph", False
        return "", False

    def _build_report_from_violations(
        self,
        violations: List[ConstraintViolation],
        scene_id: str,
    ) -> ConsistencyReport:
        symbolic_findings = [
            SymbolicFinding(
                rule_type=v.rule_type,
                message=v.message,
                severity=v.severity,
                related_ids=v.related_ids,
                facet=v.facet,
            )
            for v in violations
        ]
        blocking = [v for v in violations if v.severity.lower() == "error"]
        severity = "error" if blocking else ("warning" if violations else "info")
        rules = sorted(set(v.rule_type for v in violations))
        facets = sorted(set(v.facet for v in violations if v.facet))
        anchors = [anchor for v in violations for anchor in v.anchors]
        needs_replan = any(v.needs_replan for v in violations)
        fatal = any(v.fatal for v in violations)
        scopes = sorted(set(v.repair_scope for v in violations))
        repair_strategy = "patch_sentence"
        if needs_replan:
            repair_strategy = "needs_replan"
        elif "paragraph" in scopes:
            repair_strategy = "patch_paragraph"
        elif fatal or "scene" in scopes:
            repair_strategy = "regenerate_chunk"
        return ConsistencyReport(
            is_consistent=len(blocking) == 0,
            violation_types=rules,
            violated_facets=facets,
            violated_rules=rules,
            messages=[v.message for v in violations],
            symbolic_findings=symbolic_findings,
            neural_findings=[],
            suggested_action="accept" if not violations else "revise",
            repair_hints=[v.repair_hint for v in violations if v.repair_hint],
            conflict_spans=[],
            conflict_slots=sorted(set(x for v in violations for x in v.related_ids)),
            repair_target=facets[0] if len(facets) == 1 else ("multi_facet" if facets else "none"),
            repair_strategy=repair_strategy,
            severity=severity,
            violation_anchors=anchors,
            needs_replan=needs_replan,
            fatal=fatal,
        )
