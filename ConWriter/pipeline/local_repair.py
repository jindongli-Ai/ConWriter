"""Local repair module for scene-level inconsistency fixing."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Sequence, Set, Tuple
from urllib import error, request

from ConWriter.config.schema import LLMConfig
from ConWriter.reasoning.scene_alignment import split_scene_into_units
from ConWriter.utils.state_grounding import build_state_grounding_bundle
from ConWriter.utils.types import (
    ConstraintViolation,
    DynamicMemory,
    PatchPlan,
    SceneDraft,
    SceneExtraction,
    SentencePatch,
    ScenePlan,
    SentenceUnit,
    StaticMemory,
)


class LocalRepairer:
    """Rewrite only the current scene based on constraint violations."""

    def __init__(self, llm_config: LLMConfig, logger: logging.Logger | None = None):
        self.llm_config = llm_config
        self.logger = logger or logging.getLogger("ConWriter.local_repair")

    def repair_scene(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        attempt: int,
    ) -> SceneDraft:
        """Return repaired local scene text (not global rewrite)."""
        if self.llm_config.enabled:
            repaired = self._repair_with_llm(
                scene_plan=scene_plan,
                scene_text=scene_text,
                violations=violations,
                static_memory=static_memory,
                dynamic_memory=dynamic_memory,
            )
            if repaired.strip():
                return SceneDraft(
                    scene_id=scene_plan.scene_id,
                    chapter_id=scene_plan.chapter_id,
                    text=repaired.strip(),
                    attempt=attempt,
                    metadata={"source": "llm_repair"},
                )

        repaired = self._repair_with_rules(
            scene_text=scene_text,
            scene_plan=scene_plan,
            violations=violations,
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
        )
        return SceneDraft(
            scene_id=scene_plan.scene_id,
            chapter_id=scene_plan.chapter_id,
            text=repaired,
            attempt=attempt,
            metadata={"source": "rule_repair"},
        )

    def repair_with_patch_plan(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        extraction: SceneExtraction,
        patch_plan: PatchPlan,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        attempt: int,
        scope: str = "sentence",
        retry_context: Dict[str, object] | None = None,
        generation_control_context: Dict[str, object] | None = None,
    ) -> SceneDraft:
        """Patch-oriented local repair with sentence/paragraph/scene fallback."""
        allow_scene_rewrite_fallback = bool(
            (generation_control_context or {}).get("allow_scene_rewrite_fallback", False)
            or (retry_context or {}).get("allow_scene_rewrite_fallback", False)
        )
        if scope == "scene" and not allow_scene_rewrite_fallback:
            # Keep patch-first behavior: do not jump to scene rewrite unless explicitly enabled.
            scope = "paragraph"
        if scope == "scene":
            regen = self.repair_scene(
                scene_plan=scene_plan,
                scene_text=scene_text,
                violations=violations,
                static_memory=static_memory,
                dynamic_memory=dynamic_memory,
                attempt=attempt,
            )
            regen.metadata.update(
                {
                    "repair_scope": "scene",
                    "unchanged_ratio": 0.0,
                    "patch_plan_id": patch_plan.plan_id,
                    "applied_patches": [],
                }
            )
            return regen

        rewrite_payload = self._build_conflict_rewrite_payload(
            scene_plan=scene_plan,
            violations=violations,
            patch_plan=patch_plan,
            generation_control_context=generation_control_context,
        )
        if isinstance(retry_context, dict) and retry_context:
            rewrite_payload["retry_context"] = dict(retry_context)
        retry_expansion_meta = self._resolve_minimal_scope_expansion(
            scope=scope,
            rewrite_payload=rewrite_payload,
            retry_context=dict(retry_context or {}),
            patch_plan=patch_plan,
            units=extraction.sentences if extraction.sentences else [],
        )
        if bool(retry_expansion_meta.get("scope_expansion_triggered", False)):
            scope = str(retry_expansion_meta.get("expanded_scope", scope))
            rewrite_payload["scope_expansion_meta"] = dict(retry_expansion_meta)
            rewrite_payload["expanded_target_sentence_ids"] = list(
                retry_expansion_meta.get("expanded_target_sentence_ids", [])
            )
            rewrite_payload["expanded_local_window"] = dict(
                retry_expansion_meta.get("expanded_local_window", {})
            )
            patch_plan = self._with_expanded_patch_targets(
                patch_plan=patch_plan,
                expanded_target_sentence_ids=list(retry_expansion_meta.get("expanded_target_sentence_ids", [])),
            )
        rewrite_payload["generation_control_context"] = dict(generation_control_context or {})
        if self._requires_transition_coherence_escalation(
            rewrite_payload,
            scope,
            retry_context=dict(retry_context or {}),
        ):
            scope = "paragraph"

        original_units = list(extraction.sentences) if extraction.sentences else split_scene_into_units(scene_text, scene_plan.scene_id)
        if not original_units:
            original_units = split_scene_into_units(scene_text, scene_plan.scene_id)
        working_units: List[SentenceUnit] = [
            SentenceUnit(
                sentence_id=unit.sentence_id,
                text=unit.text,
                char_start=unit.char_start,
                char_end=unit.char_end,
                paragraph_id=unit.paragraph_id,
                source_scene_id=unit.source_scene_id,
            )
            for unit in original_units
        ]
        applied_patches: List[Dict[str, object]] = []
        paragraph_like_scope = {"paragraph", "sentence_with_neighbor", "local_paragraph_window"}
        if scope in paragraph_like_scope:
            block_window = 1 if scope == "paragraph" else 0
            working_units, paragraph_meta = self._patch_paragraph_units(
                scene_plan=scene_plan,
                units=working_units,
                patch_plan=patch_plan,
                violations=violations,
                static_memory=static_memory,
                dynamic_memory=dynamic_memory,
                rewrite_payload=rewrite_payload,
                block_window=block_window,
            )
            applied_patches.append(paragraph_meta)
        else:
            for patch in patch_plan.patch_sequence:
                if not patch.target_sentence_ids:
                    continue
                working_units, patch_meta = self._apply_sentence_patch(
                    scene_plan=scene_plan,
                    units=working_units,
                    patch=patch,
                    patch_plan=patch_plan,
                    violations=violations,
                    static_memory=static_memory,
                    dynamic_memory=dynamic_memory,
                    rewrite_payload=rewrite_payload,
                )
                applied_patches.append(patch_meta)

        patched_text = self._compose_scene_text(working_units)
        unchanged_ratio = self._unchanged_ratio(original_units, working_units)
        stability_score = self._stability_score(
            self._compose_scene_text(original_units),
            patched_text,
        )
        preservation = self._run_preservation_audit(
            original_units=original_units,
            patched_units=working_units,
            target_sentence_ids=set(patch_plan.target_sentence_ids),
            protected_sentence_ids=set(patch_plan.protected_sentence_ids),
        )
        change_stats = self._compute_patch_change_stats(
            original_units=original_units,
            patched_units=working_units,
            target_sentence_ids=set(patch_plan.target_sentence_ids),
            violated_sentence_ids=set(str(x) for x in list(rewrite_payload.get("violated_sentence_ids", []))),
        )
        return SceneDraft(
            scene_id=scene_plan.scene_id,
            chapter_id=scene_plan.chapter_id,
            text=patched_text,
            attempt=attempt,
            metadata={
                "source": f"{scope}_patch",
                "repair_scope": scope,
                "patch_plan_id": patch_plan.plan_id,
                "target_sentence_ids": list(patch_plan.target_sentence_ids),
                "protected_sentence_ids": list(patch_plan.protected_sentence_ids),
                "applied_patches": applied_patches,
                "unchanged_ratio": unchanged_ratio,
                "patch_scope_size": len(patch_plan.target_sentence_ids),
                "protected_integrity_pass": preservation["protected_integrity_pass"],
                "protected_regressions": preservation["protected_regressions"],
                "spillover_count": preservation["spillover_count"],
                "preservation_details": preservation["details"],
                "stability_score": stability_score,
                "rewrite_conflict_type": str(rewrite_payload.get("rewrite_conflict_type", "unknown")),
                "rewrite_target_scope": (
                    "multi_sentence"
                    if (scope == "paragraph" and len(patch_plan.target_sentence_ids) > 1)
                    else str(scope)
                ),
                "rewrite_required_state_changes": list(rewrite_payload.get("required_state_changes", [])),
                "rewrite_forbidden_state_changes": list(rewrite_payload.get("forbidden_state_changes", [])),
                "rewrite_transition_violations": list(rewrite_payload.get("transition_violation_messages", [])),
                "rewrite_execution_spec_violations": list(rewrite_payload.get("execution_spec_messages", [])),
                "rewrite_violated_sentence_ids": list(rewrite_payload.get("violated_sentence_ids", [])),
                "rewrite_violated_constraint_ids": list(rewrite_payload.get("violated_constraint_ids", [])),
                "rewrite_conflict_spans": list(rewrite_payload.get("conflict_spans", [])),
                "rewrite_targets_execution_spec_conflict": bool(
                    rewrite_payload.get("rewrite_targets_execution_spec_conflict", False)
                ),
                "rewrite_targets_required_state_change": bool(
                    rewrite_payload.get("rewrite_targets_required_state_change", False)
                ),
                "rewrite_targets_transition_conflict": bool(
                    rewrite_payload.get("rewrite_targets_transition_conflict", False)
                ),
                "rewrite_targets_operator_post_state_conflict": bool(
                    rewrite_payload.get("rewrite_targets_operator_post_state_conflict", False)
                ),
                "rewrite_operator_required_post_states": list(
                    rewrite_payload.get("operator_required_post_states", [])
                ),
                "rewrite_canonical_required_states": list(rewrite_payload.get("canonical_required_states", [])),
                "rewrite_canonical_forbidden_states": list(rewrite_payload.get("canonical_forbidden_states", [])),
                "rewrite_canonical_operator_post_states": list(
                    rewrite_payload.get("canonical_operator_post_states", [])
                ),
                "rewrite_required_state_groundings": list(rewrite_payload.get("required_state_groundings", [])),
                "rewrite_forbidden_state_groundings": list(rewrite_payload.get("forbidden_state_groundings", [])),
                "rewrite_operator_post_state_groundings": list(
                    rewrite_payload.get("operator_post_state_groundings", [])
                ),
                "rewrite_transition_target_state_groundings": list(
                    rewrite_payload.get("transition_target_state_groundings", [])
                ),
                "rewrite_transition_grounded_cues": list(rewrite_payload.get("transition_grounded_cues", [])),
                "rewrite_transition_coherence_guidance": list(
                    rewrite_payload.get("transition_coherence_guidance", [])
                ),
                "rewrite_retry_context": dict(rewrite_payload.get("retry_context", {})),
                "rewrite_memory_binding_mode": str(rewrite_payload.get("memory_binding_mode", "normal_binding")),
                "rewrite_generation_control_mode": str(
                    rewrite_payload.get("generation_control_mode", "normal_generation")
                ),
                "rewrite_binding_decision_reasons": list(rewrite_payload.get("binding_decision_reasons", [])),
                "rewrite_strengthened_memory_blocks": list(rewrite_payload.get("strengthened_memory_blocks", [])),
                "rewrite_strengthened_constraints": list(rewrite_payload.get("strengthened_constraints", [])),
                "rewrite_generation_control_context": dict(rewrite_payload.get("generation_control_context", {})),
                "scope_expansion_triggered": bool(retry_expansion_meta.get("scope_expansion_triggered", False)),
                "original_scope": str(retry_expansion_meta.get("original_scope", scope)),
                "expanded_scope": str(retry_expansion_meta.get("expanded_scope", scope)),
                "expanded_target_sentence_ids": list(retry_expansion_meta.get("expanded_target_sentence_ids", [])),
                "expanded_local_window": dict(retry_expansion_meta.get("expanded_local_window", {})),
                "unresolved_slot_count_before": int(retry_expansion_meta.get("unresolved_slot_count_before", 0)),
                "unresolved_slot_count_after": int(retry_expansion_meta.get("unresolved_slot_count_after", 0)),
                "checklist_items_fixed_by_expansion": list(
                    retry_expansion_meta.get("checklist_items_fixed_by_expansion", [])
                ),
                "still_unresolved_after_expansion": list(
                    retry_expansion_meta.get("still_unresolved_after_expansion", [])
                ),
                "expansion_preserved_satisfied_items": bool(
                    retry_expansion_meta.get("expansion_preserved_satisfied_items", True)
                ),
                "scope_expansion_effective": bool(retry_expansion_meta.get("scope_expansion_effective", False)),
                "changed_sentence_count": int(change_stats.get("changed_sentence_count", 0)),
                "changed_sentence_ids": list(change_stats.get("changed_sentence_ids", [])),
                "added_sentence_ids": list(change_stats.get("added_sentence_ids", [])),
                "removed_sentence_ids": list(change_stats.get("removed_sentence_ids", [])),
                "changed_target_sentence_count": int(change_stats.get("changed_target_sentence_count", 0)),
                "changed_non_target_sentence_count": int(
                    change_stats.get("changed_non_target_sentence_count", 0)
                ),
                "changed_non_target_sentence_ids": list(
                    change_stats.get("changed_non_target_sentence_ids", [])
                ),
                "target_sentence_touched_ratio": float(change_stats.get("target_sentence_touched_ratio", 0.0)),
                "violation_context_touched": bool(change_stats.get("violation_context_touched", False)),
            },
        )

    def _with_expanded_patch_targets(
        self,
        *,
        patch_plan: PatchPlan,
        expanded_target_sentence_ids: Sequence[str],
    ) -> PatchPlan:
        ids = [str(x).strip() for x in list(expanded_target_sentence_ids or []) if str(x).strip()]
        if not ids:
            return patch_plan
        patch_plan.target_sentence_ids = list(dict.fromkeys(ids))
        for patch in patch_plan.patch_sequence:
            if not patch.target_sentence_ids:
                patch.target_sentence_ids = list(dict.fromkeys(ids))
                continue
            merged = list(dict.fromkeys([str(x) for x in list(patch.target_sentence_ids) + ids if str(x).strip()]))
            patch.target_sentence_ids = merged
        return patch_plan

    def _resolve_minimal_scope_expansion(
        self,
        *,
        scope: str,
        rewrite_payload: Dict[str, object],
        retry_context: Dict[str, object],
        patch_plan: PatchPlan,
        units: Sequence[SentenceUnit],
    ) -> Dict[str, object]:
        out: Dict[str, object] = {
            "scope_expansion_triggered": False,
            "original_scope": str(scope),
            "expanded_scope": str(scope),
            "expanded_target_sentence_ids": list(patch_plan.target_sentence_ids),
            "expanded_local_window": {},
            "unresolved_slot_count_before": 0,
            "unresolved_slot_count_after": 0,
            "checklist_items_fixed_by_expansion": [],
            "still_unresolved_after_expansion": [],
            "expansion_preserved_satisfied_items": True,
            "scope_expansion_effective": False,
        }
        if not retry_context:
            return out
        conflict_type = str(rewrite_payload.get("rewrite_conflict_type", "unknown"))
        if conflict_type not in {
            "mixed_conflict",
            "execution_spec_conflict",
            "operator_post_state_conflict",
            "transition_conflict",
        }:
            return out
        hits_context = bool(getattr(patch_plan, "patch_target_hits_violation_context", False))
        if not hits_context:
            return out
        unresolved_required = list(retry_context.get("unresolved_required_states", []) or [])
        unresolved_forbidden = list(retry_context.get("unresolved_forbidden_states", []) or [])
        unresolved_operator = list(retry_context.get("unresolved_operator_post_states", []) or [])
        unresolved_count = len(unresolved_required) + len(unresolved_forbidden) + len(unresolved_operator)
        unresolved_groups = int(bool(unresolved_required)) + int(bool(unresolved_forbidden)) + int(bool(unresolved_operator))
        multi_combo = (
            (bool(unresolved_required) and bool(unresolved_forbidden))
            or (bool(unresolved_required) and bool(unresolved_operator))
            or (bool(unresolved_forbidden) and bool(unresolved_operator))
        )
        should_expand = bool(
            (unresolved_count > 1 or unresolved_groups > 1 or multi_combo or conflict_type == "mixed_conflict")
            and scope in {"sentence", "paragraph"}
        )
        out["unresolved_slot_count_before"] = int(unresolved_count)
        if not should_expand:
            out["still_unresolved_after_expansion"] = list(retry_context.get("still_unresolved_state_items", []))
            out["unresolved_slot_count_after"] = int(unresolved_count)
            return out
        id_to_idx = {u.sentence_id: idx for idx, u in enumerate(list(units or []))}
        target_ids = [sid for sid in list(patch_plan.target_sentence_ids) if sid in id_to_idx]
        if not target_ids:
            out["still_unresolved_after_expansion"] = list(retry_context.get("still_unresolved_state_items", []))
            out["unresolved_slot_count_after"] = int(unresolved_count)
            return out
        target_idxs = sorted({id_to_idx[sid] for sid in target_ids})
        if scope == "sentence":
            expanded_idxs: set[int] = set(target_idxs)
            for idx in target_idxs:
                if idx - 1 >= 0:
                    expanded_idxs.add(idx - 1)
                if idx + 1 < len(units):
                    expanded_idxs.add(idx + 1)
            expanded_ids = [units[idx].sentence_id for idx in sorted(expanded_idxs)]
            out.update(
                {
                    "scope_expansion_triggered": True,
                    "expanded_scope": "sentence_with_neighbor",
                    "expanded_target_sentence_ids": list(expanded_ids),
                    "expanded_local_window": {
                        "window_type": "neighbor",
                        "center_sentence_ids": list(target_ids),
                        "expanded_sentence_ids": list(expanded_ids),
                    },
                    "still_unresolved_after_expansion": list(retry_context.get("still_unresolved_state_items", [])),
                    "unresolved_slot_count_after": int(unresolved_count),
                }
            )
            return out
        blocks = self._build_local_blocks(target_idxs, len(units), window=1)
        if not blocks:
            out["still_unresolved_after_expansion"] = list(retry_context.get("still_unresolved_state_items", []))
            out["unresolved_slot_count_after"] = int(unresolved_count)
            return out
        block_start = min(x[0] for x in blocks)
        block_end = max(x[1] for x in blocks)
        expanded_ids = [units[idx].sentence_id for idx in range(block_start, block_end + 1)]
        out.update(
            {
                "scope_expansion_triggered": True,
                "expanded_scope": "local_paragraph_window",
                "expanded_target_sentence_ids": list(expanded_ids),
                "expanded_local_window": {
                    "window_type": "paragraph_local",
                    "range": [int(block_start), int(block_end)],
                    "expanded_sentence_ids": list(expanded_ids),
                },
                "still_unresolved_after_expansion": list(retry_context.get("still_unresolved_state_items", [])),
                "unresolved_slot_count_after": int(unresolved_count),
            }
        )
        return out

    def _apply_sentence_patch(
        self,
        scene_plan: ScenePlan,
        units: List[SentenceUnit],
        patch: SentencePatch,
        patch_plan: PatchPlan,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        rewrite_payload: Dict[str, object] | None = None,
    ) -> tuple[List[SentenceUnit], Dict[str, object]]:
        by_id = {unit.sentence_id: idx for idx, unit in enumerate(units)}
        target_ids = [sid for sid in patch.target_sentence_ids if sid in by_id]
        if not target_ids:
            return units, {"patch_id": patch.patch_id, "applied": False, "reason": "no_valid_targets"}

        main_id = target_ids[0]
        main_idx = by_id[main_id]
        target = units[main_idx]
        previous_text = units[main_idx - 1].text if main_idx > 0 else ""
        next_text = units[main_idx + 1].text if main_idx + 1 < len(units) else ""

        violations_for_patch = [
            v for v in violations if any(anchor.anchor_id in patch.linked_violation_ids for anchor in v.anchors)
        ] or violations
        rewritten = self._rule_based_sentence_patch(
            sentence_text=target.text,
            patch=patch,
            violations=violations_for_patch,
            scene_plan=scene_plan,
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
            rewrite_payload=rewrite_payload,
        )
        if self.llm_config.enabled:
            llm_text = self._patch_sentence_with_llm(
                scene_plan=scene_plan,
                target_sentence=target.text,
                previous_sentence=previous_text,
                next_sentence=next_text,
                patch=patch,
                patch_plan=patch_plan,
                violations=violations_for_patch,
                static_memory=static_memory,
                dynamic_memory=dynamic_memory,
                rewrite_payload=rewrite_payload,
            )
            if llm_text.strip():
                rewritten = llm_text.strip()

        updated_units = list(units)
        op = patch.op_type.lower()
        if op == "delete":
            updated_units.pop(main_idx)
        elif op == "insert_after":
            inserted = SentenceUnit(
                sentence_id=f"{target.sentence_id}_ins",
                text=(rewritten or patch.new_text or "Then the scene satisfies the missing constraint.").strip(),
                char_start=target.char_end,
                char_end=target.char_end,
                paragraph_id=target.paragraph_id,
                source_scene_id=target.source_scene_id,
            )
            updated_units.insert(main_idx + 1, inserted)
        else:
            target.text = (rewritten or patch.new_text or target.text).strip()
            updated_units[main_idx] = target

        return updated_units, {
            "patch_id": patch.patch_id,
            "applied": True,
            "op_type": patch.op_type,
            "target_sentence_ids": list(target_ids),
            "rewritten_text": (rewritten or patch.new_text).strip(),
            "rewrite_conflict_type": str((rewrite_payload or {}).get("rewrite_conflict_type", "unknown")),
            "rewrite_required_state_changes": list((rewrite_payload or {}).get("required_state_changes", [])),
            "rewrite_forbidden_state_changes": list((rewrite_payload or {}).get("forbidden_state_changes", [])),
            "rewrite_violated_sentence_ids": list((rewrite_payload or {}).get("violated_sentence_ids", [])),
            "rewrite_violated_constraint_ids": list((rewrite_payload or {}).get("violated_constraint_ids", [])),
            "rewrite_conflict_spans": list((rewrite_payload or {}).get("conflict_spans", [])),
            "rewrite_targets_execution_spec_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_execution_spec_conflict", False)
            ),
            "rewrite_targets_required_state_change": bool(
                (rewrite_payload or {}).get("rewrite_targets_required_state_change", False)
            ),
            "rewrite_targets_transition_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_transition_conflict", False)
            ),
            "rewrite_targets_operator_post_state_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_operator_post_state_conflict", False)
            ),
            "rewrite_operator_required_post_states": list(
                (rewrite_payload or {}).get("operator_required_post_states", [])
            ),
            "rewrite_canonical_required_states": list((rewrite_payload or {}).get("canonical_required_states", [])),
            "rewrite_canonical_forbidden_states": list((rewrite_payload or {}).get("canonical_forbidden_states", [])),
            "rewrite_canonical_operator_post_states": list(
                (rewrite_payload or {}).get("canonical_operator_post_states", [])
            ),
            "rewrite_required_state_groundings": list((rewrite_payload or {}).get("required_state_groundings", [])),
            "rewrite_forbidden_state_groundings": list((rewrite_payload or {}).get("forbidden_state_groundings", [])),
            "rewrite_operator_post_state_groundings": list(
                (rewrite_payload or {}).get("operator_post_state_groundings", [])
            ),
            "rewrite_transition_target_state_groundings": list(
                (rewrite_payload or {}).get("transition_target_state_groundings", [])
            ),
            "rewrite_transition_grounded_cues": list((rewrite_payload or {}).get("transition_grounded_cues", [])),
            "rewrite_transition_coherence_guidance": list(
                (rewrite_payload or {}).get("transition_coherence_guidance", [])
            ),
            "rewrite_retry_context": dict((rewrite_payload or {}).get("retry_context", {})),
            "rewrite_memory_binding_mode": str(
                (rewrite_payload or {}).get("memory_binding_mode", "normal_binding")
            ),
            "rewrite_generation_control_mode": str(
                (rewrite_payload or {}).get("generation_control_mode", "normal_generation")
            ),
            "rewrite_binding_decision_reasons": list(
                (rewrite_payload or {}).get("binding_decision_reasons", [])
            ),
            "rewrite_strengthened_memory_blocks": list(
                (rewrite_payload or {}).get("strengthened_memory_blocks", [])
            ),
            "rewrite_strengthened_constraints": list(
                (rewrite_payload or {}).get("strengthened_constraints", [])
            ),
            "rewrite_generation_control_context": dict(
                (rewrite_payload or {}).get("generation_control_context", {})
            ),
            "scope_expansion_triggered": bool((rewrite_payload or {}).get("scope_expansion_meta", {}).get("scope_expansion_triggered", False)),
            "original_scope": str((rewrite_payload or {}).get("scope_expansion_meta", {}).get("original_scope", "")),
            "expanded_scope": str((rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_scope", "")),
            "expanded_target_sentence_ids": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_target_sentence_ids", [])
            ),
            "expanded_local_window": dict(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_local_window", {})
            ),
            "unresolved_slot_count_before": int(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("unresolved_slot_count_before", 0) or 0
            ),
            "unresolved_slot_count_after": int(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("unresolved_slot_count_after", 0) or 0
            ),
            "checklist_items_fixed_by_expansion": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("checklist_items_fixed_by_expansion", [])
            ),
            "still_unresolved_after_expansion": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("still_unresolved_after_expansion", [])
            ),
            "expansion_preserved_satisfied_items": bool(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expansion_preserved_satisfied_items", True)
            ),
            "scope_expansion_effective": bool(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("scope_expansion_effective", False)
            ),
        }

    def _patch_paragraph_units(
        self,
        scene_plan: ScenePlan,
        units: List[SentenceUnit],
        patch_plan: PatchPlan,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        rewrite_payload: Dict[str, object] | None = None,
        block_window: int = 1,
    ) -> Tuple[List[SentenceUnit], Dict[str, object]]:
        target_ids = set(patch_plan.target_sentence_ids)
        if not target_ids:
            return units, {"scope": "paragraph", "applied": False, "reason": "no_targets"}

        id_to_idx = {unit.sentence_id: idx for idx, unit in enumerate(units)}
        target_idxs = sorted({id_to_idx[sid] for sid in target_ids if sid in id_to_idx})
        if not target_idxs:
            return units, {"scope": "paragraph", "applied": False, "reason": "targets_not_found"}

        blocks = self._build_local_blocks(target_idxs, len(units), window=max(0, int(block_window)))
        updated_units = list(units)
        rewritten_blocks: List[Dict[str, object]] = []
        for block_start, block_end in blocks:
            idxs = list(range(block_start, block_end + 1))
            block_text = " ".join(updated_units[idx].text for idx in idxs).strip()
            rewritten = block_text
            for patch in patch_plan.patch_sequence:
                rewritten = self._rule_based_sentence_patch(
                    sentence_text=rewritten,
                    patch=patch,
                    violations=violations,
                    scene_plan=scene_plan,
                    static_memory=static_memory,
                    dynamic_memory=dynamic_memory,
                    rewrite_payload=rewrite_payload,
                )
            if self.llm_config.enabled:
                prev_text = updated_units[block_start - 1].text if block_start > 0 else ""
                next_text = updated_units[block_end + 1].text if block_end + 1 < len(updated_units) else ""
                llm_rewrite = self._patch_paragraph_with_llm(
                    scene_plan=scene_plan,
                    paragraph_text=block_text,
                    patch_plan=patch_plan,
                    violations=violations,
                    static_memory=static_memory,
                    dynamic_memory=dynamic_memory,
                    previous_sentence=prev_text,
                    next_sentence=next_text,
                    rewrite_payload=rewrite_payload,
                )
                if llm_rewrite.strip():
                    rewritten = llm_rewrite.strip()

            replacement = self._split_sentences(rewritten) or [rewritten]
            original_block_len = len(idxs)
            if len(replacement) < original_block_len:
                replacement.extend([updated_units[idx].text for idx in idxs[len(replacement) :]])
            if len(replacement) > original_block_len:
                if original_block_len <= 1:
                    replacement = [" ".join(replacement).strip()]
                else:
                    replacement = replacement[: original_block_len - 1] + [
                        " ".join(replacement[original_block_len - 1 :]).strip()
                    ]
            for local_idx, global_idx in enumerate(idxs):
                updated_units[global_idx].text = replacement[local_idx].strip()
            rewritten_blocks.append(
                {
                    "block_range": [block_start, block_end],
                    "block_sentence_ids": [units[idx].sentence_id for idx in idxs],
                    "rewritten": True,
                }
            )

        cleaned = [unit for unit in updated_units if unit.text.strip()]
        return cleaned, {
            "scope": "paragraph",
            "applied": True,
            "target_sentence_ids": list(patch_plan.target_sentence_ids),
            "patch_count": len(patch_plan.patch_sequence),
            "rewritten_blocks": rewritten_blocks,
            "rewrite_conflict_type": str((rewrite_payload or {}).get("rewrite_conflict_type", "unknown")),
            "rewrite_required_state_changes": list((rewrite_payload or {}).get("required_state_changes", [])),
            "rewrite_forbidden_state_changes": list((rewrite_payload or {}).get("forbidden_state_changes", [])),
            "rewrite_violated_sentence_ids": list((rewrite_payload or {}).get("violated_sentence_ids", [])),
            "rewrite_violated_constraint_ids": list((rewrite_payload or {}).get("violated_constraint_ids", [])),
            "rewrite_conflict_spans": list((rewrite_payload or {}).get("conflict_spans", [])),
            "rewrite_targets_execution_spec_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_execution_spec_conflict", False)
            ),
            "rewrite_targets_required_state_change": bool(
                (rewrite_payload or {}).get("rewrite_targets_required_state_change", False)
            ),
            "rewrite_targets_transition_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_transition_conflict", False)
            ),
            "rewrite_targets_operator_post_state_conflict": bool(
                (rewrite_payload or {}).get("rewrite_targets_operator_post_state_conflict", False)
            ),
            "rewrite_operator_required_post_states": list(
                (rewrite_payload or {}).get("operator_required_post_states", [])
            ),
            "rewrite_canonical_required_states": list((rewrite_payload or {}).get("canonical_required_states", [])),
            "rewrite_canonical_forbidden_states": list((rewrite_payload or {}).get("canonical_forbidden_states", [])),
            "rewrite_canonical_operator_post_states": list(
                (rewrite_payload or {}).get("canonical_operator_post_states", [])
            ),
            "rewrite_required_state_groundings": list((rewrite_payload or {}).get("required_state_groundings", [])),
            "rewrite_forbidden_state_groundings": list((rewrite_payload or {}).get("forbidden_state_groundings", [])),
            "rewrite_operator_post_state_groundings": list(
                (rewrite_payload or {}).get("operator_post_state_groundings", [])
            ),
            "rewrite_transition_target_state_groundings": list(
                (rewrite_payload or {}).get("transition_target_state_groundings", [])
            ),
            "rewrite_transition_grounded_cues": list((rewrite_payload or {}).get("transition_grounded_cues", [])),
            "rewrite_transition_coherence_guidance": list(
                (rewrite_payload or {}).get("transition_coherence_guidance", [])
            ),
            "rewrite_retry_context": dict((rewrite_payload or {}).get("retry_context", {})),
            "rewrite_memory_binding_mode": str(
                (rewrite_payload or {}).get("memory_binding_mode", "normal_binding")
            ),
            "rewrite_generation_control_mode": str(
                (rewrite_payload or {}).get("generation_control_mode", "normal_generation")
            ),
            "rewrite_binding_decision_reasons": list(
                (rewrite_payload or {}).get("binding_decision_reasons", [])
            ),
            "rewrite_strengthened_memory_blocks": list(
                (rewrite_payload or {}).get("strengthened_memory_blocks", [])
            ),
            "rewrite_strengthened_constraints": list(
                (rewrite_payload or {}).get("strengthened_constraints", [])
            ),
            "rewrite_generation_control_context": dict(
                (rewrite_payload or {}).get("generation_control_context", {})
            ),
            "scope_expansion_triggered": bool((rewrite_payload or {}).get("scope_expansion_meta", {}).get("scope_expansion_triggered", False)),
            "original_scope": str((rewrite_payload or {}).get("scope_expansion_meta", {}).get("original_scope", "")),
            "expanded_scope": str((rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_scope", "")),
            "expanded_target_sentence_ids": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_target_sentence_ids", [])
            ),
            "expanded_local_window": dict(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expanded_local_window", {})
            ),
            "unresolved_slot_count_before": int(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("unresolved_slot_count_before", 0) or 0
            ),
            "unresolved_slot_count_after": int(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("unresolved_slot_count_after", 0) or 0
            ),
            "checklist_items_fixed_by_expansion": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("checklist_items_fixed_by_expansion", [])
            ),
            "still_unresolved_after_expansion": list(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("still_unresolved_after_expansion", [])
            ),
            "expansion_preserved_satisfied_items": bool(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("expansion_preserved_satisfied_items", True)
            ),
            "scope_expansion_effective": bool(
                (rewrite_payload or {}).get("scope_expansion_meta", {}).get("scope_expansion_effective", False)
            ),
        }

    def _rule_based_sentence_patch(
        self,
        sentence_text: str,
        patch: SentencePatch,
        violations: List[ConstraintViolation],
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        rewrite_payload: Dict[str, object] | None = None,
    ) -> str:
        rewritten = sentence_text or ""
        rule_types = {item.rule_type for item in violations}
        if "character_consistency" in rule_types or "neural_alias_drift" in rule_types:
            rewritten = self._canonicalize_character_names(
                rewritten,
                scene_plan,
                static_memory,
                violations,
            )
        if "timeline_consistency" in rule_types or patch.op_type.lower() == "temporal_fix":
            rewritten = self._repair_timeline_violations(rewritten, scene_plan, violations)
        if "world_rule_consistency" in rule_types:
            rewritten = self._repair_world_violations(
                rewritten,
                scene_plan,
                dynamic_memory,
                violations,
            )
            if any("does not satisfy required constraint" in v.message.lower() for v in violations):
                rewritten = self._append_required_constraints(
                    rewritten,
                    list(scene_plan.required_constraints) + list(scene_plan.must_keep_constraints),
                    max_add=2,
                )
        if "transition_validity" in rule_types or "operator_validity" in rule_types:
            required = scene_plan.expected_state_changes[:1] + scene_plan.required_constraints[:1]
            for token in required:
                if token and token.lower() not in rewritten.lower():
                    rewritten = f"{rewritten} {token}."
        rewritten = self._apply_transition_execution_guidance(
            text=rewritten,
            rewrite_payload=(rewrite_payload or {}),
            scene_plan=scene_plan,
        )
        rewritten = self._remove_forbidden_constraints(rewritten, scene_plan)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if rewritten and not rewritten.endswith((".", "!", "?")):
            rewritten = f"{rewritten}."
        return rewritten

    def _append_required_constraints(
        self,
        text: str,
        constraints: Sequence[str],
        max_add: int = 1,
    ) -> str:
        rewritten = text or ""
        added = 0
        for token in constraints:
            phrase = str(token or "").strip()
            if not phrase:
                continue
            if phrase.lower() in rewritten.lower():
                continue
            rewritten = f"{rewritten.rstrip()} {phrase}."
            added += 1
            if added >= max_add:
                break
        return rewritten.strip()

    def _patch_sentence_with_llm(
        self,
        scene_plan: ScenePlan,
        target_sentence: str,
        previous_sentence: str,
        next_sentence: str,
        patch: SentencePatch,
        patch_plan: PatchPlan,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        rewrite_payload: Dict[str, object] | None = None,
    ) -> str:
        api_key = self.llm_config.api_key.strip() or os.getenv(self.llm_config.api_key_env, "").strip()
        if not api_key:
            return ""
        violation_lines = [
            f"- [{item.rule_type}/{item.severity}] {item.message}" for item in violations[:6]
        ] or ["- Keep local consistency constraints satisfied."]
        canonical_entity_table = self._build_canonical_entity_table(static_memory, dynamic_memory, scene_plan)
        dual_summary = self._build_dual_summary_for_repair(violations)
        minimal_edit_anchors = self._build_minimal_edit_anchor_lines(violations)
        conflict_requirements = self._build_conflict_specific_requirements(
            rewrite_payload=(rewrite_payload or {}),
            allow_neighbor_revision=False,
        )
        retry_context = dict((rewrite_payload or {}).get("retry_context", {}))
        retry_note = ""
        if retry_context:
            checklist_note = self._format_retry_checklist_guidance(retry_context)
            experience_note = self._format_model_experience_retry_guidance(retry_context)
            retry_note = (
                f"\nRetry mode: This is a same-scope retry.\n"
                f"First-pass failed checks: {list(retry_context.get('failed_checks', []))}\n"
                f"Retry reason: {str(retry_context.get('retry_reason', 'state_realization_post_check_failed'))}\n"
                f"Missing required states from first-pass: {list(retry_context.get('missing_required_states', []))}\n"
                f"Remaining forbidden states from first-pass: {list(retry_context.get('remaining_forbidden_states', []))}\n"
                f"First-pass state realization match type: {str(retry_context.get('state_realization_match_type', 'no_match'))}\n"
                f"First-pass forbidden removal match type: {str(retry_context.get('forbidden_state_removal_match_type', 'no_match'))}\n"
                f"First-pass operator post-state match type: {str(retry_context.get('operator_post_state_match_type', 'no_match'))}\n"
                f"{checklist_note}\n"
                f"{experience_note}"
                "You MUST explicitly fix the listed first-pass failures in this retry.\n"
            )
        expansion_meta = dict((rewrite_payload or {}).get("scope_expansion_meta", {}))
        expansion_note = ""
        if bool(expansion_meta.get("scope_expansion_triggered", False)):
            expansion_note = (
                f"Scope expansion rationale: unresolved slots > 1 and current scope insufficient.\n"
                f"Original scope: {str(expansion_meta.get('original_scope', 'sentence'))}\n"
                f"Expanded scope: {str(expansion_meta.get('expanded_scope', 'sentence_with_neighbor'))}\n"
                f"Expanded sentence ids: {list(expansion_meta.get('expanded_target_sentence_ids', []))}\n"
                "Only modify expanded local context when necessary to fix unresolved checklist slots.\n"
                "Do not alter unrelated context.\n"
            )
        user = (
            "Patch ONLY the target sentence(s). Keep untargeted sentences unchanged.\n\n"
            f"Scene: {scene_plan.scene_id}\n"
            f"Objective: {scene_plan.objective}\n"
            f"Patch op: {patch.op_type}\n"
            f"Patch constraints: {patch.constraints_to_satisfy}\n"
            f"Rewrite conflict type: {str((rewrite_payload or {}).get('rewrite_conflict_type', 'unknown'))}\n"
            f"Required state changes to realize: {list((rewrite_payload or {}).get('required_state_changes', []))}\n"
            f"Canonical required state forms: {list((rewrite_payload or {}).get('canonical_required_states', []))}\n"
            f"Required state grounded aliases: {list((rewrite_payload or {}).get('required_state_groundings', []))}\n"
            f"Forbidden state changes to remove: {list((rewrite_payload or {}).get('forbidden_state_changes', []))}\n"
            f"Canonical forbidden state forms: {list((rewrite_payload or {}).get('canonical_forbidden_states', []))}\n"
            f"Forbidden state grounded aliases: {list((rewrite_payload or {}).get('forbidden_state_groundings', []))}\n"
            f"Violated sentence ids to directly repair: {list((rewrite_payload or {}).get('violated_sentence_ids', []))}\n"
            f"Violated constraint ids / rule anchors: {list((rewrite_payload or {}).get('violated_constraint_ids', []))}\n"
            f"Conflict spans/entities/tokens: {list((rewrite_payload or {}).get('conflict_spans', []))}\n"
            f"Transition violations: {list((rewrite_payload or {}).get('transition_violation_messages', []))}\n"
            f"Execution-spec violations: {list((rewrite_payload or {}).get('execution_spec_messages', []))}\n"
            f"Operator-required post-states: {list((rewrite_payload or {}).get('operator_required_post_states', []))}\n"
            f"Canonical operator post-state forms: {list((rewrite_payload or {}).get('canonical_operator_post_states', []))}\n"
            f"Operator post-state grounded aliases: {list((rewrite_payload or {}).get('operator_post_state_groundings', []))}\n"
            f"Transition coherence guidance: {list((rewrite_payload or {}).get('transition_coherence_guidance', []))}\n"
            f"Grounded transition cues: {list((rewrite_payload or {}).get('transition_grounded_cues', []))}\n"
            f"Memory binding mode: {str((rewrite_payload or {}).get('memory_binding_mode', 'normal_binding'))}\n"
            f"Generation control mode: {str((rewrite_payload or {}).get('generation_control_mode', 'normal_generation'))}\n"
            f"Binding decision reasons: {list((rewrite_payload or {}).get('binding_decision_reasons', []))}\n"
            f"Strengthened memory blocks: {list((rewrite_payload or {}).get('strengthened_memory_blocks', []))}\n"
            f"Strengthened constraints: {list((rewrite_payload or {}).get('strengthened_constraints', []))}\n"
            f"Unified generation control context: {dict((rewrite_payload or {}).get('generation_control_context', {}))}\n"
            f"Canonical entity table (scene-scoped):\n{self._format_canonical_entity_table(canonical_entity_table)}\n"
            f"Dual consistency decision hint: {str(dual_summary.get('decision', 'symbolic_only'))}\n"
            f"Dual consistency summary: {dual_summary}\n"
            f"Minimal edit anchors: {minimal_edit_anchors}\n"
            f"{expansion_note}"
            f"Protected sentence ids: {patch_plan.protected_sentence_ids}\n"
            f"Required constraints: {scene_plan.required_constraints}\n"
            f"Forbidden constraints: {scene_plan.forbidden_constraints}\n"
            f"Current world state: {dynamic_memory.world_setting.current_setting_state}\n"
            f"World invariants: {static_memory.world_setting.world_invariants[:5]}\n\n"
            f"Previous sentence:\n{previous_sentence}\n\n"
            f"Target sentence:\n{target_sentence}\n\n"
            f"Next sentence:\n{next_sentence}\n\n"
            f"Violations:\n{chr(10).join(violation_lines)}\n\n"
            "Rewrite requirements:\n"
            "- Resolve the specified conflict(s), not generic style edits.\n"
            "- Ensure required state change is textualized if missing, using at least one grounded canonical/alias form.\n"
            "- Remove conflicting/forbidden state evidence including grounded aliases.\n"
            "- Prefer token-level substitutions on conflict anchors; keep unaffected clauses unchanged.\n"
            "- Preserve unaffected context and avoid introducing new contradictions.\n\n"
            f"Conflict-specific requirements:\n{chr(10).join(conflict_requirements)}\n\n"
            f"{retry_note}"
            "Return ONLY the patched target sentence text."
        )
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": self.llm_config.system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.llm_config.request_temperature or 0.2),
            "max_tokens": int(self.llm_config.request_max_tokens or 220),
        }
        if self.llm_config.extra_request_body:
            payload.update(self.llm_config.extra_request_body)
        model_id = str(self.llm_config.model or "").strip().lower()
        if "max_completion_tokens" in payload:
            payload.pop("max_tokens", None)
        elif model_id.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens", int(self.llm_config.request_max_tokens or 220))

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for key, value in self.llm_config.extra_headers.items():
            headers[str(key)] = str(value)

        url = self.llm_config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.llm_config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
        except (error.URLError, error.HTTPError):
            return ""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return ""
        return self._extract_text(parsed)

    def _patch_paragraph_with_llm(
        self,
        scene_plan: ScenePlan,
        paragraph_text: str,
        patch_plan: PatchPlan,
        violations: Sequence[ConstraintViolation],
        static_memory: StaticMemory | None = None,
        dynamic_memory: DynamicMemory | None = None,
        previous_sentence: str = "",
        next_sentence: str = "",
        rewrite_payload: Dict[str, object] | None = None,
    ) -> str:
        api_key = self.llm_config.api_key.strip() or os.getenv(self.llm_config.api_key_env, "").strip()
        if not api_key:
            return ""
        violation_lines = [f"- {v.rule_type}: {v.message}" for v in violations[:6]]
        canonical_entity_table = (
            self._build_canonical_entity_table(static_memory, dynamic_memory, scene_plan)
            if static_memory is not None and dynamic_memory is not None
            else {}
        )
        dual_summary = self._build_dual_summary_for_repair(violations)
        minimal_edit_anchors = self._build_minimal_edit_anchor_lines(violations)
        conflict_requirements = self._build_conflict_specific_requirements(
            rewrite_payload=(rewrite_payload or {}),
            allow_neighbor_revision=True,
        )
        retry_context = dict((rewrite_payload or {}).get("retry_context", {}))
        retry_note = ""
        if retry_context:
            checklist_note = self._format_retry_checklist_guidance(retry_context)
            experience_note = self._format_model_experience_retry_guidance(retry_context)
            retry_note = (
                f"\nRetry mode: This is a same-scope retry.\n"
                f"First-pass failed checks: {list(retry_context.get('failed_checks', []))}\n"
                f"Retry reason: {str(retry_context.get('retry_reason', 'state_realization_post_check_failed'))}\n"
                f"Missing required states from first-pass: {list(retry_context.get('missing_required_states', []))}\n"
                f"Remaining forbidden states from first-pass: {list(retry_context.get('remaining_forbidden_states', []))}\n"
                f"First-pass state realization match type: {str(retry_context.get('state_realization_match_type', 'no_match'))}\n"
                f"First-pass forbidden removal match type: {str(retry_context.get('forbidden_state_removal_match_type', 'no_match'))}\n"
                f"First-pass operator post-state match type: {str(retry_context.get('operator_post_state_match_type', 'no_match'))}\n"
                f"{checklist_note}\n"
                f"{experience_note}"
                "You MUST explicitly fix the listed first-pass failures in this retry while keeping same patch scope.\n"
            )
        expansion_meta = dict((rewrite_payload or {}).get("scope_expansion_meta", {}))
        expansion_note = ""
        if bool(expansion_meta.get("scope_expansion_triggered", False)):
            expansion_note = (
                f"Scope expansion rationale: unresolved slots > 1 and current scope insufficient.\n"
                f"Original scope: {str(expansion_meta.get('original_scope', 'paragraph'))}\n"
                f"Expanded scope: {str(expansion_meta.get('expanded_scope', 'local_paragraph_window'))}\n"
                f"Expanded sentence ids: {list(expansion_meta.get('expanded_target_sentence_ids', []))}\n"
                "Use expanded local window only for unresolved checklist slots and transition coherence.\n"
                "Do not rewrite unrelated context.\n"
            )
        user = (
            "Rewrite only this local sentence block to satisfy listed constraints.\n"
            f"Scene: {scene_plan.scene_id}\n"
            f"Patch targets: {patch_plan.target_sentence_ids}\n"
            f"Protected sentence ids (must stay unchanged outside this block): {patch_plan.protected_sentence_ids}\n"
            f"Required constraints: {scene_plan.required_constraints}\n"
            f"Forbidden constraints: {scene_plan.forbidden_constraints}\n"
            f"Rewrite conflict type: {str((rewrite_payload or {}).get('rewrite_conflict_type', 'unknown'))}\n"
            f"Required state changes to realize: {list((rewrite_payload or {}).get('required_state_changes', []))}\n"
            f"Canonical required state forms: {list((rewrite_payload or {}).get('canonical_required_states', []))}\n"
            f"Required state grounded aliases: {list((rewrite_payload or {}).get('required_state_groundings', []))}\n"
            f"Forbidden state changes to remove: {list((rewrite_payload or {}).get('forbidden_state_changes', []))}\n"
            f"Canonical forbidden state forms: {list((rewrite_payload or {}).get('canonical_forbidden_states', []))}\n"
            f"Forbidden state grounded aliases: {list((rewrite_payload or {}).get('forbidden_state_groundings', []))}\n"
            f"Violated sentence ids to directly repair: {list((rewrite_payload or {}).get('violated_sentence_ids', []))}\n"
            f"Violated constraint ids / rule anchors: {list((rewrite_payload or {}).get('violated_constraint_ids', []))}\n"
            f"Conflict spans/entities/tokens: {list((rewrite_payload or {}).get('conflict_spans', []))}\n"
            f"Transition violations: {list((rewrite_payload or {}).get('transition_violation_messages', []))}\n"
            f"Execution-spec violations: {list((rewrite_payload or {}).get('execution_spec_messages', []))}\n"
            f"Operator-required post-states: {list((rewrite_payload or {}).get('operator_required_post_states', []))}\n"
            f"Canonical operator post-state forms: {list((rewrite_payload or {}).get('canonical_operator_post_states', []))}\n"
            f"Operator post-state grounded aliases: {list((rewrite_payload or {}).get('operator_post_state_groundings', []))}\n"
            f"Transition coherence guidance: {list((rewrite_payload or {}).get('transition_coherence_guidance', []))}\n"
            f"Grounded transition cues: {list((rewrite_payload or {}).get('transition_grounded_cues', []))}\n"
            f"Memory binding mode: {str((rewrite_payload or {}).get('memory_binding_mode', 'normal_binding'))}\n"
            f"Generation control mode: {str((rewrite_payload or {}).get('generation_control_mode', 'normal_generation'))}\n"
            f"Binding decision reasons: {list((rewrite_payload or {}).get('binding_decision_reasons', []))}\n"
            f"Strengthened memory blocks: {list((rewrite_payload or {}).get('strengthened_memory_blocks', []))}\n"
            f"Strengthened constraints: {list((rewrite_payload or {}).get('strengthened_constraints', []))}\n"
            f"Unified generation control context: {dict((rewrite_payload or {}).get('generation_control_context', {}))}\n"
            f"Canonical entity table (scene-scoped):\n{self._format_canonical_entity_table(canonical_entity_table)}\n"
            f"Dual consistency decision hint: {str(dual_summary.get('decision', 'symbolic_only'))}\n"
            f"Dual consistency summary: {dual_summary}\n"
            f"Minimal edit anchors: {minimal_edit_anchors}\n"
            f"{expansion_note}"
            f"Violations:\n{chr(10).join(violation_lines)}\n\n"
            f"Previous sentence (context, unchanged):\n{previous_sentence}\n\n"
            f"Paragraph:\n{paragraph_text}\n\n"
            f"Next sentence (context, unchanged):\n{next_sentence}\n\n"
            "Return rewritten block text only. Do not rewrite outside-block context.\n"
            "Preserve non-conflict content and explicitly satisfy required state-change mentions with grounded canonical/alias forms.\n"
            "Prefer minimal span edits anchored on conflict tokens instead of broad paraphrase.\n"
            f"Conflict-specific requirements:\n{chr(10).join(conflict_requirements)}\n"
            f"{retry_note}"
        )
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": self.llm_config.system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.llm_config.request_temperature or 0.3),
            "max_tokens": int(self.llm_config.request_max_tokens or 500),
        }
        if self.llm_config.extra_request_body:
            payload.update(self.llm_config.extra_request_body)
        model_id = str(self.llm_config.model or "").strip().lower()
        if "max_completion_tokens" in payload:
            payload.pop("max_tokens", None)
        elif model_id.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens", int(self.llm_config.request_max_tokens or 500))
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = self.llm_config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.llm_config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
        except (error.URLError, error.HTTPError):
            return ""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return ""
        return self._extract_text(parsed)

    def _build_conflict_specific_requirements(
        self,
        *,
        rewrite_payload: Dict[str, object],
        allow_neighbor_revision: bool,
    ) -> List[str]:
        required = [str(x).strip() for x in list(rewrite_payload.get("required_state_changes", [])) if str(x).strip()]
        forbidden = [str(x).strip() for x in list(rewrite_payload.get("forbidden_state_changes", [])) if str(x).strip()]
        operator_post = [
            str(x).strip() for x in list(rewrite_payload.get("operator_required_post_states", [])) if str(x).strip()
        ]
        canonical_required = [
            str(x).strip() for x in list(rewrite_payload.get("canonical_required_states", [])) if str(x).strip()
        ]
        canonical_forbidden = [
            str(x).strip() for x in list(rewrite_payload.get("canonical_forbidden_states", [])) if str(x).strip()
        ]
        canonical_operator_post = [
            str(x).strip() for x in list(rewrite_payload.get("canonical_operator_post_states", [])) if str(x).strip()
        ]
        required_groundings = list(rewrite_payload.get("required_state_groundings", []) or [])
        forbidden_groundings = list(rewrite_payload.get("forbidden_state_groundings", []) or [])
        operator_groundings = list(rewrite_payload.get("operator_post_state_groundings", []) or [])
        conflict_type = str(rewrite_payload.get("rewrite_conflict_type", "unknown"))
        generation_control_mode = str(rewrite_payload.get("generation_control_mode", "normal_generation"))
        lines: List[str] = []
        if required:
            lines.append(f"- Required state changes that MUST appear: {required[:4]}")
        if canonical_required:
            lines.append(f"- Canonical required states (must realize at least one grounded form each): {canonical_required[:4]}")
            lines.extend(self._format_grounded_alias_requirements(required_groundings, label="required"))
        if forbidden:
            lines.append(f"- Conflicting/forbidden states that MUST disappear: {forbidden[:4]}")
        if canonical_forbidden:
            lines.append(f"- Canonical forbidden states (grounded forms must be removed): {canonical_forbidden[:4]}")
            lines.extend(self._format_grounded_alias_requirements(forbidden_groundings, label="forbidden"))
        if operator_post:
            lines.append(f"- Operator-required post-state to realize explicitly: {operator_post[:4]}")
        if canonical_operator_post:
            lines.append(f"- Canonical operator post-states (must be explicit grounded realization): {canonical_operator_post[:4]}")
            lines.extend(self._format_grounded_alias_requirements(operator_groundings, label="operator_post_state"))
        if conflict_type in {"execution_spec_conflict", "operator_post_state_conflict", "mixed_conflict"}:
            lines.append("- For execution-spec conflicts, do not stop at cosmetic rewrite; realize required post-state.")
            lines.append("- You MUST realize at least one grounded form for each missing canonical required/post-state.")
        if conflict_type in {"transition_conflict", "mixed_conflict"}:
            lines.append("- For transition conflicts, restore coherent transition with explicit temporal/causal continuity.")
            lines.append("- Prefer grounded transition cues (then/after/later/therefore) over vague wording.")
        if generation_control_mode == "constrained_generation":
            lines.append("- Constrained mode: explicitly preserve invariants while realizing required state changes.")
        elif generation_control_mode == "strict_state_realization_generation":
            lines.append("- STRICT mode: explicitly textualize missing required states and operator post-states.")
            lines.append("- STRICT mode: explicitly remove forbidden/conflicting states; do not keep both inconsistent states.")
            lines.append("- STRICT mode: avoid cosmetic paraphrase and provide concrete before->after transition evidence.")
        if allow_neighbor_revision:
            lines.append("- If sentence-only edit is insufficient, revise surrounding sentence/paragraph to restore transition.")
        else:
            lines.append("- Keep scope on target sentence(s) but include explicit before->after transition evidence.")
        return lines

    def _format_grounded_alias_requirements(
        self,
        groundings: Sequence[Dict[str, object]],
        *,
        label: str,
        max_states: int = 3,
        max_aliases: int = 4,
    ) -> List[str]:
        lines: List[str] = []
        for item in list(groundings or [])[:max_states]:
            canonical = str(item.get("canonical", "")).strip()
            aliases = [
                str(x).strip()
                for x in list(item.get("aliases", []) or [])
                if str(x).strip() and str(x).strip() != canonical
            ][:max_aliases]
            if not canonical:
                continue
            lines.append(f"- Grounded {label} state `{canonical}` acceptable aliases: {aliases or [canonical]}")
        return lines

    def _format_retry_checklist_guidance(self, retry_context: Dict[str, object]) -> str:
        def _as_items(name: str) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            for item in list(retry_context.get(name, []) or []):
                if isinstance(item, dict):
                    out.append(dict(item))
            return out

        req_items = _as_items("first_pass_required_state_checklist")
        forb_items = _as_items("first_pass_forbidden_state_checklist")
        op_items = _as_items("first_pass_operator_post_state_checklist")

        def _pick(items: Sequence[Dict[str, object]], allowed: set[str]) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            for item in items:
                if str(item.get("status", "")) in allowed:
                    out.append(item)
            return out

        req_done = _pick(req_items, {"satisfied"})
        req_unresolved = _pick(req_items, {"unsatisfied", "uncertain"})
        forb_done = _pick(forb_items, {"removed"})
        forb_unresolved = _pick(forb_items, {"still_present", "uncertain"})
        op_done = _pick(op_items, {"realized"})
        op_unresolved = _pick(op_items, {"unrealized", "uncertain"})
        transition_cues = list(retry_context.get("transition_grounded_cues", []) or [])
        needs_transition = bool("transition_coherence_not_restored" in list(retry_context.get("failed_checks", [])))

        def _compact(items: Sequence[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            for item in list(items)[:limit]:
                out.append(
                    {
                        "canonical_state": str(item.get("canonical_state", "")),
                        "status": str(item.get("status", "")),
                        "match_type": str(item.get("match_type", "no_match")),
                        "matched_text_span": str(item.get("matched_text_span", "")),
                    }
                )
            return out

        priority_order = ["forbidden", "operator_post_state", "required"]
        protected_snapshot = {
            "forbidden_removed": _compact(forb_done),
            "operator_post_state_realized": _compact(op_done),
            "required_satisfied": _compact(req_done),
        }
        retry_context["retry_slot_priority_order"] = list(priority_order)
        retry_context["retry_slot_priority_unresolved"] = {
            "forbidden": _compact(forb_unresolved),
            "operator_post_state": _compact(op_unresolved),
            "required": _compact(req_unresolved),
        }
        retry_context["step_aware_preservation_guard_enabled"] = True
        retry_context["protected_items_snapshot"] = dict(protected_snapshot)
        lines = [
            "Checklist guidance for this retry (slot-by-slot with priority ordering):",
            "- Step-aware preservation guard is enabled for this retry.",
            f"- Preserve required states already satisfied: {_compact(req_done)}",
            f"- Preserve forbidden states already removed (do not reintroduce): {_compact(forb_done)}",
            f"- Preserve operator post-states already realized: {_compact(op_done)}",
            f"- Protected items snapshot (must remain valid): {protected_snapshot}",
            f"- Retry slot priority order: {priority_order}",
            f"- Step 1 (highest priority): remove unresolved forbidden states: {_compact(forb_unresolved)}",
            f"- Step 2: realize unresolved operator post-states: {_compact(op_unresolved)}",
            f"- Step 3: realize unresolved required states: {_compact(req_unresolved)}",
        ]
        if needs_transition:
            lines.append(
                f"- Transition coherence support (horizontal across steps): add grounded transition cues from {transition_cues[:6] if transition_cues else ['then', 'after', 'therefore']}."
            )
        lines.append("- Preserve all previously satisfied states across Step 2 and Step 3.")
        lines.append("- Do not reintroduce any removed forbidden state (canonical or grounded alias forms).")
        lines.append("- Do not weaken or delete already realized operator post-states.")
        lines.append("- Focus only on unresolved checklist items in the above order; do not break already satisfied items.")
        return "\n".join(lines)

    def _format_model_experience_retry_guidance(self, retry_context: Dict[str, object]) -> str:
        cautions = [
            str(x).strip()
            for x in list(retry_context.get("model_experience_cautions", []) or [])
            if str(x).strip()
        ][:3]
        if not cautions:
            return ""
        lines = ["Model-specific retry cautions (lightweight):"]
        lines.extend(f"- {line}" for line in cautions)
        lines.append("- Apply only to current conflict span; keep other text stable.")
        return "\n".join(lines) + "\n"

    def _build_conflict_rewrite_payload(
        self,
        *,
        scene_plan: ScenePlan,
        violations: Sequence[ConstraintViolation],
        patch_plan: PatchPlan,
        generation_control_context: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        control_context = dict(generation_control_context or {})
        required_state_changes: List[str] = list(scene_plan.expected_state_changes)
        forbidden_state_changes: List[str] = list(scene_plan.forbidden_state_changes)
        transition_violation_messages: List[str] = []
        execution_spec_messages: List[str] = []
        operator_required_post_states: List[str] = []
        conflict_spans: List[str] = []
        violated_sentence_ids: List[str] = []
        violated_constraint_ids: List[str] = []
        has_transition = False
        has_execution_spec = False
        has_operator_post_state = False
        has_constraint_conflict = False
        for item in violations:
            rule_type = str(item.rule_type or "").strip().lower()
            msg = str(item.message or "").strip()
            context = item.context if isinstance(item.context, dict) else {}
            msg_lower = msg.lower()
            if msg:
                if ("transition" in rule_type) or ("transition" in msg_lower):
                    has_transition = True
                    transition_violation_messages.append(msg)
                if ("execution spec" in msg_lower) or ("required state changes" in msg_lower):
                    has_execution_spec = True
                    execution_spec_messages.append(msg)
                if "operator_validity" in rule_type:
                    transition_violation_messages.append(msg)
                    if ("postcondition" in msg_lower) or ("post-state" in msg_lower) or ("required state changes" in msg_lower):
                        has_operator_post_state = True
                if ("constraint" in msg_lower) or ("forbidden" in msg_lower) or ("required" in msg_lower):
                    has_constraint_conflict = True
            for anchor in item.anchors:
                for sid in anchor.sentence_ids:
                    if sid not in violated_sentence_ids:
                        violated_sentence_ids.append(sid)
            vid = str(item.rule_type or "unknown").strip()
            if vid:
                violated_constraint_ids.append(vid)
            for rid in item.related_ids:
                token = str(rid).strip()
                if token and token not in conflict_spans:
                    conflict_spans.append(token)
            if isinstance(context, dict):
                missing_post = str(context.get("missing_postcondition", "")).strip()
                if missing_post and missing_post not in required_state_changes:
                    required_state_changes.append(missing_post)
                if missing_post and missing_post not in operator_required_post_states:
                    operator_required_post_states.append(missing_post)
                    has_operator_post_state = True
                inconsistent_state = str(context.get("inconsistent_state", "")).strip()
                if inconsistent_state and inconsistent_state not in forbidden_state_changes:
                    forbidden_state_changes.append(inconsistent_state)
                conflicting_facts = context.get("conflicting_facts", [])
                if isinstance(conflicting_facts, list):
                    for fact in conflicting_facts:
                        fact_token = str(fact).strip()
                        if not fact_token:
                            continue
                        if fact_token.lower().startswith("missing_postcondition:"):
                            post = fact_token.split(":", 1)[1].strip()
                            if post and post not in operator_required_post_states:
                                operator_required_post_states.append(post)
                                has_operator_post_state = True
                            if post and post not in required_state_changes:
                                required_state_changes.append(post)
                        if fact_token.lower().startswith("forbidden_patterns:"):
                            forbidden = fact_token.split(":", 1)[1].strip()
                            if forbidden and forbidden not in forbidden_state_changes:
                                forbidden_state_changes.append(forbidden)
        active_conflicts = int(bool(has_transition)) + int(bool(has_execution_spec)) + int(bool(has_operator_post_state))
        if active_conflicts >= 2:
            rewrite_conflict_type = "mixed_conflict"
        elif has_operator_post_state:
            rewrite_conflict_type = "operator_post_state_conflict"
        elif has_execution_spec:
            rewrite_conflict_type = "execution_spec_conflict"
        elif has_transition:
            rewrite_conflict_type = "transition_conflict"
        elif has_constraint_conflict:
            rewrite_conflict_type = "constraint_conflict"
        else:
            rewrite_conflict_type = "unknown"
        transition_coherence_guidance = self._build_transition_coherence_guidance(
            rewrite_conflict_type=rewrite_conflict_type
        )
        if patch_plan.target_sentence_ids and len(patch_plan.target_sentence_ids) > 1:
            target_scope = "multi_sentence"
        else:
            target_scope = "sentence"
        if not violated_sentence_ids:
            violated_sentence_ids = list(dict.fromkeys([str(x) for x in patch_plan.target_sentence_ids if str(x).strip()]))
        generation_control_mode = str(control_context.get("generation_control_mode", "normal_generation"))
        if generation_control_mode == "strict_state_realization_generation":
            required_state_changes = list(
                dict.fromkeys(required_state_changes + list(control_context.get("required_state_reminders", [])))
            )
            forbidden_state_changes = list(
                dict.fromkeys(forbidden_state_changes + list(control_context.get("forbidden_state_reminders", [])))
            )
        grounding_bundle = build_state_grounding_bundle(
            required_states=required_state_changes,
            forbidden_states=forbidden_state_changes,
            operator_post_states=operator_required_post_states,
            transition_target_states=required_state_changes,
        )
        return {
            "rewrite_conflict_type": rewrite_conflict_type,
            "target_scope": target_scope,
            "required_state_changes": list(dict.fromkeys([x for x in required_state_changes if str(x).strip()])),
            "forbidden_state_changes": list(dict.fromkeys([x for x in forbidden_state_changes if str(x).strip()])),
            "transition_violation_messages": list(dict.fromkeys(transition_violation_messages[:8])),
            "execution_spec_messages": list(dict.fromkeys(execution_spec_messages[:8])),
            "operator_required_post_states": list(dict.fromkeys(operator_required_post_states[:8])),
            "conflict_spans": list(dict.fromkeys(conflict_spans[:12])),
            "violated_sentence_ids": list(dict.fromkeys(violated_sentence_ids)),
            "violated_constraint_ids": list(dict.fromkeys(violated_constraint_ids)),
            "rewrite_targets_execution_spec_conflict": bool(
                has_execution_spec or rewrite_conflict_type in {"execution_spec_conflict", "mixed_conflict"}
            ),
            "rewrite_targets_required_state_change": bool(required_state_changes),
            "rewrite_targets_transition_conflict": bool(
                has_transition or rewrite_conflict_type in {"transition_conflict", "mixed_conflict"}
            ),
            "rewrite_targets_operator_post_state_conflict": bool(
                has_operator_post_state or rewrite_conflict_type in {"operator_post_state_conflict", "mixed_conflict"}
            ),
            "transition_coherence_guidance": list(transition_coherence_guidance),
            "memory_binding_mode": str(control_context.get("memory_binding_mode", "normal_binding")),
            "generation_control_mode": generation_control_mode,
            "binding_decision_reasons": list(control_context.get("decision_reasons", [])),
            "strengthened_memory_blocks": list(control_context.get("strengthened_memory_blocks", [])),
            "strengthened_constraints": list(control_context.get("strengthened_constraints", [])),
            "canonical_required_states": list(grounding_bundle.get("canonical_required_states", [])),
            "canonical_forbidden_states": list(grounding_bundle.get("canonical_forbidden_states", [])),
            "canonical_operator_post_states": list(grounding_bundle.get("canonical_operator_post_states", [])),
            "required_state_groundings": list(grounding_bundle.get("required_state_groundings", [])),
            "forbidden_state_groundings": list(grounding_bundle.get("forbidden_state_groundings", [])),
            "operator_post_state_groundings": list(grounding_bundle.get("operator_post_state_groundings", [])),
            "transition_target_state_groundings": list(grounding_bundle.get("transition_target_state_groundings", [])),
            "transition_grounded_cues": list(grounding_bundle.get("transition_grounded_cues", [])),
        }

    def _build_transition_coherence_guidance(
        self,
        *,
        rewrite_conflict_type: str,
    ) -> List[str]:
        guidance = [
            "Restore transition coherence with explicit causal and temporal progression.",
            "Avoid keeping contradictory pre-state and post-state simultaneously.",
        ]
        if rewrite_conflict_type in {"execution_spec_conflict", "operator_post_state_conflict", "mixed_conflict"}:
            guidance.append("Realize operator-required post-state explicitly in the rewritten text.")
        if rewrite_conflict_type in {"transition_conflict", "mixed_conflict"}:
            guidance.append("If sentence-only rewrite is insufficient, revise neighboring sentence/paragraph context.")
        return guidance

    def _requires_transition_coherence_escalation(
        self,
        rewrite_payload: Dict[str, object],
        scope: str,
        retry_context: Dict[str, object] | None = None,
    ) -> bool:
        if scope != "sentence":
            return False
        context = dict(retry_context or {})
        # Do not auto-escalate sentence-first pass; only escalate after explicit retry failure signals.
        failed_checks = {str(x) for x in list(context.get("failed_checks", [])) if str(x).strip()}
        if not failed_checks:
            return False
        conflict_type = str(rewrite_payload.get("rewrite_conflict_type", "unknown"))
        if conflict_type not in {
            "transition_conflict",
            "execution_spec_conflict",
            "operator_post_state_conflict",
            "mixed_conflict",
        }:
            return False
        if failed_checks.intersection(
            {
                "transition_coherence_not_restored",
                "required_state_not_realized",
                "operator_post_state_not_realized",
                "forbidden_state_not_removed",
            }
        ):
            return True
        transition_msgs = list(rewrite_payload.get("transition_violation_messages", []))
        required_state_changes = list(rewrite_payload.get("required_state_changes", []))
        operator_post_states = list(rewrite_payload.get("operator_required_post_states", []))
        return bool(
            (len(transition_msgs) >= 1 and len(required_state_changes) >= 1)
            or len(transition_msgs) >= 2
            or (operator_post_states and required_state_changes)
        )

    def _compute_patch_change_stats(
        self,
        *,
        original_units: Sequence[SentenceUnit],
        patched_units: Sequence[SentenceUnit],
        target_sentence_ids: Set[str],
        violated_sentence_ids: Set[str],
    ) -> Dict[str, object]:
        original_map = {u.sentence_id: (u.text or "").strip() for u in original_units}
        patched_map = {u.sentence_id: (u.text or "").strip() for u in patched_units}
        original_ids = set(original_map.keys())
        patched_ids = set(patched_map.keys())

        changed_ids: Set[str] = set()
        for sid in original_ids.intersection(patched_ids):
            if original_map.get(sid, "") != patched_map.get(sid, ""):
                changed_ids.add(sid)
        added_ids = set(patched_ids - original_ids)
        removed_ids = set(original_ids - patched_ids)
        changed_ids.update(added_ids)
        changed_ids.update(removed_ids)

        changed_target_ids = sorted(sid for sid in changed_ids if sid in target_sentence_ids)
        changed_non_target_ids = sorted(sid for sid in changed_ids if sid not in target_sentence_ids)
        target_touch_den = max(1, len(target_sentence_ids))
        target_touched_ratio = float(len(changed_target_ids)) / float(target_touch_den)
        violation_context_touched = bool(changed_ids.intersection(violated_sentence_ids))
        return {
            "changed_sentence_count": int(len(changed_ids)),
            "changed_sentence_ids": sorted(changed_ids),
            "added_sentence_ids": sorted(added_ids),
            "removed_sentence_ids": sorted(removed_ids),
            "changed_target_sentence_count": int(len(changed_target_ids)),
            "changed_non_target_sentence_count": int(len(changed_non_target_ids)),
            "changed_non_target_sentence_ids": changed_non_target_ids,
            "target_sentence_touched_ratio": float(target_touched_ratio),
            "violation_context_touched": bool(violation_context_touched),
        }

    def _apply_transition_execution_guidance(
        self,
        *,
        text: str,
        rewrite_payload: Dict[str, object],
        scene_plan: ScenePlan,
    ) -> str:
        rewritten = str(text or "")
        conflict_type = str(rewrite_payload.get("rewrite_conflict_type", "unknown"))
        generation_control_mode = str(rewrite_payload.get("generation_control_mode", "normal_generation"))
        strict_mode = generation_control_mode == "strict_state_realization_generation"
        if conflict_type in {
            "transition_conflict",
            "execution_spec_conflict",
            "operator_post_state_conflict",
            "mixed_conflict",
        }:
            if not re.search(
                r"\b(then|after|later|next|therefore|as a result|subsequently)\b",
                rewritten,
                flags=re.IGNORECASE,
            ):
                if rewritten:
                    rewritten = f"After a clear transition, {rewritten[0].lower() + rewritten[1:]}"
            state_limit = 6 if strict_mode else 3
            required = list(rewrite_payload.get("required_state_changes", []))[:state_limit]
            operator_post_states = list(rewrite_payload.get("operator_required_post_states", []))[:state_limit]
            for token in required + operator_post_states:
                tok = str(token).strip()
                if not tok:
                    continue
                normalized = self._normalize_state_token_for_text(tok).lower()
                if normalized and normalized not in rewritten.lower():
                    lead = "This explicitly realizes" if strict_mode else "This leads to"
                    rewritten = f"{rewritten.rstrip()} {lead} {self._normalize_state_token_for_text(tok)}."
        forbidden = list(rewrite_payload.get("forbidden_state_changes", []))[:4]
        for token in forbidden:
            tok = str(token).strip()
            if not tok:
                continue
            rewritten = re.sub(re.escape(tok), "", rewritten, flags=re.IGNORECASE)
            rewritten = re.sub(
                re.escape(self._normalize_state_token_for_text(tok)),
                "",
                rewritten,
                flags=re.IGNORECASE,
            )
        if scene_plan.forbidden_state_changes:
            for token in scene_plan.forbidden_state_changes[:3]:
                tok = str(token).strip()
                if tok:
                    rewritten = re.sub(re.escape(tok), "", rewritten, flags=re.IGNORECASE)
                    rewritten = re.sub(
                        re.escape(self._normalize_state_token_for_text(tok)),
                        "",
                        rewritten,
                        flags=re.IGNORECASE,
                    )
        rewritten = re.sub(
            r"\b(?:teleport(?:s|ed|ing)?|without any transition|without transition)\b",
            "after a clear transition",
            rewritten,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", rewritten).strip()

    def _normalize_state_token_for_text(self, token: str) -> str:
        return re.sub(r"\s+", " ", str(token or "").replace("_", " ").strip())

    def _compose_scene_text(self, units: Sequence[SentenceUnit]) -> str:
        if not units:
            return ""
        lines: List[str] = []
        current_paragraph = units[0].paragraph_id
        buffer: List[str] = []
        for unit in units:
            if unit.paragraph_id != current_paragraph:
                if buffer:
                    lines.append(" ".join(buffer).strip())
                buffer = []
                current_paragraph = unit.paragraph_id
            if unit.text.strip():
                buffer.append(unit.text.strip())
        if buffer:
            lines.append(" ".join(buffer).strip())
        return "\n\n".join(line for line in lines if line).strip()

    def _unchanged_ratio(
        self,
        original_units: Sequence[SentenceUnit],
        patched_units: Sequence[SentenceUnit],
    ) -> float:
        if not original_units:
            return 0.0
        original_map = {unit.sentence_id: unit.text.strip() for unit in original_units}
        patched_map = {unit.sentence_id: unit.text.strip() for unit in patched_units}
        unchanged = 0
        for sentence_id, text in original_map.items():
            if sentence_id in patched_map and patched_map[sentence_id] == text:
                unchanged += 1
        return float(unchanged) / float(max(1, len(original_map)))

    def _build_local_blocks(
        self,
        target_idxs: Sequence[int],
        total_units: int,
        window: int = 1,
    ) -> List[Tuple[int, int]]:
        if not target_idxs:
            return []
        blocks: List[Tuple[int, int]] = []
        for idx in target_idxs:
            start = max(0, int(idx) - window)
            end = min(total_units - 1, int(idx) + window)
            if not blocks:
                blocks.append((start, end))
                continue
            last_start, last_end = blocks[-1]
            if start <= last_end + 1:
                blocks[-1] = (last_start, max(last_end, end))
            else:
                blocks.append((start, end))
        return blocks

    def _run_preservation_audit(
        self,
        original_units: Sequence[SentenceUnit],
        patched_units: Sequence[SentenceUnit],
        target_sentence_ids: Set[str],
        protected_sentence_ids: Set[str],
    ) -> Dict[str, object]:
        original_map = {u.sentence_id: u.text for u in original_units}
        patched_map = {u.sentence_id: u.text for u in patched_units}
        protected_integrity_pass = True
        regressions: List[Dict[str, object]] = []
        spillover_count = 0

        for sid in sorted(protected_sentence_ids):
            before = original_map.get(sid, "")
            after = patched_map.get(sid, "")
            if not before or not after:
                continue
            detail: Dict[str, object] = {"sentence_id": sid}
            entity_before = self._extract_entity_mentions(before)
            entity_after = self._extract_entity_mentions(after)
            temporal_before = self._extract_temporal_cues(before)
            temporal_after = self._extract_temporal_cues(after)
            attr_before = self._extract_key_attributes(before)
            attr_after = self._extract_key_attributes(after)
            event_before = self._extract_event_markers(before)
            event_after = self._extract_event_markers(after)

            if entity_before and not entity_before.issubset(entity_after):
                detail["entity_drop"] = sorted(entity_before - entity_after)
            if temporal_before and not temporal_before.issubset(temporal_after):
                detail["temporal_drop"] = sorted(temporal_before - temporal_after)
            if attr_before and not attr_before.issubset(attr_after):
                detail["attribute_drop"] = sorted(attr_before - attr_after)
            if event_before and not event_before.issubset(event_after):
                detail["event_drop"] = sorted(event_before - event_after)

            if len(detail) > 1:
                regressions.append(detail)
                protected_integrity_pass = False

        ordered_ids = [u.sentence_id for u in patched_units]
        idx_map = {sid: idx for idx, sid in enumerate(ordered_ids)}
        for sid in target_sentence_ids:
            idx = idx_map.get(sid)
            if idx is None:
                continue
            neighbors = []
            if idx > 0:
                neighbors.append(ordered_ids[idx - 1])
            if idx + 1 < len(ordered_ids):
                neighbors.append(ordered_ids[idx + 1])
            for nid in neighbors:
                if nid in target_sentence_ids:
                    continue
                text = patched_map.get(nid, "")
                if self._likely_coref_break(text, patched_units, idx_map.get(nid, -1)):
                    spillover_count += 1
                if self._likely_causal_break(text, patched_units, idx_map.get(nid, -1)):
                    spillover_count += 1

        if spillover_count > 0:
            protected_integrity_pass = False

        return {
            "protected_integrity_pass": protected_integrity_pass,
            "protected_regressions": regressions,
            "spillover_count": spillover_count,
            "details": {
                "num_protected_checked": len(protected_sentence_ids),
                "num_regressions": len(regressions),
            },
        }

    def _extract_entity_mentions(self, text: str) -> Set[str]:
        ids = set(re.findall(r"\b(?:char|ent)_[a-z0-9_]+\b", text.lower()))
        names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        return ids.union({item.lower() for item in names})

    def _extract_temporal_cues(self, text: str) -> Set[str]:
        lowered = text.lower()
        cues = {"before", "after", "then", "later", "earlier", "meanwhile", "finally"}
        return {cue for cue in cues if cue in lowered}

    def _extract_key_attributes(self, text: str) -> Set[str]:
        attrs = set(re.findall(r"\b\d{1,4}\b", text))
        attrs.update(re.findall(r"\b(?:old|young|injured|alive|dead|missing|armed)\b", text.lower()))
        return attrs

    def _extract_event_markers(self, text: str) -> Set[str]:
        markers = {"arrive", "leave", "reveal", "discover", "attack", "resolve", "move", "travel"}
        lowered = text.lower()
        return {marker for marker in markers if marker in lowered}

    def _likely_coref_break(
        self,
        text: str,
        patched_units: Sequence[SentenceUnit],
        index: int,
    ) -> bool:
        lowered = f" {text.lower()} "
        pronouns = (" he ", " she ", " they ", " him ", " her ", " them ", " his ", " their ")
        if not any(p in lowered for p in pronouns):
            return False
        prev = patched_units[index - 1].text if index > 0 else ""
        return len(self._extract_entity_mentions(prev)) == 0

    def _likely_causal_break(
        self,
        text: str,
        patched_units: Sequence[SentenceUnit],
        index: int,
    ) -> bool:
        lowered = text.lower()
        if not any(token in lowered for token in ("therefore", "because", "as a result", "so ")):
            return False
        prev = patched_units[index - 1].text.lower() if index > 0 else ""
        return not bool(re.search(r"\b(?:did|made|caused|happened|acted|moved|revealed)\b", prev))

    def _stability_score(self, original_text: str, patched_text: str) -> float:
        original_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (original_text or "").lower()))
        patched_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (patched_text or "").lower()))
        if not original_tokens and not patched_tokens:
            return 1.0
        if not original_tokens or not patched_tokens:
            return 0.0
        jaccard = len(original_tokens.intersection(patched_tokens)) / max(
            1, len(original_tokens.union(patched_tokens))
        )
        return float(max(0.0, min(1.0, jaccard)))

    def _repair_with_llm(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
    ) -> str:
        api_key = self.llm_config.api_key.strip() or os.getenv(self.llm_config.api_key_env, "").strip()
        if not api_key:
            return ""

        violation_lines = []
        for v in violations:
            violation_lines.append(f"- [{v.rule_type}/{v.severity}] {v.message} | hint: {v.repair_hint}")
        if not violation_lines:
            violation_lines = ["- keep local consistency with memory constraints."]
        symbolic_conflicts = self._collect_symbolic_conflicts(violations)
        directives = self._build_repair_directives(scene_plan, violations, symbolic_conflicts)
        memory_snapshot = self._build_memory_snapshot(scene_plan, static_memory, dynamic_memory)
        canonical_entity_table = self._build_canonical_entity_table(static_memory, dynamic_memory, scene_plan)
        dual_summary = self._build_dual_summary_for_repair(violations)
        minimal_edit_anchors = self._build_minimal_edit_anchor_lines(violations)
        symbolic_conflict_lines = [self._format_symbolic_conflict(item) for item in symbolic_conflicts]
        if not symbolic_conflict_lines:
            symbolic_conflict_lines = ["- (no symbolic conflict payload)"]

        user = (
            f"Task: Rewrite ONLY this scene ({scene_plan.scene_id}) to fix constraint violations.\n\n"
            f"Chapter: {scene_plan.chapter_id}\n"
            f"Scene objective: {scene_plan.objective}\n"
            f"Required characters: {scene_plan.required_characters or scene_plan.involved_characters}\n"
            f"Optional characters: {scene_plan.optional_characters}\n"
            f"Expected state changes: {scene_plan.expected_state_changes}\n"
            f"Required constraints: {scene_plan.required_constraints}\n"
            f"Must keep constraints: {scene_plan.must_keep_constraints}\n"
            f"Forbidden constraints: {scene_plan.forbidden_constraints}\n"
            f"Current world state: {dynamic_memory.world_setting.current_setting_state}\n"
            f"Current structured state id: {dynamic_memory.current_state.state_id}\n"
            f"Active inferred constraints: {[c.predicate for c in dynamic_memory.inferred_constraints[-12:]]}\n"
            f"Pending constraints from memory: {dynamic_memory.timeline_plot.pending_constraints[-8:]}\n"
            f"World invariants: {static_memory.world_setting.world_invariants[:6]}\n\n"
            f"Memory snapshot:\n{memory_snapshot}\n\n"
            f"Canonical entity table (scene-scoped):\n{self._format_canonical_entity_table(canonical_entity_table)}\n\n"
            f"Dual consistency decision hint: {str(dual_summary.get('decision', 'symbolic_only'))}\n"
            f"Dual consistency summary: {dual_summary}\n\n"
            f"Minimal edit anchors:\n{chr(10).join(minimal_edit_anchors)}\n\n"
            f"Conflict-driven symbolic repair payload:\n{chr(10).join(symbolic_conflict_lines)}\n\n"
            f"Repair directives:\n{chr(10).join(directives)}\n\n"
            f"Violations to fix:\n{chr(10).join(violation_lines)}\n\n"
            f"Original scene text:\n{scene_text}\n\n"
            "Output only the repaired scene text. Keep the same scene scope. "
            "Do not rewrite other scenes and do not output analysis. "
            "Preserve narrative intent while fixing factual, timeline, and world-rule inconsistencies. "
            "Use minimal local edits anchored on conflict tokens; do not paraphrase unaffected spans."
        )
        messages = [
            {"role": "system", "content": self.llm_config.system_prompt},
            {"role": "user", "content": user},
        ]

        payload = {
            "model": self.llm_config.model,
            "messages": messages,
            "temperature": float(self.llm_config.request_temperature or 0.4),
            "max_tokens": int(self.llm_config.request_max_tokens or 900),
        }
        if self.llm_config.extra_request_body:
            payload.update(self.llm_config.extra_request_body)
        model_id = str(self.llm_config.model or "").strip().lower()
        if "max_completion_tokens" in payload:
            payload.pop("max_tokens", None)
        elif model_id.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens", int(self.llm_config.request_max_tokens or 900))

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for key, value in self.llm_config.extra_headers.items():
            headers[str(key)] = str(value)

        url = self.llm_config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.llm_config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
        except (error.URLError, error.HTTPError) as exc:
            self.logger.warning("Local repair call failed: %s", exc)
            return ""

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return ""
        return self._extract_text(parsed)

    def _extract_text(self, response: Dict[str, object]) -> str:
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = item.get("text", "")
                    if txt:
                        chunks.append(str(txt))
            return "\n".join(chunks).strip()
        reasoning_candidates: List[object] = []
        if isinstance(msg, dict):
            reasoning_candidates.extend([msg.get("reasoning_content"), msg.get("reasoning")])
        if isinstance(first, dict):
            reasoning_candidates.extend([first.get("reasoning_content"), first.get("reasoning")])
        for candidate in reasoning_candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, list):
                chunks: List[str] = []
                for item in candidate:
                    if not isinstance(item, dict):
                        continue
                    txt = item.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
                if chunks:
                    return "\n".join(chunks).strip()
        text = first.get("text", "") if isinstance(first, dict) else ""
        return str(text).strip()

    def _repair_with_rules(
        self,
        scene_text: str,
        scene_plan: ScenePlan,
        violations: List[ConstraintViolation],
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
    ) -> str:
        """Heuristic fallback local rewrite without LLM, with violation-type directives."""
        sentences = self._split_sentences(scene_text)
        if not sentences:
            sentences = [scene_text.strip()]

        working = self._drop_forbidden_sentences(sentences, scene_plan)
        working_text = " ".join(working).strip()
        if not working_text:
            working_text = (
                f"{scene_plan.title}. {scene_plan.objective} "
                "The current scene unfolds with consistent actions and state transitions."
            )

        working_text = self._canonicalize_character_names(
            working_text,
            scene_plan,
            static_memory,
            violations,
        )
        working_text = self._repair_world_violations(
            working_text,
            scene_plan,
            dynamic_memory,
            violations,
        )
        working_text = self._repair_timeline_violations(working_text, scene_plan, violations)
        working_text = self._repair_character_violations(working_text, scene_plan, static_memory, violations)
        working_text = self._enforce_required_constraints(working_text, scene_plan, violations)
        working_text = self._remove_forbidden_constraints(working_text, scene_plan)
        working_text = self._apply_symbolic_operator_hint(working_text, violations)
        working_text = re.sub(r"\s+", " ", working_text).strip()
        if not working_text.endswith((".", "!", "?")):
            working_text = f"{working_text}."
        return working_text

    def _build_repair_directives(
        self,
        scene_plan: ScenePlan,
        violations: List[ConstraintViolation],
        symbolic_conflicts: List[Dict[str, str]],
    ) -> List[str]:
        directives: List[str] = [
            "- Keep scope local: only rewrite this scene, do not alter earlier/later scenes.",
            "- Preserve scene objective while fixing consistency issues.",
        ]
        rule_types = {v.rule_type for v in violations}
        if "character_consistency" in rule_types:
            directives.append(
                "- Character consistency: use canonical names and ensure planned characters appear with coherent states."
            )
        if "timeline_consistency" in rule_types:
            directives.append(
                "- Timeline consistency: keep event order forward-moving and explicit temporal transitions."
            )
        if "world_rule_consistency" in rule_types:
            directives.append(
                "- World-rule consistency: remove forbidden actions and keep setting transitions explicit and legal."
            )
        if any(rt.startswith("neural_") for rt in rule_types):
            directives.append(
                "- Neural consistency signals: keep one canonical naming path per entity and remove contradiction markers."
            )
            directives.append(
                "- Perform minimal span edits around conflict anchors; keep unaffected narrative content intact."
            )
        if scene_plan.required_constraints:
            directives.append("- Explicitly satisfy all required constraints in scene prose.")
        if scene_plan.forbidden_constraints:
            directives.append("- Avoid all forbidden constraints and forbidden outcomes.")
        suggested_ops = sorted(
            {row.get("suggested_repair_operator", "").strip().upper() for row in symbolic_conflicts if row.get("suggested_repair_operator")}
        )
        if suggested_ops:
            directives.append(
                f"- Symbolic repair operator guidance: realize operator-level fix using {suggested_ops[0]}."
            )
        return directives

    def _collect_symbolic_conflicts(
        self,
        violations: List[ConstraintViolation],
    ) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for violation in violations:
            if not isinstance(violation.context, dict):
                continue
            if not violation.context:
                continue
            row: Dict[str, str] = {
                "rule_type": violation.rule_type,
                "message": violation.message,
                "violated_operator": str(violation.context.get("violated_operator", "")),
                "conflicting_facts": str(violation.context.get("conflicting_facts", "")),
                "missing_postcondition": str(violation.context.get("missing_postcondition", "")),
                "inconsistent_state": str(violation.context.get("inconsistent_state", "")),
                "suggested_repair_operator": str(violation.context.get("suggested_repair_operator", "")),
            }
            payload.append(row)
        return payload

    def _format_symbolic_conflict(self, row: Dict[str, str]) -> str:
        return (
            f"- [{row.get('rule_type', '')}] violated_operator={row.get('violated_operator', '')}; "
            f"conflicting_facts={row.get('conflicting_facts', '')}; "
            f"missing_postcondition={row.get('missing_postcondition', '')}; "
            f"inconsistent_state={row.get('inconsistent_state', '')}; "
            f"suggested_repair_operator={row.get('suggested_repair_operator', '')}"
        )

    def _apply_symbolic_operator_hint(
        self,
        text: str,
        violations: List[ConstraintViolation],
    ) -> str:
        suggestions = [
            str(v.context.get("suggested_repair_operator", "")).strip().upper()
            for v in violations
            if isinstance(v.context, dict) and v.context.get("suggested_repair_operator")
        ]
        if not suggestions:
            return text
        op = suggestions[0]
        addition = ""
        if op == "MOVE":
            addition = " Characters explicitly move between locations with clear transition evidence."
        elif op == "REVEAL":
            addition = " A concrete new fact is explicitly revealed to satisfy the symbolic repair target."
        elif op == "RESOLVE":
            addition = " The scene explicitly resolves the active local tension."
        else:
            addition = " The scene explicitly manifests conflict escalation with clear causal action."
        if addition.strip().lower() in text.lower():
            return text
        return f"{text.rstrip()} {addition.strip()}"

    def _build_memory_snapshot(
        self,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
    ) -> str:
        char_lines: List[str] = []
        for cid in scene_plan.involved_characters:
            profile = static_memory.characterization.character_profiles.get(cid)
            dyn = dynamic_memory.characterization.entity_store.get(cid)
            if profile is None and dyn is None:
                continue
            canonical = profile.canonical_name if profile else dyn.name
            role = profile.role if profile else "unknown"
            status = dyn.status if dyn else "unknown"
            location = dyn.location if dyn else "unknown"
            char_lines.append(f"- {cid}: {canonical}, role={role}, status={status}, location={location}")
        if not char_lines:
            char_lines = ["- (no explicit character snapshot)"]

        recent_events = dynamic_memory.timeline_plot.event_timeline[-3:]
        event_lines = [f"- {ev.event_id}: {ev.description[:100]}" for ev in recent_events] or ["- (no recent events)"]

        return (
            "Characters:\n"
            f"{chr(10).join(char_lines)}\n"
            "Recent events:\n"
            f"{chr(10).join(event_lines)}"
        )

    def _build_canonical_entity_table(
        self,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        scene_plan: ScenePlan,
        max_rows: int = 12,
    ) -> Dict[str, Dict[str, object]]:
        table: Dict[str, Dict[str, object]] = {}
        ordered_ids = list(dict.fromkeys(list(scene_plan.involved_characters) + list(scene_plan.optional_characters)))
        for entity_id in ordered_ids[:max_rows]:
            profile = static_memory.characterization.character_profiles.get(entity_id)
            if profile is None:
                continue
            state = dynamic_memory.characterization.entity_store.get(entity_id)
            aliases = [str(x).strip() for x in list(profile.aliases or []) if str(x).strip()]
            if profile.canonical_name and profile.canonical_name not in aliases:
                aliases.insert(0, profile.canonical_name)
            dedup_aliases: List[str] = []
            seen = set()
            for alias in aliases:
                key = alias.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup_aliases.append(alias)
            table[str(entity_id)] = {
                "canonical": str(profile.canonical_name),
                "aliases": dedup_aliases[:6],
                "current_name": str(state.name) if state is not None else "",
                "current_location": str(state.location) if state is not None else "",
            }
        return table

    def _format_canonical_entity_table(self, table: Dict[str, Dict[str, object]], max_rows: int = 8) -> str:
        if not table:
            return "- (empty)"
        lines: List[str] = []
        for idx, (entity_id, payload) in enumerate(table.items()):
            if idx >= max_rows:
                break
            lines.append(
                f"- {entity_id}: canonical={payload.get('canonical', '')}; aliases={list(payload.get('aliases', []))}; "
                f"current_name={payload.get('current_name', '')}; current_location={payload.get('current_location', '')}"
            )
        return "\n".join(lines) if lines else "- (empty)"

    def _build_dual_summary_for_repair(self, violations: Sequence[ConstraintViolation]) -> Dict[str, object]:
        symbolic_errors = 0
        symbolic_warnings = 0
        neural_structured = 0
        for item in violations:
            rule = str(item.rule_type or "").strip().lower()
            sev = str(item.severity or "warning").strip().lower()
            if rule.startswith("neural_"):
                neural_structured += 1
                continue
            if sev == "error":
                symbolic_errors += 1
            else:
                symbolic_warnings += 1
        if symbolic_errors > 0 and neural_structured > 0:
            decision = "must_repair_dual_confirmed"
        elif symbolic_errors > 0:
            decision = "must_repair_symbolic_only"
        elif neural_structured > 0:
            decision = "neural_risk_watch"
        else:
            decision = "accept_or_minimal_edit"
        return {
            "decision": decision,
            "symbolic_error_count": symbolic_errors,
            "symbolic_warning_count": symbolic_warnings,
            "neural_structured_count": neural_structured,
            "dual_confirmed": bool(symbolic_errors > 0 and neural_structured > 0),
            "must_repair": bool(symbolic_errors > 0),
        }

    def _build_minimal_edit_anchor_lines(
        self,
        violations: Sequence[ConstraintViolation],
        max_lines: int = 12,
    ) -> List[str]:
        lines: List[str] = []
        for item in violations:
            related = [str(x).strip() for x in list(item.related_ids or []) if str(x).strip()][:4]
            anchors: List[str] = []
            for anchor in list(item.anchors or []):
                anchors.extend([str(s).strip() for s in list(anchor.sentence_ids or []) if str(s).strip()])
            uniq_anchors = list(dict.fromkeys(anchors))[:4]
            lines.append(
                f"- {item.rule_type}: related={related}; anchor_sentence_ids={uniq_anchors}; "
                f"hint={str(item.repair_hint or '').strip()}"
            )
            if len(lines) >= max_lines:
                break
        return lines or ["- no explicit anchor found; edit only the smallest conflicting span."]

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _drop_forbidden_sentences(self, sentences: List[str], scene_plan: ScenePlan) -> List[str]:
        lowered_forbidden = [token.lower().strip() for token in scene_plan.forbidden_constraints if token.strip()]
        if not lowered_forbidden:
            return list(sentences)
        kept: List[str] = []
        for sent in sentences:
            lower = sent.lower()
            if any(token in lower for token in lowered_forbidden):
                continue
            kept.append(sent)
        return kept

    def _canonicalize_character_names(
        self,
        text: str,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        violations: List[ConstraintViolation],
    ) -> str:
        rewritten = text
        canonical_by_id = {
            cid: profile.canonical_name
            for cid, profile in static_memory.characterization.character_profiles.items()
        }
        for violation in violations:
            if violation.rule_type != "character_consistency":
                continue
            related = [rid for rid in violation.related_ids if rid in canonical_by_id]
            for char_id in related:
                canonical = canonical_by_id[char_id]
                pattern = rf"\b{re.escape(char_id.replace('char_', ''))}\b"
                rewritten = re.sub(pattern, canonical, rewritten, flags=re.IGNORECASE)
        for cid in scene_plan.involved_characters:
            if cid not in canonical_by_id:
                continue
            canonical = canonical_by_id[cid]
            rewritten = re.sub(
                rf"\b{re.escape(cid)}\b",
                canonical,
                rewritten,
                flags=re.IGNORECASE,
            )
        return rewritten

    def _repair_world_violations(
        self,
        text: str,
        scene_plan: ScenePlan,
        dynamic_memory: DynamicMemory,
        violations: List[ConstraintViolation],
    ) -> str:
        if not any(v.rule_type == "world_rule_consistency" for v in violations):
            return text

        rewritten = text
        for forbidden in scene_plan.forbidden_constraints:
            token = forbidden.strip()
            if token:
                rewritten = re.sub(re.escape(token), "", rewritten, flags=re.IGNORECASE)

        rewritten = re.sub(r"\bteleport(?:s|ed|ing)?\b", "moves with a clear transition", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bwithout any transition\b", "after a clear transition", rewritten, flags=re.IGNORECASE)

        world_state = (dynamic_memory.world_setting.current_setting_state or "").strip()
        if world_state and world_state.lower() not in rewritten.lower():
            rewritten = f"In {world_state}, {rewritten[0].lower() + rewritten[1:]}" if rewritten else f"In {world_state}, events continue."
        return rewritten

    def _repair_timeline_violations(
        self,
        text: str,
        scene_plan: ScenePlan,
        violations: List[ConstraintViolation],
    ) -> str:
        if not any(v.rule_type == "timeline_consistency" for v in violations):
            return text
        rewritten = re.sub(r"\b(back then|earlier before now|before that same moment)\b", "then", text, flags=re.IGNORECASE)
        if not re.search(r"\b(then|after|later|next)\b", rewritten, flags=re.IGNORECASE):
            rewritten = f"Then, {rewritten[0].lower() + rewritten[1:]}" if rewritten else f"Then, {scene_plan.objective}"
        return rewritten

    def _repair_character_violations(
        self,
        text: str,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        violations: List[ConstraintViolation],
    ) -> str:
        if not any(v.rule_type == "character_consistency" for v in violations):
            return text

        rewritten = text
        involved_names: List[str] = []
        for cid in scene_plan.involved_characters:
            profile = static_memory.characterization.character_profiles.get(cid)
            if profile is not None:
                involved_names.append(profile.canonical_name)
        if involved_names and not any(name.lower() in rewritten.lower() for name in involved_names):
            mention = ", ".join(involved_names[:2])
            rewritten = f"{mention} take part directly. {rewritten}"
        return rewritten

    def _enforce_required_constraints(
        self,
        text: str,
        scene_plan: ScenePlan,
        violations: List[ConstraintViolation],
    ) -> str:
        required_missing = any("required constraint" in v.message.lower() for v in violations)
        rewritten = text
        for constraint in scene_plan.required_constraints:
            token = constraint.strip()
            if not token:
                continue
            if token.lower() in rewritten.lower():
                continue
            if required_missing:
                rewritten = f"{rewritten} {token}."
        return rewritten

    def _remove_forbidden_constraints(self, text: str, scene_plan: ScenePlan) -> str:
        rewritten = text
        for forbidden in scene_plan.forbidden_constraints:
            token = forbidden.strip()
            if not token:
                continue
            rewritten = re.sub(re.escape(token), "", rewritten, flags=re.IGNORECASE)
        return rewritten
