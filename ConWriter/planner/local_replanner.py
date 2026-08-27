"""Lightweight local replanner for next-scene subplan repair."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from ConWriter.pipeline.weighted_constraints import weighted_violation_score
from ConWriter.utils.types import (
    ConstraintViolation,
    DynamicMemory,
    LocalReplanResult,
    ScenePlan,
    StaticMemory,
)


class LocalReplanner:
    """Revise the next 1..k scene plans when current plan becomes locally infeasible."""

    def __init__(self, window_scenes: int = 2):
        self.window_scenes = max(1, int(window_scenes))

    def replan(
        self,
        current_scene: ScenePlan,
        all_scenes: Sequence[ScenePlan],
        current_scene_idx: int,
        dynamic_memory: DynamicMemory,
        static_memory: StaticMemory,
        violations: Sequence[ConstraintViolation],
        failure_reason: str,
        weighted_future_constraints: Sequence[str] | None = None,
    ) -> LocalReplanResult:
        replan_id = f"local_replan_{current_scene.scene_id}_{current_scene_idx:03d}"
        future_start = int(current_scene_idx) + 1
        future_end = min(len(all_scenes), future_start + self.window_scenes)
        future_slice = list(all_scenes[future_start:future_end])
        if not future_slice:
            return LocalReplanResult(
                replan_id=replan_id,
                triggered=True,
                applied=False,
                rationale="No future scenes available for local replan.",
            )

        unresolved = [str(item) for item in dynamic_memory.timeline_plot.pending_constraints if str(item).strip()]
        unresolved = unresolved[-6:]
        static_char_ids = list(static_memory.characterization.character_profiles.keys())
        static_char_id_set = set(static_char_ids)
        active_chars = [
            cid
            for cid, ent in dynamic_memory.characterization.entity_store.items()
            if ent.status != "removed" and cid in static_char_id_set
        ]
        if not active_chars:
            active_chars = list(static_char_ids)

        violation_tokens = self._collect_violation_tokens(violations)
        weighted_sorted = sorted(
            list(violations),
            key=lambda v: (-(float(v.constraint_weight)), int(v.constraint_tier), -int(v.is_hard)),
        )
        high_priority_tokens = [
            self._extract_priority_constraint(v)
            for v in weighted_sorted
            if (v.is_hard or v.constraint_tier <= 2)
        ]
        high_priority_tokens = [x for x in high_priority_tokens if x]
        weighted_future_constraints = [str(x).strip() for x in (weighted_future_constraints or []) if str(x).strip()]
        revised_scenes: List[ScenePlan] = []
        changed_scene_ids: List[str] = []
        sacrificed_preferences: List[str] = []
        preserved_high_priority: List[str] = []
        for rel_idx, scene in enumerate(future_slice):
            required_constraints = self._merge_constraints(
                scene.required_constraints,
                unresolved[:2],
                violation_tokens[:2],
                high_priority_tokens[:3],
                weighted_future_constraints[:2],
            )
            forbidden_constraints = [c for c in scene.forbidden_constraints if c not in required_constraints]
            if not forbidden_constraints:
                forbidden_constraints = list(scene.forbidden_constraints)

            required_characters = list(scene.required_characters)
            if required_characters:
                required_characters = [cid for cid in required_characters if cid in static_char_id_set]
                if not required_characters:
                    required_characters = active_chars[:2] if active_chars else static_char_ids[:2]
            elif active_chars:
                required_characters = active_chars[:2]
            optional_characters = [
                cid for cid in active_chars if cid in static_char_id_set and cid not in required_characters
            ][:2]
            involved = required_characters + [
                cid
                for cid in scene.involved_characters
                if cid in static_char_id_set and cid not in required_characters
            ]
            involved.extend([cid for cid in optional_characters if cid not in involved])

            bridge = (
                f"Follow-up after {current_scene.scene_id}: preserve continuity with current memory state."
                if rel_idx == 0
                else "Preserve continuity with revised previous scene."
            )
            objective = f"{scene.objective} {bridge}".strip()
            expected_changes = list(scene.expected_state_changes)
            if "local_replan_bridge" not in expected_changes:
                expected_changes.append("local_replan_bridge")
            if unresolved:
                expected_changes.append(f"pending_constraint_progress:{unresolved[0]}")
            for token in high_priority_tokens[:2]:
                marker = f"preserve_high_priority:{token}"
                if marker not in expected_changes:
                    expected_changes.append(marker)

            dependency_scenes = list(scene.dependency_scenes)
            if rel_idx == 0 and current_scene.scene_id not in dependency_scenes:
                dependency_scenes.append(current_scene.scene_id)

            original_soft = [x for x in scene.must_keep_constraints if self._looks_soft_preference(x)]
            revised_soft = [x for x in required_constraints if self._looks_soft_preference(x)]
            dropped_soft = [x for x in original_soft if x not in revised_soft]
            sacrificed_preferences.extend(dropped_soft)
            preserved_high_priority.extend([x for x in high_priority_tokens[:3] if x in required_constraints])

            revised = replace(
                scene,
                objective=objective,
                required_constraints=required_constraints,
                forbidden_constraints=forbidden_constraints,
                required_characters=required_characters,
                optional_characters=optional_characters,
                involved_characters=involved,
                expected_state_changes=expected_changes,
                dependency_scenes=dependency_scenes,
            )
            revised_scenes.append(revised)
            changed_scene_ids.append(scene.scene_id)

        return LocalReplanResult(
            replan_id=replan_id,
            triggered=True,
            applied=True,
            changed_scene_ids=changed_scene_ids,
            rationale=(
                "Local replan triggered after patch failure/plan deviation: "
                f"{failure_reason or 'unknown reason'}"
            ),
            impact_summary={
                "window_size": len(revised_scenes),
                "unresolved_constraints_used": unresolved[:2],
                "violation_tokens_used": violation_tokens[:2],
                "active_character_pool_size": len(active_chars),
                "weighted_violation_score": float(weighted_violation_score(violations)),
                "preserved_high_priority_constraints": sorted(set(preserved_high_priority)),
                "sacrificed_low_priority_preferences": sorted(set(sacrificed_preferences)),
                "weighted_future_constraints_used": weighted_future_constraints[:2],
            },
            revised_scenes=revised_scenes,
        )

    def _merge_constraints(self, *groups: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        merged: List[str] = []
        for group in groups:
            for raw in group:
                item = str(raw).strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                merged.append(item)
        return merged

    def _collect_violation_tokens(self, violations: Sequence[ConstraintViolation]) -> List[str]:
        tokens: List[str] = []
        for violation in violations:
            if violation.rule_type:
                tokens.append(f"repair:{violation.rule_type}")
            if violation.repair_hint:
                tokens.append(violation.repair_hint)
            for anchor in violation.anchors:
                tokens.extend(anchor.related_entity_ids[:1])
                tokens.extend(anchor.related_event_ids[:1])
                for rel in anchor.related_relation_ids[:1]:
                    tokens.append(rel)
        return self._merge_constraints(tokens)

    def _extract_priority_constraint(self, violation: ConstraintViolation) -> str:
        msg = (violation.message or "").strip()
        if "'" in msg:
            parts = msg.split("'")
            if len(parts) >= 2 and parts[1].strip():
                return parts[1].strip()
        if violation.related_ids:
            return str(violation.related_ids[0])
        return msg[:80]

    def _looks_soft_preference(self, text: str) -> bool:
        lowered = str(text or "").lower()
        return any(token in lowered for token in ("optional", "style", "preference", "nice to have", "tone"))
