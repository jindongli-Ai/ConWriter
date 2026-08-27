"""Estimate downstream conflict risk for patch trajectory scoring."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from ConWriter.utils.types import (
    ConstraintViolation,
    DynamicMemory,
    FutureConflictEstimate,
    ScenePlan,
    WeightedConstraintItem,
)


class FutureConflictEstimator:
    """Predict future conflict penalty for next 1..k scenes."""

    def estimate(
        self,
        patched_scene_text: str,
        patched_dynamic_memory: DynamicMemory,
        future_scenes: Sequence[ScenePlan],
        active_weighted_constraints: Iterable[ConstraintViolation],
        weighted_future_constraints: Iterable[WeightedConstraintItem] | None = None,
    ) -> FutureConflictEstimate:
        text = (patched_scene_text or "").lower()
        future = list(future_scenes)[:2]
        if not future:
            return FutureConflictEstimate()

        active = list(active_weighted_constraints)
        weighted_future = list(weighted_future_constraints or [])
        breakdown: Dict[str, float] = {
            "entity_state_propagation_conflict": 0.0,
            "required_future_event_incompatibility": 0.0,
            "temporal_propagation_conflict": 0.0,
            "relation_continuity_conflict": 0.0,
            "future_plan_executability_degradation": 0.0,
        }
        impacted: List[str] = []
        critical_risks: List[str] = []
        known_entities = set(patched_dynamic_memory.characterization.entity_store.keys())
        unresolved_pending = set(
            str(item).strip().lower()
            for item in patched_dynamic_memory.timeline_plot.pending_constraints
            if str(item).strip()
        )

        for scene in future:
            scene_penalty = 0.0
            required_chars = scene.required_characters or scene.involved_characters
            for cid in required_chars:
                token = cid.replace("char_", "").replace("_", " ").lower()
                if (cid not in known_entities) and token and token not in text:
                    breakdown["entity_state_propagation_conflict"] += 1.2
                    scene_penalty += 1.2
                    critical_risks.append(f"{scene.scene_id}:missing_character:{cid}")

            for req in scene.required_constraints + scene.must_keep_constraints:
                rtxt = str(req).strip().lower()
                if not rtxt:
                    continue
                # If unresolved pending and still absent, penalize executability.
                if (rtxt in unresolved_pending or self._looks_hard_constraint(rtxt)) and rtxt not in text:
                    breakdown["future_plan_executability_degradation"] += 1.4
                    scene_penalty += 1.4
                    critical_risks.append(f"{scene.scene_id}:missing_required:{req}")

            for forb in scene.forbidden_constraints:
                ftxt = str(forb).strip().lower()
                if ftxt and ftxt in text:
                    breakdown["required_future_event_incompatibility"] += 1.6
                    scene_penalty += 1.6
                    critical_risks.append(f"{scene.scene_id}:forbidden_preintroduced:{forb}")

            if any(tok in text for tok in ("rewind", "earlier than before", "before all this")):
                breakdown["temporal_propagation_conflict"] += 1.3
                scene_penalty += 1.3
                critical_risks.append(f"{scene.scene_id}:temporal_regression_signal")

            if any(tok in text for tok in ("betray", "hostile", "enemy now")) and any(
                key in " ".join(scene.required_constraints).lower() for key in ("trust", "ally", "reconcile")
            ):
                breakdown["relation_continuity_conflict"] += 1.1
                scene_penalty += 1.1
                critical_risks.append(f"{scene.scene_id}:relation_discontinuity")

            if scene_penalty > 0.0:
                impacted.append(scene.scene_id)

        for violation in active:
            if not violation.is_hard:
                continue
            token = self._extract_required_token(violation.message)
            if token and token not in text:
                breakdown["future_plan_executability_degradation"] += 0.8 * max(1.0, violation.constraint_weight / 4.0)
                critical_risks.append(f"active_violation_at_risk:{token}")

        for item in weighted_future:
            if item.tier != 1:
                continue
            txt = item.text.lower().replace("forbid: ", "")
            if item.is_hard and txt and txt not in text:
                breakdown["future_plan_executability_degradation"] += 0.25 * float(item.weight)
                critical_risks.append(f"weighted_future_hard:{item.text}")

        total = float(sum(breakdown.values()))
        return FutureConflictEstimate(
            predicted_future_conflict_penalty=total,
            breakdown=breakdown,
            impacted_future_scene_ids=sorted(set(impacted)),
            critical_future_constraints_at_risk=sorted(set(critical_risks))[:8],
        )

    def _looks_hard_constraint(self, text: str) -> bool:
        return any(token in text for token in ("must", "required", "cannot", "must not", "never"))

    def _extract_required_token(self, message: str) -> str:
        msg = (message or "").lower()
        if "required constraint" in msg and "'" in msg:
            parts = msg.split("'")
            if len(parts) >= 2:
                return parts[1].strip()
        if "missing required events" in msg:
            return msg
        return ""

