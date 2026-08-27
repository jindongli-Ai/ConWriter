"""Repair metrics and oscillation tracking for patch-first incremental pipeline."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from ConWriter.utils.types import ConstraintViolation, LocalReplanResult


class RepairOscillationDetector:
    """Detect repeated repair patterns that indicate dead loops."""

    def __init__(self, window: int = 4, threshold: int = 2):
        self.window = max(2, int(window))
        self.threshold = max(2, int(threshold))
        self._history: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=self.window))

    def observe(
        self,
        scene_id: str,
        target_sentence_ids: Sequence[str],
        violations: Sequence[ConstraintViolation],
    ) -> Dict[str, object]:
        rules = sorted({v.rule_type for v in violations if v.rule_type})
        anchors = sorted(set(target_sentence_ids))
        signature = f"{'|'.join(anchors)}##{'|'.join(rules)}"
        buf = self._history[scene_id]
        buf.append(signature)
        repeats = sum(1 for item in buf if item == signature)
        return {
            "signature": signature,
            "repeats": repeats,
            "history_size": len(buf),
            "detected": repeats >= self.threshold and len(buf) >= self.threshold,
        }


class RepairMetricsTracker:
    """Collect per-scene and aggregate metrics for patch/replan execution."""

    def __init__(self):
        self._scene: Dict[str, Dict[str, object]] = {}
        self._agg: Dict[str, float] = defaultdict(float)

    def start_scene(self, scene_id: str) -> None:
        self._scene[scene_id] = {
            "anchor_count": 0,
            "patchable_violations": 0,
            "fatal_violations": 0,
            "execution_spec_violations": 0,
            "sentence_patch_rounds": 0,
            "paragraph_patch_rounds": 0,
            "scene_regen_rounds": 0,
            "sentence_patch_success": 0,
            "paragraph_patch_success": 0,
            "scene_regen_fallback": 0,
            "needs_replan": 0,
            "local_replan_applied": 0,
            "unchanged_ratios": [],
            "protected_integrity_passes": 0,
            "protected_integrity_fails": 0,
            "spillover_count": 0,
            "oscillation_count": 0,
            "stability_scores": [],
        }
        self._agg["scene_count"] += 1

    def record_validation(self, scene_id: str, violations: Sequence[ConstraintViolation]) -> None:
        row = self._scene.setdefault(scene_id, {})
        row["anchor_count"] = int(row.get("anchor_count", 0)) + sum(len(v.anchors) for v in violations)
        row["patchable_violations"] = int(row.get("patchable_violations", 0)) + sum(1 for v in violations if v.patchable)
        row["fatal_violations"] = int(row.get("fatal_violations", 0)) + sum(1 for v in violations if v.fatal)
        row["execution_spec_violations"] = int(row.get("execution_spec_violations", 0)) + sum(
            1 for v in violations if v.rule_type == "operator_validity"
        )

    def record_patch_round(self, scene_id: str, scope: str, metadata: Dict[str, object]) -> None:
        row = self._scene.setdefault(scene_id, {})
        scope_key = f"{scope}_patch_rounds"
        row[scope_key] = int(row.get(scope_key, 0)) + 1
        unchanged = metadata.get("unchanged_ratio")
        if isinstance(unchanged, (int, float)):
            ratios = row.setdefault("unchanged_ratios", [])
            if isinstance(ratios, list):
                ratios.append(float(unchanged))
        integrity_pass = metadata.get("protected_integrity_pass")
        if integrity_pass is True:
            row["protected_integrity_passes"] = int(row.get("protected_integrity_passes", 0)) + 1
        elif integrity_pass is False:
            row["protected_integrity_fails"] = int(row.get("protected_integrity_fails", 0)) + 1
        spillover = metadata.get("spillover_count")
        if isinstance(spillover, int):
            row["spillover_count"] = int(row.get("spillover_count", 0)) + spillover
        stability = metadata.get("stability_score")
        if isinstance(stability, (int, float)):
            scores = row.setdefault("stability_scores", [])
            if isinstance(scores, list):
                scores.append(float(stability))

    def record_patch_success(self, scene_id: str, scope: str) -> None:
        row = self._scene.setdefault(scene_id, {})
        key = f"{scope}_patch_success"
        row[key] = int(row.get(key, 0)) + 1

    def record_scene_regen(self, scene_id: str) -> None:
        row = self._scene.setdefault(scene_id, {})
        row["scene_regen_fallback"] = int(row.get("scene_regen_fallback", 0)) + 1
        row["scene_regen_rounds"] = int(row.get("scene_regen_rounds", 0)) + 1

    def record_needs_replan(self, scene_id: str) -> None:
        row = self._scene.setdefault(scene_id, {})
        row["needs_replan"] = int(row.get("needs_replan", 0)) + 1

    def record_replan_result(self, scene_id: str, result: LocalReplanResult) -> None:
        row = self._scene.setdefault(scene_id, {})
        if result.applied:
            row["local_replan_applied"] = int(row.get("local_replan_applied", 0)) + 1

    def record_oscillation(self, scene_id: str) -> None:
        row = self._scene.setdefault(scene_id, {})
        row["oscillation_count"] = int(row.get("oscillation_count", 0)) + 1

    def finalize_scene(self, scene_id: str, accepted: bool) -> Dict[str, object]:
        row = self._scene.get(scene_id, {})
        row["accepted"] = bool(accepted)
        self._scene[scene_id] = row
        self._agg["accepted_scene_count"] += 1 if accepted else 0
        self._agg["rejected_scene_count"] += 0 if accepted else 1
        return row

    def summarize(self) -> Dict[str, object]:
        scenes = list(self._scene.values())
        unchanged: List[float] = []
        stability: List[float] = []
        for row in scenes:
            ratios = row.get("unchanged_ratios", [])
            if isinstance(ratios, list):
                unchanged.extend(float(x) for x in ratios)
            scores = row.get("stability_scores", [])
            if isinstance(scores, list):
                stability.extend(float(x) for x in scores)
        return {
            "num_scenes": int(self._agg.get("scene_count", 0)),
            "num_scenes_accepted": int(self._agg.get("accepted_scene_count", 0)),
            "num_scenes_rejected": int(self._agg.get("rejected_scene_count", 0)),
            "anchor_count_total": int(sum(int(row.get("anchor_count", 0)) for row in scenes)),
            "patchable_violations_total": int(sum(int(row.get("patchable_violations", 0)) for row in scenes)),
            "fatal_violations_total": int(sum(int(row.get("fatal_violations", 0)) for row in scenes)),
            "execution_spec_violations_total": int(
                sum(int(row.get("execution_spec_violations", 0)) for row in scenes)
            ),
            "sentence_patch_rounds_total": int(sum(int(row.get("sentence_patch_rounds", 0)) for row in scenes)),
            "paragraph_patch_rounds_total": int(sum(int(row.get("paragraph_patch_rounds", 0)) for row in scenes)),
            "scene_regen_fallback_total": int(sum(int(row.get("scene_regen_fallback", 0)) for row in scenes)),
            "needs_replan_total": int(sum(int(row.get("needs_replan", 0)) for row in scenes)),
            "local_replan_applied_total": int(sum(int(row.get("local_replan_applied", 0)) for row in scenes)),
            "protected_integrity_pass_total": int(
                sum(int(row.get("protected_integrity_passes", 0)) for row in scenes)
            ),
            "protected_integrity_fail_total": int(
                sum(int(row.get("protected_integrity_fails", 0)) for row in scenes)
            ),
            "spillover_count_total": int(sum(int(row.get("spillover_count", 0)) for row in scenes)),
            "oscillation_count_total": int(sum(int(row.get("oscillation_count", 0)) for row in scenes)),
            "unchanged_ratio_mean": (sum(unchanged) / len(unchanged)) if unchanged else 0.0,
            "stability_score_mean": (sum(stability) / len(stability)) if stability else 0.0,
            "scene_metrics": self._scene,
        }
