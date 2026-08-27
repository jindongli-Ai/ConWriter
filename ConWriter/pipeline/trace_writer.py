"""Scene-level trace exporter for incremental pipeline debugging."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ConWriter.config.schema import TraceConfig


class TraceWriter:
    """Write per-story incremental trace artifacts."""

    def __init__(self, cfg: TraceConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.root = Path(cfg.output_dir).resolve()
        self.story_dir: Path | None = None

    def start_story(
        self,
        experiment_name: str,
        prompt_id: str,
        plan: Any,
        static_memory: Any,
    ) -> None:
        if not self.enabled:
            return
        safe_prompt = str(prompt_id).replace("/", "_")
        self.story_dir = self.root / experiment_name / safe_prompt
        self.story_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("plan.json", plan)
        self._write_json("static_memory.json", static_memory)

    def write_scene_prompt(self, scene_id: str, prompt_text: str) -> None:
        if not self.enabled or not self.cfg.export_prompt_text:
            return
        self._write_text(f"{scene_id}_prompt.txt", prompt_text or "")

    def write_scene_memory_before(self, scene_id: str, memory: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_memory_before.json", memory)

    def write_scene_extraction(self, scene_id: str, extraction: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_extraction.json", extraction)

    def write_scene_violations(self, scene_id: str, violations: Any, report: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "violations": self._to_jsonable(violations),
            "report": self._to_jsonable(report),
        }
        self._write_json(f"{scene_id}_violations.json", payload)

    def write_scene_transition(self, scene_id: str, transition_payload: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_transition.json", transition_payload)

    def write_scene_repair(self, scene_id: str, attempt: int, repaired_text: str) -> None:
        if not self.enabled:
            return
        self._write_text(f"{scene_id}_repair_round_{attempt}.txt", repaired_text or "")

    def write_scene_patch_plan(self, scene_id: str, patch_plan: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_patch_plan.json", patch_plan)

    def write_scene_patch(self, scene_id: str, attempt: int, patch_payload: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_patch_round_{attempt}.json", patch_payload)

    def write_scene_replan(self, scene_id: str, replan_payload: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_local_replan.json", replan_payload)

    def write_scene_memory_after(self, scene_id: str, memory: Any) -> None:
        if not self.enabled:
            return
        self._write_json(f"{scene_id}_memory_after.json", memory)

    def write_metrics_summary(self, metrics_payload: Any) -> None:
        if not self.enabled or not self.cfg.export_metrics:
            return
        self._write_json("metrics_summary.json", metrics_payload)

    def write_final_story(self, story_text: str) -> None:
        if not self.enabled:
            return
        self._write_text("final_story.txt", story_text or "")

    def _write_json(self, filename: str, obj: Any) -> None:
        if self.story_dir is None:
            return
        path = self.story_dir / filename
        path.write_text(
            json.dumps(self._to_jsonable(obj), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_text(self, filename: str, text: str) -> None:
        if self.story_dir is None:
            return
        path = self.story_dir / filename
        path.write_text(text, encoding="utf-8")

    def _to_jsonable(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, list):
            return [self._to_jsonable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        return obj
