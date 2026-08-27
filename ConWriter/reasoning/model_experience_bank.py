"""Lightweight model-specific experience bank for preventive guidance."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ConWriter.config.schema import ExperienceBankConfig


class ModelExperienceBank:
    """Cross-sample online experience bank keyed by base model id."""

    def __init__(self, config: ExperienceBankConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("ConWriter.model_experience_bank")
        self._store_path = Path(config.storage_path)
        self._loaded = False
        self._bank: Dict[str, List[Dict[str, object]]] = {}

    def retrieve_top_k(
        self,
        *,
        model_id: str,
        task_type: str,
        generation_stage: str,
        conflict_type: str = "",
        top_k: int = 3,
        exclude_sample_id: str = "",
    ) -> List[Dict[str, object]]:
        if (not self.config.enabled) or top_k <= 0:
            return []
        self._ensure_loaded()
        model_key = self._normalize_model_id(model_id)
        task_key = self._normalize_task_type(task_type)
        stage_key = self._normalize_stage(generation_stage)
        conflict_key = self._normalize_conflict_type(conflict_type)
        candidate_keys = self._candidate_model_keys(model_key)
        scored: List[Tuple[float, Dict[str, object]]] = []
        for key in candidate_keys:
            for entry in self._bank.get(key, []):
                if exclude_sample_id and str(entry.get("last_sample_id", "")) == str(exclude_sample_id):
                    continue
                entry_stage = self._normalize_stage(entry.get("generation_stage", "scene_generation"))
                if entry_stage not in {stage_key, "any"}:
                    continue
                score = self._score_entry(
                    entry=entry,
                    task_type=task_key,
                    conflict_type=conflict_key,
                    stage=stage_key,
                    exact_model=(key == model_key),
                )
                if score <= 0.0:
                    continue
                scored.append((score, dict(entry)))
        if not scored:
            return []
        scored.sort(
            key=lambda item: (
                float(item[0]),
                float(item[1].get("count", 0)),
                float(item[1].get("confidence", 0.0)),
                float(item[1].get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        cap = int(max(1, min(int(top_k), int(self._cap_for_stage(stage_key)))))
        out: List[Dict[str, object]] = []
        seen_keys: set[Tuple[str, str, str]] = set()
        for _, item in scored:
            sig = (
                str(item.get("generation_stage", "")),
                str(item.get("failure_pattern", "")),
                str(item.get("prevention_guidance", "")),
            )
            if sig in seen_keys:
                continue
            seen_keys.add(sig)
            out.append(
                {
                    "model_id": str(item.get("model_id", "")),
                    "task_type": str(item.get("task_type", "generic")),
                    "conflict_type": str(item.get("conflict_type", "any")),
                    "generation_stage": str(item.get("generation_stage", "scene_generation")),
                    "failure_pattern": str(item.get("failure_pattern", "")),
                    "prevention_guidance": str(item.get("prevention_guidance", "")),
                    "count": int(item.get("count", 1) or 1),
                    "confidence": float(item.get("confidence", 0.35) or 0.35),
                }
            )
            if len(out) >= cap:
                break
        return out

    def format_prompt_cautions(self, items: Sequence[Dict[str, object]], max_items: int = 3) -> List[str]:
        out: List[str] = []
        for item in list(items)[: max(0, int(max_items))]:
            pattern = str(item.get("failure_pattern", "")).strip()
            guidance = str(item.get("prevention_guidance", "")).strip()
            if not pattern and not guidance:
                continue
            if pattern and guidance:
                out.append(f"{pattern}; fix: {guidance}")
            elif guidance:
                out.append(guidance)
            else:
                out.append(pattern)
        return out[: max(0, int(max_items))]

    def update_from_sample(
        self,
        *,
        model_id: str,
        sample_id: str,
        task_type: str,
        scene_records: Iterable[Dict[str, object]],
        length_feedback: Dict[str, object] | None = None,
    ) -> int:
        if not self.config.enabled:
            return 0
        self._ensure_loaded()
        model_key = self._normalize_model_id(model_id)
        task_key = self._normalize_task_type(task_type)
        events: List[Tuple[str, str, str, str]] = []
        for row in list(scene_records):
            if not isinstance(row, dict):
                continue
            conflict = self._normalize_conflict_type(row.get("rewrite_conflict_type", "unknown"))
            if (conflict == "execution_spec_conflict") or (not bool(row.get("rewrite_realizes_required_state_change", True))):
                events.append(
                    (
                        "scene_generation",
                        "execution_spec_conflict",
                        "tends to omit required state realization",
                        "explicitly realize required state changes in surface text",
                    )
                )
            if (conflict in {"constraint_conflict", "mixed_conflict"}) or (
                not bool(row.get("rewrite_removes_forbidden_state", True))
            ):
                events.append(
                    (
                        "retry_rewrite",
                        "constraint_conflict",
                        "tends to retain forbidden state",
                        "remove old conflicting state before adding new state",
                    )
                )
            if (conflict == "operator_post_state_conflict") or (
                not bool(row.get("rewrite_realizes_operator_post_state", True))
            ):
                events.append(
                    (
                        "retry_rewrite",
                        "operator_post_state_conflict",
                        "operator post-state often remains implicit",
                        "make operator-required post-state explicit in the rewritten text",
                    )
                )
            if conflict == "transition_conflict":
                events.append(
                    (
                        "scene_generation",
                        "transition_conflict",
                        "timeline transition often underspecified",
                        "keep transition cue explicit and causal order clear",
                    )
                )
            no_gain_reason = str(row.get("patch_no_gain_reason", "")).strip().lower()
            if "no_gain" in no_gain_reason or "ineffective" in no_gain_reason:
                events.append(
                    (
                        "retry_rewrite",
                        conflict or "unknown",
                        "repair retries often fail to change conflict-bearing spans",
                        "edit only conflict-bearing spans and preserve already satisfied states",
                    )
                )
        if isinstance(length_feedback, dict):
            if bool(length_feedback.get("under_generation_warning_triggered", False)):
                events.append(
                    (
                        "scene_generation",
                        "length_under_generation",
                        "under-generates target length",
                        "do not end early before unresolved threads are expanded",
                    )
                )
            if bool(length_feedback.get("premature_closure_warning_triggered", False)):
                events.append(
                    (
                        "scene_generation",
                        "premature_closure",
                        "tends to close narrative too early",
                        "avoid ending language until target length is near and threads are resolved",
                    )
                )
        update_count = 0
        for stage, conflict_type, pattern, guidance in self._dedup_events(events):
            update_count += self._upsert(
                model_key=model_key,
                sample_id=sample_id,
                task_type=task_key,
                generation_stage=stage,
                conflict_type=conflict_type,
                failure_pattern=pattern,
                prevention_guidance=guidance,
            )
        if update_count > 0:
            self._persist()
        return int(update_count)

    def most_common_failure_patterns_by_model(self, top_k: int = 5) -> Dict[str, List[Dict[str, object]]]:
        if not self.config.enabled:
            return {}
        self._ensure_loaded()
        out: Dict[str, List[Dict[str, object]]] = {}
        for model_key, rows in self._bank.items():
            ranked = sorted(
                list(rows),
                key=lambda item: (
                    int(item.get("count", 0)),
                    float(item.get("confidence", 0.0)),
                    float(item.get("updated_at", 0.0)),
                ),
                reverse=True,
            )
            out[model_key] = [
                {
                    "failure_pattern": str(item.get("failure_pattern", "")),
                    "count": int(item.get("count", 1) or 1),
                    "confidence": float(item.get("confidence", 0.35) or 0.35),
                }
                for item in ranked[: max(0, int(top_k))]
            ]
        return out

    def _upsert(
        self,
        *,
        model_key: str,
        sample_id: str,
        task_type: str,
        generation_stage: str,
        conflict_type: str,
        failure_pattern: str,
        prevention_guidance: str,
    ) -> int:
        if (not failure_pattern.strip()) or (not prevention_guidance.strip()):
            return 0
        rows = self._bank.setdefault(model_key, [])
        now = float(time.time())
        for entry in rows:
            if (
                str(entry.get("task_type", "generic")) == task_type
                and str(entry.get("generation_stage", "scene_generation")) == generation_stage
                and str(entry.get("conflict_type", "any")) == conflict_type
                and str(entry.get("failure_pattern", "")) == failure_pattern
                and str(entry.get("prevention_guidance", "")) == prevention_guidance
            ):
                count = int(entry.get("count", 1) or 1) + 1
                entry["count"] = int(count)
                entry["confidence"] = float(min(0.95, 0.30 + (0.08 * min(count, 8))))
                entry["last_sample_id"] = str(sample_id)
                entry["updated_at"] = now
                self._trim_model_bucket(model_key)
                return 1
        rows.append(
            {
                "model_id": model_key,
                "task_type": task_type,
                "conflict_type": conflict_type,
                "generation_stage": generation_stage,
                "failure_pattern": failure_pattern,
                "prevention_guidance": prevention_guidance,
                "count": 1,
                "confidence": 0.35,
                "last_sample_id": str(sample_id),
                "updated_at": now,
            }
        )
        self._trim_model_bucket(model_key)
        return 1

    def _trim_model_bucket(self, model_key: str) -> None:
        rows = self._bank.get(model_key, [])
        limit = max(8, int(self.config.max_entries_per_model))
        if len(rows) <= limit:
            return
        rows.sort(
            key=lambda item: (
                int(item.get("count", 0)),
                float(item.get("confidence", 0.0)),
                float(item.get("updated_at", 0.0)),
            ),
            reverse=True,
        )
        self._bank[model_key] = rows[:limit]

    def _cap_for_stage(self, stage: str) -> int:
        if stage == "retry_rewrite":
            return int(max(1, self.config.max_retrieved_retry_items))
        return int(max(1, self.config.max_retrieved_scene_items))

    def _score_entry(
        self,
        *,
        entry: Dict[str, object],
        task_type: str,
        conflict_type: str,
        stage: str,
        exact_model: bool,
    ) -> float:
        score = 0.0
        if exact_model:
            score += 1.4
        entry_task = self._normalize_task_type(entry.get("task_type", "generic"))
        if entry_task in {"generic", "any", task_type}:
            score += 1.0 if entry_task == task_type else 0.4
        entry_conflict = self._normalize_conflict_type(entry.get("conflict_type", "any"))
        if conflict_type:
            if entry_conflict in {"any", "unknown"}:
                score += 0.3
            elif entry_conflict == conflict_type:
                score += 1.5
        else:
            score += 0.2
        entry_stage = self._normalize_stage(entry.get("generation_stage", "scene_generation"))
        if entry_stage == stage:
            score += 1.0
        count = max(1, int(entry.get("count", 1) or 1))
        score += min(2.0, 0.2 * float(count))
        score += float(entry.get("confidence", 0.35) or 0.35)
        return float(score)

    def _candidate_model_keys(self, model_key: str) -> List[str]:
        out = [model_key]
        family = self._model_family(model_key)
        if family and family in self._bank and family not in out:
            out.append(family)
        for key in list(self._bank.keys()):
            if key in out:
                continue
            if key and key in model_key:
                out.append(key)
            elif model_key and model_key in key:
                out.append(key)
        return out

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._store_path.exists():
            self._bank = {}
            return
        try:
            payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning("experience bank load failed: %s", exc)
            self._bank = {}
            return
        models = payload.get("models", {}) if isinstance(payload, dict) else {}
        if not isinstance(models, dict):
            self._bank = {}
            return
        loaded: Dict[str, List[Dict[str, object]]] = {}
        for key, rows in models.items():
            if not isinstance(rows, list):
                continue
            model_key = self._normalize_model_id(key)
            loaded_rows: List[Dict[str, object]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pattern = str(row.get("failure_pattern", "")).strip()
                guidance = str(row.get("prevention_guidance", "")).strip()
                if (not pattern) or (not guidance):
                    continue
                loaded_rows.append(
                    {
                        "model_id": model_key,
                        "task_type": self._normalize_task_type(row.get("task_type", "generic")),
                        "conflict_type": self._normalize_conflict_type(row.get("conflict_type", "any")),
                        "generation_stage": self._normalize_stage(row.get("generation_stage", "scene_generation")),
                        "failure_pattern": pattern,
                        "prevention_guidance": guidance,
                        "count": int(max(1, int(row.get("count", 1) or 1))),
                        "confidence": float(max(0.0, min(1.0, float(row.get("confidence", 0.35) or 0.35)))),
                        "last_sample_id": str(row.get("last_sample_id", "")),
                        "updated_at": float(row.get("updated_at", 0.0) or 0.0),
                    }
                )
            if loaded_rows:
                loaded[model_key] = loaded_rows[: int(max(1, self.config.max_entries_per_model))]
        self._bank = loaded

    def _persist(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": self._bank}
        try:
            self._store_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.logger.warning("experience bank save failed: %s", exc)

    @staticmethod
    def _dedup_events(events: Sequence[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        seen: set[Tuple[str, str, str, str]] = set()
        out: List[Tuple[str, str, str, str]] = []
        for item in events:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @staticmethod
    def _normalize_model_id(model_id: object) -> str:
        token = str(model_id or "").strip().lower()
        return token or "unknown_model"

    @staticmethod
    def _normalize_task_type(task_type: object) -> str:
        token = str(task_type or "").strip().lower()
        return token or "generic"

    @staticmethod
    def _normalize_conflict_type(conflict_type: object) -> str:
        token = str(conflict_type or "").strip().lower()
        return token or "any"

    @staticmethod
    def _normalize_stage(stage: object) -> str:
        token = str(stage or "").strip().lower()
        if token in {"scene", "scene_generation"}:
            return "scene_generation"
        if token in {"retry", "retry_rewrite", "rewrite"}:
            return "retry_rewrite"
        if token == "any":
            return "any"
        return "scene_generation"

    @staticmethod
    def _model_family(model_id: str) -> str:
        token = str(model_id or "").lower()
        if "gpt" in token:
            return "gpt"
        if "gemini" in token:
            return "gemini"
        if "qwen" in token:
            return "qwen"
        if "deepseek" in token:
            return "deepseek"
        return ""

