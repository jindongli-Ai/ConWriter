"""Lightweight length-aware compensation and progress warnings."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Sequence

from ConWriter.config.schema import LengthControlConfig
from ConWriter.utils.types import ScenePlan


class LengthController:
    """Model-aware target calibration and lightweight generation guidance."""

    WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    def __init__(self, config: LengthControlConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("ConWriter.length_control")
        self._online_factors: Dict[str, float] = {}

    def estimate_desired_target_length(
        self,
        *,
        planning_target_words: int,
        scenes: Sequence[ScenePlan],
        fallback_scene_target: int,
    ) -> int:
        if int(planning_target_words) > 0:
            return int(planning_target_words)
        scene_sum = sum(max(0, int(getattr(scene, "target_words", 0) or 0)) for scene in list(scenes))
        if scene_sum > 0:
            return int(scene_sum)
        if scenes:
            return int(max(0, int(fallback_scene_target)) * len(list(scenes)))
        return int(max(0, int(fallback_scene_target)))

    def compute_targets(self, *, desired_target_length: int, model_id: str) -> Dict[str, object]:
        desired = int(max(0, int(desired_target_length)))
        factor = self._resolve_factor(model_id)
        requested = int(round(float(desired) * float(factor)))
        return {
            "desired_target_length": desired,
            "length_compensation_factor": float(factor),
            "requested_target_length": int(max(desired, requested)),
        }

    def monitor_progress(
        self,
        *,
        current_word_count: int,
        requested_target_length: int,
        scene_index: int,
        total_scenes: int,
        latest_scene_text: str,
        unresolved_threads: Sequence[str],
    ) -> Dict[str, object]:
        current = int(max(0, int(current_word_count)))
        requested = int(max(1, int(requested_target_length)))
        ratio = float(current / requested)
        mid_or_late = bool(scene_index >= max(1, int(total_scenes // 2)))
        under_generation_warning = bool(mid_or_late and ratio < float(self.config.progress_warning_ratio))
        premature_closure_warning = bool(
            self.detect_premature_closure(
                latest_scene_text=latest_scene_text,
                unresolved_threads=unresolved_threads,
            )
            and ratio < 0.95
        )
        expansion_guidance: List[str] = []
        if under_generation_warning:
            expansion_guidance.append(
                "Progress is behind target length; expand unresolved threads before concluding."
            )
        if premature_closure_warning:
            expansion_guidance.append(
                "Avoid early closing language now; continue causal development before ending."
            )
        expansion_guidance = expansion_guidance[: max(0, int(self.config.max_guidance_lines))]
        return {
            "current_word_count": current,
            "requested_target_length": requested,
            "progress_ratio": float(ratio),
            "under_generation_warning_triggered": bool(under_generation_warning),
            "premature_closure_warning_triggered": bool(premature_closure_warning),
            "expansion_guidance": list(expansion_guidance),
        }

    def build_generation_guidance(
        self,
        *,
        requested_target_length: int,
        progress_ratio: float,
        expansion_guidance: Sequence[str],
    ) -> List[str]:
        lines = [
            f"Aim for about {int(max(1, requested_target_length))} words before ending the full story.",
            "Do not stop early; continue unresolved threads until target length is approached.",
        ]
        if float(progress_ratio) < float(self.config.progress_warning_ratio):
            lines.append("Current progress is behind target; continue expanding unresolved threads.")
        for line in list(expansion_guidance):
            token = str(line).strip()
            if token and token not in lines:
                lines.append(token)
        return lines[: max(1, int(self.config.max_guidance_lines))]

    def detect_premature_closure(self, *, latest_scene_text: str, unresolved_threads: Sequence[str]) -> bool:
        text = str(latest_scene_text or "").strip().lower()
        if not text:
            return False
        unresolved = [str(x).strip() for x in list(unresolved_threads or []) if str(x).strip()]
        if not unresolved:
            return False
        tail = text[-260:]
        for cue in list(self.config.premature_closure_cues):
            token = str(cue).strip().lower()
            if token and token in tail:
                return True
        return False

    def update_online_factor(
        self,
        *,
        model_id: str,
        desired_target_length: int,
        actual_generated_length: int,
    ) -> float:
        model_key = self._normalize_model_id(model_id)
        base = self._resolve_factor(model_key)
        if (not self.config.online_factor_update) or int(desired_target_length) <= 0:
            return float(base)
        actual = float(max(1, int(actual_generated_length)))
        desired = float(max(1, int(desired_target_length)))
        completion = actual / desired
        required = float(1.0 / max(0.55, completion))
        required = min(float(self.config.compensation_cap), max(1.0, required))
        alpha = float(max(0.0, min(1.0, float(self.config.online_factor_update_rate))))
        prev = float(self._online_factors.get(model_key, base))
        updated = ((1.0 - alpha) * prev) + (alpha * required)
        updated = min(float(self.config.compensation_cap), max(1.0, updated))
        self._online_factors[model_key] = float(updated)
        return float(updated)

    @classmethod
    def count_words(cls, text: str) -> int:
        return int(len(cls.WORD_RE.findall(str(text or ""))))

    def _resolve_factor(self, model_id: str) -> float:
        model_key = self._normalize_model_id(model_id)
        if model_key in self._online_factors:
            return float(self._clamp_factor(self._online_factors[model_key]))
        factors = dict(self.config.model_compensation_factors or {})
        for key in sorted(factors.keys(), key=len, reverse=True):
            token = str(key or "").strip().lower()
            if token and token in model_key:
                return float(self._clamp_factor(factors[key]))
        return float(self._clamp_factor(self.config.default_compensation_factor))

    def _clamp_factor(self, factor: float) -> float:
        return float(min(float(self.config.compensation_cap), max(1.0, float(factor))))

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        token = str(model_id or "").strip().lower()
        return token or "unknown_model"

