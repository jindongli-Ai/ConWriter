"""Uncertainty-aware risk monitor (ERM).

ERM is an auxiliary trigger mechanism only. It never decides consistency bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from statistics import mean, pvariance
from typing import Dict, List, Sequence

from ConWriter.config.schema import EntropyMonitorConfig
from ConWriter.reasoning.llm_uncertainty import build_token_uncertainties_from_raw
from ConWriter.reasoning.scene_alignment import find_sentence_ids_for_tokens
from ConWriter.utils.types import (
    EntropyRiskProfile,
    ScenePlan,
    SentenceUnit,
    TokenUncertainty,
    WeightedConstraintItem,
)


@dataclass(slots=True)
class _TextProxyStats:
    sentence_scores: Dict[str, float]
    spike_indices: List[int]
    jump_indices: List[int]
    mean_proxy: float
    spike_score: float
    jump_score: float


class EntropyRiskMonitor:
    """Compute sentence/block-level risk prior from uncertainty signals."""

    def __init__(self, cfg: EntropyMonitorConfig):
        self.cfg = cfg

    def analyze_scene(
        self,
        *,
        scene_plan: ScenePlan,
        scene_text: str,
        sentences: Sequence[SentenceUnit],
        weighted_constraints: Sequence[WeightedConstraintItem],
        previous_entropy_mean: float = 0.0,
        force_validation: bool = False,
        allow_scope_escalation: bool = False,
        allow_replan_trigger: bool = False,
        tokens: Sequence[str] | None = None,
        token_logprobs: Sequence[float] | None = None,
        top_logprobs: Sequence[Dict[str, float]] | None = None,
    ) -> EntropyRiskProfile:
        if not sentences:
            return EntropyRiskProfile(suggested_action="none", source_type="none", uncertainty_mode="none")

        selected_mode = str(self.cfg.entropy_mode).strip().lower() or "auto"
        if selected_mode not in {"auto", "model_logprob", "text_proxy"}:
            selected_mode = "auto"

        token_uncertainties = build_token_uncertainties_from_raw(
            tokens=list(tokens or []),
            token_logprobs=list(token_logprobs or []),
            top_logprobs=list(top_logprobs or []),
        )
        model_uncertainty_available = bool(token_uncertainties)

        source_type = "text_proxy"
        is_proxy_signal = True
        mode_used = "text_proxy"
        sentence_base_scores: Dict[str, float] = {}
        sentence_uncertainty_scores: Dict[str, Dict[str, float]] = {}
        sentence_spikes: List[Dict[str, object]] = []
        spike_indices: List[int] = []
        jump_indices: List[int] = []
        spike_score = 0.0
        jump_score = 0.0
        scene_uncertainty_mean = 0.0
        scene_uncertainty_peak = 0.0
        block_scores: List[Dict[str, float]] = []
        token_entropy_stats: List[Dict[str, object]] = []
        uncertainty_truncated = False

        if selected_mode in {"auto", "model_logprob"} and model_uncertainty_available:
            mode_used = "model_logprob"
            source_type = "model_logprob"
            is_proxy_signal = False
            model_stats = self._compute_model_uncertainty_stats(
                sentences=sentences,
                token_uncertainties=token_uncertainties,
            )
            sentence_base_scores = dict(model_stats["sentence_mean_entropy"])
            sentence_uncertainty_scores = dict(model_stats["sentence_uncertainty_scores"])
            sentence_spikes = list(model_stats["sentence_spikes"])
            spike_indices = list(model_stats["spike_indices"])
            jump_indices = list(model_stats["jump_indices"])
            spike_score = float(model_stats["spike_score"])
            jump_score = float(model_stats["jump_score"])
            scene_uncertainty_mean = float(model_stats["scene_uncertainty_mean"])
            scene_uncertainty_peak = float(model_stats["scene_uncertainty_peak"])
            block_scores = list(self._build_block_scores_from_sentence_metric(sentences, sentence_base_scores))
            token_entropy_stats = [self._token_uncertainty_dict(item) for item in token_uncertainties]
            uncertainty_truncated = bool(model_stats["uncertainty_truncated"])
        elif selected_mode == "auto":
            proxy = self._compute_text_proxy_stats(sentences)
            sentence_base_scores = dict(proxy.sentence_scores)
            sentence_uncertainty_scores = self._convert_proxy_sentence_scores(proxy.sentence_scores)
            sentence_spikes = self._build_proxy_sentence_spikes(sentences, proxy.sentence_scores, proxy.spike_indices)
            spike_indices = list(proxy.spike_indices)
            jump_indices = list(proxy.jump_indices)
            spike_score = float(proxy.spike_score)
            jump_score = float(proxy.jump_score)
            scene_uncertainty_mean = float(proxy.mean_proxy)
            scene_uncertainty_peak = float(max(proxy.sentence_scores.values()) if proxy.sentence_scores else 0.0)
            block_scores = self._build_block_scores_from_sentence_metric(sentences, proxy.sentence_scores)
        else:
            # Strict semantics for model_logprob mode: unavailable model uncertainty
            # should not silently degrade to text proxy.
            return EntropyRiskProfile(
                source_type="none",
                is_proxy_signal=False,
                uncertainty_mode="model_logprob",
                uncertainty_available=False,
                uncertainty_truncated=False,
                suggested_action="none",
            )

        sensitivity = self._constraint_sensitivity(
            scene_plan=scene_plan,
            sentences=sentences,
            weighted_constraints=weighted_constraints,
        )

        final_scores: Dict[str, float] = {}
        for sid, base_h in sentence_base_scores.items():
            sens = float(sensitivity.get(sid, 0.0))
            idx = self._sid_to_idx(sentences, sid)
            jump = self._jump_value(sentence_base_scores, idx, sentences)
            # R_t = alpha * U_t + beta * DeltaU_t + gamma * C_t
            score = (
                float(self.cfg.entropy_spike_weight) * float(base_h)
                + float(self.cfg.entropy_jump_weight) * float(jump)
                + float(self.cfg.constraint_sensitivity_weight) * float(sens)
            )
            final_scores[sid] = score

        final_risk = max(final_scores.values()) if final_scores else 0.0
        risk_tier = self._risk_tier(final_risk)
        sentence_uncertainty_variance = self._sentence_uncertainty_variance(sentence_base_scores)

        high_risk_sentence_ids = [
            sid for sid, score in final_scores.items() if score >= float(self.cfg.risk_high_threshold)
        ]
        high_risk_block_ranges = self._build_high_risk_blocks(sentences, high_risk_sentence_ids)
        linked_constraints = self._linked_constraints(
            sentences=sentences,
            scene_plan=scene_plan,
            weighted_constraints=weighted_constraints,
            target_sentence_ids=high_risk_sentence_ids,
        )
        linked_sentence_ids = list(high_risk_sentence_ids)
        constraint_conditioned_uncertainty = self._constraint_conditioned_uncertainty(
            sentences=sentences,
            sentence_scores=sentence_base_scores,
            linked_constraints=linked_constraints,
        )
        local_constraint_uncertainty = self._local_constraint_uncertainty(
            sentences=sentences,
            sentence_scores=sentence_base_scores,
            sentence_uncertainty_scores=sentence_uncertainty_scores,
            linked_constraints=linked_constraints,
        )
        local_vs_sentence_uncertainty_gap = {
            key: float(local_constraint_uncertainty.get(key, 0.0) - constraint_conditioned_uncertainty.get(key, 0.0))
            for key in set(local_constraint_uncertainty.keys()).union(set(constraint_conditioned_uncertainty.keys()))
        }
        critical_vals = [float(v) for v in constraint_conditioned_uncertainty.values()]
        critical_constraint_uncertainty_peak = float(max(critical_vals) if critical_vals else 0.0)
        critical_constraint_uncertainty_mean = float(mean(critical_vals) if critical_vals else 0.0)
        should_trigger_validation = bool(force_validation and risk_tier == "high_risk")
        variance_is_high = sentence_uncertainty_variance >= float(
            max(0.0, self.cfg.sentence_uncertainty_variance_threshold)
        )
        should_trigger_scope = bool(
            allow_scope_escalation
            and linked_constraints
            and (risk_tier == "high_risk" or (risk_tier == "medium_risk" and variance_is_high))
        )
        should_trigger_replan = bool(
            allow_replan_trigger and risk_tier == "high_risk" and linked_constraints
        )
        suggested_action = "none"
        if should_trigger_replan:
            suggested_action = "escalate_replan_gate"
        elif should_trigger_scope:
            suggested_action = "escalate_patch_scope"
        elif should_trigger_validation:
            suggested_action = "escalate_validation"
        elif risk_tier == "medium_risk":
            suggested_action = "prioritize_validation_focus"

        scene_entropy_mean = float(scene_uncertainty_mean)
        if previous_entropy_mean > 0:
            scene_entropy_mean = float((scene_entropy_mean + previous_entropy_mean) / 2.0)

        return EntropyRiskProfile(
            scene_entropy_mean=scene_entropy_mean,
            sentence_entropy_scores=dict(sentence_base_scores),
            block_entropy_scores=block_scores,
            entropy_spike_indices=list(spike_indices),
            entropy_jump_indices=list(jump_indices),
            entropy_spike_score=float(spike_score),
            entropy_jump_score=float(jump_score),
            constraint_sensitive_risk_scores=final_scores,
            final_risk_score=float(final_risk),
            final_risk_tier=risk_tier,
            suggested_action=suggested_action,
            linked_sentence_ids=linked_sentence_ids,
            linked_constraint_ids=linked_constraints,
            high_risk_sentence_ids=high_risk_sentence_ids,
            high_risk_block_ranges=high_risk_block_ranges,
            source_type=source_type,
            is_proxy_signal=bool(is_proxy_signal),
            token_entropy_stats=token_entropy_stats,
            sentence_uncertainty_scores=sentence_uncertainty_scores,
            sentence_uncertainty_spikes=sentence_spikes,
            constraint_conditioned_uncertainty=constraint_conditioned_uncertainty,
            critical_constraint_uncertainty_peak=critical_constraint_uncertainty_peak,
            critical_constraint_uncertainty_mean=critical_constraint_uncertainty_mean,
            local_constraint_uncertainty=local_constraint_uncertainty,
            local_vs_sentence_uncertainty_gap=local_vs_sentence_uncertainty_gap,
            scene_uncertainty_mean=float(scene_uncertainty_mean),
            scene_uncertainty_peak=float(scene_uncertainty_peak),
            sentence_uncertainty_variance=float(sentence_uncertainty_variance),
            uncertainty_mode=mode_used if selected_mode == "auto" else selected_mode,
            uncertainty_available=bool(model_uncertainty_available) if mode_used == "model_logprob" else False,
            uncertainty_truncated=bool(uncertainty_truncated),
            triggered_patch=should_trigger_scope,
            triggered_validation=should_trigger_validation,
            triggered_patch_escalation=should_trigger_scope,
            triggered_replan=should_trigger_replan,
        )

    def _compute_model_uncertainty_stats(
        self,
        *,
        sentences: Sequence[SentenceUnit],
        token_uncertainties: Sequence[TokenUncertainty],
    ) -> Dict[str, object]:
        sentence_tokens = self._align_tokens_to_sentences(sentences=sentences, token_uncertainties=token_uncertainties)
        sentence_mean_entropy: Dict[str, float] = {}
        sentence_uncertainty_scores: Dict[str, Dict[str, float]] = {}
        sentence_spikes: List[Dict[str, object]] = []
        all_sentence_means: List[float] = []
        all_token_values: List[float] = []
        all_truncated = False
        threshold = float(max(0.1, self.cfg.model_token_high_uncertainty_threshold))
        density_threshold = float(max(0.05, self.cfg.sentence_spike_density_threshold))

        for idx, unit in enumerate(sentences):
            tok_rows = sentence_tokens.get(unit.sentence_id, [])
            if not tok_rows:
                sentence_mean_entropy[unit.sentence_id] = 0.0
                sentence_uncertainty_scores[unit.sentence_id] = {
                    "mean_uncertainty": 0.0,
                    "max_uncertainty": 0.0,
                    "high_uncertainty_density": 0.0,
                    "spike_ratio": 0.0,
                    "token_count": 0.0,
                }
                continue
            vals = [float(item.entropy if item.entropy > 0 else item.uncertainty) for item in tok_rows]
            max_u = max(vals) if vals else 0.0
            mean_u = float(mean(vals) if vals else 0.0)
            high_count = sum(1 for v in vals if v >= threshold)
            density = float(high_count / max(1, len(vals)))
            spike_ratio = density
            sentence_mean_entropy[unit.sentence_id] = mean_u
            sentence_uncertainty_scores[unit.sentence_id] = {
                "mean_uncertainty": mean_u,
                "max_uncertainty": float(max_u),
                "high_uncertainty_density": density,
                "spike_ratio": float(spike_ratio),
                "token_count": float(len(vals)),
            }
            if density >= density_threshold or max_u >= (threshold + 0.5):
                sentence_spikes.append(
                    {
                        "sentence_id": unit.sentence_id,
                        "sentence_index": idx,
                        "mean_uncertainty": mean_u,
                        "max_uncertainty": float(max_u),
                        "high_uncertainty_density": density,
                    }
                )
            all_sentence_means.append(mean_u)
            all_token_values.extend(vals)
            if any(item.truncated_entropy for item in tok_rows):
                all_truncated = True

        raw = [float(sentence_mean_entropy.get(u.sentence_id, 0.0)) for u in sentences]
        spike_indices, jump_indices, spike_score, jump_score = self._spike_jump_indices(raw)
        return {
            "sentence_mean_entropy": sentence_mean_entropy,
            "sentence_uncertainty_scores": sentence_uncertainty_scores,
            "sentence_spikes": sentence_spikes,
            "spike_indices": spike_indices,
            "jump_indices": jump_indices,
            "spike_score": float(spike_score),
            "jump_score": float(jump_score),
            "scene_uncertainty_mean": float(mean(all_sentence_means) if all_sentence_means else 0.0),
            "scene_uncertainty_peak": float(max(all_token_values) if all_token_values else 0.0),
            "uncertainty_truncated": bool(all_truncated),
        }

    def _align_tokens_to_sentences(
        self,
        *,
        sentences: Sequence[SentenceUnit],
        token_uncertainties: Sequence[TokenUncertainty],
    ) -> Dict[str, List[TokenUncertainty]]:
        mapping: Dict[str, List[TokenUncertainty]] = {unit.sentence_id: [] for unit in sentences}
        if not sentences or not token_uncertainties:
            return mapping
        sentence_texts = [unit.text or "" for unit in sentences]
        sentence_index = 0
        consumed = ""
        for item in token_uncertainties:
            if sentence_index >= len(sentences):
                mapping[sentences[-1].sentence_id].append(item)
                continue
            tok = item.token or ""
            normalized = tok.replace("\n", " ")
            consumed += normalized
            mapping[sentences[sentence_index].sentence_id].append(item)
            target_text = sentence_texts[sentence_index]
            # Lightweight alignment by progressive text length and sentence boundary cues.
            if len(consumed.strip()) >= max(1, len(target_text.strip()) - 2):
                sentence_index += 1
                consumed = ""
                continue
            if tok.strip().endswith((".", "!", "?")):
                sentence_index += 1
                consumed = ""
        return mapping

    def _compute_text_proxy_stats(self, sentences: Sequence[SentenceUnit]) -> _TextProxyStats:
        raw: List[float] = []
        by_sid: Dict[str, float] = {}
        for unit in sentences:
            h = self._text_uncertainty_proxy(unit.text)
            raw.append(h)
            by_sid[unit.sentence_id] = h
        mean_h = float(mean(raw) if raw else 0.0)
        spike_indices, jump_indices, spike_score, jump_score = self._spike_jump_indices(raw)
        return _TextProxyStats(
            sentence_scores=by_sid,
            spike_indices=spike_indices,
            jump_indices=jump_indices,
            mean_proxy=mean_h,
            spike_score=float(spike_score),
            jump_score=float(jump_score),
        )

    def _spike_jump_indices(self, raw: Sequence[float]) -> tuple[List[int], List[int], float, float]:
        mean_h = float(mean(raw) if raw else 0.0)
        spike_indices: List[int] = []
        jump_indices: List[int] = []
        jump_vals: List[float] = []
        for idx, val in enumerate(raw):
            if val >= (mean_h + 0.12):
                spike_indices.append(idx)
            if idx > 0:
                jump = abs(val - raw[idx - 1])
                jump_vals.append(jump)
                if jump >= 0.10:
                    jump_indices.append(idx)
        spike_score = float((len(spike_indices) / max(1, len(raw))) * (mean_h + 1e-6))
        jump_score = float(mean(jump_vals) if jump_vals else 0.0)
        return spike_indices, jump_indices, spike_score, jump_score

    def _text_uncertainty_proxy(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z0-9_]+|[^\w\s]", (text or "").lower())
        if not tokens:
            return 0.0
        freq: Dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
        n = float(len(tokens))
        ent = 0.0
        for c in freq.values():
            p = c / n
            ent -= p * math.log(max(p, 1e-9), 2)
        norm = ent / max(1.0, math.log(max(2.0, float(len(freq))), 2))
        punct = sum(1 for t in tokens if re.match(r"[^\w\s]", t))
        punct_ratio = punct / n
        return float(max(0.0, min(1.6, norm + (0.2 * punct_ratio))))

    def _constraint_sensitivity(
        self,
        *,
        scene_plan: ScenePlan,
        sentences: Sequence[SentenceUnit],
        weighted_constraints: Sequence[WeightedConstraintItem],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {unit.sentence_id: 0.0 for unit in sentences}
        required_tokens: List[str] = list(scene_plan.required_constraints[:6])
        required_tokens.extend(scene_plan.must_keep_constraints[:4])
        weighted_tokens: List[str] = []
        for item in weighted_constraints:
            if item.is_hard or item.tier <= 2 or item.weight >= 4.0:
                weighted_tokens.append(str(item.text))
        for token in list(required_tokens) + weighted_tokens:
            sids = find_sentence_ids_for_tokens(sentences, [token])
            if not sids:
                continue
            base = 0.35
            if token in weighted_tokens:
                base = 0.6
            for sid in sids:
                scores[sid] = max(scores.get(sid, 0.0), base)
        for token in scene_plan.expected_state_changes[:4]:
            sids = find_sentence_ids_for_tokens(sentences, [token])
            for sid in sids:
                scores[sid] = max(scores.get(sid, 0.0), 0.5)
        return scores

    def _risk_tier(self, score: float) -> str:
        if score >= float(self.cfg.risk_high_threshold):
            return "high_risk"
        if score >= float(self.cfg.risk_low_threshold):
            return "medium_risk"
        return "low_risk"

    def _build_high_risk_blocks(
        self,
        sentences: Sequence[SentenceUnit],
        high_risk_sentence_ids: Sequence[str],
    ) -> List[Dict[str, int]]:
        sid_to_idx = {u.sentence_id: i for i, u in enumerate(sentences)}
        idxs = sorted(sid_to_idx[sid] for sid in high_risk_sentence_ids if sid in sid_to_idx)
        if not idxs:
            return []
        blocks: List[Dict[str, int]] = []
        start = idxs[0]
        prev = idxs[0]
        for idx in idxs[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            blocks.append({"start_idx": int(start), "end_idx": int(prev)})
            start = idx
            prev = idx
        blocks.append({"start_idx": int(start), "end_idx": int(prev)})
        return blocks

    def _linked_constraints(
        self,
        *,
        sentences: Sequence[SentenceUnit],
        scene_plan: ScenePlan,
        weighted_constraints: Sequence[WeightedConstraintItem],
        target_sentence_ids: Sequence[str],
    ) -> List[str]:
        if not target_sentence_ids:
            return []
        sid_set = set(target_sentence_ids)
        linked: List[str] = []
        candidates: List[str] = list(scene_plan.required_constraints[:8]) + list(scene_plan.must_keep_constraints[:6])
        for item in weighted_constraints:
            if item.is_hard or item.tier <= 2 or item.weight >= 4.0:
                candidates.append(str(item.text))
        for token in candidates:
            sids = find_sentence_ids_for_tokens(sentences, [token])
            if sid_set.intersection(set(sids)):
                if token not in linked:
                    linked.append(token)
        return linked[:12]

    def _build_block_scores_from_sentence_metric(
        self,
        sentences: Sequence[SentenceUnit],
        sentence_metric_scores: Dict[str, float],
    ) -> List[Dict[str, float]]:
        blocks: Dict[int, List[float]] = {}
        for unit in sentences:
            blocks.setdefault(int(unit.paragraph_id), []).append(
                float(sentence_metric_scores.get(unit.sentence_id, 0.0))
            )
        output: List[Dict[str, float]] = []
        for pid in sorted(blocks.keys()):
            vals = blocks[pid]
            output.append(
                {
                    "paragraph_id": float(pid),
                    "block_entropy": float(mean(vals) if vals else 0.0),
                    "num_sentences": float(len(vals)),
                }
            )
        return output

    def _convert_proxy_sentence_scores(self, sentence_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for sid, val in sentence_scores.items():
            payload[sid] = {
                "mean_uncertainty": float(val),
                "max_uncertainty": float(val),
                "high_uncertainty_density": 0.0,
                "spike_ratio": 0.0,
                "token_count": 0.0,
            }
        return payload

    def _build_proxy_sentence_spikes(
        self,
        sentences: Sequence[SentenceUnit],
        sentence_scores: Dict[str, float],
        spike_indices: Sequence[int],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for idx in spike_indices:
            if idx < 0 or idx >= len(sentences):
                continue
            sid = sentences[idx].sentence_id
            val = float(sentence_scores.get(sid, 0.0))
            out.append(
                {
                    "sentence_id": sid,
                    "sentence_index": idx,
                    "mean_uncertainty": val,
                    "max_uncertainty": val,
                    "high_uncertainty_density": 0.0,
                }
            )
        return out

    def _sid_to_idx(self, sentences: Sequence[SentenceUnit], sid: str) -> int:
        for idx, unit in enumerate(sentences):
            if unit.sentence_id == sid:
                return idx
        return -1

    def _jump_value(
        self,
        sentence_scores: Dict[str, float],
        idx: int,
        sentences: Sequence[SentenceUnit],
    ) -> float:
        if idx <= 0 or idx >= len(sentences):
            return 0.0
        sid = sentences[idx].sentence_id
        prev_sid = sentences[idx - 1].sentence_id
        return float(abs(float(sentence_scores.get(sid, 0.0)) - float(sentence_scores.get(prev_sid, 0.0))))

    def _token_uncertainty_dict(self, item: TokenUncertainty) -> Dict[str, object]:
        return {
            "token": item.token,
            "token_logprob": float(item.logprob),
            "token_uncertainty": float(item.uncertainty),
            "token_entropy": float(item.entropy),
            "top_logprobs": dict(item.top_logprobs),
            "whether_truncated": bool(item.truncated_entropy),
            "source_type": item.source_type,
        }

    def _constraint_conditioned_uncertainty(
        self,
        *,
        sentences: Sequence[SentenceUnit],
        sentence_scores: Dict[str, float],
        linked_constraints: Sequence[str],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token in linked_constraints:
            sids = find_sentence_ids_for_tokens(sentences, [token])
            vals = [float(sentence_scores.get(sid, 0.0)) for sid in sids if sid in sentence_scores]
            if vals:
                out[str(token)] = float(mean(vals))
        return out

    def _local_constraint_uncertainty(
        self,
        *,
        sentences: Sequence[SentenceUnit],
        sentence_scores: Dict[str, float],
        sentence_uncertainty_scores: Dict[str, Dict[str, float]],
        linked_constraints: Sequence[str],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token in linked_constraints:
            token_s = str(token).strip().lower()
            if not token_s:
                continue
            sids = find_sentence_ids_for_tokens(sentences, [token])
            if not sids:
                continue
            vals: List[float] = []
            for sid in sids:
                base = float(sentence_scores.get(sid, 0.0))
                max_u = float(sentence_uncertainty_scores.get(sid, {}).get("max_uncertainty", base))
                mean_u = float(sentence_uncertainty_scores.get(sid, {}).get("mean_uncertainty", base))
                sentence_text = ""
                for unit in sentences:
                    if unit.sentence_id == sid:
                        sentence_text = (unit.text or "").lower()
                        break
                # lightweight local anchoring: mention hit boosts local uncertainty with peak-biased blending
                mention_hit = 1.0 if token_s in sentence_text else 0.0
                local_val = (0.55 * mean_u) + (0.35 * max_u) + (0.10 * base * mention_hit)
                vals.append(float(local_val))
            if vals:
                out[str(token)] = float(mean(vals))
        return out

    def _sentence_uncertainty_variance(self, sentence_scores: Dict[str, float]) -> float:
        vals = [float(v) for v in sentence_scores.values()]
        if len(vals) <= 1:
            return 0.0
        return float(pvariance(vals))
