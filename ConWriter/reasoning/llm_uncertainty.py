"""Shared LLM response parsing helpers for model-side uncertainty signals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ConWriter.config.schema import LLMConfig
from ConWriter.utils.types import TokenUncertainty


@dataclass(slots=True)
class LLMGenerationResult:
    """Normalized generation payload from chat-completions style responses."""

    text: str
    tokens: List[str] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = field(default_factory=list)
    token_uncertainties: List[TokenUncertainty] = field(default_factory=list)
    uncertainty_source: str = "none"
    uncertainty_available: bool = False
    uncertainty_truncated: bool = False
    supports_logprobs: Optional[bool] = None


def inject_logprob_request(payload: Dict[str, Any], llm_cfg: LLMConfig) -> None:
    """Attach logprob request options to chat-completions payload."""
    if not bool(llm_cfg.request_logprobs):
        return
    model_id = str(getattr(llm_cfg, "model", "") or "").strip().lower()
    # Some OpenAI-compatible endpoints reject top_logprobs entirely.
    # Keep logprobs on so we can still derive uncertainty from token surprisal.
    # When request_top_logprobs is <= 0, do not send top_logprobs at all.
    payload["logprobs"] = True
    topk = int(llm_cfg.request_top_logprobs)
    if topk <= 0:
        return
    payload["top_logprobs"] = int(topk)


def parse_generation_response(
    response: Dict[str, Any],
    fallback_text: str = "",
) -> LLMGenerationResult:
    """Parse chat completion response into text + optional token uncertainty."""
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return LLMGenerationResult(text=(fallback_text or "").strip(), supports_logprobs=False)

    first = choices[0] if isinstance(choices[0], dict) else {}
    text = _extract_text(first, fallback_text=fallback_text)

    logprobs_payload = first.get("logprobs")
    tokens, token_logprobs, top_logprobs, supports_logprobs = _extract_token_logprobs(logprobs_payload)
    token_unc = _build_token_uncertainties(tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs)
    truncated = any(item.truncated_entropy for item in token_unc)
    available = bool(token_unc)
    source = "model_logprob" if available else "none"
    return LLMGenerationResult(
        text=text,
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        token_uncertainties=token_unc,
        uncertainty_source=source,
        uncertainty_available=available,
        uncertainty_truncated=truncated,
        supports_logprobs=supports_logprobs,
    )


def build_token_uncertainties_from_raw(
    *,
    tokens: Sequence[str],
    token_logprobs: Sequence[float],
    top_logprobs: Sequence[Dict[str, float]],
) -> List[TokenUncertainty]:
    """Build token-level uncertainty objects from raw token/logprob lists."""
    return _build_token_uncertainties(tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs)


def _extract_text(choice: Dict[str, Any], fallback_text: str = "") -> str:
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip().lower()
            # Keep only final answer-like text blocks; skip reasoning traces.
            if item_type and "reason" in item_type:
                continue
            txt = item.get("text", "")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt)
        if chunks:
            return "\n".join(chunks).strip()
    reasoning_candidates: List[Any] = []
    if isinstance(message, dict):
        reasoning_candidates.extend([message.get("reasoning_content"), message.get("reasoning")])
    if isinstance(choice, dict):
        reasoning_candidates.extend([choice.get("reasoning_content"), choice.get("reasoning")])
    for candidate in reasoning_candidates:
        normalized = _normalize_text_candidate(candidate)
        if normalized and not _looks_like_prompt_analysis(normalized):
            return normalized
    txt = choice.get("text", "") if isinstance(choice, dict) else ""
    candidate = str(txt or fallback_text)
    return candidate.strip()


def _normalize_text_candidate(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip().lower()
            if item_type and "reason" in item_type:
                continue
            txt = item.get("text", "")
            if isinstance(txt, str) and txt.strip():
                chunks.append(txt)
        return "\n".join(chunks).strip()
    return ""


def _looks_like_prompt_analysis(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    head_markers = (
        "the user ",
        "the user says",
        "i need to ",
        "based on the provided context",
        "the previous context",
    )
    if lowered.startswith(head_markers):
        return True
    markers = [
        "story tail for continuation",
        "current story length",
        "minimum required length",
        "write the next section",
        "continue directly from the current ending",
        "do not restart the story",
        "the snippet ends",
        "provided context",
    ]
    hits = sum(1 for marker in markers if marker in lowered)
    return hits >= 2


def _extract_token_logprobs(
    logprobs_payload: Any,
) -> Tuple[List[str], List[float], List[Dict[str, float]], Optional[bool]]:
    if not isinstance(logprobs_payload, dict):
        return [], [], [], False
    content = logprobs_payload.get("content")
    if not isinstance(content, list):
        # Some backends return empty dict or null content even when request includes logprobs.
        return [], [], [], True
    tokens: List[str] = []
    token_logprobs: List[float] = []
    top_logprobs: List[Dict[str, float]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        tok = str(item.get("token", ""))
        lp_raw = item.get("logprob", None)
        try:
            lp = float(lp_raw)
        except (TypeError, ValueError):
            lp = 0.0
        tops = _parse_top_logprobs(item.get("top_logprobs", []))
        tokens.append(tok)
        token_logprobs.append(lp)
        top_logprobs.append(tops)
    return tokens, token_logprobs, top_logprobs, True


def _parse_top_logprobs(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, list):
        return {}
    out: Dict[str, float] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        tok = str(row.get("token", ""))
        lp = row.get("logprob", None)
        if not tok:
            continue
        try:
            out[tok] = float(lp)
        except (TypeError, ValueError):
            continue
    return out


def _build_token_uncertainties(
    *,
    tokens: Sequence[str],
    token_logprobs: Sequence[float],
    top_logprobs: Sequence[Dict[str, float]],
) -> List[TokenUncertainty]:
    n = min(len(tokens), len(token_logprobs), len(top_logprobs))
    out: List[TokenUncertainty] = []
    for i in range(n):
        tok = str(tokens[i])
        lp = float(token_logprobs[i])
        tops = dict(top_logprobs[i] or {})
        if tok and tok not in tops:
            tops[tok] = lp
        uncertainty = float(max(0.0, -lp))
        entropy, truncated = _entropy_from_logprobs(chosen_logprob=lp, top_logprobs=tops)
        out.append(
            TokenUncertainty(
                token=tok,
                logprob=lp,
                uncertainty=uncertainty,
                entropy=entropy,
                top_logprobs=tops,
                truncated_entropy=truncated,
                source_type="model_logprob",
            )
        )
    return out


def _entropy_from_logprobs(
    *,
    chosen_logprob: float,
    top_logprobs: Dict[str, float],
) -> Tuple[float, bool]:
    if not top_logprobs:
        # Fallback to surprisal proxy when only chosen token logprob is available.
        return float(max(0.0, -chosen_logprob)), True

    lps = list(top_logprobs.values())
    if not lps:
        return float(max(0.0, -chosen_logprob)), True

    max_lp = max(lps)
    exp_vals = [math.exp(lp - max_lp) for lp in lps]
    z = sum(exp_vals)
    if z <= 0.0:
        return float(max(0.0, -chosen_logprob)), True
    probs = [v / z for v in exp_vals]
    ent = 0.0
    for p in probs:
        if p > 0.0:
            ent -= p * math.log(max(p, 1e-12), 2)
    is_truncated = len(top_logprobs) > 1
    return float(ent), bool(is_truncated)
