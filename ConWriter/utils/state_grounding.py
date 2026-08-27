"""Lightweight state canonicalization and alias grounding helpers."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence


_STATE_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "and",
    "or",
    "that",
    "this",
    "it",
    "as",
}

_WORD_ALIAS = {
    "reduced": ["reduced", "lowered", "decreased", "eased", "calmed"],
    "decreased": ["decreased", "reduced", "lowered", "eased"],
    "increased": ["increased", "raised", "grew", "heightened"],
    "resolved": ["resolved", "settled", "addressed", "fixed"],
    "revealed": ["revealed", "disclosed", "made clear", "uncovered"],
    "confirmed": ["confirmed", "verified", "made certain", "established"],
    "known": ["known", "clear", "established", "recognized"],
    "unknown": ["unknown", "uncertain", "unclear"],
    "appears": ["appears", "emerges", "shows up", "is present"],
    "appear": ["appear", "emerge", "show up", "be present"],
    "removed": ["removed", "eliminated", "cleared", "erased"],
    "teleport": ["teleport", "teleported", "teleporting", "instantly moved", "instant travel"],
}

_PHRASE_ALIAS = {
    "major tension reduced": [
        "major tension was reduced",
        "major tension eased",
        "tension was reduced",
        "tension eased",
        "tension calmed",
    ],
    "teleport": [
        "teleported",
        "teleporting",
        "instant travel",
        "instantly moved",
        "suddenly appeared elsewhere",
    ],
}

_TRANSITION_CUES = [
    "then",
    "after",
    "later",
    "next",
    "therefore",
    "as a result",
    "subsequently",
    "eventually",
]


def normalize_state_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_state_phrase(value: object) -> str:
    return normalize_state_text(value)


def _dedup(values: Sequence[str], limit: int = 24) -> List[str]:
    out: List[str] = []
    for value in values:
        token = normalize_state_text(value)
        if not token or token in out:
            continue
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _core_tokens(canonical: str) -> List[str]:
    return [
        token
        for token in canonical.split()
        if token and token not in _STATE_STOPWORDS and (len(token) >= 3 or token.isdigit())
    ]


def _single_token_alias_variants(words: List[str], idx: int, alias_limit: int) -> List[str]:
    source = words[idx]
    alts = _WORD_ALIAS.get(source, [])
    out: List[str] = []
    for alt in alts[:alias_limit]:
        alt_words = list(words)
        alt_words[idx] = alt
        out.append(" ".join(alt_words).strip())
    return out


def _generate_aliases(canonical: str, alias_limit: int = 12) -> List[str]:
    words = [w for w in canonical.split() if w]
    if not words:
        return []
    out: List[str] = [canonical]
    out.extend(list(_PHRASE_ALIAS.get(canonical, [])))
    if len(words) >= 2:
        out.append(" ".join(words[-2:]))
    if len(words) >= 3:
        out.append(" ".join(words[1:]))
    if words and words[-1] in _WORD_ALIAS and len(words) >= 2:
        subject = " ".join(words[:-1]).strip()
        for verb in _WORD_ALIAS.get(words[-1], [])[:4]:
            out.append(f"{subject} {verb}".strip())
            out.append(f"{subject} was {verb}".strip())
    for idx in range(min(len(words), 4)):
        out.extend(_single_token_alias_variants(words, idx, alias_limit=3))
    return _dedup(out, limit=alias_limit)


def build_state_groundings(states: Sequence[object] | None, *, alias_limit: int = 12) -> List[Dict[str, object]]:
    grouped: List[Dict[str, object]] = []
    seen: set[str] = set()
    for value in list(states or []):
        raw = str(value or "").strip()
        canonical = canonicalize_state_phrase(raw)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        grouped.append(
            {
                "raw": raw,
                "canonical": canonical,
                "aliases": _generate_aliases(canonical, alias_limit=alias_limit),
                "core_tokens": _core_tokens(canonical),
            }
        )
    return grouped


def build_state_grounding_bundle(
    *,
    required_states: Sequence[object] | None = None,
    forbidden_states: Sequence[object] | None = None,
    operator_post_states: Sequence[object] | None = None,
    transition_target_states: Sequence[object] | None = None,
    alias_limit: int = 12,
) -> Dict[str, object]:
    required_groundings = build_state_groundings(required_states, alias_limit=alias_limit)
    forbidden_groundings = build_state_groundings(forbidden_states, alias_limit=alias_limit)
    operator_groundings = build_state_groundings(operator_post_states, alias_limit=alias_limit)
    transition_groundings = build_state_groundings(transition_target_states, alias_limit=alias_limit)
    return {
        "canonical_required_states": [str(item.get("canonical", "")) for item in required_groundings],
        "canonical_forbidden_states": [str(item.get("canonical", "")) for item in forbidden_groundings],
        "canonical_operator_post_states": [str(item.get("canonical", "")) for item in operator_groundings],
        "canonical_transition_target_states": [str(item.get("canonical", "")) for item in transition_groundings],
        "required_state_groundings": required_groundings,
        "forbidden_state_groundings": forbidden_groundings,
        "operator_post_state_groundings": operator_groundings,
        "transition_target_state_groundings": transition_groundings,
        "transition_grounded_cues": list(_TRANSITION_CUES),
    }


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return bool(re.search(rf"\b{re.escape(phrase)}\b", normalized_text))


def _match_single_grounding(
    normalized_text: str,
    grounding: Dict[str, object],
) -> Dict[str, object]:
    canonical = str(grounding.get("canonical", "")).strip()
    aliases = [str(x).strip() for x in list(grounding.get("aliases", []) or []) if str(x).strip()]
    core_tokens = [str(x).strip() for x in list(grounding.get("core_tokens", []) or []) if str(x).strip()]
    if canonical and _contains_phrase(normalized_text, canonical):
        return {
            "canonical": canonical,
            "matched": True,
            "match_type": "canonical_exact",
            "matched_forms": [canonical],
            "matched_text_span": canonical,
        }
    for alias in aliases:
        if alias == canonical:
            continue
        if _contains_phrase(normalized_text, alias):
            return {
                "canonical": canonical,
                "matched": True,
                "match_type": "alias_grounded",
                "matched_forms": [alias],
                "matched_text_span": alias,
            }
    if core_tokens:
        hits = [token for token in core_tokens if _contains_phrase(normalized_text, token)]
        if core_tokens:
            required_hits = 1 if len(core_tokens) == 1 else max(2, int(round(len(core_tokens) * 0.67)))
            required_hits = min(required_hits, len(core_tokens))
            if len(hits) >= required_hits:
                return {
                    "canonical": canonical,
                    "matched": True,
                    "match_type": "weak_proxy",
                    "matched_forms": hits,
                    "matched_text_span": " ".join(hits),
                }
    return {
        "canonical": canonical,
        "matched": False,
        "match_type": "no_match",
        "matched_forms": [],
        "matched_text_span": "",
    }


def _aggregate_positive_match_type(match_types: Sequence[str], *, require_all: bool) -> str:
    types = [str(x) for x in match_types]
    if not types:
        return "no_match"
    if require_all and any(t == "no_match" for t in types):
        return "no_match"
    if all(t == "canonical_exact" for t in types):
        return "canonical_exact"
    if any(t == "weak_proxy" for t in types):
        return "weak_proxy"
    if any(t == "alias_grounded" for t in types):
        return "alias_grounded"
    if any(t == "canonical_exact" for t in types):
        return "canonical_exact"
    return "no_match"


def evaluate_state_realization_with_grounding(
    *,
    rewritten_text: str,
    required_groundings: Sequence[Dict[str, object]] | None = None,
    forbidden_groundings: Sequence[Dict[str, object]] | None = None,
    operator_groundings: Sequence[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    normalized_text = normalize_state_text(rewritten_text or "")
    req_items = [_match_single_grounding(normalized_text, item) for item in list(required_groundings or [])]
    op_items = [_match_single_grounding(normalized_text, item) for item in list(operator_groundings or [])]
    for_items = [_match_single_grounding(normalized_text, item) for item in list(forbidden_groundings or [])]

    req_missing = [str(item.get("canonical", "")) for item in req_items if not bool(item.get("matched", False))]
    op_missing = [str(item.get("canonical", "")) for item in op_items if not bool(item.get("matched", False))]
    for_present = [str(item.get("canonical", "")) for item in for_items if bool(item.get("matched", False))]

    req_realized = bool((not req_items) or (not req_missing))
    op_realized = bool((not op_items) or (not op_missing))
    forbidden_removed = bool((not for_items) or (not for_present))

    req_types = [str(item.get("match_type", "no_match")) for item in req_items]
    op_types = [str(item.get("match_type", "no_match")) for item in op_items]
    for_positive_types = [str(item.get("match_type", "no_match")) for item in for_items if bool(item.get("matched", False))]

    state_realization_match_type = _aggregate_positive_match_type(req_types, require_all=True)
    operator_post_state_match_type = _aggregate_positive_match_type(op_types, require_all=True)
    forbidden_state_removal_match_type = (
        "no_match"
        if forbidden_removed
        else _aggregate_positive_match_type(for_positive_types, require_all=False)
    )

    grounded_alias_matches: List[Dict[str, object]] = []
    for state_type, items in (
        ("required", req_items),
        ("operator_post_state", op_items),
        ("forbidden", for_items),
    ):
        for item in items:
            match_type = str(item.get("match_type", "no_match"))
            if match_type == "no_match":
                continue
            grounded_alias_matches.append(
                {
                    "state_type": state_type,
                    "canonical_state": str(item.get("canonical", "")),
                    "match_type": match_type,
                    "matched_forms": list(item.get("matched_forms", []) or []),
                    "matched_text_span": str(item.get("matched_text_span", "")),
                }
            )

    def _item_alias_groundings(raw_item: Dict[str, object]) -> List[str]:
        return [str(x) for x in list(raw_item.get("aliases", []) or []) if str(x).strip()]

    def _required_status(match_type: str) -> str:
        if match_type in {"canonical_exact", "alias_grounded"}:
            return "satisfied"
        if match_type == "weak_proxy":
            return "uncertain"
        return "unsatisfied"

    def _forbidden_status(match_type: str) -> str:
        if match_type == "no_match":
            return "removed"
        if match_type == "weak_proxy":
            return "uncertain"
        return "still_present"

    def _operator_status(match_type: str) -> str:
        if match_type in {"canonical_exact", "alias_grounded"}:
            return "realized"
        if match_type == "weak_proxy":
            return "uncertain"
        return "unrealized"

    required_state_checklist: List[Dict[str, object]] = []
    for raw_item, match_item in zip(list(required_groundings or []), req_items):
        match_type = str(match_item.get("match_type", "no_match"))
        required_state_checklist.append(
            {
                "canonical_state": str(raw_item.get("canonical", "")),
                "alias_groundings": _item_alias_groundings(raw_item),
                "status": _required_status(match_type),
                "matched_text_span": str(match_item.get("matched_text_span", "")),
                "match_type": match_type,
            }
        )

    forbidden_state_checklist: List[Dict[str, object]] = []
    for raw_item, match_item in zip(list(forbidden_groundings or []), for_items):
        match_type = str(match_item.get("match_type", "no_match"))
        forbidden_state_checklist.append(
            {
                "canonical_state": str(raw_item.get("canonical", "")),
                "alias_groundings": _item_alias_groundings(raw_item),
                "status": _forbidden_status(match_type),
                "matched_text_span": str(match_item.get("matched_text_span", "")),
                "match_type": match_type,
            }
        )

    operator_post_state_checklist: List[Dict[str, object]] = []
    for raw_item, match_item in zip(list(operator_groundings or []), op_items):
        match_type = str(match_item.get("match_type", "no_match"))
        operator_post_state_checklist.append(
            {
                "canonical_state": str(raw_item.get("canonical", "")),
                "alias_groundings": _item_alias_groundings(raw_item),
                "status": _operator_status(match_type),
                "matched_text_span": str(match_item.get("matched_text_span", "")),
                "match_type": match_type,
            }
        )

    def _completion_rate(items: Sequence[Dict[str, object]], done_status: str) -> float:
        n = len(items)
        if n == 0:
            return 1.0
        done = sum(1 for item in items if str(item.get("status", "")) == done_status)
        return float(done / max(1, n))

    required_completion_rate = _completion_rate(required_state_checklist, "satisfied")
    forbidden_completion_rate = _completion_rate(forbidden_state_checklist, "removed")
    operator_completion_rate = _completion_rate(operator_post_state_checklist, "realized")

    checklist_total_items = (
        len(required_state_checklist)
        + len(forbidden_state_checklist)
        + len(operator_post_state_checklist)
    )
    checklist_completed_items = (
        sum(1 for item in required_state_checklist if str(item.get("status", "")) == "satisfied")
        + sum(1 for item in forbidden_state_checklist if str(item.get("status", "")) == "removed")
        + sum(1 for item in operator_post_state_checklist if str(item.get("status", "")) == "realized")
    )
    checklist_completion_rate = float(checklist_completed_items / max(1, checklist_total_items))

    unresolved_required_states = [
        str(item.get("canonical_state", ""))
        for item in required_state_checklist
        if str(item.get("status", "")) in {"unsatisfied", "uncertain"}
    ]
    unresolved_forbidden_states = [
        str(item.get("canonical_state", ""))
        for item in forbidden_state_checklist
        if str(item.get("status", "")) in {"still_present", "uncertain"}
    ]
    unresolved_operator_post_states = [
        str(item.get("canonical_state", ""))
        for item in operator_post_state_checklist
        if str(item.get("status", "")) in {"unrealized", "uncertain"}
    ]
    still_unresolved_state_items: List[Dict[str, object]] = []
    for item in required_state_checklist:
        if str(item.get("status", "")) in {"unsatisfied", "uncertain"}:
            still_unresolved_state_items.append({"state_type": "required", **dict(item)})
    for item in forbidden_state_checklist:
        if str(item.get("status", "")) in {"still_present", "uncertain"}:
            still_unresolved_state_items.append({"state_type": "forbidden", **dict(item)})
    for item in operator_post_state_checklist:
        if str(item.get("status", "")) in {"unrealized", "uncertain"}:
            still_unresolved_state_items.append({"state_type": "operator_post_state", **dict(item)})

    return {
        "rewrite_realizes_required_state_change": bool(req_realized),
        "rewrite_realizes_operator_post_state": bool(op_realized),
        "rewrite_removes_forbidden_state": bool(forbidden_removed),
        "state_realization_match_type": str(state_realization_match_type),
        "operator_post_state_match_type": str(operator_post_state_match_type),
        "forbidden_state_removal_match_type": str(forbidden_state_removal_match_type),
        "missing_required_states": list(req_missing),
        "missing_operator_post_states": list(op_missing),
        "remaining_forbidden_states": list(for_present),
        "grounded_alias_matches": grounded_alias_matches,
        "required_state_checklist": required_state_checklist,
        "forbidden_state_checklist": forbidden_state_checklist,
        "operator_post_state_checklist": operator_post_state_checklist,
        "required_checklist_completion_rate": float(required_completion_rate),
        "forbidden_checklist_completion_rate": float(forbidden_completion_rate),
        "operator_post_checklist_completion_rate": float(operator_completion_rate),
        "checklist_completion_rate": float(checklist_completion_rate),
        "checklist_total_items": int(checklist_total_items),
        "checklist_completed_items": int(checklist_completed_items),
        "unresolved_required_states": unresolved_required_states,
        "unresolved_forbidden_states": unresolved_forbidden_states,
        "unresolved_operator_post_states": unresolved_operator_post_states,
        "still_unresolved_state_items": still_unresolved_state_items,
    }


__all__ = [
    "build_state_grounding_bundle",
    "build_state_groundings",
    "canonicalize_state_phrase",
    "evaluate_state_realization_with_grounding",
    "normalize_state_text",
]
