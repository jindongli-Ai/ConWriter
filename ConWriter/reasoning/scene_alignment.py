"""Sentence/span alignment helpers for patch-based local repair."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from ConWriter.utils.types import SentenceUnit, ViolationAnchor


_SENT_SPLIT_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", flags=re.MULTILINE)
_PRONOUNS = {
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "their",
    "theirs",
}
_TEMPORAL_CUES = (
    "before",
    "after",
    "earlier",
    "later",
    "meanwhile",
    "then",
    "finally",
    "eventually",
    "previously",
)


def split_scene_into_units(scene_text: str, scene_id: str) -> List[SentenceUnit]:
    """Split scene text into stable sentence units with character spans."""
    text = scene_text or ""
    if not text.strip():
        return [
            SentenceUnit(
                sentence_id=f"{scene_id}_sent_000",
                text="",
                char_start=0,
                char_end=0,
                paragraph_id=0,
                source_scene_id=scene_id,
            )
        ]

    units: List[SentenceUnit] = []
    paragraph_ranges = _paragraph_ranges(text)
    for sent_idx, match in enumerate(_SENT_SPLIT_RE.finditer(text)):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = int(match.start())
        end = int(match.end())
        paragraph_id = _find_paragraph_id(start, paragraph_ranges)
        units.append(
            SentenceUnit(
                sentence_id=f"{scene_id}_sent_{sent_idx:03d}",
                text=stripped,
                char_start=start,
                char_end=end,
                paragraph_id=paragraph_id,
                source_scene_id=scene_id,
            )
        )
    if not units:
        units.append(
            SentenceUnit(
                sentence_id=f"{scene_id}_sent_000",
                text=text.strip(),
                char_start=0,
                char_end=len(text),
                paragraph_id=0,
                source_scene_id=scene_id,
            )
        )
    return units


def map_entities_to_sentences(
    sentences: Sequence[SentenceUnit],
    entity_aliases: Dict[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """Map sentence_id -> entity ids with local coreference carry-over."""
    mapping: Dict[str, List[str]] = {}
    recent_entities: List[str] = []
    for sentence in sentences:
        lowered = f" {sentence.text.lower()} "
        found: List[str] = []
        for entity_id, aliases in entity_aliases.items():
            for alias in aliases:
                token = str(alias).strip().lower()
                if not token:
                    continue
                if f" {token} " in lowered:
                    found.append(entity_id)
                    break
        if found:
            recent_entities = list(sorted(set(found)))[:2]
        elif _contains_pronoun(lowered) and recent_entities:
            # Light-weight cross-sentence grounding for pronouns.
            found.extend(recent_entities)
        mapping[sentence.sentence_id] = sorted(set(found))
    return mapping


def map_events_to_sentences(
    sentences: Sequence[SentenceUnit],
    event_signals: Dict[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """Map sentence_id -> event ids based on explicit signal matching."""
    explicit, _ = map_events_with_inference(sentences, event_signals)
    return explicit


def map_events_with_inference(
    sentences: Sequence[SentenceUnit],
    event_signals: Dict[str, Sequence[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (explicit_event_map, inferred_event_map) for sentence grounding."""
    explicit: Dict[str, List[str]] = {}
    inferred: Dict[str, List[str]] = {}
    latest_explicit: List[str] = []
    mapping: Dict[str, List[str]] = {}
    for sentence in sentences:
        lowered = sentence.text.lower()
        hits: List[str] = []
        for event_id, signals in event_signals.items():
            if any(sig and str(sig).lower() in lowered for sig in signals):
                hits.append(event_id)
        explicit[sentence.sentence_id] = sorted(set(hits))
        inferred_hits: List[str] = []
        if hits:
            latest_explicit = explicit[sentence.sentence_id]
        elif latest_explicit and any(cue in lowered for cue in _TEMPORAL_CUES):
            inferred_hits = list(latest_explicit[:2])
        inferred[sentence.sentence_id] = sorted(set(inferred_hits))
        mapping[sentence.sentence_id] = sorted(set(hits))
    return explicit, inferred


def find_sentence_ids_for_tokens(
    sentences: Sequence[SentenceUnit],
    tokens: Iterable[str],
) -> List[str]:
    """Return sentence ids that contain any token."""
    norm_tokens = [str(token).strip().lower() for token in tokens if str(token).strip()]
    if not norm_tokens:
        return []
    hits: List[str] = []
    for sentence in sentences:
        lowered = sentence.text.lower()
        if any(token in lowered for token in norm_tokens):
            hits.append(sentence.sentence_id)
    return sorted(set(hits))


def sentence_ids_to_char_spans(
    sentences: Sequence[SentenceUnit],
    sentence_ids: Sequence[str],
) -> List[Dict[str, int]]:
    """Map sentence ids back to text spans."""
    sent_by_id = {unit.sentence_id: unit for unit in sentences}
    spans: List[Dict[str, int]] = []
    for sentence_id in sentence_ids:
        unit = sent_by_id.get(sentence_id)
        if unit is None:
            continue
        spans.append({"char_start": int(unit.char_start), "char_end": int(unit.char_end)})
    return spans


def build_anchor(
    anchor_id: str,
    rule_type: str,
    severity: str,
    sentence_ids: Sequence[str],
    sentences: Sequence[SentenceUnit],
    related_entity_ids: Sequence[str] | None = None,
    related_event_ids: Sequence[str] | None = None,
    related_relation_ids: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
    temporal_conflicts: Sequence[Dict[str, str]] | None = None,
    textual_realization: str = "explicit",
    grounding_confidence: float = 0.5,
    confidence_score: float | None = None,
    source_type: str | None = None,
) -> ViolationAnchor:
    """Build a ViolationAnchor with auto span backfill."""
    normalized_source = str(source_type or textual_realization or "heuristic").strip().lower()
    if normalized_source not in {"explicit", "inferred", "heuristic"}:
        normalized_source = "heuristic"
    conf = float(max(0.0, min(1.0, grounding_confidence)))
    if confidence_score is not None:
        conf = float(max(0.0, min(1.0, confidence_score)))
    return ViolationAnchor(
        anchor_id=anchor_id,
        sentence_ids=sorted(set(sentence_ids)),
        char_spans=sentence_ids_to_char_spans(sentences, sentence_ids),
        rule_type=rule_type,
        severity=severity,
        related_entity_ids=sorted(set(related_entity_ids or [])),
        related_event_ids=sorted(set(related_event_ids or [])),
        related_relation_ids=sorted(set(related_relation_ids or [])),
        temporal_conflicts=list(temporal_conflicts or []),
        textual_realization=textual_realization,
        grounding_confidence=conf,
        confidence_score=conf,
        source_type=normalized_source,
        notes=list(notes or []),
    )


def build_sentence_coref_links(
    sentences: Sequence[SentenceUnit],
    sentence_entity_mentions: Dict[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """Build local sentence->antecedent sentence ids for pronoun/coref traces."""
    links: Dict[str, List[str]] = {}
    last_entity_sentence: Dict[str, str] = {}
    for sent in sentences:
        sid = sent.sentence_id
        text = f" {sent.text.lower()} "
        mentions = list(sentence_entity_mentions.get(sid, []))
        local_links: List[str] = []
        for entity in mentions:
            anchor_sid = last_entity_sentence.get(entity)
            if anchor_sid and anchor_sid != sid:
                local_links.append(anchor_sid)
            last_entity_sentence[entity] = sid
        if _contains_pronoun(text) and not mentions:
            for entity, anchor_sid in list(last_entity_sentence.items())[-2:]:
                if anchor_sid != sid:
                    local_links.append(anchor_sid)
        links[sid] = sorted(set(local_links))
    return links


def find_temporal_conflict_sentences(
    sentences: Sequence[SentenceUnit],
    tokens: Sequence[str],
) -> List[str]:
    """Ground temporal conflict candidates to sentences via tokens + temporal cues."""
    seeds = find_sentence_ids_for_tokens(sentences, tokens)
    if seeds:
        return seeds
    hits: List[str] = []
    for sent in sentences:
        lowered = sent.text.lower()
        if any(cue in lowered for cue in _TEMPORAL_CUES):
            hits.append(sent.sentence_id)
    return sorted(set(hits))


def detect_textual_event_realization(
    sentences: Sequence[SentenceUnit],
    event_phrase: str,
) -> Tuple[bool, List[str]]:
    """Return whether an event phrase is textually realized and where."""
    phrase = str(event_phrase or "").strip().lower()
    if not phrase:
        return True, []
    hits: List[str] = []
    for sentence in sentences:
        if phrase in sentence.text.lower():
            hits.append(sentence.sentence_id)
    return bool(hits), sorted(set(hits))


def _paragraph_ranges(text: str) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    cursor = 0
    paragraph_id = 0
    for block in re.split(r"\n\s*\n", text):
        start = text.find(block, cursor)
        if start < 0:
            continue
        end = start + len(block)
        ranges.append((start, end, paragraph_id))
        paragraph_id += 1
        cursor = end
    if not ranges:
        ranges.append((0, len(text), 0))
    return ranges


def _find_paragraph_id(position: int, ranges: Sequence[Tuple[int, int, int]]) -> int:
    for start, end, pid in ranges:
        if start <= position < end:
            return pid
    return int(ranges[-1][2]) if ranges else 0


def _contains_pronoun(lowered_sentence: str) -> bool:
    return any(f" {pron} " in lowered_sentence for pron in _PRONOUNS)
