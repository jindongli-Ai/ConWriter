"""Output conversion helpers for ConWriter pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict

from ConWriter.utils.types import ConWriterOutputRecord, GenerationState


_ARTIFACT_MARKERS = [
    "this scene must satisfy:",
    "story tail for continuation",
    "we are continuing a story",
    "we need to continue the story",
    "analyze the request",
    "output only story prose",
    "continue directly from the current ending",
    "char_set advances local objective",
    "scene progression:c",
    "thinking. 1.",
]


def _is_artifact_paragraph(text: str) -> bool:
    para = (text or "").strip()
    if not para:
        return False
    lowered = para.lower()
    if re.match(r"^\[scene_\d+\]", para, flags=re.IGNORECASE):
        return True
    if re.match(r"^#\s*scene\s+\d+", para, flags=re.IGNORECASE):
        return True
    if re.match(r"^#\s*c\d+s\d+\s*-\s*you\s+pov\b", para, flags=re.IGNORECASE):
        return True
    if lowered.startswith(("thinking.", "let me work through", "we need to continue", "we are continuing")):
        return True
    hits = sum(1 for marker in _ARTIFACT_MARKERS if marker in lowered)
    if hits >= 2:
        return True
    return False


def clean_generated_story(text: str) -> str:
    """Remove internal revision/debug artifacts from exported story text."""
    raw = (text or "").strip()
    if not raw:
        return ""

    paragraphs = re.split(r"\n\s*\n", raw)
    kept_paragraphs = []
    for para in paragraphs:
        paragraph = para.strip()
        if not paragraph:
            continue
        if re.match(r"^\s*\[Revision\s+\d+\]\s*", paragraph):
            continue
        if _is_artifact_paragraph(paragraph):
            continue
        kept_lines = []
        for line in paragraph.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^\[scene_\d+\]", stripped, flags=re.IGNORECASE):
                continue
            if re.match(r"^#\s*scene\s+\d+", stripped, flags=re.IGNORECASE):
                continue
            if re.match(r"^#\s*c\d+s\d+\s*-\s*you\s+pov\b", stripped, flags=re.IGNORECASE):
                continue
            if re.match(r"^\[Revision\s+\d+\]\s*", stripped, flags=re.IGNORECASE):
                continue
            low = stripped.lower()
            if any(marker in low for marker in _ARTIFACT_MARKERS):
                continue
            kept_lines.append(stripped)
        normalized = "\n".join(kept_lines).strip()
        if normalized and not _is_artifact_paragraph(normalized):
            kept_paragraphs.append(normalized)

    cleaned = "\n\n".join(kept_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned.strip()


def build_output_record(
    state: GenerationState,
    prompt_id: str,
    prompt_text: str,
    language: str,
    task_type: str,
    model_name: str,
) -> ConWriterOutputRecord:
    """Build final output record after one generation run."""
    accepted_story = "\n\n".join(chunk.text for chunk in state.story_chunks if chunk.accepted)
    # Fallback: if all scenes were rejected, keep non-empty scene text so caller
    # does not enter infinite retry loops on empty generated_story.
    fallback_story = "\n\n".join(
        (chunk.text or "").strip()
        for chunk in state.story_chunks
        if (chunk.text or "").strip()
    )

    final_story = accepted_story
    if not final_story.strip():
        final_story = fallback_story
    cleaned_story = clean_generated_story(final_story)
    # Keep long raw fallback if cleaning removes everything; avoids empty-story loops.
    if not cleaned_story.strip() and fallback_story.strip():
        cleaned_story = fallback_story.strip()
    return ConWriterOutputRecord(
        id=prompt_id,
        language=language,
        task_type=task_type,
        prompt=prompt_text,
        model_name=model_name,
        generated_story=cleaned_story,
        generation_error=None,
        metadata={
            "num_chunks": len(state.story_chunks),
            "last_consistency": state.last_report.suggested_action if state.last_report else None,
        },
    )


def generation_state_to_dict(state: GenerationState) -> Dict[str, Any]:
    """Convert runtime state dataclass tree to dictionary."""
    return asdict(state)
