"""Generation state helpers."""

from __future__ import annotations

from copy import deepcopy

from ConWriter.utils.types import DynamicMemory, GenerationState, StaticMemory


def initialize_generation_state(static_memory: StaticMemory, dynamic_memory: DynamicMemory) -> GenerationState:
    """Create initial generation state object."""
    return GenerationState(
        current_step=0,
        story_chunks=[],
        static_memory=static_memory,
        dynamic_memory=dynamic_memory,
        last_delta=None,
        last_report=None,
        is_finished=False,
        initial_dynamic_memory=deepcopy(dynamic_memory),
        proposed_deltas=[],
    )
