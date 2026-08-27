"""Dynamic memory initialization and update interface."""

from __future__ import annotations

from copy import deepcopy

from ConWriter.memory.memory_graph import MemoryGraph
from ConWriter.memory.memory_update import MemoryUpdater
from ConWriter.utils.types import (
    CharacterStateSnapshot,
    CharacterMemory,
    ConsistencyReport,
    DynamicMemory,
    EntityState,
    MemoryDelta,
    PlotThread,
    StaticMemory,
    StoryState,
    TimelineStateSnapshot,
    TransitionAction,
    TransitionConstraint,
    WorldStateSnapshot,
)


class DynamicMemoryManager:
    """Manager for lifecycle of evolving dynamic memory."""

    def __init__(self, max_history_entries: int = 200):
        self.max_history_entries = max_history_entries
        self._updater = MemoryUpdater()

    def initialize(
        self,
        static_memory: StaticMemory,
        seed_characters: bool = True,
    ) -> DynamicMemory:
        """Create initial dynamic memory from static memory constraints."""
        memory = DynamicMemory()

        if seed_characters:
            self._seed_character_memory(memory.characterization, static_memory)

        memory.narrative_style.current_pov = static_memory.style.narrative_pov
        memory.narrative_style.current_tense = static_memory.style.tense_style
        if static_memory.style.tone != "unspecified":
            memory.narrative_style.tone_trace.append(static_memory.style.tone)

        memory.world_setting.current_setting_state = (
            static_memory.world_setting.setting_description or "unknown"
        )
        memory.world_setting.global_world_facts.extend(static_memory.world_setting.world_invariants)

        if static_memory.timeline_plot.global_story_goal:
            memory.timeline_plot.active_goals.append(static_memory.timeline_plot.global_story_goal)
        memory.timeline_plot.plot_checkpoints.extend(static_memory.timeline_plot.required_plot_points)
        memory.timeline_plot.pending_constraints.extend(static_memory.timeline_plot.required_plot_points[:6])
        for idx, conflict in enumerate(static_memory.timeline_plot.core_conflicts):
            memory.timeline_plot.unresolved_threads.append(
                PlotThread(
                    thread_id=f"thread_{idx:03d}",
                    title=conflict[:80],
                    goal=conflict,
                    status="active",
                )
            )
            memory.timeline_plot.active_plot_threads.append(conflict)

        relation_graph = MemoryGraph.from_relations(memory.characterization.relations)
        memory.memory_history.append(
            {
                "chunk_id": "init",
                "accepted": True,
                "note": "Initialized dynamic five-facet memory from static constraints.",
                "relation_graph": relation_graph.as_summary(),
            }
        )
        self._refresh_state_memory(memory, step_index=0, append_history=True)
        return memory

    def preview_update(
        self,
        memory: DynamicMemory,
        delta: MemoryDelta,
    ) -> DynamicMemory:
        """Preview a memory update without changing current memory."""
        preview = self._updater.preview(memory, delta)
        self._refresh_state_memory(preview, append_history=False)
        return preview

    def apply_update(
        self,
        memory: DynamicMemory,
        delta: MemoryDelta,
        report: ConsistencyReport | None = None,
        action: TransitionAction | None = None,
        inferred_constraints: list[TransitionConstraint] | None = None,
    ) -> DynamicMemory:
        """Apply update and clip history length."""
        updated = self._updater.apply(memory, delta, report=report)
        if inferred_constraints is not None:
            updated.inferred_constraints = list(inferred_constraints)
        if action is not None:
            updated.transition_history.append(action)
        self._refresh_state_memory(updated, append_history=True)
        if len(updated.memory_history) > self.max_history_entries:
            updated.memory_history = updated.memory_history[-self.max_history_entries :]
        return updated

    def reject_update(
        self,
        memory: DynamicMemory,
        delta: MemoryDelta,
        report: ConsistencyReport | None = None,
        reason: str = "rejected_by_consistency_policy",
        action: TransitionAction | None = None,
        inferred_constraints: list[TransitionConstraint] | None = None,
    ) -> DynamicMemory:
        """Record one rejected update without mutating facet states."""
        updated = self._updater.record_rejected(memory, delta, report=report, reason=reason)
        if inferred_constraints is not None:
            updated.inferred_constraints = list(inferred_constraints)
        if action is not None:
            updated.transition_history.append(action)
        self._refresh_state_memory(updated, append_history=False)
        if len(updated.memory_history) > self.max_history_entries:
            updated.memory_history = updated.memory_history[-self.max_history_entries :]
        return updated

    def get_state(self, memory: DynamicMemory) -> StoryState:
        """Return current structured state, synthesizing from dynamic facets when needed."""
        if memory.current_state.character_states:
            return deepcopy(memory.current_state)
        self._refresh_state_memory(memory, append_history=not memory.state_history)
        return deepcopy(memory.current_state)

    def _seed_character_memory(self, target: CharacterMemory, static_memory: StaticMemory) -> None:
        for char_id, profile in static_memory.characterization.character_profiles.items():
            entity = EntityState(
                entity_id=char_id,
                name=profile.canonical_name,
                status="introduced",
                attributes={"source": "static_prompt", "role": profile.role},
                location="unspecified",
                goals=[],
                knowledge=[],
                motivations=[],
                abilities=list(static_memory.characterization.known_abilities.get(char_id, [])),
                last_updated_step=-1,
            )
            target.entity_store[char_id] = entity
            target.current_states[char_id] = entity.status
            target.locations[char_id] = entity.location
            target.ability_states[char_id] = list(entity.abilities)
            target.knowledge_states[char_id] = []
            target.motivation_states[char_id] = []
            target.character_arcs[char_id] = ["introduced"]

        for source, rels in static_memory.characterization.initial_relations.items():
            target.relations.setdefault(source, {})
            target.relations[source].update(rels)

    def _refresh_state_memory(
        self,
        memory: DynamicMemory,
        step_index: int | None = None,
        append_history: bool = True,
    ) -> None:
        state = self._synthesize_state(memory, step_index=step_index)
        memory.current_state = state
        if not append_history:
            if not memory.state_history:
                memory.state_history.append(deepcopy(state))
            return
        if not memory.state_history:
            memory.state_history.append(deepcopy(state))
            return
        prev = memory.state_history[-1]
        if prev.state_id == state.state_id and prev.step_index == state.step_index:
            memory.state_history[-1] = deepcopy(state)
            return
        memory.state_history.append(deepcopy(state))

    def _synthesize_state(self, memory: DynamicMemory, step_index: int | None = None) -> StoryState:
        timeline = memory.timeline_plot.event_timeline
        inferred_step = step_index
        if inferred_step is None:
            if timeline:
                inferred_step = max(0, int(timeline[-1].order))
            else:
                inferred_step = max(0, int(memory.current_state.step_index))

        char_states = {
            entity_id: CharacterStateSnapshot(
                entity_id=entity_id,
                name=entity.name,
                status=entity.status,
                location=entity.location,
                last_event_order=int(entity.last_updated_step),
            )
            for entity_id, entity in memory.characterization.entity_store.items()
        }
        last_event_id = timeline[-1].event_id if timeline else ""
        last_event_order = int(timeline[-1].order) if timeline else -1
        recent_event_ids = [event.event_id for event in timeline[-8:]]

        active_locations = list(memory.world_setting.location_states.keys())
        current_location = (memory.world_setting.current_setting_state or "").strip()
        if current_location and current_location not in active_locations:
            active_locations.append(current_location)

        world = WorldStateSnapshot(
            current_setting_state=current_location or "unknown",
            active_locations=active_locations,
            global_facts=list(memory.world_setting.global_world_facts[-12:]),
            world_rule_activations=list(memory.world_setting.world_rule_activations[-12:]),
        )
        timeline_state = TimelineStateSnapshot(
            last_event_id=last_event_id,
            last_event_order=last_event_order,
            recent_event_ids=recent_event_ids,
            pending_constraints=list(memory.timeline_plot.pending_constraints[-20:]),
        )
        state = StoryState(
            state_id=f"state_{int(inferred_step):03d}",
            step_index=int(inferred_step),
            character_states=char_states,
            relations={k: dict(v) for k, v in memory.characterization.relations.items()},
            world_state=world,
            timeline=timeline_state,
            active_constraints=list(memory.inferred_constraints),
            derived_facts={
                "num_numeric_facts": len(memory.factual_detail.numeric_facts),
                "num_object_states": len(memory.factual_detail.object_states),
                "active_goals": list(memory.timeline_plot.active_goals[-8:]),
            },
        )
        return state
