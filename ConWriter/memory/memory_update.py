"""Memory update utilities for DynamicMemory with MemoryDelta payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from ConWriter.memory.memory_graph import MemoryGraph
from ConWriter.utils.types import ConsistencyReport, DynamicMemory, EntityState, MemoryDelta


class MemoryUpdater:
    """Apply memory deltas with copy-safe operations."""

    def preview(self, memory: DynamicMemory, delta: MemoryDelta) -> DynamicMemory:
        """Return a preview memory without mutating the input memory."""
        clone = deepcopy(memory)
        return self.apply(clone, delta, report=None)

    def apply(
        self,
        memory: DynamicMemory,
        delta: MemoryDelta,
        report: Optional[ConsistencyReport] = None,
    ) -> DynamicMemory:
        """Mutate and return dynamic memory using one MemoryDelta."""
        added_entity_ids: List[str] = []
        updated_entity_ids: List[str] = []

        for entity in delta.new_entities:
            self._upsert_entity(memory, entity)
            added_entity_ids.append(entity.entity_id)

        for entity in delta.updated_entities:
            self._upsert_entity(memory, entity)
            updated_entity_ids.append(entity.entity_id)

        added_event_ids: List[str] = []
        for event in delta.new_events:
            memory.timeline_plot.event_timeline.append(event)
            added_event_ids.append(event.event_id)

        if delta.temporal_links:
            memory.timeline_plot.temporal_links.extend(delta.temporal_links)
        if delta.causal_links:
            memory.timeline_plot.causal_links.extend(delta.causal_links)

        self._apply_fact_updates(memory, delta.new_facts)
        self._apply_style_updates(memory, delta.style_updates)
        self._apply_plot_updates(memory, delta.plot_updates)
        self._apply_world_updates(memory, delta.world_updates)

        relation_graph = MemoryGraph.from_relations(memory.characterization.relations)
        relation_graph.merge(delta.new_relations)
        memory.characterization.relations = relation_graph.edges

        world_overwrites, fact_overwrites = self._classify_overwrites(delta.overwritten_states)
        if delta.overwritten_states:
            memory.revision_records.append(
                {
                    "chunk_id": delta.chunk_id,
                    "overwritten_states": list(delta.overwritten_states),
                    "world_overwrites": world_overwrites,
                    "fact_overwrites": fact_overwrites,
                    "note": "State overwrite requested by memory delta.",
                }
            )

        history_item: Dict[str, Any] = {
            "chunk_id": delta.chunk_id,
            "accepted": True,
            "new_entity_ids": added_entity_ids,
            "updated_entity_ids": updated_entity_ids,
            "new_event_ids": added_event_ids,
            "new_fact_keys": sorted(delta.new_facts.keys()),
            "plot_update_keys": sorted(delta.plot_updates.keys()),
            "world_update_keys": sorted(delta.world_updates.keys()),
            "world_overwrites": world_overwrites,
            "fact_overwrites": fact_overwrites,
            "confidence": delta.confidence,
            "evidence": list(delta.raw_evidence_spans),
            "relation_graph": relation_graph.as_summary(),
        }
        memory.memory_history.append(history_item)
        memory.accepted_deltas.append(delta)

        if report is not None:
            memory.consistency_reports.append(report)
        return memory

    def record_rejected(
        self,
        memory: DynamicMemory,
        delta: MemoryDelta,
        report: Optional[ConsistencyReport] = None,
        reason: str = "rejected_by_policy",
    ) -> DynamicMemory:
        """Record a rejected delta without mutating facet states."""
        world_overwrites, fact_overwrites = self._classify_overwrites(delta.overwritten_states)
        memory.rejected_deltas.append(delta)
        memory.memory_history.append(
            {
                "chunk_id": delta.chunk_id,
                "accepted": False,
                "reason": reason,
                "new_fact_keys": sorted(delta.new_facts.keys()),
                "plot_update_keys": sorted(delta.plot_updates.keys()),
                "world_update_keys": sorted(delta.world_updates.keys()),
                "world_overwrites": world_overwrites,
                "fact_overwrites": fact_overwrites,
                "confidence": delta.confidence,
                "evidence": list(delta.raw_evidence_spans),
            }
        )
        if report is not None:
            memory.consistency_reports.append(report)
        return memory

    def _upsert_entity(self, memory: DynamicMemory, entity: EntityState) -> None:
        memory.characterization.entity_store[entity.entity_id] = entity
        memory.characterization.current_states[entity.entity_id] = entity.status
        memory.characterization.locations[entity.entity_id] = entity.location
        memory.characterization.knowledge_states[entity.entity_id] = list(entity.knowledge)
        memory.characterization.motivation_states[entity.entity_id] = list(entity.motivations)
        memory.characterization.ability_states[entity.entity_id] = list(entity.abilities)
        if entity.relations:
            memory.characterization.relations.setdefault(entity.entity_id, {})
            memory.characterization.relations[entity.entity_id].update(entity.relations)

    def _apply_fact_updates(self, memory: DynamicMemory, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        if "stable_facts" in updates and isinstance(updates["stable_facts"], list):
            memory.factual_detail.stable_facts.extend(str(v) for v in updates["stable_facts"])
        if "surface_attributes" in updates and isinstance(updates["surface_attributes"], dict):
            for key, payload in updates["surface_attributes"].items():
                if isinstance(payload, dict):
                    memory.factual_detail.surface_attributes.setdefault(key, {}).update(payload)
        if "numeric_facts" in updates and isinstance(updates["numeric_facts"], dict):
            for key, value in updates["numeric_facts"].items():
                try:
                    memory.factual_detail.numeric_facts[key] = float(value)
                except (TypeError, ValueError):
                    continue
        if "object_states" in updates and isinstance(updates["object_states"], dict):
            for key, value in updates["object_states"].items():
                memory.factual_detail.object_states[str(key)] = str(value)
        if "name_references" in updates and isinstance(updates["name_references"], dict):
            for key, refs in updates["name_references"].items():
                if isinstance(refs, list):
                    memory.factual_detail.name_references[str(key)] = [str(v) for v in refs]

    def _apply_style_updates(self, memory: DynamicMemory, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        if "current_pov" in updates:
            memory.narrative_style.current_pov = str(updates["current_pov"])
        if "current_tense" in updates:
            memory.narrative_style.current_tense = str(updates["current_tense"])
        if "tone" in updates:
            memory.narrative_style.tone_trace.append(str(updates["tone"]))
        if "style_note" in updates:
            memory.narrative_style.recent_style_notes.append(str(updates["style_note"]))
        if "style_signature" in updates and isinstance(updates["style_signature"], dict):
            memory.narrative_style.style_signature.update(updates["style_signature"])
        if "style_violations" in updates and isinstance(updates["style_violations"], list):
            memory.narrative_style.style_violations_history.extend(
                str(v) for v in updates["style_violations"]
            )

    def _apply_world_updates(self, memory: DynamicMemory, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        if "current_setting_state" in updates:
            memory.world_setting.current_setting_state = str(updates["current_setting_state"])
        if "location_states" in updates and isinstance(updates["location_states"], dict):
            for key, value in updates["location_states"].items():
                memory.world_setting.location_states[str(key)] = str(value)
        if "norm_status" in updates and isinstance(updates["norm_status"], dict):
            for key, value in updates["norm_status"].items():
                memory.world_setting.norm_status[str(key)] = str(value)
        if "world_rule_activations" in updates and isinstance(updates["world_rule_activations"], list):
            memory.world_setting.world_rule_activations.extend(
                str(v) for v in updates["world_rule_activations"]
            )
        if "environment_changes" in updates and isinstance(updates["environment_changes"], list):
            memory.world_setting.environment_changes.extend(str(v) for v in updates["environment_changes"])
        if "global_world_facts" in updates and isinstance(updates["global_world_facts"], list):
            memory.world_setting.global_world_facts.extend(str(v) for v in updates["global_world_facts"])

    def _apply_plot_updates(self, memory: DynamicMemory, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        if "current_chapter_id" in updates:
            memory.current_chapter_id = str(updates["current_chapter_id"])
        if "current_scene_id" in updates:
            memory.current_scene_id = str(updates["current_scene_id"])
        if "active_goals" in updates and isinstance(updates["active_goals"], list):
            memory.timeline_plot.active_goals = [str(v) for v in updates["active_goals"]]
        if "resolved_goals" in updates and isinstance(updates["resolved_goals"], list):
            memory.timeline_plot.resolved_goals = [str(v) for v in updates["resolved_goals"]]
        if "active_plot_threads" in updates and isinstance(updates["active_plot_threads"], list):
            memory.timeline_plot.active_plot_threads = [str(v) for v in updates["active_plot_threads"]]
        if "resolved_plot_threads" in updates and isinstance(updates["resolved_plot_threads"], list):
            memory.timeline_plot.resolved_plot_threads.extend(str(v) for v in updates["resolved_plot_threads"])
            memory.timeline_plot.resolved_plot_threads = sorted(
                set(memory.timeline_plot.resolved_plot_threads)
            )
        if "opened_plot_threads" in updates and isinstance(updates["opened_plot_threads"], list):
            for value in updates["opened_plot_threads"]:
                token = str(value).strip()
                if token and token not in memory.timeline_plot.active_plot_threads:
                    memory.timeline_plot.active_plot_threads.append(token)
        if "closed_plot_threads" in updates and isinstance(updates["closed_plot_threads"], list):
            closed = {str(v).strip() for v in updates["closed_plot_threads"] if str(v).strip()}
            if closed:
                memory.timeline_plot.active_plot_threads = [
                    t for t in memory.timeline_plot.active_plot_threads if t not in closed
                ]
                memory.timeline_plot.resolved_plot_threads.extend(sorted(closed))
                memory.timeline_plot.resolved_plot_threads = sorted(
                    set(memory.timeline_plot.resolved_plot_threads)
                )
        if "pending_constraints" in updates and isinstance(updates["pending_constraints"], list):
            memory.timeline_plot.pending_constraints = [str(v) for v in updates["pending_constraints"]]
        if "unresolved_foreshadowing" in updates and isinstance(updates["unresolved_foreshadowing"], list):
            memory.timeline_plot.unresolved_foreshadowing = [str(v) for v in updates["unresolved_foreshadowing"]]

    def _classify_overwrites(self, overwritten_states: List[str]) -> Tuple[List[str], List[str]]:
        world_overwrites: List[str] = []
        fact_overwrites: List[str] = []
        for slot in overwritten_states:
            text = str(slot)
            if text.startswith(("world_setting.", "world_updates.", "location_states.", "current_setting_state")):
                world_overwrites.append(text)
                continue
            if text.startswith(
                (
                    "factual_detail.",
                    "numeric_facts.",
                    "object_states.",
                    "surface_attributes.",
                    "name_references.",
                )
            ):
                fact_overwrites.append(text)
        return sorted(world_overwrites), sorted(fact_overwrites)
