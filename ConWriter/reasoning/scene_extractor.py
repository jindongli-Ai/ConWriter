"""Scene extraction module for incremental generation.

This module converts one generated scene text into a structured SceneExtraction
object that can be transformed into MemoryDelta.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Set, Tuple
from urllib import error, request

from ConWriter.config.schema import LLMConfig
from ConWriter.reasoning.scene_alignment import (
    build_sentence_coref_links,
    map_entities_to_sentences,
    map_events_with_inference,
    split_scene_into_units,
)
from ConWriter.utils.common import short_text, slugify_token
from ConWriter.utils.types import (
    EntityState,
    GenerationState,
    SceneExtraction,
    ScenePlan,
    StoryEvent,
)

_LOCATION_HINTS = [
    "home",
    "house",
    "city",
    "harbor",
    "forest",
    "school",
    "castle",
    "village",
    "room",
    "kitchen",
    "street",
]

_NAME_STOPWORDS = {
    "The",
    "This",
    "That",
    "Then",
    "Later",
    "After",
    "Before",
    "Meanwhile",
    "Finally",
    "When",
    "Where",
    "He",
    "She",
    "They",
    "His",
    "Her",
    "Their",
}


class SceneExtractor:
    """Extract structured memory signals from one scene text.

    Priority path:
    1) LLM JSON schema extraction + normalization
    2) heuristic fallback extractor
    """

    def __init__(self, llm_config: LLMConfig | None = None, logger: logging.Logger | None = None):
        self.llm_config = llm_config or LLMConfig()
        self.logger = logger or logging.getLogger("ConWriter.scene_extractor")

    def extract_scene(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> SceneExtraction:
        """Extract scene-level entities, events, and memory updates."""
        text = (scene_text or "").strip()
        llm_extraction = self._extract_with_llm_json(scene_plan, text, state)
        if llm_extraction is not None:
            return self._attach_sentence_alignment(llm_extraction, scene_plan, text, state)
        extraction = self._extract_with_heuristics(scene_plan, text, state)
        return self._attach_sentence_alignment(extraction, scene_plan, text, state)

    def _extract_with_llm_json(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> SceneExtraction | None:
        if os.getenv("CONWRITER_DISABLE_LLM_EXTRACTION", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None
        if not self.llm_config.enabled:
            return None
        api_key = self.llm_config.api_key.strip() or os.getenv(self.llm_config.api_key_env, "").strip()
        if not api_key:
            return None

        payload = {
            "model": self.llm_config.model,
            "messages": self._build_extraction_messages(scene_plan, scene_text, state),
            "temperature": 0.0,
            "max_tokens": int(self.llm_config.request_max_tokens or 1200),
        }
        if self.llm_config.extra_request_body:
            payload.update(self.llm_config.extra_request_body)
        model_id = str(self.llm_config.model or "").strip().lower()
        if "max_completion_tokens" in payload:
            payload.pop("max_tokens", None)
        elif model_id.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens", int(self.llm_config.request_max_tokens or 1200))

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for key, value in self.llm_config.extra_headers.items():
            headers[str(key)] = str(value)

        url = self.llm_config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.llm_config.timeout_seconds)) as resp:
                body = resp.read().decode("utf-8")
        except (error.URLError, error.HTTPError) as exc:
            self.logger.warning("Scene extraction LLM call failed: %s", exc)
            return None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None

        raw_text = self._extract_text(parsed)
        if not raw_text:
            return None

        extraction_obj = self._parse_json_object(raw_text)
        if not isinstance(extraction_obj, dict):
            return None

        try:
            normalized = self._normalize_llm_extraction(scene_plan, scene_text, extraction_obj, state)
            return normalized
        except Exception as exc:  # pragma: no cover - defensive path
            self.logger.warning("Scene extraction normalization failed: %s", exc)
            return None

    def _build_extraction_messages(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> List[Dict[str, str]]:
        static_profiles = state.static_memory.characterization.character_profiles
        profile_lines = []
        for cid in scene_plan.required_characters or scene_plan.involved_characters:
            profile = static_profiles.get(cid)
            if profile is None:
                continue
            profile_lines.append(f"- {cid}: canonical_name={profile.canonical_name}, aliases={profile.aliases}")
        if not profile_lines:
            profile_lines = ["- (none)"]

        recent_events = [
            f"- {evt.event_id}: {short_text(evt.description, 120)}"
            for evt in state.dynamic_memory.timeline_plot.event_timeline[-4:]
        ] or ["- (none)"]

        schema = {
            "entities": [
                {
                    "entity_id": "char_name",
                    "name": "Canonical Name",
                    "status": "active|offstage|injured|...",
                    "location": "city_square",
                    "relations": {"char_other": "friend"},
                    "knowledge": ["..."],
                    "motivations": ["..."],
                    "abilities": ["..."]
                }
            ],
            "events": [
                {
                    "event_id": "evt_scene_001",
                    "description": "short event",
                    "order": 1,
                    "participants": ["char_a"],
                    "location": "city_square"
                }
            ],
            "relation_updates": {"char_a": {"char_b": "trusts"}},
            "fact_updates": {
                "stable_facts": ["..."],
                "name_references": {"char_a": ["Alice"]},
                "numeric_facts": {"hero_age": 18},
                "object_states": {"amulet": "intact"}
            },
            "style_updates": {"current_pov": "third_person", "current_tense": "past"},
            "world_updates": {
                "current_setting_state": "city_square",
                "location_states": {"city_square": "active_scene"}
            },
            "plot_updates": {
                "opened_plot_threads": ["..."],
                "closed_plot_threads": ["..."],
                "pending_constraints": ["..."],
                "unresolved_foreshadowing": ["..."]
            },
            "temporal_links": [{"from": "evt_001", "to": "evt_002", "relation": "before"}],
            "causal_links": [{"cause": "evt_001", "effect": "evt_002", "relation": "causes"}],
            "overwritten_states": [],
            "raw_evidence_spans": ["..."],
            "extraction_notes": ["..."],
            "confidence": 0.7
        }

        user = (
            f"You are a strict information extractor. Return ONLY one JSON object.\n\n"
            f"Scene ID: {scene_plan.scene_id}\n"
            f"Scene objective: {scene_plan.objective}\n"
            f"Required characters: {scene_plan.required_characters or scene_plan.involved_characters}\n"
            f"Expected state changes: {scene_plan.expected_state_changes}\n"
            f"Must keep constraints: {scene_plan.must_keep_constraints}\n\n"
            f"Static character profiles:\n{chr(10).join(profile_lines)}\n\n"
            f"Recent events:\n{chr(10).join(recent_events)}\n\n"
            f"Scene text:\n{scene_text}\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Rules:"
            "\n1) Use canonicalized entity IDs when possible (char_xxx)."
            "\n2) Do not hallucinate unrelated entities/events."
            "\n3) Keep arrays/dicts present even if empty."
            "\n4) Output valid JSON only."
        )

        return [
            {"role": "system", "content": "You extract structured story state into JSON."},
            {"role": "user", "content": user},
        ]

    def _extract_with_heuristics(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> SceneExtraction:
        inferred_location = self._infer_location(scene_text, state)

        new_entities, updated_entities, mentioned_ids = self._extract_entities(
            scene_plan=scene_plan,
            scene_text=scene_text,
            inferred_location=inferred_location,
            state=state,
        )
        extra_entities = self._extract_open_world_entities(scene_text, state, scene_plan, inferred_location)
        for entity in extra_entities:
            if entity.entity_id not in {e.entity_id for e in new_entities + updated_entities}:
                new_entities.append(entity)

        relation_updates = self._build_relation_updates(new_entities, updated_entities)
        new_event = self._build_event(
            scene_plan=scene_plan,
            scene_text=scene_text,
            state=state,
            participants=[e.entity_id for e in (new_entities + updated_entities)],
            inferred_location=inferred_location,
        )
        temporal_links = self._build_temporal_links(new_event.event_id, state)
        causal_links = self._build_causal_links(new_event.event_id, scene_text)

        fact_updates = self._build_fact_updates(scene_text, new_entities, updated_entities)
        style_updates = self._build_style_updates(scene_text)
        world_updates = self._build_world_updates(inferred_location, scene_plan, new_entities, updated_entities)
        plot_updates = self._build_plot_updates(scene_plan, scene_text, state)
        confidence = self._estimate_confidence(
            scene_plan=scene_plan,
            scene_text=scene_text,
            mentioned_ids=mentioned_ids,
            inferred_location=inferred_location,
        )

        evidence = [short_text(scene_text, 240)] if scene_text else []
        notes = [
            "SceneExtractor heuristic parse",
            f"involved={len(scene_plan.involved_characters)}",
            f"mentioned={len(mentioned_ids)}",
        ]

        return SceneExtraction(
            scene_id=scene_plan.scene_id,
            new_entities=new_entities,
            updated_entities=updated_entities,
            new_events=[new_event],
            relation_updates=relation_updates,
            fact_updates=fact_updates,
            style_updates=style_updates,
            plot_updates=plot_updates,
            temporal_links=temporal_links,
            causal_links=causal_links,
            world_updates=world_updates,
            overwritten_states=[],
            raw_evidence_spans=evidence,
            extraction_notes=notes,
            confidence=confidence,
            scene_text=scene_text,
        )

    def _normalize_llm_extraction(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        payload: Dict[str, Any],
        state: GenerationState,
    ) -> SceneExtraction:
        def _to_list(value: Any) -> List[Any]:
            if isinstance(value, list):
                return value
            if value is None:
                return []
            return [value]

        def _to_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        entities = _to_list(payload.get("entities"))
        new_entities: List[EntityState] = []
        updated_entities: List[EntityState] = []
        dynamic_store = state.dynamic_memory.characterization.entity_store

        for item in entities:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not entity_id:
                token = name or "character"
                entity_id = f"char_{slugify_token(token)}"
            if not name:
                name = entity_id.replace("char_", "").replace("_", " ").title()
            entity = EntityState(
                entity_id=entity_id,
                name=name,
                status=str(item.get("status") or "active"),
                location=str(item.get("location") or "unspecified"),
                relations=_to_dict(item.get("relations")),
                goals=[],
                knowledge=[str(v) for v in _to_list(item.get("knowledge"))],
                motivations=[str(v) for v in _to_list(item.get("motivations"))],
                abilities=[str(v) for v in _to_list(item.get("abilities"))],
                attributes={"source": "llm_structured_extraction", "scene_id": scene_plan.scene_id},
                last_updated_step=scene_plan.scene_index,
            )
            if entity_id in dynamic_store:
                updated_entities.append(entity)
            else:
                new_entities.append(entity)

        if not (new_entities or updated_entities):
            # Backfill minimum entity coverage for required characters.
            fallback = self._extract_entities(
                scene_plan,
                scene_text,
                self._infer_location(scene_text, state),
                state,
            )
            new_entities, updated_entities, _ = fallback

        events_payload = _to_list(payload.get("events"))
        new_events: List[StoryEvent] = []
        for idx, event_item in enumerate(events_payload):
            if not isinstance(event_item, dict):
                continue
            event_id = str(event_item.get("event_id") or f"evt_scene_{scene_plan.scene_index:03d}_{idx}")
            description = str(event_item.get("description") or short_text(scene_text, 320))
            order = int(event_item.get("order") or scene_plan.scene_index)
            participants = [str(v) for v in _to_list(event_item.get("participants"))]
            location = str(event_item.get("location") or "unknown")
            new_events.append(
                StoryEvent(
                    event_id=event_id,
                    description=description,
                    order=order,
                    participants=participants,
                    location=location,
                    evidence_chunk_id=scene_plan.scene_id,
                )
            )
        if not new_events:
            new_events = [
                self._build_event(
                    scene_plan,
                    scene_text,
                    state,
                    [e.entity_id for e in (new_entities + updated_entities)],
                    self._infer_location(scene_text, state),
                )
            ]

        relation_updates = _to_dict(payload.get("relation_updates"))
        fact_updates = _to_dict(payload.get("fact_updates"))
        style_updates = _to_dict(payload.get("style_updates"))
        world_updates = _to_dict(payload.get("world_updates"))
        plot_updates = _to_dict(payload.get("plot_updates"))

        if "current_chapter_id" not in plot_updates:
            plot_updates["current_chapter_id"] = scene_plan.chapter_id
        if "current_scene_id" not in plot_updates:
            plot_updates["current_scene_id"] = scene_plan.scene_id

        temporal_links = [v for v in _to_list(payload.get("temporal_links")) if isinstance(v, dict)]
        causal_links = [v for v in _to_list(payload.get("causal_links")) if isinstance(v, dict)]
        overwritten_states = [str(v) for v in _to_list(payload.get("overwritten_states"))]
        evidence = [str(v) for v in _to_list(payload.get("raw_evidence_spans")) if str(v).strip()]
        if not evidence and scene_text:
            evidence = [short_text(scene_text, 240)]
        notes = [str(v) for v in _to_list(payload.get("extraction_notes")) if str(v).strip()]
        if not notes:
            notes = ["llm_structured_extractor"]

        confidence = payload.get("confidence", 0.72)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.72
        confidence = max(0.2, min(0.95, confidence))

        return SceneExtraction(
            scene_id=scene_plan.scene_id,
            new_entities=new_entities,
            updated_entities=updated_entities,
            new_events=new_events,
            relation_updates=relation_updates,
            fact_updates=fact_updates,
            style_updates=style_updates,
            plot_updates=plot_updates,
            temporal_links=temporal_links,
            causal_links=causal_links,
            world_updates=world_updates,
            overwritten_states=overwritten_states,
            raw_evidence_spans=evidence,
            extraction_notes=notes,
            confidence=confidence,
            scene_text=scene_text,
        )

    def _attach_sentence_alignment(
        self,
        extraction: SceneExtraction,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> SceneExtraction:
        """Attach sentence units and sentence-level mention maps."""
        sentences = split_scene_into_units(scene_text, scene_plan.scene_id)
        extraction.scene_text = scene_text
        extraction.sentences = sentences

        entity_aliases: Dict[str, List[str]] = {}
        for entity_id, profile in state.static_memory.characterization.character_profiles.items():
            aliases = [profile.canonical_name] + list(profile.aliases)
            aliases.extend([entity_id, entity_id.replace("char_", ""), entity_id.replace("_", " ")])
            entity_aliases[entity_id] = [str(item) for item in aliases if str(item).strip()]
        for entity in extraction.new_entities + extraction.updated_entities:
            row = entity_aliases.setdefault(entity.entity_id, [])
            row.extend([entity.name, entity.entity_id, entity.entity_id.replace("char_", "")])
            entity_aliases[entity.entity_id] = [str(item) for item in row if str(item).strip()]

        event_signals: Dict[str, List[str]] = {}
        for event in extraction.new_events:
            signals = [event.event_id, short_text(event.description, 60)]
            signals.extend([str(p) for p in event.participants[:3]])
            event_signals[event.event_id] = [str(item) for item in signals if str(item).strip()]
        for idx, req in enumerate(scene_plan.required_constraints[:3]):
            token = str(req).strip()
            if token:
                event_signals[f"required_evt_{idx}"] = [token]
        for recent in state.dynamic_memory.timeline_plot.event_timeline[-2:]:
            event_signals.setdefault(recent.event_id, []).extend([recent.event_id, short_text(recent.description, 60)])

        extraction.sentence_entity_mentions = map_entities_to_sentences(sentences, entity_aliases)
        explicit_events, inferred_events = map_events_with_inference(sentences, event_signals)
        extraction.sentence_event_mentions = explicit_events
        extraction.sentence_inferred_event_mentions = inferred_events
        extraction.sentence_coref_links = build_sentence_coref_links(
            sentences,
            extraction.sentence_entity_mentions,
        )
        return extraction

    def _extract_text(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = item.get("text", "")
                    if txt:
                        chunks.append(str(txt))
            return "\n".join(chunks).strip()
        reasoning_candidates: List[Any] = []
        if isinstance(message, dict):
            reasoning_candidates.extend([message.get("reasoning_content"), message.get("reasoning")])
        if isinstance(first, dict):
            reasoning_candidates.extend([first.get("reasoning_content"), first.get("reasoning")])
        for candidate in reasoning_candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, list):
                chunks: List[str] = []
                for item in candidate:
                    if not isinstance(item, dict):
                        continue
                    txt = item.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
                if chunks:
                    return "\n".join(chunks).strip()
        fallback = first.get("text", "") if isinstance(first, dict) else ""
        if isinstance(fallback, str):
            return fallback.strip()
        return ""

    def _parse_json_object(self, text: str) -> Dict[str, Any] | None:
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _extract_entities(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        inferred_location: str,
        state: GenerationState,
    ) -> Tuple[List[EntityState], List[EntityState], Set[str]]:
        static_profiles = state.static_memory.characterization.character_profiles
        dynamic_store = state.dynamic_memory.characterization.entity_store

        required = scene_plan.required_characters or scene_plan.involved_characters
        if not required:
            required = ["char_protagonist"]

        new_entities: List[EntityState] = []
        updated_entities: List[EntityState] = []
        mentioned_ids: Set[str] = set()

        lower_text = scene_text.lower()
        for char_id in required:
            profile = static_profiles.get(char_id)
            existing = dynamic_store.get(char_id)
            canonical = profile.canonical_name if profile else self._char_name_from_id(char_id)
            aliases = list(profile.aliases) if profile else []
            names_to_match = [canonical] + aliases
            mentioned = any(name and name.lower() in lower_text for name in names_to_match)
            if mentioned:
                mentioned_ids.add(char_id)

            status = "active" if mentioned else (existing.status if existing else "offstage")
            location = inferred_location or (existing.location if existing else "unspecified")
            abilities = list(existing.abilities) if existing else []
            knowledge = list(existing.knowledge) if existing else []
            motivations = list(existing.motivations) if existing else []
            relations = dict(existing.relations) if existing else {}

            entity = EntityState(
                entity_id=char_id,
                name=canonical,
                status=status,
                location=location,
                attributes={
                    "scene_id": scene_plan.scene_id,
                    "scene_objective": scene_plan.objective,
                    "mentioned_in_scene": mentioned,
                },
                relations=relations,
                goals=[],
                knowledge=knowledge,
                motivations=motivations,
                abilities=abilities,
                last_updated_step=scene_plan.scene_index,
            )

            if existing is None:
                new_entities.append(entity)
            else:
                updated_entities.append(entity)

        return new_entities, updated_entities, mentioned_ids

    def _extract_open_world_entities(
        self,
        scene_text: str,
        state: GenerationState,
        scene_plan: ScenePlan,
        inferred_location: str,
    ) -> List[EntityState]:
        existing_ids = set(state.dynamic_memory.characterization.entity_store.keys())
        static_ids = set(state.static_memory.characterization.character_profiles.keys())
        blocked_ids = existing_ids.union(static_ids).union(set(scene_plan.involved_characters))

        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", scene_text)
        entities: List[EntityState] = []
        for token in candidates:
            if token in _NAME_STOPWORDS:
                continue
            cid = f"char_{slugify_token(token)}"
            if cid in blocked_ids:
                continue
            entities.append(
                EntityState(
                    entity_id=cid,
                    name=token,
                    status="active",
                    location=inferred_location or "unspecified",
                    attributes={"source": "scene_surface_name"},
                    last_updated_step=scene_plan.scene_index,
                )
            )
            blocked_ids.add(cid)
            if len(entities) >= 2:
                break
        return entities

    def _build_event(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
        participants: List[str],
        inferred_location: str,
    ) -> StoryEvent:
        existing_events = state.dynamic_memory.timeline_plot.event_timeline
        next_order = existing_events[-1].order + 1 if existing_events else scene_plan.scene_index

        base_id = f"evt_scene_{scene_plan.scene_index:03d}"
        existing_ids = {ev.event_id for ev in existing_events}
        event_id = base_id
        suffix = 1
        while event_id in existing_ids:
            suffix += 1
            event_id = f"{base_id}_{suffix}"

        return StoryEvent(
            event_id=event_id,
            description=short_text(scene_text, 320),
            order=next_order,
            participants=participants[:6],
            location=inferred_location or "unknown",
            evidence_chunk_id=scene_plan.scene_id,
        )

    def _build_relation_updates(
        self,
        new_entities: List[EntityState],
        updated_entities: List[EntityState],
    ) -> Dict[str, Dict[str, str]]:
        pool = new_entities + updated_entities
        relation_updates: Dict[str, Dict[str, str]] = {}
        ids = [entity.entity_id for entity in pool]
        if len(ids) < 2:
            return relation_updates
        first, second = ids[0], ids[1]
        relation_updates[first] = {second: "co_present"}
        relation_updates[second] = {first: "co_present"}
        return relation_updates

    def _build_temporal_links(self, new_event_id: str, state: GenerationState) -> List[Dict[str, str]]:
        timeline = state.dynamic_memory.timeline_plot.event_timeline
        if not timeline:
            return []
        return [{"from": timeline[-1].event_id, "to": new_event_id, "relation": "before"}]

    def _build_causal_links(self, new_event_id: str, scene_text: str) -> List[Dict[str, str]]:
        lowered = scene_text.lower()
        if any(token in lowered for token in ("because", "therefore", "so that", "as a result")):
            return [{"cause": new_event_id, "effect": new_event_id, "relation": "local_causal_signal"}]
        return []

    def _build_fact_updates(
        self,
        scene_text: str,
        new_entities: List[EntityState],
        updated_entities: List[EntityState],
    ) -> Dict[str, object]:
        refs: Dict[str, List[str]] = {}
        for entity in new_entities + updated_entities:
            refs[entity.entity_id] = [entity.name]

        numeric_facts: Dict[str, float] = {}
        for idx, token in enumerate(re.findall(r"\b\d+(?:\.\d+)?\b", scene_text)[:6]):
            numeric_facts[f"num_scene_{idx}"] = float(token)

        updates: Dict[str, object] = {
            "stable_facts": [short_text(scene_text, 180)] if scene_text else [],
            "name_references": refs,
        }
        if numeric_facts:
            updates["numeric_facts"] = numeric_facts
        return updates

    def _build_style_updates(self, scene_text: str) -> Dict[str, object]:
        lowered = f" {scene_text.lower()} "
        updates: Dict[str, object] = {"style_note": "scene_extractor style trace update"}
        if " i " in lowered:
            updates["current_pov"] = "first_person"
        elif " you " in lowered:
            updates["current_pov"] = "second_person"
        else:
            updates["current_pov"] = "third_person_or_unspecified"

        if " was " in lowered or " were " in lowered:
            updates["current_tense"] = "past"
        elif " is " in lowered or " are " in lowered:
            updates["current_tense"] = "present"
        return updates

    def _build_world_updates(
        self,
        inferred_location: str,
        scene_plan: ScenePlan,
        new_entities: List[EntityState],
        updated_entities: List[EntityState],
    ) -> Dict[str, object]:
        setting = inferred_location or "unknown"
        location_states = {setting: "active_scene"} if setting and setting != "unknown" else {}
        entity_locations = {
            entity.entity_id: [entity.location]
            for entity in (new_entities + updated_entities)
            if entity.location and entity.location != "unknown"
        }

        updates: Dict[str, object] = {
            "current_setting_state": setting,
            "location_states": location_states,
            "environment_changes": [f"{scene_plan.scene_id}: scene progression"],
        }
        if entity_locations:
            updates["entity_locations"] = entity_locations
        return updates

    def _build_plot_updates(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        state: GenerationState,
    ) -> Dict[str, object]:
        opened = self._extract_plot_markers(scene_text, "open_thread")
        closed = self._extract_plot_markers(scene_text, "resolve_thread")
        foreshadow = self._extract_plot_markers(scene_text, "foreshadow")

        pending = list(state.dynamic_memory.timeline_plot.pending_constraints)
        pending.extend(scene_plan.required_constraints)
        if closed:
            closed_set = set(closed)
            pending = [item for item in pending if item not in closed_set]

        active_threads = list(state.dynamic_memory.timeline_plot.active_plot_threads)
        for item in opened:
            if item not in active_threads:
                active_threads.append(item)
        if closed:
            active_threads = [item for item in active_threads if item not in set(closed)]

        return {
            "current_chapter_id": scene_plan.chapter_id,
            "current_scene_id": scene_plan.scene_id,
            "opened_plot_threads": opened,
            "closed_plot_threads": closed,
            "active_plot_threads": active_threads,
            "pending_constraints": pending[-12:],
            "unresolved_foreshadowing": foreshadow[-12:],
        }

    def _extract_plot_markers(self, scene_text: str, marker: str) -> List[str]:
        values: List[str] = []
        pattern = rf"\[{re.escape(marker)}:([^\]]+)\]"
        for token in re.findall(pattern, scene_text):
            cleaned = str(token).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
        return values

    def _estimate_confidence(
        self,
        scene_plan: ScenePlan,
        scene_text: str,
        mentioned_ids: Set[str],
        inferred_location: str,
    ) -> float:
        involved = len(scene_plan.involved_characters)
        mention_ratio = (len(mentioned_ids) / involved) if involved > 0 else 0.6
        has_location = 1.0 if inferred_location else 0.0
        length_signal = 1.0 if len(scene_text.split()) >= 20 else 0.0
        req_hits = 0
        for req in scene_plan.required_constraints:
            token = req.strip().lower()
            if token and token in scene_text.lower():
                req_hits += 1
        req_ratio = (req_hits / len(scene_plan.required_constraints)) if scene_plan.required_constraints else 0.5

        score = 0.35 + (0.25 * mention_ratio) + (0.2 * has_location) + (0.1 * length_signal) + (0.1 * req_ratio)
        return max(0.35, min(0.92, score))

    def _infer_location(self, scene_text: str, state: GenerationState) -> str:
        lowered = scene_text.lower()
        for token in _LOCATION_HINTS:
            if token in lowered:
                return token.replace(" ", "_")

        for phrase in re.findall(r"\b(?:in|at|inside)\s+([A-Za-z][A-Za-z0-9_\- ]{1,24})", scene_text):
            cleaned = re.sub(r"[^A-Za-z0-9_ ]+", "", phrase).strip().lower()
            cleaned = re.sub(r"\s+", "_", cleaned)
            if cleaned:
                return cleaned

        current = (state.dynamic_memory.world_setting.current_setting_state or "").strip().lower()
        if current:
            return re.sub(r"\s+", "_", current)
        return ""

    def _char_name_from_id(self, char_id: str) -> str:
        token = char_id.replace("char_", "").replace("_", " ").strip()
        return token.title() if token else "Character"
