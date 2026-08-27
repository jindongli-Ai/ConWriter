"""Hierarchical story planner for incremental scene generation."""

from __future__ import annotations

import math
from typing import List

from ConWriter.config.schema import IncrementalConfig, PlanningConfig
from ConWriter.utils.common import short_text, slugify_token
from ConWriter.utils.types import (
    ChapterPlan,
    ConWriterPromptSample,
    ScenePlan,
    StaticMemory,
    StoryPlan,
)


class StoryPlanner:
    """Build chapter/scene-level plan from prompt and static memory."""

    def __init__(
        self,
        cfg: IncrementalConfig | None = None,
        planning_cfg: PlanningConfig | None = None,
    ):
        self.cfg = cfg or IncrementalConfig()
        self.planning_cfg = planning_cfg or PlanningConfig()

    def build_plan(self, sample: ConWriterPromptSample, static_memory: StaticMemory) -> StoryPlan:
        """Produce a minimal but structured StoryPlan."""
        num_chapters = max(1, int(self.cfg.num_chapters))
        scenes_per_chapter = max(1, int(self.cfg.scenes_per_chapter))
        scene_target_words = max(int(self.cfg.scene_target_words), int(self.planning_cfg.min_scene_words))

        if self.planning_cfg.auto_adjust_scene_count and int(self.planning_cfg.target_story_words) > 0:
            target_story_words = int(self.planning_cfg.target_story_words)
            needed_scenes = max(1, int(math.ceil(target_story_words / max(1, scene_target_words))))
            scenes_per_chapter = max(scenes_per_chapter, int(math.ceil(needed_scenes / num_chapters)))

        character_ids = self._validated_character_ids(static_memory)
        if not character_ids:
            character_ids = ["char_protagonist"]

        required_plot_points = list(static_memory.timeline_plot.required_plot_points)
        forbidden_outcomes = list(static_memory.timeline_plot.forbidden_plot_outcomes)
        world_invariants = list(static_memory.world_setting.world_invariants)
        style_directives = list(static_memory.style.style_constraints)
        if not style_directives and static_memory.style.tone != "unspecified":
            style_directives.append(f"Maintain tone: {static_memory.style.tone}")

        chapters: List[ChapterPlan] = []
        scene_counter = 0
        for cidx in range(num_chapters):
            chapter_id = f"chapter_{cidx + 1:02d}"
            chapter = ChapterPlan(
                chapter_id=chapter_id,
                chapter_index=cidx,
                title=self._chapter_title(cidx, num_chapters),
                objective=self._chapter_objective(cidx, num_chapters, static_memory),
                scenes=[],
            )

            for sidx in range(scenes_per_chapter):
                scene_id = f"scene_{scene_counter:03d}"
                required_characters = self._pick_characters(character_ids, scene_counter)
                optional_characters = [cid for cid in character_ids if cid not in required_characters][:2]
                involved = required_characters + optional_characters
                key_events = self._pick_key_events(required_plot_points, cidx, sidx)
                required_constraints = self._scene_required_constraints(
                    static_memory,
                    key_events,
                )
                must_keep_constraints = self._must_keep_constraints(static_memory)
                expected_state_changes = self._expected_state_changes(required_characters, cidx, sidx)
                preconditions = self._preconditions(chapter_id=chapter_id, scene_index=scene_counter)
                forbidden_state_changes = self._forbidden_state_changes(static_memory)
                dependency_scenes = self._dependency_scenes(scene_counter)

                chapter.scenes.append(
                    ScenePlan(
                        scene_id=scene_id,
                        chapter_id=chapter_id,
                        scene_index=scene_counter,
                        title=self._scene_title(cidx, sidx),
                        objective=self._scene_objective(cidx, sidx, num_chapters, scenes_per_chapter),
                        key_events=key_events,
                        required_characters=required_characters,
                        optional_characters=optional_characters,
                        involved_characters=involved,
                        preconditions=preconditions,
                        expected_state_changes=expected_state_changes,
                        forbidden_state_changes=forbidden_state_changes,
                        dependency_scenes=dependency_scenes,
                        must_keep_constraints=must_keep_constraints,
                        required_constraints=required_constraints,
                        forbidden_constraints=forbidden_outcomes[:3],
                        target_words=max(120, scene_target_words),
                    )
                )
                scene_counter += 1

            chapters.append(chapter)

        return StoryPlan(
            premise=short_text(sample.prompt, 200),
            global_objective=static_memory.story_spec.goal,
            chapters=chapters,
            style_directives=style_directives[:6],
            world_invariants=world_invariants[:8],
        )

    def _chapter_title(self, chapter_index: int, num_chapters: int) -> str:
        if chapter_index == 0:
            return "Setup"
        if chapter_index == num_chapters - 1:
            return "Resolution"
        return f"Escalation-{chapter_index}"

    def _chapter_objective(
        self,
        chapter_index: int,
        num_chapters: int,
        static_memory: StaticMemory,
    ) -> str:
        theme = static_memory.story_spec.theme or "core conflict"
        if chapter_index == 0:
            return f"Establish premise and characters around {theme}."
        if chapter_index == num_chapters - 1:
            return "Resolve major tension while preserving established facts."
        return "Increase stakes and causal pressure with consistent world/character states."

    def _scene_title(self, chapter_index: int, scene_index: int) -> str:
        return f"C{chapter_index + 1}S{scene_index + 1}"

    def _scene_objective(
        self,
        chapter_index: int,
        scene_index: int,
        num_chapters: int,
        scenes_per_chapter: int,
    ) -> str:
        if chapter_index == 0 and scene_index == 0:
            return "Introduce setting, cast, and initial objective."
        if chapter_index == num_chapters - 1 and scene_index == scenes_per_chapter - 1:
            return "Deliver coherent closure and preserve continuity."
        if scene_index == 0:
            return "Transition from previous events with explicit causal continuity."
        return "Advance conflict through one concrete event while maintaining consistency."

    def _pick_characters(self, character_ids: List[str], scene_index: int) -> List[str]:
        if len(character_ids) <= 2:
            return list(character_ids)
        first = character_ids[scene_index % len(character_ids)]
        second = character_ids[(scene_index + 1) % len(character_ids)]
        return [first, second]

    def _validated_character_ids(self, static_memory: StaticMemory) -> List[str]:
        valid_ids: List[str] = []
        blocked_names = {
            "after",
            "aim",
            "begin",
            "continue",
            "create",
            "explore",
            "generate",
            "imagine",
            "include",
            "introduce",
            "set",
            "start",
            "story",
            "write",
        }
        for cid, profile in static_memory.characterization.character_profiles.items():
            canonical = str(profile.canonical_name or "").strip()
            expected_id = f"char_{slugify_token(canonical)}"
            if canonical.lower() in blocked_names and str(profile.role).lower() == "character":
                continue
            if not cid.startswith("char_"):
                continue
            if cid != expected_id and profile.traits.get("source") == "prompt_heuristic":
                continue
            valid_ids.append(cid)
        return valid_ids

    def _pick_key_events(self, required_plot_points: List[str], chapter_index: int, scene_index: int) -> List[str]:
        if not required_plot_points:
            return [f"scene_event_{chapter_index}_{scene_index}"]

        ptr = (chapter_index + scene_index) % len(required_plot_points)
        total = len(required_plot_points)
        for offset in range(total):
            candidate = str(required_plot_points[(ptr + offset) % total] or "").strip()
            if not candidate:
                continue
            # Prompt-style instructions are too broad to be hard scene-level events.
            if self._is_instructional_plot_point(candidate):
                continue
            return [short_text(candidate, 120)]

        return [f"scene_event_{chapter_index}_{scene_index}"]

    def _scene_required_constraints(self, static_memory: StaticMemory, key_events: List[str]) -> List[str]:
        constraints: List[str] = []
        constraints.extend(static_memory.world_setting.world_invariants[:2])
        constraints.extend(static_memory.characterization.identity_constraints[:2])
        for event in key_events[:1]:
            token = str(event or "").strip()
            # Placeholder key events should not become hard lexical constraints.
            if token and not token.startswith("scene_event_"):
                constraints.append(token)
        return [c for c in constraints if c]

    def _is_instructional_plot_point(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        prefixes = (
            "write a story",
            "start with",
            "include ",
            "the story should",
            "make sure",
        )
        return lowered.startswith(prefixes)

    def _must_keep_constraints(self, static_memory: StaticMemory) -> List[str]:
        constraints: List[str] = []
        constraints.extend(static_memory.world_setting.world_invariants[:3])
        constraints.extend(static_memory.timeline_plot.forbidden_plot_outcomes[:3])
        constraints.extend(static_memory.style.forbidden_style_shifts[:2])
        return [c for c in constraints if c]

    def _expected_state_changes(
        self,
        required_characters: List[str],
        chapter_index: int,
        scene_index: int,
    ) -> List[str]:
        changes: List[str] = []
        if required_characters:
            changes.append(f"{required_characters[0]} advances local objective")
        if len(required_characters) >= 2:
            changes.append(f"{required_characters[0]} and {required_characters[1]} relation updates")
        changes.append(f"scene_progression:c{chapter_index + 1}s{scene_index + 1}")
        return changes

    def _preconditions(self, chapter_id: str, scene_index: int) -> List[str]:
        if scene_index == 0:
            return [f"{chapter_id}:story_initialized"]
        return [f"scene_{scene_index - 1:03d}:accepted", f"{chapter_id}:continuity_preserved"]

    def _forbidden_state_changes(self, static_memory: StaticMemory) -> List[str]:
        items: List[str] = []
        for token in static_memory.timeline_plot.forbidden_plot_outcomes[:3]:
            if token:
                items.append(f"forbidden_outcome:{short_text(token, 120)}")
        for token in static_memory.world_setting.world_invariants[:2]:
            lowered = token.lower()
            if "cannot" in lowered or "must not" in lowered or "never" in lowered:
                items.append(f"world_invariant:{short_text(token, 120)}")
        return items

    def _dependency_scenes(self, scene_index: int) -> List[str]:
        if scene_index <= 0:
            return []
        return [f"scene_{scene_index - 1:03d}"]
