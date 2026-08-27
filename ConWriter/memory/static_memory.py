"""Static memory construction from initial prompt."""

from __future__ import annotations

import re
from typing import Dict, List

from ConWriter.utils.common import (
    extract_sentences_with_keywords,
    normalize_whitespace,
    slugify_token,
)
from ConWriter.utils.types import (
    CharacterProfile,
    CharacterSpecMemory,
    ConWriterPromptSample,
    FactConstraintMemory,
    PlotConstraintMemory,
    StaticMemory,
    StorySpec,
    StyleConstraintMemory,
    WorldRuleMemory,
)


class StaticMemoryBuilder:
    """Build immutable `StaticMemory` from prompt metadata and text."""

    _NON_CHARACTER_TOKENS = {
        "after",
        "aim",
        "alternate",
        "avoid",
        "begin",
        "build",
        "continue",
        "create",
        "describe",
        "end",
        "explore",
        "focus",
        "generate",
        "imagine",
        "include",
        "introduce",
        "make",
        "set",
        "start",
        "story",
        "the",
        "this",
        "use",
        "weave",
        "write",
    }

    _ROLE_HINTS = [
        ("original poster", "Original Poster", "narrator", ["OP", "narrator"]),
        ("narrator", "Narrator", "narrator", ["narrator"]),
        ("protagonist", "Protagonist", "protagonist", ["protagonist"]),
        ("well-meaning friend", "Well-Meaning Friend", "friend", ["friend"]),
        ("best friend", "Best Friend", "friend", ["friend"]),
        ("friend", "Friend", "friend", ["friend"]),
        ("troubled roommate", "Troubled Roommate", "roommate", ["roommate"]),
        ("fourth roommate", "Fourth Roommate", "roommate", ["roommate"]),
        ("roommate", "Roommate", "roommate", ["roommate"]),
        ("girlfriend", "Girlfriend", "partner", ["girlfriend"]),
        ("boyfriend", "Boyfriend", "partner", ["boyfriend"]),
        ("brother", "Brother", "family", ["brother"]),
        ("sister", "Sister", "family", ["sister"]),
        ("mother", "Mother", "family", ["mother"]),
        ("father", "Father", "family", ["father"]),
        ("mentor", "Mentor", "mentor", ["mentor"]),
        ("sidekick", "Sidekick", "companion", ["sidekick"]),
        ("stranger", "Stranger", "character", ["stranger"]),
        ("hero", "Hero", "protagonist", ["hero"]),
    ]

    _GENRE_HINTS = [
        "fantasy",
        "sci-fi",
        "science fiction",
        "thriller",
        "romance",
        "mystery",
        "horror",
        "historical",
        "comedy",
        "drama",
    ]

    def build(self, sample: ConWriterPromptSample) -> StaticMemory:
        """Parse one prompt into static dual-memory schema."""
        prompt = normalize_whitespace(sample.prompt)

        story_spec = StorySpec(
            prompt_id=sample.prompt_id,
            raw_prompt=prompt,
            language=sample.language,
            task_type=sample.task_type,
            theme=self._extract_theme(prompt),
            genre=self._detect_genre(prompt),
            goal=self._extract_goal(prompt),
            target_length_hint=self._extract_target_length_hint(prompt),
        )

        character_profiles = self._build_character_profiles(prompt)
        style_memory = StyleConstraintMemory(
            narrative_pov=self._detect_pov(prompt),
            tense_style=self._detect_tense(prompt),
            tone=self._detect_tone(prompt),
            register=self._detect_register(prompt),
            style_constraints=extract_sentences_with_keywords(
                prompt,
                ["style", "tone", "voice", "warmth", "humor", "formal", "informal"],
            ),
            forbidden_style_shifts=extract_sentences_with_keywords(
                prompt,
                ["do not", "avoid", "never", "without switching", "must remain"],
            ),
        )

        characterization = CharacterSpecMemory(
            character_profiles=character_profiles,
            initial_relations=self._extract_initial_relations(prompt, character_profiles),
            known_abilities=self._extract_known_abilities(prompt, character_profiles),
            identity_constraints=extract_sentences_with_keywords(
                prompt,
                ["identity", "name", "must be", "is a", "who"],
            ),
            knowledge_constraints=extract_sentences_with_keywords(
                prompt,
                ["know", "aware", "unaware", "secret", "memory"],
            ),
        )

        factual_detail = FactConstraintMemory(
            initial_facts=extract_sentences_with_keywords(
                prompt,
                ["include", "start with", "find", "return", "object", "fact"],
            ),
            name_map=self._build_name_map(character_profiles),
            numeric_facts=self._extract_numeric_facts(prompt),
            object_facts=self._extract_object_facts(prompt),
        )

        world_setting = WorldRuleMemory(
            setting_description=self._extract_setting_description(prompt),
            location_constraints=extract_sentences_with_keywords(
                prompt,
                ["at", "in", "home", "city", "village", "room", "location", "setting"],
            ),
            social_norms=extract_sentences_with_keywords(
                prompt,
                ["should", "norm", "social", "custom", "etiquette"],
            ),
            physical_rules=extract_sentences_with_keywords(
                prompt,
                ["physics", "realistic", "possible", "impossible"],
            ),
            magic_rules=extract_sentences_with_keywords(
                prompt,
                ["magic", "spell", "supernatural"],
            ),
            world_invariants=extract_sentences_with_keywords(
                prompt,
                ["must", "cannot", "never", "always"],
            ),
        )

        timeline_plot = PlotConstraintMemory(
            initial_plot_setup=self._extract_initial_plot_setup(prompt),
            global_story_goal=story_spec.goal,
            required_plot_points=extract_sentences_with_keywords(
                prompt,
                ["include", "start with", "then", "finally", "flashback", "ending"],
            ),
            forbidden_plot_outcomes=extract_sentences_with_keywords(
                prompt,
                ["cannot", "never", "must not", "forbidden"],
            ),
            core_conflicts=self._extract_core_conflicts(prompt),
        )

        # TODO: Replace heuristic parsing with robust IE + constrained parser.
        return StaticMemory(
            story_spec=story_spec,
            style=style_memory,
            characterization=characterization,
            factual_detail=factual_detail,
            world_setting=world_setting,
            timeline_plot=timeline_plot,
        )

    def _detect_genre(self, prompt: str) -> str:
        lower = prompt.lower()
        for hint in self._GENRE_HINTS:
            if hint in lower:
                return hint
        return "unknown"

    def _extract_goal(self, prompt: str) -> str:
        match = re.search(r"(the story should .*?)(?:\.|$)", prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Generate a coherent long story consistent with prompt constraints."

    def _extract_theme(self, prompt: str) -> str:
        match = re.search(r"about\s+(.+?)(?:\.|,|;| and )", prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "long-form narrative consistency"

    def _extract_target_length_hint(self, prompt: str) -> str:
        ranged = re.search(r"\b\d[\d,]*\s*-\s*\d[\d,]*\s*words\b", prompt, flags=re.IGNORECASE)
        if ranged:
            return ranged.group(0)
        single = re.search(r"\b\d[\d,]*\s*words\b", prompt, flags=re.IGNORECASE)
        if single:
            return single.group(0)
        return ""

    def _build_character_profiles(self, prompt: str) -> Dict[str, CharacterProfile]:
        names = self._extract_character_mentions(prompt)
        profiles: Dict[str, CharacterProfile] = {}
        for name in names:
            cid = f"char_{slugify_token(name)}"
            profiles[cid] = CharacterProfile(
                character_id=cid,
                canonical_name=name,
                aliases=[name],
                role="character",
                traits={"source": "prompt_heuristic"},
                constraints=[],
            )
        if not profiles:
            for role_name, canonical_name, role, aliases in self._extract_role_mentions(prompt):
                cid = f"char_{slugify_token(role_name)}"
                profiles[cid] = CharacterProfile(
                    character_id=cid,
                    canonical_name=canonical_name,
                    aliases=[canonical_name, *aliases],
                    role=role,
                    traits={"source": "prompt_role_heuristic"},
                    constraints=[],
                )
        if not profiles:
            # Ensure downstream planner/precheck always has at least one valid
            # static character profile for prompts without explicit name tokens.
            profiles["char_protagonist"] = CharacterProfile(
                character_id="char_protagonist",
                canonical_name="Protagonist",
                aliases=["protagonist"],
                role="protagonist",
                traits={"source": "fallback_default"},
                constraints=[],
            )
        return profiles

    def _extract_character_mentions(self, prompt: str) -> List[str]:
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
        filtered: List[str] = []
        for name in names:
            if not self._looks_like_character_name(name):
                continue
            if name not in filtered:
                filtered.append(name)
        return filtered[:16]

    def _looks_like_character_name(self, name: str) -> bool:
        token = (name or "").strip()
        lower = token.lower()
        if not token or lower in self._NON_CHARACTER_TOKENS:
            return False
        if len(token) < 3 or len(token) > 32:
            return False
        return bool(re.fullmatch(r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?", token))

    def _extract_role_mentions(self, prompt: str) -> List[tuple[str, str, str, List[str]]]:
        lower = f" {prompt.lower()} "
        roles: List[tuple[str, str, str, List[str]]] = []
        seen_canonicals: set[str] = set()
        seen_roles: set[str] = set()
        for phrase, canonical_name, role, aliases in self._ROLE_HINTS:
            if phrase not in lower:
                continue
            if canonical_name in seen_canonicals or (role in seen_roles and role in {"friend", "roommate"}):
                continue
            roles.append((canonical_name, canonical_name, role, aliases))
            seen_canonicals.add(canonical_name)
            seen_roles.add(role)
            if len(roles) >= 4:
                break
        return roles

    def _detect_pov(self, prompt: str) -> str:
        lower = prompt.lower()
        if "first person" in lower or " i " in f" {lower} ":
            return "first_person"
        if "second person" in lower or " you " in f" {lower} ":
            return "second_person"
        if "third person" in lower:
            return "third_person"
        return "unspecified"

    def _detect_tense(self, prompt: str) -> str:
        lower = prompt.lower()
        if "past tense" in lower:
            return "past"
        if "present tense" in lower:
            return "present"
        return "unspecified"

    def _detect_tone(self, prompt: str) -> str:
        lower = prompt.lower()
        if "humor" in lower or "funny" in lower or "comedic" in lower:
            return "humorous"
        if "dark" in lower or "grim" in lower:
            return "dark"
        if "warm" in lower or "heart" in lower:
            return "warm"
        return "unspecified"

    def _detect_register(self, prompt: str) -> str:
        lower = prompt.lower()
        if "formal" in lower:
            return "formal"
        if "informal" in lower or "casual" in lower:
            return "informal"
        return "neutral"

    def _extract_initial_relations(
        self,
        prompt: str,
        character_profiles: Dict[str, CharacterProfile],
    ) -> Dict[str, Dict[str, str]]:
        relation_map: Dict[str, Dict[str, str]] = {}
        names = [p.canonical_name for p in character_profiles.values()]
        if len(names) >= 2 and " and " in prompt.lower():
            first = f"char_{slugify_token(names[0])}"
            second = f"char_{slugify_token(names[1])}"
            relation_map[first] = {second: "co_present"}
            relation_map[second] = {first: "co_present"}
        return relation_map

    def _extract_known_abilities(
        self,
        prompt: str,
        character_profiles: Dict[str, CharacterProfile],
    ) -> Dict[str, List[str]]:
        ability_map: Dict[str, List[str]] = {}
        lower = prompt.lower()
        for char_id, profile in character_profiles.items():
            abilities: List[str] = []
            if "catch" in lower and profile.canonical_name.lower() in lower:
                abilities.append("catch")
            if abilities:
                ability_map[char_id] = abilities
        return ability_map

    def _build_name_map(self, character_profiles: Dict[str, CharacterProfile]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for profile in character_profiles.values():
            mapping[profile.canonical_name.lower()] = profile.canonical_name
            for alias in profile.aliases:
                mapping[alias.lower()] = profile.canonical_name
        return mapping

    def _extract_numeric_facts(self, prompt: str) -> Dict[str, float]:
        numeric_facts: Dict[str, float] = {}
        matches = re.findall(r"\b\d[\d,]*\b", prompt)
        for idx, token in enumerate(matches):
            value = float(token.replace(",", ""))
            numeric_facts[f"num_{idx}"] = value
        return numeric_facts

    def _extract_object_facts(self, prompt: str) -> Dict[str, str]:
        object_facts: Dict[str, str] = {}
        lower = prompt.lower()
        for obj in ["mouse", "dog", "cat", "door", "house", "clue", "letter"]:
            if obj in lower:
                object_facts[obj] = "mentioned_in_prompt"
        return object_facts

    def _extract_setting_description(self, prompt: str) -> str:
        snippets = extract_sentences_with_keywords(
            prompt,
            ["at", "in", "home", "city", "world", "setting"],
        )
        return snippets[0] if snippets else ""

    def _extract_initial_plot_setup(self, prompt: str) -> str:
        snippets = extract_sentences_with_keywords(prompt, ["start with", "begin", "opening"])
        if snippets:
            return snippets[0]
        return prompt[:200]

    def _extract_core_conflicts(self, prompt: str) -> List[str]:
        return extract_sentences_with_keywords(
            prompt,
            ["conflict", "rivalry", "versus", "against", "obstacle", "challenge"],
        )
