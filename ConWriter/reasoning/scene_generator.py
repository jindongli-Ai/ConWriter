"""Scene-level generator conditioned on static and dynamic memory."""

from __future__ import annotations

import json
import logging
import os
import re
import math
from typing import Callable, Dict, List
from urllib import error, request

from ConWriter.config.schema import LLMConfig
from ConWriter.reasoning.llm_uncertainty import (
    LLMGenerationResult,
    inject_logprob_request,
    parse_generation_response,
)
from ConWriter.pipeline.weighted_constraints import build_weighted_tiered_constraints
from ConWriter.utils.common import short_text
from ConWriter.utils.types import DynamicMemory, SceneDraft, ScenePlan, StaticMemory, StoryChunk


class SceneGenerator:
    """Generate one scene using plan + memory-conditioned prompts."""

    _ABSOLUTE_TIME_MARKERS = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )

    def __init__(self, llm_config: LLMConfig, logger: logging.Logger | None = None):
        self.llm_config = llm_config
        self.logger = logger or logging.getLogger("ConWriter.scene_generator")

    def generate_scene(
        self,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        recent_chunks: List[StoryChunk],
        prepared_constraints: Dict[str, object] | None = None,
        attempt: int = 0,
        trace_prompt_callback: Callable[[str], None] | None = None,
    ) -> SceneDraft:
        """Return one scene draft text."""
        messages, user_prompt = self._build_messages(
            scene_plan,
            static_memory,
            dynamic_memory,
            recent_chunks,
            prepared_constraints=prepared_constraints,
        )
        if trace_prompt_callback is not None:
            trace_prompt_callback(user_prompt)

        if self.llm_config.enabled:
            generation = self._call_llm(messages=messages)
            text = generation.text
            if text.strip():
                reviewed_text = self._self_review_scene(
                    scene_plan=scene_plan,
                    static_memory=static_memory,
                    dynamic_memory=dynamic_memory,
                    recent_chunks=recent_chunks,
                    scene_text=text.strip(),
                )
                final_text = reviewed_text.strip() or text.strip()
                return SceneDraft(
                    scene_id=scene_plan.scene_id,
                    chapter_id=scene_plan.chapter_id,
                    text=final_text,
                    attempt=attempt,
                    metadata={
                        "source": "llm",
                        "prompt": user_prompt,
                        "self_review_applied": bool(final_text != text.strip()),
                        "uncertainty_source": generation.uncertainty_source,
                        "uncertainty_available": generation.uncertainty_available,
                        "uncertainty_truncated": generation.uncertainty_truncated,
                        "llm_supports_logprobs": generation.supports_logprobs,
                    },
                    generated_text=final_text,
                    tokens=list(generation.tokens),
                    token_logprobs=list(generation.token_logprobs),
                    top_logprobs=list(generation.top_logprobs),
                    uncertainty_source=generation.uncertainty_source,
                    uncertainty_available=bool(generation.uncertainty_available),
                )

        return SceneDraft(
            scene_id=scene_plan.scene_id,
            chapter_id=scene_plan.chapter_id,
            text=self._stub_scene(scene_plan, static_memory, dynamic_memory, recent_chunks),
            attempt=attempt,
            metadata={"source": "stub", "prompt": user_prompt},
            uncertainty_source="none",
            uncertainty_available=False,
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> LLMGenerationResult:
        api_key = self.llm_config.api_key.strip() or os.getenv(self.llm_config.api_key_env, "").strip()
        if not api_key:
            self.logger.warning("Missing API key env=%s, fallback to stub scene.", self.llm_config.api_key_env)
            return LLMGenerationResult(text="")

        payload = {
            "model": self.llm_config.model,
            "messages": messages,
            "temperature": float(self.llm_config.request_temperature or 0.7),
            "max_tokens": int(self.llm_config.request_max_tokens or 900),
        }
        inject_logprob_request(payload, self.llm_config)
        if self.llm_config.extra_request_body:
            payload.update(self.llm_config.extra_request_body)
        model_id = str(self.llm_config.model or "").strip().lower()
        if "max_completion_tokens" in payload:
            payload.pop("max_tokens", None)
        elif model_id.startswith("gpt-5"):
            payload["max_completion_tokens"] = payload.pop("max_tokens", int(self.llm_config.request_max_tokens or 900))

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for k, v in self.llm_config.extra_headers.items():
            headers[str(k)] = str(v)

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
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            self.logger.warning("Scene generation HTTP error: %s", detail)
            return LLMGenerationResult(text="")
        except error.URLError as exc:
            self.logger.warning("Scene generation URL error: %s", exc)
            return LLMGenerationResult(text="")

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return LLMGenerationResult(text="")
        return parse_generation_response(parsed)

    def _self_review_scene(
        self,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        recent_chunks: List[StoryChunk],
        scene_text: str,
    ) -> str:
        """Ask the model for a minimal surface-consistency pass on its own scene."""
        if not scene_text.strip() or not self.llm_config.enabled:
            return scene_text

        canonical_lines: List[str] = []
        for cid in scene_plan.involved_characters:
            profile = static_memory.characterization.character_profiles.get(cid)
            dyn = dynamic_memory.characterization.entity_store.get(cid)
            names = []
            if profile is not None:
                names.extend([profile.canonical_name, *profile.aliases])
            if dyn is not None and dyn.name:
                names.append(dyn.name)
            names = [str(name).strip() for name in names if str(name).strip()]
            if names:
                canonical_lines.append(f"- {cid}: {sorted(set(names))}")
        if not canonical_lines:
            canonical_lines = ["- (no fixed canonical names; preserve names already used in this scene)"]

        recent_story = "\n".join(chunk.text for chunk in recent_chunks[-2:]) or "[none]"
        scene_pov_owner = self._scene_pov_owner(scene_plan, static_memory)
        review_prompt = (
            "You are the ConWriter scene self-checker. Perform a minimal consistency edit only.\n\n"
            "Do not add plot events, new characters, explanations, summaries, or analysis. "
            "Do not shorten the scene. Return only the revised scene prose.\n\n"
            f"Original prompt:\n{static_memory.story_spec.raw_prompt}\n\n"
            f"Current scene id: {scene_plan.scene_id}\n"
            f"Scene objective: {scene_plan.objective}\n"
            f"Scene POV owner: {scene_pov_owner}\n"
            f"Planned/involved character references:\n{chr(10).join(canonical_lines)}\n\n"
            f"Recent story context:\n{recent_story}\n\n"
            "Self-check checklist before returning:\n"
            "- Nomenclature: if the same character appears with near-spelling variants inside this scene, choose one spelling and make it consistent.\n"
            f"- POV: keep this entire scene in {scene_pov_owner}'s viewpoint; remove access to other characters' private thoughts inside the same section.\n"
            "- Timeline: do not invent exact months, weekdays, or dates unless already established.\n"
            "- Local time facts: do not state the same obligation or event as both future and past in the same scene; choose one coherent status.\n"
            "- Plot closure: if this scene introduces someone saying they need to talk, have proof, or know a secret, resolve that promise in the same scene.\n"
            "- Preserve all substantive events and wording unless needed for the above fixes.\n\n"
            f"Scene draft to minimally revise:\n{scene_text}"
        )

        reviewed = self._call_llm(
            messages=[
                {"role": "system", "content": "You are a precise narrative consistency self-reviewer."},
                {"role": "user", "content": review_prompt},
            ]
        ).text.strip()
        if not reviewed:
            return scene_text
        lowered = reviewed.lower()
        if "self-check checklist" in lowered or "analysis" in lowered[:200]:
            return scene_text
        original_words = self._word_count(scene_text)
        reviewed_words = self._word_count(reviewed)
        if original_words >= 200 and reviewed_words < int(original_words * 0.92):
            return scene_text
        return reviewed

    def _build_messages(
        self,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        recent_chunks: List[StoryChunk],
        prepared_constraints: Dict[str, object] | None = None,
    ) -> tuple[List[Dict[str, str]], str]:
        profiles = static_memory.characterization.character_profiles
        character_lines: List[str] = []
        required_characters = scene_plan.required_characters or scene_plan.involved_characters
        optional_characters = scene_plan.optional_characters

        for cid in required_characters + optional_characters:
            profile = profiles.get(cid)
            if profile is None:
                continue
            dyn = dynamic_memory.characterization.entity_store.get(cid)
            state = dyn.status if dyn else "unknown"
            loc = dyn.location if dyn else "unknown"
            character_lines.append(
                f"- {cid}: name={profile.canonical_name}, role={profile.role}, state={state}, location={loc}"
            )
        if not character_lines:
            character_lines = ["- (no explicit character constraints)"]

        static_relation_lines: List[str] = []
        for src, rels in static_memory.characterization.initial_relations.items():
            for tgt, rel in rels.items():
                static_relation_lines.append(f"- {src} -> {tgt}: {rel}")
        if not static_relation_lines:
            static_relation_lines = ["- (no static relation constraints)"]

        relation_lines: List[str] = []
        for src, rels in dynamic_memory.characterization.relations.items():
            if src not in scene_plan.involved_characters:
                continue
            for tgt, rel in rels.items():
                relation_lines.append(f"- {src} -> {tgt}: {rel}")
        if not relation_lines:
            relation_lines = ["- (no explicit relation constraints)"]

        recent_event_lines = [
            f"- {ev.event_id}: {short_text(ev.description, 140)}"
            for ev in dynamic_memory.timeline_plot.event_timeline[-5:]
        ] or ["- (no prior events)"]

        full_story_so_far = "\n\n".join(chunk.text for chunk in recent_chunks).strip()
        actual_story_words_so_far = self._word_count(full_story_so_far)
        reported_story_words_so_far = self._inflated_word_report(actual_story_words_so_far)
        recent_story = "\n".join(chunk.text for chunk in recent_chunks[-2:]) or "[none]"
        scene_pov_owner = self._scene_pov_owner(scene_plan, static_memory)
        model_id = self.llm_config.model.lower()
        is_gpt5_mini = "gpt-5-mini" in model_id
        scene_target_words = max(400, int(scene_plan.target_words))
        scene_request_factor = 2.0
        if is_gpt5_mini:
            # GPT-5-mini tends to under-generate in constrained scene writing.
            # Use a stronger per-scene request so fewer runs depend on fragile extension.
            scene_request_factor = 2.6
        scene_requested_words = int(scene_target_words * scene_request_factor)
        scene_requested_words = min(scene_requested_words, 2200)
        max_tokens = int(self.llm_config.request_max_tokens or 0) if self.llm_config.request_max_tokens is not None else 0
        if max_tokens > 0:
            max_feasible_words = max(350, int(math.floor(float(max_tokens) * 0.72)))
            scene_requested_words = min(scene_requested_words, max_feasible_words)
        scene_requested_words = max(scene_target_words, scene_requested_words)
        scene_stretch_words = max(scene_requested_words, int(math.ceil(float(scene_requested_words) * 1.12)))
        scene_min_words = max(320, int(math.floor(scene_requested_words * 0.88)))
        scene_paragraphs = max(6, int(math.ceil(scene_requested_words / 120)))
        if is_gpt5_mini:
            scene_stretch_words = max(scene_stretch_words, int(math.ceil(float(scene_requested_words) * 1.20)))
            scene_min_words = max(scene_min_words, int(math.floor(scene_requested_words * 0.92)))
            scene_paragraphs = max(scene_paragraphs, 8)
        mini_length_reminder = ""
        if is_gpt5_mini:
            mini_length_reminder = (
                "- GPT-5-mini reminder: this story must ultimately reach at least 3000 words, "
                "so each scene should be expansive rather than compressed.\n"
            )
        length_contract_lines = [
            f"- scene_target_words: {scene_target_words}",
            f"- scene_requested_words_for_this_call: {scene_requested_words} (intentional over-request)",
            f"- scene_minimum_words: {scene_min_words}",
            f"- scene_stretch_words: {scene_stretch_words}",
            f"- write_at_least_paragraphs: {scene_paragraphs}",
            (
                "- reported_story_words_so_far_for_pacing: "
                f"{reported_story_words_so_far} (high-side estimate, can be above literal count)"
            ),
            "- This scene should be a complete dramatic unit, not a compressed bridge.",
            "- Use concrete action, dialogue, interior reaction, and setting detail to fill the planned length.",
            "- Do not end early with summary language such as in the end, finally, or from that day on unless this is the final resolution scene.",
            "- If you are under the scene minimum, continue the same scene with more grounded beats rather than closing it.",
        ]
        if is_gpt5_mini:
            length_contract_lines.extend(
                [
                    "- GPT-5-mini reminder: write generously and avoid short bridge scenes.",
                    f"- HARD LENGTH RULE: return at least {scene_min_words} words in this single response.",
                    "- Do not stop at a natural break if under the minimum; continue with concrete in-scene beats.",
                    "- Do not output short placeholders, terse summaries, or compressed bridges.",
                ]
            )
        pov_lock_lines = [
            f"- scene_pov_owner: {scene_pov_owner}",
            "- Treat this scene as one POV section; write the entire scene from this POV owner's viewpoint.",
            f"- Start the scene with a heading that includes this POV owner, for example: # {scene_plan.title} - {scene_pov_owner} POV",
            "- Do not enter another character's private thoughts, perceptions, or hallucinations inside the same unlabeled POV section.",
            "- Other characters may appear through dialogue, visible actions, sounds, texts, and what the POV owner can plausibly observe.",
            "- If the prompt requires alternating POVs, alternate by scene; do not switch viewpoint inside this scene.",
        ]
        prompt_time_markers = self._absolute_time_markers(static_memory.story_spec.raw_prompt)
        recent_time_markers = self._absolute_time_markers(recent_story)
        time_marker_lines = [
            f"- prompt_absolute_time_markers: {prompt_time_markers or ['(none)']}",
            f"- recent_absolute_time_markers: {recent_time_markers or ['(none)']}",
            "- If the prompt has no exact month, weekday, date, or calendar anchor, do not invent one.",
            "- Prefer relative time cues such as later that night, the next morning, after rent was due, or by finals week.",
            "- If an exact time marker is already established, keep chronology monotonic; do not jump to an earlier month/day unless it is explicitly labeled as a flashback.",
        ]
        closure_lines = [
            "- Do not introduce a new named side character only to deliver a secret, warning, or revelation unless the same scene resolves that information.",
            "- If someone says they need to talk, knows something, has proof, or has a secret, the scene must reveal or close that thread before ending.",
            "- Late scenes should resolve existing roommate/friend/house fallout rather than opening new external subplots.",
        ]
        static_goal = static_memory.story_spec.goal
        world_state = dynamic_memory.world_setting.current_setting_state or "unknown"
        world_rules = static_memory.world_setting.world_invariants[:6]
        world_rule_lines = [f"- {rule}" for rule in world_rules] or ["- (no explicit world invariant)"]
        timeline_anchors = static_memory.timeline_plot.required_plot_points[:6]
        timeline_anchor_lines = [f"- {item}" for item in timeline_anchors] or ["- (no explicit timeline anchors)"]
        plot_commitments = static_memory.timeline_plot.core_conflicts[:6]
        plot_commitment_lines = [f"- {item}" for item in plot_commitments] or ["- (no explicit plot commitments)"]
        style_constraints = static_memory.style.style_constraints[:5]
        forbidden_style = static_memory.style.forbidden_style_shifts[:4]
        style_lines = [f"- {rule}" for rule in style_constraints] or ["- (no explicit style constraints)"]
        forbidden_style_lines = [f"- {rule}" for rule in forbidden_style] or ["- (no forbidden style shift)"]
        dynamic_style_lines = [
            f"- pov={dynamic_memory.narrative_style.current_pov}",
            f"- tense={dynamic_memory.narrative_style.current_tense}",
        ]
        if dynamic_memory.narrative_style.tone_trace:
            dynamic_style_lines.append(
                f"- recent_tones={dynamic_memory.narrative_style.tone_trace[-3:]}"
            )
        if dynamic_memory.narrative_style.recent_style_notes:
            dynamic_style_lines.append(
                f"- recent_style_notes={dynamic_memory.narrative_style.recent_style_notes[-2:]}"
            )

        static_plot_points = static_memory.timeline_plot.required_plot_points[:5]
        static_plot_lines = [f"- {point}" for point in static_plot_points] or ["- (no explicit plot anchors)"]
        active_goals = dynamic_memory.timeline_plot.active_goals[-3:] or ["(no active goals)"]
        active_goal_lines = [f"- {goal}" for goal in active_goals]
        active_threads = dynamic_memory.timeline_plot.active_plot_threads[-6:] or ["(no active plot threads)"]
        active_thread_lines = [f"- {thread}" for thread in active_threads]
        pending_constraints = (
            dynamic_memory.timeline_plot.pending_constraints[-8:]
            or scene_plan.required_constraints
            or ["(no pending constraints)"]
        )
        pending_constraint_lines = [f"- {value}" for value in pending_constraints]
        foreshadowing = dynamic_memory.timeline_plot.unresolved_foreshadowing[-6:] or ["(none)"]
        foreshadow_lines = [f"- {item}" for item in foreshadowing]

        prepared_required = []
        prepared_forbidden = []
        prepared_keep = []
        prepared_allowed_transitions: List[str] = []
        prepared_forbidden_transitions: List[str] = []
        prepared_required_state_changes: List[str] = []
        prepared_forbidden_state_changes: List[str] = []
        prepared_inferred_constraints: List[str] = []
        prepared_state_summary: Dict[str, object] = {}
        prepared_candidate_operators: List[Dict[str, object]] = []
        prepared_forbidden_operators: List[str] = []
        prepared_selected_operator: Dict[str, object] = {}
        prepared_execution_spec: Dict[str, object] = {}
        prepared_propagated_constraints: List[str] = []
        prepared_conflict_candidates: List[str] = []
        prepared_hard_constraints: List[str] = []
        prepared_soft_constraints: List[str] = []
        prepared_high_conf_violations: List[str] = []
        prepared_deferred_constraints: List[str] = []
        prepared_entropy_critical_constraints: List[str] = []
        prepared_memory_binding_mode = "normal_binding"
        prepared_generation_control_mode = "normal_generation"
        prepared_binding_decision_reasons: List[str] = []
        prepared_strengthened_memory_blocks: List[str] = []
        prepared_strengthened_constraints: List[str] = []
        prepared_required_state_reminders: List[str] = []
        prepared_forbidden_state_reminders: List[str] = []
        prepared_canonical_state_phrasing: List[str] = []
        prepared_risk_control_guidance: List[str] = []
        prepared_static_memory_reinforcement: List[str] = []
        prepared_dynamic_memory_reinforcement: List[str] = []
        prepared_critical_constraints_frontloaded = False
        prepared_generation_control_context: Dict[str, object] = {}
        prepared_model_experience_cautions: List[str] = []
        prepared_length_control_guidance: List[str] = []
        prepared_length_expansion_guidance: List[str] = []
        prepared_desired_target_words = 0
        prepared_requested_target_words = 0
        prepared_length_compensation_factor = 1.0
        prepared_length_progress_ratio = 0.0
        if isinstance(prepared_constraints, dict):
            prepared_required = [str(v) for v in prepared_constraints.get("required", [])]
            prepared_forbidden = [str(v) for v in prepared_constraints.get("forbidden", [])]
            prepared_keep = [str(v) for v in prepared_constraints.get("must_keep", [])]
            prepared_allowed_transitions = [
                str(v) for v in prepared_constraints.get("allowed_transitions", [])
            ]
            prepared_forbidden_transitions = [
                str(v) for v in prepared_constraints.get("forbidden_transitions", [])
            ]
            prepared_required_state_changes = [
                str(v) for v in prepared_constraints.get("required_state_changes", [])
            ]
            prepared_forbidden_state_changes = [
                str(v) for v in prepared_constraints.get("forbidden_state_changes", [])
            ]
            prepared_inferred_constraints = [
                str(v) for v in prepared_constraints.get("inferred_constraints", [])
            ]
            if isinstance(prepared_constraints.get("state_summary"), dict):
                prepared_state_summary = dict(prepared_constraints.get("state_summary", {}))
            candidates = prepared_constraints.get("candidate_operators", [])
            if isinstance(candidates, list):
                prepared_candidate_operators = [
                    item for item in candidates if isinstance(item, dict)
                ]
            prepared_forbidden_operators = [
                str(v) for v in prepared_constraints.get("forbidden_operators", [])
            ]
            if isinstance(prepared_constraints.get("selected_operator"), dict):
                prepared_selected_operator = dict(prepared_constraints.get("selected_operator", {}))
            if isinstance(prepared_constraints.get("execution_spec"), dict):
                prepared_execution_spec = dict(prepared_constraints.get("execution_spec", {}))
            prepared_propagated_constraints = [
                str(v) for v in prepared_constraints.get("propagated_constraints", [])
            ]
            prepared_conflict_candidates = [
                str(v) for v in prepared_constraints.get("conflict_candidates", [])
            ]
            prepared_hard_constraints = [
                str(v) for v in prepared_constraints.get("hard_constraints", [])
            ]
            prepared_soft_constraints = [
                str(v) for v in prepared_constraints.get("soft_constraints", [])
            ]
            prepared_high_conf_violations = [
                str(v) for v in prepared_constraints.get("high_confidence_violations", [])
            ]
            prepared_deferred_constraints = [
                str(v) for v in prepared_constraints.get("deferred_constraints", [])
            ]
            prepared_entropy_critical_constraints = [
                str(v) for v in prepared_constraints.get("entropy_critical_constraints", [])
            ]
            prepared_memory_binding_mode = str(prepared_constraints.get("memory_binding_mode", "normal_binding"))
            prepared_generation_control_mode = str(
                prepared_constraints.get("generation_control_mode", "normal_generation")
            )
            prepared_binding_decision_reasons = [
                str(v) for v in prepared_constraints.get("memory_binding_decision_reasons", [])
            ]
            prepared_strengthened_memory_blocks = [
                str(v) for v in prepared_constraints.get("strengthened_memory_blocks", [])
            ]
            prepared_strengthened_constraints = [
                str(v) for v in prepared_constraints.get("strengthened_constraints", [])
            ]
            prepared_required_state_reminders = [
                str(v) for v in prepared_constraints.get("required_state_reminders", [])
            ]
            prepared_forbidden_state_reminders = [
                str(v) for v in prepared_constraints.get("forbidden_state_reminders", [])
            ]
            prepared_canonical_state_phrasing = [
                str(v) for v in prepared_constraints.get("canonical_state_phrasing", [])
            ]
            prepared_risk_control_guidance = [
                str(v) for v in prepared_constraints.get("risk_control_guidance", [])
            ]
            prepared_static_memory_reinforcement = [
                str(v) for v in prepared_constraints.get("static_memory_reinforcement", [])
            ]
            prepared_dynamic_memory_reinforcement = [
                str(v) for v in prepared_constraints.get("dynamic_memory_reinforcement", [])
            ]
            prepared_critical_constraints_frontloaded = bool(
                prepared_constraints.get("critical_constraints_frontloaded", False)
            )
            if isinstance(prepared_constraints.get("generation_control_context"), dict):
                prepared_generation_control_context = dict(prepared_constraints.get("generation_control_context", {}))
            prepared_model_experience_cautions = [
                str(v) for v in prepared_constraints.get("model_experience_cautions", [])
            ]
            prepared_length_control_guidance = [
                str(v) for v in prepared_constraints.get("length_control_guidance", [])
            ]
            prepared_length_expansion_guidance = [
                str(v) for v in prepared_constraints.get("length_expansion_guidance", [])
            ]
            prepared_desired_target_words = int(prepared_constraints.get("length_desired_target_words", 0) or 0)
            prepared_requested_target_words = int(prepared_constraints.get("length_requested_target_words", 0) or 0)
            prepared_length_compensation_factor = float(
                prepared_constraints.get("length_compensation_factor", 1.0) or 1.0
            )
            prepared_length_progress_ratio = float(prepared_constraints.get("length_progress_ratio", 0.0) or 0.0)
        generation_mode = str(prepared_constraints.get("generation_mode", "plan_memory_patch")) if isinstance(
            prepared_constraints, dict
        ) else "plan_memory_patch"

        selected_operator_type = str(prepared_selected_operator.get("operator_type", "")).strip() or "UNSPECIFIED"
        selected_operator_preconditions = prepared_selected_operator.get("preconditions", [])
        if not isinstance(selected_operator_preconditions, list):
            selected_operator_preconditions = []
        selected_operator_postconditions = prepared_selected_operator.get("postconditions", [])
        if not isinstance(selected_operator_postconditions, list):
            selected_operator_postconditions = []
        selected_operator_required_effects = prepared_selected_operator.get("required_effects", [])
        if not isinstance(selected_operator_required_effects, list):
            selected_operator_required_effects = []
        spec_required_entities = prepared_execution_spec.get("required_entities", [])
        if not isinstance(spec_required_entities, list):
            spec_required_entities = []
        spec_required_events = prepared_execution_spec.get("required_events", [])
        if not isinstance(spec_required_events, list):
            spec_required_events = []
        spec_required_state_changes = prepared_execution_spec.get("required_state_changes", [])
        if not isinstance(spec_required_state_changes, list):
            spec_required_state_changes = []
        spec_forbidden_patterns = prepared_execution_spec.get("forbidden_patterns", [])
        if not isinstance(spec_forbidden_patterns, list):
            spec_forbidden_patterns = []
        if not prepared_hard_constraints:
            prepared_hard_constraints = (
                list(prepared_required[:6])
                + [str(v) for v in spec_required_entities[:4]]
                + [str(v) for v in spec_required_events[:4]]
                + [str(v) for v in spec_required_state_changes[:4]]
            )
            prepared_hard_constraints = [v for v in prepared_hard_constraints if str(v).strip()]
        if not prepared_soft_constraints:
            prepared_soft_constraints = (
                list(prepared_inferred_constraints[:6])
                + list(prepared_propagated_constraints[:4])
                + [str(v) for v in scene_plan.expected_state_changes[:4]]
            )
            prepared_soft_constraints = [v for v in prepared_soft_constraints if str(v).strip()]
        if not prepared_forbidden_state_changes:
            prepared_forbidden_state_changes = [str(v) for v in scene_plan.forbidden_state_changes]
        weighted_items = build_weighted_tiered_constraints(
            required=prepared_required,
            must_keep=prepared_keep,
            forbidden=prepared_forbidden,
            inferred=prepared_inferred_constraints,
            propagated=prepared_propagated_constraints,
            high_conf_violations=prepared_high_conf_violations,
            deferred_constraints=prepared_deferred_constraints,
        )
        if generation_mode == "plan_only":
            prepared_required = list(scene_plan.required_constraints)
            prepared_forbidden = list(scene_plan.forbidden_constraints)
            prepared_keep = list(scene_plan.must_keep_constraints)
            prepared_inferred_constraints = []
            prepared_propagated_constraints = []
            prepared_hard_constraints = list(scene_plan.required_constraints[:8])
            prepared_soft_constraints = list(scene_plan.must_keep_constraints[:8])
            weighted_items = []
        elif generation_mode == "plain_generate_only":
            prepared_required = []
            prepared_forbidden = []
            prepared_keep = []
            prepared_inferred_constraints = []
            prepared_propagated_constraints = []
            prepared_hard_constraints = []
            prepared_soft_constraints = []
            prepared_high_conf_violations = []
            prepared_deferred_constraints = []
            weighted_items = []

        tier_1 = [item for item in weighted_items if item.tier == 1]
        tier_2 = [item for item in weighted_items if item.tier == 2]
        tier_3 = [item for item in weighted_items if item.tier >= 3]
        if tier_1:
            prepared_hard_constraints = [f"[w={item.weight:.2f}] {item.text}" for item in tier_1[:14]]
        if prepared_entropy_critical_constraints:
            merged = list(prepared_entropy_critical_constraints[:6]) + list(prepared_hard_constraints)
            prepared_hard_constraints = list(dict.fromkeys([x for x in merged if str(x).strip()]))[:16]
        if tier_2:
            prepared_soft_constraints = [f"[w={item.weight:.2f}] {item.text}" for item in tier_2[:14]]
        tier_3_lines = [f"[w={item.weight:.2f}] {item.text}" for item in tier_3[:14]]
        candidate_operator_lines = [
            (
                f"- {str(item.get('operator_type', 'UNKNOWN'))}: "
                f"pre={item.get('preconditions', [])} "
                f"post={item.get('postconditions', [])}"
            )
            for item in prepared_candidate_operators
        ]
        if not isinstance(prepared_generation_control_context, dict):
            prepared_generation_control_context = {}
        if not prepared_generation_control_context:
            prepared_generation_control_context = {
                "memory_binding_mode": prepared_memory_binding_mode,
                "generation_control_mode": prepared_generation_control_mode,
                "decision_reasons": list(prepared_binding_decision_reasons),
                "strengthened_memory_blocks": list(prepared_strengthened_memory_blocks),
                "strengthened_constraints": list(prepared_strengthened_constraints),
                "required_state_reminders": list(prepared_required_state_reminders),
                "forbidden_state_reminders": list(prepared_forbidden_state_reminders),
                "canonical_state_phrasing": list(prepared_canonical_state_phrasing),
                "control_guidance": list(prepared_risk_control_guidance),
                "critical_constraints_frontloaded": bool(prepared_critical_constraints_frontloaded),
                "static_memory_reinforcement": list(prepared_static_memory_reinforcement),
                "dynamic_memory_reinforcement": list(prepared_dynamic_memory_reinforcement),
            }
        if "reasoning_constraints" not in prepared_generation_control_context:
            prepared_generation_control_context["reasoning_constraints"] = {
                "required_constraints": list(prepared_required),
                "must_keep_constraints": list(prepared_keep),
                "forbidden_constraints": list(prepared_forbidden),
                "hard_constraints": list(prepared_hard_constraints),
                "soft_constraints": list(prepared_soft_constraints),
                "required_state_changes": list(prepared_required_state_changes),
                "forbidden_state_changes": list(prepared_forbidden_state_changes),
                "allowed_transitions": list(prepared_allowed_transitions),
                "forbidden_transitions": list(prepared_forbidden_transitions),
                "execution_spec_required_entities": list(spec_required_entities),
                "execution_spec_required_events": list(spec_required_events),
                "execution_spec_required_state_changes": list(spec_required_state_changes),
                "execution_spec_forbidden_patterns": list(spec_forbidden_patterns),
                "operator_required_effects": list(selected_operator_required_effects),
                "operator_postconditions": list(selected_operator_postconditions),
            }
        if "uncertainty_guided_control" not in prepared_generation_control_context:
            prepared_generation_control_context["uncertainty_guided_control"] = {
                "memory_binding_mode": prepared_memory_binding_mode,
                "generation_control_mode": prepared_generation_control_mode,
                "decision_reasons": list(prepared_binding_decision_reasons),
                "strengthened_memory_blocks": list(prepared_strengthened_memory_blocks),
                "strengthened_constraints": list(prepared_strengthened_constraints),
                "required_state_reminders": list(prepared_required_state_reminders),
                "forbidden_state_reminders": list(prepared_forbidden_state_reminders),
                "canonical_state_phrasing": list(prepared_canonical_state_phrasing),
                "control_guidance": list(prepared_risk_control_guidance),
                "critical_constraints_frontloaded": bool(prepared_critical_constraints_frontloaded),
            }
        if "dual_memory" not in prepared_generation_control_context:
            prepared_generation_control_context["dual_memory"] = {
                "static_memory_blocks_used": list(prepared_static_memory_reinforcement),
                "dynamic_memory_blocks_used": list(prepared_dynamic_memory_reinforcement),
                "static_memory_blocks_strengthened": list(prepared_static_memory_reinforcement),
                "dynamic_memory_blocks_strengthened": list(prepared_dynamic_memory_reinforcement),
            }

        user = (
            f"Original Prompt:\n{static_memory.story_spec.raw_prompt}\n\n"
            f"Global Story Premise:\n{static_memory.story_spec.theme or static_memory.story_spec.raw_prompt}\n\n"
            f"Global Objective:\n{static_goal}\n\n"
            f"Current Scene Plan:\n"
            f"- scene_id: {scene_plan.scene_id}\n"
            f"- chapter_id: {scene_plan.chapter_id}\n"
            f"- title: {scene_plan.title}\n"
            f"- objective: {scene_plan.objective}\n"
            f"- key_events: {scene_plan.key_events}\n"
            f"- required_characters: {required_characters}\n"
            f"- optional_characters: {optional_characters}\n"
            f"- preconditions: {scene_plan.preconditions}\n"
            f"- expected_state_changes: {scene_plan.expected_state_changes}\n"
            f"- forbidden_state_changes: {scene_plan.forbidden_state_changes}\n"
            f"- dependency_scenes: {scene_plan.dependency_scenes}\n"
            f"- must_keep_constraints: {scene_plan.must_keep_constraints}\n"
            f"- target_words: {scene_plan.target_words}\n"
            f"- required_constraints: {scene_plan.required_constraints}\n"
            f"- forbidden_constraints: {scene_plan.forbidden_constraints}\n\n"
            f"Scene POV Lock:\n{chr(10).join(pov_lock_lines)}\n\n"
            f"Scene Length Contract:\n{chr(10).join(length_contract_lines)}\n\n"
            f"Character Profiles + Dynamic States:\n{chr(10).join(character_lines)}\n\n"
            f"Static Character Relations:\n{chr(10).join(static_relation_lines)}\n\n"
            f"Character Relations:\n{chr(10).join(relation_lines)}\n\n"
            f"World Rules (Static):\n{chr(10).join(world_rule_lines)}\n\n"
            f"Timeline Anchors (Static):\n{chr(10).join(timeline_anchor_lines)}\n\n"
            f"Timeline Continuity Guardrails:\n{chr(10).join(time_marker_lines)}\n\n"
            f"Plot Commitments (Static):\n{chr(10).join(plot_commitment_lines)}\n\n"
            f"Plot Closure Guardrails:\n{chr(10).join(closure_lines)}\n\n"
            f"Current World State (Dynamic):\n- {world_state}\n\n"
            f"Recent Events (Dynamic Timeline):\n{chr(10).join(recent_event_lines)}\n\n"
            f"Static Plot Anchors:\n{chr(10).join(static_plot_lines)}\n\n"
            f"Dynamic Active Goals:\n{chr(10).join(active_goal_lines)}\n\n"
            f"Dynamic Active Plot Threads:\n{chr(10).join(active_thread_lines)}\n\n"
            f"Dynamic Pending Constraints:\n{chr(10).join(pending_constraint_lines)}\n\n"
            f"Unresolved Foreshadowing:\n{chr(10).join(foreshadow_lines)}\n\n"
            f"Static Style Constraints:\n{chr(10).join(style_lines)}\n\n"
            f"Forbidden Style Shifts:\n{chr(10).join(forbidden_style_lines)}\n\n"
            f"Dynamic Style State:\n{chr(10).join(dynamic_style_lines)}\n\n"
            f"Prepared Constraints Before Generation:\n"
            f"- required: {prepared_required}\n"
            f"- forbidden: {prepared_forbidden}\n"
            f"- must_keep: {prepared_keep}\n"
            f"- inferred_constraints: {prepared_inferred_constraints}\n\n"
            f"Hybrid Constraints (Hard MUST satisfy):\n"
            f"{chr(10).join(f'- {v}' for v in prepared_hard_constraints) if prepared_hard_constraints else '- (none)'}\n\n"
            f"Tier-2 Weighted Constraints (Strong SHOULD satisfy):\n"
            f"{chr(10).join(f'- {v}' for v in prepared_soft_constraints) if prepared_soft_constraints else '- (none)'}\n\n"
            f"Tier-3 Weighted Constraints (Deferred / low priority):\n"
            f"{chr(10).join(f'- {v}' for v in tier_3_lines) if tier_3_lines else '- (none)'}\n\n"
            f"High-confidence Violations To Avoid:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_high_conf_violations) if prepared_high_conf_violations else '- (none)'}\n\n"
            f"Risk-Adaptive Memory Binding:\n"
            f"- memory_binding_mode: {prepared_memory_binding_mode}\n"
            f"- generation_control_mode: {prepared_generation_control_mode}\n"
            f"- decision_reasons: {prepared_binding_decision_reasons}\n"
            f"- strengthened_memory_blocks: {prepared_strengthened_memory_blocks}\n"
            f"- strengthened_constraints: {prepared_strengthened_constraints}\n"
            f"- critical_constraints_frontloaded: {prepared_critical_constraints_frontloaded}\n\n"
            f"Required State Reminders:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_required_state_reminders) if prepared_required_state_reminders else '- (none)'}\n\n"
            f"Forbidden State Reminders:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_forbidden_state_reminders) if prepared_forbidden_state_reminders else '- (none)'}\n\n"
            f"Canonical State Phrasing:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_canonical_state_phrasing) if prepared_canonical_state_phrasing else '- (none)'}\n\n"
            f"Static Memory Reinforcement:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_static_memory_reinforcement) if prepared_static_memory_reinforcement else '- (none)'}\n\n"
            f"Dynamic Memory Reinforcement:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_dynamic_memory_reinforcement) if prepared_dynamic_memory_reinforcement else '- (none)'}\n\n"
            f"Model-Specific Experience Cautions:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_model_experience_cautions) if prepared_model_experience_cautions else '- (none)'}\n\n"
            f"Length-Aware Control:\n"
            f"- desired_target_words: {prepared_desired_target_words}\n"
            f"- requested_target_words: {prepared_requested_target_words}\n"
            f"- length_compensation_factor: {prepared_length_compensation_factor:.3f}\n"
            f"- length_progress_ratio: {prepared_length_progress_ratio:.3f}\n"
            f"- base_guidance: {prepared_length_control_guidance}\n"
            f"- expansion_guidance: {prepared_length_expansion_guidance}\n\n"
            f"Generation Control Guidance:\n"
            f"{chr(10).join(f'- {v}' for v in prepared_risk_control_guidance) if prepared_risk_control_guidance else '- (none)'}\n\n"
            f"Unified Generation Control Context:\n"
            f"{json.dumps(prepared_generation_control_context, ensure_ascii=False)}\n\n"
            f"Current Structured State (State_t):\n"
            f"{json.dumps(prepared_state_summary, ensure_ascii=False)}\n\n"
            f"Allowed Transitions:\n{chr(10).join(f'- {v}' for v in prepared_allowed_transitions) if prepared_allowed_transitions else '- (none)'}\n\n"
            f"Forbidden Transitions:\n{chr(10).join(f'- {v}' for v in prepared_forbidden_transitions) if prepared_forbidden_transitions else '- (none)'}\n\n"
            f"Required State Changes:\n{chr(10).join(f'- {v}' for v in prepared_required_state_changes) if prepared_required_state_changes else '- (none)'}\n\n"
            f"Candidate Operators:\n{chr(10).join(candidate_operator_lines) if candidate_operator_lines else '- (none)'}\n\n"
            f"Forbidden Operators:\n{chr(10).join(f'- {v}' for v in prepared_forbidden_operators) if prepared_forbidden_operators else '- (none)'}\n\n"
            f"Propagated Constraints (Graph Inference):\n{chr(10).join(f'- {v}' for v in prepared_propagated_constraints) if prepared_propagated_constraints else '- (none)'}\n\n"
            f"Conflict Candidates (Graph Inference):\n{chr(10).join(f'- {v}' for v in prepared_conflict_candidates) if prepared_conflict_candidates else '- (none)'}\n\n"
            "[Selected Operator]\n"
            f"- Type: {selected_operator_type}\n"
            f"- Preconditions: {selected_operator_preconditions}\n"
            f"- Required Effects: {selected_operator_required_effects}\n"
            f"- Postconditions: {selected_operator_postconditions}\n\n"
            "[Operator Execution Spec]\n"
            f"- required_entities: {spec_required_entities}\n"
            f"- required_events: {spec_required_events}\n"
            f"- required_state_changes: {spec_required_state_changes}\n"
            f"- forbidden_patterns: {spec_forbidden_patterns}\n\n"
            f"Recent Story Context:\n{recent_story}\n\n"
            "Write ONLY the current scene prose. "
            f"{mini_length_reminder}"
            f"Write this scene as a substantial narrative scene of about {scene_requested_words}-{scene_stretch_words} words, prioritizing the upper half, "
            f"with at least {scene_min_words} words and at least {scene_paragraphs} developed paragraphs, "
            f"because the full story target is about {max(int(prepared_requested_target_words), int(prepared_desired_target_words), int(scene_plan.target_words))} words. "
            "Do not output a placeholder, outline, meta-analysis, or one-paragraph summary. "
            "Do not summarize. Do not restart the story. "
            "Respect required constraints and avoid forbidden constraints. "
            "Do not add unsupported exact dates/months/weekdays, and do not leave newly introduced revelations unresolved. "
            "Before ending, internally check scene length; if under target, continue this same scene with additional grounded beats instead of closing. "
            f"Keep this whole scene in {scene_pov_owner}'s POV; alternate POV by later scenes, not inside this scene. "
            "You MUST generate a valid transition from State_t. "
            "You MUST realize this operator in the scene. "
            "You MUST satisfy all hard constraints. "
            "You SHOULD satisfy soft constraints unless they conflict with hard constraints."
        )
        if prepared_generation_control_mode in {"constrained_generation", "strict_state_realization_generation"}:
            user = (
                f"{user} "
                "Preserve global invariants and avoid conflicting parallel states. "
                "If state updates are required, textualize required state changes explicitly. "
                "Keep timeline/operator post-state coherent."
            )
        if prepared_generation_control_mode == "strict_state_realization_generation":
            user = (
                f"{user} "
                "STRICT: explicitly realize required state changes and operator-required post-state. "
                "STRICT: explicitly remove forbidden/conflicting states and do not keep inconsistent pre/post-state together."
            )

        messages = [
            {"role": "system", "content": self.llm_config.system_prompt},
            {"role": "user", "content": user},
        ]
        return messages, user

    def _absolute_time_markers(self, text: str) -> List[str]:
        if not text:
            return []
        found: List[str] = []
        for marker in self._ABSOLUTE_TIME_MARKERS:
            if re.search(rf"\b{re.escape(marker)}\b", text):
                found.append(marker)
        if re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text):
            found.append("numeric_date")
        return found

    def _scene_pov_owner(self, scene_plan: ScenePlan, static_memory: StaticMemory) -> str:
        profiles = static_memory.characterization.character_profiles
        candidates = list(scene_plan.required_characters or scene_plan.involved_characters)
        if not candidates:
            candidates = list(profiles.keys())
        if candidates:
            cid = candidates[int(scene_plan.scene_index or 0) % len(candidates)]
            profile = profiles.get(cid)
            if profile is not None and profile.canonical_name:
                return str(profile.canonical_name)
        return "the current viewpoint character"

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    def _inflated_word_report(self, actual_words: int, factor: float = 2.0, bonus_words: int = 120) -> int:
        """Return a conservative high-side word-count estimate for pacing prompts."""
        actual = max(0, int(actual_words))
        if actual <= 0:
            return 0
        return max(int(actual + max(0, bonus_words)), int(math.ceil(actual * max(1.0, factor))))

    def _stub_scene(
        self,
        scene_plan: ScenePlan,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        recent_chunks: List[StoryChunk],
    ) -> str:
        lead = static_memory.story_spec.theme or "story"
        world_state = dynamic_memory.world_setting.current_setting_state or "unknown setting"
        prev = short_text(recent_chunks[-1].text, 120) if recent_chunks else "the story just begins"
        constraints = ", ".join(scene_plan.required_constraints[:2]) or "core consistency constraints"
        return (
            f"[{scene_plan.scene_id}] {scene_plan.title}. "
            f"{scene_plan.objective}. "
            f"In {world_state}, characters pursue {short_text(lead, 80)}. "
            f"Previous context: {prev}. "
            f"This scene must satisfy: {constraints}. "
            "A concrete event happens, relationships evolve, and the scene ends with a causal hook. "
            "The characters respond through observable actions, preserve established facts, and carry "
            "the consequences into the next scene without changing the setting or timeline abruptly."
        )
