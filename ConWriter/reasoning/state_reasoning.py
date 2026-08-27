"""State-level reasoning for transition-based incremental generation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Protocol

from ConWriter.reasoning.symbolic_state_graph import SymbolicStateGraph
from ConWriter.utils.types import (
    OperatorExecutionSpec,
    SceneExtraction,
    ScenePlan,
    StateReasoningResult,
    StoryState,
    TransitionAction,
    TransitionConstraint,
    TransitionOperator,
)


class _ReasoningRule(Protocol):
    """Protocol for one structured reasoning rule."""

    rule_id: str

    def infer(self, state: StoryState, scene_plan: ScenePlan) -> List[TransitionConstraint]:
        """Infer structured constraints from current state."""


class _RequiredCharacterRule:
    """Infer participation constraints from required scene characters."""

    rule_id = "required_character_rule"

    def infer(self, state: StoryState, scene_plan: ScenePlan) -> List[TransitionConstraint]:
        constraints: List[TransitionConstraint] = []
        required = scene_plan.required_characters or scene_plan.involved_characters
        for cid in required:
            char_state = state.character_states.get(cid)
            if char_state is None:
                constraints.append(
                    TransitionConstraint(
                        constraint_id=f"{self.rule_id}:{scene_plan.scene_id}:{cid}:introduce",
                        facet="characterization",
                        predicate="character_must_be_introduced_or_acted",
                        arguments={"character_id": cid},
                        expected="character appears as actor or explicit state target",
                        severity="error",
                        source=self.rule_id,
                    )
                )
                continue
            if char_state.status.lower() in {"dead", "removed"}:
                constraints.append(
                    TransitionConstraint(
                        constraint_id=f"{self.rule_id}:{scene_plan.scene_id}:{cid}:forbid_actor",
                        facet="characterization",
                        predicate="character_cannot_be_actor",
                        arguments={"character_id": cid, "status": char_state.status},
                        expected="dead/removed character should not drive active action",
                        severity="error",
                        source=self.rule_id,
                    )
                )
        return constraints


class _TimelineProgressRule:
    """Infer timeline constraints that enforce forward event progression."""

    rule_id = "timeline_progress_rule"

    def infer(self, state: StoryState, scene_plan: ScenePlan) -> List[TransitionConstraint]:
        expected_next = max(0, int(state.timeline.last_event_order) + 1)
        return [
            TransitionConstraint(
                constraint_id=f"{self.rule_id}:{scene_plan.scene_id}",
                facet="timeline_plot",
                predicate="min_next_event_order",
                arguments={"min_order": expected_next},
                expected=f"new event order >= {expected_next}",
                severity="error",
                source=self.rule_id,
            )
        ]


class _WorldContinuityRule:
    """Infer world continuity constraints from current world state."""

    rule_id = "world_continuity_rule"

    def infer(self, state: StoryState, scene_plan: ScenePlan) -> List[TransitionConstraint]:
        current_location = (state.world_state.current_setting_state or "").strip()
        if not current_location:
            return []
        return [
            TransitionConstraint(
                constraint_id=f"{self.rule_id}:{scene_plan.scene_id}",
                facet="world_setting",
                predicate="location_transition_requires_evidence",
                arguments={"from_location": current_location},
                expected="if location changes, action must contain explicit transition evidence",
                severity="warning",
                source=self.rule_id,
            )
        ]


class _DependencyRule:
    """Infer scene dependency constraints from scene plan metadata."""

    rule_id = "scene_dependency_rule"

    def infer(self, state: StoryState, scene_plan: ScenePlan) -> List[TransitionConstraint]:
        constraints: List[TransitionConstraint] = []
        if not scene_plan.dependency_scenes:
            return constraints
        completed = set(state.timeline.recent_event_ids)
        for dep in scene_plan.dependency_scenes:
            constraints.append(
                TransitionConstraint(
                    constraint_id=f"{self.rule_id}:{scene_plan.scene_id}:{dep}",
                    facet="timeline_plot",
                    predicate="dependency_scene_should_be_reflected",
                    arguments={"dependency_scene_id": dep, "known_events": sorted(completed)},
                    expected="dependent scene/event should have already appeared in timeline",
                    severity="warning",
                    source=self.rule_id,
                )
            )
        return constraints


class StateReasoner:
    """Structured reasoner that maps State_t -> constraints/inferences."""

    def __init__(self):
        self._rules: List[_ReasoningRule] = [
            _RequiredCharacterRule(),
            _TimelineProgressRule(),
            _WorldContinuityRule(),
            _DependencyRule(),
        ]

    def reason(self, state: StoryState, scene_plan: ScenePlan) -> StateReasoningResult:
        """Infer structured constraints and transition bands from current state."""
        graph = SymbolicStateGraph(state)
        graph_inference = graph.infer()

        inferred: List[TransitionConstraint] = []
        for rule in self._rules:
            inferred.extend(rule.infer(state, scene_plan))
        inferred.extend(
            self._constraints_from_propagation(
                scene_id=scene_plan.scene_id,
                propagated_constraints=graph_inference.propagated_constraints,
            )
        )

        candidate_operators = self._infer_candidate_operators(
            state=state,
            scene_plan=scene_plan,
            propagated_constraints=graph_inference.propagated_constraints,
            conflict_candidates=graph_inference.conflict_candidates,
        )
        forbidden_operators = self._infer_forbidden_operators(
            state=state,
            scene_plan=scene_plan,
            conflict_candidates=graph_inference.conflict_candidates,
        )
        selected_operator = self._select_operator(
            candidate_operators=candidate_operators,
            forbidden_operators=forbidden_operators,
            scene_plan=scene_plan,
        )
        execution_spec = selected_operator.execution_spec

        allowed = self._build_allowed_transitions(state, scene_plan, selected_operator)
        forbidden = self._build_forbidden_transitions(state, scene_plan)
        required_state_changes = list(scene_plan.expected_state_changes)
        if execution_spec.required_state_changes:
            required_state_changes.extend(
                [item for item in execution_spec.required_state_changes if item not in required_state_changes]
            )

        notes = [
            f"state_id={state.state_id}",
            f"step_index={state.step_index}",
            f"rules_applied={len(self._rules)}",
            f"inferred_constraints={len(inferred)}",
            f"candidate_operators={len(candidate_operators)}",
            f"selected_operator={selected_operator.operator_type}",
            f"propagated_constraints={len(graph_inference.propagated_constraints)}",
            f"conflict_candidates={len(graph_inference.conflict_candidates)}",
        ]
        return StateReasoningResult(
            state=state,
            inferred_constraints=inferred,
            candidate_operators=candidate_operators,
            forbidden_operators=forbidden_operators,
            selected_operator=selected_operator,
            execution_spec=execution_spec,
            allowed_transitions=allowed,
            forbidden_transitions=forbidden,
            required_state_changes=required_state_changes,
            propagated_constraints=list(graph_inference.propagated_constraints),
            conflict_candidates=list(graph_inference.conflict_candidates),
            notes=notes,
        )

    def derive_action(
        self,
        state: StoryState,
        scene_plan: ScenePlan,
        scene_text: str,
        extraction: SceneExtraction,
        selected_operator: TransitionOperator | None = None,
    ) -> TransitionAction:
        """Build structured action representation from one generated scene."""
        event_ids = [event.event_id for event in extraction.new_events]
        event_orders = [int(event.order) for event in extraction.new_events]
        actors = sorted(
            {
                entity.entity_id
                for entity in (extraction.new_entities + extraction.updated_entities)
            }
        )
        summary = extraction.new_events[0].description if extraction.new_events else scene_text[:280]
        location_from = state.world_state.current_setting_state
        location_to = str(extraction.world_updates.get("current_setting_state", location_from)).strip()
        if not location_to:
            location_to = location_from

        state_changes = list(scene_plan.expected_state_changes)
        for entity in extraction.updated_entities:
            state_changes.append(f"{entity.entity_id} -> {entity.status}@{entity.location}")
        for entity in extraction.new_entities:
            state_changes.append(f"new_entity:{entity.entity_id} -> {entity.status}@{entity.location}")
        for event in extraction.new_events:
            state_changes.append(f"event_added:{event.event_id}@order{event.order}")

        realized_postconditions = self._infer_realized_postconditions(
            extraction=extraction,
            scene_text=scene_text,
            selected_operator=selected_operator,
        )

        return TransitionAction(
            action_id=f"action_{scene_plan.scene_id}",
            scene_id=scene_plan.scene_id,
            chapter_id=scene_plan.chapter_id,
            summary=summary,
            operator_id=selected_operator.operator_id if selected_operator else "",
            operator_type=selected_operator.operator_type if selected_operator else "",
            declared_preconditions=list(selected_operator.preconditions) if selected_operator else [],
            expected_postconditions=list(selected_operator.postconditions) if selected_operator else [],
            realized_postconditions=realized_postconditions,
            execution_spec=(
                selected_operator.execution_spec if selected_operator else OperatorExecutionSpec()
            ),
            actors=actors,
            event_ids=event_ids,
            event_orders=event_orders,
            relation_updates=extraction.relation_updates,
            state_changes=state_changes,
            location_from=location_from,
            location_to=location_to,
            required_constraints=list(scene_plan.required_constraints) + list(scene_plan.must_keep_constraints),
            forbidden_constraints=list(scene_plan.forbidden_constraints) + list(scene_plan.forbidden_state_changes),
            raw_evidence_spans=list(extraction.raw_evidence_spans),
            confidence=float(extraction.confidence),
            sentence_ids=[unit.sentence_id for unit in extraction.sentences],
        )

    def state_to_prompt_payload(self, state: StoryState) -> Dict[str, object]:
        """Serialize state in prompt-friendly compact payload."""
        return asdict(state)

    def constraints_to_prompt_lines(self, constraints: List[TransitionConstraint]) -> List[str]:
        """Render structured constraints as concise lines for prompts."""
        lines: List[str] = []
        for item in constraints:
            lines.append(
                f"- [{item.facet}/{item.severity}] {item.predicate} "
                f"args={item.arguments} expected={item.expected}"
            )
        return lines

    def _build_allowed_transitions(
        self,
        state: StoryState,
        scene_plan: ScenePlan,
        selected_operator: TransitionOperator,
    ) -> List[str]:
        allowed: List[str] = []
        required_chars = scene_plan.required_characters or scene_plan.involved_characters
        if required_chars:
            allowed.append(f"actors_include_any({required_chars})")

        current_loc = state.world_state.current_setting_state or "unknown"
        allowed.append(f"location_stay_or_explicit_transition(from={current_loc})")

        expected_next = max(0, int(state.timeline.last_event_order) + 1)
        allowed.append(f"event_order_gte({expected_next})")
        allowed.extend(f"state_change:{item}" for item in scene_plan.expected_state_changes)
        allowed.append(f"operator_type={selected_operator.operator_type}")
        for item in selected_operator.postconditions:
            allowed.append(f"operator_postcondition:{item}")
        return allowed

    def _build_forbidden_transitions(self, state: StoryState, scene_plan: ScenePlan) -> List[str]:
        forbidden: List[str] = []
        forbidden.extend(scene_plan.forbidden_constraints)
        forbidden.extend(scene_plan.forbidden_state_changes)
        for cid, char_state in state.character_states.items():
            if char_state.status.lower() in {"dead", "removed"}:
                forbidden.append(f"actor={cid}")
        return sorted({item for item in forbidden if str(item).strip()})

    def _infer_candidate_operators(
        self,
        state: StoryState,
        scene_plan: ScenePlan,
        propagated_constraints: List[str],
        conflict_candidates: List[str],
    ) -> List[TransitionOperator]:
        text = " ".join(
            [
                scene_plan.objective,
                scene_plan.title,
                " ".join(scene_plan.expected_state_changes),
            ]
        ).lower()
        ordered: List[str] = []
        if any(token in text for token in ("introduce", "reveal", "discover")):
            ordered.append("REVEAL")
        if any(token in text for token in ("transition", "travel", "move", "enter", "leave")):
            ordered.append("MOVE")
        if any(token in text for token in ("conflict", "clash", "fight", "stakes", "pressure")):
            ordered.append("CONFLICT")
        if any(token in text for token in ("resolve", "closure", "reconcile", "end")):
            ordered.append("RESOLVE")
        if not ordered:
            ordered.append("CONFLICT")

        # Keep small but explicit operator set for one-step selection.
        for fallback in ("MOVE", "REVEAL", "RESOLVE"):
            if fallback not in ordered:
                ordered.append(fallback)
            if len(ordered) >= 3:
                break

        base_preconditions = list(scene_plan.preconditions)
        if not base_preconditions:
            base_preconditions = [f"{scene_plan.scene_id}:scene_ready"]
        location = state.world_state.current_setting_state or "unknown"

        candidates: List[TransitionOperator] = []
        for idx, op_type in enumerate(ordered[:3]):
            postconditions = self._operator_postconditions(op_type, scene_plan)
            preconditions = list(base_preconditions)
            preconditions.append(f"world_location_known:{location}")
            execution_spec = self._build_execution_spec(
                operator_type=op_type,
                scene_plan=scene_plan,
                propagated_constraints=propagated_constraints,
                conflict_candidates=conflict_candidates,
            )
            candidates.append(
                TransitionOperator(
                    operator_id=f"op_{scene_plan.scene_id}_{idx}",
                    operator_type=op_type,
                    scene_id=scene_plan.scene_id,
                    chapter_id=scene_plan.chapter_id,
                    preconditions=preconditions,
                    postconditions=postconditions,
                    required_effects=list(postconditions),
                    execution_spec=execution_spec,
                    rationale=f"Selected from scene objective and expected state changes for {scene_plan.scene_id}.",
                    priority=idx,
                )
            )
        return candidates

    def _infer_forbidden_operators(
        self,
        state: StoryState,
        scene_plan: ScenePlan,
        conflict_candidates: List[str],
    ) -> List[str]:
        forbidden: List[str] = []
        text = " ".join(scene_plan.forbidden_constraints + scene_plan.forbidden_state_changes).lower()
        if "forbidden_outcome" in text or "must not resolve" in text:
            forbidden.append("RESOLVE")
        if any("dead" in cs.status.lower() for cs in state.character_states.values()):
            forbidden.append("REVEAL")
        if any("relation_conflict:" in item for item in conflict_candidates):
            forbidden.append("RESOLVE")
        return sorted(set(forbidden))

    def _select_operator(
        self,
        candidate_operators: List[TransitionOperator],
        forbidden_operators: List[str],
        scene_plan: ScenePlan,
    ) -> TransitionOperator:
        forbidden = {item.upper() for item in forbidden_operators}
        for operator in candidate_operators:
            if operator.operator_type.upper() not in forbidden:
                return operator
        # Ensure every scene has one selected operator.
        return TransitionOperator(
            operator_id=f"op_{scene_plan.scene_id}_fallback",
            operator_type="CONFLICT",
            scene_id=scene_plan.scene_id,
            chapter_id=scene_plan.chapter_id,
            preconditions=list(scene_plan.preconditions) or [f"{scene_plan.scene_id}:scene_ready"],
            postconditions=self._operator_postconditions("CONFLICT", scene_plan),
            required_effects=self._operator_postconditions("CONFLICT", scene_plan),
            execution_spec=self._build_execution_spec(
                operator_type="CONFLICT",
                scene_plan=scene_plan,
                propagated_constraints=[],
                conflict_candidates=[],
            ),
            rationale="Fallback operator because all candidates were forbidden.",
            priority=99,
        )

    def _operator_postconditions(self, operator_type: str, scene_plan: ScenePlan) -> List[str]:
        base = list(scene_plan.expected_state_changes)
        op = operator_type.upper()
        if op == "MOVE":
            base.append("location_transition_evidence_present")
        elif op == "REVEAL":
            base.append("new_information_revealed")
        elif op == "RESOLVE":
            base.append("major_tension_reduced")
        else:
            base.append("stakes_increase_or_conflict_visible")
        return sorted({item for item in base if str(item).strip()})

    def _infer_realized_postconditions(
        self,
        extraction: SceneExtraction,
        scene_text: str,
        selected_operator: TransitionOperator | None,
    ) -> List[str]:
        if selected_operator is None:
            return []
        realized: List[str] = []
        lowered = (scene_text or "").lower()
        expected = list(selected_operator.postconditions)
        expected.extend(selected_operator.execution_spec.required_state_changes)
        for post in sorted(set(expected)):
            token = post.lower()
            if token in lowered:
                realized.append(post)
                continue
            if "location_transition_evidence_present" in token:
                cues = ("move", "travel", "arrive", "enter", "leave", "return", "transition")
                if any(cue in lowered for cue in cues):
                    realized.append(post)
                    continue
            if "new_information_revealed" in token and extraction.new_events:
                realized.append(post)
                continue
            if "stakes_increase_or_conflict_visible" in token and extraction.new_events:
                realized.append(post)
                continue
            if "major_tension_reduced" in token and any(
                "resolve" in evt.description.lower() or "calm" in evt.description.lower()
                for evt in extraction.new_events
            ):
                realized.append(post)
                continue
            if any(change.lower() in token for change in extraction.world_updates.keys()):
                realized.append(post)
        return sorted(set(realized))

    def _constraints_from_propagation(
        self,
        scene_id: str,
        propagated_constraints: List[str],
    ) -> List[TransitionConstraint]:
        inferred: List[TransitionConstraint] = []
        for idx, item in enumerate(propagated_constraints):
            token = str(item).strip()
            if not token:
                continue
            severity = "warning"
            if token.startswith("forbid_actor:") or token.startswith("next_event_order_min:"):
                severity = "error"
            inferred.append(
                TransitionConstraint(
                    constraint_id=f"graph_prop_rule:{scene_id}:{idx}",
                    facet="timeline_plot",
                    predicate="graph_propagated_constraint",
                    arguments={"constraint": token},
                    expected=f"must satisfy propagated graph constraint: {token}",
                    severity=severity,
                    source="symbolic_state_graph",
                )
            )
        return inferred

    def _build_execution_spec(
        self,
        operator_type: str,
        scene_plan: ScenePlan,
        propagated_constraints: List[str],
        conflict_candidates: List[str],
    ) -> OperatorExecutionSpec:
        required_entities = list(scene_plan.required_characters or scene_plan.involved_characters)
        required_events = [
            event
            for event in list(scene_plan.key_events[:2])
            if self._is_scene_hard_event(event)
        ]
        required_state_changes = self._operator_postconditions(operator_type, scene_plan)
        forbidden_patterns = list(scene_plan.forbidden_constraints)
        forbidden_patterns.extend(scene_plan.forbidden_state_changes)
        forbidden_patterns.extend([f"conflict:{item}" for item in conflict_candidates[:2]])
        return OperatorExecutionSpec(
            required_entities=sorted(set(required_entities)),
            required_events=sorted(set(required_events)),
            required_state_changes=sorted(set(required_state_changes)),
            forbidden_patterns=sorted(set(item for item in forbidden_patterns if str(item).strip())),
            require_event_keyword_match=False,
            require_actor_coverage=True,
            allow_fuzzy_event_match=True,
        )

    def _is_scene_hard_event(self, event: object) -> bool:
        text = str(event or "").strip()
        if not text:
            return False
        if text.startswith("scene_event_"):
            return False
        lowered = text.lower()
        instructional_prefixes = (
            "write a story",
            "start with",
            "include ",
            "explore themes",
            "the story should",
            "make sure",
            "aim for",
        )
        if lowered.startswith(instructional_prefixes):
            return False
        # Overly long prompt-like clauses are poor hard-event anchors.
        if len(text.split()) > 18:
            return False
        return True
