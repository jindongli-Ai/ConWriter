"""Transition validity checker: Valid(State_t, Action) -> bool/violations."""

from __future__ import annotations

from typing import List, Tuple

from ConWriter.reasoning.scene_alignment import (
    build_anchor,
    detect_textual_event_realization,
    find_sentence_ids_for_tokens,
    find_temporal_conflict_sentences,
)
from ConWriter.pipeline.weighted_constraints import weight_transition_constraint, weight_violation
from ConWriter.utils.types import (
    ConstraintViolation,
    OperatorExecutionSpec,
    SentenceUnit,
    StateReasoningResult,
    StoryState,
    TransitionAction,
    TransitionConstraint,
    TransitionOperator,
    ViolationAnchor,
)


class TransitionValidator:
    """Validate structured transitions against state-bound constraints."""

    def precheck_state(self, reasoning: StateReasoningResult) -> List[ConstraintViolation]:
        """State-only precheck before generation to catch impossible transition setup."""
        violations: List[ConstraintViolation] = []
        if not reasoning.allowed_transitions:
            violations.append(
                ConstraintViolation(
                    rule_type="transition_validity",
                    message="No allowed transitions inferred from current state.",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[reasoning.state.state_id],
                    repair_hint="Ensure state reasoning produces at least one actionable transition.",
                    repair_scope="plan",
                    patchable=False,
                    fatal=True,
                    needs_replan=True,
                )
            )
        if reasoning.selected_operator is None:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message="No selected operator inferred for current scene.",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[reasoning.state.state_id],
                    repair_hint="StateReasoner must select one operator before generation.",
                    repair_scope="plan",
                    patchable=False,
                    fatal=True,
                    needs_replan=True,
                )
            )
        elif reasoning.selected_operator.operator_type.upper() in {
            item.upper() for item in reasoning.forbidden_operators
        }:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=(
                        f"Selected operator '{reasoning.selected_operator.operator_type}' "
                        "is forbidden by current state."
                    ),
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[reasoning.state.state_id, reasoning.selected_operator.operator_id],
                    repair_hint="Choose an operator outside forbidden_operators before generation.",
                    context={
                        "violated_operator": reasoning.selected_operator.operator_type,
                        "inconsistent_state": "forbidden_operator_selected",
                    },
                    repair_scope="plan",
                    patchable=False,
                    fatal=True,
                    needs_replan=True,
                )
            )
        elif not reasoning.execution_spec.required_state_changes and not reasoning.execution_spec.required_entities:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message="Selected operator has empty execution specification.",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[reasoning.state.state_id, reasoning.selected_operator.operator_id],
                    repair_hint="StateReasoner must provide non-empty OperatorExecutionSpec.",
                    context={
                        "violated_operator": reasoning.selected_operator.operator_type,
                        "inconsistent_state": "empty_execution_spec",
                    },
                    repair_scope="plan",
                    patchable=False,
                    fatal=True,
                    needs_replan=True,
                )
            )
        for violation in violations:
            weight_violation(violation)
        return violations

    def validate(
        self,
        state: StoryState,
        action: TransitionAction,
        constraints: List[TransitionConstraint],
        forbidden_transitions: List[str],
        selected_operator: TransitionOperator | None = None,
        sentence_units: List[SentenceUnit] | None = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """Run structured Valid(State_t, Action) checks."""
        violations: List[ConstraintViolation] = []
        violations.extend(self._check_operator_realization(state, action, selected_operator, sentence_units or []))
        for constraint in constraints:
            weight_transition_constraint(constraint)
            violations.extend(self._check_constraint(state, action, constraint))
        violations.extend(self._check_forbidden_patterns(action, forbidden_transitions))
        for idx, violation in enumerate(violations):
            self._classify_violation(violation)
            if sentence_units:
                anchor = self._build_violation_anchor(
                    violation=violation,
                    sentence_units=sentence_units,
                    action=action,
                    idx=idx,
                )
                violation.anchors = [anchor]
            weight_violation(violation)
        return len([v for v in violations if v.severity.lower() == "error"]) == 0, violations

    def _check_constraint(
        self,
        state: StoryState,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        predicate = constraint.predicate
        if predicate == "min_next_event_order":
            return self._check_min_next_event_order(action, constraint)
        if predicate == "character_must_be_introduced_or_acted":
            return self._check_required_character_action(state, action, constraint)
        if predicate == "character_cannot_be_actor":
            return self._check_forbidden_actor(action, constraint)
        if predicate == "location_transition_requires_evidence":
            return self._check_location_transition_evidence(action, constraint)
        if predicate == "dependency_scene_should_be_reflected":
            return self._check_dependency_reflection(state, action, constraint)
        if predicate == "graph_propagated_constraint":
            return self._check_graph_propagated_constraint(action, constraint)
        return []

    def _check_min_next_event_order(
        self,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        expected_min = int(constraint.arguments.get("min_order", 0))
        if not action.event_orders:
            return [
                self._violation(
                    constraint,
                    message=f"Action has no event order; expected order >= {expected_min}.",
                    related_ids=[action.scene_id],
                )
            ]
        observed_max = max(action.event_orders)
        if observed_max < expected_min:
            return [
                self._violation(
                    constraint,
                    message=f"Event order regression: observed {observed_max} < expected {expected_min}.",
                    related_ids=[action.scene_id],
                )
            ]
        return []

    def _check_required_character_action(
        self,
        state: StoryState,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        character_id = str(constraint.arguments.get("character_id", "")).strip()
        if not character_id:
            return []
        if character_id in action.actors:
            return []
        if character_id in state.character_states and state.character_states[character_id].status.lower() == "introduced":
            return [
                self._violation(
                    constraint,
                    message=f"Required character '{character_id}' not acted in transition action.",
                    related_ids=[character_id, action.scene_id],
                )
            ]
        return []

    def _check_forbidden_actor(
        self,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        character_id = str(constraint.arguments.get("character_id", "")).strip()
        if character_id and character_id in action.actors:
            return [
                self._violation(
                    constraint,
                    message=f"Forbidden actor '{character_id}' appears in transition action.",
                    related_ids=[character_id, action.scene_id],
                )
            ]
        return []

    def _check_location_transition_evidence(
        self,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        src = (action.location_from or "").strip().lower()
        dst = (action.location_to or "").strip().lower()
        if not src or not dst or src == dst:
            return []
        lowered = action.summary.lower()
        cues = ("move", "travel", "arrive", "enter", "leave", "return", "transition")
        if any(token in lowered for token in cues):
            return []
        return [
            self._violation(
                constraint,
                message=(
                    f"Location changed from '{action.location_from}' to '{action.location_to}' "
                    "without explicit transition evidence."
                ),
                related_ids=[action.scene_id],
            )
        ]

    def _check_dependency_reflection(
        self,
        state: StoryState,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        dep_id = str(constraint.arguments.get("dependency_scene_id", "")).strip()
        if not dep_id:
            return []
        known_ids = set(state.timeline.recent_event_ids)
        if dep_id in known_ids:
            return []
        # weak check: allow if action summary explicitly references dependency id.
        if dep_id.lower() in action.summary.lower():
            return []
        return [
            self._violation(
                constraint,
                message=f"Dependency '{dep_id}' not reflected in current transition context.",
                related_ids=[action.scene_id, dep_id],
            )
        ]

    def _check_forbidden_patterns(
        self,
        action: TransitionAction,
        forbidden_transitions: List[str],
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        lowered = action.summary.lower()
        for token in forbidden_transitions:
            txt = str(token).strip()
            if not txt:
                continue
            # Support actor=char_x form in structured forbidden transition.
            if txt.startswith("actor="):
                actor = txt.split("=", 1)[1].strip()
                if actor and actor in action.actors:
                    violations.append(
                        ConstraintViolation(
                            rule_type="transition_validity",
                            message=f"Forbidden transition matched: {txt}",
                            severity="error",
                            facet="characterization",
                            related_ids=[actor, action.scene_id],
                            repair_hint="Remove forbidden actor from current scene transition.",
                        )
                    )
                continue
            if txt.lower() in lowered:
                violations.append(
                    ConstraintViolation(
                        rule_type="transition_validity",
                        message=f"Forbidden transition matched in action summary: '{txt}'.",
                        severity="error",
                        facet="world_setting",
                        related_ids=[action.scene_id],
                        repair_hint="Rewrite scene action to avoid forbidden transition content.",
                    )
                )
        return violations

    def _check_operator_realization(
        self,
        state: StoryState,
        action: TransitionAction,
        selected_operator: TransitionOperator | None,
        sentence_units: List[SentenceUnit],
    ) -> List[ConstraintViolation]:
        if selected_operator is None:
            return []
        violations: List[ConstraintViolation] = []
        execution_spec = selected_operator.execution_spec
        op_type = selected_operator.operator_type.upper()
        action_type = (action.operator_type or "").upper()
        if action_type and action_type != op_type:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Selected operator '{op_type}' mismatches realized action operator '{action_type}'.",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Rewrite scene to realize the selected operator type.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"action_operator_type={action_type}", f"selected_operator_type={op_type}"],
                        "inconsistent_state": "operator_type_mismatch",
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "operator_type_mismatch"),
                    },
                )
            )

        spec_violations = self._check_operator_execution_spec(
            action=action,
            selected_operator=selected_operator,
            execution_spec=execution_spec,
            sentence_units=sentence_units,
        )
        violations.extend(spec_violations)

        missing_pre = self._missing_preconditions(state, selected_operator.preconditions)
        for pre in missing_pre:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Operator precondition not met: '{pre}'.",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Revise scene or planning state to satisfy selected operator preconditions.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"missing_precondition:{pre}"],
                        "inconsistent_state": pre,
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_precondition"),
                    },
                )
            )

        missing_post = self._missing_postconditions(action, selected_operator.postconditions)
        for post in missing_post:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Operator postcondition missing: '{post}'.",
                    severity="error",
                    facet="world_setting",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Rewrite scene to realize required operator effects.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"missing_postcondition:{post}"],
                        "missing_postcondition": post,
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_postcondition"),
                    },
                )
            )
        return violations

    def _check_operator_execution_spec(
        self,
        action: TransitionAction,
        selected_operator: TransitionOperator,
        execution_spec: OperatorExecutionSpec,
        sentence_units: List[SentenceUnit],
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        op_type = selected_operator.operator_type.upper()
        lowered_summary = (action.summary or "").lower()
        state_changes = [str(item).lower() for item in action.state_changes]

        missing_entities: List[str] = []
        if execution_spec.require_actor_coverage:
            missing_entities = [
                entity for entity in execution_spec.required_entities if entity and entity not in action.actors
            ]
        if missing_entities:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Execution spec missing required entities: {missing_entities}",
                    severity="error",
                    facet="characterization",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Add missing required entities into the scene action.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"missing_entities:{missing_entities}"],
                        "inconsistent_state": "required_entities_not_realized",
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_entities"),
                    },
                )
            )

        missing_events: List[str] = []
        inferred_only_events: List[str] = []
        for event in execution_spec.required_events:
            token = str(event).strip().lower()
            if not token:
                continue
            explicit_realized, explicit_sent_ids = detect_textual_event_realization(sentence_units, token)
            if execution_spec.require_event_keyword_match and explicit_realized:
                continue
            if (not execution_spec.require_event_keyword_match) and (token in lowered_summary):
                continue
            if execution_spec.allow_fuzzy_event_match and action.event_ids:
                norm_token = " ".join(token.replace("_", " ").split())
                if any(
                    norm_token in " ".join(str(event_id).lower().replace("_", " ").split())
                    for event_id in action.event_ids
                ):
                    inferred_only_events.append(str(event))
                    continue
            missing_events.append(str(event))
        if missing_events:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Execution spec missing required events: {missing_events}",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Realize required events explicitly in this scene.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"missing_events:{missing_events}"],
                        "inconsistent_state": "required_events_not_realized",
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_events"),
                        "required_event_tokens": list(missing_events),
                        "textual_realization": "missing",
                    },
                )
            )
        elif inferred_only_events:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Execution spec required events inferred but not textually realized: {inferred_only_events}",
                    severity="error",
                    facet="timeline_plot",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Make required events explicit in scene text, not only inferred from IDs.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"inferred_only_events:{inferred_only_events}"],
                        "inconsistent_state": "required_events_inferred_only",
                        "required_event_tokens": list(inferred_only_events),
                        "textual_realization": "inferred_only",
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_events"),
                    },
                )
            )

        missing_state_changes: List[str] = []
        for change in execution_spec.required_state_changes:
            token = str(change).strip().lower()
            if not token:
                continue
            if token in lowered_summary:
                continue
            if any(token in row for row in state_changes):
                continue
            if token in {item.lower() for item in action.realized_postconditions}:
                continue
            missing_state_changes.append(str(change))
        if missing_state_changes:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Execution spec missing required state changes: {missing_state_changes}",
                    severity="error",
                    facet="world_setting",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Rewrite scene to realize required state change effects.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"missing_state_changes:{missing_state_changes}"],
                        "missing_postcondition": ",".join(missing_state_changes[:2]),
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "missing_state_changes"),
                    },
                )
            )

        forbidden_hits: List[str] = []
        for pattern in execution_spec.forbidden_patterns:
            token = str(pattern).strip().lower()
            if token and token in lowered_summary:
                forbidden_hits.append(str(pattern))
        if forbidden_hits:
            violations.append(
                ConstraintViolation(
                    rule_type="operator_validity",
                    message=f"Execution spec hit forbidden patterns: {forbidden_hits}",
                    severity="error",
                    facet="world_setting",
                    related_ids=[action.scene_id, selected_operator.operator_id],
                    repair_hint="Remove forbidden patterns while preserving selected operator effects.",
                    context={
                        "violated_operator": op_type,
                        "conflicting_facts": [f"forbidden_patterns:{forbidden_hits}"],
                        "inconsistent_state": "forbidden_pattern_hit",
                        "suggested_repair_operator": self._suggest_repair_operator(op_type, "forbidden_pattern"),
                    },
                )
            )
        return violations

    def _missing_preconditions(
        self,
        state: StoryState,
        preconditions: List[str],
    ) -> List[str]:
        missing: List[str] = []
        known_events = set(state.timeline.recent_event_ids)
        current_location = (state.world_state.current_setting_state or "").strip().lower()
        for item in preconditions:
            token = str(item).strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered.endswith(":accepted"):
                dep = token.split(":", 1)[0].strip()
                dep_event_id = f"evt_{dep}" if dep else ""
                if dep and dep not in known_events and dep_event_id not in known_events:
                    missing.append(token)
                continue
            if lowered.startswith("world_location_known"):
                if not current_location or current_location == "unknown":
                    missing.append(token)
                continue
        return missing

    def _check_graph_propagated_constraint(
        self,
        action: TransitionAction,
        constraint: TransitionConstraint,
    ) -> List[ConstraintViolation]:
        token = str(constraint.arguments.get("constraint", "")).strip().lower()
        if not token:
            return []
        if token.startswith("forbid_actor:"):
            actor = token.split(":", 1)[1].strip()
            if actor and actor in action.actors:
                return [
                    self._violation(
                        constraint,
                        message=f"Graph propagated constraint violated: forbidden actor '{actor}' appears.",
                        related_ids=[action.scene_id, actor],
                    )
                ]
            return []
        if token.startswith("next_event_order_min:"):
            expected = int(token.split(":", 1)[1])
            if action.event_orders and max(action.event_orders) >= expected:
                return []
            return [
                self._violation(
                    constraint,
                    message=f"Graph propagated temporal constraint violated: expected event order >= {expected}.",
                    related_ids=[action.scene_id],
                )
            ]
        # Non-critical graph constraints are treated as advisory.
        return []

    def _missing_postconditions(
        self,
        action: TransitionAction,
        postconditions: List[str],
    ) -> List[str]:
        missing: List[str] = []
        realized = {str(item).strip().lower() for item in action.realized_postconditions if str(item).strip()}
        lowered_summary = (action.summary or "").lower()
        lowered_changes = [str(item).lower() for item in action.state_changes]
        for item in postconditions:
            token = str(item).strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered in realized:
                continue
            if any(lowered in change for change in lowered_changes):
                continue
            if lowered in lowered_summary:
                continue
            missing.append(token)
        return missing

    def _violation(
        self,
        constraint: TransitionConstraint,
        message: str,
        related_ids: List[str],
    ) -> ConstraintViolation:
        violation = ConstraintViolation(
            rule_type="transition_validity",
            message=message,
            severity=constraint.severity,
            facet=constraint.facet,
            related_ids=related_ids,
            repair_hint=f"Repair to satisfy predicate={constraint.predicate}.",
            constraint_weight=float(constraint.constraint_weight),
            constraint_tier=int(constraint.constraint_tier),
            is_hard=bool(constraint.is_hard),
            weighted_priority_source=str(constraint.weighted_priority_source),
        )
        violation.weighted_cost = float(violation.constraint_weight) * (1.25 if violation.is_hard else 1.0)
        return violation

    def _classify_violation(self, violation: ConstraintViolation) -> None:
        message = violation.message.lower()
        violation.patchable = True
        violation.fatal = False
        violation.needs_replan = False
        if "precondition not met" in message:
            violation.repair_scope = "plan"
            violation.needs_replan = True
            violation.patchable = False
            violation.fatal = True
            return
        if "execution spec missing required events" in message:
            violation.repair_scope = "paragraph"
            return
        if "operator postcondition missing" in message or "required state changes" in message:
            violation.repair_scope = "sentence"
            return
        if "event order regression" in message:
            violation.repair_scope = "sentence"
            return
        if "forbidden transition matched" in message:
            violation.repair_scope = "sentence"
            return
        violation.repair_scope = "sentence"

    def _build_violation_anchor(
        self,
        violation: ConstraintViolation,
        sentence_units: List[SentenceUnit],
        action: TransitionAction,
        idx: int,
    ) -> ViolationAnchor:
        tokens = list(violation.related_ids)
        tokens.append(violation.rule_type)
        if isinstance(violation.context, dict):
            tokens.extend(str(x) for x in violation.context.get("required_event_tokens", []))
        tokens.extend(action.required_constraints[:2])
        tokens.extend(action.forbidden_constraints[:2])
        sentence_ids = find_sentence_ids_for_tokens(sentence_units, tokens)
        if violation.rule_type == "transition_validity":
            sentence_ids.extend(find_temporal_conflict_sentences(sentence_units, tokens))
        if not sentence_ids and sentence_units:
            sentence_ids = [sentence_units[-1].sentence_id]
        related_entities = [rid for rid in violation.related_ids if rid.startswith(("char_", "ent_"))]
        related_events = [rid for rid in violation.related_ids if rid.startswith("evt_")]
        textual_realization = "explicit"
        if isinstance(violation.context, dict):
            textual_realization = str(violation.context.get("textual_realization", "explicit"))
        return build_anchor(
            anchor_id=f"anchor_{action.scene_id}_{idx:03d}",
            rule_type=violation.rule_type,
            severity=violation.severity,
            sentence_ids=sentence_ids,
            sentences=sentence_units,
            related_entity_ids=related_entities,
            related_event_ids=related_events,
            related_relation_ids=[],
            textual_realization=textual_realization,
            grounding_confidence=0.8 if textual_realization == "explicit" else (0.6 if textual_realization == "inferred" else 0.4),
            confidence_score=0.8 if textual_realization == "explicit" else (0.6 if textual_realization == "inferred" else 0.4),
            source_type=textual_realization if textual_realization in {"explicit", "inferred"} else "heuristic",
            notes=[violation.message],
        )

    def _suggest_repair_operator(self, operator_type: str, reason: str) -> str:
        op = operator_type.upper()
        if reason in {"missing_entities", "missing_events"}:
            return "REVEAL"
        if reason in {"missing_postcondition", "missing_state_changes"}:
            if op == "RESOLVE":
                return "RESOLVE"
            if op == "MOVE":
                return "MOVE"
            return "CONFLICT"
        if reason == "forbidden_pattern":
            return "REVEAL" if op == "CONFLICT" else op
        return op
