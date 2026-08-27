"""Pseudo-symbolic constraint checker for incremental scene generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ConWriter.reasoning.scene_alignment import (
    build_anchor,
    find_temporal_conflict_sentences,
    find_sentence_ids_for_tokens,
    split_scene_into_units,
)
from ConWriter.pipeline.weighted_constraints import weight_violation
from ConWriter.utils.types import (
    ConsistencyReport,
    ConstraintViolation,
    DynamicMemory,
    MemoryDelta,
    NeuralFinding,
    SceneExtraction,
    ScenePlan,
    StaticMemory,
    SymbolicFinding,
    ViolationAnchor,
)


class ConstraintChecker:
    """Run scene-level character/timeline/world consistency checks."""

    RULE_TYPES = (
        "character_consistency",
        "timeline_consistency",
        "world_rule_consistency",
    )

    _CONSTRAINT_STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "must",
        "should",
        "scene",
        "event",
    }

    def build_scene_constraints(
        self,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        scene_plan: ScenePlan,
    ) -> Dict[str, object]:
        """Collect constraints consumed by generator and pre-check."""
        required = list(scene_plan.required_constraints)
        required.extend(scene_plan.must_keep_constraints)
        required = [str(c) for c in required if str(c).strip()]
        forbidden = [str(c) for c in scene_plan.forbidden_constraints if str(c).strip()]
        pending = list(dynamic_memory.timeline_plot.pending_constraints)
        pending.extend(required)
        state_constraints = [
            f"{item.predicate}:{item.arguments}"
            for item in dynamic_memory.inferred_constraints
        ]
        return {
            "required": sorted(set(required)),
            "forbidden": sorted(set(forbidden)),
            "must_keep": [str(c) for c in scene_plan.must_keep_constraints if str(c).strip()],
            "world_invariants": list(static_memory.world_setting.world_invariants),
            "required_characters": list(scene_plan.required_characters or scene_plan.involved_characters),
            "optional_characters": list(scene_plan.optional_characters),
            "involved_characters": list(scene_plan.involved_characters),
            "current_world_state": dynamic_memory.world_setting.current_setting_state,
            "pending_constraints": sorted(set(str(c) for c in pending if str(c).strip())),
            "expected_state_changes": list(scene_plan.expected_state_changes),
            "state_id": dynamic_memory.current_state.state_id,
            "state_constraints": state_constraints,
        }

    def precheck_scene(
        self,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        scene_plan: ScenePlan,
    ) -> List[ConstraintViolation]:
        """Run checks before scene generation."""
        violations: List[ConstraintViolation] = []
        static_char_ids = set(static_memory.characterization.character_profiles.keys())
        for cid in scene_plan.involved_characters:
            if cid not in static_char_ids:
                violations.append(
                    ConstraintViolation(
                        rule_type="character_consistency",
                        message=f"Planned character '{cid}' missing in static memory profiles.",
                        severity="error",
                        facet="characterization",
                        related_ids=[cid],
                        repair_hint="Use only known character IDs from static memory.",
                        repair_scope="plan",
                        patchable=False,
                        fatal=True,
                        needs_replan=True,
                    )
                )

        overlap = set(scene_plan.required_constraints).intersection(set(scene_plan.forbidden_constraints))
        overlap.update(set(scene_plan.must_keep_constraints).intersection(set(scene_plan.forbidden_constraints)))
        if overlap:
            violations.append(
                    ConstraintViolation(
                        rule_type="world_rule_consistency",
                        message=f"Scene has contradictory required/forbidden constraints: {sorted(overlap)}",
                        severity="warning",
                        facet="world_setting",
                        related_ids=[scene_plan.scene_id],
                        repair_hint="Prefer removing contradictory constraints before generation.",
                        repair_scope="plan",
                        patchable=True,
                        fatal=False,
                        needs_replan=False,
                    )
                )

        if not dynamic_memory.world_setting.current_setting_state:
            violations.append(
                ConstraintViolation(
                    rule_type="world_rule_consistency",
                    message="Dynamic world state is empty before scene generation.",
                    severity="warning",
                    facet="world_setting",
                    related_ids=[scene_plan.scene_id],
                    repair_hint="Initialize world state from static setting description.",
                )
            )
        for violation in violations:
            weight_violation(violation)
        return violations

    def check_scene(
        self,
        static_memory: StaticMemory,
        candidate_memory: DynamicMemory,
        scene_plan: ScenePlan,
        delta: MemoryDelta,
        scene_text: str,
        scene_extraction: SceneExtraction | None = None,
    ) -> Tuple[ConsistencyReport, List[ConstraintViolation]]:
        """Run post-generation checks before accepting scene update."""
        violations: List[ConstraintViolation] = []
        violations.extend(self._check_character_consistency(static_memory, candidate_memory, scene_plan, delta))
        violations.extend(self._check_timeline_consistency(candidate_memory, delta))
        violations.extend(self._check_world_rule_consistency(static_memory, candidate_memory, scene_plan, delta, scene_text))
        canonical_entity_table = self._build_canonical_entity_table(static_memory, candidate_memory)
        neural_findings = self._collect_neural_structured_findings(
            scene_text=scene_text,
            scene_plan=scene_plan,
            canonical_entity_table=canonical_entity_table,
        )
        violations.extend(self._neural_findings_to_violations(neural_findings))

        sentence_units = (
            list(scene_extraction.sentences)
            if scene_extraction is not None and scene_extraction.sentences
            else split_scene_into_units(scene_text, scene_plan.scene_id)
        )
        violation_anchors: List[ViolationAnchor] = []
        for idx, violation in enumerate(violations):
            self._classify_violation_scope(violation)
            anchor = self._build_violation_anchor(
                violation=violation,
                scene_plan=scene_plan,
                sentence_units=sentence_units,
                scene_extraction=scene_extraction,
                idx=idx,
            )
            violation.anchors = [anchor]
            weight_violation(violation)
            violation_anchors.append(anchor)

        symbolic_findings = [
            SymbolicFinding(
                rule_type=v.rule_type,
                message=v.message,
                severity=v.severity,
                related_ids=v.related_ids,
                facet=v.facet,
            )
            for v in violations
        ]

        blocking = [v for v in violations if self._is_blocking(v)]
        fatal = any(v.fatal for v in violations)
        needs_replan = any(v.needs_replan for v in violations)
        severity = "error" if blocking else ("warning" if violations else "info")
        is_consistent = len(blocking) == 0
        violated_facets = sorted({v.facet for v in violations if v.facet})
        violated_rules = sorted({v.rule_type for v in violations})
        messages = [v.message for v in violations]
        repair_hints = [v.repair_hint for v in violations if v.repair_hint]
        repair_strategy = self._repair_strategy(
            is_consistent=is_consistent,
            needs_replan=needs_replan,
            fatal=fatal,
            violations=violations,
        )

        canonical_entity_table = self._build_canonical_entity_table(static_memory, candidate_memory)
        dual_decision, dual_summary = self._dual_consistency_decision(
            symbolic_findings=symbolic_findings,
            neural_findings=neural_findings,
            blocking_count=len(blocking),
        )

        report = ConsistencyReport(
            is_consistent=is_consistent,
            violation_types=violated_rules,
            violated_facets=violated_facets,
            violated_rules=violated_rules,
            messages=messages,
            symbolic_findings=symbolic_findings,
            neural_findings=neural_findings,
            suggested_action="accept" if is_consistent else "revise",
            repair_hints=repair_hints,
            conflict_spans=list(delta.raw_evidence_spans),
            conflict_slots=self._collect_conflict_slots(violations),
            repair_target=violated_facets[0] if len(violated_facets) == 1 else ("multi_facet" if violated_facets else "none"),
            repair_strategy=repair_strategy,
            facet_reports=self._build_facet_reports(violations),
            severity=severity,
            violation_anchors=violation_anchors,
            needs_replan=needs_replan,
            fatal=fatal,
            canonical_entity_table=canonical_entity_table,
            dual_consistency_decision=dual_decision,
            dual_consistency_summary=dual_summary,
        )
        if report.needs_replan:
            report.suggested_action = "replan"
        return report, violations

    def _build_facet_reports(self, violations: List[ConstraintViolation]) -> Dict[str, Dict[str, object]]:
        payload: Dict[str, Dict[str, object]] = {}
        for violation in violations:
            facet = violation.facet or "unscoped"
            row = payload.setdefault(
                facet,
                {
                    "is_consistent": True,
                    "num_findings": 0,
                    "violated_rules": [],
                    "messages": [],
                    "weighted_violation_score": 0.0,
                },
            )
            row["is_consistent"] = False
            row["num_findings"] = int(row["num_findings"]) + 1
            row["violated_rules"].append(violation.rule_type)
            row["messages"].append(violation.message)
            row["weighted_violation_score"] = float(row["weighted_violation_score"]) + float(
                violation.weighted_cost
            )
        for row in payload.values():
            row["violated_rules"] = sorted(set(row["violated_rules"]))
        return payload

    def _check_character_consistency(
        self,
        static_memory: StaticMemory,
        candidate_memory: DynamicMemory,
        scene_plan: ScenePlan,
        delta: MemoryDelta,
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        static_profiles = static_memory.characterization.character_profiles
        candidate_entities = candidate_memory.characterization.entity_store
        delta_ids = {e.entity_id for e in (delta.new_entities + delta.updated_entities)}

        for cid in scene_plan.involved_characters:
            if cid not in static_profiles:
                continue
            if cid not in candidate_entities and cid not in delta_ids:
                violations.append(
                    ConstraintViolation(
                        rule_type="character_consistency",
                        message=f"Planned character '{cid}' not represented in candidate dynamic memory.",
                        severity="error",
                        facet="characterization",
                        related_ids=[cid],
                        repair_hint="Include planned characters in the scene actions or state updates.",
                    )
                )

        for entity in delta.updated_entities:
            if entity.entity_id in static_profiles:
                canonical = static_profiles[entity.entity_id].canonical_name
                if entity.name.strip().lower() != canonical.strip().lower():
                    violations.append(
                        ConstraintViolation(
                            rule_type="character_consistency",
                            message=(
                                f"Entity '{entity.entity_id}' name drift: "
                                f"'{entity.name}' != canonical '{canonical}'."
                            ),
                            severity="warning",
                            facet="characterization",
                            related_ids=[entity.entity_id],
                            repair_hint="Keep canonical names stable in local scene rewrite.",
                        )
                    )
        return violations

    def _check_timeline_consistency(
        self,
        candidate_memory: DynamicMemory,
        delta: MemoryDelta,
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        timeline = candidate_memory.timeline_plot.event_timeline
        orders = [evt.order for evt in timeline]
        if orders != sorted(orders):
            violations.append(
                    ConstraintViolation(
                        rule_type="timeline_consistency",
                        message="Event order is non-monotonic in candidate timeline.",
                        severity="error",
                        facet="timeline_plot",
                        related_ids=[evt.event_id for evt in timeline],
                        repair_hint="Rewrite scene to preserve temporal progression.",
                    )
                )

        if delta.new_events and len(timeline) >= 2:
            prev = timeline[-2].order
            cur = timeline[-1].order
            if cur < prev:
                violations.append(
                    ConstraintViolation(
                        rule_type="timeline_consistency",
                        message=f"New event order regresses ({cur} < {prev}).",
                        severity="error",
                        facet="timeline_plot",
                        related_ids=[timeline[-2].event_id, timeline[-1].event_id],
                        repair_hint="Ensure this scene advances timeline instead of regressing it.",
                    )
                )
        return violations

    def _check_world_rule_consistency(
        self,
        static_memory: StaticMemory,
        candidate_memory: DynamicMemory,
        scene_plan: ScenePlan,
        delta: MemoryDelta,
        scene_text: str,
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        text = scene_text.lower()

        must_satisfy = list(scene_plan.required_constraints) + list(scene_plan.must_keep_constraints)
        for required in must_satisfy:
            constraint = required.strip()
            if not constraint:
                continue
            if not self._constraint_satisfied(constraint, scene_text, delta):
                violations.append(
                    ConstraintViolation(
                        rule_type="world_rule_consistency",
                        message=f"Scene does not satisfy required constraint: '{required}'.",
                        severity="error",
                        facet="world_setting",
                        related_ids=[scene_plan.scene_id, f"required:{required}"],
                        repair_hint="Locally rewrite scene to explicitly satisfy required constraints.",
                    )
                )

        for forbidden in scene_plan.forbidden_constraints:
            token = forbidden.strip().lower()
            if token and token in text:
                violations.append(
                    ConstraintViolation(
                        rule_type="world_rule_consistency",
                        message=f"Scene text hits forbidden constraint: '{forbidden}'.",
                        severity="error",
                        facet="world_setting",
                        related_ids=[scene_plan.scene_id, f"forbidden:{forbidden}"],
                        repair_hint="Remove or rewrite forbidden content in the current scene.",
                    )
                )

        constraints = (
            list(static_memory.world_setting.world_invariants)
            + list(static_memory.world_setting.physical_rules)
            + list(static_memory.world_setting.magic_rules)
        )
        activation_blob = " ".join(str(x).lower() for x in delta.world_updates.get("world_rule_activations", []))
        for constraint in constraints:
            forbidden_phrase = self._extract_forbidden_phrase(str(constraint).lower())
            if forbidden_phrase and (forbidden_phrase in text or forbidden_phrase in activation_blob):
                violations.append(
                    ConstraintViolation(
                        rule_type="world_rule_consistency",
                        message=f"Potential world invariant violation: '{constraint}'.",
                        severity="error",
                        facet="world_setting",
                        related_ids=[scene_plan.scene_id],
                        repair_hint="Rewrite scene to satisfy world invariants and physical rules.",
                    )
                )

        incoming_world = str(delta.world_updates.get("current_setting_state", "")).strip()
        if incoming_world and not candidate_memory.world_setting.current_setting_state:
            violations.append(
                ConstraintViolation(
                    rule_type="world_rule_consistency",
                    message="World state update is inconsistent with candidate memory state.",
                    severity="warning",
                    facet="world_setting",
                    related_ids=[scene_plan.scene_id],
                    repair_hint="Align world updates with explicit setting transitions in scene text.",
                )
            )
        return violations

    def _classify_violation_scope(self, violation: ConstraintViolation) -> None:
        """Classify violation repair scope for patch planner."""
        message = violation.message.lower()
        rule_type = violation.rule_type.lower()
        violation.patchable = True
        violation.fatal = False
        violation.needs_replan = False

        if "contradictory required/forbidden constraints" in message:
            violation.repair_scope = "plan"
            violation.patchable = False
            violation.fatal = True
            violation.needs_replan = True
            return
        if "non-monotonic" in message:
            violation.repair_scope = "paragraph"
            return
        if "event order regresses" in message:
            violation.repair_scope = "sentence"
            return
        if "world invariant" in message and violation.severity.lower() == "error":
            violation.repair_scope = "paragraph"
            return
        if "planned character" in message and "not represented" in message:
            violation.repair_scope = "sentence"
            return
        if "missing in static memory profiles" in message:
            violation.repair_scope = "plan"
            violation.patchable = False
            violation.fatal = True
            violation.needs_replan = True
            return
        if rule_type == "timeline_consistency" and violation.severity.lower() == "error":
            violation.repair_scope = "paragraph"
            return
        violation.repair_scope = "sentence"

    def _build_violation_anchor(
        self,
        violation: ConstraintViolation,
        scene_plan: ScenePlan,
        sentence_units,
        scene_extraction: SceneExtraction | None,
        idx: int,
    ) -> ViolationAnchor:
        tokens = list(violation.related_ids)
        tokens.extend(scene_plan.required_constraints[:2])
        tokens.extend(scene_plan.forbidden_constraints[:2])
        tokens.extend([violation.rule_type, violation.facet])
        sentence_ids = find_sentence_ids_for_tokens(sentence_units, tokens)
        explicit_sentence_ids: List[str] = []
        inferred_sentence_ids: List[str] = []
        if scene_extraction is not None:
            related_entities = [rid for rid in violation.related_ids if rid.startswith(("char_", "ent_"))]
            related_events = [rid for rid in violation.related_ids if rid.startswith("evt_")]
            for sid, entities in scene_extraction.sentence_entity_mentions.items():
                if set(entities).intersection(set(related_entities)):
                    explicit_sentence_ids.append(sid)
            for sid, events in scene_extraction.sentence_event_mentions.items():
                if set(events).intersection(set(related_events)):
                    explicit_sentence_ids.append(sid)
            for sid, events in scene_extraction.sentence_inferred_event_mentions.items():
                if set(events).intersection(set(related_events)):
                    inferred_sentence_ids.append(sid)
        if violation.rule_type == "timeline_consistency":
            temporal_ids = find_temporal_conflict_sentences(sentence_units, tokens)
            sentence_ids.extend(temporal_ids)
        sentence_ids.extend(explicit_sentence_ids)
        textual_realization = "explicit"
        if not explicit_sentence_ids and inferred_sentence_ids:
            sentence_ids.extend(inferred_sentence_ids)
            textual_realization = "inferred"
        if not sentence_ids and sentence_units:
            sentence_ids = [sentence_units[-1].sentence_id]
        related_entities = [rid for rid in violation.related_ids if rid.startswith(("char_", "ent_"))]
        related_events = [rid for rid in violation.related_ids if rid.startswith("evt_")]
        related_relations = [
            rid for rid in violation.related_ids if rid.startswith(("relation:", "characterization.relations"))
        ]
        temporal_conflicts: List[Dict[str, str]] = []
        if len(related_events) >= 2:
            temporal_conflicts.append({"event_a": related_events[0], "event_b": related_events[1], "type": "order_conflict"})
        return build_anchor(
            anchor_id=f"anchor_{scene_plan.scene_id}_{idx:03d}",
            rule_type=violation.rule_type,
            severity=violation.severity,
            sentence_ids=sentence_ids,
            sentences=sentence_units,
            related_entity_ids=related_entities,
            related_event_ids=related_events,
            related_relation_ids=related_relations,
            temporal_conflicts=temporal_conflicts,
            textual_realization=textual_realization,
            grounding_confidence=0.84 if explicit_sentence_ids else (0.62 if inferred_sentence_ids else 0.42),
            confidence_score=0.84 if explicit_sentence_ids else (0.62 if inferred_sentence_ids else 0.42),
            source_type=("explicit" if explicit_sentence_ids else ("inferred" if inferred_sentence_ids else "heuristic")),
            notes=[violation.message],
        )

    def _repair_strategy(
        self,
        is_consistent: bool,
        needs_replan: bool,
        fatal: bool,
        violations: List[ConstraintViolation],
    ) -> str:
        if is_consistent:
            return "update_memory_only"
        if needs_replan:
            return "needs_replan"
        scopes = {v.repair_scope for v in violations}
        if "sentence" in scopes and scopes == {"sentence"}:
            return "patch_sentence"
        if "paragraph" in scopes:
            return "patch_paragraph"
        if fatal or "scene" in scopes:
            return "regenerate_chunk"
        return "patch_sentence"

    def _build_canonical_entity_table(
        self,
        static_memory: StaticMemory,
        candidate_memory: DynamicMemory,
    ) -> Dict[str, Dict[str, Any]]:
        table: Dict[str, Dict[str, Any]] = {}
        for entity_id, profile in static_memory.characterization.character_profiles.items():
            aliases = [str(x).strip() for x in list(profile.aliases or []) if str(x).strip()]
            if profile.canonical_name and profile.canonical_name not in aliases:
                aliases.insert(0, profile.canonical_name)
            dedup_aliases: List[str] = []
            seen = set()
            for item in aliases:
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup_aliases.append(item)
            state = candidate_memory.characterization.entity_store.get(entity_id)
            table[str(entity_id)] = {
                "canonical": str(profile.canonical_name),
                "aliases": dedup_aliases,
                "role": str(profile.role),
                "current_name": str(state.name) if state is not None else "",
            }
        return table

    def _collect_neural_structured_findings(
        self,
        *,
        scene_text: str,
        scene_plan: ScenePlan,
        canonical_entity_table: Dict[str, Dict[str, Any]],
    ) -> List[NeuralFinding]:
        findings: List[NeuralFinding] = []
        lowered = str(scene_text or "").lower()
        if not lowered:
            return findings

        # Lightweight neural-side alias drift signal: same entity appears with multiple aliases in one scene.
        for entity_id, payload in canonical_entity_table.items():
            aliases = [str(x).strip() for x in list(payload.get("aliases", []) or []) if str(x).strip()]
            canonical = str(payload.get("canonical", "")).strip()
            if len(aliases) <= 1:
                continue
            matched: List[str] = []
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias.lower())}\b", lowered):
                    matched.append(alias)
            unique = sorted({m.lower(): m for m in matched}.values(), key=lambda x: x.lower())
            if len(unique) <= 1:
                continue
            findings.append(
                NeuralFinding(
                    checker_name="neural_alias_guard",
                    message=(
                        f"Potential alias drift for {entity_id}: {unique}. "
                        f"Keep one canonical identity naming path per scene."
                    ),
                    score=0.62,
                    conflict_type="alias_drift",
                    facet="factual_detail",
                    related_ids=[str(entity_id), str(scene_plan.scene_id)],
                    evidence_spans=unique[:4],
                    confidence=0.72,
                )
            )

        if "[inconsistent]" in lowered:
            findings.append(
                NeuralFinding(
                    checker_name="neural_marker_guard",
                    message="Scene includes explicit inconsistency marker.",
                    score=0.2,
                    conflict_type="marker_inconsistency",
                    facet="timeline_plot",
                    related_ids=[str(scene_plan.scene_id)],
                    evidence_spans=["[inconsistent]"],
                    confidence=0.9,
                )
            )
        return findings

    def _neural_findings_to_violations(self, findings: List[NeuralFinding]) -> List[ConstraintViolation]:
        out: List[ConstraintViolation] = []
        for item in findings:
            conflict_type = str(item.conflict_type or "").strip().lower()
            if not conflict_type:
                continue
            severity = "warning"
            if conflict_type == "marker_inconsistency":
                severity = "error"
            elif float(item.confidence or 0.0) >= 0.85 and (item.score is None or float(item.score) <= 0.35):
                severity = "error"
            out.append(
                ConstraintViolation(
                    rule_type=f"neural_{conflict_type}",
                    message=str(item.message),
                    severity=severity,
                    facet=str(item.facet or "factual_detail"),
                    related_ids=[str(x) for x in list(item.related_ids or [])],
                    repair_hint="Apply minimal local edit to remove neural-side contradiction signal.",
                    repair_scope="sentence",
                    patchable=True,
                    fatal=False,
                    needs_replan=False,
                    context={
                        "source": "neural_checker",
                        "checker_name": str(item.checker_name),
                        "confidence": float(item.confidence or 0.0),
                        "score": float(item.score if item.score is not None else 0.0),
                        "evidence_spans": list(item.evidence_spans or []),
                    },
                )
            )
        return out

    def _dual_consistency_decision(
        self,
        *,
        symbolic_findings: List[SymbolicFinding],
        neural_findings: List[NeuralFinding],
        blocking_count: int,
    ) -> tuple[str, Dict[str, Any]]:
        symbolic_error_count = int(len([f for f in symbolic_findings if f.severity == "error"]))
        symbolic_warning_count = int(len([f for f in symbolic_findings if f.severity != "error"]))
        neural_structured_count = int(len([f for f in neural_findings if f.conflict_type]))
        dual_confirmed = bool(symbolic_error_count > 0 and neural_structured_count > 0)
        if dual_confirmed:
            decision = "must_repair_dual_confirmed"
        elif symbolic_error_count > 0:
            decision = "must_repair_symbolic_only"
        elif neural_structured_count > 0:
            decision = "neural_risk_watch"
        else:
            decision = "accept_or_minimal_edit"
        return decision, {
            "symbolic_error_count": symbolic_error_count,
            "symbolic_warning_count": symbolic_warning_count,
            "symbolic_total_count": int(len(symbolic_findings)),
            "neural_structured_count": neural_structured_count,
            "dual_confirmed": bool(dual_confirmed),
            "must_repair": bool(blocking_count > 0 or dual_confirmed),
        }

    def _collect_conflict_slots(self, violations: List[ConstraintViolation]) -> List[str]:
        slots: List[str] = []
        for violation in violations:
            slots.extend(str(x) for x in violation.related_ids)
        return sorted(set(slots))

    def _extract_forbidden_phrase(self, rule_text: str) -> str:
        match = re.search(r"(?:cannot|must not|forbidden|never)\s+([a-z0-9_\- ]{2,40})", rule_text)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _is_blocking(self, violation: ConstraintViolation) -> bool:
        return violation.severity.lower() == "error"

    def _constraint_satisfied(self, constraint: str, scene_text: str, delta: MemoryDelta) -> bool:
        lowered_text = scene_text.lower()
        if constraint.lower() in lowered_text:
            return True

        token_candidates = re.findall(r"[a-zA-Z0-9_]{4,}", constraint.lower())
        tokens = [t for t in token_candidates if t not in self._CONSTRAINT_STOPWORDS]
        if not tokens:
            return True

        evidence_blob = " ".join(
            [
                lowered_text,
                " ".join(str(span).lower() for span in delta.raw_evidence_spans),
                " ".join(str(v).lower() for v in delta.world_updates.values()),
                " ".join(str(v).lower() for v in delta.new_facts.values()),
            ]
        )
        token_hits = sum(1 for token in tokens if token in evidence_blob)
        return token_hits >= min(2, len(tokens))
