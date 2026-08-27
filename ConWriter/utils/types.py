"""Core data structures shared across ConWriter modules.

This module defines the formal dual-memory + five-facet schema used by the
pipeline scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

FacetName = Literal[
    "characterization",
    "factual_detail",
    "narrative_style",
    "timeline_plot",
    "world_setting",
]

RepairStrategyName = Literal[
    "minimal_edit",
    "patch_sentence",
    "patch_paragraph",
    "regenerate_chunk",
    "update_memory_only",
    "needs_replan",
]

FIVE_FACETS: List[str] = [
    "characterization",
    "factual_detail",
    "narrative_style",
    "timeline_plot",
    "world_setting",
]


@dataclass(slots=True)
class CharacterProfile:
    """Static character profile extracted from prompt-level constraints."""

    character_id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    role: str = "unknown"
    traits: Dict[str, Any] = field(default_factory=dict)
    background: str = ""
    constraints: List[str] = field(default_factory=list)


@dataclass(slots=True)
class EntityState:
    """Dynamic state of an entity during generation."""

    entity_id: str
    name: str
    status: str = "unknown"
    location: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, str] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    abilities: List[str] = field(default_factory=list)
    last_updated_step: int = -1


@dataclass(slots=True)
class StoryEvent:
    """Atomic event representation for timeline and causal tracking."""

    event_id: str
    description: str
    order: int
    timestamp: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    location: str = "unknown"
    temporal_links: List[str] = field(default_factory=list)
    causal_links: List[str] = field(default_factory=list)
    evidence_chunk_id: Optional[str] = None


@dataclass(slots=True)
class PlotThread:
    """Represents one unresolved or resolved plot thread."""

    thread_id: str
    title: str
    status: str = "active"
    goal: str = ""
    related_events: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StorySpec:
    """Prompt-level story specification (benchmark metadata + high-level intent)."""

    prompt_id: str
    raw_prompt: str
    language: str = "en"
    task_type: str = "continuation"
    theme: str = ""
    genre: str = ""
    goal: str = ""
    target_length_hint: str = ""


@dataclass(slots=True)
class StyleConstraintMemory:
    """Static style constraints that should hold globally."""

    narrative_pov: str = "unspecified"
    tense_style: str = "unspecified"
    tone: str = "unspecified"
    register: str = "unspecified"
    style_constraints: List[str] = field(default_factory=list)
    forbidden_style_shifts: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CharacterSpecMemory:
    """Static characterization constraints from prompt."""

    character_profiles: Dict[str, CharacterProfile] = field(default_factory=dict)
    initial_relations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    known_abilities: Dict[str, List[str]] = field(default_factory=dict)
    identity_constraints: List[str] = field(default_factory=list)
    knowledge_constraints: List[str] = field(default_factory=list)


@dataclass(slots=True)
class FactConstraintMemory:
    """Static fact-level constraints from prompt instructions."""

    initial_facts: List[str] = field(default_factory=list)
    name_map: Dict[str, str] = field(default_factory=dict)
    numeric_facts: Dict[str, float] = field(default_factory=dict)
    object_facts: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WorldRuleMemory:
    """Static world-setting constraints and invariants."""

    setting_description: str = ""
    location_constraints: List[str] = field(default_factory=list)
    social_norms: List[str] = field(default_factory=list)
    physical_rules: List[str] = field(default_factory=list)
    magic_rules: List[str] = field(default_factory=list)
    world_invariants: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PlotConstraintMemory:
    """Static plot-level constraints and forbidden outcomes."""

    initial_plot_setup: str = ""
    global_story_goal: str = ""
    required_plot_points: List[str] = field(default_factory=list)
    forbidden_plot_outcomes: List[str] = field(default_factory=list)
    core_conflicts: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StaticMemory:
    """Dual-memory static side: immutable prompt-derived constraints."""

    story_spec: StorySpec
    style: StyleConstraintMemory = field(default_factory=StyleConstraintMemory)
    characterization: CharacterSpecMemory = field(default_factory=CharacterSpecMemory)
    factual_detail: FactConstraintMemory = field(default_factory=FactConstraintMemory)
    world_setting: WorldRuleMemory = field(default_factory=WorldRuleMemory)
    timeline_plot: PlotConstraintMemory = field(default_factory=PlotConstraintMemory)


@dataclass(slots=True)
class StoryChunk:
    """Generated story chunk plus lightweight extraction traces."""

    chunk_id: str
    text: str
    position: int
    accepted: bool = False
    revised: bool = False
    planner_goal: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    extracted_events: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SentenceUnit:
    """Stable sentence unit with character-span coordinates."""

    sentence_id: str
    text: str
    char_start: int
    char_end: int
    paragraph_id: int = 0
    source_scene_id: str = ""


@dataclass(slots=True)
class ViolationAnchor:
    """Localized anchor for one consistency violation."""

    anchor_id: str
    sentence_ids: List[str] = field(default_factory=list)
    char_spans: List[Dict[str, int]] = field(default_factory=list)
    rule_type: str = ""
    severity: str = "warning"
    related_entity_ids: List[str] = field(default_factory=list)
    related_event_ids: List[str] = field(default_factory=list)
    related_relation_ids: List[str] = field(default_factory=list)
    temporal_conflicts: List[Dict[str, str]] = field(default_factory=list)
    textual_realization: str = "explicit"
    grounding_confidence: float = 0.5
    confidence_score: float = 0.5
    source_type: str = "heuristic"
    constraint_tier: int = 3
    constraint_weight: float = 1.0
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SentencePatch:
    """One local patch operation over sentence units."""

    patch_id: str
    op_type: str
    target_sentence_ids: List[str] = field(default_factory=list)
    new_text: str = ""
    rationale: str = ""
    linked_violation_ids: List[str] = field(default_factory=list)
    constraints_to_satisfy: List[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchPlan:
    """Structured plan for local patch-based repair."""

    plan_id: str
    target_sentence_ids: List[str] = field(default_factory=list)
    protected_sentence_ids: List[str] = field(default_factory=list)
    patch_sequence: List[SentencePatch] = field(default_factory=list)
    fallback_level: str = "scene"
    requires_neighbor_adjustment: bool = False
    expected_fixed_violations: List[str] = field(default_factory=list)
    needs_replan: bool = False
    candidate_score: float = 0.0
    expected_violation_reduction: float = 0.0
    expected_preservation_cost: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    chosen_rationale: str = ""
    trajectory_length: int = 1
    deferred_low_confidence_violations: List[str] = field(default_factory=list)
    future_conflict_penalty: float = 0.0
    weighted_remaining_violation_score: float = 0.0
    critical_constraints_preserved: List[str] = field(default_factory=list)
    deferred_constraints: List[str] = field(default_factory=list)
    global_objective_breakdown: Dict[str, float] = field(default_factory=dict)
    impacted_future_scene_ids: List[str] = field(default_factory=list)
    critical_future_constraints_at_risk: List[str] = field(default_factory=list)
    violation_context_sentence_ids: List[str] = field(default_factory=list)
    violation_context_constraint_ids: List[str] = field(default_factory=list)
    entropy_context_sentence_ids: List[str] = field(default_factory=list)
    patch_target_hits_violation_context: bool = False
    patch_target_hits_entropy_context: bool = False
    patch_target_joint_alignment_score: float = 0.0
    patch_alignment_score_breakdown: Dict[str, float] = field(default_factory=dict)
    patch_before_transition_violation_count: int = 0
    patch_after_transition_violation_count: int = 0
    patch_before_constraint_violation_count: int = 0
    patch_after_constraint_violation_count: int = 0
    patch_before_violation_count: int = 0
    patch_after_violation_count: int = 0
    patch_before_symbolic_state_proxy: float = 0.0
    patch_after_symbolic_state_proxy: float = 0.0
    patch_before_uncertainty: float = 0.0
    patch_after_uncertainty: float = 0.0
    patch_rewrites_key_conflict_span: bool = False
    patch_changes_symbolic_state_proxy: bool = False
    patch_reduces_transition_violations: bool = False
    patch_reduces_constraint_violations: bool = False
    patch_reduces_uncertainty: bool = False
    patch_effectiveness_label: str = "unknown"
    patch_no_gain_reason: str = ""
    patch_execution_id: str = ""
    patch_execution_status: str = "not_applied"
    patch_execution_round: int = 0
    patch_execution_scope: str = ""
    patch_execution_applied: bool = False
    patch_execution_skipped_reason: str = ""
    rewrite_conflict_type: str = "unknown"
    rewrite_target_scope: str = "sentence"
    rewrite_hits_required_state_change: bool = False
    rewrite_removes_conflicting_state: bool = False
    rewrite_preserves_non_conflict_content: bool = True
    rewrite_targets_execution_spec_conflict: bool = False
    rewrite_targets_required_state_change: bool = False
    rewrite_targets_transition_conflict: bool = False
    rewrite_targets_operator_post_state_conflict: bool = False
    rewrite_operator_required_post_states: List[str] = field(default_factory=list)
    rewrite_memory_binding_mode: str = "normal_binding"
    rewrite_generation_control_mode: str = "normal_generation"
    rewrite_binding_decision_reasons: List[str] = field(default_factory=list)
    rewrite_strengthened_memory_blocks: List[str] = field(default_factory=list)
    rewrite_strengthened_constraints: List[str] = field(default_factory=list)
    rewrite_generation_control_context: Dict[str, Any] = field(default_factory=dict)
    rewrite_realizes_required_state_change: bool = False
    rewrite_removes_forbidden_state: bool = False
    rewrite_restores_transition_coherence_proxy: bool = False
    rewrite_realizes_operator_post_state: bool = False
    state_realization_match_type: str = "no_match"
    forbidden_state_removal_match_type: str = "no_match"
    operator_post_state_match_type: str = "no_match"
    canonical_required_states: List[str] = field(default_factory=list)
    canonical_forbidden_states: List[str] = field(default_factory=list)
    canonical_operator_post_states: List[str] = field(default_factory=list)
    grounded_alias_matches: List[Dict[str, Any]] = field(default_factory=list)
    first_pass_required_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    first_pass_forbidden_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    first_pass_operator_post_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    retry_required_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    retry_forbidden_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    retry_operator_post_state_checklist: List[Dict[str, Any]] = field(default_factory=list)
    first_pass_checklist_completion_rate: float = 0.0
    retry_checklist_completion_rate: float = 0.0
    checklist_items_fixed_by_retry: List[Dict[str, Any]] = field(default_factory=list)
    checklist_items_still_unresolved: List[Dict[str, Any]] = field(default_factory=list)
    retry_preserved_satisfied_items: bool = True
    scope_expansion_triggered: bool = False
    original_scope: str = ""
    expanded_scope: str = ""
    expanded_target_sentence_ids: List[str] = field(default_factory=list)
    expanded_local_window: Dict[str, Any] = field(default_factory=dict)
    unresolved_slot_count_before: int = 0
    unresolved_slot_count_after: int = 0
    checklist_items_fixed_by_expansion: List[Dict[str, Any]] = field(default_factory=list)
    still_unresolved_after_expansion: List[Dict[str, Any]] = field(default_factory=list)
    expansion_preserved_satisfied_items: bool = True
    scope_expansion_effective: bool = False
    patch_retry_attempted: bool = False
    patch_retry_scope: str = ""
    patch_retry_reason: str = ""
    patch_retry_conflict_type: str = "unknown"
    patch_retry_effective: bool = False
    patch_retry_realizes_required_state_change: bool = False
    patch_retry_removes_forbidden_state: bool = False
    patch_retry_restores_transition_coherence_proxy: bool = False
    retry_slot_priority_order: List[str] = field(default_factory=list)
    retry_slot_priority_unresolved: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    slot_priority_fix_progress: Dict[str, int] = field(default_factory=dict)
    slot_priority_preserve_result: bool = True
    priority_step_where_failure_remains: str = ""
    slot_type_rebroken_after_retry: str = ""
    step_aware_preservation_guard_enabled: bool = False
    protected_items_snapshot: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    protected_items_preserved_after_retry: bool = True
    protected_items_broken_after_retry: List[Dict[str, Any]] = field(default_factory=list)
    forbidden_reintroduced_after_step: bool = False
    operator_post_state_weakened_after_step: bool = False
    required_state_regressed_after_step: bool = False
    patch_first_pass_effective: bool = False
    patch_still_ineffective_after_retry: bool = False


@dataclass(slots=True)
class SymbolicFinding:
    """Structured finding from symbolic consistency checks."""

    rule_type: str
    message: str
    severity: str = "warning"
    related_ids: List[str] = field(default_factory=list)
    facet: str = ""


@dataclass(slots=True)
class NeuralFinding:
    """Placeholder finding from neural/LLM consistency checks."""

    checker_name: str
    message: str
    score: Optional[float] = None
    conflict_type: str = ""
    facet: str = ""
    related_ids: List[str] = field(default_factory=list)
    evidence_spans: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class MemoryDelta:
    """Standard chunk->memory update representation used across modules."""

    chunk_id: str
    new_entities: List[EntityState] = field(default_factory=list)
    updated_entities: List[EntityState] = field(default_factory=list)
    new_events: List[StoryEvent] = field(default_factory=list)
    new_relations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    new_facts: Dict[str, Any] = field(default_factory=dict)
    style_updates: Dict[str, Any] = field(default_factory=dict)
    plot_updates: Dict[str, Any] = field(default_factory=dict)
    temporal_links: List[Dict[str, str]] = field(default_factory=list)
    causal_links: List[Dict[str, str]] = field(default_factory=list)
    world_updates: Dict[str, Any] = field(default_factory=dict)
    overwritten_states: List[str] = field(default_factory=list)
    raw_evidence_spans: List[str] = field(default_factory=list)
    extraction_notes: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(slots=True)
class ConsistencyReport:
    """Facet-aware consistency report aligned with benchmark taxonomy."""

    is_consistent: bool
    violation_types: List[str] = field(default_factory=list)
    violated_facets: List[str] = field(default_factory=list)
    violated_rules: List[str] = field(default_factory=list)
    conflict_entities: List[str] = field(default_factory=list)
    conflict_events: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    symbolic_findings: List[SymbolicFinding] = field(default_factory=list)
    neural_findings: List[NeuralFinding] = field(default_factory=list)
    suggested_action: str = "accept"
    repair_hints: List[str] = field(default_factory=list)
    conflict_spans: List[str] = field(default_factory=list)
    conflict_slots: List[str] = field(default_factory=list)
    repair_target: str = ""
    repair_strategy: RepairStrategyName = "minimal_edit"
    facet_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    severity: str = "info"
    violation_anchors: List[ViolationAnchor] = field(default_factory=list)
    patch_plan: Optional[PatchPlan] = None
    needs_replan: bool = False
    fatal: bool = False
    canonical_entity_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dual_consistency_decision: str = "symbolic_only"
    dual_consistency_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CharacterMemory:
    """Dynamic characterization facet memory."""

    entity_store: Dict[str, EntityState] = field(default_factory=dict)
    current_states: Dict[str, str] = field(default_factory=dict)
    locations: Dict[str, str] = field(default_factory=dict)
    relations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    knowledge_states: Dict[str, List[str]] = field(default_factory=dict)
    motivation_states: Dict[str, List[str]] = field(default_factory=dict)
    ability_states: Dict[str, List[str]] = field(default_factory=dict)
    character_arcs: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(slots=True)
class FactMemory:
    """Dynamic factual-detail facet memory."""

    stable_facts: List[str] = field(default_factory=list)
    surface_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    numeric_facts: Dict[str, float] = field(default_factory=dict)
    object_states: Dict[str, str] = field(default_factory=dict)
    name_references: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(slots=True)
class StyleMemory:
    """Dynamic narrative-style facet memory."""

    current_pov: str = "unspecified"
    current_tense: str = "unspecified"
    tone_trace: List[str] = field(default_factory=list)
    style_signature: Dict[str, Any] = field(default_factory=dict)
    recent_style_notes: List[str] = field(default_factory=list)
    style_violations_history: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TimelinePlotMemory:
    """Dynamic timeline/plot facet memory."""

    event_timeline: List[StoryEvent] = field(default_factory=list)
    temporal_links: List[Dict[str, str]] = field(default_factory=list)
    causal_links: List[Dict[str, str]] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)
    resolved_goals: List[str] = field(default_factory=list)
    active_plot_threads: List[str] = field(default_factory=list)
    resolved_plot_threads: List[str] = field(default_factory=list)
    unresolved_threads: List[PlotThread] = field(default_factory=list)
    unresolved_foreshadowing: List[str] = field(default_factory=list)
    pending_constraints: List[str] = field(default_factory=list)
    plot_checkpoints: List[str] = field(default_factory=list)
    last_story_position: str = "start"


@dataclass(slots=True)
class WorldMemory:
    """Dynamic world-setting facet memory."""

    current_setting_state: str = "unknown"
    location_states: Dict[str, str] = field(default_factory=dict)
    norm_status: Dict[str, str] = field(default_factory=dict)
    world_rule_activations: List[str] = field(default_factory=list)
    environment_changes: List[str] = field(default_factory=list)
    global_world_facts: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TransitionConstraint:
    """Structured state-bound constraint used in transition reasoning/validation."""

    constraint_id: str
    facet: str
    predicate: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    expected: str = ""
    severity: str = "warning"
    source: str = "state_reasoner"
    constraint_weight: float = 1.0
    constraint_tier: int = 3
    is_hard: bool = False
    weighted_priority_source: str = "state_reasoning"


@dataclass(slots=True)
class TransitionOperator:
    """Forward operator selected before generation for one scene transition."""

    operator_id: str
    operator_type: str
    scene_id: str = ""
    chapter_id: str = ""
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    required_effects: List[str] = field(default_factory=list)
    execution_spec: "OperatorExecutionSpec" = field(default_factory=lambda: OperatorExecutionSpec())
    rationale: str = ""
    priority: int = 0


@dataclass(slots=True)
class OperatorExecutionSpec:
    """Verifiable symbolic execution spec for one selected transition operator."""

    required_entities: List[str] = field(default_factory=list)
    required_events: List[str] = field(default_factory=list)
    required_state_changes: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    require_event_keyword_match: bool = True
    require_actor_coverage: bool = True
    allow_fuzzy_event_match: bool = False


@dataclass(slots=True)
class CharacterStateSnapshot:
    """Compact per-character state in structured generation state."""

    entity_id: str
    name: str = ""
    status: str = "unknown"
    location: str = "unknown"
    last_event_order: int = -1


@dataclass(slots=True)
class WorldStateSnapshot:
    """Compact world snapshot for one structured generation step."""

    current_setting_state: str = "unknown"
    active_locations: List[str] = field(default_factory=list)
    global_facts: List[str] = field(default_factory=list)
    world_rule_activations: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TimelineStateSnapshot:
    """Compact timeline snapshot for one structured generation step."""

    last_event_id: str = ""
    last_event_order: int = -1
    recent_event_ids: List[str] = field(default_factory=list)
    pending_constraints: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StoryState:
    """Structured generation state used by transition-based generation."""

    state_id: str = "state_init"
    step_index: int = 0
    character_states: Dict[str, CharacterStateSnapshot] = field(default_factory=dict)
    relations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    world_state: WorldStateSnapshot = field(default_factory=WorldStateSnapshot)
    timeline: TimelineStateSnapshot = field(default_factory=TimelineStateSnapshot)
    active_constraints: List[TransitionConstraint] = field(default_factory=list)
    derived_facts: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransitionAction:
    """Structured action extracted from a scene to represent State_t -> State_t+1."""

    action_id: str
    scene_id: str
    chapter_id: str
    summary: str
    operator_id: str = ""
    operator_type: str = ""
    declared_preconditions: List[str] = field(default_factory=list)
    expected_postconditions: List[str] = field(default_factory=list)
    realized_postconditions: List[str] = field(default_factory=list)
    execution_spec: OperatorExecutionSpec = field(default_factory=OperatorExecutionSpec)
    actors: List[str] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    event_orders: List[int] = field(default_factory=list)
    relation_updates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    state_changes: List[str] = field(default_factory=list)
    location_from: str = ""
    location_to: str = ""
    required_constraints: List[str] = field(default_factory=list)
    forbidden_constraints: List[str] = field(default_factory=list)
    raw_evidence_spans: List[str] = field(default_factory=list)
    confidence: float = 0.5
    sentence_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StateReasoningResult:
    """Output of state-level reasoning before one scene generation step."""

    state: StoryState
    inferred_constraints: List[TransitionConstraint] = field(default_factory=list)
    candidate_operators: List[TransitionOperator] = field(default_factory=list)
    forbidden_operators: List[str] = field(default_factory=list)
    selected_operator: Optional[TransitionOperator] = None
    execution_spec: OperatorExecutionSpec = field(default_factory=OperatorExecutionSpec)
    allowed_transitions: List[str] = field(default_factory=list)
    forbidden_transitions: List[str] = field(default_factory=list)
    required_state_changes: List[str] = field(default_factory=list)
    propagated_constraints: List[str] = field(default_factory=list)
    conflict_candidates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DynamicMemory:
    """Dual-memory dynamic side: evolving five-facet story state."""

    characterization: CharacterMemory = field(default_factory=CharacterMemory)
    factual_detail: FactMemory = field(default_factory=FactMemory)
    narrative_style: StyleMemory = field(default_factory=StyleMemory)
    timeline_plot: TimelinePlotMemory = field(default_factory=TimelinePlotMemory)
    world_setting: WorldMemory = field(default_factory=WorldMemory)
    current_chapter_id: str = ""
    current_scene_id: str = ""
    current_state: StoryState = field(default_factory=StoryState)
    state_history: List[StoryState] = field(default_factory=list)
    transition_history: List[TransitionAction] = field(default_factory=list)
    inferred_constraints: List[TransitionConstraint] = field(default_factory=list)
    memory_history: List[Dict[str, Any]] = field(default_factory=list)
    accepted_deltas: List[MemoryDelta] = field(default_factory=list)
    rejected_deltas: List[MemoryDelta] = field(default_factory=list)
    revision_records: List[Dict[str, Any]] = field(default_factory=list)
    consistency_reports: List[ConsistencyReport] = field(default_factory=list)


@dataclass(slots=True)
class ConWriterPromptSample:
    """A single prompt sample in ConWriter-friendly format."""

    prompt_id: str
    prompt: str
    language: str = "en"
    task_type: str = "continuation"


@dataclass(slots=True)
class ConWriterOutputRecord:
    """Final export structure produced by ConWriter."""

    id: str
    language: str
    task_type: str
    prompt: str
    model_name: str
    generated_story: str
    generation_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationState:
    """Runtime state for one prompt generation trajectory."""

    current_step: int
    story_chunks: List[StoryChunk]
    static_memory: StaticMemory
    dynamic_memory: DynamicMemory
    last_delta: Optional[MemoryDelta] = None
    last_report: Optional[ConsistencyReport] = None
    is_finished: bool = False
    initial_dynamic_memory: Optional[DynamicMemory] = None
    proposed_deltas: List[MemoryDelta] = field(default_factory=list)
    story_plan: Optional["StoryPlan"] = None


@dataclass(slots=True)
class ScenePlan:
    """Plan unit for one incremental generation scene."""

    scene_id: str
    chapter_id: str
    scene_index: int
    title: str
    objective: str
    key_events: List[str] = field(default_factory=list)
    required_characters: List[str] = field(default_factory=list)
    optional_characters: List[str] = field(default_factory=list)
    involved_characters: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    expected_state_changes: List[str] = field(default_factory=list)
    forbidden_state_changes: List[str] = field(default_factory=list)
    dependency_scenes: List[str] = field(default_factory=list)
    must_keep_constraints: List[str] = field(default_factory=list)
    required_constraints: List[str] = field(default_factory=list)
    forbidden_constraints: List[str] = field(default_factory=list)
    target_words: int = 220


@dataclass(slots=True)
class ChapterPlan:
    """Plan unit for one chapter, composed of multiple scenes."""

    chapter_id: str
    chapter_index: int
    title: str
    objective: str
    scenes: List[ScenePlan] = field(default_factory=list)


@dataclass(slots=True)
class StoryPlan:
    """Hierarchical plan for incremental story generation."""

    premise: str
    global_objective: str
    chapters: List[ChapterPlan] = field(default_factory=list)
    style_directives: List[str] = field(default_factory=list)
    world_invariants: List[str] = field(default_factory=list)

    def iter_scenes(self) -> List[ScenePlan]:
        """Flatten all chapter scenes in planning order."""
        ordered: List[ScenePlan] = []
        for chapter in self.chapters:
            ordered.extend(chapter.scenes)
        return ordered

    def scene_index_map(self) -> Dict[str, int]:
        """Return scene_id -> flattened index map."""
        mapping: Dict[str, int] = {}
        for idx, scene in enumerate(self.iter_scenes()):
            mapping[scene.scene_id] = idx
        return mapping

    def replace_scenes(self, replacements: Dict[str, ScenePlan]) -> None:
        """Replace scenes by scene_id in-place while preserving chapter layout."""
        if not replacements:
            return
        for chapter in self.chapters:
            chapter.scenes = [replacements.get(scene.scene_id, scene) for scene in chapter.scenes]


@dataclass(slots=True)
class LocalReplanResult:
    """Result payload from local replanning over next 1..k scenes."""

    replan_id: str
    triggered: bool = False
    applied: bool = False
    changed_scene_ids: List[str] = field(default_factory=list)
    rationale: str = ""
    impact_summary: Dict[str, Any] = field(default_factory=dict)
    revised_scenes: List[ScenePlan] = field(default_factory=list)


@dataclass(slots=True)
class SceneDraft:
    """Generated draft text for one planned scene."""

    scene_id: str
    chapter_id: str
    text: str
    attempt: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_text: str = ""
    tokens: List[str] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = field(default_factory=list)
    uncertainty_source: str = "none"
    uncertainty_available: bool = False


@dataclass(slots=True)
class TokenUncertainty:
    """Model-side token uncertainty signal extracted from generation response."""

    token: str
    logprob: float = 0.0
    uncertainty: float = 0.0
    entropy: float = 0.0
    top_logprobs: Dict[str, float] = field(default_factory=dict)
    truncated_entropy: bool = False
    source_type: str = "model_logprob"


@dataclass(slots=True)
class EntropyRiskProfile:
    """Uncertainty-aware risk profile for one generated scene."""

    scene_entropy_mean: float = 0.0
    sentence_entropy_scores: Dict[str, float] = field(default_factory=dict)
    block_entropy_scores: List[Dict[str, Any]] = field(default_factory=list)
    entropy_spike_indices: List[int] = field(default_factory=list)
    entropy_jump_indices: List[int] = field(default_factory=list)
    entropy_spike_score: float = 0.0
    entropy_jump_score: float = 0.0
    constraint_sensitive_risk_scores: Dict[str, float] = field(default_factory=dict)
    final_risk_score: float = 0.0
    final_risk_tier: str = "low_risk"
    suggested_action: str = "none"
    linked_sentence_ids: List[str] = field(default_factory=list)
    linked_constraint_ids: List[str] = field(default_factory=list)
    high_risk_sentence_ids: List[str] = field(default_factory=list)
    high_risk_block_ranges: List[Dict[str, int]] = field(default_factory=list)
    source_type: str = "none"
    is_proxy_signal: bool = False
    token_entropy_stats: List[Dict[str, Any]] = field(default_factory=list)
    sentence_uncertainty_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sentence_uncertainty_spikes: List[Dict[str, Any]] = field(default_factory=list)
    constraint_conditioned_uncertainty: Dict[str, float] = field(default_factory=dict)
    critical_constraint_uncertainty_peak: float = 0.0
    critical_constraint_uncertainty_mean: float = 0.0
    local_constraint_uncertainty: Dict[str, float] = field(default_factory=dict)
    local_vs_sentence_uncertainty_gap: Dict[str, float] = field(default_factory=dict)
    scene_uncertainty_mean: float = 0.0
    scene_uncertainty_peak: float = 0.0
    delta_uncertainty: float = 0.0
    sentence_uncertainty_variance: float = 0.0
    round_uncertainty_trend: float = 0.0
    uncertainty_control_score: float = 0.0
    symbolic_pressure_score: float = 0.0
    memory_volatility_score: float = 0.0
    uncertainty_contribution: float = 0.0
    symbolic_contribution: float = 0.0
    memory_contribution: float = 0.0
    joint_risk_score: float = 0.0
    joint_action_selector: str = "do_nothing"
    joint_weight_template_used: str = "balanced"
    joint_local_failure_signal: float = 0.0
    joint_persistent_risk_steps: int = 0
    joint_patch_failure_proxy_score: float = 0.0
    joint_validation_signal_u: float = 0.0
    joint_validation_pre_gate_score: float = 0.0
    joint_patch_pre_gate_score: float = 0.0
    joint_replan_pre_gate_score: float = 0.0
    joint_validation_gate_passed: bool = False
    joint_patch_gate_passed: bool = False
    joint_replan_gate_passed: bool = False
    joint_validation_threshold_reached: bool = False
    joint_validation_dual_signal_satisfied: bool = False
    joint_validation_symbolic_ok: bool = False
    joint_validation_uncertainty_ok: bool = False
    joint_validation_low_violation_guard_blocked: bool = False
    joint_validation_fail_reasons: List[str] = field(default_factory=list)
    joint_patch_threshold_reached: bool = False
    joint_patch_symbolic_ok: bool = False
    joint_patch_local_failure_ok: bool = False
    joint_patch_fail_reasons: List[str] = field(default_factory=list)
    joint_replan_threshold_reached: bool = False
    joint_replan_persistence_ok: bool = False
    joint_replan_patch_failure_ok: bool = False
    joint_replan_requires_patch_failure_blocked: bool = False
    joint_replan_fail_reasons: List[str] = field(default_factory=list)
    uncertainty_mode: str = "none"
    uncertainty_available: bool = False
    uncertainty_truncated: bool = False
    triggered_patch: bool = False
    triggered_validation_mode: str = "standard"
    triggered_validation_budget: int = 1
    triggered_validation: bool = False
    triggered_patch_escalation: bool = False
    triggered_replan: bool = False


@dataclass(slots=True)
class SceneExtraction:
    """Structured extraction result from one generated scene text."""

    scene_id: str
    new_entities: List[EntityState] = field(default_factory=list)
    updated_entities: List[EntityState] = field(default_factory=list)
    new_events: List[StoryEvent] = field(default_factory=list)
    relation_updates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fact_updates: Dict[str, Any] = field(default_factory=dict)
    style_updates: Dict[str, Any] = field(default_factory=dict)
    plot_updates: Dict[str, Any] = field(default_factory=dict)
    temporal_links: List[Dict[str, str]] = field(default_factory=list)
    causal_links: List[Dict[str, str]] = field(default_factory=list)
    world_updates: Dict[str, Any] = field(default_factory=dict)
    overwritten_states: List[str] = field(default_factory=list)
    raw_evidence_spans: List[str] = field(default_factory=list)
    extraction_notes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    scene_text: str = ""
    sentences: List[SentenceUnit] = field(default_factory=list)
    sentence_entity_mentions: Dict[str, List[str]] = field(default_factory=dict)
    sentence_event_mentions: Dict[str, List[str]] = field(default_factory=dict)
    sentence_inferred_event_mentions: Dict[str, List[str]] = field(default_factory=dict)
    sentence_coref_links: Dict[str, List[str]] = field(default_factory=dict)

    def to_memory_delta(self) -> MemoryDelta:
        """Convert scene extraction to generic MemoryDelta."""
        return MemoryDelta(
            chunk_id=self.scene_id,
            new_entities=self.new_entities,
            updated_entities=self.updated_entities,
            new_events=self.new_events,
            new_relations=self.relation_updates,
            new_facts=self.fact_updates,
            style_updates=self.style_updates,
            plot_updates=self.plot_updates,
            temporal_links=self.temporal_links,
            causal_links=self.causal_links,
            world_updates=self.world_updates,
            overwritten_states=self.overwritten_states,
            raw_evidence_spans=self.raw_evidence_spans,
            extraction_notes=self.extraction_notes,
            confidence=self.confidence,
        )


@dataclass(slots=True)
class ConstraintViolation:
    """Constraint violation payload used by local repair."""

    rule_type: str
    message: str
    severity: str = "warning"
    facet: str = ""
    related_ids: List[str] = field(default_factory=list)
    repair_hint: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    anchors: List[ViolationAnchor] = field(default_factory=list)
    repair_scope: str = "sentence"
    patchable: bool = True
    fatal: bool = False
    needs_replan: bool = False
    constraint_weight: float = 1.0
    constraint_tier: int = 3
    is_hard: bool = False
    weighted_cost: float = 1.0
    weighted_priority_source: str = "default"


@dataclass(slots=True)
class WeightedConstraintItem:
    """One weighted constraint item for tiered generation conditioning."""

    text: str
    weight: float
    tier: int
    is_hard: bool = False
    source: str = "unknown"


@dataclass(slots=True)
class FutureConflictEstimate:
    """Predicted downstream conflict risk for current patch trajectory."""

    predicted_future_conflict_penalty: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    impacted_future_scene_ids: List[str] = field(default_factory=list)
    critical_future_constraints_at_risk: List[str] = field(default_factory=list)


@dataclass(slots=True)
class IncrementalState:
    """Runtime state for scene-by-scene incremental generation."""

    plan: StoryPlan
    static_memory: StaticMemory
    dynamic_memory: DynamicMemory
    scene_cursor: int = 0
    accepted_scenes: List[StoryChunk] = field(default_factory=list)
    proposed_deltas: List[MemoryDelta] = field(default_factory=list)
    scene_reports: List[ConsistencyReport] = field(default_factory=list)
    scene_revisions: List[Dict[str, Any]] = field(default_factory=list)
    failed_scenes: List[str] = field(default_factory=list)
    is_finished: bool = False


# Backward-compatible aliases used by some scaffold modules/tests.
Event = StoryEvent
MemoryUpdateProposal = MemoryDelta
