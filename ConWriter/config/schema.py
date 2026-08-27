"""Dataclass-based configuration schema for ConWriter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(slots=True)
class PathConfig:
    """Filesystem paths used by ConWriter."""

    prompt_parquet_path: str
    outputs_dir: str
    logs_dir: str
    records_jsonl: str
    judge_ready_parquet: str


@dataclass(slots=True)
class ChunkGenerationConfig:
    """Generation policy for story chunks."""

    max_steps: int = 1
    model_name: str = "ConWriter"
    temperature: float = 0.7
    max_chunk_tokens: int = 512
    target_chunk_words: int = 120


@dataclass(slots=True)
class LLMConfig:
    """Single-entry API configuration for model-backed generation.

    This block is intentionally centralized so users only need to modify one
    place when switching platform/model/key setup.
    """

    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 1
    retry_backoff_seconds: float = 5.0
    request_temperature: Optional[float] = None
    request_max_tokens: Optional[int] = None
    request_logprobs: bool = True
    request_top_logprobs: int = 5
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_request_body: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = (
        "You are ConWriter generation model. Continue the story with coherent, "
        "constraint-consistent narrative."
    )
    fallback_to_stub: bool = True


@dataclass(slots=True)
class MemoryConfig:
    """Memory tracking parameters."""

    seed_characters_into_dynamic: bool = True
    max_history_entries: int = 200
    enable_graph_summary: bool = True


@dataclass(slots=True)
class IncrementalConfig:
    """Scene-level incremental generation configuration."""

    enabled: bool = True
    num_chapters: int = 2
    scenes_per_chapter: int = 2
    scene_target_words: int = 220
    max_repair_rounds: int = 2
    strict_precheck: bool = True
    stop_on_failed_scene: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Pipeline routing controls."""

    mode: str = "incremental_v2"
    allow_legacy_fallback: bool = False


@dataclass(slots=True)
class PlanningConfig:
    """Planning-time controls for hierarchical incremental generation."""

    auto_adjust_scene_count: bool = True
    target_story_words: int = 0
    min_scene_words: int = 180


@dataclass(slots=True)
class GenerationControlConfig:
    """Generation/repair controls for per-scene loop."""

    max_scene_regen_rounds: int = 1
    max_scene_repair_rounds: int = 2
    max_sentence_patch_rounds: int = 3
    max_paragraph_patch_rounds: int = 1
    max_patch_targets_per_round: int = 3
    allow_neighbor_adjustment: bool = True
    repair_preserve_unchanged: bool = True
    enable_replan_hook: bool = True
    local_replan_window_scenes: int = 2
    max_local_replans_per_story: int = 2
    oscillation_window: int = 4
    oscillation_threshold: int = 2
    max_patch_rounds_per_scene: int = 8
    enable_repair_metrics: bool = True
    require_textual_event_realization: bool = True
    patch_lookahead_top_k: int = 3
    low_confidence_anchor_threshold: float = 0.45
    min_stability_score: float = 0.55
    enable_facet_adaptive_repair: bool = True
    adaptive_replan_error_threshold: int = 3
    adaptive_replan_round_threshold: int = 2
    # Prefer patch-first behavior: exhaust sentence-level repair before any replan/regen.
    prefer_sentence_repair_first: bool = True
    # When patch-first is enabled, allow replan/regen only for fatal violations.
    regen_only_on_fatal: bool = True


@dataclass(slots=True)
class SymbolicSwitches:
    """Switches for symbolic checker sub-modules."""

    enable_temporal_rules: bool = True
    enable_entity_rules: bool = True
    enable_causal_rules: bool = True


@dataclass(slots=True)
class ConsistencyConfig:
    """Consistency checking policy."""

    neural_checker_enabled: bool = True
    max_revision_rounds: int = 1
    reject_on_violation: bool = True
    symbolic: SymbolicSwitches = field(default_factory=SymbolicSwitches)


@dataclass(slots=True)
class LoggingConfig:
    """Logging behavior."""

    level: str = "INFO"
    file_name: str = "ConWriter.log"


@dataclass(slots=True)
class TraceConfig:
    """Trace export options for reproducible scene-level debugging."""

    enabled: bool = True
    output_dir: str = "experiments/outputs/traces"
    export_prompt_text: bool = True
    export_metrics: bool = True


@dataclass(slots=True)
class VariantConfig:
    """Unified ablation/variant switches for experiment framework."""

    variant_name: str = "full_default"
    generation_layer: str = "plan_memory_patch"
    repair_granularity: str = "full_patch_pipeline"
    optimization_layer: str = "global_aware_patch_selection"
    constraint_layer: str = "weighted_constraints"
    replan_layer: str = "local_replan_weighted"
    preservation_layer: str = "preservation_check_on"
    grounding_layer: str = "confidence_aware_grounding"
    entropy_layer: str = "entropy_monitor_off"


@dataclass(slots=True)
class EntropyMonitorConfig:
    """Lightweight entropy-aware risk monitor (ERM) controls."""

    enable_entropy_monitor: bool = False
    entropy_mode: str = "auto"
    entropy_monitor_mode: str = "sentence_block"
    model_token_high_uncertainty_threshold: float = 1.5
    sentence_spike_density_threshold: float = 0.34
    risk_low_threshold: float = 0.58
    risk_high_threshold: float = 0.92
    entropy_spike_weight: float = 0.35
    entropy_jump_weight: float = 0.45
    constraint_sensitivity_weight: float = 0.65
    entropy_high_risk_escalate_validation: bool = True
    entropy_high_risk_escalate_patch_scope: bool = True
    entropy_high_risk_replan_trigger: bool = True
    validation_escalation_extra_checks: int = 1
    replan_min_patch_rounds: int = 2
    delta_uncertainty_uptrend_threshold: float = 0.08
    sentence_uncertainty_variance_threshold: float = 0.05
    round_uncertainty_trend_threshold: float = 0.12
    joint_weight_template: str = "balanced"
    joint_uncertainty_weight: float = 0.45
    joint_symbolic_weight: float = 0.4
    joint_memory_weight: float = 0.15
    joint_validation_threshold: float = 0.35
    joint_validation_min_symbolic_pressure: float = 0.24
    joint_validation_min_uncertainty_trend: float = 0.06
    joint_validation_max_low_violation_guard: int = 1
    joint_validation_require_dual_signal: bool = True
    joint_patch_threshold: float = 0.52
    joint_patch_min_symbolic_pressure: float = 0.32
    joint_patch_min_local_failure_signal: float = 0.25
    joint_patch_escalation_threshold: float = 0.68
    joint_replan_threshold: float = 0.82
    joint_replan_min_persistent_risk_steps: int = 2
    joint_replan_requires_patch_failure: bool = True
    delayed_gain_min_uncertainty_drop: float = 0.03
    delayed_gain_min_joint_risk_drop: float = 0.04


@dataclass(slots=True)
class ExperienceBankConfig:
    """Model-specific lightweight experience bank controls."""

    enabled: bool = True
    storage_path: str = "experiments/evaluations/experience_bank/model_experience_bank.json"
    max_entries_per_model: int = 120
    max_retrieved_scene_items: int = 3
    max_retrieved_retry_items: int = 3


@dataclass(slots=True)
class LengthControlConfig:
    """Length-aware lightweight compensation and progress warnings."""

    enabled: bool = True
    compensation_cap: float = 1.6
    default_compensation_factor: float = 1.2
    model_compensation_factors: Dict[str, float] = field(
        default_factory=lambda: {
            "gpt-5": 1.08,
            "gemini": 1.16,
            "qwen": 1.30,
            "deepseek": 1.35,
        }
    )
    progress_warning_ratio: float = 0.72
    under_generation_completion_ratio: float = 0.85
    max_guidance_lines: int = 3
    premature_closure_cues: List[str] = field(
        default_factory=lambda: [
            "in the end",
            "in conclusion",
            "to sum up",
            "finally",
            "最后",
            "最终",
            "总之",
        ]
    )
    online_factor_update: bool = True
    online_factor_update_rate: float = 0.12


@dataclass(slots=True)
class DiagnosticsConfig:
    """Export and aggregation settings for experiment diagnosis artifacts."""

    output_dir: str = "experiments/evaluations/diagnostics"
    export_scene_jsonl: bool = True
    export_story_json: bool = True
    export_summary_csv: bool = True


@dataclass(slots=True)
class ConWriterConfig:
    """Top-level ConWriter configuration object."""

    experiment_name: str
    paths: PathConfig
    chunk_generation: ChunkGenerationConfig = field(default_factory=ChunkGenerationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    generation_controls: GenerationControlConfig = field(default_factory=GenerationControlConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    variant: VariantConfig = field(default_factory=VariantConfig)
    entropy_monitor: EntropyMonitorConfig = field(default_factory=EntropyMonitorConfig)
    experience_bank: ExperienceBankConfig = field(default_factory=ExperienceBankConfig)
    length_control: LengthControlConfig = field(default_factory=LengthControlConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    random_seed: int = 42

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ConWriterConfig":
        """Parse nested dictionary into typed dataclasses."""
        paths = PathConfig(**raw["paths"])
        chunk_generation = ChunkGenerationConfig(**raw.get("chunk_generation", {}))
        llm = LLMConfig(**raw.get("llm", {}))
        memory = MemoryConfig(**raw.get("memory", {}))
        pipeline = PipelineConfig(**raw.get("pipeline", {}))
        planning = PlanningConfig(**raw.get("planning", {}))
        generation_controls = GenerationControlConfig(**raw.get("generation_controls", {}))
        incremental = IncrementalConfig(**raw.get("incremental", {}))
        symbolic = SymbolicSwitches(**raw.get("consistency", {}).get("symbolic", {}))
        consistency_payload = dict(raw.get("consistency", {}))
        consistency_payload["symbolic"] = symbolic
        consistency = ConsistencyConfig(**consistency_payload)
        logging_cfg = LoggingConfig(**raw.get("logging", {}))
        trace = TraceConfig(**raw.get("trace", {}))
        variant = VariantConfig(**raw.get("variant", {}))
        entropy_monitor = EntropyMonitorConfig(**raw.get("entropy_monitor", {}))
        experience_bank = ExperienceBankConfig(**raw.get("experience_bank", {}))
        length_control = LengthControlConfig(**raw.get("length_control", {}))
        diagnostics = DiagnosticsConfig(**raw.get("diagnostics", {}))
        return cls(
            experiment_name=raw["experiment_name"],
            paths=paths,
            chunk_generation=chunk_generation,
            llm=llm,
            memory=memory,
            pipeline=pipeline,
            planning=planning,
            generation_controls=generation_controls,
            incremental=incremental,
            consistency=consistency,
            logging=logging_cfg,
            trace=trace,
            variant=variant,
            entropy_monitor=entropy_monitor,
            experience_bank=experience_bank,
            length_control=length_control,
            diagnostics=diagnostics,
            random_seed=raw.get("random_seed", 42),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConWriterConfig":
        """Load config from YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        return cls.from_dict(payload)

    def resolve_paths(self, project_root: str | Path) -> "ConWriterConfig":
        """Resolve all relative paths against project root."""
        root = Path(project_root)
        self.paths = PathConfig(
            prompt_parquet_path=str(_resolve(root, self.paths.prompt_parquet_path)),
            outputs_dir=str(_resolve(root, self.paths.outputs_dir)),
            logs_dir=str(_resolve(root, self.paths.logs_dir)),
            records_jsonl=str(_resolve(root, self.paths.records_jsonl)),
            judge_ready_parquet=str(_resolve(root, self.paths.judge_ready_parquet)),
        )
        self.trace = TraceConfig(
            enabled=self.trace.enabled,
            output_dir=str(_resolve(root, self.trace.output_dir)),
            export_prompt_text=self.trace.export_prompt_text,
            export_metrics=self.trace.export_metrics,
        )
        self.experience_bank = ExperienceBankConfig(
            enabled=self.experience_bank.enabled,
            storage_path=str(_resolve(root, self.experience_bank.storage_path)),
            max_entries_per_model=self.experience_bank.max_entries_per_model,
            max_retrieved_scene_items=self.experience_bank.max_retrieved_scene_items,
            max_retrieved_retry_items=self.experience_bank.max_retrieved_retry_items,
        )
        self.diagnostics = DiagnosticsConfig(
            output_dir=str(_resolve(root, self.diagnostics.output_dir)),
            export_scene_jsonl=self.diagnostics.export_scene_jsonl,
            export_story_json=self.diagnostics.export_story_json,
            export_summary_csv=self.diagnostics.export_summary_csv,
        )
        return self

    def default_log_file(self) -> str:
        """Return resolved log file path."""
        return str(Path(self.paths.logs_dir) / self.logging.file_name)


def _resolve(root: Path, value: Optional[str]) -> Path:
    if not value:
        return root
    path = Path(value)
    return path if path.is_absolute() else root / path
