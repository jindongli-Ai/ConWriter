"""Main ConWriter engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from ConWriter.config.schema import ConWriterConfig
from ConWriter.pipeline.incremental_writer import IncrementalWriter
from ConWriter.utils.logging import setup_logger
from ConWriter.utils.types import ConWriterOutputRecord, ConWriterPromptSample, GenerationState


class ConWriterEngine:
    """Orchestrates full ConWriter pipeline in a modular way."""

    def __init__(self, config: ConWriterConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or setup_logger(
            "ConWriter",
            level=config.logging.level,
            log_file=config.default_log_file(),
        )

        self.incremental_writer = IncrementalWriter(config=config, logger=self.logger)

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        project_root: str | Path,
    ) -> "ConWriterEngine":
        """Build engine from config YAML."""
        cfg = ConWriterConfig.from_yaml(config_path).resolve_paths(project_root)
        return cls(cfg)

    def run_single(self, sample: ConWriterPromptSample) -> Tuple[GenerationState, ConWriterOutputRecord]:
        """Run one sample with the scene-level incremental pipeline."""
        mode = (self.config.pipeline.mode or "").strip().lower()
        if mode in {"incremental_v2", "incremental", "scene_incremental"}:
            return self.incremental_writer.run_single(sample)
        raise ValueError(f"Unsupported pipeline mode: {self.config.pipeline.mode!r}")
