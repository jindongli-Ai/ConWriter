#!/usr/bin/env python3
"""Run ConWriter on one prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ConWriter.config.schema import ConWriterConfig
from ConWriter.pipeline.engine import ConWriterEngine
from ConWriter.utils.types import ConWriterPromptSample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ConWriter on one prompt.")
    parser.add_argument("prompt", help="Story-writing prompt.")
    parser.add_argument(
        "--config",
        default="ConWriter/config/default.yaml",
        help="YAML configuration path relative to the repository root.",
    )
    args = parser.parse_args()

    config = ConWriterConfig.from_yaml(PROJECT_ROOT / args.config).resolve_paths(PROJECT_ROOT)
    sample = ConWriterPromptSample(
        prompt_id="example",
        prompt=args.prompt,
        language="en",
        task_type="generation",
    )
    _, output = ConWriterEngine(config).run_single(sample)
    print(output.generated_story)


if __name__ == "__main__":
    main()
