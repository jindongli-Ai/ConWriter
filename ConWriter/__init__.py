"""ConWriter research package.

ConWriter is a modular long-story generation framework centered on
dual-memory tracking and consistency-aware generation.
"""

from ConWriter.config.schema import ConWriterConfig
from ConWriter.pipeline.engine import ConWriterEngine
from ConWriter.pipeline.incremental_writer import IncrementalWriter

__all__ = ["ConWriterConfig", "ConWriterEngine", "IncrementalWriter"]
