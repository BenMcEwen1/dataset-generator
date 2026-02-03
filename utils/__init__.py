"""
Dataset generator utilities for converting audio datasets to BaseAL format.
"""

from .helpers import convert_for_json
from .embeddings import initialise, generate_embeddings, Embedder
from .segment_labels import (
    SegmentConfig,
    split_metadata_to_segments,
    split_metadata_with_adapter,
    create_labels_csv,
    create_labels_csv_with_adapter,
    filter_by_existing_audio,
)
from .adapters import (
    DatasetAdapter,
    AdapterConfig,
    HSNAdapter,
    ESC50Adapter,
    get_adapter,
    ADAPTERS,
)

__all__ = [
    # Helpers
    "convert_for_json",
    # Embeddings
    "initialise",
    "generate_embeddings",
    "Embedder",
    # Segment labels
    "SegmentConfig",
    "split_metadata_to_segments",
    "split_metadata_with_adapter",
    "create_labels_csv",
    "create_labels_csv_with_adapter",
    "filter_by_existing_audio",
    # Adapters
    "DatasetAdapter",
    "AdapterConfig",
    "HSNAdapter",
    "ESC50Adapter",
    "get_adapter",
    "ADAPTERS",
]
