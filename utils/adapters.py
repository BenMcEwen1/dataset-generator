"""
Dataset adapters for converting various audio dataset formats to BaseAL format.

This module provides a modular adapter system that abstracts dataset-specific
details (metadata format, label structure, event annotations) into a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class AdapterConfig:
    """Configuration for dataset adapters."""
    validation_fraction: float = 0.1
    validation_fold: Optional[int] = None  # If set, use fold-based validation
    random_seed: int = 42
    label_separator: str = ";"
    no_event_label: str = "no_call"
    audio_length: Optional[float] = None  # Fixed length for pre-segmented datasets


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Each adapter handles the specifics of loading and processing a particular
    dataset format, converting it to a standardized internal representation.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass

    @abstractmethod
    def load_metadata(self, metadata_path: Path | str) -> pd.DataFrame:
        """
        Load metadata from the dataset's native format.

        Args:
            metadata_path: Path to the metadata file

        Returns:
            DataFrame with at minimum: filepath, length columns
        """
        pass

    @abstractmethod
    def get_label(self, row: pd.Series) -> str:
        """
        Extract label(s) from a metadata row.

        Args:
            row: A row from the metadata DataFrame

        Returns:
            Label string (potentially multi-label with separator)
        """
        pass

    @abstractmethod
    def get_events(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Extract event onset/offset times from a metadata row.

        Args:
            row: A row from the metadata DataFrame

        Returns:
            Array of [onset, offset] pairs, or None if pre-segmented
        """
        pass

    @abstractmethod
    def get_event_clusters(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Extract event cluster IDs from a metadata row.

        Args:
            row: A row from the metadata DataFrame

        Returns:
            Array of cluster IDs, or None if not applicable
        """
        pass

    @abstractmethod
    def get_extra_metadata(self, row: pd.Series) -> dict:
        """
        Extract additional metadata fields to propagate to segments.

        Args:
            row: A row from the metadata DataFrame

        Returns:
            Dictionary of additional metadata fields
        """
        pass

    def get_audio_length(self, row: pd.Series) -> float:
        """
        Get audio duration for a row.

        Override this for datasets with fixed-length audio.

        Args:
            row: A row from the metadata DataFrame

        Returns:
            Duration in seconds
        """
        if self.config.audio_length is not None:
            return self.config.audio_length
        return row["length"]

    def get_validation_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Create validation split mask.

        Override this for datasets with pre-defined folds.

        Args:
            df: The metadata DataFrame

        Returns:
            Boolean Series indicating validation samples
        """
        np.random.seed(self.config.random_seed)
        return pd.Series(
            np.random.random(len(df)) < self.config.validation_fraction,
            index=df.index
        )

    def is_presegmented(self) -> bool:
        """
        Check if this dataset has pre-segmented audio.

        Pre-segmented datasets have fixed-length clips where the entire
        clip represents the event (no onset/offset annotations).

        Returns:
            True if audio is pre-segmented
        """
        return self.config.audio_length is not None


class HSNAdapter(DatasetAdapter):
    """
    Adapter for the HSN (BirdSet) dataset format.

    HSN format:
    - Metadata: Parquet file
    - Audio: Variable-length .ogg files
    - Labels: ebird_code_multilabel + ebird_code_secondary (multi-label)
    - Events: detected_events array with [onset, offset] pairs
    - Validation: Random fraction-based split
    """

    @property
    def name(self) -> str:
        return "HSN"

    def load_metadata(self, metadata_path: Path | str) -> pd.DataFrame:
        metadata_path = Path(metadata_path)
        df = pd.read_parquet(metadata_path)
        return df

    def get_label(self, row: pd.Series) -> str:
        labels = []

        # Get primary labels from ebird_code_multilabel
        multilabel = row.get("ebird_code_multilabel")
        if multilabel is not None:
            if isinstance(multilabel, np.ndarray):
                labels.extend(multilabel.tolist())
            elif isinstance(multilabel, list):
                labels.extend(multilabel)

        # Get secondary labels
        secondary = row.get("ebird_code_secondary")
        if secondary is not None:
            if isinstance(secondary, np.ndarray):
                labels.extend(secondary.tolist())
            elif isinstance(secondary, list):
                labels.extend(secondary)

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label and label not in seen:
                seen.add(label)
                unique_labels.append(label)

        return self.config.label_separator.join(unique_labels) if unique_labels else row.get("ebird_code", "")

    def get_events(self, row: pd.Series) -> Optional[np.ndarray]:
        events = row.get("detected_events")
        if events is None or (isinstance(events, np.ndarray) and len(events) == 0):
            return None
        return np.array(events) if not isinstance(events, np.ndarray) else events

    def get_event_clusters(self, row: pd.Series) -> Optional[np.ndarray]:
        clusters = row.get("event_cluster")
        if clusters is None or (isinstance(clusters, np.ndarray) and len(clusters) == 0):
            return None
        return np.array(clusters) if not isinstance(clusters, np.ndarray) else clusters

    def get_extra_metadata(self, row: pd.Series) -> dict:
        return {
            "ebird_code_multilabel": row.get("ebird_code_multilabel"),
            "ebird_code_secondary": row.get("ebird_code_secondary"),
            "lat": row.get("lat"),
            "long": row.get("long"),
            "call_type": row.get("call_type"),
            "sex": row.get("sex"),
            "license": row.get("license"),
            "local_time": row.get("local_time"),
            "quality": row.get("quality"),
            "microphone": row.get("microphone"),
            "source": row.get("source"),
            "recordist": row.get("recordist"),
            "order": row.get("order"),
            "species_group": row.get("species_group"),
            "genus": row.get("genus"),
        }


class ESC50Adapter(DatasetAdapter):
    """
    Adapter for the ESC-50 dataset format.

    ESC-50 format:
    - Metadata: CSV file (meta/esc50.csv)
    - Audio: Pre-segmented 5-second .wav files at 44.1kHz
    - Labels: category column (single label, 50 classes)
    - Events: None (entire clip is the event)
    - Validation: Fold-based (fold column, typically fold 5 for validation)
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        if config is None:
            config = AdapterConfig()
        # ESC-50 has fixed 5-second clips
        config.audio_length = 5.0
        # Default to fold 5 for validation if not specified
        if config.validation_fold is None:
            config.validation_fold = 5
        super().__init__(config)

    @property
    def name(self) -> str:
        return "ESC50"

    def load_metadata(self, metadata_path: Path | str) -> pd.DataFrame:
        metadata_path = Path(metadata_path)
        df = pd.read_csv(metadata_path)

        # Add filepath column matching the audio directory structure
        df["filepath"] = df["filename"]

        # Add length column (fixed 5 seconds for ESC-50)
        df["length"] = self.config.audio_length

        return df

    def get_label(self, row: pd.Series) -> str:
        return str(row.get("category", "unknown"))

    def get_events(self, row: pd.Series) -> Optional[np.ndarray]:
        # ESC-50 is pre-segmented - no event annotations
        # The entire clip IS the event
        return None

    def get_event_clusters(self, row: pd.Series) -> Optional[np.ndarray]:
        return None

    def get_extra_metadata(self, row: pd.Series) -> dict:
        return {
            "target": row.get("target"),
            "category": row.get("category"),
            "fold": row.get("fold"),
            "esc10": row.get("esc10"),
            "src_file": row.get("src_file"),
            "take": row.get("take"),
        }

    def get_validation_mask(self, df: pd.DataFrame) -> pd.Series:
        """Use fold-based validation for ESC-50."""
        if self.config.validation_fold is not None and "fold" in df.columns:
            return df["fold"] == self.config.validation_fold
        return super().get_validation_mask(df)


# Registry of available adapters
ADAPTERS = {
    "hsn": HSNAdapter,
    "esc50": ESC50Adapter,
}


def get_adapter(dataset_name: str, config: Optional[AdapterConfig] = None) -> DatasetAdapter:
    """
    Get an adapter instance by dataset name.

    Args:
        dataset_name: Name of the dataset (case-insensitive)
        config: Optional adapter configuration

    Returns:
        Configured DatasetAdapter instance

    Raises:
        ValueError: If dataset name is not recognized
    """
    name_lower = dataset_name.lower()
    if name_lower not in ADAPTERS:
        available = ", ".join(ADAPTERS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    return ADAPTERS[name_lower](config)
