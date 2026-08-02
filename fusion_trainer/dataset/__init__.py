"""Dataset module — data loading, transforms, format conversion.

Fusion-Trainer does NOT manage the model hub. It only reads data
from file paths. The fusion-model-hub integration is purely a
path convention: {model_hub_path}/datasets/{name}.jsonl.
"""

from .formats import detect_format, load_dataset_file, save_dataset_file
from .loader import DatasetLoader
from .transforms import (
    clean_dataset,
    deduplicate,
    filter_by_length,
    split_dataset,
    to_chat_format,
    truncate_fields,
)
from .versioning import DatasetVersion

__all__ = [
    "DatasetLoader",
    "DatasetVersion",
    "detect_format",
    "load_dataset_file",
    "save_dataset_file",
    "clean_dataset",
    "deduplicate",
    "filter_by_length",
    "split_dataset",
    "to_chat_format",
    "truncate_fields",
]
