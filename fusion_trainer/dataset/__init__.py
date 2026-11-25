"""Dataset module — data loading, transforms, format conversion.

Fusion-Trainer does NOT manage the model hub. It only reads data
from file paths. The fusion-model-hub integration is purely a
path convention: {model_hub_path}/datasets/{name}.jsonl.
"""

from .formats import (
    SCHEMA_DPO_PAIR,
    SCHEMA_GRPO_PROMPT,
    SCHEMA_INPUT_OUTPUT,
    SCHEMA_MESSAGES,
    SCHEMA_SHAREGPT,
    VALID_SCHEMAS,
    copy_message,
    detect_dataset_schema,
    detect_format,
    load_dataset_file,
    save_dataset_file,
    serialize_chatml,
    sharegpt_to_messages,
    validate_chat_messages,
    validate_format,
)
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
    "SCHEMA_DPO_PAIR",
    "SCHEMA_GRPO_PROMPT",
    "SCHEMA_INPUT_OUTPUT",
    "SCHEMA_MESSAGES",
    "SCHEMA_SHAREGPT",
    "VALID_SCHEMAS",
    "detect_dataset_schema",
    "detect_format",
    "load_dataset_file",
    "save_dataset_file",
    "serialize_chatml",
    "copy_message",
    "sharegpt_to_messages",
    "validate_chat_messages",
    "validate_format",
    "clean_dataset",
    "deduplicate",
    "filter_by_length",
    "split_dataset",
    "to_chat_format",
    "truncate_fields",
]
