"""Dataset format detection and loading."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# D1 cleaned-dataset schemas (architecture/future-mlx-landing-plan.md A2):
#   SFT  : messages-JSONL  -> {"messages": [{"role","content"}, ...]}
#          ShareGPT        -> {"conversations": [{"from","value"}, ...]}
#   DPO  : DPO-pair        -> {"prompt","chosen","rejected"}
#   GRPO : GRPO-prompt     -> {"prompt", ...} (optional "answer"/"solution"/"tests")
SCHEMA_MESSAGES = "messages"
SCHEMA_SHAREGPT = "sharegpt"
SCHEMA_DPO_PAIR = "dpo_pair"
SCHEMA_GRPO_PROMPT = "grpo_prompt"
SCHEMA_INPUT_OUTPUT = "input_output"

VALID_SCHEMAS = (
    SCHEMA_MESSAGES,
    SCHEMA_SHAREGPT,
    SCHEMA_DPO_PAIR,
    SCHEMA_GRPO_PROMPT,
    SCHEMA_INPUT_OUTPUT,
)

# ShareGPT "from" -> standard role mapping
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def detect_format(path: Path) -> str:
    """Detect the format of a dataset file by its extension."""
    suffix = path.suffix.lower()
    mapping = {
        ".jsonl": "jsonl",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
        ".tsv": "tsv",
        ".arrow": "arrow",
    }
    return mapping.get(suffix, "")


def detect_dataset_schema(item: dict[str, Any]) -> str:
    """Detect the D1 cleaned-dataset schema of a single sample.

    Returns one of VALID_SCHEMAS, or "" if the schema is unrecognized.
    Inspection order matters: messages/sharegpt are chat formats, dpo is a
    preference triple, grpo is a single prompt, input_output is the legacy
    flat-text shape.
    """
    if not isinstance(item, dict):
        return ""
    if "messages" in item and isinstance(item["messages"], list):
        return SCHEMA_MESSAGES
    if "conversations" in item and isinstance(item["conversations"], list):
        return SCHEMA_SHAREGPT
    if "chosen" in item and "rejected" in item and "prompt" in item:
        return SCHEMA_DPO_PAIR
    if "prompt" in item and "chosen" not in item and "rejected" not in item:
        # GRPO-prompt: a prompt with optional reference answer / tests, but
        # no chosen/rejected (that would be DPO).
        return SCHEMA_GRPO_PROMPT
    if "input" in item or "output" in item or "text" in item:
        return SCHEMA_INPUT_OUTPUT
    return ""


def validate_format(data: list[dict[str, Any]], schema: str = "") -> tuple[bool, str]:
    """Validate that all samples conform to the given (or detected) schema.

    Returns (ok, message). When schema is empty, the first sample's detected
    schema is used and every sample must match it. Empty data is invalid.
    """
    if not data:
        return False, "dataset is empty"
    detected = schema
    if not detected:
        detected = detect_dataset_schema(data[0])
        if not detected:
            return (
                False,
                f"could not detect schema of first sample: {list(data[0].keys())}",
            )
    if detected not in VALID_SCHEMAS:
        return False, f"unknown schema: {detected}"
    bad = 0
    for i, item in enumerate(data):
        if detect_dataset_schema(item) != detected:
            bad += 1
            if bad <= 3:
                logger.warning(
                    "sample #%d does not match schema %s: keys=%s",
                    i,
                    detected,
                    list(item.keys()),
                )
    if bad > 0:
        return False, f"{bad}/{len(data)} samples do not match schema '{detected}'"
    logger.info("validated %d samples against schema '%s'", len(data), detected)
    return True, detected


def sharegpt_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one ShareGPT sample {"conversations":[{"from","value"}]} to
    standard messages [{"role","content"}]. Unknown 'from' values map to
    'user' as a safe default."""
    convs = item.get("conversations", [])
    messages = []
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = _SHAREGPT_ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        content = turn.get("value", turn.get("content", ""))
        if content:
            messages.append({"role": role, "content": str(content)})
    return messages


def load_dataset_file(path: Path, format: str = "") -> list[dict[str, Any]]:
    """Load a dataset file into a list of dicts."""
    if not format:
        format = detect_format(path)
    if not format:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    loaders = {
        "jsonl": _load_jsonl,
        "json": _load_json,
        "csv": _load_csv,
        "tsv": _load_csv,
        "parquet": _load_parquet,
        "arrow": _load_parquet,
    }

    loader = loaders.get(format)
    if loader is None:
        raise ValueError(f"Unsupported format: {format}")

    return loader(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file into a list of dicts.

    Note: Only flattens ONE level of nesting. If the JSON has a deeply
    nested structure like {"data": {"data": [...]}}, only the top-level
    "data" key is extracted — the inner list is NOT walked recursively.
    """
    with open(path, encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        # Try common keys for list-wrapped datasets
        for key in ("data", "entries", "samples", "records", "train"):
            if key in content and isinstance(content[key], list):
                return content[key]
        return [content]
    raise ValueError(f"Unexpected JSON structure in {path}")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
    return data


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required to load parquet files")

    table = pq.read_table(path)
    return table.to_pylist()


def save_dataset_file(
    data: list[dict[str, Any]],
    path: Path,
    format: str = "jsonl",
) -> None:
    """Save a list of dicts to a file."""
    writers = {
        "jsonl": _save_jsonl,
        "json": _save_json,
    }
    writer = writers.get(format)
    if writer is None:
        raise ValueError(f"Unsupported output format: {format}")
    writer(data, path)


def _save_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(_json.dumps(item, ensure_ascii=False) + "\n")


def _save_json(data: list[dict[str, Any]], path: Path) -> None:
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
