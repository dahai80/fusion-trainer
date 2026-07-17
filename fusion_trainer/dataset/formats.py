"""Dataset format detection and loading."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


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
    with open(path, "r", encoding="utf-8") as f:
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
    with open(path, "r", encoding="utf-8") as f:
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
    with open(path, "r", encoding="utf-8") as f:
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