"""Dataset transforms — cleaning, filtering, deduplication, splitting."""

from __future__ import annotations

import re
from typing import Any


def clean_dataset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply basic cleaning to a dataset."""
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Strip whitespace from string values
        cleaned_item = {}
        for k, v in item.items():
            if isinstance(v, str):
                v = v.strip()
            cleaned_item[k] = v
        cleaned.append(cleaned_item)
    return cleaned


def deduplicate(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate entries based on text content hash."""
    seen = set()
    result = []
    for item in data:
        # Use the text/input/output as dedup key
        text = (
            str(item.get("text", ""))
            or str(item.get("input", ""))
            or str(item.get("prompt", ""))
        )
        # Normalize whitespace for comparison
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def filter_by_length(
    data: list[dict[str, Any]],
    min_len: int = 2,
    max_len: int = 0,
    text_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter dataset entries by text length."""
    if text_fields is None:
        text_fields = ["text", "input", "output", "prompt", "completion"]

    filtered = []
    for item in data:
        for field in text_fields:
            val = item.get(field, "")
            if isinstance(val, str) and len(val) >= min_len:
                if max_len > 0 and len(val) > max_len:
                    continue
                filtered.append(item)
                break
    return filtered


def truncate_fields(
    data: list[dict[str, Any]],
    max_length: int,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Truncate specified fields to max_length."""
    if fields is None:
        fields = [
            "text",
            "input",
            "output",
            "prompt",
            "completion",
            "chosen",
            "rejected",
        ]

    result = []
    for item in data:
        new_item = dict(item)
        for field in fields:
            if field in new_item and isinstance(new_item[field], str):
                new_item[field] = new_item[field][:max_length]
        result.append(new_item)
    return result


def split_dataset(
    data: list[dict[str, Any]],
    val_ratio: float = 0.05,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into training and validation sets."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(data)))
    if shuffle:
        rng.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]
    return train, val


def to_chat_format(
    data: list[dict[str, Any]],
    system_prompt: str = "",
    input_field: str = "input",
    output_field: str = "output",
) -> list[dict[str, Any]]:
    """Convert raw text data to chat-format messages."""
    result = []
    for item in data:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": item.get(input_field, "")})
        messages.append({"role": "assistant", "content": item.get(output_field, "")})
        result.append({"messages": messages})
    return result
