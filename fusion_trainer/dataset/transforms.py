"""Dataset transforms — cleaning, filtering, deduplication, splitting."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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
    """Remove duplicate entries based on input+output content hash.

    E5 (round-2 audit): the old key hashed ONLY the prompt/input side, so two
    samples with the same prompt but different completions (a real SFT case —
    one right answer, one wrong answer to the same question) collapsed into
    one. That silently shrank + corrupted the dataset. The key now includes
    the completion/output side; a sample is a duplicate only if BOTH sides
    match. Empty-key samples (no detectable text on either side) are kept
    rather than dropped-by-dedup — load_messages handles schema filtering.
    """
    seen: set[str] = set()
    result = []
    dropped = 0
    for item in data:
        prompt_text = (
            str(item.get("text", ""))
            or str(item.get("input", ""))
            or str(item.get("prompt", ""))
            or str(item.get("instruction", ""))
        )
        completion_text = (
            str(item.get("output", ""))
            or str(item.get("completion", ""))
            or str(item.get("response", ""))
            or str(item.get("answer", ""))
        )
        norm_prompt = re.sub(r"\s+", " ", prompt_text).strip().lower()
        norm_completion = re.sub(r"\s+", " ", completion_text).strip().lower()
        if not norm_prompt and not norm_completion:
            # Chat formats (messages/ShareGPT) carry their text in nested
            # fields the flat keys above can't see — the E5 drop silently
            # deleted every messages sample (a real SFT load returned 0
            # samples). Serialize the chat turns into the key instead:
            # identical conversations dedup, distinct ones survive.
            chat = item.get("messages") or item.get("conversations")
            if isinstance(chat, list) and chat:
                try:
                    import json as _json

                    norm_prompt = (
                        re.sub(
                            r"\s+",
                            " ",
                            _json.dumps(chat, ensure_ascii=False, sort_keys=True),
                        )
                        .strip()
                        .lower()
                    )
                except (TypeError, ValueError):
                    norm_prompt = ""
            if not norm_prompt:
                # Genuinely no detectable text anywhere — keep (schema
                # filtering is load_messages' job, per the E5 contract).
                result.append(item)
                continue
            key = norm_prompt + "\x00chat"
        else:
            key = norm_prompt + "\x00" + norm_completion
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        result.append(item)
    if dropped:
        logger.info("deduplicate: removed %d duplicate samples", dropped)
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
