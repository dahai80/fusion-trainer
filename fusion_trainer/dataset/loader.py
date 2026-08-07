"""Dataset loaders — load data from file paths into the training pipeline.

Fusion-Trainer does NOT manage the model hub. It reads datasets from
user-provided paths or from the fusion-model-hub directory convention:

    {model_hub_path}/datasets/{category}/{name}.jsonl

The hub directory is treated as an external resource — the Trainer
never creates, modifies, or version-manages it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .formats import (
    SCHEMA_DPO_PAIR,
    SCHEMA_GRPO_PROMPT,
    SCHEMA_MESSAGES,
    SCHEMA_SHAREGPT,
    detect_dataset_schema,
    load_dataset_file,
    sharegpt_to_messages,
    validate_format,
)
from .transforms import deduplicate

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Reads datasets from local file paths for training.

    This is a pure reader — no hub directory management, no versioning,
    no file writing. The dataset is treated as an external resource.
    """

    def __init__(self, model_hub_path: str = ""):
        self._model_hub_path = model_hub_path

    def load(
        self,
        path: str = "",
        hub_dataset: str = "",
        model_hub_path: str = "",
        split: str = "train",
        val_ratio: float = 0.05,
        max_samples: int = 0,
        max_seq_length: int = 4096,
        input_field: str = "text",
        output_field: str = "output",
        remove_duplicates: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load a dataset, returning (train_data, val_data).

        Args:
            path: Direct file path to the dataset.
            hub_dataset: Dataset name within the model hub, e.g. "sft/my_data".
                Resolved as: {model_hub_path}/datasets/{hub_dataset}.jsonl
            model_hub_path: Override the hub path for this call.
            val_ratio: Fraction of data to hold out for validation.
            max_samples: Max samples to load (0 = all).
            max_seq_length: Truncate text fields to this length.
            input_field: Field name for input/prompt text.
            output_field: Field name for output/target text.
            remove_duplicates: Drop exact-duplicate text entries (whitespace-
                normalized, case-insensitive) before the train/val split.
                Default True to match DatasetConfig.remove_duplicates.

        Returns:
            (train_data, val_data), each a list of dicts with 'input' and 'output' keys.
        """
        # Resolve the actual file path
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)

        # Load raw data
        data = load_dataset_file(resolved)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]

        # Deduplicate before the split so identical samples don't leak across
        # train/val and val counts stay correct.
        if remove_duplicates:
            before = len(data)
            data = deduplicate(data)
            if len(data) < before:
                logger.info(
                    "removed %d duplicate samples from %s (%d → %d)",
                    before - len(data),
                    resolved,
                    before,
                    len(data),
                )

        # Split train/val
        if val_ratio > 0:
            split_idx = int(len(data) * (1 - val_ratio))
            raw_train = data[:split_idx]
            raw_val = data[split_idx:]
        else:
            raw_train = data
            raw_val = []

        # Normalize field names
        train = self._normalize(raw_train, input_field, output_field, max_seq_length)
        val = self._normalize(raw_val, input_field, output_field, max_seq_length)
        return train, val

    def load_preference(
        self,
        path: str = "",
        hub_dataset: str = "",
        model_hub_path: str = "",
        max_samples: int = 0,
        max_seq_length: int = 4096,
    ) -> list[dict[str, Any]]:
        """Load a preference dataset (for DPO/ORPO/RLSL).

        Expected format per sample:
            {"prompt": "...", "chosen": "...", "rejected": "..."}
        """
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)
        data = load_dataset_file(resolved)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]

        ok, msg = validate_format(data, SCHEMA_DPO_PAIR)
        if not ok:
            logger.warning("DPO dataset validation: %s", msg)

        result = []
        for item in data:
            prompt = item.get("prompt") or item.get("input", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not prompt or not chosen or not rejected:
                continue
            if max_seq_length > 0:
                prompt = prompt[:max_seq_length]
                chosen = chosen[:max_seq_length]
                rejected = rejected[:max_seq_length]
            result.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        logger.info("loaded %d DPO preference pairs from %s", len(result), resolved)
        return result

    def load_messages(
        self,
        path: str = "",
        hub_dataset: str = "",
        model_hub_path: str = "",
        split: str = "train",
        val_ratio: float = 0.05,
        max_samples: int = 0,
        max_seq_length: int = 4096,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load a D1 SFT chat dataset (messages-JSONL or ShareGPT).

        Accepts both {"messages":[{"role","content"}]} and
        {"conversations":[{"from","value"}]} samples, normalizing the latter
        to standard messages. Returns (train, val) where each sample is
        {"messages": [...]}. Downstream stage_dataset passes messages through
        unchanged so the tokenizer chat template applies.
        """
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)
        data = load_dataset_file(resolved)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]

        normalized = []
        for item in data:
            schema = detect_dataset_schema(item)
            if schema == SCHEMA_MESSAGES:
                messages = item["messages"]
            elif schema == SCHEMA_SHAREGPT:
                messages = sharegpt_to_messages(item)
            else:
                logger.warning(
                    "skip non-chat sample (schema=%s): %s", schema, list(item.keys())
                )
                continue
            if not messages:
                continue
            if max_seq_length > 0:
                messages = [
                    {"role": m["role"], "content": str(m["content"])[:max_seq_length]}
                    for m in messages
                ]
            normalized.append({"messages": messages})

        if val_ratio > 0 and len(normalized) > 1:
            split_idx = int(len(normalized) * (1 - val_ratio))
            train = normalized[:split_idx]
            val = normalized[split_idx:]
        else:
            train = normalized
            val = []
        logger.info(
            "loaded %d chat samples (train=%d val=%d) from %s",
            len(normalized),
            len(train),
            len(val),
            resolved,
        )
        return train, val

    def load_grpo(
        self,
        path: str = "",
        hub_dataset: str = "",
        model_hub_path: str = "",
        max_samples: int = 0,
        max_seq_length: int = 4096,
    ) -> list[dict[str, Any]]:
        """Load a D1 GRPO-prompt dataset.

        Each sample is {"prompt": str, ...} with optional reference fields
        ("answer", "solution", "tests", "reward") preserved for sandbox
        feedback. Samples missing a prompt are skipped.
        """
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)
        data = load_dataset_file(resolved)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]

        ok, msg = validate_format(data, SCHEMA_GRPO_PROMPT)
        if not ok:
            logger.warning("GRPO dataset validation: %s", msg)

        result = []
        for item in data:
            prompt = item.get("prompt", "")
            if not prompt:
                continue
            entry = {
                "prompt": str(prompt)[:max_seq_length]
                if max_seq_length > 0
                else str(prompt)
            }
            for ref_key in ("answer", "solution", "tests", "reward", "metadata"):
                if ref_key in item:
                    entry[ref_key] = item[ref_key]
            result.append(entry)
        logger.info("loaded %d GRPO prompts from %s", len(result), resolved)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(
        self,
        path: str = "",
        hub_dataset: str = "",
        model_hub_path: str = "",
    ) -> Path:
        """Resolve a dataset file path from either a direct path or hub name."""
        if path and os.path.exists(path):
            return Path(path)

        if hub_dataset:
            hub = model_hub_path or self._model_hub_path
            if not hub:
                raise ValueError(
                    "hub_dataset requires model_hub_path to be set. "
                    "Set FUSION_MODEL_HUB env var or pass model_hub_path."
                )
            # Convention: {hub}/datasets/{hub_dataset}.jsonl
            candidate = Path(hub) / "datasets" / f"{hub_dataset}.jsonl"
            if candidate.exists():
                return candidate
            # Try with category prefix already in hub_dataset name
            # e.g. "sft/code_alpaca" → {hub}/datasets/sft/code_alpaca.jsonl
            if "/" in hub_dataset:
                candidate2 = Path(hub) / "datasets" / f"{hub_dataset}.jsonl"
                if candidate2.exists():
                    return candidate2
            raise FileNotFoundError(
                f"Dataset '{hub_dataset}' not found in hub at {hub}/datasets/"
            )

        raise ValueError(
            "Either `path` (to a dataset file) or `hub_dataset` (name in the model hub) "
            "must be provided."
        )

    @staticmethod
    def _normalize(
        raw: list[dict[str, Any]],
        input_field: str,
        output_field: str,
        max_seq_length: int,
    ) -> list[dict[str, Any]]:
        result = []
        for item in raw:
            input_text = item.get(input_field, "")
            output_text = item.get(output_field, "")
            if not input_text or not output_text:
                # Fallback: try common alternative field names
                if not input_text:
                    input_text = (
                        item.get("prompt")
                        or item.get("instruction")
                        or item.get("text")
                        or item.get("input", "")
                    )
                if not output_text:
                    output_text = (
                        item.get("completion")
                        or item.get("response")
                        or item.get("output", "")
                    )
            if not input_text or not output_text:
                continue
            if max_seq_length > 0:
                input_text = input_text[:max_seq_length]
                output_text = output_text[:max_seq_length]
            result.append({"input": input_text, "output": output_text})
        return result
