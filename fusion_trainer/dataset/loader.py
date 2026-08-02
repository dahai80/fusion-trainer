"""Dataset loaders — load data from file paths into the training pipeline.

Fusion-Trainer does NOT manage the model hub. It reads datasets from
user-provided paths or from the fusion-model-hub directory convention:

    {model_hub_path}/datasets/{category}/{name}.jsonl

The hub directory is treated as an external resource — the Trainer
never creates, modifies, or version-manages it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .formats import load_dataset_file


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

        Returns:
            (train_data, val_data), each a list of dicts with 'input' and 'output' keys.
        """
        # Resolve the actual file path
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)

        # Load raw data
        data = load_dataset_file(resolved)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]

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
            result.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
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
