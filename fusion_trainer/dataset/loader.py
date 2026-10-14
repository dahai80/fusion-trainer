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
        data = load_dataset_file(resolved, max_samples=max_samples)

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
        data = load_dataset_file(resolved, max_samples=max_samples)

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
        data = load_dataset_file(resolved, max_samples=max_samples)

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

    def load_sft(
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
        """Schema-aware SFT load: chat datasets route through load_messages.

        The flat load() drops messages/ShareGPT samples in _normalize (no
        input/output fields) — a chat-format SFT dataset loaded "successfully"
        with 0 samples and then trained on an empty file. Sniff the dominant
        schema first: chat → load_messages (messages pass through to
        stage_dataset for the tokenizer chat template), else → load().
        """
        resolved = self._resolve_path(path, hub_dataset, model_hub_path)
        data = load_dataset_file(resolved, max_samples=max_samples)
        chat = 0
        flat = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            schema = detect_dataset_schema(item)
            if schema in (SCHEMA_MESSAGES, SCHEMA_SHAREGPT):
                chat += 1
            elif schema:
                flat += 1
        if chat and flat:
            logger.warning(
                "mixed-schema SFT dataset (%d chat + %d flat samples) — "
                "routing through the chat loader; flat samples are skipped "
                "with a per-sample warning",
                chat,
                flat,
            )
        if chat:
            logger.info(
                "load_sft: chat schema detected (%d/%d samples) — using load_messages",
                chat,
                len(data),
            )
            return self.load_messages(
                path=path,
                hub_dataset=hub_dataset,
                model_hub_path=model_hub_path,
                split=split,
                val_ratio=val_ratio,
                max_samples=max_samples,
                max_seq_length=max_seq_length,
            )
        return self.load(
            path=path,
            hub_dataset=hub_dataset,
            model_hub_path=model_hub_path,
            split=split,
            val_ratio=val_ratio,
            max_samples=max_samples,
            max_seq_length=max_seq_length,
            input_field=input_field,
            output_field=output_field,
            remove_duplicates=remove_duplicates,
        )

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
        data = load_dataset_file(resolved, max_samples=max_samples)

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
            # R3: path-traversal guard. hub_dataset is user-controlled (CLI
            # --dataset hub:NAME / GUI). Reject absolute paths, drive letters,
            # and any '..' segment so "hub:../config/secrets" can't escape the
            # hub datasets dir to read arbitrary .jsonl files (keys/config).
            # The resolved candidate must end up INSIDE {hub}/datasets/.
            if os.path.isabs(hub_dataset):
                raise ValueError(
                    f"hub_dataset must be a relative name, got absolute: {hub_dataset!r}"
                )
            parts = Path(hub_dataset).parts
            if any(p == ".." for p in parts):
                raise ValueError(
                    f"hub_dataset must not contain '..' segments: {hub_dataset!r}"
                )
            hub_root = Path(hub).resolve()
            datasets_dir = (hub_root / "datasets").resolve()
            # Convention: {hub}/datasets/{hub_dataset}.jsonl
            candidate = (datasets_dir / f"{hub_dataset}.jsonl").resolve()
            try:
                candidate.relative_to(datasets_dir)
            except ValueError:
                raise ValueError(
                    f"hub_dataset {hub_dataset!r} resolves outside the hub "
                    f"datasets directory — path traversal blocked"
                )
            if candidate.exists():
                return candidate
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
        dropped = 0
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
                # E5 (round-2 audit): a sample missing input OR output was
                # silently dropped with zero log — a half-empty dataset loaded
                # "successfully" and trained on missing targets. Count + warn
                # so the operator sees the data loss before a wasted run.
                dropped += 1
                continue
            if max_seq_length > 0:
                input_text = input_text[:max_seq_length]
                output_text = output_text[:max_seq_length]
            result.append({"input": input_text, "output": output_text})
        if dropped:
            logger.warning(
                "_normalize: dropped %d/%d samples missing input or output "
                "field (input_field=%r output_field=%r) — dataset may be "
                "smaller than expected",
                dropped,
                len(raw),
                input_field,
                output_field,
            )
        return result
