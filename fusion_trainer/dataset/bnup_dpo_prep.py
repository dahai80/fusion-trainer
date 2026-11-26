"""BNUP DPO data preparation (issue #56, uses #57 negatives).

Generates DPO preference pairs from SFT samples. Each SFT sample
produces up to neg_ratio negative variants via bnup_negatives.generate_negatives,
producing {conversations, chosen, rejected} pairs compatible with
DPOTrainer's expected input format.

Output format (ShareGPT-style, mlx-lm compatible):
  {
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."}
    ],
    "chosen": {"from": "gpt", "value": "..."},
    "rejected": {"from": "gpt", "value": "..."}
  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fusion_trainer.dataset.bnup_negatives import generate_negatives

logger = logging.getLogger(__name__)


def _sft_to_dpo_pair(
    sft_item: dict[str, str],
    neg_ratio: int = 3,
) -> list[dict[str, Any]]:
    """Convert one SFT sample into up to neg_ratio DPO pairs.

    Each pair shares the same prompt + chosen, with a different rejected.
    """
    prompt = sft_item.get("instruction", "")
    system = sft_item.get("system", "")
    chosen_response = sft_item.get("output", "")

    if not prompt or not chosen_response:
        logger.debug("_sft_to_dpo_pair: empty prompt/chosen, skipping")
        return []

    question = prompt
    formula = ""
    answer = _try_extract_answer(chosen_response)

    negatives = generate_negatives(
        chosen_response=chosen_response,
        question=question,
        formula=formula,
        answer=answer,
        neg_ratio=neg_ratio,
    )

    pairs: list[dict[str, Any]] = []
    for rejected_response in negatives:
        pair = {
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": prompt},
            ],
            "chosen": {"from": "gpt", "value": chosen_response},
            "rejected": {"from": "gpt", "value": rejected_response},
        }
        pairs.append(pair)

    logger.debug(
        "_sft_to_dpo_pair: %d pairs from 1 SFT sample (ratio=%d)",
        len(pairs),
        neg_ratio,
    )
    return pairs


def _try_extract_answer(response: str) -> str:
    """Best-effort extract a numeric answer from <thinking> or formula text.

    Looks for patterns like '<var> = 40' or a Chinese result label followed
    by a number. Returns "" if not found.
    Used to make direct-answer negatives more realistic.
    """
    import re

    # i18n: user-facing
    match = re.search(r"(?:=|结果[：:])\s*(\d+(?:\.\d+)?)", response)
    if match:
        return match.group(1)
    return ""


def create_dpo_pairs(
    sft_file: str,
    dpo_output: str,
    neg_ratio: int = 3,
) -> dict[str, int]:
    """Generate DPO preference pairs from an SFT JSONL file.

    Reads SFT samples (one JSON per line, {instruction, input, output, system}),
    generates neg_ratio negatives per sample, writes ShareGPT-format JSONL.

    Returns: {pairs, samples, neg_ratio}.
    """
    sft_path = Path(sft_file)
    if not sft_path.is_file():
        raise FileNotFoundError(f"SFT file not found: {sft_file}")

    dpo_samples: list[dict[str, Any]] = []
    sample_count = 0

    with open(sft_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sft_item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("create_dpo_pairs: line %d not JSON, skipping", line_num)
                continue

            if not isinstance(sft_item, dict):
                continue

            sample_count += 1
            pairs = _sft_to_dpo_pair(sft_item, neg_ratio=neg_ratio)
            dpo_samples.extend(pairs)

    output_path = Path(dpo_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in dpo_samples:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        "create_dpo_pairs: %d pairs from %d samples (ratio=%d) -> %s",
        len(dpo_samples),
        sample_count,
        neg_ratio,
        dpo_output,
    )
    print(
        f"✅ DPO dataset build complete: generated {len(dpo_samples)} preference pairs "
        f"({sample_count} samples x {neg_ratio} negatives). Written to {dpo_output}"
    )
    return {
        "pairs": len(dpo_samples),
        "samples": sample_count,
        "neg_ratio": neg_ratio,
    }


def normalize_for_trainer(dpo_pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert ShareGPT-format pairs to DPOTrainer's {prompt, chosen, rejected}.

    DPOTrainer._normalize_pairs expects flat {prompt, chosen, rejected} dicts.
    This bridges the ShareGPT conversations format to that flat format.
    """
    flat: list[dict[str, str]] = []
    for pair in dpo_pairs:
        convs = pair.get("conversations", [])
        system_msg = ""
        human_msg = ""
        for msg in convs:
            role = msg.get("from", "")
            value = msg.get("value", "")
            if role == "system":
                system_msg = value
            elif role == "human":
                human_msg = value

        prompt = human_msg
        if system_msg:
            prompt = f"{system_msg}\n\n{human_msg}"

        chosen = pair.get("chosen", {}).get("value", "")
        rejected = pair.get("rejected", {}).get("value", "")

        if not prompt or not chosen or not rejected:
            continue

        flat.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    logger.debug("normalize_for_trainer: %d -> %d pairs", len(dpo_pairs), len(flat))
    return flat
