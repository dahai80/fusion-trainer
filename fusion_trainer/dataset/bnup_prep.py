"""BNUP SFT data preparation (issue #56).

Reads raw BNUP math problems, validates the <thinking> + <dsl> format
via fusion-core's validate_dsl_str, and packages valid samples into
Alpaca/ShareGPT-compatible JSONL for mlx-lm SFT training.

Pipeline:
  raw JSON ([{question, thinking, dsl}])
    -> validate_dsl_json (fusion-core schema check)
    -> build_sft_sample (assemble <thinking>+<dsl> response)
    -> process_raw_sft (batch, skip invalid, log count)
    -> JSONL output (compatible with DatasetLoader.load)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是一名北师大版 K-12 数学苏格拉底讲师。"
    "请分析题目，先在 <thinking> 标签内输出思考链，"
    "再在 <dsl> 标签内输出标准的 JSON DSL。"
)

_THINKING_OPEN = "<thinking>"
_THINKING_CLOSE = "</thinking>"
_DSL_OPEN = "<dsl>"
_DSL_CLOSE = "</dsl>"

_REQUIRED_RAW_KEYS = ("question", "thinking", "dsl")


def validate_dsl_json(dsl_str: str) -> bool:
    """Validate a DSL JSON string via fusion-core validate_dsl_str.

    Returns True only if the string is valid JSON AND passes the
    Socratic DSL v3 schema. Delegates to fusion-core (no duplicated
    schema — single source of truth).
    """
    try:
        from fusion_core import validate_dsl_str
    except ImportError:
        logger.error(
            "validate_dsl_json: fusion-core not installed (pip install -e fusion-core)"
        )
        return False

    ok, err = validate_dsl_str(dsl_str)
    if not ok:
        logger.debug("validate_dsl_json: invalid DSL: %s", err)
        return False
    return True


def build_sft_sample(
    system_prompt: str,
    question: str,
    thinking_content: str,
    dsl_dict: dict[str, Any],
) -> dict[str, str]:
    """Assemble an Alpaca/ShareGPT-compatible SFT sample.

    Output format: {instruction, input, output, system} where output
    contains <thinking>...</thinking><dsl>...</dsl>.
    """
    dsl_json_str = json.dumps(dsl_dict, ensure_ascii=False, indent=2)
    assistant_response = (
        f"{_THINKING_OPEN}\n{thinking_content.strip()}\n{_THINKING_CLOSE}\n"
        f"{_DSL_OPEN}\n{dsl_json_str}\n{_DSL_CLOSE}"
    )
    return {
        "instruction": question,
        "input": "",
        "output": assistant_response,
        "system": system_prompt,
    }


def process_raw_sft(
    input_file: str,
    output_file: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, int]:
    """Batch-process raw BNUP problems into SFT JSONL.

    Reads a JSON array of {question, thinking, dsl} objects, validates
    each, skips invalid (logs count), writes valid samples as JSONL.

    Returns a summary dict: {valid, skipped, total}.
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_file}")

    raw_text = input_path.read_text(encoding="utf-8")
    try:
        raw_items = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input file is not valid JSON: {exc}") from exc

    if not isinstance(raw_items, list):
        raise ValueError(f"expected JSON array, got {type(raw_items).__name__}")

    sft_data: list[dict[str, str]] = []
    skipped = 0

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            skipped += 1
            logger.warning("raw[%d]: not a dict, skipping", idx)
            continue

        question = item.get("question", "")
        thinking = item.get("thinking", "")
        dsl_data = item.get("dsl")

        if not question or not thinking or not dsl_data:
            skipped += 1
            logger.debug("raw[%d]: missing required field, skipping", idx)
            continue

        if not isinstance(dsl_data, dict):
            skipped += 1
            logger.debug("raw[%d]: dsl not a dict, skipping", idx)
            continue

        dsl_str = json.dumps(dsl_data, ensure_ascii=False)
        if not validate_dsl_json(dsl_str):
            skipped += 1
            logger.warning("raw[%d]: DSL schema validation failed, skipping", idx)
            continue

        sample = build_sft_sample(system_prompt, question, thinking, dsl_data)
        sft_data.append(sample)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in sft_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total = len(raw_items)
    logger.info(
        "process_raw_sft: valid=%d skipped=%d total=%d -> %s",
        len(sft_data),
        skipped,
        total,
        output_file,
    )
    print(
        f"✅ SFT 数据集构建完成：有效样本 {len(sft_data)} 条，"
        f"跳过非法样本 {skipped} 条。已输出至 {output_file}"
    )
    return {"valid": len(sft_data), "skipped": skipped, "total": total}
