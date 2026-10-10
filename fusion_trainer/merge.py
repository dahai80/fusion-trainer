"""Model merge + export module (issue #58).

Merges a trained LoRA/DoRA adapter with the base model via fusion-mlx's
merge-adapter API, optionally dequantizes for Apple Silicon deployment,
generates a post-merge inference sample, and validates the merged model
produces valid <thinking>+<dsl> output.

Pure HTTP delegation — never imports mlx_lm directly. All merge work
happens inside fusion-mlx (convert_routes.py POST /v1/merge-adapter).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError

logger = logging.getLogger(__name__)

_THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
_DSL_PATTERN = re.compile(r"<dsl>.*?</dsl>", re.DOTALL)

DEFAULT_VALIDATION_PROMPT = (
    "You are a Socratic math lecturer. Solve this step by step, then emit "
    "the final answer as a Socratic DSL JSON inside <dsl>...</dsl>. "
    "Problem: A basket has 12 apples. 3 are eaten. How many remain? "
    "Wrap your reasoning in <thinking>...</thinking>."
)


def merge_adapter(
    client: MLXFineTuneClient,
    base_model: str,
    adapter_path: str,
    output_dir: str = "",
    dequantize: bool = False,
) -> dict[str, Any]:
    """Merge adapter into base model via fusion-mlx merge-adapter API.

    Args:
        client: MLXFineTuneClient (HTTP boundary to fusion-mlx).
        base_model: Base model name/alias (fusion-mlx model id).
        adapter_path: Local path to trained adapter weights.
        output_dir: Output directory for merged SafeTensors. Empty = auto.
        dequantize: If True, produce float weights (dequantize frozen base).

    Returns:
        {status, model, output_path} from fusion-mlx.
    """
    logger.info(
        "merge_adapter: base=%s adapter=%s output=%s dequant=%s",
        base_model,
        adapter_path,
        output_dir or "(auto)",
        dequantize,
    )
    result = client.merge_adapter(
        model=base_model,
        adapter_path=adapter_path,
        output_path=output_dir,
        dequantize=dequantize,
    )
    logger.info("merge_adapter: done -> %s", result.get("output_path", "?"))
    return result


def export_merged_model(
    client: MLXFineTuneClient,
    merged_model_path: str,
    quantize: str = "",
) -> dict[str, Any]:
    """Optional post-merge quantization for Apple Silicon deployment.

    Calls fusion-mlx /v1/quantize to produce 4-bit or 8-bit quantized
    weights from the merged float model. Empty quantize = no-op (merged
    model is already deployment-ready).

    Args:
        client: MLXFineTuneClient.
        merged_model_path: Path to the merged model (from merge_adapter output_path).
        quantize: "4bit", "8bit", or "" (no quantization).

    Returns:
        Quantization result dict, or {"status": "skipped"} if no quantize.
    """
    if not quantize:
        logger.info("export_merged_model: no quantization requested, skipping")
        return {"status": "skipped", "model": merged_model_path}

    bits_map = {"4bit": 4, "8bit": 8}
    bits = bits_map.get(quantize)
    if bits is None:
        raise ValueError(f"quantize must be '4bit' or '8bit', got: {quantize}")

    logger.info("export_merged_model: quantizing %s to %d-bit", merged_model_path, bits)
    body: dict[str, Any] = {
        "model": merged_model_path,
        "q_bits": bits,
        "q_group_size": 64,
    }
    result = client._request(
        "POST",
        f"{client._base}/v1/quantize",
        json_body=body,
        max_retries=1,
    )
    if isinstance(result, dict):
        logger.info("export_merged_model: quantized -> %s", result)
        return result
    raise MLXFineTuneError("quantize returned non-dict")


def validate_merged_model(
    response_text: str,
) -> dict[str, bool]:
    """Validate that a merged model's output has valid <thinking>+<dsl> format.

    Checks:
      has_thinking  — <thinking>...</thinking> block present
      has_dsl       — <dsl>...</dsl> block present
      dsl_valid_json — <dsl> content is parseable JSON

    This is a format check on generation output, not a full DSL schema
    validation (use fusion-core validate_dsl_str for that). Returns a
    dict of bool checks for the caller to act on.
    """
    thinking_match = _THINKING_PATTERN.search(response_text)
    dsl_match = _DSL_PATTERN.search(response_text)

    has_thinking = thinking_match is not None
    has_dsl = dsl_match is not None

    dsl_valid_json = False
    if has_dsl:
        import json

        raw_dsl = dsl_match.group(0)
        inner = raw_dsl[len("<dsl>") : -len("</dsl>")].strip()
        try:
            json.loads(inner)
            dsl_valid_json = True
        except json.JSONDecodeError:
            logger.warning("validate_merged_model: <dsl> block not valid JSON")

    result = {
        "has_thinking": has_thinking,
        "has_dsl": has_dsl,
        "dsl_valid_json": dsl_valid_json,
        "valid": has_thinking and has_dsl and dsl_valid_json,
    }
    if not result["valid"]:
        logger.warning(
            "validate_merged_model: format check failed (thinking=%s dsl=%s json=%s)",
            has_thinking,
            has_dsl,
            dsl_valid_json,
        )
    return result


def generate_validation_sample(
    client: MLXFineTuneClient,
    model_id: str,
    prompt: str = "",
    *,
    max_tokens: int = 512,
) -> str:
    """Generate a post-merge inference sample from the merged model.

    Calls fusion-mlx /v1/chat/completions with the merged model path/id
    and a Socratic-format validation prompt, returning the response text
    for validate_merged_model(). Fail-visible: a generation error logs
    loudly and returns "" (the merge itself still succeeded — validation
    is best-effort, not a merge gate).

    Args:
        client: MLXFineTuneClient (HTTP boundary to fusion-mlx).
        model_id: Merged model path or alias (fusion-mlx resolves it).
        prompt: Validation prompt; empty = DEFAULT_VALIDATION_PROMPT.
        max_tokens: Max generation tokens.
    """
    use_prompt = prompt or DEFAULT_VALIDATION_PROMPT
    logger.info(
        "generate_validation_sample: model=%s prompt_len=%d", model_id, len(use_prompt)
    )
    messages = [{"role": "user", "content": use_prompt}]
    try:
        completions = client.chat_completions(
            model_id=model_id,
            messages=messages,
            n=1,
            temperature=0.3,
            max_tokens=max_tokens,
        )
    except MLXFineTuneError as exc:
        logger.warning(
            "generate_validation_sample: generation failed (non-blocking, "
            "merge itself succeeded): %s",
            exc,
        )
        return ""
    if not completions:
        logger.warning("generate_validation_sample: empty completion list")
        return ""
    return completions[0]


def merge_and_export(
    client: MLXFineTuneClient,
    base_model: str,
    adapter_path: str,
    output_dir: str = "",
    quantize: str = "",
    dequantize: bool = False,
    validate: bool = True,
    validation_prompt: str = "",
) -> dict[str, Any]:
    """Full merge + optional quantize + post-merge validation pipeline.

    Orchestrates merge_adapter → export_merged_model → generate_validation_sample
    → validate_merged_model in one call. The validation step generates a
    Socratic-format sample from the merged model and checks <thinking>+<dsl>
    format — best-effort (fail-visible warning on generation error, never
    gates the merge result).
    """
    merge_result = merge_adapter(
        client=client,
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        dequantize=dequantize,
    )

    merged_path = merge_result.get("output_path", output_dir)
    quantize_result = export_merged_model(
        client=client,
        merged_model_path=merged_path,
        quantize=quantize,
    )

    validation: dict[str, Any] = {"status": "skipped"}
    if validate:
        sample = generate_validation_sample(
            client=client,
            model_id=merged_path,
            prompt=validation_prompt,
        )
        if sample:
            validation = dict(validate_merged_model(sample))
            validation["status"] = "completed"
            validation["sample"] = sample[:200]
        else:
            validation = {
                "status": "failed",
                "reason": "generation returned empty sample",
            }

    return {
        "merge": merge_result,
        "quantize": quantize_result,
        "validation": validation,
        "final_model_path": merged_path,
    }
