"""Pre-flight unified-memory admission gate (capability plan §1.3).

Before submitting a fine-tune job, estimate the training memory budget
from the model id + quantization + sequence length (deterministic, Rule 5 —
no model routing) and compare against psutil-reported available memory.
Insufficient → fail-visible with concrete de-escalation suggestions;
the guard never mutates the config by itself.

Heuristic (Apple Silicon unified memory):
    weights    = params_b * quant_factor   (4bit 0.55 / 8bit 1.05 / fp16 2.1)
    activation = batch * seq * params_b * k / 1024   (k 1.0, 0.30 w/ ckpt)
    overhead   = 4 GB (python + mlx runtime + temp buffers)
Calibrated against the recorded 27B-8bit incidents (#76: OOM panic without
--grad-checkpoint at seq 4096; completion with it at seq 1024).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.IGNORECASE)
QUANT_FACTORS = {0: 2.1, 4: 0.55, 8: 1.05}
CHECKPOINT_FACTOR = 0.30
RUNTIME_OVERHEAD_GB = 4.0
# matches fusion-mlx's upstream default budget when memory_limit_gb is unset
DEFAULT_BUDGET_FRACTION = 0.6


def estimate_training_gb(
    model: str,
    quant_bits: int = 0,
    batch_size: int = 4,
    max_seq_length: int = 4096,
    gradient_checkpointing: bool = False,
) -> float:
    """Deterministic training-memory estimate in GB; 0.0 = model unknown."""
    match = MODEL_SIZE_RE.search(model or "")
    if not match:
        logger.info("memory_guard: no param count parsed from model id %r", model)
        return 0.0
    params_b = float(match.group(1))
    factor = QUANT_FACTORS.get(int(quant_bits or 0), QUANT_FACTORS[0])
    weights = params_b * factor
    ckpt = CHECKPOINT_FACTOR if gradient_checkpointing else 1.0
    activation = batch_size * max_seq_length * params_b * ckpt / 1024.0
    return round(weights + activation + RUNTIME_OVERHEAD_GB, 1)


def precheck(
    model: str,
    quant_bits: int = 0,
    batch_size: int = 4,
    max_seq_length: int = 4096,
    gradient_checkpointing: bool = False,
    memory_limit_gb: float = 0.0,
) -> dict[str, Any]:
    """Admission check: estimated need vs available unified memory.

    Returns {ok, estimate_gb, available_gb, budget_gb, suggestions}.
    ok=False means fail-visible stop BEFORE submission (never auto-mutate).

    Opt-out: FUSION_TRAINER_MEMORY_GUARD=off disables the gate entirely
    (logged loudly). Intended for constrained CI runners where the 4GB
    runtime-overhead constant makes every real-training estimate exceed
    physical availability; keep it unset on operator machines.
    """
    if os.environ.get("FUSION_TRAINER_MEMORY_GUARD", "").strip().lower() == "off":
        logger.warning(
            "memory_guard: disabled via FUSION_TRAINER_MEMORY_GUARD=off — "
            "skipping admission check (CI/constrained-runner opt-out)"
        )
        return {
            "ok": True,
            "estimate_gb": 0.0,
            "available_gb": 0.0,
            "budget_gb": 0.0,
            "suggestions": [],
        }
    estimate = estimate_training_gb(
        model, quant_bits, batch_size, max_seq_length, gradient_checkpointing
    )
    try:
        import psutil

        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
    except Exception as exc:  # pragma: no cover - psutil is a hard dep
        logger.warning("memory_guard: psutil unavailable (%s) — gate passes", exc)
        return {
            "ok": True,
            "estimate_gb": estimate,
            "available_gb": 0.0,
            "budget_gb": 0.0,
            "suggestions": [],
        }
    budget = (
        memory_limit_gb if memory_limit_gb > 0 else total_gb * DEFAULT_BUDGET_FRACTION
    )
    if estimate <= 0:
        # unknown model size — cannot gate, pass with a note
        return {
            "ok": True,
            "estimate_gb": 0.0,
            "available_gb": round(available_gb, 1),
            "budget_gb": round(budget, 1),
            "suggestions": [],
        }
    # two constraints: physical availability AND the training budget (explicit
    # memory_limit_gb, or upstream's default 60%-of-total). Both must hold.
    ok = available_gb >= estimate and estimate <= budget
    suggestions: list[str] = []
    if not ok:
        if estimate > budget:
            suggestions.append(
                f"raise memory_limit_gb (estimate {estimate:.1f}GB > budget {budget:.1f}GB)"
            )
        if not gradient_checkpointing:
            suggestions.append("pass --grad-checkpoint (gradient_checkpointing: true)")
        if max_seq_length > 1024:
            suggestions.append("lower --max-seq-length (e.g. 1024)")
        if quant_bits != 8:
            suggestions.append("train 8-bit (--quant-bits 8 / quantize_base)")
        elif quant_bits == 8:
            suggestions.append("use a smaller base model or free resident models")
        suggestions.append("close other resident models / apps to free memory")
    logger.info(
        "memory_guard: ok=%s estimate=%.1fGB available=%.1fGB budget=%.1fGB model=%s",
        ok,
        estimate,
        available_gb,
        budget,
        model,
    )
    return {
        "ok": ok,
        "estimate_gb": estimate,
        "available_gb": round(available_gb, 1),
        "budget_gb": round(budget, 1),
        "suggestions": suggestions,
    }
