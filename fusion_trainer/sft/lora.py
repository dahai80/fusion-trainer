"""LoRA configuration for SFT fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA / QLoRA fine-tuning."""

    rank: int = 16
    """LoRA rank (r)."""

    alpha: float = 32.0
    """LoRA scaling factor (alpha)."""

    dropout: float = 0.05
    """Dropout rate for LoRA layers."""

    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    """Which modules to apply LoRA to."""

    quantize_base: bool = False
    """Whether to quantize the base model (QLoRA)."""

    quant_bits: int = 4
    """Quantization bits for QLoRA (4 or 8)."""

    use_dora: bool = False
    """Use DoRA (Weight-Decomposed Low-Rank Adaptation) if supported."""

    @classmethod
    def from_method(cls, method: str) -> LoRAConfig:
        """Create a LoRAConfig based on training method name."""
        if method == "full":
            return cls(rank=0, alpha=0, quantize_base=False)
        if method == "qlora":
            return cls(rank=16, alpha=32.0, quantize_base=True, quant_bits=4)
        # lora (default)
        return cls(rank=16, alpha=32.0, quantize_base=False)
