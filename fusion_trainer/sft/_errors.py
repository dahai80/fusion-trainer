"""Shared error types for the SFT module.

Kept in a leaf module so the trainers can import it without pulling the rest
of the sft package (which would create an import cycle through __init__.py).
"""

from __future__ import annotations


class SFTUnsupportedError(RuntimeError):
    """Raised when an SFT feature needs fusion-mlx support that is not landed.

    QLoRA (quantized-base LoRA) and MXFP8 mixed-precision training require
    server-side support in fusion-mlx's FineTuneConfig. Per the project rule
    ("integrate with fusion-mlx only ... for upstream issues, file an issue
    first, then a PR"), fusion-trainer talks to
    fusion-mlx over HTTP only and never imports mlx-lm directly, so these
    capabilities must live upstream. They do not exist yet
    (tracked in fusion-mlx#402). Until it lands, _build_mlx_config fails
    visibly instead of silently dropping the flags / faking a lora fallback.
    """
