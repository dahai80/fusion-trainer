"""Shared error types for the RLSL module.

Kept in a leaf module to avoid the circular import that would arise from
defining these in rlsl/__init__.py (which imports the trainers, which in turn
need the error types).
"""

from __future__ import annotations


class RLSLUnsupportedError(RuntimeError):
    """Raised when an RLSL method needs fusion-mlx support that is not landed.

    DPO/ORPO/reward-model training require in-model logprob + gradient updates.
    Per the project rule ("integrate with fusion-mlx only ... for upstream
    issues, file an issue first, then a PR"),
    these run server-side in fusion-mlx; the corresponding upstream endpoints do
    not exist yet (tracked in fusion-mlx#399). Until they land, these trainers
    fail visibly instead of silently returning fake "completed" results.
    """
