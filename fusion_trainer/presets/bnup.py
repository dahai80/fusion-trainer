"""BNUP two-stage training presets (issue #59).

Single source for BNUP SFT + DPO + pipeline orchestration presets
(北师大版 K-12 math). SFT/DPO preset values are sourced from the existing
sft/presets.py and rlsl/presets.py modules (no duplication); this module
aggregates them and adds BNUP_PIPELINE for pipeline orchestration.
"""

from __future__ import annotations

from typing import Any

from fusion_trainer.rlsl.presets import RLSL_PRESETS
from fusion_trainer.sft.presets import SFT_PRESETS

BNUP_SFT_32B: dict[str, Any] = dict(SFT_PRESETS["bnup_sft_32b"])
BNUP_DPO_32B: dict[str, Any] = dict(RLSL_PRESETS["bnup_dpo_32b"])

BNUP_PIPELINE: dict[str, Any] = {
    "description": "BNUP two-stage pipeline (SFT QLoRA 8-bit → DPO → merge, "
    "北师大版 K-12 math). DPO chains on SFT adapter (fusion-mlx #1149).",
    "sft_preset": "bnup_sft_32b",
    "dpo_preset": "bnup_dpo_32b",
    "chain_adapter": True,
    "merge": True,
}

BNUP_PRESETS: dict[str, dict[str, Any]] = {
    "BNUP_SFT_32B": BNUP_SFT_32B,
    "BNUP_DPO_32B": BNUP_DPO_32B,
    "BNUP_PIPELINE": BNUP_PIPELINE,
}
