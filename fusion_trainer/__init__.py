"""Fusion-Trainer — lightweight supervised + reinforcement learning trainer on Apple MLX.

Built on top of fusion-mlx, this product provides:
- Dataset Hub: local dataset management, versioning, format conversion
- SFT Pipeline: Supervised Fine-Tuning with LoRA/QLoRA support
- RLSL Pipeline: Reward Model training + DPO/ORPO alignment
- Experiment Dashboard: training metrics, loss curves, reproducibility tracking

Designed for Mac M5 unified memory (96 GB), targeting 7B–32B model fine-tuning
and RLHF-style alignment without cloud dependencies.
"""

from ._version import __version__
from .config import TrainerConfig, DatasetConfig, SFTConfig, RLSLConfig

__all__ = [
    "__version__",
    "TrainerConfig",
    "DatasetConfig",
    "SFTConfig",
    "RLSLConfig",
]