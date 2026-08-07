"""SFT module — supervised fine-tuning pipeline."""

from ._errors import SFTUnsupportedError
from .lora import LoRAConfig
from .presets import SFT_PRESETS
from .trainer import SFTTrainer

__all__ = [
    "SFTTrainer",
    "LoRAConfig",
    "SFT_PRESETS",
    "SFTUnsupportedError",
]
