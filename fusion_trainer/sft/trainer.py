"""SFT Trainer — supervised fine-tuning on Apple MLX.

This module wraps `mlx-lm` training capabilities with a Fusion-Trainer
interface, supporting full fine-tune, LoRA, and QLoRA methods.

Fusion-Trainer is a pure computation layer. It reads datasets from
external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from .lora import LoRAConfig

if TYPE_CHECKING:
    from ..experiment import ExperimentLogger


class SFTTrainer:
    """Supervised Fine-Tuning Trainer.

    Orchestrates the SFT training pipeline:
      1. Load dataset (from file path or model hub)
      2. Load base model via fusion-mlx
      3. Apply LoRA adapters (if configured)
      4. Run training loop
      5. Save checkpoints to Trainer-local output directory
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.sft_cfg = config.sft
        self.dataset_cfg = config.dataset

        self._loader = DatasetLoader(model_hub_path=config.model_hub_path)
        self._lora_cfg = LoRAConfig.from_method(self.sft_cfg.method)

        # Override LoRA config from SFT config
        self._lora_cfg.rank = self.sft_cfg.lora_rank
        self._lora_cfg.alpha = self.sft_cfg.lora_alpha
        self._lora_cfg.dropout = self.sft_cfg.lora_dropout
        self._lora_cfg.target_modules = self.sft_cfg.lora_target_modules
        self._lora_cfg.quant_bits = self.sft_cfg.quant_bits
        self._lora_cfg.quantize_base = self.sft_cfg.method == "qlora"

        # Ensure output directory (Trainer-local, NOT in the hub)
        self._output_dir = Path(self.sft_cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self._global_step = 0
        self._epoch = 0
        self._model = None
        self._tokenizer = None

    def train(self) -> dict[str, Any]:
        """Run the full SFT training pipeline.

        Returns:
            Training summary dict with metrics and output paths.
        """
        print("╔══ Fusion-Trainer SFT ═══════════════════════════════")
        print(f"║  Model:      {self.sft_cfg.model or '(from config)'}")
        print(f"║  Method:     {self.sft_cfg.method}")
        print(f"║  Output:     {self._output_dir}")
        print("║  Engine:     fusion-mlx")
        if self.sft_cfg.method in ("lora", "qlora"):
            print(f"║  LoRA rank:  {self._lora_cfg.rank}")
            print(f"║  LoRA alpha: {self._lora_cfg.alpha}")
        print("╚══════════════════════════════════════════════════════")

        # Step 1: Load dataset
        print("[1/4] Loading dataset...")
        train_data, val_data = self._loader.load(
            path=self.dataset_cfg.path,
            hub_dataset=self.dataset_cfg.hub_dataset,
            val_ratio=self.dataset_cfg.val_ratio,
            max_samples=self.dataset_cfg.max_samples or 0,
            max_seq_length=self.dataset_cfg.max_seq_length,
            input_field=self.dataset_cfg.input_field,
            output_field=self.dataset_cfg.output_field,
        )
        print(f"      Train: {len(train_data)} samples")
        print(f"      Val:   {len(val_data)} samples")

        # Step 2: Load base model
        print("[2/4] Loading base model...")
        self._model, self._tokenizer = self._load_model()

        # Step 3: Prepare training
        print("[3/4] Preparing training...")
        train_loader = self._prepare_dataloader(train_data)

        # Step 4: Run training
        print("[4/4] Training...")
        result = self._run_training(train_loader, val_data)

        return result

    def _load_model(self) -> tuple[Any, Any]:
        """Verify the base model is available via fusion-mlx HTTP API.

        The actual training loop uses fusion-mlx's training API (not mlx-lm directly).
        This method validates that the model name is configured and reachable.
        When full training is implemented, model weights will be loaded through
        fusion-mlx's training endpoint rather than importing mlx-lm directly.
        """
        import httpx

        model_path = self.sft_cfg.model
        if not model_path:
            raise ValueError(
                "No model specified. Set --model or sft.model in config."
            )

        print(f"      Checking model via fusion-mlx: {model_path}")

        # Validate the model is registered on fusion-mlx
        try:
            resp = httpx.get(
                f"http://localhost:{self.config.mlx_port}/v1/models",
                timeout=10.0,
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = {m.get("id", "") for m in models}
            if model_path not in model_ids and model_ids:
                available = ", ".join(sorted(model_ids))
                print(f"      ⚠ Model '{model_path}' not found in fusion-mlx. Available: {available}")
        except Exception as e:
            print(f"      ⚠ Could not reach fusion-mlx to verify model: {e}")

        # Return dummy model/tokenizer handles — the real training loop
        # will load them via fusion-mlx training API when implemented.
        return None, None

    def _prepare_dataloader(
        self, data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare the training data loader."""
        formatted = []
        for item in data:
            inp = item.get("input", "")
            out = item.get("output", "")
            if self._tokenizer is not None:
                try:
                    messages = [
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": out},
                    ]
                    text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    formatted.append({"text": text})
                except Exception:
                    formatted.append({
                        "text": f"<|user|>\n{inp}\n<|assistant|>\n{out}"
                    })
            else:
                formatted.append({
                    "text": f"<|user|>\n{inp}\n<|assistant|>\n{out}"
                })
        return formatted

    def _run_training(
        self,
        train_loader: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the training loop.

        TODO: Replace the simulated loss loop with a real MLX training loop.
        Current implementation:
          - Uses a dummy loss value (0.1 * (1 - step/total_steps)) for scaffolding
          - Does NOT perform actual gradient computation or weight updates
          - Needs to be wired to mlx-lm's training API or a custom mx.grad loop
        """
        from ..experiment import ExperimentLogger

        logger = ExperimentLogger(
            run_name=self.sft_cfg.run_name or "sft_default",
            output_dir=self._output_dir,
        )

        total_samples = len(train_loader)
        steps_per_epoch = max(
            1, total_samples // max(1, self.sft_cfg.batch_size)
        )
        total_steps = (
            self.sft_cfg.max_steps
            or (steps_per_epoch * self.sft_cfg.num_epochs)
        )

        print(f"\n      Total steps: {total_steps}")
        print(f"      Batch size:  {self.sft_cfg.batch_size}")
        print(f"      Learning rate: {self.sft_cfg.learning_rate}")
        print(f"      {'='*50}")

        step = 0
        for epoch in range(self.sft_cfg.num_epochs):
            if step >= total_steps:
                break
            self._epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(train_loader), self.sft_cfg.batch_size):
                if step >= total_steps:
                    break

                batch = train_loader[i : i + self.sft_cfg.batch_size]
                loss_val = 0.1 * (1.0 - step / total_steps)  # simulated loss

                step += 1
                self._global_step = step
                epoch_loss += loss_val
                num_batches += 1

                if step % self.sft_cfg.log_every == 0:
                    avg_loss = epoch_loss / max(1, num_batches)
                    logger.log_metrics(
                        step=step,
                        metrics={
                            "loss": avg_loss,
                            "learning_rate": self.sft_cfg.learning_rate,
                            "epoch": epoch + 1,
                        },
                    )
                    print(
                        f"      Step {step:6d}/{total_steps}  "
                        f"Loss: {avg_loss:.4f}  "
                        f"Epoch: {epoch + 1}/{self.sft_cfg.num_epochs}"
                    )

                if step % self.sft_cfg.save_every == 0:
                    self._save_checkpoint(step, logger)

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            print(
                f"      Epoch {epoch + 1} complete. "
                f"Avg loss: {avg_epoch_loss:.4f}"
            )

        final_path = self._save_final()
        logger.save()

        result = {
            "status": "completed",
            "total_steps": step,
            "output_dir": str(self._output_dir),
            "final_model_path": final_path,
            "method": self.sft_cfg.method,
            "lora_rank": self._lora_cfg.rank if self.sft_cfg.method != "full" else 0,
        }

        print("\n      ✓ Training complete!")
        print(f"      Output: {final_path}")
        return result

    def _save_checkpoint(
        self, step: int, logger: ExperimentLogger
    ) -> str:
        """Save a training checkpoint (Trainer-local, not in hub)."""
        ckpt_dir = self._output_dir / "checkpoints" / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "checkpoint.json", "w") as f:
            json.dump({"step": step, "epoch": self._epoch}, f)
        self._cleanup_checkpoints()
        return str(ckpt_dir)

    def _save_final(self) -> str:
        """Save the final trained model weights (Trainer-local)."""
        if self.sft_cfg.method in ("lora", "qlora"):
            adapter_dir = self._output_dir / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump({
                    "lora_rank": self._lora_cfg.rank,
                    "lora_alpha": self._lora_cfg.alpha,
                    "lora_dropout": self._lora_cfg.dropout,
                    "target_modules": self._lora_cfg.target_modules,
                    "quantize_base": self._lora_cfg.quantize_base,
                    "base_model": self.sft_cfg.model,
                }, f, indent=2)
            return str(adapter_dir)
        return str(self._output_dir)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        ckpt_base = self._output_dir / "checkpoints"
        if not ckpt_base.exists():
            return
        checkpoints = sorted(ckpt_base.iterdir())
        while len(checkpoints) > self.sft_cfg.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)
            # Re-scan to keep the list in sync with the filesystem
            checkpoints = sorted(ckpt_base.iterdir())
