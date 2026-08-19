"""SFT Trainer — supervised fine-tuning on Apple MLX.

This module wraps `mlx-lm` training capabilities with a Fusion-Trainer
interface, supporting full fine-tune, LoRA, and QLoRA methods.

Fusion-Trainer is a pure computation layer. It reads datasets from
external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from .lora import LoRAConfig

if TYPE_CHECKING:
    from ..experiment import ExperimentLogger

logger = logging.getLogger(__name__)


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
        self._train_samples: list[dict[str, Any]] = []
        self._adapter_path: str = ""

        # Optional live-progress sink (RunManager). If set, raw fusion-mlx job
        # events are forwarded here in addition to ExperimentLogger, so the GUI
        # can stream loss curves. None = CLI mode (no sink).
        self.on_event = None

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
            remove_duplicates=self.dataset_cfg.remove_duplicates,
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
            raise ValueError("No model specified. Set --model or sft.model in config.")

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
                print(
                    f"      ⚠ Model '{model_path}' not found in fusion-mlx. Available: {available}"
                )
        except Exception as e:
            print(f"      ⚠ Could not reach fusion-mlx to verify model: {e}")

        # Return dummy model/tokenizer handles — the real training loop
        # will load them via fusion-mlx training API when implemented.
        return None, None

    def _prepare_dataloader(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare the training data loader.

        Preserves {input, output} so stage_dataset can emit chat messages
        (required by mask_prompt). Pre-flattening to {text} breaks prompt
        masking because the user/assistant boundary is lost.
        """
        formatted = []
        for item in data:
            if "messages" in item or "text" in item:
                formatted.append(item)
                continue
            inp = item.get("input", "")
            out = item.get("output", "")
            formatted.append({"input": str(inp), "output": str(out)})
        return formatted

    def _build_mlx_config(self) -> dict[str, Any]:
        """Map SFTConfig → fusion-mlx FineTuneConfig dict.

        fusion-mlx #402 已落地:fine_tune_type = lora | dora | full | qlora,
        FineTuneConfig 含 quantize_base / quant_bits / mxfp8 字段。
        - QLoRA:fine_tune_type="qlora" + quantize_base=True,真实量化基座。
        - MXFP8:上游 service.py 仍 raise(字段已加但训练未实现)。本仓库同步
          fail-visible,不静默降精度(Rule 12)。
        """
        method = self.sft_cfg.method
        upstream_issue = "https://github.com/dahai80/fusion-mlx/issues/402"

        if self.sft_cfg.use_mxfp8:
            # 上游 FineTuneConfig.mxfp8 字段已存在,但 service.py 仍 raise
            # "mxfp8 mixed-precision training is not yet supported"。
            # 不向上游发送 mxfp8=True(否则 HTTP 报错);按默认精度训练并告警。
            # 仍是 fail-visible:告警写明上游未落地,不静默吞掉用户意图(Rule 12)。
            logger.warning(
                "MXFP8 mixed-precision training is not yet supported by fusion-mlx "
                "(service.py still raises). Training will run at default precision "
                "— peak memory may exceed the preset estimate. Set use_mxfp8=False "
                "to silence. Tracked in %s.",
                upstream_issue,
            )

        if method == "qlora":
            fine_tune_type = "qlora"
        elif method == "full":
            fine_tune_type = "full"
        else:
            fine_tune_type = "lora"

        total_samples = max(1, len(self._train_samples))
        steps_per_epoch = max(1, total_samples // max(1, self.sft_cfg.batch_size))
        iters = self.sft_cfg.max_steps or (steps_per_epoch * self.sft_cfg.num_epochs)

        cfg: dict[str, Any] = {
            "fine_tune_type": fine_tune_type,
            "lora_rank": self._lora_cfg.rank,
            "lora_alpha": self._lora_cfg.alpha,
            "lora_dropout": self._lora_cfg.dropout,
            "lora_layers": 16,
            "optimizer": "adamw",
            "learning_rate": self.sft_cfg.learning_rate,
            "batch_size": self.sft_cfg.batch_size,
            "iters": iters,
            "val_batches": 5,
            "steps_per_report": self.sft_cfg.log_every,
            "steps_per_eval": max(self.sft_cfg.log_every * 5, 50),
            "steps_per_save": self.sft_cfg.save_every,
            "max_seq_length": self.dataset_cfg.max_seq_length,
            "grad_accumulation_steps": self.sft_cfg.gradient_accumulation_steps,
            "mask_prompt": True,
            "seed": self.config.seed,
        }
        if fine_tune_type == "qlora":
            # QLoRA:量化基座 + LoRA。quant_bits 默认 4(上游 FineTuneConfig.quant_bits)。
            cfg["quantize_base"] = True
            cfg["quant_bits"] = 4
            logger.info("QLoRA: fine_tune_type=qlora quantize_base=True quant_bits=4")
        return cfg

    def _run_training(
        self,
        train_loader: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute training by delegating to fusion-mlx via HTTP.

        Real gradient computation happens inside fusion-mlx's
        FineTuneService (which wraps mlx_lm.tuner). This method:
          1. Stages the dataset to a local jsonl
          2. Submits a fine-tune job to /admin/api/fine-tune/jobs
          3. Polls job status, streaming real loss into ExperimentLogger
          4. Records the adapter path returned by fusion-mlx
        """
        from ..experiment import ExperimentLogger
        from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

        self._train_samples = train_loader
        exp_logger = ExperimentLogger(
            run_name=self.sft_cfg.run_name or "sft_default",
            output_dir=self._output_dir,
        )

        staged_dir = self._output_dir / "staged_data"
        dataset_path = MLXFineTuneClient.stage_dataset(
            train_loader, staged_dir, filename="train.jsonl"
        )
        if val_data:
            MLXFineTuneClient.stage_dataset(
                val_data, staged_dir, filename="valid.jsonl"
            )

        mlx_cfg = self._build_mlx_config()
        adapter_name = self.sft_cfg.run_name or "sft_run"

        client = MLXFineTuneClient(port=self.config.mlx_port)
        job = client.create_job(
            model_id=self.sft_cfg.model,
            dataset=dataset_path,
            config=mlx_cfg,
            adapter_name=adapter_name,
        )
        job_id = job.get("job_id", "")
        print(f"      Submitted fine-tune job: {job_id}")
        print(
            f"      iters={mlx_cfg['iters']} batch_size={mlx_cfg['batch_size']} "
            f"lr={mlx_cfg['learning_rate']} type={mlx_cfg['fine_tune_type']}"
        )
        print(f"      {'=' * 50}")

        if self.on_event is not None:
            self.on_event(
                {
                    "type": "job",
                    "job_id": job_id,
                    "kind": "sft",
                    "iters": mlx_cfg["iters"],
                }
            )

        self._global_step = 0
        try:
            final_job = client.wait_for_job(
                job_id,
                on_progress=lambda j, ev: self._ingest_progress(j, ev, exp_logger),
                poll_interval=2.0,
            )
        except MLXFineTuneError as exc:
            print(f"      ✗ Training failed: {exc}")
            exp_logger.save()
            return {
                "status": "failed",
                "error": str(exc),
                "output_dir": str(self._output_dir),
                "method": self.sft_cfg.method,
            }

        status = final_job.get("status", "")
        if status != "completed":
            print(f"      ✗ Job ended with status: {status}")
            exp_logger.save()
            return {
                "status": status,
                "error": final_job.get("error", ""),
                "output_dir": str(self._output_dir),
                "method": self.sft_cfg.method,
            }

        adapter_path = final_job.get("adapter_path", "")
        self._adapter_path = adapter_path
        self._save_final(adapter_path=adapter_path)
        exp_logger.save()

        result = {
            "status": "completed",
            "total_steps": self._global_step,
            "output_dir": str(self._output_dir),
            "final_model_path": adapter_path,
            "adapter_name": adapter_name,
            "method": self.sft_cfg.method,
            "lora_rank": self._lora_cfg.rank if self.sft_cfg.method != "full" else 0,
            "job_id": job_id,
        }
        print("\n      ✓ Training complete!")
        print(f"      Adapter: {adapter_path}")
        return result

    def _ingest_progress(
        self,
        job: dict[str, Any],
        new_events: list[dict[str, Any]],
        exp_logger: ExperimentLogger,
    ) -> None:
        """Convert fusion-mlx job events into ExperimentLogger metrics."""
        for ev in new_events:
            etype = ev.get("type", "")
            if self.on_event is not None:
                self.on_event(dict(ev))
            if etype == "train_loss":
                step = ev.get("step", 0)
                self._global_step = step
                exp_logger.log_metrics(
                    step=step,
                    metrics={
                        "loss": ev.get("train_loss", 0.0),
                        "learning_rate": ev.get("learning_rate", 0.0),
                        "tokens_per_second": ev.get("tokens_per_second", 0.0),
                        "it_sec": ev.get("it_sec", 0.0),
                        "peak_memory_gb": ev.get("peak_memory_gb", 0.0),
                    },
                )
                print(
                    f"      Step {step:6d}  "
                    f"Loss: {ev.get('train_loss', 0.0):.4f}  "
                    f"LR: {ev.get('learning_rate', 0.0):.2e}  "
                    f"tok/s: {ev.get('tokens_per_second', 0.0):.0f}"
                )
            elif etype == "val_loss":
                exp_logger.log_metrics(
                    step=ev.get("step", self._global_step),
                    metrics={"val_loss": ev.get("val_loss", 0.0)},
                )
                print(f"      Val loss: {ev.get('val_loss', 0.0):.4f}")

    def _save_checkpoint(self, step: int, logger: ExperimentLogger) -> str:
        """Save a training checkpoint (Trainer-local, not in hub)."""
        ckpt_dir = self._output_dir / "checkpoints" / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "checkpoint.json", "w") as f:
            json.dump({"step": step, "epoch": self._epoch}, f)
        self._cleanup_checkpoints()
        return str(ckpt_dir)

    def _save_final(self, adapter_path: str = "") -> str:
        """Record final adapter metadata (Trainer-local pointer).

        The real adapter weights live in fusion-mlx's adapter directory
        (~/.fusion-mlx/adapters/{model}/{name}/). This writes a local
        pointer config so the Trainer output_dir is self-describing.
        """
        if self.sft_cfg.method in ("lora", "qlora"):
            adapter_dir = self._output_dir / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(
                    {
                        "lora_rank": self._lora_cfg.rank,
                        "lora_alpha": self._lora_cfg.alpha,
                        "lora_dropout": self._lora_cfg.dropout,
                        "target_modules": self._lora_cfg.target_modules,
                        "quantize_base": self._lora_cfg.quantize_base,
                        "base_model": self.sft_cfg.model,
                        "adapter_path": adapter_path,
                        "loadable_via": "fusion-mlx --lora-path or request.adapters",
                    },
                    f,
                    indent=2,
                )
            return adapter_path or str(adapter_dir)
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
