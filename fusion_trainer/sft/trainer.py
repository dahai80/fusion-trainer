"""SFT Trainer — supervised fine-tuning on Apple MLX.

This module wraps `mlx-lm` training capabilities with a Fusion-Trainer
interface, supporting full fine-tune, LoRA, and QLoRA methods.

Fusion-Trainer is a pure computation layer. It reads datasets from
external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ..mlx_client import MLXFineTuneError
from .lora import LoRAConfig

if TYPE_CHECKING:
    from ..experiment import ExperimentLogger

logger = logging.getLogger(__name__)

# #65: numbered checkpoints written by mlx_lm every steps_per_save steps,
# e.g. 0000700_adapters.safetensors.
_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")

# fusion-mlx ADAPTER_BASE_DIR (fusion_mlx/training/service.py).
MLX_ADAPTER_BASE_DIR = Path.home() / ".fusion-mlx" / "adapters"


def find_latest_checkpoint(adapter_dir: Path | str) -> tuple[Path, int] | None:
    # #65: locate the newest numbered checkpoint in a fusion-mlx adapter
    # dir. Returns (checkpoint_file, step) or None when the dir holds no
    # numbered checkpoint (fresh run or cleaned up).
    best: tuple[int, Path] | None = None
    for p in Path(adapter_dir).glob("*_adapters.safetensors"):
        m = _CHECKPOINT_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, p)
    if best is None:
        return None
    return best[1], best[0]


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

        # #65 resume state (resolved in _run_training before job submission)
        self._resume_dir: str = ""
        self._resume_step: int = 0
        self._last_persisted_step: int = -1
        # issue #75: (step, val_loss) pairs from job events — the val_loss
        # fallback signal for checkpoint selection after training.
        self._val_losses: list[tuple[int, float]] = []

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

        # Step 1: Load dataset (schema-aware: chat formats route through
        # load_messages — the flat load drops messages/ShareGPT samples)
        print("[1/4] Loading dataset...")
        train_data, val_data = self._loader.load_sft(
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
        model_path = self.sft_cfg.model
        if not model_path:
            raise ValueError("No model specified. Set --model or sft.model in config.")

        print(f"      Checking model via fusion-mlx: {model_path}")

        # E3 (round-2 audit): model validation was a sham — raw httpx.get to
        # localhost (broke multi-node + bypassed the pooled client), empty
        # model_ids silently skipped the check, and `except Exception` swallowed
        # connection/JSON/timeout errors with only a print. Use the pooled
        # client's health() (base_url-aware), warn when no models are loaded,
        # and catch only the transport/MLX error types so real bugs surface.
        from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

        client = MLXFineTuneClient(port=self.config.mlx_port)
        try:
            try:
                info = client.health()
            except MLXFineTuneError as e:
                logger.warning("could not reach fusion-mlx to verify model: %s", e)
                return None, None
            models = info.get("data", []) if isinstance(info, dict) else []
            model_ids = {m.get("id", "") for m in models if isinstance(m, dict)}
            if not model_ids:
                logger.warning(
                    "fusion-mlx reachable but no models loaded — cannot verify "
                    "'%s' is registered; training will fail at job submit if the "
                    "model is missing",
                    model_path,
                )
            elif model_path not in model_ids:
                available = ", ".join(sorted(x for x in model_ids if x))
                logger.warning(
                    "model '%s' not found in fusion-mlx. Available: %s",
                    model_path,
                    available,
                )
        finally:
            client.close()

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
            flat = {"input": str(inp), "output": str(out)}
            # #69: keep the system field so stage_dataset emits it as a
            # chat message (models are served with a system prompt).
            if item.get("system"):
                flat["system"] = str(item["system"])
            formatted.append(flat)
        return formatted

    def _build_mlx_config(self) -> dict[str, Any]:
        """Map SFTConfig → fusion-mlx FineTuneConfig dict.

        fusion-mlx #402/#425 已落地:fine_tune_type = lora | dora | full | qlora,
        FineTuneConfig 含 quantize_base / quant_bits / mxfp8 字段。
        - QLoRA:fine_tune_type="qlora" + quantize_base=True,真实量化基座。
        - MXFP8:上游 service.py 路由到 8-bit QLoRA 路径(quantize_base=True,
          quant_bits=8, fine_tune_type=qlora)。MLX 0.32 无原生 fp8 dtype,
          语义=8-bit frozen base + LoRA 省显存,非 fp8 计算。mxfp8+full 矛盾
          (full 解冻基座无物可量化),上游 fail-visible raise。本仓库透传 mxfp8
          字段,由上游决定语义。
        """
        method = self.sft_cfg.method

        if method == "qlora":
            fine_tune_type = "qlora"
        elif method == "qdora":
            # #63: quantized DoRA — fine_tune_type="dora" + quantize_base=True
            # + quant_bits. fusion-mlx quantizes the frozen base then attaches
            # DoRA adapters (mlx_lm 0.31.3 DoRALinear.from_base accepts
            # QuantizedLinear — verified forward + grads flow). PRD baseline
            # for Qwen2.5-32B BNUP training.
            fine_tune_type = "dora"
        elif method == "full":
            fine_tune_type = "full"
        elif method == "dora":
            # P1-5 (round-3 audit): DoRA was a declared-but-unreachable dead
            # field (use_dora never read, method had no "dora"). Upstream
            # fusion-mlx #402 accepts fine_tune_type=dora; wire it through.
            fine_tune_type = "dora"
        else:
            fine_tune_type = "lora"

        total_samples = max(1, len(self._train_samples))
        steps_per_epoch = max(1, total_samples // max(1, self.sft_cfg.batch_size))
        iters = self.sft_cfg.max_steps or (steps_per_epoch * self.sft_cfg.num_epochs)

        # #65: fusion-mlx #1182 restarts the step counter at 0 on resume, so
        # subtract the already-trained steps to avoid over-training.
        if self._resume_step > 0:
            logger.info(
                "resume: %d steps already trained; iters %d -> %d",
                self._resume_step,
                iters,
                max(iters - self._resume_step, 0),
            )
            iters = max(iters - self._resume_step, 0)

        # A7: fusion-mlx FineTuneConfig accepts ONLY its declared fields
        # (FineTuneConfig(**body) raises TypeError on unknown keys → 400).
        # Confirmed supported: lr_schedule (dict), quant_bits. NOT supported
        # (upstream has no field): lora_target_modules (only lora_layers count),
        # weight_decay, max_grad_norm. The old code silently dropped ALL of
        # these. Now: lr_schedule + quant_bits are wired through; the
        # unsupported three warn LOUDLY when the user set a non-default value
        # (fail-visible, never silent drop) — upstream issues filed.
        unsupported_warned: list[str] = []
        if (
            self.sft_cfg.lora_target_modules
            and self.sft_cfg.lora_target_modules
            not in (["q_proj", "v_proj", "k_proj", "o_proj"],)
        ):
            unsupported_warned.append(
                f"lora_target_modules={self.sft_cfg.lora_target_modules} "
                "(fusion-mlx only supports a layer COUNT via lora_layers, "
                "not per-module targeting — ignored)"
            )
        if self.sft_cfg.weight_decay != 0.01:
            unsupported_warned.append(
                f"weight_decay={self.sft_cfg.weight_decay} "
                "(fusion-mlx optimizer_config is empty — ignored)"
            )
        if self.sft_cfg.max_grad_norm != 1.0:
            unsupported_warned.append(
                f"max_grad_norm={self.sft_cfg.max_grad_norm} "
                "(fusion-mlx has no gradient-clip field — ignored)"
            )
        for msg in unsupported_warned:
            logger.warning(
                "SFT config field not consumed by fusion-mlx: %s. "
                "Training proceeds with fusion-mlx defaults for this field. "
                "Upstream support tracked in fusion-mlx issues.",
                msg,
            )

        lr_schedule = self._build_lr_schedule(iters)

        # Tiny runs (smoke tests, demos) have fewer steps than log_every —
        # report every step then, or the run completes with zero progress
        # events and the UI shows 0/3.
        steps_per_report = max(1, min(self.sft_cfg.log_every, iters))

        cfg: dict[str, Any] = {
            "fine_tune_type": fine_tune_type,
            "optimizer": "adamw",
            "learning_rate": self.sft_cfg.learning_rate,
            "batch_size": self.sft_cfg.batch_size,
            "iters": iters,
            "val_batches": 5,
            "steps_per_report": steps_per_report,
            "steps_per_eval": max(steps_per_report * 5, 50),
            "steps_per_save": self.sft_cfg.save_every,
            "max_seq_length": self.dataset_cfg.max_seq_length,
            "grad_accumulation_steps": self.sft_cfg.gradient_accumulation_steps,
            "mask_prompt": True,
            "seed": self.config.seed,
        }
        # Explicit memory budget pass-through (fusion-mlx FineTuneConfig.
        # memory_limit_gb). 0 keeps the upstream default.
        if self.sft_cfg.memory_limit_gb > 0:
            cfg["memory_limit_gb"] = float(self.sft_cfg.memory_limit_gb)
        # P3-4 (round-3 audit): only send LoRA fields for lora/dora/qlora. A
        # full fine-tune sending lora_rank/lora_layers is meaningless and could
        # confuse upstream validation. Strip them on the full path.
        if fine_tune_type != "full":
            cfg["lora_rank"] = self._lora_cfg.rank
            cfg["lora_alpha"] = self._lora_cfg.alpha
            cfg["lora_dropout"] = self._lora_cfg.dropout
            cfg["lora_layers"] = 16
        # A7: pass the lr_schedule dict (warmup + cosine/linear) through to
        # fusion-mlx's build_schedule. Was previously dropped → no warmup, no
        # decay despite the preset specifying warmup_steps/lr_scheduler.
        if lr_schedule is not None:
            cfg["lr_schedule"] = lr_schedule
        if method in ("qlora", "qdora"):
            # A7: send quant_bits from config (was hardcoded 4 → 8-bit QLoRA
            # impossible). fusion-mlx validates quant_bits in (4, 8).
            # #63: qdora shares the gate — same quantized-base mechanics,
            # DoRA adapters instead of LoRA on top.
            qb = int(self.sft_cfg.quant_bits)
            if qb not in (4, 8):
                logger.warning(
                    "quant_bits=%d not in (4,8); clamping to 4 for %s", qb, method
                )
                qb = 4
            cfg["quantize_base"] = True
            cfg["quant_bits"] = qb
            logger.info(
                "%s: fine_tune_type=%s quantize_base=True quant_bits=%d",
                method.upper(),
                fine_tune_type,
                qb,
            )
        # #402/#425 已落地:透传 mxfp8。上游路由到 8-bit QLoRA(quantize_base=True,
        # quant_bits=8, fine_tune_type=qlora),语义=8-bit frozen base LoRA 省显存,
        # 非 fp8 计算。mxfp8+full 矛盾→上游 fail-visible raise。
        if self.sft_cfg.use_mxfp8:
            if fine_tune_type == "full":
                raise MLXFineTuneError(
                    "use_mxfp8 is incompatible with method='full' (full fine-tuning "
                    "unfreezes the base, so there is no frozen base to quantize); "
                    "use lora/dora/qlora with --mxfp8 (fusion-mlx #425)."
                )
            cfg["mxfp8"] = True
            logger.info(
                "MXFP8: passing mxfp8=True → fusion-mlx routes to 8-bit QLoRA "
                "(8-bit frozen base + LoRA for memory saving, NOT fp8 compute, "
                "MLX 0.32 has no fp8 dtype; fusion-mlx #425)."
            )
        # #75/#1400: sequence packing — fusion-mlx 0.10.17 supplies the
        # packed batch iterator + block-diagonal masked loss. Only sent
        # when explicitly on: older fusion-mlx (< 0.10.17) has no
        # FineTuneConfig.pack_sequences and would 400 on the unknown key.
        if self.sft_cfg.pack_sequences:
            cfg["pack_sequences"] = True
            logger.info(
                "Sequence packing enabled: rows up to max_seq_length=%d, "
                "block-diagonal attention mask (fusion-mlx #1400)",
                self.dataset_cfg.max_seq_length,
            )
        # #65: resume an interrupted run — fusion-mlx #1182 loads
        # adapters.safetensors from this dir into the freshly built LoRA
        # layers before train(). _resume_dir is staged in _run_training
        # (numbered checkpoints are copied to resume_stage/adapters.safetensors
        # because upstream loads exactly that filename).
        if self._resume_dir:
            cfg["init_adapter_path"] = self._resume_dir
        return cfg

    def _build_lr_schedule(self, iters: int) -> dict[str, Any] | None:
        """A7: build the mlx-lm lr_schedule dict from lr_scheduler + warmup_steps.

        mlx-lm build_schedule expects {"name": <fn>, "arguments": [...],
        "warmup": <steps>, "warmup_init": 0.0}. Supported schedulers:
        cosine_decay(init, decay_steps, end=0.0), linear_schedule(init, end,
        steps), exponential_decay(init, decay_rate), step_decay(init,
        decay_rate, step_size). Returns None for "constant"/unknown → fusion-mlx
        uses the flat learning_rate.
        """
        name = (self.sft_cfg.lr_scheduler or "").lower().strip()
        lr = float(self.sft_cfg.learning_rate)
        warmup = max(0, int(self.sft_cfg.warmup_steps))
        if name in ("", "constant", "none"):
            if warmup > 0:
                # Warmup only, then hold at lr: linear_schedule(lr, lr, warmup)
                # would decay to lr over warmup steps then stay — but we want
                # warmup THEN constant. Use a linear warmup to lr via the
                # warmup key with a constant base. mlx-lm joins warmup_fn with
                # bound_schedule_fn at warmup+1; for constant, bound = cosine
                # decay over a huge horizon so it stays ~lr.
                return {
                    "name": "cosine_decay",
                    "arguments": [lr, max(iters * 100, 1), lr],
                    "warmup": warmup,
                    "warmup_init": 0.0,
                }
            return None
        if name in ("cosine", "cosine_decay"):
            return {
                "name": "cosine_decay",
                "arguments": [lr, max(iters, 1), 0.0],
                "warmup": warmup,
                "warmup_init": 0.0,
            }
        if name in ("linear", "linear_decay"):
            return {
                "name": "linear_schedule",
                "arguments": [lr, 0.0, max(iters, 1)],
                "warmup": warmup,
                "warmup_init": 0.0,
            }
        logger.warning(
            "lr_scheduler=%r not mapped to an mlx-lm schedule; using flat lr",
            self.sft_cfg.lr_scheduler,
        )
        return None

    def _build_vlm_config(self) -> dict[str, Any]:
        """Map SFTConfig → fusion-mlx VLMFineTuneConfig dict (#55, #797/#800).

        VLMFineTuneConfig (fusion_mlx/training/vlm_service.py) drives
        mlx_vlm.trainer SFT/LoRA. Its field set is DIFFERENT from the text
        FineTuneConfig — it has no lr_schedule, mask_prompt, quant_bits,
        quantize_base, mxfp8, or lora_target_modules. Sending any of those
        = upstream 400 (TypeError on VLMFineTuneConfig(**body)). Only the
        declared fields are emitted here.
        """
        total_samples = max(1, len(self._train_samples))
        steps_per_epoch = max(1, total_samples // max(1, self.sft_cfg.batch_size))
        iters = self.sft_cfg.max_steps or (steps_per_epoch * self.sft_cfg.num_epochs)

        method = self.sft_cfg.method
        if method == "qlora":
            # VLMFineTuneConfig has NO quantize field — mlx-vlm QLoRA
            # semantics differ from mlx-lm and are not exposed upstream
            # (#797). Degrade to plain LoRA + fail-visible warning so the
            # user knows VLM QLoRA did not happen.
            logger.warning(
                "VLM method='qlora' is not supported by fusion-mlx's VLM "
                "endpoint (VLMFineTuneConfig has no quantize field, #797); "
                "falling back to plain LoRA (full_finetune=False). "
                "No base quantization will occur."
            )
            full_finetune = False
        elif method == "full":
            full_finetune = True
        else:
            full_finetune = False

        cfg: dict[str, Any] = {
            "lora_rank": self._lora_cfg.rank,
            "lora_alpha": self._lora_cfg.alpha,
            "lora_dropout": self._lora_cfg.dropout,
            "lora_layers": 16,
            "full_finetune": full_finetune,
            "train_vision": False,
            "optimizer": "adamw",
            "learning_rate": self.sft_cfg.learning_rate,
            "batch_size": self.sft_cfg.batch_size,
            "iters": iters,
            "val_batches": 5,
            "steps_per_report": self.sft_cfg.log_every,
            "steps_per_eval": max(self.sft_cfg.log_every * 5, 50),
            "steps_per_save": self.sft_cfg.save_every,
            "max_seq_length": self.dataset_cfg.max_seq_length,
            "grad_checkpoint": False,
            "grad_accumulation_steps": self.sft_cfg.gradient_accumulation_steps,
            "seed": self.config.seed,
            "num_processes": 1,
        }
        logger.info(
            "VLM config built: full_finetune=%s lora_rank=%d iters=%d "
            "batch_size=%d lr=%s (no lr_schedule/mask_prompt/quant fields — "
            "VLMFineTuneConfig does not accept them)",
            full_finetune,
            self._lora_cfg.rank,
            iters,
            self.sft_cfg.batch_size,
            self.sft_cfg.learning_rate,
        )
        return cfg

    def _resolve_vlm_dataset_ref(self) -> str:
        """Resolve the dataset reference for the VLM endpoint.

        mlx_vlm.lora.load_dataset accepts a HF dataset id string OR a local
        dir (images + jsonl). Unlike text SFT, the VLM endpoint consumes the
        raw reference directly — NOT the staged chat-messages jsonl that
        stage_dataset produces. Resolve from dataset_cfg: a direct path wins
        (local dir or HF id passed as path); else the hub convention
        ({model_hub_path}/datasets/{hub_dataset}) as a directory reference.
        """
        if self.dataset_cfg.path:
            return self.dataset_cfg.path
        if self.dataset_cfg.hub_dataset:
            hub = self.config.model_hub_path
            if not hub:
                raise ValueError(
                    "hub_dataset requires model_hub_path to be set for VLM "
                    "(set FUSION_MODEL_HUB or pass model_hub_path)."
                )
            # Reuse the same path-traversal guard as DatasetLoader._resolve_path.
            import os
            from pathlib import Path as _Path

            if os.path.isabs(self.dataset_cfg.hub_dataset):
                raise ValueError(
                    f"hub_dataset must be a relative name, got absolute: "
                    f"{self.dataset_cfg.hub_dataset!r}"
                )
            if any(p == ".." for p in _Path(self.dataset_cfg.hub_dataset).parts):
                raise ValueError(
                    f"hub_dataset must not contain '..' segments: "
                    f"{self.dataset_cfg.hub_dataset!r}"
                )
            hub_root = _Path(hub).resolve()
            datasets_dir = (hub_root / "datasets").resolve()
            candidate = (datasets_dir / self.dataset_cfg.hub_dataset).resolve()
            try:
                candidate.relative_to(datasets_dir)
            except ValueError:
                raise ValueError(
                    f"hub_dataset {self.dataset_cfg.hub_dataset!r} resolves "
                    f"outside the hub datasets directory — path traversal blocked"
                )
            if not candidate.exists():
                raise FileNotFoundError(
                    f"VLM dataset '{self.dataset_cfg.hub_dataset}' not found "
                    f"in hub at {hub}/datasets/"
                )
            return str(candidate)
        raise ValueError(
            "VLM training requires a dataset: set dataset.path (HF id or local "
            "dir) or dataset.hub_dataset."
        )

    def _submit_vlm_job(self, exp_logger: ExperimentLogger) -> dict[str, Any]:
        """Submit + poll a VLM fine-tune job (#55, fusion-mlx #797/#800).

        Pure HTTP delegation to /admin/api/fine-tune/vlm/jobs. Does NOT call
        stage_dataset (mlx_vlm consumes the raw dataset ref). Shares the
        _ingest_progress event path with text SFT so the ExperimentLogger +
        GUI sink see the same loss/val curves.
        """
        from ..mlx_client import MLXFineTuneClient

        mlx_cfg = self._build_vlm_config()
        adapter_name = self.sft_cfg.run_name or "vlm_run"
        dataset_ref = self._resolve_vlm_dataset_ref()

        client = MLXFineTuneClient(port=self.config.mlx_port)
        job = client.create_vlm_job(
            model_id=self.sft_cfg.model,
            dataset=dataset_ref,
            config=mlx_cfg,
            adapter_name=adapter_name,
        )
        job_id = job.get("job_id", "")
        print(f"      Submitted VLM fine-tune job: {job_id}")
        print(
            f"      iters={mlx_cfg['iters']} batch_size={mlx_cfg['batch_size']} "
            f"lr={mlx_cfg['learning_rate']} full_finetune="
            f"{mlx_cfg['full_finetune']}"
        )
        print(f"      dataset_ref={dataset_ref}")
        print(f"      {'=' * 50}")

        if self.on_event is not None:
            self.on_event(
                {
                    "type": "job",
                    "job_id": job_id,
                    "kind": "vlm",
                    "iters": mlx_cfg["iters"],
                }
            )

        self._global_step = 0
        try:
            final_job = client.wait_for_vlm_job(
                job_id,
                on_progress=lambda j, ev: self._ingest_progress(j, ev, exp_logger),
                poll_interval=2.0,
                is_canceled=getattr(self, "is_canceled", None),
            )
        except MLXFineTuneError as exc:
            print(f"      ✗ VLM training failed: {exc}")
            return {
                "status": "failed",
                "error": str(exc),
                "output_dir": str(self._output_dir),
                "method": self.sft_cfg.method,
                "model_type": "vlm",
            }

        status = final_job.get("status", "")
        if status != "completed":
            print(f"      ✗ VLM job ended with status: {status}")
            return {
                "status": status,
                "error": final_job.get("error", ""),
                "output_dir": str(self._output_dir),
                "method": self.sft_cfg.method,
                "model_type": "vlm",
            }

        adapter_path = final_job.get("adapter_path", "")
        self._adapter_path = adapter_path
        self._save_final(adapter_path=adapter_path)

        result = {
            "status": "completed",
            "total_steps": self._global_step,
            "output_dir": str(self._output_dir),
            "final_model_path": adapter_path,
            "adapter_name": adapter_name,
            "method": self.sft_cfg.method,
            "model_type": "vlm",
            "lora_rank": self._lora_cfg.rank if self.sft_cfg.method != "full" else 0,
            "job_id": job_id,
        }
        print("\n      ✓ VLM training complete!")
        print(f"      Adapter: {adapter_path}")
        return result

    def _run_training(
        self,
        train_loader: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute training by delegating to fusion-mlx via HTTP.

        Real gradient computation happens inside fusion-mlx's
        FineTuneService (which wraps mlx_lm.tuner) for text LM models, or
        VLMFineTuneService (which wraps mlx_vlm.trainer) for vision-language
        models. This method:
          1. Stages the dataset to a local jsonl (text LM only)
          2. Submits a fine-tune job to /admin/api/fine-tune/jobs (text LM)
             or /admin/api/fine-tune/vlm/jobs (VLM)
          3. Polls job status, streaming real loss into ExperimentLogger
          4. Records the adapter path returned by fusion-mlx
        """
        from ..experiment import ExperimentLogger
        from ..mlx_client import MLXFineTuneClient

        self._train_samples = train_loader
        exp_logger = ExperimentLogger(
            run_name=self.sft_cfg.run_name or "sft_default",
            output_dir=self._output_dir,
        )

        # E4 (round-2 audit): wrap the whole body in try/finally so a non-
        # MLXFineTuneError (stage_dataset OSError, create_job ConnectError)
        # still flushes + closes the logger (metrics fd + TensorBoard async
        # writer thread + event fd) and closes the pooled client. Only
        # MLXFineTuneError was caught before, so other exceptions leaked fds.
        client = None
        try:
            # #55 (fusion-mlx #797/#800): model_type=="vlm" drives the VLM
            # fine-tune endpoint (mlx_vlm SFT/LoRA over HTTP). The VLM path
            # does NOT stage a chat-messages jsonl — mlx_vlm.lora.load_dataset
            # consumes a HF dataset id or a local dir of images + jsonl
            # directly. Branch early, keep one try/finally.
            if self.sft_cfg.model_type == "vlm":
                if self.sft_cfg.init_adapter_path:
                    logger.warning(
                        "VLM resume is not supported upstream "
                        "(VLMFineTuneConfig has no init_adapter_path) — "
                        "ignoring init_adapter_path=%s, starting fresh (#65)",
                        self.sft_cfg.init_adapter_path,
                    )
                result = self._submit_vlm_job(exp_logger)
                return result

            self._resolve_resume()

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

            # #65: resume landed on an already-finished step count — nothing
            # to train; surface the existing checkpoint instead of submitting
            # a zero-iter job.
            if self._resume_dir and int(mlx_cfg.get("iters", 0)) <= 0:
                ckpt = find_latest_checkpoint(
                    MLX_ADAPTER_BASE_DIR / self.sft_cfg.model / adapter_name
                )
                src = str(ckpt[0]) if ckpt else self._resume_dir
                print(
                    f"      ✓ Run already trained {self._resume_step} iters "
                    f"— nothing left to do."
                )
                print(f"      Adapter: {src}")
                return {
                    "status": "completed",
                    "resumed_noop": True,
                    "total_steps": self._resume_step,
                    "output_dir": str(self._output_dir),
                    "final_model_path": src,
                    "adapter_name": adapter_name,
                    "method": self.sft_cfg.method,
                }

            client = MLXFineTuneClient(port=self.config.mlx_port)
            job = client.create_job(
                model_id=self.sft_cfg.model,
                dataset=dataset_path,
                config=mlx_cfg,
                adapter_name=adapter_name,
            )
            job_id = job.get("job_id", "")
            self._persist_job_state("running", job_id)
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
                    is_canceled=getattr(self, "is_canceled", None),
                )
            except MLXFineTuneError as exc:
                if getattr(exc, "status_code", None) == 404:
                    # #65: engine restarted and dropped the persisted job.
                    # Fail NOW as interrupted — never zombie-poll to timeout.
                    ckpt = find_latest_checkpoint(
                        MLX_ADAPTER_BASE_DIR / self.sft_cfg.model / adapter_name
                    )
                    hint = f"{ckpt[0]} (step {ckpt[1]})" if ckpt else "(none)"
                    print(
                        f"      ✗ Run interrupted: job {job_id} lost "
                        f"(fusion-mlx restart dropped it). "
                        f"Latest checkpoint: {hint}"
                    )
                    print(
                        "        Rerun with --resume to continue from the checkpoint."
                    )
                    self._persist_job_state("interrupted", job_id)
                    exp_logger.save()
                    return {
                        "status": "interrupted",
                        "error": str(exc),
                        "job_id": job_id,
                        "resume_checkpoint": str(ckpt[0]) if ckpt else "",
                        "resume_step": ckpt[1] if ckpt else 0,
                        "output_dir": str(self._output_dir),
                        "method": self.sft_cfg.method,
                    }
                print(f"      ✗ Training failed: {exc}")
                self._persist_job_state("failed", job_id)
                exp_logger.save()
                return {
                    "status": "failed",
                    "error": str(exc),
                    "output_dir": str(self._output_dir),
                    "method": self.sft_cfg.method,
                }

            status = final_job.get("status", "")
            if status != "completed":
                # #65: a returned "cancelled" on this path is engine-side —
                # fusion-mlx restart recovery marks RUNNING/QUEUED jobs
                # cancelled (user cancel raises instead). Same interrupted
                # treatment: surface the checkpoint, suggest --resume.
                if status == "cancelled":
                    ckpt = find_latest_checkpoint(
                        MLX_ADAPTER_BASE_DIR / self.sft_cfg.model / adapter_name
                    )
                    hint = f"{ckpt[0]} (step {ckpt[1]})" if ckpt else "(none)"
                    print(
                        f"      ✗ Run interrupted: job {job_id} cancelled "
                        f"engine-side (fusion-mlx restart recovery). "
                        f"Latest checkpoint: {hint}"
                    )
                    print(
                        "        Rerun with --resume to continue from the checkpoint."
                    )
                    self._persist_job_state("interrupted", job_id)
                    exp_logger.save()
                    return {
                        "status": "interrupted",
                        "error": final_job.get("error", "") or "cancelled engine-side",
                        "job_id": job_id,
                        "resume_checkpoint": str(ckpt[0]) if ckpt else "",
                        "resume_step": ckpt[1] if ckpt else 0,
                        "output_dir": str(self._output_dir),
                        "method": self.sft_cfg.method,
                    }
                print(f"      ✗ Job ended with status: {status}")
                self._persist_job_state(status or "failed", job_id)
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
            self._persist_job_state("completed", job_id)
            exp_logger.save()

            result = {
                "status": "completed",
                "total_steps": self._global_step,
                "output_dir": str(self._output_dir),
                "final_model_path": adapter_path,
                "adapter_name": adapter_name,
                "method": self.sft_cfg.method,
                "lora_rank": self._lora_cfg.rank
                if self.sft_cfg.method != "full"
                else 0,
                "job_id": job_id,
            }
            print("\n      ✓ Training complete!")
            print(f"      Adapter: {adapter_path}")

            if self.sft_cfg.eval_select:
                selection = self._run_checkpoint_eval(adapter_name)
                if selection is not None:
                    result["checkpoint_selection"] = selection
            return result
        finally:
            # E4: flush + close logger + pooled client on EVERY exit path,
            # including non-MLXFineTuneError exceptions that previously leaked.
            try:
                exp_logger.save()
                exp_logger.close()
            except Exception as e:
                logger.warning("ExperimentLogger close failed: %s", e)
            if client is not None:
                client.close()

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
                # #65 pt4: persist last_seen_step incrementally (per step
                # change), not only at start/end — engine restarts bypass
                # atexit, so state is only as durable as the last poll.
                if step != self._last_persisted_step:
                    self._last_persisted_step = step
                    self._persist_job_state("running", job.get("job_id", ""))
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
                val_step = ev.get("step", self._global_step)
                self._val_losses.append((val_step, ev.get("val_loss", 0.0)))
                exp_logger.log_metrics(
                    step=val_step,
                    metrics={"val_loss": ev.get("val_loss", 0.0)},
                )
                print(f"      Val loss: {ev.get('val_loss', 0.0):.4f}")

    def _run_checkpoint_eval(self, adapter_name: str) -> dict[str, Any] | None:
        # issue #75: benchmark every numbered checkpoint and pick the
        # winner. Never raises — a bench outage falls back to val_loss
        # (persisted in the selection table) and the run result survives.
        from ..eval_loop import BenchClient, run_checkpoint_eval

        adapter_dir = MLX_ADAPTER_BASE_DIR / self.sft_cfg.model / adapter_name
        bench = None
        if self.sft_cfg.bench_url or self.sft_cfg.bench_api_key:
            bench = BenchClient(
                base_url=self.sft_cfg.bench_url,
                api_key=self.sft_cfg.bench_api_key,
                tenant_id=self.sft_cfg.bench_tenant_id,
            )
        else:
            logger.info(
                "eval_select on but no bench_url/bench_api_key configured "
                "— selecting on val_loss only"
            )
        print(f"      Evaluating checkpoints in {adapter_dir}")
        try:
            selection = run_checkpoint_eval(
                model=self.sft_cfg.model,
                adapter_dir=adapter_dir,
                val_losses=self._val_losses,
                bench=bench,
                bench_executor=self.sft_cfg.bench_executor,
                bench_suite=self.sft_cfg.bench_suite,
                bench_params=self.sft_cfg.bench_params or None,
                task_wait_timeout=self.sft_cfg.bench_task_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — eval must never kill the run
            logger.warning("checkpoint eval failed: %s", exc)
            print(f"      ⚠ Checkpoint eval failed: {exc}")
            return None
        for row in selection["table"]:
            mark = "← chosen" if row.get("chosen") else ""
            print(
                f"        {row.get('checkpoint', '?'):32s} "
                f"step={row.get('step', '-')}  "
                f"val_loss={row.get('val_loss', '-')}  "
                f"bench={row.get('bench_score', '-')} [{row.get('bench_status', '')}] "
                f"{mark}"
            )
        print(
            f"      Selection basis: {selection['basis']} "
            f"(table: {selection['path']})"
        )
        if selection.get("fallback_reason"):
            print(f"      ⚠ {selection['fallback_reason']}")
        return selection

    def _resolve_resume(self) -> None:
        # #65: stage the resume checkpoint for fusion-mlx #1182. Upstream
        # loads exactly {init_adapter_path}/adapters.safetensors, while the
        # run's own checkpoints are numbered (NNNNNN_adapters.safetensors) in
        # the same dir — copy the newest one to resume_stage/ so the resumed
        # job does not overwrite the source checkpoint at its first save.
        # Step counter restarts at 0 upstream; _resume_step lets
        # _build_mlx_config subtract already-trained iters.
        if not self.sft_cfg.init_adapter_path:
            return
        adapter_dir = Path(self.sft_cfg.init_adapter_path)
        ckpt = find_latest_checkpoint(adapter_dir)
        if ckpt is not None:
            src, step = ckpt
            staged = self._output_dir / "resume_stage"
            staged.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, staged / "adapters.safetensors")
            self._resume_dir = str(staged)
            self._resume_step = step
            logger.info("resume: checkpoint step %d staged %s -> %s", step, src, staged)
            print(f"      ↻ Resume: checkpoint step {step} ({src})")
        elif (adapter_dir / "adapters.safetensors").exists():
            # Explicit dir holding final weights (e.g. a completed run's
            # adapter). Step unknown — train the full iters on loaded weights.
            self._resume_dir = str(adapter_dir)
            self._resume_step = 0
            logger.info(
                "resume: adapters.safetensors in %s (step unknown)", adapter_dir
            )
            print(f"      ↻ Resume: adapter weights from {adapter_dir}")
        else:
            logger.warning(
                "init_adapter_path=%s holds no adapters.safetensors and no "
                "numbered checkpoint — starting fresh (#65)",
                adapter_dir,
            )
            print(f"      ⚠ --resume: no checkpoint in {adapter_dir}; starting fresh")

    def _persist_job_state(self, status: str, job_id: str = "") -> None:
        # #65 (pt3/4): liveness state under output_dir so a restarted trainer
        # can decide resume-vs-fresh without operator archaeology. Written
        # incrementally (on submit + every step change + on every terminal
        # path) because engine-side os._exit bypasses atexit persistence —
        # trainer-side state is only as durable as the last poll.
        adapter_name = self.sft_cfg.run_name or "sft_run"
        state = {
            "job_id": job_id,
            "model": self.sft_cfg.model,
            "adapter_name": adapter_name,
            "adapter_dir": str(
                MLX_ADAPTER_BASE_DIR / self.sft_cfg.model / adapter_name
            ),
            "step": self._global_step,
            "resume_step": self._resume_step,
            "status": status,
            "updated_at": time.time(),
        }
        p = self._output_dir / "job_state.json"
        tmp = p.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            os.replace(tmp, p)
        except OSError as e:
            logger.warning("job state persist failed: %s", e)

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
