"""Voice (TTS voice-pack) trainer — issue #74, wired on fusion-mlx #1336.

Trains a Kokoro voice package from paired (audio, transcript) samples by
delegating to fusion-mlx's /admin/api/fine-tune/voice/jobs endpoint. Same
pure-HTTP-delegation rule as every other trainer: fusion-trainer never
imports mlx_audio. The job contract mirrors SFT/DPO (submit → poll →
cancel); the trained package is served directly by fusion-mlx as a
`voice_ref` on POST /v1/audio/speech.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import asdict, dataclass
from typing import Any

from .dataset.voice_prep import validate_voice_dataset
from .mlx_client import MLXFineTuneClient, MLXFineTuneError

logger = logging.getLogger(__name__)

VOICE_FAMILY = "kokoro"


@dataclass
class VoiceTrainConfig:
    # Field set mirrors upstream fusion_mlx.training.voice.VoiceTrainConfig
    # (SPSA voice-pack optimization). Unknown keys 400 upstream, so keep
    # this list exactly aligned.
    steps: int = 300
    learning_rate: float = 0.02
    perturbation: float = 0.005
    decay: float = 0.101
    base_voice: str = "af_heart"
    lang_code: str = "a"
    max_samples: int = 0
    dynamics_weight: float = 0.5
    level_weight: float = 0.3
    anchor_weight: float = 0.02
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceTrainer:
    """Orchestrate a voice fine-tune job on fusion-mlx (#1336 endpoint)."""

    def __init__(
        self,
        model: str,
        dataset_dir: str,
        voice_id: str = "",
        tenant_id: str = "default",
        config: VoiceTrainConfig | None = None,
        mlx_port: int | None = None,
    ):
        if not model:
            raise ValueError("voice training needs a fusion-mlx model id")
        if VOICE_FAMILY not in model.lower():
            # Upstream enforces the same gate (only kokoro ships a trainable
            # voice pack); fail here so the user learns before a round trip.
            raise ValueError(
                f"voice fine-tune supports the {VOICE_FAMILY} family only, "
                f"got {model!r}"
            )
        self._model = model
        self._dataset_dir = dataset_dir
        self._voice_id = voice_id
        self._tenant_id = tenant_id
        self._config = config or VoiceTrainConfig()
        from .config import _default_mlx_port

        self._client = MLXFineTuneClient(port=mlx_port or _default_mlx_port())

        # Optional live-progress sink (RunManager). None = CLI mode.
        self.on_event = None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()

    def train(self) -> dict[str, Any]:
        """Validate the dataset, submit the job, poll to terminal.

        Returns {status, voice_id, voice_ref, voice_path, metrics, job_id,
        model}. status is the upstream terminal status — never faked
        "completed" (fail-visible, Rule 12).
        """
        report = validate_voice_dataset(self._dataset_dir)
        print("  [voice] TTS voice-pack training via fusion-mlx #1336")
        print(f"       Dataset: {report['samples']} pair(s) in {self._dataset_dir}")
        print(
            f"       Steps: {self._config.steps}  base voice: {self._config.base_voice}"
        )

        job = self._client.create_voice_job(
            model_id=self._model,
            dataset_dir=str(self._dataset_dir),
            voice_id=self._voice_id,
            tenant_id=self._tenant_id,
            config=self._config.to_dict(),
        )
        job_id = job.get("job_id", "")
        logger.info("voice job submitted: %s", job_id)
        print(f"       Submitted voice job: {job_id}")

        if self.on_event is not None:
            self.on_event({"type": "job", "job_id": job_id, "kind": "voice"})

        try:
            final_job = self._client.wait_for_voice_job(
                job_id,
                poll_interval=2.0,
            )
        except MLXFineTuneError as exc:
            logger.error("voice job failed: %s", exc)
            return {
                "status": "failed",
                "error": str(exc),
                "job_id": job_id,
                "model": self._model,
            }

        status = final_job.get("status", "")
        metrics = final_job.get("metrics", {}) or {}
        result = {
            "status": status,
            "job_id": job_id,
            "model": self._model,
            "voice_id": final_job.get("voice_id", ""),
            "voice_ref": final_job.get("voice_ref", ""),
            "voice_path": final_job.get("voice_path", ""),
            "metrics": metrics,
        }
        if status != "completed":
            result["error"] = final_job.get("error", f"job ended {status}")
            logger.error("voice job %s ended %s: %s", job_id, status, result["error"])
            return result

        logger.info(
            "voice job %s completed: voice_ref=%s path=%s improved=%s",
            job_id,
            result["voice_ref"],
            result["voice_path"],
            metrics.get("improved"),
        )
        print(
            f"       Voice package: {result['voice_path']} "
            f"(improved={metrics.get('improved')})"
        )
        print(f"       Serve with: voice_ref={result['voice_ref']}")
        return result
