"""Experiment logger — records training metrics, hyperparameters, and artifacts.

Metrics are streamed to disk incrementally to avoid OOM on long training runs.
Only a bounded window of recent metrics is kept in memory for fast access.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logs training metrics and experiment metadata.

    Supports:
    - Scalar metrics (loss, accuracy, learning rate) — streamed to disk
    - Hyperparameter tracking
    - Reproducibility metadata (dataset version, seed, config)
    - TensorBoard export (optional)

    Memory-safe: metrics are flushed to disk every N steps, and only a
    bounded window (max_in_memory=1000) is retained in RAM.
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
        max_in_memory: int = 1000,
        flush_every: int = 100,
    ):
        self.run_name = run_name
        self._output_dir = Path(output_dir) / "logs" / run_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Bounded in-memory window (recent metrics for fast access)
        self._metrics: deque[dict[str, Any]] = deque(maxlen=max_in_memory)
        self._max_in_memory = max_in_memory
        self._flush_every = flush_every
        self._steps_since_flush = 0

        # Streaming file handle (append mode)
        self._metrics_path = self._output_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "a", encoding="utf-8")  # noqa: SIM115

        self._metadata: dict[str, Any] = metadata or {}
        self._metadata["run_name"] = run_name
        self._metadata["started_at"] = datetime.now(UTC).isoformat()

        # TensorBoard writer (optional)
        self._tb_writer = None

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        prefix: str = "",
    ) -> None:
        """Log training metrics at a given step.

        Metrics are written to the JSONL file immediately and kept in
        a bounded in-memory window for live access.
        """
        entry = {
            "step": step,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        for key, value in metrics.items():
            entry[f"{prefix}{key}"] = value

        # Stream to disk immediately (append a line)
        line = json.dumps(entry, ensure_ascii=False)
        self._metrics_file.write(line + "\n")
        self._metrics_file.flush()
        self._steps_since_flush += 1

        # Keep a bounded window in memory
        self._metrics.append(entry)

        # Also log to TensorBoard if available
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(f"{prefix}{key}", value, step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the experiment."""
        self._metadata["hyperparameters"] = params

    def log_dataset_info(self, info: dict[str, Any]) -> None:
        """Log dataset information for reproducibility."""
        self._metadata["dataset"] = info

    def save(self) -> Path:
        """Finalize and save all logged data to disk."""
        # Close the streaming metrics file
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()

        # Save metadata (overwrites on each save to keep it current)
        meta = dict(self._metadata)
        meta["completed_at"] = datetime.now(UTC).isoformat()
        meta["num_metrics_logged"] = self._count_total_metrics()

        meta_path = self._output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()

        return self._output_dir

    @property
    def log_dir(self) -> Path:
        return self._output_dir

    @property
    def recent_metrics(self) -> list[dict[str, Any]]:
        """Return the in-memory window of recent metrics."""
        return list(self._metrics)

    def _count_total_metrics(self) -> int:
        """Count total metric entries written to disk."""
        try:
            count = 0
            with open(self._metrics_path, encoding="utf-8") as f:
                for _ in f:
                    count += 1
            return count
        except Exception:
            return len(self._metrics)
