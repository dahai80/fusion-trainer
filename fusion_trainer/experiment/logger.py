"""Experiment logger — records training metrics, hyperparameters, and artifacts.

Metrics are streamed to disk incrementally to avoid OOM on long training runs.
Only a bounded window of recent metrics is kept in memory for fast access.
TensorBoard events are written alongside the JSONL stream (opt-in via
enable_tensorboard, default on when torch.utils.tensorboard is importable).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs training metrics and experiment metadata.

    Supports:
    - Scalar metrics (loss, accuracy, learning rate) — streamed to disk
    - Hyperparameter tracking
    - Reproducibility metadata (dataset version, seed, config)
    - TensorBoard export (optional, default on when the backend is importable)

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
        enable_tensorboard: bool = True,
    ):
        self.run_name = run_name
        self._output_dir = Path(output_dir) / "logs" / run_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Bounded in-memory window (recent metrics for fast access)
        self._metrics: deque[dict[str, Any]] = deque(maxlen=max_in_memory)
        self._max_in_memory = max_in_memory
        self._flush_every = flush_every
        self._steps_since_flush = 0
        # E8 (round-2 audit): running count of log_metrics calls so save()
        # doesn't re-read the whole metrics.jsonl to size it. On a 100k-step
        # run that re-read was O(N) at finalize for a number already known.
        self._total_logged = 0

        # Streaming file handle (append mode)
        self._metrics_path = self._output_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "a", encoding="utf-8")  # noqa: SIM115

        self._metadata: dict[str, Any] = metadata or {}
        self._metadata["run_name"] = run_name
        self._metadata["started_at"] = datetime.now(UTC).isoformat()

        # TensorBoard writer (optional). Lazy-imported so a missing
        # torch/tensorboard never breaks training; degrades to None + warning.
        self._tb_writer = None
        self._tb_dir: Path | None = None
        if enable_tensorboard:
            self._tb_writer = self._init_tensorboard()

    def _init_tensorboard(self):
        """Create a SummaryWriter writing to log_dir/tensorboard.

        Returns None (with a warning) if the tensorboard backend is not
        importable, so the rest of logging keeps working.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # ImportError or tensorboard backend issues
            logger.warning(
                "TensorBoard disabled (import failed: %s); metrics still "
                "stream to JSONL. Install torch/tensorboard to enable.",
                exc,
            )
            return None
        self._tb_dir = self._output_dir / "tensorboard"
        self._tb_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TensorBoard writer -> %s", self._tb_dir)
        return SummaryWriter(log_dir=str(self._tb_dir))

    @property
    def tensorboard_dir(self) -> Path | None:
        """Directory holding TensorBoard event files (None if disabled)."""
        return self._tb_dir

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        prefix: str = "",
    ) -> None:
        """Log training metrics at a given step.

        Metrics are written to the JSONL file immediately and kept in
        a bounded in-memory window for live access. When TensorBoard is
        enabled, each scalar is also written as a TB event.
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
        self._total_logged += 1

        # Keep a bounded window in memory
        self._metrics.append(entry)

        # Also log to TensorBoard if available
        if self._tb_writer is not None:
            for key, value in metrics.items():
                try:
                    self._tb_writer.add_scalar(f"{prefix}{key}", float(value), step)
                except (TypeError, ValueError):
                    logger.debug("skip non-scalar TB metric %s=%r", key, value)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the experiment.

        Stored in metadata (JSONL). Hyperparameters are not written to the
        TensorBoard event stream: TB's add_hparams needs a non-empty metric
        dict and its async writer raises on empty metrics, so we keep hparams
        in the JSON metadata only (metrics still stream to TB via log_metrics).
        """
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
        meta["tensorboard"] = self._tb_dir is not None

        meta_path = self._output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Close TensorBoard writer + release event-file handle
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception as exc:
                logger.debug("TB writer close failed: %s", exc)
            self._tb_writer = None

        return self._output_dir

    @property
    def log_dir(self) -> Path:
        return self._output_dir

    @property
    def recent_metrics(self) -> list[dict[str, Any]]:
        """Return the in-memory window of recent metrics."""
        return list(self._metrics)

    def close(self) -> None:
        """Release resources (metrics file + TensorBoard writer).

        Safe to call multiple times. Called automatically by save(); call this
        directly if discarding a logger without save() so the TB async writer
        flushes and releases its event-file handle (avoids fd leaks and
        tempdir cleanup races on macOS).
        """
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception as exc:
                logger.debug("TB writer close failed: %s", exc)
            self._tb_writer = None

    def __del__(self) -> None:
        # Best-effort cleanup if save()/close() was never called.
        import contextlib

        with contextlib.suppress(Exception):
            self.close()

    def _count_total_metrics(self) -> int:
        """Count total metric entries written to disk.

        E8 (round-2 audit): tracked incrementally in log_metrics — no re-read
        of the (possibly huge) metrics.jsonl at finalize. Fall back to a file
        line-count only if the counter was lost (e.g. a logger reloaded state
        without replay), which is not the normal path.
        """
        if self._total_logged > 0:
            return self._total_logged
        try:
            count = 0
            with open(self._metrics_path, encoding="utf-8") as f:
                for _ in f:
                    count += 1
            return count
        except Exception:
            return len(self._metrics)

    # ------------------------------------------------------------------
    # Prometheus text-format export (enterprise #3)
    # ------------------------------------------------------------------

    def latest_metrics(self) -> dict[str, float]:
        """Return the most recent value of each scalar metric key.

        Deterministic scan of the in-memory window (Rule 5): the last entry
        that has a key wins. Used by export_prometheus so the .prom file
        reflects the current state, not a moving average.
        """
        latest: dict[str, float] = {}
        for entry in self._metrics:
            for key, value in entry.items():
                if key in ("step", "timestamp"):
                    continue
                try:
                    latest[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return latest

    def export_prometheus(
        self,
        path: str | Path,
        run_id: str = "",
        extra_series: dict[str, float] | None = None,
    ) -> Path:
        """Write a Prometheus text-format (.prom) snapshot for this run.

        One gauge per scalar metric key, plus any extra_series (e.g.
        run_steps_total / run_status supplied by RunManager). A scrape
        endpoint is out of scope (lives in fusion-agent-studio upstream); this
        is a file export a node-exporter textfile collector or a sidecar can
        ship. Fail-visible: logs on write error, never raises into a run.
        """
        return export_prometheus_snapshot(
            path, self.latest_metrics(), run_id=run_id or self.run_name,
            extra_series=extra_series,
        )


def export_prometheus_snapshot(
    path: str | Path,
    metrics: dict[str, float],
    run_id: str = "",
    extra_series: dict[str, float] | None = None,
) -> Path:
    """Write a Prometheus text-format (.prom) snapshot from a flat metrics dict.

    Module-level helper (enterprise #3) so RunManager can emit a .prom file at
    run end without constructing an ExperimentLogger. One gauge per scalar key
    + any extra_series (run_steps_total / run_status_code). Fail-visible: logs
    on write error, never raises.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            series[key] = float(value)
        except (TypeError, ValueError):
            continue
    if extra_series:
        for k, v in extra_series.items():
            try:
                series[k] = float(v)
            except (TypeError, ValueError):
                continue
    run_label = run_id
    lines: list[str] = []
    for key in sorted(series):
        name = f"fusion_trainer_{_prom_safe(key)}"
        lines.append(f"# HELP {name} {run_label}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f'{name}{{run="{run_label}"}} {series[key]}')
    text = "\n".join(lines) + ("\n" if lines else "")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception as exc:
        logger.warning("export_prometheus_snapshot write to %s failed: %s", out_path, exc)
    return out_path


def _prom_safe(key: str) -> str:
    """Sanitize a metric key into a Prometheus-valid metric name segment.

    Prom metric names: [a-zA-Z_:][a-zA-Z0-9_:]*. Replace anything else with _.
    """
    out: list[str] = []
    for ch in key:
        if ch.isalnum() or ch in ("_", ":"):
            out.append(ch)
        else:
            out.append("_")
    if out and not (out[0].isalpha() or out[0] == "_"):
        out.insert(0, "_")
    return "".join(out) or "metric"
