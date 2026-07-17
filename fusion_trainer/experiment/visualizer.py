"""Training visualizer — loss curves, metrics plots, and dashboard utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingVisualizer:
    """Generates training visualizations from experiment logs.

    Produces:
    - Loss curves (training vs validation)
    - Learning rate schedules
    - Metric comparison plots
    """

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)

    def plot_loss(
        self,
        output_path: str | Path = "loss_curve.png",
        smooth: bool = True,
        smooth_window: int = 10,
    ) -> str:
        """Plot training loss curve from metrics log."""
        metrics = self._load_metrics()
        if not metrics:
            return ""

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            steps = [m["step"] for m in metrics if "loss" in m]
            losses = [m["loss"] for m in metrics if "loss" in m]

            if not steps:
                return ""

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, losses, alpha=0.3, color="blue", label="Raw loss")

            if smooth and len(losses) > smooth_window:
                # Simple moving average
                arr = np.array(losses)
                smoothed = np.convolve(
                    arr, np.ones(smooth_window) / smooth_window, mode="valid"
                )
                smooth_steps = steps[len(steps) - len(smoothed):]
                ax.plot(
                    smooth_steps, smoothed,
                    color="red", linewidth=2,
                    label=f"Smoothed (window={smooth_window})",
                )

            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

            output_path = str(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path

        except ImportError:
            return ""

    def plot_metrics(
        self,
        metric_keys: list[str] | None = None,
        output_path: str | Path = "metrics.png",
    ) -> str:
        """Plot multiple training metrics."""
        metrics = self._load_metrics()
        if not metrics:
            return ""

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if metric_keys is None:
                # Auto-detect numeric keys
                metric_keys = [
                    k for k in metrics[0].keys()
                    if k not in ("step", "timestamp")
                ]

            n = len(metric_keys)
            fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)

            for i, key in enumerate(metric_keys):
                ax = axes[i][0]
                steps = [m["step"] for m in metrics if key in m]
                values = [m[key] for m in metrics if key in m]
                if steps:
                    ax.plot(steps, values, linewidth=1.5)
                    ax.set_xlabel("Step")
                    ax.set_ylabel(key)
                    ax.set_title(key)
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = str(output_path)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path

        except ImportError:
            return ""

    def _load_metrics(self) -> list[dict[str, Any]]:
        metrics_file = self._log_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []
        metrics = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
        return metrics