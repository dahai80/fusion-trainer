"""Training visualizer — loss curves, metrics plots, and dashboard utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# E8 (round-2 audit): cap metrics loaded for a single plot. A 100k-step run
# produces a 100k-line metrics.jsonl; the old _load_metrics materialized the
# whole file into a list AND plot_loss/plot_metrics built parallel
# steps/values lists on top — doubling memory for a one-shot PNG render.
# Most curves are visually identical past ~5k points; gate the load and log
# the downsample so the operator knows the curve is sampled, not exhaustive.
_MAX_PLOT_POINTS = 20000


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
                smooth_steps = steps[len(steps) - len(smoothed) :]
                ax.plot(
                    smooth_steps,
                    smoothed,
                    color="red",
                    linewidth=2,
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
                metric_keys = [k for k in metrics[0] if k not in ("step", "timestamp")]

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
        """Stream metrics.jsonl, keeping only numeric metric fields per row.

        E8 (round-2 audit): the old version materialized every full entry
        (including the timestamp string) into a list, then plot_loss /
        plot_metrics built more parallel lists on top — O(N) memory ×2-3 for
        a one-shot PNG. Now each row is parsed, non-numeric / metadata fields
        (timestamp) are dropped at parse time, and the total is capped at
        _MAX_PLOT_POINTS with a logged downsample note. A corrupt line is
        skipped (not fatal) so a half-written tail never blanks the curve.
        """
        metrics_file = self._log_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []
        metrics: list[dict[str, Any]] = []
        bad = 0
        with open(metrics_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    bad += 1
                    continue
                if not isinstance(row, dict):
                    bad += 1
                    continue
                # Keep step + numeric metric values only; drop timestamp etc.
                slim: dict[str, Any] = {}
                for k, v in row.items():
                    if k == "timestamp":
                        continue
                    if isinstance(v, (int, float)):
                        slim[k] = v
                if slim:
                    metrics.append(slim)
                if len(metrics) >= _MAX_PLOT_POINTS:
                    logger.info(
                        "visualizer: capped plot metrics at %d points "
                        "(metrics.jsonl has more) — curve is sampled",
                        _MAX_PLOT_POINTS,
                    )
                    break
        if bad:
            logger.warning("visualizer: skipped %d corrupt metrics lines", bad)
        return metrics
