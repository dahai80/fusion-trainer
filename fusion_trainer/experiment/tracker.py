"""Experiment tracker — manages experiment lifecycle and reproducibility."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Tracks experiment runs for reproducibility and comparison.

    Each experiment gets a unique ID with metadata including:
    - Git commit hash
    - Dataset version
    - Hyperparameters
    - Training metrics summary
    - Output paths
    """

    def __init__(self, experiments_dir: str | Path = "experiments"):
        self._experiments_dir = Path(experiments_dir)
        self._experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        config: dict[str, Any],
        description: str = "",
    ) -> str:
        """Create a new experiment record.

        Returns:
            Experiment ID string.
        """
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        record = {
            "experiment_id": exp_id,
            "name": name,
            "description": description,
            "config": config,
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._get_git_commit(),
            "metrics_summary": {},
            "output_paths": {},
        }

        exp_dir = self._experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "experiment.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return exp_id

    def update_status(
        self, experiment_id: str, status: str
    ) -> None:
        """Update experiment status."""
        record = self._load(experiment_id)
        if record:
            record["status"] = status
            self._save(experiment_id, record)

    def update_metrics(
        self, experiment_id: str, metrics: dict[str, Any]
    ) -> None:
        """Update metrics summary."""
        record = self._load(experiment_id)
        if record:
            record["metrics_summary"].update(metrics)
            self._save(experiment_id, record)

    def update_outputs(
        self, experiment_id: str, outputs: dict[str, str]
    ) -> None:
        """Record output paths."""
        record = self._load(experiment_id)
        if record:
            record["output_paths"].update(outputs)
            self._save(experiment_id, record)

    def list_experiments(
        self, status: str | None = None
    ) -> list[dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        results = []
        for exp_dir in sorted(self._experiments_dir.iterdir()):
            if exp_dir.is_dir():
                record = self._load(exp_dir.name)
                if record:
                    if status is None or record.get("status") == status:
                        results.append(record)
        return results

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get a specific experiment record."""
        return self._load(experiment_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, experiment_id: str) -> dict[str, Any] | None:
        path = self._experiments_dir / experiment_id / "experiment.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, experiment_id: str, record: dict[str, Any]) -> None:
        path = self._experiments_dir / experiment_id / "experiment.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_git_commit() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return ""