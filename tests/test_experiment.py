"""Tests for the experiment module — logger, visualizer, tracker.

Targets: logger.py, visualizer.py, tracker.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fusion_trainer.experiment import (
    ExperimentLogger,
    ExperimentTracker,
    TrainingVisualizer,
)

# ===========================================================================
# ExperimentLogger
# ===========================================================================


class TestExperimentLogger:
    def test_basic_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("test_run", tmpdir)
            logger.log_metrics(step=1, metrics={"loss": 0.5})
            logger.log_metrics(step=2, metrics={"loss": 0.3})
            log_dir = logger.save()
            assert (log_dir / "metrics.jsonl").exists()
            assert (log_dir / "metadata.json").exists()

    def test_metrics_streamed_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("stream_test", tmpdir, max_in_memory=5)
            for i in range(20):
                logger.log_metrics(step=i, metrics={"loss": 1.0 / (i + 1)})
            log_dir = logger.save()
            # Verify all 20 entries are on disk
            with open(log_dir / "metrics.jsonl") as f:
                lines = f.readlines()
            assert len(lines) == 20

    def test_bounded_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("mem_test", tmpdir, max_in_memory=3)
            for i in range(10):
                logger.log_metrics(step=i, metrics={"loss": float(i)})
            assert len(logger.recent_metrics) == 3
            assert logger.recent_metrics[-1]["step"] == 9

    def test_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("hp_test", tmpdir)
            logger.log_hyperparameters({"lr": 1e-5, "batch": 4})
            logger.save()
            with open(logger.log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["hyperparameters"]["lr"] == 1e-5

    def test_dataset_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("ds_test", tmpdir)
            logger.log_dataset_info({"name": "test", "version": "v1"})
            logger.save()
            with open(logger.log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["dataset"]["name"] == "test"

    def test_metadata_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("ts_test", tmpdir)
            logger.save()
            with open(logger.log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert "started_at" in meta
            assert "completed_at" in meta

    def test_metrics_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("pref_test", tmpdir)
            logger.log_metrics(step=1, metrics={"loss": 0.1}, prefix="train/")
            logger.save()
            with open(logger.log_dir / "metrics.jsonl") as f:
                line = json.loads(f.readline())
            assert "train/loss" in line

    def test_log_dir_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("prop_test", tmpdir)
            assert logger.log_dir.name == "prop_test"
            assert logger.log_dir.exists()

    def test_metadata_run_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("custom_name", tmpdir)
            logger.save()
            with open(logger.log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["run_name"] == "custom_name"

    def test_count_total_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("count_test", tmpdir)
            for i in range(7):
                logger.log_metrics(step=i, metrics={"loss": 0.0})
            # _count_total_metrics is called inside save()
            log_dir = logger.save()
            with open(log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["num_metrics_logged"] == 7

    def test_metadata_with_custom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("meta_test", tmpdir, metadata={"user": "alice"})
            logger.save()
            with open(logger.log_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["user"] == "alice"

    def test_user_provided_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "custom" / "path"
            logger = ExperimentLogger("deep_test", output_dir)
            assert logger.log_dir == output_dir / "logs" / "deep_test"
            logger.save()
            assert logger.log_dir.exists()


# ===========================================================================
# ExperimentTracker
# ===========================================================================


class TestExperimentTracker:
    def test_create_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("test", {"lr": 1e-5})
            assert exp_id.startswith("test_")
            assert (Path(tmpdir) / exp_id / "experiment.json").exists()

    def test_update_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("t", {})
            tracker.update_status(exp_id, "running")
            record = tracker.get_experiment(exp_id)
            assert record["status"] == "running"

    def test_update_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("t", {})
            tracker.update_metrics(exp_id, {"loss": 0.1})
            record = tracker.get_experiment(exp_id)
            assert record["metrics_summary"]["loss"] == 0.1

    def test_update_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("t", {})
            tracker.update_outputs(exp_id, {"model": "/path/to/model"})
            record = tracker.get_experiment(exp_id)
            assert record["output_paths"]["model"] == "/path/to/model"

    def test_list_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            tracker.create_experiment("a", {})
            tracker.create_experiment("b", {})
            assert len(tracker.list_experiments()) == 2

    def test_list_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            e1 = tracker.create_experiment("a", {})
            e2 = tracker.create_experiment("b", {})
            tracker.update_status(e1, "completed")
            assert len(tracker.list_experiments(status="completed")) == 1
            assert len(tracker.list_experiments(status="running")) == 0

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            assert tracker.get_experiment("nonexistent") is None

    def test_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("t", {}, description="my desc")
            record = tracker.get_experiment(exp_id)
            assert record["description"] == "my desc"

    def test_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            assert tracker.list_experiments() == []

    def test_metrics_accumulate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(experiments_dir=tmpdir)
            exp_id = tracker.create_experiment("t", {})
            tracker.update_metrics(exp_id, {"loss": 0.5})
            tracker.update_metrics(exp_id, {"acc": 0.9})
            record = tracker.get_experiment(exp_id)
            assert record["metrics_summary"] == {"loss": 0.5, "acc": 0.9}


# ===========================================================================
# TrainingVisualizer (requires matplotlib)
# ===========================================================================


class TestTrainingVisualizer:
    def test_plot_loss_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_test", tmpdir)
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            result = viz.plot_loss()
            assert result == ""  # No data, returns empty

    def test_plot_loss_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_loss", tmpdir)
            for i in range(1, 21):
                logger.log_metrics(step=i, metrics={"loss": 1.0 / i})
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            out = Path(tmpdir) / "loss.png"
            result = viz.plot_loss(output_path=str(out))
            assert result != ""
            assert out.exists()
            assert out.stat().st_size > 0

    def test_plot_loss_smoothed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_smooth", tmpdir)
            for i in range(1, 31):
                logger.log_metrics(step=i, metrics={"loss": 0.5})
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            out = Path(tmpdir) / "smooth.png"
            result = viz.plot_loss(output_path=str(out), smooth=True, smooth_window=5)
            assert result != ""

    def test_plot_metrics_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_none", tmpdir)
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            result = viz.plot_metrics()
            assert result == ""

    def test_plot_metrics_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_metrics", tmpdir)
            for i in range(1, 11):
                logger.log_metrics(
                    step=i, metrics={"loss": 0.1 * i, "acc": 1.0 - 0.1 * i}
                )
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            out = Path(tmpdir) / "metrics.png"
            result = viz.plot_metrics(output_path=str(out))
            assert result != ""
            assert out.exists()

    def test_plot_metrics_specific_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("viz_keys", tmpdir)
            for i in range(1, 6):
                logger.log_metrics(
                    step=i, metrics={"loss": 0.1, "acc": 0.9, "val_loss": 0.2}
                )
            log_dir = logger.save()
            viz = TrainingVisualizer(log_dir)
            out = Path(tmpdir) / "key_metrics.png"
            result = viz.plot_metrics(
                metric_keys=["loss", "val_loss"], output_path=str(out)
            )
            assert result != ""

    def test_load_metrics_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = TrainingVisualizer(Path(tmpdir))
            assert viz._load_metrics() == []
