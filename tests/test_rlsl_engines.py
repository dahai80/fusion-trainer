"""Tests for RLSL engines — reward, dpo, orpo trainers and pipeline.

Targets: rlsl/reward.py, rlsl/dpo.py, rlsl/orpo.py, rlsl/pipeline.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fusion_trainer.config import DatasetConfig, RLSLConfig, SFTConfig, TrainerConfig
from fusion_trainer.rlsl import (
    DPOTrainer,
    ORPOTrainer,
    RewardTrainer,
    RLSLEngine,
    RLSLUnsupportedError,
)


def _make_preference_data(n: int = 5) -> list[dict]:
    return [
        {"prompt": f"Q{i}", "chosen": f"Good answer {i}", "rejected": f"Bad answer {i}"}
        for i in range(n)
    ]


def _make_config(method: str = "dpo", **kwargs) -> TrainerConfig:
    return TrainerConfig(
        dataset=DatasetConfig(path="/tmp/test.jsonl"),
        sft=SFTConfig(model="test-model"),
        rlsl=RLSLConfig(method=method, **kwargs),
        cache_path="/tmp",
    )


# ===========================================================================
# RewardTrainer
# ===========================================================================


class TestRewardTrainer:
    def test_instantiate(self):
        trainer = RewardTrainer(_make_config())
        assert trainer is not None

    def test_train_raises_upstream_unsupported(self):
        # RM training needs fusion-mlx endpoint (fusion-mlx#399) -> fail visibly.
        trainer = RewardTrainer(_make_config("rlsl"))
        data = _make_preference_data(3)
        with pytest.raises(RLSLUnsupportedError, match="fusion-mlx/issues/399"):
            trainer.train(data)

    def test_train_empty(self):
        trainer = RewardTrainer(_make_config("rlsl"))
        result = trainer.train([])
        assert result["num_pairs"] == 0

    def test_predict_raises_upstream_unsupported(self):
        trainer = RewardTrainer(_make_config("rlsl"))
        with pytest.raises(RLSLUnsupportedError, match="fusion-mlx/issues/399"):
            trainer.predict("/fake/model", ["hello", "world"])


# ===========================================================================
# DPOTrainer
# ===========================================================================


class TestDPOTrainer:
    def test_instantiate(self):
        trainer = DPOTrainer(_make_config("dpo"))
        assert trainer is not None

    def test_train_raises_upstream_unsupported(self):
        # DPO needs fusion-mlx endpoint (fusion-mlx#399) -> fail visibly.
        trainer = DPOTrainer(_make_config("dpo"))
        data = _make_preference_data(4)
        with pytest.raises(RLSLUnsupportedError, match="fusion-mlx/issues/399"):
            trainer.train(data)

    def test_train_empty(self):
        trainer = DPOTrainer(_make_config("dpo"))
        result = trainer.train([])
        assert result["num_pairs"] == 0


# ===========================================================================
# ORPOTrainer
# ===========================================================================


class TestORPOTrainer:
    def test_instantiate(self):
        trainer = ORPOTrainer(_make_config("orpo"))
        assert trainer is not None

    def test_train_raises_upstream_unsupported(self):
        # ORPO needs fusion-mlx endpoint (fusion-mlx#399) -> fail visibly.
        trainer = ORPOTrainer(_make_config("orpo"))
        data = _make_preference_data(3)
        with pytest.raises(RLSLUnsupportedError, match="fusion-mlx/issues/399"):
            trainer.train(data)

    def test_train_empty(self):
        trainer = ORPOTrainer(_make_config("orpo"))
        result = trainer.train([])
        assert result["num_pairs"] == 0


# ===========================================================================
# RLSLEngine
# ===========================================================================


class TestRLSLEngine:
    def test_instantiate(self):
        config = _make_config("dpo")
        engine = RLSLEngine(config)
        assert engine is not None

    def test_run_dpo_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "pref.jsonl"
            import json

            with open(data_path, "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps({"prompt": f"Q{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            config = TrainerConfig(
                dataset=DatasetConfig(path=str(data_path)),
                rlsl=RLSLConfig(method="dpo", run_name="test_dpo"),
                cache_path=tmpdir,
            )
            engine = RLSLEngine(config)
            with pytest.raises(RLSLUnsupportedError, match="DPO alignment"):
                engine.run()

    def test_run_orpo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "pref.jsonl"
            import json

            with open(data_path, "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps({"prompt": f"Q{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            config = TrainerConfig(
                dataset=DatasetConfig(path=str(data_path)),
                rlsl=RLSLConfig(method="orpo", run_name="test_orpo"),
                cache_path=tmpdir,
            )
            engine = RLSLEngine(config)
            with pytest.raises(RLSLUnsupportedError, match="ORPO alignment"):
                engine.run()

    def test_run_rlsl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "pref.jsonl"
            import json

            with open(data_path, "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps({"prompt": f"Q{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            config = TrainerConfig(
                dataset=DatasetConfig(path=str(data_path)),
                rlsl=RLSLConfig(
                    method="rlsl", reward_model="test-rm", run_name="test_rlsl"
                ),
                cache_path=tmpdir,
            )
            engine = RLSLEngine(config)
            # RLSL = RM training (Phase 1) -> unsupported upstream.
            with pytest.raises(RLSLUnsupportedError, match="Reward-model training"):
                engine.run()

    def test_run_empty_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "empty.jsonl"
            # Create empty file
            with open(data_path, "w") as f:
                pass
            config = TrainerConfig(
                dataset=DatasetConfig(path=str(data_path)),
                rlsl=RLSLConfig(method="dpo"),
                cache_path=tmpdir,
            )
            engine = RLSLEngine(config)
            with pytest.raises(ValueError, match="No valid preference data found"):
                engine.run()

    def test_run_unknown_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "pref.jsonl"
            import json

            with open(data_path, "w") as f:
                f.write(
                    json.dumps({"prompt": "q", "chosen": "c", "rejected": "r"}) + "\n"
                )
            config = TrainerConfig(
                dataset=DatasetConfig(path=str(data_path)),
                rlsl=RLSLConfig(method="unknown"),  # type: ignore[arg-type]
                cache_path=tmpdir,
            )
            engine = RLSLEngine(config)
            with pytest.raises(ValueError, match="Unknown alignment method"):
                engine.run()

    def test_run_dpo_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            ds_dir = hub_dir / "datasets"
            ds_dir.mkdir()
            import json

            with open(ds_dir / "pref_data.jsonl", "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps({"prompt": f"Q{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            config = TrainerConfig(
                dataset=DatasetConfig(hub_dataset="pref_data"),
                rlsl=RLSLConfig(method="dpo", run_name="hub_dpo"),
                cache_path=tmpdir,
                model_hub_path=str(hub_dir),
            )
            engine = RLSLEngine(config)
            with pytest.raises(RLSLUnsupportedError, match="DPO alignment"):
                engine.run()
