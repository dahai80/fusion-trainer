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


def _patch_pref_client(monkeypatch, final_status="completed"):
    # Patch MLXFineTuneClient 让 DPO/ORPO 跑通,不连真实 fusion-mlx。
    # 记录 create_dpo_job/create_orpo_job 调用供断言。
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def create_dpo_job(
            self, model_id, preference_pairs, config=None, adapter_name="", method="dpo"
        ):
            calls.append(
                {
                    "method": method,
                    "model_id": model_id,
                    "preference_pairs": preference_pairs,
                    "config": config,
                    "adapter_name": adapter_name,
                }
            )
            return {"job_id": "fake-pref-job", "status": "queued"}

        def create_orpo_job(
            self, model_id, preference_pairs, config=None, adapter_name=""
        ):
            return self.create_dpo_job(
                model_id, preference_pairs, config, adapter_name, method="orpo"
            )

        def wait_for_dpo_job(self, job_id, on_progress=None, **kw):
            return {
                "status": final_status,
                "adapter_path": "/fake/adapters/dpo/adapter",
                "adapter_name": "fake-adapter",
            }

    import fusion_trainer.rlsl.dpo as dpo_mod
    import fusion_trainer.rlsl.orpo as orpo_mod

    monkeypatch.setattr(dpo_mod, "MLXFineTuneClient", _FakeClient)
    monkeypatch.setattr(orpo_mod, "MLXFineTuneClient", _FakeClient)
    return calls


def _patch_reward_client(monkeypatch, final_status="completed", progress=None):
    # Patch MLXFineTuneClient 让 RewardTrainer 跑通,不连真实 fusion-mlx。
    # 记录 create_reward_job 调用供断言。
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def create_reward_job(
            self, model_id, preference_pairs, config=None, adapter_name=""
        ):
            calls.append(
                {
                    "model_id": model_id,
                    "preference_pairs": preference_pairs,
                    "config": config,
                    "adapter_name": adapter_name,
                }
            )
            return {"job_id": "fake-reward-job", "status": "queued"}

        def wait_for_reward_job(self, job_id, on_progress=None, **kw):
            return {
                "status": final_status,
                "adapter_path": "/fake/adapters/reward/adapter",
                "adapter_name": "fake-reward-adapter",
                "progress": progress
                or {"iter": 4, "loss": 0.01, "reward_margin": 5.0, "acc_chosen": 1.0},
            }

        def score_reward(self, model_id, adapter_name, prompt, completions):
            # 默认真实 RM 打分(非 length fallback),供 predict 走 #431。
            return {
                "rewards": [0.9, 0.1],
                "model_id": model_id,
                "adapter_name": adapter_name,
            }

    import fusion_trainer.rlsl.reward as reward_mod

    monkeypatch.setattr(reward_mod, "MLXFineTuneClient", _FakeClient)
    return calls


def _patch_grpo_and_score_client(monkeypatch):
    # patch fusion_trainer.mlx_client.MLXFineTuneClient(Phase2 在 _run_rlsl 内
    # 本地 import),提供 create_grpo_job/wait_for_grpo_job/score_reward。
    # score_reward 会被 RMRewardServer 回调调用(返回真实 reward,非 length fallback)。
    calls: dict = {"create": [], "score": []}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def create_grpo_job(self, model_id, prompts, config=None, adapter_name=""):
            calls["create"].append(
                {
                    "model_id": model_id,
                    "prompts": prompts,
                    "config": config or {},
                    "adapter_name": adapter_name,
                }
            )
            return {"job_id": "fake-grpo-job", "status": "queued"}

        def wait_for_grpo_job(self, job_id, on_progress=None, **kw):
            return {
                "status": "completed",
                "adapter_path": "/fake/adapters/rlsl_grpo",
                "adapter_name": "test_rlsl_grpo",
                "progress": {"mean_reward": 0.83, "loss": 1.2, "mean_ratio": 1.0},
            }

        def score_reward(self, model_id, adapter_name, prompt, completions):
            calls["score"].append(
                {
                    "model_id": model_id,
                    "adapter_name": adapter_name,
                    "prompt": prompt,
                    "completions": completions,
                }
            )
            # 真实 RM 分数(非 length fallback 58.0),验证闭环打分走的是 RM。
            return {
                "rewards": [0.9, 0.1],
                "model_id": model_id,
                "adapter_name": adapter_name,
            }

    import fusion_trainer.mlx_client as client_mod

    monkeypatch.setattr(client_mod, "MLXFineTuneClient", _FakeClient)
    return calls


# ===========================================================================
# RewardTrainer
# ===========================================================================


class TestRewardTrainer:
    def test_instantiate(self):
        trainer = RewardTrainer(_make_config())
        assert trainer is not None

    def test_train_real(self, monkeypatch):
        # fusion-mlx #424 已落地:reward 经 /admin/api/fine-tune/reward/jobs 真实训练。
        # patch 须在实例化前(RewardTrainer.__init__ 创建 client)。
        calls = _patch_reward_client(monkeypatch)
        trainer = RewardTrainer(_make_config("rlsl"))
        data = _make_preference_data(4)
        result = trainer.train(data)
        assert result["status"] == "completed"
        assert result["method"] == "reward_model"
        assert result["num_pairs"] == 4
        assert result["loss"] is not None
        assert result["reward_margin"] == 5.0
        assert calls[0]["model_id"] == "test-model"
        assert len(calls[0]["preference_pairs"]) == 4
        assert calls[0]["preference_pairs"][0]["prompt"] == "Q0"
        assert "learning_rate" in calls[0]["config"]

    def test_train_no_model_raises(self, monkeypatch):
        _patch_reward_client(monkeypatch)
        config = _make_config("rlsl")
        config.sft.model = ""
        trainer = RewardTrainer(config)
        with pytest.raises(ValueError, match="sft.model"):
            trainer.train(_make_preference_data(2))

    def test_train_empty(self):
        trainer = RewardTrainer(_make_config("rlsl"))
        result = trainer.train([])
        assert result["num_pairs"] == 0

    def test_train_failure_propagates(self, monkeypatch):
        _patch_reward_client(monkeypatch, final_status="failed")
        trainer = RewardTrainer(_make_config("rlsl"))
        result = trainer.train(_make_preference_data(2))
        assert result["status"] == "failed"
        assert result["method"] == "reward_model"

    def test_predict_without_trained_adapter_raises(self):
        # 未先 train() -> 没有 adapter -> fail visibly(指向 #431)。
        trainer = RewardTrainer(_make_config("rlsl"))
        with pytest.raises(RLSLUnsupportedError, match="train\\(\\) first"):
            trainer.predict("/fake/model", ["hello", "world"])

    def test_predict_real_via_score_reward(self, monkeypatch):
        # train() 后 predict 经上游 reward/score(#431)真实打分。
        _patch_reward_client(monkeypatch)
        trainer = RewardTrainer(_make_config("rlsl"))
        trainer.train(_make_preference_data(2))
        rewards = trainer.predict("/fake/model", ["good ans", "bad ans"])
        assert rewards == [0.9, 0.1]
        assert trainer._adapter_name == "fake-reward-adapter"


# ===========================================================================
# DPOTrainer
# ===========================================================================


class TestDPOTrainer:
    def test_instantiate(self):
        trainer = DPOTrainer(_make_config("dpo"))
        assert trainer is not None

    def test_train_real(self, monkeypatch):
        # fusion-mlx #399 已落地:DPO 经 /admin/api/fine-tune/dpo/jobs 真实训练。
        # patch 须在实例化前(DPOTrainer.__init__ 创建 client)。
        calls = _patch_pref_client(monkeypatch)
        trainer = DPOTrainer(_make_config("dpo"))
        data = _make_preference_data(4)
        result = trainer.train(data)
        assert result["status"] == "completed"
        assert result["method"] == "dpo"
        assert result["num_pairs"] == 4
        assert calls[0]["method"] == "dpo"
        assert calls[0]["model_id"] == "test-model"
        assert len(calls[0]["preference_pairs"]) == 4
        assert calls[0]["preference_pairs"][0]["prompt"] == "Q0"
        assert "beta" in calls[0]["config"]

    def test_train_no_model_raises(self, monkeypatch):
        _patch_pref_client(monkeypatch)
        config = _make_config("dpo")
        config.sft.model = ""
        trainer = DPOTrainer(config)
        with pytest.raises(ValueError, match="sft.model"):
            trainer.train(_make_preference_data(2))

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

    def test_train_real(self, monkeypatch):
        # fusion-mlx #399 已落地:ORPO 经 /admin/api/fine-tune/orpo/jobs 真实训练。
        calls = _patch_pref_client(monkeypatch)
        trainer = ORPOTrainer(_make_config("orpo"))
        data = _make_preference_data(3)
        result = trainer.train(data)
        assert result["status"] == "completed"
        assert result["method"] == "orpo"
        assert result["num_pairs"] == 3
        assert calls[0]["method"] == "orpo"
        assert calls[0]["model_id"] == "test-model"
        assert "lambda_odds" in calls[0]["config"]

    def test_train_no_model_raises(self, monkeypatch):
        _patch_pref_client(monkeypatch)
        config = _make_config("orpo")
        config.sft.model = ""
        trainer = ORPOTrainer(config)
        with pytest.raises(ValueError, match="sft.model"):
            trainer.train(_make_preference_data(2))

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

    def test_run_dpo_with_path(self, monkeypatch):
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
                sft=SFTConfig(model="test-model"),
                rlsl=RLSLConfig(method="dpo", run_name="test_dpo"),
                cache_path=tmpdir,
            )
            calls = _patch_pref_client(monkeypatch)
            engine = RLSLEngine(config)
            result = engine.run()
            assert result["status"] == "completed"
            assert calls[0]["method"] == "dpo"
            assert len(calls[0]["preference_pairs"]) == 3

    def test_run_orpo(self, monkeypatch):
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
                sft=SFTConfig(model="test-model"),
                rlsl=RLSLConfig(method="orpo", run_name="test_orpo"),
                cache_path=tmpdir,
            )
            calls = _patch_pref_client(monkeypatch)
            engine = RLSLEngine(config)
            result = engine.run()
            assert result["status"] == "completed"
            assert calls[0]["method"] == "orpo"

    def test_run_rlsl(self, monkeypatch):
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
                sft=SFTConfig(model="test-model"),
                rlsl=RLSLConfig(
                    method="rlsl", reward_model="test-rm", run_name="test_rlsl"
                ),
                cache_path=tmpdir,
            )
            # RLSL 闭环: Phase1 RM 训练(#424) -> Phase2 真实 GRPO(#363),
            # reward 信号走本机 RM reward server -> 上游 reward/score(#431)。
            _patch_reward_client(monkeypatch)
            grpo_calls = _patch_grpo_and_score_client(monkeypatch)
            engine = RLSLEngine(config)
            result = engine.run()
            assert result["status"] == "completed"
            assert result["method"] == "rlsl"
            assert result["reward_model"]["status"] == "completed"
            assert result["phase2"]["method"] == "grpo"
            assert result["phase2"]["backend"] == "grpo"
            assert result["phase2"]["mean_reward"] == 0.83
            assert grpo_calls["create"][0]["model_id"] == "test-model"
            assert grpo_calls["create"][0]["adapter_name"] == "test_rlsl_grpo"
            assert "reward_endpoint" in grpo_calls["create"][0]["config"]

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

    def test_run_dpo_from_hub(self, monkeypatch):
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
                sft=SFTConfig(model="test-model"),
                rlsl=RLSLConfig(method="dpo", run_name="hub_dpo"),
                cache_path=tmpdir,
                model_hub_path=str(hub_dir),
            )
            calls = _patch_pref_client(monkeypatch)
            engine = RLSLEngine(config)
            result = engine.run()
            assert result["status"] == "completed"
            assert calls[0]["method"] == "dpo"
            assert len(calls[0]["preference_pairs"]) == 3
