"""Tests for GRPO trainer + sandbox reward.

Targets: rlsl/sandbox.py, rlsl/grpo.py, rlsl/pipeline.py (grpo branch)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_trainer.config import DatasetConfig, RLSLConfig, SFTConfig, TrainerConfig
from fusion_trainer.rlsl import GRPOTrainer, RLSLEngine, sandbox

# ===========================================================================
# Sandbox reward
# ===========================================================================


class TestExtractAnswer:
    def test_answer_tag(self):
        assert sandbox.extract_answer("blah <answer>42</answer> tail") == "42"

    def test_boxed(self):
        assert sandbox.extract_answer(r"The result is $\boxed{7}$ done") == "7"

    def test_final_answer_label(self):
        assert sandbox.extract_answer("Final answer: 100") == "100"

    def test_last_line_fallback(self):
        assert sandbox.extract_answer("work\n\n33") == "33"

    def test_normalizes_commas(self):
        assert sandbox.extract_answer("<answer>1,000.5</answer>") == "1000.5"


class TestRewardMath:
    def test_match(self):
        r = sandbox.reward_math({"answer": "42"}, "<answer>42</answer>")
        assert r == 1.0

    def test_mismatch(self):
        r = sandbox.reward_math({"answer": "42"}, "<answer>7</answer>")
        assert r == 0.0

    def test_numeric_close(self):
        # 3.1416 normalizes to "3.1416", 3.14159 to "3.14159" — different strings, close floats
        r = sandbox.reward_math({"answer": "3.14159"}, "<answer>3.1416</answer>")
        assert r == 0.8

    def test_no_reference(self):
        r = sandbox.reward_math({}, "<answer>5</answer>")
        assert r == 0.5


class TestRewardCode:
    def test_passing_code(self):
        code = "x = 2 + 3\nassert x == 5"
        r = sandbox.reward_code({}, f"```python\n{code}\n```")
        assert r == 1.0

    def test_failing_code(self):
        code = "assert 1 == 2"
        r = sandbox.reward_code({}, f"```python\n{code}\n```")
        assert r == 0.0

    def test_no_code(self):
        assert sandbox.reward_code({}, "just text, no code") == 0.0

    def test_with_tests_field(self):
        code = "def add(a,b):\n    return a+b"
        tests = "assert add(2,3)==5"
        r = sandbox.reward_code({"tests": tests}, f"```python\n{code}\n```")
        assert r == 1.0


class TestRewardCompletion:
    def test_auto_picks_code_when_tests(self):
        sample = {"tests": "assert True"}
        r = sandbox.reward_completion(
            sample, "```python\nassert True\n```", mode="auto"
        )
        assert r == 1.0

    def test_auto_picks_math_when_answer(self):
        r = sandbox.reward_completion(
            {"answer": "5"}, "<answer>5</answer>", mode="auto"
        )
        assert r == 1.0

    def test_off_uniform(self):
        assert sandbox.reward_completion({}, "anything", mode="off") == 0.5

    def test_scale_applied(self):
        r = sandbox.reward_completion(
            {"answer": "5"}, "<answer>5</answer>", mode="math", scale=2.0
        )
        assert r == 2.0


# ===========================================================================
# GRPOTrainer — unit tests with monkeypatched HTTP
# ===========================================================================


def _make_grpo_config(**rlsl_kw) -> TrainerConfig:
    return TrainerConfig(
        dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
        sft=SFTConfig(model="test-model"),
        rlsl=RLSLConfig(method="grpo", run_name="grpo_test", **rlsl_kw),
        cache_path="/tmp",
    )


class TestGRPOTrainerUnit:
    def test_instantiate(self):
        t = GRPOTrainer(_make_grpo_config())
        assert t is not None

    def test_empty_dataset_noop(self):
        t = GRPOTrainer(_make_grpo_config())
        result = t.train([])
        assert result["status"] == "completed"
        assert result["iterations"] == 0

    def test_no_model_raises(self):
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            rlsl=RLSLConfig(method="grpo"),
            cache_path="/tmp",
        )
        t = GRPOTrainer(cfg)
        with pytest.raises(ValueError, match="GRPO needs sft.model"):
            t.train([{"prompt": "q"}])

    def test_select_top_k(self):
        samples = [
            {
                "prompt": "p",
                "completion": "c",
                "reward": 0.1,
                "advantage": -1.0,
                "sample": {},
            },
            {
                "prompt": "p",
                "completion": "c",
                "reward": 0.9,
                "advantage": 1.0,
                "sample": {},
            },
            {
                "prompt": "p",
                "completion": "c",
                "reward": 0.5,
                "advantage": 0.0,
                "sample": {},
            },
        ]
        kept = GRPOTrainer._select_top_k(samples, 2)
        assert len(kept) == 2
        assert kept[0]["advantage"] == 1.0
        assert kept[1]["advantage"] == 0.0

    def test_stage_sft_writes_messages(self, tmp_path):
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(method="grpo", run_name="g"),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        kept = [
            {
                "prompt": "q1",
                "completion": "a1",
                "reward": 1.0,
                "advantage": 1.0,
                "sample": {},
            },
        ]
        dest = t._stage_sft(kept, 0)
        lines = Path(dest, "train.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["messages"][0]["role"] == "user"
        assert obj["messages"][0]["content"] == "q1"
        assert obj["messages"][1]["content"] == "a1"


class TestGRPOTrainerRun:
    def _patch_http(self, monkeypatch, completions, job_status="completed"):
        """Stub _generate and _submit_sft_job so no real HTTP."""

        def fake_generate(self, prompt, n):
            return [completions[i % len(completions)] for i in range(n)]

        def fake_submit(self, dataset_dir, iteration, kept_count):
            return {
                "status": job_status,
                "adapter_path": f"/fake/adapter_iter_{iteration}",
                "adapter_name": f"grpo_iter_{iteration}",
                "job_id": f"job-{iteration}",
            }

        monkeypatch.setattr(GRPOTrainer, "_generate", fake_generate)
        monkeypatch.setattr(GRPOTrainer, "_submit_sft_job", fake_submit)

    def test_math_run_converges(self, monkeypatch, tmp_path):
        self._patch_http(monkeypatch, ["<answer>5</answer>", "<answer>6</answer>"])
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(
                method="grpo",
                run_name="g",
                sandbox_mode="math",
                group_size=2,
                reward_top_k=1,
                grpo_iterations=2,
            ),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        data = [{"prompt": "2+3=?", "answer": "5"}, {"prompt": "2+4=?", "answer": "6"}]
        result = t.train(data)
        assert result["status"] == "completed"
        assert result["method"] == "grpo"
        assert len(result["reward_trajectory"]) == 2
        assert result["final_mean_reward"] > 0.0
        assert result["adapter_path"] == "/fake/adapter_iter_1"

    def test_code_run(self, monkeypatch, tmp_path):
        good = "```python\nassert True\n```"
        bad = "```python\nassert 1==2\n```"
        self._patch_http(monkeypatch, [good, bad])
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(
                method="grpo",
                run_name="g",
                sandbox_mode="code",
                group_size=2,
                reward_top_k=1,
                grpo_iterations=1,
            ),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        data = [{"prompt": "write assert True", "tests": ""}]
        result = t.train(data)
        assert result["status"] == "completed"
        assert result["reward_trajectory"][0] > 0.0

    def test_failed_job_propagates(self, monkeypatch, tmp_path):
        self._patch_http(
            monkeypatch,
            ["<answer>5</answer>", "<answer>6</answer>"],
            job_status="failed",
        )
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(
                method="grpo",
                run_name="g",
                sandbox_mode="math",
                group_size=2,
                reward_top_k=1,
                grpo_iterations=3,
            ),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        data = [{"prompt": "q", "answer": "5"}]
        result = t.train(data)
        assert result["status"] == "failed"


# ===========================================================================
# RLSLEngine — grpo dispatch
# ===========================================================================


class TestRLSLEngineGRPO:
    def test_run_grpo_dispatch(self, monkeypatch, tmp_path):
        data_path = tmp_path / "grpo.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps({"prompt": "2+2=?", "answer": "4"}) + "\n")

        called = {}

        def fake_train(self, dataset):
            called["dataset"] = dataset
            return {
                "status": "completed",
                "method": "grpo",
                "num_prompts": len(dataset),
            }

        monkeypatch.setattr(GRPOTrainer, "train", fake_train)

        cfg = TrainerConfig(
            dataset=DatasetConfig(path=str(data_path)),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(method="grpo", run_name="g"),
            cache_path=str(tmp_path),
        )
        engine = RLSLEngine(cfg)
        result = engine.run()
        assert result["method"] == "grpo"
        assert called["dataset"][0]["prompt"] == "2+2=?"

    def test_run_grpo_empty_raises(self, tmp_path):
        data_path = tmp_path / "empty.jsonl"
        with open(data_path, "w") as f:
            pass
        cfg = TrainerConfig(
            dataset=DatasetConfig(path=str(data_path)),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(method="grpo"),
            cache_path=str(tmp_path),
        )
        engine = RLSLEngine(cfg)
        with pytest.raises(ValueError, match="No valid GRPO prompt data found"):
            engine.run()


# ===========================================================================
# RewardServer — local HTTP reward endpoint for true GRPO
# ===========================================================================


class TestRewardServer:
    def test_scores_via_sandbox(self):
        import httpx

        from fusion_trainer.rlsl.reward_server import RewardServer

        dataset = [{"prompt": "2+3=?", "answer": "5"}]
        srv = RewardServer(dataset, mode="math", scale=1.0)
        endpoint = srv.start()
        try:
            resp = httpx.post(
                endpoint,
                json={
                    "prompt": "2+3=?",
                    "completions": ["<answer>5</answer>", "<answer>7</answer>"],
                },
                timeout=5.0,
            )
            assert resp.status_code == 200
            rewards = resp.json()["rewards"]
            assert rewards == [1.0, 0.0]
        finally:
            srv.stop()

    def test_unknown_prompt_falls_back(self):
        import httpx

        from fusion_trainer.rlsl.reward_server import RewardServer

        srv = RewardServer([{"prompt": "known", "answer": "5"}], mode="math")
        endpoint = srv.start()
        try:
            resp = httpx.post(
                endpoint,
                json={"prompt": "unknown prompt", "completions": ["x", "y"]},
                timeout=5.0,
            )
            assert resp.status_code == 200
            # auto/math with no reference sample -> 0.5 per completion
            rewards = resp.json()["rewards"]
            assert rewards == [0.5, 0.5]
        finally:
            srv.stop()

    def test_stop_is_idempotent(self):
        from fusion_trainer.rlsl.reward_server import RewardServer

        srv = RewardServer([], mode="off")
        srv.start()
        srv.stop()
        srv.stop()  # no error


# ===========================================================================
# RMRewardServer — RM-backed reward endpoint for RLSL Phase 2 (#431)
# ===========================================================================


class TestRMRewardServer:
    def test_forwards_to_score_reward(self):
        # GRPO 回调 {prompt, completions} -> 转调 client.score_reward(#431)
        # -> 返回真实 RM 分数。验证闭环转发与参数传递。
        import httpx

        from fusion_trainer.rlsl.rm_reward_server import RMRewardServer

        captured = {}

        class _FakeClient:
            def score_reward(self, model_id, adapter_name, prompt, completions):
                captured["model_id"] = model_id
                captured["adapter_name"] = adapter_name
                captured["prompt"] = prompt
                captured["completions"] = completions
                return {
                    "rewards": [0.9, 0.1],
                    "model_id": model_id,
                    "adapter_name": adapter_name,
                }

        srv = RMRewardServer(
            _FakeClient(), model_id="qwen-0.6b", adapter_name="rm-adapter"
        )
        endpoint = srv.start()
        try:
            resp = httpx.post(
                endpoint,
                json={"prompt": "2+3=?", "completions": ["5", "7"]},
                timeout=5.0,
            )
            assert resp.status_code == 200
            assert resp.json()["rewards"] == [0.9, 0.1]
            assert captured["model_id"] == "qwen-0.6b"
            assert captured["adapter_name"] == "rm-adapter"
            assert captured["prompt"] == "2+3=?"
            assert captured["completions"] == ["5", "7"]
        finally:
            srv.stop()

    def test_score_error_returns_500(self):
        import httpx

        from fusion_trainer.rlsl.rm_reward_server import RMRewardServer

        class _FakeClient:
            def score_reward(self, model_id, adapter_name, prompt, completions):
                raise RuntimeError("upstream down")

        srv = RMRewardServer(_FakeClient(), model_id="m", adapter_name="a")
        endpoint = srv.start()
        try:
            resp = httpx.post(
                endpoint,
                json={"prompt": "p", "completions": ["c"]},
                timeout=5.0,
            )
            assert resp.status_code == 500
            assert "upstream down" in resp.json()["error"]
        finally:
            srv.stop()

    def test_stop_is_idempotent(self):
        from fusion_trainer.rlsl.rm_reward_server import RMRewardServer

        class _FakeClient:
            def score_reward(self, *a, **kw):
                return {"rewards": []}

        srv = RMRewardServer(_FakeClient(), model_id="m", adapter_name="a")
        srv.start()
        srv.stop()
        srv.stop()  # no error


# ===========================================================================
# GRPOTrainer — true GRPO backend (grpo_backend="grpo")
# ===========================================================================


class TestGRPOTrueBackend:
    def _patch_grpo_client(self, monkeypatch, job_status="completed", events=None):
        """Stub MLXFineTuneClient GRPO methods so no real HTTP."""
        events = events or [
            {
                "type": "grpo_step",
                "iter": 0,
                "loss": 1.2,
                "mean_reward": 0.5,
                "mean_ratio": 1.0,
                "n_samples": 4,
            },
            {
                "type": "grpo_step",
                "iter": 1,
                "loss": 0.8,
                "mean_reward": 0.8,
                "mean_ratio": 0.9,
                "n_samples": 4,
            },
            {"type": "done", "adapter_path": "/fake/grpo_adapter"},
        ]

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def create_grpo_job(self, model_id, prompts, config, adapter_name=""):
                return {
                    "job_id": "grpo-job-1",
                    "model_id": model_id,
                    "prompts": prompts,
                    "adapter_name": adapter_name,
                }

            def wait_for_grpo_job(
                self, job_id, on_progress=None, poll_interval=2.0, timeout=86400.0
            ):
                if on_progress is not None:
                    on_progress(
                        {"job_id": job_id, "status": job_status, "events": events},
                        list(events),
                    )
                return {
                    "job_id": job_id,
                    "status": job_status,
                    "adapter_path": "/fake/grpo_adapter",
                    "adapter_name": "grpo_run",
                    "events": events,
                }

        monkeypatch.setattr("fusion_trainer.rlsl.grpo.MLXFineTuneClient", FakeClient)

    def test_true_grpo_completed(self, monkeypatch, tmp_path):
        self._patch_grpo_client(monkeypatch)
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="qwen-7b", lora_rank=8, lora_alpha=16.0),
            rlsl=RLSLConfig(
                method="grpo",
                grpo_backend="grpo",
                run_name="true_grpo",
                sandbox_mode="math",
                group_size=4,
                grpo_iterations=2,
                response_length=64,
            ),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        data = [{"prompt": "2+3=?", "answer": "5"}]
        result = t.train(data)
        assert result["status"] == "completed"
        assert result["method"] == "grpo"
        assert result["backend"] == "grpo"
        assert result["adapter_path"] == "/fake/grpo_adapter"
        assert result["num_prompts"] == 1

    def test_true_grpo_empty_prompts_noop(self, monkeypatch, tmp_path):
        self._patch_grpo_client(monkeypatch)
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(method="grpo", grpo_backend="grpo", run_name="g"),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        result = t.train([{"prompt": ""}, {}])
        assert result["status"] == "completed"
        assert result["iterations"] == 0
        assert result["num_prompts"] == 0

    def test_true_grpo_failed_propagates(self, monkeypatch, tmp_path):
        self._patch_grpo_client(monkeypatch, job_status="failed")
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(
                method="grpo",
                grpo_backend="grpo",
                run_name="g",
                sandbox_mode="off",
            ),
            cache_path=str(tmp_path),
        )
        t = GRPOTrainer(cfg)
        result = t.train([{"prompt": "q", "answer": "5"}])
        assert result["status"] == "failed"

    def test_default_backend_is_rft(self, tmp_path):
        # Without grpo_backend set, RFT path is used (existing behavior).
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/tmp/grpo.jsonl"),
            sft=SFTConfig(model="m"),
            rlsl=RLSLConfig(method="grpo", run_name="g"),
            cache_path=str(tmp_path),
        )
        assert cfg.rlsl.grpo_backend == "rft"
