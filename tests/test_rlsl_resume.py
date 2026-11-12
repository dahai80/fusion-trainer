"""RL resume tests (capability plan T1)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fusion_trainer.config import DatasetConfig, RLSLConfig, SFTConfig, TrainerConfig
from fusion_trainer.mlx_client import MLXFineTuneError
from fusion_trainer.rlsl.pipeline import RLSLEngine


def _engine(tmp_path: Path, method: str = "dpo") -> RLSLEngine:
    cfg = TrainerConfig(
        dataset=DatasetConfig(path=str(tmp_path / "data.jsonl")),
        sft=SFTConfig(model="llama-3.2-1b-4bit"),
        rlsl=RLSLConfig(
            method=method,  # type: ignore[arg-type]
            run_name="rl_test",
            output_dir=str(tmp_path / "out"),
        ),
    )
    cfg.rlsl.output_dir = str(tmp_path / "out")
    return RLSLEngine(cfg)


def _make_checkpoint(tmp_path: Path, step: int) -> Path:
    ckpt = tmp_path / ".fusion-mlx" / "adapters" / "llama-3.2-1b-4bit" / "rl_test"
    ckpt.mkdir(parents=True, exist_ok=True)
    f = ckpt / f"{step:06d}_adapters.safetensors"
    f.write_bytes(b"fake-weights")
    return f


def test_latest_checkpoint_picks_highest_step(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _make_checkpoint(tmp_path, 10)
    _make_checkpoint(tmp_path, 20)
    engine = _engine(tmp_path)
    found = engine._latest_checkpoint()
    assert found is not None
    ckpt, step = found
    assert step == 20
    assert ckpt.name == "000020_adapters.safetensors"


def test_latest_checkpoint_none_without_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    engine = _engine(tmp_path)
    assert engine._latest_checkpoint() is None


def test_resolve_resume_stages_and_sets_init(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ckpt = _make_checkpoint(tmp_path, 30)
    engine = _engine(tmp_path)
    engine._resolve_rl_resume()
    staged = Path(engine.rlsl_cfg.output_dir) / "resume_stage" / "adapters.safetensors"
    assert staged.is_file()
    assert staged.read_bytes() == ckpt.read_bytes()
    assert engine.rlsl_cfg.init_adapter_path == str(staged)


def test_resolve_resume_emits_event(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _make_checkpoint(tmp_path, 30)
    engine = _engine(tmp_path)
    events: list[dict] = []
    engine.on_event = events.append
    engine._resolve_rl_resume()
    assert events == [
        {
            "type": "resume",
            "resume_step": 30,
            "checkpoint": str(Path(engine.rlsl_cfg.output_dir) / "resume_stage" / "adapters.safetensors"),
        }
    ]


def test_resolve_resume_grpo_warns_fresh(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _make_checkpoint(tmp_path, 30)
    engine = _engine(tmp_path, method="grpo")
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.pipeline"):
        engine._resolve_rl_resume()
    assert engine.rlsl_cfg.init_adapter_path == ""
    assert any("GRPO resume is not supported" in r.message for r in caplog.records)


def test_resolve_resume_no_checkpoint_warns_fresh(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    engine = _engine(tmp_path)
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.pipeline"):
        engine._resolve_rl_resume()
    assert engine.rlsl_cfg.init_adapter_path == ""
    assert any("no numbered checkpoint found" in r.message for r in caplog.records)


def _bypass_guard(monkeypatch):
    # CI runners report ~7GB RAM; the admission gate must not gate offline
    # pipeline tests on host memory.
    monkeypatch.setattr(
        "fusion_trainer.memory_guard.precheck",
        lambda **kw: {
            "ok": True,
            "estimate_gb": 0.0,
            "available_gb": 0.0,
            "budget_gb": 0.0,
            "suggestions": [],
        },
    )


def test_interrupted_result_on_404(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _bypass_guard(monkeypatch)
    ckpt = _make_checkpoint(tmp_path, 40)
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine._loader,
        "load_preference",
        lambda **kwargs: [{"prompt": "p", "chosen": "c", "rejected": "r"}],
    )

    def _boom(dataset):
        raise MLXFineTuneError("job gone", status_code=404)

    monkeypatch.setattr(engine, "_run_dpo", _boom)
    result = engine.run()
    assert result["status"] == "interrupted"
    assert result["method"] == "dpo"
    assert result["resume_step"] == 40
    assert result["resume_checkpoint"] == str(ckpt)
    assert result["resume_hint"] == "ft rlsl --resume"


def test_non_404_error_reraises(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _bypass_guard(monkeypatch)
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine._loader,
        "load_preference",
        lambda **kwargs: [{"prompt": "p", "chosen": "c", "rejected": "r"}],
    )

    def _boom(dataset):
        raise MLXFineTuneError("bad request", status_code=400)

    monkeypatch.setattr(engine, "_run_dpo", _boom)
    with pytest.raises(MLXFineTuneError):
        engine.run()
