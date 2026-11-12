"""Memory admission gate tests (capability plan §1.3)."""

from __future__ import annotations

import logging

import pytest

from fusion_trainer.memory_guard import estimate_training_gb, precheck


def test_estimate_known_model_sizes():
    assert estimate_training_gb(
        "qwen3-27b-8bit", quant_bits=8, batch_size=1
    ) == pytest.approx(27 * 1.05 + 4096 * 27 / 1024 + 4.0, abs=0.2)
    assert estimate_training_gb(
        "llama-3.2-1b-4bit", quant_bits=4, batch_size=1
    ) == pytest.approx(1 * 0.55 + 4096 * 1 / 1024 + 4.0, abs=0.2)


def test_estimate_unknown_model_returns_zero():
    assert estimate_training_gb("mystery-model") == 0.0
    assert estimate_training_gb("") == 0.0


def test_estimate_gradient_checkpointing_cuts_activation():
    full = estimate_training_gb("qwen3-27b-8bit", quant_bits=8)
    ckpt = estimate_training_gb("qwen3-27b-8bit", quant_bits=8, gradient_checkpointing=True)
    assert ckpt < full


def test_estimate_seq_length_scales_activation():
    short = estimate_training_gb("qwen3-27b-8bit", quant_bits=8, max_seq_length=1024)
    long = estimate_training_gb("qwen3-27b-8bit", quant_bits=8, max_seq_length=4096)
    assert long > short


def test_precheck_passes_small_model():
    result = precheck(
        model="llama-3.2-1b-4bit",
        quant_bits=4,
        batch_size=1,
        max_seq_length=1024,
    )
    assert result["ok"] is True
    assert result["estimate_gb"] > 0
    assert result["suggestions"] == []


def test_precheck_passes_unknown_model():
    result = precheck(model="mystery-model")
    assert result["ok"] is True
    assert result["estimate_gb"] == 0.0


def test_precheck_fails_huge_model(monkeypatch):
    import psutil

    fake = type("VM", (), {"total": 96 * 1024**3, "available": 8 * 1024**3})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)
    result = precheck(
        model="qwen3-27b-8bit",
        quant_bits=8,
        batch_size=1,
        max_seq_length=4096,
        gradient_checkpointing=False,
    )
    assert result["ok"] is False
    joined = "; ".join(result["suggestions"])
    assert "--grad-checkpoint" in joined
    assert "--max-seq-length" in joined
    assert "free resident models" in joined


def test_precheck_checkpointed_short_seq_passes(monkeypatch):
    import psutil

    fake = type("VM", (), {"total": 96 * 1024**3, "available": 48 * 1024**3})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)
    result = precheck(
        model="qwen3-27b-8bit",
        quant_bits=8,
        batch_size=1,
        max_seq_length=1024,
        gradient_checkpointing=True,
    )
    assert result["ok"] is True


def test_precheck_memory_limit_override_budget(monkeypatch):
    import psutil

    fake = type("VM", (), {"total": 96 * 1024**3, "available": 10 * 1024**3})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)
    result = precheck(
        model="llama-3.2-1b-4bit",
        quant_bits=4,
        batch_size=1,
        max_seq_length=512,
        memory_limit_gb=2.0,
    )
    assert result["budget_gb"] == 2.0
    # estimate ~5GB exceeds the explicit 2GB budget even though 10GB free
    assert result["ok"] is False
    assert any("memory_limit_gb" in s for s in result["suggestions"])


def test_precheck_logs_decision(caplog):
    with caplog.at_level(logging.INFO, logger="fusion_trainer.memory_guard"):
        precheck(model="llama-3.2-1b-4bit", quant_bits=4)
    assert any("memory_guard: ok=" in r.message for r in caplog.records)
