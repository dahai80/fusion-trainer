"""Rollout-sink ingestion tests (issue #86).

Fully offline: synthetic sink JSONL in tmp_path, no gateway / fusion-mlx.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_trainer.dataset.rollout_ingest import (
    RolloutIngestError,
    build_from_rollouts,
    find_rollout_manifest,
    load_rollout_sink,
    positive_rollouts,
    reward_score,
    rollouts_to_grpo,
    rollouts_to_sft,
)


def _rec(
    rollout_id="ro-1",
    signal="task_success",
    score=None,
    prompt="p",
    messages=None,
    model="test-model",
    **extra,
):
    rec = {
        "schema_version": 1,
        "rollout_id": rollout_id,
        "model": model,
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": f"did {rollout_id}"},
        ],
    }
    if prompt is not None:
        rec["prompt"] = prompt
    if signal is not None:
        outcome = {"signal": signal}
        if score is not None:
            outcome["score"] = score
        rec["outcome"] = outcome
    rec.update(extra)
    return rec


def _write_sink(tmp_path: Path, records: list[dict], name="rollouts.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def test_load_sink_valid(tmp_path):
    p = _write_sink(tmp_path, [_rec(), _rec("ro-2", "user_rejected")])
    records, digest = load_rollout_sink(p)
    assert len(records) == 2
    assert len(digest) == 64


def test_load_sink_missing_file_raises(tmp_path):
    with pytest.raises(RolloutIngestError, match="not found"):
        load_rollout_sink(tmp_path / "nope.jsonl")


def test_load_sink_empty_raises(tmp_path):
    p = _write_sink(tmp_path, [])
    with pytest.raises(RolloutIngestError, match="no valid rollout records"):
        load_rollout_sink(p)


def test_load_sink_skips_bad_lines(tmp_path, caplog):
    p = tmp_path / "rollouts.jsonl"
    rec_no_model = _rec("ro-bad")
    del rec_no_model["model"]
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps(rec_no_model) + "\n")
        fh.write(json.dumps({"schema_version": 2, "rollout_id": "x"}) + "\n")
        fh.write(json.dumps(_rec("ro-bad", messages=[])) + "\n")
        fh.write(json.dumps(_rec("ro-ok")) + "\n")
    records, _ = load_rollout_sink(p)
    assert len(records) == 1
    assert records[0]["rollout_id"] == "ro-ok"
    assert "skipped 4 bad lines" in caplog.text


def test_reward_score_signal_map():
    assert reward_score({"signal": "task_success"}) == 1.0
    assert reward_score({"signal": "user_accepted"}) == 1.0
    assert reward_score({"signal": "user_edited"}) == 0.5
    assert reward_score({"signal": "user_rejected"}) == 0.0
    assert reward_score({"signal": "error"}) == 0.0
    assert reward_score({"signal": "none"}) is None
    assert reward_score(None) is None
    assert reward_score("junk") is None


def test_reward_score_explicit_overrides_signal():
    assert reward_score({"signal": "user_rejected", "score": 0.9}) == 0.9
    assert reward_score({"signal": "task_success", "score": 0.2}) == 0.2


def test_reward_score_out_of_range_falls_back_to_signal():
    assert reward_score({"signal": "user_edited", "score": 1.5}) == 0.5
    assert reward_score({"signal": "user_edited", "score": True}) == 0.5


def test_positive_rollouts_min_score(tmp_path):
    records = [
        _rec("a", "task_success"),
        _rec("b", "user_accepted"),
        _rec("c", "user_edited"),
        _rec("d", "none"),
        _rec("e", "user_rejected"),
        _rec("f", "user_edited", score=0.9),
    ]
    assert [r["rollout_id"] for r in positive_rollouts(records)] == ["a", "b"]
    assert [r["rollout_id"] for r in positive_rollouts(records, min_score=0.5)] == [
        "a",
        "b",
        "c",
        "f",
    ]


def test_sft_messages_preserved():
    recs = positive_rollouts([_rec("a"), _rec("b", "user_rejected")])
    samples = rollouts_to_sft(recs)
    assert len(samples) == 1
    assert samples[0]["messages"][0]["role"] == "user"
    assert samples[0]["messages"][1]["content"] == "did a"


def test_grpo_prompts_reward_ref_and_metadata():
    recs = [
        _rec("a", "task_success", prompt="q1"),
        _rec("b", "user_edited", prompt="q2"),
        _rec("c", "none", prompt="", messages=[{"role": "assistant", "content": "x"}]),
        _rec("d", "none", prompt=None),
    ]
    samples = rollouts_to_grpo(recs)
    assert len(samples) == 3
    assert samples[0]["prompt"] == "q1"
    assert samples[0]["reward"] == 1.0
    assert samples[1]["reward"] == 0.5
    assert "reward" not in samples[2]
    assert samples[2]["metadata"]["rollout_id"] == "d"
    assert samples[2]["metadata"]["model"] == "test-model"


def test_grpo_prompt_falls_back_to_first_user_turn():
    rec = _rec("a", "task_success", prompt=None)
    samples = rollouts_to_grpo([rec])
    assert samples[0]["prompt"] == "do thing"


def test_build_sft_and_grpo_writes_manifest(tmp_path):
    sink = _write_sink(
        tmp_path,
        [
            _rec("a", "task_success"),
            _rec("b", "user_rejected"),
            _rec("c", "none", prompt="q3"),
        ],
    )
    sft_out = tmp_path / "sft.jsonl"
    grpo_out = tmp_path / "grpo.jsonl"
    manifest = build_from_rollouts(sink, sft_out, grpo_out)
    sft_lines = [json.loads(l) for l in sft_out.read_text().splitlines()]
    grpo_lines = [json.loads(l) for l in grpo_out.read_text().splitlines()]
    assert len(sft_lines) == 1
    assert len(grpo_lines) == 3
    assert manifest["sft_samples"] == 1
    assert manifest["grpo_prompts"] == 3
    assert manifest["signals"] == {"task_success": 1, "user_rejected": 1, "none": 1}
    assert len(manifest["sink_sha256"]) == 64
    sidecar = tmp_path / "sft.jsonl.from_rollouts.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["source"] == "gateway-rollout-tap"


def test_build_grpo_only_manifest_beside_grpo_output(tmp_path):
    sink = _write_sink(tmp_path, [_rec("a", "none", prompt="q")])
    grpo_out = tmp_path / "prompts.jsonl"
    manifest = build_from_rollouts(sink, None, grpo_out)
    assert manifest["sft_path"] == ""
    assert (tmp_path / "prompts.jsonl.from_rollouts.json").is_file()


def test_build_requires_at_least_one_output(tmp_path):
    sink = _write_sink(tmp_path, [_rec()])
    with pytest.raises(RolloutIngestError, match="nothing to write"):
        build_from_rollouts(sink, None, None)


def test_build_sft_no_positives_raises(tmp_path):
    sink = _write_sink(tmp_path, [_rec("a", "user_rejected"), _rec("b", "none")])
    with pytest.raises(RolloutIngestError, match="no positive rollouts"):
        build_from_rollouts(sink, tmp_path / "sft.jsonl")


def test_build_grpo_no_prompts_raises(tmp_path):
    sink = _write_sink(
        tmp_path,
        [_rec("a", "task_success", prompt=None, messages=[{"role": "assistant", "content": "x"}])],
    )
    with pytest.raises(RolloutIngestError, match="no prompts extractable"):
        build_from_rollouts(sink, None, tmp_path / "grpo.jsonl")


def test_find_rollout_manifest(tmp_path):
    sink = _write_sink(tmp_path, [_rec("a", "task_success")])
    sft_out = tmp_path / "sft.jsonl"
    build_from_rollouts(sink, sft_out)
    found = find_rollout_manifest(sft_out)
    assert found is not None
    assert found["sft_samples"] == 1
    assert find_rollout_manifest(tmp_path / "missing.jsonl") is None


def test_find_rollout_manifest_unreadable_returns_none(tmp_path):
    p = tmp_path / "sft.jsonl.from_rollouts.json"
    p.write_text("not json")
    assert find_rollout_manifest(tmp_path / "sft.jsonl") is None


def test_trainer_find_manifest_picks_rollout_sidecar(tmp_path):
    """SFTTrainer._find_eval_manifest falls through to the rollout sidecar (#86)."""
    from fusion_trainer.dataset.loader import DatasetLoader

    sink = _write_sink(tmp_path, [_rec("a", "user_accepted")])
    sft_out = tmp_path / "sft.jsonl"
    build_from_rollouts(sink, sft_out)

    loader = DatasetLoader()
    resolved = loader._resolve_path(path=str(sft_out))
    from fusion_trainer.dataset.eval_ingest import find_eval_manifest
    from fusion_trainer.dataset.rollout_ingest import find_rollout_manifest

    assert find_eval_manifest(resolved) is None
    manifest = find_rollout_manifest(resolved)
    assert manifest is not None
    assert manifest.get("sink_sha256")
