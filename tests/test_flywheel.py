"""Eval-trajectory ingestion + flywheel registry tests (issue #76).

Fully offline: synthetic exports in tmp_path, no fusion-mlx / fusion-eval.
"""

from __future__ import annotations

import json

import pytest

from fusion_trainer.dataset.eval_ingest import (
    EvalIngestError,
    build_from_eval_dataset,
    eval_records_to_sft,
    find_eval_manifest,
    load_eval_export,
    mine_preference_pairs,
)
from fusion_trainer.flywheel import (
    FlywheelError,
    format_status,
    load_registry,
    parse_eval_summary,
    record_round,
)


def _rec(
    task_id: str,
    passed: bool,
    messages=None,
    task_type: str = "bugfix",
    prompt: str = "fix it",
):
    return {
        "schema_version": 1,
        "task_id": task_id,
        "task_type": task_type,
        "arm": "fc-local",
        "model": "test/test-model",
        "passed": passed,
        "prompt": prompt,
        "messages": messages
        or [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "fixed it"},
        ],
        "final_diff": None,
    }


def _write_export(tmp_path, records, name="trajectories.jsonl"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


# ---------------------------------------------------------------------------
# load_eval_export
# ---------------------------------------------------------------------------


def test_load_export_valid(tmp_path):
    p = _write_export(tmp_path, [_rec("t1", True), _rec("t2", False)])
    records, digest = load_eval_export(p)
    assert len(records) == 2
    assert len(digest) == 64


def test_load_export_skips_bad_schema(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps(_rec("t1", True))
        + "\n"
        + json.dumps({"schema_version": 99, "task_id": "t2"})
        + "\n"
        + "not json\n"
        + json.dumps({"schema_version": 1, "task_id": "t3", "passed": True})
        + "\n",
        encoding="utf-8",
    )
    records, _ = load_eval_export(p)
    assert [r["task_id"] for r in records] == ["t1"]


def test_load_export_missing_file(tmp_path):
    with pytest.raises(EvalIngestError):
        load_eval_export(tmp_path / "nope.jsonl")


def test_load_export_empty_raises(tmp_path):
    p = _write_export(tmp_path, [])
    with pytest.raises(EvalIngestError):
        load_eval_export(p)


# ---------------------------------------------------------------------------
# SFT conversion
# ---------------------------------------------------------------------------


def test_sft_passed_only_multi_turn_preserved():
    multi = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    samples = eval_records_to_sft([_rec("t1", True, messages=multi), _rec("t2", False)])
    assert len(samples) == 1
    assert [m["role"] for m in samples[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_sft_skips_passed_with_invalid_messages():
    bad = _rec("t1", True, messages=[{"role": "user", "content": ""}])
    assert eval_records_to_sft([bad]) == []


# ---------------------------------------------------------------------------
# DPO mining
# ---------------------------------------------------------------------------


def test_mine_pairs_passed_vs_failed_same_task():
    records = [
        _rec("t1", True),
        _rec(
            "t1",
            False,
            messages=[
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "broke it"},
            ],
        ),
        _rec("t2", False),
    ]
    pairs = mine_preference_pairs(records)
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "fix it"
    assert pairs[0]["chosen"] == "fixed it"
    assert pairs[0]["rejected"] == "broke it"
    assert pairs[0]["task_id"] == "t1"


def test_mine_pairs_no_passed_sibling_skips():
    assert mine_preference_pairs([_rec("t1", False)]) == []


def test_mine_pairs_passed_only_yields_none():
    assert mine_preference_pairs([_rec("t1", True)]) == []


# ---------------------------------------------------------------------------
# build_from_eval_dataset + manifest
# ---------------------------------------------------------------------------


def test_build_dataset_writes_manifest(tmp_path):
    p = _write_export(
        tmp_path,
        [
            _rec("t1", True),
            _rec(
                "t1",
                False,
                messages=[
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "content": "broke it"},
                ],
            ),
            _rec("t2", True, task_type="feature"),
            _rec("t3", False),
        ],
    )
    out = tmp_path / "ds" / "sft.jsonl"
    dpo_out = tmp_path / "ds" / "dpo.jsonl"
    manifest = build_from_eval_dataset(p, out, dpo_out)
    assert manifest["sft_samples"] == 2
    assert manifest["records_passed"] == 2
    assert manifest["records_failed"] == 2
    assert manifest["dpo_pairs"] == 1
    assert manifest["dpo_structurally_empty"] is False
    assert manifest["arms"] == ["fc-local"]
    assert manifest["per_type"]["bugfix"] == {"total": 3, "passed": 1, "failed": 2}
    assert out.is_file()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "messages" in json.loads(lines[0])
    assert len(manifest["export_sha256"]) == 64
    found = find_eval_manifest(out)
    assert found is not None and found["export_sha256"] == manifest["export_sha256"]


def test_build_dataset_dpo_gate_single_arm_raises(tmp_path):
    p = _write_export(tmp_path, [_rec("t1", True), _rec("t2", False)])
    with pytest.raises(EvalIngestError, match="single-arm"):
        build_from_eval_dataset(p, tmp_path / "sft.jsonl", tmp_path / "dpo.jsonl")
    assert not (tmp_path / "dpo.jsonl").exists()


def test_build_dataset_dpo_gate_allow_empty(tmp_path):
    p = _write_export(tmp_path, [_rec("t1", True), _rec("t2", False)])
    out = tmp_path / "sft.jsonl"
    dpo_out = tmp_path / "dpo.jsonl"
    manifest = build_from_eval_dataset(p, out, dpo_out, dpo_allow_empty=True)
    assert manifest["dpo_pairs"] == 0
    assert manifest["dpo_structurally_empty"] is True
    assert dpo_out.read_text(encoding="utf-8").strip() == ""


def test_build_dataset_no_passed_raises(tmp_path):
    p = _write_export(tmp_path, [_rec("t1", False)])
    with pytest.raises(EvalIngestError):
        build_from_eval_dataset(p, tmp_path / "sft.jsonl")


def test_find_manifest_missing_returns_none(tmp_path):
    p = tmp_path / "plain.jsonl"
    p.write_text('{"messages": []}\n', encoding="utf-8")
    assert find_eval_manifest(p) is None
    assert find_eval_manifest("") is None


# ---------------------------------------------------------------------------
# flywheel registry
# ---------------------------------------------------------------------------


def _summary(
    tmp_path, run_id="r1", pass_rate=0.5, arm="fc-local", model="m1", by_type=None
):
    data = {
        "run_id": run_id,
        "models": {arm: model},
        "arms": {
            arm: {
                "total": 30,
                "passed": int(pass_rate * 30),
                "pass_rate": pass_rate,
                "by_type": by_type
                or {"bugfix": {"total": 3, "passed": 1, "pass_rate": 0.3333}},
            }
        },
    }
    p = tmp_path / f"{run_id}-summary.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_parse_summary_single_arm(tmp_path):
    parsed = parse_eval_summary(_summary(tmp_path))
    assert parsed["eval_run_id"] == "r1"
    assert parsed["pass_rate"] == 0.5
    assert parsed["model"] == "m1"
    assert parsed["by_type"]["bugfix"]["pass_rate"] == 0.3333


def test_parse_summary_multi_arm_defaults_sorted_first(tmp_path):
    p = _summary(tmp_path, arm="b-arm")
    p.write_text(
        p.read_text(encoding="utf-8")
        .replace('"b-arm"', '"b-arm"')
        .replace(
            json.dumps({"b-arm": {"total": 30}}), json.dumps({"b-arm": {"total": 30}})
        ),
        encoding="utf-8",
    )
    data = json.loads(p.read_text(encoding="utf-8"))
    data["arms"]["a-arm"] = dict(data["arms"]["b-arm"], pass_rate=0.1)
    data["models"]["a-arm"] = "ma"
    p.write_text(json.dumps(data), encoding="utf-8")
    parsed = parse_eval_summary(p)
    assert parsed["arm"] == "a-arm"
    assert parse_eval_summary(p, arm="b-arm")["arm"] == "b-arm"


def test_parse_summary_bad_arm_raises(tmp_path):
    with pytest.raises(FlywheelError):
        parse_eval_summary(_summary(tmp_path), arm="nope")


def test_record_and_status_curve(tmp_path):
    cache = str(tmp_path / "cache")
    record_round(
        cache,
        label="baseline",
        checkpoint="base-model",
        summary_path=_summary(tmp_path, "r0", 0.5333),
    )
    record_round(
        cache,
        label="round-1-sft",
        checkpoint="/adapters/round1",
        summary_path=_summary(tmp_path, "r1", 0.6),
        export_path=_write_export(tmp_path, [_rec("t1", True)], name="e1.jsonl"),
    )
    record_round(
        cache,
        label="round-2-sft",
        checkpoint="/adapters/round2",
        summary_path=_summary(tmp_path, "r2", 0.7),
    )
    reg = load_registry(cache)
    assert len(reg["rounds"]) == 3
    assert reg["rounds"][1]["export_sha256"]
    text = format_status(cache)
    assert "baseline" in text and "round-1-sft" in text and "round-2-sft" in text
    assert "UP" in text
    assert "(baseline)" in text


def test_record_export_sha_computed_from_file(tmp_path):
    cache = str(tmp_path / "cache")
    export = _write_export(tmp_path, [_rec("t1", True)])
    entry = record_round(
        cache,
        label="r1",
        checkpoint="c",
        summary_path=_summary(tmp_path),
        export_path=str(export),
    )
    assert len(entry["export_sha256"]) == 64


def test_status_empty_registry(tmp_path):
    text = format_status(str(tmp_path / "empty"))
    assert "No flywheel rounds" in text


def test_registry_corrupt_raises(tmp_path):
    cache = tmp_path
    (cache / "flywheel.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FlywheelError):
        load_registry(str(cache))


# ---------------------------------------------------------------------------
# SFTTrainer source-hash hook (#76 requirement 3)
# ---------------------------------------------------------------------------


def test_sft_trainer_finds_eval_manifest(tmp_path):
    from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
    from fusion_trainer.sft.trainer import SFTTrainer

    export = _write_export(tmp_path, [_rec("t1", True)])
    out = tmp_path / "sft.jsonl"
    manifest = build_from_eval_dataset(export, out)
    cfg = TrainerConfig(
        dataset=DatasetConfig(path=str(out), max_seq_length=128),
        sft=SFTConfig(model="test-model", method="lora"),
        cache_path=str(tmp_path),
    )
    trainer = SFTTrainer(cfg)
    found = trainer._find_eval_manifest()
    assert found is not None
    assert found["export_sha256"] == manifest["export_sha256"]


def test_sft_trainer_no_manifest_plain_dataset(tmp_path):
    from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
    from fusion_trainer.sft.trainer import SFTTrainer

    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"input": "q", "output": "a"}\n', encoding="utf-8")
    cfg = TrainerConfig(
        dataset=DatasetConfig(path=str(plain), max_seq_length=128),
        sft=SFTConfig(model="test-model", method="lora"),
        cache_path=str(tmp_path),
    )
    assert SFTTrainer(cfg)._find_eval_manifest() is None
