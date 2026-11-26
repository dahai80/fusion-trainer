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
    failure_rollouts,
    find_rollout_manifest,
    load_rollout_sink,
    positive_rollouts,
    reward_score,
    rollouts_to_grpo,
    rollouts_to_sft,
    struggle_features,
    tool_call_stats,
    weakness_profile,
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
        [
            _rec(
                "a",
                "task_success",
                prompt=None,
                messages=[{"role": "assistant", "content": "x"}],
            )
        ],
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


# ---------------------------------------------------------------------------
# outcome sidecar join (#97)
# ---------------------------------------------------------------------------

from fusion_trainer.dataset.rollout_ingest import (  # noqa: E402
    join_outcomes,
    load_outcome_sidecar,
)


def _sidecar_line(rollout_id, signal="task_success", score=None):
    rec = {
        "schema_version": 1,
        "rollout_id": rollout_id,
        "model": "test-model",
        "messages": [{"role": "user", "content": "x"}],
        "outcome": {"signal": signal},
    }
    if score is not None:
        rec["outcome"]["score"] = score
    return rec


def test_load_outcome_sidecar_latest_wins_and_loud_skips(tmp_path):
    p = tmp_path / "outcomes.jsonl"
    lines = [
        _sidecar_line("ro-1", "user_rejected"),
        '{"broken',
        json.dumps({"schema_version": 1, "rollout_id": "ro-x", "messages": []}),
        _sidecar_line("ro-1", "task_success", 1.0),
        _sidecar_line("ro-2", "user_accepted"),
    ]
    p.write_text(
        "\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n",
        encoding="utf-8",
    )
    outcomes = load_outcome_sidecar(p)
    assert set(outcomes) == {"ro-1", "ro-2"}
    assert outcomes["ro-1"]["signal"] == "task_success"


def test_load_outcome_sidecar_missing_raises(tmp_path):
    with pytest.raises(RolloutIngestError, match="not found"):
        load_outcome_sidecar(tmp_path / "nope.jsonl")


def test_join_outcomes_inline_wins():
    records = [
        _rec("ro-1", signal=None),
        _rec("ro-2", signal="user_edited"),
    ]
    outcomes = {"ro-1": {"signal": "task_success"}, "ro-2": {"signal": "error"}}
    joined, n = join_outcomes(records, outcomes)
    assert n == 1
    assert joined[0]["outcome"]["signal"] == "task_success"
    assert joined[1]["outcome"]["signal"] == "user_edited"


def test_build_sft_with_outcomes_join(tmp_path):
    tap = _write_sink(
        tmp_path,
        [_rec("ro-1", signal=None), _rec("ro-2", signal=None)],
        name="tap.jsonl",
    )
    side = tmp_path / "outcomes.jsonl"
    side.write_text(
        json.dumps(_sidecar_line("ro-1", "task_success", 1.0)) + "\n", encoding="utf-8"
    )
    sft_out = tmp_path / "sft.jsonl"
    manifest = build_from_rollouts(tap, sft_out, outcomes_path=side)
    assert manifest["outcomes_joined"] == 1
    assert manifest["sft_samples"] == 1
    assert manifest["lineage"]["transforms"][0] == "join_outcomes"
    samples = [json.loads(l) for l in sft_out.read_text().splitlines()]
    assert samples[0]["messages"][1]["content"] == "did ro-1"


def test_build_sft_outcomes_join_still_no_positives(tmp_path):
    tap = _write_sink(tmp_path, [_rec("ro-1", signal=None)], name="tap.jsonl")
    side = tmp_path / "outcomes.jsonl"
    side.write_text(
        json.dumps(_sidecar_line("ro-9", "task_success", 1.0)) + "\n", encoding="utf-8"
    )
    with pytest.raises(RolloutIngestError, match="no positive rollouts"):
        build_from_rollouts(tap, tmp_path / "sft.jsonl", outcomes_path=side)


# ---------------------------------------------------------------------------
# failure channel + tool-call efficiency stats (#111)
# ---------------------------------------------------------------------------


def test_failure_rollouts_scored_negatives_only():
    records = [
        _rec("ro-1", signal="task_success"),
        _rec("ro-2", signal="user_rejected"),
        _rec("ro-3", signal="error", score=0.2),
        _rec("ro-4", signal=None),  # unobserved: neither positive nor failure
    ]
    failures = failure_rollouts(records)
    assert [r["rollout_id"] for r in failures] == ["ro-2", "ro-3"]


def test_weakness_profile_counts_and_caps_samples():
    records = [_rec(f"ro-{i}", signal="user_rejected") for i in range(15)]
    records += [_rec("err-1", signal="error")]
    profile = weakness_profile(failure_rollouts(records))
    assert profile["failure_signals"] == {"user_rejected": 15, "error": 1}
    assert len(profile["sample_rollout_ids"]["user_rejected"]) == 10
    assert profile["sample_rollout_ids"]["error"] == ["err-1"]


def test_tool_call_stats_consecutive_duplicates():
    dup_call = {"type": "function", "function": {"name": "read", "arguments": '{"path": "a.xlsx"}'}}
    other = {"type": "function", "function": {"name": "write", "arguments": {"path": "b.docx"}}}
    records = [
        {
            "rollout_id": "ro-1",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [dup_call, dup_call, other]},
                {"role": "user", "content": "again"},
                {
                    "role": "assistant",
                    "content": "",
                    # same key as turn 1 but a NEW user turn resets the anchor
                    "tool_calls": [dup_call],
                },
            ],
        }
    ]
    stats = tool_call_stats(records)
    assert stats["records_with_tool_calls"] == 1
    assert stats["tool_calls"] == 4
    assert stats["redundant_consecutive"] == 1
    assert stats["redundant_rate"] == 0.25


def test_tool_call_stats_empty_and_unparseable():
    assert tool_call_stats([])["tool_calls"] == 0
    assert tool_call_stats([_rec("ro-1")])["redundant_rate"] == 0.0
    bad = {
        "rollout_id": "ro-2",
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]}
        ],
    }
    # arguments missing -> key still forms with None-ish args
    assert tool_call_stats([bad])["tool_calls"] == 1


def test_build_failures_only_channel(tmp_path):
    tap = _write_sink(
        tmp_path,
        [
            _rec("ro-1", signal="task_success"),
            _rec("ro-2", signal="user_rejected"),
            _rec("ro-3", signal=None),
        ],
    )
    fail_out = tmp_path / "failures.jsonl"
    manifest = build_from_rollouts(tap, failures_output_path=fail_out)
    assert manifest["failures_records"] == 1
    assert manifest["failures_path"] == str(fail_out)
    assert manifest["weakness"]["failure_signals"] == {"user_rejected": 1}
    assert manifest["weakness"]["sample_rollout_ids"]["user_rejected"] == ["ro-2"]
    lines = [json.loads(l) for l in fail_out.read_text().splitlines()]
    assert lines[0]["rollout_id"] == "ro-2"
    assert lines[0]["outcome"]["signal"] == "user_rejected"
    assert manifest["lineage"]["transforms"] == ["failure_rollouts"]


def test_build_failures_empty_set_writes_empty_file(tmp_path):
    tap = _write_sink(tmp_path, [_rec("ro-1", signal="task_success")])
    fail_out = tmp_path / "failures.jsonl"
    manifest = build_from_rollouts(tap, failures_output_path=fail_out)
    assert manifest["failures_records"] == 0
    assert fail_out.exists() and fail_out.read_text() == ""


def test_build_manifest_carries_efficiency(tmp_path):
    call = {"type": "function", "function": {"name": "read", "arguments": "{}"}}
    tap = _write_sink(
        tmp_path,
        [
            _rec(
                "ro-1",
                signal="task_success",
                messages=[
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": "", "tool_calls": [call, call]},
                ],
            )
        ],
    )
    manifest = build_from_rollouts(tap, tmp_path / "sft.jsonl")
    assert manifest["efficiency"]["tool_calls"] == 2
    assert manifest["efficiency"]["redundant_consecutive"] == 1


def test_build_failures_with_sft_and_lineage(tmp_path):
    tap = _write_sink(
        tmp_path,
        [
            _rec("ro-1", signal="task_success"),
            _rec("ro-2", signal="user_rejected"),
        ],
    )
    sft_out = tmp_path / "sft.jsonl"
    fail_out = tmp_path / "failures.jsonl"
    manifest = build_from_rollouts(tap, sft_out, failures_output_path=fail_out)
    assert manifest["sft_samples"] == 1 and manifest["failures_records"] == 1
    assert manifest["lineage"]["transforms"] == [
        "positive_rollouts",
        "rollouts_to_sft",
        "failure_rollouts",
    ]
    assert manifest["lineage"]["output"] == str(sft_out)


def test_struggle_features_scores_and_groups():
    dup = {"type": "function", "function": {"name": "read", "arguments": "{}"}}
    fail_a = {
        "rollout_id": "a",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [dup, dup]},
        ],
        "outcome": {"signal": "error"},
    }
    clean_fail = {
        "rollout_id": "b",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "nope"},
        ],
        "outcome": {"signal": "user_rejected"},
    }
    feats = struggle_features([fail_a, clean_fail])
    assert feats["a"]["redundant_consecutive"] == 1
    assert feats["a"]["error_signals"] == 1
    assert feats["a"]["struggle_score"] == round((1 + 2) / 1, 4)
    assert feats["b"]["struggle_score"] == 0.0


def test_weakness_profile_task_breakdown_when_present():
    def _task_rec(rid, signal, task_id):
        rec = _rec(rid, signal=signal)
        rec["outcome"]["task_id"] = task_id
        return rec

    records = [
        _task_rec("ro-1", "user_rejected", "t-1"),
        _task_rec("ro-2", "user_rejected", "t-2"),
        _task_rec("ro-3", "error", "t-1"),
    ]
    profile = weakness_profile(failure_rollouts(records))
    assert profile["failure_by_task"] == {"t-1": 2, "t-2": 1}


def test_weakness_profile_no_task_breakdown_when_absent():
    profile = weakness_profile(failure_rollouts([_rec("ro-1", signal="error")]))
    assert "failure_by_task" not in profile


def test_build_manifest_struggle_only_nonzero(tmp_path):
    dup = {"type": "function", "function": {"name": "read", "arguments": "{}"}}
    fail_rec = {
        "schema_version": 1,
        "rollout_id": "a",
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [dup, dup]},
        ],
        "prompt": "go",
        "outcome": {"signal": "error"},
    }
    clean_fail = _rec("ro-ok", signal="user_rejected")
    tap = _write_sink(tmp_path, [fail_rec, clean_fail])
    manifest = build_from_rollouts(tap, failures_output_path=tmp_path / "f.jsonl")
    assert set(manifest["struggle"]) == {"a"}
    assert manifest["struggle"]["a"]["redundant_consecutive"] == 1
