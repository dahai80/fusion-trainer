"""Tests for skill mining (#123, gradient-free path 2)."""

import json
from pathlib import Path

import pytest

from fusion_trainer.skills import (
    DEFAULT_MIN_SKILL_SUCCESSES,
    SkillMiningError,
    mine_skills,
    write_skill_library,
)


def _rec(rid, calls, score=1.0, task=None):
    messages = [{"role": "user", "content": "go"}]
    for name, args in calls:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            }
        )
    outcome: dict = {"signal": "task_success", "score": score}
    if task:
        outcome["task_id"] = task
    return {
        "schema_version": 1,
        "rollout_id": rid,
        "model": "m",
        "messages": messages,
        "outcome": outcome,
    }


def test_mine_admits_repeated_success_chain():
    records = [
        _rec("r1", [("open", {"p": 1}), ("save", {"q": 2})]),
        _rec("r2", [("open", {"p": 1}), ("save", {"q": 2})]),
        _rec("r3", [("open", {"p": 9})]),
    ]
    skills = mine_skills(records)
    assert len(skills) == 1
    assert skills[0]["tools"] == ["open", "save"]
    assert skills[0]["success_count"] == 2
    assert skills[0]["example_rollout_id"] == "r1"
    assert len(skills[0]["skill_id"]) == 12


def test_evidence_citations_and_admission_block():
    records = [
        _rec("r1", [("open", {"p": 1})], task="t-a"),
        _rec("r2", [("open", {"p": 1})], task="t-a"),
    ]
    skills = mine_skills(records)
    s = skills[0]
    assert s["schema_version"] == 2
    assert [e["rollout_id"] for e in s["evidence"]] == ["r1", "r2"]
    assert all(
        e["signal"] == "task_success" and e["score"] == 1.0 for e in s["evidence"]
    )
    assert all(e["verifier"] == "outcome_signal" for e in s["evidence"])
    assert s["admission"]["min_successes"] == 2
    assert "reward>=1.0" in s["admission"]["gate"]
    assert s["admission"]["mined_at"]


def test_validator_verified_evidence():
    rec = _rec("r1", [("open", {"p": 1})])
    rec["outcome"]["validator"] = {
        "name": "docx_check",
        "version": "1",
        "status": "passing",
    }
    rec2 = _rec("r2", [("open", {"p": 1})])
    skills = mine_skills([rec, rec2])
    verifiers = {e["verifier"] for e in skills[0]["evidence"]}
    assert verifiers == {"docx_check@v1", "outcome_signal"}


def test_min_successes_default_is_two():
    assert DEFAULT_MIN_SKILL_SUCCESSES == 2
    records = [_rec("r1", [("open", {"p": 1})])]
    assert mine_skills(records) == []
    assert mine_skills(records, min_successes=1)[0]["success_count"] == 1


def test_user_turn_splits_segments_and_counts_distinct_rollouts():
    # Two segments with the same chain inside ONE rollout count once —
    # success_count = distinct successful rollouts (anti-overfit semantic).
    rec = {
        "schema_version": 1,
        "rollout_id": "r1",
        "model": "m",
        "messages": [
            {"role": "user", "content": "a"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "open", "arguments": {}}}],
            },
            {"role": "user", "content": "b"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "open", "arguments": {}}}],
            },
        ],
        "outcome": {"signal": "task_success", "score": 1.0},
    }
    assert mine_skills([rec], min_successes=2) == []
    skills = mine_skills([rec, {**rec, "rollout_id": "r2"}], min_successes=2)
    assert len(skills) == 1
    assert skills[0]["success_count"] == 2


def test_negatives_and_unobserved_excluded():
    records = [
        _rec("r1", [("open", {"p": 1})], score=0.0),
        _rec("r2", [("open", {"p": 1})], score=1.0),
        {
            "schema_version": 1,
            "rollout_id": "r3",
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
        },
    ]
    assert mine_skills(records) == []
    assert mine_skills(records, min_successes=1)[0]["example_rollout_id"] == "r2"


def test_task_ids_collected_when_present():
    records = [
        _rec("r1", [("open", {"p": 1})], task="t-a"),
        _rec("r2", [("open", {"p": 1})], task="t-a"),
        _rec("r3", [("open", {"p": 1})], task="t-b"),
    ]
    skills = mine_skills(records)
    assert skills[0]["task_ids"] == ["t-a", "t-b"]


def test_sorted_by_success_count_desc():
    records = [
        _rec("r1", [("a", {})]),
        _rec("r2", [("a", {})]),
        _rec("r3", [("a", {})]),
        _rec("r4", [("b", {})]),
        _rec("r5", [("b", {})]),
    ]
    skills = mine_skills(records)
    assert [s["success_count"] for s in skills] == [3, 2]


def test_min_successes_validation():
    with pytest.raises(SkillMiningError, match="min_successes"):
        mine_skills([], min_successes=0)


def test_write_skill_library(tmp_path: Path):
    path = tmp_path / "skills.jsonl"
    skills = mine_skills(
        [
            _rec("r1", [("open", {"p": 1})]),
            _rec("r2", [("open", {"p": 1})]),
        ],
    )
    out = write_skill_library(skills, path)
    assert out == path
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["skill_id"] == skills[0]["skill_id"]
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_write_empty_library_honest(tmp_path: Path):
    path = tmp_path / "skills.jsonl"
    write_skill_library([], path)
    assert path.read_text(encoding="utf-8") == ""


def test_load_skill_library_roundtrip_and_validation(tmp_path):
    from fusion_trainer.skills import SKILL_SCHEMA_VERSION, load_skill_library

    path = tmp_path / "skills.jsonl"
    skills = mine_skills([_rec("r1", [("open", {"p": 1})]), _rec("r2", [("open", {"p": 1})])])
    write_skill_library(skills, path)
    loaded = load_skill_library(path)
    assert [s["skill_id"] for s in loaded] == [s["skill_id"] for s in skills]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"skill_id": "x", "chain": ["a"]}\nnot json\n{"skill_id": "y"}\n'
        f'{{"skill_id": "z", "chain": ["a"], "schema_version": {SKILL_SCHEMA_VERSION + 1}}}\n',
        encoding="utf-8",
    )
    loaded_bad = load_skill_library(bad)
    assert [s["skill_id"] for s in loaded_bad] == ["x"]


def test_load_skill_library_missing_raises(tmp_path):
    from fusion_trainer.skills import SkillMiningError, load_skill_library

    with pytest.raises(SkillMiningError, match="not found"):
        load_skill_library(tmp_path / "nope.jsonl")


def test_render_skill_block_deterministic():
    from fusion_trainer.skills import render_skill_block

    assert render_skill_block([]) == ""
    skills = mine_skills(
        [_rec("r1", [("open", {"p": 1}), ("save", {})]), _rec("r2", [("open", {"p": 1}), ("save", {})])]
    )
    block = render_skill_block(skills)
    assert block == render_skill_block(skills)
    assert "Proven procedural skills" in block
    assert "open -> save" in block
    assert "used 2x" in block
    assert "outcome_signal" in block


def test_render_skill_block_budget_respected():
    from fusion_trainer.skills import render_skill_block

    records = []
    for i in range(6):
        for j in range(2):
            records.append(_rec(f"r{i}_{j}", [(f"tool{i}", {"p": 1})]))
    skills = mine_skills(records)
    assert len(skills) == 6
    block = render_skill_block(skills, max_skills=3)
    assert "omitted" in block
    assert "+3 more skills omitted" in block
    assert block.count("used 2x") == 3
