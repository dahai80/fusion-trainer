"""Tests for per-task efficiency budgets (#117 CARE-PO)."""

import json
from pathlib import Path

import pytest

from fusion_trainer.dataset.rollout_ingest import (
    RolloutIngestError,
    budget_penalty,
    build_from_rollouts,
    efficiency_budgets,
)


def _pos(rid, task, calls):
    msgs = []
    for i in range(calls):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "open", "arguments": {"p": i}}}],
            }
        )
    return {
        "schema_version": 1,
        "rollout_id": rid,
        "model": "m",
        "messages": msgs,
        "outcome": {"signal": "task_success", "score": 1.0, "task_id": task},
    }


def test_budget_is_p50_of_positive_baselines():
    records = [
        _pos("r1", "t-a", 3),
        _pos("r2", "t-a", 5),
        _pos("r3", "t-a", 7),
    ]
    out = efficiency_budgets(records)
    assert out["tasks"]["t-a"]["budget_tool_calls"] == 5
    assert out["tasks"]["t-a"]["positives"] == 3
    assert out["tasks"]["t-a"]["max"] == 7
    assert out["omitted"] == []


def test_thin_baselines_omitted_loudly():
    records = [_pos("r1", "t-a", 3), _pos("r2", "t-a", 4), _pos("r3", "t-b", 9)]
    out = efficiency_budgets(records)
    assert out["tasks"] == {}
    assert out["omitted"] == [
        {"task_id": "t-a", "positives": 2},
        {"task_id": "t-b", "positives": 1},
    ]


def test_min_positives_override():
    records = [_pos("r1", "t-a", 3), _pos("r2", "t-a", 4)]
    out = efficiency_budgets(records, min_positives=2)
    assert out["tasks"]["t-a"]["budget_tool_calls"] in (3, 4)


def test_negatives_and_unobserved_excluded():
    records = [
        _pos("r1", "t-a", 3),
        {
            "schema_version": 1,
            "rollout_id": "r2",
            "model": "m",
            "messages": [],
            "outcome": {"signal": "user_rejected", "score": 0.0, "task_id": "t-a"},
        },
        {"schema_version": 1, "rollout_id": "r3", "model": "m", "messages": []},
    ]
    out = efficiency_budgets(records, min_positives=1)
    assert out["positives_total"] == 1
    assert out["tasks"]["t-a"]["positives"] == 1


def test_budget_penalty_bonus_only():
    assert budget_penalty(5, 6, 0) == 0.1
    assert budget_penalty(7, 6, 0) == 0.0
    assert budget_penalty(5, 6, 2) == 0.0
    assert budget_penalty(0, 6, 0) == 0.0


def test_build_from_rollouts_writes_budgets(tmp_path: Path):
    sink = tmp_path / "tap.jsonl"
    with open(sink, "w", encoding="utf-8") as fh:
        for rid, task, calls in [("r1", "t-a", 3), ("r2", "t-a", 5), ("r3", "t-a", 7)]:
            fh.write(json.dumps(_pos(rid, task, calls)) + "\n")
    budgets_path = tmp_path / "budgets.json"
    manifest = build_from_rollouts(sink, budgets_output_path=budgets_path)
    data = json.loads(budgets_path.read_text(encoding="utf-8"))
    assert data["tasks"]["t-a"]["budget_tool_calls"] == 5
    assert manifest["budgets_tasks"] == 1
    assert manifest["budgets_omitted"] == 0
    assert "efficiency_budgets" in manifest["lineage"]["transforms"]
    assert manifest["lineage"]["output"] == str(budgets_path)


def test_build_from_rollouts_empty_budgets_honest(tmp_path: Path):
    sink = tmp_path / "tap.jsonl"
    rec = {
        "schema_version": 1,
        "rollout_id": "r1",
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    with open(sink, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    budgets_path = tmp_path / "budgets.json"
    manifest = build_from_rollouts(sink, budgets_output_path=budgets_path)
    assert json.loads(budgets_path.read_text(encoding="utf-8"))["tasks"] == {}
    assert manifest["budgets_tasks"] == 0


def test_build_from_rollouts_still_requires_something(tmp_path: Path):
    with pytest.raises(RolloutIngestError, match="nothing to write"):
        build_from_rollouts(tmp_path / "tap.jsonl")
