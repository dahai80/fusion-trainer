"""Tests for the office task registry module (#120)."""

from pathlib import Path

import pytest

from fusion_trainer.tasks import (
    TaskRegistryError,
    gate_report,
    load_task_registry,
    synthesize_stage2_tasks,
    upsert_task,
    validator_gate,
)


def _task(tid="t-1", stage=1, status="passing", solutions=None, **extra):
    rec = {
        "task_id": tid,
        "curriculum_stage": stage,
        "validator": {"name": f"val-{tid}", "version": "1", "status": status},
        "known_valid_solutions": solutions
        if solutions is not None
        else ["sol-a", "sol-b"],
    }
    rec.update(extra)
    return rec


def test_validator_gate_passing_with_corpus():
    ok, reason = validator_gate(_task())
    assert ok
    assert "passing validator v1" in reason
    assert "2 valid solutions" in reason


def test_validator_gate_missing_validator():
    task = _task()
    del task["validator"]
    ok, reason = validator_gate(task)
    assert not ok
    assert reason == "no validator record"


def test_validator_gate_untested():
    ok, reason = validator_gate(_task(status="untested"))
    assert not ok
    assert "untested" in reason


def test_validator_gate_failing():
    ok, reason = validator_gate(_task(status="failing"))
    assert not ok
    assert "rejects its own known-valid corpus" in reason


def test_validator_gate_empty_corpus():
    ok, reason = validator_gate(_task(solutions=[]))
    assert not ok
    assert reason == "empty known-valid solution corpus"


def test_upsert_validates(tmp_path: Path):
    with pytest.raises(TaskRegistryError):
        upsert_task(tmp_path / "registry.jsonl", {"task_id": "x"})
    with pytest.raises(TaskRegistryError):
        upsert_task(tmp_path / "registry.jsonl", _task(stage=9))
    with pytest.raises(TaskRegistryError):
        upsert_task(tmp_path / "registry.jsonl", _task(status="bogus"))
    with pytest.raises(TaskRegistryError):
        upsert_task(tmp_path / "registry.jsonl", _task(solutions=["", "  "]))


def test_upsert_then_load_latest_wins(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    upsert_task(path, _task(tid="t-1", status="untested", solutions=[]))
    upsert_task(path, _task(tid="t-1", status="passing"))
    upsert_task(path, _task(tid="t-2"))
    tasks = load_task_registry(path)
    assert set(tasks) == {"t-1", "t-2"}
    assert tasks["t-1"]["validator"]["status"] == "passing"
    assert tasks["t-1"]["created_at"]


def test_load_registry_missing_raises(tmp_path: Path):
    with pytest.raises(TaskRegistryError, match="not found"):
        load_task_registry(tmp_path / "nope.jsonl")


def test_load_registry_empty_raises(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    path.write_text("\n{not json}\n", encoding="utf-8")
    with pytest.raises(TaskRegistryError, match="no task records"):
        load_task_registry(path)


def test_gate_report_masks_and_groups(tmp_path: Path):
    tasks = {
        "a": _task(tid="a"),
        "b": _task(tid="b", status="untested", solutions=[]),
        "c": _task(tid="c", stage=2),
    }
    report = gate_report(tasks)
    assert report["total"] == 3
    assert report["allowed"] == ["a", "c"]
    assert len(report["masked"]) == 1
    assert report["masked"][0]["task_id"] == "b"
    assert report["by_stage"] == {"1": 2, "2": 1}


def test_synthesize_stage2_from_graph_edges():
    graph = {
        "nodes": [{"id": "doc-tool"}],
        "edges": [
            {
                "producer": "doc-tool",
                "consumer": "sheet-tool",
                "artifact": "docx",
                "stage": 2,
                "source_artifacts": ["s1"],
            },
            {
                "from": "mail-tool",
                "to": "cal-tool",
                "artifact_type": "invite",
                "stage": 3,
            },
            {"producer": "x", "consumer": "y", "artifact": "z", "stage": 1},
        ],
    }
    tasks = synthesize_stage2_tasks(graph)
    assert [t["task_id"] for t in tasks] == [
        "edge-doc-tool-sheet-tool-docx",
        "edge-mail-tool-cal-tool-invite",
        "edge-x-y-z",
    ]
    assert all(t["curriculum_stage"] == 2 for t in tasks)
    assert all(t["validator"]["status"] == "untested" for t in tasks)
    assert all(t["known_valid_solutions"] == [] for t in tasks)
    assert tasks[0]["edge"]["producer"] == "doc-tool"
    assert tasks[0]["source_artifacts"] == ["s1"]


def test_synthesize_requires_edges_list():
    with pytest.raises(TaskRegistryError, match="no edges list"):
        synthesize_stage2_tasks({"nodes": []})
    with pytest.raises(TaskRegistryError, match="no edges list"):
        synthesize_stage2_tasks("nope")


def test_synthesized_tasks_start_masked(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    for task in synthesize_stage2_tasks(
        {"edges": [{"producer": "a", "consumer": "b", "artifact": "c", "stage": 2}]}
    ):
        upsert_task(path, task)
    tasks = load_task_registry(path)
    ok, reason = validator_gate(tasks["edge-a-b-c"])
    assert not ok
    assert "untested" in reason
