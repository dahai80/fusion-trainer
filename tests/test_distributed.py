"""Distributed orchestration surface tests (capability plan 2.4, T5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_trainer.distributed import (
    DistributedPlanError,
    aggregate_progress,
    load_nodes,
    plan_shards,
)

NODES = [
    {"node_id": "mac-studio", "host": "studio.local", "port": 11434},
    {"node_id": "mac-mini", "host": "mini.local", "port": 11434},
]


def test_load_nodes_from_list():
    nodes = load_nodes(NODES)
    assert [n["node_id"] for n in nodes] == ["mac-studio", "mac-mini"]


def test_load_nodes_from_file(tmp_path):
    p = tmp_path / "nodes.json"
    p.write_text(json.dumps(NODES), encoding="utf-8")
    assert len(load_nodes(p)) == 2


def test_load_nodes_missing_file(tmp_path):
    with pytest.raises(DistributedPlanError, match="not found"):
        load_nodes(tmp_path / "nope.json")


def test_load_nodes_rejects_duplicates():
    dup = [NODES[0], NODES[0]]
    with pytest.raises(DistributedPlanError, match="duplicate node_id"):
        load_nodes(dup)


def test_load_nodes_rejects_bad_port():
    with pytest.raises(DistributedPlanError, match="port"):
        load_nodes([{"node_id": "x", "host": "h", "port": 99999}])


def test_load_nodes_rejects_empty_host():
    with pytest.raises(DistributedPlanError, match="host"):
        load_nodes([{"node_id": "x", "host": "", "port": 1}])


def test_plan_shards_round_robin(tmp_path):
    src = tmp_path / "data.jsonl"
    rows = [json.dumps({"input": f"Q{i}"}) for i in range(10)]
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    plan = plan_shards(src, [n["node_id"] for n in NODES], tmp_path / "shards")
    assert plan["total_lines"] == 10
    assert plan["num_nodes"] == 2
    s0 = Path(plan["shards"][0]["path"])
    s1 = Path(plan["shards"][1]["path"])
    assert s0.read_text().strip().count("\n") == 4  # lines 0,2,4,6,8
    assert s1.read_text().strip().count("\n") == 4
    assert plan["shards"][0]["lines"] == 5
    assert plan["shards"][1]["lines"] == 5
    manifest = json.loads((tmp_path / "shards" / "shard_plan.json").read_text())
    assert manifest["dataset_sha256"]
    for shard in manifest["shards"]:
        content = Path(shard["path"]).read_text(encoding="utf-8")
        import hashlib

        assert hashlib.sha256(content.encode()).hexdigest() == shard["sha256"]


def test_plan_shards_missing_dataset(tmp_path):
    with pytest.raises(DistributedPlanError, match="dataset not found"):
        plan_shards(tmp_path / "nope.jsonl", ["a"], tmp_path)


def test_plan_shards_no_nodes(tmp_path):
    src = tmp_path / "d.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DistributedPlanError, match="no nodes"):
        plan_shards(src, [], tmp_path)


def test_aggregate_progress_all_completed():
    out = aggregate_progress(
        [
            {"node_id": "a", "status": "completed", "step": 100, "total_steps": 100},
            {"node_id": "b", "status": "completed", "step": 100, "total_steps": 100},
        ]
    )
    assert out["status"] == "completed"
    assert out["step"] == 200
    assert out["by_status"] == {"completed": 2}
    assert out["errors"] == []


def test_aggregate_progress_worst_status_wins():
    out = aggregate_progress(
        [
            {"node_id": "a", "status": "completed", "step": 50, "total_steps": 100},
            {"node_id": "b", "status": "failed", "step": 10, "total_steps": 100, "error": "OOM"},
        ]
    )
    assert out["status"] == "failed"
    assert out["errors"] == ["b: OOM"]
    assert out["step"] == 60


def test_aggregate_progress_empty():
    assert aggregate_progress([])["status"] == "unknown"
    assert aggregate_progress([])["nodes"] == 0
