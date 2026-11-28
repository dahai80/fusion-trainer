"""Steps-shape cowork run ingestion (#136, handoff doc §1).

Fully offline: synthetic run exports in tmp_path, no fusion-cowork import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_trainer.dataset.cowork_ingest import (
    CoworkIngestError,
    build_from_cowork_steps_dataset,
    cowork_runs_to_sft,
    detect_cowork_export_shape,
    load_cowork_runs,
)


def _run(
    run_id="r1", success=True, classification="internal", steps=None, task="do work"
):
    return {
        "run_id": run_id,
        "task": task,
        "success": success,
        "classification": classification,
        "steps": steps
        if steps is not None
        else [
            {
                "node_id": "n1",
                "node_name": "fetch",
                "output_data": {"ok": True},
                "error": "",
                "status": "success",
                "upstream": [],
            },
            {
                "node_id": "n2",
                "node_name": "render",
                "output_data": {"path": "out.md"},
                "error": "",
                "status": "success",
                "upstream": ["n1"],
            },
        ],
    }


def test_detect_shape_steps_and_node_runs(tmp_path):
    p = tmp_path / "export.jsonl"
    p.write_text(json.dumps(_run()) + "\n", encoding="utf-8")
    assert detect_cowork_export_shape(p) == "steps"
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "flow_id": "f",
                "node_id": "n",
                "passed": True,
                "messages": [{"role": "user", "content": "x"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert detect_cowork_export_shape(legacy) == "node-runs"
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps({"foo": 1}) + "\n", encoding="utf-8")
    with pytest.raises(CoworkIngestError):
        detect_cowork_export_shape(unknown)


def test_load_cowork_runs_validation(tmp_path):
    p = tmp_path / "runs.jsonl"
    lines = [
        json.dumps(_run(run_id="ok1")),
        json.dumps({"run_id": "bad"}),
        "not json",
        json.dumps(_run(run_id="ok2", steps=[{"node_id": "a"}])),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs, digest = load_cowork_runs(p)
    assert [r["run_id"] for r in runs] == ["ok1", "ok2"]
    assert len(digest) == 64
    with pytest.raises(CoworkIngestError):
        load_cowork_runs(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("not json\n", encoding="utf-8")
    with pytest.raises(CoworkIngestError):
        load_cowork_runs(empty)


def test_convert_one_sample_per_successful_run():
    samples, stats = cowork_runs_to_sft([_run(run_id="r1"), _run(run_id="r2")])
    assert len(samples) == 2
    assert stats["runs_kept"] == 2
    msgs = samples[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "do work"}
    assert msgs[1]["role"] == "assistant"
    call = msgs[1]["tool_calls"][0]["function"]
    assert call == {"name": "fetch", "arguments": {}}
    tool = msgs[2]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "n1"
    assert json.loads(tool["content"]) == {"ok": True}


def test_convert_respects_topological_order():
    steps = [
        {
            "node_id": "n2",
            "node_name": "render",
            "output_data": {},
            "status": "success",
            "upstream": ["n1"],
        },
        {
            "node_id": "n1",
            "node_name": "fetch",
            "output_data": {},
            "status": "success",
            "upstream": [],
        },
    ]
    samples, _ = cowork_runs_to_sft([_run(steps=steps)])
    names = [
        m["tool_calls"][0]["function"]["name"]
        for m in samples[0]["messages"]
        if m["role"] == "assistant"
    ]
    assert names == ["fetch", "render"]


def test_filters_loud_and_counted():
    runs = [
        _run(run_id="fail", success=False),
        _run(run_id="conf", classification="confidential"),
        _run(run_id="unclass", classification="topsecret"),
        _run(run_id="noclass", classification=""),
        _run(
            run_id="inconsistent",
            steps=[
                {
                    "node_id": "n1",
                    "node_name": "a",
                    "output_data": {},
                    "status": "success",
                    "upstream": [],
                },
                {
                    "node_id": "n2",
                    "node_name": "b",
                    "output_data": {},
                    "error": "boom",
                    "status": "failed",
                    "upstream": [],
                },
            ],
        ),
        _run(
            run_id="cycle",
            steps=[
                {
                    "node_id": "a",
                    "node_name": "x",
                    "output_data": {},
                    "status": "success",
                    "upstream": ["b"],
                },
                {
                    "node_id": "b",
                    "node_name": "y",
                    "output_data": {},
                    "status": "success",
                    "upstream": ["a"],
                },
            ],
        ),
        _run(
            run_id="skips",
            steps=[
                {
                    "node_id": "n1",
                    "node_name": "fetch",
                    "output_data": {},
                    "status": "success",
                    "upstream": [],
                },
                {
                    "node_id": "n2",
                    "node_name": "gate",
                    "output_data": None,
                    "status": "skipped",
                    "upstream": ["n1"],
                },
                {
                    "node_id": "n3",
                    "node_name": "render",
                    "output_data": {},
                    "status": "success",
                    "upstream": ["n1", "n2"],
                },
            ],
        ),
    ]
    samples, stats = cowork_runs_to_sft(runs)
    kept = [s["run_id"] for s in samples]
    assert kept == ["skips"]
    assert stats["dropped_unsuccessful"] == 1
    assert stats["dropped_classification"] == 3
    assert stats["dropped_inconsistent"] == 1
    assert stats["dropped_cycle"] == 1
    assert stats["steps_skipped"] == 1
    names = [
        m["tool_calls"][0]["function"]["name"]
        for m in samples[0]["messages"]
        if m["role"] == "assistant"
    ]
    assert names == ["fetch", "render"]


def test_build_dataset_manifest_and_sidecar(tmp_path):
    export = tmp_path / "export.jsonl"
    export.write_text(
        "\n".join(json.dumps(_run(run_id=f"r{i}")) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "ds" / "sft.jsonl"
    manifest = build_from_cowork_steps_dataset(export, out)
    assert manifest["shape"] == "steps"
    assert manifest["sft_samples"] == 3
    assert manifest["cowork_export_sha256"]
    assert manifest["stats"]["runs_kept"] == 3
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        msgs = json.loads(line)["messages"]
        assert msgs[0]["role"] == "user"
    sidecar = Path(str(out) + ".from_cowork.json")
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["lineage"]["transforms"] == ["cowork_runs_to_sft"]


def test_build_dataset_all_filtered_raises(tmp_path):
    export = tmp_path / "export.jsonl"
    export.write_text(
        json.dumps(_run(run_id="r1", success=False)) + "\n", encoding="utf-8"
    )
    with pytest.raises(CoworkIngestError, match="no successful cowork runs"):
        build_from_cowork_steps_dataset(export, tmp_path / "sft.jsonl")
    assert not (tmp_path / "sft.jsonl").exists()
