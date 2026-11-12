"""Cowork flow-export ingestion + datasets info CLI tests (capability plan §3.1 / T4).

Fully offline: synthetic exports in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fusion_trainer.cli import app
from fusion_trainer.dataset.cowork_ingest import (
    CoworkIngestError,
    build_from_cowork_dataset,
    cowork_records_to_eval,
    find_cowork_manifest,
    load_cowork_export,
)


def _rec(flow="f1", node="n1", passed=True, prompt="q", messages=None, **extra):
    rec = {
        "schema_version": 1,
        "flow_id": flow,
        "node_id": node,
        "passed": passed,
        "prompt": prompt,
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"answer {node}"},
        ],
    }
    rec.update(extra)
    return rec


def _write(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_load_cowork_export_valid_and_bad_lines(tmp_path):
    p = _write(tmp_path / "cowork.jsonl", [_rec(), _rec(node="n2", passed=False)])
    records, digest = load_cowork_export(p)
    assert len(records) == 2
    assert len(digest) == 64


def test_load_cowork_export_missing_raises(tmp_path):
    with pytest.raises(CoworkIngestError, match="not found"):
        load_cowork_export(tmp_path / "nope.jsonl")


def test_load_cowork_export_empty_raises(tmp_path):
    p = _write(tmp_path / "cowork.jsonl", [])
    with pytest.raises(CoworkIngestError, match="no valid cowork records"):
        load_cowork_export(p)


def test_cowork_to_eval_shape_task_id_and_arm():
    recs = cowork_records_to_eval(
        [_rec(flow="fx", node="ny", task_type="codegen", arm="a2")]
    )
    assert recs[0]["task_id"] == "fx/ny"
    assert recs[0]["task_type"] == "codegen"
    assert recs[0]["arm"] == "a2"
    assert recs[0]["passed"] is True


def test_cowork_defaults_task_type_and_arm():
    recs = cowork_records_to_eval([_rec()])
    assert recs[0]["task_type"] == "cowork"
    assert recs[0]["arm"] == "default"


def test_build_sft_from_cowork_with_lineage_manifest(tmp_path):
    export = _write(
        tmp_path / "cowork.jsonl",
        [_rec(node="n1", passed=True), _rec(node="n2", passed=False)],
    )
    out = tmp_path / "sft.jsonl"
    manifest = build_from_cowork_dataset(export, out)
    assert manifest["source"] == "fusion-cowork"
    assert manifest["sft_samples"] == 1
    assert manifest["lineage"]["source"] == "fusion-cowork"
    assert "cowork_records_to_eval" in manifest["lineage"]["transforms"]
    assert len(manifest["cowork_export_sha256"]) == 64
    sidecar = tmp_path / "sft.jsonl.from_cowork.json"
    assert sidecar.is_file()
    # staged eval-shaped temp file must not survive
    assert not (tmp_path / "cowork.jsonl.evalshaped.jsonl").exists()


def test_build_dpo_from_cowork_siblings(tmp_path):
    # retry semantics: same flow/node run twice, one passed one failed
    export = _write(
        tmp_path / "cowork.jsonl",
        [_rec(node="n1", passed=True, prompt="same"),
         _rec(node="n1", passed=False, prompt="same",
              messages=[{"role": "user", "content": "same"},
                        {"role": "assistant", "content": "broken"}])],
    )
    out = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    manifest = build_from_cowork_dataset(export, out, dpo)
    assert manifest["dpo_pairs"] == 1
    assert dpo.is_file()


def test_find_cowork_manifest(tmp_path):
    export = _write(tmp_path / "cowork.jsonl", [_rec()])
    out = tmp_path / "sft.jsonl"
    build_from_cowork_dataset(export, out)
    found = find_cowork_manifest(out)
    assert found is not None
    assert found["source"] == "fusion-cowork"
    assert find_cowork_manifest(tmp_path / "missing.jsonl") is None


def test_cli_from_cowork_happy_path(tmp_path):
    export = _write(tmp_path / "cowork.jsonl", [_rec()])
    out = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(
        app,
        ["datasets", "from-cowork", "--export", str(export), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "SFT dataset: 1 samples" in result.output
    assert out.is_file()


def test_cli_from_cowork_error_exits_1(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "from-cowork",
            "--export",
            str(tmp_path / "nope.jsonl"),
            "--output",
            str(tmp_path / "sft.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_cli_datasets_info_with_lineage(tmp_path):
    export = _write(tmp_path / "cowork.jsonl", [_rec()])
    out = tmp_path / "sft.jsonl"
    build_from_cowork_dataset(export, out)
    result = CliRunner().invoke(app, ["datasets", "info", str(out)])
    assert result.exit_code == 0, result.output
    assert "Lineage[cowork]:" in result.output
    assert "cowork_records_to_eval" in result.output
    assert "Schema:" in result.output
