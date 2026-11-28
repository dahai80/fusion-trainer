"""Office-pilot contract gate (issue #133). Fully offline."""

from __future__ import annotations

import json

import pytest

from fusion_trainer.contract_gate import (
    DEFAULT_MUST_NOT_FLIP,
    ContractGateError,
    gate_contract_probes,
    load_gate_summary,
)


def _summary(pass_rate, contracts, model="m"):
    return {
        "total": 30,
        "passed": round(pass_rate * 30),
        "pass_rate": pass_rate,
        "contracts": contracts,
    }


def _contracts(rate_by_cid):
    return {
        cid: {"passed": round(rate * 4), "total": 4, "rate": rate}
        for cid, rate in rate_by_cid.items()
    }


FULL = {cid: 1.0 for cid in DEFAULT_MUST_NOT_FLIP}


def test_load_gate_summary_extracts_arm(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(
        json.dumps({"arms": {"fc-local": _summary(0.8, {})}}), encoding="utf-8"
    )
    arm = load_gate_summary(p)
    assert arm["pass_rate"] == 0.8
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"arms": {}}), encoding="utf-8")
    with pytest.raises(ContractGateError):
        load_gate_summary(empty)
    multi = tmp_path / "multi.json"
    multi.write_text(
        json.dumps({"arms": {"a": _summary(0.5, {}), "b": _summary(0.6, {})}}),
        encoding="utf-8",
    )
    with pytest.raises(ContractGateError, match="2 arms"):
        load_gate_summary(multi)


def test_gate_all_pass():
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)),
        _summary(0.85, _contracts(FULL)),
        token_per_task_pre=1000.0,
        token_per_task_post=650.0,
        hacking_clean=True,
    )
    assert d["promoted"] is True
    assert d["unmeasured"] == []


def test_gate_overall_below_threshold_rejects():
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)),
        _summary(0.75, _contracts(FULL)),
        token_per_task_pre=1000.0,
        token_per_task_post=650.0,
        hacking_clean=True,
    )
    assert d["promoted"] is False
    assert "overall" in d["reason"]


def test_gate_contract_regression_rejects():
    contracts = dict(FULL)
    contracts["OB-NOOP-NO-MUTATION"] = 0.75
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)),
        _summary(0.95, _contracts(contracts)),
        token_per_task_pre=1000.0,
        token_per_task_post=650.0,
        hacking_clean=True,
    )
    assert d["promoted"] is False
    assert "must-not-flip" in d["reason"]
    assert "OB-NOOP-NO-MUTATION" in d["checks"][1]["detail"]


def test_gate_token_budget_rejects():
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)),
        _summary(0.85, _contracts(FULL)),
        token_per_task_pre=1000.0,
        token_per_task_post=750.0,
        hacking_clean=True,
    )
    assert d["promoted"] is False
    assert "token-cost" in d["reason"]


def test_gate_hacking_detection_rejects():
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)),
        _summary(0.85, _contracts(FULL)),
        token_per_task_pre=1000.0,
        token_per_task_post=650.0,
        hacking_clean=False,
    )
    assert d["promoted"] is False
    assert "hacking-monitor" in d["reason"]


def test_gate_unmeasured_fails_loud():
    d = gate_contract_probes(
        _summary(0.719, _contracts(FULL)), _summary(0.85, _contracts(FULL))
    )
    assert d["promoted"] is False
    assert sorted(d["unmeasured"]) == ["hacking-monitor", "token-per-task"]
    assert all(c["detail"] == "unmeasured" for c in d["checks"][2:])
