"""Wilson CI promotion gate + claims registry rows (issue #134).

Fully offline: synthetic counts in tmp_path, no fusion-mlx / fusion-eval.
"""

from __future__ import annotations

import json

import pytest

from fusion_trainer.flywheel import load_registry, record_round
from fusion_trainer.promotion import (
    append_claims_row,
    ci_non_overlap,
    gate_promotion,
    load_floor_band,
    next_claims_id,
    wilson_ci,
)


def test_wilson_known_values():
    # 25/30 = 83.3%, Wilson 95% [66.4, 92.7] — matches CL-006 (E-2 floor doc).
    lo, hi = wilson_ci(25, 30)
    assert lo == pytest.approx(0.664, abs=0.005)
    assert hi == pytest.approx(0.927, abs=0.005)
    # 0/0 never fabricates a tight band.
    assert wilson_ci(0, 0) == (0.0, 1.0)
    # Perfect score keeps a non-degenerate lower bound (Wilson property):
    # 30/30 -> [0.886, 1.0]; only larger n tightens it.
    lo, hi = wilson_ci(30, 30)
    assert hi == pytest.approx(1.0, abs=1e-6)
    assert lo == pytest.approx(0.886, abs=0.005)


def test_ci_non_overlap():
    assert ci_non_overlap((0.8, 0.95), (0.5, 0.7))
    assert not ci_non_overlap((0.6, 0.8), (0.55, 0.75))
    assert not ci_non_overlap((0.5, 0.7), (0.8, 0.95))


def test_gate_first_round_promotes():
    d = gate_promotion({"passed": 20, "total": 30}, None)
    assert d["promoted"] is True
    assert d["reason"]


def test_gate_non_overlap_promotes():
    # 27/30 vs 15/30: CIs clearly separated.
    d = gate_promotion({"passed": 27, "total": 30}, {"passed": 15, "total": 30})
    assert d["promoted"] is True
    assert d["candidate_ci"][0] > d["incumbent_ci"][1]


def test_gate_overlap_discards():
    # 16/30 vs 15/30: one-task swing, CIs overlap — statistical discard.
    d = gate_promotion({"passed": 16, "total": 30}, {"passed": 15, "total": 30})
    assert d["promoted"] is False
    assert "overlaps" in d["reason"]


def test_gate_floor_band_discards():
    # Candidate beats the incumbent but sits inside the E-2 floor band.
    floor = {"passed": 25, "total": 30}
    d = gate_promotion(
        {"passed": 24, "total": 30},
        {"passed": 15, "total": 30},
        floor=floor,
    )
    assert d["promoted"] is False
    assert "floor" in d["reason"]


def test_gate_floor_band_promotes_above_floor():
    # n=300 keeps the Wilson bands tight enough to separate cleanly.
    d = gate_promotion(
        {"passed": 280, "total": 300},
        {"passed": 150, "total": 300},
        floor={"passed": 250, "total": 300},
    )
    assert d["promoted"] is True
    assert d["candidate_ci"][0] > d["floor_ci"][1]


def test_gate_missing_counts_degrades_loudly():
    d = gate_promotion(
        {"passed": 0, "total": 0, "pass_rate": 0.7},
        {"passed": 0, "total": 0, "pass_rate": 0.5},
    )
    assert d["gate_mode"] == "regression-fallback"
    assert d["promoted"] is True
    assert "counts unavailable" in d["reason"]


def test_next_claims_id_sequential(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(
        "| ID | Date | Claim | Evidence | Status | Review |\n"
        "|----|------|-------|----------|--------|--------|\n"
        "| CL-003 | 2026-10-31 | old | run | refuted | 2026-11-01 |\n",
        encoding="utf-8",
    )
    assert next_claims_id(p) == "CL-004"
    assert next_claims_id(tmp_path / "absent.md") == "CL-001"


def test_append_claims_row_preserves_history(tmp_path):
    p = tmp_path / "claims.md"
    p.write_text(
        "| ID | Date | Claim | Evidence | Status | Review |\n"
        "|----|------|-------|----------|--------|--------|\n"
        "| CL-001 | 2026-11-01 | seed | run-1 | verified | 2026-11-01 |\n",
        encoding="utf-8",
    )
    row = append_claims_row(
        p,
        claim="flywheel round round-1 gate promoted: pass_rate 0.9000 (27/30)",
        evidence="eval_run=e1 trainer_run=t1 checkpoint=adp gate_mode=wilson",
        status="verified",
    )
    assert row["id"] == "CL-002"
    text = p.read_text(encoding="utf-8")
    assert "CL-001" in text and "CL-002" in text
    # Rejected decisions register too (kept forever, never deleted).
    row2 = append_claims_row(
        p, claim="round-2 rejected", evidence="eval_run=e2", status="rejected"
    )
    assert row2["id"] == "CL-003"
    assert "rejected" in p.read_text(encoding="utf-8")


def test_append_claims_row_missing_registry_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        append_claims_row(tmp_path / "absent.md", claim="x", evidence="y")


def test_load_floor_band(tmp_path):
    p = tmp_path / "floor.json"
    p.write_text(json.dumps({"passed": 25, "total": 30}), encoding="utf-8")
    assert load_floor_band(p) == {"passed": 25, "total": 30}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"p50": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_floor_band(bad)


def _summary(tmp_path, run_id="r1", passed=27, total=30, arm="fc-local", model="m1"):
    data = {
        "run_id": run_id,
        "models": {arm: model},
        "arms": {
            arm: {
                "total": total,
                "passed": passed,
                "pass_rate": passed / total,
                "by_type": {"bugfix": {"total": 3, "passed": 1, "pass_rate": 0.3333}},
            }
        },
    }
    p = tmp_path / f"{run_id}-summary.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_record_round_wilson_gate_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| ID | Date | Claim | Evidence | Status | Review |\n"
        "|----|------|-------|----------|--------|--------|\n",
        encoding="utf-8",
    )
    # Incumbent: weak baseline round.
    record_round(
        str(cache),
        label="baseline",
        checkpoint="base",
        summary_path=_summary(tmp_path, run_id="r0", passed=15, total=30),
        baseline=True,
    )
    # Candidate: overlapping CI — statistically indistinguishable, discarded.
    e1 = record_round(
        str(cache),
        label="round-1",
        checkpoint="adp1",
        summary_path=_summary(tmp_path, run_id="r1", passed=16, total=30),
        run_id="t1",
        gate=True,
        gate_mode="wilson",
        claims_registry=claims,
    )
    assert e1["status"] == "discarded"
    assert e1["gate_mode"] == "wilson"
    assert e1["claims_row"] == "CL-001"
    # Candidate: clearly separated CI — promoted, next CL row.
    e2 = record_round(
        str(cache),
        label="round-2",
        checkpoint="adp2",
        summary_path=_summary(tmp_path, run_id="r2", passed=27, total=30),
        run_id="t2",
        gate=True,
        gate_mode="wilson",
        claims_registry=claims,
    )
    assert e2["status"] == "promoted"
    assert e2["claims_row"] == "CL-002"
    text = claims.read_text(encoding="utf-8")
    assert "rejected" in text and "verified" in text
    reg = load_registry(str(cache))
    statuses = [r.get("status", "promoted") for r in reg["rounds"]]
    assert statuses == ["promoted", "discarded", "promoted"]


def test_record_round_wilson_floor_band(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    record_round(
        str(cache),
        label="baseline",
        checkpoint="base",
        summary_path=_summary(tmp_path, run_id="r0", passed=15, total=30),
        baseline=True,
    )
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps({"passed": 25, "total": 30}), encoding="utf-8")
    # Beats incumbent, inside the floor band — discarded.
    e = record_round(
        str(cache),
        label="round-1",
        checkpoint="adp1",
        summary_path=_summary(tmp_path, run_id="r1", passed=24, total=30),
        run_id="t1",
        gate=True,
        gate_mode="wilson",
        floor_band={"passed": 25, "total": 30},
    )
    assert e["status"] == "discarded"
    assert e["floor_ci"] is not None
    assert "floor" in e["gate_reason"]


def test_record_round_regression_mode_unchanged(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    record_round(
        str(cache),
        label="baseline",
        checkpoint="base",
        summary_path=_summary(tmp_path, run_id="r0", passed=15, total=30),
        baseline=True,
    )
    # Default gate_mode stays the #82 point regression check.
    e = record_round(
        str(cache),
        label="round-1",
        checkpoint="adp1",
        summary_path=_summary(tmp_path, run_id="r1", passed=16, total=30),
        run_id="t1",
        gate=True,
    )
    # Point rate 16/30 >= 15/30 — promoted under regression semantics.
    assert e["status"] == "promoted"
    assert "gate_mode" not in e
