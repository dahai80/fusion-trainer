"""Distributed orchestration surface (capability plan 2.4, T5).

Orchestration ONLY — trainer-side node manifest / shard plan / aggregated
progress. The training itself stays delegated to fusion-mlx (its mx.distributed
job API is the tracked upstream dependency U4); this module never trains and
never imports mlx. All computations are deterministic file/JSON work (Rule 5).

Consumption shape stays one of the three frozen forms (CLI flag / YAML /
RunManager call) — no new RPC surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NODE_MANIFEST_VERSION = 1


class DistributedPlanError(ValueError):
    """Raised when a node manifest or shard plan is invalid (fail-visible)."""


def load_nodes(source: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate + normalize a node manifest (JSON file or ready list).

    Schema: [{"node_id": str, "host": str, "port": int, "max_model_gb"?: float}]
    node_id must be unique; host non-empty; port 1-65535.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DistributedPlanError(f"node manifest not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise DistributedPlanError(f"node manifest not JSON: {exc}") from exc
    else:
        data = source
    if not isinstance(data, list) or not data:
        raise DistributedPlanError("node manifest must be a non-empty list")
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise DistributedPlanError(f"node #{i} is not an object")
        node_id = str(raw.get("node_id", "")).strip()
        host = str(raw.get("host", "")).strip()
        port = raw.get("port")
        if not node_id:
            raise DistributedPlanError(f"node #{i} missing node_id")
        if node_id in seen:
            raise DistributedPlanError(f"duplicate node_id: {node_id}")
        seen.add(node_id)
        if not host:
            raise DistributedPlanError(f"node {node_id} missing host")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise DistributedPlanError(f"node {node_id} port invalid: {port!r}")
        node = {"node_id": node_id, "host": host, "port": port}
        if raw.get("max_model_gb") is not None:
            node["max_model_gb"] = float(raw["max_model_gb"])
        nodes.append(node)
    logger.info("distributed: %d nodes validated", len(nodes))
    return nodes


def plan_shards(
    dataset_path: str | Path,
    node_ids: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Split a JSONL dataset into one contiguous shard per node.

    Deterministic: line i goes to shard i % len(nodes) (round-robin keeps
    shards balanced for sorted or clustered data). Each shard file is
    {output_dir}/shard_{i}_of_{n}.jsonl; the plan manifest carries per-shard
    line counts + sha256 so any Mac can verify its slice before training.
    """
    if not node_ids:
        raise DistributedPlanError("no nodes to shard across")
    src = Path(dataset_path)
    if not src.is_file():
        raise DistributedPlanError(f"dataset not found: {src}")
    lines = [
        ln
        for ln in src.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(node_ids)
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, ln in enumerate(lines):
        buckets[i % n].append(ln)
    shards = []
    for i, bucket in enumerate(buckets):
        shard_path = out_dir / f"shard_{i}_of_{n}.jsonl"
        content = "\n".join(bucket) + ("\n" if bucket else "")
        shard_path.write_text(content, encoding="utf-8")
        shards.append(
            {
                "shard": i,
                "node_id": node_ids[i],
                "path": str(shard_path),
                "lines": len(bucket),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    plan = {
        "schema_version": NODE_MANIFEST_VERSION,
        "dataset": str(src),
        "dataset_sha256": hashlib.sha256(
            src.read_bytes()
        ).hexdigest(),
        "total_lines": len(lines),
        "num_nodes": n,
        "shards": shards,
    }
    plan_path = out_dir / "shard_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "distributed: plan written %s (%d lines across %d nodes)",
        plan_path,
        len(lines),
        n,
    )
    return plan


def aggregate_progress(node_statuses: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-node run statuses to one aggregated view (deterministic).

    Each entry: {"node_id", "status", "step", "total_steps", "error"?}.
    Aggregation: counts by status, summed steps, worst-status summary —
    any failed/unknown node fails the aggregate visibly (never fakes green).
    """
    if not node_statuses:
        return {
            "nodes": 0,
            "status": "unknown",
            "by_status": {},
            "step": 0,
            "total_steps": 0,
            "errors": [],
        }
    by_status: dict[str, int] = {}
    step = 0
    total = 0
    errors: list[str] = []
    worst = "completed"
    rank = {"failed": 0, "cancelled": 0, "interrupted": 1, "unknown": 2,
            "running": 3, "queued": 4, "completed": 5}
    for ns in node_statuses:
        status = str(ns.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        step += int(ns.get("step", 0) or 0)
        total += int(ns.get("total_steps", 0) or 0)
        if ns.get("error"):
            errors.append(f"{ns.get('node_id', '?')}: {ns['error']}")
        if rank.get(status, 2) < rank.get(worst, 5):
            worst = status
    return {
        "nodes": len(node_statuses),
        "status": worst,
        "by_status": by_status,
        "step": step,
        "total_steps": total,
        "errors": errors,
    }
