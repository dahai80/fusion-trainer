"""Office task registry: task definitions, validator audit gate, synthesis (#120).

Owns what cannot live in fusion-office (handed off from office#49/#50):

1. Task definitions + known-valid solution corpora. For every task with
   a generated validator, 2-3 human-verified correct solutions live
   beside the definition; the audit gate refuses reward signal until
   the validator accepts them all (VerifierBench: a verifier that
   accepts nothing silently poisons the batch while looking like
   "strong rejection").
2. Deterministic cross-app task synthesis on the office-api entity-tool
   graph (GET /v1/graph, office-api >= 0.1.27): stage-2 tasks on graph
   edges. Synthesis is template-based and LLM-free (Rule 5) — new tasks
   start with an untested validator and an empty corpus, so the gate
   blocks them until both are filled in.

Registry file: append-only JSONL at {cache}/tasks/registry.jsonl, latest
record per task_id wins (same contract as the outcome sidecar).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TASK_REGISTRY_FILENAME = "registry.jsonl"
VALID_CURRICULUM_STAGES = (1, 2, 3)
VALID_VALIDATOR_STATUSES = ("untested", "passing", "failing")


class TaskRegistryError(Exception):
    pass


def _required_task_errors(rec: dict[str, Any]) -> str | None:
    if not isinstance(rec.get("task_id"), str) or not rec["task_id"].strip():
        return "task_id must be a non-empty string"
    stage = rec.get("curriculum_stage")
    if stage not in VALID_CURRICULUM_STAGES:
        return f"curriculum_stage must be one of {list(VALID_CURRICULUM_STAGES)}"
    validator = rec.get("validator")
    if not isinstance(validator, dict):
        return "validator must be an object with name, version, status"
    for key in ("name", "version"):
        value = validator.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"validator.{key} must be a non-empty string"
    if validator.get("status") not in VALID_VALIDATOR_STATUSES:
        return f"validator.status must be one of {list(VALID_VALIDATOR_STATUSES)}"
    solutions = rec.get("known_valid_solutions")
    if not isinstance(solutions, list) or any(
        not isinstance(s, str) or not s.strip() for s in solutions
    ):
        return "known_valid_solutions must be a list of non-empty strings"
    return None


def load_task_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load the registry JSONL -> {task_id: task}, latest record wins.

    Malformed lines are skipped loudly; an unreadable/empty registry is
    an error (fail-visible, never silently empty).
    """
    if not path.is_file():
        raise TaskRegistryError(f"task registry not found: {path}")
    tasks: dict[str, dict[str, Any]] = {}
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                logger.warning("skip task registry line %d: not JSON", lineno)
                continue
            if not isinstance(rec, dict) or not rec.get("task_id"):
                bad += 1
                logger.warning("skip task registry line %d: no task_id", lineno)
                continue
            tasks[str(rec["task_id"])] = rec
    if not tasks:
        raise TaskRegistryError(f"no task records in {path} (skipped {bad} bad lines)")
    if bad:
        logger.warning("task registry: %d tasks loaded, %d bad lines", len(tasks), bad)
    logger.info("loaded %d tasks from registry %s", len(tasks), path)
    return tasks


def upsert_task(path: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Validate and append a task record (atomic append, latest-wins)."""
    if not isinstance(task, dict):
        raise TaskRegistryError("task must be an object")
    err = _required_task_errors(task)
    if err:
        raise TaskRegistryError(f"invalid task {task.get('task_id', '?')!r}: {err}")
    record = dict(task)
    record.setdefault("created_at", datetime.now(UTC).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("upserted task %s into %s", record["task_id"], path)
    return record


def validator_gate(task: dict[str, Any]) -> tuple[bool, str]:
    """Validator audit gate (VerifierBench, #120 item 1).

    A task may produce training signal only when:
    - its validator exists and is status=passing (self-tested against
      the known-valid corpus), and
    - the known-valid solution corpus is non-empty.

    Everything else must be masked from the positive set — an overly
    strict validator rejecting valid trajectories is the most common
    batch-poisoning failure and masquerades as "strong rejection".
    Returns (ok, reason).
    """
    validator = task.get("validator")
    if not isinstance(validator, dict):
        return False, "no validator record"
    status = validator.get("status")
    if status == "untested":
        return False, "validator untested (run the self-test gate first)"
    if status == "failing":
        return False, "validator failing (rejects its own known-valid corpus)"
    solutions = task.get("known_valid_solutions") or []
    if not solutions:
        return False, "empty known-valid solution corpus"
    return (
        True,
        f"passing validator v{validator.get('version')} + {len(solutions)} valid solutions",
    )


def gate_report(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Gate status over the whole registry: allowed vs masked, by stage."""
    allowed: list[str] = []
    masked: list[dict[str, str]] = []
    by_stage: dict[str, int] = {}
    for tid, task in sorted(tasks.items()):
        ok, reason = validator_gate(task)
        stage = str(task.get("curriculum_stage"))
        by_stage[stage] = by_stage.get(stage, 0) + 1
        if ok:
            allowed.append(tid)
        else:
            masked.append({"task_id": tid, "reason": reason, "stage": stage})
    return {
        "total": len(tasks),
        "allowed": allowed,
        "masked": masked,
        "by_stage": by_stage,
    }


def synthesize_stage2_tasks(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Template stage-2 tasks on cross-app graph edges (office#50 handoff).

    `graph` is the office-api GET /v1/graph response: nodes are endpoints
    with consumed/produced artifact types, edges join producers to
    consumers with stage >= 2 (cross-app). One template task per edge,
    deterministic, no LLM: each starts validator=untested + empty
    corpus, so the audit gate masks it until both are filled in.
    """
    edges = graph.get("edges") if isinstance(graph, dict) else None
    if not isinstance(edges, list):
        raise TaskRegistryError(
            "graph payload has no edges list — expected the office-api GET /v1/graph response"
        )
    tasks: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        stage = edge.get("stage")
        producer = edge.get("producer") or edge.get("from")
        consumer = edge.get("consumer") or edge.get("to")
        if stage is None or not producer or not consumer:
            continue
        artifact = edge.get("artifact") or edge.get("artifact_type") or "artifact"
        tid = f"edge-{producer}-{consumer}-{artifact}".replace(" ", "-").lower()
        tasks.append(
            {
                "task_id": tid,
                "description": (
                    f"Cross-app flow: produce {artifact} via {producer}, "
                    f"consume it via {consumer} (synthesized from the "
                    "office-api entity-tool graph)"
                ),
                "curriculum_stage": 2,
                "edge": {
                    "producer": producer,
                    "consumer": consumer,
                    "artifact": artifact,
                    "stage": stage,
                },
                "validator": {"name": tid, "version": "1", "status": "untested"},
                "known_valid_solutions": [],
                "source_artifacts": edge.get("source_artifacts", []),
            }
        )
    logger.info(
        "synthesized %d stage-2 tasks from %d graph edges", len(tasks), len(edges)
    )
    return tasks
