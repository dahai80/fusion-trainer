"""Skill mining from positive rollouts (issue #123, GUI-SkillEvo pattern).

Gradient-free path 2: mine successful procedural patterns (consecutive
assistant tool-call sequences) from positive rollouts into a skill
library the harness can retrieve — weights never change. Deterministic
and LLM-free (rule 5): a skill is identified by its normalized tool-call
chain (same normalization as the redundant-call detector in
rollout_ingest), deduplicated across rollouts, and only admitted when it
succeeded at least `min_successes` times (default 2 — a single success
is anecdote, not a skill).

Library file: append-safe JSONL at the CLI-provided path, one skill per
line, sorted by success_count descending. Skills reference their
example rollout by id — the trajectory itself stays in the tap, the
library stays small and auditable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset.rollout_ingest import _tool_call_key, reward_score

logger = logging.getLogger(__name__)

DEFAULT_MIN_SKILL_SUCCESSES = 2
SKILL_SCHEMA_VERSION = 2


class SkillMiningError(Exception):
    pass


def _call_chain(messages: list[Any]) -> list[str]:
    """Normalized keys of all tool calls in one assistant segment.

    A user turn ends the segment (a skill cannot span user turns).
    """
    keys: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            break
        for call in m.get("tool_calls") or []:
            key = _tool_call_key(call)
            if key is not None:
                keys.append(key)
    return keys


def mine_skills(
    records: list[dict[str, Any]], min_successes: int = DEFAULT_MIN_SKILL_SUCCESSES
) -> list[dict[str, Any]]:
    """Mine procedural skills from positive rollouts (#123 path 2).

    For every positive rollout (reward >= 1.0) each assistant tool-call
    segment (consecutive calls, ended by a user turn) becomes a skill
    candidate keyed by its normalized call chain. Candidates seen in
    >= min_successes distinct positive rollouts are admitted with:
    success_count, tool sequence, contributing task_ids (when outcomes
    carry task_id), the example rollout_id, and a stable skill_id
    (sha256 of the chain, 12 hex chars).

    Trust requirements (frontier watch #16, Zep Agent Skills pattern):
    every admitted skill carries field-level evidence citations — the
    supporting rollout events (rollout_id, task_id, signal, score, and
    the office validator name@version when the outcome was
    validator-verified, sidecar schema v2) — plus a versioned admission
    block recording the gate that let it in. Verified-attempts-only is
    structural: only outcomes with reward >= 1.0 are ever candidates.
    """
    if min_successes < 1:
        raise SkillMiningError("min_successes must be >= 1")
    seen: dict[str, dict[str, Any]] = {}
    positives = 0
    for rec in records:
        score = reward_score(rec.get("outcome"))
        if score is None or score < 1.0:
            continue
        positives += 1
        messages = rec.get("messages")
        if not isinstance(messages, list):
            continue
        rid = str(rec.get("rollout_id", ""))
        outcome = rec.get("outcome") if isinstance(rec.get("outcome"), dict) else {}
        task_id = str(outcome.get("task_id") or "")
        validator = outcome.get("validator")
        verifier = (
            f"{validator.get('name')}@v{validator.get('version')}"
            if isinstance(validator, dict) and validator.get("name")
            else "outcome_signal"
        )
        evidence = {
            "rollout_id": rid,
            "task_id": task_id,
            "signal": str(outcome.get("signal", "")),
            "score": score,
            "verifier": verifier,
        }
        # Split the message list into assistant segments at user turns.
        segment: list[Any] = []
        segments: list[list[Any]] = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append(m)
        if segment:
            segments.append(segment)
        for seg in segments:
            chain = _call_chain(seg)
            if not chain:
                continue
            key = hashlib.sha256("\n".join(chain).encode()).hexdigest()[:12]
            entry = seen.setdefault(
                key,
                {
                    "skill_id": key,
                    "chain": chain,
                    "rollout_ids": set(),
                    "task_ids": set(),
                    "evidence": [],
                    "example_rollout_id": rid,
                },
            )
            entry["rollout_ids"].add(rid)
            entry["evidence"].append(evidence)
            if task_id:
                entry["task_ids"].add(task_id)
    skills: list[dict[str, Any]] = []
    for entry in seen.values():
        if len(entry["rollout_ids"]) < min_successes:
            continue
        skills.append(
            {
                "schema_version": SKILL_SCHEMA_VERSION,
                "skill_id": entry["skill_id"],
                "tools": [k.split("|", 1)[0] for k in entry["chain"]],
                "chain": entry["chain"],
                "success_count": len(entry["rollout_ids"]),
                "task_ids": sorted(entry["task_ids"]),
                "evidence": sorted(entry["evidence"], key=lambda e: e["rollout_id"]),
                "example_rollout_id": entry["example_rollout_id"],
                "admission": {
                    "gate": f"reward>=1.0 in >={min_successes} distinct rollouts",
                    "min_successes": min_successes,
                    "mined_at": datetime.now(UTC).isoformat(),
                },
            }
        )
    skills.sort(key=lambda s: (-s["success_count"], s["skill_id"]))
    logger.info(
        "skill mining: %d skill(s) admitted from %d positive rollout(s) "
        "(%d candidates, min_successes=%d)",
        len(skills),
        positives,
        len(seen),
        min_successes,
    )
    return skills


def write_skill_library(skills: list[dict[str, Any]], output_path: Path) -> Path:
    """Write the skill library JSONL (one skill per line, atomic)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for skill in skills:
            fh.write(json.dumps(skill, ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    logger.info("wrote %d skill(s) to %s", len(skills), output_path)
    return output_path
