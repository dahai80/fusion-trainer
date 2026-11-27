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
    records: list[dict[str, Any]],
    min_successes: int = DEFAULT_MIN_SKILL_SUCCESSES,
    polarity: str = "success",
) -> list[dict[str, Any]]:
    """Mine procedural skill patterns from scored rollouts (#123 path 2).

    Polarity pools (DualPM pattern, frontier watch #16): polarity="success"
    (default) mines chains from positive rollouts (reward >= 1.0) —
    patterns to follow; polarity="failure" mines chains from scored
    negative rollouts (reward < 1.0, unobserved excluded) — patterns to
    avoid / recovery-strategy seeds (WASP-Mem). Both pools share the same
    candidate/admission/evidence machinery.

    For every candidate rollout each assistant tool-call segment
    (consecutive calls, ended by a user turn) becomes a pattern keyed by
    its normalized call chain. Candidates seen in >= min_successes
    (success) / min_failures (failure) DISTINCT rollouts are admitted
    with: success_count, tool sequence, contributing task_ids (when
    outcomes carry task_id), the example rollout_id, and a stable
    skill_id (sha256 of the chain, 12 hex chars).

    Trust requirements (frontier watch #16, Zep Agent Skills pattern):
    every admitted pattern carries field-level evidence citations — the
    supporting rollout events (rollout_id, task_id, signal, score, and
    the office validator name@version when the outcome was
    validator-verified, sidecar schema v2) — plus a versioned admission
    block recording the gate that let it in. Scored-attempts-only is
    structural: unobserved outcomes are never candidates.
    """
    if min_successes < 1:
        raise SkillMiningError("min_successes must be >= 1")
    if polarity not in ("success", "failure"):
        raise SkillMiningError("polarity must be 'success' or 'failure'")
    seen: dict[str, dict[str, Any]] = {}
    positives = 0
    for rec in records:
        score = reward_score(rec.get("outcome"))
        if score is None:
            continue
        if polarity == "success" and score < 1.0:
            continue
        if polarity == "failure" and score >= 1.0:
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
                "polarity": polarity,
                "admission": {
                    "gate": (
                        f"reward>=1.0 in >={min_successes} distinct rollouts"
                        if polarity == "success"
                        else f"reward<1.0 (scored) in >={min_successes} distinct rollouts"
                    ),
                    "min_successes": min_successes,
                    "polarity": polarity,
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


def load_skill_library(path: Path) -> list[dict[str, Any]]:
    """Load a skill-library JSONL into validated skill dicts (#123).

    Fail-visible: a missing library raises; malformed lines are skipped
    loudly; a schema_version we do not understand is skipped loudly too
    (forward compatibility — older readers never misread newer fields).
    """
    if not path.is_file():
        raise SkillMiningError(f"skill library not found: {path}")
    skills: list[dict[str, Any]] = []
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
                logger.warning("skip skill library line %d: not JSON", lineno)
                continue
            if (
                not isinstance(rec, dict)
                or not rec.get("skill_id")
                or not isinstance(rec.get("chain"), list)
                or not rec["chain"]
            ):
                bad += 1
                logger.warning(
                    "skip skill library line %d: missing skill_id/chain", lineno
                )
                continue
            version = rec.get("schema_version")
            if version is not None and version > SKILL_SCHEMA_VERSION:
                bad += 1
                logger.warning(
                    "skip skill library line %d: schema_version %s newer than reader (%s)",
                    lineno,
                    version,
                    SKILL_SCHEMA_VERSION,
                )
                continue
            skills.append(rec)
    if bad:
        logger.warning(
            "skill library: %d skill(s) loaded, %d bad line(s)", len(skills), bad
        )
    logger.info("loaded %d skill(s) from %s", len(skills), path)
    return skills


def render_skill_block(
    skills: list[dict[str, Any]], max_skills: int = 5, max_chars: int = 2000
) -> str:
    """Render the deterministic prompt-injection block for a skill set (#123).

    The harness contract: retrieved skills become a compact instruction
    block the agent sees before acting. Deterministic, LLM-free; the
    block cites each skill's success_count and verifier so the model can
    weigh provenance. Truncated at max_skills / max_chars — the budget
    is explicit, never silently exceeded.
    """
    if not skills:
        return ""
    lines = [
        "Proven procedural skills (successfully used before — follow when applicable):"
    ]
    rendered = 0
    for rendered, s in enumerate(skills):
        if rendered >= max_skills:
            lines.append(f"(+{len(skills) - max_skills} more skills omitted)")
            break
        verifier = "outcome_signal"
        for ev in s.get("evidence") or []:
            if ev.get("verifier") and ev["verifier"] != "outcome_signal":
                verifier = str(ev["verifier"])
                break
        tools = " -> ".join(str(t) for t in s.get("tools") or [])
        line = (
            f"- {tools} (used {s.get('success_count', '?')}x, verified by {verifier})"
        )
        if len("\n".join(lines)) + len(line) > max_chars:
            lines.append("(further skills omitted: character budget reached)")
            break
        lines.append(line)
        rendered += 1
    return "\n".join(lines)
