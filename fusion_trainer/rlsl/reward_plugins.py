"""Pluggable deterministic reward functions (issue #90).

Beyond the built-in math/code sandboxes, GRPO reward dispatch supports
sample-level plugin specs:

    "reward": {"plugin": "office_xlsx", "args": {...}}

Built-ins:
    office_xlsx / office_docx / office_pptx — structural validators for
        task result files produced by office agents. Pure stdlib
        (zipfile + ElementTree over OOXML), deterministic: same input
        file + args -> same reward.
    rubric_judge — LLM-judge over a deterministic rubric prompt; score
        normalized to [0, 1]. NOT deterministic (model call); intended
        for open-ended rubrics the file validators cannot express.

Custom plugins register via register_reward_fn(name, fn); fn signature:
    fn(sample: dict, completion: str, args: dict) -> float   # 0..1
"""

from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fusion_trainer.mlx_client import MLXFineTuneClient

logger = logging.getLogger(__name__)

RewardFn = Callable[[dict[str, Any], str, dict[str, Any]], float]

_REGISTRY: dict[str, RewardFn] = {}

# OOXML namespaces (spreadsheetML / wordprocessingML / presentationML)
_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_WML = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
_NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def register_reward_fn(name: str, fn: RewardFn) -> None:
    """Register a reward plugin under a dotted-lowercase name."""
    key = str(name).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
        raise ValueError(f"invalid reward plugin name: {name!r}")
    if not callable(fn):
        raise TypeError("reward plugin must be callable")
    _REGISTRY[key] = fn
    logger.info("registered reward plugin: %s", key)


def get_reward_fn(name: str) -> RewardFn | None:
    return _REGISTRY.get(str(name).strip().lower())


def list_reward_fns() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

# A completion references its result file either explicitly (last path-like
# token) or via a plugin-supplied regex with one capture group.
_PATH_TOKEN_RE = re.compile(r"[/~][\w./~-]+\.[A-Za-z0-9]{2,8}")


def extract_result_path(completion: str, args: dict[str, Any]) -> str:
    """Deterministically pull the result-file path out of a completion.

    args["path_regex"] (one capture group) wins; otherwise the LAST
    path-like token in the text. Raises ValueError when nothing matches —
    fail-visible, never a guessed default.
    """
    pattern = args.get("path_regex")
    if pattern:
        m = re.search(str(pattern), completion)
        if not m:
            raise ValueError(f"path_regex {pattern!r} matched nothing in completion")
        return str(m.group(1) if m.groups() else m.group(0))
    matches = _PATH_TOKEN_RE.findall(completion)
    if not matches:
        raise ValueError("no path-like token found in completion")
    return matches[-1]


def _require_file(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"result file not found: {path}")
    return path


def _normalize_score(parts: list[tuple[str, bool, str]]) -> float:
    """All-or-nothing structural score with per-check detail logging."""
    failed = [name for name, ok, _ in parts if not ok]
    for name, ok, detail in parts:
        if not ok:
            logger.info("office validator check %s FAILED: %s", name, detail)
    return 1.0 if not failed else 0.0


# ---------------------------------------------------------------------------
# office validators (deterministic, stdlib OOXML)
# ---------------------------------------------------------------------------


def reward_office_xlsx(sample: dict, completion: str, args: dict) -> float:
    """Structural checks on an .xlsx result: openable zip, required sheets
    present, formula-vs-hardcode policy (args.formula_policy: "require"
    demands at least one formula cell in the first required sheet; "strict"
    also fails on hardcoded numeric constants)."""
    path = _require_file(extract_result_path(completion, args))
    required = [str(s) for s in (args.get("required_sheets") or [])]
    checks: list[tuple[str, bool, str]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            ns = {"m": _NS_MAIN, "r": _NS_REL_DOC}
            sheets: dict[str, str] = {}
            for sh in wb.findall("m:sheets/m:sheet", ns):
                sheets[sh.get("name", "")] = sh.get(f"{{{_NS_REL_DOC}}}id", "")
            for name in required:
                checks.append(
                    (f"sheet:{name}", name in sheets, f"sheets={sorted(sheets)}")
                )
            policy = str(args.get("formula_policy", "") or "")
            if policy and required:
                target = _sheet_target(zf, required[0], sheets)
                if target is None:
                    checks.append(("formulas", False, "sheet part unresolved"))
                else:
                    formulas, values = _sheet_formula_counts(zf, target)
                    ok = formulas > 0 and (policy != "strict" or values == 0)
                    checks.append(
                        (
                            "formulas",
                            ok,
                            f"policy={policy} formula_cells={formulas} "
                            f"constant_numeric={values}",
                        )
                    )
    except (zipfile.BadZipFile, KeyError) as exc:
        checks.append(("open", False, f"not a readable xlsx: {exc}"))
    score = _normalize_score(checks)
    logger.info("office_xlsx reward=%.1f file=%s", score, path)
    return score


def _sheet_target(
    zf: zipfile.ZipFile, sheet_name: str, sheets: dict[str, str]
) -> str | None:
    rid = sheets.get(sheet_name, "")
    for rel in ET.fromstring(zf.read("xl/_rels/workbook.xml.rels")).iter(
        f"{{{_NS_PKG_REL}}}Relationship"
    ):
        if rel.get("Id") == rid:
            target = rel.get("Target", "worksheets/sheet1.xml")
            return target if target.startswith("xl/") else f"xl/{target}"
    return None


def _sheet_formula_counts(zf: zipfile.ZipFile, target: str) -> tuple[int, int]:
    root = ET.fromstring(zf.read(target))
    ns = {"m": _NS_MAIN}
    formulas = len(root.findall(".//m:f", ns))
    values = 0
    for c in root.findall(".//m:c", ns):
        if c.get("t") == "str" or c.find("m:f", ns) is not None:
            continue
        v = c.find("m:v", ns)
        if v is not None and v.text and re.fullmatch(r"-?\d+(\.\d+)?", v.text.strip()):
            values += 1
    return formulas, values


def reward_office_docx(sample: dict, completion: str, args: dict) -> float:
    """Structural checks on a .docx result: openable zip, required section
    headings present (args.required_sections, case-insensitive substring
    over paragraph text)."""
    path = _require_file(extract_result_path(completion, args))
    required = [str(s) for s in (args.get("required_sections") or [])]
    checks: list[tuple[str, bool, str]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            doc = ET.fromstring(zf.read("word/document.xml"))
            text = "\n".join(
                "".join(t.text or "" for t in p.iter(f"{{{_NS_WML}}}t"))
                for p in doc.iter(f"{{{_NS_WML}}}p")
            )
            for sec in required:
                checks.append(
                    (
                        f"section:{sec}",
                        sec.lower() in text.lower(),
                        f"{sec!r} not found in document text",
                    )
                )
    except (zipfile.BadZipFile, KeyError) as exc:
        checks.append(("open", False, f"not a readable docx: {exc}"))
    score = _normalize_score(checks)
    logger.info("office_docx reward=%.1f file=%s", score, path)
    return score


def reward_office_pptx(sample: dict, completion: str, args: dict) -> float:
    """Structural checks on a .pptx result: openable zip, minimum slide
    count (args.min_slides, default 1), optional required slide titles
    (args.required_titles, substring match)."""
    path = _require_file(extract_result_path(completion, args))
    min_slides = int(args.get("min_slides", 1))
    required_titles = [str(t) for t in (args.get("required_titles") or [])]
    checks: list[tuple[str, bool, str]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            slides = sorted(
                n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)
            )
            checks.append(
                ("slides", len(slides) >= min_slides, f"{len(slides)} < {min_slides}")
            )
            titles = []
            for name in slides:
                root = ET.fromstring(zf.read(name))
                # slide text lives in both drawingml (`a:t`) and
                # presentationml (`p:t`) parts — match any namespaced `t`
                titles.append(
                    " ".join(
                        el.text or ""
                        for el in root.iter()
                        if isinstance(el.tag, str) and el.tag.endswith("}t")
                    )
                )
            joined = "\n".join(titles).lower()
            for title in required_titles:
                checks.append(
                    (
                        f"title:{title}",
                        title.lower() in joined,
                        f"{title!r} not found on any slide",
                    )
                )
    except (zipfile.BadZipFile, KeyError) as exc:
        checks.append(("open", False, f"not a readable pptx: {exc}"))
    score = _normalize_score(checks)
    logger.info("office_pptx reward=%.1f file=%s", score, path)
    return score


# ---------------------------------------------------------------------------
# rubric LLM judge (deterministic prompt template, normalized score)
# ---------------------------------------------------------------------------

_RUBRIC_PROMPT_TEMPLATE = (
    "You are grading one agent completion against a rubric.\n"
    "TASK:\n{task}\n\nCOMPLETION:\n{completion}\n\nRUBRIC (weighted criteria):\n"
    "{rubric}\n\n"
    "Reply with ONLY a JSON object: "
    '{{"scores": {{"<criterion>": <0..1>}}, "total": <0..1>}}'
)


def reward_rubric_judge(sample: dict, completion: str, args: dict) -> float:
    """LLM-judge over args["rubric"] = [{"criterion": str, "weight": float}].

    Deterministic prompt template (Rule 5: the TEMPLATE is code); the score
    comes from the configured model over MLXClient HTTP. Score normalized:
    weighted mean of per-criterion scores, clamped to [0, 1]. Any failure
    (no model configured, endpoint down, unparseable reply) -> 0.0 with a
    loud warning — fail-visible, never a fabricated pass.
    """
    rubric = args.get("rubric") or sample.get("rubric") or []
    if not rubric:
        logger.warning("rubric_judge: empty rubric — reward 0.0")
        return 0.0
    rubric_text = "\n".join(
        f"- {item.get('criterion', '?')} (weight {float(item.get('weight', 1.0)):.2f})"
        for item in rubric
        if isinstance(item, dict)
    )
    prompt = _RUBRIC_PROMPT_TEMPLATE.format(
        task=str(args.get("task") or sample.get("prompt") or ""),
        completion=completion[:8000],
        rubric=rubric_text,
    )
    model = str(
        args.get("model") or os.environ.get("FUSION_RUBRIC_JUDGE_MODEL", "") or ""
    )
    if not model:
        logger.warning(
            "rubric_judge: no judge model configured (args.model or FUSION_RUBRIC_JUDGE_MODEL) — reward 0.0"
        )
        return 0.0
    try:
        client = MLXFineTuneClient()
        try:
            reply = client.chat_completions(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
        finally:
            client.close()
        content = reply[0]
        data = json.loads(content[content.index("{") : content.rindex("}") + 1])
        total = float(data.get("total", 0.0))
    except Exception as exc:
        logger.warning("rubric_judge failed (%s) — reward 0.0", exc)
        return 0.0
    score = max(0.0, min(1.0, total))
    logger.info("rubric_judge reward=%.2f model=%s", score, model)
    return score


def _extract_tool_calls(completion: str) -> tuple[int, int]:
    """Count tool calls + redundant-consecutive repeats in a completion.

    Two deterministic shapes (issue #117): the in-repo serialization
    (``TOOL_CALL {"name": ..., "arguments": ...}`` lines, formats.py
    serialize_chatml) and engine-native JSON tool calls (a single object
    with function/name, a list of them, or {"tool_calls": [...]}). The
    redundancy rule matches rollout_ingest: a call repeating the
    immediately preceding call (same name + normalized arguments) counts
    as redundant; a completion is one assistant turn, so no user-turn
    reset is needed. Unparsable text -> (0, 0) — no fabricated calls.
    """
    keys: list[str] = []

    def _key_from_call(call: Any) -> str | None:
        if not isinstance(call, dict):
            return None
        fn = call.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            fn = call  # engine-native flat shape: {"name": ..., "arguments": ...}
        if not fn.get("name"):
            return None
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return f"{fn['name']}|{args}"
        if isinstance(args, dict):
            args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return f"{fn['name']}|{args}"

    text = completion.strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if parsed is not None:
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            parsed = parsed["tool_calls"]
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            keys.extend(k for k in (_key_from_call(c) for c in parsed) if k)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("TOOL_CALL "):
                continue
            try:
                call = json.loads(line[len("TOOL_CALL ") :])
            except ValueError:
                continue
            key = _key_from_call(call)
            if key:
                keys.append(key)
    redundant = sum(1 for prev, cur in zip(keys, keys[1:], strict=False) if prev == cur)
    return len(keys), redundant


def reward_efficiency_budget(sample: dict, completion: str, args: dict) -> float:
    """Frozen-budget efficiency reward (issue #117, CARE-PO recipe).

    args: {"budgets": {"<task_id>": int, ...}, optional "_default": int}.
    The budgets MUST be the frozen p50 baselines from
    ``ft datasets from-rollouts --budgets-output`` — never policy-relative.
    Bonus-only: within budget + zero redundant consecutive calls earns the
    full bonus (normalized to 1.0; the dispatcher applies scale); a
    no-tool-call completion earns 0, never negative — the CARE-PO collapse
    (no-tool-call optimal policy) is structurally impossible. No budget for
    the sample's task -> 0.0 with a loud log (VerifierBench lesson: silent
    rejection masquerading as strictness is a poisoning mode).
    """
    from ..dataset.rollout_ingest import _BUDGET_PENALTY_BONUS, budget_penalty

    budgets = args.get("budgets")
    if not isinstance(budgets, dict) or not budgets:
        logger.warning("efficiency_budget: no budgets in args — reward 0.0")
        return 0.0
    task_id = str(sample.get("task_id") or "")
    budget = budgets.get(task_id, budgets.get("_default"))
    if budget is None:
        logger.warning(
            "efficiency_budget: no frozen budget for task %r (have %d task(s)) — reward 0.0",
            task_id or "<none>",
            len(budgets),
        )
        return 0.0
    calls, redundant = _extract_tool_calls(completion)
    bonus = budget_penalty(calls, int(budget), redundant)
    score = bonus / _BUDGET_PENALTY_BONUS if bonus else 0.0
    logger.info(
        "efficiency_budget task=%s calls=%d redundant=%d budget=%s -> %.2f",
        task_id or "<none>",
        calls,
        redundant,
        budget,
        score,
    )
    return score


register_reward_fn("office_xlsx", reward_office_xlsx)
register_reward_fn("office_docx", reward_office_docx)
register_reward_fn("office_pptx", reward_office_pptx)
register_reward_fn("rubric_judge", reward_rubric_judge)
register_reward_fn("efficiency_budget", reward_efficiency_budget)
