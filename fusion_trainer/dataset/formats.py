"""Dataset format detection and loading."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# S3 (round-3 audit): depth-of-defense caps against attacker-controlled
# dataset files. A single multi-GB line / object could OOM the loader even
# with max_samples set (the bad line is read fully before the count check).
# These are generous for real training data (1MB/line is a huge document;
# 2GB file is a large dataset) but stop a malicious file from exhausting
# unified memory. Override via env for legitimate oversized inputs.
MAX_LINE_BYTES = int(os.environ.get("FUSION_MAX_LINE_BYTES", str(2 * 1024 * 1024)))
MAX_JSON_FILE_BYTES = int(
    os.environ.get("FUSION_MAX_JSON_FILE_BYTES", str(2 * 1024 * 1024 * 1024))
)

# D1 cleaned-dataset schemas (architecture/future-mlx-landing-plan.md A2):
#   SFT  : messages-JSONL  -> {"messages": [{"role","content"}, ...]}
#          ShareGPT        -> {"conversations": [{"from","value"}, ...]}
#   DPO  : DPO-pair        -> {"prompt","chosen","rejected"}
#   GRPO : GRPO-prompt     -> {"prompt", ...} (optional "answer"/"solution"/"tests")
SCHEMA_MESSAGES = "messages"
SCHEMA_SHAREGPT = "sharegpt"
SCHEMA_DPO_PAIR = "dpo_pair"
SCHEMA_GRPO_PROMPT = "grpo_prompt"
SCHEMA_INPUT_OUTPUT = "input_output"

VALID_SCHEMAS = (
    SCHEMA_MESSAGES,
    SCHEMA_SHAREGPT,
    SCHEMA_DPO_PAIR,
    SCHEMA_GRPO_PROMPT,
    SCHEMA_INPUT_OUTPUT,
)

# ShareGPT "from" -> standard role mapping
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}

# Issue #88: OpenAI-compatible tool-calling fields preserved on chat
# messages. mlx_lm.tuner ChatDataset passes the whole message dict to
# tokenizer.apply_chat_template, so these keys are visible to templates
# that render tool calls (Llama-3.x, Qwen2.5+). Content may also be a
# list of typed parts, passed through verbatim.
TOOL_MESSAGE_FIELDS = ("tool_calls", "tool_call_id", "name")
CHAT_ROLES = ("system", "user", "assistant", "tool")


def copy_message(m: dict[str, Any]) -> dict[str, Any]:
    """Copy one chat message preserving tool-calling fields verbatim.

    The old round-trips rebuilt {"role","content"} and silently amputated
    tool_calls / tool_call_id / name (issue #88).
    """
    out = {"role": m["role"], "content": m.get("content", "")}
    for key in TOOL_MESSAGE_FIELDS:
        if key in m:
            out[key] = m[key]
    return out


def validate_chat_messages(messages: Any) -> tuple[bool, str]:
    """Validate a chat conversation including tool-calling turns (issue #88).

    Accepts: text-only messages; assistant turns with
    tool_calls=[{id, type, function:{name, arguments}}]; tool-result turns
    (role "tool") with tool_call_id. Content may be a string or a list of
    typed parts. Returns (ok, reason).
    """
    if not isinstance(messages, list) or not messages:
        return False, "messages must be a non-empty list"
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False, f"message #{i} is not an object"
        if m.get("role") not in CHAT_ROLES:
            return False, f"message #{i} has invalid role {m.get('role')!r}"
        content = m.get("content", "")
        if isinstance(content, str) or isinstance(content, list) and all(
            isinstance(p, dict) and "type" in p for p in content
        ):
            pass
        else:
            return False, f"message #{i} content must be string or typed parts"
        calls = m.get("tool_calls")
        if calls is not None:
            if m.get("role") != "assistant":
                return False, f"message #{i}: tool_calls only valid on assistant"
            if not isinstance(calls, list) or not calls:
                return False, f"message #{i}: tool_calls must be a non-empty list"
            for j, tc in enumerate(calls):
                fn = (tc or {}).get("function") if isinstance(tc, dict) else None
                if (
                    not isinstance(fn, dict)
                    or not str(fn.get("name", "")).strip()
                    or "arguments" not in fn
                ):
                    return False, (
                        f"message #{i} tool_call #{j}: needs function.name+arguments"
                    )
        if m.get("role") == "tool" and not str(m.get("tool_call_id", "")).strip():
            return False, f"message #{i}: role 'tool' requires tool_call_id"
    return True, ""


def serialize_chatml(messages: list[dict[str, Any]]) -> str:
    """Deterministically render messages (incl. tool calls) to training text.

    Fallback for model templates that cannot render tool_calls themselves:
    each tool call becomes a "TOOL_CALL " line with sorted-keys JSON
    (Rule 5: no model routing); tool results become "TOOL_RESULT[id] "
    lines. Byte-stable for equal input.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        calls = m.get("tool_calls") if role == "assistant" else None
        if calls:
            rendered = "\n".join(
                "TOOL_CALL "
                + json.dumps(
                    {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for tc in calls
            )
            text = rendered if not content else f"{content}\n{rendered}"
            parts.append(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{text}<|eot_id|>"
            )
        elif role == "tool":
            parts.append(
                "<|start_header_id|>tool<|end_header_id|>\n\n"
                f"TOOL_RESULT[{m.get('tool_call_id', '')}] {content}<|eot_id|>"
            )
        else:
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
    return "<|begin_of_text|>" + "".join(parts)


def detect_format(path: Path) -> str:
    """Detect the format of a dataset file by its extension."""
    suffix = path.suffix.lower()
    mapping = {
        ".jsonl": "jsonl",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
        ".tsv": "tsv",
        ".arrow": "arrow",
    }
    return mapping.get(suffix, "")


def detect_dataset_schema(item: dict[str, Any]) -> str:
    """Detect the D1 cleaned-dataset schema of a single sample.

    Returns one of VALID_SCHEMAS, or "" if the schema is unrecognized.
    Inspection order matters: messages/sharegpt are chat formats, dpo is a
    preference triple, grpo is a single prompt, input_output is the legacy
    flat-text shape.
    """
    if not isinstance(item, dict):
        return ""
    if "messages" in item and isinstance(item["messages"], list):
        return SCHEMA_MESSAGES
    if "conversations" in item and isinstance(item["conversations"], list):
        return SCHEMA_SHAREGPT
    if "chosen" in item and "rejected" in item and "prompt" in item:
        return SCHEMA_DPO_PAIR
    if "prompt" in item and "chosen" not in item and "rejected" not in item:
        # GRPO-prompt: a prompt with optional reference answer / tests, but
        # no chosen/rejected (that would be DPO).
        return SCHEMA_GRPO_PROMPT
    if "input" in item or "output" in item or "text" in item:
        return SCHEMA_INPUT_OUTPUT
    return ""


def validate_format(data: list[dict[str, Any]], schema: str = "") -> tuple[bool, str]:
    """Validate that all samples conform to the given (or detected) schema.

    Returns (ok, message). When schema is empty, the first sample's detected
    schema is used and every sample must match it. Empty data is invalid.
    """
    if not data:
        return False, "dataset is empty"
    detected = schema
    if not detected:
        detected = detect_dataset_schema(data[0])
        if not detected:
            return (
                False,
                f"could not detect schema of first sample: {list(data[0].keys())}",
            )
    if detected not in VALID_SCHEMAS:
        return False, f"unknown schema: {detected}"
    bad = 0
    for i, item in enumerate(data):
        if detect_dataset_schema(item) != detected:
            bad += 1
            if bad <= 3:
                logger.warning(
                    "sample #%d does not match schema %s: keys=%s",
                    i,
                    detected,
                    list(item.keys()),
                )
    if bad > 0:
        return False, f"{bad}/{len(data)} samples do not match schema '{detected}'"
    logger.info("validated %d samples against schema '%s'", len(data), detected)
    return True, detected


def sharegpt_to_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one ShareGPT sample {"conversations":[{"from","value"}]} to
    standard messages [{"role","content"}]. Unknown 'from' values map to
    'user' as a safe default."""
    convs = item.get("conversations", [])
    messages = []
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = _SHAREGPT_ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        content = turn.get("value", turn.get("content", ""))
        if content:
            messages.append({"role": role, "content": str(content)})
    return messages


def load_dataset_file(
    path: Path, format: str = "", max_samples: int = 0
) -> list[dict[str, Any]]:
    """Load a dataset file into a list of dicts.

    R4: max_samples > 0 streams + stops early instead of materializing the
    whole file then slicing. A 50GB JSONL / 5GB parquet with max_samples=1000
    now reads only ~1000 rows, not the whole file (was an OOM risk).
    """
    if not format:
        format = detect_format(path)
    if not format:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if format in ("jsonl",):
        return _load_jsonl(path, max_samples)
    if format in ("csv", "tsv"):
        return _load_csv(path, max_samples)
    if format in ("parquet", "arrow"):
        return _load_parquet(path, max_samples)
    # json (single object) can't be streamed row-by-row; load whole then slice.
    if format == "json":
        data = _load_json(path)
        if max_samples > 0 and len(data) > max_samples:
            data = data[:max_samples]
        return data
    raise ValueError(f"Unsupported format: {format}")


def _load_jsonl(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    # R4: stream line-by-line; break as soon as max_samples is reached so a
    # multi-GB JSONL with max_samples set doesn't OOM.
    # R19: a single corrupt line no longer fails the whole file (was: 10M-row
    # dataset unusable because of 1 bad line). Bad lines are skipped with a
    # logged line number + a running count, so the operator sees how much was
    # dropped rather than getting a raw JSONDecodeError.
    data = []
    bad = 0
    oversize = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # S3 (round-3 audit): per-line byte cap. A single huge line is read
            # fully before max_samples can stop the loop, so cap the line itself.
            # Skipping is fail-visible (logged), not a silent truncation.
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                oversize += 1
                if oversize <= 5:
                    logger.warning(
                        "skip oversize JSONL line %d in %s: %d bytes > cap %d",
                        lineno,
                        path,
                        len(line.encode("utf-8")),
                        MAX_LINE_BYTES,
                    )
                continue
            try:
                item = json.loads(line)
            except (ValueError, json.JSONDecodeError) as exc:
                bad += 1
                if bad <= 5:
                    logger.warning(
                        "skip corrupt JSONL line %d in %s: %s", lineno, path, exc
                    )
                continue
            if not isinstance(item, dict):
                bad += 1
                if bad <= 5:
                    logger.warning("skip non-object JSONL line %d in %s", lineno, path)
                continue
            data.append(item)
            if max_samples > 0 and len(data) >= max_samples:
                break
    if bad or oversize:
        logger.warning(
            "loaded %d samples from %s, skipped %d corrupt + %d oversize lines",
            len(data),
            path,
            bad,
            oversize,
        )
    return data


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file into a list of dicts.

    Note: Only flattens ONE level of nesting. If the JSON has a deeply
    nested structure like {"data": {"data": [...]}}, only the top-level
    "data" key is extracted — the inner list is NOT walked recursively.
    """
    # S3 (round-3 audit): whole-file json.load materializes the entire object
    # in memory with no cap — guard the file size before reading.
    size = path.stat().st_size
    if size > MAX_JSON_FILE_BYTES:
        raise ValueError(
            f"JSON file {path} is {size} bytes, exceeds cap "
            f"{MAX_JSON_FILE_BYTES} (set FUSION_MAX_JSON_FILE_BYTES to raise)"
        )
    with open(path, encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        # Try common keys for list-wrapped datasets
        for key in ("data", "entries", "samples", "records", "train"):
            if key in content and isinstance(content[key], list):
                return content[key]
        return [content]
    raise ValueError(f"Unexpected JSON structure in {path}")


def _load_csv(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
            if max_samples > 0 and len(data) >= max_samples:
                break
    return data


def _load_parquet(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required to load parquet files")

    # R4: stream via iter_batches instead of read_table().to_pylist(), which
    # materialized every row (10M-row parquet → 10M dicts → OOM) before the
    # caller's max_samples slice. With max_samples set we read only enough
    # batches. Without it we still read the whole table, but batch-iteration
    # is far more memory-friendly than one giant to_pylist().
    data: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        rows = batch.to_pylist()
        if max_samples > 0 and len(data) + len(rows) > max_samples:
            data.extend(rows[: max_samples - len(data)])
            break
        data.extend(rows)
        if max_samples > 0 and len(data) >= max_samples:
            break
    return data


def save_dataset_file(
    data: list[dict[str, Any]],
    path: Path,
    format: str = "jsonl",
) -> None:
    """Save a list of dicts to a file."""
    writers = {
        "jsonl": _save_jsonl,
        "json": _save_json,
    }
    writer = writers.get(format)
    if writer is None:
        raise ValueError(f"Unsupported output format: {format}")
    writer(data, path)


def _save_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(_json.dumps(item, ensure_ascii=False) + "\n")


def _save_json(data: list[dict[str, Any]], path: Path) -> None:
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
