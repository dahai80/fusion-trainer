"""Tool-calling trajectory contract tests (issue #88).

Golden path: trajectory JSONL with tool_calls -> DatasetLoader ->
serialized training text; ingest modules preserve tool fields verbatim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fusion_trainer.dataset.formats import (
    copy_message,
    serialize_chatml,
    validate_chat_messages,
)
from fusion_trainer.dataset.loader import DatasetLoader

_TRAJECTORY = {
    "messages": [
        {"role": "system", "content": "You are an office agent."},
        {"role": "user", "content": "Make the sheet bold"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "sheet_bold",
                        "arguments": {"range": "A1:B2"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "Done, A1:B2 is bold."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "sheet_bold",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
}


def _write_jsonl(path: Path, samples: list[dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# formats helpers
# ---------------------------------------------------------------------------


def test_validate_accepts_tool_trajectory():
    ok, reason = validate_chat_messages(_TRAJECTORY["messages"])
    assert ok, reason


def test_validate_rejects_tool_role_without_call_id():
    ok, reason = validate_chat_messages(
        [{"role": "tool", "content": "result"}]
    )
    assert not ok
    assert "tool_call_id" in reason


def test_validate_rejects_tool_call_missing_function():
    ok, reason = validate_chat_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        ]
    )
    assert not ok
    assert "function.name+arguments" in reason


def test_validate_rejects_tool_calls_on_user_role():
    ok, reason = validate_chat_messages(
        [
            {"role": "user", "content": "q", "tool_calls": [
                {"id": "c", "function": {"name": "f", "arguments": {}}}
            ]},
        ]
    )
    assert not ok
    assert "only valid on assistant" in reason


def test_validate_accepts_typed_content_parts():
    ok, reason = validate_chat_messages(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        ]
    )
    assert ok, reason


def test_copy_message_preserves_tool_fields():
    m = _TRAJECTORY["messages"][2]
    copied = copy_message(m)
    assert copied["tool_calls"] == m["tool_calls"]
    assert "tool_call_id" not in copied  # assistant turn had none
    tool_msg = copy_message(_TRAJECTORY["messages"][3])
    assert tool_msg["tool_call_id"] == "call_1"


def test_serialize_chatml_deterministic_and_complete():
    text1 = serialize_chatml(_TRAJECTORY["messages"])
    text2 = serialize_chatml(json.loads(json.dumps(_TRAJECTORY["messages"])))
    assert text1 == text2  # byte-stable for equal input
    assert "TOOL_CALL " in text1
    assert '"name": "sheet_bold"' in text1
    assert '"arguments": {"range": "A1:B2"}' in text1
    assert "TOOL_RESULT[call_1] ok" in text1
    assert "Done, A1:B2 is bold." in text1


# ---------------------------------------------------------------------------
# loader round-trip
# ---------------------------------------------------------------------------


def test_loader_preserves_tool_calls_and_tools(tmp_path):
    src = _write_jsonl(tmp_path / "traj.jsonl", [_TRAJECTORY])
    train, val = DatasetLoader().load_messages(src, max_seq_length=4096)
    assert len(train) == 1 and not val
    sample = train[0]
    msgs = sample["messages"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "sheet_bold"
    assert msgs[3]["tool_call_id"] == "call_1"
    # sample-level tools schema passes through for apply_chat_template
    assert sample["tools"] == _TRAJECTORY["tools"]


def test_loader_trim_path_preserves_tool_fields(tmp_path):
    tiny = {
        "messages": [
            {"role": "user", "content": "x" * 40},
            {
                "role": "assistant",
                "content": "y" * 40,
                "tool_calls": [
                    {"id": "c", "function": {"name": "f", "arguments": {}}}
                ],
            },
        ]
    }
    src = _write_jsonl(tmp_path / "tiny.jsonl", [tiny])
    train, _ = DatasetLoader().load_messages(src, max_seq_length=8)
    kept = train[0]["messages"]
    assert kept[-1]["tool_calls"][0]["function"]["name"] == "f"


def test_loader_drops_invalid_tool_structure_loudly(tmp_path, caplog):
    bad = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "orphan result"},
        ]
    }
    src = _write_jsonl(tmp_path / "bad.jsonl", [bad])
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.dataset.loader"):
        train, _ = DatasetLoader().load_messages(src, max_seq_length=4096)
    assert train == []
    assert any("invalid tool-call structure" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ingest entry points
# ---------------------------------------------------------------------------


def _rollout_record(messages, tools=None):
    return {
        "schema_version": 1,
        "rollout_id": "ro-1",
        "model": "test-model",
        "task_id": "t-1",
        "created_at": "2026-11-25T00:00:00Z",
        "messages": messages,
        "outcome": {"signal": "task_complete", "score": 1.0},
        **({"tools": tools} if tools else {}),
    }


def test_rollout_ingest_preserves_tool_calls(tmp_path):
    from fusion_trainer.dataset.rollout_ingest import load_rollout_sink, rollouts_to_sft

    rec = _rollout_record(_TRAJECTORY["messages"], tools=_TRAJECTORY["tools"])
    sink = tmp_path / "sink.jsonl"
    sink.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records, _ = load_rollout_sink(sink)
    samples = rollouts_to_sft(records)
    assert len(samples) == 1
    assert samples[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "sheet_bold"
    assert samples[0]["tools"] == _TRAJECTORY["tools"]


def test_rollout_ingest_accepts_empty_content_with_tool_calls():
    from fusion_trainer.dataset.rollout_ingest import _valid_messages

    assert _valid_messages(_TRAJECTORY["messages"]) is True


def test_rollout_ingest_rejects_orphan_tool_turn():
    from fusion_trainer.dataset.rollout_ingest import _valid_messages

    assert (
        _valid_messages(
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "r"},
            ]
        )
        is False
    )


def test_eval_ingest_preserves_tool_calls():
    from fusion_trainer.dataset.eval_ingest import eval_records_to_sft

    rec = {
        "task_id": "t-1",
        "passed": True,
        "messages": _TRAJECTORY["messages"],
        "tools": _TRAJECTORY["tools"],
    }
    samples = eval_records_to_sft([rec])
    assert len(samples) == 1
    assert samples[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "sheet_bold"
    assert samples[0]["tools"] == _TRAJECTORY["tools"]


def test_eval_dpo_mining_renders_tool_calls():
    from fusion_trainer.dataset.eval_ingest import mine_preference_pairs

    passed = {
        "task_id": "t-1",
        "passed": True,
        "prompt": "do it",
        "messages": _TRAJECTORY["messages"],
    }
    failed = {
        "task_id": "t-1",
        "passed": False,
        "prompt": "do it",
        "messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "I did nothing."},
        ],
    }
    pairs = mine_preference_pairs([passed, failed])
    assert len(pairs) == 1
    assert "TOOL_CALL " in pairs[0]["chosen"]
    assert "sheet_bold" in pairs[0]["chosen"]
    assert pairs[0]["rejected"] == "I did nothing."
