import json

from fusion_trainer.struggle import (
    judge_struggle_steps,
    render_expert_block,
    struggling_recently,
)


def _call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def _tool_result(content, call_id="c1"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _msgs(*items):
    return list(items)


def test_clean_steps_score_zero():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
        _tool_result("ok"),
        {"role": "assistant", "content": "", "tool_calls": [_call("save", {"p": 2})]},
        _tool_result("ok"),
    )
    steps = judge_struggle_steps(msgs)
    assert len(steps) == 2
    assert all(s["struggle_score"] == 0 for s in steps)


def test_repeated_prev_call_flagged():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
        _tool_result("ok"),
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
    )
    steps = judge_struggle_steps(msgs)
    assert steps[0]["struggle_score"] == 0
    assert steps[1]["repeated_prev"] == 1
    assert steps[1]["struggle_score"] == 1


def test_error_result_counts_double():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
        _tool_result("Error: file not found"),
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 2})]},
    )
    steps = judge_struggle_steps(msgs)
    # the error attributes to the ISSUING step (steps[0]) — the step that
    # must change course; the retry step itself is clean
    assert steps[0]["error_results"] == 1
    assert steps[0]["struggle_score"] == 2
    assert steps[1]["struggle_score"] == 0


def test_user_turn_resets_window():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
        _tool_result("Error: boom"),
        {"role": "user", "content": "new task"},
        {"role": "assistant", "content": "", "tool_calls": [_call("save", {})]},
    )
    steps = judge_struggle_steps(msgs)
    assert steps[1]["error_results"] == 0
    assert steps[1]["struggle_score"] == 0


def test_struggling_recently_window():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("a", {"p": 1})]},
        _tool_result("Error: x"),
        {"role": "assistant", "content": "", "tool_calls": [_call("b", {})]},
        _tool_result("ok"),
        {"role": "assistant", "content": "", "tool_calls": [_call("c", {})]},
    )
    # error attributes to assistant step 0; window counts assistant steps
    assert struggling_recently(msgs, window=1) == []  # only the clean last step
    recent = struggling_recently(msgs, window=3)
    assert len(recent) == 1 and recent[0]["step_index"] == 1


def test_render_block_deterministic_and_budgeted():
    msgs = _msgs(
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
        _tool_result("Error: no such file"),
        {"role": "assistant", "content": "", "tool_calls": [_call("open", {"p": 1})]},
    )
    steps = struggling_recently(msgs)
    block = render_expert_block(steps)
    assert block == render_expert_block(steps)
    assert "Expert guidance" in block
    assert "Step 1" in block and "error" in block.lower()
    assert "Step 3" in block and "repeated" in block


def test_render_block_empty_when_clean():
    assert render_expert_block([]) == ""
    assert (
        render_expert_block(
            [
                {
                    "step_index": 0,
                    "repeated_prev": 0,
                    "error_results": 0,
                    "struggle_score": 0,
                }
            ]
        )
        == ""
    )


def test_render_block_max_steps_omission():
    msgs = [{"role": "user", "content": "go"}]
    for i in range(5):
        msgs += [
            {"role": "assistant", "content": "", "tool_calls": [_call(f"t{i}", {})]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"Error {i}"},
        ]
    steps = struggling_recently(msgs, window=10)
    block = render_expert_block(steps)
    assert "+2 more struggling steps omitted" in block
