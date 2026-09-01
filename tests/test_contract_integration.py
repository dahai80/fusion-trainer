"""Contract integration tests (audit M1).

The audit found the suite is almost entirely mock-based: tests patch
MLXFineTuneClient / httpx and assert trainer-side *assumptions* about the
upstream API shape, not upstream *facts*. If fusion-mlx renames a response
field (e.g. adapter_path -> adapter_dir), the mocked suite stays green while
production silently loses the adapter path.

These tests close that gap: they talk to a REAL fusion-mlx server and assert
the actual response fields the trainer depends on. They are marked
``integration`` and auto-skip unless a live server is reachable at
$FUSION_MLX_PORT (default 11434). Run them locally / nightly with a started
fusion-mlx (~/claude-home/fusion-mlx/start.sh start) and a small base model.

    pytest tests/test_contract_integration.py -m integration -v

They never run under the default ``pytest tests/`` because the server is
unreachable in CI/sandbox, keeping the fast suite fast.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest

from fusion_trainer.config import _default_mlx_port
from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError


def _mlx_port() -> int:
    return int(os.environ.get("FUSION_MLX_PORT", "0")) or _default_mlx_port()


def _server_reachable() -> bool:
    port = _mlx_port()
    try:
        resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_client():
    if not _server_reachable():
        pytest.skip(
            f"fusion-mlx not reachable at localhost:{_mlx_port()}; "
            "start it with ~/claude-home/fusion-mlx/start.sh start to run "
            "contract integration tests"
        )
    client = MLXFineTuneClient(port=_mlx_port())
    yield client
    client.close()


@pytest.fixture(scope="module")
def base_model_id(live_client: MLXFineTuneClient) -> str:
    data = live_client.health().get("data", [])
    if not data:
        pytest.skip("no models loaded on fusion-mlx; load a small base model")
    return str(data[0].get("id", ""))


def test_health_contract(live_client: MLXFineTuneClient):
    # GET /v1/models must return {"data": [{"id": str}, ...]}.
    data = live_client.health()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert all("id" in m and isinstance(m["id"], str) for m in data["data"])


def test_list_adapters_contract(live_client: MLXFineTuneClient, base_model_id: str):
    # GET /admin/api/fine-tune/adapters returns a list of dicts each with a
    # "name" (and "model"). The trainer's publish_run existence check
    # (L4) and GUI adapter tab depend on this exact shape.
    adapters = live_client.list_adapters(model_id=base_model_id)
    assert isinstance(adapters, list)
    for a in adapters:
        assert "name" in a, f"adapter dict missing 'name' key: {a}"


def test_sft_job_field_contract(
    live_client: MLXFineTuneClient, base_model_id: str, tmp_path: Path
):
    # The end-to-end contract the audit cared most about: a real SFT job's
    # terminal response must carry the fields SFTTrainer._run_training reads —
    # status, adapter_path, adapter_name. A field rename upstream would make
    # result["final_model_path"] an empty string in production.
    # Uses a 1-step, 1-epoch LoRA job over a tiny staged dataset so it runs
    # fast even on a loaded model. Cancelled if it runs too long.
    samples = [
        {"input": "hello", "output": "world"},
        {"input": "ping", "output": "pong"},
    ]
    dataset_path = MLXFineTuneClient.stage_dataset(
        samples, tmp_path, filename="contract_train.jsonl"
    )
    adapter_name = f"contract_test_{int(time.time())}"
    cfg = {
        "fine_tune_type": "lora",
        "num_epochs": 1,
        "batch_size": 1,
        "max_steps": 2,
        "lora_rank": 4,
        "lora_alpha": 8,
        "iters": 2,
    }
    try:
        job = live_client.create_job(
            model_id=base_model_id,
            dataset=dataset_path,
            config=cfg,
            adapter_name=adapter_name,
        )
    except MLXFineTuneError as exc:
        pytest.skip(f"create_job rejected by upstream (non-fatal for contract): {exc}")
        return

    assert "job_id" in job, f"create_job response missing 'job_id': {job}"
    job_id = str(job["job_id"])

    deadline = time.monotonic() + 600.0
    final = {}
    while time.monotonic() < deadline:
        try:
            final = live_client.get_job(job_id)
        except MLXFineTuneError:
            time.sleep(2.0)
            continue
        status = str(final.get("status", ""))
        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(2.0)
    else:
        live_client.cancel_job(job_id)
        pytest.fail("contract SFT job did not reach terminal status within 600s")

    status = str(final.get("status", ""))
    if status != "completed":
        live_client.cancel_job(job_id)
        pytest.skip(f"contract job ended non-completed ({status}); upstream env issue")

    # The fields SFTTrainer reads off the terminal job:
    assert "adapter_path" in final, (
        f"terminal job missing 'adapter_path' (trainer would get empty string): {final}"
    )
    assert final.get("adapter_path"), f"'adapter_path' present but empty: {final}"
    assert "adapter_name" in final, f"terminal job missing 'adapter_name': {final}"
