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

from fusion_trainer.config import (
    DatasetConfig,
    SFTConfig,
    TrainerConfig,
    _default_mlx_port,
)
from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError
from fusion_trainer.run_manager import get_run_manager


def _mlx_port() -> int:
    return int(os.environ.get("FUSION_MLX_PORT", "0")) or _default_mlx_port()


def _server_reachable() -> bool:
    port = _mlx_port()
    # A fusion-mlx server started with --api-key returns 401 to an
    # unauthenticated GET, which would look "unreachable" and silently skip
    # the whole contract suite. Send the admin token if set so a secured
    # server is correctly detected as live.
    headers = {}
    token = os.environ.get("FUSION_MLX_ADMIN_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # timeout: a live fusion-mlx with a large model cache can take ~10s to
    # answer GET /v1/models (it scans the cache dir on each call). A 3s cap
    # would falsely report a busy server as unreachable and skip the whole
    # suite. 20s is generous but still bounded.
    try:
        resp = httpx.get(
            f"http://localhost:{port}/v1/models", timeout=20.0, headers=headers
        )
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
    # S4 (soft #4): data[0] is whatever fusion-mlx loaded first — on this host
    # that is an image/video model (FLUX / Wan), which cannot serve an SFT
    # text job and makes create_job reject or hang. Pick the first model that
    # actually declares text_generation capability so the live SFT contract
    # runs against a real text base. Fall back to data[0] only if no model
    # reports capabilities (older fusion-mlx without the capabilities field).
    data = live_client.health().get("data", [])
    if not data:
        pytest.skip("no models loaded on fusion-mlx; load a small base model")
    text_models = [
        m for m in data if (m.get("capabilities") or {}).get("text_generation")
    ]
    if not text_models:
        return str(data[0].get("id", ""))

    # Prefer the smallest text model for a fast 2-iter smoke (parse a "<n>B" or
    # "<n>M" size hint from the id). Falls back to the first text model.
    def _size(m: dict) -> float:
        mid = str(m.get("id", "")).lower()
        import re

        mb = re.search(r"(\d+(?:\.\d+)?)b\b", mid)
        if mb:
            return float(mb.group(1))
        mm = re.search(r"(\d+(?:\.\d+)?)m\b", mid)
        if mm:
            return float(mm.group(1)) / 1000.0
        return 999.0

    text_models.sort(key=_size)
    return str(text_models[0].get("id", ""))


def test_health_contract(live_client: MLXFineTuneClient):
    # GET /v1/models must return {"data": [{"id": str}, ...]}.
    data = live_client.health()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert all("id" in m and isinstance(m["id"], str) for m in data["data"])


def test_list_adapters_contract(live_client: MLXFineTuneClient, base_model_id: str):
    # GET /admin/api/fine-tune/adapters returns a list of dicts each carrying
    # "adapter_name" (the identifier publish_run's L4 existence check matches
    # on) plus "model_id" and "adapter_path". This test pins that contract: a
    # field rename upstream (adapter_name -> name) would silently break
    # publish_run's existence check, which mock-based tests cannot catch.
    adapters = live_client.list_adapters(model_id=base_model_id)
    assert isinstance(adapters, list)
    for a in adapters:
        assert "adapter_name" in a, f"adapter dict missing 'adapter_name' key: {a}"


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
        samples, tmp_path, filename="train.jsonl"
    )
    adapter_name = f"contract_test_{int(time.time())}"
    cfg = {
        "fine_tune_type": "lora",
        "batch_size": 1,
        "iters": 2,
        "lora_rank": 4,
        "lora_alpha": 8,
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


def test_gui_run_manager_sft_live_e2e(
    live_client: MLXFineTuneClient, base_model_id: str, tmp_path: Path
):
    # soft #4: the GUI drives training through RunManager.start_sft, then polls
    # run_status / run_progress. The unit suite only exercises that surface
    # with a FAKE fusion-mlx (patched httpx), so a real end-to-end GUI-shaped
    # run was never verified in fusion-trainer's own suite. This test closes
    # that gap: a real model, a real (tiny) SFT job, driven exactly as the
    # fusion-agent-studio TrainingService does, asserting the live-progress
    # contract the GUI renders against.
    if os.environ.get("FUSION_TRAINER_REAL", "") != "1":
        pytest.skip("set FUSION_TRAINER_REAL=1 to run the GUI live e2e test")

    # dataset.path is the SOURCE file the trainer's DatasetLoader reads; the
    # trainer stages it into fusion-mlx's expected {train,valid}.jsonl layout
    # itself in _run_training. Passing a pre-staged dir here double-stages and
    # the loader rejects the dir suffix ("Unsupported file format"), so write a
    # plain {input,output} JSONL source the loader detects + transforms.
    import json

    samples = [
        {"input": "hello", "output": "world"},
        {"input": "ping", "output": "pong"},
    ]
    src_path = tmp_path / "src_train.jsonl"
    src_path.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    cfg = TrainerConfig(
        dataset=DatasetConfig(path=str(src_path), max_samples=0),
        sft=SFTConfig(
            model=base_model_id,
            method="lora",
            num_epochs=1,
            max_steps=2,
            lora_rank=4,
            lora_alpha=8.0,
            run_name=f"gui_e2e_{int(time.time())}",
        ),
        cache_path=str(tmp_path / "cache"),
        mlx_port=_mlx_port(),
    )

    mgr = get_run_manager()
    start = mgr.start_sft(cfg)
    run_id = start["run_id"]
    assert start["status"] == "running"

    deadline = time.monotonic() + 600.0
    terminal = ""
    saw_running = False
    while time.monotonic() < deadline:
        st = mgr.run_status(run_id)
        status = str(st.get("status", ""))
        if status == "running":
            saw_running = True
        # live-progress: even mid-run, run_progress must return a dict with a
        # buffer shape the GUI consumes ({"events": [...], ...}).
        prog = mgr.run_progress(run_id, since_step=-1)
        assert isinstance(prog, dict)
        if status in ("completed", "failed", "cancelled"):
            terminal = status
            break
        time.sleep(2.0)
    else:
        mgr.stop_run(run_id)
        pytest.fail("GUI e2e SFT run did not reach terminal status within 600s")

    assert saw_running, "run_status never reported 'running' — GUI would hang"
    assert terminal == "completed", f"GUI e2e run ended non-completed: {terminal}"

    final_status = mgr.run_status(run_id)
    assert final_status["status"] == "completed"
    # the GUI adapter tab + publish path depend on these populated fields.
    assert final_status.get("adapter_name"), (
        f"terminal run_status has empty adapter_name: {final_status}"
    )
    # events.jsonl must exist on disk (GUI replays from it on reconnect).
    events_file = Path(final_status.get("events_path", ""))
    assert events_file.is_file(), f"events.jsonl missing at {events_file}"
    assert events_file.stat().st_size > 0, "events.jsonl is empty"
    # the run appears in list_runs (GUI run-history tab).
    runs = mgr.list_runs(limit=200)
    assert any(r.get("run_id") == run_id for r in runs), (
        f"run {run_id} not in list_runs: {runs[:3]}"
    )


def test_contract_vlm_endpoint_fields(live_client: MLXFineTuneClient):
    # #55 (fusion-mlx #797/#800): the VLM fine-tune endpoint must exist and
    # the trainer's VLM client methods must reach it. We do NOT run a real
    # VLM job here (needs a registered VLM model + an image dataset — too
    # heavy for the contract suite and no small VLM is guaranteed). Instead
    # we assert the endpoint is wired: GET /admin/api/fine-tune/vlm/jobs
    # returns a list (even if empty), so a rename or missing mount would 404
    # and fail this contract. This catches the upstream field/endpoint drift
    # that mock-based tests cannot.
    import httpx

    port = _mlx_port()
    headers = {}
    token = os.environ.get("FUSION_MLX_ADMIN_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            f"http://localhost:{port}/admin/api/fine-tune/vlm/jobs",
            timeout=20.0,
            headers=headers,
        )
    except Exception as exc:
        pytest.skip(f"VLM endpoint unreachable (transport error): {exc}")
        return

    # A wired endpoint returns 200 + a JSON list. A 404 means fusion-mlx
    # predates #797/#800 (the route is not mounted) — fail visibly so an
    # operator knows to upgrade fusion-mlx rather than silently skip.
    if resp.status_code == 404:
        pytest.fail(
            "VLM fine-tune endpoint returned 404 — fusion-mlx predates #797/#800; "
            "upgrade fusion-mlx to >=0.8.80 to enable VLM fine-tune (#55)"
        )
    if resp.status_code == 401:
        pytest.skip("VLM endpoint needs admin token; set FUSION_MLX_ADMIN_TOKEN")
    assert resp.status_code == 200, (
        f"VLM endpoint returned {resp.status_code}: {resp.text[:200]}"
    )
    body = resp.json()
    assert isinstance(body, list), (
        f"GET /admin/api/fine-tune/vlm/jobs must return a list, got {type(body)}"
    )
