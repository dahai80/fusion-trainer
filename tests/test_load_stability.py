"""Load / stability verification (enterprise #4).

Consecutive-run stability: drives several back-to-back SFT runs through
RunManager against a REAL fusion-mlx and asserts no state leaks between
rounds (each run reaches `running` then `completed`, each appears in
list_runs, no cross-run file-handle or registry corruption).

Mock-based suites patch httpx and assert trainer-side invariants; they
cannot catch the class of bugs that only surface on the Nth consecutive
run — a leaked file handle, a singleton that accumulates state, an event
log that appends across runs. This test closes that gap.

Marked ``integration`` AND ``load``; gated behind ``FUSION_TRAINER_LOAD=1``.
Auto-skips otherwise (keeps the fast suite fast and CI cheap). Run locally
against a started fusion-mlx:

    FUSION_TRAINER_LOAD=1 FUSION_MLX_ADMIN_TOKEN=<key> \
        pytest tests/test_load_stability.py -m load -v
"""

from __future__ import annotations

import json
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
from fusion_trainer.run_manager import get_run_manager


def _mlx_port() -> int:
    return int(os.environ.get("FUSION_MLX_PORT", "0")) or _default_mlx_port()


def _server_reachable() -> bool:
    port = _mlx_port()
    headers = {}
    token = os.environ.get("FUSION_MLX_ADMIN_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            f"http://localhost:{port}/v1/models", timeout=20.0, headers=headers
        )
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [pytest.mark.integration, pytest.mark.load]

# Default 3 rounds: enough to surface leaks without burning an hour. Override
# with FUSION_TRAINER_LOAD_ROUNDS for a longer soak.
def _rounds() -> int:
    return int(os.environ.get("FUSION_TRAINER_LOAD_ROUNDS", "3"))


@pytest.fixture(scope="module")
def live_env(tmp_path_factory):
    if not _server_reachable():
        pytest.skip(
            f"fusion-mlx not reachable at localhost:{_mlx_port()}; "
            "start it with ~/claude-home/fusion-mlx/start.sh start to run "
            "load/stability tests"
        )
    if os.environ.get("FUSION_TRAINER_LOAD", "") != "1":
        pytest.skip("set FUSION_TRAINER_LOAD=1 to run load/stability tests")
    cache_root = tmp_path_factory.mktemp("load_cache")
    # Fresh singleton cache dir so prior runs in the process don't taint it.
    os.environ["FUSION_TRAINER_CACHE"] = str(cache_root)
    mgr = get_run_manager()
    yield mgr, cache_root


@pytest.fixture(scope="module")
def base_model_id(live_env) -> str:
    mgr, _ = live_env
    data = mgr._client.health().get("data", [])
    if not data:
        pytest.skip("no models loaded on fusion-mlx; load a small base model")
    text_models = [
        m for m in data if (m.get("capabilities") or {}).get("text_generation")
    ]
    if not text_models:
        return str(data[0].get("id", ""))

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


def _wait_terminal(mgr, run_id: str, timeout: float = 600.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = mgr.run_status(run_id)
        status = str(st.get("status", ""))
        if status in ("completed", "failed", "cancelled"):
            return st
        time.sleep(2.0)
    mgr.stop_run(run_id)
    pytest.fail(f"run {run_id} did not reach terminal status within {timeout}s")


def test_consecutive_sft_no_state_leak(live_env, base_model_id: str):
    # Drive N consecutive SFT runs. Each must reach `running` then `completed`,
    # appear in list_runs, and produce a non-empty events.jsonl. A leaked file
    # handle or accumulated singleton state would surface as a run that fails
    # to start, a missing event file, or a registry that forgets earlier runs.
    mgr, cache_root = live_env
    seen_run_ids: list[str] = []
    src_samples = [
        {"input": "hello", "output": "world"},
        {"input": "ping", "output": "pong"},
    ]
    src_path = cache_root / "src_train.jsonl"
    src_path.write_text("\n".join(json.dumps(s) for s in src_samples) + "\n")

    for i in range(_rounds()):
        cfg = TrainerConfig(
            dataset=DatasetConfig(path=str(src_path), max_samples=0),
            sft=SFTConfig(
                model=base_model_id,
                method="lora",
                num_epochs=1,
                max_steps=2,
                lora_rank=4,
                lora_alpha=8.0,
                run_name=f"load_round_{i}_{int(time.time())}",
            ),
            cache_path=str(cache_root),
            mlx_port=_mlx_port(),
        )
        start = mgr.start_sft(cfg)
        run_id = start["run_id"]
        assert start["status"] == "running", (
            f"round {i}: start_sft did not report running: {start}"
        )
        seen_run_ids.append(run_id)

        final = _wait_terminal(mgr, run_id)
        assert final["status"] == "completed", (
            f"round {i}: run ended non-completed: {final}"
        )
        assert final.get("adapter_name"), (
            f"round {i}: empty adapter_name in terminal status: {final}"
        )
        events_file = Path(final.get("events_path", ""))
        assert events_file.is_file() and events_file.stat().st_size > 0, (
            f"round {i}: events.jsonl missing or empty at {events_file}"
        )

        # Registry remembers every prior run — no accumulation drops earlier
        # entries (a bounded-but-evicting registry is fine; a state leak that
        # overwrites the registry is not). Assert at least this run + the last
        # one are both present once N>=2.
        runs = mgr.list_runs(limit=200)
        run_ids = {r.get("run_id") for r in runs}
        assert run_id in run_ids, f"round {i}: run not in list_runs"
        if i >= 1:
            assert seen_run_ids[i - 1] in run_ids, (
                f"round {i}: previous run {seen_run_ids[i - 1]} dropped from registry"
            )

    # Cleanup intermediate run artifacts per the verify-then-clean rule: keep
    # only final outputs (adapter pointers on fusion-mlx) + logs. The cache
    # dir is a pytest tmp_path, auto-cleaned after the session; nothing to do.


def test_load_test_script_exists_and_is_executable():
    # Guard: the companion scripts/load_test.sh must ship executable so an
    # operator can run the shell soak without a pytest harness.
    script = Path(__file__).resolve().parent.parent / "scripts" / "load_test.sh"
    assert script.is_file(), f"load_test.sh missing at {script}"
    assert os.access(script, os.X_OK), f"load_test.sh not executable: {script}"
