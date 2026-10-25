"""Benchmark-based checkpoint selection (issue #75, eval-loop gear).

Closes the training flywheel's missing verification step: after an SFT run
that saved 2+ numbered checkpoints, run a fusion-bench task per candidate
checkpoint and pick the winner by bench score — not just lowest val loss.
The selection table (checkpoint, val_loss, bench scores, chosen) is
persisted next to the run as checkpoint_selection.json.

Pure HTTP: the trainer talks only to fusion-bench's task API
(POST /api/v1/tasks, GET /api/v1/tasks/{id}) — no bench imports. Bench
auth is fusion-identity tenant auth (X-Tenant-Id + Bearer JWT), so a
deployment routes this through an operator-issued token.

Upstream note (filed, not worked around): fusion-bench executors do not
yet forward a LoRA adapter path to fusion-mlx chat completions. The task
params carry `adapter_path` for each candidate so the moment bench lands
passthrough (fusion-mlx /v1/chat/completions already accepts `adapters`)
the loop scores the true checkpoint weights; until then bench scores
reflect the base model and the selector falls back to val_loss with a
loud, persisted reason — never a silent mis-selection.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from .config import validate_service_url

logger = logging.getLogger(__name__)

# Literal "localhost", not 127.0.0.1 — validate_service_url (S6) rejects
# numeric loopback hosts written as anything but the literal string.
DEFAULT_BENCH_URL = "http://localhost:11467"
BENCH_TASKS_PATH = "/api/v1/tasks"
SELECTION_FILENAME = "checkpoint_selection.json"

# Terminal task statuses in fusion-bench's task store (api/app.py
# _run_task): completed / failed / cancelled. Anything else keeps polling.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}

_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


class BenchError(RuntimeError):
    pass


def _default_bench_url() -> str:
    return os.environ.get("FUSION_BENCH_URL", DEFAULT_BENCH_URL)


def list_checkpoints(adapter_dir: Path | str) -> list[dict[str, Any]]:
    """All numbered checkpoints in a fusion-mlx adapter dir, step-ascending.

    Same NNNNNN_adapters.safetensors layout find_latest_checkpoint (#65)
    matches, generalized: eval needs every candidate, not just the newest.
    """
    out: list[dict[str, Any]] = []
    for p in sorted(Path(adapter_dir).glob("*_adapters.safetensors")):
        m = _CHECKPOINT_RE.match(p.name)
        if not m:
            continue
        out.append({"step": int(m.group(1)), "path": str(p)})
    return out


def nearest_val_loss(val_losses: list[tuple[int, float]], step: int) -> float | None:
    """val_loss recorded closest to (and not after, when possible) a step.

    val_losses is [(step, val_loss)] in report order from the run's events.
    A checkpoint saved between two eval points attributes the closer one —
    deterministic, no interpolation (Rule 5).
    """
    if not val_losses:
        return None
    return min(val_losses, key=lambda sv: abs(sv[0] - step))[1]


def select_best_checkpoint(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Deterministic winner pick over a selection table.

    Primary: bench score (higher wins; pass_rate preferred, metric_value
    fallback). Only rows whose bench run completed with a numeric score
    compete. Tie or no bench scores: lowest val_loss; tie again: the later
    step (more training). No val_loss either: the last checkpoint.
    Returns (chosen_row, basis) where basis is one of
    "bench" / "val_loss" / "last_checkpoint".
    """
    if not rows:
        raise BenchError("select_best_checkpoint: empty selection table")
    scored = [r for r in rows if r.get("bench_score") is not None]
    if scored:
        best = max(
            scored,
            key=lambda r: (
                float(r["bench_score"]),
                int(r.get("step", 0)),
            ),
        )
        return best, "bench"
    with_loss = [r for r in rows if r.get("val_loss") is not None]
    if with_loss:
        best = min(
            with_loss,
            key=lambda r: (
                float(r["val_loss"]),
                -int(r.get("step", 0)),
            ),
        )
        return best, "val_loss"
    return rows[-1], "last_checkpoint"


class BenchClient:
    """Thin HTTP wrapper around fusion-bench's task API (issue #75)."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        tenant_id: str = "",
        timeout: float = 15.0,
    ):
        # S6 pattern from ModelHubClient: validate before storing so a
        # tainted URL can't exfiltrate the bearer token.
        self._base = validate_service_url(
            base_url or _default_bench_url(), field_name="bench_url"
        ).rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if tenant_id:
            # fusion-bench's tenant middleware (fail-closed) requires the
            # header on every /api/v1/* call.
            self._headers["X-Tenant-Id"] = tenant_id
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        logger.info(
            "BenchClient initialized: base=%s tenant=%s auth=%s",
            self._base,
            bool(tenant_id),
            bool(api_key),
        )

    def create_task(
        self,
        model: str,
        executor_key: str = "speed",
        suite: str = "",
        level: str = "L1",
        params: dict[str, Any] | None = None,
        timeout_seconds: int = 1800,
        max_samples: int | None = None,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "executor_key": executor_key,
            "level": level,
            "timeout_seconds": timeout_seconds,
            "params": params or {},
        }
        if suite:
            body["suite"] = suite
        if max_samples is not None:
            body["max_samples"] = max_samples
        if dataset:
            body["dataset"] = dataset
        url = f"{self._base}{BENCH_TASKS_PATH}"
        logger.info("bench create_task: POST %s model=%s executor=%s", url, model, executor_key)
        try:
            resp = httpx.post(url, json=body, headers=self._headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise BenchError(f"bench POST transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise BenchError(
                f"bench POST {url} failed: {resp.status_code} {resp.text[:300]}"
            )
        try:
            task = resp.json()
        except ValueError as exc:
            raise BenchError(f"bench POST {url} returned non-JSON: {resp.text[:200]}") from exc
        if not task.get("task_id"):
            raise BenchError(f"bench POST {url} returned no task_id: {task}")
        return task

    def get_task(self, task_id: str) -> dict[str, Any]:
        url = f"{self._base}{BENCH_TASKS_PATH}/{task_id}"
        try:
            resp = httpx.get(url, headers=self._headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise BenchError(f"bench GET transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise BenchError(
                f"bench GET {url} failed: {resp.status_code} {resp.text[:300]}"
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise BenchError(f"bench GET {url} returned non-dict: {type(data).__name__}")
        return data

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> dict[str, Any]:
        """Poll until terminal status. A single failed poll retries with
        backoff (A1 pattern); a 404 raises immediately — the task is gone.
        """
        deadline = time.monotonic() + timeout
        backoff = poll_interval
        while time.monotonic() < deadline:
            try:
                task = self.get_task(task_id)
                backoff = poll_interval
            except BenchError as exc:
                if "failed: 404" in str(exc):
                    raise
                logger.warning("bench poll %s failed (retrying): %s", task_id, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                continue
            status = str(task.get("status", ""))
            if status in _TERMINAL_STATUSES:
                return task
            time.sleep(poll_interval)
        raise BenchError(f"bench task {task_id} did not reach terminal status within {timeout}s")

    def run_benchmark(
        self,
        model: str,
        adapter_path: str = "",
        executor_key: str = "speed",
        suite: str = "",
        level: str = "L1",
        params: dict[str, Any] | None = None,
        timeout_seconds: int = 1800,
        poll_interval: float = 5.0,
        wait_timeout: float = 3600.0,
    ) -> dict[str, Any]:
        """Submit one bench task and wait for it; extract the score.

        Returns {status, task_id, metric_name, metric_value, pass_rate,
        duration_seconds, errors}. A failed task returns status="failed"
        with its errors — the caller decides (selector skips unscored
        rows); never a faked score.
        """
        body_params = dict(params or {})
        if adapter_path:
            body_params["adapter_path"] = adapter_path
        task = self.create_task(
            model=model,
            executor_key=executor_key,
            suite=suite,
            level=level,
            params=body_params,
            timeout_seconds=timeout_seconds,
        )
        task_id = task["task_id"]
        final = self.wait_for_task(
            task_id, poll_interval=poll_interval, timeout=wait_timeout
        )
        status = str(final.get("status", ""))
        result = final.get("result") or {}
        out = {
            "status": status,
            "task_id": task_id,
            "metric_name": result.get("metric_name", ""),
            "metric_value": result.get("metric_value"),
            "pass_rate": result.get("pass_rate"),
            "duration_seconds": result.get("duration_seconds", 0.0),
            "errors": list(result.get("errors", []) or []),
        }
        # Score for ranking: pass_rate is the quality signal; metric_value
        # (e.g. decode_speed) is the fallback for executors without cases.
        score = out["pass_rate"]
        if score is None:
            score = out["metric_value"]
        if score is not None:
            try:
                out["score"] = float(score)
            except (TypeError, ValueError):
                out["score"] = None
        else:
            out["score"] = None
        if status != "completed" or out["score"] is None:
            logger.warning(
                "bench task %s ended %s score=%s errors=%s",
                task_id,
                status,
                out["score"],
                out["errors"][:2],
            )
        return out


def stage_checkpoint_for_serving(
    checkpoint_file: Path | str, stage_dir: Path | str
) -> str:
    """Copy a numbered checkpoint into a dir holding adapters.safetensors.

    fusion-mlx serves an adapter from a directory containing exactly
    adapters.safetensors; numbered checkpoints (NNNNNN_adapters.safetensors)
    are files inside the run dir. Staging (same trick as #65 resume_stage)
    also protects the source checkpoint from anything writing to the stage.
    """
    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    dest = stage / "adapters.safetensors"
    shutil.copy2(checkpoint_file, dest)
    logger.info(
        "staged checkpoint %s -> %s", checkpoint_file, dest,
    )
    return str(stage)


def run_checkpoint_eval(
    model: str,
    adapter_dir: Path | str,
    val_losses: list[tuple[int, float]] | None = None,
    bench: BenchClient | None = None,
    bench_executor: str = "speed",
    bench_suite: str = "",
    bench_params: dict[str, Any] | None = None,
    poll_interval: float = 5.0,
    task_wait_timeout: float = 3600.0,
    stage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate every numbered checkpoint in adapter_dir, pick the winner.

    Builds the persisted selection table (checkpoint, step, val_loss,
    bench task/score, chosen), writes it atomically to
    {adapter_dir}/checkpoint_selection.json, and returns
    {table, chosen, basis, path, bench_status}.

    Fail-visible, never raises: a bench outage marks every row
    bench_status="unavailable" and the selector falls back to val_loss —
    the run's result is never blocked by a bench outage, and the fallback
    is recorded in the table, not silent.
    """
    val_losses = val_losses or []
    candidates = list_checkpoints(adapter_dir)
    table: list[dict[str, Any]] = []
    bench_status = "not_configured"
    if not candidates:
        # A run with no numbered checkpoints still gets a table recording
        # why no selection happened (single final save = nothing to choose).
        table.append(
            {
                "checkpoint": "final",
                "step": None,
                "path": str(Path(adapter_dir) / "adapters.safetensors"),
                "val_loss": nearest_val_loss(val_losses, 10**9),
                "bench_task_id": "",
                "bench_status": "skipped",
                "bench_score": None,
                "chosen": True,
            }
        )
        chosen, basis = table[0], "single_checkpoint"
    else:
        for cand in candidates:
            table.append(
                {
                    "checkpoint": Path(cand["path"]).name,
                    "step": cand["step"],
                    "path": cand["path"],
                    "val_loss": nearest_val_loss(val_losses, cand["step"]),
                    "bench_task_id": "",
                    "bench_status": "pending",
                    "bench_score": None,
                    "chosen": False,
                }
            )
        chosen, basis = None, "last_checkpoint"
        if bench is not None:
            bench_status = "ok"
            for row in table:
                stage_dir = (
                    Path(stage_root) / f"eval_stage_{row['step']}"
                    if stage_root
                    else Path(adapter_dir) / f"eval_stage_{row['step']}"
                )
                try:
                    served_dir = stage_checkpoint_for_serving(row["path"], stage_dir)
                    res = bench.run_benchmark(
                        model=model,
                        adapter_path=served_dir,
                        executor_key=bench_executor,
                        suite=bench_suite,
                        params=bench_params,
                        poll_interval=poll_interval,
                        wait_timeout=task_wait_timeout,
                    )
                    row["bench_task_id"] = res.get("task_id", "")
                    row["bench_status"] = res.get("status", "")
                    row["bench_score"] = res.get("score")
                    row["bench_metric_name"] = res.get("metric_name", "")
                    row["bench_pass_rate"] = res.get("pass_rate")
                except BenchError as exc:
                    # One unreachable bench = every remaining row unscorable.
                    logger.warning(
                        "checkpoint eval: bench unavailable at %s — %s", row["checkpoint"], exc,
                    )
                    bench_status = "unavailable"
                    for rest in table:
                        if rest["bench_status"] in ("pending",):
                            rest["bench_status"] = "unavailable"
                    break
        chosen, basis = select_best_checkpoint(table)
    for row in table:
        row["chosen"] = row is chosen

    result = {
        "table": table,
        "chosen": chosen,
        "basis": basis,
        "bench_status": bench_status,
    }
    if bench_status == "unavailable":
        result["fallback_reason"] = (
            "bench unreachable or rejected the task — selection fell back to "
            "val_loss; see logs"
        )
    out_path = Path(adapter_dir) / SELECTION_FILENAME
    tmp = out_path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, out_path)
        logger.info("checkpoint selection persisted: %s (basis=%s)", out_path, basis)
    except OSError as exc:
        # Persistence failure must not kill the run result, but must be loud.
        logger.warning("checkpoint selection persist failed: %s", exc)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    result["path"] = str(out_path)
    return result
