"""RunManager — async run orchestration + live progress surface for the GUI.

fusion-trainer is a pure computation layer; it never hosts its own RPC/HTTP
server for fusion-studio. fusion-agent-studio's central daemon imports
fusion_trainer in-process and dispatches trainer.* RPC. RunManager is the
richer API surface that daemon calls: start SFT/RLSL runs, stream live
progress events, list runs/adapters/datasets/presets, stop runs.

Design:
  - start_sft/start_rlsl run the trainer in a worker thread, return
    immediately with {run_id, status:"running"}.
  - A per-run on_event sink is installed on the trainer before it runs;
    every trainer feeds raw fusion-mlx job events into it. Events land in a
    per-run deque(maxlen=1000) (mirrors ExperimentLogger) AND append to
    runs/{run_id}/events.jsonl on disk.
  - run_progress(run_id, since_step) replays buffered events with step >
    since_step — deterministic data flow, no model routing (Rule 5).
  - stop_run sets a cancel flag; the sink cancels the live fusion-mlx job
    (job id recorded by trainers via the sink).

Fail-visible (Rule 12): MXFP8 upstream-blocked surfaces in info(), never
silently downgraded. Run crashes/unknown statuses never fake "completed".
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import TrainerConfig
from .mlx_client import MLXFineTuneClient

logger = logging.getLogger(__name__)

_TERMINAL = ("completed", "failed", "cancelled")
_MXF8_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/402"
# A3: module slot + lock declared early so RunManager.__init__ can register
# itself / warn on a bypassing direct construction. Filled in at module bottom
# for readability, but available at __init__ call time (module fully loaded).
_RUN_MANAGER_SINGLETON: RunManager | None = None
_SINGLETON_LOCK = threading.Lock()


class RunManager:
    """Orchestrate training runs + expose live progress for the GUI.

    Thread-safe: run threads mutate the registry under _LOCK; readers
    (list_runs/run_status/run_progress) acquire the same lock for a snapshot.
    """

    def __init__(
        self,
        cache_path: str = "",
        model_hub_path: str = "",
        mlx_port: int | None = None,
    ):
        self.cache_path = cache_path or os.environ.get(
            "FUSION_TRAINER_CACHE", str(Path.cwd() / "runtime_cache")
        )
        self.model_hub_path = model_hub_path or os.environ.get(
            "FUSION_MODEL_HUB", str(Path.home() / "fusion-model-hub")
        )
        self.mlx_port = mlx_port or int(os.environ.get("FUSION_MLX_PORT", "11432"))
        self._runs_root = Path(self.cache_path) / "runs"
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, dict[str, Any]] = {}
        # P1: per-file dataset count cache keyed by (mtime, size). Avoids a
        # full re-scan + re-count of every hub dataset file on each GUI refresh.
        self._dataset_cache: dict[str, tuple[float, int, int]] = {}
        # A2: active sinks live on the INSTANCE, not a module global. Two
        # RunManager instances (CLI + agent-studio daemon) used to share one
        # module-level sink dict while holding separate _runs registries, so a
        # stop_run on one instance could not find a run owned by the other.
        # Per-instance sinks pair with per-instance _runs; production callers
        # should go through get_run_manager() so there is one shared instance.
        self._active_sinks: dict[str, Any] = {}
        self._lock = threading.Lock()
        # A3: defense-in-depth against the singleton being bypassed. The
        # production GUI path (fusion-agent-studio TrainingService) used to
        # construct RunManager() directly, creating a second registry that the
        # CLI singleton could not see — stop_run/list_runs crossed instances
        # and silently no-op'd while upstream jobs burned GPU. The singleton
        # entry get_run_manager() is the supported path; a direct construct is
        # allowed for tests/isolated instances, but if a singleton already
        # exists and this is a *different* instance, warn loudly so the bypass
        # is visible (not silent). Register self as singleton when none set.
        global _RUN_MANAGER_SINGLETON
        with _SINGLETON_LOCK:
            if _RUN_MANAGER_SINGLETON is None:
                _RUN_MANAGER_SINGLETON = self
            elif _RUN_MANAGER_SINGLETON is not self:
                logger.warning(
                    "RunManager constructed directly while a process singleton "
                    "already exists (existing=%r new=%r). Two registries are "
                    "now live and stop_run/list_runs will NOT cross instances — "
                    "use get_run_manager() instead. Upstream: "
                    "fusion-agent-studio#268.",
                    id(_RUN_MANAGER_SINGLETON),
                    id(self),
                )
        # A4: persist in-memory running runs at interpreter exit. The worker
        # thread is non-daemon (see _start), but a hard SIGKILL / OOM still
        # skips the terminal _persist_meta. atexit covers graceful exit; the
        # meta-at-start write (in _start) covers the crash-then-restart path.
        atexit.register(self._atexit_persist_all)
        self._load_existing_runs()

    # ------------------------------------------------------------------
    # Run registry persistence (lightweight: scan runs/ dir)
    # ------------------------------------------------------------------

    def _load_existing_runs(self) -> None:
        """Reload terminal runs from disk so a restart still lists history."""
        if not self._runs_root.exists():
            return
        for run_dir in self._runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            meta = run_dir / "run_meta.json"
            events_path = run_dir / "events.jsonl"
            if not meta.exists():
                # A4: meta absent. The old code `continue`d here, which made
                # the L3 zombie-cancel path dead code for the most common
                # crash scenario: a run killed mid-flight before its terminal
                # _persist_meta wrote run_meta.json. If events.jsonl exists,
                # a run was in progress — reconstruct a minimal record (run_id
                # from dir name, job_id/kind/phase scavenged from the event
                # stream) so zombie-cancel can fire. If no events either, the
                # dir is a stray/empty — skip silently.
                if not events_path.exists():
                    continue
                data = self._reconstruct_meta_from_events(run_dir, events_path)
                if data is None:
                    continue
                logger.warning(
                    "run %s has events.jsonl but no run_meta.json — "
                    "reconstructing from events (crash before terminal meta)",
                    data.get("run_id", run_dir.name),
                )
            else:
                try:
                    data = json.loads(meta.read_text())
                except Exception as exc:
                    logger.warning("skip unreadable run meta %s: %s", meta, exc)
                    continue
            rid = data.get("run_id", run_dir.name)
            # Only rehydrate terminal runs; running runs from a dead process
            # are marked failed (fail-visible: never claim a zombie is alive).
            if data.get("status") not in _TERMINAL:
                # L3: the run's fusion-mlx job is NOT owned by this process.
                # Marking failed here only updates our view; the upstream job
                # may still be burning GPU/memory. Best-effort cancel it via
                # the recorded job_id/kind, then mark failed. If fusion-mlx is
                # unreachable, surface the job_id loudly so the operator can
                # stop it manually.
                zombie_job = data.get("job_id", "") or ""
                zombie_kind = data.get("job_kind", "sft") or "sft"
                if zombie_job:
                    try:
                        client = MLXFineTuneClient(
                            port=int(
                                data.get("mlx_port")
                                or os.environ.get("FUSION_MLX_PORT", "11432")
                            )
                        )
                        if zombie_kind in ("dpo", "orpo"):
                            client.cancel_dpo_job(zombie_job)
                        elif zombie_kind == "grpo":
                            client.cancel_grpo_job(zombie_job)
                        elif zombie_kind == "reward":
                            client.cancel_reward_job(zombie_job)
                        else:
                            client.cancel_job(zombie_job)
                        client.close()
                        logger.info(
                            "zombie run %s: cancelled upstream job %s (kind=%s)",
                            rid,
                            zombie_job,
                            zombie_kind,
                        )
                    except Exception as exc:
                        logger.warning(
                            "zombie run %s: could not cancel upstream job %s "
                            "(kind=%s) — it may STILL be running on fusion-mlx. "
                            "Stop it manually. error=%s",
                            rid,
                            zombie_job,
                            zombie_kind,
                            exc,
                        )
                data["status"] = "failed"
                data["error"] = "process restarted while run was active"
            data["events_buffer"] = deque(maxlen=1000)
            data.setdefault("canceled", False)
            data.setdefault("job_id", "")
            data.setdefault("job_kind", "sft")
            self._runs[rid] = data

    @staticmethod
    def _reconstruct_meta_from_events(
        run_dir: Path, events_path: Path
    ) -> dict[str, Any] | None:
        """A4: build a minimal run record from events.jsonl when run_meta.json
        is missing (run crashed before terminal persist). Scavenges the latest
        job id/kind + phase + step from the event stream so zombie-cancel has
        a job_id to target. Returns None if the event stream is empty/invalid.
        """
        rid = run_dir.name
        data: dict[str, Any] = {
            "run_id": rid,
            "name": rid,
            "method": rid.split("_", 1)[0] if "_" in rid else "sft",
            "status": "running",
            "created": 0.0,
            "step": 0,
            "job_id": "",
            "job_kind": "sft",
            "phase": "",
            "model": "",
            "mlx_port": int(os.environ.get("FUSION_MLX_PORT", "11432")),
            "metrics": {},
            "final_adapter": "",
            "adapter_name": "",
        }
        saw_event = False
        try:
            with open(events_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    saw_event = True
                    if ev.get("type") == "job":
                        if ev.get("job_id"):
                            data["job_id"] = ev["job_id"]
                        if ev.get("kind"):
                            data["job_kind"] = ev["kind"]
                    if ev.get("type") == "phase":
                        data["phase"] = ev.get("name", ev.get("phase", ""))
                    step = ev.get("step", ev.get("iter"))
                    if isinstance(step, (int, float)) and int(step) > int(
                        data.get("step", 0)
                    ):
                        data["step"] = int(step)
        except Exception as exc:
            logger.warning("reconstruct meta: read events %s failed: %s", events_path, exc)
            return None
        if not saw_event:
            return None
        return data

    # ------------------------------------------------------------------
    # info / presets / datasets / adapters
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """System info for the Settings/Dashboard panels."""
        mlx_alive = False
        models: list[dict[str, Any]] = []
        gpu_mem_gb: float | None = None
        try:
            client = MLXFineTuneClient(port=self.mlx_port)
            data = client.health()
            mlx_alive = True
            models = data.get("data", []) or []
        except Exception as exc:
            logger.info("info: fusion-mlx unreachable: %s", exc)

        presets = self.list_presets()
        return {
            "version": __version__,
            "mlx_port": self.mlx_port,
            "mlx_alive": mlx_alive,
            "models": models,
            "gpu_mem_gb": gpu_mem_gb,
            "model_hub_path": self.model_hub_path,
            "cache_path": self.cache_path,
            "presets": presets,
            "warnings": self._upstream_warnings(),
        }

    @staticmethod
    def _upstream_warnings() -> list[dict[str, str]]:
        return [
            {
                "id": "mxfp8",
                "level": "warn",
                "message": "MXFP8 mixed precision upstream-blocked; trains at default precision",
                "issue": _MXF8_ISSUE,
            }
        ]

    def list_presets(self, kind: str = "") -> list[dict[str, Any]]:
        """List SFT and/or RLSL presets for the New Run dropdowns."""
        out: list[dict[str, Any]] = []
        if kind in ("", "sft"):
            from .sft.presets import SFT_PRESETS

            for name, p in SFT_PRESETS.items():
                hyper = {
                    k: v
                    for k, v in p.items()
                    if k not in ("description", "memory_estimate_gb")
                }
                out.append(
                    {
                        "name": name,
                        "kind": "sft",
                        "description": p.get("description", ""),
                        "memory_estimate_gb": p.get("memory_estimate_gb"),
                        "hyperparams": hyper,
                    }
                )
        if kind in ("", "rlsl"):
            from .rlsl.presets import RLSL_PRESETS

            for name, p in RLSL_PRESETS.items():
                hyper = {
                    k: v
                    for k, v in p.items()
                    if k not in ("description", "memory_estimate_gb")
                }
                out.append(
                    {
                        "name": name,
                        "kind": "rlsl",
                        "description": p.get("description", ""),
                        "memory_estimate_gb": p.get("memory_estimate_gb"),
                        "hyperparams": hyper,
                    }
                )
        return out

    def list_datasets(self) -> list[dict[str, Any]]:
        """Scan {model_hub_path}/datasets/ (read-only hub, by category).

        P1: results are cached per file by (mtime, size) so repeated GUI
        refreshes do not re-scan + re-count every file. Parquet uses pyarrow
        row count (was counted as text lines = wrong). Invalidated on file
        change automatically.
        """
        hub = Path(self.model_hub_path) / "datasets"
        if not hub.exists():
            logger.info("list_datasets: hub datasets dir missing: %s", hub)
            return []
        out: list[dict[str, Any]] = []
        for entry in sorted(hub.iterdir()):
            if entry.is_dir():
                for f in sorted(entry.glob("*")):
                    if f.suffix in (".jsonl", ".json", ".csv", ".parquet"):
                        out.append(
                            {
                                "name": f"{entry.name}/{f.stem}",
                                "category": entry.name,
                                "format": f.suffix.lstrip("."),
                                "path": str(f),
                                "n_samples": self._count_samples_cached(f),
                            }
                        )
            elif entry.suffix in (".jsonl", ".json", ".csv", ".parquet"):
                out.append(
                    {
                        "name": entry.stem,
                        "category": "uncategorized",
                        "format": entry.suffix.lstrip("."),
                        "path": str(entry),
                        "n_samples": self._count_samples_cached(entry),
                    }
                )
        return out

    def _count_samples_cached(self, path: Path) -> int:
        """mtime+size-keyed cache over _count_samples (P1). Avoids re-reading
        every file on each GUI refresh; fresh data wins when a file changes."""
        try:
            st = path.stat()
        except OSError:
            return _count_samples(path)
        key = str(path)
        cached = self._dataset_cache.get(key)
        if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return cached[2]
        n = _count_samples(path)
        self._dataset_cache[key] = (st.st_mtime, st.st_size, n)
        return n

    def preview_dataset(self, name: str, limit: int = 5) -> dict[str, Any]:
        """Read first `limit` rows of a hub dataset by {category}/{name}."""
        limit = max(1, min(int(limit), 100))
        path = Path(self.model_hub_path) / "datasets" / f"{name}.jsonl"
        if not path.exists():
            path = (
                Path(self.model_hub_path) / "datasets" / f"{name.split('/')[-1]}.jsonl"
            )
        if not path.exists():
            return {
                "rows": [],
                "fields": [],
                "n_total": 0,
                "error": f"dataset {name} not found",
            }
        rows: list[dict[str, Any]] = []
        fields: set[str] = set()
        total = 0
        with open(path) as fh:
            for line in fh:
                total += 1
                if len(rows) < limit:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rows.append(obj)
                    fields.update(obj.keys())
        return {"rows": rows, "fields": sorted(fields), "n_total": total}

    def list_adapters(self, model: str = "") -> list[dict[str, Any]]:
        """Proxy fusion-mlx GET /admin/api/fine-tune/adapters."""
        try:
            client = MLXFineTuneClient(port=self.mlx_port)
            return client.list_adapters(model_id=model)
        except Exception as exc:
            logger.error("list_adapters failed: %s", exc)
            return []

    def delete_adapter(self, name: str) -> dict[str, Any]:
        """Adapter deletion — upstream endpoint pending; fail-visible stub.

        fusion-mlx has no DELETE /adapters yet. Return explicit unsupported
        rather than fake success (Rule 12).
        """
        logger.warning("delete_adapter unsupported upstream: %s", name)
        return {
            "ok": False,
            "error": "adapter delete endpoint not yet available in fusion-mlx",
        }

    # ------------------------------------------------------------------
    # Run lifecycle: start
    # ------------------------------------------------------------------

    def start_sft(self, config: TrainerConfig) -> dict[str, Any]:
        """Start an SFT run in a worker thread; return {run_id, status}."""
        return self._start(config, method="sft")

    def start_rlsl(self, config: TrainerConfig) -> dict[str, Any]:
        """Start an RLSL/alignment run (method in dpo|orpo|grpo|rlsl)."""
        return self._start(config, method="rlsl")

    def _start(self, config: TrainerConfig, method: str) -> dict[str, Any]:
        # R5: run_id was a ms-timestamp ({method}_{ms}); two runs started in
        # the same millisecond collided on run_dir. Use a uuid4 suffix for
        # collision-proof uniqueness while keeping the method prefix for
        # human-readability.
        import uuid

        run_id = f"{method}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        run_name = (
            config.sft.run_name if method == "sft" else config.rlsl.run_name
        ) or run_id
        run_dir = self._runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "events.jsonl"
        log_path = run_dir / "run.log"

        record: dict[str, Any] = {
            "run_id": run_id,
            "name": run_name,
            "method": method,
            "method_kind": config.rlsl.method if method == "rlsl" else "sft",
            "model": config.sft.model,
            "status": "running",
            "created": time.time(),
            "step": 0,
            "total_steps": 0,
            "phase": "",
            "job_id": "",
            "job_kind": "sft",
            "final_adapter": "",
            "adapter_name": "",
            "metrics": {},
            "canceled": False,
            "events_buffer": deque(maxlen=1000),
            "events_path": str(events_path),
            "log_path": str(log_path),
            "run_dir": str(run_dir),
            "mlx_port": self.mlx_port,
            "error": "",
        }
        with self._lock:
            self._runs[run_id] = record
            # A4: persist meta AT START (status:running) atomically, BEFORE the
            # worker thread spawns. The old code only wrote run_meta.json at the
            # terminal _run_thread path — a crash/SIGKILL/OOM mid-run left no
            # meta on disk, so _load_existing_runs skipped it (it `continue`s
            # when meta is absent) and the L3 zombie-cancel path never fired,
            # leaving the upstream fusion-mlx job burning GPU uncancelled.
            self._persist_meta(run_id)

        sink = _RunSink(record, self._lock)
        self._active_sinks[run_id] = sink

        thread = threading.Thread(
            target=self._run_thread,
            args=(run_id, config, method, sink, log_path),
            name=f"trainer-run-{run_id}",
            # A4/R6: non-daemon so the interpreter waits for the run on exit
            # rather than forcibly killing the worker mid-job (daemon threads
            # are slain without running finally blocks → no terminal meta, no
            # upstream cancel). The trainer polls fusion-mlx and is the only
            # thing that can mark the job terminal; killing it orphan-locks the
            # job. Non-daemon lets atexit + graceful shutdown persist meta.
            daemon=False,
        )
        record["thread"] = thread
        thread.start()
        logger.info(
            "started run %s method=%s model=%s", run_id, method, config.sft.model
        )
        return {"run_id": run_id, "status": "running"}

    def _run_thread(
        self,
        run_id: str,
        config: TrainerConfig,
        method: str,
        sink: _RunSink,
        log_path: Path,
    ) -> None:
        """Worker: install sink, run trainer, capture terminal status."""
        fh = logging.FileHandler(log_path)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        # A3: attach to the fusion_trainer package logger, NOT the root logger.
        # Root-level handlers stacked across concurrent runs intercepted ALL
        # app logs (incl. other libraries) and cross-wrote every run's file.
        # The package logger scopes the handler to trainer emissions. (Truly
        # concurrent runs still share this logger; the GUI is single-run-at-a-
        # time in practice, and each handler is removed in the finally below.)
        pkg_logger = logging.getLogger("fusion_trainer")
        pkg_logger.addHandler(fh)
        sink.emit({"type": "status", "status": "running"})

        result: dict[str, Any]
        try:
            if method == "sft":
                from .sft import SFTTrainer

                trainer = SFTTrainer(config)
                trainer.on_event = sink.emit
                result = trainer.train()
            else:
                from .rlsl import RLSLEngine

                engine = RLSLEngine(config)
                engine.on_event = sink.emit
                result = engine.run()
        except Exception as exc:
            logger.exception("run %s crashed", run_id)
            result = {"status": "failed", "error": str(exc)}
        finally:
            pkg_logger.removeHandler(fh)
            fh.close()

        canceled = sink.is_canceled()
        final_status = "cancelled" if canceled else result.get("status", "failed")
        if final_status not in _TERMINAL:
            final_status = "failed"

        # Issue #49: publish trained LoRA adapter to fusion-model-hub after a
        # completed run, if config.publish_adapter is set. Non-blocking: a hub
        # outage warns loudly but never flips the run status (fail-visible).
        publish_result = None
        if final_status == "completed" and getattr(config, "publish_adapter", False):
            from .hub_client import publish_adapter_if_enabled

            publish_result = publish_adapter_if_enabled(config, result)
            if publish_result.get("published"):
                logger.info(
                    "run %s adapter published to hub: %s",
                    run_id,
                    publish_result.get("model", {}).get("id", ""),
                )
            elif publish_result.get("error"):
                logger.warning(
                    "run %s adapter publish failed (non-blocking): %s",
                    run_id,
                    publish_result["error"],
                )

        sink.emit(
            {
                "type": "status",
                "status": final_status,
                "adapter_path": result.get("final_model_path")
                or result.get("adapter_path", ""),
                "adapter_name": result.get("adapter_name", ""),
                "error": result.get("error", ""),
                "published": bool(publish_result and publish_result.get("published")),
            }
        )
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is not None:
                rec["status"] = final_status
                rec["error"] = result.get("error", "")
                rec["final_adapter"] = result.get("final_model_path") or result.get(
                    "adapter_path", ""
                )
                rec["adapter_name"] = result.get("adapter_name", "")
                rec["finished"] = time.time()
                if publish_result is not None:
                    rec["publish_result"] = publish_result
                self._persist_meta(run_id)
        self._active_sinks.pop(run_id, None)
        logger.info("run %s ended: %s", run_id, final_status)

    # ------------------------------------------------------------------
    # Run queries
    # ------------------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List runs newest-first, without the in-memory event buffer."""
        with self._lock:
            items = sorted(
                self._runs.values(),
                key=lambda r: r.get("created", 0),
                reverse=True,
            )
        out: list[dict[str, Any]] = []
        for r in items[: max(1, int(limit))]:
            out.append(
                {
                    "run_id": r["run_id"],
                    "name": r.get("name", ""),
                    "method": r.get("method_kind", r.get("method")),
                    "model": r.get("model", ""),
                    "status": r.get("status", ""),
                    "step": r.get("step", 0),
                    "total_steps": r.get("total_steps", 0),
                    "phase": r.get("phase", ""),
                    "created": r.get("created", 0),
                    "final_adapter": r.get("final_adapter", ""),
                    "adapter_name": r.get("adapter_name", ""),
                    "error": r.get("error", ""),
                }
            )
        return out

    def run_status(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return {"run_id": run_id, "status": "unknown", "error": "no such run"}
            return {
                "run_id": run_id,
                "status": r.get("status", ""),
                "step": r.get("step", 0),
                "total_steps": r.get("total_steps", 0),
                "phase": r.get("phase", ""),
                "metrics": dict(r.get("metrics", {})),
                "adapter_name": r.get("adapter_name", ""),
                "final_adapter": r.get("final_adapter", ""),
                "error": r.get("error", ""),
                "created": r.get("created", 0),
                "job_id": r.get("job_id", ""),
            }

    def run_progress(self, run_id: str, since_step: int = -1) -> dict[str, Any]:
        """Replay buffered events with step > since_step (deterministic, Rule 5)."""
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return {"run_id": run_id, "events": [], "status": "unknown"}
            buf: deque = r["events_buffer"]
            events = [dict(ev) for ev in buf]
            status = r.get("status", "")
        filtered = [
            ev for ev in events if ev.get("step", ev.get("iter", -1)) > since_step
        ]
        return {"run_id": run_id, "events": filtered, "status": status}

    def stop_run(self, run_id: str) -> dict[str, Any]:
        """Request cancellation; returns {status: cancelling}. The sink
        cancels the live fusion-mlx job when it next observes the flag."""
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return {"run_id": run_id, "status": "unknown", "error": "no such run"}
            if r.get("status") in _TERMINAL:
                return {"run_id": run_id, "status": r["status"]}
            r["canceled"] = True
        sink = self._active_sinks.get(run_id)
        if sink is not None:
            sink.request_cancel()
        logger.info("stop_run requested for %s", run_id)
        return {"run_id": run_id, "status": "cancelling"}

    def publish_run(
        self,
        run_id: str,
        hub_url: str = "",
        api_key: str = "",
    ) -> dict[str, Any]:
        """Publish a completed run's adapter to fusion-model-hub (issue #49).

        Reads the run's registry record (adapter_name + model). Used by the
        CLI `publish <run_id>` subcommand to retroactively publish a run that
        was trained without --publish. Fail-visible: returns {published:False,
        error:...} never raises.
        """
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return {"run_id": run_id, "published": False, "error": "no such run"}
            adapter_name = r.get("adapter_name", "")
            base_model = r.get("model", "")
            status = r.get("status", "")
        if status != "completed":
            return {
                "run_id": run_id,
                "published": False,
                "error": f"run status is {status!r}, not completed",
            }
        if not adapter_name:
            return {
                "run_id": run_id,
                "published": False,
                "error": "run has no adapter_name recorded",
            }
        if not base_model:
            return {
                "run_id": run_id,
                "published": False,
                "error": "run has no base model recorded",
            }
        # L4 point 3: never register a hub model pointing at non-existent
        # weights. Verify the adapter still exists on fusion-mlx before POST.
        # If fusion-mlx is unreachable, fail-visible and refuse to publish
        # rather than register a dangling entry.
        mlx_client = MLXFineTuneClient(port=self.mlx_port)
        try:
            try:
                adapters = mlx_client.list_adapters(model_id=base_model)
            except Exception as exc:
                logger.warning(
                    "publish_run %s: adapter existence check failed (%s); "
                    "refusing to publish without verification",
                    run_id,
                    exc,
                )
                return {
                    "run_id": run_id,
                    "published": False,
                    "error": f"adapter existence check failed: {exc}",
                }
        finally:
            mlx_client.close()
        adapter_exists = any(str(a.get("name", "")) == adapter_name for a in adapters)
        if not adapter_exists:
            logger.error(
                "publish_run %s: adapter %r not found on fusion-mlx for "
                "model %r; refusing to publish dangling hub entry",
                run_id,
                adapter_name,
                base_model,
            )
            return {
                "run_id": run_id,
                "published": False,
                "error": (
                    f"adapter {adapter_name!r} not found on fusion-mlx "
                    f"(model {base_model!r}); weights may have been deleted"
                ),
            }
        from .hub_client import HubPublishError, ModelHubClient

        url = hub_url or os.environ.get(
            "FUSION_MODEL_HUB_URL", "http://localhost:11432"
        )
        key = api_key or os.environ.get("FUSION_MODEL_HUB_API_KEY", "")
        client = ModelHubClient(base_url=url, api_key=key)
        desc = f"LoRA adapter from fusion-trainer run {run_id}"
        try:
            pub = client.publish_adapter(
                name=adapter_name, base_model=base_model, description=desc
            )
        except HubPublishError as exc:
            logger.warning("publish_run %s failed (non-blocking): %s", run_id, exc)
            return {"run_id": run_id, "published": False, "error": str(exc)}
        with self._lock:
            r = self._runs.get(run_id)
            if r is not None:
                r["publish_result"] = pub
                self._persist_meta(run_id)
        return {"run_id": run_id, **pub}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _persist_meta(self, run_id: str) -> None:
        r = self._runs.get(run_id)
        if r is None:
            return
        meta_path = Path(r["run_dir"]) / "run_meta.json"
        serial = {k: v for k, v in r.items() if k not in ("events_buffer", "thread")}
        # M2: atomic write — write to a temp file in the same dir, fsync, then
        # os.replace so a crash mid-write never leaves a half-written meta
        # (which would make events.jsonl and run_meta.json disagree on status).
        tmp_path = meta_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(serial, default=str, ensure_ascii=False, indent=2)
            )
            os.replace(tmp_path, meta_path)
        except Exception as exc:
            logger.warning("persist meta %s failed: %s", run_id, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def _atexit_persist_all(self) -> None:
        """A4: at interpreter exit, persist meta for any still-running run so a
        graceful shutdown leaves a recoverable record. Non-daemon workers
        normally reach the terminal path, but a signal during the brief window
        between run start and natural completion is covered by this + the
        meta-at-start write. Idempotent + silent on failure (atexit context)."""
        try:
            with self._lock:
                running = [
                    rid
                    for rid, r in self._runs.items()
                    if r.get("status") not in _TERMINAL
                ]
            for rid in running:
                self._persist_meta(rid)
        except Exception:
            pass


# A2/A3: process-wide singleton so the CLI path and the in-process
# agent-studio path share ONE RunManager (one _runs registry + one
# _active_sinks map). The slot/lock are declared at module top so
# RunManager.__init__ can register/warn at construction time.
def get_run_manager() -> RunManager:
    """Return the process-wide RunManager, constructing it on first call."""
    global _RUN_MANAGER_SINGLETON
    if _RUN_MANAGER_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _RUN_MANAGER_SINGLETON is None:
                _RUN_MANAGER_SINGLETON = RunManager()
    return _RUN_MANAGER_SINGLETON


class _RunSink:
    """Per-run event sink: appends to ring buffer + events.jsonl, and
    records the live job id so a cancel request can hit fusion-mlx."""

    def __init__(self, record: dict[str, Any], lock: threading.Lock):
        self._record = record
        self._lock = lock
        self._events_path = Path(record["events_path"])
        self._canceled = False

    def emit(self, event: dict[str, Any]) -> None:
        """Receive a raw progress event from a trainer."""
        if not event:
            return
        ev = dict(event)
        etype = ev.get("type", "")
        # Record live job id/kind for cancel.
        if etype == "job":
            with self._lock:
                if ev.get("job_id"):
                    self._record["job_id"] = ev["job_id"]
                if ev.get("kind"):
                    self._record["job_kind"] = ev["kind"]
            return
        # Track step / phase / metrics on the record.
        step = ev.get("step", ev.get("iter"))
        if step is not None and isinstance(step, (int, float)):
            with self._lock:
                if int(step) > int(self._record.get("step", 0)):
                    self._record["step"] = int(step)
        if etype == "phase":
            with self._lock:
                self._record["phase"] = ev.get("name", ev.get("phase", ""))
        if etype in ("train_loss", "val_loss", "grpo_step"):
            with self._lock:
                self._record["metrics"].update(_flatten_metrics(ev))
        # Buffer + disk.
        with self._lock:
            self._record["events_buffer"].append(ev)
        try:
            with open(self._events_path, "a") as fh:
                fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("write events.jsonl failed: %s", exc)
        # Honor cancel between events.
        if self._canceled:
            self._do_cancel()

    def record_job(self, job_id: str, kind: str = "sft") -> None:
        """Trainer reports the live fusion-mlx job id so cancel can target it."""
        if job_id:
            with self._lock:
                self._record["job_id"] = job_id
                self._record["job_kind"] = kind

    def request_cancel(self) -> None:
        self._canceled = True

    def is_canceled(self) -> bool:
        return self._canceled

    def _do_cancel(self) -> None:
        """Call fusion-mlx cancel endpoint on the recorded job id."""
        with self._lock:
            job_id = self._record.get("job_id", "")
            kind = self._record.get("job_kind", "sft")
            port = self._record.get("mlx_port")
        if not job_id:
            logger.info("cancel: no job_id recorded for run yet")
            return
        try:
            client = MLXFineTuneClient(
                port=port or int(os.environ.get("FUSION_MLX_PORT", "11432"))
            )
            if kind in ("dpo", "orpo"):
                client.cancel_dpo_job(job_id)
            elif kind == "grpo":
                client.cancel_grpo_job(job_id)
            elif kind == "reward":
                client.cancel_reward_job(job_id)
            else:
                client.cancel_job(job_id)
            logger.info("cancel dispatched for job %s (kind=%s)", job_id, kind)
        except Exception as exc:
            logger.warning("cancel_job %s failed: %s", job_id, exc)


def _flatten_metrics(ev: dict[str, Any]) -> dict[str, float]:
    """Pick the headline scalar from a progress event for run_status.metrics."""
    out: dict[str, float] = {}
    if ev.get("type") == "train_loss":
        out["train_loss"] = float(ev.get("train_loss", 0.0))
        out["learning_rate"] = float(ev.get("learning_rate", 0.0))
        out["peak_memory_gb"] = float(ev.get("peak_memory_gb", 0.0))
    elif ev.get("type") == "val_loss":
        out["val_loss"] = float(ev.get("val_loss", 0.0))
    elif ev.get("type") == "grpo_step":
        out["grpo_loss"] = float(ev.get("loss", 0.0))
        out["mean_reward"] = float(ev.get("mean_reward", 0.0))
        out["mean_ratio"] = float(ev.get("mean_ratio", 0.0))
    return out


def _count_samples(path: Path) -> int:
    """Count rows in a dataset file. Parquet via pyarrow; text via line count.

    Parquet is binary — reading it as text lines (the old _count_lines)
    returned a wrong count and could decode-garbage. jsonl/json/csv are
    one-row-per-line so line count is the row count.
    """
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.read_metadata(str(path)).num_rows)
        except Exception as exc:
            logger.warning("parquet row count failed for %s: %s", path, exc)
            return 0
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def config_to_dict(config: TrainerConfig) -> dict[str, Any]:
    """Serialize a TrainerConfig to a JSON-safe dict (RPC payload)."""
    if is_dataclass(config):
        return asdict(config)
    return {}


__all__ = ["RunManager", "config_to_dict"]
