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
import contextlib
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
from .config import TrainerConfig, _default_mlx_port
from .mlx_client import MLXFineTuneClient

logger = logging.getLogger(__name__)

_TERMINAL = ("completed", "failed", "cancelled")
# R10: cap on in-memory (terminal) run records kept by a long-lived daemon.
# Beyond this the oldest terminal records are evicted from self._runs (full
# history remains on disk). Running runs are always kept regardless.
_MAX_INMEM_RUNS = 200
_MXF8_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/402"
# A3: module slot + lock declared early so RunManager.__init__ can register
# itself / warn on a bypassing direct construction. Filled in at module bottom
# for readability, but available at __init__ call time (module fully loaded).
_RUN_MANAGER_SINGLETON: RunManager | None = None
# RLock, not Lock: get_run_manager() holds this lock while calling
# RunManager(), whose __init__ re-acquires it (A3 singleton-bypass warning).
# A plain Lock self-deadlocks on that re-acquire — get_run_manager() (the
# supported singleton entry) hung forever while a direct RunManager() worked.
# RLock lets the same thread reenter, so construction completes.
_SINGLETON_LOCK = threading.RLock()


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
        self.mlx_port = mlx_port or _default_mlx_port()
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
        # R12: RLock (reentrant). _persist_meta takes this lock to snapshot a
        # record, then does the disk IO outside it — but several callers
        # (_start, _run_thread terminal paths, publish_run) already hold the
        # lock when they call _persist_meta. A plain Lock deadlocks on that
        # re-acquire; RLock lets the nested acquire reenter (same thread) so
        # the snapshot still runs under the caller's held lock and the IO
        # happens once the inner `with` returns (still held by the caller for
        # those paths, lock-free for direct _persist_meta callers).
        self._lock = threading.RLock()
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
        # R11: zombie-job cancels collected during _load_existing_runs are
        # dispatched on this background thread (not inline in __init__).
        self._pending_zombies: list[tuple[str, str, str, int | None]] = []
        self._load_existing_runs()
        if self._pending_zombies:
            t = threading.Thread(
                target=self._cancel_zombies, daemon=True, name="ft-zombie-cancel"
            )
            t.start()

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
                # R11: the cancel HTTP call (30s timeout × 5 retries) used to
                # run synchronously inside __init__ (under the singleton lock),
                # so a restart with N zombie runs blocked the first ft info /
                # get_run_manager() call for minutes. Collect zombies here and
                # cancel them on a background thread after construction; the
                # record is already marked failed synchronously so the GUI sees
                # the correct status immediately.
                zombie_job = data.get("job_id", "") or ""
                zombie_kind = data.get("job_kind", "sft") or "sft"
                if zombie_job:
                    self._pending_zombies.append(
                        (rid, zombie_job, zombie_kind, data.get("mlx_port"))
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
            "mlx_port": _default_mlx_port(),
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
            logger.warning(
                "reconstruct meta: read events %s failed: %s", events_path, exc
            )
            return None
        if not saw_event:
            return None
        return data

    # ------------------------------------------------------------------
    # info / presets / datasets / adapters
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """System info for the Settings/Dashboard panels.

        P3-3 (round-3 audit): builds a short-lived MLXFineTuneClient per call
        rather than sharing a daemon-wide one. info()/list_adapters() are read
        paths hit on GUI refresh; a shared pooled client would be contended by
        the worker threads driving active runs (create_job/wait_for_job) and
        risk serializing a multi-hour run behind a refresh. The per-call client
        is closed in finally (R7) so no pool leaks. Revisit only if profiling
        shows the connect cost on refresh is material.
        """
        mlx_alive = False
        models: list[dict[str, Any]] = []
        gpu_mem_gb: float | None = None
        try:
            client = MLXFineTuneClient(port=self.mlx_port)
            try:
                data = client.health()
                mlx_alive = True
                models = data.get("data", []) or []
            finally:
                # R7: close the pooled client so the daemon doesn't leak a
                # connection on every info() refresh.
                client.close()
        except Exception as exc:
            logger.info("info: fusion-mlx unreachable: %s", exc)

        # P1-4 (round-3 audit): gpu_mem_gb was hard-wired to None — the GUI
        # Dashboard showed a permanent blank. Best-effort fill via psutil's
        # physical memory (Apple Silicon unified memory = the pool MLX shares).
        # Optional dep: import-guarded so a missing psutil degrades to None.
        try:
            import psutil

            gpu_mem_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            gpu_mem_gb = None

        presets = self.list_presets()
        warnings = self._upstream_warnings()
        # P1-4: surface a loud warning when fusion-mlx is alive but loaded no
        # models — every SFT/RLSL run would fail to find a base model.
        if mlx_alive and not models:
            warnings.append(
                {
                    "id": "no_models",
                    "level": "warn",
                    "message": "fusion-mlx is running but no models are loaded; pull a model before training",
                }
            )
        return {
            "version": __version__,
            "mlx_port": self.mlx_port,
            "mlx_alive": mlx_alive,
            "models": models,
            "gpu_mem_gb": gpu_mem_gb,
            "model_hub_path": self.model_hub_path,
            "cache_path": self.cache_path,
            "presets": presets,
            "warnings": warnings,
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
        """Read first `limit` rows of a hub dataset by {category}/{name}.

        P2-3 (round-3 audit): n_total comes from _count_samples_cached (mtime-
        keyed) instead of a full-file line scan on every preview call — a GUI
        refresh over a 10M-row dataset no longer re-reads the whole file. Also
        now supports .parquet/.csv/.json preview, not just .jsonl.
        """
        limit = max(1, min(int(limit), 100))
        # P2 (round-3 audit): path-traversal guard. `name` is caller-supplied
        # (GUI/CLI) and was joined straight into the datasets path — a crafted
        # "../" name could read arbitrary files under model_hub_path. Resolve
        # the candidate and confirm it stays inside the datasets dir.
        datasets_dir = (Path(self.model_hub_path) / "datasets").resolve()
        if not datasets_dir.exists():
            return {
                "rows": [],
                "fields": [],
                "n_total": 0,
                "error": "datasets directory not found",
            }

        def _safe_join(base: Path, leaf: str) -> Path | None:
            # P2-3: try the known dataset suffixes, not just .jsonl, so a
            # parquet/csv dataset is previewable too.
            for suffix in (".jsonl", ".json", ".csv", ".parquet"):
                cand = (base / f"{leaf}{suffix}").resolve()
                try:
                    cand.relative_to(base)
                except ValueError:
                    logger.warning("preview_dataset: rejecting traversal name=%r", leaf)
                    return None
                if cand.exists():
                    return cand
            return None

        path = _safe_join(datasets_dir, name)
        if path is None or not path.exists():
            path = _safe_join(datasets_dir, name.split("/")[-1])
        if path is None or not path.exists():
            return {
                "rows": [],
                "fields": [],
                "n_total": 0,
                "error": f"dataset {name} not found",
            }

        n_total = self._count_samples_cached(path)
        rows, fields = _preview_rows(path, limit)
        return {"rows": rows, "fields": sorted(fields), "n_total": n_total}

    def list_adapters(self, model: str = "") -> dict[str, Any]:
        """Proxy fusion-mlx GET /admin/api/fine-tune/adapters.

        P2-6 (round-3 audit): on fusion-mlx down, the GUI saw an empty list
        and rendered "no adapters trained yet" — indistinguishable from a
        genuinely empty server. Return an explicit error carrier so the GUI
        can surface "fusion-mlx unreachable" instead of a misleading empty
        state. Callers that only want the list can read the ``items`` key.
        """
        try:
            client = MLXFineTuneClient(port=self.mlx_port)
            try:
                items = client.list_adapters(model_id=model)
                return {"items": items, "error": None}
            finally:
                # R7: close the pooled client (daemon leaks on every refresh).
                client.close()
        except Exception as exc:
            logger.error("list_adapters failed: %s", exc)
            return {"items": [], "error": f"fusion-mlx unreachable: {exc}"}

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
                # P1-3 (round-3 audit): propagate the sink cancel flag so the
                # pooled-client poll loops honor stop_run between polls.
                trainer.is_canceled = sink.is_canceled
                result = trainer.train()
            else:
                from .rlsl import RLSLEngine

                engine = RLSLEngine(config)
                engine.on_event = sink.emit
                engine.is_canceled = sink.is_canceled
                result = engine.run()
        except BaseException as exc:  # P1-3 (round-3 audit): catch
            # KeyboardInterrupt/SystemExit too — `except Exception` missed them,
            # so Ctrl-C / interpreter shutdown left the run stuck "running".
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
            from .hub_client import HubPublishError, publish_adapter_if_enabled

            # P1-2 (round-3 audit): publish_adapter_if_enabled raises
            # HubPublishError on a *known* hub failure (already handled →
            # returned as {published:False, error}). But any OTHER exception
            # (a bug in the publish path, an unexpected transport error) used
            # to propagate uncaught past this point and kill the worker thread
            # mid-run, leaving the run stuck "running" with no terminal status
            # ever emitted. Catch everything, log loud, still emit terminal.
            try:
                publish_result = publish_adapter_if_enabled(config, result)
            except HubPublishError as exc:
                logger.warning(
                    "run %s adapter publish failed (non-blocking): %s",
                    run_id,
                    exc,
                )
                publish_result = {"published": False, "error": str(exc)}
            except Exception as exc:
                logger.exception(
                    "run %s adapter publish raised unexpectedly (non-blocking)",
                    run_id,
                )
                publish_result = {"published": False, "error": str(exc)}
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
            # R10: self._runs was never pruned — a long-lived daemon accumulated
            # every run record + its events_buffer deque(maxlen=1000) forever,
            # so RSS climbed until restart. Keep the most recent MAX_INMEM_RUNS
            # records plus any still-running ones; drop old terminal records
            # (their full history stays on disk in events.jsonl/run_meta.json).
            all_items = sorted(
                self._runs.values(),
                key=lambda r: r.get("created", 0),
                reverse=True,
            )
            kept: list[dict[str, Any]] = []
            dropped = 0
            for r in all_items:
                if r.get("status") not in _TERMINAL or len(kept) < _MAX_INMEM_RUNS:
                    kept.append(r)
                else:
                    self._runs.pop(r.get("run_id", ""), None)
                    dropped += 1
            if dropped:
                logger.info("pruned %d terminal runs from in-memory registry", dropped)
            items = kept
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
            metrics = dict(r.get("metrics", {}))
            # O18 (round-3 audit): derive it_sec / progress_pct locally so a
            # status poll answers "how fast / how far" without re-parsing
            # events.jsonl. Deterministic from record state (Rule 5): no model
            # routing. it_sec = step / elapsed since created; progress_pct =
            # step / total_steps (guard zero divides).
            elapsed_s = time.time() - float(r.get("created", 0.0))
            step = int(r.get("step", 0))
            total_steps = int(r.get("total_steps", 0))
            if "it_sec" not in metrics and elapsed_s > 0 and step > 0:
                metrics["it_sec"] = round(step / elapsed_s, 4)
            if "elapsed_s" not in metrics and elapsed_s > 0:
                metrics["elapsed_s"] = round(elapsed_s, 2)
            if total_steps > 0:
                metrics["progress_pct"] = round(
                    min(step, total_steps) / total_steps * 100, 2
                )
            return {
                "run_id": run_id,
                "status": r.get("status", ""),
                "step": step,
                "total_steps": total_steps,
                "phase": r.get("phase", ""),
                "metrics": metrics,
                "adapter_name": r.get("adapter_name", ""),
                "final_adapter": r.get("final_adapter", ""),
                "error": r.get("error", ""),
                "created": r.get("created", 0),
                "job_id": r.get("job_id", ""),
                "events_path": r.get("events_path", ""),
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
        # R8: events without a step/iter key (status, job, initial-running) are
        # control-plane events. The old filter `ev.get("step", ev.get("iter",-1))
        # > since_step` dropped ALL of them — including the terminal
        # {"type":"status","status":"completed"} — so the GUI polling
        # run_progress never saw completion and stayed "running" forever. Such
        # events are always passed through; only step-bearing metric events are
        # gated by since_step.
        filtered = []
        for ev in events:
            step = ev.get("step")
            if step is None:
                step = ev.get("iter")
            if step is None or step > since_step:
                filtered.append(ev)
        return {"run_id": run_id, "events": filtered, "status": status}

    def stop_run(
        self, run_id: str, wait: bool = False, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Request cancellation; returns {status: cancelling}.

        R1: sets the cancel flag AND dispatches the upstream fusion-mlx cancel
        IMMEDIATELY (cancel_now) rather than waiting for the next trainer
        event. A job blocked in a long silent eval or a long poll would
        otherwise keep burning GPU for minutes until the next event trickles
        in. If the job_id isn't recorded yet, the sink retries on the next
        emit (see _do_cancel).

        O19 (round-3 audit): wait=True blocks until the run worker reaches a
        terminal status (joins its thread with a timeout), then returns the
        final status instead of fire-and-forget "cancelling". Callers that
        need to chain on actual termination (e.g. CLI `&&`, GUI "stop then
        release model") pass wait=True. The default stays fire-and-forget so
        long GUI cancel flows don't block the RPC thread.
        """
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return {"run_id": run_id, "status": "unknown", "error": "no such run"}
            if r.get("status") in _TERMINAL:
                return {"run_id": run_id, "status": r["status"]}
            r["canceled"] = True
            thread = r.get("thread")
        sink = self._active_sinks.get(run_id)
        if sink is not None:
            sink.cancel_now()
        logger.info("stop_run requested for %s (active cancel dispatched)", run_id)
        if not wait:
            return {"run_id": run_id, "status": "cancelling"}
        # O19: block on the worker thread reaching terminal. Non-daemon thread
        # so join() is safe; timeout keeps a hung upstream from wedging the
        # caller forever.
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            r = self._runs.get(run_id)
            status = r.get("status", "cancelling") if r else "cancelling"
        logger.info("stop_run wait settled for %s -> %s", run_id, status)
        return {"run_id": run_id, "status": status}

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
        # fusion-mlx GET /admin/api/fine-tune/adapters returns each adapter as
        # {model_id, adapter_name, adapter_path, has_weights, has_config} — the
        # identifier key is "adapter_name", NOT "name". Reading a.get("name")
        # always returned "" so every publish_run falsely concluded the adapter
        # was missing and refused to publish (L4 existence check defeated by an
        # upstream field name the mock-based suite never verified).
        adapter_exists = any(
            str(a.get("adapter_name", "") or a.get("name", "")) == adapter_name
            for a in adapters
        )
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
        from .config import URLGuardError
        from .hub_client import HubPublishError, ModelHubClient

        url = hub_url or os.environ.get(
            "FUSION_MODEL_HUB_URL", "http://localhost:11432"
        )
        key = api_key or os.environ.get("FUSION_MODEL_HUB_API_KEY", "")
        try:
            client = ModelHubClient(base_url=url, api_key=key)
        except URLGuardError as exc:
            logger.warning(
                "publish_run %s rejected hub_url (non-blocking): %s", run_id, exc
            )
            return {"run_id": run_id, "published": False, "error": str(exc)}
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
        # R12: snapshot the record under the lock, then do the synchronous disk
        # IO (write_text + os.replace) OUTSIDE the lock. The old code held
        # self._lock across the fsync+rename, which on NFS/SMB cache_path
        # blocked every concurrent list_runs/run_status/run_progress/stop_run
        # and every sink emit across all runs for the duration of each persist.
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                return
            meta_path = Path(r["run_dir"]) / "run_meta.json"
            serial = {
                k: v for k, v in r.items() if k not in ("events_buffer", "thread")
            }
        # M2: atomic write — write to a temp file in the same dir, fsync, then
        # os.replace so a crash mid-write never leaves a half-written meta
        # (which would make events.jsonl and run_meta.json disagree on status).
        # P3-1 (round-3 audit): fsync the tmp file contents before replace, and
        # clean up the residual tmp on every path (the old success path left
        # .json.tmp around because os.replace moves it — but a crashed peer that
        # died after write_text left a stale tmp that this run never cleared).
        tmp_path = meta_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(serial, default=str, ensure_ascii=False, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, meta_path)
        except Exception as exc:
            logger.warning("persist meta %s failed: %s", run_id, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def _cancel_zombies(self) -> None:
        """R11: cancel upstream fusion-mlx jobs left running by a dead process,
        off the construction path so get_run_manager() / ft info don't block.
        Each cancel is best-effort + fail-visible; a transient fusion-mlx
        outage surfaces the job_id so the operator can stop it manually."""
        while True:
            try:
                rid, job_id, kind, mlx_port = self._pending_zombies.pop()
            except IndexError:
                return
            try:
                client = MLXFineTuneClient(
                    port=int(mlx_port or os.environ.get("FUSION_MLX_PORT", "11434"))
                )
                if kind in ("dpo", "orpo"):
                    client.cancel_dpo_job(job_id)
                elif kind == "grpo":
                    client.cancel_grpo_job(job_id)
                elif kind == "reward":
                    client.cancel_reward_job(job_id)
                else:
                    client.cancel_job(job_id)
                client.close()
                logger.info(
                    "zombie run %s: cancelled upstream job %s (kind=%s)",
                    rid,
                    job_id,
                    kind,
                )
            except Exception as exc:
                logger.warning(
                    "zombie run %s: could not cancel upstream job %s (kind=%s) "
                    "— it may STILL be running on fusion-mlx. Stop manually. "
                    "error=%s",
                    rid,
                    job_id,
                    kind,
                    exc,
                )

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
        except Exception as exc:
            # P2 (round-3 audit): was bare `except: pass` — a persist failure
            # at exit was swallowed with zero trace. Log loud (fail-visible);
            # atexit can't propagate, but the operator gets the reason.
            logger.warning("atexit persist-all failed: %s", exc)


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

    def __init__(self, record: dict[str, Any], lock: threading.RLock):
        self._record = record
        self._lock = lock
        self._events_path = Path(record["events_path"])
        # R2: a threading.Event replaces the plain bool. set()/is_set() are
        # internally synchronized, eliminating the read→act TOCTOU that let a
        # cancel request race between an emit check and the next event.
        self._cancel_event = threading.Event()
        # R1: guard so we only DISPATCH one upstream cancel per run, even if
        # stop_run + a late emit both try. The Event tracks intent; this flag
        # tracks the dispatched action.
        self._cancel_dispatched = False

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
                # P2-11 (round-3 audit): total_steps was always 0 — the job
                # event carries the planned iteration count; record it so the
                # GUI progress bar has a denominator.
                iters = ev.get("iters")
                if iters is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        self._record["total_steps"] = int(iters)
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
            # P2 (round-3 audit): serialize the append under the same lock and
            # fsync so concurrent emitters never interleave half-lines and a
            # crash leaves complete records on disk (atomic-ish per event).
            try:
                with open(self._events_path, "a") as fh:
                    fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception as exc:
                logger.warning("write events.jsonl failed: %s", exc)
        # Honor cancel between events.
        if self._cancel_event.is_set():
            self._do_cancel()

    def record_job(self, job_id: str, kind: str = "sft") -> None:
        """Trainer reports the live fusion-mlx job id so cancel can target it."""
        if job_id:
            with self._lock:
                self._record["job_id"] = job_id
                self._record["job_kind"] = kind

    def request_cancel(self) -> None:
        # R2: Event.set is atomic and visible to all threads; no lock needed.
        self._cancel_event.set()

    def is_canceled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel_now(self) -> None:
        """R1: dispatch the upstream cancel IMMEDIATELY rather than waiting
        for the next trainer event. Called by stop_run so a job blocked in a
        long silent eval / long poll is cancelled within seconds, not minutes.
        Idempotent: the _cancel_dispatched guard prevents a double POST.
        """
        self._cancel_event.set()
        self._do_cancel()

    def _do_cancel(self) -> None:
        """Call fusion-mlx cancel endpoint on the recorded job id."""
        with self._lock:
            job_id = self._record.get("job_id", "")
            kind = self._record.get("job_kind", "sft")
            port = self._record.get("mlx_port")
        if not job_id:
            # No job submitted yet — can't cancel upstream. Leave
            # _cancel_dispatched False so a later emit (once the job_id lands)
            # retries the cancel. The Event stays set so is_canceled() is True.
            logger.info(
                "cancel: no job_id recorded for run yet; will retry on next event"
            )
            return
        with self._lock:
            if self._cancel_dispatched:
                # R1: already dispatched an upstream cancel for this run; a
                # second emit/stop would double-POST. The Event still holds so
                # the trainer's is_canceled() keeps returning True.
                return
            self._cancel_dispatched = True
        try:
            client = MLXFineTuneClient(port=port or _default_mlx_port())
            try:
                if kind in ("dpo", "orpo"):
                    client.cancel_dpo_job(job_id)
                elif kind == "grpo":
                    client.cancel_grpo_job(job_id)
                elif kind == "reward":
                    client.cancel_reward_job(job_id)
                else:
                    client.cancel_job(job_id)
                logger.info("cancel dispatched for job %s (kind=%s)", job_id, kind)
            finally:
                # R7: close the pooled client after cancel.
                client.close()
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
    # O18 (round-3 audit): pass through any upstream throughput fields so the
    # GUI does not have to re-parse events.jsonl for them. Names are
    # best-effort — fusion-mlx may emit any subset; missing keys are skipped.
    for raw_key, out_key in (
        ("it_per_sec", "it_per_sec"),
        ("it_sec", "it_sec"),
        ("tokens_per_sec", "tokens_per_sec"),
        ("tokens_per_s", "tokens_per_sec"),
        ("throughput", "throughput"),
        ("elapsed", "elapsed_s"),
        ("elapsed_time", "elapsed_s"),
    ):
        val = ev.get(raw_key)
        if val is not None:
            with contextlib.suppress(TypeError, ValueError):
                out[out_key] = float(val)
    return out


def _preview_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], set[str]]:
    """Read first `limit` parsed rows + their field set. Format-aware so a
    parquet/csv dataset previews the same as jsonl (P2-3)."""
    rows: list[dict[str, Any]] = []
    fields: set[str] = set()
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            batch = next(
                iter(pq.ParquetFile(path).iter_batches(batch_size=limit)), None
            )
            if batch is not None:
                for obj in batch.to_pylist():
                    if len(rows) >= limit:
                        break
                    if isinstance(obj, dict):
                        rows.append(obj)
                        fields.update(obj.keys())
        elif suffix == ".json":
            import json as _json

            with open(path, encoding="utf-8") as fh:
                content = _json.load(fh)
            if isinstance(content, list):
                for obj in content[:limit]:
                    if isinstance(obj, dict):
                        rows.append(obj)
                        fields.update(obj.keys())
            elif isinstance(content, dict):
                rows.append(content)
                fields.update(content.keys())
        else:
            # jsonl / csv / text — one row per line (csv parsed by csv module).
            if suffix == ".csv":
                import csv as _csv

                with open(path, encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        if len(rows) >= limit:
                            break
                        rows.append(dict(row))
                        fields.update(row.keys())
            else:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if len(rows) >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            rows.append(obj)
                            fields.update(obj.keys())
    except Exception as exc:
        logger.warning("preview_rows failed for %s: %s", path, exc)
    return rows, fields


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
