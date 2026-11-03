"""trainer-bridge — thin FastAPI service around the fusion_trainer library.

The Trainer console GUI (bridge/trainer.html, served at /) proxies to this
service; the bridge owns the fusion-mlx admin key and the RunManager
lifecycle — the browser never sees the admin key. Runs are background
threads in THIS process, so the bridge must stay up for the duration of
a training (console restarts do not affect it).

Auth: bearer token — the static FUSION_TRAINER_BRIDGE_KEY for direct
loopback access, plus the gateway-issued fm- module key (issue #72) for
requests the fusion-gateway /shell proxy forwards. Loopback bind only;
the gateway proxy at /shell/fusion-trainer/ is the single user-facing
entry when shell registration is enabled.

Env:
  FUSION_TRAINER_BRIDGE_KEY   required bearer token for console -> bridge
  FUSION_MLX_ADMIN_TOKEN      fusion-mlx admin key (bridge -> fusion-mlx)
  FUSION_TRAINER_MLX_PORT     fusion-mlx port, default 11434 (engine direct)
  FUSION_TRAINER_BRIDGE_PORT  default 8790
  FUSION_MLX_MODEL            default base model alias
  FUSION_TRAINER_BRIDGE_PRESETS  preset registry JSON (editable without release)
  FUSION_SHELL_GATEWAY_URL    gateway base — enables shell registration (#72)
  FUSION_ADMIN_KEY            gateway key authorized for /api/v1/shell/*
                              (fusion_core.config.admin_key, #80)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# fusion_trainer is importable because the bridge runs inside the
# fusion-trainer repo's venv (scripts/trainer-bridge.sh guarantees it).
from bridge import shell_registry  # noqa: E402
from fusion_trainer import RunManager, TrainerConfig  # noqa: E402
from fusion_trainer.dataset import formats as ds_formats  # noqa: E402
from fusion_trainer.merge import merge_and_export  # noqa: E402
from fusion_trainer.mlx_client import MLXFineTuneClient  # noqa: E402
from fusion_trainer.run_manager import get_run_manager  # noqa: E402
from fusion_trainer.sft.presets import SFT_PRESETS  # noqa: E402

BRIDGE_PORT = int(os.environ.get("FUSION_TRAINER_BRIDGE_PORT", "8790"))
# fusion-mlx's fine-tune admin API is served DIRECTLY on the engine port;
# the gateway (11432) does not proxy /admin — so the bridge uses its OWN
# env var name and never inherits an ambient FUSION_MLX_PORT (which points
# at the gateway in console shells).
MLX_PORT = int(os.environ.get("FUSION_TRAINER_MLX_PORT", "11434"))
# Console -> bridge auth: the gateway client key (config-conventions §5.1 —
# the only credential a business process holds); the legacy
# FUSION_TRAINER_BRIDGE_KEY is accepted as a transition fallback.
try:
    from fusion_core.config import client_key as _client_key
except ImportError:  # pragma: no cover

    def _client_key() -> str:
        return ""


BRIDGE_KEY = os.environ.get("FUSION_TRAINER_BRIDGE_KEY", "") or _client_key()

# Default tuning target: the assistant base model. The bridge stays
# generic — any fusion-mlx model alias is accepted per request.
DEFAULT_BASE_MODEL = os.environ.get(
    "FUSION_MLX_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"
)

CACHE_DIR = Path(
    os.environ.get("FUSION_TRAINER_CACHE", str(Path.home() / ".fusion-trainer-bridge"))
)
DATASETS_DIR = CACHE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
# Merged models must live under fusion-mlx's allowed model dirs
# (/v1/merge-adapter rejects paths outside them) — the standard models dir
# also gets the merged model discovered and servable by alias for free,
# i.e. published tuned models auto-enter the /v1/models catalog.
MERGED_DIR = Path.home() / ".fusion-mlx" / "models"
MERGED_DIR.mkdir(parents=True, exist_ok=True)
# Published-tuned-model state (one record; empty file = nothing published).
PUBLISHED_PATH = CACHE_DIR / "published.json"
# Preset registry — a plain JSON file so operators can add/edit presets
# without cutting a release. Seeded from SFT_PRESETS on first start;
# re-read on every request so edits apply live.
PRESETS_FILE = Path(
    os.environ.get("FUSION_TRAINER_BRIDGE_PRESETS", str(CACHE_DIR / "presets.json"))
)

# One active training at a time — this Mac trains models for the console
# while also serving chat; two concurrent LoRA runs would thrash memory.
MAX_ACTIVE_RUNS = 1
MAX_DATASET_BYTES = 50 * 1024 * 1024  # 50 MB per upload
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# CSV pair upload → chat JSONL conversion
_PAIR_COLUMNS = ("instruction", "input", "output")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: Any):
    # Issue #72: register the console as a fusion-gateway shell module and
    # keep it heartbeating, so the shell sidebar lists 训练 and the gateway
    # proxy (/shell/fusion-trainer/) is the user-facing entry. Env-gated —
    # never fatal to the bridge itself.
    task = None
    if shell_registry.shell_registration_enabled():
        task = asyncio.create_task(shell_registry.shell_registry_loop())
        logger.info("shell registration loop started (module fusion-trainer)")
    else:
        logger.info(
            "shell registration disabled — set FUSION_SHELL_GATEWAY_URL + "
            "FUSION_ADMIN_KEY to list the console in the shell sidebar"
        )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            logger.info("shell registration loop stopped")


app = FastAPI(title="fusion-trainer bridge", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loopback-only service behind token
    allow_methods=["*"],
    allow_headers=["*"],
)

_run_manager: RunManager | None = None


def _rm() -> RunManager:
    global _run_manager
    if _run_manager is None:
        _run_manager = get_run_manager()
    return _run_manager


def _mlx() -> MLXFineTuneClient:
    # source="model-hub": merge/serve are model-management endpoints gated
    # on X-Fusion-Source (fusion-mlx auth.py:531) — the bridge manages
    # models on the store's behalf.
    return MLXFineTuneClient(port=MLX_PORT, source="model-hub")


def _auth(authorization: str) -> None:
    """Accept the static bridge key OR the gateway-issued fm- module key
    (issue #72 — the gateway /shell proxy injects the fm- key as Bearer).
    Constant-time compare per candidate; fail closed when no key exists."""
    keys = shell_registry.accepted_bridge_keys()
    if not keys:
        raise HTTPException(
            status_code=503,
            detail="bridge key not configured — set FUSION_TRAINER_BRIDGE_KEY",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not any(secrets.compare_digest(token, k) for k in keys):
        raise HTTPException(status_code=401, detail="invalid bridge token")


async def _require_auth(
    authorization: str = Header(default=""),
) -> None:
    _auth(authorization)


# ── models ──────────────────────────────────────────────────────────────


class DatasetUpload(BaseModel):
    name: str
    content: str  # raw JSONL or CSV text
    fmt: str = "auto"  # auto | jsonl | csv


class RunStart(BaseModel):
    name: str
    dataset: str  # dataset name as registered by /datasets
    model: str = ""  # default DEFAULT_BASE_MODEL
    preset: str = ""  # SFT_PRESETS key; empty = 4B defaults
    lora_rank: int | None = None
    learning_rate: float | None = None
    num_epochs: int | None = None
    memory_limit_gb: float | None = None


class MergeRequest(BaseModel):
    quantize: str = ""  # "" | 4bit | 8bit
    validation_prompt: str = ""


# ── helpers ─────────────────────────────────────────────────────────────


def _dataset_path(name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise HTTPException(422, "dataset name must match [A-Za-z0-9_-]")
    return DATASETS_DIR / f"{name}.jsonl"


def _csv_to_messages(content: str) -> list[dict[str, str]]:
    """Convert instruction,input,output CSV rows into chat-JSONL dicts."""
    import csv
    import io

    rows: list[dict[str, str]] = []
    reader = csv.DictReader(io.StringIO(content))
    missing = [c for c in _PAIR_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise HTTPException(
            422, f"CSV must have columns {_PAIR_COLUMNS}; missing: {missing}"
        )
    for i, row in enumerate(reader, start=2):  # line 1 = header
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if not instruction or not out:
            raise HTTPException(
                422, f"line {i}: 'instruction' and 'output' are required"
            )
        user = f"{instruction}\n\n{inp}" if inp else instruction
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": out},
                ],
            }
        )
    if not rows:
        raise HTTPException(422, "CSV contains no data rows")
    return rows


def _validate_and_store(name: str, content: str, fmt: str) -> dict[str, Any]:
    if len(content.encode("utf-8")) > MAX_DATASET_BYTES:
        raise HTTPException(413, "dataset exceeds the 50 MB upload cap")
    path = _dataset_path(name)

    if fmt in ("", "auto"):
        fmt = (
            "csv"
            if content.lstrip().startswith(("instruction,", '"instruction"'))
            else "jsonl"
        )

    if fmt == "csv":
        rows = _csv_to_messages(content)
    else:
        rows = []
        for i, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            import json

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise HTTPException(422, f"line {i}: invalid JSON ({e})")

    if not rows:
        raise HTTPException(422, "dataset contains no rows")
    # detect_dataset_schema works on a single sample dict (formats.py:70)
    schema = ds_formats.detect_dataset_schema(rows[0])
    if not schema:
        raise HTTPException(
            422,
            "cannot detect a supported schema (messages / sharegpt / "
            "dpo_pair / grpo_prompt / input_output)",
        )
    # validate_format returns (ok, message) (formats.py:95)
    ok, message = ds_formats.validate_format(rows, str(schema))
    if not ok:
        raise HTTPException(422, {"schema": str(schema), "error": message})

    import json

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "path": str(path),
        "schema": str(schema),
        "samples": len(rows),
        "bytes": path.stat().st_size,
    }


def _build_config(req: RunStart) -> TrainerConfig:
    dpath = _dataset_path(req.dataset)
    if not dpath.exists():
        raise HTTPException(404, f"dataset '{req.dataset}' not found")

    cfg = TrainerConfig()
    # Never inherit ambient FUSION_MLX_PORT (the gateway) — training jobs
    # must reach the engine's admin API directly.
    cfg.mlx_port = MLX_PORT
    cfg.sft.model = req.model or DEFAULT_BASE_MODEL
    cfg.sft.run_name = req.name
    cfg.sft.method = "lora"
    # 4B defaults; a preset bundle overrides them wholesale.
    cfg.sft.lora_rank = req.lora_rank or 16
    cfg.sft.learning_rate = req.learning_rate or 2e-4
    cfg.sft.num_epochs = req.num_epochs or 3
    cfg.dataset.path = str(dpath)
    cfg.dataset.max_seq_length = 2048

    if req.preset:
        registry = _load_presets()
        if req.preset not in registry:
            raise HTTPException(
                422,
                f"unknown preset '{req.preset}' "
                f"(available: {', '.join(sorted(registry))})",
            )
        preset = registry[req.preset]
        for key, value in preset.items():  # hyperparam bundle → SFTConfig
            if hasattr(cfg.sft, key) and key != "model":
                setattr(cfg.sft, key, value)

    # request-level overrides beat the preset
    if req.lora_rank:
        cfg.sft.lora_rank = req.lora_rank
    if req.learning_rate:
        cfg.sft.learning_rate = req.learning_rate
    if req.num_epochs:
        cfg.sft.num_epochs = req.num_epochs
    # Training-memory budget: fusion-mlx's admission gate projects
    # resident usage + job budget against the physical safety line. Its
    # default budget (60% of physical) assumes a dedicated training box —
    # wrong here, where 27B+4B serving models stay resident (~37GB) and a
    # 0.6B smoke run got rejected with "projected 114.1GB". Compute a
    # budget that fits: safety line − resident − headroom.
    if req.memory_limit_gb:
        cfg.sft.memory_limit_gb = float(req.memory_limit_gb)
    elif cfg.sft.memory_limit_gb <= 0:
        import psutil

        vm = psutil.virtual_memory()
        line_gb = vm.total / 2**30 * 0.85
        budget_gb = line_gb - vm.used / 2**30 - 6.0
        cfg.sft.memory_limit_gb = round(max(8.0, budget_gb), 1)
    return cfg


def _active_run_count() -> int:
    active = 0
    for run in _rm().list_runs(limit=50):
        if run.get("status") in ("running", "pending"):
            active += 1
    return active


# ── preset registry ─────────────────────────────────────────────────────


def _seed_presets() -> None:
    """First start: write the in-code presets to the registry file."""
    import json

    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(
        json.dumps(SFT_PRESETS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_presets() -> dict[str, Any]:
    """Preset registry = the JSON file, re-read per request (live edits).

    Falls back to in-code SFT_PRESETS if the file is missing or corrupt —
    a broken registry edit must not take the bridge down (fail-visible in
    the log, not fail-closed).
    """
    import json

    if not PRESETS_FILE.exists():
        try:
            _seed_presets()
        except OSError as e:
            logging.getLogger(__name__).warning("preset seed failed: %s", e)
            return dict(SFT_PRESETS)
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
        logging.getLogger(__name__).warning(
            "preset registry %s empty/invalid — using built-ins", PRESETS_FILE
        )
        return dict(SFT_PRESETS)
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger(__name__).warning(
            "preset registry unreadable (%s) — using built-ins", e
        )
        return dict(SFT_PRESETS)


# ── published tuned-model state ─────────────────────────────────────────


def _read_published() -> dict[str, Any]:
    import json

    if not PUBLISHED_PATH.exists():
        return {}
    try:
        data = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger(__name__).warning("published state unreadable: %s", e)
        return {}


def _write_published(record: dict[str, Any]) -> None:
    import json

    PUBLISHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PUBLISHED_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(PUBLISHED_PATH)  # atomic — never half-written state


# ── endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    # trainer health() probes GET /v1/models and returns the raw dict
    # (mlx_client.py:227-235) — {"object": "list", "data": [...]}
    mlx_ok, mlx_info = False, ""
    client = _mlx()
    try:
        info = client.health()
        models = info.get("data") if isinstance(info, dict) else None
        mlx_ok = isinstance(models, list)
        mlx_info = f"{len(models)} model(s) registered" if mlx_ok else str(info)[:200]
    except Exception as e:  # health must never raise
        mlx_info = f"probe failed: {e}"
    finally:
        client.close()
    return {
        "ok": mlx_ok,
        "bridge_port": BRIDGE_PORT,
        "mlx_port": MLX_PORT,
        "mlx": mlx_info,
        "presets": len(_load_presets()),
        "shell_registered": shell_registry.shell_registration_enabled(),
    }


@app.get("/presets", dependencies=[Depends(_require_auth)])
async def presets() -> dict[str, Any]:
    registry = _load_presets()
    return {
        "presets": sorted(registry.keys()),
        "registry_file": str(PRESETS_FILE),
        "default_base_model": DEFAULT_BASE_MODEL,
    }


@app.get("/catalog", dependencies=[Depends(_require_auth)])
async def catalog() -> dict[str, Any]:
    logger.info("GET /catalog: listing fusion-mlx model catalog")
    client = _mlx()
    try:
        info = client.health()
    except Exception as e:
        logger.warning(f"GET /catalog: fusion-mlx unreachable: {e}")
        raise HTTPException(502, f"fusion-mlx unreachable: {e}")
    finally:
        client.close()
    data = info.get("data") if isinstance(info, dict) else None
    if not isinstance(data, list):
        logger.warning(f"GET /catalog: unexpected /v1/models reply: {str(info)[:200]}")
        raise HTTPException(502, f"unexpected /v1/models reply: {str(info)[:200]}")
    ids = sorted(
        str(m.get("id", "")) for m in data if isinstance(m, dict) and m.get("id")
    )
    logger.info(f"GET /catalog: {len(ids)} models in catalog")
    return {"models": ids, "default_base_model": DEFAULT_BASE_MODEL}


@app.post("/datasets", dependencies=[Depends(_require_auth)])
async def upload_dataset(body: DatasetUpload) -> dict[str, Any]:
    return _validate_and_store(body.name, body.content, body.fmt)


@app.get("/datasets", dependencies=[Depends(_require_auth)])
async def list_datasets() -> dict[str, Any]:
    items = []
    for p in sorted(DATASETS_DIR.glob("*.jsonl")):
        try:
            lines = sum(1 for _ in p.open(encoding="utf-8"))
            items.append({"name": p.stem, "samples": lines, "bytes": p.stat().st_size})
        except OSError:
            continue
    return {"datasets": items}


@app.delete("/datasets/{name}", dependencies=[Depends(_require_auth)])
async def delete_dataset(name: str) -> dict[str, Any]:
    path = _dataset_path(name)
    if not path.exists():
        raise HTTPException(404, "dataset not found")
    path.unlink()
    return {"deleted": name}


@app.post("/runs", dependencies=[Depends(_require_auth)])
async def start_run(req: RunStart) -> dict[str, Any]:
    if _active_run_count() >= MAX_ACTIVE_RUNS:
        raise HTTPException(
            409,
            "another training run is active — stop it or wait "
            "(one active run per machine)",
        )
    if not _NAME_RE.match(req.name or ""):
        raise HTTPException(422, "run name must match [A-Za-z0-9_-]")
    cfg = _build_config(req)
    result = _rm().start_sft(cfg)
    return {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "model": cfg.sft.model,
        "dataset": req.dataset,
    }


@app.get("/runs", dependencies=[Depends(_require_auth)])
async def list_runs(limit: int = 20) -> dict[str, Any]:
    return {"runs": _rm().list_runs(limit=min(limit, 50))}


@app.get("/runs/{run_id}", dependencies=[Depends(_require_auth)])
async def run_detail(run_id: str) -> dict[str, Any]:
    status = _rm().run_status(run_id)
    if status.get("status") == "unknown":
        raise HTTPException(404, "no such run")
    return status


@app.post("/runs/{run_id}/stop", dependencies=[Depends(_require_auth)])
async def stop_run(run_id: str) -> dict[str, Any]:
    try:
        result = _rm().stop_run(run_id, wait=False)
    except KeyError:
        raise HTTPException(404, "no such run")
    return result


@app.post("/runs/{run_id}/merge", dependencies=[Depends(_require_auth)])
async def merge_run(run_id: str, req: MergeRequest) -> dict[str, Any]:
    status = _rm().run_status(run_id)
    if status.get("status") != "completed":
        raise HTTPException(
            409, f"run is '{status.get('status')}', merge needs 'completed'"
        )
    adapter = status.get("final_adapter") or ""
    if not adapter:
        raise HTTPException(409, "run has no final adapter path")
    base_model = status.get("model") or DEFAULT_BASE_MODEL
    # The fine-tune registry lists ids in dash form (mlx-community--X);
    # /v1/merge-adapter resolves HF-style slash ids (mlx-community/X).
    if "/" not in base_model and "--" in base_model:
        base_model = base_model.replace("--", "/", 1)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = MERGED_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', run_id)}-{stamp}"
    client = _mlx()
    try:
        result = merge_and_export(
            client=client,
            base_model=base_model,
            adapter_path=adapter,
            output_dir=str(out),
            quantize=req.quantize,
            validation_prompt=req.validation_prompt or "",
        )
    except Exception as e:
        raise HTTPException(502, f"merge failed: {e}")
    finally:
        client.close()
    return {
        "model_id": out.name,
        "final_model_path": result.get("final_model_path", str(out)),
        "merge": result.get("merge"),
        "quantize": result.get("quantize"),
        "validation": result.get("validation"),
    }


@app.post("/models/serve", dependencies=[Depends(_require_auth)])
async def serve_model(body: dict[str, Any]) -> dict[str, Any]:
    base_model = str(body.get("base_model") or "")
    adapter_name = str(body.get("adapter_name") or "")
    if not base_model or not adapter_name:
        raise HTTPException(422, "base_model and adapter_name are required")
    # Hot-mount the LoRA adapter on its base (verified on fusion-mlx
    # 0.10.x): returns a composite chat id "<base>::lora::<adapter_path>"
    # the gateway accepts on /v1/chat/completions. No merge needed; the
    # base model keeps serving unchanged for everyone else.
    client = _mlx()
    try:
        result = client.serve_adapter(base_model, adapter_name)
    except Exception as e:
        raise HTTPException(502, f"fusion-mlx serve failed: {e}")
    finally:
        client.close()
    return {
        "served_model_id": result.get("served_model_id", ""),
        "base_model_id": result.get("base_model_id", base_model),
        "adapter_name": adapter_name,
    }


@app.get("/models", dependencies=[Depends(_require_auth)])
async def models() -> dict[str, Any]:
    client = _mlx()
    try:
        adapters = client.list_adapters()
    except Exception as e:
        raise HTTPException(502, f"fusion-mlx unreachable: {e}")
    finally:
        client.close()
    return {"adapters": adapters}


# ── publish / published state ───────────────────────────────────────────


class PublishRequest(BaseModel):
    run_id: str = ""  # empty = revert to base model


@app.post("/publish", dependencies=[Depends(_require_auth)])
async def publish(req: PublishRequest) -> dict[str, Any]:
    if not req.run_id:
        # Revert: clear the published slot; the assistant falls back to
        # the base model. Merged models already in the catalog stay there.
        _write_published({})
        return {"published": False, "reverted": True}
    status = _rm().run_status(req.run_id)
    if status.get("status") != "completed":
        raise HTTPException(
            409, f"run is '{status.get('status')}', publish needs 'completed'"
        )
    # Publish = promote the run's MERGED standalone model (the merge step
    # already wrote it into fusion-mlx's models dir, so it auto-enters the
    # /v1/models catalog). Hot-mounting the raw adapter is not used: the
    # engine's adapter registry is 2-level (adapters/<model>/<name>) and a
    # slash model id (mlx-community/X) nests one level deeper, leaving the
    # adapter invisible to serve (fusion-mlx#1317).
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", req.run_id)
    merged = sorted(
        (
            p
            for p in MERGED_DIR.iterdir()
            if p.is_dir() and p.name.startswith(f"{prefix}-")
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not merged:
        raise HTTPException(409, "run has no merged model — run Merge first")
    model_id = merged[-1].name
    record = {
        "run_id": req.run_id,
        "model_id": model_id,
        "final_model_path": str(merged[-1]),
        "base_model": status.get("model") or DEFAULT_BASE_MODEL,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_published(record)
    logging.getLogger(__name__).info("published run %s: %s", req.run_id, model_id)
    return {"published": True, "record": record}


@app.get("/published", dependencies=[Depends(_require_auth)])
async def published() -> dict[str, Any]:
    return _read_published()


# ── evaluation: base vs tuned, same instruction ─────────────────────────


class EvalRequest(BaseModel):
    instruction: str
    max_tokens: int = 512


@app.post("/eval", dependencies=[Depends(_require_auth)])
async def evaluate(req: EvalRequest) -> dict[str, Any]:
    instruction = (req.instruction or "").strip()
    if not instruction:
        raise HTTPException(422, "instruction is required")
    rec = _read_published()
    base_model = rec.get("base_model") or DEFAULT_BASE_MODEL
    tuned_model = rec.get("model_id") or ""
    client = _mlx()
    try:
        messages = [{"role": "user", "content": instruction}]
        base_answer = client.chat_completions(
            base_model, messages, max_tokens=req.max_tokens
        )
        tuned_answer: list[str] = []
        if tuned_model:
            tuned_answer = client.chat_completions(
                tuned_model, messages, max_tokens=req.max_tokens
            )
    except Exception as e:
        raise HTTPException(502, f"fusion-mlx generation failed: {e}")
    finally:
        client.close()
    return {
        "base_model": base_model,
        "tuned_model": tuned_model,
        "base_answer": base_answer[0] if base_answer else "",
        "tuned_answer": tuned_answer[0] if tuned_answer else "",
    }


# ── console GUI (static page) ───────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def index():
    from fastapi.responses import FileResponse

    page = Path(__file__).resolve().parent / "trainer.html"
    if not page.exists():
        raise HTTPException(404, "trainer.html missing from bridge package")
    return FileResponse(page)


if __name__ == "__main__":
    import uvicorn

    # Show the product name in ps/Activity Monitor instead of bare "Python"
    # (#77). setproctitle is the only reliable way on macOS homebrew Python
    # (exec -a is overwritten by the Python.app argv). Optional dep: keep
    # running with the default title if it is missing.
    try:
        import setproctitle

        setproctitle.setproctitle(f"fusion-trainer bridge :{BRIDGE_PORT}")
    except Exception as exc:
        logging.getLogger("fusion_trainer.bridge").warning(
            "setproctitle unavailable, process title stays default: %s", exc
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if not BRIDGE_KEY:
        raise SystemExit(
            "FUSION_TRAINER_BRIDGE_KEY is required (and must match the "
            "console's .env.local)"
        )
    uvicorn.run(app, host="127.0.0.1", port=BRIDGE_PORT)
