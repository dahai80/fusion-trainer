"""Append-only audit log for RunManager operations (enterprise item #2).

Records who did what, when, to which run. One JSONL line per action:
  {"timestamp", "actor", "action", "run_id", "detail", "ok"}

Actor comes from FUSION_TRAINER_ACTOR (default "local"). The file is
append-only — no rewriting history. Tamper-protection (signing) is out of
scope for a local-first file; the audit trail is for accountability, not
forensic non-repudiation.

API-key gate: FUSION_TRAINER_API_KEY, when set, gates mutating RunManager
operations (start/stop/publish/delete). An unset env means local-trust mode
(all ops allowed) — appropriate for single-user Mac. Set it for shared/multi
operator deployments.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_ACTOR_ENV = "FUSION_TRAINER_ACTOR"
_API_KEY_ENV = "FUSION_TRAINER_API_KEY"


def _default_actor() -> str:
    return os.environ.get(_ACTOR_ENV, "") or "local"


class AuditLog:
    """Append-only JSONL audit trail. Thread-safe."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(
        self,
        action: str,
        run_id: str = "",
        detail: str = "",
        ok: bool = True,
        actor: str = "",
    ) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": actor or _default_actor(),
            "action": action,
            "run_id": run_id,
            "detail": detail,
            "ok": ok,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    def read(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self._path.exists():
            return entries
        with self._lock:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("audit log unparseable line skipped: %s", exc)
        return entries


class PermissionDeniedError(Exception):
    """Raised when an api_key-gated op is attempted without the right key."""


class ApiKeyGate:
    """Optional api-key gate for mutating operations.

    If FUSION_TRAINER_API_KEY is unset, every check passes (local-trust).
    If set, check(key) raises PermissionDenied on a mismatch (fail-visible,
    constant-time compare). A key may also be supplied per-instance at
    construction (tests, explicit config).
    """

    def __init__(self, key: str = ""):
        self._expected = key or os.environ.get(_API_KEY_ENV, "")

    @property
    def enabled(self) -> bool:
        return bool(self._expected)

    def check(self, key: str | None) -> None:
        if not self._expected:
            return
        if key is None or not secrets.compare_digest(str(key), self._expected):
            raise PermissionDeniedError(
                "api_key required but missing or mismatched "
                f"(set {_API_KEY_ENV} or pass api_key)"
            )
