"""Isolation backends for the GRPO code-sandbox reward (issue #53).

Two backends:
  - seatbelt: macOS sandbox-exec (real MAC isolation: deny network, read-only
    root, tmpfs workspace). Auto-selected on Darwin when the binary is present.
  - rlimit: subprocess + RLIMIT_AS/RLIMIT_CPU (bounds buggy programs, NOT real
    isolation). Fallback on non-macOS or when seatbelt is unavailable.

Selection: FUSION_CODE_SANDBOX_BACKEND=auto|seatbelt|rlimit (default auto).
seatbelt-requested-but-unavailable raises SandboxIsolationError (fail-visible,
never a silent downgrade). auto-unavailable falls back to rlimit with a warning.

Importers/callers: fusion_trainer.rlsl.sandbox.reward_code
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BACKEND_ENV = "FUSION_CODE_SANDBOX_BACKEND"
_SEATBELT_BIN = "sandbox-exec"


class SandboxIsolationError(Exception):
    pass


@dataclass
class IsolationResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class _IsolationBackend:
    def run(
        self,
        code: str,
        tmpdir: str,
        env: dict[str, str],
        timeout: float,
    ) -> IsolationResult:
        raise NotImplementedError


class _RlimitBackend(_IsolationBackend):
    def run(
        self,
        code: str,
        tmpdir: str,
        env: dict[str, str],
        timeout: float,
    ) -> IsolationResult:
        # placeholder — real impl lands in Task 2
        return IsolationResult(1, "", "not implemented", False)


class _SeatbeltBackend(_IsolationBackend):
    def run(
        self,
        code: str,
        tmpdir: str,
        env: dict[str, str],
        timeout: float,
    ) -> IsolationResult:
        # placeholder — real impl lands in Task 3
        return IsolationResult(1, "", "not implemented", False)


_BACKEND: _IsolationBackend | None = None


def _seatbelt_available() -> bool:
    return platform.system() == "Darwin" and shutil.which(_SEATBELT_BIN) is not None


def get_isolation_backend() -> _IsolationBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    choice = os.environ.get(_BACKEND_ENV, "auto").lower()
    if choice == "seatbelt":
        if not _seatbelt_available():
            raise SandboxIsolationError(
                f"FUSION_CODE_SANDBOX_BACKEND=seatbelt but {_SEATBELT_BIN} is "
                "unavailable on this platform; refusing to silently downgrade"
            )
        _BACKEND = _SeatbeltBackend()
    elif choice == "rlimit":
        _BACKEND = _RlimitBackend()
    else:
        if _seatbelt_available():
            _BACKEND = _SeatbeltBackend()
        else:
            logger.warning(
                "seatbelt isolation unavailable (platform=%s, %s=%s); "
                "falling back to rlimit-only backend — NOT real isolation",
                platform.system(),
                _SEATBELT_BIN,
                shutil.which(_SEATBELT_BIN),
            )
            _BACKEND = _RlimitBackend()
    logger.info("isolation backend selected: %s", type(_BACKEND).__name__)
    return _BACKEND
