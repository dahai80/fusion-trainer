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

# Defense-in-depth resource caps applied INSIDE the subprocess before the user
# code runs (via a generated preamble). These are NOT a real sandbox — a
# determined attacker can still fork/pipe/syscall — but they bound the blast
# radius of a buggy (not malicious) generated program: OOM and infinite-loop
# can't hang or starve the trainer host. Real OS isolation stays at #743.
_MEM_LIMIT_BYTES_ENV = "FUSION_CODE_SANDBOX_MEM_BYTES"
_CPU_LIMIT_SECONDS_ENV = "FUSION_CODE_SANDBOX_CPU_SECONDS"
_DEFAULT_MEM_BYTES = 512 * 1024 * 1024  # 512 MB virtual memory cap
_DEFAULT_CPU_SECONDS = 10  # CPU seconds (complements the wall-clock timeout)

# _ISOLATION_PRELUDE: runs inside the child before user code. Sets rlimits so a
# runaway program dies (MemoryError / SIGXCPU) instead of consuming the host.
# resource is stdlib; RLIMIT_AS caps virtual address space, RLIMIT_CPU raises
# SIGXCPU on CPU-time exhaustion. Fileds not present on non-POSIX are skipped.
# NB: env-var names are spliced with str.replace (NOT % formatting) so the
# prelude's own %s/%r literals aren't consumed as format spec.
_ISOLATION_PRELUDE = """\
import resource, sys
def _cap(name, limit):
    try:
        res = getattr(resource, name, None)
        if res is not None and limit is not None:
            resource.setrlimit(res, (int(limit), int(limit)))
    except Exception as _e:
        sys.stderr.write("[sandbox] rlimit %s failed: %s\\n" % (name, _e))
_cap("RLIMIT_AS", __import__("os").environ.get("__MEM_ENV__"))
_cap("RLIMIT_CPU", __import__("os").environ.get("__CPU_ENV__"))
sys.stderr.write("[sandbox] isolation: rlimits applied (NOT a real sandbox, fusion-mlx#743)\\n")
sys.stderr.flush()
"""


def _build_prelude(mem_env: str, cpu_env: str) -> str:
    return _ISOLATION_PRELUDE.replace("__MEM_ENV__", mem_env).replace(
        "__CPU_ENV__", cpu_env
    )


def _rlimit_env(tmpdir: str) -> dict[str, str]:
    mem_bytes = os.environ.get(_MEM_LIMIT_BYTES_ENV, "") or str(_DEFAULT_MEM_BYTES)
    cpu_seconds = os.environ.get(_CPU_LIMIT_SECONDS_ENV, "") or str(
        _DEFAULT_CPU_SECONDS
    )
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "TMPDIR": tmpdir,
        "PYTHONPATH": "",
        _MEM_LIMIT_BYTES_ENV: mem_bytes,
        _CPU_LIMIT_SECONDS_ENV: cpu_seconds,
    }


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
        import subprocess
        import sys
        from pathlib import Path

        script = Path(tmpdir) / "sol.py"
        prelude = _build_prelude(_MEM_LIMIT_BYTES_ENV, _CPU_LIMIT_SECONDS_ENV)
        script.write_text(prelude + "\n" + code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("rlimit sandbox timed out (%ss wall)", timeout)
            return IsolationResult(-1, "", "timeout", True)
        except Exception as exc:
            logger.warning("rlimit sandbox failed to run: %s", exc)
            return IsolationResult(1, "", str(exc), False)
        return IsolationResult(
            proc.returncode, proc.stdout or "", proc.stderr or "", False
        )


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
