"""Sandbox reward functions for GRPO.

Real, deterministic reward computation — no model routing (Rule 5).
Three modes:
  - code: extract a code block from the completion, exec in a sandboxed
          subprocess with optional tests, reward = pass rate.
  - math: extract a final answer, normalize, compare to the reference.
  - off:  uniform reward (baseline / debugging).

Importers/callers: fusion_trainer.rlsl.grpo.GRPOTrainer
Affected API: reward_completion(sample, completion, mode) -> float
Data schemas:
  sample: {"prompt": str, "answer"?: str, "tests"?: str, "solution"?: str, ...}
  completion: str (model output, may contain <answer>...</answer> or ``` blocks)

Security: code mode runs in a subprocess with a hard timeout and a restricted
temp dir. It is NOT a full sandbox — do not run on untrusted data. For
untrusted code, fuse with a container/seccomp sandbox upstream.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# C2: code mode execs model/data-generated Python in a subprocess with no real
# isolation (no seccomp/namespace/privilege drop). It is only safe to run on
# fully trusted data. Require an explicit opt-in env var before exec; absent
# it, refuse visibly (0.0 + loud warning) instead of silently running untrusted
# code. Real isolation is upstream (fusion-mlx#743).
_TRUSTED_ENV = "FUSION_CODE_SANDBOX_TRUSTED"

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

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|答案|最终答案)\s*[:：]?\s*(.+?)(?:\n|$)",
    re.DOTALL | re.IGNORECASE,
)


def extract_answer(text: str) -> str:
    """Pull a normalized final answer out of a model completion."""
    m = _ANSWER_RE.search(text)
    if m:
        return _normalize_answer(m.group(1))
    m = _BOXED_RE.search(text)
    if m:
        return _normalize_answer(m.group(1))
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return _normalize_answer(m.group(1))
    last_line = text.strip().splitlines()[-1] if text.strip() else ""
    return _normalize_answer(last_line)


def _normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    except ValueError:
        return s.lower()


def reward_math(sample: dict[str, Any], completion: str) -> float:
    """1.0 if extracted answer matches reference, else 0.0 (partial for numeric closeness)."""
    ref = sample.get("answer") or sample.get("solution") or ""
    ref = _normalize_answer(str(ref))
    if not ref:
        logger.warning("math reward: no reference answer in sample, neutral 0.5")
        return 0.5
    got = extract_answer(completion)
    if got == ref:
        return 1.0
    try:
        if abs(float(got) - float(ref)) < 1e-4:
            return 0.8
    except (ValueError, TypeError):
        pass
    return 0.0


def _extract_code(completion: str) -> str:
    m = _CODE_BLOCK_RE.search(completion)
    if m:
        return m.group(1)
    if "def " in completion or "import " in completion or "print(" in completion:
        return completion
    return ""


def reward_code(sample: dict[str, Any], completion: str) -> float:
    """Run the generated code (+ optional tests) in a subprocess; reward = pass rate.

    sample["tests"] is appended as assertions. Reward is 1.0 on clean exit,
    0.0 on exception/timeout, 0.5 if no code extracted but a natural-language
    answer matches a reference (degrades gracefully).

    C2: the subprocess has NO real isolation (runs as the current user with
    filesystem/network access). It must only run on fully trusted data. Before
    exec we require FUSION_CODE_SANDBOX_TRUSTED=1; absent it we refuse visibly
    (0.0 + warning) rather than silently running model-generated code.
    """
    # C2: refuse untrusted exec unless explicitly opted in.
    if os.environ.get(_TRUSTED_ENV, "") != "1":
        logger.error(
            "code reward refused: %s!=1. code mode execs model/data-generated "
            "Python with no real isolation (seccomp/container upstream, "
            "fusion-mlx#743). Set %s=1 ONLY for fully trusted data. "
            "Returning fail-visible 0.0.",
            _TRUSTED_ENV,
            _TRUSTED_ENV,
        )
        return 0.0

    code = _extract_code(completion)
    if not code:
        ref = sample.get("answer") or ""
        if ref and extract_answer(completion) == _normalize_answer(str(ref)):
            return 0.5
        return 0.0

    tests = sample.get("tests") or ""
    user_code = code + "\n\n" + str(tests) if tests else code

    mem_bytes = os.environ.get(_MEM_LIMIT_BYTES_ENV, "") or str(_DEFAULT_MEM_BYTES)
    cpu_seconds = os.environ.get(_CPU_LIMIT_SECONDS_ENV, "") or str(
        _DEFAULT_CPU_SECONDS
    )

    with tempfile.TemporaryDirectory(prefix="grpo_sandbox_") as td:
        script = Path(td) / "sol.py"
        # _ISOLATION_PRELUDE runs first (sets rlimits), then the user code.
        prelude = _build_prelude(_MEM_LIMIT_BYTES_ENV, _CPU_LIMIT_SECONDS_ENV)
        script.write_text(prelude + "\n" + user_code)
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "TMPDIR": td,
            "PYTHONPATH": "",
            _MEM_LIMIT_BYTES_ENV: mem_bytes,
            _CPU_LIMIT_SECONDS_ENV: cpu_seconds,
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=td,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("code sandbox timed out (10s wall)")
            return 0.0
        except Exception as exc:
            logger.warning("code sandbox failed to run: %s", exc)
            return 0.0
        if proc.returncode == 0:
            return 1.0
        logger.info(
            "code sandbox exit=%d stderr=%s", proc.returncode, proc.stderr[:300]
        )
        return 0.0


def reward_completion(
    sample: dict[str, Any],
    completion: str,
    mode: str = "auto",
    scale: float = 1.0,
) -> float:
    """Dispatch reward by mode. 'auto' picks by dataset fields (tests->code, answer->math)."""
    if mode == "off":
        return 0.5 * scale
    if mode == "math":
        return reward_math(sample, completion) * scale
    if mode == "code":
        return reward_code(sample, completion) * scale
    # auto
    if sample.get("tests"):
        return reward_code(sample, completion) * scale
    if sample.get("answer") or sample.get("solution"):
        return reward_math(sample, completion) * scale
    return 0.5 * scale
