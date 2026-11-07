---
doc_id: fusion-trainer/seatbelt-sandbox-isolation-plan
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "Seatbelt 沙箱隔离实施计划（历史存档）", en: "Seatbelt sandbox isolation implementation plan (archived)" }
audience: engineer
status: deprecated
superseded_by: docs/guides/grpo-sandbox-reward.md
version: 1.0.0
updated: 2026-09-02
owner: @dahai
source_of_truth:
  - docs/superpowers/plans/2026-09-02-seatbelt-sandbox-isolation.md
---

# Seatbelt Sandbox Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real OS-level isolation (macOS seatbelt `sandbox-exec`) to the GRPO code-sandbox reward, behind an isolation-backend abstraction, with rlimit fallback on non-macOS.

**Architecture:** New `fusion_trainer/rlsl/isolation.py` holds `_RlimitBackend` (extracted from `sandbox.py`) and `_SeatbeltBackend` (new), selected by `get_isolation_backend()` via `FUSION_CODE_SANDBOX_BACKEND` env + platform. `sandbox.py` `reward_code` delegates exec to the backend; the trusted-gate + reward math stay in `sandbox.py`. Both isolation layers run on macOS (seatbelt MAC outside, rlimit SIGXCPU inside).

**Tech Stack:** Python 3.12 stdlib (`subprocess`, `resource`, `tempfile`, `shutil`, `platform`), macOS `sandbox-exec` (seatbelt), pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-02-seatbelt-sandbox-isolation-design.md`

## Global Constraints

- 4-space indentation multiples, no docstrings, always include logging.
- Never import `mlx-lm`/`mlx-vlm` — trainer is pure HTTP orchestration.
- Fail-visible: never silently downgrade isolation or fake a passing reward.
- `FUSION_CODE_SANDBOX_TRUSTED=1` gate stays in `reward_code` regardless of backend.
- ruff clean (`ruff check .` + `ruff format .`).
- Env vars (verbatim from spec): `FUSION_CODE_SANDBOX_BACKEND` (`auto`|`seatbelt`|`rlimit`, default `auto`), `FUSION_CODE_SANDBOX_MEM_BYTES`, `FUSION_CODE_SANDBOX_CPU_SECONDS`, `FUSION_CODE_SANDBOX_TRUSTED`.

---

## File Structure

- **Create:** `fusion_trainer/rlsl/isolation.py` — isolation backend abstraction + selection. One responsibility: run a code string in a restricted environment, return `IsolationResult`.
- **Create:** `tests/test_isolation.py` — backend tests (seatbelt real-exec on Darwin, rlimit, selection).
- **Modify:** `fusion_trainer/rlsl/sandbox.py` — `reward_code` delegates to `get_isolation_backend()`; rlimit prelude + exec code moves to `_RlimitBackend` in `isolation.py`. `reward_completion`/`reward_math`/`extract_answer` unchanged.
- **Modify:** `tests/test_grpo.py` — confirm `TestRewardCode` stays green (no edits expected; only run).
- **Modify:** `CHANGELOG.md`, `README.md`, `README_CN.md`, `CLAUDE.md` — docs (final task).

---

### Task 1: `IsolationResult` dataclass + `SandboxIsolationError` + skeleton selection

**Files:**
- Create: `fusion_trainer/rlsl/isolation.py`
- Test: `tests/test_isolation.py`

**Interfaces:**
- Produces: `IsolationResult(returncode: int, stdout: str, stderr: str, timed_out: bool)`, `SandboxIsolationError(Exception)`, `get_isolation_backend() -> _IsolationBackend` (skeleton returning `_RlimitBackend` placeholder).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_isolation.py`:

```python
from __future__ import annotations

import os

import pytest

from fusion_trainer.rlsl import isolation


class TestIsolationResult:
    def test_fields(self):
        r = isolation.IsolationResult(
            returncode=0, stdout="hi", stderr="", timed_out=False
        )
        assert r.returncode == 0
        assert r.stdout == "hi"
        assert r.stderr == ""
        assert r.timed_out is False


class TestSandboxIsolationError:
    def test_is_exception(self):
        assert issubclass(isolation.SandboxIsolationError, Exception)


class TestGetIsolationBackendDefault:
    def test_returns_a_backend(self):
        b = isolation.get_isolation_backend()
        assert b is not None
        assert hasattr(b, "run")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isolation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fusion_trainer.rlsl.isolation'`

- [ ] **Step 3: Write minimal implementation**

Create `fusion_trainer/rlsl/isolation.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_isolation.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint**

Run: `ruff check fusion_trainer/rlsl/isolation.py tests/test_isolation.py && ruff format fusion_trainer/rlsl/isolation.py tests/test_isolation.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add fusion_trainer/rlsl/isolation.py tests/test_isolation.py
git commit -m "feat(#53): isolation backend skeleton + selection"
```

---

### Task 2: `_RlimitBackend.run` — extract the existing exec from `sandbox.py`

**Files:**
- Modify: `fusion_trainer/rlsl/isolation.py` (`_RlimitBackend.run`)
- Modify: `fusion_trainer/rlsl/sandbox.py` (remove inline exec, delegate to backend)
- Test: `tests/test_isolation.py`

**Interfaces:**
- Consumes: `_ISOLATION_PRELUDE`, `_build_prelude`, `_MEM_LIMIT_BYTES_ENV`, `_CPU_LIMIT_SECONDS_ENV`, `_DEFAULT_MEM_BYTES`, `_DEFAULT_CPU_SECONDS` (move to `isolation.py`).
- Produces: `_RlimitBackend.run(code, tmpdir, env, timeout) -> IsolationResult`. `sandbox.reward_code` calls `get_isolation_backend().run(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_isolation.py`:

```python
class TestRlimitBackendRun:
    def setup_method(self):
        os.environ["FUSION_CODE_SANDBOX_TRUSTED"] = "1"

    def teardown_method(self):
        os.environ.pop("FUSION_CODE_SANDBOX_TRUSTED", None)

    def test_passing_code(self, tmp_path):
        b = isolation._RlimitBackend()
        code = "x = 2 + 3\nassert x == 5"
        r = b.run(code, str(tmp_path), isolation._rlimit_env(str(tmp_path)), 10)
        assert r.returncode == 0
        assert not r.timed_out

    def test_failing_code(self, tmp_path):
        b = isolation._RlimitBackend()
        r = b.run(
            "assert 1 == 2", str(tmp_path), isolation._rlimit_env(str(tmp_path)), 10
        )
        assert r.returncode != 0
        assert not r.timed_out

    def test_timeout(self, tmp_path):
        b = isolation._RlimitBackend()
        r = b.run(
            "while True:\n    pass",
            str(tmp_path),
            isolation._rlimit_env(str(tmp_path)),
            1,
        )
        assert r.timed_out is True
        assert r.returncode != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isolation.py::TestRlimitBackendRun -v`
Expected: FAIL — `_rlimit_env` missing / run returns "not implemented".

- [ ] **Step 3: Move prelude + implement `_RlimitBackend.run`**

In `fusion_trainer/rlsl/isolation.py`, add the prelude constants + `_rlimit_env` helper + real `run`. Move `_ISOLATION_PRELUDE`, `_build_prelude`, `_MEM_LIMIT_BYTES_ENV`, `_CPU_LIMIT_SECONDS_ENV`, `_DEFAULT_MEM_BYTES`, `_DEFAULT_CPU_SECONDS` from `sandbox.py` to `isolation.py` (keep the same values verbatim). Add to the `_RlimitBackend` class:

```python
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
        return IsolationResult(proc.returncode, proc.stdout or "", proc.stderr or "", False)
```

Add module-level helper:

```python
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
```

- [ ] **Step 4: Refactor `sandbox.reward_code` to delegate**

In `fusion_trainer/rlsl/sandbox.py`, replace the `with tempfile.TemporaryDirectory(...)` exec block (lines ~181–213) with delegation. Keep the trusted-gate, code extraction, tests append, and no-code fallback. The new exec block:

```python
with tempfile.TemporaryDirectory(prefix="grpo_sandbox_") as td:
    from fusion_trainer.rlsl.isolation import get_isolation_backend

    backend = get_isolation_backend()
    result = backend.run(user_code, td, _rlimit_env(td), 10)
    if result.timed_out:
        logger.warning("code sandbox timed out")
        return 0.0 * scale
    if result.returncode == 0:
        return 1.0 * scale
    logger.info(
        "code sandbox exit=%d stderr=%s",
        result.returncode,
        result.stderr[:300],
    )
    return 0.0 * scale
```

Remove from `sandbox.py`: `_ISOLATION_PRELUDE`, `_build_prelude`, `_MEM_LIMIT_BYTES_ENV`, `_CPU_LIMIT_SECONDS_ENV`, `_DEFAULT_MEM_BYTES`, `_DEFAULT_CPU_SECONDS` (now in `isolation.py`). Re-export them from `sandbox.py` for the existing `test_grpo.py` references (`sandbox._CPU_LIMIT_SECONDS_ENV`, `sandbox._build_prelude`, `sandbox._MEM_LIMIT_BYTES_ENV`, `sandbox._DEFAULT_MEM_BYTES`):

```python
from fusion_trainer.rlsl.isolation import (  # re-export for tests
    _build_prelude,
    _CPU_LIMIT_SECONDS_ENV,
    _DEFAULT_CPU_SECONDS,
    _DEFAULT_MEM_BYTES,
    _MEM_LIMIT_BYTES_ENV,
)
```

Remove the now-unused `import subprocess` from `sandbox.py` if nothing else uses it.

- [ ] **Step 5: Run the isolation tests + existing sandbox tests**

Run: `pytest tests/test_isolation.py tests/test_grpo.py -v`
Expected: PASS (isolation tests + all TestRewardCode/TestRewardCompletion stay green — `reward_code` return values unchanged).

- [ ] **Step 6: Lint**

Run: `ruff check fusion_trainer/rlsl/isolation.py fusion_trainer/rlsl/sandbox.py tests/test_isolation.py && ruff format fusion_trainer/rlsl/isolation.py fusion_trainer/rlsl/sandbox.py tests/test_isolation.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add fusion_trainer/rlsl/isolation.py fusion_trainer/rlsl/sandbox.py tests/test_isolation.py
git commit -m "feat(#53): extract rlimit backend, reward_code delegates to isolation"
```

---

### Task 3: `_SeatbeltBackend.run` — real seatbelt profile + exec

**Files:**
- Modify: `fusion_trainer/rlsl/isolation.py` (`_SeatbeltBackend._build_profile` + `run`)
- Test: `tests/test_isolation.py`

**Interfaces:**
- Produces: `_SeatbeltBackend.run(code, tmpdir, env, timeout) -> IsolationResult`. Programmatic seatbelt profile written to `<tmpdir>/grpo_seatbelt.sb`, exec via `sandbox-exec -f <profile> <sys.executable> <script>`.

- [ ] **Step 1: Write the failing tests (Darwin + sandbox-exec only)**

Append to `tests/test_isolation.py`:

```python
import platform as _platform
import shutil as _shutil

_SEATBELT_OK = (
    _platform.system() == "Darwin" and _shutil.which("sandbox-exec") is not None
)
_seatbelt = pytest.mark.skipif(not _SEATBELT_OK, reason="seatbelt only on macOS")


class TestSeatbeltBackendRun:
    def setup_method(self):
        os.environ["FUSION_CODE_SANDBOX_TRUSTED"] = "1"

    def teardown_method(self):
        os.environ.pop("FUSION_CODE_SANDBOX_TRUSTED", None)

    @_seatbelt
    def test_passing_code(self, tmp_path):
        b = isolation._SeatbeltBackend()
        r = b.run(
            "assert 1 == 1", str(tmp_path), isolation._rlimit_env(str(tmp_path)), 10
        )
        assert r.returncode == 0
        assert not r.timed_out

    @_seatbelt
    def test_network_denied(self, tmp_path):
        b = isolation._SeatbeltBackend()
        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
            "    raise SystemExit(7)\n"
            "except Exception:\n"
            "    raise SystemExit(0)\n"
        )
        r = b.run(code, str(tmp_path), isolation._rlimit_env(str(tmp_path)), 10)
        assert r.returncode == 0, f"network should be denied: {r.stderr}"

    @_seatbelt
    def test_write_outside_tmpdir_denied(self, tmp_path):
        b = isolation._SeatbeltBackend()
        code = (
            "try:\n"
            "    open('/tmp/grpo_seatbelt_escape_test', 'w').write('x')\n"
            "    raise SystemExit(7)\n"
            "except Exception:\n"
            "    raise SystemExit(0)\n"
        )
        r = b.run(code, str(tmp_path), isolation._rlimit_env(str(tmp_path)), 10)
        assert r.returncode == 0, f"write outside tmpdir should be denied: {r.stderr}"

    @_seatbelt
    def test_timeout(self, tmp_path):
        b = isolation._SeatbeltBackend()
        r = b.run(
            "while True:\n    pass",
            str(tmp_path),
            isolation._rlimit_env(str(tmp_path)),
            1,
        )
        assert r.timed_out is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isolation.py::TestSeatbeltBackendRun -v`
Expected: FAIL — run returns "not implemented" (returncode 1, not 0).

- [ ] **Step 3: Implement `_SeatbeltBackend._build_profile` + `run`**

In `fusion_trainer/rlsl/isolation.py`, replace the `_SeatbeltBackend` placeholder:

```python
class _SeatbeltBackend(_IsolationBackend):
    def _build_profile(self, python_prefix: str, real_exe: str, tmpdir: str) -> str:
        lines = [
            "(version 1)",
            "(deny default)",
            f'(allow process-exec* (literal "{real_exe}"))',
            f'(allow process-exec* (subpath "{python_prefix}"))',
            f'(allow file-read* (subpath "{python_prefix}"))',
            f'(allow file-read* file-write* (subpath "{tmpdir}"))',
            "(allow process-fork)",
            "(allow signal (target self))",
            "(deny network*)",
        ]
        return "\n".join(lines) + "\n"

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

        real_exe = os.path.realpath(sys.executable)
        python_prefix = os.path.dirname(real_exe)
        profile = self._build_profile(python_prefix, real_exe, tmpdir)
        profile_path = Path(tmpdir) / "grpo_seatbelt.sb"
        profile_path.write_text(profile)
        script = Path(tmpdir) / "sol.py"
        prelude = _build_prelude(_MEM_LIMIT_BYTES_ENV, _CPU_LIMIT_SECONDS_ENV)
        script.write_text(prelude + "\n" + code)
        try:
            proc = subprocess.run(
                ["sandbox-exec", "-f", str(profile_path), real_exe, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning("seatbelt sandbox timed out (%ss wall)", timeout)
            return IsolationResult(-1, "", "timeout", True)
        except FileNotFoundError as exc:
            logger.error("seatbelt sandbox-exec missing: %s", exc)
            return IsolationResult(1, "", str(exc), False)
        except Exception as exc:
            logger.warning("seatbelt sandbox failed to run: %s", exc)
            return IsolationResult(1, "", str(exc), False)
        return IsolationResult(
            proc.returncode, proc.stdout or "", proc.stderr or "", False
        )
```

- [ ] **Step 4: Run the seatbelt tests**

Run: `pytest tests/test_isolation.py::TestSeatbeltBackendRun -v`
Expected: PASS on Darwin (4 tests). On non-macOS all 4 skip.

If `test_passing_code` fails with a profile parse error or exec denied, debug the profile: check `r.stderr` for the seatbelt backtrace, adjust `_build_profile` allow rules (common: need to also allow read on `/usr/lib`, `/System/Library`, the dyld cache). Add additional `(allow file-read* (subpath "<dir>"))` rules for the dirs Python loads shared libs from. Re-run until passing.

- [ ] **Step 5: Run the full isolation + grpo suites**

Run: `pytest tests/test_isolation.py tests/test_grpo.py -v`
Expected: PASS (seatbelt tests pass on Darwin, rlimit tests pass everywhere, grpo tests green).

- [ ] **Step 6: Lint**

Run: `ruff check fusion_trainer/rlsl/isolation.py tests/test_isolation.py && ruff format fusion_trainer/rlsl/isolation.py tests/test_isolation.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add fusion_trainer/rlsl/isolation.py tests/test_isolation.py
git commit -m "feat(#53): seatbelt backend with network/fs deny profile"
```

---

### Task 4: Backend selection tests + `seatbelt`-unavailable fail-visible

**Files:**
- Modify: `tests/test_isolation.py`
- Modify: `fusion_trainer/rlsl/isolation.py` (reset `_BACKEND` cache helper, if needed for test isolation)

**Interfaces:**
- Produces: tested behavior — `auto` picks seatbelt on Darwin, `rlimit` explicit, `seatbelt`-on-non-Darwin raises `SandboxIsolationError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_isolation.py`:

```python
class TestBackendSelection:
    def setup_method(self):
        isolation._reset_backend()

    def teardown_method(self):
        isolation._reset_backend()

    def test_explicit_rlimit(self, monkeypatch):
        monkeypatch.setenv("FUSION_CODE_SANDBOX_BACKEND", "rlimit")
        b = isolation.get_isolation_backend()
        assert isinstance(b, isolation._RlimitBackend)

    def test_seatbelt_unavailable_raises_when_explicit(self, monkeypatch):
        monkeypatch.setenv("FUSION_CODE_SANDBOX_BACKEND", "seatbelt")
        monkeypatch.setattr(isolation, "_seatbelt_available", lambda: False)
        with pytest.raises(
            isolation.SandboxIsolationError, match="refusing to silently downgrade"
        ):
            isolation.get_isolation_backend()

    def test_auto_falls_back_to_rlimit_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("FUSION_CODE_SANDBOX_BACKEND", "auto")
        monkeypatch.setattr(isolation, "_seatbelt_available", lambda: False)
        b = isolation.get_isolation_backend()
        assert isinstance(b, isolation._RlimitBackend)

    def test_auto_picks_seatbelt_when_available(self, monkeypatch):
        monkeypatch.setenv("FUSION_CODE_SANDBOX_BACKEND", "auto")
        monkeypatch.setattr(isolation, "_seatbelt_available", lambda: True)
        b = isolation.get_isolation_backend()
        assert isinstance(b, isolation._SeatbeltBackend)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isolation.py::TestBackendSelection -v`
Expected: FAIL — `_reset_backend` missing.

- [ ] **Step 3: Add `_reset_backend` to `isolation.py`**

In `fusion_trainer/rlsl/isolation.py`, add:

```python
def _reset_backend() -> None:
    global _BACKEND
    _BACKEND = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_isolation.py::TestBackendSelection -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint**

Run: `ruff check fusion_trainer/rlsl/isolation.py tests/test_isolation.py && ruff format fusion_trainer/rlsl/isolation.py tests/test_isolation.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add fusion_trainer/rlsl/isolation.py tests/test_isolation.py
git commit -m "test(#53): backend selection + seatbelt-unavailable fail-visible"
```

---

### Task 5: Docs — CHANGELOG, README, README_CN, CLAUDE.md

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the landed backend (Tasks 1–4).

- [ ] **Step 1: Update CHANGELOG.md**

In the existing `## [Unreleased]` section (top of CHANGELOG), add a new `### Security` bullet after the existing soft #2 entry:

```markdown
- **soft #2 / #53 — seatbelt isolation backend**: `reward_code` now runs
  model-generated code under a real MAC isolation layer on macOS via
  `sandbox-exec` (seatbelt): network denied, filesystem restricted to the
  tmpfs workspace + read-only interpreter prefix, killed on timeout. Selected
  automatically on Darwin (`FUSION_CODE_SANDBOX_BACKEND=auto`, default);
  `=rlimit` forces the existing rlimit-only path; `=seatbelt` on a platform
  without the binary raises `SandboxIsolationError` (fail-visible, never a
  silent downgrade). The rlimit prelude still runs *inside* the seatbelt child
  as an inner CPU cap. `FUSION_CODE_SANDBOX_TRUSTED=1` gate unchanged. Real
  container isolation for the inference-server reward path stays upstream at
  [fusion-mlx#743](https://github.com/dahai80/fusion-mlx/issues/743).
```

- [ ] **Step 2: Update README.md sandbox note**

Find the existing `--sandbox` paragraph (around line 182) and append a sentence after it:

```markdown

`reward_code` runs the generated code under a real isolation layer on macOS via the built-in seatbelt sandbox (`sandbox-exec`): network denied, filesystem locked to a tmpfs workspace, killed on timeout. Override the backend with `FUSION_CODE_SANDBOX_BACKEND` (`auto` default | `seatbelt` | `rlimit`); on non-macOS it falls back to rlimits with a loud warning. The `FUSION_CODE_SANDBOX_TRUSTED=1` gate still applies regardless of backend. See `fusion_trainer/rlsl/isolation.py`.
```

- [ ] **Step 3: Update README_CN.md**

Find the sandbox-related section and append the Chinese counterpart:

```markdown

`reward_code` 在 macOS 上通过内置 seatbelt 沙箱（`sandbox-exec`）运行生成的代码：禁用网络、文件系统锁定在 tmpfs 工作区、超时即杀。用 `FUSION_CODE_SANDBOX_BACKEND` 覆盖后端（默认 `auto` | `seatbelt` | `rlimit`）；非 macOS 回退到 rlimit 并大声告警。无论何种后端，`FUSION_CODE_SANDBOX_TRUSTED=1` 显式开关仍然生效。详见 `fusion_trainer/rlsl/isolation.py`。
```

- [ ] **Step 4: Update CLAUDE.md design-decision #6**

Find design decision #6 (Deterministic sandbox reward). Append a sentence:

```markdown
v0.1.13 adds a real macOS seatbelt isolation backend (`sandbox-exec`, `fusion_trainer/rlsl/isolation.py`) auto-selected on Darwin — network denied, tmpfs workspace, read-only interpreter prefix — with rlimit fallback on non-macOS; `FUSION_CODE_SANDBOX_BACKEND` selects. The rlimit prelude still runs inside the seatbelt child as an inner cap. Issue #53.
```

- [ ] **Step 5: Run the full suite + lint**

Run: `source /Users/dahai/fusion/.venv/bin/activate && pytest tests/ -q && ruff check . && ruff format --check .`
Expected: all pass, lint clean.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md README.md README_CN.md CLAUDE.md
git commit -m "docs(#53): seatbelt isolation backend in changelog/readme/claude.md"
```

---

### Task 6: Close issue #53

**Files:** none (GitHub)

- [ ] **Step 1: Push the branch**

```bash
git push origin master
```

- [ ] **Step 2: Close issue #53 with a comment**

Run:
```bash
gh issue close 53 --comment "Landed in <commit-sha>. \`reward_code\` now runs model-generated code under a real macOS seatbelt isolation layer (sandbox-exec: network denied, tmpfs workspace, read-only interpreter prefix, timeout-killed), auto-selected on Darwin via \`FUSION_CODE_SANDBOX_BACKEND=auto\`. rlimit fallback on non-macOS (fail-visible, no silent downgrade). rlimit prelude runs inside the seatbelt child as an inner CPU cap; \`FUSION_CODE_SANDBOX_TRUSTED=1\` gate unchanged. Real container isolation for the server-side reward path stays at fusion-mlx#743. New module: \`fusion_trainer/rlsl/isolation.py\`."
```
Replace `<commit-sha>` with the HEAD sha from Step 1.
Expected: issue closed.

- [ ] **Step 3: Clean up stale branches**

Delete the local + remote branches whose content is fully superseded by master (verified in the session that all branch diffs show master ahead):
```bash
git branch -D audit/round2-fixes feat/publish-adapter-hub release/v0.1.10
git push origin --delete fix/audit-p0-p3-0901
```
Expected: branches deleted (this closes the "合并所有分支到主干" part — all branches were already content-superseded by master, no merge needed).
