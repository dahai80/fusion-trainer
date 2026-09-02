# Seatbelt Sandbox Isolation — Design Spec (issue #53)

**Date:** 2026-09-02
**Issue:** [#53 — container-isolated code-sandbox reward primitive for GRPO](https://github.com/dahai80/fusion-trainer/issues/53)
**Status:** approved
**Path:** architectural

## Context

fusion-trainer's `reward_code` (GRPO code-mode reward) runs model-generated
Python in a subprocess. Current isolation (commit 3771848, soft #2) is
**rlimits only** — `RLIMIT_AS`/`RLIMIT_CPU` inside a restricted temp dir, gated
behind `FUSION_CODE_SANDBOX_TRUSTED=1`. This bounds a *buggy* program (OOM,
infinite loop) but is **not real OS isolation**: the child runs as the current
user with full filesystem + network access. Issue #53 asks for genuine
container isolation — no network, read-only root, tmpfs workspace, bounded
CPU/memory, killed on timeout.

**Boundary:** the real container primitive belongs in the trainer, not the
inference server (per the issue's own framing and the project-boundary rule).
fusion-mlx #743 tracks the *server-side* reward plumbing and is a separate
upstream concern — this spec does not touch fusion-mlx.

**Platform:** macOS Apple Silicon (Darwin 25.5). The local-first isolation
option is the built-in **seatbelt** sandbox (`sandbox-exec`): no daemon, no
install, Mandatory Access Control via a `.sb` profile. Docker is heavy
(daemon, violates local-first); seccomp/namespaces are Linux-only.

## Goal

A real isolation backend for `reward_code` that:
- denies network egress,
- restricts filesystem to a tmpfs workspace + read-only interpreter/stdlib,
- bounds CPU/memory,
- kills on timeout,
- is local-first (no daemon, no install) on macOS,
- degrades gracefully + loudly on non-macOS,
- keeps the existing trusted-gate (defense-in-depth, not a license to run
  untrusted code).

## Non-goals

- Full container image / namespace isolation (Docker-level). Seatbelt MAC is
  the macOS-appropriate equivalent; the issue's "container" means *isolated
  execution environment*, not specifically OCI.
- Server-side reward wiring (fusion-mlx #743).
- Running without `FUSION_CODE_SANDBOX_TRUSTED=1`. Seatbelt is
  defense-in-depth; the explicit opt-in gate stays.

## Architecture

New module `fusion_trainer/rlsl/isolation.py` holds the isolation backend
abstraction. `sandbox.py` stays focused on reward dispatch + answer extraction;
it delegates execution to the backend.

```
reward_code(sample, completion)            # sandbox.py — dispatch + trusted gate
  └─ backend = get_isolation_backend()     # isolation.py — platform/env selection
       └─ backend.run(user_code, tmpdir, env, timeout) -> IsolationResult
            ├─ _SeatbeltBackend.run   # sandbox-exec -f <profile.sb> python <script>
            │      └─ child still runs rlimit prelude (inner CPU cap)
            └─ _RlimitBackend.run     # current path: prelude + subprocess.run
```

### Components

**`IsolationResult`** (dataclass): `returncode: int`, `stdout: str`,
`stderr: str`, `timed_out: bool`. Pure return shape so `reward_code` does not
care which backend produced it.

**`_RlimitBackend`** — extracted from the current `sandbox.py` inline exec:
writes `_ISOLATION_PRELUDE` + user code to `sol.py`, runs `subprocess.run` with
the restricted env. Behavior identical to today.

**`_SeatbeltBackend`** — new:
1. Resolves `python_prefix = os.path.dirname(sys.executable)` and its parent
   (stdlib lives one dir up: `.venv/lib/python3.12/...`).
2. Builds a seatbelt profile string programmatically (paths vary per install),
   writes it to `<tmpdir>/grpo_seatbelt.sb`.
3. Runs `sandbox-exec -f <profile> <sys.executable> <script>` with the same
   restricted env + rlimit prelude inside the child.
4. Captures stdout/stderr/returncode/timeout into `IsolationResult`.

**`get_isolation_backend()`** — selection:
- `FUSION_CODE_SANDBOX_BACKEND` env: `auto` (default) | `seatbelt` | `rlimit`.
- `auto`: Darwin + `sandbox-exec` on PATH → `_SeatbeltBackend`; else
  `_RlimitBackend` + a loud warning log ("real isolation unavailable, falling
  back to rlimit-only").
- explicit `seatbelt` on non-Darwin or missing binary → raise
  `SandboxIsolationError` (fail-visible, never silent downgrade).
- explicit `rlimit` → `_RlimitBackend` on any platform.

Cached per-process (backend constructed once, reused).

### Seatbelt profile

Built programmatically in `_SeatbeltBackend._build_profile(python_prefix,
tmpdir)`:

```scheme
(version 1)
(deny default)
(allow process-exec* (subpath "<python_prefix>"))
(allow file-read* (subpath "<python_prefix>"))
(allow file-read* file-write* (subpath "<tmpdir>"))
(allow process-fork)
(allow signal (target self))
(deny network*)
(allow process-exec* (literal "<sys.executable>"))
```

- `(deny default)` — start locked down.
- exec + read only under the interpreter prefix (the venv + its python lib
  dir).
- read+write only inside the tmpdir (tmpfs workspace).
- fork + self-signal (Python runtime needs these).
- network fully denied.
- The literal interpreter path allowed explicitly as a belt-and-suspenders
  since `subpath` matching can be finicky with symlinks.

Profile written to a temp `.sb` file (not `-p` inline) because long inline
strings + quoting are fragile across shells.

### Error handling

| Condition | Result |
|---|---|
| seatbelt profile parse error / `sandbox-exec` exec failure | loud log, `IsolationResult(returncode!=0)`, reward 0.0 |
| timeout | `timed_out=True`, reward 0.0 |
| non-zero exit (test fail / crash) | reward 0.0 |
| exit 0 | reward 1.0 |
| backend unavailable + explicit `seatbelt` request | raise `SandboxIsolationError` (no silent downgrade) |
| backend unavailable + `auto` | fall back to rlimit + warning |

Fail-visible everywhere: never silently downgrades isolation strength, never
fakes a passing reward.

### Trusted gate (unchanged)

`reward_code` still refuses exec unless `FUSION_CODE_SANDBOX_TRUSTED=1`.
Seatbelt is defense-in-depth; it is not a reason to drop the explicit opt-in.
A seatbelt escape (unlikely but MAC profiles have had bugs) must not mean
untrusted code runs.

## Testing

`tests/test_isolation.py` (new):

- **Seatbelt backend (Darwin + sandbox-exec only, `@pytest.mark.skipif`):**
  - network denied: child opens a socket → exit≠0, reward 0.0.
  - filesystem denied: child writes outside tmpdir → exit≠0, reward 0.0.
  - passing code → exit 0, reward 1.0.
  - infinite loop → timeout → reward 0.0.
  - workspace write inside tmpdir → allowed, reward 1.0.
- **Rlimit backend:** existing `test_cpu_limit_kills_infinite_loop` behavior
  preserved (asserts SIGXCPU kills infinite loop → reward 0.0).
- **Selection:**
  - `auto` on Darwin with binary → `_SeatbeltBackend` (monkeypatch
    `platform.system` + `shutil.which`).
  - `auto` on non-Darwin → `_RlimitBackend` + warning captured.
  - explicit `seatbelt` on non-Darwin → raises `SandboxIsolationError`.
  - explicit `rlimit` on any platform → `_RlimitBackend`.
- **Contract:** `reward_code` returns 1.0 for passing code, 0.0 for failing,
  regardless of backend (parametrize over `seatbelt`/`rlimit` where available).

Existing `tests/test_grpo.py` `TestRewardCode` stays green — the public
`reward_code` signature + return values are unchanged; only the internal exec
plumbing moves to the backend.

Integration test (`test_contract_integration.py`) unchanged — already gated
`FUSION_TRAINER_REAL`.

## Docs

- `CHANGELOG.md` [Unreleased] — new Security entry: seatbelt isolation backend
  (#53), real MAC isolation on macOS, rlimit fallback elsewhere.
- `README.md` / `README_CN.md` sandbox section — note seatbelt auto on macOS,
  `FUSION_CODE_SANDBOX_BACKEND` env, trusted-gate unchanged.
- `CLAUDE.md` design-decision #6 — mention seatbelt backend alongside rlimit.
- Issue #53 — close with a comment linking the landing commit.

## Files touched

- `fusion_trainer/rlsl/isolation.py` — **new** (backends + selection).
- `fusion_trainer/rlsl/sandbox.py` — `reward_code` delegates to backend;
  rlimit prelude + exec extracted to `_RlimitBackend`.
- `tests/test_isolation.py` — **new**.
- `tests/test_grpo.py` — unchanged (confirm green).
- `CHANGELOG.md`, `README.md`, `README_CN.md`, `CLAUDE.md` — docs.

## Risks

- **Seatbelt profile syntax** is under-documented; profile may need iteration
  (the `import "system.sb"` path is too restrictive — blocks interpreter
  exec). Mitigation: programmatic minimal profile + real-exec tests on Darwin
  surface parse errors immediately.
- **Symlinked `sys.executable`** (venv python → real python) may need the
  resolved real path in the exec-allow rule. Mitigation: `os.path.realpath`
  both the prefix and the literal interpreter path.
- **rlimit prelude inside seatbelt** must still be allowed to call
  `resource.setrlimit` (a syscall). Seatbelt's `(deny default)` blocks
  syscalls only via file/process/network, not `setrlimit` — to be verified
  by the passing-code test in `test_isolation.py`.
