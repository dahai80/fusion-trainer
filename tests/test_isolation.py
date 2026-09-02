from __future__ import annotations

import os

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


import platform as _platform
import shutil as _shutil

import pytest

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
            "assert 1 == 1",
            str(tmp_path),
            isolation._rlimit_env(str(tmp_path)),
            10,
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
