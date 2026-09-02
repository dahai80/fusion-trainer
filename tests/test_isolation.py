from __future__ import annotations

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
