"""Cross-process run lock + in-process priority queue (T2, capability plan §1.2).

fusion-trainer has three consumers (agent-studio in-process, CLI, bridge
HTTP) that may live in different processes; before T2 two hosts submitting
jobs at once raced on fusion-mlx. The RunLock is an fcntl.flock mutex on
{cache_path}/run.lock — crash-safe by construction (the kernel drops the
lock when the holder dies), so a SIGKILLed sibling cannot wedge the queue.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RunLock:
    """fcntl.flock-based cross-process run mutex."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self, timeout: float = 0.0, poll: float = 0.2) -> bool:
        """Acquire exclusively. timeout=0 retries forever; >0 gives up after
        that many seconds and returns False (caller decides fail-visible)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.time() + timeout if timeout > 0 else None
        waited = 0.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(fd, 0)
                os.write(
                    fd,
                    json.dumps({"pid": os.getpid(), "acquired": time.time()}).encode(),
                )
                self._fd = fd
                if waited > 0:
                    logger.info("run lock acquired after %.1fs: %s", waited, self.path)
                return True
            except OSError:
                if deadline is not None and time.time() >= deadline:
                    os.close(fd)
                    return False
                # log the wait sparsely: every poll for the first 5s, then ~30s
                if waited < 5.0 or (waited % 30.0) < poll:
                    logger.info("run lock busy, waiting: %s (%.0fs)", self.path, waited)
                time.sleep(poll)
                waited += poll

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def holder(self) -> dict:
        """Best-effort holder info from the lock file (diagnostics only)."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
