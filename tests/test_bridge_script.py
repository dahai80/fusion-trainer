import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "trainer-bridge.sh"

DEFAULT_SINKS = (
    "$HOME/.fusion-gateway/rollout-tap.jsonl"
    ":"
    "$HOME/.fusion-office/rollout-outcomes.jsonl"
)


def _make_fake_venv(tmp: Path, env_file: Path) -> Path:
    venv = tmp / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    fake_py = bin_dir / "python"
    fake_py.write_text(
        textwrap.dedent(
            f"""
            #!/bin/sh
            for a in "$@"; do
                case "$a" in
                    -c) exit 0;;
                esac
            done
            env > "{env_file}"
            exec sleep 30
            """
        ).strip()
        + "\n"
    )
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IXUSR)
    return venv


def _start_bridge(tmp: Path, venv: Path, env_file: Path, extra_env: dict) -> None:
    script = tmp / "scripts" / "trainer-bridge.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env.pop("FUSION_TRAINER_ROLLOUT_SINKS", None)
    env.update(
        {
            "FUSION_TRAINER_BRIDGE_VENV": str(venv),
            "FUSION_TRAINER_BRIDGE_KEY": "test-key",
            "FUSION_MLX_ADMIN_TOKEN": "test-token",
            "FAKE_ENV_FILE": str(env_file),
        }
    )
    env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(script), "start"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "bridge started" in proc.stdout


def _stop_bridge(tmp: Path) -> None:
    script = tmp / "scripts" / "trainer-bridge.sh"
    subprocess.run(
        ["bash", str(script), "stop"],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _read_env(env_file: Path) -> dict:
    lines = env_file.read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


@pytest.fixture(autouse=True)
def _cleanup():
    yield


def test_rollout_sinks_default(tmp_path):
    env_file = tmp_path / "env.txt"
    venv = _make_fake_venv(tmp_path, env_file)
    try:
        _start_bridge(tmp_path, venv, env_file, {})
        exported = _read_env(env_file)
        expected = os.path.expandvars(DEFAULT_SINKS)
        assert exported.get("FUSION_TRAINER_ROLLOUT_SINKS") == expected
    finally:
        _stop_bridge(tmp_path)
        subprocess.run(["pkill", "-f", f"sleep 30.*{env_file}"], capture_output=True)


def test_rollout_sinks_env_override_wins(tmp_path):
    env_file = tmp_path / "env.txt"
    venv = _make_fake_venv(tmp_path, env_file)
    override = str(tmp_path / "custom.jsonl")
    try:
        _start_bridge(
            tmp_path, venv, env_file, {"FUSION_TRAINER_ROLLOUT_SINKS": override}
        )
        exported = _read_env(env_file)
        assert exported.get("FUSION_TRAINER_ROLLOUT_SINKS") == override
    finally:
        _stop_bridge(tmp_path)
        subprocess.run(["pkill", "-f", f"sleep 30.*{env_file}"], capture_output=True)
