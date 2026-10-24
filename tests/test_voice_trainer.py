"""Voice trainer tests (issue #74, fusion-mlx #1336 contract).

Client tests mock the pooled httpx client (A8 pattern from
test_mlx_client). Trainer/dataset tests run fully offline.
"""

from __future__ import annotations

import json

import pytest

from fusion_trainer.dataset.voice_prep import (
    VoiceDatasetError,
    validate_voice_dataset,
)
from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError
from fusion_trainer.voice import VoiceTrainConfig, VoiceTrainer


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


class _FakeHttp:
    def __init__(self, responses):
        # responses: list of _Resp (popped in order) or a single _Resp
        self._responses = list(responses)
        self.requests = []

    def request(self, method, url, json=None, params=None):
        self.requests.append({"method": method, "url": url, "json": json})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def close(self):
        pass


# ── voice_prep ─────────────────────────────────────────────────────────


def _make_dataset(tmp_path, layout):
    if layout == "csv":
        (tmp_path / "a.wav").write_bytes(b"\0" * 2048)
        (tmp_path / "b.wav").write_bytes(b"\0" * 2048)
        (tmp_path / "metadata.csv").write_text(
            "a.wav|hello world\nb.wav|second line\n", encoding="utf-8"
        )
    elif layout == "jsonl":
        (tmp_path / "a.wav").write_bytes(b"\0" * 2048)
        (tmp_path / "metadata.jsonl").write_text(
            '{"audio": "a.wav", "text": "hello"}\n', encoding="utf-8"
        )
    else:
        (tmp_path / "a.wav").write_bytes(b"\0" * 2048)
        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")


def test_prep_csv_layout(tmp_path):
    _make_dataset(tmp_path, "csv")
    report = validate_voice_dataset(tmp_path)
    assert report["samples"] == 2
    assert report["problems"] == []


def test_prep_jsonl_layout(tmp_path):
    _make_dataset(tmp_path, "jsonl")
    assert validate_voice_dataset(tmp_path)["samples"] == 1


def test_prep_bare_wav_txt_layout(tmp_path):
    _make_dataset(tmp_path, "bare")
    assert validate_voice_dataset(tmp_path)["samples"] == 1


def test_prep_missing_dir(tmp_path):
    with pytest.raises(VoiceDatasetError, match="not found"):
        validate_voice_dataset(tmp_path / "nope")


def test_prep_no_pairs(tmp_path):
    with pytest.raises(VoiceDatasetError, match="no training pairs"):
        validate_voice_dataset(tmp_path)


def test_prep_missing_audio_file(tmp_path):
    (tmp_path / "metadata.csv").write_text("gone.wav|hello\n", encoding="utf-8")
    with pytest.raises(VoiceDatasetError, match="audio file missing"):
        validate_voice_dataset(tmp_path)


def test_prep_empty_transcript(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"\0" * 2048)
    (tmp_path / "metadata.csv").write_text("a.wav|   \n", encoding="utf-8")
    with pytest.raises(VoiceDatasetError, match="transcript is empty"):
        validate_voice_dataset(tmp_path)


def test_prep_truncated_audio(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"\0" * 10)
    (tmp_path / "metadata.csv").write_text("a.wav|hello\n", encoding="utf-8")
    with pytest.raises(VoiceDatasetError, match="empty or truncated"):
        validate_voice_dataset(tmp_path)


# ── client ─────────────────────────────────────────────────────────────


def test_create_voice_job_body():
    resp = _Resp({"job_id": "j1", "status": "queued"})
    client = MLXFineTuneClient(port=11434)
    client._http = _FakeHttp([resp])
    job = client.create_voice_job(
        model_id="mlx-community/Kokoro-82M-bf16",
        dataset_dir="/ds",
        voice_id="my-voice",
        tenant_id="t1",
        config={"steps": 40, "seed": 42},
    )
    assert job["job_id"] == "j1"
    req = client._http.requests[0]
    assert req["url"].endswith("/admin/api/fine-tune/voice/jobs")
    assert req["json"] == {
        "model_id": "mlx-community/Kokoro-82M-bf16",
        "dataset_dir": "/ds",
        "voice_id": "my-voice",
        "tenant_id": "t1",
        "config": {"steps": 40, "seed": 42},
    }


def test_wait_for_voice_job_terminal():
    job = {
        "job_id": "j1",
        "status": "completed",
        "voice_ref": "my-voice",
        "voice_path": "/v/my-voice/voice.safetensors",
        "metrics": {"improved": True},
    }
    client = MLXFineTuneClient(port=11434)
    client._http = _FakeHttp([_Resp(job)])
    assert client.wait_for_voice_job("j1") == job


def test_wait_for_voice_job_404_raises_fast():
    client = MLXFineTuneClient(port=11434)
    client._http = _FakeHttp([_Resp({}, status_code=404)])
    with pytest.raises(MLXFineTuneError):
        client.wait_for_voice_job("gone")


def test_cancel_voice_job_tolerates_404():
    client = MLXFineTuneClient(port=11434)
    client._http = _FakeHttp([_Resp({}, status_code=404)])
    assert client.cancel_voice_job("gone") == {}
    assert client._http.requests[0]["url"].endswith(
        "/admin/api/fine-tune/voice/jobs/gone/cancel"
    )


# ── trainer ────────────────────────────────────────────────────────────


def test_trainer_rejects_non_kokoro_model():
    with pytest.raises(ValueError, match="kokoro"):
        VoiceTrainer(model="qwen-4b", dataset_dir="/ds")


def test_trainer_rejects_missing_model():
    with pytest.raises(ValueError, match="model id"):
        VoiceTrainer(model="", dataset_dir="/ds")


def test_trainer_happy_path(tmp_path, monkeypatch):
    _make_dataset(tmp_path, "csv")
    final = {
        "job_id": "j1",
        "status": "completed",
        "voice_id": "v-accept",
        "voice_ref": "v-accept",
        "voice_path": "/voices/v-accept/voice.safetensors",
        "metrics": {"improved": True, "final_spectral_distance": 0.13},
    }
    created = _Resp({"job_id": "j1", "status": "queued"})
    polled = _Resp(final)
    fake = _FakeHttp([created, polled])

    trainer = VoiceTrainer(
        model="mlx-community/Kokoro-82M-bf16",
        dataset_dir=str(tmp_path),
        voice_id="v-accept",
        config=VoiceTrainConfig(steps=5),
        mlx_port=11434,
    )
    trainer._client._http = fake
    try:
        result = trainer.train()
    finally:
        trainer.close()
    assert result["status"] == "completed"
    assert result["voice_ref"] == "v-accept"
    assert result["metrics"]["improved"] is True
    assert fake.requests[0]["json"]["config"]["steps"] == 5


def test_trainer_failed_job_fail_visible(tmp_path, monkeypatch):
    _make_dataset(tmp_path, "csv")
    final = {"job_id": "j1", "status": "failed", "error": "boom", "voice_ref": ""}
    fake = _FakeHttp([_Resp({"job_id": "j1", "status": "queued"}), _Resp(final)])
    trainer = VoiceTrainer(
        model="mlx-community/Kokoro-82M-bf16",
        dataset_dir=str(tmp_path),
        mlx_port=11434,
    )
    trainer._client._http = fake
    try:
        result = trainer.train()
    finally:
        trainer.close()
    assert result["status"] == "failed"
    assert "boom" in result["error"]


def test_trainer_rejects_bad_dataset_before_submit(tmp_path):
    with pytest.raises(VoiceDatasetError):
        VoiceTrainer(
            model="mlx-community/Kokoro-82M-bf16",
            dataset_dir=str(tmp_path / "nope"),
            mlx_port=11434,
        ).train()


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_voice_train_success(tmp_path, monkeypatch, capsys):
    _make_dataset(tmp_path, "csv")

    class _FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            return {
                "status": "completed",
                "voice_ref": "v-cli",
                "voice_path": "/voices/v-cli/voice.safetensors",
                "metrics": {
                    "improved": True,
                    "initial_spectral_distance": 0.2,
                    "final_spectral_distance": 0.1,
                },
            }

        def close(self):
            pass

    monkeypatch.setattr("fusion_trainer.voice.VoiceTrainer", _FakeTrainer)
    from typer.testing import CliRunner

    from fusion_trainer.cli import app

    result = CliRunner().invoke(
        app,
        [
            "voice-train",
            "--model",
            "mlx-community/Kokoro-82M-bf16",
            "--dataset",
            str(tmp_path),
            "--steps",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "v-cli" in result.output


def test_cli_voice_train_failure_exits_nonzero(tmp_path, monkeypatch):
    _make_dataset(tmp_path, "csv")

    class _FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self):
            return {"status": "failed", "error": "upstream down"}

        def close(self):
            pass

    monkeypatch.setattr("fusion_trainer.voice.VoiceTrainer", _FakeTrainer)
    from typer.testing import CliRunner

    from fusion_trainer.cli import app

    result = CliRunner().invoke(
        app,
        [
            "voice-train",
            "--model",
            "mlx-community/Kokoro-82M-bf16",
            "--dataset",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "failed" in result.output


# ── hub publish (#74 M-C) ──────────────────────────────────────────────


class _HubResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


def test_publish_voice_package_creates(monkeypatch):
    from fusion_trainer import hub_client as hc

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _HubResp(201, {"id": "m-1", "name": "my-voice"})

    monkeypatch.setattr(hc.httpx, "post", fake_post)
    hub = hc.ModelHubClient(base_url="http://localhost:11432", api_key="k")
    result = hub.publish_voice_package(
        name="my-voice", base_model="Kokoro-82M", voice_ref="my-voice"
    )
    assert result["published"] is True
    assert result["model"]["id"] == "m-1"
    assert captured["json"]["model_type"] == "audio"
    assert "voice_ref: my-voice" in captured["json"]["description"]


def test_publish_voice_package_409_idempotent(monkeypatch):
    from fusion_trainer import hub_client as hc

    monkeypatch.setattr(
        hc.httpx, "post", lambda *a, **k: _HubResp(409, {"detail": "exists"})
    )
    hub = hc.ModelHubClient(base_url="http://localhost:11432", api_key="k")
    monkeypatch.setattr(
        hub,
        "list_models",
        lambda **k: [{"id": "m-9", "name": "my-voice", "model_type": "audio"}],
    )
    result = hub.publish_voice_package(name="my-voice")
    assert result["published"] is True
    assert result["idempotent"] is True


def test_publish_voice_package_409_no_match_raises(monkeypatch):
    from fusion_trainer import hub_client as hc

    monkeypatch.setattr(
        hc.httpx, "post", lambda *a, **k: _HubResp(409, {"detail": "exists"})
    )
    hub = hc.ModelHubClient(base_url="http://localhost:11432", api_key="k")
    monkeypatch.setattr(hub, "list_models", lambda **k: [])
    with pytest.raises(hc.HubPublishError):
        hub.publish_voice_package(name="my-voice")


def test_publish_voice_package_short_name_raises():
    from fusion_trainer.hub_client import HubPublishError, ModelHubClient

    hub = ModelHubClient(base_url="http://localhost:11432", api_key="k")
    with pytest.raises(HubPublishError, match="too short"):
        hub.publish_voice_package(name="ab")
