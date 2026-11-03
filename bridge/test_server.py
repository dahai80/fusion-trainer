"""trainer-bridge tests (FastAPI TestClient, no real training here).

Run: .venv-bridge/bin/python -m pytest bridge/test_server.py -q
Dataset validation and config building are exercised against the real
fusion_trainer library; the run lifecycle is covered by the live smoke
(Qwen3-0.6B) rather than mocked.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FUSION_TRAINER_BRIDGE_KEY", "test-bridge-key")
os.environ.setdefault("FUSION_TRAINER_CACHE", "/tmp/f-ecommerce-bridge-test-cache")
os.environ.setdefault("FUSION_TRAINER_MLX_PORT", "11434")

import bridge.server as bs  # noqa: E402

AUTH = {"Authorization": "Bearer test-bridge-key"}
CSV_OK = (
    "instruction,input,output\n"
    'Write a product title,red mug,"Elegant Red Ceramic Mug 350ml"\n'
    'Write a product title,blue mug,"Classic Blue Ceramic Mug 350ml"\n'
)
CSV_BAD = "instruction,input,output\n,red mug,missing instruction\n"
JSONL_OK = (
    '{"messages": [{"role": "user", "content": "hi"},'
    ' {"role": "assistant", "content": "hello"}]}\n'
)


@pytest.fixture()
def client():
    bs.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return TestClient(bs.app)


def test_health_probes_mlx(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"ok", "bridge_port", "mlx_port", "presets"}
    assert body["mlx_port"] == 11434


def test_auth_gate(client):
    assert client.get("/presets").status_code == 401
    assert (
        client.get("/presets", headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    assert client.get("/presets", headers=AUTH).status_code == 200


def test_csv_upload_converts_to_messages(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "csv_ok", "content": CSV_OK, "fmt": "csv"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema"] == "messages"
    assert body["samples"] == 2
    stored = Path(body["path"]).read_text().splitlines()
    assert '"role": "user"' in stored[0]


def test_csv_missing_column_rejected(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "csv_bad", "content": "a,b\n1,2", "fmt": "csv"},
    )
    assert resp.status_code == 422
    assert "missing" in resp.text


def test_csv_missing_instruction_rejected(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "csv_bad2", "content": CSV_BAD, "fmt": "csv"},
    )
    assert resp.status_code == 422
    assert "line 2" in resp.text


def test_jsonl_upload_and_schema_detection(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "jsonl_ok", "content": JSONL_OK, "fmt": "jsonl"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["schema"] == "messages"


def test_unsupported_schema_rejected(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "junk", "content": '{"foo": 1}\n', "fmt": "jsonl"},
    )
    assert resp.status_code == 422
    assert "schema" in resp.text


def test_dataset_name_sanitized(client):
    resp = client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "../evil", "content": CSV_OK, "fmt": "csv"},
    )
    assert resp.status_code == 422


def test_dataset_delete(client):
    client.post(
        "/datasets",
        headers=AUTH,
        json={"name": "delme", "content": JSONL_OK, "fmt": "jsonl"},
    )
    assert client.delete("/datasets/delme", headers=AUTH).status_code == 200
    assert client.delete("/datasets/delme", headers=AUTH).status_code == 404


def test_build_config_defaults_and_overrides():
    # self-sufficient: create the dataset here, not via an earlier test
    # (pytest-randomly shuffles order)
    bs._validate_and_store("cfg_ds", JSONL_OK, "jsonl")
    req = bs.RunStart(
        name="cfgtest",
        dataset="cfg_ds",
        lora_rank=8,
        learning_rate=1e-4,
        num_epochs=1,
    )
    cfg = bs._build_config(req)
    assert cfg.sft.lora_rank == 8
    assert cfg.sft.learning_rate == 1e-4
    assert cfg.sft.num_epochs == 1
    assert cfg.sft.method == "lora"
    assert cfg.sft.model == bs.DEFAULT_BASE_MODEL
    assert cfg.mlx_port == 11434
    assert Path(cfg.dataset.path).name == "cfg_ds.jsonl"


def test_build_config_unknown_preset_422():
    bs._validate_and_store("cfg_ds2", JSONL_OK, "jsonl")
    req = bs.RunStart(name="cfgtest", dataset="cfg_ds2", preset="nope")
    with pytest.raises(Exception) as ei:
        bs._build_config(req)
    assert "unknown preset" in str(ei.value)


def test_build_config_missing_dataset_404():
    req = bs.RunStart(name="cfgtest", dataset="absent")
    with pytest.raises(Exception) as ei:
        bs._build_config(req)
    assert "not found" in str(ei.value)


# ── publish / published / eval (mocked RunManager + MLX client) ────────


class _FakeRM:
    def __init__(self, status):
        self._status = status

    def run_status(self, run_id):
        return dict(self._status, run_id=run_id)

    def list_runs(self, limit=20):
        return []


class _FakeMLX:
    def __init__(self, answers=None):
        self._answers = answers or ["base-reply", "tuned-reply"]
        self.calls = []

    def chat_completions(self, model_id, messages, **kw):
        self.calls.append(model_id)
        return [self._answers.pop(0)]

    def close(self):
        pass


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "PUBLISHED_PATH", tmp_path / "published.json")
    merged_dir = tmp_path / "models"
    merged_dir.mkdir()
    monkeypatch.setattr(bs, "MERGED_DIR", merged_dir)
    return merged_dir


def test_publish_needs_completed_run(client, isolated, monkeypatch):
    monkeypatch.setattr(bs, "_rm", lambda: _FakeRM({"status": "running", "model": "m"}))
    resp = client.post("/publish", headers=AUTH, json={"run_id": "r1"})
    assert resp.status_code == 409


def test_publish_without_merge_409(client, isolated, monkeypatch):
    monkeypatch.setattr(
        bs, "_rm", lambda: _FakeRM({"status": "completed", "model": "m"})
    )
    resp = client.post("/publish", headers=AUTH, json={"run_id": "r1"})
    assert resp.status_code == 409
    assert "no merged model" in resp.text


def test_publish_promotes_merged_model(client, isolated, monkeypatch):
    (isolated / "r1-20260101-000000").mkdir()
    monkeypatch.setattr(
        bs, "_rm", lambda: _FakeRM({"status": "completed", "model": "base-1"})
    )
    resp = client.post("/publish", headers=AUTH, json={"run_id": "r1"})
    assert resp.status_code == 200, resp.text
    rec = resp.json()["record"]
    assert rec["model_id"] == "r1-20260101-000000"
    assert rec["base_model"] == "base-1"
    # published state readable back
    got = client.get("/published", headers=AUTH).json()
    assert got["run_id"] == "r1"


def test_publish_picks_newest_merge(client, isolated, monkeypatch):
    import os

    for i, stamp in enumerate(("20260101-000000", "20260102-000000")):
        d = isolated / f"r1-{stamp}"
        d.mkdir()
        os.utime(d, (1000 + i, 1000 + i))
    monkeypatch.setattr(
        bs, "_rm", lambda: _FakeRM({"status": "completed", "model": "base-1"})
    )
    resp = client.post("/publish", headers=AUTH, json={"run_id": "r1"})
    assert resp.json()["record"]["model_id"] == "r1-20260102-000000"


def test_publish_revert_clears_state(client, isolated):
    bs._write_published({"run_id": "r1", "model_id": "m1"})
    resp = client.post("/publish", headers=AUTH, json={"run_id": ""})
    assert resp.json() == {"published": False, "reverted": True}
    assert client.get("/published", headers=AUTH).json() == {}


def test_eval_base_and_tuned(client, isolated, monkeypatch):
    fake = _FakeMLX()
    monkeypatch.setattr(bs, "_mlx", lambda: fake)
    bs._write_published(
        {"run_id": "r1", "model_id": "merged-1", "base_model": "base-1"}
    )
    resp = client.post(
        "/eval", headers=AUTH, json={"instruction": "hi", "max_tokens": 32}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["base_answer"] == "base-reply"
    assert body["tuned_answer"] == "tuned-reply"
    assert fake.calls == ["base-1", "merged-1"]


def test_eval_without_published_tuned(client, isolated, monkeypatch):
    fake = _FakeMLX(answers=["base-only"])
    monkeypatch.setattr(bs, "_mlx", lambda: fake)
    resp = client.post(
        "/eval", headers=AUTH, json={"instruction": "hi", "max_tokens": 32}
    )
    body = resp.json()
    assert body["tuned_answer"] == ""
    assert body["tuned_model"] == ""
    assert fake.calls == [body["base_model"]]


def test_eval_blank_instruction_422(client, isolated):
    resp = client.post("/eval", headers=AUTH, json={"instruction": "   "})
    assert resp.status_code == 422


def test_index_serves_console_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Model Trainer" in resp.content


class _FakeCatalogMLX:
    def __init__(self, data=None, raise_exc=None):
        self._data = data
        self._raise = raise_exc

    def health(self):
        if self._raise:
            raise self._raise
        return {"object": "list", "data": self._data}

    def close(self):
        pass


def test_catalog_lists_models(client, monkeypatch):
    monkeypatch.setattr(
        bs,
        "_mlx",
        lambda: _FakeCatalogMLX(
            data=[{"id": "b-model"}, {"id": "a-model"}, {"no-id": True}]
        ),
    )
    resp = client.get("/catalog", headers=AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["models"] == ["a-model", "b-model"]
    assert body["default_base_model"] == bs.DEFAULT_BASE_MODEL


def test_catalog_mlx_unreachable_502(client, monkeypatch):
    monkeypatch.setattr(
        bs, "_mlx", lambda: _FakeCatalogMLX(raise_exc=ConnectionError("down"))
    )
    resp = client.get("/catalog", headers=AUTH)
    assert resp.status_code == 502


def test_catalog_unexpected_reply_502(client, monkeypatch):
    monkeypatch.setattr(bs, "_mlx", lambda: _FakeCatalogMLX(data="garbage"))
    resp = client.get("/catalog", headers=AUTH)
    assert resp.status_code == 502


def test_console_has_model_select(client):
    html = client.get("/").content.decode()
    assert 'id="run-model"' in html
    assert '"catalog"' in html
    assert "model: " in html


def test_console_theme_follows_shell(client):
    # issue #73: class-based theming + shell theme event + no-flash boot.
    html = client.get("/").content.decode()
    assert "html.dark" in html and "html.light" in html
    # boot script reads the shell's stored preference before first paint
    assert 'localStorage.getItem("fusion.shell.theme")' in html
    assert "prefers-color-scheme" in html
    # embedded mode: shell posts {type:"event", event:"theme", payload:...}
    assert 'd.type === "event" && d.event === "theme"' in html
    assert "applyTheme" in html
    # surfaces use tokens, not hardcoded white (dark mode would break)
    assert "background: #fff" not in html
