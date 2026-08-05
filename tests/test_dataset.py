"""Tests for the dataset module — formats, transforms, loader, versioning.

Targets: formats.py, transforms.py, loader.py, versioning.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fusion_trainer.dataset import (
    DatasetLoader,
    DatasetVersion,
    clean_dataset,
    deduplicate,
    detect_format,
    filter_by_length,
    load_dataset_file,
    save_dataset_file,
    split_dataset,
    to_chat_format,
    truncate_fields,
)

# ===========================================================================
# formats.py — detect_format
# ===========================================================================


class TestDetectFormat:
    def test_jsonl(self):
        assert detect_format(Path("data.jsonl")) == "jsonl"

    def test_json(self):
        assert detect_format(Path("data.json")) == "json"

    def test_parquet(self):
        assert detect_format(Path("data.parquet")) == "parquet"

    def test_csv(self):
        assert detect_format(Path("data.csv")) == "csv"

    def test_tsv(self):
        assert detect_format(Path("data.tsv")) == "tsv"

    def test_arrow(self):
        assert detect_format(Path("data.arrow")) == "arrow"

    def test_unknown(self):
        assert detect_format(Path("data.unknown")) == ""

    def test_no_extension(self):
        assert detect_format(Path("data")) == ""

    def test_uppercase(self):
        assert detect_format(Path("data.JSONL")) == "jsonl"


# ===========================================================================
# formats.py — load_dataset_file
# ===========================================================================


class TestLoadDatasetFile:
    def test_jsonl(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for i in range(3):
                f.write(json.dumps({"id": i, "text": f"hello {i}"}) + "\n")
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 3
            assert data[1]["text"] == "hello 1"
        finally:
            Path(p).unlink(missing_ok=True)

    def test_jsonl_skip_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write('{"a": 2}\n')
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 2
        finally:
            Path(p).unlink(missing_ok=True)

    def test_json_list(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"x": 1}, {"x": 2}], f)
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 2
        finally:
            Path(p).unlink(missing_ok=True)

    def test_json_dict_with_data_key(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"data": [{"a": 1}, {"a": 2}]}, f)
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 2
        finally:
            Path(p).unlink(missing_ok=True)

    def test_json_dict_with_entries_key(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"entries": [{"b": 3}]}, f)
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 1
        finally:
            Path(p).unlink(missing_ok=True)

    def test_json_dict_single_fallback(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"key": "value"}, f)  # Not a list, no known wrapper key
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 1
            assert data[0]["key"] == "value"
        finally:
            Path(p).unlink(missing_ok=True)

    def test_json_invalid_structure(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('"just a string"')
            p = f.name
        try:
            with pytest.raises(ValueError, match="Unexpected JSON structure"):
                load_dataset_file(Path(p))
        finally:
            Path(p).unlink(missing_ok=True)

    def test_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("name,age\nAlice,30\nBob,25\n")
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 2
            assert data[0]["name"] == "Alice"
            assert data[1]["age"] == "25"
        finally:
            Path(p).unlink(missing_ok=True)

    def test_tsv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        ) as f:
            f.write("name\tage\nAlice\t30\nBob\t25\n")
            p = f.name
        try:
            data = load_dataset_file(Path(p))
            assert len(data) == 2
            assert data[0]["name"] == "Alice"
        finally:
            Path(p).unlink(missing_ok=True)

    def test_unknown_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False, encoding="utf-8"
        ) as f:
            f.write("data")
            p = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_dataset_file(Path(p))
        finally:
            Path(p).unlink(missing_ok=True)

    def test_format_override(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".custom", delete=False, encoding="utf-8"
        ) as f:
            for i in range(2):
                f.write(json.dumps({"n": i}) + "\n")
            p = f.name
        try:
            data = load_dataset_file(Path(p), format="jsonl")
            assert len(data) == 2
        finally:
            Path(p).unlink(missing_ok=True)

    def test_parquet_no_pyarrow(self):
        """Should raise ImportError if pyarrow is not installed."""
        # Force import error by checking if pyarrow is available
        try:
            import pyarrow  # noqa: F401

            pytest.skip("pyarrow is installed, can't test missing dependency")
        except ImportError:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                p = f.name
            try:
                with pytest.raises(ImportError, match="pyarrow is required"):
                    load_dataset_file(Path(p))
            finally:
                Path(p).unlink(missing_ok=True)


# ===========================================================================
# formats.py — save_dataset_file
# ===========================================================================


class TestSaveDatasetFile:
    def test_save_jsonl(self):
        data = [{"a": 1}, {"b": 2}]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.jsonl"
            save_dataset_file(data, p, format="jsonl")
            loaded = load_dataset_file(p)
            assert len(loaded) == 2
            assert loaded[0]["a"] == 1

    def test_save_json(self):
        data = [{"a": 1}, {"b": 2}]
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.json"
            save_dataset_file(data, p, format="json")
            loaded = load_dataset_file(p)
            assert len(loaded) == 2

    def test_save_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "out.csv"
            with pytest.raises(ValueError, match="Unsupported output format"):
                save_dataset_file([], p, format="csv")


# ===========================================================================
# transforms.py
# ===========================================================================


class TestCleanDataset:
    def test_strips_whitespace(self):
        data = [{"text": "  hello  ", "num": 42}]
        cleaned = clean_dataset(data)
        assert cleaned[0]["text"] == "hello"

    def test_skips_non_dict(self):
        data = [{"a": 1}, "not a dict", None, {"b": 2}]
        cleaned = clean_dataset(data)  # type: ignore[arg-type]
        assert len(cleaned) == 2

    def test_preserves_non_string(self):
        data = [{"num": 42, "flag": True}]
        cleaned = clean_dataset(data)
        assert cleaned[0]["num"] == 42
        assert cleaned[0]["flag"] is True


class TestDeduplicate:
    def test_removes_duplicates(self):
        data = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
        deduped = deduplicate(data)
        assert len(deduped) == 2

    def test_normalized_whitespace(self):
        data = [{"text": "hello world"}, {"text": "  hello world  "}]
        deduped = deduplicate(data)
        assert len(deduped) == 1

    def test_case_sensitive(self):
        data = [{"text": "Hello"}, {"text": "hello"}]
        deduped = deduplicate(data)
        assert len(deduped) == 1  # lowercased, so they are duplicates

    def test_uses_input_field(self):
        data = [{"input": "hello"}, {"input": "hello"}]
        deduped = deduplicate(data)
        assert len(deduped) == 1

    def test_uses_prompt_field(self):
        data = [{"prompt": "hello"}, {"prompt": "hello"}]
        deduped = deduplicate(data)
        assert len(deduped) == 1

    def test_empty_text_skipped(self):
        data = [{"text": ""}, {"text": "hello"}]
        deduped = deduplicate(data)
        assert len(deduped) == 1


class TestFilterByLength:
    def test_min_length(self):
        data = [{"text": "ab"}, {"text": "a"}, {"text": "abc"}]
        filtered = filter_by_length(data, min_len=2)
        assert len(filtered) == 2

    def test_max_length(self):
        data = [{"text": "ab"}, {"text": "abcde"}, {"text": "abcdefgh"}]
        filtered = filter_by_length(data, max_len=5)
        assert len(filtered) == 2

    def test_custom_field(self):
        data = [{"title": "ab"}, {"title": "a"}]
        filtered = filter_by_length(data, min_len=2, text_fields=["title"])
        assert len(filtered) == 1


class TestTruncateFields:
    def test_truncates_all_fields(self):
        data = [{"input": "hello world", "output": "foo bar"}]
        truncated = truncate_fields(data, max_length=5)
        assert truncated[0]["input"] == "hello"
        assert truncated[0]["output"] == "foo b"

    def test_custom_fields(self):
        data = [{"a": "abcdef", "b": "xyz"}]
        truncated = truncate_fields(data, max_length=3, fields=["a"])
        assert truncated[0]["a"] == "abc"
        assert truncated[0]["b"] == "xyz"  # unchanged

    def test_missing_field(self):
        data = [{"input": "hello"}]
        truncated = truncate_fields(data, max_length=3)
        assert truncated[0]["input"] == "hel"


class TestSplitDataset:
    def test_split_ratio(self):
        data = [{"id": i} for i in range(100)]
        train, val = split_dataset(data, val_ratio=0.1, seed=42)
        assert len(train) == 90
        assert len(val) == 10

    def test_no_shuffle(self):
        data = [{"id": i} for i in range(10)]
        train, val = split_dataset(data, val_ratio=0.2, shuffle=False)
        assert train[0]["id"] == 0
        assert val[0]["id"] == 8

    def test_reproducible_seed(self):
        data = [{"id": i} for i in range(100)]
        t1, v1 = split_dataset(data, val_ratio=0.1, seed=42)
        t2, v2 = split_dataset(data, val_ratio=0.1, seed=42)
        assert [d["id"] for d in t1] == [d["id"] for d in t2]

    def test_empty_data(self):
        train, val = split_dataset([], val_ratio=0.1)
        assert len(train) == 0
        assert len(val) == 0

    def test_all_train(self):
        data = [{"id": i} for i in range(10)]
        train, val = split_dataset(data, val_ratio=0.0)
        assert len(train) == 10
        assert len(val) == 0


class TestToChatFormat:
    def test_basic_conversion(self):
        data = [{"input": "hi", "output": "hello"}]
        result = to_chat_format(data)
        assert len(result[0]["messages"]) == 2
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_with_system_prompt(self):
        data = [{"input": "hi", "output": "hello"}]
        result = to_chat_format(data, system_prompt="You are a bot.")
        assert len(result[0]["messages"]) == 3
        assert result[0]["messages"][0]["role"] == "system"


# ===========================================================================
# loader.py — DatasetLoader
# ===========================================================================


class TestDatasetLoader:
    def test_load_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                for i in range(20):
                    f.write(json.dumps({"input": f"Q{i}", "output": f"A{i}"}) + "\n")
            loader = DatasetLoader()
            train, val = loader.load(path=str(p), val_ratio=0.1)
            assert len(train) == 18
            assert len(val) == 2

    def test_load_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            ds_dir = hub_dir / "datasets"
            ds_dir.mkdir()
            p = ds_dir / "my_data.jsonl"
            with open(p, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"input": f"Q{i}", "output": f"A{i}"}) + "\n")
            loader = DatasetLoader(model_hub_path=str(hub_dir))
            train, val = loader.load(hub_dataset="my_data", val_ratio=0.0)
            assert len(train) == 10

    def test_load_with_category_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            ds_dir = hub_dir / "datasets"
            ds_dir.mkdir()
            p = ds_dir / "sft_code_alpaca.jsonl"
            with open(p, "w") as f:
                for i in range(5):
                    f.write(json.dumps({"input": f"Q{i}", "output": f"A{i}"}) + "\n")
            loader = DatasetLoader(model_hub_path=str(hub_dir))
            train, val = loader.load(hub_dataset="sft_code_alpaca", val_ratio=0.0)
            assert len(train) == 5

    def test_hub_dataset_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(model_hub_path=str(tmpdir))
            with pytest.raises(FileNotFoundError, match="not found in hub"):
                loader.load(hub_dataset="nonexistent")

    def test_hub_dataset_requires_path(self):
        loader = DatasetLoader(model_hub_path="")
        with pytest.raises(ValueError, match="model_hub_path"):
            loader.load(hub_dataset="foo")

    def test_no_path_or_hub(self):
        loader = DatasetLoader()
        with pytest.raises(ValueError, match="must be provided"):
            loader.load()

    def test_path_not_found(self):
        loader = DatasetLoader()
        with pytest.raises(ValueError, match="must be provided"):
            loader.load(path="/nonexistent/path.jsonl")

    def test_max_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                for i in range(100):
                    f.write(json.dumps({"input": f"Q{i}", "output": f"A{i}"}) + "\n")
            loader = DatasetLoader()
            train, val = loader.load(path=str(p), max_samples=10, val_ratio=0.0)
            assert len(train) == 10

    def test_normalize_fallback_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"prompt": "hello", "output": "world"}) + "\n")
            loader = DatasetLoader()
            train, val = loader.load(path=str(p), val_ratio=0.0, input_field="text")
            assert len(train) == 1
            assert train[0]["input"] == "hello"

    def test_normalize_fallback_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"input": "hello", "completion": "world"}) + "\n")
            loader = DatasetLoader()
            train, val = loader.load(path=str(p), val_ratio=0.0, output_field="target")
            assert len(train) == 1
            assert train[0]["output"] == "world"

    def test_normalize_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"irrelevant": "data"}) + "\n")
            loader = DatasetLoader()
            train, val = loader.load(path=str(p), val_ratio=0.0)
            assert len(train) == 0

    def test_preference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pref.jsonl"
            with open(p, "w") as f:
                for i in range(5):
                    f.write(
                        json.dumps(
                            {"prompt": f"P{i}", "chosen": f"C{i}", "rejected": f"R{i}"}
                        )
                        + "\n"
                    )
            loader = DatasetLoader()
            result = loader.load_preference(path=str(p))
            assert len(result) == 5
            assert result[0]["chosen"] == "C0"

    def test_preference_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pref.jsonl"
            with open(p, "w") as f:
                f.write(
                    json.dumps({"prompt": "p", "chosen": "c"}) + "\n"
                )  # missing rejected
                f.write(
                    json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}) + "\n"
                )
            loader = DatasetLoader()
            result = loader.load_preference(path=str(p))
            assert len(result) == 1

    def test_preference_uses_input_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pref.jsonl"
            with open(p, "w") as f:
                f.write(
                    json.dumps({"input": "p", "chosen": "c", "rejected": "r"}) + "\n"
                )
            loader = DatasetLoader()
            result = loader.load_preference(path=str(p))
            assert result[0]["prompt"] == "p"

    def test_preference_max_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pref.jsonl"
            with open(p, "w") as f:
                for i in range(10):
                    f.write(
                        json.dumps({"prompt": f"P{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            loader = DatasetLoader()
            result = loader.load_preference(path=str(p), max_samples=3)
            assert len(result) == 3

    def test_preference_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pref.jsonl"
            with open(p, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "prompt": "hello world",
                            "chosen": "foo bar",
                            "rejected": "baz qux",
                        }
                    )
                    + "\n"
                )
            loader = DatasetLoader()
            result = loader.load_preference(path=str(p), max_seq_length=5)
            assert result[0]["prompt"] == "hello"
            assert result[0]["chosen"] == "foo b"
            assert result[0]["rejected"] == "baz q"

    def test_preference_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            ds_dir = hub_dir / "datasets"
            ds_dir.mkdir()
            p = ds_dir / "pref_data.jsonl"
            with open(p, "w") as f:
                for i in range(3):
                    f.write(
                        json.dumps({"prompt": f"P{i}", "chosen": "c", "rejected": "r"})
                        + "\n"
                    )
            loader = DatasetLoader(model_hub_path=str(hub_dir))
            result = loader.load_preference(hub_dataset="pref_data")
            assert len(result) == 3

    def test_model_hub_path_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            ds_dir = hub_dir / "datasets"
            ds_dir.mkdir()
            p = ds_dir / "data.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"input": "hi", "output": "ho"}) + "\n")
            # Override hub path at call time
            loader = DatasetLoader(model_hub_path="/wrong/path")
            train, val = loader.load(
                hub_dataset="data", model_hub_path=str(hub_dir), val_ratio=0.0
            )
            assert len(train) == 1


# ===========================================================================
# versioning.py — DatasetVersion
# ===========================================================================


class TestDatasetVersion:
    def test_constructor(self):
        v = DatasetVersion(
            name="test", version="v2", path="/p", hash="abc", num_samples=50
        )
        assert v.name == "test"
        assert v.version == "v2"
        assert v.num_samples == 50

    def test_to_dict(self):
        v = DatasetVersion(name="x", version="v1", path="/p", hash="h", num_samples=10)
        d = v.to_dict()
        assert d["name"] == "x"
        assert d["version"] == "v1"

    def test_from_dict(self):
        d = {
            "dataset": "foo",
            "version": "v3",
            "file_path": "/a",
            "file_hash": "h",
            "num_samples": 7,
        }
        v = DatasetVersion.from_dict(d)
        assert v.name == "foo"
        assert v.version == "v3"
        assert v.num_samples == 7

    def test_from_dict_minimal(self):
        v = DatasetVersion.from_dict({})
        assert v.name == ""
        assert v.version == "v1"
        assert v.num_samples == 0

    def test_from_dict_with_name_key(self):
        d = {
            "name": "bar",
            "version": "v2",
            "path": "/b",
            "hash": "x",
            "num_samples": 3,
        }
        v = DatasetVersion.from_dict(d)
        assert v.name == "bar"
        assert v.path == "/b"


# ===========================================================================
# Issue #7 — D1 cleaned-dataset schema detection / validation / loading
# (ShareGPT / messages-JSONL / DPO-pair / GRPO-prompt)
# ===========================================================================
from fusion_trainer.dataset import (
    SCHEMA_DPO_PAIR,
    SCHEMA_GRPO_PROMPT,
    SCHEMA_INPUT_OUTPUT,
    SCHEMA_MESSAGES,
    SCHEMA_SHAREGPT,
    detect_dataset_schema,
    sharegpt_to_messages,
    validate_format,
)


class TestDetectDatasetSchema:
    def test_messages(self):
        assert (
            detect_dataset_schema({"messages": [{"role": "user", "content": "hi"}]})
            == SCHEMA_MESSAGES
        )

    def test_sharegpt(self):
        assert (
            detect_dataset_schema({"conversations": [{"from": "human", "value": "hi"}]})
            == SCHEMA_SHAREGPT
        )

    def test_dpo_pair(self):
        assert (
            detect_dataset_schema({"prompt": "p", "chosen": "c", "rejected": "r"})
            == SCHEMA_DPO_PAIR
        )

    def test_grpo_prompt(self):
        assert (
            detect_dataset_schema({"prompt": "p", "answer": "a"}) == SCHEMA_GRPO_PROMPT
        )
        assert detect_dataset_schema({"prompt": "p"}) == SCHEMA_GRPO_PROMPT

    def test_input_output(self):
        assert (
            detect_dataset_schema({"input": "q", "output": "a"}) == SCHEMA_INPUT_OUTPUT
        )
        assert detect_dataset_schema({"text": "raw"}) == SCHEMA_INPUT_OUTPUT

    def test_unknown(self):
        assert detect_dataset_schema({"foo": "bar"}) == ""
        assert detect_dataset_schema("not a dict") == ""


class TestValidateFormat:
    def test_valid_messages(self):
        data = [
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
        ]
        ok, msg = validate_format(data)
        assert ok and msg == SCHEMA_MESSAGES

    def test_explicit_schema(self):
        data = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        ok, msg = validate_format(data, SCHEMA_DPO_PAIR)
        assert ok

    def test_mixed_schema_fails(self):
        data = [
            {"messages": [{"role": "user", "content": "a"}]},
            {"input": "q", "output": "a"},
        ]
        ok, msg = validate_format(data)
        assert not ok
        assert "do not match schema" in msg

    def test_empty_fails(self):
        ok, msg = validate_format([])
        assert not ok
        assert "empty" in msg

    def test_undetectable_first_sample(self):
        ok, msg = validate_format([{"foo": "bar"}])
        assert not ok


class TestSharegptToMessages:
    def test_basic(self):
        item = {
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi there"},
            ]
        }
        msgs = sharegpt_to_messages(item)
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_system_role(self):
        msgs = sharegpt_to_messages(
            {"conversations": [{"from": "system", "value": "sys"}]}
        )
        assert msgs[0]["role"] == "system"

    def test_unknown_role_defaults_user(self):
        msgs = sharegpt_to_messages(
            {"conversations": [{"from": "alien", "value": "x"}]}
        )
        assert msgs[0]["role"] == "user"

    def test_skips_empty(self):
        msgs = sharegpt_to_messages({"conversations": [{"from": "gpt", "value": ""}]})
        assert msgs == []


class TestLoadMessages:
    def _write(self, tmp_path, name, samples):
        p = tmp_path / name
        with open(p, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        return p

    def test_messages_jsonl(self, tmp_path):
        p = self._write(
            tmp_path,
            "m.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "q2"},
                        {"role": "assistant", "content": "a2"},
                    ]
                },
            ],
        )
        loader = DatasetLoader()
        train, val = loader.load_messages(path=str(p), val_ratio=0.0)
        assert len(train) == 2
        assert train[0]["messages"][0]["content"] == "q1"

    def test_sharegpt_jsonl(self, tmp_path):
        p = self._write(
            tmp_path,
            "s.jsonl",
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hi"},
                        {"from": "gpt", "value": "yo"},
                    ]
                },
            ],
        )
        loader = DatasetLoader()
        train, val = loader.load_messages(path=str(p), val_ratio=0.0)
        assert len(train) == 1
        assert train[0]["messages"][0] == {"role": "user", "content": "hi"}
        assert train[0]["messages"][1] == {"role": "assistant", "content": "yo"}

    def test_val_split(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(10)
        ]
        p = self._write(tmp_path, "m.jsonl", samples)
        loader = DatasetLoader()
        train, val = loader.load_messages(path=str(p), val_ratio=0.2)
        assert len(train) == 8
        assert len(val) == 2

    def test_max_samples(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(10)
        ]
        p = self._write(tmp_path, "m.jsonl", samples)
        loader = DatasetLoader()
        train, val = loader.load_messages(path=str(p), val_ratio=0.0, max_samples=3)
        assert len(train) == 3

    def test_skips_non_chat(self, tmp_path):
        p = self._write(
            tmp_path,
            "m.jsonl",
            [
                {"messages": [{"role": "user", "content": "ok"}]},
                {"input": "q", "output": "a"},
            ],
        )
        loader = DatasetLoader()
        train, val = loader.load_messages(path=str(p), val_ratio=0.0)
        assert len(train) == 1


class TestLoadGrpo:
    def test_basic(self, tmp_path):
        p = tmp_path / "g.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"prompt": "solve 1+1", "answer": "2"}) + "\n")
            f.write(json.dumps({"prompt": "solve 2+2"}) + "\n")
        loader = DatasetLoader()
        result = loader.load_grpo(path=str(p))
        assert len(result) == 2
        assert result[0]["prompt"] == "solve 1+1"
        assert result[0]["answer"] == "2"
        assert "answer" not in result[1]

    def test_skips_missing_prompt(self, tmp_path):
        p = tmp_path / "g.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"prompt": "ok"}) + "\n")
            f.write(json.dumps({"answer": "no prompt"}) + "\n")
        loader = DatasetLoader()
        result = loader.load_grpo(path=str(p))
        assert len(result) == 1

    def test_preserves_reference_fields(self, tmp_path):
        p = tmp_path / "g.jsonl"
        with open(p, "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "p",
                        "tests": ["t1"],
                        "reward": 1.0,
                        "metadata": {"k": "v"},
                    }
                )
                + "\n"
            )
        loader = DatasetLoader()
        result = loader.load_grpo(path=str(p))
        assert result[0]["tests"] == ["t1"]
        assert result[0]["reward"] == 1.0
        assert result[0]["metadata"] == {"k": "v"}
