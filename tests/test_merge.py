"""Tests for model merge + export module (issue #58).

Mock-based tests for merge_adapter, export_merged_model, validate_merged_model,
and merge_and_export. The merge endpoint is verified against the real
fusion-mlx in test_contract_integration.py (auto-skip if no server).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fusion_trainer.merge import (
    DEFAULT_VALIDATION_PROMPT,
    export_merged_model,
    generate_validation_sample,
    merge_adapter,
    merge_and_export,
    validate_merged_model,
)


class TestMergeAdapter:
    def test_calls_merge_endpoint(self):
        client = MagicMock()
        client.merge_adapter.return_value = {
            "status": "ok",
            "model": "Qwen2.5-32B",
            "output_path": "/models/bnup-fused",
        }
        result = merge_adapter(
            client=client,
            base_model="Qwen2.5-32B",
            adapter_path="/adapters/bnup_dpo",
            output_dir="/models/bnup-fused",
        )
        assert result["status"] == "ok"
        assert result["output_path"] == "/models/bnup-fused"
        client.merge_adapter.assert_called_once_with(
            model="Qwen2.5-32B",
            adapter_path="/adapters/bnup_dpo",
            output_path="/models/bnup-fused",
            dequantize=False,
        )

    def test_dequantize_flag(self):
        client = MagicMock()
        client.merge_adapter.return_value = {"status": "ok", "output_path": "/out"}
        merge_adapter(
            client=client,
            base_model="m",
            adapter_path="a",
            dequantize=True,
        )
        _, kwargs = client.merge_adapter.call_args
        assert kwargs["dequantize"] is True


class TestExportMergedModel:
    def test_no_quantize_skips(self):
        client = MagicMock()
        result = export_merged_model(client, "/merged", quantize="")
        assert result["status"] == "skipped"
        client._request.assert_not_called()

    def test_4bit_calls_quantize(self):
        client = MagicMock()
        client._base = "http://localhost:11434"
        client._request.return_value = {"status": "ok", "output": "/merged-4bit"}
        result = export_merged_model(client, "/merged", quantize="4bit")
        assert result["status"] == "ok"
        client._request.assert_called_once()

    def test_invalid_quantize_raises(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="quantize must be"):
            export_merged_model(client, "/merged", quantize="16bit")


class TestValidateMergedModel:
    def test_valid_response(self):
        # i18n: user-facing
        response = (
            '<thinking>\n思考过程\n</thinking>\n<dsl>\n{"meta": {"id": "x"}}\n</dsl>'
        )
        result = validate_merged_model(response)
        assert result["has_thinking"] is True
        assert result["has_dsl"] is True
        assert result["dsl_valid_json"] is True
        assert result["valid"] is True

    def test_missing_thinking(self):
        response = '<dsl>\n{"key": "val"}\n</dsl>'
        result = validate_merged_model(response)
        assert result["has_thinking"] is False
        assert result["valid"] is False

    def test_missing_dsl(self):
        # i18n: user-facing
        response = "<thinking>\n思考\n</thinking>\n答案"
        result = validate_merged_model(response)
        assert result["has_dsl"] is False
        assert result["valid"] is False

    def test_invalid_dsl_json(self):
        response = "<thinking>\nt\n</thinking>\n<dsl>\n{broken\n</dsl>"
        result = validate_merged_model(response)
        assert result["dsl_valid_json"] is False
        assert result["valid"] is False

    def test_empty_response(self):
        result = validate_merged_model("")
        assert result["valid"] is False


class TestMergeAndExport:
    def test_merge_only(self):
        client = MagicMock()
        client.merge_adapter.return_value = {
            "status": "ok",
            "output_path": "/merged",
        }
        client.chat_completions.return_value = []
        result = merge_and_export(
            client=client,
            base_model="m",
            adapter_path="a",
            output_dir="/merged",
            validate=False,
        )
        assert result["merge"]["status"] == "ok"
        assert result["quantize"]["status"] == "skipped"
        assert result["final_model_path"] == "/merged"

    def test_merge_with_quantize(self):
        client = MagicMock()
        client._base = "http://localhost:11434"
        client.merge_adapter.return_value = {
            "status": "ok",
            "output_path": "/merged",
        }
        client._request.return_value = {"status": "ok"}
        client.chat_completions.return_value = []
        result = merge_and_export(
            client=client,
            base_model="m",
            adapter_path="a",
            output_dir="/merged",
            quantize="4bit",
            validate=False,
        )
        assert result["quantize"]["status"] == "ok"

    def test_merge_validates_sample(self):
        client = MagicMock()
        client.merge_adapter.return_value = {
            "status": "ok",
            "output_path": "/merged",
        }
        valid_sample = '<thinking>step</thinking><dsl>{"a":1}</dsl>'
        client.chat_completions.return_value = [valid_sample]
        result = merge_and_export(
            client=client,
            base_model="m",
            adapter_path="a",
            output_dir="/merged",
            validate=True,
        )
        assert result["validation"]["status"] == "completed"
        assert result["validation"]["valid"] is True

    def test_merge_validation_failed_format(self):
        client = MagicMock()
        client.merge_adapter.return_value = {
            "status": "ok",
            "output_path": "/merged",
        }
        client.chat_completions.return_value = ["no format tags here"]
        result = merge_and_export(
            client=client,
            base_model="m",
            adapter_path="a",
            output_dir="/merged",
            validate=True,
        )
        assert result["validation"]["status"] == "completed"
        assert result["validation"]["valid"] is False


class TestGenerateValidationSample:
    def test_returns_completion_text(self):
        client = MagicMock()
        client.chat_completions.return_value = ["sample output"]
        text = generate_validation_sample(client, "model-id")
        assert text == "sample output"

    def test_empty_completions_returns_empty(self):
        client = MagicMock()
        client.chat_completions.return_value = []
        assert generate_validation_sample(client, "model-id") == ""

    def test_generation_error_returns_empty(self):
        from fusion_trainer.mlx_client import MLXFineTuneError

        client = MagicMock()
        client.chat_completions.side_effect = MLXFineTuneError("boom")
        assert generate_validation_sample(client, "model-id") == ""

    def test_default_prompt_used(self):
        client = MagicMock()
        client.chat_completions.return_value = ["x"]
        generate_validation_sample(client, "model-id")
        call = client.chat_completions.call_args
        assert DEFAULT_VALIDATION_PROMPT in call.kwargs["messages"][0]["content"]


class TestMlxClientMergeAdapter:
    def test_client_merge_adapter_method(self):
        from fusion_trainer.mlx_client import MERGE_ADAPTER_PATH, MLXFineTuneClient

        client = MLXFineTuneClient.__new__(MLXFineTuneClient)
        client._base = "http://localhost:11434"
        client._request = MagicMock(
            return_value={"status": "ok", "output_path": "/out"}
        )
        result = client.merge_adapter(
            model="Qwen2.5-32B",
            adapter_path="/adapters/bnup",
            output_path="/out",
        )
        assert result["status"] == "ok"
        call_args = client._request.call_args
        assert call_args[0][0] == "POST"
        assert MERGE_ADAPTER_PATH in call_args[0][1]
        body = call_args[1]["json_body"]
        assert body["model"] == "Qwen2.5-32B"
        assert body["adapter_path"] == "/adapters/bnup"
