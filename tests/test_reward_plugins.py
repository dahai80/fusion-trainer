"""Reward plugin tests (issue #90): office validators + rubric judge."""

from __future__ import annotations

import logging
import zipfile

import pytest

from fusion_trainer.rlsl.reward_plugins import (
    extract_result_path,
    get_reward_fn,
    list_reward_fns,
    register_reward_fn,
    reward_office_docx,
    reward_office_pptx,
    reward_office_xlsx,
    reward_rubric_judge,
)
from fusion_trainer.rlsl.sandbox import reward_completion

# ---------------------------------------------------------------------------
# OOXML fixtures (minimal, built with stdlib zipfile)
# ---------------------------------------------------------------------------


def _make_xlsx(path, sheets: dict[str, str]) -> str:
    """sheets: name -> sheet XML body (formulas/values)."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{name}" r:id="rId{i + 1}"/>'
                for i, name in enumerate(sheets)
            )
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i + 1}" '
                f'Target="worksheets/sheet{i + 1}.xml"/>'
                for i in range(len(sheets))
            )
            + "</Relationships>",
        )
        for i, body in enumerate(sheets.values()):
            zf.writestr(
                f"xl/worksheets/sheet{i + 1}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"{body}</worksheet>",
            )
    return str(path)


def _make_docx(path, paragraphs: list[str]) -> str:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "word/document.xml",
            '<document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"{body}</document>",
        )
    return str(path)


def _make_pptx(path, slide_texts: list[str]) -> str:
    ns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    with zipfile.ZipFile(path, "w") as zf:
        for i, text in enumerate(slide_texts, 1):
            zf.writestr(
                f"ppt/slides/slide{i}.xml",
                f'<presentation xmlns:p="{ns}"><p:txBody><a:t xmlns:a="urn:x">{text}</a:t></p:txBody></presentation>',
            )
    return str(path)


COMPLETION = "Saved the result to /tmp/out/report.xlsx"


# ---------------------------------------------------------------------------
# path extraction
# ---------------------------------------------------------------------------


def test_extract_path_last_token():
    assert (
        extract_result_path("wrote /tmp/a.xlsx then /tmp/b.xlsx", {})
        == "/tmp/b.xlsx"
    )


def test_extract_path_regex_wins():
    p = extract_result_path(
        "out=<out>/tmp/x.xlsx</out>", {"path_regex": r"<out>(.*?)</out>"}
    )
    assert p == "/tmp/x.xlsx"


def test_extract_path_no_match_raises():
    with pytest.raises(ValueError):
        extract_result_path("no path here", {})


# ---------------------------------------------------------------------------
# office validators
# ---------------------------------------------------------------------------


def test_xlsx_pass_required_sheets(tmp_path):
    f = _make_xlsx(
        tmp_path / "r.xlsx",
        {"Summary": "<sheetData><c r='B1'><f>SUM(A1:A9)</f></c></sheetData>"},
    )
    score = reward_office_xlsx(
        {}, f"Saved to {f}", {"required_sheets": ["Summary"], "formula_policy": "require"}
    )
    assert score == 1.0


def test_xlsx_fail_missing_sheet(tmp_path):
    f = _make_xlsx(tmp_path / "r.xlsx", {"Data": "<sheetData/>"})
    assert (
        reward_office_xlsx({}, f"saved {f}", {"required_sheets": ["Summary"]}) == 0.0
    )


def test_xlsx_strict_policy_fails_hardcoded(tmp_path):
    f = _make_xlsx(
        tmp_path / "r.xlsx",
        {"Summary": "<sheetData><c r='B1'><v>42</v></c></sheetData>"},
    )
    assert (
        reward_office_xlsx(
            {},
            f"saved {f}",
            {"required_sheets": ["Summary"], "formula_policy": "strict"},
        )
        == 0.0
    )


def test_xlsx_corrupt_zip_fails(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a zip")
    assert reward_office_xlsx({}, f"saved {bad}", {}) == 0.0


def test_xlsx_missing_file_fails(tmp_path):
    # plugin contract: fn may raise; the dispatcher converts to 0.0 loudly
    assert (
        reward_completion(
            {"reward": {"plugin": "office_xlsx"}},
            "saved to /tmp/none/report.xlsx",
        )
        == 0.0
    )


def test_docx_required_sections(tmp_path):
    f = _make_docx(tmp_path / "r.docx", ["Introduction", "Body text", "Conclusion"])
    assert (
        reward_office_docx(
            {}, f"saved {f}", {"required_sections": ["Introduction", "Conclusion"]}
        )
        == 1.0
    )
    assert reward_office_docx({}, f, {"required_sections": ["Budget"]}) == 0.0


def test_pptx_min_slides_and_titles(tmp_path):
    f = _make_pptx(tmp_path / "r.pptx", ["Agenda", "Quarterly Results", "QA"])
    assert reward_office_pptx({}, f"saved {f}", {"min_slides": 2}) == 1.0
    assert (
        reward_office_pptx(
            {}, f, {"min_slides": 5, "required_titles": ["Quarterly Results"]}
        )
        == 0.0
    )


# ---------------------------------------------------------------------------
# registry + dispatch
# ---------------------------------------------------------------------------


def test_registry_roundtrip_and_validation():
    assert "office_xlsx" in list_reward_fns()
    assert get_reward_fn("OFFICE_XLSX") is not None  # case-insensitive
    with pytest.raises(ValueError):
        register_reward_fn("bad name!", lambda s, c, a: 1.0)


def test_reward_completion_dispatches_plugin():
    sample = {"reward": {"plugin": "office_xlsx", "args": {}}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "fusion_trainer.rlsl.reward_plugins.reward_office_xlsx",
            lambda s, c, a: 0.75,
        )
        # registry already holds the original fn; re-register the mock
        import fusion_trainer.rlsl.reward_plugins as rp

        original = rp._REGISTRY["office_xlsx"]
        rp.register_reward_fn("office_xlsx", lambda s, c, a: 0.75)
        try:
            assert reward_completion(sample, COMPLETION) == 0.75
        finally:
            rp.register_reward_fn("office_xlsx", original)


def test_reward_completion_unknown_plugin_zero(caplog):
    sample = {"reward": {"plugin": "no_such_plugin"}}
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.sandbox"):
        assert reward_completion(sample, COMPLETION) == 0.0
    assert any("unknown reward plugin" in r.message for r in caplog.records)


def test_reward_completion_plugin_exception_zero(caplog):
    import fusion_trainer.rlsl.reward_plugins as rp

    def _boom(sample, completion, args):
        raise RuntimeError("validator blew up")

    rp.register_reward_fn("boom", _boom)
    sample = {"reward": {"plugin": "boom"}}
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.sandbox"):
        assert reward_completion(sample, COMPLETION) == 0.0
    assert any("reward plugin boom failed" in r.message for r in caplog.records)


def test_reward_completion_plugin_clamped():
    import fusion_trainer.rlsl.reward_plugins as rp

    rp.register_reward_fn("over", lambda s, c, a: 7.0)
    sample = {"reward": {"plugin": "over"}}
    assert reward_completion(sample, COMPLETION) == 1.0


# ---------------------------------------------------------------------------
# rubric judge
# ---------------------------------------------------------------------------


def test_rubric_judge_no_model_zero(caplog):

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("FUSION_RUBRIC_JUDGE_MODEL", raising=False)
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.reward_plugins"):
        assert (
            reward_rubric_judge(
                {}, "text", {"rubric": [{"criterion": "clarity", "weight": 1.0}]}
            )
            == 0.0
        )
    assert any("no judge model configured" in r.message for r in caplog.records)
    monkeypatch.undo()


def test_rubric_judge_empty_rubric_zero():
    assert reward_rubric_judge({}, "text", {"model": "m"}) == 0.0


def test_rubric_judge_parses_score(monkeypatch):
    import fusion_trainer.rlsl.reward_plugins as rp

    class _FakeClient:
        def chat_completions(self, model, messages, **kw):
            assert messages[0]["content"].count("- ") >= 1  # rubric rendered
            return ['{"scores": {"clarity": 0.9}, "total": 0.9}']

        def close(self):
            pass

    monkeypatch.setattr(rp, "MLXFineTuneClient", _FakeClient)
    score = reward_rubric_judge(
        {},
        "the completion",
        {
            "model": "judge-1",
            "task": "write report",
            "rubric": [{"criterion": "clarity", "weight": 1.0}],
        },
    )
    assert score == pytest.approx(0.9)


def test_rubric_judge_clamps_and_fails_visible(monkeypatch, caplog):
    import fusion_trainer.rlsl.reward_plugins as rp

    class _BadClient:
        def chat_completions(self, model, messages, **kw):
            return ["not json at all"]

        def close(self):
            pass

    monkeypatch.setattr(rp, "MLXFineTuneClient", _BadClient)
    with caplog.at_level(logging.WARNING, logger="fusion_trainer.rlsl.reward_plugins"):
        assert (
            reward_rubric_judge(
                {}, "text", {"model": "m", "rubric": [{"criterion": "c"}]}
            )
            == 0.0
        )
    assert any("rubric_judge failed" in r.message for r in caplog.records)
