
import builtins
import json
from pathlib import Path

import pytest

from dashboard.pdf_report import make_pdf


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "sample_audit.json"
)


def load_sample_audit():
    """Load the existing synthetic audit fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_pdf_generation_returns_valid_pdf():
    """Verify that the PDF generator produces a valid PDF file."""
    audit_data = load_sample_audit()

    pdf_bytes = make_pdf(audit_data)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000


def test_pdf_includes_withheld_score_explanation(monkeypatch):
    """Verify that the PDF includes the reason for withholding the score."""
    from reportlab.platypus import SimpleDocTemplate

    audit_data = load_sample_audit()
    score = audit_data["compliance_score"]

    assert score["score"] is None

    captured_story = []

    def capture_story(self, story):
        captured_story.extend(story)

    monkeypatch.setattr(SimpleDocTemplate, "build", capture_story)

    make_pdf(audit_data)

    paragraph_texts = [
        item.getPlainText()
        for item in captured_story
        if hasattr(item, "getPlainText")
    ]

    expected_text = (
        "Score withheld — "
        + score["score_withheld_reason"]
    )

    assert expected_text in paragraph_texts


def test_missing_reportlab_shows_clear_error(monkeypatch):
    """Verify that missing ReportLab produces a helpful error."""
    original_import = builtins.__import__

    def block_reportlab(name, *args, **kwargs):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ModuleNotFoundError(
                "No module named 'reportlab'",
                name=name,
            )

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_reportlab)

    audit_data = load_sample_audit()

    with pytest.raises(
        RuntimeError,
        match="PDF export requires ReportLab",
    ):
        make_pdf(audit_data)
