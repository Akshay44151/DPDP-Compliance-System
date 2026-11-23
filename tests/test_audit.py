"""Tests for the audit record. No model, no network."""

import json
from datetime import datetime, timezone

import pytest

from orchestrator.audit import build_audit_record, write_audit_record
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.summary import AssessmentSummary
from agents.legal_rag.schemas import FailedLLMCall, LegalAnalysisResult

MARKER = "SECRET-POLICY-TEXT-123"
FIXED_TIME = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc)


def make_state(assessment_id: str = "A1") -> AssessmentState:
    request = AssessmentRequest(
        assessment_id=assessment_id,
        policy_documents=[
            PolicyDocument(name="privacy_policy.txt", text=f"We keep data. {MARKER}")
        ],
        project_path="some/project",
    )
    return AssessmentState(request=request)


def make_summary(assessment_id: str = "A1") -> AssessmentSummary:
    return AssessmentSummary(assessment_id=assessment_id)


def test_record_has_names_and_lengths_but_not_policy_text():
    state = make_state()
    record = build_audit_record(state, make_summary(), created_at=FIXED_TIME)

    assert record["policy_documents"] == [
        {"name": "privacy_policy.txt", "chars": len(f"We keep data. {MARKER}")}
    ]
    assert MARKER not in json.dumps(record)


def test_missing_results_stay_none():
    record = build_audit_record(make_state(), make_summary(), created_at=FIXED_TIME)

    assert record["implementation_report"] is None
    assert record["alignment_result"] is None


def test_record_includes_step_log_and_summary():
    state = make_state()
    state.log("agent1_policy_extraction", StepStatus.COMPLETED, target="privacy_policy.txt")
    record = build_audit_record(state, make_summary(), created_at=FIXED_TIME)

    assert record["step_log"][0]["step"] == "agent1_policy_extraction"
    assert record["step_log"][0]["status"] == "completed"
    assert record["summary"]["assessment_id"] == "A1"
    assert record["summary"]["is_complete"] is False


def test_record_keeps_results_when_present():
    state = make_state()
    state.alignment_result = {"findings": [], "llm_failures": 1}
    state.implementation_report = {"facts": [], "llm_failures": 0}
    record = build_audit_record(state, make_summary(), created_at=FIXED_TIME)

    assert record["alignment_result"]["llm_failures"] == 1
    assert record["implementation_report"]["llm_failures"] == 0


def test_write_creates_valid_json_file(tmp_path):
    record = build_audit_record(make_state(), make_summary(), created_at=FIXED_TIME)
    path = write_audit_record(record, tmp_path / "audit")

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["assessment_id"] == "A1"
    assert loaded["audit_version"] == 1


def test_unsafe_assessment_id_cannot_escape_the_folder(tmp_path):
    state = make_state("../../evil id")
    record = build_audit_record(state, make_summary("../../evil id"), created_at=FIXED_TIME)
    folder = tmp_path / "audit"
    path = write_audit_record(record, folder)

    assert path.parent == folder
    assert ".." not in path.name


def test_write_never_overwrites(tmp_path):
    record = build_audit_record(make_state(), make_summary(), created_at=FIXED_TIME)
    write_audit_record(record, tmp_path)

    with pytest.raises(FileExistsError):
        write_audit_record(record, tmp_path)


def test_record_keeps_unknown_verdict_ids():
    state = make_state()
    state.alignment_result = {"findings": [], "llm_failures": 0,
                              "unknown_verdict_ids": ["pc_000000000000"]}
    record = build_audit_record(state, make_summary(), created_at=FIXED_TIME)

    assert record["alignment_result"]["unknown_verdict_ids"] == ["pc_000000000000"]


def test_agent4_error_message_survives_serialization_and_audit_json(tmp_path):
    error_message = "Citation excerpt does not match retrieved source text."

    legal_result = LegalAnalysisResult(
        findings=[],
        failed_calls=[
            FailedLLMCall(
                alignment_finding_id="A-007",
                attempt=2,
                error_type="ValueError",
                error_message=error_message,
            )
        ],
    )

    # Verify LegalAnalysisResult JSON serialization preserves diagnostics.
    serialized = legal_result.model_dump(mode="json")
    restored = LegalAnalysisResult.model_validate(serialized)

    assert restored.failed_calls[0].error_message == error_message

    # Verify the generated audit JSON includes the preserved diagnostic.
    state = make_state()
    state.legal_analysis_original = restored.model_dump(mode="json")

    record = build_audit_record(state, make_summary(), created_at=FIXED_TIME)
    path = write_audit_record(record, tmp_path / "audit")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    failed_call = loaded["legal_analysis_original"]["failed_calls"][0]
    assert failed_call["alignment_finding_id"] == "A-007"
    assert failed_call["error_type"] == "ValueError"
    assert failed_call["error_message"] == error_message
