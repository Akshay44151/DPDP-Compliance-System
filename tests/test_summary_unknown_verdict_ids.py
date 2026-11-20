"""The assessment summary keeps Agent 3's unknown_verdict_ids. No model, no network."""

from orchestrator.models import AssessmentRequest, AssessmentState, PolicyDocument
from orchestrator.summary import summarize_assessment


def make_state() -> AssessmentState:
    request = AssessmentRequest(
        assessment_id="A1",
        policy_documents=[PolicyDocument(name="privacy_policy.txt", text="We keep data.")],
        project_path="some/project",
    )
    return AssessmentState(request=request)


def test_summary_reports_unknown_verdict_ids():
    state = make_state()
    state.alignment_result = {
        "findings": [],
        "llm_failures": 0,
        "unknown_verdict_ids": ["C1", "pc_000000000000"],
    }

    summary = summarize_assessment(state)

    assert summary.alignment_unknown_verdict_ids == ["C1", "pc_000000000000"]
    assert summary.model_dump(mode="json")["alignment_unknown_verdict_ids"] == [
        "C1",
        "pc_000000000000",
    ]


def test_summary_unknown_verdict_ids_empty_when_key_is_missing():
    state = make_state()
    state.alignment_result = {"findings": [], "llm_failures": 0}

    assert summarize_assessment(state).alignment_unknown_verdict_ids == []


def test_summary_unknown_verdict_ids_empty_without_an_alignment_result():
    assert summarize_assessment(make_state()).alignment_unknown_verdict_ids == []