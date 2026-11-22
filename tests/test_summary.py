"""Tests for the assessment summary. Fakes only: no agents, no model."""

from orchestrator.alignment_stage import STEP_NAME as ALIGNMENT_STEP
from orchestrator.analysis_stage import WARNING_STEP
from orchestrator.implementation_stage import STEP_NAME as IMPLEMENTATION_STEP
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP
from orchestrator.summary import AssessmentSummary, summarize_assessment

DOC = "policy.txt"


def finding(status, reason_code="model_assessed"):
    return {
        "id": "A-001",
        "kind": "claim_check",
        "category": "consent",
        "status": status,
        "reason_code": reason_code,
    }


def alignment(findings, llm_failures=0, failed_calls=None, chars=100, truncated=False):
    return {
        "findings": findings,
        "llm_failures": llm_failures,
        "failed_calls": failed_calls or [],
        "policy_text_chars": chars,
        "policy_text_truncated": truncated,
    }


def make_state(findings=None, documents=(DOC,), completed=None, **alignment_kwargs):
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="s1",
            policy_documents=[
                PolicyDocument(name=name, text="We ask for consent.")
                for name in documents
            ],
        )
    )
    for name in documents if completed is None else completed:
        state.log(POLICY_STEP, StepStatus.COMPLETED, target=name)
    state.implementation_report = {"llm_failures": 0}
    state.alignment_result = alignment(
        findings if findings is not None else [finding("aligned"), finding("violated")],
        **alignment_kwargs,
    )
    return state


def reasons_text(summary):
    return " | ".join(summary.incomplete_reasons)


def test_clean_run_is_complete():
    s = summarize_assessment(make_state())

    assert s.is_complete is True
    assert s.incomplete_reasons == []
    assert s.outcome_counts == {
        "compliant": 1,
        "gap": 1,
        "insufficient_evidence": 0,
        "not_assessed": 0,
    }
    assert s.decided_percent == 100.0
    assert s.failed_steps == []
    assert s.documents_without_claims == []


def test_many_insufficient_findings_lower_decided_percent():
    findings = (
        [finding("aligned")]
        + [finding("unverifiable", "evidence_gap")] * 3
        + [finding("insufficient_evidence", "coverage_gap")]
    )

    s = summarize_assessment(make_state(findings))

    assert s.decided_percent == 20.0
    assert s.insufficient_by_reason == {"evidence_gap": 3, "coverage_gap": 1}
    # Complete means nothing failed. It does not mean much was decided.
    assert s.is_complete is True


def test_undeclared_is_counted_as_insufficient_with_its_reason():
    findings = [finding("aligned"), finding("undeclared", "undeclared_practice")]

    s = summarize_assessment(make_state(findings))

    assert s.outcome_counts["insufficient_evidence"] == 1
    assert s.insufficient_by_reason == {"undeclared_practice": 1}
    assert s.decided_percent == 50.0


def test_not_assessed_is_excluded_from_decided_percent():
    findings = [finding("aligned"), finding("out_of_scope", "out_of_scope")]

    s = summarize_assessment(make_state(findings))

    assert s.outcome_counts["not_assessed"] == 1
    assert s.decided_percent == 100.0


def test_missing_alignment_result_is_not_assessed_not_zero():
    state = make_state()
    state.alignment_result = None
    state.log(ALIGNMENT_STEP, StepStatus.FAILED, message="Skipped: no report.")

    s = summarize_assessment(state)

    assert s.outcome_counts is None
    assert s.decided_percent is None
    assert s.is_complete is False
    assert "Agent 3 produced no result" in reasons_text(s)
    assert [i.step for i in s.failed_steps] == [ALIGNMENT_STEP]


def test_missing_implementation_report_is_reported():
    state = make_state()
    state.implementation_report = None

    s = summarize_assessment(state)

    assert s.implementation_llm_failures is None
    assert s.is_complete is False
    assert "Agent 2 produced no report" in reasons_text(s)


def test_agent2_failures_make_the_run_incomplete():
    state = make_state()
    state.implementation_report = {"llm_failures": 2}

    s = summarize_assessment(state)

    assert s.implementation_llm_failures == 2
    assert s.is_complete is False
    assert "Agent 2 gave up on 2 model call(s)" in reasons_text(s)


def test_failed_undeclared_check_is_reported():
    s = summarize_assessment(
        make_state(llm_failures=1, failed_calls=["undeclared"])
    )

    assert s.alignment_llm_failures == 1
    assert s.alignment_failed_calls == ["undeclared"]
    assert s.is_complete is False
    assert "undeclared-behaviour check did not run" in reasons_text(s)


def test_truncated_policy_text_is_reported():
    s = summarize_assessment(make_state(truncated=True, chars=40000))

    assert s.policy_text_truncated is True
    assert s.policy_text_chars == 40000
    assert s.is_complete is False
    assert "cut" in reasons_text(s)


def test_model_problem_findings_are_reported_even_without_llm_failures():
    findings = [
        finding("unverifiable", "model_failure"),
        finding("unverifiable", "invalid_model_output"),
        finding("aligned"),
    ]

    s = summarize_assessment(make_state(findings, llm_failures=0))

    assert s.is_complete is False
    assert "2 finding(s)" in reasons_text(s)


def test_document_without_a_completed_extraction_is_listed():
    state = make_state(documents=("a.txt", "b.txt"), completed=("a.txt",))
    state.log(POLICY_STEP, StepStatus.RETRYING, target="b.txt", attempt=1, message="x")
    state.log(POLICY_STEP, StepStatus.FAILED, target="b.txt", attempt=2, message="boom")

    s = summarize_assessment(state)

    assert s.documents_without_claims == ["b.txt"]
    assert [(i.step, i.target) for i in s.failed_steps] == [(POLICY_STEP, "b.txt")]
    assert s.is_complete is False


def test_retry_then_completed_is_not_a_failed_step():
    state = make_state(completed=())
    state.log(POLICY_STEP, StepStatus.RETRYING, target=DOC, attempt=1, message="x")
    state.log(POLICY_STEP, StepStatus.COMPLETED, target=DOC, attempt=2)

    s = summarize_assessment(state)

    assert s.failed_steps == []
    assert s.documents_without_claims == []
    assert s.is_complete is True


def test_crash_after_retries_gives_one_failed_entry_per_step():
    state = make_state()
    state.log(
        IMPLEMENTATION_STEP, StepStatus.RETRYING, target="proj", attempt=1, message="x"
    )
    state.log(IMPLEMENTATION_STEP, StepStatus.FAILED, message="crashed")

    s = summarize_assessment(state)

    assert len(s.failed_steps) == 1
    assert s.failed_steps[0].step == IMPLEMENTATION_STEP
    assert s.failed_steps[0].status == StepStatus.FAILED
    assert s.is_complete is False


def test_analysis_warnings_are_collected():
    state = make_state()
    text = "Agent 2 is incomplete (1 model call(s) failed)."
    state.log(WARNING_STEP, StepStatus.COMPLETED, message=text)

    s = summarize_assessment(state)

    assert s.warnings == [text]
    assert s.is_complete is False


def test_zero_findings_is_not_a_clean_result():
    s = summarize_assessment(make_state(findings=[]))

    assert s.outcome_counts == {
        "compliant": 0,
        "gap": 0,
        "insufficient_evidence": 0,
        "not_assessed": 0,
    }
    assert s.decided_percent is None
    assert s.is_complete is False
    assert "no findings" in reasons_text(s)


def test_summary_round_trips_through_json():
    s = summarize_assessment(make_state(llm_failures=1, failed_calls=["undeclared"]))

    data = s.model_dump(mode="json")

    assert AssessmentSummary.model_validate(data) == s


def test_legal_status_counts_use_gated_findings():
    state = make_state()

    state.legal_analysis_original = {
        "findings": [
            {
                "id": "finding-1",
                "status": "potential_non_compliance",
            }
        ],
        "failed_calls": [],
        "retrieval_failures": [],
    }

    state.legal_analysis_gated = {
        "findings": [
            {
                "id": "finding-1",
                "status": "insufficient_evidence",
            }
        ],
    }

    summary = summarize_assessment(state)

    assert summary.legal_status_counts == {
        "insufficient_evidence": 1
    }