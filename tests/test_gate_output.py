"""The evidence gate shows up in the summary and the audit record."""

import json

from orchestrator.audit import build_audit_record
from orchestrator.gate_stage import STEP_NAME as GATE_STEP
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP
from orchestrator.summary import summarize_assessment


def make_state(gate=None) -> AssessmentState:
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="t",
            policy_documents=[PolicyDocument(name="p.txt", text="text")],
            project_path=".",
        ),
        implementation_report={"llm_failures": 0, "facts": []},
        alignment_result={"findings": [], "llm_failures": 0, "failed_calls": []},
        evidence_gate=gate,
    )
    state.log(POLICY_STEP, StepStatus.COMPLETED, target="p.txt")
    return state


def gate_reasons(summary):
    return [r for r in summary.incomplete_reasons if "evidence gate" in r]


def test_no_gate_result_means_none_and_no_gate_reason():
    summary = summarize_assessment(make_state())
    assert summary.gate_counts is None
    assert summary.gate_warnings == []
    assert gate_reasons(summary) == []


def test_review_only_is_counted_but_not_a_reason():
    gate = {"decisions": [], "warnings": [], "counts": {"review_only": 3, "pass": 1}}
    summary = summarize_assessment(make_state(gate))
    assert summary.gate_counts == {"review_only": 3, "pass": 1}
    assert gate_reasons(summary) == []


def test_blocked_findings_make_the_run_incomplete():
    gate = {"decisions": [], "warnings": [], "counts": {"blocked": 2}}
    summary = summarize_assessment(make_state(gate))
    assert summary.is_complete is False
    assert len(gate_reasons(summary)) == 1
    assert "2 finding(s)" in gate_reasons(summary)[0]


def test_gate_warnings_are_copied():
    gate = {"decisions": [], "warnings": ["The policy text was cut."], "counts": {}}
    summary = summarize_assessment(make_state(gate))
    assert summary.gate_warnings == ["The policy text was cut."]


def test_a_failed_gate_step_is_reported_as_a_failed_step():
    state = make_state()
    state.log(GATE_STEP, StepStatus.FAILED, message="TypeError: boom")
    summary = summarize_assessment(state)
    assert any(issue.step == GATE_STEP for issue in summary.failed_steps)
    assert summary.is_complete is False


def test_audit_record_includes_the_gate_result():
    gate = {"decisions": [], "warnings": [], "counts": {"pass": 1}}
    state = make_state(gate)
    record = build_audit_record(state, summarize_assessment(state))
    assert record["evidence_gate"] == gate
    json.dumps(record)  # must stay JSON-friendly


def test_audit_record_keeps_a_missing_gate_as_none():
    state = make_state()
    record = build_audit_record(state, summarize_assessment(state))
    assert record["evidence_gate"] is None