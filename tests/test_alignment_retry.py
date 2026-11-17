"""Tests for the targeted retry rule. No model, no network."""

from orchestrator.alignment_problems import (
    count_model_problem_findings,
    problem_score,
)
from orchestrator.alignment_stage import STEP_NAME, run_alignment_stage
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


class Result:
    def __init__(self, findings=(), failures=0, failed_calls=()):
        self.llm_failures = failures
        self.findings = list(findings)
        self.failed_calls = list(failed_calls)

    def model_dump(self, *, mode="python"):
        return {
            "findings": list(self.findings),
            "llm_failures": self.llm_failures,
            "failed_calls": list(self.failed_calls),
        }


def finding(reason, status="unverifiable"):
    return {"status": status, "reason_code": reason}


def make_factory(results):
    remaining = iter(results)
    calls = []

    class Agent:
        def run(self, policy, implementation, *, policy_text=None):
            calls.append(1)
            return next(remaining)

    return (lambda: Agent()), calls


def make_state():
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="r1",
            policy_documents=[
                PolicyDocument(name="p.txt", text="We keep data two years.")
            ],
        )
    )


def no_sleep(_seconds):
    pass


def statuses(state):
    return [r.status for r in state.step_log if r.step == STEP_NAME]


def test_problem_score_counts_only_model_problem_findings():
    dump = {
        "findings": [
            finding("model_failure"),
            finding("invalid_model_output"),
            finding("evidence_gap"),
            finding("unsupported_evidence"),
            finding("model_assessed", status="aligned"),
        ]
    }

    assert count_model_problem_findings(dump) == 2
    assert problem_score(0, dump) == 2


def test_problem_score_handles_missing_keys():
    assert problem_score(0, {}) == 0
    assert problem_score(0, {"findings": [], "llm_failures": 0}) == 0
    assert problem_score(2, {"findings": []}) == 2


def test_problem_score_uses_larger_of_failures_and_failed_calls():
    assert problem_score(1, {"failed_calls": ["verdict: a", "undeclared"]}) == 2
    assert problem_score(3, {"failed_calls": ["undeclared"]}) == 3


def test_clean_result_is_not_retried():
    state = make_state()
    factory, calls = make_factory([Result()])

    run_alignment_stage(state, factory, object(), sleep=no_sleep)

    assert len(calls) == 1
    assert statuses(state) == [StepStatus.COMPLETED]


def test_retries_when_a_finding_has_invalid_model_output():
    state = make_state()
    factory, calls = make_factory(
        [Result(findings=[finding("invalid_model_output")]), Result()]
    )

    run_alignment_stage(state, factory, object(), sleep=no_sleep)

    assert len(calls) == 2
    assert state.alignment_result["findings"] == []
    assert statuses(state) == [StepStatus.RETRYING, StepStatus.COMPLETED]


def test_no_retry_for_evidence_gap_or_unsupported_evidence():
    state = make_state()
    factory, calls = make_factory(
        [Result(findings=[finding("evidence_gap"), finding("unsupported_evidence")])]
    )

    run_alignment_stage(state, factory, object(), sleep=no_sleep)

    assert len(calls) == 1
    assert statuses(state) == [StepStatus.COMPLETED]


def test_retries_when_failed_calls_is_not_empty():
    state = make_state()
    factory, calls = make_factory(
        [Result(failed_calls=["undeclared"]), Result()]
    )

    run_alignment_stage(state, factory, object(), sleep=no_sleep)

    assert len(calls) == 2
    assert statuses(state) == [StepStatus.RETRYING, StepStatus.COMPLETED]


def test_keeps_attempt_with_fewest_problems_and_earlier_one_on_a_tie():
    state = make_state()
    factory, calls = make_factory(
        [
            Result(
                findings=[
                    finding("invalid_model_output", status="r1"),
                    finding("invalid_model_output", status="r1"),
                ]
            ),
            Result(findings=[finding("invalid_model_output", status="r2")]),
            Result(findings=[finding("invalid_model_output", status="r3")]),
        ]
    )

    run_alignment_stage(state, factory, object(), sleep=no_sleep)

    assert len(calls) == 3
    assert [f["status"] for f in state.alignment_result["findings"]] == ["r2"]


def test_stops_at_max_retries_and_says_problems_remain():
    state = make_state()
    factory, calls = make_factory(
        [Result(findings=[finding("model_failure")]) for _ in range(2)]
    )

    run_alignment_stage(state, factory, object(), max_retries=1, sleep=no_sleep)

    assert len(calls) == 2
    assert statuses(state) == [StepStatus.RETRYING, StepStatus.COMPLETED]
    assert "remain after retries" in state.step_log[-1].message