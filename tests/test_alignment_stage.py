from agents.alignment_agent.schemas import AlignmentResult
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim
from orchestrator.alignment_stage import STEP_NAME, run_alignment_stage
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


class FakeResult:
    def __init__(self, failures=0):
        self.llm_failures = failures
        self.findings = []

    def model_dump(self, *, mode="python"):
        return {"findings": [], "llm_failures": self.llm_failures}


class FakeAgent:
    def __init__(self, outcome, calls):
        self.outcome = outcome
        self.calls = calls

    def run(self, policy, implementation, *, policy_text=None):
        self.calls.append((policy, implementation, policy_text))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def make_factory(outcomes):
    calls = []
    made = []
    outcome_iter = iter(outcomes)

    def factory():
        agent = FakeAgent(next(outcome_iter), calls)
        made.append(agent)
        return agent

    return factory, calls, made


def make_state():
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="a1",
            policy_documents=[
                PolicyDocument(name="p.txt", text="We keep data two years.")
            ],
        ),
        policy_claims=[
            PolicyClaim(
                claim="Data is kept two years",
                category=PolicyCategory.STORAGE_RETENTION,
                evidence="We keep data two years.",
                source="p.txt",
            )
        ],
    )


def no_sleep(_seconds):
    pass


def statuses(state):
    return [r.status for r in state.step_log]


def test_success_on_first_try():
    state = make_state()
    factory, _, made = make_factory([FakeResult(0)])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert state.alignment_result == {"findings": [], "llm_failures": 0}
    assert statuses(state) == [StepStatus.COMPLETED]
    assert len(made) == 1


def test_retries_after_model_failure_then_succeeds():
    state = make_state()
    factory, _, made = make_factory([FakeResult(1), FakeResult(0)])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert state.alignment_result["llm_failures"] == 0
    assert statuses(state) == [StepStatus.RETRYING, StepStatus.COMPLETED]
    assert len(made) == 2


def test_keeps_best_result_when_failures_remain():
    state = make_state()
    factory, _, made = make_factory([FakeResult(3), FakeResult(1), FakeResult(2)])
    run_alignment_stage(state, factory, object(), max_retries=2, sleep=no_sleep)
    assert state.alignment_result["llm_failures"] == 1
    assert len(made) == 3
    last = state.step_log[-1]
    assert last.status == StepStatus.COMPLETED
    assert "incomplete" in last.message


def test_exception_then_success():
    state = make_state()
    factory, _, _ = make_factory([RuntimeError("boom"), FakeResult(0)])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert state.alignment_result is not None
    assert state.step_log[0].status == StepStatus.RETRYING
    assert "boom" in state.step_log[0].message


def test_all_attempts_raise_leaves_result_empty():
    state = make_state()
    factory, _, made = make_factory([RuntimeError("x")] * 3)
    run_alignment_stage(state, factory, object(), max_retries=2, sleep=no_sleep)
    assert state.alignment_result is None
    assert state.step_log[-1].status == StepStatus.FAILED
    assert state.step_log[-1].step == STEP_NAME
    assert len(made) == 3


def test_passes_claims_policy_text_and_implementation():
    state = make_state()
    implementation = object()
    factory, calls, _ = make_factory([FakeResult(0)])
    run_alignment_stage(state, factory, implementation, sleep=no_sleep)
    policy, impl, policy_text = calls[0]
    assert policy.claims == state.policy_claims
    assert impl is implementation
    assert "=== Document: p.txt ===" in policy_text


def test_fresh_agent_for_every_attempt():
    state = make_state()
    factory, _, made = make_factory([FakeResult(1), FakeResult(0)])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert made[0] is not made[1]


def test_no_retry_when_max_retries_is_zero():
    state = make_state()
    factory, _, made = make_factory([FakeResult(2)])
    run_alignment_stage(state, factory, object(), max_retries=0, sleep=no_sleep)
    assert len(made) == 1
    assert state.alignment_result["llm_failures"] == 2


def test_sleeps_between_retries():
    state = make_state()
    factory, _, _ = make_factory([FakeResult(1), FakeResult(0)])
    waits = []
    run_alignment_stage(
        state, factory, object(), retry_delay_seconds=0.5, sleep=waits.append
    )
    assert waits == [0.5]


def test_unknown_verdict_ids_survive_in_the_stored_result():
    """The stage stores the full model_dump, so a field added to AlignmentResult must reach the state."""
    state = make_state()
    result = AlignmentResult(findings=[], summary={}, unknown_verdict_ids=["pc_000000000000"])
    factory, _, _ = make_factory([result])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert state.alignment_result["unknown_verdict_ids"] == ["pc_000000000000"]


def test_unknown_verdict_ids_come_from_the_kept_attempt_only():
    """Documents current behaviour: only the best attempt's dump is stored, so ids from a discarded
    attempt are not kept (the same as llm_failures and failed_calls)."""
    state = make_state()
    first = AlignmentResult(
        findings=[], summary={}, llm_failures=1, unknown_verdict_ids=["pc_aaaaaaaaaaaa"]
    )
    second = AlignmentResult(findings=[], summary={}, unknown_verdict_ids=[])
    factory, _, made = make_factory([first, second])
    run_alignment_stage(state, factory, object(), sleep=no_sleep)
    assert len(made) == 2
    assert state.alignment_result["unknown_verdict_ids"] == []
