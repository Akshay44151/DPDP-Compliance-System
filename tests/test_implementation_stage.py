from pathlib import Path

import pytest

from orchestrator.implementation_stage import STEP_NAME, run_implementation_stage
from orchestrator.interfaces import ImplementationAnalyzer
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


class FakeReport:
    def __init__(self, failures: int = 0, facts: int = 2):
        self.llm_failures = failures
        self.llm_retries = 0
        self.facts = [object() for _ in range(facts)]

    def model_dump(self, *, mode: str = "python"):
        return {"llm_failures": self.llm_failures, "facts": len(self.facts)}


class ScriptedAgent:
    """One attempt's agent: returns a report or raises an error."""

    def __init__(self, outcome):
        self.outcome = outcome

    def run(self, project_path):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ScriptedFactory:
    """Hands out a fresh agent per call, following a script of outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        return ScriptedAgent(outcome)


def make_state(project_path) -> AssessmentState:
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-1",
            policy_documents=[PolicyDocument(name="p.txt", text="text")],
            project_path=None if project_path is None else str(project_path),
        )
    )


def statuses(state):
    return [(r.status, r.attempt) for r in state.step_log]


def test_first_attempt_success(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(FakeReport(failures=0))

    report = run_implementation_stage(state, factory)

    assert report is not None
    assert factory.calls == 1
    assert statuses(state) == [(StepStatus.COMPLETED, 1)]
    assert state.step_log[0].step == STEP_NAME
    assert state.implementation_report == {"llm_failures": 0, "facts": 2}


def test_retries_when_model_calls_failed_then_succeeds(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(FakeReport(failures=2), FakeReport(failures=0))

    report = run_implementation_stage(state, factory)

    assert report.llm_failures == 0
    assert statuses(state) == [
        (StepStatus.RETRYING, 1),
        (StepStatus.COMPLETED, 2),
    ]


def test_a_fresh_agent_is_built_for_every_attempt(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(
        FakeReport(failures=1), FakeReport(failures=1), FakeReport(failures=1)
    )

    run_implementation_stage(state, factory)

    assert factory.calls == 3


def test_keeps_the_attempt_with_fewest_failures_and_marks_it_incomplete(tmp_path):
    state = make_state(tmp_path)
    best = FakeReport(failures=1)
    factory = ScriptedFactory(FakeReport(failures=3), best, FakeReport(failures=2))

    report = run_implementation_stage(state, factory, max_retries=2)

    assert report is best
    assert state.step_log[-1].status == StepStatus.COMPLETED
    assert state.step_log[-1].attempt == 2
    assert "INCOMPLETE" in state.step_log[-1].message
    assert "not evidence" in state.step_log[-1].message


def test_every_attempt_raising_gives_none_and_failed(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(
        RuntimeError("one"), RuntimeError("two"), RuntimeError("three")
    )

    report = run_implementation_stage(state, factory, max_retries=2)

    assert report is None
    assert state.implementation_report is None
    assert statuses(state) == [
        (StepStatus.RETRYING, 1),
        (StepStatus.RETRYING, 2),
        (StepStatus.FAILED, 3),
    ]
    assert "RuntimeError" in state.step_log[-1].message
    assert "three" in state.step_log[-1].message


def test_recovers_after_an_exception(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(RuntimeError("boom"), FakeReport(failures=0))

    report = run_implementation_stage(state, factory)

    assert report is not None
    assert statuses(state) == [
        (StepStatus.RETRYING, 1),
        (StepStatus.COMPLETED, 2),
    ]


def test_no_project_path_fails_without_calling_the_agent():
    state = make_state(None)
    factory = ScriptedFactory(FakeReport())

    report = run_implementation_stage(state, factory)

    assert report is None
    assert factory.calls == 0
    assert statuses(state) == [(StepStatus.FAILED, 1)]
    assert "No project_path" in state.step_log[0].message


def test_a_missing_folder_fails_without_calling_the_agent(tmp_path):
    state = make_state(Path(tmp_path) / "does-not-exist")
    factory = ScriptedFactory(FakeReport())

    report = run_implementation_stage(state, factory)

    assert report is None
    assert factory.calls == 0
    assert statuses(state) == [(StepStatus.FAILED, 1)]


def test_zero_retries_means_a_single_attempt(tmp_path):
    state = make_state(tmp_path)
    factory = ScriptedFactory(FakeReport(failures=2))

    run_implementation_stage(state, factory, max_retries=0)

    assert factory.calls == 1
    assert "INCOMPLETE" in state.step_log[-1].message


def test_negative_max_retries_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_implementation_stage(
            make_state(tmp_path), ScriptedFactory(FakeReport()), max_retries=-1
        )


def test_waits_between_attempts(tmp_path):
    state = make_state(tmp_path)
    sleeps: list[float] = []
    factory = ScriptedFactory(
        FakeReport(failures=1), FakeReport(failures=1), FakeReport(failures=1)
    )

    run_implementation_stage(
        state,
        factory,
        max_retries=2,
        retry_delay_seconds=0.5,
        sleep=sleeps.append,
    )

    assert sleeps == [0.5, 0.5]


def test_a_simple_fake_satisfies_the_agent_2_contract():
    assert isinstance(ScriptedAgent(FakeReport()), ImplementationAnalyzer)


def test_project_path_is_optional_and_state_round_trips(tmp_path):
    request = AssessmentRequest(
        assessment_id="demo-1",
        policy_documents=[PolicyDocument(name="p.txt", text="text")],
    )
    assert request.project_path is None

    state = make_state(tmp_path)
    state.implementation_report = {"llm_failures": 0, "facts": 2}

    restored = AssessmentState.model_validate_json(state.model_dump_json())

    assert restored == state
    