import threading

import pytest

from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)
from agents.policy_agent.verification import ExtractionReport
from orchestrator import analysis_stage
from orchestrator.alignment_stage import STEP_NAME as ALIGNMENT_STEP
from orchestrator.analysis_stage import WARNING_STEP, run_analysis_stages
from orchestrator.implementation_stage import STEP_NAME as IMPLEMENTATION_STEP
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP


def make_claim(source: str) -> PolicyClaim:
    return PolicyClaim(
        claim="Some claim.",
        category=PolicyCategory.OTHER,
        evidence="Some evidence.",
        source=source,
    )


class FakeExtractor:
    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.calls: list[str] = []

    def extract_with_report(self, *, document_text: str, source: str):
        self.calls.append(source)
        if source in self.fail_for:
            raise RuntimeError("model failed")
        return ExtractionReport(
            result=PolicyExtractionResult(claims=[make_claim(source)])
        )


class FakeImplReport:
    def __init__(self, failures: int = 0):
        self.llm_failures = failures
        self.llm_retries = 0
        self.facts = [object()]

    def model_dump(self, *, mode: str = "python"):
        return {"llm_failures": self.llm_failures, "facts": 1}


class ImplAgent:
    def __init__(self, outcome):
        self.outcome = outcome

    def run(self, project_path):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ImplFactory:
    """Builds a fresh agent per call. The outcome is a report or an error."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return ImplAgent(self.outcome)


class FakeAlignmentResult:
    llm_failures = 0
    findings: list = []

    def model_dump(self, *, mode: str = "python"):
        return {"findings": [], "llm_failures": 0}


class AlignAgent:
    def __init__(self, seen: dict):
        self.seen = seen

    def run(self, policy, implementation, *, policy_text=None):
        self.seen.update(
            policy=policy, implementation=implementation, policy_text=policy_text
        )
        return FakeAlignmentResult()


class AlignFactory:
    def __init__(self):
        self.calls = 0
        self.seen: dict = {}

    def __call__(self):
        self.calls += 1
        return AlignAgent(self.seen)


def make_state(project_path, *names: str) -> AssessmentState:
    names = names or ("a.txt",)
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-1",
            policy_documents=[
                PolicyDocument(name=n, text=f"text of {n}") for n in names
            ],
            project_path=None if project_path is None else str(project_path),
        )
    )


def entries(state, step):
    return [r for r in state.step_log if r.step == step]


def test_full_run_stores_everything_and_calls_agent3_with_real_inputs(tmp_path):
    state = make_state(tmp_path, "a.txt", "b.txt")
    report = FakeImplReport()
    align = AlignFactory()

    run_analysis_stages(state, FakeExtractor(), ImplFactory(report), align)

    assert [c.source for c in state.policy_claims] == ["a.txt", "b.txt"]
    assert state.implementation_report == {"llm_failures": 0, "facts": 1}
    assert state.alignment_result == {"findings": [], "llm_failures": 0}
    assert align.calls == 1
    assert align.seen["implementation"] is report
    assert align.seen["policy"].claims == state.policy_claims
    assert "=== Document: a.txt ===" in align.seen["policy_text"]
    assert entries(state, WARNING_STEP) == []


def test_agent1_and_agent2_really_run_at_the_same_time(tmp_path):
    barrier = threading.Barrier(2, timeout=2)

    class MeetingExtractor:
        def extract_with_report(self, *, document_text: str, source: str):
            barrier.wait()
            return ExtractionReport(
                result=PolicyExtractionResult(claims=[make_claim(source)])
            )

    class MeetingImplAgent:
        def run(self, project_path):
            barrier.wait()
            return FakeImplReport()

    state = make_state(tmp_path)

    run_analysis_stages(
        state, MeetingExtractor(), lambda: MeetingImplAgent(), AlignFactory()
    )

    assert len(state.step_log) == 3
    assert all(r.status == StepStatus.COMPLETED for r in state.step_log)


def test_agent2_failing_skips_agent3(tmp_path):
    state = make_state(tmp_path)
    impl = ImplFactory(RuntimeError("scan failed"))
    align = AlignFactory()

    run_analysis_stages(state, FakeExtractor(), impl, align)

    assert align.calls == 0
    assert state.alignment_result is None
    assert state.implementation_report is None
    last = entries(state, ALIGNMENT_STEP)[-1]
    assert last.status == StepStatus.FAILED
    assert "Skipped" in last.message
    assert "Agent 2" in last.message


def test_missing_project_path_skips_agent3():
    state = make_state(None)
    align = AlignFactory()

    run_analysis_stages(state, FakeExtractor(), ImplFactory(FakeImplReport()), align)

    assert align.calls == 0
    assert entries(state, IMPLEMENTATION_STEP)[-1].status == StepStatus.FAILED
    assert entries(state, ALIGNMENT_STEP)[-1].status == StepStatus.FAILED


def test_every_policy_document_failing_skips_agent3(tmp_path):
    state = make_state(tmp_path, "a.txt", "b.txt")
    extractor = FakeExtractor(fail_for={"a.txt", "b.txt"})
    align = AlignFactory()

    run_analysis_stages(state, extractor, ImplFactory(FakeImplReport()), align)

    assert state.policy_claims == []
    assert align.calls == 0
    last = entries(state, ALIGNMENT_STEP)[-1]
    assert last.status == StepStatus.FAILED
    assert "Agent 1" in last.message
    assert state.implementation_report is not None


def test_one_failed_document_runs_agent3_with_a_warning(tmp_path):
    state = make_state(tmp_path, "a.txt", "b.txt")
    align = AlignFactory()

    run_analysis_stages(
        state,
        FakeExtractor(fail_for={"b.txt"}),
        ImplFactory(FakeImplReport()),
        align,
    )

    assert align.calls == 1
    assert state.alignment_result is not None
    warning = entries(state, WARNING_STEP)
    assert len(warning) == 1
    assert "b.txt" in warning[0].message


def test_incomplete_agent2_runs_agent3_with_a_warning(tmp_path):
    state = make_state(tmp_path)
    align = AlignFactory()

    run_analysis_stages(
        state, FakeExtractor(), ImplFactory(FakeImplReport(failures=2)), align
    )

    assert align.calls == 1
    warning = entries(state, WARNING_STEP)
    assert len(warning) == 1
    assert "Agent 2" in warning[0].message


def test_retry_limit_is_passed_to_both_stages(tmp_path):
    state = make_state(tmp_path)
    extractor = FakeExtractor(fail_for={"a.txt"})
    impl = ImplFactory(RuntimeError("scan failed"))

    run_analysis_stages(state, extractor, impl, AlignFactory(), max_retries=0)

    assert extractor.calls == ["a.txt"]
    assert impl.calls == 1


def test_an_unexpected_crash_in_a_stage_is_logged_and_agent3_skipped(
    tmp_path, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("stage bug")

    monkeypatch.setattr(analysis_stage, "run_policy_stage", boom)
    state = make_state(tmp_path)
    align = AlignFactory()

    run_analysis_stages(state, FakeExtractor(), ImplFactory(FakeImplReport()), align)

    failed = entries(state, POLICY_STEP)[-1]
    assert failed.status == StepStatus.FAILED
    assert "RuntimeError" in failed.message
    assert "stage bug" in failed.message
    assert align.calls == 0


def test_negative_max_retries_is_rejected_before_anything_runs(tmp_path):
    state = make_state(tmp_path)

    with pytest.raises(ValueError):
        run_analysis_stages(
            state,
            FakeExtractor(),
            ImplFactory(FakeImplReport()),
            AlignFactory(),
            max_retries=-1,
        )

    assert state.step_log == []