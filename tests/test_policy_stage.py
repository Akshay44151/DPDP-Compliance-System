from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)
from agents.policy_agent.verification import ExtractionReport
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.policy_stage import STEP_NAME, run_policy_stage


def make_claim(source: str) -> PolicyClaim:
    return PolicyClaim(
        claim="Some claim.",
        category=PolicyCategory.OTHER,
        evidence="Some evidence.",
        source=source,
    )


def make_state(*names: str) -> AssessmentState:
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-1",
            policy_documents=[
                PolicyDocument(name=n, text=f"text of {n}") for n in names
            ],
        )
    )


class FakeExtractor:
    """Returns one verified claim per document, optionally one rejected."""

    def __init__(self, with_rejected: bool = False):
        self.with_rejected = with_rejected
        self.calls: list[str] = []

    def extract_with_report(self, *, document_text: str, source: str):
        self.calls.append(source)
        rejected = [make_claim(source)] if self.with_rejected else []
        return ExtractionReport(
            result=PolicyExtractionResult(claims=[make_claim(source)]),
            rejected_claims=rejected,
        )


class FailingExtractor:
    """Fails on one named document and succeeds on the others."""

    def __init__(self, bad_name: str):
        self.bad_name = bad_name

    def extract_with_report(self, *, document_text: str, source: str):
        if source == self.bad_name:
            raise RuntimeError("model call failed")
        return ExtractionReport(
            result=PolicyExtractionResult(claims=[make_claim(source)])
        )


def test_claims_from_every_document_are_stored():
    state = make_state("a.txt", "b.txt")
    extractor = FakeExtractor()

    run_policy_stage(state, extractor)

    assert extractor.calls == ["a.txt", "b.txt"]
    assert [c.source for c in state.policy_claims] == ["a.txt", "b.txt"]


def test_each_document_gets_a_completed_log_entry():
    state = make_state("a.txt", "b.txt")

    run_policy_stage(state, FakeExtractor())

    assert [r.target for r in state.step_log] == ["a.txt", "b.txt"]
    assert all(r.step == STEP_NAME for r in state.step_log)
    assert all(r.status == StepStatus.COMPLETED for r in state.step_log)


def test_rejected_claims_are_kept_not_dropped():
    state = make_state("a.txt")

    run_policy_stage(state, FakeExtractor(with_rejected=True))

    assert len(state.policy_claims) == 1
    assert len(state.rejected_claims) == 1
    assert "1 verified, 1 rejected" in state.step_log[0].message


def test_a_failure_is_logged_and_does_not_stop_other_documents():
    state = make_state("a.txt", "bad.txt", "c.txt")

    run_policy_stage(state, FailingExtractor("bad.txt"))

    assert [c.source for c in state.policy_claims] == ["a.txt", "c.txt"]
    statuses = {r.target: r.status for r in state.step_log}
    assert statuses == {
        "a.txt": StepStatus.COMPLETED,
        "bad.txt": StepStatus.FAILED,
        "c.txt": StepStatus.COMPLETED,
    }


def test_failure_entry_records_the_error():
    state = make_state("bad.txt")

    run_policy_stage(state, FailingExtractor("bad.txt"))

    entry = state.step_log[0]
    assert entry.status == StepStatus.FAILED
    assert "RuntimeError" in entry.message
    assert "model call failed" in entry.message
    assert state.policy_claims == []


def test_returns_the_same_state_object():
    state = make_state("a.txt")
    assert run_policy_stage(state, FakeExtractor()) is state