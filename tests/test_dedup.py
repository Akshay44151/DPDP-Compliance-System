"""Tests for dropping exact duplicate claims. No model, no network."""

from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)
from agents.policy_agent.verification import ExtractionReport
from orchestrator.analysis_stage import run_analysis_stages
from orchestrator.dedup import STEP_NAME, drop_duplicate_claims
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.summary import summarize_assessment

CATEGORY_A, CATEGORY_B = list(PolicyCategory)[:2]
EVIDENCE = "We keep data for two years."


def make_claim(
    source="a.txt", category=CATEGORY_A, evidence=EVIDENCE, text="Data is kept."
) -> PolicyClaim:
    return PolicyClaim(
        claim=text, category=category, evidence=evidence, source=source
    )


def make_state(claims=()) -> AssessmentState:
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="dedup-1",
            policy_documents=[PolicyDocument(name="a.txt", text="text of a")],
            project_path=".",
        )
    )
    state.policy_claims.extend(claims)
    return state


def dedup_entries(state):
    return [r for r in state.step_log if r.step == STEP_NAME]


def test_drops_exact_duplicate_keeps_order_and_logs_it():
    other = make_claim(evidence="Another sentence.")
    state = make_state([make_claim(), make_claim(), other])

    dropped = drop_duplicate_claims(state)

    assert dropped == 1
    assert len(state.policy_claims) == 2
    assert state.policy_claims[1] is other
    entries = dedup_entries(state)
    assert len(entries) == 1
    assert entries[0].status == StepStatus.COMPLETED
    assert entries[0].target == "a.txt"
    assert make_claim_id("a.txt", CATEGORY_A, EVIDENCE) in entries[0].message


def test_same_evidence_in_another_category_is_kept():
    state = make_state([make_claim(category=CATEGORY_A), make_claim(category=CATEGORY_B)])

    assert drop_duplicate_claims(state) == 0
    assert len(state.policy_claims) == 2


def test_same_evidence_in_another_document_is_kept():
    state = make_state([make_claim(source="a.txt"), make_claim(source="b.txt")])

    assert drop_duplicate_claims(state) == 0
    assert len(state.policy_claims) == 2


def test_reworded_claim_with_same_evidence_is_a_duplicate_and_first_is_kept():
    state = make_state([make_claim(text="first"), make_claim(text="second")])

    assert drop_duplicate_claims(state) == 1
    assert [c.claim for c in state.policy_claims] == ["first"]


def test_no_duplicates_means_no_log_entries():
    state = make_state([make_claim(), make_claim(evidence="Another sentence.")])

    assert drop_duplicate_claims(state) == 0
    assert dedup_entries(state) == []


def test_three_copies_give_two_drops_and_two_log_entries():
    state = make_state([make_claim(), make_claim(), make_claim()])

    assert drop_duplicate_claims(state) == 2
    assert len(state.policy_claims) == 1
    assert len(dedup_entries(state)) == 2


class DuplicateExtractor:
    def extract_with_report(self, *, document_text: str, source: str):
        claims = [
            make_claim(source=source, text="one"),
            make_claim(source=source, text="two"),
        ]
        return ExtractionReport(result=PolicyExtractionResult(claims=claims))


class FakeImplReport:
    llm_failures = 0
    llm_retries = 0
    facts: list = []

    def model_dump(self, *, mode: str = "python"):
        return {"llm_failures": 0, "facts": []}


class FakeImplAgent:
    def run(self, project_path):
        return FakeImplReport()


class FakeAlignResult:
    llm_failures = 0
    findings: list = []

    def model_dump(self, *, mode: str = "python"):
        return {"findings": [], "llm_failures": 0}


def test_agent3_gets_deduplicated_claims_and_summary_is_not_hurt():
    state = make_state()
    seen = {}

    class AlignAgent:
        def run(self, policy, implementation, *, policy_text=None):
            seen["claims"] = list(policy.claims)
            return FakeAlignResult()

    run_analysis_stages(
        state, DuplicateExtractor(), lambda: FakeImplAgent(), lambda: AlignAgent()
    )

    assert len(seen["claims"]) == 1
    assert len(dedup_entries(state)) == 1
    summary = summarize_assessment(state)
    assert summary.failed_steps == []
    assert summary.warnings == []