"""Contract test: our alignment stage against the REAL AlignmentAgent.

The model is a fake (no network, no API key). This checks that the stage
and the agent fit together: the call signature, policy_text, the stored
result, and how failures are logged. It does not test the accuracy of
Agent 3 on a real model.
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from agents.alignment_agent.agent import AlignmentAgent
from agents.implementation_agent.schemas import (
    CodeLocation,
    ImplementationFact,
    ImplementationReport,
)
from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim
from orchestrator.alignment_stage import STEP_NAME, run_alignment_stage
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.policy_text import combined_policy_text

DOC_NAME = "policy.txt"
DOC_TEXT = "We ask for your consent before we collect your email address."
CLAIM_EVIDENCE = "We ask for your consent before we collect your email address."


class FakeChatModel:
    """Stands in for the chat model. Replies are chosen by the first prompt
    line ('CATEGORY: consent' for a verdict call, 'TASK: undeclared' for the
    global pass). A reply is a dict or an Exception that is raised."""

    def __init__(self, replies):
        self.replies = replies
        self.called = []

    def with_structured_output(self, cls):
        def run(prompt_value):
            text = prompt_value.messages[-1].content
            first = text.splitlines()[0]
            if first.startswith("TASK: undeclared"):
                key = "undeclared"
            else:
                key = first.removeprefix("CATEGORY: ")
            self.called.append(key)
            default = {"undeclared": []} if key == "undeclared" else {"verdicts": []}
            out = self.replies.get(key, default)
            if isinstance(out, Exception):
                raise out
            return cls.model_validate(out)

        return RunnableLambda(run)


def make_state(text=DOC_TEXT):
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="contract-1",
            policy_documents=[PolicyDocument(name=DOC_NAME, text=text)],
        )
    )
    state.policy_claims = [
        PolicyClaim(
            claim="Consent is asked before collecting email",
            category="consent",
            evidence=CLAIM_EVIDENCE,
            source=DOC_NAME,
        )
    ]
    return state


def claim_id():
    return make_claim_id(DOC_NAME, PolicyCategory("consent"), CLAIM_EVIDENCE)


def report(with_fact=True):
    facts = []
    if with_fact:
        facts.append(
            ImplementationFact(
                id="F1",
                category=PolicyCategory("consent"),
                statement="Registration view asks for consent before saving.",
                fact_type="present",
                evidence_ids=[],
                locations=[CodeLocation(file="views.py", line=10)],
            )
        )
    return ImplementationReport(
        project_path="x",
        files_scanned=1,
        coverage={},
        evidence=[],
        facts=facts,
    )


def verdict(status):
    return {
        "claim_id": claim_id(),
        "status": status,
        "fact_ids": ["F1"],
        "reasoning": "test",
        "confidence": 0.8,
    }


def run_stage(state, model, implementation, max_policy_chars=30000):
    run_alignment_stage(
        state,
        lambda: AlignmentAgent(model, max_policy_chars=max_policy_chars),
        implementation,
        max_retries=2,
        retry_delay_seconds=0.0,
        sleep=lambda seconds: None,
    )


def claim_checks(state):
    return [
        f for f in state.alignment_result["findings"] if f["kind"] == "claim_check"
    ]


def retrying_entries(state):
    return [
        r
        for r in state.step_log
        if r.step == STEP_NAME and r.status == StepStatus.RETRYING
    ]


def test_stage_passes_policy_text_to_the_real_agent_and_stores_the_result():
    state = make_state()
    model = FakeChatModel({"consent": {"verdicts": [verdict("aligned")]}})

    run_stage(state, model, report())

    result = state.alignment_result
    assert result is not None
    assert result["llm_failures"] == 0
    assert result["policy_text_chars"] == len(combined_policy_text(state).strip())
    assert result["policy_text_truncated"] is False
    checks = claim_checks(state)
    assert len(checks) == 1
    assert checks[0]["status"] == "aligned"
    assert retrying_entries(state) == []


def test_a_long_policy_text_is_cut_by_the_agent_and_reported():
    long_text = DOC_TEXT + " " + ("More policy text. " * 40)
    state = make_state(text=long_text)
    model = FakeChatModel({"consent": {"verdicts": [verdict("aligned")]}})

    run_stage(state, model, report(), max_policy_chars=100)

    result = state.alignment_result
    assert result is not None
    assert result["policy_text_truncated"] is True


def test_model_failure_is_retried_logged_and_never_looks_compliant():
    state = make_state()
    model = FakeChatModel({"consent": OutputParserException("bad output")})

    run_stage(state, model, report())

    result = state.alignment_result
    assert result is not None
    assert result["llm_failures"] >= 1
    assert "verdict:consent" in result["failed_calls"]
    checks = claim_checks(state)
    assert len(checks) == 1
    assert checks[0]["status"] == "unverifiable"
    assert checks[0]["reason_code"] == "model_failure"
    assert len(retrying_entries(state)) >= 1


def test_category_without_facts_is_unverifiable_with_no_model_call_and_no_retry():
    state = make_state()
    model = FakeChatModel({})

    run_stage(state, model, report(with_fact=False))

    result = state.alignment_result
    assert result is not None
    assert result["llm_failures"] == 0
    checks = claim_checks(state)
    assert len(checks) == 1
    assert checks[0]["status"] == "unverifiable"
    assert checks[0]["reason_code"] == "evidence_gap"
    assert "consent" not in model.called
    assert retrying_entries(state) == []