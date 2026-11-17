import pytest
from pydantic import ValidationError

from agents.policy_agent.schemas import PolicyCategory, PolicyClaim
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


def make_request() -> AssessmentRequest:
    return AssessmentRequest(
        assessment_id="demo-1",
        policy_documents=[
            PolicyDocument(name="policy.txt", text="We keep data 2 years.")
        ],
    )


def make_claim() -> PolicyClaim:
    return PolicyClaim(
        claim="Keeps data for two years.",
        category=PolicyCategory.STORAGE_RETENTION,
        evidence="We keep data 2 years.",
        source="policy.txt",
    )


def test_request_requires_at_least_one_document():
    with pytest.raises(ValidationError):
        AssessmentRequest(assessment_id="demo-1", policy_documents=[])


def test_request_rejects_duplicate_document_names():
    with pytest.raises(ValidationError):
        AssessmentRequest(
            assessment_id="demo-1",
            policy_documents=[
                PolicyDocument(name="p.txt", text="a"),
                PolicyDocument(name="p.txt", text="b"),
            ],
        )


def test_request_rejects_blank_assessment_id():
    with pytest.raises(ValidationError):
        AssessmentRequest(
            assessment_id="",
            policy_documents=[PolicyDocument(name="p.txt", text="a")],
        )


def test_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        PolicyDocument(name="policy.txt", text="")


def test_new_state_starts_empty():
    state = AssessmentState(request=make_request())
    assert state.policy_claims == []
    assert state.rejected_claims == []
    assert state.step_log == []


def test_states_do_not_share_lists():
    a = AssessmentState(request=make_request())
    b = AssessmentState(request=make_request())
    a.log("step", StepStatus.COMPLETED)
    assert b.step_log == []


def test_state_can_hold_agent1_claims():
    state = AssessmentState(
        request=make_request(),
        policy_claims=[make_claim()],
        rejected_claims=[make_claim()],
    )
    assert state.policy_claims[0].category == PolicyCategory.STORAGE_RETENTION
    assert len(state.rejected_claims) == 1


def test_log_appends_a_record_with_defaults():
    state = AssessmentState(request=make_request())
    record = state.log(
        "agent1_policy_extraction",
        StepStatus.FAILED,
        target="policy.txt",
        message="boom",
    )
    assert state.step_log == [record]
    assert record.attempt == 1
    assert record.status == StepStatus.FAILED
    assert record.timestamp is not None


def test_state_survives_a_json_round_trip():
    state = AssessmentState(
        request=make_request(), policy_claims=[make_claim()]
    )
    state.log("agent1_policy_extraction", StepStatus.COMPLETED, target="policy.txt")

    restored = AssessmentState.model_validate_json(state.model_dump_json())

    assert restored == state


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        AssessmentState(request=make_request(), surprise="field")