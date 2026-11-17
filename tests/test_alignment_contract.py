from orchestrator.interfaces import AlignmentChecker
from orchestrator.models import AssessmentRequest, AssessmentState, PolicyDocument


def _request() -> AssessmentRequest:
    return AssessmentRequest(
        assessment_id="a1",
        policy_documents=[PolicyDocument(name="p.txt", text="hello")],
    )


def test_alignment_result_is_empty_by_default():
    state = AssessmentState(request=_request())
    assert state.alignment_result is None


def test_alignment_result_round_trips_through_json():
    state = AssessmentState(
        request=_request(),
        alignment_result={"findings": [], "summary": {"aligned": 1}},
    )
    restored = AssessmentState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_object_with_run_method_satisfies_alignment_checker():
    class FakeAligner:
        def run(self, policy, implementation, *, policy_text=None):
            return None

    assert isinstance(FakeAligner(), AlignmentChecker)