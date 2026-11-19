"""Tests for orchestrator/gate_stage.py. No model, key or network needed."""

from agents.implementation_agent.agent import CATEGORY_KINDS as AGENT_KINDS
from orchestrator.gate_stage import CATEGORY_KINDS, STEP_NAME, run_gate_stage
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


def make_state(implementation_report=None, alignment_result=None) -> AssessmentState:
    return AssessmentState(
        request=AssessmentRequest(
            assessment_id="t",
            policy_documents=[PolicyDocument(name="p.txt", text="text")],
            project_path=".",
        ),
        implementation_report=implementation_report,
        alignment_result=alignment_result,
    )


def impl_report():
    return {
        "files_scanned": 3,
        "coverage": {"consent_logic": 1},
        "facts": [
            {"id": "F-CONSENT-01", "category": "consent", "fact_type": "present"},
        ],
        "llm_failures": 0,
        "failed_categories": [],
    }


def align_result(**overrides):
    base = {
        "findings": [
            {
                "id": "A-1",
                "kind": "claim_check",
                "category": "consent",
                "status": "violated",
                "fact_ids": ["F-CONSENT-01"],
            }
        ],
        "failed_calls": [],
        "policy_text_truncated": False,
    }
    base.update(overrides)
    return base


def test_category_kinds_copy_matches_agent_2():
    assert CATEGORY_KINDS == {c.value: k for c, k in AGENT_KINDS.items()}


def test_stage_stores_decisions_and_counts():
    state = make_state(impl_report(), align_result())
    run_gate_stage(state)
    assert state.evidence_gate["decisions"][0]["decision"] == "pass"
    assert state.evidence_gate["counts"] == {"pass": 1}


def test_clean_run_logs_completed_with_empty_message():
    state = make_state(impl_report(), align_result())
    run_gate_stage(state)
    entry = state.step_log[-1]
    assert entry.step == STEP_NAME
    assert entry.status == StepStatus.COMPLETED
    assert entry.message == ""


def test_run_warning_goes_into_the_log_message():
    state = make_state(impl_report(), align_result(policy_text_truncated=True))
    run_gate_stage(state)
    assert "policy text" in state.step_log[-1].message


def test_missing_results_do_nothing():
    for impl, align in [(None, align_result()), (impl_report(), None), (None, None)]:
        state = make_state(impl, align)
        run_gate_stage(state)
        assert state.evidence_gate is None
        assert state.step_log == []


def test_gate_crash_is_logged_and_never_looks_like_a_pass():
    state = make_state(impl_report(), align_result())
    run_gate_stage(state, category_kinds=object())  # not a mapping: forces a crash
    assert state.evidence_gate is None
    assert state.step_log[-1].status == StepStatus.FAILED


def test_alignment_result_is_not_changed():
    state = make_state(impl_report(), align_result())
    before = repr(state.alignment_result)
    run_gate_stage(state)
    assert repr(state.alignment_result) == before