"""Gate stage: run the evidence gate after Agent 3 and store its decisions.

Deterministic Python. Reads state.implementation_report and
state.alignment_result, and writes state.evidence_gate. It never changes
Agent 3's result.

Rules:
- If either result is missing, do nothing. The stage that failed has
  already logged it, so a second FAILED entry would only add noise.
- A crash inside the gate is logged as FAILED and leaves
  state.evidence_gate as None. A missing gate result must never be read
  as "every finding passed".
- The step entry has an empty message unless the gate raised run-level
  warnings, so a normal run adds no warning text to the summary.

CATEGORY_KINDS is a copy of the mapping in agents/implementation_agent/agent.py.
It is copied, not imported, because that module loads the model libraries and
fake mode must not need them. tests/test_gate_stage.py fails if the two differ.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

from orchestrator.evidence_gate import evaluate_findings
from orchestrator.models import AssessmentState, StepStatus

STEP_NAME = "evidence_gate"

CATEGORY_KINDS: dict[str, list[str]] = {
    "transparency_notice": [
        "privacy_notice_link",
        "purpose_text",
    ],
    "consent": ["consent_logic", "consent_ui", "data_write", "outbound_email"],
    "purpose_limitation": [
        "purpose_marker",
        "purpose_text",
        "data_write",
        "data_sharing",
        "outbound_email",
        "personal_data_logged",
    ],
    "data_minimization": [
        "model_field",
        "form_field",
        "form_input",
        "request_input",
        "view_response",
    ],
    "storage_retention": [
        "retention_logic",
        "retention_field",
        "setting",
        "personal_data_logged",
    ],
}


def run_gate_stage(
    state: AssessmentState,
    category_kinds: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Run the evidence gate and store its decisions in state.evidence_gate."""
    if state.implementation_report is None or state.alignment_result is None:
        return

    kinds = CATEGORY_KINDS if category_kinds is None else category_kinds

    try:
        result = evaluate_findings(
            state.implementation_report, state.alignment_result, kinds
        )
        stored = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - a bad gate run must not crash the pipeline
        state.log(
            STEP_NAME,
            StepStatus.FAILED,
            message=f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        return

    stored["counts"] = dict(Counter(d.decision for d in result.decisions))
    state.evidence_gate = stored
    state.log(
        STEP_NAME,
        StepStatus.COMPLETED,
        message=" ".join(result.warnings),
    )