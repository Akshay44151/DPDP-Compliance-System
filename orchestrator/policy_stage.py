"""Policy stage of the orchestrator (Phase A, step 3).

Runs Agent 1 (through the PolicyExtractor contract) on every policy
document in the assessment state, stores the results, and writes an
audit entry for every document. A failure on one document is recorded
and does not stop the others.

Retries come in step 4 and parallel execution in step 5.
"""

from orchestrator.interfaces import PolicyExtractor
from orchestrator.models import AssessmentState, StepStatus

STEP_NAME = "agent1_policy_extraction"


def run_policy_stage(
    state: AssessmentState,
    extractor: PolicyExtractor,
) -> AssessmentState:
    """Extract claims from each policy document and record what happened."""

    for document in state.request.policy_documents:
        try:
            report = extractor.extract_with_report(
                document_text=document.text,
                source=document.name,
            )
        except Exception as exc:
            state.log(
                STEP_NAME,
                StepStatus.FAILED,
                target=document.name,
                message=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
            continue

        state.policy_claims.extend(report.result.claims)
        state.rejected_claims.extend(report.rejected_claims)
        state.log(
            STEP_NAME,
            StepStatus.COMPLETED,
            target=document.name,
            message=(
                f"{len(report.result.claims)} verified, "
                f"{len(report.rejected_claims)} rejected"
            ),
        )

    return state