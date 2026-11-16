"""Policy stage of the orchestrator (Phase A, steps 3 and 4).

Runs Agent 1 (through the PolicyExtractor contract) on every policy
document in the assessment state, stores the results, and writes an
audit entry for every attempt. A failed attempt is retried a bounded
number of times. A document that still fails is recorded as FAILED and
does not stop the other documents.

Parallel execution comes in step 5.
"""

import time
from collections.abc import Callable

from orchestrator.interfaces import PolicyExtractor
from orchestrator.models import AssessmentState, StepStatus

STEP_NAME = "agent1_policy_extraction"


def run_policy_stage(
    state: AssessmentState,
    extractor: PolicyExtractor,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> AssessmentState:
    """Extract claims from each policy document and record what happened.

    max_retries is the number of extra attempts after the first one, so
    the default of 2 allows up to 3 attempts per document.
    """
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    total_attempts = max_retries + 1

    for document in state.request.policy_documents:
        for attempt in range(1, total_attempts + 1):
            try:
                report = extractor.extract_with_report(
                    document_text=document.text,
                    source=document.name,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"

                if attempt < total_attempts:
                    state.log(
                        STEP_NAME,
                        StepStatus.RETRYING,
                        target=document.name,
                        attempt=attempt,
                        message=error,
                    )
                    if retry_delay_seconds > 0:
                        sleep(retry_delay_seconds)
                    continue

                state.log(
                    STEP_NAME,
                    StepStatus.FAILED,
                    target=document.name,
                    attempt=attempt,
                    message=error,
                )
                break

            state.policy_claims.extend(report.result.claims)
            state.rejected_claims.extend(report.rejected_claims)
            state.log(
                STEP_NAME,
                StepStatus.COMPLETED,
                target=document.name,
                attempt=attempt,
                message=(
                    f"{len(report.result.claims)} verified, "
                    f"{len(report.rejected_claims)} rejected"
                ),
            )
            break

    return state