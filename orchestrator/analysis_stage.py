"""Front half of the pipeline: Agents 1 and 2 in parallel, then Agent 3.

Agent 1 (policy extraction) and Agent 2 (code analysis) do not depend on
each other, so they run at the same time. Agent 3 needs both, so it runs
afterwards.

Safeguards, so that a failure never looks like a clean result:
- If Agent 1 produced no usable result for ANY document, the policy is
  unknown (not empty), so Agent 3 is skipped and logged as FAILED.
- If Agent 2 produced no report, Agent 3 is skipped and logged as FAILED.
- If some documents failed, or Agent 2 is incomplete, Agent 3 still runs
  but a warning entry says its "undeclared" findings may be wrong.
- Exact duplicate claims are dropped before Agent 3, and each drop is
  logged.

The stages themselves handle retries. This function only decides what
runs, in what order, and what is skipped.
"""

import time
from collections.abc import Callable

from orchestrator.alignment_stage import STEP_NAME as ALIGNMENT_STEP
from orchestrator.alignment_stage import run_alignment_stage
from orchestrator.dedup import drop_duplicate_claims  # added: dedup step
from orchestrator.gate_stage import run_gate_stage  # added: evidence gate
from orchestrator.implementation_stage import STEP_NAME as IMPLEMENTATION_STEP
from orchestrator.implementation_stage import run_implementation_stage
from orchestrator.interfaces import (
    AlignmentChecker,
    ImplementationAnalyzer,
    PolicyExtractor,
)
from orchestrator.models import AssessmentState, StepStatus
from orchestrator.parallel import run_in_parallel
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP
from orchestrator.policy_stage import run_policy_stage

WARNING_STEP = "analysis_warning"


def _completed_documents(state: AssessmentState) -> set[str]:
    return {
        record.target
        for record in state.step_log
        if record.step == POLICY_STEP and record.status == StepStatus.COMPLETED
    }


def run_analysis_stages(
    state: AssessmentState,
    extractor: PolicyExtractor,
    implementation_factory: Callable[[], ImplementationAnalyzer],
    alignment_factory: Callable[[], AlignmentChecker],
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run Agents 1 and 2 in parallel, then Agent 3, recording everything."""
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    outcomes = run_in_parallel(
        {
            "agent1": lambda: run_policy_stage(
                state,
                extractor,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                sleep=sleep,
            ),
            "agent2": lambda: run_implementation_stage(
                state,
                implementation_factory,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                sleep=sleep,
            ),
        }
    )

    if not outcomes["agent1"].ok:
        state.log(
            POLICY_STEP, StepStatus.FAILED, message=outcomes["agent1"].error
        )

    implementation = None
    if outcomes["agent2"].ok:
        implementation = outcomes["agent2"].value
    else:
        state.log(
            IMPLEMENTATION_STEP,
            StepStatus.FAILED,
            message=outcomes["agent2"].error,
        )

    completed = _completed_documents(state)
    failed_documents = [
        document.name
        for document in state.request.policy_documents
        if document.name not in completed
    ]

    if not completed:
        state.log(
            ALIGNMENT_STEP,
            StepStatus.FAILED,
            message=(
                "Skipped: Agent 1 produced no usable result for any policy "
                "document, so the policy is unknown, not empty."
            ),
        )
        return

    if implementation is None:
        state.log(
            ALIGNMENT_STEP,
            StepStatus.FAILED,
            message=(
                "Skipped: Agent 2 produced no report, so there is nothing "
                "to compare the policy with."
            ),
        )
        return

    warnings: list[str] = []
    if failed_documents:
        warnings.append(
            f"Agent 1 failed on: {', '.join(failed_documents)}. Claims from "
            "these documents are missing, so 'undeclared' findings may be wrong."
        )
    if implementation.llm_failures > 0:
        warnings.append(
            f"Agent 2 is incomplete ({implementation.llm_failures} model "
            "call(s) failed). Categories without facts are not evidence "
            "that the code does nothing."
        )
    if warnings:
        state.log(WARNING_STEP, StepStatus.COMPLETED, message=" ".join(warnings))

    # Added: drop exact duplicate claims before Agent 3, logging each drop.
    drop_duplicate_claims(state)

    run_alignment_stage(
        state,
        alignment_factory,
        implementation,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        sleep=sleep,
    )

    # Added: check the evidence behind Agent 3's findings.
    run_gate_stage(state)