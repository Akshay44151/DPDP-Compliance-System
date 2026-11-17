"""Agent 3 stage: run the alignment check with retries.

Deterministic Python. The orchestrator calls Agent 3 through the
AlignmentChecker contract, so tests can use a fake.

Rules:
- A fresh agent is built for every attempt, because Agent 3 keeps its
  failure counters on the object.
- Retry when the model failed (llm_failures > 0) or the call raised.
- Keep the result with the fewest model failures.
- A crash or an empty result is never read as "compliant". If every
  attempt raises, alignment_result stays None.
"""

import time
from collections.abc import Callable
from typing import Any

from agents.policy_agent.schemas import PolicyExtractionResult
from orchestrator.interfaces import AlignmentChecker
from orchestrator.models import AssessmentState, StepStatus
from orchestrator.policy_text import combined_policy_text
from orchestrator.sufficiency import should_retry_alignment

STEP_NAME = "agent3_alignment"


def run_alignment_stage(
    state: AssessmentState,
    agent_factory: Callable[[], AlignmentChecker],
    implementation: Any,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run Agent 3 and store its result in state.alignment_result."""
    policy = PolicyExtractionResult(claims=state.policy_claims)
    policy_text = combined_policy_text(state)

    best = None
    last_error = ""
    attempt = 0

    while True:
        attempt += 1
        retries_done = attempt - 1
        try:
            result = agent_factory().run(
                policy, implementation, policy_text=policy_text
            )
        except Exception as exc:  # noqa: BLE001 - a bad run must not crash the pipeline
            last_error = f"{type(exc).__name__}: {exc}"
            if retries_done < max_retries:
                state.log(
                    STEP_NAME,
                    StepStatus.RETRYING,
                    attempt=attempt,
                    message=last_error,
                )
                sleep(retry_delay_seconds)
                continue
            break

        if best is None or result.llm_failures < best.llm_failures:
            best = result

        if should_retry_alignment(result.llm_failures, retries_done, max_retries):
            state.log(
                STEP_NAME,
                StepStatus.RETRYING,
                attempt=attempt,
                message=(
                    f"{result.llm_failures} model call(s) failed; "
                    "re-running Agent 3."
                ),
            )
            sleep(retry_delay_seconds)
            continue
        break

    if best is None:
        state.log(
            STEP_NAME,
            StepStatus.FAILED,
            attempt=attempt,
            message=last_error,
        )
        return

    state.alignment_result = best.model_dump(mode="json")
    message = ""
    if best.llm_failures > 0:
        message = (
            f"{best.llm_failures} model call(s) still failed after retries; "
            "the undeclared check may be incomplete."
        )
    state.log(
        STEP_NAME,
        StepStatus.COMPLETED,
        attempt=attempt,
        message=message,
    )