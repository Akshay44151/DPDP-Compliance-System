"""Agent 2 stage of the orchestrator.

Runs the implementation analysis (Agent 2) on the code project named in
the request. Agent 2 cannot rerun for a single category, so a retry means
a whole new run, with a fresh agent each time. The stage retries while
some of Agent 2's model calls failed, or the call raised an error, and it
keeps the attempt with the fewest failed calls.

Important: when a model call fails, Agent 2 returns NO facts for that
category. That is not evidence that the code does nothing there, so the
stage records the run as INCOMPLETE and the caller must not treat the
missing facts as absence.

The stage returns the real report object, because Agent 3 needs it. Only
a plain dictionary copy goes into the assessment state.
"""

import time
from collections.abc import Callable
from pathlib import Path

from orchestrator.interfaces import ImplementationAnalyzer, ImplementationOutput
from orchestrator.models import AssessmentState, StepStatus

STEP_NAME = "agent2_implementation_analysis"


def run_implementation_stage(
    state: AssessmentState,
    agent_factory: Callable[[], ImplementationAnalyzer],
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ImplementationOutput | None:
    """Run Agent 2, retrying a bounded number of times.

    Returns the best report, or None if no usable report was produced.
    """
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    project_path = state.request.project_path
    if not project_path:
        state.log(
            STEP_NAME,
            StepStatus.FAILED,
            message="No project_path was given, so Agent 2 was not run.",
        )
        return None

    if not Path(project_path).is_dir():
        state.log(
            STEP_NAME,
            StepStatus.FAILED,
            target=project_path,
            message="project_path is not a folder, so Agent 2 was not run.",
        )
        return None

    total_attempts = max_retries + 1
    best: ImplementationOutput | None = None
    best_failures = 0
    best_attempt = 0
    attempts_made = 0
    last_error = ""

    for attempt in range(1, total_attempts + 1):
        attempts_made = attempt
        try:
            report = agent_factory().run(project_path)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if attempt < total_attempts:
                state.log(
                    STEP_NAME,
                    StepStatus.RETRYING,
                    target=project_path,
                    attempt=attempt,
                    message=last_error,
                )
                if retry_delay_seconds > 0:
                    sleep(retry_delay_seconds)
                continue
            break

        last_error = ""
        failures = report.llm_failures
        if best is None or failures < best_failures:
            best, best_failures, best_attempt = report, failures, attempt

        if failures == 0:
            break

        if attempt < total_attempts:
            state.log(
                STEP_NAME,
                StepStatus.RETRYING,
                target=project_path,
                attempt=attempt,
                message=f"{failures} model call(s) failed; running Agent 2 again",
            )
            if retry_delay_seconds > 0:
                sleep(retry_delay_seconds)

    if best is None:
        state.log(
            STEP_NAME,
            StepStatus.FAILED,
            target=project_path,
            attempt=attempts_made,
            message=last_error,
        )
        return None

    state.implementation_report = best.model_dump(mode="json")

    if best_failures > 0:
        message = (
            f"INCOMPLETE: {best_failures} model call(s) failed; categories "
            "without facts are not evidence that the code does nothing."
        )
    else:
        message = f"{len(best.facts)} facts"
    if last_error:
        message += f" Last attempt raised: {last_error}"

    state.log(
        STEP_NAME,
        StepStatus.COMPLETED,
        target=project_path,
        attempt=best_attempt,
        message=message,
    )
    return best