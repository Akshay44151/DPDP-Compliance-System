"""Assessment summary (Phase A).

Reads a finished AssessmentState and answers: what was decided, what was not,
and did anything fail? This is deterministic Python. No LLM is involved.

Purpose: a result with many "insufficient evidence" findings, or with a failed
step, must never look like strong compliance. The summary reports how much was
actually decided and lists every failure it can see.

Rules:
- Failures are read from structured data (llm_failures, failed_calls,
  reason_code, step names and step statuses), never by searching message
  text. Messages are copied only so that a person can read them.
- A missing result is "not assessed", never zero counts that look clean.
- "is_complete" means every step ran and no failure was hidden. It does NOT
  say how much was decided. Read decided_percent for that.
"""

from pydantic import BaseModel, ConfigDict

from orchestrator.alignment_stage import STEP_NAME as ALIGNMENT_STEP  # noqa: F401
from orchestrator.analysis_stage import WARNING_STEP
from orchestrator.implementation_stage import STEP_NAME as IMPLEMENTATION_STEP  # noqa: F401
from orchestrator.models import AssessmentState, StepStatus
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP
from orchestrator.sufficiency import Outcome, count_outcomes, outcome_for_status

# Agent 3 reason codes that mean the MODEL failed (a retry may help).
MODEL_PROBLEM_CODES = ("model_failure", "invalid_model_output")
MESSAGE_LIMIT = 200


class StepIssue(BaseModel):
    """A step whose last log entry is not COMPLETED."""

    model_config = ConfigDict(extra="forbid")

    step: str
    target: str = ""
    status: StepStatus
    message: str = ""


class AssessmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    # None means Agent 3 produced no result: not assessed, not "zero findings".
    outcome_counts: dict[str, int] | None = None
    # Why findings ended as insufficient evidence: reason_code -> count.
    insufficient_by_reason: dict[str, int] = {}
    # Share of in-scope findings that are compliant or gap. None if no data.
    decided_percent: float | None = None
    policy_text_chars: int | None = None
    policy_text_truncated: bool | None = None
    implementation_llm_failures: int | None = None
    alignment_llm_failures: int | None = None
    alignment_failed_calls: list[str] = []
    documents_without_claims: list[str] = []
    failed_steps: list[StepIssue] = []
    warnings: list[str] = []
    # Evidence gate decisions by kind (blocked, review_only, pass, not_gated).
    # None means the gate produced no result.
    gate_counts: dict[str, int] | None = None
    gate_warnings: list[str] = []
    incomplete_reasons: list[str] = []
    is_complete: bool = False


def _short(text: str, limit: int = MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _failed_steps(state: AssessmentState) -> list[StepIssue]:
    """Steps whose LAST log entry is not COMPLETED.

    Agent 1 runs once per document, so it is keyed by document name. Agents 2
    and 3 run once per assessment, so they are keyed by step name only.
    """
    last = {}
    for record in state.step_log:
        if record.step == WARNING_STEP:
            continue
        target = record.target if record.step == POLICY_STEP else ""
        last[(record.step, target)] = record
    return [
        StepIssue(
            step=record.step,
            target=record.target,
            status=record.status,
            message=_short(record.message),
        )
        for record in last.values()
        if record.status != StepStatus.COMPLETED
    ]


def _documents_without_claims(state: AssessmentState) -> list[str]:
    done = {
        record.target
        for record in state.step_log
        if record.step == POLICY_STEP and record.status == StepStatus.COMPLETED
    }
    return [
        document.name
        for document in state.request.policy_documents
        if document.name not in done
    ]


def summarize_assessment(state: AssessmentState) -> AssessmentSummary:
    """Summarise what was decided, what was not, and what failed."""
    failed_steps = _failed_steps(state)
    missing_documents = _documents_without_claims(state)
    warnings = [r.message for r in state.step_log if r.step == WARNING_STEP]
    reasons: list[str] = []

    if failed_steps:
        names = ", ".join(
            issue.step + (f" ({issue.target})" if issue.target else "")
            for issue in failed_steps
        )
        reasons.append(f"{len(failed_steps)} step(s) did not complete: {names}.")
    if missing_documents:
        reasons.append(
            "No claims were extracted from: "
            + ", ".join(missing_documents)
            + ". Their claims were not checked."
        )
    if warnings:
        reasons.append(f"{len(warnings)} analysis warning(s) were logged; see warnings.")

    # Agent 2
    implementation = state.implementation_report
    implementation_failures = None
    if implementation is None:
        reasons.append("Agent 2 produced no report, so the code was not analysed.")
    else:
        implementation_failures = int(implementation.get("llm_failures", 0))
        if implementation_failures > 0:
            reasons.append(
                f"Agent 2 gave up on {implementation_failures} model call(s). "
                "Categories without facts are not evidence that the code does "
                "nothing there."
            )

    # Agent 3
    alignment = state.alignment_result
    outcome_counts = None
    insufficient_by_reason: dict[str, int] = {}
    decided_percent = None
    text_chars = None
    truncated = None
    alignment_failures = None
    failed_calls: list[str] = []

    if alignment is None:
        reasons.append("Agent 3 produced no result, so nothing was assessed.")
    else:
        findings = alignment.get("findings", [])
        counts = count_outcomes(f.get("status", "") for f in findings)
        outcome_counts = {outcome.value: n for outcome, n in counts.items()}

        for f in findings:
            if outcome_for_status(f.get("status", "")) == Outcome.INSUFFICIENT_EVIDENCE:
                code = f.get("reason_code", "unknown")
                insufficient_by_reason[code] = insufficient_by_reason.get(code, 0) + 1

        decided = counts[Outcome.COMPLIANT] + counts[Outcome.GAP]
        in_scope = decided + counts[Outcome.INSUFFICIENT_EVIDENCE]
        if in_scope:
            decided_percent = round(100 * decided / in_scope, 1)
        if not findings:
            reasons.append("Agent 3 returned no findings, so nothing was assessed.")

        alignment_failures = int(alignment.get("llm_failures", 0))
        failed_calls = [str(call) for call in alignment.get("failed_calls", [])]
        text_chars = alignment.get("policy_text_chars")
        truncated = bool(alignment.get("policy_text_truncated", False))

        if alignment_failures > 0 or failed_calls:
            detail = f": {', '.join(failed_calls)}" if failed_calls else ""
            reasons.append(
                f"Agent 3 gave up on {alignment_failures} model call(s){detail}."
            )
        if "undeclared" in failed_calls:
            reasons.append(
                "The undeclared-behaviour check did not run, so 'no undeclared "
                "findings' is not a clean result."
            )
        model_problems = sum(
            insufficient_by_reason.get(code, 0) for code in MODEL_PROBLEM_CODES
        )
        if model_problems:
            reasons.append(
                f"{model_problems} finding(s) could not be assessed because of a "
                "model problem; a retry may fix them."
            )
        if truncated:
            reasons.append(
                f"The policy text was cut ({text_chars} characters before the "
                "cut), so Agent 3 did not see the whole policy and 'undeclared' "
                "findings may be wrong."
            )

    # Evidence gate: checks the evidence behind Agent 3's findings.
    gate = state.evidence_gate
    gate_counts = None
    gate_warnings: list[str] = []
    if gate is not None:
        gate_counts = {
            str(key): int(value) for key, value in (gate.get("counts") or {}).items()
        }
        gate_warnings = [str(w) for w in (gate.get("warnings") or [])]
        blocked = gate_counts.get("blocked", 0)
        if blocked:
            reasons.append(
                f"{blocked} finding(s) were blocked by the evidence gate: the "
                "scan behind them was incomplete or a cited fact was missing, "
                "so they are not reported as gaps."
            )

    return AssessmentSummary(
        assessment_id=state.request.assessment_id,
        outcome_counts=outcome_counts,
        insufficient_by_reason=insufficient_by_reason,
        decided_percent=decided_percent,
        policy_text_chars=text_chars,
        policy_text_truncated=truncated,
        implementation_llm_failures=implementation_failures,
        alignment_llm_failures=alignment_failures,
        alignment_failed_calls=failed_calls,
        documents_without_claims=missing_documents,
        failed_steps=failed_steps,
        warnings=warnings,
        gate_counts=gate_counts,
        gate_warnings=gate_warnings,
        incomplete_reasons=reasons,
        is_complete=not reasons,
    )