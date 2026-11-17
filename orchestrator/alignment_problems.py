"""Count the model problems in one Agent 3 result (Phase A).

Deterministic Python. Reads a result that was already turned into a dict
with model_dump(mode="json"). Every key is optional, so an older or
smaller result still works.

A "model problem" is something a second run might fix:
- a model call gave up (llm_failures, failed_calls), or
- a finding ended as model_failure or invalid_model_output, for example
  because the model answered with a claim id that matches no claim.

Findings that ended as evidence_gap or unsupported_evidence are NOT model
problems. A second run gives the same answer, so they are never retried.
"""

from typing import Any

# Same two reason codes that summary.py reports as model problems.
MODEL_PROBLEM_CODES = ("model_failure", "invalid_model_output")


def count_model_problem_findings(dump: dict[str, Any]) -> int:
    """Count findings whose reason_code says the model failed."""
    findings = dump.get("findings") or []
    return sum(
        1
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("reason_code") in MODEL_PROBLEM_CODES
    )


def problem_score(llm_failures: int, dump: dict[str, Any]) -> int:
    """One number for how much of this result a retry might fix.

    0 means nothing a retry could fix. A higher score is worse.
    """
    failed_calls = dump.get("failed_calls") or []
    return max(llm_failures, len(failed_calls)) + count_model_problem_findings(dump)