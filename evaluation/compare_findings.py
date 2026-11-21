
"""Compare Agent 3 claim findings against evaluation ground truth."""

from collections import defaultdict
from collections.abc import Mapping

from agents.policy_agent.claim_id import make_claim_id
from evaluation.schema_mapping import (
    map_expected_category,
    map_expected_status,
)


def _get_value(item, key, default=None):
    """Read a field from either a dictionary or a Pydantic object."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _status_value(status):
    """Convert an enum status or string into its string value."""
    if status is None:
        return None
    return getattr(status, "value", status)


def compare_claim_findings(
    expected_claims: list[dict],
    actual_findings: list,
    policy_source: str,
) -> list[dict]:
    """
    Compare expected claim statuses with Agent 3 claim_check findings.

    Claims are matched using the project's existing claim ID function.
    Findings are not matched by their list position or claim wording.
    """
    findings_by_claim_id = defaultdict(list)

    for finding in actual_findings:
        if _get_value(finding, "kind") != "claim_check":
            continue

        claim_id = _get_value(finding, "claim_id")
        if claim_id:
            findings_by_claim_id[claim_id].append(finding)

    comparisons = []

    for expected in expected_claims:
        category = map_expected_category(expected["category"])
        expected_status = map_expected_status(
            expected["expected_status"]
        )

        claim_id = make_claim_id(
            policy_source,
            category,
            expected["evidence"],
        )

        matching_findings = findings_by_claim_id.get(claim_id, [])

        # A missing or ambiguous result must not be treated as a match.
        actual_statuses = {
            _status_value(_get_value(finding, "status"))
            for finding in matching_findings
        }

        if len(matching_findings) == 1:
            actual_status = next(iter(actual_statuses))
        else:
            actual_status = None

        expected_status_value = expected_status.value

        comparisons.append(
            {
                "key": expected["key"],
                "claim_id": claim_id,
                "category": category.value,
                "expected_status": expected_status_value,
                "actual_status": actual_status,
                "matched": (
                    len(matching_findings) == 1
                    and actual_status == expected_status_value
                ),
                "finding_count": len(matching_findings),
                "reason": expected.get("reason", ""),
            }
        )

    return comparisons
