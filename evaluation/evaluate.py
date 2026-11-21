
"""Run the evaluation framework for one seeded application."""

import json
from pathlib import Path
from collections.abc import Mapping

from evaluation.compare_findings import compare_claim_findings
from evaluation.compare_undeclared import compare_undeclared_practices
from evaluation.compare_seeded_violations import compare_seeded_violations
from evaluation.metrics import calculate_metrics


def load_ground_truth(path: str | Path) -> dict:
    """Load and validate a ground-truth JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_fields = {
        "schema_version",
        "application",
        "policy_source",
        "expected_claims",
    }

    missing = required_fields - data.keys()
    if missing:
        raise ValueError(
            f"Ground-truth file is missing fields: {sorted(missing)}"
        )

    if not isinstance(data["expected_claims"], list):
        raise ValueError("'expected_claims' must be a list.")

    if not isinstance(data.get("expected_undeclared_practices", []), list):
        raise ValueError(
            "'expected_undeclared_practices' must be a list."
        )

    if not isinstance(data.get("known_seeded_violations", []), list):
        raise ValueError(
            "'known_seeded_violations' must be a list."
        )

    return data


def _get_value(item, key, default=None):
    """Read a field from a dictionary or Pydantic object."""
    if isinstance(item, Mapping):
        return item.get(key, default)

    return getattr(item, key, default)


def _get_findings(alignment_result):
    """Extract Agent 3 findings."""
    return _get_value(alignment_result, "findings", []) or []


def _get_evidence(implementation_result):
    """Extract implementation scanner evidence."""
    return _get_value(implementation_result, "evidence", []) or []


def evaluate_application(
    ground_truth_path: str | Path,
    alignment_result,
    implementation_result=None,
) -> dict:
    """
    Evaluate claim alignment, undeclared practices, and seeded violations.

    Claim-level evaluation uses Agent 3 findings.

    Undeclared-practice and seeded-violation evaluations additionally
    require the implementation report containing scanner evidence.
    """
    ground_truth = load_ground_truth(ground_truth_path)

    findings = _get_findings(alignment_result)

    # 1. Compare expected policy claims with Agent 3 findings.
    comparisons = compare_claim_findings(
        expected_claims=ground_truth["expected_claims"],
        actual_findings=findings,
        policy_source=ground_truth["policy_source"],
    )

    # 2. Calculate claim-level evaluation metrics.
    metrics = calculate_metrics(comparisons)

    undeclared_evaluation = None
    seeded_violation_evaluation = None

    if implementation_result is not None:
        implementation_evidence = _get_evidence(
            implementation_result
        )

        # 3. Evaluate undeclared data practices.
        undeclared_evaluation = compare_undeclared_practices(
            expected_practices=ground_truth.get(
                "expected_undeclared_practices", []
            ),
            actual_findings=findings,
            implementation_evidence=implementation_evidence,
        )

        # 4. Evaluate known seeded violations.
        seeded_violation_evaluation = compare_seeded_violations(
            expected_violations=ground_truth.get(
                "known_seeded_violations", []
            ),
            actual_findings=findings,
            implementation_evidence=implementation_evidence,
            claim_comparisons=comparisons,
        )

    return {
        "application": ground_truth["application"],
        "schema_version": ground_truth["schema_version"],

        # Keep the original output keys for backward compatibility.
        "comparisons": comparisons,
        "metrics": metrics,

        # Grouped claim-level evaluation.
        "claim_evaluation": {
            "comparisons": comparisons,
            "metrics": metrics,
        },

        # Additional evaluation sections.
        "undeclared_practice_evaluation": undeclared_evaluation,
        "seeded_violation_evaluation": seeded_violation_evaluation,
    }
