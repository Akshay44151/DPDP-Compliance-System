
import json
from types import SimpleNamespace

import pytest

from agents.alignment_agent.schemas import AlignmentStatus
from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import PolicyCategory

from evaluation.evaluate import (
    evaluate_application,
    load_ground_truth,
)


POLICY_SOURCE = "synthetic_privacy_policy.txt"

EVIDENCE = (
    "We ask for your consent before sending marketing emails. "
    "You may withdraw your consent at any time from your account settings."
)


@pytest.fixture
def ground_truth_file(tmp_path):
    data = {
        "schema_version": "1.0",
        "application": "test_application",
        "policy_source": POLICY_SOURCE,
        "expected_claims": [
            {
                "key": "consent",
                "category": "consent",
                "evidence": EVIDENCE,
                "expected_status": "violated",
                "reason": "Marketing emails require consent.",
            }
        ],
    }

    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_alignment_result(status):
    claim_id = make_claim_id(
        POLICY_SOURCE,
        PolicyCategory.CONSENT,
        EVIDENCE,
    )

    finding = SimpleNamespace(
        kind="claim_check",
        claim_id=claim_id,
        status=status,
    )

    return SimpleNamespace(findings=[finding])


def test_load_ground_truth(ground_truth_file):
    data = load_ground_truth(ground_truth_file)

    assert data["application"] == "test_application"
    assert len(data["expected_claims"]) == 1


def test_evaluate_application_correct_result(ground_truth_file):
    alignment_result = make_alignment_result(
        AlignmentStatus.VIOLATED
    )

    result = evaluate_application(
        ground_truth_file,
        alignment_result,
    )

    assert result["application"] == "test_application"
    assert result["metrics"]["alignment_accuracy"] == 1.0
    assert result["metrics"]["true_positives"] == 1
    assert result["comparisons"][0]["matched"] is True


def test_evaluate_application_incorrect_result(ground_truth_file):
    alignment_result = make_alignment_result(
        AlignmentStatus.ALIGNED
    )

    result = evaluate_application(
        ground_truth_file,
        alignment_result,
    )

    assert result["metrics"]["alignment_accuracy"] == 0.0
    assert result["metrics"]["false_negatives"] == 1
    assert result["comparisons"][0]["matched"] is False


def test_evaluate_application_with_dictionary_result(
    ground_truth_file,
):
    alignment_result = {
        "findings": [
            {
                "kind": "claim_check",
                "claim_id": make_claim_id(
                    POLICY_SOURCE,
                    PolicyCategory.CONSENT,
                    EVIDENCE,
                ),
                "status": "violated",
            }
        ]
    }

    result = evaluate_application(
        ground_truth_file,
        alignment_result,
    )

    assert result["metrics"]["alignment_accuracy"] == 1.0


def test_invalid_ground_truth_raises_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"application": "missing_fields"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing fields"):
        load_ground_truth(path)


def test_evaluate_application_integrates_seeded_violations():
    from pathlib import Path

    ground_truth_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "ground_truth"
        / "acmenotes_violating.json"
    )

    alignment_result = {
        "findings": [
            {
                "kind": "claim_check",
                "claim_id": make_claim_id(
                    POLICY_SOURCE,
                    PolicyCategory.CONSENT,
                    EVIDENCE,
                ),
                "status": "violated",
                "evidence_ids": ["E001"],
            },
            {
                "kind": "claim_check",
                "claim_id": make_claim_id(
                    POLICY_SOURCE,
                    PolicyCategory.PURPOSE_LIMITATION,
                    (
                        "We use this information only to provide and improve "
                        "our note-taking service."
                    ),
                ),
                "status": "violated",
                "evidence_ids": ["E002"],
            },
        ]
    }

    implementation_result = {
        "evidence": [
            {
                "id": "E001",
                "kind": "outbound_email",
                "detail": {
                    "subjects": ["Weekly offers"],
                    "references_consent": False,
                },
            },
            {
                "id": "E002",
                "kind": "data_sharing",
                "detail": {
                    "target": "https://ads.example.com/track",
                },
            },
        ]
    }

    result = evaluate_application(
        ground_truth_path,
        alignment_result,
        implementation_result,
    )

    assert result["application"] == "acmenotes_violating"

    assert result["claim_evaluation"] is not None
    assert result["undeclared_practice_evaluation"] is not None
    assert result["seeded_violation_evaluation"] is not None

    seeded_results = {
        item["key"]: item["result"]
        for item in result["seeded_violation_evaluation"]["results"]
    }

    assert seeded_results["unconditional_marketing_email"] == "detected"
    assert seeded_results["external_advertising_transfer"] == "detected"

    assert seeded_results["missing_retention"] == "claim_finding_missing"
    assert (
        seeded_results["missing_withdrawal"]
        == "not_independently_evaluable"
    )

    # All entries, including unevaluable entries.
    assert result["seeded_violation_evaluation"]["total"] == 4

    # Only entries that can be independently evaluated.
    assert (
        result["seeded_violation_evaluation"]["independently_evaluable"]
        == 3
    )
