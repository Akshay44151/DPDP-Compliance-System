
from types import SimpleNamespace

from agents.alignment_agent.schemas import AlignmentStatus
from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import PolicyCategory

from evaluation.compare_findings import compare_claim_findings


POLICY_SOURCE = "synthetic_privacy_policy.txt"

EXPECTED_CLAIMS = [
    {
        "key": "marketing_consent",
        "category": "consent",
        "evidence": "We ask for your consent before sending marketing emails.",
        "expected_status": "violated",
        "reason": "Marketing emails must require consent.",
    }
]


def make_finding(status, kind="claim_check"):
    claim_id = make_claim_id(
        POLICY_SOURCE,
        PolicyCategory.CONSENT,
        EXPECTED_CLAIMS[0]["evidence"],
    )

    return SimpleNamespace(
        kind=kind,
        claim_id=claim_id,
        status=status,
    )


def test_matching_status_is_correct():
    actual = [
        make_finding(AlignmentStatus.VIOLATED)
    ]

    result = compare_claim_findings(
        EXPECTED_CLAIMS, actual, POLICY_SOURCE
    )

    assert len(result) == 1
    assert result[0]["actual_status"] == "violated"
    assert result[0]["matched"] is True


def test_different_status_is_incorrect():
    actual = [
        make_finding(AlignmentStatus.ALIGNED)
    ]

    result = compare_claim_findings(
        EXPECTED_CLAIMS, actual, POLICY_SOURCE
    )

    assert result[0]["actual_status"] == "aligned"
    assert result[0]["matched"] is False


def test_missing_finding_is_not_a_match():
    result = compare_claim_findings(
        EXPECTED_CLAIMS, [], POLICY_SOURCE
    )

    assert result[0]["actual_status"] is None
    assert result[0]["finding_count"] == 0
    assert result[0]["matched"] is False


def test_duplicate_findings_are_not_accepted():
    actual = [
        make_finding(AlignmentStatus.VIOLATED),
        make_finding(AlignmentStatus.VIOLATED),
    ]

    result = compare_claim_findings(
        EXPECTED_CLAIMS, actual, POLICY_SOURCE
    )

    assert result[0]["actual_status"] is None
    assert result[0]["finding_count"] == 2
    assert result[0]["matched"] is False


def test_undeclared_behavior_is_ignored():
    actual = [
        make_finding(
            AlignmentStatus.VIOLATED,
            kind="undeclared_behavior",
        )
    ]

    result = compare_claim_findings(
        EXPECTED_CLAIMS, actual, POLICY_SOURCE
    )

    assert result[0]["actual_status"] is None
    assert result[0]["matched"] is False
