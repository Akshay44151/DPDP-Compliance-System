
from evaluation.compare_seeded_violations import compare_seeded_violations


def test_detects_unconditional_marketing_email():
    expected = [
        {
            "key": "unconditional_marketing_email",
            "expected_claim": "marketing_consent",
            "evidence_kind": "outbound_email",
            "evidence_match": {
                "subjects_include": "Weekly offers",
                "references_consent": False,
            },
        }
    ]

    evidence = [
        {
            "id": "E001",
            "kind": "outbound_email",
            "detail": {
                "subjects": ["Weekly offers"],
                "references_consent": False,
            },
        }
    ]

    findings = [
        {
            "kind": "claim_check",
            "status": "violated",
            "evidence_ids": ["E001"],
            "claim_id": "marketing-claim",
        }
    ]

    claim_comparisons = [
        {
            "key": "marketing_consent",
            "actual_status": "violated",
            "claim_id": "marketing-claim",
        }
    ]

    result = compare_seeded_violations(
        expected, findings, evidence, claim_comparisons
    )

    assert result["results"][0]["result"] == "detected"
    assert result["counts"]["detected"] == 1


def test_detects_external_advertising_transfer():
    expected = [
        {
            "key": "external_advertising_transfer",
            "expected_claim": "purpose_limitation",
            "evidence_kind": "data_sharing",
            "evidence_match": {
                "target_contains": "ads.example.com",
            },
        }
    ]

    evidence = [
        {
            "id": "E010",
            "kind": "data_sharing",
            "detail": {
                "target": "https://ads.example.com/track",
            },
        }
    ]

    findings = [
        {
            "kind": "claim_check",
            "status": "violated",
            "evidence_ids": ["E010"],
            "claim_id": "purpose-claim",
        }
    ]

    claim_comparisons = [
        {
            "key": "purpose_limitation",
            "actual_status": "violated",
            "claim_id": "purpose-claim",
        }
    ]

    result = compare_seeded_violations(
        expected, findings, evidence, claim_comparisons
    )

    assert result["results"][0]["result"] == "detected"


def test_reports_when_scanner_found_but_agent_missed():
    expected = [
        {
            "key": "unconditional_marketing_email",
            "expected_claim": "marketing_consent",
            "evidence_kind": "outbound_email",
            "evidence_match": {
                "subjects_include": "Weekly offers",
                "references_consent": False,
            },
        }
    ]

    evidence = [
        {
            "id": "E001",
            "kind": "outbound_email",
            "detail": {
                "subjects": ["Weekly offers"],
                "references_consent": False,
            },
        }
    ]

    findings = []

    claim_comparisons = [
        {
            "key": "marketing_consent",
            "actual_status": "violated",
        }
    ]

    result = compare_seeded_violations(
        expected, findings, evidence, claim_comparisons
    )

    assert result["results"][0]["result"] == "scanner_found_agent_missed"


def test_reports_when_scanner_did_not_find_evidence():
    expected = [
        {
            "key": "unconditional_marketing_email",
            "expected_claim": "marketing_consent",
            "evidence_kind": "outbound_email",
            "evidence_match": {
                "subjects_include": "Weekly offers",
                "references_consent": False,
            },
        }
    ]

    result = compare_seeded_violations(
        expected,
        [],
        [],
        [{"key": "marketing_consent", "actual_status": "violated"}],
    )

    assert result["results"][0]["result"] == "not_found_in_scan"


def test_reports_evidence_cited_but_claim_not_violated():
    expected = [
        {
            "key": "external_advertising_transfer",
            "expected_claim": "purpose_limitation",
            "evidence_kind": "data_sharing",
            "evidence_match": {
                "target_contains": "ads.example.com",
            },
        }
    ]

    evidence = [
        {
            "id": "E010",
            "kind": "data_sharing",
            "detail": {
                "target": "https://ads.example.com/track",
            },
        }
    ]

    findings = [
        {
            "kind": "claim_check",
            "status": "partial",
            "evidence_ids": ["E010"],
            "claim_id": "purpose-claim",
        }
    ]

    claim_comparisons = [
        {
            "key": "purpose_limitation",
            "actual_status": "partial",
            "claim_id": "purpose-claim",
        }
    ]

    result = compare_seeded_violations(
        expected, findings, evidence, claim_comparisons
    )

    assert (
        result["results"][0]["result"]
        == "evidence_cited_claim_not_violated"
    )


def test_detects_missing_retention_through_claim_status():
    expected = [
        {
            "key": "missing_retention",
            "expected_claim": "retention",
        }
    ]

    claim_comparisons = [
        {
            "key": "retention",
            "actual_status": "violated",
        }
    ]

    result = compare_seeded_violations(
        expected, [], [], claim_comparisons
    )

    assert result["results"][0]["result"] == "detected"


def test_reports_missing_claim_finding():
    expected = [
        {
            "key": "missing_withdrawal",
            "expected_claim": "consent_withdrawal",
        }
    ]

    result = compare_seeded_violations(
        expected, [], [], []
    )

    assert result["results"][0]["result"] == "claim_finding_missing"


def test_unrelated_finding_evidence_does_not_count_as_detection():
    expected = [
        {
            "key": "unconditional_marketing_email",
            "expected_claim": "marketing_consent",
            "evidence_kind": "outbound_email",
            "evidence_match": {
                "subjects_include": "Weekly offers",
                "references_consent": False,
            },
        }
    ]

    evidence = [
        {
            "id": "E001",
            "kind": "outbound_email",
            "detail": {
                "subjects": ["Weekly offers"],
                "references_consent": False,
            },
        }
    ]

    # An unrelated policy finding cites the matching scanner evidence.
    findings = [
        {
            "kind": "claim_check",
            "claim_id": "unrelated-purpose-claim",
            "status": "violated",
            "evidence_ids": ["E001"],
        }
    ]

    # The expected marketing claim is violated, but its own finding
    # does not cite E001.
    claim_comparisons = [
        {
            "key": "marketing_consent",
            "claim_id": "expected-marketing-claim",
            "actual_status": "violated",
        }
    ]

    result = compare_seeded_violations(
        expected, findings, evidence, claim_comparisons
    )

    assert (
        result["results"][0]["result"]
        == "scanner_found_agent_missed"
    )
    assert result["counts"]["detected"] == 0
