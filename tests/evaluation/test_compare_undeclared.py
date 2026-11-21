
import pytest

from evaluation.compare_undeclared import compare_undeclared_practices

def make_evidence(evidence_id="E001", field="phone"):
    return {
        "id": evidence_id,
        "kind": "model_field",
        "snippet": f"{field} = models.CharField()",
        "detail": {
            "field": field,
            "model": "Profile",
            "field_type": "CharField",
        },
    }


def make_practice(key="phone_collection"):
    return {
        "key": key,
        "practice": "Collection of phone numbers",
        "evidence_kind": "model_field",
        "category": "transparency_notice",
    }


def test_detects_undeclared_practice():
    result = compare_undeclared_practices(
        [make_practice()],
        [{
            "kind": "undeclared_behavior",
            "status": "undeclared",
            "evidence_ids": ["E001"],
        }],
        [make_evidence()],
    )

    assert result["counts"]["detected_undeclared"] == 1
    assert result["comparisons"][0]["result"] == "detected_undeclared"


def test_practice_already_reported_under_violation():
    result = compare_undeclared_practices(
        [make_practice()],
        [{
            "kind": "claim_check",
            "status": "violated",
            "evidence_ids": ["E001"],
        }],
        [make_evidence()],
    )

    assert result["comparisons"][0]["result"] == "reported_under_violation"


def test_insufficient_evidence_is_reported_separately():
    result = compare_undeclared_practices(
        [make_practice()],
        [{
            "kind": "undeclared_behavior",
            "status": "insufficient_evidence",
            "evidence_ids": ["E001"],
        }],
        [make_evidence()],
    )

    assert result["counts"]["insufficient_evidence"] == 1


def test_scanned_practice_not_reported_by_agent():
    result = compare_undeclared_practices(
        [make_practice()],
        [],
        [make_evidence()],
    )

    assert result["counts"]["missed"] == 1


def test_expected_practice_not_found_by_scanner():
    result = compare_undeclared_practices(
        [make_practice()],
        [],
        [],
    )

    assert result["counts"]["not_found_in_scan"] == 1


def test_unexpected_undeclared_evidence_is_reported():
    result = compare_undeclared_practices(
        [make_practice()],
        [{
            "kind": "undeclared_behavior",
            "status": "undeclared",
            "evidence_ids": ["E002"],
        }],
        [
            make_evidence("E001", "phone"),
            make_evidence("E002", "email"),
        ],
    )

    assert len(result["unexpected_undeclared_evidence"]) == 1
    assert result["unexpected_undeclared_evidence"][0]["evidence_id"] == "E002"



@pytest.mark.parametrize(
    ("practice_key", "field_name"),
    [
        ("phone_collection", "phone"),
        ("ip_collection", "ip_address"),
        ("device_collection", "device_id"),
        ("aadhaar_collection", "aadhaar_number"),
        ("dob_collection", "date_of_birth"),
    ],
)
def test_matches_personal_data_model_fields(practice_key, field_name):
    evidence = make_evidence(field=field_name)
    practice = make_practice(key=practice_key)

    result = compare_undeclared_practices(
        [practice],
        [],
        [evidence],
    )

    assert result["comparisons"][0]["result"] == "missed"
    assert result["comparisons"][0]["matching_evidence_ids"] == ["E001"]


def test_matches_email_logging():
    evidence = {
        "id": "E010",
        "kind": "personal_data_logged",
        "snippet": "Personal data included in log statement",
        "detail": {
            "personal_tokens": ["email"],
        },
    }

    practice = {
        "key": "email_logging",
        "practice": "Email address logging",
        "evidence_kind": "personal_data_logged",
        "category": "data_minimization",
    }

    result = compare_undeclared_practices(
        [practice],
        [],
        [evidence],
    )

    assert result["comparisons"][0]["result"] == "missed"
    assert result["comparisons"][0]["matching_evidence_ids"] == ["E010"]


def test_does_not_match_unrelated_text_as_ip_address():
    evidence = make_evidence(
        field="description",
    )

    evidence["snippet"] = "shipping_information = models.CharField()"

    practice = make_practice(key="ip_collection")

    result = compare_undeclared_practices(
        [practice],
        [],
        [evidence],
    )

    assert result["comparisons"][0]["result"] == "not_found_in_scan"
    assert result["comparisons"][0]["matching_evidence_ids"] == []


def test_does_not_match_email_logging_without_email_token():
    evidence = {
        "id": "E020",
        "kind": "personal_data_logged",
        "snippet": "User information logged",
        "detail": {
            "personal_tokens": ["phone", "device_id"],
        },
    }

    practice = {
        "key": "email_logging",
        "practice": "Email address logging",
        "evidence_kind": "personal_data_logged",
        "category": "data_minimization",
    }

    result = compare_undeclared_practices(
        [practice],
        [],
        [evidence],
    )

    assert result["comparisons"][0]["result"] == "not_found_in_scan"
