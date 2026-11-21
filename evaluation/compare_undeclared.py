
"""Compare expected undeclared practices with Agent 3 findings."""

from collections.abc import Mapping


def _get_value(item, key, default=None):
    """Read a field from a dictionary or Pydantic object."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _status_value(status):
    """Convert an enum or string status into a string."""
    return getattr(status, "value", status)


def _normalise(value):
    """Normalise text for exact field-name matching."""
    return "".join(
        character.lower()
        for character in str(value or "")
        if character.isalnum()
    )


def _evidence_matches(practice, evidence):
    """Check whether scanner evidence represents an expected practice."""

    expected_kind = practice["evidence_kind"]

    if _get_value(evidence, "kind") != expected_kind:
        return False

    detail = _get_value(evidence, "detail", {}) or {}

    if not isinstance(detail, Mapping):
        detail = {}

    # Match personal data fields using structured scanner evidence.
    if expected_kind == "model_field":

        field_name = _normalise(detail.get("field", ""))

        aliases = {
            "phone_collection": {
                "phone",
                "phonenumber",
                "mobile",
                "mobilenumber",
            },
            "ip_collection": {
                "ip",
                "ipaddress",
            },
            "device_collection": {
                "deviceid",
                "deviceidentifier",
            },
            "aadhaar_collection": {
                "aadhaar",
                "aadhaarnumber",
                "uid",
                "uidnumber",
            },
            "dob_collection": {
                "dateofbirth",
                "dob",
                "birthdate",
            },
        }

        expected_fields = aliases.get(practice["key"], set())

        return field_name in expected_fields

    # Match personal data logging using scanner tokens.
    if expected_kind == "personal_data_logged":

        tokens = detail.get("personal_tokens", [])

        if not isinstance(tokens, (list, tuple, set)):
            return False

        normalized_tokens = {
            _normalise(token)
            for token in tokens
        }

        return bool(
            normalized_tokens & {"email", "emailaddress"}
        )

    return False


def compare_undeclared_practices(
    expected_practices: list[dict],
    actual_findings: list,
    implementation_evidence: list,
) -> dict:
    """
    Compare expected practices against Agent 3 undeclared findings.

    Evidence IDs are resolved against the current ImplementationReport.
    Evidence already cited by a violated claim is treated as reported
    under that violation, not as an undeclared-practice miss.
    """

    evidence_by_id = {
        _get_value(item, "id"): item
        for item in implementation_evidence
        if _get_value(item, "id")
    }

    undeclared_ids = set()
    insufficient_ids = set()
    violated_ids = set()

    # Collect evidence IDs from Agent 3 findings.
    for finding in actual_findings:

        kind = _get_value(finding, "kind")

        status = _status_value(
            _get_value(finding, "status")
        )

        evidence_ids = set(
            _get_value(finding, "evidence_ids", []) or []
        )

        if kind == "claim_check" and status == "violated":
            violated_ids.update(evidence_ids)

        elif kind == "undeclared_behavior":

            if status == "undeclared":
                undeclared_ids.update(evidence_ids)

            elif status == "insufficient_evidence":
                insufficient_ids.update(evidence_ids)

    comparisons = []
    matched_expected_ids = set()

    # Compare every expected practice with scanner evidence.
    for practice in expected_practices:

        candidates = {
            evidence_id
            for evidence_id, evidence in evidence_by_id.items()
            if _evidence_matches(practice, evidence)
        }

        matched_expected_ids.update(candidates)

        undeclared_matches = candidates & undeclared_ids
        insufficient_matches = candidates & insufficient_ids
        violated_matches = candidates & violated_ids

        if undeclared_matches:
            result = "detected_undeclared"

        elif violated_matches:
            result = "reported_under_violation"

        elif insufficient_matches:
            result = "insufficient_evidence"

        elif candidates:
            result = "missed"

        else:
            result = "not_found_in_scan"

        comparisons.append({
            "key": practice["key"],
            "practice": practice["practice"],
            "evidence_kind": practice["evidence_kind"],
            "expected_category": practice.get("category"),
            "result": result,
            "matching_evidence_ids": sorted(candidates),
            "undeclared_evidence_ids": sorted(undeclared_matches),
            "violation_evidence_ids": sorted(violated_matches),
            "insufficient_evidence_ids": sorted(
                insufficient_matches
            ),
        })

    # Find undeclared evidence that does not match an expected practice.
    unexpected = []

    for evidence_id in sorted(
        undeclared_ids - matched_expected_ids
    ):

        evidence = evidence_by_id.get(evidence_id)

        if evidence is None:
            continue

        unexpected.append({
            "evidence_id": evidence_id,
            "evidence_kind": _get_value(evidence, "kind"),
            "snippet": _get_value(evidence, "snippet", ""),
        })

    # Count evaluation outcomes.
    counts = {
        "detected_undeclared": 0,
        "reported_under_violation": 0,
        "insufficient_evidence": 0,
        "missed": 0,
        "not_found_in_scan": 0,
    }

    for item in comparisons:
        counts[item["result"]] += 1

    evaluable = (
        counts["detected_undeclared"]
        + counts["reported_under_violation"]
        + counts["insufficient_evidence"]
        + counts["missed"]
    )

    detection_rate = (
        counts["detected_undeclared"] / evaluable
        if evaluable
        else 0.0
    )

    return {
        "total_expected_practices": len(expected_practices),
        "comparisons": comparisons,
        "counts": counts,
        "unexpected_undeclared_evidence": unexpected,
        "detection_rate": detection_rate,
    }
