
"""Compare known seeded violations with scanner evidence and Agent 3 findings."""

from collections.abc import Mapping


def _get_value(item, key, default=None):
    """Read a field from either a dictionary or a Pydantic model."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalise(value):
    """Normalise text for case-insensitive matching."""
    return "".join(
        character.lower()
        for character in str(value or "")
        if character.isalnum()
    )


def _matches_evidence(expected, evidence):
    """Check whether scanner evidence matches a seeded violation."""
    expected_kind = expected.get("evidence_kind")

    if _get_value(evidence, "kind") != expected_kind:
        return False

    detail = _get_value(evidence, "detail", {}) or {}

    if not isinstance(detail, Mapping):
        return False

    evidence_match = expected.get("evidence_match", {})

    if not isinstance(evidence_match, Mapping):
        return False

    # Match an outbound email using its subject and consent reference.
    if expected_kind == "outbound_email":
        expected_subject = evidence_match.get("subjects_include")

        if expected_subject:
            subjects = detail.get("subjects", [])
            if not any(
                _normalise(expected_subject) in _normalise(subject)
                for subject in subjects
            ):
                return False

        if "references_consent" in evidence_match:
            if detail.get("references_consent") is not evidence_match[
                "references_consent"
            ]:
                return False

        return True

    # Match an external data-sharing call using its target.
    if expected_kind == "data_sharing":
        target_part = evidence_match.get("target_contains")

        if not target_part:
            return False

        return _normalise(target_part) in _normalise(
            detail.get("target", "")
        )

    return False


def compare_seeded_violations(
    expected_violations,
    actual_findings,
    implementation_evidence,
    claim_comparisons,
):
    """
    Compare seeded violations against scanner evidence and Agent 3 findings.

    Evidence-based violations require:
    1. Matching scanner evidence.
    2. The relevant Agent 3 claim finding citing that evidence.
    3. The related policy claim being marked violated.

    Evidence cited by an unrelated claim finding does not count.

    Missing-implementation violations are evaluated using their claim status.
    """
    evidence_by_id = {
        _get_value(item, "id"): item
        for item in implementation_evidence
    }

    comparisons_by_key = {
        item["key"]: item
        for item in claim_comparisons
    }

    results = []

    for expected in expected_violations:
        key = expected["key"]
        expected_claim = expected.get("expected_claim")

        claim_comparison = comparisons_by_key.get(expected_claim)
        claim_status = (
            claim_comparison.get("actual_status")
            if claim_comparison
            else None
        )

        claim_id = (
            claim_comparison.get("claim_id")
            if claim_comparison
            else None
        )

        # Collect evidence only from the relevant Agent 3 claim finding.
        cited_evidence_ids = {
            evidence_id
            for finding in actual_findings
            if (
                _get_value(finding, "kind") == "claim_check"
                and claim_id is not None
                and _get_value(finding, "claim_id") == claim_id
            )
            for evidence_id in (
                _get_value(finding, "evidence_ids", []) or []
            )
        }

        # Missing implementation is assessed through the claim finding.
        if not expected.get("evidence_kind"):
            if claim_status is None:
                result = "claim_finding_missing"
            elif claim_status == "violated":
                result = "detected"
            else:
                result = "claim_not_violated"

            results.append({
                "key": key,
                "expected_claim": expected_claim,
                "claim_status": claim_status,
                "matching_evidence_ids": [],
                "cited_evidence_ids": [],
                "result": result,
            })
            continue

        # Find scanner evidence matching this seeded behaviour.
        matching_evidence_ids = [
            evidence_id
            for evidence_id, evidence in evidence_by_id.items()
            if _matches_evidence(expected, evidence)
        ]

        cited_matches = [
            evidence_id
            for evidence_id in matching_evidence_ids
            if evidence_id in cited_evidence_ids
        ]

        if not matching_evidence_ids:
            result = "not_found_in_scan"
        elif not cited_matches:
            result = "scanner_found_agent_missed"
        elif claim_status != "violated":
            result = "evidence_cited_claim_not_violated"
        else:
            result = "detected"

        results.append({
            "key": key,
            "expected_claim": expected_claim,
            "claim_status": claim_status,
            "matching_evidence_ids": matching_evidence_ids,
            "cited_evidence_ids": cited_matches,
            "result": result,
        })

    counts = {
        "detected": 0,
        "scanner_found_agent_missed": 0,
        "not_found_in_scan": 0,
        "claim_not_violated": 0,
        "claim_finding_missing": 0,
        "evidence_cited_claim_not_violated": 0,
    }

    for item in results:
        counts[item["result"]] += 1

    return {
        "results": results,
        "counts": counts,
        "total": len(results),
    }
