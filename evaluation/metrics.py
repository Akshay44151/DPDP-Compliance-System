
"""Calculate evaluation metrics for Agent 3 findings."""


def calculate_metrics(comparisons: list[dict]) -> dict:
    """
    Calculate alignment accuracy and binary violation-detection metrics.

    Out-of-scope claims are excluded.

    Only 'violated' counts as a detected violation.

    Missing or ambiguous findings for expected non-violated claims
    are excluded from binary metrics because their actual status
    cannot be determined.

    Missing or ambiguous findings for expected violated claims
    count as false negatives because the violation was not
    successfully identified.
    """
    evaluated = [
        item
        for item in comparisons
        if item["expected_status"] != "out_of_scope"
    ]

    excluded = len(comparisons) - len(evaluated)

    tp = fp = fn = tn = 0
    exact_matches = 0
    missing_findings = 0
    ambiguous_findings = 0
    indeterminate_binary_claims = 0

    for item in evaluated:
        expected = item["expected_status"]
        actual = item["actual_status"]
        finding_count = item.get("finding_count", 0)

        if actual == expected:
            exact_matches += 1

        if finding_count == 0:
            missing_findings += 1
        elif finding_count > 1:
            ambiguous_findings += 1

        expected_violated = expected == "violated"

        # Missing or ambiguous results cannot establish
        # a negative classification for non-violated claims.
        if actual is None and not expected_violated:
            indeterminate_binary_claims += 1
            continue

        actual_violated = actual == "violated"

        if expected_violated and actual_violated:
            tp += 1
        elif not expected_violated and actual_violated:
            fp += 1
        elif expected_violated and not actual_violated:
            fn += 1
        else:
            tn += 1

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "total_claims": len(comparisons),
        "evaluated_claims": len(evaluated),
        "excluded_out_of_scope": excluded,
        "exact_matches": exact_matches,
        "alignment_accuracy": safe_divide(
            exact_matches, len(evaluated)
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "missing_findings": missing_findings,
        "ambiguous_findings": ambiguous_findings,
        "indeterminate_binary_claims": indeterminate_binary_claims,
    }
