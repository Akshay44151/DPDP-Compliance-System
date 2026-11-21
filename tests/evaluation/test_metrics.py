
from evaluation.metrics import calculate_metrics


def test_calculate_metrics():
    comparisons = [
        {
            "expected_status": "violated",
            "actual_status": "violated",
            "finding_count": 1,
        },
        {
            "expected_status": "violated",
            "actual_status": "aligned",
            "finding_count": 1,
        },
        {
            "expected_status": "aligned",
            "actual_status": "violated",
            "finding_count": 1,
        },
        {
            "expected_status": "partial",
            "actual_status": "partial",
            "finding_count": 1,
        },
        {
            "expected_status": "out_of_scope",
            "actual_status": "out_of_scope",
            "finding_count": 1,
        },
    ]

    result = calculate_metrics(comparisons)

    assert result["total_claims"] == 5
    assert result["evaluated_claims"] == 4
    assert result["excluded_out_of_scope"] == 1

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1

    assert result["alignment_accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5


def test_missing_findings_are_counted():
    comparisons = [
        {
            "expected_status": "violated",
            "actual_status": None,
            "finding_count": 0,
        }
    ]

    result = calculate_metrics(comparisons)

    assert result["missing_findings"] == 1
    assert result["false_negatives"] == 1
    assert result["alignment_accuracy"] == 0.0


def test_ambiguous_findings_are_counted():
    comparisons = [
        {
            "expected_status": "aligned",
            "actual_status": None,
            "finding_count": 2,
        }
    ]

    result = calculate_metrics(comparisons)

    assert result["ambiguous_findings"] == 1
    assert result["true_negatives"] == 0
    assert result["indeterminate_binary_claims"] == 1
    assert result["alignment_accuracy"] == 0.0


def test_missing_non_violated_findings_are_excluded_from_binary_metrics():
    comparisons = [
        {
            "expected_status": "aligned",
            "actual_status": None,
            "finding_count": 0,
        }
    ]

    result = calculate_metrics(comparisons)

    assert result["missing_findings"] == 1
    assert result["true_negatives"] == 0
    assert result["indeterminate_binary_claims"] == 1


def test_zero_division_is_handled():
    result = calculate_metrics([])

    assert result["alignment_accuracy"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["false_positive_rate"] == 0.0
