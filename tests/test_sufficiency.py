import pytest

from orchestrator.sufficiency import (
    STATUS_TO_OUTCOME,
    Agent3Status,
    Outcome,
    count_outcomes,
    outcome_for_status,
    should_retry_alignment,
)


def test_every_agent3_status_has_a_mapping():
    assert set(STATUS_TO_OUTCOME) == set(Agent3Status)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("aligned", Outcome.COMPLIANT),
        ("partial", Outcome.GAP),
        ("violated", Outcome.GAP),
        ("unverifiable", Outcome.INSUFFICIENT_EVIDENCE),
        ("insufficient_evidence", Outcome.INSUFFICIENT_EVIDENCE),
        ("out_of_scope", Outcome.NOT_ASSESSED),
        ("undeclared", Outcome.INSUFFICIENT_EVIDENCE),
    ],
)
def test_status_string_maps_to_expected_outcome(status, expected):
    assert outcome_for_status(status) == expected


def test_accepts_enum_member():
    assert outcome_for_status(Agent3Status.ALIGNED) == Outcome.COMPLIANT


def test_unknown_status_is_insufficient_evidence():
    assert outcome_for_status("something_new") == Outcome.INSUFFICIENT_EVIDENCE


def test_only_aligned_is_compliant():
    compliant = [s for s, o in STATUS_TO_OUTCOME.items() if o == Outcome.COMPLIANT]
    assert compliant == [Agent3Status.ALIGNED]


def test_status_values_match_agent3_strings():
    assert {s.value for s in Agent3Status} == {
        "aligned",
        "partial",
        "violated",
        "unverifiable",
        "out_of_scope",
        "undeclared",
        "insufficient_evidence",
    }


def test_count_outcomes_counts_each_outcome():
    counts = count_outcomes(
        ["aligned", "aligned", "violated", "partial", "unverifiable", "out_of_scope"]
    )
    assert counts[Outcome.COMPLIANT] == 2
    assert counts[Outcome.GAP] == 2
    assert counts[Outcome.INSUFFICIENT_EVIDENCE] == 1
    assert counts[Outcome.NOT_ASSESSED] == 1


def test_count_outcomes_includes_zero_counts():
    counts = count_outcomes(["aligned"])
    assert set(counts) == set(Outcome)
    assert counts[Outcome.GAP] == 0


def test_count_outcomes_empty():
    assert count_outcomes([]) == {outcome: 0 for outcome in Outcome}


@pytest.mark.parametrize(
    ("failures", "retries_done", "max_retries", "expected"),
    [
        (0, 0, 2, False),  # no model failure: nothing to retry
        (1, 0, 2, True),  # failed, retries left
        (1, 1, 2, True),  # failed, one retry left
        (1, 2, 2, False),  # failed, no retries left
        (3, 0, 0, False),  # retries disabled
    ],
)
def test_should_retry_alignment(failures, retries_done, max_retries, expected):
    assert should_retry_alignment(failures, retries_done, max_retries) == expected