import pytest

from orchestrator.sufficiency import (
    STATUS_TO_OUTCOME,
    Agent3Status,
    Outcome,
    outcome_for_status,
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
    }