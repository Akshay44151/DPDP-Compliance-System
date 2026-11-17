from agents.policy_agent.retention import parse_claim_retention, parse_retention
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim


def pairs(text: str):
    return [(c.period_days, c.trigger) for c in parse_retention(text)]


def test_two_years_after_last_login():
    text = "We keep your account data for two years after your last login and then delete it."
    assert pairs(text) == [(730, "last_login")]


def test_sample_evidence_gives_two_commitments():
    text = (
        "We keep account data for two years after your last login, unless the "
        "law requires us to keep it longer. Backups are deleted within 90 days "
        "of account deletion."
    )
    assert pairs(text) == [(730, "last_login"), (90, "account_deletion")]


def test_phrase_split_across_a_line_break():
    text = "Backups are deleted within 90 days of\naccount deletion."
    assert pairs(text) == [(90, "account_deletion")]


def test_months_from_collection():
    assert pairs("Data is kept for 6 months from collection.") == [(180, "collection")]


def test_singular_digit_year():
    text = "Records are held for 1 year after account deletion."
    assert pairs(text) == [(365, "account_deletion")]


def test_weeks_with_no_trigger_is_other():
    assert pairs("Logs are kept for two weeks.") == [(14, "other")]


def test_hyphenated_duration():
    assert pairs("We apply a 30-day retention period.") == [(30, "other")]


def test_unknown_trigger_is_other():
    assert pairs("We retain backups for 90 days.") == [(90, "other")]


def test_purpose_fulfilled_trigger():
    text = "Data is kept for 90 days after the purpose is fulfilled."
    assert pairs(text) == [(90, "purpose_fulfilled")]


def test_text_without_a_period_gives_nothing():
    assert parse_retention("We delete data when you ask us to.") == []


def test_a_frequency_is_not_a_retention_period():
    text = "We conduct a data protection impact assessment once a year."
    assert parse_retention(text) == []


def test_matching_ignores_capitalisation():
    assert pairs("TWO YEARS AFTER YOUR LAST LOGIN") == [(730, "last_login")]


def test_empty_text_gives_nothing():
    assert parse_retention("") == []
    assert parse_retention("   ") == []


def make_claim(category: PolicyCategory) -> PolicyClaim:
    return PolicyClaim(
        claim="Some claim.",
        category=category,
        evidence="We keep data for two years after your last login.",
        source="policy.txt",
    )


def test_claim_helper_only_parses_retention_claims():
    assert parse_claim_retention(make_claim(PolicyCategory.CONSENT)) == []


def test_claim_helper_parses_a_retention_claim():
    result = parse_claim_retention(make_claim(PolicyCategory.STORAGE_RETENTION))
    assert [(c.period_days, c.trigger) for c in result] == [(730, "last_login")]