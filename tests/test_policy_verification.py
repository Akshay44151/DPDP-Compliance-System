from agents.policy_agent.schemas import PolicyCategory, PolicyClaim
from agents.policy_agent.verification import evidence_in_document, split_verified

DOC = (
    "We keep your account data for two years\n"
    "after your last login and then delete it.\n"
    "We use it only for our note-taking service."
)


def make_claim(evidence: str) -> PolicyClaim:
    return PolicyClaim(
        claim="Some claim.",
        category=PolicyCategory.OTHER,
        evidence=evidence,
        source="policy.txt",
    )


def test_exact_quote_passes():
    assert evidence_in_document("We use it only for our note-taking service.", DOC)


def test_quote_spanning_a_line_break_passes():
    assert evidence_in_document(
        "We keep your account data for two years after your last login", DOC
    )


def test_odd_hyphen_character_passes():
    assert evidence_in_document("our note\u2011taking service", DOC)


def test_fabricated_quote_fails():
    assert not evidence_in_document("We never sell your data.", DOC)


def test_changed_word_fails():
    assert not evidence_in_document("We keep your account data for three years", DOC)


def test_empty_evidence_fails():
    assert not evidence_in_document("   ", DOC)


def test_split_separates_real_from_fabricated():
    good = make_claim("We use it only for our note-taking service.")
    bad = make_claim("We never sell your data.")
    verified, rejected = split_verified([good, bad], DOC)
    assert verified == [good]
    assert rejected == [bad]