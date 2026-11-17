from agents.policy_agent.claim_id import make_claim_id
from agents.policy_agent.schemas import PolicyCategory

EVIDENCE = "We keep account data for two years after your last login."


def test_same_inputs_give_the_same_id():
    a = make_claim_id("p.txt", PolicyCategory.STORAGE_RETENTION, EVIDENCE)
    b = make_claim_id("p.txt", PolicyCategory.STORAGE_RETENTION, EVIDENCE)
    assert a == b


def test_id_has_a_prefix_and_a_fixed_length():
    claim_id = make_claim_id("p.txt", PolicyCategory.CONSENT, EVIDENCE)
    assert claim_id.startswith("pc_")
    assert len(claim_id) == 15


def test_formatting_differences_do_not_change_the_id():
    plain = make_claim_id("p.txt", PolicyCategory.CONSENT, "note-taking service here")
    messy = make_claim_id(
        "p.txt", PolicyCategory.CONSENT, "note\u2011taking   service\nhere"
    )
    assert plain == messy


def test_a_different_category_changes_the_id():
    a = make_claim_id("p.txt", PolicyCategory.CONSENT, EVIDENCE)
    b = make_claim_id("p.txt", PolicyCategory.OTHER, EVIDENCE)
    assert a != b


def test_a_different_source_changes_the_id():
    a = make_claim_id("a.txt", PolicyCategory.CONSENT, EVIDENCE)
    b = make_claim_id("b.txt", PolicyCategory.CONSENT, EVIDENCE)
    assert a != b


def test_a_changed_word_changes_the_id():
    a = make_claim_id("p.txt", PolicyCategory.CONSENT, "We keep data for two years.")
    b = make_claim_id("p.txt", PolicyCategory.CONSENT, "We keep data for three years.")
    assert a != b