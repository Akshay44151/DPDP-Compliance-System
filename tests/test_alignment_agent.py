import json
from pathlib import Path

from agents.alignment_agent.agent import AlignmentAgent
from agents.alignment_agent.schemas import AlignmentStatus as S
from agents.implementation_agent.schemas import CodeLocation, ImplementationFact, ImplementationReport
from agents.policy_agent.schemas import PolicyCategory as C, PolicyExtractionResult

FIXTURE = Path(__file__).parent / "fixtures" / "agent1_sample_output.json"


def load_policy() -> PolicyExtractionResult:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return PolicyExtractionResult(claims=data["claims"])


def fact(fid, cat, text, ftype="present"):
    return ImplementationFact(id=fid, category=cat, statement=text, fact_type=ftype,
                              locations=[CodeLocation(file="views.py", line=10)])


def report(*facts):
    return ImplementationReport(project_path="x", files_scanned=1, coverage={}, evidence=[],
                                facts=list(facts))


class CategoryLLM:
    """Replies by CATEGORY line in the prompt; records which categories were called."""

    def __init__(self, replies):
        self.replies, self.called = replies, []

    def generate(self, *, system_prompt, user_prompt, response_schema=None):
        cat = user_prompt.splitlines()[0].removeprefix("CATEGORY: ")
        self.called.append(cat)
        reply = self.replies[cat]
        return reply if isinstance(reply, str) else json.dumps(reply)


def by_status(result, status):
    return [f for f in result.findings if f.status == status]


def test_out_of_scope_claims_skip_the_llm_and_in_scope_get_verdicts():
    rep = report(
        fact("F1", C.CONSENT, "Marketing email is sent with no consent check.", "absent"),
        fact("F2", C.STORAGE_RETENTION, "No deletion or retention logic exists.", "absent"),
        fact("F3", C.PURPOSE_LIMITATION, "User email is posted to ads.example.com."),
        fact("F4", C.TRANSPARENCY_NOTICE, "Signup form collects email, name and phone."),
        fact("F5", C.DATA_MINIMIZATION, "Signup collects Aadhaar number and date of birth."),
    )
    llm = CategoryLLM({
        "consent": {"verdicts": [{"claim_id": "C1", "status": "violated", "fact_ids": ["F1"],
                                  "reasoning": "No consent check.", "confidence": 0.9}]},
        "storage_retention": {"verdicts": [{"claim_id": "C1", "status": "violated",
                                            "fact_ids": ["F2"], "reasoning": "No purge."}]},
        "purpose_limitation": {"verdicts": [{"claim_id": "C1", "status": "violated",
                                             "fact_ids": ["F3"], "reasoning": "Ad tracker."}]},
        "transparency_notice": {"verdicts": [{"claim_id": "C1", "status": "partial",
                                              "fact_ids": ["F4"], "reasoning": "Missing items."}]},
        "data_minimization": {"verdicts": [], "undeclared": [
            {"fact_ids": ["F5"], "reasoning": "Aadhaar collected but never disclosed.",
             "confidence": 0.8}]},
    })
    result = AlignmentAgent(llm).run(load_policy(), rep)

    assert len(by_status(result, S.OUT_OF_SCOPE)) == 8
    assert not set(llm.called) & {"data_sharing", "cross_border_transfer", "other"}
    assert len(by_status(result, S.VIOLATED)) == 3
    assert len(by_status(result, S.PARTIAL)) == 1
    und = by_status(result, S.UNDECLARED)
    assert len(und) == 1 and und[0].claim is None and und[0].kind == "undeclared_behavior"
    assert und[0].code_locations[0].file == "views.py"
    assert sum(result.summary.values()) == len(result.findings)


def test_unsupported_verdicts_are_downgraded_and_missing_ones_filled():
    rep = report(fact("F1", C.CONSENT, "Consent is recorded.", "present"))
    llm = CategoryLLM({"consent": {"verdicts": [
        {"claim_id": "C1", "status": "aligned", "fact_ids": ["F999"], "reasoning": "ok"}]}})
    policy = PolicyExtractionResult(claims=[
        {"claim": "We obtain consent.", "category": "consent", "evidence": "e", "source": "s"},
        {"claim": "Consent can be withdrawn.", "category": "consent", "evidence": "e", "source": "s"}])
    result = AlignmentAgent(llm).run(policy, rep)
    assert [f.status for f in result.findings] == [S.UNVERIFIABLE, S.UNVERIFIABLE]
    assert "Downgraded" in result.findings[0].reasoning          # hallucinated fact id
    assert "no verdict" in result.findings[1].reasoning          # claim C2 skipped by model


def test_invalid_json_twice_makes_claims_unverifiable_and_no_facts_skips_llm():
    rep = report(fact("F1", C.CONSENT, "x"))
    llm = CategoryLLM({"consent": "not json"})
    policy = PolicyExtractionResult(claims=[
        {"claim": "We obtain consent.", "category": "consent", "evidence": "e", "source": "s"},
        {"claim": "Data kept 2 years.", "category": "storage_retention", "evidence": "e", "source": "s"}])
    result = AlignmentAgent(llm).run(policy, rep)
    assert llm.called == ["consent", "consent"]                  # one call + one retry
    assert [f.status for f in result.findings] == [S.UNVERIFIABLE, S.UNVERIFIABLE]
    assert "No implementation facts" in result.findings[1].reasoning