import json
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from agents.alignment_agent.agent import AlignmentAgent
from agents.alignment_agent.schemas import AlignmentStatus as S
from agents.implementation_agent.agent import ImplementationExtractionAgent
from agents.implementation_agent.schemas import (CodeLocation, EvidenceItem, ImplementationFact,
                                                 ImplementationReport)
from agents.policy_agent.schemas import PolicyCategory as C, PolicyExtractionResult

FIXTURE = Path(__file__).parent / "fixtures" / "agent1_sample_output.json"


def load_policy() -> PolicyExtractionResult:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return PolicyExtractionResult(claims=data["claims"])


def fact(fid, cat, text, ftype="present", ev=None):
    return ImplementationFact(id=fid, category=cat, statement=text, fact_type=ftype,
                              evidence_ids=ev or [],
                              locations=[CodeLocation(file="views.py", line=10)])


def ev(eid, kind, line, **detail):
    return EvidenceItem(id=eid, kind=kind, location=CodeLocation(file="app.py", line=line),
                        snippet="code", detail=detail)


def report(*facts, evidence=()):
    return ImplementationReport(project_path="x", files_scanned=1, coverage={},
                                evidence=list(evidence), facts=list(facts))


def verdict(status, ids, reasoning="r", confidence=0.8, claim_id="C1"):
    return {"claim_id": claim_id, "status": status, "fact_ids": ids,
            "reasoning": reasoning, "confidence": confidence}


class KeyedChatModel:
    """Stands in for ChatGroq. Replies are keyed by the first prompt line:
    'CATEGORY: x' for verdict calls and 'TASK: undeclared' for the global pass.
    A reply may be a dict (validated into the requested class) or an Exception (raised)."""

    def __init__(self, replies):
        self.replies, self.called, self.prompts = replies, [], {}

    def with_structured_output(self, cls):
        def run(prompt_value):
            text = prompt_value.messages[-1].content
            first = text.splitlines()[0]
            key = "undeclared" if first.startswith("TASK: undeclared") else first.removeprefix("CATEGORY: ")
            self.called.append(key)
            self.prompts[key] = text
            default = {"undeclared": []} if key == "undeclared" else {"verdicts": []}
            out = self.replies.get(key, default)
            if isinstance(out, Exception):
                raise out
            return cls.model_validate(out)
        return RunnableLambda(run)


def by_status(result, status):
    return [f for f in result.findings if f.status == status]


def test_full_flow_out_of_scope_skips_llm_and_undeclared_uses_raw_evidence():
    evidence = [
        ev("E5", "model_field", 9, model="Profile", field="aadhaar_number", field_type="CharField"),
        ev("E6", "model_field", 10, model="Profile", field="date_of_birth", field_type="DateField"),
        ev("E7", "data_sharing", 20, call="requests.post"),
    ]
    rep = report(
        fact("F1", C.CONSENT, "Marketing email is sent with no consent check.", "absent"),
        fact("F2", C.STORAGE_RETENTION, "No deletion or retention logic exists.", "absent"),
        fact("F3", C.PURPOSE_LIMITATION, "User email is posted to ads.example.com.", ev=["E7"]),
        # one fact bundles declared AND undeclared fields; the verdict citing it must not hide E5/E6
        fact("F4", C.TRANSPARENCY_NOTICE, "Profile stores email, name, Aadhaar and date of birth.",
             ev=["E5", "E6"]),
        evidence=evidence,
    )
    model = KeyedChatModel({
        "consent": {"verdicts": [verdict("violated", ["F1"], "No consent check.", 0.9)]},
        "storage_retention": {"verdicts": [verdict("violated", ["F2"], "No purge.")]},
        "purpose_limitation": {"verdicts": [verdict("violated", ["F3"], "Ad tracker.")]},
        "transparency_notice": {"verdicts": [verdict("partial", ["F4"], "Missing items.")]},
        "undeclared": {"undeclared": [
            {"evidence_ids": ["E5", "E6"], "reasoning": "Aadhaar and DOB never disclosed.", "confidence": 0.8},
            {"evidence_ids": ["E5"], "reasoning": "repeat of the item above", "confidence": 0.5},
            {"evidence_ids": ["E7"], "reasoning": "already a violation", "confidence": 0.5},
            {"evidence_ids": ["E999"], "reasoning": "invented id", "confidence": 0.5},
        ]},
    })
    result = AlignmentAgent(model).run(load_policy(), rep)

    assert len(by_status(result, S.OUT_OF_SCOPE)) == 8
    assert not set(model.called) & {"data_sharing", "cross_border_transfer", "other"}
    assert model.called.count("undeclared") == 1                    # one global pass
    assert len(by_status(result, S.VIOLATED)) == 3
    assert len(by_status(result, S.PARTIAL)) == 1
    und = by_status(result, S.UNDECLARED)
    assert len(und) == 1 and und[0].evidence_ids == ["E5", "E6"] and und[0].fact_ids == ["F4"]
    assert und[0].claim is None and und[0].kind == "undeclared_behavior"
    assert und[0].reason_code == "undeclared_practice"
    assert und[0].category == C.TRANSPARENCY_NOTICE
    assert [l.line for l in und[0].code_locations] == [9, 10]
    prompt = model.prompts["undeclared"]
    assert "[transparency_notice]" in prompt and "[data_sharing]" in prompt   # whole policy in view
    assert "E5 [model_field]" in prompt and "ALREADY REPORTED: E7" in prompt
    assert len({f.id for f in result.findings}) == len(result.findings)       # finding ids are unique
    assert result.llm_failures == 0 and result.llm_retries == 0 and result.failed_calls == []
    assert sum(result.summary.values()) == len(result.findings)


def test_unsupported_verdicts_are_downgraded_and_missing_ones_filled():
    rep = report(fact("F1", C.CONSENT, "Consent is recorded.", "present"))
    model = KeyedChatModel({"consent": {"verdicts": [verdict("aligned", ["F999"], "ok")]}})
    policy = PolicyExtractionResult(claims=[
        {"claim": "We obtain consent.", "category": "consent", "evidence": "e", "source": "s"},
        {"claim": "Consent can be withdrawn.", "category": "consent", "evidence": "e", "source": "s"}])
    result = AlignmentAgent(model).run(policy, rep)
    claim_checks = [f for f in result.findings if f.kind == "claim_check"]
    assert [f.status for f in claim_checks] == [S.UNVERIFIABLE, S.UNVERIFIABLE]
    assert "Downgraded" in claim_checks[0].reasoning          # hallucinated fact id
    assert "no verdict" in claim_checks[1].reasoning          # claim C2 skipped by model
    assert claim_checks[0].reason_code == "unsupported_evidence"
    assert claim_checks[1].reason_code == "invalid_model_output"


def test_unusable_output_is_counted_and_makes_claims_unverifiable():
    rep = report(fact("F1", C.CONSENT, "x"))
    model = KeyedChatModel({"consent": OutputParserException("bad json")})
    policy = PolicyExtractionResult(claims=[
        {"claim": "We obtain consent.", "category": "consent", "evidence": "e", "source": "s"},
        {"claim": "Data kept 2 years.", "category": "storage_retention", "evidence": "e", "source": "s"}])
    agent = AlignmentAgent(model)
    result = agent.run(policy, rep)
    assert model.called.count("consent") == 2                 # one call + one retry
    assert [f.status for f in result.findings] == [S.UNVERIFIABLE, S.UNVERIFIABLE]
    assert "No implementation facts" in result.findings[1].reasoning
    assert result.findings[1].reason_code == "evidence_gap"
    assert result.llm_retries == 1 and result.llm_failures == 1 and agent.last_error
    assert result.findings[0].reason_code == "model_failure"
    assert result.failed_calls == ["verdict:consent"]


def test_failed_global_undeclared_pass_is_reported_in_failed_calls():
    model = KeyedChatModel({"undeclared": OutputParserException("bad json")})
    policy = PolicyExtractionResult(claims=[])
    rep = report(evidence=[ev("E1", "model_field", 9, model="Profile", field="email")])

    result = AlignmentAgent(model).run(policy, rep)

    assert model.called.count("undeclared") == 2
    assert not by_status(result, S.UNDECLARED)
    assert result.llm_failures == 1
    assert result.failed_calls == ["undeclared"]


def test_agents_build_on_the_real_chatgroq_model_without_network():
    """Catches schema/structured-output wiring errors at construction time (no API call)."""
    from llm.config import LLMConfig
    from llm.langchain_model import get_langchain_model

    model = get_langchain_model(LLMConfig(model="openai/gpt-oss-120b", api_key="dummy-key"))
    assert ImplementationExtractionAgent(model).chain is not None
    assert AlignmentAgent(model).verdict_chain is not None