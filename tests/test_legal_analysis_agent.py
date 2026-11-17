import json

from agents.alignment_agent.schemas import (
    AlignmentFinding,
    AlignmentResult,
    AlignmentStatus,
)
from agents.implementation_agent.schemas import (
    CodeLocation,
    EvidenceItem,
    ImplementationFact,
    ImplementationReport,
)
from agents.legal_rag.agent import LegalAnalysisAgent
from agents.legal_rag.schemas import (
    LegalAnalysisRequest,
    LegalAssessmentStatus,
    RetrievedLegalChunk,
)
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, *, k=5):
        return self.chunks


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, **kwargs):
        return json.dumps(self.response)


def make_request(status=AlignmentStatus.VIOLATED, include_fact=True):
    claim = PolicyClaim(
        claim="We obtain consent before processing personal data.",
        category=PolicyCategory.CONSENT,
        evidence="Privacy policy",
        source="privacy.html",
    )

    alignment = AlignmentFinding(
        id="A1",
        kind="claim_check",
        category=PolicyCategory.CONSENT,
        status=status,
        claim=claim,
        fact_ids=["F1"] if include_fact else [],
        reasoning="Consent implementation needs review.",
        confidence=0.9,
    )

    evidence = EvidenceItem(
        id="E1",
        kind="consent_check",
        location=CodeLocation(file="app.py", line=10),
        snippet="user = create_user(email)",
    )

    fact = ImplementationFact(
        id="F1",
        category=PolicyCategory.CONSENT,
        statement="User creation has no visible consent check.",
        fact_type="absent",
        evidence_ids=["E1"],
        locations=[CodeLocation(file="app.py", line=10)],
    )

    report = ImplementationReport(
        project_path="sample",
        files_scanned=1,
        coverage={},
        evidence=[evidence],
        facts=[fact] if include_fact else [],
    )

    return LegalAnalysisRequest(
        alignment_result=AlignmentResult(
            findings=[alignment],
            summary={"violated": 1},
        ),
        implementation_report=report,
    )


def make_chunk():
    return RetrievedLegalChunk(
        source_id="LEGAL-act-P8-R1",
        text="The Data Fiduciary shall protect personal data in its possession or under its control.",
        source_file="dpdp_act.pdf",
        document_type="act",
        authority="primary",
        pdf_page=8,
        provision="Section 8",
    )


def valid_response():
    return {
        "status": "potential_non_compliance",
        "applicable_provisions": ["Section 8"],
        "legal_reasoning": "The retrieved passage establishes a duty to protect personal data. The supplied evidence requires further assessment against that duty.",
        "evidence_fact_ids": ["F1"],
        "citations": [{
            "source_id": "LEGAL-act-P8-R1",
            "provision": "Section 8",
            "document_type": "act",
            "source_file": "dpdp_act.pdf",
            "pdf_page": 8,
            "excerpt": "The Data Fiduciary shall protect personal data in its possession or under its control.",
        }],
        "confidence": 0.7,
        "limitations": ["The retrieved passage does not establish every technical safeguard required."],
    }


def test_legal_agent_returns_validated_finding():
    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(valid_response()),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert len(result.findings) == 1
    assert result.findings[0].status == LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE
    assert result.findings[0].citations[0].provision == "Section 8"
    assert result.findings[0].evidence_fact_ids == ["F1"]
    assert result.sources_consulted == ["dpdp_act.pdf"]


def test_legal_agent_rejects_fabricated_citation():
    response = valid_response()
    response["citations"][0]["excerpt"] = "This text does not exist in the law."

    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(response),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert "could not be validated" in result.findings[0].legal_reasoning


def test_legal_agent_handles_missing_implementation_facts():
    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(valid_response()),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request(include_fact=False))

    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert "No verifiable implementation fact" in result.findings[0].legal_reasoning


def test_out_of_scope_finding_does_not_call_llm_or_retriever():
    class MustNotBeCalled:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Retriever should not be called")

    class MustNotGenerate:
        def generate(self, **kwargs):
            raise AssertionError("LLM should not be called")

    agent = LegalAnalysisAgent(
        llm_client=MustNotGenerate(),
        retriever=MustNotBeCalled(),
    )

    result = agent.run(make_request(status=AlignmentStatus.OUT_OF_SCOPE))

    assert result.findings[0].status == LegalAssessmentStatus.NOT_APPLICABLE
