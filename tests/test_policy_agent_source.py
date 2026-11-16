from unittest.mock import MagicMock

import pytest

from agents.policy_agent import agent as agent_module
from agents.policy_agent.agent import PolicyExtractionAgent
from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)


@pytest.fixture
def mock_agent(monkeypatch):
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()

    prompt = MagicMock()
    chain = MagicMock()
    prompt.__or__.return_value = chain

    monkeypatch.setattr(
        agent_module.ChatPromptTemplate,
        "from_messages",
        MagicMock(return_value=prompt),
    )

    return PolicyExtractionAgent(model=model), chain


def make_claim(evidence: str, source: str) -> PolicyClaim:
    return PolicyClaim(
        claim="Some claim.",
        category=PolicyCategory.OTHER,
        evidence=evidence,
        source=source,
    )


def test_source_is_set_by_code_not_by_the_model(mock_agent):
    agent, chain = mock_agent
    chain.invoke.return_value = PolicyExtractionResult(
        claims=[make_claim("We keep data for two years.", "polcy.txt")]
    )

    result = agent.extract(
        document_text="We keep data for two years.",
        source="policy.txt",
    )

    assert result.claims[0].source == "policy.txt"


def test_rejected_claims_also_get_the_trusted_source(mock_agent):
    agent, chain = mock_agent
    chain.invoke.return_value = PolicyExtractionResult(
        claims=[make_claim("We never sell your data.", "wrong.txt")]
    )

    report = agent.extract_with_report(
        document_text="We keep data for two years.",
        source="policy.txt",
    )

    assert report.rejected_claims[0].source == "policy.txt"