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
    """Create an Agent 1 instance without calling the Groq API."""

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


REAL = PolicyClaim(
    claim="Data is kept for two years.",
    category=PolicyCategory.STORAGE_RETENTION,
    evidence="We keep data for two years.",
    source="policy.txt",
)

FAKE = PolicyClaim(
    claim="Data is never sold.",
    category=PolicyCategory.DATA_SHARING,
    evidence="We never sell your data.",
    source="policy.txt",
)


def test_report_separates_fabricated_evidence(mock_agent):
    agent, chain = mock_agent
    chain.invoke.return_value = PolicyExtractionResult(claims=[REAL, FAKE])

    report = agent.extract_with_report(
        document_text="We keep data for two years.",
        source="policy.txt",
    )

    assert report.result.claims == [REAL]
    assert report.rejected_claims == [FAKE]


def test_extract_returns_only_verified_claims(mock_agent):
    agent, chain = mock_agent
    chain.invoke.return_value = PolicyExtractionResult(claims=[FAKE])

    result = agent.extract(
        document_text="We keep data for two years.",
        source="policy.txt",
    )

    assert result.claims == []