
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
    structured_model = MagicMock()
    model.with_structured_output.return_value = structured_model

    prompt = MagicMock()
    chain = MagicMock()
    prompt.__or__.return_value = chain

    monkeypatch.setattr(
        agent_module.ChatPromptTemplate,
        "from_messages",
        MagicMock(return_value=prompt),
    )

    agent = PolicyExtractionAgent(model=model)

    return agent, chain, model


def test_extract_returns_validated_policy_claim(mock_agent):
    agent, chain, model = mock_agent

    expected_result = PolicyExtractionResult(
        claims=[
            PolicyClaim(
                claim="Users can withdraw their consent.",
                category=PolicyCategory.CONSENT,
                evidence="Users may withdraw their consent at any time.",
                source="privacy_policy.txt",
            )
        ]
    )

    chain.invoke.return_value = expected_result

    result = agent.extract(
        document_text="Users may withdraw their consent at any time.",
        source="privacy_policy.txt",
    )

    assert result == expected_result
    chain.invoke.assert_called_once()
    model.with_structured_output.assert_called_once_with(
        PolicyExtractionResult
    )


def test_extract_rejects_empty_document(mock_agent):
    agent, chain, _ = mock_agent

    with pytest.raises(ValueError, match="document text cannot be empty"):
        agent.extract(
            document_text="   ",
            source="privacy_policy.txt",
        )

    chain.invoke.assert_not_called()


def test_extract_rejects_empty_source(mock_agent):
    agent, chain, _ = mock_agent

    with pytest.raises(ValueError, match="Source identifier cannot be empty"):
        agent.extract(
            document_text="We collect personal information.",
            source=" ",
        )

    chain.invoke.assert_not_called()


def test_extract_accepts_empty_claim_list(mock_agent):
    agent, chain, _ = mock_agent

    expected_result = PolicyExtractionResult(claims=[])
    chain.invoke.return_value = expected_result

    result = agent.extract(
        document_text="This document contains no privacy-related claims.",
        source="general_document.txt",
    )

    assert result.claims == []
