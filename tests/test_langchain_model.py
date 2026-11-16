
from unittest.mock import Mock

import pytest

from llm.config import LLMConfig
from llm.langchain_model import get_langchain_model


@pytest.fixture
def chat_groq_factory(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(
        "llm.langchain_model.ChatGroq",
        factory,
    )
    return factory


def test_creates_chatgroq_with_shared_config(chat_groq_factory):
    config = LLMConfig(
        provider="groq",
        model="test-model",
        api_key="test-key",
        temperature=0.2,
        max_retries=3,
    )

    model = get_langchain_model(config)

    chat_groq_factory.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        temperature=0.2,
        max_retries=3,
    )

    assert model is chat_groq_factory.return_value


def test_missing_api_key_fails_before_model_creation(chat_groq_factory):
    config = LLMConfig(
        provider="groq",
        model="test-model",
        api_key=None,
    )

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_langchain_model(config)

    chat_groq_factory.assert_not_called()


def test_missing_model_fails_before_model_creation(chat_groq_factory):
    config = LLMConfig(
        provider="groq",
        model="not-selected",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_langchain_model(config)

    chat_groq_factory.assert_not_called()


def test_unsupported_provider_is_rejected(chat_groq_factory):
    config = LLMConfig(
        provider="ollama",
        model="test-model",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_langchain_model(config)

    chat_groq_factory.assert_not_called()
