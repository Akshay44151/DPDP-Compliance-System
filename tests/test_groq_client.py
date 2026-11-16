from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm.config import LLMConfig
from llm.groq_client import (
    GroqLLMClient,
    MalformedJSONError,
    ResponseSchemaError,
)


@pytest.fixture
def groq_factory(monkeypatch):
    """Replace the SDK constructor so tests never contact Groq."""
    factory = Mock()
    monkeypatch.setattr("llm.groq_client.Groq", factory)
    return factory


def make_completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_client(groq_factory, content):
    sdk_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=Mock(return_value=make_completion(content)))
        )
    )
    groq_factory.return_value = sdk_client
    config = LLMConfig(api_key="test-key", model="test-model")
    return GroqLLMClient(config), sdk_client


def test_ordinary_text_is_returned_unchanged(groq_factory):
    client, sdk_client = make_client(groq_factory, "  Just text.  ")

    result = client.generate(system_prompt="system", user_prompt="user")

    assert result == "  Just text.  "
    assert "response_format" not in sdk_client.chat.completions.create.call_args.kwargs


def test_valid_json_is_validated_and_returned_as_json_text(groq_factory):
    client, sdk_client = make_client(groq_factory, '{"name":"Asha","age":20}')
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    result = client.generate(
        system_prompt="system", user_prompt="user", response_schema=schema
    )

    assert result == '{"name": "Asha", "age": 20}'
    assert sdk_client.chat.completions.create.call_args.kwargs["response_format"] == {
        "type": "json_object"
    }


def test_malformed_json_raises_clear_error(groq_factory):
    client, _ = make_client(groq_factory, "{not json}")

    with pytest.raises(MalformedJSONError, match="malformed JSON"):
        client.generate(
            system_prompt="system",
            user_prompt="user",
            response_schema={"type": "object"},
        )


def test_schema_mismatch_raises_clear_error(groq_factory):
    client, _ = make_client(groq_factory, '{"age":"twenty"}')

    with pytest.raises(ResponseSchemaError, match="did not match"):
        client.generate(
            system_prompt="system",
            user_prompt="user",
            response_schema={"type": "object", "properties": {"age": {"type": "integer"}}},
        )


@pytest.mark.parametrize(
    "config, message",
    [
        (LLMConfig(api_key=None, model="test-model"), "GROQ_API_KEY"),
        (LLMConfig(api_key="test-key", model="not-selected"), "LLM_MODEL"),
    ],
)
def test_missing_credentials_or_model_fail_before_sdk_call(
    groq_factory, config, message
):
    with pytest.raises(ValueError, match=message):
        GroqLLMClient(config)

    groq_factory.assert_not_called()


def test_api_exception_is_not_reported_as_json_error(groq_factory):
    client, sdk_client = make_client(groq_factory, "unused")
    api_error = RuntimeError("simulated API failure")
    sdk_client.chat.completions.create.side_effect = api_error

    with pytest.raises(RuntimeError, match="simulated API failure") as error:
        client.generate(
            system_prompt="system",
            user_prompt="user",
            response_schema={"type": "object"},
        )

    assert error.value is api_error


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_response_content_raises_clear_error(groq_factory, content):
    client, _ = make_client(groq_factory, content)

    with pytest.raises(RuntimeError, match="empty response"):
        client.generate(system_prompt="system", user_prompt="user")
