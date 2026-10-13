
from llm.client import LLMClient


class MockLLM:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None = None,
    ) -> str:
        return '{"message": "mock response"}'


def test_mock_llm_follows_interface():
    client: LLMClient = MockLLM()

    result = client.generate(
        system_prompt="You are a test assistant.",
        user_prompt="Say hello.",
    )

    assert result == '{"message": "mock response"}'
