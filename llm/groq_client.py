
import json

from groq import Groq

from llm.client import LLMClient
from llm.config import LLMConfig, get_llm_config


class GroqLLMClient:
    """Groq implementation of the shared LLM client interface."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_llm_config()

        if not self.config.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Check your local .env file."
            )

        if not self.config.model or self.config.model == "not-selected":
            raise ValueError(
                "LLM_MODEL is not configured. Check your local .env file."
            )

        self._client = Groq(
            api_key=self.config.api_key,
            max_retries=self.config.max_retries,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None = None,
    ) -> str:
        """Generate a response using Groq's chat completions API."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if response_schema is not None:
            messages[0]["content"] += (
                "\nReturn a JSON object matching this schema:\n"
                + json.dumps(response_schema)
                + "\nDo not include Markdown code fences."
            )
            request["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**request)

        content = completion.choices[0].message.content

        if content is None:
            raise RuntimeError("Groq returned an empty response.")

        return content
