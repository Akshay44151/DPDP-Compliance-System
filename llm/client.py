
from typing import Any, Protocol


class LLMClient(Protocol):
    """Common interface that all LLM providers must follow."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate a response, optionally following a JSON schema."""
        ...
