
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Shared configuration for the project's LLM provider."""

    provider: str = "groq"
    model: str = "not-selected"
    api_key: str | None = None
    temperature: float = 0.0
    max_retries: int = 2


def get_llm_config() -> LLMConfig:
    """Load LLM settings from environment variables."""

    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "groq"),
        model=os.getenv("LLM_MODEL", "not-selected"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
