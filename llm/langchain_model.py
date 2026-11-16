
from langchain_groq import ChatGroq

from llm.config import LLMConfig, get_llm_config


def get_langchain_model(
    config: LLMConfig | None = None,
) -> ChatGroq:
    """Create a LangChain chat model using the shared LLM configuration."""

    config = config or get_llm_config()

    if config.provider.lower() != "groq":
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. "
            "Currently, only Groq is supported."
        )

    if not config.api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your local .env file."
        )

    if not config.model or config.model == "not-selected":
        raise ValueError(
            "LLM_MODEL is not configured. Check your local .env file."
        )

    return ChatGroq(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_retries=config.max_retries,
    )
