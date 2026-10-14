
from llm.config import get_llm_config


def test_default_config(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)

    config = get_llm_config()

    assert config.provider == "groq"
    assert config.model == "not-selected"
    assert config.temperature == 0.0
    assert config.max_retries == 2
    assert config.api_key is None
