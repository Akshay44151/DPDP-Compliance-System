
from llm.config import get_llm_config


def test_default_config():
    config = get_llm_config()

    assert config.provider == "groq"
    assert config.model == "not-selected"
    assert config.temperature == 0.0
    assert config.max_retries == 2
    assert config.api_key is None or isinstance(config.api_key, str)
