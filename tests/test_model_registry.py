"""Direct coverage for model_registry.py, independent of agent.py's re-export."""
import pytest

import model_registry


def test_resolve_model_known():
    assert model_registry.resolve_model("gpt-5.4") == ("openai", "openai:gpt-5.4")


def test_resolve_model_unknown_raises():
    with pytest.raises(KeyError):
        model_registry.resolve_model("does-not-exist")


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = "x"
    GEMINI_API_KEY = "g"


def test_provider_api_key_selection():
    assert model_registry.provider_api_key("openai", _Cfg) == "o"
    assert model_registry.provider_api_key("xai", _Cfg) == "x"
    assert model_registry.provider_api_key("google_genai", _Cfg) == "g"


def test_every_registered_model_has_a_label():
    for provider in model_registry.MODEL_PROVIDERS.values():
        assert provider in model_registry.PROVIDER_LABEL
