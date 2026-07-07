"""Provider-prefixed model resolution replacing MODEL_REGISTRY validation."""
import pytest
import agent


def test_known_models_map_to_expected_providers():
    assert agent.resolve_model("gpt-5") == ("openai", "openai:gpt-5")
    assert agent.resolve_model("grok-4-1-fast-reasoning") == (
        "xai", "xai:grok-4-1-fast-reasoning")
    assert agent.resolve_model("gemini-3-flash-preview") == (
        "google_genai", "google_genai:gemini-3-flash-preview")


def test_every_registered_model_has_a_label():
    for provider in agent.MODEL_PROVIDERS.values():
        assert provider in agent.PROVIDER_LABEL


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        agent.resolve_model("does-not-exist")


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = "x"
    GEMINI_API_KEY = "g"


def test_provider_api_key_selection():
    assert agent.provider_api_key("openai", _Cfg) == "o"
    assert agent.provider_api_key("xai", _Cfg) == "x"
    assert agent.provider_api_key("google_genai", _Cfg) == "g"
