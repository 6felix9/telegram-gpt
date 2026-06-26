"""Tests for MODEL_REGISTRY validation used by /model command."""
from openai_client import MODEL_REGISTRY

VALID_APIS = {"responses", "chat_completions"}
VALID_PROVIDERS = {"openai", "xai", "gemini"}


def test_registry_is_non_empty():
    assert len(MODEL_REGISTRY) > 0


def test_known_model_present():
    assert "gpt-4o-mini" in MODEL_REGISTRY


def test_unknown_model_not_in_registry():
    assert "totally-fake-model" not in MODEL_REGISTRY


def test_all_entries_have_valid_api_and_provider():
    for model_name, config in MODEL_REGISTRY.items():
        assert config.api in VALID_APIS, f"{model_name} has invalid api: {config.api}"
        assert config.provider in VALID_PROVIDERS, f"{model_name} has invalid provider: {config.provider}"
