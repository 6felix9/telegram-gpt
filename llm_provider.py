"""Abstract base class for LLM provider implementations."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface that all LLM providers must implement."""

    @abstractmethod
    async def get_completion(self, messages: list[dict], **kwargs) -> str:
        """
        Get completion from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Provider-specific options (e.g., is_group)

        Returns:
            Response text from the LLM
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test API connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return current model identifier.

        Returns:
            Model name string
        """
        pass

    @abstractmethod
    def get_max_context_tokens(self) -> int:
        """
        Return max context window for current model.

        Returns:
            Maximum context tokens
        """
        pass

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens in message list.

        Args:
            messages: List of message dicts

        Returns:
            Total token count
        """
        pass

    @abstractmethod
    def format_messages(self, messages: list[dict], is_group: bool) -> list[dict]:
        """
        Format messages for provider-specific API.

        Args:
            messages: Raw message list
            is_group: Whether this is a group chat

        Returns:
            Formatted message list
        """
        pass
