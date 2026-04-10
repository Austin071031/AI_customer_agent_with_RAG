from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers (Strategy Pattern).
    Allows decoupling business logic from specific LLM implementations
    (like DeepSeek, OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        Generate a complete chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            
        Returns:
            The complete response string
            
        Raises:
            LLMProviderError: If the API request fails
        """
        pass

    @abstractmethod
    async def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a chat response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            
        Yields:
            String chunks of the response
            
        Raises:
            LLMProviderError: If the API request fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM provider API is accessible and working.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
