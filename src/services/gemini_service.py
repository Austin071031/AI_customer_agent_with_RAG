"""
Google Gemini API Service for AI Customer Agent.

This module provides the core service for communicating with the Google Gemini API,
demonstrating how the Strategy Pattern allows swapping LLM providers seamlessly.
"""

import json
import logging
from typing import List, Dict, Optional, AsyncGenerator
import aiohttp

from src.models.config_models import APIConfig
from src.interfaces.llm_provider import LLMProvider, LLMProviderError


class GeminiAPIError(LLMProviderError):
    """Exception raised for errors in the Google Gemini API."""
    pass


class GeminiService(LLMProvider):
    """
    Service for interacting with the Google Gemini API.
    
    Implements the LLMProvider interface, allowing it to be used
    interchangeably with DeepSeekService or any other provider.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int = 2000):
        """
        Initialize the Gemini service.
        
        Args:
            api_key: Google Gemini API key
            model: Model to use (default: gemini-2.5-flash)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Convert standard OpenAI-style messages to Gemini format.
        """
        gemini_contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini handles system prompts separately
                system_instruction = {"parts": [{"text": content}]}
            else:
                # Map 'assistant' to 'model' for Gemini
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
                
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        return payload

    async def chat_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """Generate a complete chat response using Gemini."""
        await self._ensure_session()
        
        model_to_use = model or self.model
        url = f"{self.base_url}/{model_to_use}:generateContent?key={self.api_key}"
        
        payload = self._convert_messages_to_gemini_format(messages)
        headers = {"Content-Type": "application/json"}
        
        try:
            self.logger.info(f"Sending request to Gemini API model: {model_to_use}")
            async with self.session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get('error', {}).get('message', 'Unknown API error')
                    raise GeminiAPIError(
                        f"Gemini API error ({response.status}): {error_msg}",
                        status_code=response.status
                    )
                    
                # Extract text from the first candidate
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error communicating with Gemini API: {str(e)}")
            raise GeminiAPIError(f"Network error: {str(e)}") from e
        except (KeyError, IndexError) as e:
            self.logger.error(f"Unexpected response format from Gemini API: {e}")
            raise GeminiAPIError("Unexpected response format from API") from e

    async def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream a chat response using Gemini."""
        await self._ensure_session()
        
        model_to_use = model or self.model
        url = f"{self.base_url}/{model_to_use}:streamGenerateContent?key={self.api_key}"
        
        payload = self._convert_messages_to_gemini_format(messages)
        headers = {"Content-Type": "application/json"}
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown API error')
                    raise GeminiAPIError(
                        f"Gemini API error ({response.status}): {error_msg}",
                        status_code=response.status
                    )
                
                # Gemini streaming returns a JSON array of objects, but streams them iteratively
                # For simplicity in this raw aiohttp implementation, we'll read lines.
                # Note: For production streaming, the official `google-generativeai` SDK is highly recommended.
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('"text":'):
                        # Very basic JSON string extraction for demonstration
                        # e.g., "text": "Hello"
                        try:
                            text_part = line.split(':', 1)[1].strip().strip('",')
                            # Handle escaped quotes and newlines
                            text_part = text_part.replace('\\n', '\n').replace('\\"', '"')
                            if text_part:
                                yield text_part
                        except Exception:
                            continue
                            
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during streaming: {str(e)}")
            raise GeminiAPIError(f"Network error during streaming: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check if the Gemini API is accessible."""
        try:
            test_messages = [{"role": "user", "content": "Hi"}]
            await self.chat_completion(test_messages)
            return True
        except GeminiAPIError:
            return False
