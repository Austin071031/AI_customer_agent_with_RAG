"""
DeepSeek API Service for AI Customer Agent.

This module provides the core service for communicating with the DeepSeek API,
handling both standard chat completions and streaming responses with proper
error handling and async/await patterns.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime

from ..models.config_models import APIConfig
from ..models.chat_models import ChatMessage


class DeepSeekAPIError(Exception):
    """Custom exception for DeepSeek API related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)


class DeepSeekService:
    """
    Service for interacting with the DeepSeek API.
    
    This service handles both standard chat completions and streaming responses
    with proper error handling and async/await patterns.
    """
    
    def __init__(self, api_config: APIConfig):
        """
        Initialize the DeepSeek service with API configuration.
        
        Args:
            api_config: APIConfig instance containing API settings
        """
        self.api_config = api_config
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry - create session."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session."""
        if self.session:
            await self.session.close()
            
    async def _ensure_session(self):
        """Ensure a session exists, create one if not."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def _make_api_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make an API request to DeepSeek with proper error handling.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Returns:
            Dict containing API response
            
        Raises:
            DeepSeekAPIError: If the API request fails
        """
        await self._ensure_session()
        
        url = f"{self.api_config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self.api_config.get_headers()
        
        try:
            self.logger.info(f"Sending request to DeepSeek API: {url}")
            async with self.session.post(url, json=payload, headers=headers) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get('error', {}).get('message', 'Unknown API error')
                    error_type = response_data.get('error', {}).get('type', 'unknown')
                    raise DeepSeekAPIError(
                        f"DeepSeek API error ({response.status}): {error_msg}",
                        status_code=response.status,
                        error_type=error_type
                    )
                    
                return response_data
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error communicating with DeepSeek API: {str(e)}")
            raise DeepSeekAPIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from DeepSeek API: {str(e)}")
            raise DeepSeekAPIError("Invalid JSON response from API")
        except Exception as e:
            self.logger.error(f"Unexpected error in DeepSeek API call: {str(e)}")
            raise DeepSeekAPIError(f"Unexpected error: {str(e)}")
            
    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """
        Send chat completion request to DeepSeek API and return the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to config model)
            
        Returns:
            String containing the AI response
            
        Raises:
            DeepSeekAPIError: If the API request fails
        """
        model_to_use = model or self.api_config.model
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens,
            "stream": False
        }
        
        # Log the final prompt being sent to DeepSeek API
        self._log_prompt_to_terminal(messages, model_to_use)
        
        try:
            response_data = await self._make_api_request("chat/completions", payload)
            return response_data["choices"][0]["message"]["content"]
            
        except KeyError as e:
            self.logger.error(f"Unexpected response format from DeepSeek API: {e}")
            raise DeepSeekAPIError("Unexpected response format from API")
            
    def _log_prompt_to_terminal(self, messages: List[Dict], model: str) -> None:
        """
        Log the final prompt being sent to DeepSeek API to the terminal.
        
        This method outputs the complete prompt that will be sent to the DeepSeek API,
        including system messages, conversation history, and knowledge base context.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model being used for the API call
        """
        print("\n" + "="*80)
        print("DEEPSEEK API PROMPT BEING SENT:")
        print("="*80)
        print(f"Model: {model}")
        print(f"Total messages: {len(messages)}")
        print("-"*80)
        
        for i, message in enumerate(messages, 1):
            role = message.get('role', 'unknown').upper()
            content = message.get('content', '')
            print(f"{i}. [{role}]")
            print(f"   {content}")
            if i < len(messages):
                print("   " + "-"*40)
        print("="*80 + "\n")
        
    async def stream_chat(self, messages: List[Dict], model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream chat responses from DeepSeek API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to config model)
            
        Yields:
            String chunks from the streaming response
            
        Raises:
            DeepSeekAPIError: If the API request fails
        """
        model_to_use = model or self.api_config.model
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens,
            "stream": True
        }
        
        # Log the final prompt being sent to DeepSeek API
        self._log_prompt_to_terminal(messages, model_to_use)
        
        await self._ensure_session()
        url = f"{self.api_config.base_url.rstrip('/')}/chat/completions"
        headers = self.api_config.get_headers()
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown API error')
                    raise DeepSeekAPIError(
                        f"DeepSeek API error ({response.status}): {error_msg}",
                        status_code=response.status
                    )
                    
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during streaming: {str(e)}")
            raise DeepSeekAPIError(f"Network error during streaming: {str(e)}")
            
    async def process_chat_message(self, user_message: str, conversation_history: List[ChatMessage] = None) -> str:
        """
        Process a user chat message with conversation context.
        
        Args:
            user_message: The user's message to process
            conversation_history: Optional list of previous ChatMessage objects
            
        Returns:
            AI response as string
        """
        messages = []
        
        # Add system message for context
        system_message = {
            "role": "system",
            "content": "You are a helpful customer service assistant. Provide clear, accurate, and friendly responses."
        }
        messages.append(system_message)
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add the current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return await self.chat_completion(messages)
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the DeepSeek API.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Send a simple test message to check API connectivity
            test_messages = [{"role": "user", "content": "Hello"}]
            await self.chat_completion(test_messages)
            return True
        except DeepSeekAPIError:
            return False
            
    def get_usage_info(self) -> Dict:
        """
        Get usage information about the current API configuration.
        
        Returns:
            Dictionary containing API usage information
        """
        return {
            "model": self.api_config.model,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens,
            "base_url": self.api_config.base_url
        }
