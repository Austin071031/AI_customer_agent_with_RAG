"""
Unit tests for DeepSeek API Service.

This module contains comprehensive tests for the DeepSeekService class,
including API interactions, error handling, and streaming functionality.
"""

import pytest
import aiohttp
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

from src.services.deepseek_service import DeepSeekService, DeepSeekAPIError
from src.models.config_models import APIConfig
from src.models.chat_models import ChatMessage


class TestDeepSeekAPIError:
    """Test cases for DeepSeekAPIError custom exception."""
    
    def test_deepseek_api_error_creation(self):
        """Test creating DeepSeekAPIError with message and status code."""
        error = DeepSeekAPIError("Test error", status_code=400, error_type="invalid_request")
        
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.error_type == "invalid_request"
        assert str(error) == "Test error"
        
    def test_deepseek_api_error_defaults(self):
        """Test DeepSeekAPIError with default parameters."""
        error = DeepSeekAPIError("Test error")
        
        assert error.message == "Test error"
        assert error.status_code is None
        assert error.error_type is None


class TestDeepSeekService:
    """Test cases for DeepSeekService class."""
    
    @pytest.fixture
    def api_config(self):
        """Create a test API configuration."""
        return APIConfig(
            api_key="sk-test1234567890",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
    
    @pytest.fixture
    def deepseek_service(self, api_config):
        """Create a DeepSeekService instance for testing."""
        return DeepSeekService(api_config)
    
    @pytest.fixture
    def test_messages(self):
        """Create test messages for API calls."""
        return [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    
    @pytest.fixture
    def successful_response(self):
        """Create a successful API response."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
                    }
                }
            ]
        }
    
    @pytest.fixture
    def error_response(self):
        """Create an error API response."""
        return {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error"
            }
        }

    @pytest.mark.asyncio
    async def test_service_initialization(self, api_config):
        """Test that DeepSeekService initializes correctly."""
        service = DeepSeekService(api_config)
        
        assert service.api_config == api_config
        assert service.session is None
        assert service.logger.name == "src.services.deepseek_service"

    @pytest.mark.asyncio
    async def test_async_context_manager(self, api_config):
        """Test the async context manager functionality."""
        async with DeepSeekService(api_config) as service:
            assert service.session is not None
            assert isinstance(service.session, aiohttp.ClientSession)

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, deepseek_service, test_messages, successful_response):
        """Test successful chat completion."""
        with patch.object(deepseek_service, '_make_api_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = successful_response
            
            response = await deepseek_service.chat_completion(test_messages)
            
            # Verify the response is correct
            assert response == "Hello! I'm doing well, thank you for asking. How can I help you today?"
            
            # Verify the API was called with correct parameters
            mock_request.assert_called_once_with(
                "chat/completions",
                {
                    "model": "deepseek-chat",
                    "messages": test_messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": False
                }
            )

    @pytest.mark.asyncio
    async def test_chat_completion_custom_model(self, deepseek_service, test_messages, successful_response):
        """Test chat completion with custom model parameter."""
        with patch.object(deepseek_service, '_make_api_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = successful_response
            
            response = await deepseek_service.chat_completion(test_messages, model="deepseek-coder")
            
            # Verify custom model was used
            mock_request.assert_called_once()
            call_args = mock_request.call_args[0]
            payload = call_args[1]
            assert payload["model"] == "deepseek-coder"

    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, deepseek_service, test_messages, error_response):
        """Test chat completion with API error."""
        with patch.object(deepseek_service, '_make_api_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = DeepSeekAPIError(
                "API error", 
                status_code=401, 
                error_type="invalid_request"
            )
            
            # Verify the error is raised
            with pytest.raises(DeepSeekAPIError) as exc_info:
                await deepseek_service.chat_completion(test_messages)
            
            assert exc_info.value.status_code == 401
            assert exc_info.value.error_type == "invalid_request"

    @pytest.mark.asyncio
    async def test_chat_completion_unexpected_response_format(self, deepseek_service, test_messages):
        """Test chat completion with unexpected response format."""
        with patch.object(deepseek_service, '_make_api_request', new_callable=AsyncMock) as mock_request:
            # Return response without expected structure
            mock_request.return_value = {"unexpected": "format"}
            
            # Verify KeyError is caught and converted to DeepSeekAPIError
            with pytest.raises(DeepSeekAPIError) as exc_info:
                await deepseek_service.chat_completion(test_messages)
            
            assert "Unexpected response format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_chat_success(self, deepseek_service, test_messages):
        """Test successful streaming chat."""
        # Mock streaming response data
        stream_responses = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " there"}}]}',
            'data: [DONE]'
        ]
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = [resp.encode('utf-8') for resp in stream_responses]
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            # Collect streaming responses
            responses = []
            async for chunk in deepseek_service.stream_chat(test_messages):
                responses.append(chunk)
            
            # Verify streaming responses
            assert responses == ["Hello", " there"]
            
            # Verify API call
            mock_session.post.assert_called_once_with(
                "https://api.deepseek.com/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": test_messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True
                },
                headers={
                    "Authorization": "Bearer sk-test1234567890",
                    "Content-Type": "application/json"
                }
            )

    @pytest.mark.asyncio
    async def test_stream_chat_api_error(self, deepseek_service, test_messages, error_response):
        """Test streaming chat with API error."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value=error_response)
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            # Verify error is raised
            with pytest.raises(DeepSeekAPIError) as exc_info:
                async for _ in deepseek_service.stream_chat(test_messages):
                    pass
            
            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_chat_message_without_history(self, deepseek_service):
        """Test processing chat message without conversation history."""
        user_message = "Hello, how can you help me?"
        
        with patch.object(deepseek_service, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "I can help you with various customer service tasks!"
            
            response = await deepseek_service.process_chat_message(user_message)
            
            # Verify response
            assert response == "I can help you with various customer service tasks!"
            
            # Verify chat_completion was called with correct messages structure
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args[0]
            messages = call_args[0]
            
            # Should include system message and user message
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "customer service assistant" in messages[0]["content"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == user_message

    @pytest.mark.asyncio
    async def test_process_chat_message_with_history(self, deepseek_service):
        """Test processing chat message with conversation history."""
        user_message = "Tell me more about that"
        conversation_history = [
            ChatMessage(role="user", content="What services do you offer?"),
            ChatMessage(role="assistant", content="We offer customer support, technical assistance, and product information."),
            ChatMessage(role="user", content="Can you help with billing questions?"),
            ChatMessage(role="assistant", content="Yes, I can help with billing questions."),
        ]
        
        with patch.object(deepseek_service, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "For billing questions, I can help you check your account balance, review recent charges, and explain billing statements."
            
            response = await deepseek_service.process_chat_message(user_message, conversation_history)
            
            # Verify response
            assert "billing questions" in response
            
            # Verify chat_completion was called with conversation history
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args[0]
            messages = call_args[0]
            
            # Should include system message, conversation history (limited to last 10), and current message
            assert len(messages) == 6  # system + 4 history messages + current
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert messages[3]["role"] == "user"
            assert messages[4]["role"] == "assistant"
            assert messages[5]["role"] == "user"
            assert messages[5]["content"] == user_message

    @pytest.mark.asyncio
    async def test_health_check_success(self, deepseek_service):
        """Test health check when API is healthy."""
        with patch.object(deepseek_service, 'chat_completion', new_callable=AsyncMock):
            # chat_completion will succeed without raising exception
            is_healthy = await deepseek_service.health_check()
            
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, deepseek_service):
        """Test health check when API is not healthy."""
        with patch.object(deepseek_service, 'chat_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = DeepSeekAPIError("API unavailable")
            
            is_healthy = await deepseek_service.health_check()
            
            assert is_healthy is False

    def test_get_usage_info(self, deepseek_service):
        """Test getting usage information."""
        usage_info = deepseek_service.get_usage_info()
        
        expected_info = {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2000,
            "base_url": "https://api.deepseek.com"
        }
        
        assert usage_info == expected_info

    @pytest.mark.asyncio
    async def test_make_api_request_success(self, deepseek_service):
        """Test successful API request."""
        test_payload = {"test": "data"}
        expected_response = {"success": True}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=expected_response)
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            response = await deepseek_service._make_api_request("test/endpoint", test_payload)
            
            assert response == expected_response
            
            # Verify correct URL and headers
            mock_session.post.assert_called_once_with(
                "https://api.deepseek.com/test/endpoint",
                json=test_payload,
                headers={
                    "Authorization": "Bearer sk-test1234567890",
                    "Content-Type": "application/json"
                }
            )

    @pytest.mark.asyncio
    async def test_make_api_request_error_response(self, deepseek_service):
        """Test API request with error response."""
        test_payload = {"test": "data"}
        error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }
        
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value=error_response)
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            with pytest.raises(DeepSeekAPIError) as exc_info:
                await deepseek_service._make_api_request("test/endpoint", test_payload)
            
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value)
            assert exc_info.value.error_type == "rate_limit_error"

    @pytest.mark.asyncio
    async def test_make_api_request_network_error(self, deepseek_service):
        """Test API request with network error."""
        test_payload = {"test": "data"}
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Network connection failed")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            with pytest.raises(DeepSeekAPIError) as exc_info:
                await deepseek_service._make_api_request("test/endpoint", test_payload)
            
            assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_api_request_json_decode_error(self, deepseek_service):
        """Test API request with JSON decode error."""
        test_payload = {"test": "data"}
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0))
        
        # Mock the session and its post method
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            # Initialize session first
            await deepseek_service._ensure_session()
            
            with pytest.raises(DeepSeekAPIError) as exc_info:
                await deepseek_service._make_api_request("test/endpoint", test_payload)
            
            assert "Invalid JSON response" in str(exc_info.value)

    def test_ensure_session_creates_new_session(self, deepseek_service):
        """Test that _ensure_session creates a new session when needed."""
        # Initially no session
        assert deepseek_service.session is None
        
        # Create event loop and run _ensure_session
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(deepseek_service._ensure_session())
            assert deepseek_service.session is not None
            assert isinstance(deepseek_service.session, aiohttp.ClientSession)
        finally:
            loop.close()


# Integration-style tests that verify the service works with real API patterns
class TestDeepSeekServiceIntegration:
    """Integration-style tests for DeepSeekService."""
    
    @pytest.fixture
    def api_config(self):
        return APIConfig(api_key="sk-test-integration")
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, api_config):
        """Test the complete lifecycle of the service."""
        async with DeepSeekService(api_config) as service:
            # Service should have active session
            assert service.session is not None
            
            # Test that service can be used
            with patch.object(service, 'chat_completion', new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = "Test response"
                response = await service.chat_completion([{"role": "user", "content": "test"}])
                assert response == "Test response"
        
        # After context manager, session should be closed
        # Note: We can't easily test session.close() was called without more complex mocking

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, api_config):
        """Test handling multiple concurrent requests."""
        async with DeepSeekService(api_config) as service:
            with patch.object(service, '_make_api_request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = {
                    "choices": [{"message": {"content": "Response"}}]
                }
                
                # Create multiple concurrent requests
                tasks = [
                    service.chat_completion([{"role": "user", "content": f"Message {i}"}])
                    for i in range(3)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                # All requests should complete successfully
                assert len(responses) == 3
                assert all(response == "Response" for response in responses)
                assert mock_request.call_count == 3
