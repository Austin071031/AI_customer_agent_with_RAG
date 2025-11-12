"""
Unit tests for the Streamlit web interface of AI Customer Agent.

This module contains unit tests for the helper functions and components
of the Streamlit web interface, excluding the Streamlit runtime itself.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path to allow imports from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ui.streamlit_app import (
    check_api_health,
    get_system_info,
    send_chat_message,
    stream_chat_message,
    get_conversation_history,
    clear_conversation_history,
    upload_to_knowledge_base,
    search_knowledge_base,
    clear_knowledge_base,
    get_configuration,
    update_configuration,
    check_gpu_status,
    API_BASE_URL,
    CHAT_ENDPOINT,
    KB_ENDPOINT,
    CONFIG_ENDPOINT,
    HEALTH_ENDPOINT,
    INFO_ENDPOINT
)


class TestStreamlitApp:
    """Test cases for Streamlit web interface helper functions."""
    
    def test_api_endpoints_defined(self):
        """Test that API endpoints are properly defined."""
        assert API_BASE_URL == "http://localhost:8000"
        assert CHAT_ENDPOINT == f"{API_BASE_URL}/api/chat"
        assert KB_ENDPOINT == f"{API_BASE_URL}/api/knowledge-base"
        assert CONFIG_ENDPOINT == f"{API_BASE_URL}/api/config"
        assert HEALTH_ENDPOINT == f"{API_BASE_URL}/health"
        assert INFO_ENDPOINT == f"{API_BASE_URL}/info"
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_check_api_health_success(self, mock_get):
        """Test API health check when API is healthy."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is True
        mock_get.assert_called_once_with(HEALTH_ENDPOINT, timeout=5)
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_check_api_health_unhealthy_status(self, mock_get):
        """Test API health check when API returns unhealthy status."""
        # Mock unhealthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "unhealthy"}
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is False
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_check_api_health_connection_error(self, mock_get):
        """Test API health check when connection fails."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = check_api_health()
        
        assert result is False
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_system_info_success(self, mock_get):
        """Test successful retrieval of system information."""
        # Mock successful response
        expected_info = {
            "system": {
                "name": "AI Customer Agent",
                "version": "1.0.0",
                "environment": "development"
            },
            "configuration": {
                "api_enabled": True,
                "knowledge_base_enabled": True,
                "chat_enabled": True
            }
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_info
        mock_get.return_value = mock_response
        
        result = get_system_info()
        
        assert result == expected_info
        mock_get.assert_called_once_with(INFO_ENDPOINT, timeout=5)
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_system_info_failure(self, mock_get):
        """Test failure to retrieve system information."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = get_system_info()
        
        assert result is None
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_system_info_connection_error(self, mock_get):
        """Test system info retrieval when connection fails."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = get_system_info()
        
        assert result is None
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_send_chat_message_success(self, mock_post):
        """Test successful chat message sending."""
        # Mock successful response
        expected_response = {
            "response": "Hello! How can I help you?",
            "message_id": "123e4567-e89b-12d3-a456-426614174000",
            "conversation_length": 2,
            "used_knowledge_base": True
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_post.return_value = mock_response
        
        result = send_chat_message("Hello", use_kb=True, stream=False)
        
        assert result == expected_response
        mock_post.assert_called_once_with(
            f"{CHAT_ENDPOINT}/",
            json={
                "message": "Hello",
                "use_knowledge_base": True,
                "stream": False
            },
            timeout=30
        )
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_send_chat_message_stream_endpoint(self, mock_post):
        """Test chat message sending with streaming endpoint."""
        # Mock successful response
        expected_response = {
            "response": "Streaming response",
            "message_id": "123e4567-e89b-12d3-a456-426614174000",
            "conversation_length": 2,
            "used_knowledge_base": True
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_post.return_value = mock_response
        
        result = send_chat_message("Hello", use_kb=True, stream=True)
        
        assert result == expected_response
        mock_post.assert_called_once_with(
            f"{CHAT_ENDPOINT}/stream",
            json={
                "message": "Hello",
                "use_knowledge_base": True,
                "stream": True
            },
            timeout=30
        )
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_send_chat_message_api_error(self, mock_post):
        """Test chat message sending when API returns error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="API Error: 400 - Bad request"):
            send_chat_message("Hello")
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_stream_chat_message_success(self, mock_post):
        """Test successful chat message streaming."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            "data: Hello",
            "data: world",
            "data: [DONE]"
        ]
        mock_post.return_value = mock_response
        
        generator = stream_chat_message("Hello", use_kb=True)
        chunks = list(generator)
        
        expected_chunks = ["Hello", "world"]
        assert chunks == expected_chunks
        
        mock_post.assert_called_once_with(
            f"{CHAT_ENDPOINT}/stream",
            json={
                "message": "Hello",
                "use_knowledge_base": True,
                "stream": True
            },
            stream=True,
            timeout=60
        )
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_stream_chat_message_error_response(self, mock_post):
        """Test chat message streaming with error response."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response
        
        generator = stream_chat_message("Hello")
        chunks = list(generator)
        
        expected_chunk = "Error: 400 - Bad request"
        assert chunks == [expected_chunk]
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_stream_chat_message_connection_error(self, mock_post):
        """Test chat message streaming with connection error."""
        # Mock connection error
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        generator = stream_chat_message("Hello")
        chunks = list(generator)
        
        expected_chunk = "❌ Connection error: Connection failed"
        assert chunks == [expected_chunk]
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_conversation_history_success(self, mock_get):
        """Test successful retrieval of conversation history."""
        # Mock successful response
        expected_messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-01T12:00:00",
                "message_id": "msg_001"
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": "2024-01-01T12:00:01",
                "message_id": "msg_002"
            }
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": expected_messages}
        mock_get.return_value = mock_response
        
        result = get_conversation_history()
        
        assert result == expected_messages
        mock_get.assert_called_once_with(f"{CHAT_ENDPOINT}/history", timeout=5)
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_conversation_history_api_error(self, mock_get):
        """Test conversation history retrieval when API returns error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception, match="API Error: 500 - Internal server error"):
            get_conversation_history()
    
    @patch('src.ui.streamlit_app.requests.delete')
    def test_clear_conversation_history_success(self, mock_delete):
        """Test successful clearing of conversation history."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response
        
        result = clear_conversation_history()
        
        assert result is True
        mock_delete.assert_called_once_with(f"{CHAT_ENDPOINT}/history", timeout=5)
    
    @patch('src.ui.streamlit_app.requests.delete')
    def test_clear_conversation_history_failure(self, mock_delete):
        """Test failure to clear conversation history."""
        # Mock connection error
        mock_delete.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = clear_conversation_history()
        
        assert result is False
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_upload_to_knowledge_base_success(self, mock_post):
        """Test successful file upload to knowledge base."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Create a mock file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"test content"
        mock_file.type = "application/pdf"
        
        result = upload_to_knowledge_base(mock_file)
        
        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['files'] == {"files": ("test.pdf", b"test content", "application/pdf")}
    
    @patch('src.ui.streamlit_app.requests.post')
    def test_upload_to_knowledge_base_failure(self, mock_post):
        """Test failure to upload file to knowledge base."""
        # Mock connection error
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"test content"
        mock_file.type = "application/pdf"
        
        result = upload_to_knowledge_base(mock_file)
        
        assert result is False
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_search_knowledge_base_success(self, mock_get):
        """Test successful knowledge base search."""
        # Mock successful response
        expected_results = [
            {
                "id": "doc_001",
                "content": "This is a test document",
                "metadata": {"source": "test.pdf"},
                "file_path": "/path/to/test.pdf",
                "file_type": "pdf"
            }
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_results
        mock_get.return_value = mock_response
        
        result = search_knowledge_base("test query", k=3)
        
        assert result == expected_results
        mock_get.assert_called_once_with(
            f"{KB_ENDPOINT}/search",
            params={"query": "test query", "k": 3},
            timeout=10
        )
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_search_knowledge_base_api_error(self, mock_get):
        """Test knowledge base search when API returns error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception, match="API Error: 500 - Internal server error"):
            search_knowledge_base("test query")
    
    @patch('src.ui.streamlit_app.requests.delete')
    def test_clear_knowledge_base_success(self, mock_delete):
        """Test successful clearing of knowledge base."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response
        
        result = clear_knowledge_base()
        
        assert result is True
        mock_delete.assert_called_once_with(f"{KB_ENDPOINT}/", timeout=10)
    
    @patch('src.ui.streamlit_app.requests.delete')
    def test_clear_knowledge_base_failure(self, mock_delete):
        """Test failure to clear knowledge base."""
        # Mock connection error
        mock_delete.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = clear_knowledge_base()
        
        assert result is False
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_configuration_success(self, mock_get):
        """Test successful retrieval of configuration."""
        # Mock successful response
        expected_config = {
            "api_key": "test_key",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_config
        mock_get.return_value = mock_response
        
        result = get_configuration()
        
        assert result == expected_config
        mock_get.assert_called_once_with(CONFIG_ENDPOINT, timeout=5)
    
    @patch('src.ui.streamlit_app.requests.get')
    def test_get_configuration_failure(self, mock_get):
        """Test failure to retrieve configuration."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = get_configuration()
        
        assert result is None
    
    @patch('src.ui.streamlit_app.requests.put')
    def test_update_configuration_success(self, mock_put):
        """Test successful update of configuration."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response
        
        config_data = {
            "api_key": "new_key",
            "temperature": 0.8
        }
        
        result = update_configuration(config_data)
        
        assert result is True
        mock_put.assert_called_once_with(CONFIG_ENDPOINT, json=config_data, timeout=10)
    
    @patch('src.ui.streamlit_app.requests.put')
    def test_update_configuration_failure(self, mock_put):
        """Test failure to update configuration."""
        # Mock connection error
        mock_put.side_effect = requests.exceptions.RequestException("Connection failed")
        
        config_data = {"api_key": "new_key"}
        
        result = update_configuration(config_data)
        
        assert result is False
    
    @patch('src.ui.streamlit_app.torch')
    def test_check_gpu_status_available(self, mock_torch):
        """Test GPU status check when GPU is available."""
        # Mock PyTorch CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070 Ti"
        mock_torch.cuda.memory_allocated.return_value = 1024**3  # 1 GB
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024**3  # 2 GB
        
        result = check_gpu_status()
        
        assert result["available"] is True
        assert result["device_count"] == 1
        assert result["current_device"] == 0
        assert result["device_name"] == "NVIDIA GeForce RTX 4070 Ti"
        assert result["memory_allocated"] == 1.0  # GB
        assert result["memory_reserved"] == 2.0  # GB
    
    @patch('src.ui.streamlit_app.torch')
    def test_check_gpu_status_unavailable(self, mock_torch):
        """Test GPU status check when GPU is unavailable."""
        # Mock PyTorch CUDA unavailability
        mock_torch.cuda.is_available.return_value = False
        
        result = check_gpu_status()
        
        assert result["available"] is False
        assert result["device_count"] == 0
        assert result["current_device"] is None
        assert result["device_name"] is None
    
    def test_check_gpu_status_pytorch_not_installed(self):
        """Test GPU status check when PyTorch is not installed."""
        # Temporarily remove torch from imports if it exists
        with patch.dict('sys.modules', {'torch': None}):
            # Re-import the function to trigger the ImportError
            import importlib
            import src.ui.streamlit_app
            importlib.reload(src.ui.streamlit_app)
            
            result = src.ui.streamlit_app.check_gpu_status()
            
            assert result["available"] is False
            assert result["error"] == "PyTorch not installed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
