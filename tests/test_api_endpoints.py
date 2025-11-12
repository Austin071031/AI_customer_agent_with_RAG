"""
Unit tests for FastAPI Backend endpoints.

This module contains comprehensive unit tests for all API endpoints
including chat, knowledge base, and configuration endpoints.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status, UploadFile

from src.api.main import app, app_state
from src.api.dependencies import get_chat_manager, get_kb_manager, get_config_manager
from src.models.chat_models import ChatMessage, KBDocument
from src.models.config_models import APIConfig, DatabaseConfig, AppConfig
from src.services.chat_manager import ChatManagerError
from src.services.knowledge_base import KnowledgeBaseError
from src.services.config_manager import ConfigManagerError


@pytest.fixture
def client():
    """
    Test client fixture for FastAPI application.
    
    Returns:
        TestClient instance for testing API endpoints
    """
    return TestClient(app)


@pytest.fixture
def mock_services():
    """
    Mock services fixture for testing.
    
    Returns:
        Dictionary with mocked service instances
    """
    # Mock configuration manager
    mock_config_manager = MagicMock()
    mock_config_manager.get_api_config.return_value = APIConfig(
        api_key="test-api-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    mock_config_manager.get_db_config.return_value = DatabaseConfig()
    mock_config_manager.get_app_config.return_value = AppConfig(
        api_config=APIConfig(api_key="test-api-key")
    )
    mock_config_manager.get_last_modified.return_value = "2024-01-01T12:00:00Z"
    mock_config_manager.update_api_config.return_value = True
    mock_config_manager.update_db_config.return_value = True
    mock_config_manager.update_app_config.return_value = True
    mock_config_manager.reset_to_defaults.return_value = True
    mock_config_manager.validate_configuration.return_value = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    mock_config_manager.create_backup.return_value = {
        "backup_file": "/backups/config_backup_20240101.zip",
        "backup_time": "2024-01-01T12:00:00Z",
        "config_files": ["settings.yaml", ".env"]
    }
    mock_config_manager.health_check.return_value = True
    mock_config_manager.get_loaded_config_files.return_value = ["settings.yaml", ".env"]
    
    # Mock DeepSeek service
    mock_deepseek_service = AsyncMock()
    mock_deepseek_service.chat_completion.return_value = "This is a test response from the AI."
    mock_deepseek_service.stream_chat.return_value = AsyncMock()
    mock_deepseek_service.health_check.return_value = True
    
    # Mock knowledge base manager
    mock_kb_manager = MagicMock()
    mock_kb_manager.search_similar.return_value = [
        KBDocument(
            id="doc_1",
            content="This is test document content.",
            metadata={"source": "test", "similarity_score": 0.85},
            file_path="/test/doc1.pdf",
            file_type="pdf"
        )
    ]
    mock_kb_manager.add_documents.return_value = True
    mock_kb_manager.clear_knowledge_base.return_value = True
    mock_kb_manager.get_document_count.return_value = 5
    mock_kb_manager.get_statistics.return_value = {
        "total_documents": 5,
        "document_types": {"pdf": 3, "txt": 2},
        "total_size_bytes": 1024000,
        "embedding_model": "all-MiniLM-L6-v2",
        "collection_name": "documents",
        "last_updated": "2024-01-01T12:00:00Z"
    }
    mock_kb_manager.health_check.return_value = True
    
    # Mock chat manager - use MagicMock for sync methods, AsyncMock for async methods
    mock_chat_manager = MagicMock()
    mock_chat_manager.process_message = AsyncMock(return_value="This is a test AI response.")
    mock_chat_manager.stream_message = AsyncMock()
    
    # Create a proper async generator for stream_message
    async def mock_stream_generator():
        yield "Hello"
        yield " there"
        yield "!"
    
    mock_chat_manager.stream_message.return_value = mock_stream_generator()
    mock_chat_manager.get_conversation_history.return_value = [
        ChatMessage(
            role="user",
            content="Hello, how are you?",
            message_id="msg_001"
        ),
        ChatMessage(
            role="assistant",
            content="I'm doing well, thank you! How can I help you today?",
            message_id="msg_002"
        )
    ]
    mock_chat_manager.clear_conversation.return_value = None
    mock_chat_manager.get_conversation_summary.return_value = {
        "total_messages": 2,
        "user_messages": 1,
        "assistant_messages": 1,
        "max_history_length": 20,
        "history_usage_percentage": 10.0
    }
    mock_chat_manager.health_check = AsyncMock(return_value=True)
    
    return {
        "config_manager": mock_config_manager,
        "deepseek_service": mock_deepseek_service,
        "kb_manager": mock_kb_manager,
        "chat_manager": mock_chat_manager
    }


@pytest.fixture(autouse=True)
def setup_app_state(mock_services):
    """
    Automatically set up app state with mocked services before each test.
    
    Args:
        mock_services: Dictionary with mocked service instances
    """
    # Clear existing app state
    app_state.clear()
    
    # Set up mocked services
    app_state.update(mock_services)
    
    yield
    
    # Clean up after test
    app_state.clear()


class TestRootEndpoints:
    """Test cases for root endpoints."""
    
    def test_root_endpoint(self, client):
        """
        Test root endpoint returns API information.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["message"] == "AI Customer Agent API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_health_check_endpoint(self, client):
        """
        Test health check endpoint returns service status.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "api" in data["services"]
        assert "config_manager" in data["services"]
        assert "deepseek_service" in data["services"]
        assert "knowledge_base" in data["services"]
        assert "chat_manager" in data["services"]
    
    def test_system_info_endpoint(self, client):
        """
        Test system info endpoint returns configuration summary.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "system" in data
        assert "configuration" in data
        assert data["system"]["name"] == "AI Customer Agent"
        assert "api_enabled" in data["configuration"]
        assert "knowledge_base_enabled" in data["configuration"]
        assert "chat_enabled" in data["configuration"]


class TestChatEndpoints:
    """Test cases for chat endpoints."""
    
    def test_process_chat_message_success(self, client):
        """
        Test successful chat message processing.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "message": "Hello, how are you?",
            "use_knowledge_base": True,
            "stream": False
        }
        
        response = client.post("/api/chat/", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert "message_id" in data
        assert "conversation_length" in data
        assert "used_knowledge_base" in data
        assert data["response"] == "This is a test AI response."
        assert data["used_knowledge_base"] is True
    
    def test_process_chat_message_empty_message(self, client):
        """
        Test chat message processing with empty message.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "message": "",
            "use_knowledge_base": True,
            "stream": False
        }
        
        response = client.post("/api/chat/", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_process_chat_message_chat_manager_error(self, client, mock_services):
        """
        Test chat message processing with ChatManager error.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock ChatManager to raise an error
        mock_services["chat_manager"].process_message.side_effect = ChatManagerError(
            "Test error message"
        )
        
        request_data = {
            "message": "Hello",
            "use_knowledge_base": True,
            "stream": False
        }
        
        response = client.post("/api/chat/", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "Test error message" in data["detail"]
    
    def test_get_conversation_history_success(self, client):
        """
        Test successful retrieval of conversation history.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/chat/history")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "messages" in data
        assert "total_messages" in data
        assert "summary" in data
        assert len(data["messages"]) == 2
        assert data["total_messages"] == 2
        assert data["summary"]["total_messages"] == 2
    
    def test_get_conversation_history_with_limit(self, client):
        """
        Test conversation history retrieval with limit.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/chat/history?limit=1")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["messages"]) == 1  # Should be limited to 1 message
    
    def test_clear_conversation_history_success(self, client):
        """
        Test successful clearing of conversation history.
        
        Args:
            client: TestClient instance
        """
        response = client.delete("/api/chat/history")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "cleared_messages" in data
        assert "cleared successfully" in data["message"]
    
    def test_get_conversation_summary_success(self, client):
        """
        Test successful retrieval of conversation summary.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/chat/summary")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_messages" in data
        assert "user_messages" in data
        assert "assistant_messages" in data
        assert data["total_messages"] == 2
        assert data["user_messages"] == 1
        assert data["assistant_messages"] == 1
    
    def test_chat_health_check_success(self, client):
        """
        Test successful chat service health check.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/chat/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert data["service"] == "chat_manager"
    
    def test_chat_health_check_failure(self, client, mock_services):
        """
        Test chat service health check failure.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock health check to return False
        mock_services["chat_manager"].health_check.return_value = False
        
        response = client.get("/api/chat/health")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestKnowledgeBaseEndpoints:
    """Test cases for knowledge base endpoints."""
    
    def test_search_knowledge_base_success(self, client):
        """
        Test successful knowledge base search.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "query": "How to reset password?",
            "k": 3,
            "similarity_threshold": 0.5
        }
        
        response = client.post("/api/knowledge-base/search", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert "query" in data
        assert "search_time_ms" in data
        assert len(data["results"]) == 1
        assert data["query"] == "How to reset password?"
        assert data["total_results"] == 1
    
    def test_search_knowledge_base_empty_query(self, client):
        """
        Test knowledge base search with empty query.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "query": "",
            "k": 3,
            "similarity_threshold": 0.5
        }
        
        response = client.post("/api/knowledge-base/search", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_search_knowledge_base_kb_error(self, client, mock_services):
        """
        Test knowledge base search with KnowledgeBaseError.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock search_similar to raise an error
        mock_services["kb_manager"].search_similar.side_effect = KnowledgeBaseError(
            "Test KB error"
        )
        
        request_data = {
            "query": "Test query",
            "k": 3,
            "similarity_threshold": 0.5
        }
        
        response = client.post("/api/knowledge-base/search", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "Test KB error" in data["detail"]
    
    def test_upload_documents_success(self, client):
        """
        Test successful document upload to knowledge base.
        
        Args:
            client: TestClient instance
        """
        # Create a test file
        test_file_content = b"This is a test file content."
        
        files = [
            ("files", ("test.txt", test_file_content, "text/plain"))
        ]
        
        response = client.post(
            "/api/knowledge-base/documents",
            files=files,
            data={"overwrite": False}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "uploaded_files" in data
        assert "failed_files" in data
        assert "total_processed" in data
        assert "total_successful" in data
        assert "test.txt" in data["uploaded_files"]
        assert data["total_successful"] == 1
        assert data["total_processed"] == 1
    
    def test_upload_documents_unsupported_format(self, client):
        """
        Test document upload with unsupported file format.
        
        Args:
            client: TestClient instance
        """
        # Create a test file with unsupported format
        test_file_content = b"This is a test image content."
        
        files = [
            ("files", ("test.png", test_file_content, "image/png"))
        ]
        
        response = client.post(
            "/api/knowledge-base/documents",
            files=files,
            data={"overwrite": False}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_successful"] == 0
        assert len(data["failed_files"]) == 1
        assert data["failed_files"][0]["filename"] == "test.png"
        assert "Unsupported file type" in data["failed_files"][0]["reason"]
    
    def test_clear_knowledge_base_success(self, client):
        """
        Test successful clearing of knowledge base.
        
        Args:
            client: TestClient instance
        """
        response = client.delete("/api/knowledge-base/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "cleared_documents" in data
        assert "cleared successfully" in data["message"]
        assert data["cleared_documents"] == 5
    
    def test_get_knowledge_base_info_success(self, client):
        """
        Test successful retrieval of knowledge base information.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/knowledge-base/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_documents" in data
        assert "document_types" in data
        assert "total_size_bytes" in data
        assert "embedding_model" in data
        assert "collection_name" in data
        assert "last_updated" in data
        assert data["total_documents"] == 5
        assert data["document_types"] == {"pdf": 3, "txt": 2}
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
    
    def test_get_supported_formats_success(self, client):
        """
        Test successful retrieval of supported file formats.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/knowledge-base/supported-formats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
        assert "max_file_size_bytes" in data
        assert ".pdf" in data["supported_formats"]
        assert ".txt" in data["supported_formats"]
        assert ".docx" in data["supported_formats"]
    
    def test_knowledge_base_health_check_success(self, client):
        """
        Test successful knowledge base health check.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/knowledge-base/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert data["service"] == "knowledge_base"
        assert "document_count" in data


class TestConfigurationEndpoints:
    """Test cases for configuration endpoints."""
    
    def test_get_configuration_success(self, client):
        """
        Test successful retrieval of configuration.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/config/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "api_config" in data
        assert "db_config" in data
        assert "app_config" in data
        assert "last_modified" in data
        assert data["api_config"]["model"] == "deepseek-chat"
        assert data["db_config"]["collection_name"] == "documents"
        assert data["app_config"]["log_level"] == "info"
    
    def test_update_api_configuration_success(self, client):
        """
        Test successful update of API configuration.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "model": "deepseek-coder",
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        response = client.put("/api/config/api", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "updated_sections" in data
        assert "restart_required" in data
        assert "API configuration updated successfully" in data["message"]
        assert "api_config" in data["updated_sections"]
        assert data["restart_required"] is False
    
    def test_update_api_configuration_invalid_data(self, client, mock_services):
        """
        Test API configuration update with invalid data.

        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock update_api_config to raise ConfigManagerError
        mock_services["config_manager"].update_api_config.side_effect = ConfigManagerError(
            "Invalid API configuration"
        )

        request_data = {
            "temperature": 2.0,  # Invalid temperature (should be 0.0-1.0)
        }

        response = client.put("/api/config/api", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_update_database_configuration_success(self, client):
        """
        Test successful update of database configuration.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "persist_directory": "./custom_knowledge_base",
            "collection_name": "custom_documents"
        }
        
        response = client.put("/api/config/database", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "updated_sections" in data
        assert "restart_required" in data
        assert "Database configuration updated successfully" in data["message"]
        assert "db_config" in data["updated_sections"]
        assert data["restart_required"] is True
    
    def test_update_application_configuration_success(self, client):
        """
        Test successful update of application configuration.
        
        Args:
            client: TestClient instance
        """
        request_data = {
            "log_level": "debug",
            "enable_gpu": False,
            "max_conversation_history": 100
        }
        
        response = client.put("/api/config/application", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "updated_sections" in data
        assert "restart_required" in data
        assert "Application configuration updated successfully" in data["message"]
        assert "app_config" in data["updated_sections"]
        assert data["restart_required"] is True
    
    def test_reset_configuration_success(self, client):
        """
        Test successful reset of configuration to defaults.
        
        Args:
            client: TestClient instance
        """
        response = client.post("/api/config/reset")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "reset_sections" in data
        assert "default_values" in data
        assert "reset to defaults" in data["message"]
        assert "api_config" in data["reset_sections"]
        assert "db_config" in data["reset_sections"]
        assert "app_config" in data["reset_sections"]
        assert "base_url" in data["default_values"]["api_config"]
        assert "persist_directory" in data["default_values"]["db_config"]
        assert "log_level" in data["default_values"]["app_config"]
    
    def test_validate_configuration_success(self, client):
        """
        Test successful configuration validation.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/config/validate")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["warnings"] == []
    
    def test_backup_configuration_success(self, client):
        """
        Test successful configuration backup.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/config/backup")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "backup_file" in data
        assert "backup_time" in data
        assert "config_files" in data
        assert "backup created successfully" in data["message"]
        assert data["backup_file"] == "/backups/config_backup_20240101.zip"
        assert "settings.yaml" in data["config_files"]
    
    def test_get_environment_info_success(self, client):
        """
        Test successful retrieval of environment information.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/config/environment")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "environment" in data
        assert "timestamp" in data
        assert "python_version" in data["environment"]
        assert "platform" in data["environment"]
        assert "system" in data["environment"]
    
    def test_config_health_check_success(self, client):
        """
        Test successful configuration service health check.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/config/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert data["service"] == "config_manager"
        assert "config_files_loaded" in data


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_service_unavailable_error(self, client):
        """
        Test error handling when services are not available.
        
        Args:
            client: TestClient instance
        """
        # Clear app state to simulate services not being available
        app_state.clear()
        
        # Test chat endpoint without chat manager
        response = client.post("/api/chat/", json={
            "message": "Hello",
            "use_knowledge_base": True,
            "stream": False
        })
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data
        assert "Chat service not available" in data["detail"]
    
    def test_global_exception_handler(self, client, mock_services):
        """
        Test global exception handler for unhandled exceptions.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock a method to raise a generic exception
        mock_services["config_manager"].get_api_config.side_effect = Exception(
            "Unexpected error"
        )
        
        response = client.get("/api/config/")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "detail" in data
        assert data["error"] == "Internal server error"
        assert "unexpected error occurred" in data["message"]
    
    def test_invalid_json_request(self, client):
        """
        Test handling of invalid JSON request.
        
        Args:
            client: TestClient instance
        """
        # Send invalid JSON
        response = client.post(
            "/api/chat/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestStreamingEndpoints:
    """Test cases for streaming endpoints."""
    
    def test_stream_chat_message_success(self, client, mock_services):
        """
        Test successful streaming of chat messages.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock the streaming response
        async def mock_stream():
            yield "Hello"
            yield " there"
            yield "!"
        
        mock_services["chat_manager"].stream_message.return_value = mock_stream()
        
        request_data = {
            "message": "Hello",
            "use_knowledge_base": True,
            "stream": True
        }
        
        response = client.post("/api/chat/stream", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"].startswith("text/plain")
        assert "Cache-Control" in response.headers
        assert "no-cache" in response.headers["Cache-Control"]
    
    def test_stream_chat_message_error(self, client, mock_services):
        """
        Test streaming chat message with error.
        
        Args:
            client: TestClient instance
            mock_services: Dictionary with mocked service instances
        """
        # Mock stream_message to raise an error
        mock_services["chat_manager"].stream_message.side_effect = ChatManagerError(
            "Streaming error"
        )
        
        request_data = {
            "message": "Hello",
            "use_knowledge_base": True,
            "stream": True
        }
        
        response = client.post("/api/chat/stream", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "Streaming error" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
