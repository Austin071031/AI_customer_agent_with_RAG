"""
Unit tests for the core data models of the AI Customer Agent.

This module contains comprehensive tests for ChatMessage, KBDocument,
APIConfig, DatabaseConfig, and AppConfig models to ensure proper
validation and functionality.
"""

import pytest
from datetime import datetime
from uuid import UUID
from src.models.chat_models import ChatMessage, KBDocument
from src.models.config_models import APIConfig, DatabaseConfig, AppConfig


class TestChatMessage:
    """Test cases for ChatMessage model."""
    
    def test_chat_message_creation(self):
        """Test creating a ChatMessage with valid data."""
        message = ChatMessage(
            role="user",
            content="Hello, how can you help me today?"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how can you help me today?"
        assert isinstance(message.timestamp, datetime)
        assert isinstance(UUID(message.message_id), UUID)
    
    def test_chat_message_validation(self):
        """Test ChatMessage validation for invalid role."""
        with pytest.raises(ValueError):
            ChatMessage(role="invalid_role", content="test message")
    
    def test_chat_message_string_representation(self):
        """Test the string representation of ChatMessage."""
        message = ChatMessage(
            role="assistant",
            content="I'm here to help you with any questions you might have."
        )
        
        expected_str = "assistant: I'm here to help you with any questions you might ..."
        assert str(message) == expected_str
    
    def test_chat_message_json_serialization(self):
        """Test that ChatMessage can be serialized to JSON."""
        message = ChatMessage(
            role="system",
            content="You are a helpful assistant."
        )
        
        # Test that the model can be converted to dict (for JSON serialization)
        message_dict = message.dict()
        assert "role" in message_dict
        assert "content" in message_dict
        assert "timestamp" in message_dict
        assert "message_id" in message_dict


class TestKBDocument:
    """Test cases for KBDocument model."""
    
    def test_kb_document_creation(self):
        """Test creating a KBDocument with valid data."""
        document = KBDocument(
            id="doc_001",
            content="This is the content of the document.",
            metadata={
                "source": "company_handbook",
                "author": "HR Department",
                "created_date": "2024-01-01"
            },
            file_path="/documents/handbook.pdf",
            file_type="pdf"
        )
        
        assert document.id == "doc_001"
        assert document.content == "This is the content of the document."
        assert document.metadata["source"] == "company_handbook"
        assert document.file_path == "/documents/handbook.pdf"
        assert document.file_type == "pdf"
        assert document.embedding is None
    
    def test_kb_document_with_embedding(self):
        """Test creating a KBDocument with embedding data."""
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        document = KBDocument(
            id="doc_002",
            content="Document with embedding",
            metadata={"source": "test"},
            file_path="/documents/test.txt",
            file_type="txt",
            embedding=embedding_vector
        )
        
        assert document.embedding == embedding_vector
        assert len(document.embedding) == 5
    
    def test_kb_document_metadata_summary(self):
        """Test the get_metadata_summary method."""
        document = KBDocument(
            id="doc_003",
            content="Test content",
            metadata={"source": "test_source"},
            file_path="/documents/important_file.pdf",
            file_type="pdf"
        )
        
        summary = document.get_metadata_summary()
        expected_summary = "Document: important_file.pdf | Type: pdf | ID: doc_003"
        assert summary == expected_summary
    
    def test_kb_document_string_representation(self):
        """Test the string representation of KBDocument."""
        document = KBDocument(
            id="doc_004",
            content="A" * 100,  # Long content to test truncation
            metadata={"source": "test"},
            file_path="/documents/long.pdf",
            file_type="pdf"
        )
        
        expected_str = "KBDocument(id=doc_004, file_type=pdf, content_length=100)"
        assert str(document) == expected_str


class TestAPIConfig:
    """Test cases for APIConfig model."""
    
    def test_api_config_creation(self):
        """Test creating an APIConfig with valid data."""
        config = APIConfig(
            api_key="sk-test1234567890",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        assert config.api_key == "sk-test1234567890"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_api_config_default_values(self):
        """Test APIConfig uses default values when not provided."""
        config = APIConfig(api_key="sk-test123")
        
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_api_config_validation_empty_api_key(self):
        """Test APIConfig validation for empty API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            APIConfig(api_key="")
        
        with pytest.raises(ValueError, match="API key cannot be empty"):
            APIConfig(api_key="   ")
    
    def test_api_config_validation_invalid_base_url(self):
        """Test APIConfig validation for invalid base URL."""
        with pytest.raises(ValueError, match="Base URL must start with http:// or https://"):
            APIConfig(api_key="sk-test", base_url="invalid-url")
    
    def test_api_config_validation_temperature_range(self):
        """Test APIConfig validation for temperature range."""
        # Test valid temperature values
        APIConfig(api_key="sk-test", temperature=0.0)
        APIConfig(api_key="sk-test", temperature=0.5)
        APIConfig(api_key="sk-test", temperature=1.0)
        
        # Test invalid temperature values
        with pytest.raises(ValueError):
            APIConfig(api_key="sk-test", temperature=-0.1)
        
        with pytest.raises(ValueError):
            APIConfig(api_key="sk-test", temperature=1.1)
    
    def test_api_config_validation_max_tokens_range(self):
        """Test APIConfig validation for max_tokens range."""
        # Test valid max_tokens values
        APIConfig(api_key="sk-test", max_tokens=1)
        APIConfig(api_key="sk-test", max_tokens=2000)
        APIConfig(api_key="sk-test", max_tokens=4000)
        
        # Test invalid max_tokens values
        with pytest.raises(ValueError):
            APIConfig(api_key="sk-test", max_tokens=0)
        
        with pytest.raises(ValueError):
            APIConfig(api_key="sk-test", max_tokens=4001)
    
    def test_api_config_get_headers(self):
        """Test the get_headers method returns correct headers."""
        config = APIConfig(api_key="sk-test1234567890")
        
        headers = config.get_headers()
        
        assert headers == {
            "Authorization": "Bearer sk-test1234567890",
            "Content-Type": "application/json"
        }
    
    def test_api_config_string_representation(self):
        """Test the string representation hides sensitive API key."""
        config = APIConfig(
            api_key="sk-sensitive-key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.8
        )
        
        # API key should not appear in string representation
        str_repr = str(config)
        assert "sk-sensitive-key" not in str_repr
        assert "base_url=https://api.deepseek.com" in str_repr
        assert "model=deepseek-chat" in str_repr
        assert "temperature=0.8" in str_repr


class TestDatabaseConfig:
    """Test cases for DatabaseConfig model."""
    
    def test_database_config_creation(self):
        """Test creating a DatabaseConfig with valid data."""
        config = DatabaseConfig(
            persist_directory="./custom_knowledge_base",
            collection_name="custom_documents"
        )
        
        assert config.persist_directory == "./custom_knowledge_base"
        assert config.collection_name == "custom_documents"
    
    def test_database_config_default_values(self):
        """Test DatabaseConfig uses default values when not provided."""
        config = DatabaseConfig()
        
        assert config.persist_directory == "./knowledge_base"
        assert config.collection_name == "documents"
    
    def test_database_config_validation_empty_persist_directory(self):
        """Test DatabaseConfig validation for empty persist directory."""
        with pytest.raises(ValueError, match="Persist directory cannot be empty"):
            DatabaseConfig(persist_directory="")
        
        with pytest.raises(ValueError, match="Persist directory cannot be empty"):
            DatabaseConfig(persist_directory="   ")


class TestAppConfig:
    """Test cases for AppConfig model."""
    
    def test_app_config_creation(self):
        """Test creating an AppConfig with valid data."""
        api_config = APIConfig(api_key="sk-test123")
        db_config = DatabaseConfig(persist_directory="./test_db")
        
        app_config = AppConfig(
            api_config=api_config,
            db_config=db_config,
            log_level="debug",
            enable_gpu=False,
            max_conversation_history=100
        )
        
        assert app_config.api_config == api_config
        assert app_config.db_config == db_config
        assert app_config.log_level == "debug"
        assert app_config.enable_gpu is False
        assert app_config.max_conversation_history == 100
    
    def test_app_config_default_values(self):
        """Test AppConfig uses default values when not provided."""
        api_config = APIConfig(api_key="sk-test123")
        
        app_config = AppConfig(api_config=api_config)
        
        assert app_config.db_config.persist_directory == "./knowledge_base"
        assert app_config.db_config.collection_name == "documents"
        assert app_config.log_level == "info"
        assert app_config.enable_gpu is True
        assert app_config.max_conversation_history == 50
    
    def test_app_config_validation_log_level(self):
        """Test AppConfig validation for log level."""
        api_config = APIConfig(api_key="sk-test123")
        
        # Test valid log levels
        AppConfig(api_config=api_config, log_level="debug")
        AppConfig(api_config=api_config, log_level="info")
        AppConfig(api_config=api_config, log_level="warning")
        AppConfig(api_config=api_config, log_level="error")
        
        # Test invalid log level
        with pytest.raises(ValueError):
            AppConfig(api_config=api_config, log_level="invalid")
    
    def test_app_config_validation_max_conversation_history(self):
        """Test AppConfig validation for max_conversation_history range."""
        api_config = APIConfig(api_key="sk-test123")
        
        # Test valid values
        AppConfig(api_config=api_config, max_conversation_history=1)
        AppConfig(api_config=api_config, max_conversation_history=50)
        AppConfig(api_config=api_config, max_conversation_history=1000)
        
        # Test invalid values
        with pytest.raises(ValueError):
            AppConfig(api_config=api_config, max_conversation_history=0)
        
        with pytest.raises(ValueError):
            AppConfig(api_config=api_config, max_conversation_history=1001)


if __name__ == "__main__":
    # Run the tests if this file is executed directly
    pytest.main([__file__, "-v"])
