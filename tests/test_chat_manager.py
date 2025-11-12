"""
Unit tests for the ChatManager class in AI Customer Agent.

This module contains comprehensive tests for the ChatManager functionality,
including message processing, conversation history management, knowledge base
integration, and error handling scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, AsyncGenerator

from src.models.chat_models import ChatMessage
from src.services.chat_manager import ChatManager, ChatManagerError
from src.services.deepseek_service import DeepSeekService, DeepSeekAPIError
from src.services.knowledge_base import KnowledgeBaseManager, KnowledgeBaseError


class TestChatManager:
    """Test suite for ChatManager class."""
    
    @pytest.fixture
    def mock_deepseek_service(self):
        """Create a mock DeepSeekService for testing."""
        mock_service = Mock(spec=DeepSeekService)
        mock_service.chat_completion = AsyncMock(return_value="Mock AI response")
        mock_service.stream_chat = AsyncMock()
        mock_service.health_check = AsyncMock(return_value=True)
        return mock_service
        
    @pytest.fixture
    def mock_kb_manager(self):
        """Create a mock KnowledgeBaseManager for testing."""
        mock_kb = Mock(spec=KnowledgeBaseManager)
        mock_kb.search_similar = Mock(return_value=[])
        mock_kb.health_check = Mock(return_value=True)
        return mock_kb
        
    @pytest.fixture
    def chat_manager(self, mock_deepseek_service, mock_kb_manager):
        """Create a ChatManager instance with mocked dependencies."""
        return ChatManager(mock_deepseek_service, mock_kb_manager)
        
    @pytest.fixture
    def sample_kb_documents(self):
        """Create sample knowledge base documents for testing."""
        from src.models.chat_models import KBDocument
        
        return [
            KBDocument(
                id="doc1",
                content="This is document 1 about customer service policies.",
                metadata={
                    "file_path": "/docs/policy1.pdf",
                    "file_type": ".pdf",
                    "similarity_score": 0.85
                },
                file_path="/docs/policy1.pdf",
                file_type=".pdf"
            ),
            KBDocument(
                id="doc2",
                content="Document 2 contains information about product features.",
                metadata={
                    "file_path": "/docs/features.docx",
                    "file_type": ".docx",
                    "similarity_score": 0.72
                },
                file_path="/docs/features.docx",
                file_type=".docx"
            )
        ]
        
    # Test Initialization
    
    def test_initialization_success(self, mock_deepseek_service, mock_kb_manager):
        """Test successful ChatManager initialization with valid dependencies."""
        # Act
        chat_manager = ChatManager(mock_deepseek_service, mock_kb_manager)
        
        # Assert
        assert chat_manager.deepseek_service == mock_deepseek_service
        assert chat_manager.kb_manager == mock_kb_manager
        assert chat_manager.conversation_history == []
        assert chat_manager.max_history_length == 20
        assert chat_manager.knowledge_base_threshold == 0.7
        assert chat_manager.max_kb_context_length == 1500
        
    def test_initialization_invalid_deepseek_service(self, mock_kb_manager):
        """Test initialization fails with invalid DeepSeekService."""
        # Arrange
        invalid_service = "not_a_service"
        
        # Act & Assert
        with pytest.raises(ChatManagerError, match="DeepSeekService instance required"):
            ChatManager(invalid_service, mock_kb_manager)
            
    def test_initialization_invalid_kb_manager(self, mock_deepseek_service):
        """Test initialization fails with invalid KnowledgeBaseManager."""
        # Arrange
        invalid_kb = "not_a_kb_manager"
        
        # Act & Assert
        with pytest.raises(ChatManagerError, match="KnowledgeBaseManager instance required"):
            ChatManager(mock_deepseek_service, invalid_kb)
            
    # Test Message Processing
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test successful message processing without knowledge base."""
        # Arrange
        user_message = "Hello, how can you help me?"
        expected_response = "Mock AI response"
        
        # Act
        response = await chat_manager.process_message(user_message, use_knowledge_base=False)
        
        # Assert
        assert response == expected_response
        mock_deepseek_service.chat_completion.assert_called_once()
        mock_kb_manager.search_similar.assert_not_called()
        
        # Check conversation history was updated
        assert len(chat_manager.conversation_history) == 2
        assert chat_manager.conversation_history[0].role == "user"
        assert chat_manager.conversation_history[0].content == user_message
        assert chat_manager.conversation_history[1].role == "assistant"
        assert chat_manager.conversation_history[1].content == expected_response
        
    @pytest.mark.asyncio
    async def test_process_message_with_knowledge_base(self, chat_manager, mock_deepseek_service, mock_kb_manager, sample_kb_documents):
        """Test message processing with knowledge base integration."""
        # Arrange
        user_message = "Tell me about customer service policies"
        mock_kb_manager.search_similar.return_value = sample_kb_documents
        
        # Act
        response = await chat_manager.process_message(user_message, use_knowledge_base=True)
        
        # Assert
        assert response == "Mock AI response"
        mock_kb_manager.search_similar.assert_called_once_with(user_message, k=3)
        mock_deepseek_service.chat_completion.assert_called_once()
        
        # Verify the call included KB context in messages
        call_args = mock_deepseek_service.chat_completion.call_args
        messages = call_args[0][0]  # First positional argument
        system_message = messages[0]
        assert "knowledge base" in system_message["content"].lower()
        assert "customer service policies" in system_message["content"]
        
    @pytest.mark.asyncio
    async def test_process_message_empty_message(self, chat_manager):
        """Test processing empty message raises appropriate error."""
        # Act & Assert
        with pytest.raises(ChatManagerError, match="Empty user message provided"):
            await chat_manager.process_message("")
            
        with pytest.raises(ChatManagerError, match="Empty user message provided"):
            await chat_manager.process_message("   ")
            
    @pytest.mark.asyncio
    async def test_process_message_deepseek_api_error(self, chat_manager, mock_deepseek_service):
        """Test handling of DeepSeek API errors."""
        # Arrange
        user_message = "Test message"
        mock_deepseek_service.chat_completion.side_effect = DeepSeekAPIError("API timeout")
        
        # Act & Assert
        with pytest.raises(DeepSeekAPIError):
            await chat_manager.process_message(user_message)
            
    @pytest.mark.asyncio
    async def test_process_message_kb_error_fallback(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test fallback behavior when knowledge base fails."""
        # Arrange
        user_message = "Test message"
        mock_kb_manager.search_similar.side_effect = KnowledgeBaseError("KB search failed")
        
        # Act
        response = await chat_manager.process_message(user_message, use_knowledge_base=True)
        
        # Assert
        assert response == "Mock AI response"
        mock_kb_manager.search_similar.assert_called_once()
        mock_deepseek_service.chat_completion.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_process_message_history_trimming(self, chat_manager, mock_deepseek_service):
        """Test conversation history trimming when exceeding maximum length."""
        # Arrange
        chat_manager.max_history_length = 2  # Small limit for testing
        
        # Add initial messages to exceed limit
        for i in range(5):
            user_msg = ChatMessage(role="user", content=f"Message {i}")
            ai_msg = ChatMessage(role="assistant", content=f"Response {i}")
            chat_manager.conversation_history.extend([user_msg, ai_msg])
            
        initial_count = len(chat_manager.conversation_history)
        
        # Act
        await chat_manager.process_message("New message")
        
        # Assert
        # History should be trimmed to max_history_length * 2 (4 messages)
        assert len(chat_manager.conversation_history) == 4
        # Should keep the most recent messages
        assert chat_manager.conversation_history[2].content == "New message"
        
    # Test Streaming
    
    @pytest.mark.asyncio
    async def test_stream_message_success(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test successful message streaming."""
        # Arrange
        user_message = "Stream test message"
        
        # Create a mock async generator for streaming
        async def mock_stream():
            yield "Streaming "
            yield "response "
            yield "chunks"
            
        mock_deepseek_service.stream_chat = AsyncMock(return_value=mock_stream())
        
        # Act
        collected_chunks = []
        async for chunk in chat_manager.stream_message(user_message, use_knowledge_base=False):
            collected_chunks.append(chunk)
            
        full_response = "".join(collected_chunks)
        
        # Assert
        assert full_response == "Streaming response chunks"
        mock_deepseek_service.stream_chat.assert_called_once()
        mock_kb_manager.search_similar.assert_not_called()
        
        # Check conversation history was updated with full response
        assert len(chat_manager.conversation_history) == 2
        assert chat_manager.conversation_history[1].content == full_response
        
    @pytest.mark.asyncio
    async def test_stream_message_with_knowledge_base(self, chat_manager, mock_deepseek_service, mock_kb_manager, sample_kb_documents):
        """Test message streaming with knowledge base integration."""
        # Arrange
        user_message = "Stream with KB test"
        mock_kb_manager.search_similar.return_value = sample_kb_documents
        
        async def mock_stream():
            yield "Streaming with KB"
            
        mock_deepseek_service.stream_chat = AsyncMock(return_value=mock_stream())
        
        # Act
        collected_chunks = []
        async for chunk in chat_manager.stream_message(user_message, use_knowledge_base=True):
            collected_chunks.append(chunk)
            
        # Assert
        assert "".join(collected_chunks) == "Streaming with KB"
        mock_kb_manager.search_similar.assert_called_once_with(user_message, k=3)
        mock_deepseek_service.stream_chat.assert_called_once()
        
    # Test Knowledge Base Context Retrieval
    
    @pytest.mark.asyncio
    async def test_get_knowledge_base_context_success(self, chat_manager, mock_kb_manager, sample_kb_documents):
        """Test successful knowledge base context retrieval."""
        # Arrange
        user_message = "Customer service query"
        mock_kb_manager.search_similar.return_value = sample_kb_documents
        
        # Act
        context = await chat_manager._get_knowledge_base_context(user_message)
        
        # Assert
        assert context != ""
        assert "customer service policies" in context
        assert "product features" in context
        mock_kb_manager.search_similar.assert_called_once_with(user_message, k=3)
        
    @pytest.mark.asyncio
    async def test_get_knowledge_base_context_no_results(self, chat_manager, mock_kb_manager):
        """Test knowledge base context when no similar documents found."""
        # Arrange
        user_message = "Unrelated query"
        mock_kb_manager.search_similar.return_value = []
        
        # Act
        context = await chat_manager._get_knowledge_base_context(user_message)
        
        # Assert
        assert context == ""
        mock_kb_manager.search_similar.assert_called_once_with(user_message, k=3)
        
    @pytest.mark.asyncio
    async def test_get_knowledge_base_context_below_threshold(self, chat_manager, mock_kb_manager):
        """Test knowledge base context filtering by similarity threshold."""
        # Arrange
        user_message = "Low similarity query"
        low_similarity_docs = [
            Mock(
                content="Low relevance content",
                metadata={'similarity_score': 0.5},  # Below threshold of 0.7
                get_metadata_summary=Mock(return_value="Low relevance doc")
            )
        ]
        mock_kb_manager.search_similar.return_value = low_similarity_docs
        
        # Act
        context = await chat_manager._get_knowledge_base_context(user_message)
        
        # Assert
        assert context == ""  # No documents above threshold
        
    @pytest.mark.asyncio
    async def test_get_knowledge_base_context_length_limiting(self, chat_manager, mock_kb_manager):
        """Test knowledge base context length limiting."""
        # Arrange
        user_message = "Test query"
        long_content_doc = Mock(
            content="A" * 1000,  # Long content
            metadata={'similarity_score': 0.8},
            get_metadata_summary=Mock(return_value="Long document")
        )
        mock_kb_manager.search_similar.return_value = [long_content_doc]
        
        # Act
        context = await chat_manager._get_knowledge_base_context(user_message)
        
        # Assert
        assert len(context) <= chat_manager.max_kb_context_length + 3  # +3 for "..."
        
    # Test Conversation History Management
    
    def test_get_conversation_history(self, chat_manager):
        """Test retrieving conversation history."""
        # Arrange
        user_msg = ChatMessage(role="user", content="Test message")
        ai_msg = ChatMessage(role="assistant", content="Test response")
        chat_manager.conversation_history = [user_msg, ai_msg]
        
        # Act
        history = chat_manager.get_conversation_history()
        
        # Assert
        assert history == [user_msg, ai_msg]
        # Should return a copy, not the original list
        assert history is not chat_manager.conversation_history
        
    def test_clear_conversation(self, chat_manager):
        """Test clearing conversation history."""
        # Arrange
        user_msg = ChatMessage(role="user", content="Test message")
        ai_msg = ChatMessage(role="assistant", content="Test response")
        chat_manager.conversation_history = [user_msg, ai_msg]
        
        # Act
        chat_manager.clear_conversation()
        
        # Assert
        assert chat_manager.conversation_history == []
        
    def test_get_conversation_summary(self, chat_manager):
        """Test getting conversation summary statistics."""
        # Arrange
        for i in range(3):
            user_msg = ChatMessage(role="user", content=f"Message {i}")
            ai_msg = ChatMessage(role="assistant", content=f"Response {i}")
            chat_manager.conversation_history.extend([user_msg, ai_msg])
            
        # Act
        summary = chat_manager.get_conversation_summary()
        
        # Assert
        assert summary["total_messages"] == 6
        assert summary["user_messages"] == 3
        assert summary["assistant_messages"] == 3
        assert summary["max_history_length"] == 20
        assert summary["history_usage_percentage"] == (6 / 40) * 100  # 6 messages out of 40 max
        
    # Test Configuration Management
    
    def test_update_configuration_valid(self, chat_manager):
        """Test updating configuration with valid values."""
        # Act
        chat_manager.update_configuration(
            max_history_length=30,
            knowledge_base_threshold=0.8,
            max_kb_context_length=2000
        )
        
        # Assert
        assert chat_manager.max_history_length == 30
        assert chat_manager.knowledge_base_threshold == 0.8
        assert chat_manager.max_kb_context_length == 2000
        
    def test_update_configuration_partial(self, chat_manager):
        """Test updating only some configuration values."""
        # Store original values
        original_history = chat_manager.max_history_length
        original_threshold = chat_manager.knowledge_base_threshold
        
        # Act
        chat_manager.update_configuration(max_kb_context_length=1800)
        
        # Assert
        assert chat_manager.max_kb_context_length == 1800
        assert chat_manager.max_history_length == original_history  # Unchanged
        assert chat_manager.knowledge_base_threshold == original_threshold  # Unchanged
        
    def test_update_configuration_invalid_values(self, chat_manager):
        """Test updating configuration with invalid values (should be ignored)."""
        # Store original values
        original_history = chat_manager.max_history_length
        original_threshold = chat_manager.knowledge_base_threshold
        original_context = chat_manager.max_kb_context_length
        
        # Act
        chat_manager.update_configuration(
            max_history_length=-5,  # Invalid
            knowledge_base_threshold=1.5,  # Invalid
            max_kb_context_length=0  # Invalid
        )
        
        # Assert - values should remain unchanged
        assert chat_manager.max_history_length == original_history
        assert chat_manager.knowledge_base_threshold == original_threshold
        assert chat_manager.max_kb_context_length == original_context
        
    # Test Health Check
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test successful health check with all components healthy."""
        # Act
        result = await chat_manager.health_check()
        
        # Assert
        assert result is True
        mock_deepseek_service.health_check.assert_called_once()
        mock_kb_manager.health_check.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_health_check_deepseek_failure(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test health check when DeepSeek service is unhealthy."""
        # Arrange
        mock_deepseek_service.health_check.return_value = False
        
        # Act
        result = await chat_manager.health_check()
        
        # Assert
        assert result is False
        
    @pytest.mark.asyncio
    async def test_health_check_kb_failure(self, chat_manager, mock_deepseek_service, mock_kb_manager):
        """Test health check when knowledge base is unhealthy."""
        # Arrange
        mock_kb_manager.health_check.return_value = False
        
        # Act
        result = await chat_manager.health_check()
        
        # Assert
        assert result is False
        
    @pytest.mark.asyncio
    async def test_health_check_internal_state_error(self, chat_manager):
        """Test health check when internal state is inconsistent."""
        # Arrange
        chat_manager.conversation_history = "not_a_list"  # Invalid state
        
        # Act
        result = await chat_manager.health_check()
        
        # Assert
        assert result is False
        
    # Test Message Building
    
    def test_build_conversation_messages_without_kb(self, chat_manager):
        """Test building conversation messages without knowledge base context."""
        # Arrange
        user_message = "Test message"
        
        # Add some conversation history
        user_msg = ChatMessage(role="user", content="Previous message")
        ai_msg = ChatMessage(role="assistant", content="Previous response")
        chat_manager.conversation_history = [user_msg, ai_msg]
        
        # Act
        messages = chat_manager._build_conversation_messages(user_message)
        
        # Assert
        assert len(messages) == 4  # system + 2 history + current user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous message"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous response"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == user_message
        
    def test_build_conversation_messages_with_kb(self, chat_manager):
        """Test building conversation messages with knowledge base context."""
        # Arrange
        user_message = "Test message"
        kb_context = "KB context content"
        
        # Act
        messages = chat_manager._build_conversation_messages(user_message, kb_context)
        
        # Assert
        assert len(messages) == 2  # system + current user
        system_message = messages[0]
        assert system_message["role"] == "system"
        assert "knowledge base" in system_message["content"].lower()
        assert kb_context in system_message["content"]
        
    # Test Error Handling and Edge Cases
    
    @pytest.mark.asyncio
    async def test_process_message_unexpected_error(self, chat_manager, mock_deepseek_service):
        """Test handling of unexpected errors during message processing."""
        # Arrange
        user_message = "Test message"
        mock_deepseek_service.chat_completion.side_effect = Exception("Unexpected error")
        
        # Act & Assert
        with pytest.raises(ChatManagerError, match="Failed to process message"):
            await chat_manager.process_message(user_message)
            
    @pytest.mark.asyncio
    async def test_stream_message_unexpected_error(self, chat_manager, mock_deepseek_service):
        """Test handling of unexpected errors during message streaming."""
        # Arrange
        user_message = "Test message"
        mock_deepseek_service.stream_chat.side_effect = Exception("Unexpected streaming error")
        
        # Act & Assert
        with pytest.raises(ChatManagerError, match="Failed to stream message"):
            async for _ in chat_manager.stream_message(user_message):
                pass
                
    def test_conversation_history_management(self, chat_manager):
        """Test various conversation history management scenarios."""
        # Test empty history
        assert chat_manager.get_conversation_history() == []
        assert chat_manager.get_conversation_summary()["total_messages"] == 0
        
        # Test adding messages
        chat_manager._update_conversation_history("User message 1", "AI response 1")
        assert len(chat_manager.conversation_history) == 2
        
        # Test clearing history
        chat_manager.clear_conversation()
        assert len(chat_manager.conversation_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
