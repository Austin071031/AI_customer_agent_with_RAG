"""
Chat Manager for AI Customer Agent.

This module provides the core chat management functionality that orchestrates
between the DeepSeek API service and knowledge base. It handles conversation
history, context management, and response generation with knowledge base integration.
"""

import logging
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime

from ..models.chat_models import ChatMessage
from ..services.deepseek_service import DeepSeekService, DeepSeekAPIError
from ..services.knowledge_base import KnowledgeBaseManager, KnowledgeBaseError


class ChatManagerError(Exception):
    """Custom exception for Chat Manager related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class ChatManager:
    """
    Manages chat conversations with AI agent and knowledge base integration.
    
    This class orchestrates between the DeepSeek API service and knowledge base
    to provide intelligent responses with context from local documents.
    """
    
    def __init__(self, deepseek_service: DeepSeekService, kb_manager: KnowledgeBaseManager):
        """
        Initialize the chat manager with required services.
        
        Args:
            deepseek_service: DeepSeekService instance for AI responses
            kb_manager: KnowledgeBaseManager instance for knowledge base access
            
        Raises:
            ChatManagerError: If services are not properly initialized
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate and store service dependencies
        if not isinstance(deepseek_service, DeepSeekService):
            raise ChatManagerError("DeepSeekService instance required")
        if not isinstance(kb_manager, KnowledgeBaseManager):
            raise ChatManagerError("KnowledgeBaseManager instance required")
            
        self.deepseek_service = deepseek_service
        self.kb_manager = kb_manager
        
        # Initialize conversation history
        self.conversation_history: List[ChatMessage] = []
        
        # Configuration settings
        self.max_history_length = 20  # Maximum number of messages to keep in history
        self.knowledge_base_threshold = -0.5  # Similarity threshold for KB results (adjusted for negative scores)
        self.max_kb_context_length = 1500  # Maximum characters for KB context
        
        self.logger.info("ChatManager initialized successfully")
        
    async def process_message(self, user_message: str, use_knowledge_base: bool = True) -> str:
        """
        Process a user message and generate an AI response.
        
        This method orchestrates the entire response generation process:
        1. Optionally searches knowledge base for relevant context
        2. Builds the conversation context with history and KB context
        3. Calls DeepSeek API for response generation
        4. Updates conversation history
        
        Args:
            user_message: The user's input message
            use_knowledge_base: Whether to use knowledge base for context
            
        Returns:
            AI-generated response as string
            
        Raises:
            ChatManagerError: If message processing fails
            DeepSeekAPIError: If DeepSeek API call fails
        """
        if not user_message or not user_message.strip():
            raise ChatManagerError("Empty user message provided")
            
        try:
            self.logger.info(f"Processing user message: {user_message[:50]}...")
            
            # Step 1: Search knowledge base for relevant context if enabled
            kb_context = ""
            if use_knowledge_base:
                kb_context = await self._get_knowledge_base_context(user_message)
                self.logger.debug(f"Retrieved KB context: {len(kb_context)} characters")
            
            # Step 2: Build conversation messages with context
            messages = self._build_conversation_messages(user_message, kb_context)
            
            # Step 3: Get AI response from DeepSeek API
            ai_response = await self.deepseek_service.chat_completion(messages)
            
            # Step 4: Update conversation history
            self._update_conversation_history(user_message, ai_response)
            
            self.logger.info("Successfully processed user message")
            return ai_response
            
        except DeepSeekAPIError as e:
            self.logger.error(f"DeepSeek API error in process_message: {str(e)}")
            raise
        except KnowledgeBaseError as e:
            self.logger.error(f"Knowledge base error in process_message: {str(e)}")
            # Continue without KB context if KB fails
            if use_knowledge_base:
                self.logger.warning("Falling back to response without knowledge base context")
                return await self.process_message(user_message, use_knowledge_base=False)
            else:
                raise ChatManagerError(f"Knowledge base error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in process_message: {str(e)}")
            raise ChatManagerError(f"Failed to process message: {str(e)}")
            
    async def stream_message(self, user_message: str, use_knowledge_base: bool = True) -> AsyncGenerator[str, None]:
        """
        Process a user message and stream the AI response.
        
        This method provides real-time streaming of AI responses while
        maintaining the same context and knowledge base integration.
        
        Args:
            user_message: The user's input message
            use_knowledge_base: Whether to use knowledge base for context
            
        Yields:
            String chunks from the streaming AI response
            
        Raises:
            ChatManagerError: If message processing fails
            DeepSeekAPIError: If DeepSeek API call fails
        """
        if not user_message or not user_message.strip():
            raise ChatManagerError("Empty user message provided")
            
        try:
            self.logger.info(f"Streaming response for user message: {user_message[:50]}...")
            
            # Step 1: Search knowledge base for relevant context if enabled
            kb_context = ""
            if use_knowledge_base:
                kb_context = await self._get_knowledge_base_context(user_message)
                self.logger.debug(f"Retrieved KB context: {len(kb_context)} characters")
            
            # Step 2: Build conversation messages with context
            messages = self._build_conversation_messages(user_message, kb_context)
            
            # Step 3: Stream AI response from DeepSeek API
            full_response = ""
            async for chunk in self.deepseek_service.stream_chat(messages):
                full_response += chunk
                yield chunk
                
            # Step 4: Update conversation history with complete response
            self._update_conversation_history(user_message, full_response)
            
            self.logger.info("Successfully streamed response")
            
        except DeepSeekAPIError as e:
            self.logger.error(f"DeepSeek API error in stream_message: {str(e)}")
            raise
        except KnowledgeBaseError as e:
            self.logger.error(f"Knowledge base error in stream_message: {str(e)}")
            # Continue without KB context if KB fails
            if use_knowledge_base:
                self.logger.warning("Falling back to streaming without knowledge base context")
                async for chunk in self.stream_message(user_message, use_knowledge_base=False):
                    yield chunk
            else:
                raise ChatManagerError(f"Knowledge base error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in stream_message: {str(e)}")
            raise ChatManagerError(f"Failed to stream message: {str(e)}")
            
    async def _get_knowledge_base_context(self, user_message: str) -> str:
        """
        Retrieve relevant context from knowledge base for the user message.
        
        This method searches the knowledge base for documents similar to the
        user's query and formats them into a context string for the AI.
        
        Args:
            user_message: The user's message to search for
            
        Returns:
            Formatted context string from knowledge base, or empty string if no results
        """
        try:
            # Search for similar documents in knowledge base
            similar_docs = self.kb_manager.search_similar(user_message, k=3)
            
            self.logger.info(f"Found {len(similar_docs)} similar documents for query: {user_message}")
            
            if not similar_docs:
                self.logger.debug("No relevant documents found in knowledge base")
                return ""
                
            # Debug: Log all found documents and their similarity scores
            for i, doc in enumerate(similar_docs):
                similarity_score = doc.metadata.get('similarity_score', 0)
                file_name = doc.file_path.split('/')[-1] if '/' in doc.file_path else doc.file_path
                file_name = file_name.split('\\')[-1]  # Handle Windows paths
                self.logger.info(f"Document {i+1}: {file_name}, Similarity: {similarity_score:.4f}, Threshold: {self.knowledge_base_threshold}")
                
            # Format the context from similar documents
            context_parts = []
            for doc in similar_docs:
                # Extract relevant portion of document content
                content_preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                similarity_score = doc.metadata.get('similarity_score', 0)
                
                # Only include documents above similarity threshold
                if similarity_score >= self.knowledge_base_threshold:
                    # Create document summary using available attributes
                    file_name = doc.file_path.split('/')[-1] if '/' in doc.file_path else doc.file_path
                    file_name = file_name.split('\\')[-1]  # Handle Windows paths
                    doc_summary = f"Document: {file_name} | Type: {doc.file_type} | ID: {doc.id}"
                    
                    context_parts.append(
                        f"Document: {doc_summary}\n"
                        f"Relevance: {similarity_score:.2f}\n"
                        f"Content: {content_preview}\n"
                    )
                    self.logger.info(f"Included document in context: {file_name} (score: {similarity_score:.4f})")
                else:
                    file_name = doc.file_path.split('/')[-1] if '/' in doc.file_path else doc.file_path
                    file_name = file_name.split('\\')[-1]  # Handle Windows paths
                    self.logger.info(f"Excluded document from context: {file_name} (score: {similarity_score:.4f} < threshold: {self.knowledge_base_threshold})")
            
            if not context_parts:
                self.logger.info("No documents met similarity threshold after filtering")
                return ""
                
            # Combine context parts and limit total length
            full_context = "\n---\n".join(context_parts)
            if len(full_context) > self.max_kb_context_length:
                full_context = full_context[:self.max_kb_context_length] + "..."
                
            self.logger.info(f"Generated KB context with {len(context_parts)} documents, total length: {len(full_context)} characters")
            return full_context
            
        except KnowledgeBaseError as e:
            self.logger.warning(f"Knowledge base search failed: {str(e)}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error in KB context retrieval: {str(e)}")
            return ""
            
    def _build_conversation_messages(self, user_message: str, kb_context: str = "") -> List[Dict]:
        """
        Build the conversation messages for the AI API call.
        
        This method constructs the message list including system prompt,
        conversation history, knowledge base context, and current user message.
        
        Args:
            user_message: The current user message
            kb_context: Context from knowledge base (optional)
            
        Returns:
            List of message dictionaries for AI API call
        """
        messages = []
        
        # System message with instructions
        system_content = (
            "You are a helpful customer service assistant. "
            "Provide clear, accurate, and friendly responses. "
            "Use the provided context when relevant to answer questions accurately."
        )
        
        # Add knowledge base context if available
        if kb_context:
            system_content += (
                f"\n\nHere is some relevant information from our knowledge base:\n"
                f"{kb_context}\n\n"
                "Please use this information to provide accurate responses. "
                "If the information doesn't fully answer the question, "
                "provide the best answer you can based on your general knowledge."
            )
            
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history (limited to prevent token overflow)
        for msg in self.conversation_history[-self.max_history_length:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
            
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        self.logger.debug(f"Built conversation with {len(messages)} messages")
        return messages
        
    def _update_conversation_history(self, user_message: str, ai_response: str) -> None:
        """
        Update the conversation history with new messages.
        
        This method adds the user message and AI response to the conversation
        history and ensures the history doesn't exceed the maximum length.
        
        Args:
            user_message: The user's message to add
            ai_response: The AI's response to add
        """
        # Add user message to history
        user_chat_message = ChatMessage(
            role="user",
            content=user_message
        )
        self.conversation_history.append(user_chat_message)
        
        # Add AI response to history
        ai_chat_message = ChatMessage(
            role="assistant",
            content=ai_response
        )
        self.conversation_history.append(ai_chat_message)
        
        # Trim history if it exceeds maximum length
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 for user+assistant pairs
            # Keep only the most recent messages
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]
            self.logger.debug(f"Trimmed conversation history to {len(self.conversation_history)} messages")
            
    def get_conversation_history(self) -> List[ChatMessage]:
        """
        Get the current conversation history.
        
        Returns:
            List of ChatMessage objects representing the conversation history
        """
        return self.conversation_history.copy()
        
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        previous_count = len(self.conversation_history)
        self.conversation_history.clear()
        self.logger.info(f"Cleared conversation history ({previous_count} messages)")
        
    def get_conversation_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current conversation state.
        
        Returns:
            Dictionary containing conversation statistics and information
        """
        user_messages = [msg for msg in self.conversation_history if msg.role == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg.role == "assistant"]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "max_history_length": self.max_history_length,
            "history_usage_percentage": (len(self.conversation_history) / (self.max_history_length * 2)) * 100
        }
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the chat manager and its dependencies.
        
        Returns:
            True if all components are healthy, False otherwise
        """
        try:
            # Check DeepSeek service health
            deepseek_healthy = await self.deepseek_service.health_check()
            if not deepseek_healthy:
                self.logger.warning("DeepSeek service health check failed")
                return False
                
            # Check knowledge base health
            kb_healthy = self.kb_manager.health_check()
            if not kb_healthy:
                self.logger.warning("Knowledge base health check failed")
                return False
                
            # Check internal state
            if (not isinstance(self.conversation_history, list) or 
                not hasattr(self, 'max_history_length')):
                self.logger.warning("Chat manager internal state inconsistent")
                return False
                
            self.logger.debug("Chat manager health check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
            
    def update_configuration(self, max_history_length: Optional[int] = None, 
                           knowledge_base_threshold: Optional[float] = None,
                           max_kb_context_length: Optional[int] = None) -> None:
        """
        Update chat manager configuration settings.
        
        Args:
            max_history_length: New maximum history length (optional)
            knowledge_base_threshold: New KB similarity threshold (optional)
            max_kb_context_length: New maximum KB context length (optional)
        """
        if max_history_length is not None and max_history_length > 0:
            self.max_history_length = max_history_length
            self.logger.info(f"Updated max_history_length to {max_history_length}")
            
        if knowledge_base_threshold is not None and 0 <= knowledge_base_threshold <= 1:
            self.knowledge_base_threshold = knowledge_base_threshold
            self.logger.info(f"Updated knowledge_base_threshold to {knowledge_base_threshold}")
            
        if max_kb_context_length is not None and max_kb_context_length > 0:
            self.max_kb_context_length = max_kb_context_length
            self.logger.info(f"Updated max_kb_context_length to {max_kb_context_length}")
