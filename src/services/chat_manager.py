"""
Chat Manager for AI Customer Agent.

This module provides the core chat management functionality that orchestrates
between the DeepSeek API service and knowledge base. It handles conversation
history, context management, and response generation with knowledge base integration.
"""

import logging
import asyncio
import re
import json
from typing import List, Dict, Optional, AsyncGenerator, Any
from datetime import datetime

from ..models.chat_models import ChatMessage
from ..services.deepseek_service import DeepSeekService, DeepSeekAPIError
from ..services.knowledge_base import KnowledgeBaseManager, KnowledgeBaseError
from ..services.text_to_sql_service import TextToSQLService, TextToSQLError


class ChatManagerError(Exception):
    """Custom exception for Chat Manager related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class ChatManager:
    """
    Enhanced Chat Manager for AI Customer Agent with Text-to-SQL integration.
    
    This class orchestrates between the DeepSeek API service, knowledge base,
    and Text-to-SQL service to provide intelligent responses with intelligent
    query routing and mixed data source integration.
    """
    
    def __init__(self, deepseek_service: DeepSeekService, kb_manager: KnowledgeBaseManager,
                 text_to_sql_service: Optional[TextToSQLService] = None):
        """
        Initialize the enhanced chat manager with required services.
        
        Args:
            deepseek_service: DeepSeekService instance for AI responses
            kb_manager: KnowledgeBaseManager instance for knowledge base access
            text_to_sql_service: Optional TextToSQLService for Excel data queries
            
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
        self.text_to_sql_service = text_to_sql_service
        
        # Initialize conversation history
        self.conversation_history: List[ChatMessage] = []
        
        # Configuration settings
        self.max_history_length = 20  # Maximum number of messages to keep in history
        self.knowledge_base_threshold = 0.7  # Similarity threshold for KB results
        self.max_kb_context_length = 1500  # Maximum characters for KB context
        
        # Query intent detection settings
        self.excel_query_keywords = [
            'excel', 'spreadsheet', 'sheet', 'table', 'data', 'column', 'row',
            'sum', 'average', 'count', 'total', 'max', 'min', 'calculate',
            'how many', 'what is the', 'list all', 'show me', 'find',
            'records', 'entries', 'cells', 'worksheet', 'workbook', 'pivot',
            'filter', 'sort', 'group', 'aggregate', 'statistics'
        ]
        
        self.excel_query_patterns = [
            r'how many.*(row|record|entry|customer|product|item)',
            r'what is the.*(total|sum|average|count|maximum|minimum)',
            r'list all.*(data|record|entry|customer|product|item)',
            r'show me.*(data|table|sheet|information|details)',
            r'find.*(in|from).*(table|sheet|data|excel|spreadsheet)',
            r'count.*(row|record|entry|customer|product|item)',
            r'calculate.*(average|sum|total|maximum|minimum)',
            r'display.*(data|table|sheet|information)'
        ]
        
        self.logger.info("Enhanced ChatManager initialized successfully")
        
    def _detect_query_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Detect the intent of a user query to determine the appropriate service.
        
        This method analyzes the user message to determine if it's:
        - An Excel data query (should use Text-to-SQL)
        - A knowledge base query (should use KB search)
        - A general conversation (should use DeepSeek API directly)
        
        Args:
            user_message: The user's input message
            
        Returns:
            Dictionary with intent classification and confidence scores
        """
        user_message_lower = user_message.lower()
        
        # Check for Excel data query patterns
        excel_keyword_matches = sum(1 for keyword in self.excel_query_keywords 
                                  if keyword in user_message_lower)
        excel_pattern_matches = sum(1 for pattern in self.excel_query_patterns 
                                  if re.search(pattern, user_message_lower))
        
        # Enhanced Excel confidence calculation
        excel_confidence = (excel_keyword_matches * 0.2) + (excel_pattern_matches * 0.4)
        
        # Boost confidence if explicit Excel/spreadsheet terms are present
        if any(term in user_message_lower for term in ['excel', 'spreadsheet', 'sheet', 'workbook']):
            excel_confidence += 0.3
            
        # Boost confidence for data analysis terms
        if any(term in user_message_lower for term in ['total', 'sum', 'average', 'count', 'calculate']):
            excel_confidence += 0.2
            
        # Cap at 1.0
        excel_confidence = min(excel_confidence, 1.0)
        
        # Check for knowledge base query patterns
        kb_keywords = ['what is', 'how to', 'help with', 'information about', 
                      'policy', 'procedure', 'guide', 'manual', 'documentation',
                      'refund', 'reset', 'account', 'setup', 'user']
        kb_keyword_matches = sum(1 for keyword in kb_keywords 
                               if keyword in user_message_lower)
        
        kb_confidence = kb_keyword_matches * 0.2
        
        # Determine primary intent (enhanced thresholds to improve detection)
        if excel_confidence >= 0.3 and self.text_to_sql_service:
            intent = "excel_data"
            confidence = excel_confidence
        elif kb_confidence >= 0.2:
            intent = "knowledge_base"
            confidence = min(kb_confidence, 1.0)
        else:
            intent = "general"
            confidence = 0.8  # Default confidence for general queries
            
        self.logger.info(f"Query intent detected: {intent} (confidence: {confidence:.2f})")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "scores": {
                "excel_data": excel_confidence,
                "knowledge_base": kb_confidence,
                "general": 1.0 - max(excel_confidence, kb_confidence)
            }
        }
        
    async def _generate_natural_language_answer(self, user_message: str, sql_results: Dict[str, Any]) -> str:
        """
        Generate a natural language answer from SQL query results.
        
        Args:
            user_message: The original user query
            sql_results: Dictionary containing SQL query results from Text-to-SQL service
            
        Returns:
            Natural language answer based on the results
        """
        try:
            # Extract relevant information from SQL results
            result_count = sql_results.get("result_count", 0)
            results = sql_results.get("results", [])
            sql_query = sql_results.get("sql_query", "")
            
            # If no results, return a simple message
            if result_count == 0:
                return f"I searched the Excel data but didn't find any results for: {user_message}"
            
            # Prepare context for the LLM
            results_preview = results[:10]  # Limit to first 10 rows for context
            results_str = json.dumps(results_preview, indent=2, default=str)
            
            # Build messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful data analyst. Your task is to provide a clear, concise natural language answer based on SQL query results.

IMPORTANT GUIDELINES:
1. Analyze the SQL query results and answer the user's original question directly.
2. Provide specific numbers and data points from the results when relevant.
3. Keep the answer focused and avoid unnecessary explanations.
4. If the results contain multiple rows, summarize the key findings.
5. Do not mention SQL queries, databases, or technical details unless specifically asked.
6. Format numbers appropriately (e.g., use commas for thousands).
7. If the user asked for a calculation (sum, average, count, etc.), provide the computed result."""
                },
                {
                    "role": "user",
                    "content": f"""Original question: {user_message}

SQL Query Results (showing {result_count} total results):
{results_str}

Based on these results, please provide a direct answer to the user's question. Do not show raw data or SQL queries in your response."""
                }
            ]
            
            # Generate natural language answer using DeepSeek
            answer = await self.deepseek_service.chat_completion(messages)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Add a brief note about the data source if not already mentioned
            if "excel" not in answer.lower() and "data" not in answer.lower():
                answer += "\n\n(This information is based on the Excel data you uploaded.)"
            
            self.logger.info(f"Generated natural language answer from {result_count} SQL results")
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate natural language answer: {str(e)}")
            # Fall back to formatted results
            return self._format_results_fallback(user_message, sql_results)
    
    def _format_results_fallback(self, user_message: str, sql_results: Dict[str, Any]) -> str:
        """
        Fallback method to format SQL results when natural language generation fails.
        
        Args:
            user_message: The original user query
            sql_results: Dictionary containing SQL query results
            
        Returns:
            Formatted results string
        """
        result_count = sql_results.get("result_count", 0)
        results = sql_results.get("results", [])
        
        if result_count == 0:
            return f"I searched the Excel data but didn't find any results for: {user_message}"
        
        # Format results for display
        results_summary = f"Here's what I found in the Excel data for your query '{user_message}':\n\n"
        results_summary += f"Found {result_count} result(s):\n\n"
        
        # Show a preview of results (first 5 rows)
        preview_rows = results[:5]
        for i, row in enumerate(preview_rows, 1):
            results_summary += f"Row {i}: {row}\n"
        
        if result_count > 5:
            results_summary += f"\n... and {result_count - 5} more results"
        
        return results_summary

    async def _handle_excel_data_query(self, user_message: str, file_id: Optional[str] = None) -> str:
        """
        Handle Excel data queries using Text-to-SQL service.
        
        Args:
            user_message: The user's query about Excel data
            file_id: Optional specific Excel file ID to query
            
        Returns:
            Natural language answer based on Excel data results
        """
        if not self.text_to_sql_service:
            raise ChatManagerError("Text-to-SQL service not available for Excel data queries")
            
        try:
            self.logger.info(f"Processing Excel data query: {user_message}")
            
            # If no file_id provided, get the most recent Excel file
            if not file_id:
                # Get available Excel files from SQLite database
                try:
                    available_files = self.text_to_sql_service.sqlite_service.list_excel_files()
                    if not available_files:
                        return "I don't see any Excel files available for querying. Please upload an Excel file first."
                    
                    # Use the most recent file (first in the list since they're sorted by upload_time DESC)
                    file_id = available_files[0].id
                    self.logger.info(f"Using most recent Excel file: {available_files[0].file_name} (ID: {file_id})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to get available Excel files: {str(e)}")
                    return "I encountered an error while trying to access the Excel files. Please try uploading the file again."
            
            # Convert conversation history to context format
            conversation_context = []
            for msg in self.conversation_history[-4:]:  # Last 2 exchanges
                conversation_context.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Use Text-to-SQL service to process the query
            result = await self.text_to_sql_service.convert_to_sql(
                natural_language_query=user_message,
                file_id=file_id,
                conversation_context=conversation_context
            )
            
            # Generate natural language answer from the results
            response = await self._generate_natural_language_answer(user_message, result)
                
            self.logger.info(f"Successfully processed Excel data query, found {result['result_count']} results")
            return response
            
        except TextToSQLError as e:
            self.logger.error(f"Text-to-SQL query failed: {str(e)}")
            return f"I encountered an error while querying the Excel data: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error in Excel data query: {str(e)}")
            return f"Sorry, I couldn't process your Excel data query due to an unexpected error."
            
    async def process_message(self, user_message: str, use_knowledge_base: bool = True, 
                           file_id: Optional[str] = None) -> str:
        """
        Process a user message with intelligent routing to appropriate services.
        
        This enhanced method detects query intent and routes to:
        - Text-to-SQL service for Excel data queries
        - Knowledge base for document-based queries
        - DeepSeek API for general conversation
        
        Args:
            user_message: The user's input message
            use_knowledge_base: Whether to use knowledge base for context
            file_id: Optional specific Excel file ID to query
            
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
            
            # Step 1: Detect query intent for intelligent routing
            intent_result = self._detect_query_intent(user_message)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
            
            self.logger.info(f"Detected intent: {intent} (confidence: {confidence:.2f})")
            
            # Step 2: Route to appropriate service based on intent
            if intent == "excel_data" and self.text_to_sql_service:
                self.logger.info("Routing to Text-to-SQL service for Excel data query")
                response = await self._handle_excel_data_query(user_message, file_id)
            elif intent == "knowledge_base" and use_knowledge_base:
                self.logger.info("Routing to knowledge base enhanced response")
                response = await self._process_with_knowledge_base(user_message)
            else:
                self.logger.info("Routing to general conversation")
                response = await self._process_general_conversation(user_message, use_knowledge_base)
            
            # Step 3: Update conversation history
            self._update_conversation_history(user_message, response)
            
            self.logger.info("Successfully processed user message with intelligent routing")
            return response
            
        except DeepSeekAPIError as e:
            self.logger.error(f"DeepSeek API error in process_message: {str(e)}")
            raise
        except KnowledgeBaseError as e:
            self.logger.error(f"Knowledge base error in process_message: {str(e)}")
            # Fall back to general conversation without KB
            if use_knowledge_base:
                self.logger.warning("Falling back to general conversation without knowledge base")
                return await self.process_message(user_message, use_knowledge_base=False, file_id=file_id)
            else:
                raise ChatManagerError(f"Knowledge base error: {str(e)}")
        except TextToSQLError as e:
            self.logger.error(f"Text-to-SQL error in process_message: {str(e)}")
            # Fall back to general conversation for Excel data query failures
            self.logger.warning("Falling back to general conversation for failed Excel query")
            return await self._process_general_conversation(user_message, use_knowledge_base)
        except Exception as e:
            self.logger.error(f"Unexpected error in process_message: {str(e)}")
            raise ChatManagerError(f"Failed to process message: {str(e)}")
            
    async def _process_with_knowledge_base(self, user_message: str) -> str:
        """
        Process message with knowledge base context.
        
        Args:
            user_message: The user's input message
            
        Returns:
            AI-generated response with KB context
        """
        kb_context = await self._get_knowledge_base_context(user_message)
        messages = self._build_conversation_messages(user_message, kb_context)
        return await self.deepseek_service.chat_completion(messages)
        
    async def _process_general_conversation(self, user_message: str, use_knowledge_base: bool = True) -> str:
        """
        Process general conversation message.
        
        Args:
            user_message: The user's input message
            use_knowledge_base: Whether to use knowledge base for context
            
        Returns:
            AI-generated response
        """
        kb_context = ""
        if use_knowledge_base:
            kb_context = await self._get_knowledge_base_context(user_message)
        messages = self._build_conversation_messages(user_message, kb_context)
        return await self.deepseek_service.chat_completion(messages)
            
    async def stream_message(self, user_message: str, use_knowledge_base: bool = True,
                           file_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Process a user message and stream the AI response with intelligent routing.
        
        This enhanced method provides real-time streaming while routing to appropriate
        services based on query intent detection.
        
        Args:
            user_message: The user's input message
            use_knowledge_base: Whether to use knowledge base for context
            file_id: Optional specific Excel file ID to query
            
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
            
            # For Excel data queries, we can't stream Text-to-SQL results, so process normally
            intent_result = self._detect_query_intent(user_message)
            intent = intent_result["intent"]
            
            if intent == "excel_data" and self.text_to_sql_service:
                # For Excel data queries, process normally and stream the formatted result
                self.logger.info("Processing Excel data query for streaming")
                response = await self._handle_excel_data_query(user_message, file_id)
                
                # Stream the response in chunks to simulate streaming
                for i in range(0, len(response), 50):
                    yield response[i:i+50]
                    
                # Update conversation history
                self._update_conversation_history(user_message, response)
                return
            else:
                # For KB and general queries, use the original streaming approach
                kb_context = ""
                if use_knowledge_base:
                    kb_context = await self._get_knowledge_base_context(user_message)
                    self.logger.debug(f"Retrieved KB context: {len(kb_context)} characters")
                
                messages = self._build_conversation_messages(user_message, kb_context)
                
                full_response = ""
                async for chunk in self.deepseek_service.stream_chat(messages):
                    full_response += chunk
                    yield chunk
                    
                self._update_conversation_history(user_message, full_response)
            
            self.logger.info("Successfully streamed response with intelligent routing")
            
        except DeepSeekAPIError as e:
            self.logger.error(f"DeepSeek API error in stream_message: {str(e)}")
            raise
        except KnowledgeBaseError as e:
            self.logger.error(f"Knowledge base error in stream_message: {str(e)}")
            # Continue without KB context if KB fails
            if use_knowledge_base:
                self.logger.warning("Falling back to streaming without knowledge base context")
                async for chunk in self.stream_message(user_message, use_knowledge_base=False, file_id=file_id):
                    yield chunk
            else:
                raise ChatManagerError(f"Knowledge base error: {str(e)}")
        except TextToSQLError as e:
            self.logger.error(f"Text-to-SQL error in stream_message: {str(e)}")
            # Fall back to general conversation for Excel data query failures
            self.logger.warning("Falling back to general conversation for failed Excel query")
            async for chunk in self.stream_message(user_message, use_knowledge_base, file_id=None):
                yield chunk
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
                
            # Check Text-to-SQL service health if available
            if self.text_to_sql_service:
                try:
                    tts_health = await self.text_to_sql_service.health_check()
                    if not tts_health.get("status") == "healthy":
                        self.logger.warning("Text-to-SQL service health check failed")
                        return False
                except Exception as e:
                    self.logger.warning(f"Text-to-SQL service health check error: {str(e)}")
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
