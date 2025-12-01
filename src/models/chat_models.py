"""
Chat-related data models for the AI Customer Agent with Dynamic Table Support.

This module defines the core data structures for handling chat messages,
knowledge base documents, and enhanced conversation context using Pydantic for validation.
The models support dynamic table integration for Excel data queries and intelligent
query routing between different data sources.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator


class QueryIntent(str, Enum):
    """
    Enumeration of possible query intents for intelligent routing.
    
    This helps the system determine which service to use for processing
    user queries based on their content and context.
    
    Values:
        EXCEL_DATA: Query about Excel data that should use Text-to-SQL service
        KNOWLEDGE_BASE: Query that should search the knowledge base
        GENERAL: General conversation query that uses only the AI model
        MIXED: Query that combines multiple data sources
        UNKNOWN: Intent could not be determined
    """
    EXCEL_DATA = "excel_data"
    KNOWLEDGE_BASE = "knowledge_base"
    GENERAL = "general"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """
    Enumeration of data sources used in responses.
    
    This tracks which data sources contributed to a response
    for transparency and debugging purposes.
    """
    DEEPSEEK_API = "deepseek_api"
    KNOWLEDGE_BASE = "knowledge_base"
    EXCEL_DATA = "excel_data"
    TEXT_TO_SQL = "text_to_sql"
    COMBINED = "combined"


class EnhancedChatMessage(BaseModel):
    """
    Enhanced chat message with additional context for dynamic table and Excel data support.
    
    This extends the basic ChatMessage with fields for tracking query intent,
    data sources used, and any SQL queries executed during processing.
    
    Attributes:
        role: The role of the message sender (user, assistant, or system)
        content: The text content of the message
        timestamp: When the message was created (auto-generated)
        message_id: Unique identifier for the message (auto-generated)
        query_intent: Detected intent of the user query
        data_sources: List of data sources used to generate the response
        sql_queries: List of SQL queries executed for Excel data (if any)
        excel_file_references: References to Excel files mentioned in the query
        confidence_score: Confidence score for intent detection (0.0 to 1.0)
        processing_time: Time taken to process the message (seconds)
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "role": "user",
            "content": "What were the total sales for product X last month?",
            "timestamp": "2024-01-01T12:00:00",
            "message_id": "123e4567-e89b-12d3-a456-426614174000",
            "query_intent": "excel_data",
            "data_sources": ["excel_data", "text_to_sql"],
            "sql_queries": ["SELECT SUM(Revenue) FROM sales_data WHERE Product = 'X'"],
            "excel_file_references": ["excel_123e4567-e89b-12d3-a456-426614174000"],
            "confidence_score": 0.95,
            "processing_time": 2.3
        }
    })
    
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    query_intent: QueryIntent = QueryIntent.UNKNOWN
    data_sources: List[DataSource] = Field(default_factory=list)
    sql_queries: List[str] = Field(default_factory=list)
    excel_file_references: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float = Field(default=0.0, ge=0.0)
    
    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v):
        """Validate that message content is not empty."""
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()
    
    @field_validator('excel_file_references')
    @classmethod
    def excel_references_must_have_correct_format(cls, v):
        """Validate that Excel file references follow the expected format."""
        for ref in v:
            if not ref.startswith('excel_'):
                raise ValueError('Excel file references must start with "excel_" prefix')
        return v
    
    def add_data_source(self, source: DataSource) -> None:
        """Add a data source to the message."""
        if source not in self.data_sources:
            self.data_sources.append(source)
    
    def add_sql_query(self, sql_query: str) -> None:
        """Add an executed SQL query to the message."""
        self.sql_queries.append(sql_query)
    
    def add_excel_reference(self, file_id: str) -> None:
        """Add an Excel file reference to the message."""
        if file_id not in self.excel_file_references:
            self.excel_file_references.append(file_id)
    
    def get_processing_summary(self) -> str:
        """
        Get a summary of the message processing.
        
        Returns:
            String containing processing details
        """
        sources = ", ".join(self.data_sources) if self.data_sources else "none"
        sql_count = len(self.sql_queries)
        excel_count = len(self.excel_file_references)
        return f"Intent: {self.query_intent.value} | Sources: {sources} | SQL: {sql_count} | Excel: {excel_count}"
    
    def __str__(self) -> str:
        """String representation of the enhanced chat message."""
        return f"EnhancedChatMessage(role={self.role}, intent={self.query_intent}, sources={len(self.data_sources)})"


class ConversationContext(BaseModel):
    """
    Represents the context of a conversation with dynamic table and Excel data support.
    
    This model tracks the conversation history, current context, and any
    Excel data references that are relevant to the ongoing conversation.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        messages: List of messages in the conversation
        current_intent: Current detected intent of the conversation
        excel_files_mentioned: Excel files that have been referenced in this conversation
        active_table_schemas: Dynamic table schemas currently relevant to the conversation
        conversation_start_time: When the conversation started
        last_activity_time: When the conversation was last active
        metadata: Additional metadata about the conversation
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174000",
            "messages": [],
            "current_intent": "general",
            "excel_files_mentioned": ["excel_123e4567-e89b-12d3-a456-426614174000"],
            "active_table_schemas": ["sales_data_sheet1"],
            "conversation_start_time": "2024-01-01T12:00:00",
            "last_activity_time": "2024-01-01T12:05:00",
            "metadata": {
                "user_id": "user_123",
                "session_duration": 300
            }
        }
    })
    
    conversation_id: str = Field(default_factory=lambda: f"conv_{str(uuid4())}")
    messages: List[EnhancedChatMessage] = Field(default_factory=list)
    current_intent: QueryIntent = QueryIntent.UNKNOWN
    excel_files_mentioned: List[str] = Field(default_factory=list)
    active_table_schemas: List[str] = Field(default_factory=list)
    conversation_start_time: datetime = Field(default_factory=datetime.now)
    last_activity_time: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('excel_files_mentioned')
    @classmethod
    def excel_files_must_have_correct_format(cls, v):
        """Validate that Excel file references follow the expected format."""
        for ref in v:
            if not ref.startswith('excel_'):
                raise ValueError('Excel file references must start with "excel_" prefix')
        return v
    
    def add_message(self, message: EnhancedChatMessage) -> None:
        """Add a message to the conversation and update context."""
        self.messages.append(message)
        self.last_activity_time = datetime.now()
        
        # Update current intent based on the latest message
        if message.query_intent != QueryIntent.UNKNOWN:
            self.current_intent = message.query_intent
        
        # Track Excel file references
        for file_id in message.excel_file_references:
            if file_id not in self.excel_files_mentioned:
                self.excel_files_mentioned.append(file_id)
    
    def get_recent_messages(self, count: int = 10) -> List[EnhancedChatMessage]:
        """
        Get the most recent messages from the conversation.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent EnhancedChatMessage objects
        """
        return self.messages[-count:] if self.messages else []
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            String containing conversation statistics
        """
        message_count = len(self.messages)
        user_messages = len([m for m in self.messages if m.role == "user"])
        assistant_messages = len([m for m in self.messages if m.role == "assistant"])
        excel_files = len(self.excel_files_mentioned)
        duration = (self.last_activity_time - self.conversation_start_time).total_seconds()
        
        return f"Messages: {message_count} (U:{user_messages}/A:{assistant_messages}) | Excel: {excel_files} | Duration: {duration:.1f}s"
    
    def clear_conversation(self) -> None:
        """Clear the conversation history while preserving the context."""
        self.messages.clear()
        self.last_activity_time = datetime.now()
    
    def __str__(self) -> str:
        """String representation of the conversation context."""
        return f"ConversationContext(id={self.conversation_id}, messages={len(self.messages)}, intent={self.current_intent})"


class QueryResult(BaseModel):
    """
    Represents the result of a query from any data source.
    
    This model standardizes the format for query results from different
    services (knowledge base, Excel data, Text-to-SQL, etc.) to enable
    consistent response generation.
    
    Attributes:
        success: Whether the query was successful
        data: The actual result data (varies by source)
        data_source: Source of the data
        query: Original query that generated this result
        execution_time: Time taken to execute the query (seconds)
        confidence: Confidence score for the result (0.0 to 1.0)
        metadata: Additional metadata about the result
        error_message: Error message if the query failed
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "data": [
                {"Product": "Widget A", "Total_Sales": 15000.0},
                {"Product": "Widget B", "Total_Sales": 12000.0}
            ],
            "data_source": "excel_data",
            "query": "What are the total sales by product?",
            "execution_time": 1.2,
            "confidence": 0.95,
            "metadata": {
                "rows_returned": 2,
                "table_used": "sales_data"
            },
            "error_message": None
        }
    })
    
    success: bool
    data: Any
    data_source: DataSource
    query: str
    execution_time: float = Field(ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v):
        """Validate that query is not empty."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    def is_from_excel(self) -> bool:
        """Check if the result is from Excel data."""
        return self.data_source == DataSource.EXCEL_DATA
    
    def is_from_knowledge_base(self) -> bool:
        """Check if the result is from the knowledge base."""
        return self.data_source == DataSource.KNOWLEDGE_BASE
    
    def get_result_summary(self) -> str:
        """
        Get a summary of the query result.
        
        Returns:
            String containing result statistics
        """
        status = "SUCCESS" if self.success else "FAILED"
        data_type = type(self.data).__name__
        
        if isinstance(self.data, list):
            data_info = f"List[{len(self.data)} items]"
        elif isinstance(self.data, dict):
            data_info = f"Dict[{len(self.data)} keys]"
        else:
            data_info = data_type
        
        return f"QueryResult(status={status}, source={self.data_source.value}, data={data_info}, time={self.execution_time:.2f}s)"
    
    def __str__(self) -> str:
        """String representation of the query result."""
        return f"QueryResult(success={self.success}, source={self.data_source.value}, time={self.execution_time:.2f}s)"


class ChatMessage(BaseModel):
    """
    Represents a single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender (user, assistant, or system)
        content: The text content of the message
        timestamp: When the message was created (auto-generated)
        message_id: Unique identifier for the message (auto-generated)
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "role": "user",
            "content": "Hello, how can you help me?",
            "timestamp": "2024-01-01T12:00:00",
            "message_id": "123e4567-e89b-12d3-a456-426614174000"
        }
    })
    
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    
    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v):
        """Validate that message content is not empty."""
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()
    
    def __str__(self) -> str:
        """String representation of the chat message."""
        return f"{self.role}: {self.content[:50]}..."


class KBDocument(BaseModel):
    """
    Represents a document in the knowledge base with metadata and embeddings.
    
    Attributes:
        id: Unique identifier for the document
        content: The text content extracted from the document
        metadata: Additional information about the document (source, author, etc.)
        file_path: Path to the original document file
        file_type: Type of the document (pdf, txt, docx, etc.)
        embedding: Optional vector embedding of the document content
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "doc_123",
            "content": "This is sample document content...",
            "metadata": {
                "source": "company_handbook",
                "author": "HR Department",
                "created_date": "2024-01-01"
            },
            "file_path": "/documents/company_handbook.pdf",
            "file_type": "pdf",
            "embedding": None
        }
    })
    
    id: str
    content: str
    metadata: Dict[str, Any]
    file_path: str
    file_type: str
    embedding: Optional[List[float]] = None
    
    def get_metadata_summary(self) -> str:
        """
        Get a summary of the document metadata.
        
        Returns:
            String containing key metadata information
        """
        file_name = self.file_path.split('/')[-1] if '/' in self.file_path else self.file_path
        return f"Document: {file_name} | Type: {self.file_type} | ID: {self.id}"
    
    def __str__(self) -> str:
        """String representation of the knowledge base document."""
        return f"KBDocument(id={self.id}, file_type={self.file_type}, content_length={len(self.content)})"
