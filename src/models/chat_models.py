"""
Chat-related data models for the AI Customer Agent.

This module defines the core data structures for handling chat messages
and knowledge base documents using Pydantic for validation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict


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
