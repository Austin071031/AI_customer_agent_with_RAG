"""
Chat endpoints for AI Customer Agent API.

This module provides REST endpoints for chat functionality including
message processing, streaming, and conversation history management.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...models.chat_models import ChatMessage
from ...services.chat_manager import ChatManager, ChatManagerError
from ..dependencies import get_chat_manager

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class ChatRequest(BaseModel):
    """
    Request model for chat message processing.
    
    Attributes:
        message: The user's message to process
        use_knowledge_base: Whether to use knowledge base for context (default: True)
        stream: Whether to stream the response (default: False)
    """
    
    message: str = Field(..., min_length=1, max_length=4000, description="User message to process")
    use_knowledge_base: bool = Field(default=True, description="Use knowledge base for context")
    stream: bool = Field(default=False, description="Stream the response in real-time")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "How can I reset my password?",
                "use_knowledge_base": True,
                "stream": False
            }
        }
    }


class ChatResponse(BaseModel):
    """
    Response model for non-streaming chat responses.
    
    Attributes:
        response: The AI-generated response
        message_id: Unique identifier for the response message
        conversation_length: Current number of messages in the conversation
        used_knowledge_base: Whether knowledge base was used for context
    """
    
    response: str = Field(..., description="AI-generated response")
    message_id: str = Field(..., description="Unique identifier for the response message")
    conversation_length: int = Field(..., description="Current number of messages in conversation")
    used_knowledge_base: bool = Field(..., description="Whether knowledge base was used")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "To reset your password, please visit the account settings page...",
                "message_id": "123e4567-e89b-12d3-a456-426614174000",
                "conversation_length": 4,
                "used_knowledge_base": True
            }
        }
    }


class ConversationHistoryResponse(BaseModel):
    """
    Response model for conversation history.
    
    Attributes:
        messages: List of chat messages in the conversation
        total_messages: Total number of messages in history
        summary: Summary of conversation statistics
    """
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    total_messages: int = Field(..., description="Total number of messages")
    summary: dict = Field(..., description="Conversation statistics summary")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2024-01-01T12:00:00",
                        "message_id": "msg_001"
                    },
                    {
                        "role": "assistant",
                        "content": "Hi there! How can I help you?",
                        "timestamp": "2024-01-01T12:00:01",
                        "message_id": "msg_002"
                    }
                ],
                "total_messages": 2,
                "summary": {
                    "user_messages": 1,
                    "assistant_messages": 1,
                    "max_history_length": 20
                }
            }
        }
    }


class ClearHistoryResponse(BaseModel):
    """
    Response model for clearing conversation history.
    
    Attributes:
        message: Confirmation message
        cleared_messages: Number of messages cleared
    """
    
    message: str = Field(..., description="Confirmation message")
    cleared_messages: int = Field(..., description="Number of messages cleared")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Conversation history cleared successfully",
                "cleared_messages": 10
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error: Error type
        message: Human-readable error message
        detail: Additional error details (optional)
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process chat message",
    description="Process a user message and return an AI-generated response with optional knowledge base context.",
    responses={
        200: {"description": "Successfully processed message"},
        400: {"model": ErrorResponse, "description": "Invalid request or empty message"},
        503: {"model": ErrorResponse, "description": "Chat service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def process_chat_message(request: ChatRequest) -> ChatResponse:
    """
    Process a user message and generate an AI response.
    
    This endpoint handles both regular and streaming responses. For streaming,
    use the /stream endpoint instead.
    
    Args:
        request: Chat request containing message and options
        
    Returns:
        ChatResponse with AI-generated response and metadata
        
    Raises:
        HTTPException: If processing fails or service is unavailable
    """
    try:
        logger.info(f"Processing chat message (KB: {request.use_knowledge_base}, Stream: {request.stream})")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        # Process the message
        response_text = await chat_manager.process_message(
            user_message=request.message,
            use_knowledge_base=request.use_knowledge_base
        )
        
        # Get updated conversation history
        conversation_history = chat_manager.get_conversation_history()
        
        # Find the latest assistant message (the response we just generated)
        latest_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg.role == "assistant":
                latest_assistant_msg = msg
                break
        
        if not latest_assistant_msg:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve generated response"
            )
        
        return ChatResponse(
            response=response_text,
            message_id=latest_assistant_msg.message_id,
            conversation_length=len(conversation_history),
            used_knowledge_base=request.use_knowledge_base
        )
        
    except ChatManagerError as e:
        logger.error(f"Chat manager error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process message: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_chat_message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Stream chat response",
    description="Process a user message and stream the AI response in real-time.",
    responses={
        200: {"description": "Successfully streaming response"},
        400: {"model": ErrorResponse, "description": "Invalid request or empty message"},
        503: {"model": ErrorResponse, "description": "Chat service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def stream_chat_message(request: ChatRequest) -> StreamingResponse:
    """
    Stream AI response for a user message in real-time.
    
    This endpoint provides real-time streaming of AI responses, which is
    useful for creating responsive chat interfaces.
    
    Args:
        request: Chat request containing message and options
        
    Returns:
        StreamingResponse with real-time text chunks
        
    Raises:
        HTTPException: If processing fails or service is unavailable
    """
    try:
        logger.info(f"Streaming chat message (KB: {request.use_knowledge_base})")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        async def generate_stream():
            """Generator function for streaming response chunks."""
            try:
                async for chunk in chat_manager.stream_message(
                    user_message=request.message,
                    use_knowledge_base=request.use_knowledge_base
                ):
                    yield f"data: {chunk}\n\n"
                
                # Send completion signal
                yield "data: [DONE]\n\n"
                
            except ChatManagerError as e:
                error_message = f"data: ERROR: {str(e)}\n\n"
                yield error_message
            except Exception as e:
                error_message = f"data: ERROR: Internal server error: {str(e)}\n\n"
                yield error_message
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except ChatManagerError as e:
        logger.error(f"Chat manager error in stream: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to stream message: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in stream_chat_message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/history",
    response_model=ConversationHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation history",
    description="Retrieve the current conversation history with statistics.",
    responses={
        200: {"description": "Successfully retrieved conversation history"},
        503: {"model": ErrorResponse, "description": "Chat service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_conversation_history(
    limit: Optional[int] = Query(
        default=None, 
        ge=1, 
        le=1000, 
        description="Limit number of messages returned (optional)"
    )
) -> ConversationHistoryResponse:
    """
    Get the current conversation history.
    
    This endpoint returns the complete conversation history or a limited
    subset if the limit parameter is provided.
    
    Args:
        limit: Optional limit on number of messages to return
        
    Returns:
        ConversationHistoryResponse with messages and summary
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Retrieving conversation history")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        # Get conversation history
        conversation_history = chat_manager.get_conversation_history()
        
        # Apply limit if specified
        if limit and limit < len(conversation_history):
            conversation_history = conversation_history[-limit:]
        
        # Get conversation summary
        summary = chat_manager.get_conversation_summary()
        
        return ConversationHistoryResponse(
            messages=conversation_history,
            total_messages=len(conversation_history),
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.delete(
    "/history",
    response_model=ClearHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear conversation history",
    description="Clear the current conversation history and start fresh.",
    responses={
        200: {"description": "Successfully cleared conversation history"},
        503: {"model": ErrorResponse, "description": "Chat service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def clear_conversation_history() -> ClearHistoryResponse:
    """
    Clear the current conversation history.
    
    This endpoint removes all messages from the conversation history,
    effectively starting a new conversation.
    
    Returns:
        ClearHistoryResponse with confirmation and count of cleared messages
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Clearing conversation history")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        # Get current history count before clearing
        conversation_history = chat_manager.get_conversation_history()
        previous_count = len(conversation_history)
        
        # Clear the history
        chat_manager.clear_conversation()
        
        return ClearHistoryResponse(
            message="Conversation history cleared successfully",
            cleared_messages=previous_count
        )
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation history: {str(e)}"
        )


@router.get(
    "/summary",
    summary="Get conversation summary",
    description="Get statistics and summary of the current conversation.",
    responses={
        200: {"description": "Successfully retrieved conversation summary"},
        503: {"model": ErrorResponse, "description": "Chat service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_conversation_summary():
    """
    Get a summary of the current conversation.
    
    This endpoint provides statistics about the conversation without
    returning the actual message content.
    
    Returns:
        Dictionary with conversation statistics
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Retrieving conversation summary")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        # Get conversation summary
        summary = chat_manager.get_conversation_summary()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error retrieving conversation summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation summary: {str(e)}"
        )


# Health check endpoint for chat service
@router.get(
    "/health",
    summary="Chat service health check",
    description="Check the health status of the chat service.",
    responses={
        200: {"description": "Chat service is healthy"},
        503: {"model": ErrorResponse, "description": "Chat service is unhealthy"}
    }
)
async def chat_health_check():
    """
    Health check for the chat service.
    
    This endpoint performs a health check on the chat manager and its
    dependencies to ensure they are functioning properly.
    
    Returns:
        Dictionary with health status
        
    Raises:
        HTTPException: If chat service is unhealthy
    """
    try:
        logger.debug("Performing chat service health check")
        
        # Get chat manager instance
        chat_manager = get_chat_manager()
        
        # Perform health check
        is_healthy = await chat_manager.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat service health check failed"
            )
        
        return {
            "status": "healthy",
            "service": "chat_manager",
            "timestamp": "2024-01-01T12:00:00Z"  # This would be dynamic in production
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat service health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service health check failed: {str(e)}"
        )
