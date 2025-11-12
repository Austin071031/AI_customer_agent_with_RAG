"""
Dependency injection module for FastAPI backend.

This module provides dependency functions for accessing services from the
application state, avoiding circular imports between main.py and endpoints.
"""

from fastapi import HTTPException, status
from ..services.chat_manager import ChatManager
from ..services.knowledge_base import KnowledgeBaseManager
from ..services.config_manager import ConfigManager

# Import the global app_state from state module
from .state import app_state


def get_chat_manager() -> ChatManager:
    """
    Get the chat manager instance from application state.
    
    Returns:
        ChatManager instance
        
    Raises:
        HTTPException: If chat manager is not available
    """
    chat_manager = app_state.get("chat_manager")
    if not chat_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not available"
        )
    return chat_manager


def get_kb_manager() -> KnowledgeBaseManager:
    """
    Get the knowledge base manager instance from application state.
    
    Returns:
        KnowledgeBaseManager instance
        
    Raises:
        HTTPException: If knowledge base manager is not available
    """
    kb_manager = app_state.get("kb_manager")
    if not kb_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base service not available"
        )
    return kb_manager


def get_config_manager() -> ConfigManager:
    """
    Get the configuration manager instance from application state.
    
    Returns:
        ConfigManager instance
        
    Raises:
        HTTPException: If configuration manager is not available
    """
    config_manager = app_state.get("config_manager")
    if not config_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration service not available"
        )
    return config_manager
