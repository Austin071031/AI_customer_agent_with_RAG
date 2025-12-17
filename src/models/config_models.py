"""
Configuration data models for the AI Customer Agent.

This module defines the data structures for API configuration and
application settings using Pydantic for validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class APIConfig(BaseModel):
    """
    Configuration model for DeepSeek API settings.
    
    Attributes:
        api_key: The API key for DeepSeek API authentication
        base_url: Base URL for the DeepSeek API (default: "https://api.deepseek.com")
        model: The model to use for chat completions (default: "deepseek-chat")
        temperature: Controls randomness in response generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2000
        }
    })
    
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=1, le=4000)
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        """Validate that API key is not empty."""
        if not v or not v.strip():
            raise ValueError('API key cannot be empty')
        return v.strip()
    
    @field_validator('base_url')
    @classmethod
    def base_url_must_be_valid(cls, v):
        """Validate that base URL is a valid HTTP/HTTPS URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v
    
    def get_headers(self) -> dict:
        """
        Get the headers required for API authentication.
        
        Returns:
            Dictionary containing Authorization header with API key
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def __str__(self) -> str:
        """String representation of API configuration (hides sensitive API key)."""
        return f"APIConfig(base_url={self.base_url}, model={self.model}, temperature={self.temperature})"


class DatabaseConfig(BaseModel):
    """
    Configuration model for database settings.
    
    Attributes:
        persist_directory: Directory path for persisting the vector database
        collection_name: Name of the collection in the vector database
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "persist_directory": "./knowledge_base",
            "collection_name": "documents"
        }
    })
    
    persist_directory: str = "./knowledge_base"
    collection_name: str = "documents"
    
    @field_validator('persist_directory')
    @classmethod
    def persist_directory_must_be_valid(cls, v):
        """Validate that persist directory is a non-empty string."""
        if not v or not v.strip():
            raise ValueError('Persist directory cannot be empty')
        return v.strip()


class AppConfig(BaseModel):
    """
    Main application configuration model.
    
    Attributes:
        api_config: Configuration for DeepSeek API
        db_config: Configuration for the vector database
        log_level: Logging level (debug, info, warning, error)
        max_conversation_history: Maximum number of messages to keep in conversation history
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "api_config": {
                "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "db_config": {
                "persist_directory": "./knowledge_base",
                "collection_name": "documents"
            },
            "log_level": "info",
            "max_conversation_history": 50
        }
    })
    
    api_config: APIConfig
    db_config: DatabaseConfig = Field(default_factory=DatabaseConfig)
    log_level: str = Field(default="info", pattern="^(debug|info|warning|error)$")
    max_conversation_history: int = Field(default=50, ge=1, le=1000)
    
    @field_validator('log_level')
    @classmethod
    def log_level_must_be_valid(cls, v):
        """Validate that log level is one of the allowed values."""
        allowed_levels = ['debug', 'info', 'warning', 'error']
        if v.lower() not in allowed_levels:
            raise ValueError(f'Log level must be one of: {", ".join(allowed_levels)}')
        return v.lower()
