"""
Configuration endpoints for AI Customer Agent API.

This module provides REST endpoints for managing application configuration
including API settings, database configuration, and application preferences.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...models.config_models import APIConfig, DatabaseConfig, AppConfig
from ...services.config_manager import ConfigManager, ConfigManagerError
from ..dependencies import get_config_manager

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response Models
class UpdateAPIConfigRequest(BaseModel):
    """
    Request model for updating API configuration.
    
    Attributes:
        api_key: DeepSeek API key (optional, will not update if not provided)
        base_url: Base URL for DeepSeek API (optional)
        model: Model name for chat completions (optional)
        temperature: Temperature setting (0.0 to 1.0, optional)
        max_tokens: Maximum tokens in response (optional)
    """
    
    api_key: Optional[str] = Field(None, min_length=1, description="DeepSeek API key")
    base_url: Optional[str] = Field(None, min_length=1, description="Base URL for DeepSeek API")
    model: Optional[str] = Field(None, min_length=1, description="Model name for chat completions")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Temperature setting")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens in response")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }
    }


class UpdateDBConfigRequest(BaseModel):
    """
    Request model for updating database configuration.
    
    Attributes:
        persist_directory: Directory path for persisting vector database (optional)
        collection_name: Name of the collection in vector database (optional)
    """
    
    persist_directory: Optional[str] = Field(None, min_length=1, description="Persist directory path")
    collection_name: Optional[str] = Field(None, min_length=1, description="Collection name")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "persist_directory": "./knowledge_base",
                "collection_name": "documents"
            }
        }
    }


class UpdateAppConfigRequest(BaseModel):
    """
    Request model for updating application configuration.
    
    Attributes:
        log_level: Logging level (debug, info, warning, error, optional)
        enable_gpu: Whether to enable GPU acceleration (optional)
        max_conversation_history: Maximum conversation history length (optional)
    """
    
    log_level: Optional[str] = Field(None, pattern="^(debug|info|warning|error)$", description="Logging level")
    enable_gpu: Optional[bool] = Field(None, description="Enable GPU acceleration")
    max_conversation_history: Optional[int] = Field(None, ge=1, le=1000, description="Max conversation history length")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "log_level": "info",
                "enable_gpu": True,
                "max_conversation_history": 50
            }
        }
    }


class ConfigResponse(BaseModel):
    """
    Response model for configuration retrieval.
    
    Attributes:
        api_config: API configuration settings
        db_config: Database configuration settings
        app_config: Application configuration settings
        last_modified: Last modification timestamp (ISO format)
    """
    
    api_config: APIConfig = Field(..., description="API configuration")
    db_config: DatabaseConfig = Field(..., description="Database configuration")
    app_config: AppConfig = Field(..., description="Application configuration")
    last_modified: str = Field(..., description="Last modification timestamp")
    
    model_config = {
        "json_schema_extra": {
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
                "app_config": {
                    "log_level": "info",
                    "enable_gpu": True,
                    "max_conversation_history": 50
                },
                "last_modified": "2024-01-01T12:00:00Z"
            }
        }
    }


class UpdateConfigResponse(BaseModel):
    """
    Response model for configuration updates.
    
    Attributes:
        message: Success message
        updated_sections: List of configuration sections that were updated
        restart_required: Whether application restart is required for changes to take effect
    """
    
    message: str = Field(..., description="Update result message")
    updated_sections: list = Field(..., description="List of updated configuration sections")
    restart_required: bool = Field(..., description="Whether restart is required")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Configuration updated successfully",
                "updated_sections": ["api_config", "app_config"],
                "restart_required": False
            }
        }
    }


class ResetConfigResponse(BaseModel):
    """
    Response model for configuration reset.
    
    Attributes:
        message: Success message
        reset_sections: List of configuration sections that were reset
        default_values: The default configuration values applied
    """
    
    message: str = Field(..., description="Reset result message")
    reset_sections: list = Field(..., description="List of reset configuration sections")
    default_values: dict = Field(..., description="Default configuration values applied")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Configuration reset to defaults",
                "reset_sections": ["api_config", "db_config", "app_config"],
                "default_values": {
                    "api_config": {
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-chat",
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    "db_config": {
                        "persist_directory": "./knowledge_base",
                        "collection_name": "documents"
                    },
                    "app_config": {
                        "log_level": "info",
                        "enable_gpu": True,
                        "max_conversation_history": 50
                    }
                }
            }
        }
    }


class ValidationResponse(BaseModel):
    """
    Response model for configuration validation.
    
    Attributes:
        valid: Whether configuration is valid
        errors: List of validation errors (if any)
        warnings: List of validation warnings (if any)
    """
    
    valid: bool = Field(..., description="Configuration validity")
    errors: list = Field(..., description="List of validation errors")
    warnings: list = Field(..., description="List of validation warnings")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "valid": True,
                "errors": [],
                "warnings": ["API key not configured"]
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


@router.get(
    "/",
    response_model=ConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current configuration",
    description="Retrieve the current application configuration including API, database, and application settings.",
    responses={
        200: {"description": "Successfully retrieved configuration"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_configuration() -> ConfigResponse:
    """
    Get the current application configuration.
    
    This endpoint returns the complete application configuration including
    API settings, database configuration, and application preferences.
    
    Returns:
        ConfigResponse with all configuration settings
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Retrieving application configuration")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Get configuration components
        api_config = config_manager.get_api_config()
        db_config = config_manager.get_db_config()
        app_config = config_manager.get_app_config()
        
        # Get last modification time
        last_modified = config_manager.get_last_modified()
        
        logger.debug("Successfully retrieved configuration")
        
        return ConfigResponse(
            api_config=api_config,
            db_config=db_config,
            app_config=app_config,
            last_modified=last_modified
        )
        
    except ConfigManagerError as e:
        logger.error(f"Configuration manager error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error retrieving configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


@router.put(
    "/api",
    response_model=UpdateConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Update API configuration",
    description="Update DeepSeek API configuration settings.",
    responses={
        200: {"description": "Successfully updated API configuration"},
        400: {"model": ErrorResponse, "description": "Invalid configuration values"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_api_configuration(request: UpdateAPIConfigRequest) -> UpdateConfigResponse:
    """
    Update DeepSeek API configuration settings.
    
    This endpoint allows updating individual API configuration settings
    without affecting other configuration sections.
    
    Args:
        request: API configuration update request
        
    Returns:
        UpdateConfigResponse with update results
        
    Raises:
        HTTPException: If update fails or service is unavailable
    """
    try:
        logger.info("Updating API configuration")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Prepare update data
        update_data = {}
        if request.api_key is not None:
            update_data["api_key"] = request.api_key
        if request.base_url is not None:
            update_data["base_url"] = request.base_url
        if request.model is not None:
            update_data["model"] = request.model
        if request.temperature is not None:
            update_data["temperature"] = request.temperature
        if request.max_tokens is not None:
            update_data["max_tokens"] = request.max_tokens
        
        # Update API configuration
        success = config_manager.update_api_config(update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update API configuration"
            )
        
        logger.info("Successfully updated API configuration")
        
        return UpdateConfigResponse(
            message="API configuration updated successfully",
            updated_sections=["api_config"],
            restart_required=False  # API config changes typically don't require restart
        )
        
    except ConfigManagerError as e:
        logger.error(f"Configuration manager error in API update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid API configuration: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_api_configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.put(
    "/database",
    response_model=UpdateConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Update database configuration",
    description="Update database configuration settings for the vector store.",
    responses={
        200: {"description": "Successfully updated database configuration"},
        400: {"model": ErrorResponse, "description": "Invalid configuration values"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_database_configuration(request: UpdateDBConfigRequest) -> UpdateConfigResponse:
    """
    Update database configuration settings.
    
    This endpoint allows updating database configuration settings
    including persist directory and collection name.
    
    Args:
        request: Database configuration update request
        
    Returns:
        UpdateConfigResponse with update results
        
    Raises:
        HTTPException: If update fails or service is unavailable
    """
    try:
        logger.info("Updating database configuration")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Prepare update data
        update_data = {}
        if request.persist_directory is not None:
            update_data["persist_directory"] = request.persist_directory
        if request.collection_name is not None:
            update_data["collection_name"] = request.collection_name
        
        # Update database configuration
        success = config_manager.update_db_config(update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update database configuration"
            )
        
        logger.info("Successfully updated database configuration")
        
        return UpdateConfigResponse(
            message="Database configuration updated successfully",
            updated_sections=["db_config"],
            restart_required=True  # Database config changes may require restart
        )
        
    except ConfigManagerError as e:
        logger.error(f"Configuration manager error in DB update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid database configuration: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_database_configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.put(
    "/application",
    response_model=UpdateConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Update application configuration",
    description="Update application configuration settings including logging and behavior.",
    responses={
        200: {"description": "Successfully updated application configuration"},
        400: {"model": ErrorResponse, "description": "Invalid configuration values"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_application_configuration(request: UpdateAppConfigRequest) -> UpdateConfigResponse:
    """
    Update application configuration settings.
    
    This endpoint allows updating application-level settings including
    logging level, GPU acceleration, and conversation history limits.
    
    Args:
        request: Application configuration update request
        
    Returns:
        UpdateConfigResponse with update results
        
    Raises:
        HTTPException: If update fails or service is unavailable
    """
    try:
        logger.info("Updating application configuration")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Prepare update data
        update_data = {}
        if request.log_level is not None:
            update_data["log_level"] = request.log_level
        if request.enable_gpu is not None:
            update_data["enable_gpu"] = request.enable_gpu
        if request.max_conversation_history is not None:
            update_data["max_conversation_history"] = request.max_conversation_history
        
        # Update application configuration
        success = config_manager.update_app_config(update_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update application configuration"
            )
        
        logger.info("Successfully updated application configuration")
        
        return UpdateConfigResponse(
            message="Application configuration updated successfully",
            updated_sections=["app_config"],
            restart_required=True  # App config changes may require restart
        )
        
    except ConfigManagerError as e:
        logger.error(f"Configuration manager error in app update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid application configuration: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_application_configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/reset",
    response_model=ResetConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset configuration to defaults",
    description="Reset all configuration settings to their default values.",
    responses={
        200: {"description": "Successfully reset configuration to defaults"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def reset_configuration() -> ResetConfigResponse:
    """
    Reset all configuration settings to default values.
    
    This endpoint resets the entire configuration to factory defaults,
    including API, database, and application settings.
    
    Returns:
        ResetConfigResponse with reset results and default values
        
    Raises:
        HTTPException: If reset fails or service is unavailable
    """
    try:
        logger.info("Resetting configuration to defaults")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Reset configuration to defaults
        success = config_manager.reset_to_defaults()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset configuration"
            )
        
        # Get default values for response
        default_api_config = APIConfig(api_key="")  # Default without API key
        default_db_config = DatabaseConfig()
        default_app_config = AppConfig(api_config=default_api_config)
        
        logger.info("Successfully reset configuration to defaults")
        
        return ResetConfigResponse(
            message="Configuration reset to defaults successfully",
            reset_sections=["api_config", "db_config", "app_config"],
            default_values={
                "api_config": {
                    "base_url": default_api_config.base_url,
                    "model": default_api_config.model,
                    "temperature": default_api_config.temperature,
                    "max_tokens": default_api_config.max_tokens
                },
                "db_config": {
                    "persist_directory": default_db_config.persist_directory,
                    "collection_name": default_db_config.collection_name
                },
                "app_config": {
                    "log_level": default_app_config.log_level,
                    "enable_gpu": default_app_config.enable_gpu,
                    "max_conversation_history": default_app_config.max_conversation_history
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset configuration: {str(e)}"
        )


@router.get(
    "/validate",
    response_model=ValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate configuration",
    description="Validate the current configuration for errors and warnings.",
    responses={
        200: {"description": "Successfully validated configuration"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def validate_configuration() -> ValidationResponse:
    """
    Validate the current configuration.
    
    This endpoint performs validation on the current configuration
    and returns any errors or warnings that need attention.
    
    Returns:
        ValidationResponse with validation results
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Validating configuration")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Validate configuration
        validation_result = config_manager.validate_configuration()
        
        return ValidationResponse(
            valid=validation_result.get("valid", False),
            errors=validation_result.get("errors", []),
            warnings=validation_result.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}"
        )


@router.get(
    "/backup",
    summary="Backup configuration",
    description="Create a backup of the current configuration.",
    responses={
        200: {"description": "Successfully created configuration backup"},
        503: {"model": ErrorResponse, "description": "Configuration service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def backup_configuration():
    """
    Create a backup of the current configuration.
    
    This endpoint creates a backup of the current configuration files
    that can be restored later if needed.
    
    Returns:
        Dictionary with backup information
        
    Raises:
        HTTPException: If backup fails or service is unavailable
    """
    try:
        logger.info("Creating configuration backup")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Create backup
        backup_info = config_manager.create_backup()
        
        return {
            "message": "Configuration backup created successfully",
            "backup_file": backup_info.get("backup_file"),
            "backup_time": backup_info.get("backup_time"),
            "config_files": backup_info.get("config_files", [])
        }
        
    except Exception as e:
        logger.error(f"Error creating configuration backup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create configuration backup: {str(e)}"
        )


@router.get(
    "/environment",
    summary="Get environment information",
    description="Retrieve information about the runtime environment and system.",
    responses={
        200: {"description": "Successfully retrieved environment information"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_environment_info():
    """
    Get runtime environment and system information.
    
    This endpoint provides information about the runtime environment
    including Python version, system information, and available resources.
    
    Returns:
        Dictionary with environment information
    """
    try:
        import platform
        import sys
        import os
        
        logger.debug("Retrieving environment information")
        
        # Get system information
        system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "working_directory": os.getcwd(),
            "user": os.getenv("USER", os.getenv("USERNAME", "unknown"))
        }
        
        # Get memory information (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["memory_total_gb"] = round(memory.total / (1024**3), 2)
            system_info["memory_available_gb"] = round(memory.available / (1024**3), 2)
            system_info["memory_used_percent"] = memory.percent
        except ImportError:
            system_info["memory_info"] = "psutil not available"
        
        # Get GPU information (if available)
        try:
            import torch
            if torch.cuda.is_available():
                system_info["gpu_available"] = True
                system_info["gpu_count"] = torch.cuda.device_count()
                system_info["gpu_devices"] = [
                    {
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                    }
                    for i in range(torch.cuda.device_count())
                ]
            else:
                system_info["gpu_available"] = False
        except ImportError:
            system_info["gpu_info"] = "torch not available"
        
        return {
            "environment": system_info,
            "timestamp": "2024-01-01T12:00:00Z"  # This would be dynamic in production
        }
        
    except Exception as e:
        logger.error(f"Error retrieving environment info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve environment information: {str(e)}"
        )


# Health check endpoint for configuration service
@router.get(
    "/health",
    summary="Configuration service health check",
    description="Check the health status of the configuration service.",
    responses={
        200: {"description": "Configuration service is healthy"},
        503: {"model": ErrorResponse, "description": "Configuration service is unhealthy"}
    }
)
async def config_health_check():
    """
    Health check for the configuration service.
    
    This endpoint performs a health check on the configuration manager
    to ensure it is functioning properly.
    
    Returns:
        Dictionary with health status
        
    Raises:
        HTTPException: If configuration service is unhealthy
    """
    try:
        logger.debug("Performing configuration service health check")
        
        # Get configuration manager instance
        config_manager = get_config_manager()
        
        # Perform health check
        is_healthy = config_manager.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configuration service health check failed"
            )
        
        return {
            "status": "healthy",
            "service": "config_manager",
            "timestamp": "2024-01-01T12:00:00Z",  # This would be dynamic in production
            "config_files_loaded": config_manager.get_loaded_config_files()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration service health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration service health check failed: {str(e)}"
        )
