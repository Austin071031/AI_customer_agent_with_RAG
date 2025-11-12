"""
Centralized Error Handler for AI Customer Agent.

This module provides comprehensive error handling utilities for the application,
including custom exception classes, error formatting, and user-friendly error messages.
"""

import traceback
import sys
from typing import Dict, Any, Optional, Type
from enum import Enum

from loguru import logger


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ApplicationError(Exception):
    """
    Base application error class with enhanced error handling capabilities.
    
    This class provides structured error information including severity,
    error codes, and user-friendly messages.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        user_message: Optional[str] = None,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the application error.
        
        Args:
            message: Technical error message for logging
            severity: Error severity level
            error_code: Unique error code for categorization
            user_message: User-friendly error message
            original_exception: Original exception that caused this error
            context: Additional context information about the error
        """
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.user_message = user_message or "An unexpected error occurred. Please try again."
        self.original_exception = original_exception
        self.context = context or {}
        self.timestamp = None
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization.
        
        Returns:
            Dictionary containing error information
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class ConfigurationError(ApplicationError):
    """Error raised for configuration-related issues."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs
    ):
        context = {"config_key": config_key, "config_value": config_value}
        if "context" in kwargs:
            context.update(kwargs.pop("context"))
            
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            error_code="CONFIG_ERROR",
            user_message="Configuration error. Please check your settings.",
            context=context,
            **kwargs
        )


class APIConnectionError(ApplicationError):
    """Error raised for API connection failures."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = {"endpoint": endpoint, "status_code": status_code}
        if "context" in kwargs:
            context.update(kwargs.pop("context"))
            
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            error_code="API_CONNECTION_ERROR",
            user_message="Unable to connect to AI service. Please check your internet connection and try again.",
            context=context,
            **kwargs
        )


class KnowledgeBaseError(ApplicationError):
    """Error raised for knowledge base operations."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        context = {"operation": operation, "file_path": file_path}
        if "context" in kwargs:
            context.update(kwargs.pop("context"))
            
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            error_code="KB_ERROR",
            user_message="Knowledge base operation failed. Please try again or contact support.",
            context=context,
            **kwargs
        )


class ValidationError(ApplicationError):
    """Error raised for data validation failures."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        context = {"field": field, "value": value}
        if "context" in kwargs:
            context.update(kwargs.pop("context"))
            
        super().__init__(
            message=message,
            severity=ErrorSeverity.LOW,
            error_code="VALIDATION_ERROR",
            user_message="Invalid input provided. Please check your data and try again.",
            context=context,
            **kwargs
        )


class ErrorHandler:
    """
    Centralized error handler for the application.
    
    This class provides utilities for handling exceptions consistently
    across the application, including logging, formatting, and recovery.
    """
    
    # Error code to user message mapping
    ERROR_MESSAGES = {
        "CONFIG_ERROR": "Configuration error. Please check your settings.",
        "API_CONNECTION_ERROR": "Unable to connect to AI service. Please check your connection.",
        "KB_ERROR": "Knowledge base operation failed. Please try again.",
        "VALIDATION_ERROR": "Invalid input provided. Please check your data.",
        "UNKNOWN_ERROR": "An unexpected error occurred. Please try again."
    }
    
    @classmethod
    def handle_exception(
        cls,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        logger_instance: Optional[logger] = None
    ) -> ApplicationError:
        """
        Handle an exception and convert it to an ApplicationError.
        
        Args:
            exception: The exception to handle
            context: Additional context information
            logger_instance: Optional logger instance (uses default if None)
            
        Returns:
            ApplicationError instance with proper formatting
        """
        log = logger_instance or logger
        
        # If it's already an ApplicationError, just log and return
        if isinstance(exception, ApplicationError):
            cls._log_application_error(exception, log)
            return exception
        
        # Convert standard exceptions to ApplicationError
        app_error = cls._convert_to_application_error(exception, context)
        cls._log_application_error(app_error, log)
        
        return app_error
    
    @classmethod
    def _convert_to_application_error(
        cls,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ApplicationError:
        """
        Convert a standard exception to an ApplicationError.
        
        Args:
            exception: Standard exception to convert
            context: Additional context information
            
        Returns:
            ApplicationError instance
        """
        error_context = context or {}
        error_context.update({
            "exception_type": exception.__class__.__name__,
            "traceback": traceback.format_exc()
        })
        
        # Map common exception types to specific ApplicationError subclasses
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return APIConnectionError(
                message=str(exception),
                original_exception=exception,
                context=error_context
            )
        elif isinstance(exception, (ValueError, TypeError)):
            return ValidationError(
                message=str(exception),
                original_exception=exception,
                context=error_context
            )
        elif isinstance(exception, (KeyError, AttributeError)):
            return ConfigurationError(
                message=str(exception),
                original_exception=exception,
                context=error_context
            )
        else:
            # Generic application error for unhandled exceptions
            return ApplicationError(
                message=str(exception),
                severity=ErrorSeverity.HIGH,
                error_code="UNKNOWN_ERROR",
                original_exception=exception,
                context=error_context
            )
    
    @classmethod
    def _log_application_error(cls, error: ApplicationError, log: logger) -> None:
        """
        Log an ApplicationError with appropriate severity.
        
        Args:
            error: ApplicationError to log
            log: Logger instance to use
        """
        log_context = {
            "error_code": error.error_code,
            "severity": error.severity.value,
            "context": error.context
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            log.critical(f"{error.__class__.__name__}: {error.message}", **log_context)
        elif error.severity == ErrorSeverity.HIGH:
            log.error(f"{error.__class__.__name__}: {error.message}", **log_context)
        elif error.severity == ErrorSeverity.MEDIUM:
            log.warning(f"{error.__class__.__name__}: {error.message}", **log_context)
        else:
            log.info(f"{error.__class__.__name__}: {error.message}", **log_context)
    
    @classmethod
    def get_user_friendly_message(cls, error: ApplicationError) -> str:
        """
        Get a user-friendly message for an error.
        
        Args:
            error: ApplicationError instance
            
        Returns:
            User-friendly error message
        """
        return error.user_message
    
    @classmethod
    def format_error_for_api(cls, error: ApplicationError, include_details: bool = False) -> Dict[str, Any]:
        """
        Format error for API response.
        
        Args:
            error: ApplicationError to format
            include_details: Whether to include technical details
            
        Returns:
            Dictionary suitable for API response
        """
        response = {
            "error": error.error_code or "UNKNOWN_ERROR",
            "message": cls.get_user_friendly_message(error)
        }
        
        if include_details:
            response["details"] = {
                "technical_message": error.message,
                "severity": error.severity.value,
                "context": error.context
            }
        
        return response


def error_handler(
    reraise: bool = True,
    return_value: Any = None,
    log_level: str = "ERROR"
):
    """
    Decorator for handling exceptions in functions.
    
    Args:
        reraise: Whether to re-raise the exception after handling
        return_value: Value to return if exception occurs and reraise is False
        log_level: Log level to use for exceptions
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the exception
                app_error = ErrorHandler.handle_exception(e)
                
                # Log with specified level
                log_method = getattr(logger, log_level.lower())
                log_method(f"Error in {func.__name__}: {app_error.message}")
                
                if reraise:
                    raise app_error
                else:
                    return return_value
        
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Handle the exception
                app_error = ErrorHandler.handle_exception(e)
                
                # Log with specified level
                log_method = getattr(logger, log_level.lower())
                log_method(f"Error in {func.__name__}: {app_error.message}")
                
                if reraise:
                    raise app_error
                else:
                    return return_value
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator
