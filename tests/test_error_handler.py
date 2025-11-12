"""
Unit tests for error handler utilities.

This module tests the centralized error handling functionality including
custom exception classes, error formatting, and error handling decorators.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.utils.error_handler import (
    ApplicationError,
    ConfigurationError,
    APIConnectionError,
    KnowledgeBaseError,
    ValidationError,
    ErrorHandler,
    ErrorSeverity,
    error_handler
)


class TestApplicationError:
    """Test cases for ApplicationError base class."""
    
    def test_application_error_creation(self):
        """Test basic ApplicationError creation."""
        error = ApplicationError("Test error message")
        
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_code is None
        assert error.user_message == "An unexpected error occurred. Please try again."
        assert error.original_exception is None
        assert error.context == {}
    
    def test_application_error_with_all_parameters(self):
        """Test ApplicationError creation with all parameters."""
        original_exception = ValueError("Original error")
        context = {"key": "value"}
        
        error = ApplicationError(
            message="Test error",
            severity=ErrorSeverity.HIGH,
            error_code="TEST_ERROR",
            user_message="User friendly message",
            original_exception=original_exception,
            context=context
        )
        
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "TEST_ERROR"
        assert error.user_message == "User friendly message"
        assert error.original_exception == original_exception
        assert error.context == context
    
    def test_application_error_to_dict(self):
        """Test converting ApplicationError to dictionary."""
        error = ApplicationError(
            message="Test error",
            severity=ErrorSeverity.CRITICAL,
            error_code="TEST_ERROR"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error"] == "ApplicationError"
        assert error_dict["message"] == "Test error"
        assert error_dict["severity"] == "critical"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert "timestamp" in error_dict


class TestSpecificErrorClasses:
    """Test cases for specific error subclasses."""
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            message="Config error",
            config_key="api_key",
            config_value="invalid"
        )
        
        assert error.message == "Config error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "CONFIG_ERROR"
        assert error.context["config_key"] == "api_key"
        assert error.context["config_value"] == "invalid"
    
    def test_api_connection_error(self):
        """Test APIConnectionError creation."""
        error = APIConnectionError(
            message="Connection failed",
            endpoint="/api/test",
            status_code=500
        )
        
        assert error.message == "Connection failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "API_CONNECTION_ERROR"
        assert error.context["endpoint"] == "/api/test"
        assert error.context["status_code"] == 500
    
    def test_knowledge_base_error(self):
        """Test KnowledgeBaseError creation."""
        error = KnowledgeBaseError(
            message="KB operation failed",
            operation="search",
            file_path="/path/to/file"
        )
        
        assert error.message == "KB operation failed"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_code == "KB_ERROR"
        assert error.context["operation"] == "search"
        assert error.context["file_path"] == "/path/to/file"
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError(
            message="Invalid input",
            field="username",
            value=""
        )
        
        assert error.message == "Invalid input"
        assert error.severity == ErrorSeverity.LOW
        assert error.error_code == "VALIDATION_ERROR"
        assert error.context["field"] == "username"
        assert error.context["value"] == ""


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def test_handle_application_error(self):
        """Test handling an ApplicationError."""
        original_error = ConfigurationError("Config error")
        
        with patch('src.utils.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_exception(original_error)
            
            # Should return the same error
            assert result == original_error
            # Should log the error (ConfigurationError has HIGH severity -> error level)
            mock_logger.error.assert_called_once()
    
    def test_handle_connection_error(self):
        """Test handling a ConnectionError."""
        connection_error = ConnectionError("Connection refused")
        
        with patch('src.utils.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_exception(connection_error)
            
            # Should convert to APIConnectionError
            assert isinstance(result, APIConnectionError)
            assert result.severity == ErrorSeverity.HIGH
            assert "Connection refused" in result.message
            # Should log the error
            mock_logger.error.assert_called_once()
    
    def test_handle_value_error(self):
        """Test handling a ValueError."""
        value_error = ValueError("Invalid value")
        
        with patch('src.utils.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_exception(value_error)
            
            # Should convert to ValidationError
            assert isinstance(result, ValidationError)
            assert result.severity == ErrorSeverity.LOW
            assert "Invalid value" in result.message
            # Should log the error
            mock_logger.info.assert_called_once()
    
    def test_handle_unknown_error(self):
        """Test handling an unknown exception type."""
        custom_error = Exception("Custom error")
        
        with patch('src.utils.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_exception(custom_error)
            
            # Should convert to generic ApplicationError
            assert isinstance(result, ApplicationError)
            assert result.severity == ErrorSeverity.HIGH
            assert result.error_code == "UNKNOWN_ERROR"
            # Should log the error
            mock_logger.error.assert_called_once()
    
    def test_handle_exception_with_context(self):
        """Test handling exception with additional context."""
        value_error = ValueError("Invalid value")
        context = {"user_id": 123, "operation": "test"}
        
        result = ErrorHandler.handle_exception(value_error, context=context)
        
        # Context should be preserved
        assert result.context["user_id"] == 123
        assert result.context["operation"] == "test"
        assert "exception_type" in result.context
        assert "traceback" in result.context
    
    def test_get_user_friendly_message(self):
        """Test getting user-friendly error message."""
        error = ConfigurationError("Config error")
        user_message = ErrorHandler.get_user_friendly_message(error)
        
        assert user_message == "Configuration error. Please check your settings."
    
    def test_format_error_for_api(self):
        """Test formatting error for API response."""
        error = ValidationError("Invalid input")
        
        # Test without details
        response = ErrorHandler.format_error_for_api(error)
        assert response["error"] == "VALIDATION_ERROR"
        assert "Invalid input provided" in response["message"]
        assert "details" not in response
        
        # Test with details
        response_with_details = ErrorHandler.format_error_for_api(error, include_details=True)
        assert response_with_details["error"] == "VALIDATION_ERROR"
        assert "details" in response_with_details
        assert response_with_details["details"]["technical_message"] == "Invalid input"
        assert response_with_details["details"]["severity"] == "low"


class TestErrorHandlerDecorator:
    """Test cases for error_handler decorator."""
    
    def test_error_handler_decorator_sync_success(self):
        """Test error_handler decorator with successful sync function."""
        
        @error_handler()
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_error_handler_decorator_sync_error_reraises(self):
        """Test error_handler decorator with sync function that raises error."""
        
        @error_handler(reraise=True)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValidationError):  # Should be converted to ValidationError
            failing_function()
    
    def test_error_handler_decorator_sync_error_no_reraises(self):
        """Test error_handler decorator with sync function that returns default."""
        
        @error_handler(reraise=False, return_value="default")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "default"
    
    @pytest.mark.asyncio
    async def test_error_handler_decorator_async_success(self):
        """Test error_handler decorator with successful async function."""
        
        @error_handler()
        async def successful_async_function():
            await asyncio.sleep(0.01)
            return "async success"
        
        result = await successful_async_function()
        assert result == "async success"
    
    @pytest.mark.asyncio
    async def test_error_handler_decorator_async_error_reraises(self):
        """Test error_handler decorator with async function that raises error."""
        
        @error_handler(reraise=True)
        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise ValueError("Async test error")
        
        with pytest.raises(ValidationError):  # Should be converted to ValidationError
            await failing_async_function()
    
    def test_error_handler_with_custom_log_level(self):
        """Test error_handler decorator with custom log level."""
        
        @error_handler(log_level="WARNING")
        def function_with_warning():
            raise ValueError("Warning level error")
        
        with patch('src.utils.error_handler.logger') as mock_logger:
            with pytest.raises(ValidationError):
                function_with_warning()
            
            # Should use warning level logging
            mock_logger.warning.assert_called_once()


class TestErrorSeverity:
    """Test cases for ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_severity_comparison(self):
        """Test ErrorSeverity comparison."""
        assert ErrorSeverity.LOW != ErrorSeverity.HIGH
        assert ErrorSeverity.MEDIUM == ErrorSeverity.MEDIUM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
