"""
Unit tests for structured logging utilities.

This module tests the logging configuration, structured logging functions,
and loguru integration for the AI Customer Agent application.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.utils.logger import (
    LogConfig,
    setup_global_logging,
    get_logger,
    get_log_stats,
    update_log_level,
    log_api_request,
    log_chat_interaction,
    log_knowledge_base_operation,
    log_system_health,
    log_startup,
    log_shutdown,
    log_configuration_change,
    log_text_to_sql_operation,
    log_sql_execution,
    log_excel_table_operation,
    log_query_intent_detection
)


class TestLogConfig:
    """Test cases for LogConfig class."""
    
    def test_log_config_initialization(self):
        """Test LogConfig initialization with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir, environment="development")
            
            assert config.log_dir == Path(temp_dir)
            assert config.environment == "development"
            assert config.log_level == "DEBUG"
    
    def test_log_config_environment_levels(self):
        """Test LogConfig environment-specific log levels."""
        # Test development environment
        dev_config = LogConfig(environment="development")
        assert dev_config.log_level == "DEBUG"
        
        # Test testing environment
        test_config = LogConfig(environment="testing")
        assert test_config.log_level == "INFO"
        
        # Test production environment
        prod_config = LogConfig(environment="production")
        assert prod_config.log_level == "WARNING"
        
        # Test unknown environment (should default to INFO)
        unknown_config = LogConfig(environment="unknown")
        assert unknown_config.log_level == "INFO"
    
    def test_log_config_creates_directory(self):
        """Test that LogConfig creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "new_logs"
            
            # Directory shouldn't exist initially
            assert not log_dir.exists()
            
            # Create config - should create directory
            config = LogConfig(log_dir=str(log_dir))
            assert log_dir.exists()
    
    @patch('src.utils.logger.logger')
    def test_setup_logging(self, mock_logger):
        """Test logging setup with all components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir, environment="development")
            
            # Mock the internal setup methods to avoid actual file operations
            with patch.object(config, '_setup_console_logging') as mock_console, \
                 patch.object(config, '_setup_file_logging') as mock_file, \
                 patch.object(config, '_setup_error_logging') as mock_error:
                
                config.setup_logging(
                    enable_file_logging=True,
                    enable_console_logging=True
                )
                
                # All setup methods should be called
                mock_console.assert_called_once()
                mock_file.assert_called_once()
                mock_error.assert_called_once()
                
                # Logger should be called with success message
                mock_logger.info.assert_called_once()
    
    @patch('src.utils.logger.logger')
    def test_setup_logging_console_only(self, mock_logger):
        """Test logging setup with console logging only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir)
            
            with patch.object(config, '_setup_console_logging') as mock_console, \
                 patch.object(config, '_setup_file_logging') as mock_file, \
                 patch.object(config, '_setup_error_logging') as mock_error:
                
                config.setup_logging(
                    enable_file_logging=False,
                    enable_console_logging=True
                )
                
                # Only console logging should be set up
                mock_console.assert_called_once()
                mock_file.assert_not_called()
                mock_error.assert_not_called()
    
    @patch('src.utils.logger.logger')
    def test_setup_logging_file_only(self, mock_logger):
        """Test logging setup with file logging only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir)
            
            with patch.object(config, '_setup_console_logging') as mock_console, \
                 patch.object(config, '_setup_file_logging') as mock_file, \
                 patch.object(config, '_setup_error_logging') as mock_error:
                
                config.setup_logging(
                    enable_file_logging=True,
                    enable_console_logging=False
                )
                
                # Only file and error logging should be set up
                mock_console.assert_not_called()
                mock_file.assert_called_once()
                mock_error.assert_called_once()
    
    def test_get_logger(self):
        """Test getting logger instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir)
            
            # Should return the loguru logger
            logger_instance = config.get_logger()
            assert logger_instance is not None
    
    def test_update_log_level_valid(self):
        """Test updating log level with valid level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir, environment="development")
            original_level = config.log_level
            
            # Update to valid level
            config.update_log_level("WARNING")
            
            assert config.log_level == "WARNING"
            assert original_level != config.log_level
    
    def test_update_log_level_invalid(self):
        """Test updating log level with invalid level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir)
            
            with pytest.raises(ValueError, match="Invalid log level"):
                config.update_log_level("INVALID_LEVEL")
    
    def test_get_log_stats(self):
        """Test getting logging statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LogConfig(log_dir=temp_dir, environment="testing")
            
            stats = config.get_log_stats()
            
            assert stats["environment"] == "testing"
            assert stats["log_level"] == "INFO"
            assert stats["log_directory"] == temp_dir
            assert isinstance(stats["log_files"], list)
            assert isinstance(stats["total_log_files"], int)
            assert isinstance(stats["log_file_sizes"], dict)


class TestGlobalLoggingFunctions:
    """Test cases for global logging functions."""
    
    @patch('src.utils.logger.LogConfig')
    def test_setup_global_logging(self, mock_log_config_class):
        """Test setting up global logging."""
        mock_log_config = MagicMock()
        mock_log_config_class.return_value = mock_log_config
        
        # Test with default parameters
        logger_instance = setup_global_logging()
        
        mock_log_config_class.assert_called_once_with(
            log_dir="./logs",
            environment="development"
        )
        mock_log_config.setup_logging.assert_called_once_with(
            enable_file_logging=True,
            enable_console_logging=True
        )
        assert logger_instance == mock_log_config.get_logger.return_value
    
    @patch('src.utils.logger.LogConfig')
    def test_setup_global_logging_with_environment_variable(self, mock_log_config_class):
        """Test setting up global logging with environment variable."""
        mock_log_config = MagicMock()
        mock_log_config_class.return_value = mock_log_config
        
        with patch.dict(os.environ, {"APP_ENVIRONMENT": "production"}):
            setup_global_logging()
            
            mock_log_config_class.assert_called_once_with(
                log_dir="./logs",
                environment="production"
            )
    
    @patch('src.utils.logger._log_config')
    def test_get_logger_not_configured(self, mock_global_config):
        """Test getting logger when not configured."""
        mock_global_config.__bool__.return_value = False
        
        with patch('src.utils.logger.setup_global_logging') as mock_setup:
            get_logger()
            
            mock_setup.assert_called_once()
    
    @patch('src.utils.logger._log_config')
    def test_get_logger_configured(self, mock_global_config):
        """Test getting logger when already configured."""
        mock_config_instance = MagicMock()
        mock_global_config.__bool__.return_value = True
        mock_global_config.get_logger.return_value = "mock_logger"
        
        logger_instance = get_logger()
        
        assert logger_instance == "mock_logger"
        mock_global_config.get_logger.assert_called_once()
    
    @patch('src.utils.logger._log_config')
    def test_get_log_stats_not_configured(self, mock_global_config):
        """Test getting log stats when not configured."""
        mock_global_config.__bool__.return_value = False
        
        stats = get_log_stats()
        
        assert stats == {"error": "Logging not configured"}
    
    @patch('src.utils.logger._log_config')
    def test_get_log_stats_configured(self, mock_global_config):
        """Test getting log stats when configured."""
        mock_config_instance = MagicMock()
        mock_config_instance.get_log_stats.return_value = {"test": "stats"}
        mock_global_config.__bool__.return_value = True
        mock_global_config.get_log_stats.return_value = {"test": "stats"}
        
        stats = get_log_stats()
        
        assert stats == {"test": "stats"}
        mock_global_config.get_log_stats.assert_called_once()
    
    @patch('src.utils.logger._log_config')
    def test_update_log_level_not_configured(self, mock_global_config):
        """Test updating log level when not configured."""
        mock_global_config.__bool__.return_value = False
        
        with pytest.raises(RuntimeError, match="Logging not configured"):
            update_log_level("DEBUG")
    
    @patch('src.utils.logger._log_config')
    def test_update_log_level_configured(self, mock_global_config):
        """Test updating log level when configured."""
        mock_config_instance = MagicMock()
        mock_global_config.__bool__.return_value = True
        
        update_log_level("WARNING")
        
        mock_global_config.update_log_level.assert_called_once_with("WARNING")


class TestContextAwareLogging:
    """Test cases for context-aware logging functions."""
    
    @patch('src.utils.logger.get_logger')
    def test_log_api_request_success(self, mock_get_logger):
        """Test logging successful API request."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_api_request(
            method="POST",
            endpoint="/api/chat",
            status_code=200,
            response_time=0.5,
            user_agent="test-agent",
            user_id="user123"
        )
        
        # Should log at INFO level for success
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "API POST /api/chat completed" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "api_request"
        assert call_args[1]["extra"]["method"] == "POST"
        assert call_args[1]["extra"]["status_code"] == 200
    
    @patch('src.utils.logger.get_logger')
    def test_log_api_request_error(self, mock_get_logger):
        """Test logging failed API request."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_api_request(
            method="GET",
            endpoint="/api/health",
            status_code=500,
            response_time=0.1
        )
        
        # Should log at WARNING level for error
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "API GET /api/health returned 500" in call_args[0][0]
        assert call_args[1]["extra"]["status_code"] == 500
    
    @patch('src.utils.logger.get_logger')
    def test_log_chat_interaction_success(self, mock_get_logger):
        """Test logging successful chat interaction."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_chat_interaction(
            user_message="Hello, how are you?",
            response_time=1.2,
            message_id="msg_123",
            used_knowledge_base=True
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Chat response generated for message msg_123" in call_args[0][0]
        assert call_args[1]["extra"]["message_id"] == "msg_123"
        assert call_args[1]["extra"]["used_knowledge_base"] is True
        assert call_args[1]["extra"]["user_message_length"] == 18
    
    @patch('src.utils.logger.get_logger')
    def test_log_chat_interaction_error(self, mock_get_logger):
        """Test logging failed chat interaction."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_chat_interaction(
            user_message="Test message",
            response_time=0.5,
            message_id="msg_456",
            error="API timeout"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Chat error for message msg_456: API timeout" in call_args[0][0]
        assert call_args[1]["extra"]["error"] == "API timeout"
    
    @patch('src.utils.logger.get_logger')
    def test_log_knowledge_base_operation_success(self, mock_get_logger):
        """Test logging successful knowledge base operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_knowledge_base_operation(
            operation="add",
            file_path="/docs/manual.pdf",
            document_count=5
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Knowledge base add completed" in call_args[0][0]
        assert call_args[1]["extra"]["operation"] == "add"
        assert call_args[1]["extra"]["file_path"] == "/docs/manual.pdf"
        assert call_args[1]["extra"]["document_count"] == 5
    
    @patch('src.utils.logger.get_logger')
    def test_log_knowledge_base_operation_error(self, mock_get_logger):
        """Test logging failed knowledge base operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_knowledge_base_operation(
            operation="search",
            error="Vector database unavailable"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Knowledge base search failed: Vector database unavailable" in call_args[0][0]
        assert call_args[1]["extra"]["error"] == "Vector database unavailable"
    
    @patch('src.utils.logger.get_logger')
    def test_log_system_health_healthy(self, mock_get_logger):
        """Test logging healthy system status."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_system_health(
            component="database",
            status="healthy",
            details={"connections": 10}
        )
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "System component database is healthy" in call_args[0][0]
        assert call_args[1]["extra"]["status"] == "healthy"
    
    @patch('src.utils.logger.get_logger')
    def test_log_system_health_degraded(self, mock_get_logger):
        """Test logging degraded system status."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_system_health(
            component="api",
            status="degraded"
        )
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "System component api is degraded" in call_args[0][0]
    
    @patch('src.utils.logger.get_logger')
    def test_log_system_health_unhealthy(self, mock_get_logger):
        """Test logging unhealthy system status."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_system_health(
            component="cache",
            status="unhealthy",
            details={"error": "Connection lost"}
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "System component cache is unhealthy" in call_args[0][0]
        assert call_args[1]["extra"]["details"]["error"] == "Connection lost"


class TestConvenienceLoggingFunctions:
    """Test cases for convenience logging functions."""
    
    @patch('src.utils.logger.get_logger')
    def test_log_startup(self, mock_get_logger):
        """Test logging application startup."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_startup()
        
        mock_logger.info.assert_called_once_with(
            "Application starting up",
            type="startup"
        )
    
    @patch('src.utils.logger.get_logger')
    def test_log_shutdown(self, mock_get_logger):
        """Test logging application shutdown."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_shutdown()
        
        mock_logger.info.assert_called_once_with(
            "Application shutting down",
            type="shutdown"
        )
    
    @patch('src.utils.logger.get_logger')
    def test_log_configuration_change(self, mock_get_logger):
        """Test logging configuration change."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_configuration_change(
            setting="api_timeout",
            old_value=30,
            new_value=60,
            user="admin"
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Configuration changed: api_timeout" in call_args[0][0]
        assert call_args[1]["extra"]["setting"] == "api_timeout"
        assert call_args[1]["extra"]["old_value"] == "30"
        assert call_args[1]["extra"]["new_value"] == "60"
        assert call_args[1]["extra"]["user"] == "admin"


class TestTextToSQLLoggingFunctions:
    """Test cases for Text-to-SQL specific logging functions."""
    
    @patch('src.utils.logger.get_logger')
    def test_log_text_to_sql_operation_success(self, mock_get_logger):
        """Test logging successful Text-to-SQL operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_text_to_sql_operation(
            natural_language_query="Show me sales data for Q1",
            generated_sql="SELECT * FROM sales WHERE quarter = 'Q1'",
            execution_time=0.75,
            result_count=150,
            table_name="sales",
            query_complexity="moderate"
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Text-to-SQL completed:" in call_args[0][0]
        assert "150 results" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "text_to_sql"
        assert call_args[1]["extra"]["natural_language_query_length"] == 28
        assert call_args[1]["extra"]["generated_sql_length"] == 47
        assert call_args[1]["extra"]["execution_time"] == 0.75
        assert call_args[1]["extra"]["result_count"] == 150
        assert call_args[1]["extra"]["table_name"] == "sales"
        assert call_args[1]["extra"]["query_complexity"] == "moderate"
    
    @patch('src.utils.logger.get_logger')
    def test_log_text_to_sql_operation_error(self, mock_get_logger):
        """Test logging failed Text-to-SQL operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_text_to_sql_operation(
            natural_language_query="Show me sales data for Q1",
            generated_sql="SELECT * FROM non_existent_table",
            execution_time=0.25,
            table_name="non_existent_table",
            error="Table not found",
            query_complexity="simple"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Text-to-SQL failed for query:" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "text_to_sql"
        assert call_args[1]["extra"]["error"] == "Table not found"
        assert call_args[1]["extra"]["table_name"] == "non_existent_table"
    
    @patch('src.utils.logger.get_logger')
    def test_log_sql_execution_success(self, mock_get_logger):
        """Test logging successful SQL execution."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_sql_execution(
            sql_query="SELECT * FROM customers WHERE status = 'active'",
            execution_time=0.15,
            row_count=250,
            table_name="customers",
            operation_type="SELECT"
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "SQL execution completed: SELECT on customers -> 250 rows" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "sql_execution"
        assert call_args[1]["extra"]["sql_query_length"] == 48
        assert call_args[1]["extra"]["execution_time"] == 0.15
        assert call_args[1]["extra"]["row_count"] == 250
        assert call_args[1]["extra"]["table_name"] == "customers"
        assert call_args[1]["extra"]["operation_type"] == "SELECT"
    
    @patch('src.utils.logger.get_logger')
    def test_log_sql_execution_error(self, mock_get_logger):
        """Test logging failed SQL execution."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_sql_execution(
            sql_query="SELECT * FROM non_existent_table",
            execution_time=0.05,
            table_name="non_existent_table",
            error="no such table: non_existent_table",
            operation_type="SELECT"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "SQL execution failed:" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "sql_execution"
        assert call_args[1]["extra"]["error"] == "no such table: non_existent_table"
        assert call_args[1]["extra"]["operation_type"] == "SELECT"
    
    @patch('src.utils.logger.get_logger')
    def test_log_excel_table_operation_success(self, mock_get_logger):
        """Test logging successful Excel table operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_excel_table_operation(
            operation="create",
            file_name="sales_data.xlsx",
            table_name="sales_q1",
            row_count=500,
            column_count=10
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Excel table operation completed: create on sales_q1 in sales_data.xlsx" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "excel_table_operation"
        assert call_args[1]["extra"]["operation"] == "create"
        assert call_args[1]["extra"]["file_name"] == "sales_data.xlsx"
        assert call_args[1]["extra"]["table_name"] == "sales_q1"
        assert call_args[1]["extra"]["row_count"] == 500
        assert call_args[1]["extra"]["column_count"] == 10
    
    @patch('src.utils.logger.get_logger')
    def test_log_excel_table_operation_error(self, mock_get_logger):
        """Test logging failed Excel table operation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_excel_table_operation(
            operation="search",
            file_name="data.xlsx",
            table_name="sheet1",
            error="Table not found"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Excel table operation failed: search on sheet1 in data.xlsx" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "excel_table_operation"
        assert call_args[1]["extra"]["error"] == "Table not found"
    
    @patch('src.utils.logger.get_logger')
    def test_log_query_intent_detection_success(self, mock_get_logger):
        """Test logging successful query intent detection."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_query_intent_detection(
            user_query="What were the sales figures for last quarter?",
            detected_intent="excel_data",
            confidence=0.85,
            routed_service="text_to_sql"
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Query routed to text_to_sql with 0.85 confidence:" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "query_intent_detection"
        assert call_args[1]["extra"]["user_query_length"] == 45
        assert call_args[1]["extra"]["detected_intent"] == "excel_data"
        assert call_args[1]["extra"]["confidence"] == 0.85
        assert call_args[1]["extra"]["routed_service"] == "text_to_sql"
    
    @patch('src.utils.logger.get_logger')
    def test_log_query_intent_detection_error(self, mock_get_logger):
        """Test logging failed query intent detection."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_query_intent_detection(
            user_query="What were the sales figures for last quarter?",
            detected_intent="unknown",
            confidence=0.3,
            routed_service="chat",
            error="Intent detection model failed"
        )
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Query intent detection failed for:" in call_args[0][0]
        assert call_args[1]["extra"]["type"] == "query_intent_detection"
        assert call_args[1]["extra"]["error"] == "Intent detection model failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
