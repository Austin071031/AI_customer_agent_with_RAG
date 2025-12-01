"""
Structured Logging Configuration for AI Customer Agent.

This module provides comprehensive logging configuration using loguru
with structured logging, log rotation, and environment-specific settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger


class LogConfig:
    """
    Configuration class for structured logging.
    
    This class handles loguru logger configuration with rotation,
    retention, and structured formatting for different environments.
    """
    
    # Log levels for different environments
    LOG_LEVELS = {
        "development": "DEBUG",
        "testing": "INFO", 
        "production": "WARNING"
    }
    
    # Log format for structured logging
    LOG_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> | "
        "{extra}"
    )
    
    def __init__(self, log_dir: str = "./logs", environment: str = "development"):
        """
        Initialize logging configuration.
        
        Args:
            log_dir: Directory to store log files
            environment: Current environment (development, testing, production)
        """
        self.log_dir = Path(log_dir)
        self.environment = environment
        self.log_level = self.LOG_LEVELS.get(environment, "INFO")
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
    def setup_logging(self, enable_file_logging: bool = True, enable_console_logging: bool = True) -> None:
        """
        Set up logging configuration.
        
        Args:
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
        """
        # Configure console logging
        if enable_console_logging:
            self._setup_console_logging()
        
        # Configure file logging and error logging only if file logging is enabled
        if enable_file_logging:
            self._setup_file_logging()
            self._setup_error_logging()
        
        logger.info(
            "Logging configured successfully",
            environment=self.environment,
            log_level=self.log_level,
            log_dir=str(self.log_dir)
        )
    
    def _setup_console_logging(self) -> None:
        """Set up console logging with colored output."""
        logger.add(
            sys.stderr,
            level=self.log_level,
            format=self.LOG_FORMAT,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    def _setup_file_logging(self) -> None:
        """Set up file logging with rotation and retention."""
        log_file = self.log_dir / "app.log"
        
        logger.add(
            str(log_file),
            level=self.log_level,
            format=self.LOG_FORMAT,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe logging
        )
    
    def _setup_error_logging(self) -> None:
        """Set up separate error logging."""
        error_log_file = self.log_dir / "errors.log"
        
        logger.add(
            str(error_log_file),
            level="ERROR",
            format=self.LOG_FORMAT,
            rotation="5 MB",
            retention="90 days",  # Keep error logs longer
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
    
    def get_logger(self) -> logger:
        """
        Get the configured logger instance.
        
        Returns:
            Configured loguru logger instance
        """
        return logger
    
    def update_log_level(self, new_level: str) -> None:
        """
        Update the log level dynamically.
        
        Args:
            new_level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if new_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {new_level}. Must be one of {valid_levels}")
        
        self.log_level = new_level.upper()
        logger.info(f"Log level updated to {self.log_level}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics and configuration.
        
        Returns:
            Dictionary with logging statistics
        """
        return {
            "environment": self.environment,
            "log_level": self.log_level,
            "log_directory": str(self.log_dir),
            "log_files": [f.name for f in self.log_dir.glob("*.log")],
            "total_log_files": len(list(self.log_dir.glob("*.log"))),
            "log_file_sizes": {
                f.name: f.stat().st_size 
                for f in self.log_dir.glob("*.log")
            }
        }


# Global logger configuration instance
_log_config: Optional[LogConfig] = None


def setup_global_logging(
    log_dir: str = "./logs",
    environment: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> logger:
    """
    Set up global logging configuration.
    
    This function should be called once at application startup
    to configure logging for the entire application.
    
    Args:
        log_dir: Directory to store log files
        environment: Current environment (development, testing, production)
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    global _log_config
    
    # Determine environment if not specified
    if environment is None:
        environment = os.getenv("APP_ENVIRONMENT", "development")
    
    # Create and configure logging
    _log_config = LogConfig(log_dir=log_dir, environment=environment)
    _log_config.setup_logging(
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging
    )
    
    return _log_config.get_logger()


def get_logger() -> logger:
    """
    Get the global logger instance.
    
    If logging hasn't been configured yet, this will set up
    default logging configuration.
    
    Returns:
        Configured logger instance
    """
    global _log_config
    
    if _log_config is None:
        # Set up default logging if not configured
        return setup_global_logging()
    
    return _log_config.get_logger()


def get_log_stats() -> Dict[str, Any]:
    """
    Get current logging statistics.
    
    Returns:
        Dictionary with logging statistics
    """
    global _log_config
    
    if _log_config is None:
        return {"error": "Logging not configured"}
    
    return _log_config.get_log_stats()


def update_log_level(new_level: str) -> None:
    """
    Update the global log level.
    
    Args:
        new_level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _log_config
    
    if _log_config is None:
        raise RuntimeError("Logging not configured. Call setup_global_logging() first.")
    
    _log_config.update_log_level(new_level)


# Context-aware logging functions
def log_api_request(
    method: str,
    endpoint: str,
    status_code: int,
    response_time: float,
    user_agent: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Log API request with structured context.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        status_code: HTTP status code
        response_time: Response time in seconds
        user_agent: User agent string (optional)
        user_id: User ID (optional)
    """
    log = get_logger()
    
    log_context = {
        "type": "api_request",
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "response_time": response_time,
        "user_agent": user_agent,
        "user_id": user_id
    }
    
    if status_code >= 400:
        log.warning(f"API {method} {endpoint} returned {status_code}", **log_context)
    else:
        log.info(f"API {method} {endpoint} completed", **log_context)


def log_chat_interaction(
    user_message: str,
    response_time: float,
    message_id: str,
    used_knowledge_base: bool = False,
    error: Optional[str] = None
) -> None:
    """
    Log chat interaction with structured context.
    
    Args:
        user_message: User's message (truncated for privacy)
        response_time: Response time in seconds
        message_id: Unique message identifier
        used_knowledge_base: Whether knowledge base was used
        error: Error message if any (optional)
    """
    log = get_logger()
    
    # Truncate user message for privacy
    truncated_message = user_message[:100] + "..." if len(user_message) > 100 else user_message
    
    log_context = {
        "type": "chat_interaction",
        "message_id": message_id,
        "user_message_length": len(user_message),
        "response_time": response_time,
        "used_knowledge_base": used_knowledge_base,
        "error": error
    }
    
    if error:
        log.error(f"Chat error for message {message_id}: {error}", **log_context)
    else:
        log.info(f"Chat response generated for message {message_id}", **log_context)


def log_knowledge_base_operation(
    operation: str,
    file_path: Optional[str] = None,
    document_count: Optional[int] = None,
    error: Optional[str] = None
) -> None:
    """
    Log knowledge base operations with structured context.
    
    Args:
        operation: Operation type (add, search, delete, etc.)
        file_path: File path if applicable (optional)
        document_count: Number of documents affected (optional)
        error: Error message if any (optional)
    """
    log = get_logger()
    
    log_context = {
        "type": "knowledge_base",
        "operation": operation,
        "file_path": file_path,
        "document_count": document_count,
        "error": error
    }
    
    if error:
        log.error(f"Knowledge base {operation} failed: {error}", **log_context)
    else:
        log.info(f"Knowledge base {operation} completed", **log_context)


def log_system_health(
    component: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log system health status with structured context.
    
    Args:
        component: System component name
        status: Health status (healthy, degraded, unhealthy)
        details: Additional health details (optional)
    """
    log = get_logger()
    
    log_context = {
        "type": "system_health",
        "component": component,
        "status": status,
        "details": details or {}
    }
    
    if status == "unhealthy":
        log.error(f"System component {component} is unhealthy", **log_context)
    elif status == "degraded":
        log.warning(f"System component {component} is degraded", **log_context)
    else:
        log.debug(f"System component {component} is healthy", **log_context)


# Convenience functions for common logging patterns
def log_startup() -> None:
    """Log application startup."""
    log = get_logger()
    log.info("Application starting up", type="startup")


def log_shutdown() -> None:
    """Log application shutdown."""
    log = get_logger()
    log.info("Application shutting down", type="shutdown")


def log_configuration_change(
    setting: str,
    old_value: Any,
    new_value: Any,
    user: Optional[str] = None
) -> None:
    """
    Log configuration changes.

    Args:
        setting: Configuration setting name
        old_value: Previous value
        new_value: New value
        user: User who made the change (optional)
    """
    log = get_logger()

    log_context = {
        "type": "configuration_change",
        "setting": setting,
        "old_value": str(old_value),
        "new_value": str(new_value),
        "user": user
    }

    log.info(f"Configuration changed: {setting}", **log_context)


def log_text_to_sql_operation(
    natural_language_query: str,
    generated_sql: str,
    execution_time: float,
    result_count: Optional[int] = None,
    table_name: Optional[str] = None,
    error: Optional[str] = None,
    query_complexity: Optional[str] = None
) -> None:
    """
    Log Text-to-SQL operations with structured context.

    Args:
        natural_language_query: Original natural language query
        generated_sql: Generated SQL query
        execution_time: Query execution time in seconds
        result_count: Number of results returned (optional)
        table_name: Table involved in the query (optional)
        error: Error message if any (optional)
        query_complexity: Query complexity level (simple, moderate, complex)
    """
    log = get_logger()

    # Truncate long queries for privacy and readability
    truncated_nl_query = natural_language_query[:200] + "..." if len(natural_language_query) > 200 else natural_language_query
    truncated_sql = generated_sql[:500] + "..." if len(generated_sql) > 500 else generated_sql

    log_context = {
        "type": "text_to_sql",
        "natural_language_query_length": len(natural_language_query),
        "generated_sql_length": len(generated_sql),
        "execution_time": execution_time,
        "result_count": result_count,
        "table_name": table_name,
        "error": error,
        "query_complexity": query_complexity
    }

    if error:
        log.error(
            f"Text-to-SQL failed for query: {truncated_nl_query}",
            **log_context
        )
    else:
        log.info(
            f"Text-to-SQL completed: {truncated_nl_query} -> {result_count} results",
            **log_context
        )


def log_sql_execution(
    sql_query: str,
    execution_time: float,
    row_count: Optional[int] = None,
    table_name: Optional[str] = None,
    error: Optional[str] = None,
    operation_type: Optional[str] = None
) -> None:
    """
    Log SQL execution operations with structured context.

    Args:
        sql_query: SQL query executed
        execution_time: Query execution time in seconds
        row_count: Number of rows affected/returned (optional)
        table_name: Table involved in the query (optional)
        error: Error message if any (optional)
        operation_type: Type of SQL operation (SELECT, INSERT, UPDATE, DELETE)
    """
    log = get_logger()

    # Truncate long SQL queries
    truncated_sql = sql_query[:300] + "..." if len(sql_query) > 300 else sql_query

    log_context = {
        "type": "sql_execution",
        "sql_query_length": len(sql_query),
        "execution_time": execution_time,
        "row_count": row_count,
        "table_name": table_name,
        "error": error,
        "operation_type": operation_type
    }

    if error:
        log.error(
            f"SQL execution failed: {truncated_sql}",
            **log_context
        )
    else:
        log.info(
            f"SQL execution completed: {operation_type} on {table_name} -> {row_count} rows",
            **log_context
        )


def log_excel_table_operation(
    operation: str,
    file_name: str,
    table_name: str,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
    error: Optional[str] = None
) -> None:
    """
    Log Excel table operations with structured context.

    Args:
        operation: Operation type (create, read, update, delete, search)
        file_name: Excel file name
        table_name: Table/sheet name
        row_count: Number of rows (optional)
        column_count: Number of columns (optional)
        error: Error message if any (optional)
    """
    log = get_logger()

    log_context = {
        "type": "excel_table_operation",
        "operation": operation,
        "file_name": file_name,
        "table_name": table_name,
        "row_count": row_count,
        "column_count": column_count,
        "error": error
    }

    if error:
        log.error(
            f"Excel table operation failed: {operation} on {table_name} in {file_name}",
            **log_context
        )
    else:
        log.info(
            f"Excel table operation completed: {operation} on {table_name} in {file_name}",
            **log_context
        )


def log_query_intent_detection(
    user_query: str,
    detected_intent: str,
    confidence: float,
    routed_service: str,
    error: Optional[str] = None
) -> None:
    """
    Log query intent detection and routing.

    Args:
        user_query: User's original query
        detected_intent: Detected intent (excel_data, knowledge_base, general)
        confidence: Confidence score (0.0 to 1.0)
        routed_service: Service routed to (text_to_sql, knowledge_base, chat)
        error: Error message if any (optional)
    """
    log = get_logger()

    # Truncate user query for privacy
    truncated_query = user_query[:150] + "..." if len(user_query) > 150 else user_query

    log_context = {
        "type": "query_intent_detection",
        "user_query_length": len(user_query),
        "detected_intent": detected_intent,
        "confidence": confidence,
        "routed_service": routed_service,
        "error": error
    }

    if error:
        log.error(
            f"Query intent detection failed for: {truncated_query}",
            **log_context
        )
    else:
        log.info(
            f"Query routed to {routed_service} with {confidence:.2f} confidence: {truncated_query}",
            **log_context
        )
