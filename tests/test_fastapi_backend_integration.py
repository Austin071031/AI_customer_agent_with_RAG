"""
Integration tests for FastAPI Backend with Text-to-SQL and Relational Storage.

This module contains comprehensive integration tests for the enhanced FastAPI backend
with Text-to-SQL service integration, dynamic table creation, and relational storage
for Excel files.
"""

import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from src.api.main import app, app_state
from src.models.excel_models import ExcelDocument, ExcelSheetData
from src.services.text_to_sql_service import TextToSQLService, TextToSQLError
from src.services.sqlite_database_service import SQLiteDatabaseError


@pytest.fixture
def client():
    """
    Test client fixture for FastAPI application.
    
    Returns:
        TestClient instance for testing API endpoints
    """
    return TestClient(app)


@pytest.fixture
def mock_enhanced_services():
    """
    Enhanced mock services fixture with Text-to-SQL and relational storage support.
    
    Returns:
        Dictionary with mocked service instances including Text-to-SQL service
    """
    # Mock configuration manager
    mock_config_manager = MagicMock()
    mock_config_manager.get_api_config.return_value = MagicMock(
        api_key="test-api-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    mock_config_manager.get_db_config.return_value = MagicMock()
    mock_config_manager.get_app_config.return_value = MagicMock(
        api_config=MagicMock(api_key="test-api-key")
    )
    mock_config_manager.get_last_modified.return_value = "2024-01-01T12:00:00Z"
    mock_config_manager.update_api_config.return_value = True
    mock_config_manager.update_db_config.return_value = True
    mock_config_manager.update_app_config.return_value = True
    mock_config_manager.reset_to_defaults.return_value = True
    mock_config_manager.validate_configuration.return_value = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    mock_config_manager.create_backup.return_value = {
        "backup_file": "/backups/config_backup_20240101.zip",
        "backup_time": "2024-01-01T12:00:00Z",
        "config_files": ["settings.yaml", ".env"]
    }
    mock_config_manager.health_check.return_value = True
    mock_config_manager.get_loaded_config_files.return_value = ["settings.yaml", ".env"]
    
    # Mock DeepSeek service
    mock_deepseek_service = AsyncMock()
    mock_deepseek_service.chat_completion.return_value = "This is a test response from the AI."
    mock_deepseek_service.stream_chat.return_value = AsyncMock()
    mock_deepseek_service.health_check.return_value = True
    
    # Mock knowledge base manager with enhanced Excel support
    mock_kb_manager = MagicMock()
    
    # Mock Excel files for relational storage testing
    mock_excel_files = [
        ExcelDocument(
            file_id="excel_001",
            file_name="sales_data.xlsx",
            file_size=1024000,
            sheet_names=["Sheet1", "Sheet2"],
            metadata={"uploaded_by": "test_user"},
            upload_date="2024-01-01T12:00:00Z"
        ),
        ExcelDocument(
            file_id="excel_002",
            file_name="customer_data.xlsx",
            file_size=2048000,
            sheet_names=["Customers", "Orders"],
            metadata={"uploaded_by": "test_user"},
            upload_date="2024-01-01T12:30:00Z"
        )
    ]
    
    mock_kb_manager.list_excel_files.return_value = mock_excel_files
    mock_kb_manager.get_excel_file.return_value = mock_excel_files[0]
    mock_kb_manager.delete_excel_file.return_value = True
    
    # Mock sheet data with dynamic table schema
    mock_sheet_data = [
        ExcelSheetData(
            file_id="excel_001",
            sheet_name="Sheet1",
            headers=["id", "name", "amount", "date"],
            column_types=["INTEGER", "TEXT", "REAL", "TEXT"],
            data_sample=[{"id": 1, "name": "Product A", "amount": 100.50, "date": "2024-01-01"}],
            row_count=100,
            table_name="excel_001_sheet1"
        )
    ]
    
    mock_kb_manager.get_sheet_data.return_value = mock_sheet_data
    mock_kb_manager.search_excel_data.return_value = [
        {
            "file_id": "excel_001",
            "sheet_name": "Sheet1",
            "row_data": {"id": 1, "name": "Product A", "amount": 100.50},
            "matched_column": "name",
            "matched_value": "Product A"
        }
    ]
    
    # Mock knowledge base operations
    mock_kb_manager.search_similar.return_value = []
    mock_kb_manager.add_documents.return_value = {
        "documents": ["test.pdf"],
        "excel_files": ["sales_data.xlsx"],
        "errors": []
    }
    mock_kb_manager.clear_knowledge_base.return_value = True
    mock_kb_manager.get_document_count.return_value = 5
    mock_kb_manager.get_statistics.return_value = {
        "total_documents": 5,
        "document_types": {"pdf": 2, "xlsx": 2, "txt": 1},
        "total_size_bytes": 3072000,
        "embedding_model": "all-MiniLM-L6-v2",
        "collection_name": "documents",
        "last_updated": "2024-01-01T12:00:00Z"
    }
    mock_kb_manager.health_check.return_value = True
    
    # Mock Text-to-SQL service with relational query execution
    mock_text_to_sql_service = MagicMock()
    mock_text_to_sql_service.convert_to_sql.return_value = {
        "sql_query": "SELECT name, amount FROM excel_001_sheet1 WHERE amount > 100",
        "explanation": "Find products with amount greater than 100",
        "results": [
            {"name": "Product A", "amount": 100.50},
            {"name": "Product B", "amount": 200.75}
        ],
        "error": None
    }
    mock_text_to_sql_service.get_table_schema.return_value = {
        "table_name": "excel_001_sheet1",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
            {"name": "amount", "type": "REAL"},
            {"name": "date", "type": "TEXT"}
        ],
        "row_count": 100
    }
    mock_text_to_sql_service.list_tables.return_value = [
        "excel_001_sheet1",
        "excel_001_sheet2",
        "excel_002_customers"
    ]
    mock_text_to_sql_service.health_check.return_value = True
    
    # Mock chat manager with Text-to-SQL integration
    mock_chat_manager = MagicMock()
    mock_chat_manager.process_message = AsyncMock(return_value="This is a test AI response.")
    mock_chat_manager.stream_message = AsyncMock()
    
    # Create a proper async generator for stream_message
    async def mock_stream_generator():
        yield "Hello"
        yield " there"
        yield "!"
    
    mock_chat_manager.stream_message.return_value = mock_stream_generator()
    mock_chat_manager.get_conversation_history.return_value = []
    mock_chat_manager.clear_conversation.return_value = None
    mock_chat_manager.get_conversation_summary.return_value = {
        "total_messages": 0,
        "user_messages": 0,
        "assistant_messages": 0,
        "max_history_length": 20,
        "history_usage_percentage": 0.0
    }
    mock_chat_manager.health_check = AsyncMock(return_value=True)
    
    return {
        "config_manager": mock_config_manager,
        "deepseek_service": mock_deepseek_service,
        "kb_manager": mock_kb_manager,
        "text_to_sql_service": mock_text_to_sql_service,
        "chat_manager": mock_chat_manager
    }


@pytest.fixture(autouse=True)
def setup_enhanced_app_state(mock_enhanced_services):
    """
    Automatically set up app state with enhanced mocked services before each test.
    
    Args:
        mock_enhanced_services: Dictionary with enhanced mocked service instances
    """
    # Clear existing app state
    app_state.clear()
    
    # Set up enhanced mocked services
    app_state.update(mock_enhanced_services)
    
    yield
    
    # Clean up after test
    app_state.clear()


class TestEnhancedHealthCheck:
    """Test cases for enhanced health check with Text-to-SQL service."""
    
    def test_health_check_includes_text_to_sql_service(self, client):
        """
        Test health check endpoint includes Text-to-SQL service status.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "services" in data
        services = data["services"]
        
        # Check that Text-to-SQL service is included in health check
        assert "text_to_sql_service" in services
        assert services["text_to_sql_service"] == "healthy"
        
        # Verify all expected services are present
        expected_services = [
            "api", "config_manager", "deepseek_service", 
            "knowledge_base", "chat_manager", "text_to_sql_service"
        ]
        for service in expected_services:
            assert service in services


class TestExcelFileManagementWithRelationalStorage:
    """Test cases for Excel file management with relational storage."""
    
    def test_list_excel_files_success(self, client):
        """
        Test successful listing of Excel files stored in relational database.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/excel-files/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Verify Excel file structure
        first_file = data[0]
        assert "file_id" in first_file
        assert "file_name" in first_file
        assert "sheet_names" in first_file
        assert "metadata" in first_file
        assert first_file["file_name"] == "sales_data.xlsx"
        assert "Sheet1" in first_file["sheet_names"]
        assert "Sheet2" in first_file["sheet_names"]
    
    def test_get_excel_file_details_success(self, client):
        """
        Test successful retrieval of specific Excel file details.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/excel-files/excel_001")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["file_id"] == "excel_001"
        assert data["file_name"] == "sales_data.xlsx"
        assert "sheet_names" in data
        assert "metadata" in data
    
    def test_get_excel_file_not_found(self, client, mock_enhanced_services):
        """
        Test handling of non-existent Excel file.
        
        Args:
            client: TestClient instance
            mock_enhanced_services: Dictionary with mocked service instances
        """
        # Mock get_excel_file to return None (file not found)
        mock_enhanced_services["kb_manager"].get_excel_file.return_value = None
        
        response = client.get("/api/excel-files/nonexistent_id")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_delete_excel_file_success(self, client):
        """
        Test successful deletion of Excel file from relational storage.
        
        Args:
            client: TestClient instance
        """
        response = client.delete("/api/excel-files/excel_001")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "deleted successfully" in data["message"]
        assert data["file_id"] == "excel_001"
        assert data["status"] == "deleted"
    
    def test_get_excel_sheet_data_success(self, client):
        """
        Test successful retrieval of Excel sheet data with dynamic table schema.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/excel-files/excel_001/sheets")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        
        # Verify sheet data structure with dynamic table schema
        sheet_data = data[0]
        assert "file_id" in sheet_data
        assert "sheet_name" in sheet_data
        assert "headers" in sheet_data
        assert "column_types" in sheet_data
        assert "data_sample" in sheet_data
        assert "row_count" in sheet_data
        assert "table_name" in sheet_data
        
        # Verify dynamic table creation features
        assert sheet_data["file_id"] == "excel_001"
        assert sheet_data["sheet_name"] == "Sheet1"
        assert sheet_data["headers"] == ["id", "name", "amount", "date"]
        assert sheet_data["column_types"] == ["INTEGER", "TEXT", "REAL", "TEXT"]
        assert sheet_data["table_name"] == "excel_001_sheet1"
    
    def test_search_excel_data_success(self, client):
        """
        Test successful search within Excel data using relational queries.
        
        Args:
            client: TestClient instance
        """
        search_data = {
            "query": "Product A",
            "file_id": "excel_001",
            "sheet_name": "Sheet1",
            "column_name": "name",
            "case_sensitive": False,
            "max_results": 10
        }
        
        response = client.post("/api/excel-files/search", json=search_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "result_count" in data
        assert "filters" in data
        
        # Verify search results structure
        results = data["results"]
        assert len(results) == 1
        first_result = results[0]
        assert "file_id" in first_result
        assert "sheet_name" in first_result
        assert "row_data" in first_result
        assert "matched_column" in first_result
        assert "matched_value" in first_result


class TestTextToSQLIntegration:
    """Test cases for Text-to-SQL service integration."""
    
    def test_chat_with_excel_data_query(self, client):
        """
        Test chat functionality with Excel data query that should route to Text-to-SQL.
        
        Args:
            client: TestClient instance
        """
        # Mock chat manager to simulate Text-to-SQL routing
        chat_request = {
            "message": "Show me products with sales greater than 100",
            "use_knowledge_base": True,
            "stream": False
        }
        
        response = client.post("/api/chat/", json=chat_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert "message_id" in data
        assert "conversation_length" in data
        assert "used_knowledge_base" in data
    
    def test_text_to_sql_service_available(self, client):
        """
        Test that Text-to-SQL service is properly initialized and available.
        
        Args:
            client: TestClient instance
        """
        # The service should be available in app_state
        text_to_sql_service = app_state.get("text_to_sql_service")
        assert text_to_sql_service is not None
        assert hasattr(text_to_sql_service, "convert_to_sql")
        assert hasattr(text_to_sql_service, "get_table_schema")
        assert hasattr(text_to_sql_service, "list_tables")


class TestKnowledgeBaseWithDynamicTableCreation:
    """Test cases for knowledge base with dynamic table creation for Excel files."""
    
    def test_upload_excel_file_with_dynamic_tables(self, client):
        """
        Test uploading Excel file that should create dynamic tables in SQLite.
        
        Args:
            client: TestClient instance
        """
        # Create a test Excel file content
        test_excel_content = b"fake excel content"
        
        files = [
            ("files", ("test_data.xlsx", test_excel_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
        ]
        
        response = client.post(
            "/api/knowledge-base/documents",
            files=files,
            data={"overwrite": False}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "uploaded_files" in data
        assert "failed_files" in data
        assert "total_processed" in data
        assert "total_successful" in data
        
        # Verify Excel file was processed (should be in uploaded_files)
        assert "test_data.xlsx" in data["uploaded_files"]
        assert data["total_successful"] == 1
    
    def test_knowledge_base_info_includes_excel_stats(self, client):
        """
        Test knowledge base information includes Excel file statistics.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/api/knowledge-base/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_documents" in data
        assert "document_types" in data
        assert "total_size_bytes" in data
        
        # Verify Excel files are included in document types
        document_types = data["document_types"]
        assert "xlsx" in document_types
        assert document_types["xlsx"] == 2  # From our mock data


class TestErrorHandlingWithEnhancedServices:
    """Test cases for error handling with enhanced services."""
    
    def test_text_to_sql_service_unavailable(self, client):
        """
        Test error handling when Text-to-SQL service is not available.
        
        Args:
            client: TestClient instance
        """
        # Remove Text-to-SQL service from app_state
        app_state.pop("text_to_sql_service", None)
        
        # Health check should reflect Text-to-SQL service as unhealthy
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        services = data["services"]
        assert services["text_to_sql_service"] == "unhealthy"
        # Overall status should be degraded, not healthy
        assert data["status"] == "degraded"
    
    def test_excel_file_operation_sqlite_error(self, client, mock_enhanced_services):
        """
        Test error handling for SQLite database errors in Excel operations.
        
        Args:
            client: TestClient instance
            mock_enhanced_services: Dictionary with mocked service instances
        """
        # Mock SQLiteDatabaseError for list_excel_files
        mock_enhanced_services["kb_manager"].list_excel_files.side_effect = SQLiteDatabaseError(
            "Database connection failed"
        )
        
        response = client.get("/api/excel-files/")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Failed to list Excel files" in data["detail"]
    
    def test_text_to_sql_service_error(self, client, mock_enhanced_services):
        """
        Test error handling for Text-to-SQL service errors.
        
        Args:
            client: TestClient instance
            mock_enhanced_services: Dictionary with mocked service instances
        """
        # Mock TextToSQLError for convert_to_sql
        mock_enhanced_services["text_to_sql_service"].convert_to_sql.side_effect = TextToSQLError(
            "Failed to generate SQL query"
        )
        
        # This would be called internally by chat manager for Excel data queries
        # We don't have a direct endpoint for Text-to-SQL, so we test through chat
        chat_request = {
            "message": "Show me sales data for last month",
            "use_knowledge_base": True,
            "stream": False
        }
        
        # The chat manager should handle the TextToSQLError gracefully
        response = client.post("/api/chat/", json=chat_request)
        
        # Should still return a successful response with fallback behavior
        assert response.status_code == status.HTTP_200_OK


class TestConfigurationEndpointsWithEnhancedFeatures:
    """Test cases for configuration endpoints with enhanced features."""
    
    def test_system_info_includes_enhanced_features(self, client):
        """
        Test system information includes enhanced feature status.
        
        Args:
            client: TestClient instance
        """
        response = client.get("/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "system" in data
        assert "configuration" in data
        
        config = data["configuration"]
        # Verify enhanced features are reflected in configuration
        assert "api_enabled" in config
        assert "knowledge_base_enabled" in config
        assert "chat_enabled" in config
        assert "log_level" in config
        assert "enable_gpu" in config
        assert "max_conversation_history" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
