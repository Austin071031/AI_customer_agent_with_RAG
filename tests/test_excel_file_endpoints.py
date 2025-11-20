"""
Unit tests for Excel file management endpoints.

This module contains tests for the Excel file management API endpoints,
including listing, retrieving, deleting, and searching Excel files.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, status

from src.api.main import app
from src.models.excel_models import ExcelDocument, ExcelSheetData, ExcelSearchQuery
from src.services.sqlite_database_service import SQLiteDatabaseError

# Create test client
client = TestClient(app)


class TestExcelFileEndpoints:
    """Test cases for Excel file management endpoints."""

    @pytest.fixture
    def sample_excel_document(self):
        """Create a sample Excel document for testing."""
        return ExcelDocument(
            id="excel_test_123",
            file_name="test_data.xlsx",
            file_size=10240,
            sheet_names=["Sheet1", "Sheet2"],
            metadata={"description": "Test data", "source": "test"}
        )

    @pytest.fixture
    def sample_excel_sheet_data(self):
        """Create sample Excel sheet data for testing."""
        return ExcelSheetData(
            file_id="excel_test_123",
            sheet_name="Sheet1",
            headers=["Name", "Age", "City"],
            row_count=100,
            column_count=3,
            sample_data=[
                {"Name": "Alice", "Age": 30, "City": "New York"},
                {"Name": "Bob", "Age": 25, "City": "London"}
            ],
            data_types={"Name": "string", "Age": "integer", "City": "string"}
        )

    @pytest.fixture
    def sample_excel_search_query(self):
        """Create sample Excel search query for testing."""
        return ExcelSearchQuery(
            query="Alice",
            file_id="excel_test_123",
            sheet_name="Sheet1",
            column_name="Name",
            case_sensitive=False,
            max_results=10
        )

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_list_excel_files_success(self, mock_get_kb_manager, sample_excel_document):
        """Test successful listing of Excel files."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.list_excel_files.return_value = [sample_excel_document]
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "excel_test_123"
        assert data[0]["file_name"] == "test_data.xlsx"
        mock_kb_manager.list_excel_files.assert_called_once()

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_list_excel_files_with_pagination(self, mock_get_kb_manager, sample_excel_document):
        """Test listing Excel files with pagination parameters."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.list_excel_files.return_value = [sample_excel_document] * 5
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request with pagination
        response = client.get("/api/excel-files/?skip=1&limit=2")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2
        mock_kb_manager.list_excel_files.assert_called_once()

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_list_excel_files_database_error(self, mock_get_kb_manager):
        """Test error handling when listing Excel files fails."""
        # Mock the knowledge base manager to raise an error
        mock_kb_manager = Mock()
        mock_kb_manager.list_excel_files.side_effect = SQLiteDatabaseError("Database connection failed")
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/")

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to list Excel files" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_success(self, mock_get_kb_manager, sample_excel_document):
        """Test successful retrieval of specific Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/excel_test_123")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "excel_test_123"
        assert data["file_name"] == "test_data.xlsx"
        mock_kb_manager.get_excel_file.assert_called_once_with("excel_test_123")

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_not_found(self, mock_get_kb_manager):
        """Test retrieval of non-existent Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = None
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/nonexistent_id")

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_database_error(self, mock_get_kb_manager):
        """Test error handling when retrieving Excel file fails."""
        # Mock the knowledge base manager to raise an error
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.side_effect = SQLiteDatabaseError("Database error")
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/excel_test_123")

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve Excel file" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_delete_excel_file_success(self, mock_get_kb_manager, sample_excel_document):
        """Test successful deletion of Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        mock_kb_manager.delete_excel_file.return_value = True
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.delete("/api/excel-files/excel_test_123")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Excel file with ID 'excel_test_123' deleted successfully"
        assert data["status"] == "deleted"
        mock_kb_manager.delete_excel_file.assert_called_once_with("excel_test_123")

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_delete_excel_file_not_found(self, mock_get_kb_manager):
        """Test deletion of non-existent Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = None
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.delete("/api/excel-files/nonexistent_id")

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_delete_excel_file_failure(self, mock_get_kb_manager, sample_excel_document):
        """Test deletion failure when database operation returns False."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        mock_kb_manager.delete_excel_file.return_value = False
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.delete("/api/excel-files/excel_test_123")

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to delete" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_sheets_success(self, mock_get_kb_manager, sample_excel_document, sample_excel_sheet_data):
        """Test successful retrieval of Excel file sheets."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        mock_kb_manager.get_sheet_data.return_value = [sample_excel_sheet_data]
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/excel_test_123/sheets")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["sheet_name"] == "Sheet1"
        assert data[0]["file_id"] == "excel_test_123"
        mock_kb_manager.get_sheet_data.assert_called_once_with("excel_test_123", None)

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_sheets_with_sheet_name(self, mock_get_kb_manager, sample_excel_document, sample_excel_sheet_data):
        """Test retrieval of specific Excel file sheet."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        mock_kb_manager.get_sheet_data.return_value = [sample_excel_sheet_data]
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request with specific sheet name
        response = client.get("/api/excel-files/excel_test_123/sheets?sheet_name=Sheet1")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["sheet_name"] == "Sheet1"
        mock_kb_manager.get_sheet_data.assert_called_once_with("excel_test_123", "Sheet1")

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_sheets_not_found(self, mock_get_kb_manager):
        """Test retrieval of sheets for non-existent Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = None
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/nonexistent_id/sheets")

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_data_success(self, mock_get_kb_manager, sample_excel_search_query):
        """Test successful search across Excel files."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        search_results = [
            {
                "file_name": "test_data.xlsx",
                "sheet_name": "Sheet1",
                "column_name": "Name",
                "cell_value": "Alice",
                "file_id": "excel_test_123"
            }
        ]
        mock_kb_manager.search_excel_data.return_value = search_results
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        search_data = sample_excel_search_query.model_dump()
        response = client.post("/api/excel-files/search", json=search_data)

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == "Alice"
        assert data["result_count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["cell_value"] == "Alice"
        mock_kb_manager.search_excel_data.assert_called_once()

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_file_data_success(self, mock_get_kb_manager, sample_excel_document):
        """Test successful search within specific Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        search_results = [
            {
                "file_name": "test_data.xlsx",
                "sheet_name": "Sheet1",
                "column_name": "Name",
                "cell_value": "Alice",
                "file_id": "excel_test_123"
            }
        ]
        mock_kb_manager.search_excel_data.return_value = search_results
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/excel_test_123/search?query=Alice")

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["file_id"] == "excel_test_123"
        assert data["query"] == "Alice"
        assert data["result_count"] == 1
        assert data["results"][0]["cell_value"] == "Alice"

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_file_data_with_filters(self, mock_get_kb_manager, sample_excel_document):
        """Test search within Excel file with filters."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_document
        search_results = [
            {
                "file_name": "test_data.xlsx",
                "sheet_name": "Sheet1",
                "column_name": "Name",
                "cell_value": "Alice",
                "file_id": "excel_test_123"
            }
        ]
        mock_kb_manager.search_excel_data.return_value = search_results
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request with filters
        response = client.get(
            "/api/excel-files/excel_test_123/search"
            "?query=Alice"
            "&sheet_name=Sheet1"
            "&column_name=Name"
            "&case_sensitive=true"
            "&max_results=5"
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["filters"]["sheet_name"] == "Sheet1"
        assert data["filters"]["column_name"] == "Name"
        assert data["filters"]["case_sensitive"] is True
        assert data["filters"]["max_results"] == 5

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_file_data_not_found(self, mock_get_kb_manager):
        """Test search within non-existent Excel file."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = None
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/nonexistent_id/search?query=test")

        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_data_database_error(self, mock_get_kb_manager, sample_excel_search_query):
        """Test error handling when search fails."""
        # Mock the knowledge base manager to raise an error
        mock_kb_manager = Mock()
        mock_kb_manager.search_excel_data.side_effect = SQLiteDatabaseError("Search error")
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        search_data = sample_excel_search_query.model_dump()
        response = client.post("/api/excel-files/search", json=search_data)

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to search Excel data" in data["detail"]

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_knowledge_base_manager_unavailable(self, mock_get_kb_manager):
        """Test error handling when knowledge base manager is not available."""
        # Mock get_kb_manager to raise HTTPException
        mock_get_kb_manager.side_effect = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base manager not available"
        )

        # Make request
        response = client.get("/api/excel-files/")

        # Verify error response
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "Knowledge base manager not available" in data["detail"]

    def test_search_excel_data_validation_error(self):
        """Test validation error for invalid search query."""
        # Make request with invalid data (empty query)
        invalid_data = {
            "query": "",  # Empty query should fail validation
            "max_results": 50
        }
        response = client.post("/api/excel-files/search", json=invalid_data)

        # Verify validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_excel_files_validation_error(self):
        """Test validation error for invalid pagination parameters."""
        # Make request with invalid pagination
        response = client.get("/api/excel-files/?skip=-1&limit=0")

        # Verify validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_unexpected_error_handling(self, mock_get_kb_manager):
        """Test handling of unexpected errors."""
        # Mock the knowledge base manager to raise unexpected error
        mock_kb_manager = Mock()
        mock_kb_manager.list_excel_files.side_effect = Exception("Unexpected error")
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request
        response = client.get("/api/excel-files/")

        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "unexpected error" in data["detail"].lower()


class TestExcelFileEndpointsIntegration:
    """Integration tests for Excel file endpoints with actual API calls."""

    def test_api_root_includes_excel_files_info(self):
        """Test that API root includes information about Excel files endpoints."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "docs" in data
        assert "/docs" in data["docs"]  # FastAPI docs endpoint

    def test_api_documentation_includes_excel_files(self):
        """Test that API documentation includes Excel files endpoints."""
        response = client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        # The docs should be accessible (HTML response)
        assert "text/html" in response.headers.get("content-type", "")

    def test_health_check_includes_knowledge_base(self):
        """Test that health check includes knowledge base status."""
        response = client.get("/health")
        
        # Health check should be accessible
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
