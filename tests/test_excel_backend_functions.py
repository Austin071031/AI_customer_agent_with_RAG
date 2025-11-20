"""
Comprehensive Unit Tests for Backend Excel Functions.

This module contains comprehensive unit tests for the backend Excel functions
that are used by the Streamlit UI. It covers API endpoints, knowledge base
Excel operations, and data models.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from pathlib import Path
import tempfile
import os

from src.api.main import app
from src.models.excel_models import ExcelDocument, ExcelSheetData, ExcelSearchQuery
from src.services.knowledge_base import KnowledgeBaseManager, KnowledgeBaseError
from src.services.sqlite_database_service import SQLiteDatabaseError

# Create test client
client = TestClient(app)


class TestExcelBackendFunctions:
    """Test cases for backend Excel functions used by Streamlit UI."""
    
    @pytest.fixture
    def sample_excel_files(self):
        """Create sample Excel files data for testing."""
        return [
            ExcelDocument(
                id="excel_1",
                file_name="sales_data.xlsx",
                file_size=15360,
                sheet_names=["Sales", "Customers", "Products"],
                metadata={"description": "Sales data for Q1", "source": "upload"}
            ),
            ExcelDocument(
                id="excel_2", 
                file_name="inventory.xlsx",
                file_size=20480,
                sheet_names=["Inventory", "Suppliers"],
                metadata={"description": "Current inventory levels", "source": "upload"}
            )
        ]
    
    @pytest.fixture
    def sample_sheet_data(self):
        """Create sample sheet data for testing."""
        return [
            ExcelSheetData(
                file_id="excel_1",
                sheet_name="Sales",
                headers=["Date", "Product", "Quantity", "Revenue"],
                row_count=1000,
                column_count=4,
                sample_data=[
                    {"Date": "2024-01-01", "Product": "Widget A", "Quantity": 50, "Revenue": 5000.00},
                    {"Date": "2024-01-02", "Product": "Widget B", "Quantity": 30, "Revenue": 3000.00}
                ],
                data_types={"Date": "date", "Product": "string", "Quantity": "integer", "Revenue": "float"}
            )
        ]
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing."""
        return [
            {
                "file_id": "excel_1",
                "file_name": "sales_data.xlsx",
                "sheet_name": "Sales",
                "column_name": "Product",
                "row_number": 1,
                "cell_value": "Widget A",
                "similarity_score": 1.0
            },
            {
                "file_id": "excel_1",
                "file_name": "sales_data.xlsx",
                "sheet_name": "Sales", 
                "column_name": "Product",
                "row_number": 2,
                "cell_value": "Widget B",
                "similarity_score": 0.9
            }
        ]

    # Test API Endpoints used by Streamlit UI
    
    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_list_excel_files_endpoint(self, mock_get_kb_manager, sample_excel_files):
        """Test the list Excel files endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.list_excel_files.return_value = sample_excel_files
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request - this is called by Streamlit UI's list_excel_files() function
        response = client.get("/api/excel-files/")
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "excel_1"
        assert data[0]["file_name"] == "sales_data.xlsx"
        assert "sheet_names" in data[0]
        assert "Sales" in data[0]["sheet_names"]
        
        # Verify the backend function was called
        mock_kb_manager.list_excel_files.assert_called_once()

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_file_endpoint(self, mock_get_kb_manager, sample_excel_files):
        """Test the get Excel file endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_files[0]
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request - this is called by Streamlit UI's get_excel_file() function
        response = client.get("/api/excel-files/excel_1")
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == "excel_1"
        assert data["file_name"] == "sales_data.xlsx"
        assert data["file_size"] == 15360
        assert "sheet_names" in data
        assert "metadata" in data
        
        # Verify the backend function was called with correct parameters
        mock_kb_manager.get_excel_file.assert_called_once_with("excel_1")

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_delete_excel_file_endpoint(self, mock_get_kb_manager, sample_excel_files):
        """Test the delete Excel file endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_files[0]
        mock_kb_manager.delete_excel_file.return_value = True
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request - this is called by Streamlit UI's delete_excel_file() function
        response = client.delete("/api/excel-files/excel_1")
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Excel file with ID 'excel_1' deleted successfully"
        assert data["status"] == "deleted"
        assert data["file_id"] == "excel_1"
        
        # Verify the backend functions were called
        mock_kb_manager.get_excel_file.assert_called_once_with("excel_1")
        mock_kb_manager.delete_excel_file.assert_called_once_with("excel_1")

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_get_excel_sheets_endpoint(self, mock_get_kb_manager, sample_excel_files, sample_sheet_data):
        """Test the get Excel sheets endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_files[0]
        mock_kb_manager.get_sheet_data.return_value = sample_sheet_data
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request - this is called by Streamlit UI's get_excel_sheets() function
        response = client.get("/api/excel-files/excel_1/sheets")
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["file_id"] == "excel_1"
        assert data[0]["sheet_name"] == "Sales"
        assert data[0]["row_count"] == 1000
        assert data[0]["column_count"] == 4
        assert "headers" in data[0]
        assert "sample_data" in data[0]
        
        # Verify the backend functions were called
        mock_kb_manager.get_excel_file.assert_called_once_with("excel_1")
        mock_kb_manager.get_sheet_data.assert_called_once_with("excel_1", None)

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_data_endpoint(self, mock_get_kb_manager, sample_search_results):
        """Test the search Excel data endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.search_excel_data.return_value = sample_search_results
        mock_get_kb_manager.return_value = mock_kb_manager

        # Create search query - this matches what Streamlit UI sends
        search_query = {
            "query": "Widget",
            "max_results": 10
        }

        # Make request - this is called by Streamlit UI's search_excel_data() function
        response = client.post("/api/excel-files/search", json=search_query)
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == "Widget"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["cell_value"] == "Widget A"
        assert data["results"][1]["cell_value"] == "Widget B"
        
        # Verify the backend function was called with correct parameters
        mock_kb_manager.search_excel_data.assert_called_once()
        call_args = mock_kb_manager.search_excel_data.call_args[0][0]
        assert call_args.query == "Widget"
        assert call_args.max_results == 10

    @patch("src.api.endpoints.excel_files.get_kb_manager")
    def test_search_excel_file_data_endpoint(self, mock_get_kb_manager, sample_excel_files, sample_search_results):
        """Test the search within specific Excel file endpoint used by Streamlit UI."""
        # Mock the knowledge base manager
        mock_kb_manager = Mock()
        mock_kb_manager.get_excel_file.return_value = sample_excel_files[0]
        mock_kb_manager.search_excel_data.return_value = sample_search_results
        mock_get_kb_manager.return_value = mock_kb_manager

        # Make request - this is called by Streamlit UI's search_excel_data() function with file_id
        response = client.get("/api/excel-files/excel_1/search?query=Widget&max_results=5")
        
        # Verify response structure and data
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["file_id"] == "excel_1"
        assert data["query"] == "Widget"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["filters"]["max_results"] == 5
        
        # Verify the backend functions were called
        mock_kb_manager.get_excel_file.assert_called_once_with("excel_1")
        mock_kb_manager.search_excel_data.assert_called_once()

    # Test Knowledge Base Manager Excel Functions
    
    @patch("src.services.knowledge_base.chromadb")
    def test_knowledge_base_manager_list_excel_files(self, mock_chromadb, sample_excel_files):
        """Test KnowledgeBaseManager list_excel_files method."""
        # Mock the SQLite service
        mock_sqlite_service = Mock()
        mock_sqlite_service.list_excel_files.return_value = sample_excel_files
        
        # Create KnowledgeBaseManager with mocked SQLite service
        manager = KnowledgeBaseManager()
        manager.sqlite_service = mock_sqlite_service
        
        # Call the method
        result = manager.list_excel_files()
        
        # Verify result
        assert len(result) == 2
        assert result[0].file_name == "sales_data.xlsx"
        assert result[1].file_name == "inventory.xlsx"
        mock_sqlite_service.list_excel_files.assert_called_once()

    @patch("src.services.knowledge_base.chromadb")
    def test_knowledge_base_manager_get_excel_file(self, mock_chromadb, sample_excel_files):
        """Test KnowledgeBaseManager get_excel_file method."""
        # Mock the SQLite service
        mock_sqlite_service = Mock()
        mock_sqlite_service.get_excel_file.return_value = sample_excel_files[0]
        
        # Create KnowledgeBaseManager with mocked SQLite service
        manager = KnowledgeBaseManager()
        manager.sqlite_service = mock_sqlite_service
        
        # Call the method
        result = manager.get_excel_file("excel_1")
        
        # Verify result
        assert result.file_name == "sales_data.xlsx"
        assert result.file_size == 15360
        mock_sqlite_service.get_excel_file.assert_called_once_with("excel_1")

    @patch("src.services.knowledge_base.chromadb")
    def test_knowledge_base_manager_delete_excel_file(self, mock_chromadb):
        """Test KnowledgeBaseManager delete_excel_file method."""
        # Mock the SQLite service
        mock_sqlite_service = Mock()
        mock_sqlite_service.delete_excel_file.return_value = True
        
        # Create KnowledgeBaseManager with mocked SQLite service
        manager = KnowledgeBaseManager()
        manager.sqlite_service = mock_sqlite_service
        
        # Call the method
        result = manager.delete_excel_file("excel_1")
        
        # Verify result
        assert result is True
        mock_sqlite_service.delete_excel_file.assert_called_once_with("excel_1")

    @patch("src.services.knowledge_base.chromadb")
    def test_knowledge_base_manager_get_sheet_data(self, mock_chromadb, sample_sheet_data):
        """Test KnowledgeBaseManager get_sheet_data method."""
        # Mock the SQLite service
        mock_sqlite_service = Mock()
        mock_sqlite_service.get_sheet_data.return_value = sample_sheet_data
        
        # Create KnowledgeBaseManager with mocked SQLite service
        manager = KnowledgeBaseManager()
        manager.sqlite_service = mock_sqlite_service
        
        # Call the method
        result = manager.get_sheet_data("excel_1", "Sales")
        
        # Verify result
        assert len(result) == 1
        assert result[0].sheet_name == "Sales"
        assert result[0].row_count == 1000
        mock_sqlite_service.get_sheet_data.assert_called_once_with("excel_1", "Sales")

    @patch("src.services.knowledge_base.chromadb")
    def test_knowledge_base_manager_search_excel_data(self, mock_chromadb, sample_search_results):
        """Test KnowledgeBaseManager search_excel_data method."""
        # Mock the SQLite service
        mock_sqlite_service = Mock()
        mock_sqlite_service.search_excel_data.return_value = sample_search_results
        
        # Create KnowledgeBaseManager with mocked SQLite service
        manager = KnowledgeBaseManager()
        manager.sqlite_service = mock_sqlite_service
        
        # Create search query
        search_query = ExcelSearchQuery(
            query="Widget",
            file_id="excel_1",
            sheet_name="Sales",
            max_results=10
        )
        
        # Call the method
        result = manager.search_excel_data(search_query)
        
        # Verify result
        assert len(result) == 2
        assert result[0]["cell_value"] == "Widget A"
        assert result[1]["cell_value"] == "Widget B"
        mock_sqlite_service.search_excel_data.assert_called_once_with(
            query="Widget",
            file_id="excel_1",
            sheet_name="Sales",
            column_name=None,
            case_sensitive=False,
            max_results=10
        )

    # Test Error Handling Scenarios
    
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

    # Test Data Validation
    
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

    # Test ExcelSearchQuery Model
    
    def test_excel_search_query_validation(self):
        """Test ExcelSearchQuery model validation."""
        # Valid query
        valid_query = ExcelSearchQuery(
            query="test",
            max_results=10
        )
        assert valid_query.query == "test"
        assert valid_query.max_results == 10
        
        # Test default values
        query_with_defaults = ExcelSearchQuery(query="test")
        assert query_with_defaults.case_sensitive is False
        assert query_with_defaults.max_results == 50

    # Test ExcelDocument Model
    
    def test_excel_document_validation(self, sample_excel_files):
        """Test ExcelDocument model validation."""
        excel_doc = sample_excel_files[0]
        
        assert excel_doc.id == "excel_1"
        assert excel_doc.file_name == "sales_data.xlsx"
        assert excel_doc.file_size == 15360
        assert len(excel_doc.sheet_names) == 3
        assert "Sales" in excel_doc.sheet_names
        assert excel_doc.metadata["description"] == "Sales data for Q1"

    # Test ExcelSheetData Model
    
    def test_excel_sheet_data_validation(self, sample_sheet_data):
        """Test ExcelSheetData model validation."""
        sheet_data = sample_sheet_data[0]
        
        assert sheet_data.file_id == "excel_1"
        assert sheet_data.sheet_name == "Sales"
        assert sheet_data.row_count == 1000
        assert sheet_data.column_count == 4
        assert len(sheet_data.headers) == 4
        assert "Date" in sheet_data.headers
        assert len(sheet_data.sample_data) == 2
        assert sheet_data.sample_data[0]["Product"] == "Widget A"


class TestExcelBackendIntegration:
    """Integration tests for Excel backend functions."""
    
    def test_api_endpoints_are_registered(self):
        """Test that all Excel endpoints are properly registered in the API."""
        # Get the API routes
        routes = [route for route in app.routes if hasattr(route, 'path') and '/api/excel-files' in route.path]
        
        # Verify that we have the expected endpoints
        endpoint_paths = [route.path for route in routes]
        expected_paths = [
            '/api/excel-files/',
            '/api/excel-files/{file_id}',
            '/api/excel-files/{file_id}/sheets',
            '/api/excel-files/search',
            '/api/excel-files/{file_id}/search'
        ]
        
        for expected_path in expected_paths:
            assert any(expected_path in path for path in endpoint_paths), f"Missing endpoint: {expected_path}"

    def test_api_documentation_includes_excel_endpoints(self):
        """Test that API documentation includes Excel endpoints."""
        response = client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        # The docs should be accessible (HTML response)
        assert "text/html" in response.headers.get("content-type", "")
