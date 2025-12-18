"""
Test script for Excel data models to verify they work correctly.
This script tests the ExcelDocument, ExcelSheetData, ExcelFileUpload, and ExcelSearchQuery models.
"""

import sys
import os

# Add the src directory to the path to import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.excel_models import ExcelDocument, ExcelSheetData, ExcelFileUpload, ExcelSearchQuery
from datetime import datetime


def test_excel_document():
    """Test ExcelDocument model creation and validation."""
    print("Testing ExcelDocument model...")
    
    # Valid data
    valid_data = {
        "file_name": "sales_data.xlsx",
        "file_size": 10240,
        "sheet_names": ["Sales", "Customers", "Products"],
        "metadata": {
            "source": "company_reports",
            "uploaded_by": "admin",
            "description": "Monthly sales data"
        }
    }
    
    try:
        doc = ExcelDocument(**valid_data)
        print(f"✓ ExcelDocument created successfully: {doc}")
        print(f"  File info: {doc.get_file_info()}")
        
        # Test validation
        assert doc.file_name == "sales_data.xlsx"
        assert doc.file_size == 10240
        assert len(doc.sheet_names) == 3
        assert doc.storage_type == "sqlite"
        assert "excel_" in doc.id
        print("✓ All validations passed")
        
    except Exception as e:
        print(f"✗ ExcelDocument creation failed: {e}")
        return False
    
    # Test invalid data
    print("\nTesting invalid ExcelDocument data...")
    
    # Invalid file extension
    try:
        invalid_data = valid_data.copy()
        invalid_data["file_name"] = "sales_data.txt"
        ExcelDocument(**invalid_data)
        print("✗ Should have failed with invalid file extension")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid file extension: {e}")
    
    # Empty sheet names
    try:
        invalid_data = valid_data.copy()
        invalid_data["sheet_names"] = []
        ExcelDocument(**invalid_data)
        print("✗ Should have failed with empty sheet names")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected empty sheet names: {e}")
    
    return True


def test_excel_sheet_data():
    """Test ExcelSheetData model creation and validation."""
    print("\nTesting ExcelSheetData model...")
    
    # Valid data
    valid_data = {
        "file_id": "excel_123e4567-e89b-12d3-a456-426614174000",
        "sheet_name": "Sales",
        "headers": ["Date", "Product", "Quantity", "Revenue"],
        "row_count": 150,
        "column_count": 4,
        "sample_data": [
            {"Date": "2024-01-01", "Product": "Widget A", "Quantity": 10, "Revenue": 1000.0},
            {"Date": "2024-01-02", "Product": "Widget B", "Quantity": 5, "Revenue": 750.0}
        ],
        "data_types": {
            "Date": "date",
            "Product": "string",
            "Quantity": "integer",
            "Revenue": "float"
        }
    }
    
    try:
        sheet_data = ExcelSheetData(**valid_data)
        print(f"✓ ExcelSheetData created successfully: {sheet_data}")
        print(f"  Sheet summary: {sheet_data.get_sheet_summary()}")
        print(f"  Column info: {sheet_data.get_column_info()}")
        
        # Test validation
        assert sheet_data.file_id == "excel_123e4567-e89b-12d3-a456-426614174000"
        assert sheet_data.sheet_name == "Sales"
        assert sheet_data.row_count == 150
        assert sheet_data.column_count == 4
        assert len(sheet_data.sample_data) == 2
        print("✓ All validations passed")
        
    except Exception as e:
        print(f"✗ ExcelSheetData creation failed: {e}")
        return False
    
    # Test invalid data
    print("\nTesting invalid ExcelSheetData data...")
    
    # Invalid file_id format
    try:
        invalid_data = valid_data.copy()
        invalid_data["file_id"] = "invalid_id"
        ExcelSheetData(**invalid_data)
        print("✗ Should have failed with invalid file_id format")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid file_id format: {e}")
    
    # Empty headers
    try:
        invalid_data = valid_data.copy()
        invalid_data["headers"] = []
        ExcelSheetData(**invalid_data)
        print("✗ Should have failed with empty headers")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected empty headers: {e}")
    
    # Sample data doesn't match headers
    try:
        invalid_data = valid_data.copy()
        invalid_data["sample_data"] = [{"InvalidColumn": "value"}]
        ExcelSheetData(**invalid_data)
        print("✗ Should have failed with mismatched sample data")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected mismatched sample data: {e}")
    
    return True


def test_excel_file_upload():
    """Test ExcelFileUpload model creation and validation."""
    print("\nTesting ExcelFileUpload model...")
    
    # Valid data
    valid_data = {
        "file_name": "inventory_data.xlsx",
        "file_size": 20480,
        "description": "Current inventory levels and stock information",
        "tags": ["inventory", "stock", "products"]
    }
    
    try:
        upload = ExcelFileUpload(**valid_data)
        print(f"✓ ExcelFileUpload created successfully: {upload}")
        
        # Test validation
        assert upload.file_name == "inventory_data.xlsx"
        assert upload.file_size == 20480
        assert upload.description == "Current inventory levels and stock information"
        assert upload.tags == ["inventory", "stock", "products"]
        print("✓ All validations passed")
        
    except Exception as e:
        print(f"✗ ExcelFileUpload creation failed: {e}")
        return False
    
    # Test invalid data
    print("\nTesting invalid ExcelFileUpload data...")
    
    # Invalid file extension
    try:
        invalid_data = valid_data.copy()
        invalid_data["file_name"] = "inventory_data.pdf"
        ExcelFileUpload(**invalid_data)
        print("✗ Should have failed with invalid file extension")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid file extension: {e}")
    
    # File size too small
    try:
        invalid_data = valid_data.copy()
        invalid_data["file_size"] = 0
        ExcelFileUpload(**invalid_data)
        print("✗ Should have failed with file size 0")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected file size 0: {e}")
    
    return True


def test_excel_search_query():
    """Test ExcelSearchQuery model creation and validation."""
    print("\nTesting ExcelSearchQuery model...")
    
    # Valid data
    valid_data = {
        "query": "sales revenue",
        "file_id": "excel_123e4567-e89b-12d3-a456-426614174000",
        "sheet_name": "Sales",
        "column_name": "Product",
        "case_sensitive": False,
        "max_results": 50
    }
    
    try:
        search_query = ExcelSearchQuery(**valid_data)
        print(f"✓ ExcelSearchQuery created successfully: {search_query}")
        
        # Test validation
        assert search_query.query == "sales revenue"
        assert search_query.file_id == "excel_123e4567-e89b-12d3-a456-426614174000"
        assert search_query.sheet_name == "Sales"
        assert search_query.column_name == "Product"
        assert search_query.case_sensitive == False
        assert search_query.max_results == 50
        print("✓ All validations passed")
        
    except Exception as e:
        print(f"✗ ExcelSearchQuery creation failed: {e}")
        return False
    
    # Test invalid data
    print("\nTesting invalid ExcelSearchQuery data...")
    
    # Empty query
    try:
        invalid_data = valid_data.copy()
        invalid_data["query"] = ""
        ExcelSearchQuery(**invalid_data)
        print("✗ Should have failed with empty query")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected empty query: {e}")
    
    # Whitespace-only query
    try:
        invalid_data = valid_data.copy()
        invalid_data["query"] = "   "
        ExcelSearchQuery(**invalid_data)
        print("✗ Should have failed with whitespace-only query")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected whitespace-only query: {e}")
    
    # Invalid max_results
    try:
        invalid_data = valid_data.copy()
        invalid_data["max_results"] = 0
        ExcelSearchQuery(**invalid_data)
        print("✗ Should have failed with max_results 0")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected max_results 0: {e}")
    
    return True


def main():
    """Run all tests."""
    print("Testing Excel data models...\n")
    
    all_passed = True
    
    # Run all test functions
    all_passed &= test_excel_document()
    all_passed &= test_excel_sheet_data()
    all_passed &= test_excel_file_upload()
    all_passed &= test_excel_search_query()
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed! Excel models are working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
