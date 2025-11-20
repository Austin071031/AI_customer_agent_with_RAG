#!/usr/bin/env python3
"""
Simple test runner for Excel file upload unit tests.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def run_basic_tests():
    """Run basic tests to verify the test suite works."""
    print("Running basic Excel model tests...")
    
    try:
        from tests.test_excel_file_upload import TestExcelModels
        test = TestExcelModels()
        
        # Test Excel document creation
        test.test_excel_document_creation()
        print("✓ Excel document creation test passed")
        
        # Test invalid file extension
        test.test_excel_document_invalid_file_extension()
        print("✓ Invalid file extension test passed")
        
        # Test empty sheet names
        test.test_excel_document_empty_sheet_names()
        print("✓ Empty sheet names test passed")
        
        # Test Excel sheet data creation
        test.test_excel_sheet_data_creation()
        print("✓ Excel sheet data creation test passed")
        
        # Test Excel file upload creation
        test.test_excel_file_upload_creation()
        print("✓ Excel file upload creation test passed")
        
        # Test Excel search query creation
        test.test_excel_search_query_creation()
        print("✓ Excel search query creation test passed")
        
        print("\n🎉 All basic Excel model tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_basic_tests()
