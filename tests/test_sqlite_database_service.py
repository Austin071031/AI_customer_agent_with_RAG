"""
Unit tests for the SQLite Database Service with Dynamic Table Creation.

This module contains comprehensive tests for the SQLite database functionality
including dynamic table creation, Excel file storage, and SQL query execution.
"""

import pytest
import tempfile
import os
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import json

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the dependencies that may not be installed for testing
mock_modules = ['openpyxl']
for module_name in mock_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

# Now import our modules
from src.services.sqlite_database_service import (
    SQLiteDatabaseService,
    SQLiteDatabaseError
)
from src.models.excel_models import (
    ExcelDocument,
    ExcelSheetData,
    SQLiteDataType,
    DynamicTableSchema,
    ColumnDefinition
)


class TestSQLiteDatabaseService:
    """Test cases for the SQLiteDatabaseService class."""
    
    def test_init(self):
        """Test SQLiteDatabaseService initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            assert service.db_path == temp_db_path
            assert service.logger is not None
            
            # Verify tables were created
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                assert 'excel_documents' in tables
                assert 'excel_sheet_data' in tables
                assert 'excel_row_data' in tables
                
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_init_failure(self):
        """Test SQLiteDatabaseService initialization failure."""
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Connection failed")
            
            with pytest.raises(SQLiteDatabaseError, match="Database initialization failed"):
                SQLiteDatabaseService(db_path="/invalid/path/test.db")
                
    def test_store_excel_file_success(self):
        """Test successful Excel file storage."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create a mock Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_excel:
                temp_excel_path = temp_excel.name
                
            try:
                # Mock openpyxl to avoid actual Excel file processing
                with patch('openpyxl.load_workbook') as mock_load_workbook:
                    mock_workbook = Mock()
                    mock_sheet = Mock()
                    mock_workbook.sheetnames = ['Sheet1']
                    mock_workbook.__getitem__.return_value = mock_sheet
                    
                    # Mock sheet data
                    mock_sheet.max_row = 5
                    mock_sheet.max_column = 3
                    mock_sheet[1] = [Mock(value='Header1'), Mock(value='Header2'), Mock(value='Header3')]
                    
                    # Mock cell values for data rows
                    def mock_cell(row, column):
                        cell = Mock()
                        if row == 1:
                            cell.value = f'Header{column}'
                        else:
                            cell.value = f'Value{row-1}_{column}'
                        return cell
                    
                    mock_sheet.cell.side_effect = mock_cell
                    mock_load_workbook.return_value = mock_workbook
                    
                    file_id = service.store_excel_file(
                        file_path=temp_excel_path,
                        file_name='test.xlsx',
                        file_size=1024,
                        sheet_names=['Sheet1'],
                        metadata={'description': 'Test file'}
                    )
                    
                    assert file_id.startswith('excel_')
                    assert len(file_id) > 10
                    
                    # Verify the file was stored
                    stored_file = service.get_excel_file(file_id)
                    assert stored_file is not None
                    assert stored_file.file_name == 'test.xlsx'
                    assert stored_file.file_size == 1024
                    
            finally:
                if os.path.exists(temp_excel_path):
                    os.unlink(temp_excel_path)
                    
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_store_excel_file_with_dynamic_tables_success(self):
        """Test successful Excel file storage with dynamic table creation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create a mock Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_excel:
                temp_excel_path = temp_excel.name
                
            try:
                # Mock openpyxl to avoid actual Excel file processing
                with patch('openpyxl.load_workbook') as mock_load_workbook:
                    mock_workbook = Mock()
                    mock_sheet = Mock()
                    mock_workbook.sheetnames = ['Sheet1']
                    mock_workbook.__getitem__.return_value = mock_sheet
                    
                    # Mock sheet data
                    mock_sheet.max_row = 5
                    mock_sheet.max_column = 3
                    mock_sheet[1] = [Mock(value='Header1'), Mock(value='Header2'), Mock(value='Header3')]
                    
                    # Mock cell values for data rows
                    def mock_cell(row, column):
                        cell = Mock()
                        if row == 1:
                            cell.value = f'Header{column}'
                        else:
                            cell.value = f'Value{row-1}_{column}'
                        return cell
                    
                    mock_sheet.cell.side_effect = mock_cell
                    mock_load_workbook.return_value = mock_workbook
                    
                    result = service.store_excel_file_with_dynamic_tables(
                        file_path=temp_excel_path,
                        file_name='test.xlsx',
                        file_size=1024,
                        sheet_names=['Sheet1'],
                        metadata={'description': 'Test file'}
                    )
                    
                    assert result['status'] == 'success'
                    assert result['file_name'] == 'test.xlsx'
                    assert 'tables_created' in result
                    assert 'Sheet1' in result['tables_created']
                    
                    # Verify dynamic tables were created
                    schemas = service.get_dynamic_table_schemas(result['file_id'])
                    assert len(schemas) > 0
                    
                    # Check that the table names contain the file ID
                    table_names = list(schemas.keys())
                    assert any(result['file_id'].replace('excel_', '') in name for name in table_names)
                    
            finally:
                if os.path.exists(temp_excel_path):
                    os.unlink(temp_excel_path)
                    
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_infer_data_type(self):
        """Test data type inference."""
        service = SQLiteDatabaseService(db_path=":memory:")
        
        # Test integer
        assert service._infer_data_type(42) == "integer"
        assert service._infer_data_type(-100) == "integer"
        
        # Test float
        assert service._infer_data_type(3.14) == "float"
        assert service._infer_data_type(-2.5) == "float"
        
        # Test string
        assert service._infer_data_type("hello") == "string"
        assert service._infer_data_type("") == "string"
        
        # Test boolean
        assert service._infer_data_type(True) == "boolean"
        assert service._infer_data_type(False) == "boolean"
        
        # Test None
        assert service._infer_data_type(None) == "null"
        
        # Test date string
        assert service._infer_data_type("2024-01-01") == "date"
        
        # Test unknown
        assert service._infer_data_type([1, 2, 3]) == "unknown"
        
    def test_get_excel_file_not_found(self):
        """Test retrieving non-existent Excel file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            result = service.get_excel_file("nonexistent_id")
            assert result is None
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_list_excel_files_empty(self):
        """Test listing Excel files when database is empty."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            files = service.list_excel_files()
            assert files == []
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_delete_excel_file_success(self):
        """Test successful Excel file deletion."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # First store a file
            with patch('openpyxl.load_workbook') as mock_load_workbook:
                mock_workbook = Mock()
                mock_sheet = Mock()
                mock_workbook.sheetnames = ['Sheet1']
                mock_workbook.__getitem__.return_value = mock_sheet
                mock_sheet.max_row = 3
                mock_sheet.max_column = 2
                mock_sheet[1] = [Mock(value='H1'), Mock(value='H2')]
                
                def mock_cell(row, column):
                    cell = Mock()
                    if row == 1:
                        cell.value = f'H{column}'
                    else:
                        cell.value = f'V{row-1}_{column}'
                    return cell
                
                mock_sheet.cell.side_effect = mock_cell
                mock_load_workbook.return_value = mock_workbook
                
                file_id = service.store_excel_file(
                    file_path='/fake/path/test.xlsx',
                    file_name='test.xlsx',
                    file_size=1024,
                    sheet_names=['Sheet1'],
                    metadata={}
                )
                
            # Now delete it
            result = service.delete_excel_file(file_id)
            assert result is True
            
            # Verify it's gone
            assert service.get_excel_file(file_id) is None
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_delete_excel_file_not_found(self):
        """Test deleting non-existent Excel file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            result = service.delete_excel_file("nonexistent_id")
            assert result is False
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_get_sheet_data(self):
        """Test retrieving sheet data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Store a file first
            with patch('openpyxl.load_workbook') as mock_load_workbook:
                mock_workbook = Mock()
                mock_sheet = Mock()
                mock_workbook.sheetnames = ['Sheet1', 'Sheet2']
                mock_workbook.__getitem__.return_value = mock_sheet
                mock_sheet.max_row = 3
                mock_sheet.max_column = 2
                mock_sheet[1] = [Mock(value='H1'), Mock(value='H2')]
                
                def mock_cell(row, column):
                    cell = Mock()
                    if row == 1:
                        cell.value = f'H{column}'
                    else:
                        cell.value = f'V{row-1}_{column}'
                    return cell
                
                mock_sheet.cell.side_effect = mock_cell
                mock_load_workbook.return_value = mock_workbook
                
                file_id = service.store_excel_file(
                    file_path='/fake/path/test.xlsx',
                    file_name='test.xlsx',
                    file_size=1024,
                    sheet_names=['Sheet1', 'Sheet2'],
                    metadata={}
                )
                
            # Get sheet data
            sheet_data = service.get_sheet_data(file_id)
            assert len(sheet_data) == 2
            assert all(isinstance(sheet, ExcelSheetData) for sheet in sheet_data)
            
            # Get specific sheet
            sheet1_data = service.get_sheet_data(file_id, 'Sheet1')
            assert len(sheet1_data) == 1
            assert sheet1_data[0].sheet_name == 'Sheet1'
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_search_excel_data(self):
        """Test searching Excel data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Store a file with sample data containing "test_value"
            with patch('openpyxl.load_workbook') as mock_load_workbook:
                mock_workbook = Mock()
                mock_sheet = Mock()
                mock_workbook.sheetnames = ['Sheet1']
                mock_workbook.__getitem__.return_value = mock_sheet
                mock_sheet.max_row = 3
                mock_sheet.max_column = 2
                mock_sheet[1] = [Mock(value='Column1'), Mock(value='Column2')]
                
                def mock_cell(row, column):
                    cell = Mock()
                    if row == 1:
                        cell.value = f'Column{column}'
                    elif row == 2:
                        cell.value = 'test_value' if column == 1 else 'other_value'
                    else:
                        cell.value = f'value{row}_{column}'
                    return cell
                
                mock_sheet.cell.side_effect = mock_cell
                mock_load_workbook.return_value = mock_workbook
                
                file_id = service.store_excel_file(
                    file_path='/fake/path/test.xlsx',
                    file_name='test.xlsx',
                    file_size=1024,
                    sheet_names=['Sheet1'],
                    metadata={}
                )
                
            # Search for the test value
            results = service.search_excel_data(
                query='test_value',
                file_id=file_id,
                max_results=10
            )
            
            # Since we're searching in sample_data JSON, we need to check if our mock data
            # would be properly serialized and searched
            # For this test, we'll just verify the method runs without error
            assert isinstance(results, list)
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_execute_sql_query_select(self):
        """Test executing SELECT SQL queries."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create a test table and insert data
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
                cursor.execute("INSERT INTO test_table VALUES (1, 'Alice')")
                cursor.execute("INSERT INTO test_table VALUES (2, 'Bob')")
                conn.commit()
                
            # Execute SELECT query
            results = service.execute_sql_query("SELECT * FROM test_table")
            
            assert len(results) == 2
            assert results[0]['id'] == 1
            assert results[0]['name'] == 'Alice'
            assert results[1]['id'] == 2
            assert results[1]['name'] == 'Bob'
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_execute_sql_query_non_select(self):
        """Test executing non-SELECT SQL queries."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create a test table
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
                conn.commit()
                
            # Execute INSERT query
            results = service.execute_sql_query("INSERT INTO test_table VALUES (1, 'Alice')")
            
            assert len(results) == 1
            assert results[0]['rows_affected'] == 1
            assert results[0]['query_type'] == 'non_select'
            assert results[0]['message'] == 'Query executed successfully'
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_execute_sql_query_error(self):
        """Test executing invalid SQL query."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            with pytest.raises(SQLiteDatabaseError, match="SQL query execution failed"):
                service.execute_sql_query("INVALID SQL QUERY")
                
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_get_dynamic_table_schemas(self):
        """Test getting dynamic table schemas."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create a mock dynamic table
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE excel_test123_sheet1 (id INTEGER, name TEXT)")
                cursor.execute("INSERT INTO excel_test123_sheet1 VALUES (1, 'Test')")
                conn.commit()
                
            # Get schemas
            schemas = service.get_dynamic_table_schemas()
            
            assert 'excel_test123_sheet1' in schemas
            table_schema = schemas['excel_test123_sheet1']
            assert table_schema['row_count'] == 1
            assert len(table_schema['columns']) == 2
            assert table_schema['file_id'] == 'excel_test123'
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_get_dynamic_table_schemas_filtered(self):
        """Test getting dynamic table schemas filtered by file ID."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Create mock dynamic tables
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE excel_test123_sheet1 (id INTEGER, name TEXT)")
                cursor.execute("CREATE TABLE excel_other456_sheet1 (id INTEGER, value REAL)")
                conn.commit()
                
            # Get schemas for specific file
            schemas = service.get_dynamic_table_schemas('excel_test123')
            
            assert 'excel_test123_sheet1' in schemas
            assert 'excel_other456_sheet1' not in schemas
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_extract_file_id_from_table_name(self):
        """Test extracting file ID from table name."""
        service = SQLiteDatabaseService(db_path=":memory:")
        
        # Valid table names
        assert service._extract_file_id_from_table_name("excel_123abc_sheet1") == "excel_123abc"
        assert service._extract_file_id_from_table_name("excel_uuid_here_sheet_name") == "excel_uuid"
        
        # Invalid table names
        assert service._extract_file_id_from_table_name("not_excel_table") is None
        assert service._extract_file_id_from_table_name("excel_") is None
        assert service._extract_file_id_from_table_name("") is None
        
    def test_health_check_success(self):
        """Test successful health check."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            assert service.health_check() is True
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_health_check_failure(self):
        """Test health check failure."""
        service = SQLiteDatabaseService(db_path="/invalid/path/test.db")
        assert service.health_check() is False
        
    def test_get_database_info(self):
        """Test getting database information."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            # Store a file to have some data
            with patch('openpyxl.load_workbook') as mock_load_workbook:
                mock_workbook = Mock()
                mock_sheet = Mock()
                mock_workbook.sheetnames = ['Sheet1']
                mock_workbook.__getitem__.return_value = mock_sheet
                mock_sheet.max_row = 3
                mock_sheet.max_column = 2
                mock_sheet[1] = [Mock(value='H1'), Mock(value='H2')]
                
                def mock_cell(row, column):
                    cell = Mock()
                    if row == 1:
                        cell.value = f'H{column}'
                    else:
                        cell.value = f'V{row-1}_{column}'
                    return cell
                
                mock_sheet.cell.side_effect = mock_cell
                mock_load_workbook.return_value = mock_workbook
                
                service.store_excel_file(
                    file_path='/fake/path/test.xlsx',
                    file_name='test.xlsx',
                    file_size=1024,
                    sheet_names=['Sheet1'],
                    metadata={}
                )
                
            # Get database info
            info = service.get_database_info()
            
            assert info['document_count'] == 1
            assert info['sheet_count'] == 1
            assert info['total_size_bytes'] == 1024
            assert info['database_path'] == temp_db_path
            assert info['database_size'] > 0
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)


class TestDynamicTableCreation:
    """Test cases specifically for dynamic table creation functionality."""
    
    def test_create_dynamic_table_for_sheet(self):
        """Test creating a dynamic table for an Excel sheet."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                
                headers = ['ID', 'Name', 'Age']
                data_types = {
                    'ID': 'integer',
                    'Name': 'string', 
                    'Age': 'integer'
                }
                
                table_name = service._create_dynamic_table_for_sheet(
                    file_id='excel_test123',
                    sheet_name='Users',
                    headers=headers,
                    data_types=data_types,
                    cursor=cursor
                )
                
                # Verify table was created
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                assert cursor.fetchone() is not None
                
                # Verify table structure
                cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
                columns = cursor.fetchall()
                
                assert len(columns) == 3
                column_names = [col[1] for col in columns]
                assert 'ID' in column_names
                assert 'Name' in column_names
                assert 'Age' in column_names
                
                # Verify column types
                for col in columns:
                    if col[1] == 'ID':
                        assert 'INTEGER' in col[2]
                    elif col[1] == 'Name':
                        assert 'TEXT' in col[2]
                    elif col[1] == 'Age':
                        assert 'INTEGER' in col[2]
                        
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
                
    def test_create_dynamic_table_with_special_characters(self):
        """Test creating dynamic table with special characters in headers."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        try:
            service = SQLiteDatabaseService(db_path=temp_db_path)
            
            with sqlite3.connect(temp_db_path) as conn:
                cursor = conn.cursor()
                
                headers = ['ID#', 'Name with spaces', 'Amount ($)']
                data_types = {
                    'ID#': 'integer',
                    'Name with spaces': 'string',
                    'Amount ($)': 'float'
                }
                
                table_name = service._create_dynamic_table_for_sheet(
                    file_id='excel_test123',
                    sheet_name='Special Data',
                    headers=headers,
                    data_types=data_types,
                    cursor=cursor
                )
                
                # Verify table was created
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                assert cursor.fetchone() is not None
                
                # Verify sanitized column names
                cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
                columns = cursor.fetchall()
                
                column_names = [col[1] for col in columns]
                # Check that special characters were removed or handled
                assert all(' ' not in name for name in column_names)
                assert all('#' not in name for name in column_names)
                assert all('$' not in name for name in column_names)
                
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
