"""
SQLite Database Service for Excel File Storage.

This module provides the SQLite database service for storing Excel file metadata
and data as required by US-009. It handles structured storage of Excel files
with efficient querying capabilities.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import openpyxl
from datetime import datetime

from ..models.excel_models import ExcelDocument, ExcelSheetData


class SQLiteDatabaseError(Exception):
    """Custom exception for SQLite database related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class SQLiteDatabaseService:
    """
    Manages SQLite database operations for Excel file storage.
    
    This service handles storing Excel file metadata, sheet data, and provides
    search capabilities for structured Excel data.
    """
    
    def __init__(self, db_path: str = "./excel_database.db"):
        """
        Initialize the SQLite database service.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self) -> None:
        """
        Initialize SQLite database with required tables.
        
        Creates tables for Excel documents and sheet data if they don't exist.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create Excel documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS excel_documents (
                        id TEXT PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        sheet_names TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        upload_time TIMESTAMP NOT NULL,
                        storage_type TEXT DEFAULT 'sqlite'
                    )
                """)
                
                # Create Excel sheet data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS excel_sheet_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT NOT NULL,
                        sheet_name TEXT NOT NULL,
                        headers TEXT NOT NULL,
                        row_count INTEGER NOT NULL,
                        column_count INTEGER NOT NULL,
                        sample_data TEXT NOT NULL,
                        data_types TEXT NOT NULL,
                        FOREIGN KEY (file_id) REFERENCES excel_documents (id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON excel_sheet_data(file_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sheet_name ON excel_sheet_data(sheet_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_time ON excel_documents(upload_time)")
                
                conn.commit()
                self.logger.info(f"SQLite database initialized at: {self.db_path}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize SQLite database: {str(e)}")
            raise SQLiteDatabaseError(f"Database initialization failed: {str(e)}")
            
    def store_excel_file(self, file_path: str, file_name: str, file_size: int,
                        sheet_names: List[str], metadata: Dict[str, Any]) -> str:
        """
        Store Excel file metadata and content in SQLite database.
        
        Args:
            file_path: Path to the Excel file
            file_name: Name of the Excel file
            file_size: Size of the file in bytes
            sheet_names: List of sheet names in the Excel file
            metadata: Additional metadata about the file
            
        Returns:
            File ID of the stored Excel document
            
        Raises:
            SQLiteDatabaseError: If storage operation fails
        """
        try:
            # Create ExcelDocument object for validation
            excel_doc = ExcelDocument(
                file_name=file_name,
                file_size=file_size,
                sheet_names=sheet_names,
                metadata=metadata
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert Excel document
                cursor.execute("""
                    INSERT INTO excel_documents 
                    (id, file_name, file_size, sheet_names, metadata, upload_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    excel_doc.id,
                    excel_doc.file_name,
                    excel_doc.file_size,
                    json.dumps(excel_doc.sheet_names),
                    json.dumps(excel_doc.metadata),
                    excel_doc.upload_time.isoformat()
                ))
                
                # Process and store sheet data
                self._process_and_store_sheet_data(file_path, excel_doc.id, cursor)
                
                conn.commit()
                self.logger.info(f"Stored Excel file: {file_name} with ID: {excel_doc.id}")
                return excel_doc.id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store Excel file: {str(e)}")
            raise SQLiteDatabaseError(f"Excel file storage failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error storing Excel file: {str(e)}")
            raise SQLiteDatabaseError(f"Unexpected error: {str(e)}")
            
    def _process_and_store_sheet_data(self, file_path: str, file_id: str, cursor: sqlite3.Cursor) -> None:
        """
        Process Excel file and store sheet data in database.
        
        Args:
            file_path: Path to the Excel file
            file_id: ID of the Excel document
            cursor: Database cursor for executing queries
        """
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                
                # Extract headers (first row)
                headers = []
                for cell in sheet[1]:
                    headers.append(str(cell.value) if cell.value else f"Column_{cell.column}")
                
                # Count rows and columns
                row_count = sheet.max_row - 1  # Exclude header
                column_count = sheet.max_column
                
                # Extract sample data (first 5 rows after header)
                sample_data = []
                data_types = {}
                
                for row_idx in range(2, min(7, sheet.max_row + 1)):
                    row_data = {}
                    for col_idx, header in enumerate(headers, 1):
                        cell_value = sheet.cell(row=row_idx, column=col_idx).value
                        row_data[header] = cell_value
                        
                        # Infer data type for this column
                        if header not in data_types:
                            data_types[header] = self._infer_data_type(cell_value)
                    
                    sample_data.append(row_data)
                
                # Create ExcelSheetData object for validation
                sheet_data = ExcelSheetData(
                    file_id=file_id,
                    sheet_name=sheet_name,
                    headers=headers,
                    row_count=row_count,
                    column_count=column_count,
                    sample_data=sample_data,
                    data_types=data_types
                )
                
                # Insert sheet data
                cursor.execute("""
                    INSERT INTO excel_sheet_data 
                    (file_id, sheet_name, headers, row_count, column_count, sample_data, data_types)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sheet_data.file_id,
                    sheet_data.sheet_name,
                    json.dumps(sheet_data.headers),
                    sheet_data.row_count,
                    sheet_data.column_count,
                    json.dumps(sheet_data.sample_data),
                    json.dumps(sheet_data.data_types)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to process sheet data: {str(e)}")
            raise SQLiteDatabaseError(f"Sheet data processing failed: {str(e)}")
            
    def _infer_data_type(self, value: Any) -> str:
        """
        Infer data type from cell value.
        
        Args:
            value: Cell value from Excel
            
        Returns:
            Inferred data type as string
        """
        if value is None:
            return "null"
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                return "integer"
            else:
                return "float"
        elif isinstance(value, str):
            # Check if it's a date string
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return "date"
            except (ValueError, TypeError):
                return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, datetime):
            return "datetime"
        else:
            return "unknown"
            
    def get_excel_file(self, file_id: str) -> Optional[ExcelDocument]:
        """
        Retrieve Excel file data from database.
        
        Args:
            file_id: ID of the Excel file to retrieve
            
        Returns:
            ExcelDocument object if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_name, file_size, sheet_names, metadata, upload_time, storage_type
                    FROM excel_documents 
                    WHERE id = ?
                """, (file_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                return ExcelDocument(
                    id=row[0],
                    file_name=row[1],
                    file_size=row[2],
                    sheet_names=json.loads(row[3]),
                    metadata=json.loads(row[4]),
                    upload_time=datetime.fromisoformat(row[5]),
                    storage_type=row[6]
                )
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve Excel file: {str(e)}")
            raise SQLiteDatabaseError(f"Excel file retrieval failed: {str(e)}")
            
    def list_excel_files(self) -> List[ExcelDocument]:
        """
        List all Excel files in database.
        
        Returns:
            List of ExcelDocument objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_name, file_size, sheet_names, metadata, upload_time, storage_type
                    FROM excel_documents 
                    ORDER BY upload_time DESC
                """)
                
                excel_files = []
                for row in cursor.fetchall():
                    excel_files.append(ExcelDocument(
                        id=row[0],
                        file_name=row[1],
                        file_size=row[2],
                        sheet_names=json.loads(row[3]),
                        metadata=json.loads(row[4]),
                        upload_time=datetime.fromisoformat(row[5]),
                        storage_type=row[6]
                    ))
                    
                return excel_files
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to list Excel files: {str(e)}")
            raise SQLiteDatabaseError(f"Excel file listing failed: {str(e)}")
            
    def delete_excel_file(self, file_id: str) -> bool:
        """
        Delete Excel file from database.
        
        Args:
            file_id: ID of the Excel file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM excel_documents WHERE id = ?", (file_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.info(f"Deleted Excel file with ID: {file_id}")
                else:
                    self.logger.warning(f"No Excel file found with ID: {file_id}")
                    
                return deleted
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete Excel file: {str(e)}")
            raise SQLiteDatabaseError(f"Excel file deletion failed: {str(e)}")
            
    def get_sheet_data(self, file_id: str, sheet_name: Optional[str] = None) -> List[ExcelSheetData]:
        """
        Get sheet data for an Excel file.
        
        Args:
            file_id: ID of the Excel file
            sheet_name: Optional specific sheet name
            
        Returns:
            List of ExcelSheetData objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if sheet_name:
                    cursor.execute("""
                        SELECT file_id, sheet_name, headers, row_count, column_count, sample_data, data_types
                        FROM excel_sheet_data 
                        WHERE file_id = ? AND sheet_name = ?
                    """, (file_id, sheet_name))
                else:
                    cursor.execute("""
                        SELECT file_id, sheet_name, headers, row_count, column_count, sample_data, data_types
                        FROM excel_sheet_data 
                        WHERE file_id = ?
                    """, (file_id,))
                
                sheet_data_list = []
                for row in cursor.fetchall():
                    sheet_data_list.append(ExcelSheetData(
                        file_id=row[0],
                        sheet_name=row[1],
                        headers=json.loads(row[2]),
                        row_count=row[3],
                        column_count=row[4],
                        sample_data=json.loads(row[5]),
                        data_types=json.loads(row[6])
                    ))
                    
                return sheet_data_list
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get sheet data: {str(e)}")
            raise SQLiteDatabaseError(f"Sheet data retrieval failed: {str(e)}")
            
    def search_excel_data(self, query: str, file_id: Optional[str] = None, 
                         sheet_name: Optional[str] = None, column_name: Optional[str] = None,
                         case_sensitive: bool = False, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for data within Excel files.
        
        Args:
            query: Search query string
            file_id: Optional specific file to search within
            sheet_name: Optional specific sheet to search within
            column_name: Optional specific column to search within
            case_sensitive: Whether search should be case sensitive
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with file and location information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the search query
                base_query = """
                    SELECT ed.file_name, esd.sheet_name, esd.sample_data
                    FROM excel_sheet_data esd
                    JOIN excel_documents ed ON esd.file_id = ed.id
                    WHERE esd.sample_data LIKE ?
                """
                
                params = [f"%{query}%"]
                
                # Add filters
                if file_id:
                    base_query += " AND esd.file_id = ?"
                    params.append(file_id)
                    
                if sheet_name:
                    base_query += " AND esd.sheet_name = ?"
                    params.append(sheet_name)
                
                # Add limit
                base_query += " LIMIT ?"
                params.append(max_results)
                
                cursor.execute(base_query, params)
                
                results = []
                for row in cursor.fetchall():
                    file_name, sheet_name, sample_data_json = row
                    sample_data = json.loads(sample_data_json)
                    
                    # Search within sample data
                    for row_data in sample_data:
                        for col_name, cell_value in row_data.items():
                            if column_name and col_name != column_name:
                                continue
                                
                            if cell_value and query.lower() in str(cell_value).lower():
                                results.append({
                                    'file_name': file_name,
                                    'sheet_name': sheet_name,
                                    'column_name': col_name,
                                    'cell_value': cell_value,
                                    'file_id': file_id
                                })
                                if len(results) >= max_results:
                                    return results
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to search Excel data: {str(e)}")
            raise SQLiteDatabaseError(f"Excel data search failed: {str(e)}")
            
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the SQLite database.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get document count
                cursor.execute("SELECT COUNT(*) FROM excel_documents")
                doc_count = cursor.fetchone()[0]
                
                # Get sheet count
                cursor.execute("SELECT COUNT(*) FROM excel_sheet_data")
                sheet_count = cursor.fetchone()[0]
                
                # Get total file size
                cursor.execute("SELECT SUM(file_size) FROM excel_documents")
                total_size = cursor.fetchone()[0] or 0
                
                return {
                    'document_count': doc_count,
                    'sheet_count': sheet_count,
                    'total_size_bytes': total_size,
                    'database_path': self.db_path,
                    'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get database info: {str(e)}")
            return {'error': str(e)}
            
    def health_check(self) -> bool:
        """
        Perform a health check on the SQLite database.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except sqlite3.Error:
            return False
