"""
Excel file data models for the AI Customer Agent.

This module defines the data structures for handling Excel file documents
and sheet data using Pydantic for validation, specifically designed for
SQLite storage of Excel files as required by US-009.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ExcelDocument(BaseModel):
    """
    Represents an Excel file document stored in SQLite database.
    
    Attributes:
        id: Unique identifier for the Excel file
        file_name: Original name of the Excel file
        file_size: Size of the file in bytes
        sheet_names: List of sheet names in the Excel file
        metadata: Additional metadata about the file
        upload_time: When the file was uploaded (auto-generated)
        storage_type: Type of storage (always "sqlite" for Excel files)
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "excel_123e4567-e89b-12d3-a456-426614174000",
            "file_name": "sales_data.xlsx",
            "file_size": 10240,
            "sheet_names": ["Sales", "Customers", "Products"],
            "metadata": {
                "source": "company_reports",
                "uploaded_by": "admin",
                "description": "Monthly sales data"
            },
            "upload_time": "2024-01-01T12:00:00",
            "storage_type": "sqlite"
        }
    })
    
    id: str = Field(default_factory=lambda: f"excel_{str(uuid4())}")
    file_name: str
    file_size: int = Field(ge=0, description="File size in bytes")
    sheet_names: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    upload_time: datetime = Field(default_factory=datetime.now)
    storage_type: Literal["sqlite"] = "sqlite"
    
    @field_validator('file_name')
    @classmethod
    def file_name_must_have_extension(cls, v):
        """Validate that file name has a valid Excel extension."""
        valid_extensions = ('.xlsx', '.xls', '.xlsm', '.xlsb')
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f'File name must have a valid Excel extension: {valid_extensions}')
        return v
    
    @field_validator('sheet_names')
    @classmethod
    def sheet_names_must_not_be_empty(cls, v):
        """Validate that sheet names list is not empty."""
        if not v:
            raise ValueError('Excel file must contain at least one sheet')
        return v
    
    def get_file_info(self) -> str:
        """
        Get a summary of the Excel file information.
        
        Returns:
            String containing file name, size, and sheet count
        """
        size_kb = self.file_size / 1024
        return f"File: {self.file_name} | Size: {size_kb:.1f} KB | Sheets: {len(self.sheet_names)}"
    
    def __str__(self) -> str:
        """String representation of the Excel document."""
        return f"ExcelDocument(id={self.id}, file_name={self.file_name}, sheets={len(self.sheet_names)})"


class ExcelSheetData(BaseModel):
    """
    Represents data from a specific sheet in an Excel file.
    
    Attributes:
        file_id: Reference to the parent Excel document
        sheet_name: Name of the sheet
        headers: List of column headers
        row_count: Number of data rows (excluding header)
        column_count: Number of columns
        sample_data: Sample of the data (first few rows)
        data_types: Dictionary mapping column names to data types
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
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
    })
    
    file_id: str
    sheet_name: str
    headers: List[str]
    row_count: int = Field(ge=0, description="Number of data rows")
    column_count: int = Field(ge=0, description="Number of columns")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    data_types: Dict[str, str] = Field(default_factory=dict)
    
    @field_validator('file_id')
    @classmethod
    def file_id_must_start_with_excel(cls, v):
        """Validate that file_id follows the expected format."""
        if not v.startswith('excel_'):
            raise ValueError('file_id must start with "excel_" prefix')
        return v
    
    @field_validator('headers')
    @classmethod
    def headers_must_not_be_empty(cls, v):
        """Validate that headers list is not empty."""
        if not v:
            raise ValueError('Sheet must have at least one column header')
        return v
    
    @field_validator('sample_data')
    @classmethod
    def sample_data_must_match_headers(cls, v, info):
        """Validate that sample data keys match the headers."""
        if 'headers' in info.data and v:
            headers = set(info.data['headers'])
            for row in v:
                if not set(row.keys()).issubset(headers):
                    raise ValueError('Sample data keys must match the headers')
        return v
    
    def get_sheet_summary(self) -> str:
        """
        Get a summary of the sheet data.
        
        Returns:
            String containing sheet name, dimensions, and sample info
        """
        sample_size = len(self.sample_data)
        return f"Sheet: {self.sheet_name} | Rows: {self.row_count} | Columns: {self.column_count} | Sample: {sample_size} rows"
    
    def get_column_info(self) -> Dict[str, str]:
        """
        Get information about each column's data type.
        
        Returns:
            Dictionary mapping column names to data types
        """
        return self.data_types
    
    def __str__(self) -> str:
        """String representation of the Excel sheet data."""
        return f"ExcelSheetData(file_id={self.file_id}, sheet={self.sheet_name}, rows={self.row_count}, cols={self.column_count})"


class ExcelFileUpload(BaseModel):
    """
    Model for Excel file upload requests.
    
    Attributes:
        file_name: Name of the file being uploaded
        file_size: Size of the file in bytes
        description: Optional description of the file
        tags: Optional tags for categorization
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "file_name": "inventory_data.xlsx",
            "file_size": 20480,
            "description": "Current inventory levels and stock information",
            "tags": ["inventory", "stock", "products"]
        }
    })
    
    file_name: str
    file_size: int = Field(ge=1, description="File size must be at least 1 byte")
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('file_name')
    @classmethod
    def file_name_must_have_extension(cls, v):
        """Validate that file name has a valid Excel extension."""
        valid_extensions = ('.xlsx', '.xls', '.xlsm', '.xlsb')
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f'File name must have a valid Excel extension: {valid_extensions}')
        return v


class ExcelSearchQuery(BaseModel):
    """
    Model for searching Excel file data.
    
    Attributes:
        query: Search query string
        file_id: Optional specific file to search within
        sheet_name: Optional specific sheet to search within
        column_name: Optional specific column to search within
        case_sensitive: Whether search should be case sensitive
        max_results: Maximum number of results to return
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "sales revenue",
            "file_id": "excel_123e4567-e89b-12d3-a456-426614174000",
            "sheet_name": "Sales",
            "column_name": "Product",
            "case_sensitive": False,
            "max_results": 50
        }
    })
    
    query: str
    file_id: Optional[str] = None
    sheet_name: Optional[str] = None
    column_name: Optional[str] = None
    case_sensitive: bool = False
    max_results: int = Field(default=50, ge=1, le=1000)
    
    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v):
        """Validate that search query is not empty."""
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty')
        return v.strip()
