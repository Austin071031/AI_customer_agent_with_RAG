"""
Excel file data models for the AI Customer Agent with Dynamic Table Support.

This module defines the data structures for handling Excel file documents
and sheet data using Pydantic for validation, specifically designed for
SQLite storage of Excel files with dynamic table creation as required by US-009.
The models support relational table schemas with proper column types for
efficient SQL query execution and Text-to-SQL operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SQLiteDataType(str, Enum):
    """
    Enumeration of SQLite data types for dynamic table column definitions.
    
    These types correspond to the five storage classes in SQLite:
    - INTEGER: Signed integers, stored in 1, 2, 3, 4, 6, or 8 bytes
    - REAL: Floating point values, stored as 8-byte IEEE floating point numbers
    - TEXT: Text strings, stored using the database encoding (UTF-8, UTF-16BE or UTF-16LE)
    - BLOB: Binary data, stored exactly as it was input
    - NULL: Null value
    
    Note: SQLite uses dynamic typing, but these types provide hints for storage optimization.
    """
    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BLOB = "BLOB"
    NULL = "NULL"


class ColumnDefinition(BaseModel):
    """
    Defines a column in a dynamic SQLite table for Excel sheet storage.
    
    Attributes:
        name: Name of the column (derived from Excel header)
        sqlite_type: SQLite data type for the column
        is_primary_key: Whether this column is a primary key
        is_nullable: Whether the column allows NULL values
        default_value: Default value for the column
        description: Optional description of the column
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Revenue",
            "sqlite_type": "REAL",
            "is_primary_key": False,
            "is_nullable": True,
            "default_value": "0.0",
            "description": "Sales revenue in dollars"
        }
    })
    
    name: str
    sqlite_type: SQLiteDataType
    is_primary_key: bool = False
    is_nullable: bool = True
    default_value: Optional[str] = None
    description: Optional[str] = None
    
    @field_validator('name')
    @classmethod
    def name_must_be_valid_sql_identifier(cls, v):
        """Validate that column name is a valid SQL identifier."""
        # Basic SQL identifier validation
        if not v or not v.strip():
            raise ValueError('Column name cannot be empty')
        if len(v) > 63:  # SQLite identifier limit
            raise ValueError('Column name must be 63 characters or less')
        # Check for SQL keywords or invalid characters (simplified)
        invalid_chars = ['"', "'", ";", ",", "(", ")", "[", "]", "{", "}", "\\"]
        if any(char in v for char in invalid_chars):
            raise ValueError('Column name contains invalid characters')
        return v.strip()
    
    def get_sql_definition(self) -> str:
        """
        Generate SQL column definition for table creation.
        
        Returns:
            SQL column definition string
        """
        parts = [f'"{self.name}" {self.sqlite_type.value}']
        
        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        
        if not self.is_nullable:
            parts.append("NOT NULL")
        
        if self.default_value is not None:
            parts.append(f"DEFAULT {self.default_value}")
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        """String representation of column definition."""
        return f"ColumnDefinition(name={self.name}, type={self.sqlite_type.value})"


class DynamicTableSchema(BaseModel):
    """
    Represents the schema for a dynamic SQLite table created from an Excel sheet.
    
    Attributes:
        table_name: Name of the dynamic table (sanitized sheet name)
        file_id: Reference to the parent Excel document
        sheet_name: Original sheet name from Excel file
        columns: List of column definitions for the table
        row_count: Number of rows in the table
        created_at: When the table was created
        primary_key: Optional primary key column name
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "table_name": "sales_data_sheet1",
            "file_id": "excel_123e4567-e89b-12d3-a456-426614174000",
            "sheet_name": "Sales Data",
            "columns": [
                {
                    "name": "Date",
                    "sqlite_type": "TEXT",
                    "is_primary_key": False,
                    "is_nullable": True,
                    "description": "Transaction date"
                },
                {
                    "name": "Revenue",
                    "sqlite_type": "REAL",
                    "is_primary_key": False,
                    "is_nullable": True,
                    "default_value": "0.0",
                    "description": "Sales amount"
                }
            ],
            "row_count": 150,
            "created_at": "2024-01-01T12:00:00",
            "primary_key": None
        }
    })
    
    table_name: str
    file_id: str
    sheet_name: str
    columns: List[ColumnDefinition]
    row_count: int = Field(ge=0, description="Number of rows in the table")
    created_at: datetime = Field(default_factory=datetime.now)
    primary_key: Optional[str] = None
    
    @field_validator('table_name')
    @classmethod
    def table_name_must_be_valid(cls, v):
        """Validate that table name is a valid SQL identifier."""
        if not v or not v.strip():
            raise ValueError('Table name cannot be empty')
        if len(v) > 63:  # SQLite identifier limit
            raise ValueError('Table name must be 63 characters or less')
        # Remove any potentially dangerous characters
        v = "".join(c for c in v if c.isalnum() or c in ['_'])
        if not v:
            raise ValueError('Table name must contain valid characters')
        return v
    
    @field_validator('file_id')
    @classmethod
    def file_id_must_start_with_excel(cls, v):
        """Validate that file_id follows the expected format."""
        if not v.startswith('excel_'):
            raise ValueError('file_id must start with "excel_" prefix')
        return v
    
    @field_validator('columns')
    @classmethod
    def columns_must_not_be_empty(cls, v):
        """Validate that columns list is not empty."""
        if not v:
            raise ValueError('Table must have at least one column')
        return v
    
    def get_create_table_sql(self) -> str:
        """
        Generate CREATE TABLE SQL statement for this schema.
        
        Returns:
            SQL CREATE TABLE statement
        """
        column_defs = [col.get_sql_definition() for col in self.columns]
        
        # Add primary key constraint if specified
        if self.primary_key:
            pk_column = next((col for col in self.columns if col.name == self.primary_key), None)
            if pk_column:
                column_defs.append(f"PRIMARY KEY ({self.primary_key})")
        
        columns_sql = ", ".join(column_defs)
        return f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({columns_sql})'
    
    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]
    
    def get_column_by_name(self, column_name: str) -> Optional[ColumnDefinition]:
        """Get column definition by name."""
        return next((col for col in self.columns if col.name == column_name), None)
    
    def __str__(self) -> str:
        """String representation of dynamic table schema."""
        return f"DynamicTableSchema(table={self.table_name}, columns={len(self.columns)}, rows={self.row_count})"


class TableCreationResult(BaseModel):
    """
    Represents the result of creating a dynamic table from an Excel sheet.
    
    Attributes:
        success: Whether table creation was successful
        table_name: Name of the created table
        rows_inserted: Number of rows successfully inserted
        error_message: Error message if creation failed
        warnings: List of warnings during table creation
        execution_time: Time taken to create the table (seconds)
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "table_name": "sales_data_sheet1",
            "rows_inserted": 150,
            "error_message": None,
            "warnings": ["Some empty cells found in column 'Notes'"],
            "execution_time": 2.5
        }
    })
    
    success: bool
    table_name: str
    rows_inserted: int = Field(ge=0, description="Number of rows inserted")
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    execution_time: float = Field(ge=0.0, description="Execution time in seconds")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def __str__(self) -> str:
        """String representation of table creation result."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"TableCreationResult(table={self.table_name}, status={status}, rows={self.rows_inserted})"


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
    Represents data from a specific sheet in an Excel file with enhanced dynamic table support.
    
    This model now integrates with SQLite data types and can generate dynamic table schemas
    for relational database storage, supporting Text-to-SQL operations.
    
    Attributes:
        file_id: Reference to the parent Excel document
        sheet_name: Name of the sheet
        headers: List of column headers
        row_count: Number of data rows (excluding header)
        column_count: Number of columns
        sample_data: Sample of the data (first few rows)
        data_types: Dictionary mapping column names to SQLite data types
        table_schema: Optional dynamic table schema for SQLite storage
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
                "Date": "TEXT",
                "Product": "TEXT", 
                "Quantity": "INTEGER",
                "Revenue": "REAL"
            },
            "table_schema": None
        }
    })
    
    file_id: str
    sheet_name: str
    headers: List[str]
    row_count: int = Field(ge=0, description="Number of data rows")
    column_count: int = Field(ge=0, description="Number of columns")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    data_types: Dict[str, SQLiteDataType] = Field(default_factory=dict)
    table_schema: Optional[DynamicTableSchema] = None
    
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
    
    @field_validator('data_types')
    @classmethod
    def data_types_must_match_headers(cls, v, info):
        """Validate that data types keys match the headers."""
        if 'headers' in info.data and v:
            headers = set(info.data['headers'])
            data_type_keys = set(v.keys())
            if not data_type_keys.issubset(headers):
                raise ValueError('Data types must only contain keys that are in headers')
        return v
    
    def get_sheet_summary(self) -> str:
        """
        Get a summary of the sheet data.
        
        Returns:
            String containing sheet name, dimensions, and sample info
        """
        sample_size = len(self.sample_data)
        return f"Sheet: {self.sheet_name} | Rows: {self.row_count} | Columns: {self.column_count} | Sample: {sample_size} rows"
    
    def get_column_info(self) -> Dict[str, SQLiteDataType]:
        """
        Get information about each column's data type.
        
        Returns:
            Dictionary mapping column names to SQLite data types
        """
        return self.data_types
    
    def generate_table_schema(self, table_name: Optional[str] = None) -> DynamicTableSchema:
        """
        Generate a dynamic table schema from the Excel sheet data.
        
        Args:
            table_name: Optional custom table name (default: sanitized sheet name)
            
        Returns:
            DynamicTableSchema object ready for SQLite table creation
        """
        if not table_name:
            # Generate table name from sheet name - preserve underscores
            table_name = "".join(c for c in self.sheet_name if c.isalnum() or c == '_').lower()
            if not table_name:
                table_name = f"sheet_{self.file_id[-8:]}"
        
        # Create column definitions from headers and data types
        columns = []
        for header in self.headers:
            sqlite_type = self.data_types.get(header, SQLiteDataType.TEXT)
            column_def = ColumnDefinition(
                name=header,
                sqlite_type=sqlite_type,
                is_primary_key=False,
                is_nullable=True,
                description=f"Column from Excel sheet: {self.sheet_name}"
            )
            columns.append(column_def)
        
        # Create the dynamic table schema
        schema = DynamicTableSchema(
            table_name=table_name,
            file_id=self.file_id,
            sheet_name=self.sheet_name,
            columns=columns,
            row_count=self.row_count
        )
        
        # Store the schema in the instance
        self.table_schema = schema
        return schema
    
    def get_sqlite_column_types(self) -> Dict[str, str]:
        """
        Get SQLite column types as strings for database operations.
        
        Returns:
            Dictionary mapping column names to SQLite type strings
        """
        return {col: self.data_types.get(col, SQLiteDataType.TEXT).value for col in self.headers}
    
    def validate_for_sqlite_storage(self) -> List[str]:
        """
        Validate the sheet data for SQLite storage compatibility.
        
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        
        # Check for duplicate column names (case-insensitive)
        lower_headers = [h.lower() for h in self.headers]
        if len(lower_headers) != len(set(lower_headers)):
            warnings.append("Duplicate column names detected (case-insensitive)")
        
        # Check for SQL keywords in column names
        sql_keywords = ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "TABLE"]
        for header in self.headers:
            if header.upper() in sql_keywords:
                warnings.append(f"Column name '{header}' is a SQL keyword")
        
        # Check data type consistency
        for header, data_type in self.data_types.items():
            if header not in self.headers:
                warnings.append(f"Data type defined for non-existent column: {header}")
        
        return warnings
    
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
