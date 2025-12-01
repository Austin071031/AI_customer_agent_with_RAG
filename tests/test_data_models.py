"""
Unit tests for the core data models with dynamic table support.

This module contains comprehensive tests for all data models in the
AI Customer Agent application, including chat models, configuration models,
and Excel models with dynamic table support.
"""

import pytest
from datetime import datetime
from uuid import UUID
from typing import Dict, Any, List

from src.models.chat_models import (
    ChatMessage, EnhancedChatMessage, ConversationContext, QueryResult,
    QueryIntent, DataSource, KBDocument
)
from src.models.config_models import APIConfig, DatabaseConfig, AppConfig
from src.models.excel_models import (
    SQLiteDataType, ColumnDefinition, DynamicTableSchema, TableCreationResult,
    ExcelDocument, ExcelSheetData, ExcelFileUpload, ExcelSearchQuery
)


class TestChatModels:
    """Test cases for chat-related data models."""
    
    def test_chat_message_creation(self):
        """Test basic ChatMessage creation and validation."""
        message = ChatMessage(
            role="user",
            content="Hello, how can you help me?"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how can you help me?"
        assert isinstance(message.timestamp, datetime)
        assert isinstance(UUID(message.message_id), UUID)
    
    def test_chat_message_validation(self):
        """Test ChatMessage validation rules."""
        # Test empty content validation
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(role="user", content="")
        
        # Test whitespace content validation
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatMessage(role="user", content="   ")
    
    def test_enhanced_chat_message_creation(self):
        """Test EnhancedChatMessage creation with additional context."""
        message = EnhancedChatMessage(
            role="user",
            content="What were the total sales last month?",
            query_intent=QueryIntent.EXCEL_DATA,
            confidence_score=0.95,
            processing_time=2.3
        )
        
        assert message.role == "user"
        assert message.query_intent == QueryIntent.EXCEL_DATA
        assert message.confidence_score == 0.95
        assert message.processing_time == 2.3
        assert message.data_sources == []
        assert message.sql_queries == []
        assert message.excel_file_references == []
    
    def test_enhanced_chat_message_methods(self):
        """Test EnhancedChatMessage helper methods."""
        message = EnhancedChatMessage(
            role="user",
            content="Show me sales data"
        )
        
        # Test adding data sources
        message.add_data_source(DataSource.EXCEL_DATA)
        message.add_data_source(DataSource.TEXT_TO_SQL)
        assert DataSource.EXCEL_DATA in message.data_sources
        assert DataSource.TEXT_TO_SQL in message.data_sources
        
        # Test adding SQL queries
        message.add_sql_query("SELECT * FROM sales_data")
        assert "SELECT * FROM sales_data" in message.sql_queries
        
        # Test adding Excel references
        message.add_excel_reference("excel_123")
        assert "excel_123" in message.excel_file_references
        
        # Test processing summary
        summary = message.get_processing_summary()
        assert "excel_data" in summary
        assert "text_to_sql" in summary
    
    def test_enhanced_chat_message_validation(self):
        """Test EnhancedChatMessage validation rules."""
        # Test invalid Excel reference format
        with pytest.raises(ValueError, match='Excel file references must start with "excel_" prefix'):
            EnhancedChatMessage(
                role="user",
                content="test",
                excel_file_references=["invalid_ref"]
            )
    
    def test_conversation_context_creation(self):
        """Test ConversationContext creation and basic operations."""
        context = ConversationContext()
        
        assert isinstance(context.conversation_id, str)
        assert context.conversation_id.startswith("conv_")
        assert context.messages == []
        assert context.current_intent == QueryIntent.UNKNOWN
        assert context.excel_files_mentioned == []
        assert context.active_table_schemas == []
        assert isinstance(context.conversation_start_time, datetime)
        assert isinstance(context.last_activity_time, datetime)
    
    def test_conversation_context_message_management(self):
        """Test ConversationContext message management methods."""
        context = ConversationContext()
        
        # Create test messages
        message1 = EnhancedChatMessage(
            role="user",
            content="Show sales data",
            query_intent=QueryIntent.EXCEL_DATA
        )
        message1.add_excel_reference("excel_123")
        
        message2 = EnhancedChatMessage(
            role="assistant",
            content="Here are the sales figures...",
            query_intent=QueryIntent.EXCEL_DATA
        )
        
        # Add messages to context
        context.add_message(message1)
        context.add_message(message2)
        
        # Verify context was updated
        assert len(context.messages) == 2
        assert context.current_intent == QueryIntent.EXCEL_DATA
        assert "excel_123" in context.excel_files_mentioned
        assert context.last_activity_time >= context.conversation_start_time
    
    def test_conversation_context_methods(self):
        """Test ConversationContext helper methods."""
        context = ConversationContext()
        
        # Add some messages
        for i in range(15):
            message = EnhancedChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
            context.add_message(message)
        
        # Test getting recent messages
        recent = context.get_recent_messages(5)
        assert len(recent) == 5
        
        # Test conversation summary
        summary = context.get_conversation_summary()
        assert "Messages: 15" in summary
        
        # Test clearing conversation
        context.clear_conversation()
        assert len(context.messages) == 0
        assert isinstance(context.last_activity_time, datetime)
    
    def test_query_result_creation(self):
        """Test QueryResult creation and validation."""
        result_data = [
            {"Product": "A", "Sales": 100},
            {"Product": "B", "Sales": 200}
        ]
        
        result = QueryResult(
            success=True,
            data=result_data,
            data_source=DataSource.EXCEL_DATA,
            query="Show top products",
            execution_time=1.5,
            confidence=0.9,
            metadata={"rows_returned": 2}
        )
        
        assert result.success is True
        assert result.data == result_data
        assert result.data_source == DataSource.EXCEL_DATA
        assert result.query == "Show top products"
        assert result.execution_time == 1.5
        assert result.confidence == 0.9
        assert result.metadata["rows_returned"] == 2
        assert result.error_message is None
    
    def test_query_result_methods(self):
        """Test QueryResult helper methods."""
        result = QueryResult(
            success=True,
            data=[{"test": "data"}],
            data_source=DataSource.EXCEL_DATA,
            query="test query",
            execution_time=1.0
        )
        
        # Test source checking methods
        assert result.is_from_excel() is True
        assert result.is_from_knowledge_base() is False
        
        # Test result summary
        summary = result.get_result_summary()
        assert "SUCCESS" in summary
        assert "excel_data" in summary
    
    def test_kb_document_creation(self):
        """Test KBDocument creation and methods."""
        document = KBDocument(
            id="doc_123",
            content="This is sample document content.",
            metadata={
                "source": "company_handbook",
                "author": "HR"
            },
            file_path="/documents/handbook.pdf",
            file_type="pdf"
        )
        
        assert document.id == "doc_123"
        assert document.content == "This is sample document content."
        assert document.metadata["source"] == "company_handbook"
        assert document.file_path == "/documents/handbook.pdf"
        assert document.file_type == "pdf"
        assert document.embedding is None
        
        # Test metadata summary
        summary = document.get_metadata_summary()
        assert "handbook.pdf" in summary
        assert "pdf" in summary


class TestConfigModels:
    """Test cases for configuration data models."""
    
    def test_api_config_creation(self):
        """Test APIConfig creation and validation."""
        config = APIConfig(
            api_key="sk-test123",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        
        assert config.api_key == "sk-test123"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
    
    def test_api_config_validation(self):
        """Test APIConfig validation rules."""
        # Test empty API key validation
        with pytest.raises(ValueError, match="API key cannot be empty"):
            APIConfig(api_key="")
        
        # Test invalid base URL validation
        with pytest.raises(ValueError, match="Base URL must start with http:// or https://"):
            APIConfig(api_key="test", base_url="invalid-url")
        
        # Test temperature bounds
        with pytest.raises(ValueError):
            APIConfig(api_key="test", temperature=1.5)
        
        # Test max_tokens bounds
        with pytest.raises(ValueError):
            APIConfig(api_key="test", max_tokens=5000)
    
    def test_api_config_methods(self):
        """Test APIConfig helper methods."""
        config = APIConfig(api_key="sk-test123")
        
        # Test headers generation
        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer sk-test123"
        assert headers["Content-Type"] == "application/json"
        
        # Test string representation (should hide API key)
        str_repr = str(config)
        assert "sk-test123" not in str_repr
    
    def test_database_config_creation(self):
        """Test DatabaseConfig creation and validation."""
        config = DatabaseConfig(
            persist_directory="./knowledge_base",
            collection_name="documents"
        )
        
        assert config.persist_directory == "./knowledge_base"
        assert config.collection_name == "documents"
    
    def test_database_config_validation(self):
        """Test DatabaseConfig validation rules."""
        # Test empty persist directory validation
        with pytest.raises(ValueError, match="Persist directory cannot be empty"):
            DatabaseConfig(persist_directory="")
    
    def test_app_config_creation(self):
        """Test AppConfig creation and validation."""
        api_config = APIConfig(api_key="sk-test123")
        db_config = DatabaseConfig()
        
        app_config = AppConfig(
            api_config=api_config,
            db_config=db_config,
            log_level="info",
            enable_gpu=True,
            max_conversation_history=50
        )
        
        assert app_config.api_config == api_config
        assert app_config.db_config == db_config
        assert app_config.log_level == "info"
        assert app_config.enable_gpu is True
        assert app_config.max_conversation_history == 50
    
    def test_app_config_validation(self):
        """Test AppConfig validation rules."""
        api_config = APIConfig(api_key="sk-test123")
        
        # Test invalid log level
        with pytest.raises(ValueError):
            AppConfig(api_config=api_config, log_level="invalid")
        
        # Test conversation history bounds
        with pytest.raises(ValueError):
            AppConfig(api_config=api_config, max_conversation_history=0)


class TestExcelModels:
    """Test cases for Excel-related data models with dynamic table support."""
    
    def test_sqlite_data_type_enum(self):
        """Test SQLiteDataType enum values."""
        assert SQLiteDataType.INTEGER.value == "INTEGER"
        assert SQLiteDataType.REAL.value == "REAL"
        assert SQLiteDataType.TEXT.value == "TEXT"
        assert SQLiteDataType.BLOB.value == "BLOB"
        assert SQLiteDataType.NULL.value == "NULL"
    
    def test_column_definition_creation(self):
        """Test ColumnDefinition creation and validation."""
        column = ColumnDefinition(
            name="Revenue",
            sqlite_type=SQLiteDataType.REAL,
            is_primary_key=False,
            is_nullable=True,
            default_value="0.0",
            description="Sales revenue in dollars"
        )
        
        assert column.name == "Revenue"
        assert column.sqlite_type == SQLiteDataType.REAL
        assert column.is_primary_key is False
        assert column.is_nullable is True
        assert column.default_value == "0.0"
        assert column.description == "Sales revenue in dollars"
    
    def test_column_definition_validation(self):
        """Test ColumnDefinition validation rules."""
        # Test empty column name
        with pytest.raises(ValueError, match="Column name cannot be empty"):
            ColumnDefinition(name="", sqlite_type=SQLiteDataType.TEXT)
        
        # Test column name with invalid characters
        with pytest.raises(ValueError, match="Column name contains invalid characters"):
            ColumnDefinition(name="col;umn", sqlite_type=SQLiteDataType.TEXT)
    
    def test_column_definition_sql_generation(self):
        """Test ColumnDefinition SQL generation."""
        # Basic column
        column1 = ColumnDefinition(name="Name", sqlite_type=SQLiteDataType.TEXT)
        assert column1.get_sql_definition() == '"Name" TEXT'
        
        # Column with constraints
        column2 = ColumnDefinition(
            name="ID",
            sqlite_type=SQLiteDataType.INTEGER,
            is_primary_key=True,
            is_nullable=False,
            default_value="0"
        )
        sql_def = column2.get_sql_definition()
        assert '"ID" INTEGER' in sql_def
        assert "PRIMARY KEY" in sql_def
        assert "NOT NULL" in sql_def
        assert "DEFAULT 0" in sql_def
    
    def test_dynamic_table_schema_creation(self):
        """Test DynamicTableSchema creation and validation."""
        columns = [
            ColumnDefinition(name="Date", sqlite_type=SQLiteDataType.TEXT),
            ColumnDefinition(name="Revenue", sqlite_type=SQLiteDataType.REAL)
        ]
        
        schema = DynamicTableSchema(
            table_name="sales_data",
            file_id="excel_123",
            sheet_name="Sales",
            columns=columns,
            row_count=150,
            primary_key="Date"
        )
        
        assert schema.table_name == "sales_data"
        assert schema.file_id == "excel_123"
        assert schema.sheet_name == "Sales"
        assert len(schema.columns) == 2
        assert schema.row_count == 150
        assert schema.primary_key == "Date"
        assert isinstance(schema.created_at, datetime)
    
    def test_dynamic_table_schema_validation(self):
        """Test DynamicTableSchema validation rules."""
        columns = [ColumnDefinition(name="test", sqlite_type=SQLiteDataType.TEXT)]
        
        # Test invalid table name
        with pytest.raises(ValueError, match="Table name cannot be empty"):
            DynamicTableSchema(
                table_name="",
                file_id="excel_123",
                sheet_name="test",
                columns=columns
            )
        
        # Test invalid file_id format
        with pytest.raises(ValueError, match='file_id must start with "excel_" prefix'):
            DynamicTableSchema(
                table_name="test",
                file_id="invalid",
                sheet_name="test",
                columns=columns
            )
        
        # Test empty columns
        with pytest.raises(ValueError, match="Table must have at least one column"):
            DynamicTableSchema(
                table_name="test",
                file_id="excel_123",
                sheet_name="test",
                columns=[]
            )
    
    def test_dynamic_table_schema_methods(self):
        """Test DynamicTableSchema helper methods."""
        columns = [
            ColumnDefinition(name="ID", sqlite_type=SQLiteDataType.INTEGER),
            ColumnDefinition(name="Name", sqlite_type=SQLiteDataType.TEXT)
        ]
        
        schema = DynamicTableSchema(
            table_name="test_table",
            file_id="excel_123",
            sheet_name="Test",
            columns=columns,
            row_count=100
        )
        
        # Test CREATE TABLE SQL generation
        create_sql = schema.get_create_table_sql()
        assert "CREATE TABLE IF NOT EXISTS" in create_sql
        assert "test_table" in create_sql
        assert "ID" in create_sql
        assert "Name" in create_sql
        
        # Test column names
        column_names = schema.get_column_names()
        assert "ID" in column_names
        assert "Name" in column_names
        
        # Test column retrieval
        column = schema.get_column_by_name("ID")
        assert column is not None
        assert column.name == "ID"
        
        # Test non-existent column
        column = schema.get_column_by_name("NonExistent")
        assert column is None
    
    def test_table_creation_result_creation(self):
        """Test TableCreationResult creation and methods."""
        result = TableCreationResult(
            success=True,
            table_name="sales_data",
            rows_inserted=150,
            execution_time=2.5
        )
        
        assert result.success is True
        assert result.table_name == "sales_data"
        assert result.rows_inserted == 150
        assert result.execution_time == 2.5
        assert result.error_message is None
        assert result.warnings == []
        
        # Test adding warnings
        result.add_warning("Some empty cells found")
        assert len(result.warnings) == 1
        assert "empty cells" in result.warnings[0]
    
    def test_excel_document_creation(self):
        """Test ExcelDocument creation and validation."""
        document = ExcelDocument(
            file_name="sales_data.xlsx",
            file_size=10240,
            sheet_names=["Sales", "Customers"],
            metadata={
                "source": "company_reports",
                "description": "Monthly sales data"
            }
        )
        
        assert document.file_name == "sales_data.xlsx"
        assert document.file_size == 10240
        assert document.sheet_names == ["Sales", "Customers"]
        assert document.metadata["source"] == "company_reports"
        assert document.storage_type == "sqlite"
        assert isinstance(document.upload_time, datetime)
        assert document.id.startswith("excel_")
    
    def test_excel_document_validation(self):
        """Test ExcelDocument validation rules."""
        # Test invalid file extension
        with pytest.raises(ValueError, match="File name must have a valid Excel extension"):
            ExcelDocument(
                file_name="data.txt",
                file_size=100,
                sheet_names=["Sheet1"]
            )
        
        # Test empty sheet names
        with pytest.raises(ValueError, match="Excel file must contain at least one sheet"):
            ExcelDocument(
                file_name="data.xlsx",
                file_size=100,
                sheet_names=[]
            )
    
    def test_excel_document_methods(self):
        """Test ExcelDocument helper methods."""
        document = ExcelDocument(
            file_name="data.xlsx",
            file_size=2048,  # 2KB
            sheet_names=["Sheet1", "Sheet2"]
        )
        
        # Test file info summary
        file_info = document.get_file_info()
        assert "data.xlsx" in file_info
        assert "2.0 KB" in file_info
        assert "Sheets: 2" in file_info
    
    def test_excel_sheet_data_creation(self):
        """Test ExcelSheetData creation and validation."""
        sheet_data = ExcelSheetData(
            file_id="excel_123",
            sheet_name="Sales",
            headers=["Date", "Product", "Revenue"],
            row_count=150,
            column_count=3,
            sample_data=[
                {"Date": "2024-01-01", "Product": "A", "Revenue": 1000.0},
                {"Date": "2024-01-02", "Product": "B", "Revenue": 1500.0}
            ],
            data_types={
                "Date": SQLiteDataType.TEXT,
                "Product": SQLiteDataType.TEXT,
                "Revenue": SQLiteDataType.REAL
            }
        )
        
        assert sheet_data.file_id == "excel_123"
        assert sheet_data.sheet_name == "Sales"
        assert sheet_data.headers == ["Date", "Product", "Revenue"]
        assert sheet_data.row_count == 150
        assert sheet_data.column_count == 3
        assert len(sheet_data.sample_data) == 2
        assert sheet_data.data_types["Revenue"] == SQLiteDataType.REAL
    
    def test_excel_sheet_data_validation(self):
        """Test ExcelSheetData validation rules."""
        # Test invalid file_id format
        with pytest.raises(ValueError, match='file_id must start with "excel_" prefix'):
            ExcelSheetData(
                file_id="invalid",
                sheet_name="test",
                headers=["test"]
            )
        
        # Test empty headers
        with pytest.raises(ValueError, match="Sheet must have at least one column header"):
            ExcelSheetData(
                file_id="excel_123",
                sheet_name="test",
                headers=[]
            )
        
        # Test sample data validation
        with pytest.raises(ValueError, match="Sample data keys must match the headers"):
            ExcelSheetData(
                file_id="excel_123",
                sheet_name="test",
                headers=["A", "B"],
                sample_data=[{"C": "value"}]  # Invalid key
            )
        
        # Test data types validation
        with pytest.raises(ValueError, match="Data types must only contain keys that are in headers"):
            ExcelSheetData(
                file_id="excel_123",
                sheet_name="test",
                headers=["A", "B"],
                data_types={"C": SQLiteDataType.TEXT}  # Invalid key
            )
    
    def test_excel_sheet_data_methods(self):
        """Test ExcelSheetData helper methods."""
        sheet_data = ExcelSheetData(
            file_id="excel_123",
            sheet_name="Sales Data",
            headers=["Date", "Revenue"],
            row_count=100,
            column_count=2,
            data_types={
                "Date": SQLiteDataType.TEXT,
                "Revenue": SQLiteDataType.REAL
            }
        )
        
        # Test sheet summary
        summary = sheet_data.get_sheet_summary()
        assert "Sales Data" in summary
        assert "Rows: 100" in summary
        assert "Columns: 2" in summary
        
        # Test column info
        column_info = sheet_data.get_column_info()
        assert column_info["Date"] == SQLiteDataType.TEXT
        assert column_info["Revenue"] == SQLiteDataType.REAL
        
        # Test SQLite column types
        sqlite_types = sheet_data.get_sqlite_column_types()
        assert sqlite_types["Date"] == "TEXT"
        assert sqlite_types["Revenue"] == "REAL"
        
        # Test table schema generation
        schema = sheet_data.generate_table_schema()
        assert schema is not None
        assert schema.table_name == "salesdata"  # Sanitized name (spaces are removed)
        assert len(schema.columns) == 2
        assert sheet_data.table_schema == schema
        
        # Test validation for SQLite storage
        warnings = sheet_data.validate_for_sqlite_storage()
        assert isinstance(warnings, list)
    
    def test_excel_file_upload_creation(self):
        """Test ExcelFileUpload creation and validation."""
        upload = ExcelFileUpload(
            file_name="data.xlsx",
            file_size=10240,
            description="Test data file",
            tags=["test", "sample"]
        )
        
        assert upload.file_name == "data.xlsx"
        assert upload.file_size == 10240
        assert upload.description == "Test data file"
        assert upload.tags == ["test", "sample"]
    
    def test_excel_search_query_creation(self):
        """Test ExcelSearchQuery creation and validation."""
        search = ExcelSearchQuery(
            query="sales revenue",
            file_id="excel_123",
            sheet_name="Sales",
            column_name="Product",
            case_sensitive=False,
            max_results=50
        )
        
        assert search.query == "sales revenue"
        assert search.file_id == "excel_123"
        assert search.sheet_name == "Sales"
        assert search.column_name == "Product"
        assert search.case_sensitive is False
        assert search.max_results == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
