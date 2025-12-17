# AI Customer Agent - Task Breakdown

## Implementation Tasks

### Task 1: Project Setup and Configuration
**File**: `requirements.txt`, `config/settings.yaml`, `config/.env`

**Requirements**: US-003, US-007, US-008

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python DevOps Engineer

**Task**: Set up the project structure, dependencies, and configuration files for the AI customer service agent. Create the initial project layout with all necessary configuration files and dependency specifications.

**Restrictions**: 
- Do not implement any business logic yet
- Focus only on project structure and configuration
- Use Python 3.8+ compatible dependencies
- Ensure Windows compatibility
                                                                                                                                    
**_Leverage**: 
- requirements.txt for dependency management
- config/settings.yaml for application settings
- config/.env for sensitive configuration

**_Requirements**:
- US-003: Windows Local Execution
- US-007: Python Implementation  
- US-008: Error Handling

**Success**: 
- Project structure created with all necessary directories
- requirements.txt with all dependencies specified
- Configuration files created with proper structure
- Environment variable setup for API keys

- [x] **Task 1**: Project Setup and Configuration

### Task 2: Core Data Models with Dynamic Table Support
**File**: `src/models/chat_models.py`, `src/models/config_models.py`, `src/models/excel_models.py`

**Requirements**: US-001, US-002, US-007, US-009

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Data Engineer

**Task**: Create the core data models for chat messages, knowledge base documents, Excel file documents with dynamic table support, and API configuration using Pydantic. These models will form the foundation for all data handling in the application, including dynamic SQLite table creation for Excel sheets.

**Restrictions**:
- Use Pydantic for data validation
- Include proper type hints and documentation
- Follow Python naming conventions
- No business logic in models
- Support dynamic table schema for Excel sheets

**_Leverage**:
- Pydantic for data validation
- Python typing for type hints
- UUID for unique identifiers
- SQLite database models for dynamic Excel table creation
- Support for relational table schemas with proper column types

**_Requirements**:
- US-001: DeepSeek API Integration
- US-002: Local Knowledge Base Integration
- US-007: Python Implementation
- US-009: Excel File Upload with Dynamic Table Creation

**Success**:
- ChatMessage model with role, content, timestamp, message_id
- KBDocument model with content, metadata, file info
- ExcelDocument model for SQLite storage with file_name, file_size, sheet_names, metadata
- ExcelSheetData model for dynamic table schema with headers, column types, and relational storage
- APIConfig model for API settings
- Support for dynamic table creation with proper column types (integer, real, text, etc.)
- All models properly validated and documented

- [x] **Task 2**: Core Data Models with Dynamic Table Support

### Task 3: DeepSeek API Service
**File**: `src/services/deepseek_service.py`

**Requirements**: US-001, US-008

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python API Integration Engineer

**Task**: Implement the DeepSeek API service that handles communication with the DeepSeek API. Include both standard chat completion and streaming responses, with proper error handling and configuration management.

**Restrictions**:
- Use async/await for API calls
- Implement proper error handling
- Support both streaming and non-streaming responses
- No UI components

**_Leverage**:
- requests or aiohttp for API calls
- asyncio for async operations
- Configuration manager for API settings

**_Requirements**:
- US-001: DeepSeek API Integration
- US-008: Error Handling

**Success**:
- DeepSeekService class with chat_completion method
- Streaming response support
- Proper error handling for API failures
- Configuration-based API settings

- [-] **Task 3**: DeepSeek API Service

### Task 4: Knowledge Base Manager with Dynamic Table Creation and Document Chunking
**File**: `src/services/knowledge_base.py`, `src/services/document_chunking_service.py`

**Requirements**: US-002, US-006, US-009

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python AI/ML Engineer

**Task**: Create the enhanced knowledge base manager that handles document processing with file type detection, intelligent chunking for long documents, and dynamic table creation for Excel files. Support multiple file formats with intelligent routing: Excel files to SQLite database with dynamic tables for each sheet, other documents (PDF, TXT, DOCX) to vector storage with proper chunking to ensure all content is stored for RAG. Implement the Document Chunking Service to handle long documents with configurable chunk sizes and overlap.

**Restrictions**:
- Use ChromaDB for vector storage of non-Excel files
- Use SQLite for structured Excel file storage with dynamic table creation
- Support common document formats (PDF, TXT, DOCX, XLSX)
- No chat functionality
- Create dynamic tables for each Excel sheet with proper column types
- Implement document chunking for PDF, TXT, DOCX files with configurable chunk size and overlap
- Validate no content loss during chunking

**_Leverage**:
- ChromaDB for vector database
- SQLite for Excel file storage with dynamic table creation
- sentence-transformers for embeddings
- File processing libraries (pypdf2, python-docx)
- openpyxl for Excel file processing
- Dynamic table creation with proper column type inference
- Document chunking strategies with metadata preservation

**_Requirements**:
- US-002: Local Knowledge Base Integration
- US-006: Knowledge Base Management
- US-009: Excel File Upload with Dynamic Table Creation

**Success**:
- Enhanced KnowledgeBaseManager class with file type detection and routing
- DocumentChunkingService class with configurable chunk size and overlap, supporting PDF, TXT, DOCX
- Excel file processing and dynamic table creation in SQLite database for each sheet
- Document chunking for long PDF, TXT, DOCX files with metadata preservation (page numbers, section headers)
- Vector search functionality for non-Excel documents (including chunked documents)
- Excel data search functionality in SQLite database using relational tables
- Dynamic table creation with proper column types (integer, real, text, etc.)
- Validation of no content loss during chunking process

- [-] **Task 4**: Knowledge Base Manager with Dynamic Table Creation and Document Chunking

### Task 5: Text-to-SQL Service with Actual SQL Execution
**File**: `src/services/text_to_sql_service.py`

**Requirements**: US-010, US-001, US-009

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python AI/ML Engineer

**Task**: Implement the Text-to-SQL service that converts natural language queries about Excel data into actual SQL queries executed on relational tables. The service should use DeepSeek API for SQL generation and execute queries on the SQLite database with dynamic Excel tables, supporting standard SQL operations.

**Restrictions**:
- Use DeepSeek API for SQL query generation
- Execute actual SQL queries on relational tables in SQLite database
- Handle dynamic table schema discovery for Excel sheets
- Support standard SQL operations (SELECT, WHERE, JOIN, GROUP BY, etc.)
- No UI components

**_Leverage**:
- DeepSeekService for SQL generation
- SQLiteDatabaseService for query execution on relational tables
- Dynamic table schema discovery and validation
- Natural language to SQL conversion with relational table support
- Standard SQL query execution

**_Requirements**:
- US-010: Text-to-SQL Query Service with Actual SQL Execution
- US-001: DeepSeek API Integration
- US-009: Excel File Upload with Dynamic Table Creation

**Success**:
- TextToSQLService class with convert_to_sql method for relational tables
- Actual SQL query execution and result processing on relational data
- Dynamic table schema discovery and validation for Excel sheets
- Natural language to SQL conversion using DeepSeek API with relational context
- Support for standard SQL operations on Excel data

- [-] **Task 5**: Text-to-SQL Service with Actual SQL Execution

### Task 6: Enhanced Chat Manager with Text-to-SQL
**File**: `src/services/chat_manager.py`

**Requirements**: US-001, US-002, US-005, US-008, US-010, US-011

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Backend Engineer

**Task**: Implement the enhanced chat manager with intelligent query routing and Text-to-SQL integration. The manager should detect query intent and route to appropriate services (Excel data via Text-to-SQL, knowledge base, or general conversation).

**Restrictions**:
- Integrate with DeepSeekService, KnowledgeBaseManager, and TextToSQLService
- Implement query intent detection
- Manage conversation history with mixed data sources
- No UI components

**_Leverage**:
- DeepSeekService for API calls
- KnowledgeBaseManager for knowledge base context
- TextToSQLService for Excel data queries
- Query intent detection and routing
- Conversation history management

**_Requirements**:
- US-001: DeepSeek API Integration
- US-002: Local Knowledge Base Integration
- US-005: Chat Interface
- US-008: Error Handling
- US-010: Text-to-SQL Query Service
- US-011: Intelligent Excel Data Query Routing

**Success**:
- Enhanced ChatManager class with query intent detection
- Intelligent routing to Text-to-SQL, knowledge base, or general conversation
- Integration with all three service types
- Proper error handling and fallback mechanisms

- [x] **Task 6**: Enhanced Chat Manager with Text-to-SQL

### Task 7: Configuration Manager
**File**: `src/services/config_manager.py`

**Requirements**: US-003, US-007, US-008

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Systems Engineer

**Task**: Create the configuration manager that handles application settings, API keys, and environment variables. Support both YAML configuration files and environment variables for sensitive data.

**Restrictions**:
- Support multiple configuration sources
- Encrypt sensitive data
- Provide validation for settings
- No business logic

**_Leverage**:
- python-dotenv for environment variables
- PyYAML for configuration files
- Pydantic for settings validation

**_Requirements**:
- US-003: Windows Local Execution
- US-007: Python Implementation
- US-008: Error Handling

**Success**:
- ConfigManager class with settings loading
- Environment variable support
- Configuration validation
- Secure API key handling

- [x] **Task 7**: Configuration Manager

### Task 8: FastAPI Backend with Relational Storage and Document Chunking
**File**: `src/api/main.py`, `src/api/endpoints/chat.py`, `src/api/endpoints/knowledge_base.py`, `src/api/endpoints/config.py`, `src/api/endpoints/excel_files.py`

**Requirements**: US-001, US-002, US-005, US-006, US-008, US-009, US-010, US-011

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Backend API Developer

**Task**: Create the FastAPI backend with REST endpoints for chat, knowledge base management with document chunking for long files, Excel file management with relational storage, and configuration. Implement proper request validation, error handling, and API documentation with dynamic table creation, Text-to-SQL integration for relational data, and document chunking configuration.

**Restrictions**:
- Use FastAPI framework
- Implement proper HTTP status codes
- Include API documentation
- No frontend UI
- Support dynamic table creation for Excel files
- Include Text-to-SQL service integration with relational table support
- Support document chunking configuration for PDF, TXT, DOCX files

**_Leverage**:
- FastAPI for web framework
- Pydantic for request/response models
- Service classes for business logic
- File upload handling with dynamic table creation and document chunking
- Text-to-SQL service for Excel data queries on relational tables
- Relational database operations for Excel data
- Document chunking service for long documents

**_Requirements**:
- US-001: DeepSeek API Integration
- US-002: Local Knowledge Base Integration
- US-005: Chat Interface
- US-006: Knowledge Base Management
- US-008: Error Handling
- US-009: Excel File Upload with Dynamic Table Creation
- US-010: Text-to-SQL Query Service with Actual SQL Execution
- US-011: Intelligent Excel Data Query Routing

**Success**:
- FastAPI application with all endpoints
- Chat endpoints with streaming support and Text-to-SQL integration for relational data
- Knowledge base management endpoints with file type routing, dynamic table creation for Excel, and document chunking for PDF/TXT/DOCX
- Excel file management endpoints (list, get, delete, search) with relational storage
- Configuration endpoints including chunking configuration (chunk size, overlap)
- Proper API documentation with dynamic table creation examples, Text-to-SQL usage on relational data, and document chunking examples

- [-] **Task 8**: FastAPI Backend with Relational Storage and Document Chunking

### Task 9: Streamlit Web Interface with Document Chunking Configuration
**File**: `src/ui/streamlit_app.py`

**Requirements**: US-005, US-006, US-003, US-009, US-010, US-011

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Frontend Developer

**Task**: Create the Streamlit web interface for the AI customer service agent. Include chat interface, knowledge base management with Excel file upload and document chunking configuration for PDF/TXT/DOCX files, Excel file management panel, and configuration panel with chunking settings.

**Restrictions**:
- Use Streamlit for web interface
- Implement real-time chat with streaming
- Include file upload for knowledge base with Excel file support and chunking options for long documents
- Include Excel file management interface
- No backend API changes
- Include chunking configuration (chunk size, overlap) in settings panel

**_Leverage**:
- Streamlit for web framework
- FastAPI client for backend communication
- Real-time updates for chat
- File upload components with type detection and chunking options
- Data table display for Excel file management
- Configuration sliders for chunking parameters

**_Requirements**:
- US-005: Chat Interface
- US-006: Knowledge Base Management
- US-003: Windows Local Execution
- US-009: Excel File Upload with SQLite Storage

**Success**:
- Streamlit application with chat interface
- Real-time message streaming
- Knowledge base file upload with Excel file support and chunking configuration for PDF/TXT/DOCX
- Excel file management panel (list, view, delete, search)
- Configuration management panel with chunking settings (chunk size, overlap)
- Chunking progress and status display during file upload

- [-] **Task 9**: Streamlit Web Interface with Document Chunking Configuration

### Task 10: Error Handling and Logging
**File**: `src/utils/error_handler.py`, `src/utils/logger.py`

**Requirements**: US-008, US-003, US-007, US-010

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Reliability Engineer

**Task**: Implement comprehensive error handling and logging throughout the application. Create centralized error handling, user-friendly error messages, and structured logging with Text-to-SQL error handling.

**Restrictions**:
- Use loguru for logging
- Implement graceful error handling including Text-to-SQL failures
- Provide user-friendly error messages
- No business logic changes

**_Leverage**:
- loguru for structured logging
- Python exception handling
- Custom exception classes for Text-to-SQL errors

**_Requirements**:
- US-008: Error Handling
- US-003: Windows Local Execution
- US-007: Python Implementation
- US-010: Text-to-SQL Query Service

**Success**:
- Centralized error handling including Text-to-SQL service
- Structured logging with rotation
- User-friendly error messages for Text-to-SQL queries
- Comprehensive exception coverage

- [x] **Task 10**: Error Handling and Logging

### Task 11: Main Application Entry Points
**File**: `main.py`, `run_api.py`, `run_ui.py`

**Requirements**: US-003, US-007, US-008, US-010

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Python Application Developer

**Task**: Create the main application entry points for running the API server, Streamlit UI, and combined application. Include proper startup configuration and service management with Text-to-SQL service initialization.

**Restrictions**:
- Create separate entry points for different components
- Handle graceful shutdown
- Include health checks
- No new functionality

**_Leverage**:
- uvicorn for ASGI server
- Streamlit for UI server
- Python subprocess for service management
- Text-to-SQL service initialization

**_Requirements**:
- US-003: Windows Local Execution
- US-007: Python Implementation
- US-008: Error Handling
- US-010: Text-to-SQL Query Service

**Success**:
- Main application entry point with Text-to-SQL service
- Separate API and UI entry points
- Graceful startup and shutdown
- Health check endpoints including Text-to-SQL service

- [x] **Task 11**: Main Application Entry Points

### Task 12: Documentation and Examples
**File**: `README.md`, `examples/`, `docs/`

**Requirements**: US-003, US-007, US-008, US-010, US-011

**_Prompt**:
Implement the task for spec ai-customer-agent, first run spec-workflow-guide to get the workflow guide then implement the task:

**Role**: Technical Writer and Developer

**Task**: Create comprehensive documentation, usage examples, and setup instructions. Include API documentation, configuration guides, troubleshooting information, and Text-to-SQL usage examples.

**Restrictions**:
- Focus on user documentation
- Include practical examples including Text-to-SQL
- Provide troubleshooting guide
- No code changes

**_Leverage**:
- Markdown for documentation
- Code examples in Python including Text-to-SQL
- Configuration examples

**_Requirements**:
- US-003: Windows Local Execution
- US-007: Python Implementation
- US-008: Error Handling
- US-010: Text-to-SQL Query Service
- US-011: Intelligent Excel Data Query Routing

**Success**:
- Comprehensive README.md with Text-to-SQL examples
- Usage examples and tutorials including Excel data queries
- API documentation with Text-to-SQL endpoints
- Troubleshooting guide for Text-to-SQL queries

- [x] **Task 12**: Documentation and Examples
