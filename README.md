# AI Customer Agent

A Python-based AI customer service agent that uses DeepSeek API and can connect to local knowledge bases, running locally on Windows with NVIDIA 4070Ti GPU support. The system now includes enhanced capabilities for Excel file processing with SQLite storage, Text-to-SQL querying, and intelligent document chunking for long documents.

## Features

- 🤖 **DeepSeek AI Integration**: Connect to DeepSeek API for intelligent customer service responses
- 📚 **Local Knowledge Base**: Index and search local documents (PDF, TXT, DOCX, XLSX) with intelligent chunking
- 💬 **Real-time Chat**: Stream responses with conversation history and intelligent query routing
- 🖥️ **Web Interface**: Streamlit-based web UI with comprehensive management panels
- 🔧 **REST API**: FastAPI backend with comprehensive endpoints including Text-to-SQL and Excel management
- 🔒 **Local Execution**: All data stays on your local machine
- 📊 **Excel File Processing**: Upload Excel files stored in SQLite database with dynamic table creation for each sheet
- 🗣️ **Text-to-SQL Service**: Convert natural language queries to actual SQL executed on relational Excel tables
- 🔍 **Intelligent Query Routing**: Automatically detect query intent and route to appropriate service (Excel data, knowledge base, or general conversation)
- 📄 **Document Chunking**: Intelligent chunking of long documents (PDF, TXT, DOCX) with configurable chunk size and overlap

## Enhanced System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web/Desktop   │◄──►│   FastAPI Backend │◄──►│  DeepSeek API   │
│    Interface    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ├───────────────────┐
                                ▼                   ▼
                       ┌──────────────────┐ ┌──────────────────┐
                       │ Local Knowledge  │ │   SQLite DB      │
                       │     Base         │ │  (Excel Files)   │
                       │   (Vector DB)    │ │                  │
                       └──────────────────┘ └──────────────────┘
                                │                         │
                                │                         │
                                │                         ▼
                                │                ┌──────────────────┐
                                │                │  Text-to-SQL     │
                                │                │    Service       │
                                │                │                  │
                                │                └──────────────────┘
                                │                         │
                                │                         ▼
                                │                ┌──────────────────┐
                                │                │ Query Results    │
                                │                │  Processor       │
                                │                │                  │
                                │                └──────────────────┘
                                │                         │
                                └─────────────────────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │   Response       │
                                  │   Generator      │
                                  │                  │
                                  └──────────────────┘
```

### File Upload Processing Flow with Document Chunking
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───►│  File Type       │───►│   Excel File?   │
│                 │    │  Detection       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                 ┌──────────────┴─────────────┐    ┌──────────────────┐
                 ▼                            ▼    │  Store in        │
        ┌──────────────────┐       ┌──────────────────┐ │  SQLite Database │
        │  Document        │       │  Store in        │ │                  │
        │  Chunking        │──────►│ Knowledge Base   │ └──────────────────┘
        │  (PDF/TXT/DOCX)  │       │ (Vector DB)      │
        └──────────────────┘       └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  Chunk Storage   │
        │  with Metadata   │
        └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Windows 10/11
- NVIDIA GPU with CUDA support (recommended)
- DeepSeek API key

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd ai-customer-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure your environment:**
   - Copy `config/.env.example` to `config/.env`
   - Add your DeepSeek API key:
```env
DEEPSEEK_API_KEY=your_api_key_here
```

4. **Run the application:**
```bash
# Run both API and UI
python main.py

# Or run separately:
# API server only
python run_api.py

# UI only  
python run_ui.py
```

5. **Access the application:**
   - Web UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## Usage Examples

### Basic Chat
1. Open the web interface at http://localhost:8501
2. Start chatting with the AI agent in the main chat area
3. Use the sidebar to configure settings and manage knowledge base

### Knowledge Base Integration with Document Chunking
1. Upload documents (PDF, TXT, DOCX) via the web interface
2. System automatically chunks long documents with configurable settings
3. The AI will automatically use relevant information from your documents
4. Manage uploaded files in the knowledge base section

### Excel File Upload with SQLite Storage
1. Upload Excel files (.xlsx) via the web interface
2. System automatically creates dynamic SQLite tables for each sheet
3. Each sheet is stored as a relational table with proper column types
4. Excel data is available for natural language queries via Text-to-SQL

### Text-to-SQL Queries on Excel Data
1. Ask natural language questions about your Excel data
2. System automatically converts questions to SQL queries
3. Queries are executed on relational SQLite tables
4. Results are processed and returned in natural language format

### API Usage
```python
import requests

# Send a chat message
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "Hello, how can you help me?", "use_kb": True}
)
print(response.json())

# Search knowledge base
response = requests.get(
    "http://localhost:8000/api/knowledge-base/search",
    params={"query": "customer service policy", "k": 3}
)
print(response.json())

# Upload Excel file
files = {"files": open("data.xlsx", "rb")}
response = requests.post(
    "http://localhost:8000/api/knowledge-base/documents",
    files=files
)
print(response.json())

# Query Excel data using Text-to-SQL
response = requests.post(
    "http://localhost:8000/api/excel-files/search",
    json={"file_id": "excel-file-id", "query": "Show me total sales by region"}
)
print(response.json())

# List all Excel files
response = requests.get("http://localhost:8000/api/excel-files")
print(response.json())
```

## Configuration

### Environment Variables
Create `config/.env` with:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LOG_LEVEL=INFO
```

### Settings File
Modify `config/settings.yaml` for advanced configuration:
```yaml
api:
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 2000
  
knowledge_base:
  persist_directory: "./knowledge_base/chroma_db"
  supported_formats: [".pdf", ".txt", ".docx", ".xlsx"]
  
document_chunking:
  chunk_size: 1000
  chunk_overlap: 200
  
sqlite_database:
  db_path: "./excel_database.db"
  
ui:
  port: 8501
  theme: "light"
  
api_server:
  port: 8000
  host: "0.0.0.0"
```

## Project Structure

```
ai-customer-agent/
├── src/
│   ├── services/          # Core business logic
│   │   ├── deepseek_service.py
│   │   ├── knowledge_base.py
│   │   ├── chat_manager.py
│   │   ├── config_manager.py
│   │   ├── document_chunking_service.py  # New: Document chunking service
│   │   ├── sqlite_database_service.py    # New: SQLite database service for Excel files
│   │   └── text_to_sql_service.py        # New: Text-to-SQL service
│   ├── api/              # FastAPI backend
│   │   ├── main.py
│   │   ├── state.py
│   │   ├── dependencies.py
│   │   └── endpoints/
│   │       ├── chat.py
│   │       ├── config.py
│   │       ├── knowledge_base.py
│   │       └── excel_files.py           # New: Excel file management endpoints
│   ├── ui/               # Streamlit frontend
│   │   └── streamlit_app.py
│   ├── models/           # Data models
│   │   ├── chat_models.py
│   │   ├── config_models.py
│   │   └── excel_models.py              # New: Excel data models
│   └── utils/            # Utilities
│       ├── gpu_utils.py
│       ├── cache.py
│       ├── error_handler.py
│       └── logger.py
├── config/               # Configuration files
│   ├── settings.yaml
│   └── .env
├── knowledge_base/       # Vector database storage
│   └── chroma_db/
├── excel_database.db     # SQLite database for Excel files
├── logs/                 # Application logs
├── examples/             # Usage examples
├── docs/                 # Additional documentation
├── main.py              # Main application entry point
├── run_api.py           # API server entry point
├── run_ui.py            # UI server entry point
└── requirements.txt     # Python dependencies
```

## API Endpoints

### Chat Endpoints
- `POST /api/chat` - Send chat message with intelligent routing
- `GET /api/chat/history` - Get conversation history  
- `DELETE /api/chat/history` - Clear conversation history

### Knowledge Base Endpoints
- `POST /api/knowledge-base/documents` - Upload documents with automatic file type detection and routing (Excel → SQLite, others → Vector DB with chunking)
- `GET /api/knowledge-base/search` - Search knowledge base (excludes Excel files)
- `DELETE /api/knowledge-base` - Clear knowledge base (does not affect Excel files)

### Excel File Endpoints
- `POST /api/knowledge-base/documents` - Upload Excel files (automatically routed to SQLite storage)
- `GET /api/excel-files` - List all Excel files stored in SQLite database
- `GET /api/excel-files/{file_id}` - Get specific Excel file details
- `DELETE /api/excel-files/{file_id}` - Delete Excel file from SQLite database
- `POST /api/excel-files/{file_id}/search` - Search for data within specific Excel file using natural language
- `GET /api/excel-files/{file_id}/schemas` - Get table schemas for Excel file

### Text-to-SQL Endpoints
- `POST /api/text-to-sql/convert` - Convert natural language query to SQL
- `POST /api/text-to-sql/execute` - Execute SQL query on Excel data
- `POST /api/text-to-sql/process` - Process natural language query about Excel data and return formatted results

### Configuration Endpoints
- `GET /api/config` - Get current configuration including chunking settings
- `PUT /api/config` - Update configuration including chunking parameters
- `GET /api/config/chunking` - Get document chunking configuration
- `PUT /api/config/chunking` - Update document chunking configuration

## Technology Stack

### Core Technologies
- **Programming Language**: Python 3.8+
- **Web Framework**: FastAPI (for API) + Streamlit (for UI)
- **Vector Database**: ChromaDB (local, lightweight)
- **Relational Database**: SQLite (for Excel file storage)
- **Embedding Model**: sentence-transformers (all-MiniLM-L6-v2)

### Key Dependencies
```python
# Core AI/ML
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
langchain>=0.0.200
chromadb>=0.4.0

# API & Web
fastapi>=0.100.0
uvicorn>=0.20.0
streamlit>=1.25.0
requests>=2.28.0

# File Processing
pypdf2>=3.0.0
python-docx>=0.8.11
openpyxl>=3.1.0

# Database & SQL
sqlite3>=3.35.0  # Built-in
sqlalchemy>=2.0.0
pandas>=1.5.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
pydantic>=2.0.0
```


## License

[Add your license here]

## Support

For issues and feature requests, please create an issue in the project repository.

## Changelog

### Recent Enhancements
- **Excel File Support**: Upload Excel files stored in SQLite database with dynamic table creation
- **Text-to-SQL Service**: Convert natural language queries to actual SQL executed on relational tables
- **Intelligent Query Routing**: Automatic detection and routing to appropriate service
- **Document Chunking**: Intelligent chunking of long documents with configurable settings
- **Enhanced API**: Comprehensive endpoints for Excel file management and Text-to-SQL queries
- **Updated Architecture**: Support for both vector storage (ChromaDB) and relational storage (SQLite)
