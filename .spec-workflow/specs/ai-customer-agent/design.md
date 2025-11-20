# AI Customer Agent - Design Specification

## System Architecture

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

### File Upload Processing Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───►│  File Type       │───►│   Excel File?   │
│                 │    │  Detection       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌──────────────────┐
                       │  Store in        │    │  Store in        │
                       │ Knowledge Base   │    │  SQLite Database │
                       │ (Vector DB)      │    │                  │
                       └──────────────────┘    └──────────────────┘
```

### Component Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
├─────────────────────────────────────────────────────────────┤
│  • Streamlit Web Interface                                  │
│  • Desktop GUI (Tkinter/PyQt)                              │
│  • REST API Endpoints                                       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  • Chat Manager (with Query Routing)                       │
│  • Knowledge Base Manager                                  │
│  • Session Manager                                         │
│  • Configuration Manager                                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Service Layer                            │
├─────────────────────────────────────────────────────────────┤
│  • DeepSeek API Service                                    │
│  • Embedding Service                                       │
│  • Vector Search Service                                   │
│  • File Processing Service                                 │
│  • SQLite Database Service                                 │
│  • Text-to-SQL Service                                     │
│  • Query Results Processor                                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  • Chroma Vector Database                                  │
│  • SQLite Database (Excel Files)                           │
│  • Local File System                                       │
│  • Configuration Files                                     │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Technologies
- **Programming Language**: Python 3.8+
- **Web Framework**: FastAPI (for API) + Streamlit (for UI)
- **Vector Database**: ChromaDB (local, lightweight)
- **Embedding Model**: sentence-transformers (all-MiniLM-L6-v2)
- **GPU Acceleration**: CUDA + PyTorch with GPU support

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

## Core Components Design

### 1. DeepSeek API Service
```python
class DeepSeekService:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        
    async def chat_completion(self, messages: List[Dict], model: str = "deepseek-chat") -> str:
        """Send chat completion request to DeepSeek API"""
        
    async def stream_chat(self, messages: List[Dict], model: str = "deepseek-chat") -> AsyncGenerator:
        """Stream chat responses from DeepSeek API"""
```

### 2. Knowledge Base Manager
```python
class KnowledgeBaseManager:
    def __init__(self, persist_directory: str = "./knowledge_base"):
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=SentenceTransformerEmbeddings()
        )
        
    def add_documents(self, file_paths: List[str]) -> bool:
        """Process and add documents to knowledge base"""
        
    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents in knowledge base"""
        
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from knowledge base"""
```

### 3. Chat Manager
```python
class ChatManager:
    def __init__(self, deepseek_service: DeepSeekService, kb_manager: KnowledgeBaseManager):
        self.deepseek_service = deepseek_service
        self.kb_manager = kb_manager
        self.conversation_history = []
        
    async def process_message(self, user_message: str, use_knowledge_base: bool = True) -> str:
        """Process user message with optional knowledge base context"""
        
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        
    def clear_conversation(self) -> None:
        """Clear conversation history"""
```

### 4. SQLite Database Service
```python
class SQLiteDatabaseService:
    def __init__(self, db_path: str = "./excel_database.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        
    def store_excel_file(self, file_path: str, file_name: str, file_size: int, 
                        sheet_names: List[str], metadata: Dict[str, Any]) -> str:
        """Store Excel file metadata and content in SQLite database"""
        
    def get_excel_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve Excel file data from database"""
        
    def list_excel_files(self) -> List[Dict[str, Any]]:
        """List all Excel files in database"""
        
    def delete_excel_file(self, file_id: str) -> bool:
        """Delete Excel file from database"""
        
    def search_excel_data(self, query: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for data within Excel files"""
```

### 5. Enhanced Knowledge Base Manager
```python
class KnowledgeBaseManager:
    def __init__(self, persist_directory: str = "./knowledge_base", 
                 sqlite_service: Optional[SQLiteDatabaseService] = None):
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=SentenceTransformerEmbeddings()
        )
        self.sqlite_service = sqlite_service
        
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process and add documents to appropriate storage based on file type"""
        
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type and route to appropriate storage"""
        
    def _process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """Process Excel file and store in SQLite database"""
        
    def _process_other_files(self, file_path: str) -> bool:
        """Process non-Excel files and store in knowledge base"""
        
    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents in knowledge base (excludes Excel files)"""
        
    def search_excel_data(self, query: str) -> List[Dict[str, Any]]:
        """Search for data in Excel files stored in SQLite database"""
        
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from knowledge base (does not affect Excel files)"""
```

### 6. Text-to-SQL Service
```python
class TextToSQLService:
    def __init__(self, deepseek_service: DeepSeekService, sqlite_service: SQLiteDatabaseService):
        self.deepseek_service = deepseek_service
        self.sqlite_service = sqlite_service
        
    async def convert_to_sql(self, natural_language_query: str, table_schema: Dict[str, Any]) -> str:
        """Convert natural language query to SQL using DeepSeek API"""
        
    async def execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query on SQLite database and return results"""
        
    async def process_excel_query(self, user_query: str) -> Dict[str, Any]:
        """Process natural language query about Excel data and return formatted results"""
        
    def get_table_schemas(self) -> Dict[str, Any]:
        """Get schema information for all Excel tables in database"""
```

### 7. Enhanced Chat Manager with Text-to-SQL
```python
class ChatManager:
    def __init__(self, deepseek_service: DeepSeekService, kb_manager: KnowledgeBaseManager, 
                 text_to_sql_service: TextToSQLService):
        self.deepseek_service = deepseek_service
        self.kb_manager = kb_manager
        self.text_to_sql_service = text_to_sql_service
        self.conversation_history = []
        
    async def process_message(self, user_message: str, use_knowledge_base: bool = True) -> str:
        """Process user message with intelligent routing to appropriate services"""
        
    def _detect_query_intent(self, user_message: str) -> str:
        """Detect if query is about Excel data, knowledge base, or general conversation"""
        
    async def _handle_excel_query(self, user_message: str) -> str:
        """Handle Excel data queries using Text-to-SQL service"""
        
    async def _handle_knowledge_base_query(self, user_message: str) -> str:
        """Handle knowledge base queries"""
        
    async def _handle_general_query(self, user_message: str) -> str:
        """Handle general conversation queries"""
        
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        
    def clear_conversation(self) -> None:
        """Clear conversation history"""
```

### 8. Configuration Manager
```python
class ConfigManager:
    def __init__(self, config_path: str = "./config"):
        self.config_path = config_path
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict:
        """Load configuration from YAML/JSON files"""
        
    def update_setting(self, key: str, value: Any) -> bool:
        """Update configuration setting"""
        
    def get_deepseek_api_key(self) -> str:
        """Get DeepSeek API key from configuration"""
```

## Data Models

### Chat Message
```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

### Knowledge Base Document
```python
class KBDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    file_path: str
    file_type: str
    embedding: Optional[List[float]] = None
```

### Excel File Document
```python
class ExcelDocument(BaseModel):
    id: str
    file_name: str
    file_size: int
    sheet_names: List[str]
    metadata: Dict[str, Any]
    upload_time: datetime = Field(default_factory=datetime.now)
    storage_type: Literal["sqlite"] = "sqlite"
```

### Excel Sheet Data
```python
class ExcelSheetData(BaseModel):
    file_id: str
    sheet_name: str
    headers: List[str]
    row_count: int
    column_count: int
    sample_data: List[Dict[str, Any]]
```

### API Configuration
```python
class APIConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
```

## API Endpoints Design

### FastAPI Endpoints
```python
# Chat endpoints
@app.post("/api/chat")
async def chat_endpoint(message: str, use_kb: bool = True) -> Dict:
    """Process chat message with optional knowledge base"""

@app.get("/api/chat/history")
async def get_chat_history() -> List[ChatMessage]:
    """Get conversation history"""

@app.delete("/api/chat/history")
async def clear_chat_history() -> Dict:
    """Clear conversation history"""

# Knowledge base endpoints
@app.post("/api/knowledge-base/documents")
async def add_documents(files: List[UploadFile]) -> Dict:
    """Add documents to appropriate storage based on file type (Excel → SQLite, others → Vector DB)"""

@app.get("/api/knowledge-base/search")
async def search_knowledge_base(query: str, k: int = 3) -> List[KBDocument]:
    """Search knowledge base (excludes Excel files)"""

@app.delete("/api/knowledge-base")
async def clear_knowledge_base() -> Dict:
    """Clear knowledge base (does not affect Excel files)"""

# Excel file endpoints
@app.get("/api/excel-files")
async def list_excel_files() -> List[ExcelDocument]:
    """List all Excel files stored in SQLite database"""

@app.get("/api/excel-files/{file_id}")
async def get_excel_file(file_id: str) -> ExcelDocument:
    """Get specific Excel file details"""

@app.delete("/api/excel-files/{file_id}")
async def delete_excel_file(file_id: str) -> Dict:
    """Delete Excel file from SQLite database"""

@app.get("/api/excel-files/{file_id}/search")
async def search_excel_data(file_id: str, query: str, sheet_name: Optional[str] = None) -> List[Dict]:
    """Search for data within specific Excel file"""

# Configuration endpoints
@app.get("/api/config")
async def get_configuration() -> Dict:
    """Get current configuration"""

@app.put("/api/config")
async def update_configuration(config: Dict) -> Dict:
    """Update configuration"""
```

## User Interface Design

### Streamlit Web Interface
```python
# Main components
- Sidebar: Configuration and knowledge base management
- Main area: Chat interface with message history
- Header: Application status and GPU utilization

# Features
- Real-time chat with streaming responses
- File upload for knowledge base
- Configuration settings panel
- Conversation history management
- GPU status monitoring
```

### Desktop GUI (Alternative)
- **Framework**: Tkinter or PyQt
- **Features**: Native Windows application with system tray integration
- **Advantages**: Better performance, system integration

## GPU Optimization

### CUDA Configuration
```python
def setup_gpu():
    """Configure PyTorch for NVIDIA 4070Ti GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
    return torch.device("cpu")
```

### Batch Processing
- Use batch processing for document embeddings
- Implement async operations for API calls
- Optimize vector search with GPU acceleration

## Security Considerations

### Data Protection
- All data stored locally
- API keys encrypted in configuration
- No external data transmission beyond DeepSeek API

### Input Validation
- Sanitize file uploads
- Validate API inputs
- Rate limiting for API calls

## Performance Optimization

### Caching Strategy
```python
class ResponseCache:
    def __init__(self, max_size: int = 1000):
        self.cache = LRUCache(max_size=max_size)
        
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for query"""
        
    def cache_response(self, query: str, response: str) -> None:
        """Cache response for future use"""
```

### Async Operations
- Use async/await for all I/O operations
- Implement connection pooling for API calls
- Background tasks for knowledge base indexing

## Deployment Architecture

### Local Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                    Windows Local Machine                    │
├─────────────────────────────────────────────────────────────┤
│  • Python Virtual Environment                              │
│  • FastAPI Server (localhost:8000)                         │
│  • Streamlit UI (localhost:8501)                           │
│  • ChromaDB Vector Store (local files)                     │
│  • GPU Acceleration (NVIDIA 4070Ti)                        │
└─────────────────────────────────────────────────────────────┘
```

### File Structure
```
ai-customer-agent/
├── src/
│   ├── services/
│   │   ├── deepseek_service.py
│   │   ├── knowledge_base.py
│   │   └── chat_manager.py
│   ├── api/
│   │   ├── main.py
│   │   └── endpoints/
│   ├── ui/
│   │   └── streamlit_app.py
│   └── models/
│       ├── chat_models.py
│       └── config_models.py
├── config/
│   ├── settings.yaml
│   └── .env
├── knowledge_base/
│   └── chroma_db/
├── logs/
└── requirements.txt
```

## Error Handling & Logging

### Error Handling Strategy
- Comprehensive exception handling for all components
- Graceful degradation when services are unavailable
- User-friendly error messages

### Logging Configuration
```python
import loguru

logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)
```

## Testing Strategy

### Test Types
- Unit tests for individual components
- Integration tests for API endpoints
- End-to-end tests for user workflows
- Performance tests for GPU utilization

### Test Framework
- pytest for Python tests
- pytest-asyncio for async tests
- unittest for component tests

## Monitoring & Maintenance

### Health Checks
- API endpoint health monitoring
- GPU utilization tracking
- Knowledge base integrity checks

### Maintenance Tasks
- Regular knowledge base optimization
- Log file rotation and cleanup
- Configuration backup
