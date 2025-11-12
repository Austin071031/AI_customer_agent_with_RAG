# Project Structure

## Directory Organization

```
ai-customer-agent/
├── src/                    # Main source code
│   ├── api/               # FastAPI backend
│   │   ├── endpoints/     # API route handlers
│   │   ├── main.py        # API server entry point
│   │   ├── dependencies.py # Dependency injection
│   │   └── state.py       # Application state management
│   ├── services/          # Core business logic
│   │   ├── chat_manager.py
│   │   ├── knowledge_base.py
│   │   ├── deepseek_service.py
│   │   └── config_manager.py
│   ├── models/            # Data models and validation
│   │   ├── chat_models.py
│   │   └── config_models.py
│   ├── ui/                # Streamlit frontend
│   │   └── streamlit_app.py
│   └── utils/             # Shared utilities
│       ├── logger.py
│       ├── cache.py
│       ├── error_handler.py
│       └── gpu_utils.py
├── config/                # Configuration files
│   ├── settings.yaml
│   └── .env
├── knowledge_base/        # Vector database storage
│   └── chroma_db/
├── logs/                  # Application logs
├── examples/              # Usage examples
├── docs/                  # Additional documentation
├── tests/                 # Test suite
├── main.py               # Main application entry point
├── run_api.py            # API server entry point
├── run_ui.py             # UI server entry point
└── requirements.txt      # Python dependencies
```

## Naming Conventions

### Files
- **Components/Modules**: `snake_case` for Python files and directories
- **Services/Handlers**: `snake_case` with descriptive names (e.g., `chat_manager.py`, `knowledge_base.py`)
- **Utilities/Helpers**: `snake_case` with utility-focused names (e.g., `gpu_utils.py`, `error_handler.py`)
- **Tests**: `test_` prefix with same directory structure as source (e.g., `test_chat_manager.py`)

### Code
- **Classes/Types**: `PascalCase` (e.g., `ChatManager`, `KnowledgeBaseService`)
- **Functions/Methods**: `snake_case` (e.g., `process_document`, `send_chat_message`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODEL`, `MAX_TOKENS`)
- **Variables**: `snake_case` (e.g., `conversation_history`, `api_key`)

## Import Patterns

### Import Order
1. Standard library imports
2. Third-party library imports
3. Local application imports
4. Type imports (if using type annotations)

### Module/Package Organization
- **Absolute imports** from project root using `src` as package
- **Relative imports** within the same module directory
- **Clear separation** between internal and external dependencies

## Code Structure Patterns

### Module/Class Organization
```
1. Imports (standard lib, third-party, local)
2. Constants and configuration
3. Type definitions and data models
4. Main class/function definitions
5. Helper/utility functions
6. Main execution block (if applicable)
```

### Function/Method Organization
- **Input validation** at function start
- **Core business logic** in the middle
- **Error handling** with specific exceptions
- **Clear return values** with consistent types

### File Organization Principles
- **One primary class per file** for services and managers
- **Related functionality grouped** in same directory
- **Public API clearly defined** at module level
- **Implementation details** in private functions

## Code Organization Principles

1. **Single Responsibility**: Each service and module has one clear purpose
2. **Modularity**: Services are independent and can be tested separately
3. **Testability**: Dependency injection enables easy mocking and testing
4. **Consistency**: Follow established patterns across the codebase

## Module Boundaries

- **API Layer**: Handles HTTP requests, validation, and response formatting
- **Service Layer**: Contains business logic and coordinates between services
- **Data Layer**: Manages data storage, retrieval, and persistence
- **UI Layer**: Handles user interface and interaction
- **Utility Layer**: Provides shared functionality across all layers

### Dependency Flow
```
UI Layer → API Layer → Service Layer → Data Layer
                    ↘ Utility Layer ↗
```

## Code Size Guidelines

- **File size**: Maximum 300-400 lines per file
- **Function/Method size**: Maximum 50 lines per function
- **Class/Module complexity**: Maximum 10 methods per class
- **Nesting depth**: Maximum 3 levels of indentation

## Dashboard/Monitoring Structure

### Streamlit UI Organization
```
src/ui/
└── streamlit_app.py     # Main Streamlit application
    ├── Sidebar components
    ├── Main chat interface
    ├── Knowledge base management
    └── Configuration panels
```

### Separation of Concerns
- **UI isolated** from business logic via API calls
- **State management** handled by Streamlit session state
- **Configuration** loaded from external files
- **Real-time updates** via WebSocket connections

## Documentation Standards

- **All public APIs** must have docstrings following Google style
- **Complex logic** includes inline comments explaining the approach
- **Configuration files** include comments for each setting
- **README files** for major modules and the overall project
- **Type hints** used throughout for better code clarity and IDE support

### Documentation Format
```python
def process_document(file_path: str, chunk_size: int = 1000) -> List[Document]:
    """Process a document and split it into chunks for the knowledge base.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each text chunk in characters
        
    Returns:
        List of Document objects with metadata
        
    Raises:
        FileNotFoundError: If the document doesn't exist
        ValueError: If the file format is not supported
    """
    # Implementation here
```

## Testing Structure

- **Unit tests** in `tests/` directory mirroring source structure
- **Integration tests** for API endpoints and service interactions
- **Test data** separated from test logic
- **Mock external dependencies** for reliable testing
