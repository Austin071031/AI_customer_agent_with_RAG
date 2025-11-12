# AI Customer Agent

A Python-based AI customer service agent that uses DeepSeek API and can connect to local knowledge bases, running locally on Windows with NVIDIA 4070Ti GPU support.

## Features

- 🤖 **DeepSeek AI Integration**: Connect to DeepSeek API for intelligent customer service responses
- 📚 **Local Knowledge Base**: Index and search local documents (PDF, TXT, DOCX, XLSX)
- 💬 **Real-time Chat**: Stream responses with conversation history
- 🚀 **GPU Acceleration**: Optimized for NVIDIA 4070Ti GPU
- 🖥️ **Web Interface**: Streamlit-based web UI
- 🔧 **REST API**: FastAPI backend with comprehensive endpoints
- 🔒 **Local Execution**: All data stays on your local machine

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

### Knowledge Base Integration
1. Upload documents (PDF, TXT, DOCX, XLSX) via the web interface
2. The AI will automatically use relevant information from your documents
3. Manage uploaded files in the knowledge base section

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
│   │   └── config_manager.py
│   ├── api/              # FastAPI backend
│   │   ├── main.py
│   │   └── endpoints/
│   ├── ui/               # Streamlit frontend
│   │   └── streamlit_app.py
│   ├── models/           # Data models
│   │   ├── chat_models.py
│   │   └── config_models.py
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
- `POST /api/chat` - Send chat message
- `GET /api/chat/history` - Get conversation history  
- `DELETE /api/chat/history` - Clear conversation history

### Knowledge Base Endpoints
- `POST /api/knowledge-base/documents` - Upload documents
- `GET /api/knowledge-base/search` - Search knowledge base
- `DELETE /api/knowledge-base` - Clear knowledge base

### Configuration Endpoints
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration

## GPU Optimization

The application automatically detects and utilizes NVIDIA GPUs for:
- Document embedding generation
- Vector search operations
- Model inference acceleration

Check GPU status in the web interface sidebar.

## Troubleshooting

### Common Issues

**API Key Error**
- Ensure `DEEPSEEK_API_KEY` is set in `config/.env`
- Verify the API key is valid and has sufficient credits

**GPU Not Detected**
- Install NVIDIA drivers and CUDA toolkit
- Ensure PyTorch with CUDA support is installed
- Check GPU compatibility with CUDA version

**Knowledge Base Not Working**
- Verify document formats are supported (PDF, TXT, DOCX, XLSX)
- Check file permissions in knowledge_base directory
- Ensure sufficient disk space for vector database

**Port Already in Use**
- Change ports in `config/settings.yaml`
- Ensure no other services are using ports 8000 or 8501

### Logs

Check application logs in the `logs/` directory:
- `logs/app.log` - General application logs
- `logs/api_server.log` - API server logs
- `logs/ui_server.log` - UI server logs

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- **Services**: Core business logic (chat, knowledge base, configuration)
- **API**: REST endpoints and request handling
- **UI**: Streamlit web interface
- **Models**: Data validation and serialization
- **Utils**: Shared utilities and helpers

### Adding New Features
1. Follow the existing code patterns and structure
2. Add tests in the `tests/` directory
3. Update documentation in `README.md` and `docs/`

## License

[Add your license here]

## Support

For issues and feature requests, please create an issue in the project repository.
