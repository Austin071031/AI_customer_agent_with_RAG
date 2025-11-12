# API Reference

Complete reference for the AI Customer Agent REST API.

## Base URL

All API endpoints are relative to:
```
http://localhost:8000
```

## Authentication

Currently, the API runs locally without authentication. All endpoints are accessible from localhost.

## Common Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Chat Endpoints

### POST /api/chat

Send a chat message to the AI agent.

**Request Body:**
```json
{
  "message": "Hello, how can you help me?",
  "use_kb": true
}
```

**Parameters:**
- `message` (string, required): The user's message
- `use_kb` (boolean, optional): Whether to use knowledge base context (default: true)

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Hello! I'm here to help with customer service inquiries...",
    "response_time": 2.45,
    "message_id": "123e4567-e89b-12d3-a456-426614174000",
    "knowledge_base_used": true,
    "relevant_documents": [
      {
        "id": "doc_123",
        "content": "Customer service hours are 9 AM to 6 PM...",
        "file_path": "docs/customer_service.pdf",
        "similarity_score": 0.85
      }
    ]
  },
  "message": "Message processed successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /api/chat/history

Get the current conversation history.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "message_id": "123e4567-e89b-12d3-a456-426614174000",
      "role": "user",
      "content": "Hello, how can you help me?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "message_id": "123e4567-e89b-12d3-a456-426614174001",
      "role": "assistant",
      "content": "Hello! I'm here to help with customer service...",
      "timestamp": "2024-01-01T12:00:01Z"
    }
  ],
  "message": "Conversation history retrieved",
  "timestamp": "2024-01-01T12:00:02Z"
}
```

### DELETE /api/chat/history

Clear the conversation history.

**Response:**
```json
{
  "success": true,
  "data": {
    "cleared_messages": 5
  },
  "message": "Conversation history cleared",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Knowledge Base Endpoints

### POST /api/knowledge-base/documents

Upload documents to the knowledge base.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: One or more files

**Parameters:**
- `files` (array of files, required): Documents to upload (PDF, TXT, DOCX, XLSX)

**Response:**
```json
{
  "success": true,
  "data": {
    "uploaded_documents": 3,
    "failed_documents": 0,
    "total_documents": 150,
    "processing_time": 12.5
  },
  "message": "Documents uploaded successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /api/knowledge-base/search

Search the knowledge base for relevant documents.

**Parameters:**
- `query` (string, required): Search query
- `k` (integer, optional): Number of results to return (default: 3, max: 10)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "doc_123",
      "content": "Our customer service policy states that...",
      "metadata": {
        "file_path": "docs/policy.pdf",
        "file_type": "pdf",
        "file_size": 102400,
        "last_modified": "2024-01-01T10:00:00Z"
      },
      "similarity_score": 0.92
    }
  ],
  "message": "Search completed successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### DELETE /api/knowledge-base

Clear the entire knowledge base.

**Response:**
```json
{
  "success": true,
  "data": {
    "deleted_documents": 150
  },
  "message": "Knowledge base cleared",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Configuration Endpoints

### GET /api/config

Get current application configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "api": {
      "model": "deepseek-chat",
      "temperature": 0.7,
      "max_tokens": 2000,
      "base_url": "https://api.deepseek.com"
    },
    "knowledge_base": {
      "status": "active",
      "document_count": 150,
      "supported_formats": [".pdf", ".txt", ".docx", ".xlsx"],
      "persist_directory": "./knowledge_base/chroma_db"
    },
    "system": {
      "gpu_available": true,
      "gpu_name": "NVIDIA GeForce RTX 4070 Ti",
      "python_version": "3.9.0",
      "platform": "Windows"
    },
    "ui": {
      "port": 8501,
      "theme": "light"
    }
  },
  "message": "Configuration retrieved",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### PUT /api/config

Update application configuration.

**Request Body:**
```json
{
  "api": {
    "temperature": 0.8,
    "max_tokens": 1500
  },
  "ui": {
    "theme": "dark"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "updated_settings": ["api.temperature", "api.max_tokens", "ui.theme"],
    "current_config": { ... }
  },
  "message": "Configuration updated successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## System Endpoints

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-01-01T12:00:00Z",
    "services": {
      "deepseek_api": "connected",
      "knowledge_base": "active",
      "gpu": "available"
    }
  },
  "message": "System is healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /api/status

Get detailed system status.

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 65.8,
      "disk_usage": 23.1,
      "gpu_usage": 78.5,
      "gpu_memory": 12.3
    },
    "application": {
      "uptime": 3600,
      "active_sessions": 3,
      "total_messages": 150,
      "average_response_time": 2.1
    }
  },
  "message": "System status retrieved",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "success": false,
  "error": {
    "code": "API_ERROR",
    "message": "Failed to connect to DeepSeek API",
    "details": "Connection timeout after 30 seconds"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid request parameters
- `API_ERROR`: DeepSeek API connection or response error
- `FILE_PROCESSING_ERROR`: Document processing failed
- `KNOWLEDGE_BASE_ERROR`: Knowledge base operation failed
- `CONFIGURATION_ERROR`: Configuration loading/saving failed
- `SYSTEM_ERROR`: Internal server error

## Rate Limiting

Currently, there are no rate limits for local usage. For production deployments, consider implementing rate limiting based on your requirements.

## Streaming Responses

The chat endpoint supports streaming responses for real-time interaction:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "Hello", "use_kb": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

Streaming responses are sent as Server-Sent Events (SSE) with the following format:

```
data: {"chunk": "Hello", "finished": false}
data: {"chunk": " there", "finished": false}
data: {"chunk": "!", "finished": true}
```

## WebSocket Support

For real-time bidirectional communication, WebSocket support is available at:

```
ws://localhost:8000/ws/chat
```

WebSocket messages follow the same format as REST API requests and responses.
