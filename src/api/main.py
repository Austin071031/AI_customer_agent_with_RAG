"""
FastAPI Backend for AI Customer Agent.

This module provides the main FastAPI application with REST endpoints
for chat, knowledge base management, and configuration.
"""

import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from ..services.config_manager import ConfigManager
from ..services.deepseek_service import DeepSeekService
from ..services.knowledge_base import KnowledgeBaseManager
from ..services.chat_manager import ChatManager
from ..services.text_to_sql_service import TextToSQLService
from .endpoints import chat, knowledge_base, config, excel_files
from .state import app_state

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events for the application.
    """
    # Startup: Initialize services
    try:
        logger.info("Starting AI Customer Agent FastAPI backend...")
        
        # Initialize configuration manager
        config_manager = ConfigManager()
        app_state["config_manager"] = config_manager
        
        # Get settings
        settings = config_manager.settings
        
        # Initialize DeepSeek service
        from ..models.config_models import APIConfig
        api_config = APIConfig(
            api_key=settings.deepseek.api_key,
            base_url=settings.deepseek.base_url,
            model=settings.deepseek.model,
            temperature=settings.deepseek.temperature,
            max_tokens=settings.deepseek.max_tokens
        )
        deepseek_service = DeepSeekService(api_config)
        app_state["deepseek_service"] = deepseek_service
        
        # Initialize knowledge base manager
        kb_manager = KnowledgeBaseManager(
            persist_directory=settings.knowledge_base.persist_directory,
            sqlite_db_path=settings.database.sqlite_db_path
        )
        app_state["kb_manager"] = kb_manager
        
        # Initialize Text-to-SQL service with relational database support
        from ..services.sqlite_database_service import SQLiteDatabaseService
        sqlite_service = SQLiteDatabaseService(db_path=settings.database.sqlite_db_path)
        text_to_sql_service = TextToSQLService(deepseek_service, sqlite_service)
        app_state["text_to_sql_service"] = text_to_sql_service
        
        # Initialize chat manager with Text-to-SQL integration for intelligent query routing
        chat_manager = ChatManager(deepseek_service, kb_manager, text_to_sql_service)
        app_state["chat_manager"] = chat_manager
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Clean up resources
    logger.info("Shutting down AI Customer Agent FastAPI backend...")
    app_state.clear()


# Create FastAPI application
app = FastAPI(
    title="AI Customer Agent API",
    description="A local AI customer service agent with DeepSeek API integration and knowledge base support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.
    
    Returns a standardized error response for all unhandled exceptions.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )


# Include routers
app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"]
)

app.include_router(
    knowledge_base.router,
    prefix="/api/knowledge-base",
    tags=["Knowledge Base"]
)

app.include_router(
    config.router,
    prefix="/api/config",
    tags=["Configuration"]
)

app.include_router(
    excel_files.router,
    prefix="/api/excel-files",
    tags=["Excel Files"]
)


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        Dictionary with API information and status
    """
    return {
        "message": "AI Customer Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for the API and its services.
    
    Returns:
        Dictionary with health status of all components
    """
    try:
        # Check if services are initialized
        config_manager = app_state.get("config_manager")
        deepseek_service = app_state.get("deepseek_service")
        kb_manager = app_state.get("kb_manager")
        chat_manager = app_state.get("chat_manager")
        text_to_sql_service = app_state.get("text_to_sql_service")
        
        health_status = {
            "api": "healthy",
            "config_manager": "healthy" if config_manager else "unhealthy",
            "deepseek_service": "healthy" if deepseek_service else "unhealthy",
            "knowledge_base": "healthy" if kb_manager else "unhealthy",
            "chat_manager": "healthy" if chat_manager else "unhealthy",
            "text_to_sql_service": "healthy" if text_to_sql_service else "unhealthy"
        }
        
        
        # Determine overall health
        overall_health = all(
            status == "healthy" 
            for service, status in health_status.items() 
            if service != "api"  # API itself is always healthy if running
        )
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "timestamp": "2024-01-01T12:00:00Z",  # This would be dynamic in production
            "services": health_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "services": {
                    "api": "unhealthy",
                    "config_manager": "unknown",
                    "deepseek_service": "unknown",
                    "knowledge_base": "unknown",
                    "chat_manager": "unknown"
                }
            }
        )


@app.get("/info")
async def get_system_info():
    """
    Get system information and configuration summary.
    
    Returns:
        Dictionary with system information and configuration
    """
    try:
        config_manager = app_state.get("config_manager")
        if not config_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configuration manager not available"
            )
        
        settings = config_manager.settings
        
        return {
            "system": {
                "name": "AI Customer Agent",
                "version": "1.0.0",
                "environment": "development"  # This would be dynamic in production
            },
            "configuration": {
                "api_enabled": bool(app_state.get("deepseek_service")),
                "knowledge_base_enabled": bool(app_state.get("kb_manager")),
                "chat_enabled": bool(app_state.get("chat_manager")),
                "log_level": settings.app.log_level,
                "enable_gpu": settings.gpu.enable_cuda,
                "max_conversation_history": settings.chat.max_history_length
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system information: {str(e)}"
        )




if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
