"""
Knowledge Base endpoints for AI Customer Agent API.

This module provides REST endpoints for managing the knowledge base including
document upload, search, and management operations.
"""

import logging
import os
import tempfile
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...models.chat_models import KBDocument
from ...services.knowledge_base import KnowledgeBaseManager, KnowledgeBaseError
from ..dependencies import get_kb_manager

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF Document',
    '.txt': 'Text File',
    '.docx': 'Word Document',
    '.doc': 'Word Document',
    '.md': 'Markdown File',
    '.json': 'JSON File',
    '.csv': 'CSV File',
    '.xlsx': 'Excel File',
    '.xls': 'Excel File',
    '.xlsm': 'Excel File',
    '.xlsb': 'Excel File'
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


# Request/Response Models
class SearchRequest(BaseModel):
    """
    Request model for knowledge base search.
    
    Attributes:
        query: Search query string
        k: Number of similar documents to return (default: 3)
        similarity_threshold: Minimum similarity score (0.0 to 1.0, default: 0.5)
    """
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    k: int = Field(default=3, ge=1, le=1000, description="Number of similar documents to return")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How to reset password?",
                "k": 3,
                "similarity_threshold": 0.7
            }
        }
    }


class SearchResult(BaseModel):
    """
    Individual search result model.
    
    Attributes:
        document: The knowledge base document
        similarity_score: Similarity score (0.0 to 1.0)
        relevance_percentage: Relevance as percentage (0-100)
    """
    
    document: KBDocument = Field(..., description="Knowledge base document")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    relevance_percentage: int = Field(..., ge=0, le=100, description="Relevance percentage")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document": {
                    "id": "doc_123",
                    "content": "To reset your password, visit the settings page...",
                    "metadata": {"source": "user_guide"},
                    "file_path": "/documents/guide.pdf",
                    "file_type": "pdf",
                    "embedding": None
                },
                "similarity_score": 0.85,
                "relevance_percentage": 85
            }
        }
    }


class SearchResponse(BaseModel):
    """
    Response model for knowledge base search.
    
    Attributes:
        results: List of search results
        total_results: Total number of results found
        query: The original search query
        search_time_ms: Time taken for search in milliseconds
    """
    
    results: List[SearchResult] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "document": {
                            "id": "doc_123",
                            "content": "To reset your password...",
                            "metadata": {"source": "user_guide"},
                            "file_path": "/documents/guide.pdf",
                            "file_type": "pdf",
                            "embedding": None
                        },
                        "similarity_score": 0.85,
                        "relevance_percentage": 85
                    }
                ],
                "total_results": 1,
                "query": "How to reset password?",
                "search_time_ms": 150.5
            }
        }
    }


class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload.
    
    Attributes:
        message: Success message
        uploaded_files: List of successfully uploaded files
        failed_files: List of files that failed to upload with reasons
        total_processed: Total number of files processed
        total_successful: Number of successfully processed files
    """
    
    message: str = Field(..., description="Upload result message")
    uploaded_files: List[str] = Field(..., description="List of successfully uploaded files")
    failed_files: List[dict] = Field(..., description="List of failed files with reasons")
    total_processed: int = Field(..., description="Total number of files processed")
    total_successful: int = Field(..., description="Number of successfully processed files")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Documents processed successfully",
                "uploaded_files": ["user_guide.pdf", "faq.txt"],
                "failed_files": [
                    {"filename": "image.png", "reason": "Unsupported file type"}
                ],
                "total_processed": 3,
                "total_successful": 2
            }
        }
    }


class ClearKBResponse(BaseModel):
    """
    Response model for clearing knowledge base.
    
    Attributes:
        message: Confirmation message
        cleared_documents: Number of documents cleared
    """
    
    message: str = Field(..., description="Confirmation message")
    cleared_documents: int = Field(..., description="Number of documents cleared")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Knowledge base cleared successfully",
                "cleared_documents": 15
            }
        }
    }


class KBInfoResponse(BaseModel):
    """
    Response model for knowledge base information.
    
    Attributes:
        total_documents: Total number of documents in knowledge base
        document_types: Count of documents by file type
        total_size_bytes: Total size of documents in bytes
        embedding_model: Name of the embedding model used
        collection_name: Name of the vector collection
        last_updated: Last update timestamp (ISO format)
    """
    
    total_documents: int = Field(..., description="Total number of documents")
    document_types: dict = Field(..., description="Count of documents by file type")
    total_size_bytes: int = Field(..., description="Total size in bytes")
    embedding_model: str = Field(..., description="Embedding model name")
    collection_name: str = Field(..., description="Vector collection name")
    last_updated: str = Field(..., description="Last update timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_documents": 15,
                "document_types": {"pdf": 5, "txt": 8, "docx": 2},
                "total_size_bytes": 5242880,
                "embedding_model": "all-MiniLM-L6-v2",
                "collection_name": "documents",
                "last_updated": "2024-01-01T12:00:00Z"
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error: Error type
        message: Human-readable error message
        detail: Additional error details (optional)
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")


def validate_file_extension(filename: str) -> bool:
    """
    Validate if file extension is supported.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if file extension is supported, False otherwise
    """
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """
    Validate if file size is within limits.
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        True if file size is within limits, False otherwise
    """
    return file_size <= MAX_FILE_SIZE


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search knowledge base",
    description="Search the knowledge base for documents similar to the query.",
    responses={
        200: {"description": "Successfully searched knowledge base"},
        400: {"model": ErrorResponse, "description": "Invalid request or empty query"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def search_knowledge_base(request: SearchRequest) -> SearchResponse:
    """
    Search the knowledge base for similar documents.
    
    This endpoint performs semantic search on the knowledge base using
    vector embeddings to find documents similar to the query.
    
    Args:
        request: Search request containing query and parameters
        
    Returns:
        SearchResponse with search results and metadata
        
    Raises:
        HTTPException: If search fails or service is unavailable
    """
    import time
    
    try:
        logger.info(f"Searching knowledge base for: {request.query[:50]}...")
        start_time = time.time()
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Search for similar documents
        similar_docs = kb_manager.search_similar(
            query=request.query,
            k=request.k
        )
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        # Format results with similarity scores
        results = []
        for doc in similar_docs:
            similarity_score = doc.metadata.get('similarity_score', 0.0)
            
            # Filter by similarity threshold
            if similarity_score >= request.similarity_threshold:
                relevance_percentage = int(similarity_score * 100)
                results.append(SearchResult(
                    document=doc,
                    similarity_score=similarity_score,
                    relevance_percentage=relevance_percentage
                ))
        
        logger.info(f"Found {len(results)} relevant documents in {search_time_ms:.2f}ms")
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            search_time_ms=search_time_ms
        )
        
    except KnowledgeBaseError as e:
        logger.error(f"Knowledge base search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Search failed: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search_knowledge_base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload documents to knowledge base",
    description="Upload and process documents to add to the knowledge base.",
    responses={
        200: {"description": "Successfully processed documents"},
        400: {"model": ErrorResponse, "description": "Invalid files or unsupported formats"},
        413: {"model": ErrorResponse, "description": "File too large"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Files to upload to knowledge base"),
    overwrite: bool = Form(default=False, description="Overwrite existing documents with same name"),
    chunk_size: int = Form(default=1000, description="Chunk size for document chunking (in characters)"),
    chunk_overlap: int = Form(default=200, description="Overlap between chunks (in characters)")
) -> DocumentUploadResponse:
    """
    Upload documents to the knowledge base with intelligent file type routing.
    
    This endpoint accepts multiple files, validates them, and processes them based on file type:
    - Excel files (.xlsx, .xls, .xlsm, .xlsb): Processed and stored in SQLite database with 
      dynamic table creation for each sheet, supporting relational data operations
    - Other document types (PDF, TXT, DOCX, etc.): Processed and stored in vector database 
      with semantic search capabilities
    
    Args:
        files: List of files to upload
        overwrite: Whether to overwrite existing documents with same names
        
    Returns:
        DocumentUploadResponse with upload results, including separate tracking for 
        Excel files (stored in relational tables) and other documents (stored in vector DB)
        
    Raises:
        HTTPException: If upload fails or service is unavailable
    """
    try:
        logger.info(f"Processing {len(files)} files for knowledge base upload")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        uploaded_files = []
        failed_files = []
        
        for file in files:
            try:
                # Validate file extension
                if not validate_file_extension(file.filename):
                    failed_files.append({
                        "filename": file.filename,
                        "reason": f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
                    })
                    continue
                
                # Validate file size
                file_content = await file.read()
                if not validate_file_size(len(file_content)):
                    failed_files.append({
                        "filename": file.filename,
                        "reason": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                    })
                    continue
                
                # Create temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                try:
                    # Add document to knowledge base with chunking parameters
                    result = kb_manager.add_documents([temp_file_path], chunk_size, chunk_overlap)
                    
                    # Check if file was successfully processed (either as Excel or document)
                    if result.get('excel_files') or result.get('documents'):
                        uploaded_files.append(file.filename)
                        logger.info(f"Successfully processed: {file.filename}")
                    else:
                        # Check for specific errors
                        if result.get('errors'):
                            error_msg = result['errors'][0] if result['errors'] else "Unknown processing error"
                            failed_files.append({
                                "filename": file.filename,
                                "reason": error_msg
                            })
                        else:
                            failed_files.append({
                                "filename": file.filename,
                                "reason": "Failed to process document content - no results returned"
                            })
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                failed_files.append({
                    "filename": file.filename,
                    "reason": f"Processing error: {str(e)}"
                })
        
        # Prepare response
        total_processed = len(uploaded_files) + len(failed_files)
        total_successful = len(uploaded_files)
        
        if total_successful > 0:
            message = f"Successfully processed {total_successful} out of {total_processed} files"
        else:
            message = "No files were successfully processed"
        
        return DocumentUploadResponse(
            message=message,
            uploaded_files=uploaded_files,
            failed_files=failed_files,
            total_processed=total_processed,
            total_successful=total_successful
        )
        
    except KnowledgeBaseError as e:
        logger.error(f"Knowledge base upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload failed: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/",
    response_model=ClearKBResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear knowledge base",
    description="Remove all documents from the knowledge base (both vector store and SQLite database).",
    responses={
        200: {"description": "Successfully cleared knowledge base"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def clear_knowledge_base() -> ClearKBResponse:
    """
    Clear all documents from the knowledge base.
    
    This endpoint removes all documents and their embeddings from the
    knowledge base, effectively resetting it to an empty state.
    
    Returns:
        ClearKBResponse with confirmation and count of cleared documents
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Clearing knowledge base (both vector store and SQLite)")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Get current document count before clearing
        current_count = kb_manager.get_document_count()
        
        # Clear the knowledge base (both vector store and SQLite)
        success = kb_manager.clear_knowledge_base()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear knowledge base"
            )
        
        return ClearKBResponse(
            message="Knowledge base cleared successfully (both vector store and SQLite database)",
            cleared_documents=current_count
        )
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.delete(
    "/vector-store",
    response_model=ClearKBResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear vector store only",
    description="Remove only non-Excel documents from the vector store (does not affect Excel files in SQLite).",
    responses={
        200: {"description": "Successfully cleared vector store"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def clear_vector_store() -> ClearKBResponse:
    """
    Clear only the vector store for non-Excel documents.
    
    This endpoint removes all documents from the vector store (PDF, TXT, DOCX, etc.)
    but does not affect Excel files stored in SQLite database.
    
    Returns:
        ClearKBResponse with confirmation and count of cleared documents
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Clearing vector store only (non-Excel documents)")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Get current vector document count before clearing
        stats = kb_manager.get_statistics()
        vector_count = stats.get('vector_documents', 0)
        
        # Clear only the vector store
        success = kb_manager.clear_vector_store()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear vector store"
            )
        
        return ClearKBResponse(
            message="Vector store cleared successfully (Excel files in SQLite are unaffected)",
            cleared_documents=vector_count
        )
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {str(e)}"
        )


@router.delete(
    "/sqlite-database",
    response_model=ClearKBResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear SQLite database only",
    description="Remove only Excel files from SQLite database (does not affect non-Excel documents in vector store).",
    responses={
        200: {"description": "Successfully cleared SQLite database"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def clear_sqlite_database() -> ClearKBResponse:
    """
    Clear only the SQLite database for Excel files.
    
    This endpoint removes all Excel files from the SQLite database
    but does not affect non-Excel documents stored in the vector store.
    
    Returns:
        ClearKBResponse with confirmation and count of cleared documents
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Clearing SQLite database only (Excel files)")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Get current Excel document count before clearing
        stats = kb_manager.get_statistics()
        excel_count = stats.get('excel_documents', 0)
        
        # Clear only the SQLite database
        success = kb_manager.clear_sqlite_database()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear SQLite database"
            )
        
        return ClearKBResponse(
            message="SQLite database cleared successfully (non-Excel documents in vector store are unaffected)",
            cleared_documents=excel_count
        )
        
    except Exception as e:
        logger.error(f"Error clearing SQLite database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear SQLite database: {str(e)}"
        )


@router.get(
    "/info",
    response_model=KBInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get knowledge base information",
    description="Retrieve information and statistics about the knowledge base.",
    responses={
        200: {"description": "Successfully retrieved knowledge base information"},
        503: {"model": ErrorResponse, "description": "Knowledge base service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_knowledge_base_info() -> KBInfoResponse:
    """
    Get information and statistics about the knowledge base.
    
    This endpoint provides detailed information about the knowledge base
    including document counts, file types, size, and configuration.
    
    Returns:
        KBInfoResponse with knowledge base information
        
    Raises:
        HTTPException: If service is unavailable or error occurs
    """
    try:
        logger.info("Retrieving knowledge base information")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Get knowledge base statistics
        stats = kb_manager.get_statistics()
        
        return KBInfoResponse(
            total_documents=stats.get('total_documents', 0),
            document_types=stats.get('document_types', {}),
            total_size_bytes=stats.get('total_size_bytes', 0),
            embedding_model=stats.get('embedding_model', 'unknown'),
            collection_name=stats.get('collection_name', 'unknown'),
            last_updated=stats.get('last_updated', '2024-01-01T12:00:00Z')
        )
        
    except Exception as e:
        logger.error(f"Error retrieving knowledge base info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge base information: {str(e)}"
        )


@router.get(
    "/supported-formats",
    summary="Get supported file formats",
    description="Retrieve list of supported file formats for knowledge base upload.",
    responses={
        200: {"description": "Successfully retrieved supported formats"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_supported_formats():
    """
    Get list of supported file formats for knowledge base upload.
    
    Returns:
        Dictionary with supported file formats and their descriptions
    """
    try:
        logger.debug("Retrieving supported file formats")
        
        return {
            "supported_formats": SUPPORTED_EXTENSIONS,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "max_file_size_bytes": MAX_FILE_SIZE
        }
        
    except Exception as e:
        logger.error(f"Error retrieving supported formats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve supported formats: {str(e)}"
        )


# Health check endpoint for knowledge base service
@router.get(
    "/health",
    summary="Knowledge base health check",
    description="Check the health status of the knowledge base service.",
    responses={
        200: {"description": "Knowledge base service is healthy"},
        503: {"model": ErrorResponse, "description": "Knowledge base service is unhealthy"}
    }
)
async def knowledge_base_health_check():
    """
    Health check for the knowledge base service.
    
    This endpoint performs a health check on the knowledge base manager
    to ensure it is functioning properly.
    
    Returns:
        Dictionary with health status
        
    Raises:
        HTTPException: If knowledge base service is unhealthy
    """
    try:
        logger.debug("Performing knowledge base health check")
        
        # Get knowledge base manager instance
        kb_manager = get_kb_manager()
        
        # Perform health check
        is_healthy = kb_manager.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge base health check failed"
            )
        
        return {
            "status": "healthy",
            "service": "knowledge_base",
            "timestamp": "2024-01-01T12:00:00Z",  # This would be dynamic in production
            "document_count": kb_manager.get_document_count()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge base health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Knowledge base health check failed: {str(e)}"
        )
