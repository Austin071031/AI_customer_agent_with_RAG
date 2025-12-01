"""
Excel File Management Endpoints for AI Customer Agent.

This module provides REST API endpoints for managing Excel files stored in SQLite database
with dynamic table creation for relational data storage. Each Excel sheet is automatically
converted to a relational table with proper column type inference (integer, real, text, etc.),
enabling SQL queries and data analysis operations.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from ...models.excel_models import ExcelDocument, ExcelSheetData, ExcelSearchQuery
from ...services.knowledge_base import KnowledgeBaseManager
from ...services.sqlite_database_service import SQLiteDatabaseError
from ..state import app_state

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def get_kb_manager() -> KnowledgeBaseManager:
    """
    Get the knowledge base manager from application state.
    
    Returns:
        KnowledgeBaseManager instance
        
    Raises:
        HTTPException: If knowledge base manager is not available
    """
    kb_manager = app_state.get("kb_manager")
    if not kb_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base manager not available"
        )
    return kb_manager


@router.get("/", response_model=List[ExcelDocument])
async def list_excel_files(
    skip: int = Query(0, ge=0, description="Number of files to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of files to return")
) -> List[ExcelDocument]:
    """
    List all Excel files stored in SQLite database.
    
    Args:
        skip: Number of files to skip (for pagination)
        limit: Maximum number of files to return
        
    Returns:
        List of ExcelDocument objects
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        kb_manager = get_kb_manager()
        files = kb_manager.list_excel_files()
        
        # Apply pagination
        paginated_files = files[skip:skip + limit]
        
        logger.info(f"Listed {len(paginated_files)} Excel files (skip={skip}, limit={limit})")
        return paginated_files
        
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to list Excel files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list Excel files: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error listing Excel files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing Excel files"
        )


@router.get("/{file_id}", response_model=ExcelDocument)
async def get_excel_file(file_id: str) -> ExcelDocument:
    """
    Get specific Excel file details.
    
    Args:
        file_id: ID of the Excel file to retrieve
        
    Returns:
        ExcelDocument object
        
    Raises:
        HTTPException: If file not found or operation fails
    """
    try:
        kb_manager = get_kb_manager()
        excel_file = kb_manager.get_excel_file(file_id)
        
        if not excel_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Excel file with ID '{file_id}' not found"
            )
            
        logger.info(f"Retrieved Excel file: {file_id}")
        return excel_file
        
    except HTTPException:
        raise
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to retrieve Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Excel file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the Excel file"
        )


@router.delete("/{file_id}")
async def delete_excel_file(file_id: str) -> JSONResponse:
    """
    Delete Excel file from SQLite database.
    
    Args:
        file_id: ID of the Excel file to delete
        
    Returns:
        JSON response with deletion status
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        kb_manager = get_kb_manager()
        
        # Check if file exists first
        existing_file = kb_manager.get_excel_file(file_id)
        if not existing_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Excel file with ID '{file_id}' not found"
            )
        
        # Delete the file
        success = kb_manager.delete_excel_file(file_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete Excel file with ID '{file_id}'"
            )
            
        logger.info(f"Deleted Excel file: {file_id}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Excel file with ID '{file_id}' deleted successfully",
                "file_id": file_id,
                "status": "deleted"
            }
        )
        
    except HTTPException:
        raise
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to delete Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete Excel file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the Excel file"
        )


@router.get("/{file_id}/sheets", response_model=List[ExcelSheetData])
async def get_excel_file_sheets(
    file_id: str,
    sheet_name: Optional[str] = Query(None, description="Specific sheet name to retrieve")
) -> List[ExcelSheetData]:
    """
    Get sheet data for an Excel file.
    
    Args:
        file_id: ID of the Excel file
        sheet_name: Optional specific sheet name
        
    Returns:
        List of ExcelSheetData objects
        
    Raises:
        HTTPException: If file not found or operation fails
    """
    try:
        kb_manager = get_kb_manager()
        
        # Check if file exists first
        existing_file = kb_manager.get_excel_file(file_id)
        if not existing_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Excel file with ID '{file_id}' not found"
            )
        
        # Get sheet data
        sheets = kb_manager.get_sheet_data(file_id, sheet_name)
        
        logger.info(f"Retrieved {len(sheets)} sheets for Excel file: {file_id}")
        return sheets
        
    except HTTPException:
        raise
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to retrieve sheets for Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sheet data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving sheets for Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving sheet data"
        )


@router.post("/search")
async def search_excel_data(search_query: ExcelSearchQuery) -> JSONResponse:
    """
    Search for data within Excel files.
    
    Args:
        search_query: ExcelSearchQuery object with search parameters
        
    Returns:
        JSON response with search results
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        kb_manager = get_kb_manager()
        
        # Perform search
        results = kb_manager.search_excel_data(search_query)
        
        logger.info(f"Excel data search completed: '{search_query.query}' - {len(results)} results")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "query": search_query.query,
                "results": results,
                "result_count": len(results),
                "filters": {
                    "file_id": search_query.file_id,
                    "sheet_name": search_query.sheet_name,
                    "column_name": search_query.column_name,
                    "case_sensitive": search_query.case_sensitive,
                    "max_results": search_query.max_results
                }
            }
        )
        
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to search Excel data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search Excel data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error searching Excel data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while searching Excel data"
        )


@router.get("/{file_id}/search")
async def search_excel_file_data(
    file_id: str,
    query: str = Query(..., description="Search query string"),
    sheet_name: Optional[str] = Query(None, description="Specific sheet name to search within"),
    column_name: Optional[str] = Query(None, description="Specific column name to search within"),
    case_sensitive: bool = Query(False, description="Whether search should be case sensitive"),
    max_results: int = Query(50, ge=1, le=1000, description="Maximum number of results to return")
) -> JSONResponse:
    """
    Search for data within a specific Excel file.
    
    Args:
        file_id: ID of the Excel file to search within
        query: Search query string
        sheet_name: Optional specific sheet name to search within
        column_name: Optional specific column name to search within
        case_sensitive: Whether search should be case sensitive
        max_results: Maximum number of results to return
        
    Returns:
        JSON response with search results
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        kb_manager = get_kb_manager()
        
        # Check if file exists first
        existing_file = kb_manager.get_excel_file(file_id)
        if not existing_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Excel file with ID '{file_id}' not found"
            )
        
        # Create search query
        search_query = ExcelSearchQuery(
            query=query,
            file_id=file_id,
            sheet_name=sheet_name,
            column_name=column_name,
            case_sensitive=case_sensitive,
            max_results=max_results
        )
        
        # Perform search
        results = kb_manager.search_excel_data(search_query)
        
        logger.info(f"Excel file search completed: '{query}' in file {file_id} - {len(results)} results")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "file_id": file_id,
                "query": query,
                "results": results,
                "result_count": len(results),
                "filters": {
                    "sheet_name": sheet_name,
                    "column_name": column_name,
                    "case_sensitive": case_sensitive,
                    "max_results": max_results
                }
            }
        )
        
    except HTTPException:
        raise
    except SQLiteDatabaseError as e:
        logger.error(f"Failed to search Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search Excel file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error searching Excel file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while searching the Excel file"
        )
