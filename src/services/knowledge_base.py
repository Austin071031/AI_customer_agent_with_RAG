"""
Enhanced Knowledge Base Manager for AI Customer Agent.

This module provides enhanced knowledge base management with file type detection
and intelligent routing: Excel files to SQLite database, other documents to vector storage.
Supports multiple file formats and GPU-accelerated embeddings.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from ..models.chat_models import KBDocument
from ..models.excel_models import ExcelDocument, ExcelSearchQuery, ExcelSheetData
from .sqlite_database_service import SQLiteDatabaseService, SQLiteDatabaseError


class KnowledgeBaseError(Exception):
    """Custom exception for knowledge base related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class DocumentProcessor:
    """
    Handles document processing for various file formats.
    
    This class provides methods to extract text from different document formats
    including PDF, TXT, DOCX, and supports additional formats as needed.
    """
    
    def __init__(self):
        """Initialize the document processor with required libraries."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.pdf', '.txt', '.docx', '.doc', '.md'}
        
    def extract_text_from_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content from a file based on its format.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            KnowledgeBaseError: If file format is not supported or processing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise KnowledgeBaseError(f"File not found: {file_path}")
            
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise KnowledgeBaseError(f"Unsupported file format: {file_extension}")
            
        try:
            # Extract basic file metadata
            metadata = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_extension': file_extension,
                'last_modified': file_path.stat().st_mtime
            }
            
            # Process based on file type
            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path), metadata
            elif file_extension == '.txt' or file_extension == '.md':
                return self._extract_text_file(file_path), metadata
            elif file_extension in ['.docx', '.doc']:
                return self._extract_docx_text(file_path), metadata
            else:
                raise KnowledgeBaseError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise KnowledgeBaseError(f"Failed to process file {file_path}: {str(e)}")
            
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            raise KnowledgeBaseError("PyPDF2 not installed for PDF processing")
        except Exception as e:
            raise KnowledgeBaseError(f"PDF processing error: {str(e)}")
            
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                raise KnowledgeBaseError(f"Text file encoding error: {str(e)}")
        except Exception as e:
            raise KnowledgeBaseError(f"Text file processing error: {str(e)}")
            
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files using python-docx."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            raise KnowledgeBaseError("python-docx not installed for DOCX processing")
        except Exception as e:
            raise KnowledgeBaseError(f"DOCX processing error: {str(e)}")


class EmbeddingService:
    """
    Handles text embedding generation with GPU acceleration.
    
    This service uses sentence-transformers models to generate embeddings
    and automatically utilizes GPU when available for faster processing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = self._setup_device()
        self.model = self._load_model()
        
    def _setup_device(self) -> torch.device:
        """
        Configure PyTorch for GPU acceleration if available.
        
        Returns:
            torch.device: The device to use for computations
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU for embeddings")
        return device
        
    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence-transformers model.
        
        Returns:
            SentenceTransformer: Loaded model instance
            
        Raises:
            KnowledgeBaseError: If model loading fails
        """
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise KnowledgeBaseError(f"Failed to load embedding model: {str(e)}")
            
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            KnowledgeBaseError: If embedding generation fails
        """
        if not texts:
            return []
            
        try:
            # Use batch processing for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Convert numpy arrays to lists
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return [embedding.tolist() for embedding in embeddings]
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise KnowledgeBaseError(f"Embedding generation failed: {str(e)}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model and device."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "gpu_available": torch.cuda.is_available()
        }


class EnhancedKnowledgeBaseManager:
    """
    Enhanced Knowledge Base Manager with file type detection and routing.
    
    This class provides intelligent file routing: Excel files to SQLite database,
    other documents to vector storage. Supports multiple file formats and
    GPU-accelerated embeddings.
    """
    
    def __init__(self, persist_directory: str = "./knowledge_base/chroma_db",
                 sqlite_db_path: str = "./excel_database.db"):
        """
        Initialize the enhanced knowledge base manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            sqlite_db_path: Path to SQLite database for Excel files
        """
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.sqlite_service = SQLiteDatabaseService(sqlite_db_path)
        self.vector_store = self._initialize_vector_store()
        
    def _initialize_vector_store(self) -> chromadb.Collection:
        """
        Initialize the ChromaDB vector store.
        
        Returns:
            chromadb.Collection: ChromaDB collection for document storage
            
        Raises:
            KnowledgeBaseError: If vector store initialization fails
        """
        try:
            # Create persistence directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            collection = client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "AI Customer Agent Knowledge Base"}
            )
            
            self.logger.info(f"Initialized vector store at: {self.persist_directory}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise KnowledgeBaseError(f"Vector store initialization failed: {str(e)}")
            
    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect file type and determine storage method.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type category: 'excel' or 'document'
        """
        file_extension = Path(file_path).suffix.lower()
        excel_extensions = {'.xlsx', '.xls', '.xlsm', '.xlsb'}
        
        if file_extension in excel_extensions:
            return 'excel'
        else:
            return 'document'
            
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process and add documents to appropriate storage based on file type.
        
        Args:
            file_paths: List of file paths to add to the knowledge base
            
        Returns:
            Dictionary with processing results for both Excel and document files
            
        Raises:
            KnowledgeBaseError: If document processing or addition fails
        """
        if not file_paths:
            raise KnowledgeBaseError("No file paths provided")
            
        results = {
            'excel_files': [],
            'documents': [],
            'errors': []
        }
        
        try:
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.exists():
                    error_msg = f"File not found: {file_path}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    continue
                    
                file_type = self._detect_file_type(str(file_path))
                
                try:
                    if file_type == 'excel':
                        excel_result = self._process_excel_file(str(file_path))
                        results['excel_files'].append(excel_result)
                    else:
                        document_result = self._process_document_file(str(file_path))
                        results['documents'].append(document_result)
                        
                except Exception as e:
                    error_msg = f"Failed to process file {file_path}: {str(e)}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
            self.logger.info(f"Processing complete: {len(results['excel_files'])} Excel files, "
                           f"{len(results['documents'])} documents, {len(results['errors'])} errors")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise KnowledgeBaseError(f"Document addition failed: {str(e)}")
            
    def _process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process Excel file and store in SQLite database with dynamic table creation.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with processing results including dynamic table information
            
        Raises:
            KnowledgeBaseError: If Excel file processing fails
        """
        try:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            file_size = file_path_obj.stat().st_size
            
            # Extract sheet names from Excel file
            import openpyxl
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            sheet_names = workbook.sheetnames
            workbook.close()
            
            # Prepare metadata
            metadata = {
                'file_name': file_name,
                'file_size': file_size,
                'file_path': file_path,
                'uploaded_by': 'system',
                'description': f'Excel file: {file_name}',
                'dynamic_tables_created': True
            }
            
            # Store in SQLite database with dynamic table creation
            storage_result = self.sqlite_service.store_excel_file_with_dynamic_tables(
                file_path=file_path,
                file_name=file_name,
                file_size=file_size,
                sheet_names=sheet_names,
                metadata=metadata
            )
            
            result = {
                'file_id': storage_result['file_id'],
                'file_name': file_name,
                'file_size': file_size,
                'sheet_names': sheet_names,
                'storage_type': 'sqlite',
                'status': 'success',
                'tables_created': storage_result['tables_created'],
                'total_sheets': storage_result['total_sheets']
            }
            
            self.logger.info(f"Successfully processed Excel file with dynamic tables: {file_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process Excel file {file_path}: {str(e)}")
            raise KnowledgeBaseError(f"Excel file processing failed: {str(e)}")
            
    def _process_document_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process document file and store in vector database.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Extract text and metadata
            text, file_metadata = self.document_processor.extract_text_from_file(file_path)
            
            if not text.strip():
                raise KnowledgeBaseError(f"Empty content in file: {file_path}")
                
            # Create document ID
            doc_id = str(uuid.uuid4())
            
            # Create KBDocument
            kb_doc = KBDocument(
                id=doc_id,
                content=text,
                metadata=file_metadata,
                file_path=file_path,
                file_type=Path(file_path).suffix.lower()
            )
            
            # Generate embedding
            embedding = self.embedding_service.generate_embeddings([text])[0]
            kb_doc.embedding = embedding
            
            # Prepare data for ChromaDB
            document_metadata = {
                "file_path": file_path,
                "file_type": kb_doc.file_type,
                "file_size": file_metadata.get('file_size', 0),
                "document_id": doc_id
            }
            
            # Add to vector store
            self.vector_store.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[document_metadata],
                ids=[doc_id]
            )
            
            result = {
                'document_id': doc_id,
                'file_name': file_metadata['file_name'],
                'file_size': file_metadata['file_size'],
                'file_type': kb_doc.file_type,
                'storage_type': 'vector',
                'status': 'success'
            }
            
            self.logger.info(f"Successfully processed document file: {file_metadata['file_name']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process document file {file_path}: {str(e)}")
            raise KnowledgeBaseError(f"Document file processing failed: {str(e)}")
            
    def search_similar(self, query: str, k: int = 3) -> List[KBDocument]:
        """
        Search for similar documents in the knowledge base (vector storage only).
        
        Args:
            query: Search query string
            k: Number of similar documents to return
            
        Returns:
            List of KBDocument objects sorted by similarity
            
        Raises:
            KnowledgeBaseError: If search operation fails
        """
        if not query.strip():
            raise KnowledgeBaseError("Empty query provided")
            
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embeddings([query])[0]
            
            # Search in vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to KBDocument objects
            similar_documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_content, metadata, distance) in enumerate(
                    zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                ):
                    kb_doc = KBDocument(
                        id=metadata['document_id'],
                        content=doc_content,
                        metadata=metadata,
                        file_path=metadata['file_path'],
                        file_type=metadata['file_type'],
                        embedding=None  # Not returning embeddings for search results
                    )
                    # Store similarity score in metadata
                    kb_doc.metadata['similarity_score'] = 1 - distance
                    similar_documents.append(kb_doc)
                    
            self.logger.info(f"Found {len(similar_documents)} similar documents for query")
            return similar_documents
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise KnowledgeBaseError(f"Search operation failed: {str(e)}")
            
    def search_excel_data(self, search_query: ExcelSearchQuery) -> List[Dict[str, Any]]:
        """
        Search for data within Excel files stored in SQLite database.
        
        Args:
            search_query: ExcelSearchQuery object with search parameters
            
        Returns:
            List of search results with file and location information
            
        Raises:
            KnowledgeBaseError: If search operation fails
        """
        try:
            results = self.sqlite_service.search_excel_data(
                query=search_query.query,
                file_id=search_query.file_id,
                sheet_name=search_query.sheet_name,
                column_name=search_query.column_name,
                case_sensitive=search_query.case_sensitive,
                max_results=search_query.max_results
            )
            
            self.logger.info(f"Found {len(results)} Excel data matches for query")
            return results
            
        except SQLiteDatabaseError as e:
            self.logger.error(f"Excel data search failed: {str(e)}")
            raise KnowledgeBaseError(f"Excel data search failed: {str(e)}")
            
    def list_excel_files(self) -> List[ExcelDocument]:
        """
        List all Excel files stored in SQLite database.
        
        Returns:
            List of ExcelDocument objects
        """
        try:
            return self.sqlite_service.list_excel_files()
        except SQLiteDatabaseError as e:
            self.logger.error(f"Failed to list Excel files: {str(e)}")
            raise KnowledgeBaseError(f"Failed to list Excel files: {str(e)}")
            
    def get_excel_file(self, file_id: str) -> Optional[ExcelDocument]:
        """
        Get specific Excel file details from SQLite database.
        
        Args:
            file_id: ID of the Excel file to retrieve
            
        Returns:
            ExcelDocument object if found, None otherwise
        """
        try:
            return self.sqlite_service.get_excel_file(file_id)
        except SQLiteDatabaseError as e:
            self.logger.error(f"Failed to get Excel file: {str(e)}")
            raise KnowledgeBaseError(f"Failed to get Excel file: {str(e)}")
            
    def delete_excel_file(self, file_id: str) -> bool:
        """
        Delete Excel file from SQLite database.
        
        Args:
            file_id: ID of the Excel file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            return self.sqlite_service.delete_excel_file(file_id)
        except SQLiteDatabaseError as e:
            self.logger.error(f"Failed to delete Excel file: {str(e)}")
            raise KnowledgeBaseError(f"Failed to delete Excel file: {str(e)}")
            
    def get_sheet_data(self, file_id: str, sheet_name: Optional[str] = None) -> List[ExcelSheetData]:
        """
        Get sheet data for an Excel file.
        
        Args:
            file_id: ID of the Excel file
            sheet_name: Optional specific sheet name
            
        Returns:
            List of ExcelSheetData objects
        """
        try:
            return self.sqlite_service.get_sheet_data(file_id, sheet_name)
        except SQLiteDatabaseError as e:
            self.logger.error(f"Failed to get sheet data: {str(e)}")
            raise KnowledgeBaseError(f"Failed to get sheet data: {str(e)}")
            
    def clear_knowledge_base(self) -> Dict[str, bool]:
        """
        Clear all documents from both storage systems.
        
        Returns:
            Dictionary with clear operation results for both storage systems
        """
        results = {
            'vector_store': False,
            'sqlite_database': False
        }
        
        try:
            # Clear vector store
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection("knowledge_base")
            self.vector_store = self._initialize_vector_store()
            results['vector_store'] = True
            self.logger.info("Vector store cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear vector store: {str(e)}")
            
        try:
            # Note: SQLite database clear would require recreating tables
            # For now, we'll just log that SQLite clear is not implemented
            self.logger.warning("SQLite database clear requires manual table recreation")
            results['sqlite_database'] = False
            
        except Exception as e:
            self.logger.error(f"Failed to clear SQLite database: {str(e)}")
            
        return results
            
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the knowledge base.
        
        Returns:
            Dictionary containing knowledge base statistics and information
        """
        try:
            # Get vector store info
            vector_count = self.vector_store.count()
            vector_info = {
                "document_count": vector_count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_service.get_model_info(),
                "supported_formats": list(self.document_processor.supported_formats)
            }
            
            # Get SQLite database info
            sqlite_info = self.sqlite_service.get_database_info()
            
            return {
                "vector_store": vector_info,
                "sqlite_database": sqlite_info,
                "total_documents": vector_count + sqlite_info.get('document_count', 0),
                "health_status": self.health_check()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base info: {str(e)}")
            return {"error": str(e)}
            
    def get_document_count(self) -> int:
        """
        Get the total number of documents in both storage systems.

        Returns:
            Total number of documents
        """
        try:
            vector_count = self.vector_store.count()
            sqlite_info = self.sqlite_service.get_database_info()
            sqlite_count = sqlite_info.get('document_count', 0)
            return vector_count + sqlite_count
        except Exception as e:
            self.logger.error(f"Failed to get document count: {str(e)}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the knowledge base.

        Returns:
            Dictionary containing knowledge base statistics
        """
        try:
            vector_count = self.vector_store.count()
            sqlite_info = self.sqlite_service.get_database_info()
            
            return {
                'total_documents': vector_count + sqlite_info.get('document_count', 0),
                'vector_documents': vector_count,
                'excel_documents': sqlite_info.get('document_count', 0),
                'total_sheets': sqlite_info.get('sheet_count', 0),
                'total_size_bytes': sqlite_info.get('total_size_bytes', 0),
                'embedding_model': self.embedding_service.model_name,
                'gpu_available': torch.cuda.is_available(),
                'last_updated': '2024-01-01T12:00:00Z'  # This would need to be tracked
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {
                'total_documents': 0,
                'vector_documents': 0,
                'excel_documents': 0,
                'total_sheets': 0,
                'total_size_bytes': 0,
                'embedding_model': self.embedding_service.model_name,
                'gpu_available': False,
                'last_updated': '2024-01-01T12:00:00Z'
            }

    def health_check(self) -> bool:
        """
        Perform a comprehensive health check on the knowledge base.

        Returns:
            True if knowledge base is healthy, False otherwise
        """
        try:
            # Test vector store
            vector_info = self.get_knowledge_base_info()
            embedding_info = self.embedding_service.get_model_info()
            
            # Test SQLite database
            sqlite_health = self.sqlite_service.health_check()
            
            # Check if all essential components are working
            if (isinstance(vector_info, dict) and 
                isinstance(embedding_info, dict) and 
                self.vector_store is not None and
                sqlite_health):
                return True
            return False

        except Exception:
            return False

# For backward compatibility, alias the enhanced class as KnowledgeBaseManager
KnowledgeBaseManager = EnhancedKnowledgeBaseManager
