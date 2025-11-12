"""
Knowledge Base Manager for AI Customer Agent.

This module provides the knowledge base management functionality including
document processing, embedding generation, and vector search using ChromaDB.
Supports multiple file formats and GPU-accelerated embeddings.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from ..models.chat_models import KBDocument


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


class KnowledgeBaseManager:
    """
    Manages the knowledge base with document processing and vector search.
    
    This class provides the main interface for knowledge base operations
    including adding documents, searching, and managing the vector database.
    """
    
    def __init__(self, persist_directory: str = "./knowledge_base/chroma_db"):
        """
        Initialize the knowledge base manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
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
            
    def add_documents(self, file_paths: List[str]) -> List[KBDocument]:
        """
        Process and add documents to the knowledge base.
        
        Args:
            file_paths: List of file paths to add to the knowledge base
            
        Returns:
            List of KBDocument objects representing the processed documents
            
        Raises:
            KnowledgeBaseError: If document processing or addition fails
        """
        if not file_paths:
            raise KnowledgeBaseError("No file paths provided")
            
        processed_documents = []
        texts_to_embed = []
        document_metadatas = []
        
        try:
            # Process each document
            for file_path in file_paths:
                self.logger.info(f"Processing document: {file_path}")
                
                # Extract text and metadata
                text, file_metadata = self.document_processor.extract_text_from_file(file_path)
                
                if not text.strip():
                    self.logger.warning(f"Empty content in file: {file_path}")
                    continue
                    
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
                
                processed_documents.append(kb_doc)
                texts_to_embed.append(text)
                document_metadatas.append({
                    "file_path": file_path,
                    "file_type": kb_doc.file_type,
                    "file_size": file_metadata.get('file_size', 0),
                    "document_id": doc_id
                })
                
            if not processed_documents:
                raise KnowledgeBaseError("No valid documents found to process")
                
            # Generate embeddings in batch for efficiency
            self.logger.info(f"Generating embeddings for {len(texts_to_embed)} documents")
            embeddings = self.embedding_service.generate_embeddings(texts_to_embed)
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in processed_documents]
            documents = [doc.content for doc in processed_documents]
            metadatas = document_metadatas
            
            # Add to vector store
            self.vector_store.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update KBDocuments with their embeddings
            for i, doc in enumerate(processed_documents):
                doc.embedding = embeddings[i]
                
            self.logger.info(f"Successfully added {len(processed_documents)} documents to knowledge base")
            return processed_documents
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise KnowledgeBaseError(f"Document addition failed: {str(e)}")
            
    def search_similar(self, query: str, k: int = 3) -> List[KBDocument]:
        """
        Search for similar documents in the knowledge base.
        
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
            
    def clear_knowledge_base(self) -> bool:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # ChromaDB doesn't have a direct clear method, so we recreate the collection
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection("knowledge_base")
            
            # Reinitialize the collection
            self.vector_store = self._initialize_vector_store()
            
            self.logger.info("Knowledge base cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear knowledge base: {str(e)}")
            return False
            
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the knowledge base.
        
        Returns:
            Dictionary containing knowledge base statistics and information
        """
        try:
            # Get collection count (this might be approximate in ChromaDB)
            count = self.vector_store.count()
            
            return {
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_service.get_model_info(),
                "supported_formats": list(self.document_processor.supported_formats)
            }
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base info: {str(e)}")
            return {"error": str(e)}
            
    def get_document_count(self) -> int:
        """
        Get the number of documents in the knowledge base.

        Returns:
            Number of documents in the knowledge base
        """
        try:
            return self.vector_store.count()
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
            count = self.vector_store.count()
            
            # Get document type distribution (this would require querying metadata)
            # For now, we return basic stats
            return {
                'total_documents': count,
                'document_types': {},  # This would require additional logic to populate
                'total_size_bytes': 0,  # This would require additional logic
                'embedding_model': self.embedding_service.model_name,
                'collection_name': 'knowledge_base',
                'last_updated': '2024-01-01T12:00:00Z'  # This would need to be tracked
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {
                'total_documents': 0,
                'document_types': {},
                'total_size_bytes': 0,
                'embedding_model': self.embedding_service.model_name,
                'collection_name': 'knowledge_base',
                'last_updated': '2024-01-01T12:00:00Z'
            }

    def health_check(self) -> bool:
        """
        Perform a health check on the knowledge base.

        Returns:
            True if knowledge base is healthy, False otherwise
        """
        try:
            # Test basic operations
            info = self.get_knowledge_base_info()
            embedding_info = self.embedding_service.get_model_info()

            # Check if essential components are working
            if (isinstance(info, dict) and 
                isinstance(embedding_info, dict) and 
                self.vector_store is not None):
                return True
            return False

        except Exception:
            return False
