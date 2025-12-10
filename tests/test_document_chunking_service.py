"""
Unit tests for Document Chunking Service.

This module provides comprehensive unit tests for the DocumentChunkingService
and related classes to ensure proper functionality of document chunking for
PDF, TXT, and DOCX files.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.document_chunking_service import (
    DocumentChunkingService,
    DocumentChunkingError,
    SentenceAwareChunker,
    ParagraphChunker,
    Chunk
)


class TestSentenceAwareChunker(unittest.TestCase):
    """Test cases for SentenceAwareChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = SentenceAwareChunker(chunk_size=100, chunk_overlap=20)
        
    def test_split_into_sentences(self):
        """Test splitting text into sentences."""
        text = "Hello world. This is a test! Another sentence? Yes, it is."
        sentences = self.chunker._split_into_sentences(text)
        expected = ["Hello world.", "This is a test!", "Another sentence?", "Yes, it is."]
        self.assertEqual(sentences, expected)
        
    def test_split_into_sentences_empty(self):
        """Test splitting empty text."""
        text = ""
        sentences = self.chunker._split_into_sentences(text)
        self.assertEqual(sentences, [])
        
    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = "This is a short text that should fit in one chunk."
        metadata = {"source": "test"}
        chunks = self.chunker.chunk(text, metadata)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].metadata["source"], "test")
        self.assertEqual(chunks[0].chunk_index, 0)
        
    def test_chunk_large_text(self):
        """Test chunking text larger than chunk size."""
        # Create text with multiple sentences that will exceed chunk size
        sentences = [f"Sentence {i}. " * 10 for i in range(10)]
        text = "".join(sentences)
        
        metadata = {"source": "test"}
        chunks = self.chunker.chunk(text, metadata)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks have correct indices
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)
            
    def test_chunk_overlap(self):
        """Test that chunks have appropriate overlap."""
        # Create a long text
        sentences = [f"This is sentence {i}. " for i in range(20)]
        text = "".join(sentences)
        
        metadata = {"source": "test"}
        chunks = self.chunker.chunk(text, metadata)
        
        # We need at least 2 chunks to test overlap
        if len(chunks) >= 2:
            # Check that chunk boundaries respect sentence boundaries
            # This is more of a sanity check
            self.assertTrue(len(chunks[0].text) > 0)
            self.assertTrue(len(chunks[1].text) > 0)


class TestParagraphChunker(unittest.TestCase):
    """Test cases for ParagraphChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = ParagraphChunker(chunk_size=100, chunk_overlap=20)
        
    def test_chunk_by_paragraphs(self):
        """Test chunking text by paragraphs."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        metadata = {"source": "test"}
        chunks = self.chunker.chunk(text, metadata)
        
        # Should create at least one chunk
        self.assertGreaterEqual(len(chunks), 1)
        
        # Check that chunk metadata includes paragraph count
        if chunks:
            self.assertIn('paragraph_count', chunks[0].metadata)
            
    def test_chunk_single_paragraph(self):
        """Test chunking single paragraph text."""
        text = "This is a single paragraph without blank lines."
        metadata = {"source": "test"}
        chunks = self.chunker.chunk(text, metadata)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)


class TestDocumentChunkingService(unittest.TestCase):
    """Test cases for DocumentChunkingService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = DocumentChunkingService(chunk_size=1000, chunk_overlap=200)
        
    def test_init(self):
        """Test service initialization."""
        self.assertEqual(self.service.chunk_size, 1000)
        self.assertEqual(self.service.chunk_overlap, 200)
        self.assertIsNotNone(self.service.sentence_chunker)
        self.assertIsNotNone(self.service.paragraph_chunker)
        
    @patch('src.services.document_chunking_service.DocumentChunkingService._extract_txt_text')
    def test_chunk_txt_file(self, mock_extract):
        """Test chunking a text file."""
        # Mock text extraction
        mock_extract.return_value = (
            "Sentence one. Sentence two. Sentence three.",
            {"file_name": "test.txt", "file_type": "txt", "line_count": 1}
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sentence one. Sentence two. Sentence three.")
            temp_path = f.name
            
        try:
            chunks = self.service.chunk_document(temp_path, 'txt')
            
            # Should return list of chunks
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertIn('text', chunks[0])
                self.assertIn('metadata', chunks[0])
                self.assertEqual(chunks[0]['metadata']['file_type'], 'txt')
        finally:
            os.unlink(temp_path)
            
    @patch('src.services.document_chunking_service.DocumentChunkingService._extract_pdf_text')
    def test_chunk_pdf_file(self, mock_extract):
        """Test chunking a PDF file."""
        # Mock PDF extraction
        mock_extract.return_value = (
            "Page 1:\nPDF content here.\n\nPage 2:\nMore content.",
            {"file_name": "test.pdf", "file_type": "pdf", "pages": 2}
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("dummy pdf content")
            temp_path = f.name
            
        try:
            chunks = self.service.chunk_document(temp_path, 'pdf')
            
            # Should return list of chunks
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertIn('text', chunks[0])
                self.assertIn('metadata', chunks[0])
                self.assertEqual(chunks[0]['metadata']['file_type'], 'pdf')
        finally:
            os.unlink(temp_path)
            
    @patch('src.services.document_chunking_service.DocumentChunkingService._extract_docx_text')
    def test_chunk_docx_file(self, mock_extract):
        """Test chunking a DOCX file."""
        # Mock DOCX extraction
        mock_extract.return_value = (
            "Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            {"file_name": "test.docx", "file_type": "docx", "paragraph_count": 3}
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            f.write("dummy docx content")
            temp_path = f.name
            
        try:
            chunks = self.service.chunk_document(temp_path, 'docx')
            
            # Should return list of chunks
            self.assertIsInstance(chunks, list)
            if chunks:
                self.assertIn('text', chunks[0])
                self.assertIn('metadata', chunks[0])
                self.assertEqual(chunks[0]['metadata']['file_type'], 'docx')
        finally:
            os.unlink(temp_path)
            
    def test_chunk_unsupported_file_type(self):
        """Test chunking with unsupported file type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unsupported', delete=False) as f:
            f.write("dummy content")
            temp_path = f.name
            
        try:
            # Should default to sentence-aware chunking
            with patch.object(self.service, '_extract_txt_text') as mock_extract:
                mock_extract.return_value = ("dummy text", {"file_name": "test.unsupported"})
                chunks = self.service.chunk_document(temp_path, 'unsupported')
                self.assertIsInstance(chunks, list)
        finally:
            os.unlink(temp_path)
            
    def test_update_chunking_config(self):
        """Test updating chunking configuration."""
        self.service.update_chunking_config(2000, 400)
        
        self.assertEqual(self.service.chunk_size, 2000)
        self.assertEqual(self.service.chunk_overlap, 400)
        self.assertEqual(self.service.sentence_chunker.chunk_size, 2000)
        self.assertEqual(self.service.sentence_chunker.chunk_overlap, 400)
        self.assertEqual(self.service.paragraph_chunker.chunk_size, 2000)
        self.assertEqual(self.service.paragraph_chunker.chunk_overlap, 400)
        
    def test_update_chunking_config_invalid(self):
        """Test updating chunking config with invalid values."""
        with self.assertRaises(ValueError):
            self.service.update_chunking_config(0, 100)  # chunk_size must be positive
            
        with self.assertRaises(ValueError):
            self.service.update_chunking_config(1000, -1)  # chunk_overlap cannot be negative
            
        with self.assertRaises(ValueError):
            self.service.update_chunking_config(100, 200)  # overlap must be less than size
            
    def test_validate_chunks(self):
        """Test chunk validation."""
        original_text = "This is a test text for validation."
        chunks = [
            {
                'text': 'This is a test',
                'metadata': {},
                'start_position': 0,
                'end_position': 14
            },
            {
                'text': ' text for validation.',
                'metadata': {},
                'start_position': 14,
                'end_position': len(original_text)
            }
        ]
        
        # Should return True for valid chunks
        result = self.service.validate_chunks(original_text, chunks)
        self.assertTrue(result)
        
    def test_validate_chunks_content_loss(self):
        """Test chunk validation with content loss."""
        original_text = "This is a test text for validation."
        chunks = [
            {
                'text': 'This is a test',
                'metadata': {},
                'start_position': 0,
                'end_position': 14
            }
            # Missing the second part
        ]
        
        # Should return False due to content loss
        result = self.service.validate_chunks(original_text, chunks)
        self.assertFalse(result)


class TestDocumentChunkingIntegration(unittest.TestCase):
    """Integration tests for document chunking with knowledge base."""
    
    @patch('src.services.knowledge_base.DocumentChunkingService')
    @patch('src.services.knowledge_base.EmbeddingService')
    @patch('src.services.knowledge_base.SQLiteDatabaseService')
    @patch('src.services.knowledge_base.chromadb.PersistentClient')
    def test_knowledge_base_chunking_integration(self, mock_client, mock_sqlite, 
                                                 mock_embedding, mock_chunking):
        """Test that knowledge base uses chunking service for non-Excel files."""
        from src.services.knowledge_base import EnhancedKnowledgeBaseManager
        
        # Mock the chunking service
        mock_chunk_instance = Mock()
        mock_chunk_instance.chunk_document.return_value = [
            {
                'text': 'Chunk 1 content',
                'metadata': {'chunking_strategy': 'sentence_aware'},
                'chunk_index': 0,
                'start_position': 0,
                'end_position': 15
            },
            {
                'text': 'Chunk 2 content',
                'metadata': {'chunking_strategy': 'sentence_aware'},
                'chunk_index': 1,
                'start_position': 15,
                'end_position': 30
            }
        ]
        mock_chunking.return_value = mock_chunk_instance
        
        # Mock embedding service
        mock_embedding_instance = Mock()
        mock_embedding_instance.generate_embeddings.return_value = [[0.1] * 384, [0.2] * 384]
        mock_embedding.return_value = mock_embedding_instance
        
        # Mock vector store
        mock_collection = Mock()
        mock_collection.add = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create knowledge base manager
        kb_manager = EnhancedKnowledgeBaseManager()
        
        # Mock the vector store initialization to avoid actual ChromaDB calls
        kb_manager.vector_store = mock_collection
        
        # Test processing a text file (non-Excel)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample text content for chunking.")
            temp_path = f.name
            
        try:
            # Mock the document processor
            with patch.object(kb_manager.document_processor, 'extract_text_from_file') as mock_extract:
                mock_extract.return_value = (
                    "Sample text content for chunking.",
                    {'file_name': 'test.txt', 'file_size': 100}
                )
                
                # Process the document file
                result = kb_manager._process_document_file(temp_path)
                
                # Verify chunking service was called
                mock_chunk_instance.chunk_document.assert_called_once()
                
                # Verify vector store add was called with chunks
                mock_collection.add.assert_called_once()
                
                # Verify result includes chunk information
                self.assertEqual(result['storage_type'], 'vector')
                self.assertIn('chunks_created', result)
                self.assertEqual(result['chunks_created'], 2)
                
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
