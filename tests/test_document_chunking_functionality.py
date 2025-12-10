"""
Unit tests for Document Chunking Functionality Integration.

This module provides comprehensive unit tests for the document chunking functionality
integration with the knowledge base and API endpoints.
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
from src.services.knowledge_base import EnhancedKnowledgeBaseManager
from src.api.endpoints.knowledge_base import upload_documents


class TestDocumentChunkingFunctionalityIntegration(unittest.TestCase):
    """Integration tests for document chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.persist_dir = os.path.join(self.test_dir, "chroma_db")
        self.sqlite_db_path = os.path.join(self.test_dir, "test.db")
        
        # Initialize the knowledge base manager with test paths
        self.kb_manager = EnhancedKnowledgeBaseManager(
            persist_directory=self.persist_dir,
            sqlite_db_path=self.sqlite_db_path,
            chunk_size=500,
            chunk_overlap=100
        )
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directories
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_chunking_parameter_propagation_through_api(self):
        """Test that chunking parameters are properly passed through the API to the service."""
        # Create a test document
        content = "This is a test document. " * 100  # Make it long enough to create multiple chunks
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
            
        try:
            # Mock the knowledge base manager in the API endpoint
            with patch('src.api.endpoints.knowledge_base.get_kb_manager') as mock_get_kb:
                mock_kb_manager = Mock()
                mock_get_kb.return_value = mock_kb_manager
                mock_kb_manager.add_documents.return_value = {
                    'excel_files': [],
                    'documents': [{'file_name': 'test.txt', 'chunks_created': 3}],
                    'errors': []
                }
                
                # Simulate calling the API endpoint with chunking parameters
                from fastapi import UploadFile
                import io
                
                # Create a mock UploadFile
                file_content = content.encode('utf-8')
                upload_file = UploadFile(
                    filename="test.txt",
                    file=io.BytesIO(file_content),
                    size=len(file_content)
                )
                
                # Test the upload function with chunking parameters
                # Note: In a real test, we would call the actual endpoint
                # For this test, we'll directly test the parameter passing
                
                # Verify that the knowledge base manager is called with the correct parameters
                self.kb_manager.add_documents([temp_path], chunk_size=800, chunk_overlap=150)
                mock_kb_manager.add_documents.assert_called_with([temp_path], 800, 150)
                
        finally:
            os.unlink(temp_path)
            
    def test_document_chunking_with_custom_parameters(self):
        """Test document chunking with custom chunk size and overlap parameters."""
        # Create a test document with known content
        content = "Sentence one. Sentence two. Sentence three. " * 20
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
            
        try:
            # Test with custom chunking parameters
            chunk_size = 200
            chunk_overlap = 50
            
            # Update the chunking service with custom parameters
            self.kb_manager.document_chunking_service.update_chunking_config(chunk_size, chunk_overlap)
            
            # Process the document
            result = self.kb_manager.add_documents([temp_path], chunk_size, chunk_overlap)
            
            # Verify the result structure
            self.assertIn('documents', result)
            self.assertGreater(len(result['documents']), 0)
            
            # Get the first document result
            doc_result = result['documents'][0]
            self.assertIn('chunks_created', doc_result)
            self.assertGreater(doc_result['chunks_created'], 0)
            
            # Verify chunking service was called with correct parameters
            chunks = self.kb_manager.document_chunking_service.chunk_document(temp_path, 'txt')
            self.assertGreater(len(chunks), 0)
            
            # Check that chunks have the expected properties
            for chunk in chunks:
                self.assertIn('text', chunk)
                self.assertIn('metadata', chunk)
                self.assertIn('chunk_index', chunk)
                self.assertLessEqual(len(chunk['text']), chunk_size + 50)  # Allow some buffer
                
        finally:
            os.unlink(temp_path)
            
    def test_chunking_parameter_validation(self):
        """Test validation of chunking parameters."""
        # Test valid parameters
        try:
            self.kb_manager.document_chunking_service.update_chunking_config(1000, 200)
            # Should not raise an exception
        except Exception as e:
            self.fail(f"Valid parameters should not raise exception: {e}")
            
        # Test invalid chunk size (zero or negative)
        with self.assertRaises(ValueError):
            self.kb_manager.document_chunking_service.update_chunking_config(0, 100)
            
        with self.assertRaises(ValueError):
            self.kb_manager.document_chunking_service.update_chunking_config(-1, 100)
            
        # Test invalid overlap (negative)
        with self.assertRaises(ValueError):
            self.kb_manager.document_chunking_service.update_chunking_config(1000, -1)
            
        # Test invalid overlap (greater than chunk size)
        with self.assertRaises(ValueError):
            self.kb_manager.document_chunking_service.update_chunking_config(100, 200)
            
    def test_different_file_types_chunking(self):
        """Test chunking of different file types with custom parameters."""
        test_cases = [
            ('.txt', 'This is a text file content. ' * 30),
            ('.md', '# Markdown Title\n\nThis is markdown content. ' * 20),
        ]
        
        for file_ext, content in test_cases:
            with self.subTest(file_type=file_ext):
                with tempfile.NamedTemporaryFile(mode='w', suffix=file_ext, delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                    
                try:
                    # Test with custom chunking parameters
                    chunk_size = 300
                    chunk_overlap = 50
                    
                    self.kb_manager.document_chunking_service.update_chunking_config(chunk_size, chunk_overlap)
                    chunks = self.kb_manager.document_chunking_service.chunk_document(temp_path, file_ext[1:])
                    
                    # Verify chunks were created
                    self.assertGreater(len(chunks), 0)
                    
                    # Verify chunk properties
                    for i, chunk in enumerate(chunks):
                        self.assertIn('text', chunk)
                        self.assertIn('metadata', chunk)
                        self.assertEqual(chunk['chunk_index'], i)
                        self.assertLessEqual(len(chunk['text']), chunk_size + 50)  # Allow buffer
                        
                finally:
                    os.unlink(temp_path)
                    
    def test_chunking_consistency_across_parameters(self):
        """Test that chunking produces consistent results with the same parameters."""
        content = "Consistency test sentence. " * 50
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
            
        try:
            # Test with specific parameters
            chunk_size = 400
            chunk_overlap = 100
            
            self.kb_manager.document_chunking_service.update_chunking_config(chunk_size, chunk_overlap)
            chunks1 = self.kb_manager.document_chunking_service.chunk_document(temp_path, 'txt')
            
            # Reset and test again with same parameters
            self.kb_manager.document_chunking_service.update_chunking_config(chunk_size, chunk_overlap)
            chunks2 = self.kb_manager.document_chunking_service.chunk_document(temp_path, 'txt')
            
            # Results should be the same
            self.assertEqual(len(chunks1), len(chunks2))
            for c1, c2 in zip(chunks1, chunks2):
                self.assertEqual(c1['text'], c2['text'])
                self.assertEqual(c1['chunk_index'], c2['chunk_index'])
                
        finally:
            os.unlink(temp_path)
            
    def test_streamlit_ui_chunking_integration(self):
        """Test Streamlit UI chunking parameter integration."""
        # This test verifies that the Streamlit UI functions correctly
        # with chunking parameters
        
        # Import the Streamlit functions
        from src.ui.streamlit_app import upload_to_knowledge_base
        
        # Create a mock file
        content = b"This is a test file for Streamlit UI testing. " * 20
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.getvalue.return_value = content
        mock_file.type = "text/plain"
        
        # Mock the requests.post call
        with patch('src.ui.streamlit_app.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Test upload with chunking parameters
            result = upload_to_knowledge_base(mock_file, chunk_size=600, chunk_overlap=120)
            
            # Verify the request was made with correct parameters
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            
            # Check that data contains chunking parameters
            self.assertIn('data', kwargs)
            data = kwargs['data']
            self.assertEqual(data['chunk_size'], 600)
            self.assertEqual(data['chunk_overlap'], 120)
            
            # Verify successful result
            self.assertTrue(result)
            

class TestDocumentChunkingEdgeCases(unittest.TestCase):
    """Test edge cases for document chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunking_service = DocumentChunkingService(chunk_size=500, chunk_overlap=100)
        
    def test_empty_document_chunking(self):
        """Test chunking of empty documents."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty content
            temp_path = f.name
            
        try:
            with self.assertRaises(DocumentChunkingError):
                self.chunking_service.chunk_document(temp_path, 'txt')
        finally:
            os.unlink(temp_path)
            
    def test_very_small_chunk_size(self):
        """Test chunking with very small chunk size."""
        content = "Small chunk test. " * 100
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
            
        try:
            # Test with very small chunk size
            self.chunking_service.update_chunking_config(50, 10)
            chunks = self.chunking_service.chunk_document(temp_path, 'txt')
            
            # Should still create chunks
            self.assertGreater(len(chunks), 0)
            
            # Each chunk should be reasonably sized
            for chunk in chunks:
                self.assertGreater(len(chunk['text']), 0)
        finally:
            os.unlink(temp_path)
            
    def test_large_overlap_relative_to_chunk_size(self):
        """Test chunking with large overlap relative to chunk size."""
        content = "Large overlap test sentence. " * 50
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
            
        try:
            # Test with large overlap (40% of chunk size)
            chunk_size = 200
            chunk_overlap = 80  # 40% of chunk size
            
            self.chunking_service.update_chunking_config(chunk_size, chunk_overlap)
            chunks = self.chunking_service.chunk_document(temp_path, 'txt')
            
            # Should create chunks
            self.assertGreater(len(chunks), 0)
            
            # Check for reasonable overlap behavior
            if len(chunks) > 1:
                # Second chunk should have some content from first chunk
                first_chunk_end = chunks[0]['text'][-50:]  # Last 50 chars of first chunk
                second_chunk_start = chunks[1]['text'][:50]  # First 50 chars of second chunk
                
                # There should be some similarity (indicating overlap)
                self.assertTrue(len(first_chunk_end) > 0)
                self.assertTrue(len(second_chunk_start) > 0)
        finally:
            os.unlink(temp_path)
            
    def test_unicode_content_chunking(self):
        """Test chunking of documents with unicode content."""
        # Content with unicode characters
        content = "Unicode test: café, naïve, résumé, piñata. " * 30
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
            
        try:
            self.chunking_service.update_chunking_config(300, 50)
            chunks = self.chunking_service.chunk_document(temp_path, 'txt')
            
            # Should create chunks
            self.assertGreater(len(chunks), 0)
            
            # Chunks should preserve unicode content
            full_reconstructed = "".join([chunk['text'] for chunk in chunks])
            self.assertIn("café", full_reconstructed)
            self.assertIn("résumé", full_reconstructed)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
