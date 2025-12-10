"""
Simple test script to verify document chunking functionality.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.document_chunking_service import DocumentChunkingService

def test_document_chunking():
    """Test document chunking with a simple text file."""
    # Create a test document with known content
    content = "This is sentence one. This is sentence two. This is sentence three. " * 20
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        # Initialize the document chunking service
        chunking_service = DocumentChunkingService(chunk_size=200, chunk_overlap=50)
        
        # Chunk the document
        chunks = chunking_service.chunk_document(temp_path, 'txt')
        
        # Print results
        print(f"Original content length: {len(content)}")
        print(f"Number of chunks created: {len(chunks)}")
        
        if chunks:
            print("\nFirst chunk:")
            print(f"Text: {chunks[0]['text']}")
            print(f"Length: {len(chunks[0]['text'])}")
            print(f"Metadata: {chunks[0]['metadata']}")
            
        # Test with different parameters
        chunking_service.update_chunking_config(300, 75)
        chunks2 = chunking_service.chunk_document(temp_path, 'txt')
        
        print(f"\nWith different parameters - Number of chunks: {len(chunks2)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    print("Testing document chunking functionality...")
    success = test_document_chunking()
    if success:
        print("\n✅ Document chunking test passed!")
    else:
        print("\n❌ Document chunking test failed!")
