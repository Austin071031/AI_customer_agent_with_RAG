"""
Unit tests for caching system.

Tests the ResponseCache, EmbeddingCache, QueryResponseCache, and related 
utility functions for the AI Customer Agent application.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.cache import (
    ResponseCache,
    EmbeddingCache,
    QueryResponseCache,
    CacheEntry,
    CacheStats,
    CacheError,
    get_embedding_cache,
    get_query_cache,
    get_general_cache,
    clear_all_caches,
    get_cache_stats,
    cache_response
)


class TestCacheEntry:
    """Test cases for CacheEntry data class."""
    
    def test_cache_entry_creation(self):
        """Test creating CacheEntry instance with valid data."""
        entry = CacheEntry(
            value="test_value",
            timestamp=1234567890.0,
            access_count=5,
            size=1024,
            ttl=3600
        )
        
        assert entry.value == "test_value"
        assert entry.timestamp == 1234567890.0
        assert entry.access_count == 5
        assert entry.size == 1024
        assert entry.ttl == 3600
        
    def test_cache_entry_without_ttl(self):
        """Test creating CacheEntry instance without TTL."""
        entry = CacheEntry(
            value="test_value",
            timestamp=1234567890.0,
            access_count=5,
            size=1024,
            ttl=None
        )
        
        assert entry.value == "test_value"
        assert entry.timestamp == 1234567890.0
        assert entry.access_count == 5
        assert entry.size == 1024
        assert entry.ttl is None


class TestCacheStats:
    """Test cases for CacheStats data class."""
    
    def test_cache_stats_creation(self):
        """Test creating CacheStats instance with valid data."""
        stats = CacheStats(
            hits=100,
            misses=50,
            evictions=10,
            total_size=1024000,
            max_size=1000,
            hit_ratio=0.6667
        )
        
        assert stats.hits == 100
        assert stats.misses == 50
        assert stats.evictions == 10
        assert stats.total_size == 1024000
        assert stats.max_size == 1000
        assert stats.hit_ratio == 0.6667


class TestResponseCache:
    """Test cases for ResponseCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization with default parameters."""
        cache = ResponseCache()
        
        assert cache.max_size == 1000
        assert cache.max_memory_bytes == 100 * 1024 * 1024  # 100MB in bytes
        assert cache.default_ttl == 3600
        assert len(cache._cache) == 0
        
    def test_cache_initialization_custom_params(self):
        """Test cache initialization with custom parameters."""
        cache = ResponseCache(max_size=500, max_memory_mb=50, default_ttl=1800)
        
        assert cache.max_size == 500
        assert cache.max_memory_bytes == 50 * 1024 * 1024  # 50MB in bytes
        assert cache.default_ttl == 1800
        
    def test_generate_key_string(self):
        """Test key generation with string data."""
        cache = ResponseCache()
        key = cache._generate_key("test_string")
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length
        
    def test_generate_key_dict(self):
        """Test key generation with dictionary data."""
        cache = ResponseCache()
        test_dict = {"key1": "value1", "key2": "value2"}
        key = cache._generate_key(test_dict)
        
        assert isinstance(key, str)
        assert len(key) == 32
        
    def test_generate_key_consistent(self):
        """Test that same input generates same key."""
        cache = ResponseCache()
        key1 = cache._generate_key("test_input")
        key2 = cache._generate_key("test_input")
        
        assert key1 == key2
        
    def test_estimate_size_string(self):
        """Test size estimation for string values."""
        cache = ResponseCache()
        size = cache._estimate_size("test_string")
        
        assert size == len("test_string")
        
    def test_estimate_size_list(self):
        """Test size estimation for list values."""
        cache = ResponseCache()
        test_list = [1, 2, 3, 4, 5]
        size = cache._estimate_size(test_list)
        
        assert size > 0
        
    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = ResponseCache()
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Get the value
        result = cache.get("test_key")
        
        assert result == "test_value"
        
    def test_get_nonexistent_key(self):
        """Test getting a non-existent key."""
        cache = ResponseCache()
        result = cache.get("nonexistent_key")
        
        assert result is None
        
    def test_set_overwrite(self):
        """Test overwriting an existing key."""
        cache = ResponseCache()
        
        # Set initial value
        cache.set("test_key", "initial_value")
        
        # Overwrite with new value
        cache.set("test_key", "new_value")
        
        # Get the value
        result = cache.get("test_key")
        
        assert result == "new_value"
        
    def test_expired_entry(self):
        """Test that expired entries are not returned."""
        cache = ResponseCache()
        
        # Set value with very short TTL
        cache.set("test_key", "test_value", ttl=0.1)  # 0.1 seconds
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Try to get expired value
        result = cache.get("test_key")
        
        assert result is None
        
    def test_cache_response_and_get_cached_response(self):
        """Test caching and retrieving responses."""
        cache = ResponseCache()
        
        # Cache a response
        cache.cache_response("test_query", "test_response", ttl=3600, param1="value1")
        
        # Retrieve the cached response
        result = cache.get_cached_response("test_query", param1="value1")
        
        assert result == "test_response"
        
    def test_delete_existing_key(self):
        """Test deleting an existing key."""
        cache = ResponseCache()
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Delete the key
        result = cache.delete("test_key")
        
        assert result is True
        assert cache.get("test_key") is None
        
    def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key."""
        cache = ResponseCache()
        result = cache.delete("nonexistent_key")
        
        assert result is False
        
    def test_clear_cache(self):
        """Test clearing all cache entries."""
        cache = ResponseCache()
        
        # Add some entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Clear cache
        cache.clear()
        
        # Verify cache is empty
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache._cache) == 0
        
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ResponseCache()
        
        # Add some entries and perform operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.evictions == 0
        assert stats.total_size > 0
        
    def test_get_cache_info(self):
        """Test getting detailed cache information."""
        cache = ResponseCache()
        
        # Add some entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        info = cache.get_cache_info()
        
        assert info["current_size"] == 2
        assert info["max_size"] == 1000
        assert "memory_used_mb" in info
        assert "hits" in info
        assert "misses" in info
        assert "hit_ratio" in info
        assert "top_items" in info
        
    def test_health_check(self):
        """Test cache health check."""
        cache = ResponseCache()
        result = cache.health_check()
        
        assert result is True
        
    def test_lru_eviction(self):
        """Test LRU eviction when cache reaches size limit."""
        cache = ResponseCache(max_size=2, max_memory_mb=1)  # Small cache for testing
        
        # Add entries to fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add third entry, should evict key2 (least recently used)
        cache.set("key3", "value3")
        
        # key2 should be evicted, key1 and key3 should remain
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"


class TestEmbeddingCache:
    """Test cases for EmbeddingCache class."""
    
    def test_embedding_cache_initialization(self):
        """Test embedding cache initialization."""
        cache = EmbeddingCache()
        
        assert cache.max_size == 500
        assert cache.max_memory_bytes == 200 * 1024 * 1024  # 200MB in bytes
        assert cache.default_ttl == 86400  # 24 hours
        
    def test_estimate_size_embedding_vector(self):
        """Test size estimation for embedding vectors."""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5-dimensional vector
        
        size = cache._estimate_size(embedding)
        
        # Should be approximately 5 * 8 bytes = 40 bytes
        assert size == 40
        
    def test_estimate_size_list_of_embeddings(self):
        """Test size estimation for list of embedding vectors."""
        cache = EmbeddingCache()
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]  # 3 vectors of 3 dimensions each
        
        size = cache._estimate_size(embeddings)
        
        # Should be approximately 3 * 3 * 8 bytes = 72 bytes
        assert size == 72
        
    def test_cache_embedding_and_get_cached_embedding(self):
        """Test caching and retrieving embeddings."""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Cache embedding
        cache.cache_embedding("test_text", embedding, ttl=3600)
        
        # Retrieve cached embedding
        result = cache.get_cached_embedding("test_text")
        
        assert result == embedding


class TestQueryResponseCache:
    """Test cases for QueryResponseCache class."""
    
    def test_query_response_cache_initialization(self):
        """Test query response cache initialization."""
        cache = QueryResponseCache()
        
        assert cache.max_size == 1000
        assert cache.max_memory_bytes == 100 * 1024 * 1024  # 100MB in bytes
        assert cache.default_ttl == 3600  # 1 hour
        
    def test_normalize_query(self):
        """Test query normalization."""
        cache = QueryResponseCache()
        
        # Test various query formats
        assert cache._normalize_query("Hello World") == "hello world"
        assert cache._normalize_query("  Hello   World  ") == "hello world"
        assert cache._normalize_query("HELLO WORLD") == "hello world"
        assert cache._normalize_query("Hello\tWorld") == "hello world"
        
    def test_cache_query_response_and_get_cached_query_response(self):
        """Test caching and retrieving query responses."""
        cache = QueryResponseCache()
        
        # Cache query response
        cache.cache_query_response(
            "What is AI?",
            "AI stands for Artificial Intelligence.",
            use_knowledge_base=True,
            ttl=3600
        )
        
        # Retrieve cached response
        result = cache.get_cached_query_response(
            "what is ai?",  # Normalized query
            use_knowledge_base=True
        )
        
        assert result == "AI stands for Artificial Intelligence."
        
    def test_cache_query_response_different_kb_settings(self):
        """Test that queries with different KB settings are cached separately."""
        cache = QueryResponseCache()
        
        # Cache same query with different KB settings
        cache.cache_query_response(
            "What is AI?",
            "Response with KB",
            use_knowledge_base=True
        )
        cache.cache_query_response(
            "What is AI?",
            "Response without KB",
            use_knowledge_base=False
        )
        
        # Retrieve both responses
        result_with_kb = cache.get_cached_query_response("what is ai?", use_knowledge_base=True)
        result_without_kb = cache.get_cached_query_response("what is ai?", use_knowledge_base=False)
        
        assert result_with_kb == "Response with KB"
        assert result_without_kb == "Response without KB"


class TestCacheError:
    """Test cases for CacheError exception."""
    
    def test_cache_error_creation(self):
        """Test creating CacheError instance."""
        error = CacheError("Test error message", "test_error")
        
        assert str(error) == "Test error message"
        assert error.error_type == "test_error"
        
    def test_cache_error_without_error_type(self):
        """Test creating CacheError without error type."""
        error = CacheError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.error_type is None


class TestCacheResponseDecorator:
    """Test cases for cache_response decorator."""
    
    def test_cache_response_decorator(self):
        """Test the cache_response decorator."""
        # Create a mock cache instance
        mock_cache = Mock()
        mock_cache._generate_key.return_value = "test_key"
        mock_cache.get.return_value = None  # First call: cache miss
        
        # Apply decorator to test function
        @cache_response(ttl=3600, cache_instance=mock_cache)
        def test_function(param1, param2):
            return f"result_{param1}_{param2}"
            
        # Call the decorated function
        result = test_function("value1", "value2")
        
        # Verify cache operations
        mock_cache.get.assert_called_once_with("test_key")
        mock_cache.set.assert_called_once_with("test_key", "result_value1_value2", 3600)
        assert result == "result_value1_value2"
        
    def test_cache_response_decorator_cache_hit(self):
        """Test the cache_response decorator with cache hit."""
        # Create a mock cache instance
        mock_cache = Mock()
        mock_cache._generate_key.return_value = "test_key"
        mock_cache.get.return_value = "cached_result"  # Cache hit
        
        # Apply decorator to test function
        @cache_response(ttl=3600, cache_instance=mock_cache)
        def test_function(param1, param2):
            return f"result_{param1}_{param2}"  # This should not be executed
            
        # Call the decorated function
        result = test_function("value1", "value2")
        
        # Verify cache operations
        mock_cache.get.assert_called_once_with("test_key")
        mock_cache.set.assert_not_called()  # Should not set when cache hit
        assert result == "cached_result"  # Should return cached result


class TestGlobalCacheFunctions:
    """Test cases for global cache functions."""
    
    def test_get_embedding_cache_singleton(self):
        """Test that get_embedding_cache returns a singleton instance."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, EmbeddingCache)
        
    def test_get_query_cache_singleton(self):
        """Test that get_query_cache returns a singleton instance."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, QueryResponseCache)
        
    def test_get_general_cache_singleton(self):
        """Test that get_general_cache returns a singleton instance."""
        cache1 = get_general_cache()
        cache2 = get_general_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, ResponseCache)
        
    def test_clear_all_caches(self):
        """Test clearing all global caches."""
        # Get cache instances
        embedding_cache = get_embedding_cache()
        query_cache = get_query_cache()
        general_cache = get_general_cache()
        
        # Mock the clear method
        with patch.object(embedding_cache, 'clear') as mock_embedding_clear, \
             patch.object(query_cache, 'clear') as mock_query_clear, \
             patch.object(general_cache, 'clear') as mock_general_clear:
            
            clear_all_caches()
            
            # Verify all clear methods were called
            mock_embedding_clear.assert_called_once()
            mock_query_clear.assert_called_once()
            mock_general_clear.assert_called_once()
            
    def test_get_cache_stats(self):
        """Test getting statistics from all cache instances."""
        # Mock cache instances
        mock_embedding_cache = Mock()
        mock_embedding_cache.get_cache_info.return_value = {"embedding_stats": "test"}
        
        mock_query_cache = Mock()
        mock_query_cache.get_cache_info.return_value = {"query_stats": "test"}
        
        mock_general_cache = Mock()
        mock_general_cache.get_cache_info.return_value = {"general_stats": "test"}
        
        # Patch global instances
        with patch('src.utils.cache._embedding_cache', mock_embedding_cache), \
             patch('src.utils.cache._query_cache', mock_query_cache), \
             patch('src.utils.cache._general_cache', mock_general_cache):
            
            stats = get_cache_stats()
            
            assert "embedding_cache" in stats
            assert "query_cache" in stats
            assert "general_cache" in stats
            assert stats["embedding_cache"] == {"embedding_stats": "test"}
            assert stats["query_cache"] == {"query_stats": "test"}
            assert stats["general_cache"] == {"general_stats": "test"}


if __name__ == "__main__":
    pytest.main([__file__])
