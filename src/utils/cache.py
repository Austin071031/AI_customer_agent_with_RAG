"""
Caching System for AI Customer Agent.

This module provides intelligent caching strategies to improve performance
by caching frequent queries, embeddings, and API responses.
Includes LRU cache implementation with TTL and memory management.
"""

import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from functools import wraps
from collections import OrderedDict
import json
import pickle
from pathlib import Path


@dataclass
class CacheEntry:
    """Data class to store cache entry information."""
    value: Any
    timestamp: float
    access_count: int
    size: int  # Estimated size in bytes
    ttl: Optional[float] = None  # Time to live in seconds


@dataclass
class CacheStats:
    """Data class to store cache statistics."""
    hits: int
    misses: int
    evictions: int
    total_size: int
    max_size: int
    hit_ratio: float


class CacheError(Exception):
    """Custom exception for cache related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class ResponseCache:
    """
    LRU Cache implementation with TTL and memory management.
    
    This cache provides intelligent caching for frequent queries,
    embeddings, and API responses to improve application performance.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, default_ttl: int = 3600):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of items in cache
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # Cache storage using OrderedDict for LRU functionality
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(hits=0, misses=0, evictions=0, total_size=0, max_size=max_size, hit_ratio=0.0)
        
        self.logger.info(
            f"Cache initialized: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, default_ttl={default_ttl}s"
        )
        
    def _generate_key(self, data: Any) -> str:
        """
        Generate a cache key from input data.
        
        Args:
            data: Input data to generate key from
            
        Returns:
            MD5 hash string as cache key
        """
        try:
            # Convert data to bytes representation for hashing
            if isinstance(data, (str, int, float, bool)):
                data_bytes = str(data).encode('utf-8')
            elif isinstance(data, (list, dict)):
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            else:
                data_bytes = pickle.dumps(data)
                
            # Generate MD5 hash from bytes
            return hashlib.md5(data_bytes).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Key generation failed: {str(e)}")
            raise CacheError(f"Key generation failed: {str(e)}")
            
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate the memory size of a value.
        
        Args:
            value: The value to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8  # Approximate size for primitive types
            elif isinstance(value, (list, dict, tuple)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
                
        except Exception:
            return 1024  # Default estimate if calculation fails
            
    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry has expired.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if expired, False otherwise
        """
        if entry.ttl is None:
            return False
            
        current_time = time.time()
        return (current_time - entry.timestamp) > entry.ttl
        
    def _evict_lru(self) -> None:
        """
        Evict least recently used items until within memory limits.
        
        Removes expired items first, then LRU items if still over limits.
        """
        current_time = time.time()
        
        # First pass: remove expired items
        expired_keys = []
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_entry(key)
            self.logger.debug(f"Evicted expired entry: {key}")
            
        # Second pass: remove LRU items if still over limits
        while (len(self._cache) > self.max_size or 
               self._stats.total_size > self.max_memory_bytes):
            if not self._cache:
                break
                
            # Remove least recently used item
            key, entry = self._cache.popitem(last=False)
            self._stats.total_size -= entry.size
            self._stats.evictions += 1
            
            self.logger.debug(f"Evicted LRU entry: {key}, size: {entry.size} bytes")
            
    def _remove_entry(self, key: str) -> None:
        """
        Remove a specific entry from cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            entry = self._cache[key]
            self._stats.total_size -= entry.size
            del self._cache[key]
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        try:
            # Calculate entry size
            size = self._estimate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=0,
                size=size,
                ttl=ttl if ttl is not None else self.default_ttl
            )
            
            # Check if key already exists and remove it first
            if key in self._cache:
                self._remove_entry(key)
                
            # Add to cache (most recently used)
            self._cache[key] = entry
            self._stats.total_size += size
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Evict if necessary
            self._evict_lru()
            
            self.logger.debug(f"Cached entry: {key}, size: {size} bytes")
            
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {str(e)}")
            raise CacheError(f"Cache set failed: {str(e)}")
            
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats.misses += 1
            return None
            
        entry = self._cache[key]
        
        # Check if expired
        if self._is_expired(entry):
            self._remove_entry(key)
            self._stats.misses += 1
            self.logger.debug(f"Cache miss (expired): {key}")
            return None
            
        # Update access count and move to end (most recently used)
        entry.access_count += 1
        self._cache.move_to_end(key)
        
        self._stats.hits += 1
        self._update_hit_ratio()
        
        self.logger.debug(f"Cache hit: {key}, access count: {entry.access_count}")
        return entry.value
        
    def _update_hit_ratio(self) -> None:
        """Update the cache hit ratio statistic."""
        total_requests = self._stats.hits + self._stats.misses
        if total_requests > 0:
            self._stats.hit_ratio = self._stats.hits / total_requests
        else:
            self._stats.hit_ratio = 0.0
            
    def get_cached_response(self, query: str, **kwargs) -> Optional[Any]:
        """
        Get cached response for a query with additional parameters.
        
        Args:
            query: The query string
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cached response or None
        """
        cache_key = self._generate_key((query, kwargs))
        return self.get(cache_key)
        
    def cache_response(self, query: str, response: Any, ttl: Optional[int] = None, **kwargs) -> None:
        """
        Cache a response for a query.
        
        Args:
            query: The query string
            response: Response to cache
            ttl: Time to live in seconds
            **kwargs: Additional parameters for cache key generation
        """
        cache_key = self._generate_key((query, kwargs))
        self.set(cache_key, response, ttl)
        
    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            self._remove_entry(key)
            self.logger.debug(f"Deleted cache entry: {key}")
            return True
        return False
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats.total_size = 0
        self.logger.info("Cache cleared")
        
    def get_stats(self) -> CacheStats:
        """
        Get current cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        self._update_hit_ratio()
        return self._stats
        
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information.
        
        Returns:
            Dictionary with cache information and statistics
        """
        stats = self.get_stats()
        
        # Get top accessed items
        top_items = []
        for key, entry in list(self._cache.items())[-10:]:  # Last 10 (most recent)
            top_items.append({
                'key': key[:16] + '...' if len(key) > 16 else key,
                'access_count': entry.access_count,
                'size_bytes': entry.size,
                'age_seconds': time.time() - entry.timestamp
            })
            
        return {
            'current_size': len(self._cache),
            'max_size': self.max_size,
            'memory_used_mb': stats.total_size / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'hits': stats.hits,
            'misses': stats.misses,
            'evictions': stats.evictions,
            'hit_ratio': stats.hit_ratio,
            'top_items': top_items
        }
        
    def health_check(self) -> bool:
        """
        Perform a health check on the cache.
        
        Returns:
            True if cache is healthy, False otherwise
        """
        try:
            # Test basic operations
            test_key = "health_check"
            test_value = "test_value"
            
            self.set(test_key, test_value, ttl=10)
            retrieved = self.get(test_key)
            self.delete(test_key)
            
            return retrieved == test_value
            
        except Exception:
            return False


class EmbeddingCache(ResponseCache):
    """
    Specialized cache for embedding vectors.
    
    Provides optimized caching for embedding generation operations
    with specific handling for vector data.
    """
    
    def __init__(self, max_size: int = 500, max_memory_mb: int = 200, default_ttl: int = 86400):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embedding entries
            max_memory_mb: Maximum memory for embeddings
            default_ttl: Default TTL for embeddings (24 hours)
        """
        super().__init__(max_size, max_memory_mb, default_ttl)
        self.logger.info("Embedding cache initialized")
        
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of embedding vectors.
        
        Args:
            value: Embedding vector or list of vectors
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, list):
                # Estimate based on number of elements and typical float size
                if value and isinstance(value[0], (int, float)):
                    return len(value) * 8  # 8 bytes per float
                elif value and isinstance(value[0], list):
                    total_elements = sum(len(sublist) for sublist in value)
                    return total_elements * 8
            return super()._estimate_size(value)
            
        except Exception:
            return 1024  # Default estimate
            
    def cache_embedding(self, text: str, embedding: List[float], ttl: Optional[int] = None) -> None:
        """
        Cache an embedding for a text.
        
        Args:
            text: The input text
            embedding: Generated embedding vector
            ttl: Time to live in seconds
        """
        cache_key = self._generate_key(text)
        self.set(cache_key, embedding, ttl)
        
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for a text.
        
        Args:
            text: The input text
            
        Returns:
            Cached embedding or None
        """
        cache_key = self._generate_key(text)
        return self.get(cache_key)


class QueryResponseCache(ResponseCache):
    """
    Specialized cache for query responses.
    
    Provides caching for AI responses and search results
    with intelligent key generation for similar queries.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, default_ttl: int = 3600):
        """
        Initialize the query response cache.
        
        Args:
            max_size: Maximum number of query responses
            max_memory_mb: Maximum memory for responses
            default_ttl: Default TTL for responses (1 hour)
        """
        super().__init__(max_size, max_memory_mb, default_ttl)
        self.logger.info("Query response cache initialized")
        
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent caching.
        
        Args:
            query: Original query string
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase and remove extra whitespace
        normalized = ' '.join(query.lower().split())
        return normalized
        
    def cache_query_response(self, query: str, response: str, use_knowledge_base: bool = True, ttl: Optional[int] = None) -> None:
        """
        Cache a query response.
        
        Args:
            query: The user query
            response: AI-generated response
            use_knowledge_base: Whether knowledge base was used
            ttl: Time to live in seconds
        """
        normalized_query = self._normalize_query(query)
        cache_data = {
            'query': normalized_query,
            'use_knowledge_base': use_knowledge_base
        }
        
        cache_key = self._generate_key(cache_data)
        self.set(cache_key, response, ttl)
        
    def get_cached_query_response(self, query: str, use_knowledge_base: bool = True) -> Optional[str]:
        """
        Get cached response for a query.
        
        Args:
            query: The user query
            use_knowledge_base: Whether knowledge base was used
            
        Returns:
            Cached response or None
        """
        normalized_query = self._normalize_query(query)
        cache_data = {
            'query': normalized_query,
            'use_knowledge_base': use_knowledge_base
        }
        
        cache_key = self._generate_key(cache_data)
        return self.get(cache_key)


def cache_response(ttl: int = 3600, cache_instance: Optional[ResponseCache] = None):
    """
    Decorator to cache function responses.
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Cache instance to use (creates new if None)
        
    Returns:
        Decorated function with caching
    """
    if cache_instance is None:
        cache_instance = ResponseCache()
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key_data = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = cache_instance._generate_key(cache_key_data)
            
            # Try to get cached result
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            
            return result
            
        return wrapper
    return decorator


# Global cache instances
_embedding_cache: Optional[EmbeddingCache] = None
_query_cache: Optional[QueryResponseCache] = None
_general_cache: Optional[ResponseCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """
    Get or create the global embedding cache instance.
    
    Returns:
        EmbeddingCache: Global embedding cache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_query_cache() -> QueryResponseCache:
    """
    Get or create the global query response cache instance.
    
    Returns:
        QueryResponseCache: Global query response cache instance
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResponseCache()
    return _query_cache


def get_general_cache() -> ResponseCache:
    """
    Get or create the global general cache instance.
    
    Returns:
        ResponseCache: Global general cache instance
    """
    global _general_cache
    if _general_cache is None:
        _general_cache = ResponseCache()
    return _general_cache


def clear_all_caches() -> None:
    """Clear all global cache instances."""
    global _embedding_cache, _query_cache, _general_cache
    
    if _embedding_cache:
        _embedding_cache.clear()
    if _query_cache:
        _query_cache.clear()
    if _general_cache:
        _general_cache.clear()
        
    logging.getLogger(__name__).info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics from all cache instances.
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {}
    
    if _embedding_cache:
        stats['embedding_cache'] = _embedding_cache.get_cache_info()
    if _query_cache:
        stats['query_cache'] = _query_cache.get_cache_info()
    if _general_cache:
        stats['general_cache'] = _general_cache.get_cache_info()
        
    return stats
