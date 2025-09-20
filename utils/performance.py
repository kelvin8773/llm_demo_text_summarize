# utils/performance.py - Performance optimization utilities

"""
Performance optimization utilities for the LLM Text Summarization Tool.

This module provides caching, memory management, and performance monitoring
capabilities to improve the application's speed and efficiency.
"""

import time
import logging
import functools
import threading
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import pickle
import hashlib
import gc
import psutil
import os

logger = logging.getLogger(__name__)

# Import configuration
from .config import get_performance_config

# Performance configuration
perf_config = get_performance_config()
CACHE_SIZE_LIMIT = perf_config.get('cache_size_limit', 100)
CACHE_TTL = perf_config.get('cache_ttl', 3600)
MEMORY_THRESHOLD = perf_config.get('memory_threshold', 0.8)
CLEANUP_INTERVAL = perf_config.get('cleanup_interval', 300)
ENABLE_BACKGROUND_CLEANUP = perf_config.get('enable_background_cleanup', True)
ENABLE_MEMORY_MONITORING = perf_config.get('enable_memory_monitoring', True)
ENABLE_PERFORMANCE_TRACKING = perf_config.get('enable_performance_tracking', True)


class ModelCache:
    """Thread-safe model cache with TTL and size limits."""
    
    def __init__(self, max_size: int = CACHE_SIZE_LIMIT, ttl: int = CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and hasn't expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            item, timestamp = self._cache[key]
            
            # Check if item has expired
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._access_times.pop(key, None)
                logger.debug(f"Cache item '{key}' expired and removed")
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            logger.debug(f"Cache hit for '{key}'")
            return item
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        with self._lock:
            # Remove oldest items if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[key] = (value, time.time())
            self._access_times[key] = time.time()
            logger.debug(f"Cached item '{key}'")
    
    def _evict_oldest(self) -> None:
        """Remove the least recently used item."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._cache.pop(oldest_key, None)
        self._access_times.pop(oldest_key, None)
        logger.debug(f"Evicted oldest cache item '{oldest_key}'")
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info("Model cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count of removed items."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._access_times.pop(key, None)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
            
            return len(expired_keys)


class MemoryManager:
    """Memory usage monitoring and management."""
    
    def __init__(self, threshold: float = MEMORY_THRESHOLD):
        self.threshold = threshold
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': self.process.memory_percent(),
            'system_memory_percent': system_memory.percent,
            'available_memory_mb': system_memory.available / 1024 / 1024
        }
    
    def is_memory_pressure(self) -> bool:
        """Check if memory usage is above threshold."""
        memory_stats = self.get_memory_usage()
        return memory_stats['system_memory_percent'] > (self.threshold * 100)
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup and return cleanup statistics."""
        initial_memory = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear model cache if memory pressure is high
        cache_cleared = False
        if self.is_memory_pressure():
            model_cache.clear()
            cache_cleared = True
        
        final_memory = self.get_memory_usage()
        
        cleanup_stats = {
            'objects_collected': collected,
            'cache_cleared': cache_cleared,
            'memory_freed_mb': initial_memory['process_memory_mb'] - final_memory['process_memory_mb'],
            'initial_memory_mb': initial_memory['process_memory_mb'],
            'final_memory_mb': final_memory['process_memory_mb']
        }
        
        logger.info(f"Memory cleanup completed: {cleanup_stats}")
        return cleanup_stats


class PerformanceMonitor:
    """Performance monitoring and profiling utilities."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation '{operation}'")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]
        
        logger.debug(f"Operation '{operation}' took {duration:.3f} seconds")
        return duration
    
    def get_average_time(self, operation: str) -> Optional[float]:
        """Get average time for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return summary


def cached_model_loader(cache_key_func: Callable[[], str]):
    """Decorator for caching model loading operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_key_func()
            
            # Try to get from cache first
            cached_model = model_cache.get(cache_key)
            if cached_model is not None:
                logger.debug(f"Using cached model for '{cache_key}'")
                return cached_model
            
            # Load model and cache it
            logger.debug(f"Loading model for '{cache_key}'")
            model = func(*args, **kwargs)
            model_cache.set(cache_key, model)
            
            return model
        
        return wrapper
    return decorator


def performance_timer(operation_name: str):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_timer(operation_name)
        
        return wrapper
    return decorator


def memory_aware(func: Callable) -> Callable:
    """Decorator for memory-aware function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory before execution
        if memory_manager.is_memory_pressure():
            logger.warning("High memory usage detected, performing cleanup")
            memory_manager.cleanup_memory()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Check memory after execution
        if memory_manager.is_memory_pressure():
            logger.warning("High memory usage after execution, performing cleanup")
            memory_manager.cleanup_memory()
        
        return result
    
    return wrapper


def optimize_text_chunking(text: str, max_tokens: int, tokenizer) -> list:
    """Optimized text chunking with better memory management."""
    if not text or not text.strip():
        return []
    
    try:
        # Use more efficient tokenization
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        if not token_ids:
            return []
        
        chunks = []
        for i in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            if chunk_text and chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error in optimized text chunking: {e}")
        # Fallback to simple splitting
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_tokens * 4:  # Rough estimate
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    chunks.append(word)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def batch_process(items: list, batch_size: int = 5, delay: float = 0.1) -> list:
    """Process items in batches with optional delay."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for item in batch:
            try:
                # This would be replaced with actual processing logic
                batch_results.append(item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        
        # Optional delay between batches
        if delay > 0 and i + batch_size < len(items):
            time.sleep(delay)
        
        # Memory cleanup between batches
        if memory_manager.is_memory_pressure():
            memory_manager.cleanup_memory()
    
    return results


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        'cache_size': model_cache.size(),
        'max_cache_size': model_cache.max_size,
        'cache_ttl': model_cache.ttl,
        'memory_usage': memory_manager.get_memory_usage(),
        'performance_summary': performance_monitor.get_performance_summary()
    }


def cleanup_resources() -> Dict[str, Any]:
    """Clean up all resources and return cleanup statistics."""
    cleanup_stats = {
        'cache_cleared': model_cache.size(),
        'memory_cleanup': memory_manager.cleanup_memory(),
        'performance_reset': len(performance_monitor.metrics)
    }
    
    # Clear cache
    model_cache.clear()
    
    # Reset performance metrics
    performance_monitor.metrics.clear()
    performance_monitor.start_times.clear()
    
    logger.info(f"Resource cleanup completed: {cleanup_stats}")
    return cleanup_stats


# Global instances
model_cache = ModelCache()
memory_manager = MemoryManager()
performance_monitor = PerformanceMonitor()

# Background cleanup thread
def background_cleanup():
    """Background thread for periodic cleanup."""
    while True:
        try:
            time.sleep(CLEANUP_INTERVAL)
            
            # Cleanup expired cache items
            expired_count = model_cache.cleanup_expired()
            
            # Memory cleanup if needed
            if memory_manager.is_memory_pressure():
                memory_manager.cleanup_memory()
            
            if expired_count > 0:
                logger.debug(f"Background cleanup: removed {expired_count} expired items")
        
        except Exception as e:
            logger.error(f"Error in background cleanup: {e}")


# Start background cleanup thread if enabled
if ENABLE_BACKGROUND_CLEANUP:
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Performance optimization module initialized with background cleanup")
else:
    logger.info("Performance optimization module initialized (background cleanup disabled)")