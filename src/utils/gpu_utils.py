"""
GPU Optimization Utilities for AI Customer Agent.

This module provides GPU detection, configuration, and performance monitoring
utilities to optimize the application for NVIDIA 4070Ti GPU and other CUDA devices.
Includes batch processing utilities and performance monitoring.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import wraps

import torch
import psutil
import GPUtil


@dataclass
class GPUInfo:
    """Data class to store GPU information."""
    name: str
    memory_total: float  # in MB
    memory_used: float   # in MB
    memory_free: float   # in MB
    utilization: float   # percentage
    temperature: float   # in Celsius
    driver_version: str
    cuda_version: str


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics."""
    inference_time: float
    memory_usage: float
    gpu_utilization: float
    batch_size: int
    throughput: float  # items per second


class GPUOptimizationError(Exception):
    """Custom exception for GPU optimization related errors."""
    
    def __init__(self, message: str, error_type: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


class GPUManager:
    """
    Manages GPU detection, configuration, and optimization.
    
    This class provides utilities for GPU setup, memory management,
    and performance monitoring specifically optimized for NVIDIA 4070Ti.
    """
    
    def __init__(self):
        """Initialize the GPU manager with logging and device detection."""
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.gpu_info = None
        self._initialize_device()
        
    def _initialize_device(self) -> None:
        """
        Configure PyTorch for optimal GPU performance.
        
        Sets up CUDA device with performance optimizations for NVIDIA 4070Ti.
        Falls back to CPU if GPU is not available.
        """
        try:
            if torch.cuda.is_available():
                # Set device to GPU
                self.device = torch.device("cuda")
                
                # Performance optimizations for NVIDIA 4070Ti
                torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
                torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
                
                # Get GPU information
                self.gpu_info = self._get_gpu_info()
                
                self.logger.info(f"GPU initialized: {self.gpu_info.name}")
                self.logger.info(f"GPU Memory: {self.gpu_info.memory_used:.1f}/{self.gpu_info.memory_total:.1f} MB")
                self.logger.info(f"CUDA Version: {self.gpu_info.cuda_version}")
                
            else:
                self.device = torch.device("cpu")
                self.logger.warning("CUDA not available, using CPU for computations")
                
        except Exception as e:
            self.logger.error(f"GPU initialization failed: {str(e)}")
            self.device = torch.device("cpu")
            raise GPUOptimizationError(f"GPU initialization failed: {str(e)}")
            
    def _get_gpu_info(self) -> GPUInfo:
        """
        Get detailed information about the available GPU.
        
        Returns:
            GPUInfo: Detailed GPU information including memory and utilization
            
        Raises:
            GPUOptimizationError: If GPU information cannot be retrieved
        """
        try:
            # Get CUDA version
            cuda_version = torch.version.cuda or "Unknown"
            
            # Get GPU details using GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                raise GPUOptimizationError("No GPU found")
                
            gpu = gpus[0]  # Use first available GPU
            
            return GPUInfo(
                name=gpu.name,
                memory_total=gpu.memoryTotal,
                memory_used=gpu.memoryUsed,
                memory_free=gpu.memoryFree,
                utilization=gpu.load * 100,  # Convert to percentage
                temperature=gpu.temperature,
                driver_version="Unknown",  # GPUtil doesn't provide driver version
                cuda_version=cuda_version
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU info: {str(e)}")
            raise GPUOptimizationError(f"Failed to get GPU info: {str(e)}")
            
    def get_device(self) -> torch.device:
        """
        Get the current computation device.
        
        Returns:
            torch.device: The current device (GPU or CPU)
        """
        return self.device
        
    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU status and statistics.
        
        Returns:
            Dictionary containing GPU status information
        """
        if self.device.type != "cuda" or not self.gpu_info:
            return {"gpu_available": False}
            
        try:
            # Update GPU information
            self.gpu_info = self._get_gpu_info()
            
            # Get additional PyTorch memory statistics
            if torch.cuda.is_available():
                torch_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                torch_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                torch_max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                torch_memory_allocated = 0
                torch_memory_reserved = 0
                torch_max_memory_allocated = 0
                
            return {
                "gpu_available": True,
                "gpu_name": self.gpu_info.name,
                "memory_used_mb": self.gpu_info.memory_used,
                "memory_total_mb": self.gpu_info.memory_total,
                "memory_free_mb": self.gpu_info.memory_free,
                "utilization_percent": self.gpu_info.utilization,
                "temperature_c": self.gpu_info.temperature,
                "cuda_version": self.gpu_info.cuda_version,
                "torch_memory_allocated_mb": torch_memory_allocated,
                "torch_memory_reserved_mb": torch_memory_reserved,
                "torch_max_memory_allocated_mb": torch_max_memory_allocated,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU status: {str(e)}")
            return {"gpu_available": False, "error": str(e)}
            
    def clear_gpu_cache(self) -> None:
        """
        Clear GPU memory cache to free up memory.
        
        Useful after large batch processing operations to prevent memory leaks.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
            
    def optimize_for_batch_processing(self, batch_size: int) -> Dict[str, Any]:
        """
        Optimize GPU settings for batch processing.
        
        Args:
            batch_size: The intended batch size for processing
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {}
        
        if self.device.type != "cuda":
            recommendations["warning"] = "GPU not available, using CPU"
            return recommendations
            
        # Memory-based batch size recommendations
        available_memory = self.gpu_info.memory_free
        estimated_memory_per_item = 100  # MB - conservative estimate
        
        max_recommended_batch = max(1, int(available_memory * 0.8 / estimated_memory_per_item))
        
        if batch_size > max_recommended_batch:
            recommendations["batch_size_adjustment"] = {
                "current": batch_size,
                "recommended": max_recommended_batch,
                "reason": f"Memory constraint: {available_memory:.1f} MB available"
            }
            
        # Performance optimizations
        recommendations["optimizations"] = {
            "cudnn_benchmark": True,
            "tf32_enabled": True,
            "memory_pinning": True if batch_size > 1 else False
        }
        
        return recommendations
        
    def monitor_performance(self, func):
        """
        Decorator to monitor performance of GPU-accelerated functions.
        
        Args:
            func: The function to monitor
            
        Returns:
            Wrapped function with performance monitoring
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if self.device.type == "cuda" else 0
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if self.device.type == "cuda" else 0
                
                # Calculate metrics
                inference_time = end_time - start_time
                memory_usage = (end_memory - start_memory) / 1024**2  # MB
                
                # Determine batch size from function arguments or result
                batch_size = 1
                if args and hasattr(args[0], 'shape') and len(args[0].shape) > 0:
                    batch_size = args[0].shape[0]
                elif kwargs.get('batch_size'):
                    batch_size = kwargs['batch_size']
                    
                throughput = batch_size / inference_time if inference_time > 0 else 0
                
                metrics = PerformanceMetrics(
                    inference_time=inference_time,
                    memory_usage=memory_usage,
                    gpu_utilization=self.gpu_info.utilization if self.gpu_info else 0,
                    batch_size=batch_size,
                    throughput=throughput
                )
                
                # Log performance metrics
                self.logger.info(
                    f"Performance - Function: {func.__name__}, "
                    f"Time: {inference_time:.3f}s, "
                    f"Memory: {memory_usage:.1f}MB, "
                    f"Throughput: {throughput:.1f} items/s"
                )
                
                # Store metrics for analysis
                if hasattr(wrapper, 'performance_history'):
                    wrapper.performance_history.append(metrics)
                else:
                    wrapper.performance_history = [metrics]
                    
                return result
                
            except Exception as e:
                self.logger.error(f"Performance monitoring failed for {func.__name__}: {str(e)}")
                raise
                
        return wrapper


class BatchProcessor:
    """
    Handles batch processing operations with GPU optimization.
    
    Provides utilities for efficient batch processing of embeddings
    and other GPU-accelerated operations.
    """
    
    def __init__(self, gpu_manager: GPUManager):
        """
        Initialize the batch processor.
        
        Args:
            gpu_manager: GPUManager instance for device management
        """
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.get_device()
        
    def process_batches(self, items: List[Any], batch_size: int, process_func) -> List[Any]:
        """
        Process items in batches with GPU optimization.
        
        Args:
            items: List of items to process
            batch_size: Size of each batch
            process_func: Function to process each batch
            
        Returns:
            List of processed results
        """
        if not items:
            return []
            
        # Get optimization recommendations
        optimizations = self.gpu_manager.optimize_for_batch_processing(batch_size)
        
        # Adjust batch size if recommended
        if "batch_size_adjustment" in optimizations:
            recommended_batch = optimizations["batch_size_adjustment"]["recommended"]
            if recommended_batch < batch_size:
                self.logger.warning(
                    f"Reducing batch size from {batch_size} to {recommended_batch} "
                    f"due to memory constraints"
                )
                batch_size = recommended_batch
                
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {len(items)} items in {total_batches} batches (size: {batch_size})")
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                # Process batch with performance monitoring
                @self.gpu_manager.monitor_performance
                def process_batch():
                    return process_func(batch)
                    
                batch_result = process_batch()
                results.extend(batch_result)
                
                # Clear cache periodically to prevent memory buildup
                if (i // batch_size) % 10 == 0:  # Every 10 batches
                    self.gpu_manager.clear_gpu_cache()
                    
            except Exception as e:
                self.logger.error(f"Batch processing failed at batch {i//batch_size + 1}: {str(e)}")
                raise GPUOptimizationError(f"Batch processing failed: {str(e)}")
                
        return results
        
    def calculate_optimal_batch_size(self, item_size_estimate: float) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            item_size_estimate: Estimated memory per item in MB
            
        Returns:
            Optimal batch size
        """
        if self.device.type != "cuda":
            return 32  # Default for CPU
            
        gpu_status = self.gpu_manager.get_gpu_status()
        if not gpu_status.get("gpu_available", False):
            return 32
            
        available_memory = gpu_status["memory_free_mb"]
        
        # Use 70% of available memory for safety
        usable_memory = available_memory * 0.7
        
        # Calculate batch size
        batch_size = max(1, int(usable_memory / item_size_estimate))
        
        # Cap at reasonable maximum
        batch_size = min(batch_size, 256)
        
        self.logger.info(
            f"Optimal batch size: {batch_size} "
            f"(available: {available_memory:.1f}MB, "
            f"item size: {item_size_estimate:.1f}MB)"
        )
        
        return batch_size


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """
    Get or create the global GPU manager instance.
    
    Returns:
        GPUManager: Global GPU manager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def get_gpu_status() -> Dict[str, Any]:
    """
    Get current GPU status.
    
    Returns:
        Dictionary with GPU status information
    """
    return get_gpu_manager().get_gpu_status()


def setup_gpu() -> torch.device:
    """
    Set up GPU for optimal performance.
    
    Returns:
        torch.device: The configured device
    """
    return get_gpu_manager().get_device()
