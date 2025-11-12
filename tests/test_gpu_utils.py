"""
Unit tests for GPU optimization utilities.

Tests the GPUManager, BatchProcessor, and related utility functions
for the AI Customer Agent application.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.gpu_utils import (
    GPUManager, 
    BatchProcessor, 
    GPUInfo, 
    PerformanceMetrics,
    GPUOptimizationError,
    get_gpu_manager,
    get_gpu_status,
    setup_gpu
)


class TestGPUInfo:
    """Test cases for GPUInfo data class."""
    
    def test_gpu_info_creation(self):
        """Test creating GPUInfo instance with valid data."""
        gpu_info = GPUInfo(
            name="NVIDIA GeForce RTX 4070 Ti",
            memory_total=12288.0,
            memory_used=2048.0,
            memory_free=10240.0,
            utilization=45.5,
            temperature=65.0,
            driver_version="546.17",
            cuda_version="12.2"
        )
        
        assert gpu_info.name == "NVIDIA GeForce RTX 4070 Ti"
        assert gpu_info.memory_total == 12288.0
        assert gpu_info.memory_used == 2048.0
        assert gpu_info.memory_free == 10240.0
        assert gpu_info.utilization == 45.5
        assert gpu_info.temperature == 65.0
        assert gpu_info.driver_version == "546.17"
        assert gpu_info.cuda_version == "12.2"


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics data class."""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance with valid data."""
        metrics = PerformanceMetrics(
            inference_time=0.123,
            memory_usage=512.5,
            gpu_utilization=45.5,
            batch_size=32,
            throughput=260.16
        )
        
        assert metrics.inference_time == 0.123
        assert metrics.memory_usage == 512.5
        assert metrics.gpu_utilization == 45.5
        assert metrics.batch_size == 32
        assert metrics.throughput == 260.16


class TestGPUManager:
    """Test cases for GPUManager class."""
    
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    @patch('src.utils.gpu_utils.GPUtil.getGPUs')
    def test_gpu_manager_initialization_with_gpu(self, mock_get_gpus, mock_cuda_available):
        """Test GPUManager initialization when GPU is available."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        # Mock GPU information
        mock_gpu = Mock()
        mock_gpu.name = "NVIDIA GeForce RTX 4070 Ti"
        mock_gpu.memoryTotal = 12288.0
        mock_gpu.memoryUsed = 2048.0
        mock_gpu.memoryFree = 10240.0
        mock_gpu.load = 0.455
        mock_gpu.temperature = 65.0
        mock_get_gpus.return_value = [mock_gpu]
        
        # Mock CUDA version
        with patch('torch.version.cuda', '12.2'):
            gpu_manager = GPUManager()
            
        assert gpu_manager.device.type == "cuda"
        assert gpu_manager.gpu_info is not None
        assert gpu_manager.gpu_info.name == "NVIDIA GeForce RTX 4070 Ti"
        
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    def test_gpu_manager_initialization_without_gpu(self, mock_cuda_available):
        """Test GPUManager initialization when GPU is not available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        
        gpu_manager = GPUManager()
        
        assert gpu_manager.device.type == "cpu"
        assert gpu_manager.gpu_info is None
        
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    @patch('src.utils.gpu_utils.GPUtil.getGPUs')
    def test_get_gpu_status_with_gpu(self, mock_get_gpus, mock_cuda_available):
        """Test getting GPU status when GPU is available."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        # Mock GPU information
        mock_gpu = Mock()
        mock_gpu.name = "NVIDIA GeForce RTX 4070 Ti"
        mock_gpu.memoryTotal = 12288.0
        mock_gpu.memoryUsed = 2048.0
        mock_gpu.memoryFree = 10240.0
        mock_gpu.load = 0.455
        mock_gpu.temperature = 65.0
        mock_get_gpus.return_value = [mock_gpu]
        
        # Mock CUDA version and memory functions
        with patch('torch.version.cuda', '12.2'), \
             patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 1024), \
             patch('torch.cuda.memory_reserved', return_value=2048 * 1024 * 1024), \
             patch('torch.cuda.max_memory_allocated', return_value=3072 * 1024 * 1024):
            
            gpu_manager = GPUManager()
            status = gpu_manager.get_gpu_status()
            
            assert status["gpu_available"] == True
            assert status["gpu_name"] == "NVIDIA GeForce RTX 4070 Ti"
            assert status["memory_used_mb"] == 2048.0
            assert status["memory_total_mb"] == 12288.0
            assert status["memory_free_mb"] == 10240.0
            assert status["utilization_percent"] == 45.5
            assert status["temperature_c"] == 65.0
            assert status["cuda_version"] == "12.2"
            
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    def test_get_gpu_status_without_gpu(self, mock_cuda_available):
        """Test getting GPU status when GPU is not available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        
        gpu_manager = GPUManager()
        status = gpu_manager.get_gpu_status()
        
        assert status["gpu_available"] == False
        
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    def test_clear_gpu_cache_with_gpu(self, mock_cuda_available):
        """Test clearing GPU cache when GPU is available."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            gpu_manager = GPUManager()
            gpu_manager.device = torch.device("cuda")
            gpu_manager.clear_gpu_cache()
            
            mock_empty_cache.assert_called_once()
            
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    def test_clear_gpu_cache_without_gpu(self, mock_cuda_available):
        """Test clearing GPU cache when GPU is not available."""
        # Mock no GPU available
        mock_cuda_available.return_value = False
        
        gpu_manager = GPUManager()
        gpu_manager.device = torch.device("cpu")
        
        # Should not raise an exception
        gpu_manager.clear_gpu_cache()
        
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    @patch('src.utils.gpu_utils.GPUtil.getGPUs')
    def test_optimize_for_batch_processing(self, mock_get_gpus, mock_cuda_available):
        """Test batch processing optimization recommendations."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        # Mock GPU with limited memory
        mock_gpu = Mock()
        mock_gpu.name = "NVIDIA GeForce RTX 4070 Ti"
        mock_gpu.memoryTotal = 12288.0
        mock_gpu.memoryUsed = 2048.0
        mock_gpu.memoryFree = 100.0  # Only 100MB free
        mock_gpu.load = 0.455
        mock_gpu.temperature = 65.0
        mock_get_gpus.return_value = [mock_gpu]
        
        with patch('torch.version.cuda', '12.2'):
            gpu_manager = GPUManager()
            recommendations = gpu_manager.optimize_for_batch_processing(batch_size=100)
            
            # Should recommend smaller batch size due to memory constraints
            assert "batch_size_adjustment" in recommendations
            assert recommendations["batch_size_adjustment"]["current"] == 100
            assert recommendations["batch_size_adjustment"]["recommended"] < 100
            
    def test_monitor_performance_decorator(self):
        """Test the performance monitoring decorator."""
        gpu_manager = GPUManager()
        gpu_manager.device = torch.device("cpu")  # Use CPU for testing
        
        # Create a mock function to monitor
        @gpu_manager.monitor_performance
        def test_function(data):
            return [x * 2 for x in data]
            
        # Test the decorated function
        test_data = [1, 2, 3, 4, 5]
        result = test_function(test_data)
        
        assert result == [2, 4, 6, 8, 10]
        # Check that performance history was stored
        assert hasattr(test_function, 'performance_history')
        assert len(test_function.performance_history) == 1


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    def test_process_batches_empty_input(self):
        """Test batch processing with empty input."""
        gpu_manager = GPUManager()
        batch_processor = BatchProcessor(gpu_manager)
        
        result = batch_processor.process_batches([], 10, lambda x: x)
        
        assert result == []
        
    def test_process_batches_single_batch(self):
        """Test batch processing with single batch."""
        gpu_manager = GPUManager()
        batch_processor = BatchProcessor(gpu_manager)
        
        items = [1, 2, 3, 4, 5]
        process_func = lambda batch: [x * 2 for x in batch]
        
        result = batch_processor.process_batches(items, 10, process_func)
        
        assert result == [2, 4, 6, 8, 10]
        
    def test_process_batches_multiple_batches(self):
        """Test batch processing with multiple batches."""
        gpu_manager = GPUManager()
        batch_processor = BatchProcessor(gpu_manager)
        
        items = list(range(10))  # [0, 1, 2, ..., 9]
        process_func = lambda batch: [x * 2 for x in batch]
        
        result = batch_processor.process_batches(items, 3, process_func)
        
        expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert result == expected
        
    @patch('src.utils.gpu_utils.torch.cuda.is_available')
    def test_calculate_optimal_batch_size_with_gpu(self, mock_cuda_available):
        """Test calculating optimal batch size with GPU available."""
        # Mock GPU availability
        mock_cuda_available.return_value = True
        
        gpu_manager = GPUManager()
        gpu_manager.device = torch.device("cuda")
        
        # Mock GPU status with specific memory
        with patch.object(gpu_manager, 'get_gpu_status') as mock_status:
            mock_status.return_value = {
                "gpu_available": True,
                "memory_free_mb": 700.0  # 700MB free
            }
            
            batch_processor = BatchProcessor(gpu_manager)
            
            # With 10MB per item estimate, should get batch size around 49 (700 * 0.7 / 10)
            optimal_batch = batch_processor.calculate_optimal_batch_size(10.0)
            
            assert optimal_batch > 0
            assert optimal_batch <= 256  # Should be capped at 256
            
    def test_calculate_optimal_batch_size_without_gpu(self):
        """Test calculating optimal batch size without GPU."""
        gpu_manager = GPUManager()
        gpu_manager.device = torch.device("cpu")
        
        batch_processor = BatchProcessor(gpu_manager)
        
        # Should return default CPU batch size
        optimal_batch = batch_processor.calculate_optimal_batch_size(10.0)
        
        assert optimal_batch == 32


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_get_gpu_manager_singleton(self):
        """Test that get_gpu_manager returns a singleton instance."""
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        
        assert manager1 is manager2
        
    @patch('src.utils.gpu_utils.get_gpu_manager')
    def test_get_gpu_status(self, mock_get_gpu_manager):
        """Test the get_gpu_status utility function."""
        mock_manager = Mock()
        mock_manager.get_gpu_status.return_value = {"gpu_available": True}
        mock_get_gpu_manager.return_value = mock_manager
        
        status = get_gpu_status()
        
        assert status == {"gpu_available": True}
        mock_manager.get_gpu_status.assert_called_once()
        
    @patch('src.utils.gpu_utils.get_gpu_manager')
    def test_setup_gpu(self, mock_get_gpu_manager):
        """Test the setup_gpu utility function."""
        mock_manager = Mock()
        mock_manager.get_device.return_value = torch.device("cuda")
        mock_get_gpu_manager.return_value = mock_manager
        
        device = setup_gpu()
        
        assert device.type == "cuda"
        mock_manager.get_device.assert_called_once()


class TestGPUOptimizationError:
    """Test cases for GPUOptimizationError exception."""
    
    def test_gpu_optimization_error_creation(self):
        """Test creating GPUOptimizationError instance."""
        error = GPUOptimizationError("Test error message", "test_error")
        
        assert str(error) == "Test error message"
        assert error.error_type == "test_error"
        
    def test_gpu_optimization_error_without_error_type(self):
        """Test creating GPUOptimizationError without error type."""
        error = GPUOptimizationError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.error_type is None


if __name__ == "__main__":
    pytest.main([__file__])
