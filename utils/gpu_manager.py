#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Manager for SRT Generator
Handles GPU detection, optimization, and resource management
"""

import os
import sys
import gc
from typing import Dict, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class GPUManager:
    """
    Manages GPU detection, optimization, and resource management for ASR models
    """
    
    def __init__(self):
        self.gpu_available = self._detect_gpu()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else {}
        
    def _detect_gpu(self) -> bool:
        """
        Detect if CUDA GPU is available
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, GPU support disabled")
            return False
            
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("No CUDA GPU detected, will use CPU")
            return False
    
    def _get_gpu_info(self) -> Dict:
        """
        Get detailed GPU information
        
        Returns:
            Dict: GPU information including name, memory, etc.
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}
            
        try:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            
            # Get multiprocessor count with fallback for older PyTorch versions
            try:
                multiprocessors = properties.multiprocessor_count
            except AttributeError:
                multiprocessors = "Unknown"
            
            return {
                "available": True,
                "name": torch.cuda.get_device_name(device),
                "device_id": device,
                "total_memory": properties.total_memory,
                "free_memory": properties.total_memory - torch.cuda.memory_allocated(device),
                "allocated_memory": torch.cuda.memory_allocated(device),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multiprocessor_count": multiprocessors
            }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return {"available": False, "error": str(e)}
    
    def get_optimal_device(self) -> str:
        """
        Get the optimal device for model inference
        
        Returns:
            str: Device string ('cuda' or 'cpu')
        """
        if self.gpu_available:
            # Check if we have enough GPU memory
            if self.gpu_info.get("free_memory", 0) > 1024 * 1024 * 1024:  # 1GB minimum
                return "cuda"
            else:
                print("Warning: Low GPU memory, falling back to CPU")
                return "cpu"
        return "cpu"
    
    def optimize_model_for_gpu(self, model, use_half_precision: bool = True):
        """
        Optimize model for GPU inference
        
        Args:
            model: The model to optimize
            use_half_precision: Whether to use FP16 for faster inference
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, returning original model")
            return model
            
        if not self.gpu_available:
            print("GPU not available, keeping model on CPU")
            return model
            
        try:
            device = self.get_optimal_device()
            
            if device == "cuda":
                model = model.cuda()
                
                # Use half precision for faster inference if requested
                if use_half_precision and hasattr(model, 'half'):
                    try:
                        model = model.half()
                        print("Model optimized with half precision (FP16)")
                    except Exception as e:
                        print(f"Could not enable half precision: {e}")
                        print("Using full precision (FP32)")
                
                # Enable optimizations
                if hasattr(torch.backends.cudnn, 'benchmark'):
                    torch.backends.cudnn.benchmark = True
                    print("Enabled cuDNN benchmark mode")
                
                print(f"Model optimized for GPU: {self.gpu_info.get('name', 'Unknown')}")
            else:
                print("Keeping model on CPU")
                
            return model
            
        except Exception as e:
            print(f"Error optimizing model for GPU: {e}")
            print("Falling back to CPU")
            return model.cpu() if hasattr(model, 'cpu') else model
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU cache cleared")
    
    def get_memory_usage(self) -> Dict:
        """
        Get current memory usage
        
        Returns:
            Dict: Memory usage information
        """
        memory_info = {}
        
        # GPU memory
        if self.gpu_available:
            device = torch.cuda.current_device()
            memory_info.update({
                "gpu_allocated": torch.cuda.memory_allocated(device),
                "gpu_reserved": torch.cuda.memory_reserved(device),
                "gpu_free": self.gpu_info.get("total_memory", 0) - torch.cuda.memory_allocated(device)
            })
        
        # CPU memory
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_info.update({
                "cpu_total": memory.total,
                "cpu_available": memory.available,
                "cpu_percent": memory.percent
            })
        
        return memory_info
    
    def print_gpu_info(self):
        """Print detailed GPU information"""
        if self.gpu_available:
            print("\n" + "="*50)
            print("GPU INFORMATION")
            print("="*50)
            print(f"Name: {self.gpu_info.get('name', 'Unknown')}")
            print(f"Total Memory: {self.gpu_info.get('total_memory', 0) / 1024**3:.2f} GB")
            print(f"Free Memory: {self.gpu_info.get('free_memory', 0) / 1024**3:.2f} GB")
            print(f"Compute Capability: {self.gpu_info.get('compute_capability', 'Unknown')}")
            print(f"Multiprocessors: {self.gpu_info.get('multiprocessor_count', 'Unknown')}")
            print("="*50)
        else:
            print("No GPU available or detected")
    
    def estimate_model_memory(self, model_name: str) -> int:
        """
        Estimate memory requirements for different models
        
        Args:
            model_name: Name of the model (whisper-tiny, whisper-base, etc.)
            
        Returns:
            int: Estimated memory in bytes
        """
        # Rough estimates for model sizes
        model_sizes = {
            "whisper-tiny": 150 * 1024 * 1024,    # 150MB
            "whisper-base": 300 * 1024 * 1024,    # 300MB
            "whisper-small": 800 * 1024 * 1024,   # 800MB
            "whisper-medium": 1500 * 1024 * 1024, # 1.5GB
            "whisper-large": 3000 * 1024 * 1024,  # 3GB
            "typhoon-asr": 2000 * 1024 * 1024,    # 2GB (estimate)
        }
        
        # Account for additional memory during inference
        base_size = model_sizes.get(model_name.lower(), 1000 * 1024 * 1024)
        inference_overhead = base_size * 0.5  # 50% overhead
        
        return int(base_size + inference_overhead)
    
    def can_fit_model(self, model_name: str) -> bool:
        """
        Check if a model can fit in available GPU memory
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if model can fit, False otherwise
        """
        if not self.gpu_available:
            return False
            
        required_memory = self.estimate_model_memory(model_name)
        available_memory = self.gpu_info.get("free_memory", 0)
        
        # Add some buffer (10% of available memory)
        safe_memory = available_memory * 0.9
        
        return required_memory < safe_memory


# Singleton instance for easy access
_gpu_manager = None

def get_gpu_manager() -> GPUManager:
    """
    Get the singleton GPU manager instance
    
    Returns:
        GPUManager: The GPU manager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def print_system_info():
    """Print comprehensive system information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # Python info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # GPU info
    gpu_manager = get_gpu_manager()
    gpu_manager.print_gpu_info()
    
    # Memory info
    memory = gpu_manager.get_memory_usage()
    if "cpu_total" in memory:
        print(f"\nCPU Memory: {memory['cpu_total'] / 1024**3:.2f} GB")
        print(f"CPU Available: {memory['cpu_available'] / 1024**3:.2f} GB")
        print(f"CPU Usage: {memory['cpu_percent']:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    # Test the GPU manager
    print("Testing GPU Manager...")
    print_system_info()
    
    gpu_manager = get_gpu_manager()
    print(f"Optimal device: {gpu_manager.get_optimal_device()}")
    
    # Test memory estimation
    models_to_test = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large"]
    print("\nModel Memory Requirements:")
    for model in models_to_test:
        can_fit = gpu_manager.can_fit_model(model)
        required = gpu_manager.estimate_model_memory(model) / 1024**3
        print(f"{model}: {required:.2f} GB - {'✓ Can fit' if can_fit else '✗ Too large'}")