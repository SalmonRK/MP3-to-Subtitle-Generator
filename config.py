#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for SRT Generator
Handles model selection, GPU settings, and other parameters
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for SRT Generator
    """
    
    DEFAULT_CONFIG = {
        # ASR Model Settings
        "asr_model": "whisper",  # "whisper" or "typhoon"
        "whisper_model_size": "large",  # "tiny", "base", "small", "medium", "large"
        "typhoon_model": "scb10x/typhoon-asr-realtime",
        
        # GPU Settings
        "use_gpu": True,
        "gpu_memory_fraction": 0.7,
        "use_half_precision": True,
        "fallback_to_cpu": True,
        
        # Transcription Settings
        "language": "th",  # Target language code
        "source_language": "auto",  # Source language or "auto"
        "task": "transcribe",  # "transcribe" or "translate"
        "chunk_duration_ms": 30000,  # Duration of audio chunks
        
        # Audio Processing
        "sample_rate": 16000,
        "audio_format": "wav",
        
        # Output Settings
        "output_format": "srt",
        "output_encoding": "utf-8",
        "word_timestamps": True,
        
        # Performance Settings
        "batch_size": 1,
        "num_workers": 1,
        "max_memory_usage": 0.8,  # Maximum memory usage fraction
        
        # Text Segmentation Settings
        "max_chars_per_segment": 42,  # Maximum characters per subtitle segment
        "max_segment_duration": 7.0,  # Maximum duration in seconds per segment
        
        # Paths
        "models_dir": "models",
        "output_dir": "output",
        "temp_dir": "temp",
        
        # Debug Settings
        "verbose": True,
        "save_debug_info": False,
        "log_level": "INFO"
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file (JSON format)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_file = config_file or "config.json"
        
        # Load configuration from file if it exists
        self.load_config()
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def load_config(self):
        """Load configuration from file"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Merge with default config
                self._merge_config(self.config, user_config)
                print(f"Configuration loaded from {config_path}")
                
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                print("Using default configuration")
        else:
            print(f"Configuration file {config_path} not found, using defaults")
            self.save_config()  # Save default config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            "SRT_ASR_MODEL": "asr_model",
            "SRT_WHISPER_MODEL_SIZE": "whisper_model_size",
            "SRT_USE_GPU": "use_gpu",
            "SRT_GPU_MEMORY_FRACTION": "gpu_memory_fraction",
            "SRT_LANGUAGE": "language",
            "SRT_CHUNK_DURATION": "chunk_duration_ms",
            "SRT_VERBOSE": "verbose",
            "SRT_LOG_LEVEL": "log_level"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key in ["use_gpu", "verbose"]:
                    self.config[config_key] = env_value.lower() in ("true", "1", "yes", "on")
                elif config_key in ["gpu_memory_fraction", "chunk_duration_ms", "max_memory_usage"]:
                    try:
                        self.config[config_key] = float(env_value)
                    except ValueError:
                        print(f"Invalid value for {env_var}: {env_value}")
                else:
                    self.config[config_key] = env_value
                
                print(f"Environment override: {config_key} = {self.config[config_key]}")
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.config["models_dir"],
            self.config["output_dir"],
            self.config["temp_dir"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        model_type = self.get("asr_model")
        
        if model_type == "whisper":
            return {
                "model_name": self.get("whisper_model_size"),
                "use_gpu": self.get("use_gpu"),
                "use_half_precision": self.get("use_half_precision"),
                "language": self.get("language"),
                "task": self.get("task")
            }
        elif model_type == "typhoon":
            return {
                "model_name": self.get("typhoon_model"),
                "use_gpu": self.get("use_gpu"),
                "use_half_precision": self.get("use_half_precision"),
                "language": self.get("language")
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU-specific configuration"""
        return {
            "use_gpu": self.get("use_gpu"),
            "memory_fraction": self.get("gpu_memory_fraction"),
            "use_half_precision": self.get("use_half_precision"),
            "fallback_to_cpu": self.get("fallback_to_cpu")
        }
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        
        # Model settings
        print(f"ASR Model: {self.get('asr_model')}")
        if self.get('asr_model') == 'whisper':
            print(f"Whisper Model Size: {self.get('whisper_model_size')}")
        
        # GPU settings
        print(f"Use GPU: {self.get('use_gpu')}")
        if self.get('use_gpu'):
            print(f"GPU Memory Fraction: {self.get('gpu_memory_fraction')}")
            print(f"Half Precision: {self.get('use_half_precision')}")
        
        # Transcription settings
        print(f"Language: {self.get('language')}")
        print(f"Chunk Duration: {self.get('chunk_duration_ms')}ms")
        
        # Paths
        print(f"Models Directory: {self.get('models_dir')}")
        print(f"Output Directory: {self.get('output_dir')}")
        
        print("="*50)


# Global configuration instance
_config = None

def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config: Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


def reset_config():
    """Reset the global configuration instance"""
    global _config
    _config = None


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.print_config()
    
    # Test getting and setting values
    print(f"\nModel config: {config.get_model_config()}")
    print(f"GPU config: {config.get_gpu_config()}")
    
    # Test environment override
    os.environ["SRT_USE_GPU"] = "false"
    config2 = Config()
    print(f"\nGPU setting after env override: {config2.get('use_gpu')}")