#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper ASR Model Implementation
Extends the base ASRModel class for OpenAI Whisper
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from scripts.base_asr import ASRModel
from utils.gpu_manager import get_gpu_manager


class WhisperModel(ASRModel):
    """
    Whisper ASR model implementation
    """
    
    def __init__(self, model_name: str = "large", use_gpu: bool = True, **kwargs):
        """
        Initialize Whisper model
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            use_gpu: Whether to use GPU if available
            **kwargs: Additional parameters
        """
        super().__init__(model_name, use_gpu, **kwargs)
        
        # Whisper-specific settings
        self.task = kwargs.get("task", "transcribe")
        self.language = kwargs.get("language", "th")
        self.word_timestamps = kwargs.get("word_timestamps", True)
        
        # GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Check if Whisper is available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import whisper
            self.whisper_module = whisper
        except ImportError:
            raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load Whisper model
        
        Args:
            model_path: Path to local model file (optional)
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.is_loaded:
            print(f"Whisper model {self.model_name} already loaded")
            return True
        
        try:
            start_time = time.time()
            
            # Determine device
            if self.use_gpu and self.gpu_manager.gpu_available:
                self.device = "cuda"
                print(f"Loading Whisper model {self.model_name} on GPU...")
            else:
                self.device = "cpu"
                print(f"Loading Whisper model {self.model_name} on CPU...")
            
            # Try to load local model first
            if model_path and Path(model_path).exists():
                print(f"Loading model from local path: {model_path}")
                self.model = self.whisper_module.load_model("large")  # Load base structure
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Download/load from internet
                print(f"Loading model {self.model_name} (may download if not cached)...")
                self.model = self.whisper_module.load_model(self.model_name, device=self.device)
            
            # Optimize for GPU if available (disable half-precision to avoid errors)
            if self.device == "cuda":
                self.model = self.gpu_manager.optimize_model_for_gpu(
                    self.model,
                    use_half_precision=False  # Disable half-precision to avoid errors
                )
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            print(f"Whisper model {self.model_name} loaded successfully in {self.load_time:.2f}s")
            print(f"Model device: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.is_loaded = False
            return False
    
    def transcribe(self, audio_path: str, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (overrides instance default)
            **kwargs: Additional transcription parameters
            
        Returns:
            Dict containing transcription results
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        # Validate audio file
        if not self.validate_audio_file(audio_path):
            return {"error": "Invalid audio file"}
        
        # Use provided language or default
        lang = language or self.language
        
        # Get audio duration
        audio_duration = self.get_audio_duration(audio_path)
        
        # Preprocess audio
        temp_path = None
        try:
            temp_path = self.preprocess_audio(audio_path, target_sample_rate=16000)
            audio_to_transcribe = temp_path
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            audio_to_transcribe = audio_path
        
        try:
            start_time = time.time()
            
            # Prepare transcription options (disable fp16 to avoid errors)
            options = {
                "task": kwargs.get("task", self.task),
                "language": lang,
                "word_timestamps": kwargs.get("word_timestamps", self.word_timestamps),
                "fp16": False  # Disable half-precision to avoid errors
            }
            
            # Transcribe
            print(f"Transcribing with Whisper {self.model_name} (language: {lang})...")
            try:
                result = self.model.transcribe(audio_to_transcribe, **options)
            except RuntimeError as e:
                if "expected scalar type Float but found Half" in str(e):
                    # Fallback to full precision if half precision fails
                    print("Half precision failed, falling back to full precision...")
                    options["fp16"] = False
                    result = self.model.transcribe(audio_to_transcribe, **options)
                else:
                    raise
            
            # Track performance
            processing_time = self._track_transcription(start_time, audio_duration)
            
            # Format results
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "words": segment.get("words", [])
                })
            
            return {
                "text": result.get("text", "").strip(),
                "segments": segments,
                "language": result.get("language", lang),
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "model": self.model_name,
                "device": str(self.device)
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"error": str(e)}
        
        finally:
            # Clean up temporary files
            if temp_path:
                self.cleanup_temp_files(temp_path)
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for Whisper"""
        return self.gpu_manager.gpu_available
    
    def load_local_model(self, model_path: str) -> bool:
        """
        Load Whisper model from local file
        
        Args:
            model_path: Path to local model file (.pt)
            
        Returns:
            bool: True if model loaded successfully
        """
        return self.load_model(model_path=model_path)
    
    def download_model(self, model_name: str = None, save_path: str = None) -> str:
        """
        Download Whisper model and save locally
        
        Args:
            model_name: Model size to download
            save_path: Path to save the model
            
        Returns:
            str: Path to downloaded model
        """
        model_to_download = model_name or self.model_name
        
        if save_path is None:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            save_path = str(models_dir / f"{model_to_download}.pt")
        
        try:
            print(f"Downloading Whisper model {model_to_download}...")
            
            # Load model (this downloads if not cached)
            model = self.whisper_module.load_model(model_to_download)
            
            # Save to local file
            torch.save(model.state_dict(), save_path)
            
            print(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        # Whisper supports many languages, returning common ones
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def set_language(self, language: str):
        """
        Set default language for transcription
        
        Args:
            language: Language code
        """
        self.language = language
    
    def set_task(self, task: str):
        """
        Set task type (transcribe or translate)
        
        Args:
            task: Task type
        """
        if task in ["transcribe", "translate"]:
            self.task = task
        else:
            raise ValueError("Task must be 'transcribe' or 'translate'")
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size and requirements
        
        Returns:
            Dict with model size information
        """
        size_info = {
            "tiny": {"params": "39M", "size": "150MB", "english_only": False},
            "base": {"params": "74M", "size": "300MB", "english_only": False},
            "small": {"params": "244M", "size": "800MB", "english_only": False},
            "medium": {"params": "769M", "size": "1.5GB", "english_only": False},
            "large": {"params": "1550M", "size": "3GB", "english_only": False},
            "large-v1": {"params": "1550M", "size": "3GB", "english_only": False},
            "large-v2": {"params": "1550M", "size": "3GB", "english_only": False},
            "large-v3": {"params": "1550M", "size": "3GB", "english_only": False}
        }
        
        return size_info.get(self.model_name, {"params": "Unknown", "size": "Unknown"})