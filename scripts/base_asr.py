#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base ASR Model Abstract Class
Defines the interface for all ASR model implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time


class ASRModel(ABC):
    """
    Abstract base class for ASR model implementations
    """
    
    def __init__(self, model_name: str, use_gpu: bool = True, **kwargs):
        """
        Initialize ASR model
        
        Args:
            model_name: Name or path of the model
            use_gpu: Whether to use GPU if available
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.is_loaded = False
        self.device = None
        
        # Performance tracking
        self.load_time = 0
        self.total_transcription_time = 0
        self.total_audio_duration = 0
        self.transcription_count = 0
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the ASR model
        
        Args:
            model_path: Path to model file (optional)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "th", **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            **kwargs: Additional transcription parameters
            
        Returns:
            Dict containing transcription results with keys:
            - 'text': Full transcription text
            - 'segments': List of segments with timestamps
            - 'language': Detected language
            - 'processing_time': Time taken for transcription
        """
        pass
    
    @abstractmethod
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available for this model
        
        Returns:
            bool: True if GPU is available
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        return {
            "name": self.model_name,
            "type": self.__class__.__name__,
            "device": str(self.device) if self.device else "unknown",
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "use_gpu": self.use_gpu
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dict containing performance metrics
        """
        avg_time = 0
        if self.transcription_count > 0:
            avg_time = self.total_transcription_time / self.transcription_count
        
        real_time_factor = 0
        if self.total_audio_duration > 0:
            real_time_factor = self.total_transcription_time / self.total_audio_duration
        
        return {
            "transcription_count": self.transcription_count,
            "total_transcription_time": self.total_transcription_time,
            "total_audio_duration": self.total_audio_duration,
            "average_transcription_time": avg_time,
            "real_time_factor": real_time_factor
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.total_transcription_time = 0
        self.total_audio_duration = 0
        self.transcription_count = 0
    
    def _track_transcription(self, start_time: float, audio_duration: float):
        """
        Track transcription performance
        
        Args:
            start_time: Start time of transcription
            audio_duration: Duration of audio in seconds
        """
        processing_time = time.time() - start_time
        
        self.total_transcription_time += processing_time
        self.total_audio_duration += audio_duration
        self.transcription_count += 1
        
        return processing_time
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in SRT format (HH:MM:SS,mmm)
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio file duration
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            float: Duration in seconds
        """
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0
    
    def preprocess_audio(self, audio_path: str, target_sample_rate: int = 16000) -> str:
        """
        Preprocess audio file for transcription
        
        Args:
            audio_path: Path to input audio file
            target_sample_rate: Target sample rate
            
        Returns:
            str: Path to preprocessed audio file
        """
        try:
            from pydub import AudioSegment
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert sample rate if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Export to temporary file
            temp_path = f"temp_preprocessed_{Path(audio_path).stem}.wav"
            audio.export(temp_path, format="wav")
            
            return temp_path
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio_path  # Return original if preprocessing fails
    
    def cleanup_temp_files(self, temp_path: str):
        """
        Clean up temporary files
        
        Args:
            temp_path: Path to temporary file
        """
        try:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception as e:
            print(f"Error cleaning up temp file {temp_path}: {e}")
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            bool: True if file is valid
        """
        if not Path(audio_path).exists():
            print(f"Audio file not found: {audio_path}")
            return False
        
        try:
            from pydub import AudioSegment
            AudioSegment.from_file(audio_path)
            return True
        except Exception as e:
            print(f"Invalid audio file {audio_path}: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device}, loaded={self.is_loaded})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', use_gpu={self.use_gpu}, is_loaded={self.is_loaded})"