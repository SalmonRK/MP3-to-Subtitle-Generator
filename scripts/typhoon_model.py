#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typhoon ASR Model Implementation
Extends the base ASRModel class for SCB10X Typhoon ASR Realtime
"""

import os
import sys
import time
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from scripts.base_asr import ASRModel
from utils.gpu_manager import get_gpu_manager


class TyphoonModel(ASRModel):
    """
    Typhoon ASR model implementation for Thai speech recognition
    """
    
    def __init__(self, model_name: str = "scb10x/typhoon-asr-realtime", use_gpu: bool = True, **kwargs):
        """
        Initialize Typhoon ASR model
        
        Args:
            model_name: Hugging Face model identifier
            use_gpu: Whether to use GPU if available
            **kwargs: Additional parameters
        """
        super().__init__(model_name, use_gpu, **kwargs)
        
        # Typhoon-specific settings
        self.language = kwargs.get("language", "th")
        self.sample_rate = 16000  # Typhoon typically uses 16kHz
        
        # GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Model components
        self.model = None
        self.processor = None
        
        # Check if dependencies are available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        missing_deps = []
        
        try:
            import transformers
            self.transformers = transformers
        except ImportError:
            missing_deps.append("transformers")
        
        try:
            import torch
            self.torch = torch
        except ImportError:
            missing_deps.append("torch")
        
        try:
            import torchaudio
            self.torchaudio = torchaudio
        except ImportError:
            missing_deps.append("torchaudio")
        
        if missing_deps:
            raise ImportError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load Typhoon ASR model
        
        Args:
            model_path: Path to local model (optional, defaults to Hugging Face model)
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.is_loaded:
            print(f"Typhoon model {self.model_name} already loaded")
            return True
        
        try:
            start_time = time.time()
            
            # Determine device
            if self.use_gpu and self.gpu_manager.gpu_available:
                self.device = "cuda"
                print(f"Loading Typhoon model {self.model_name} on GPU...")
            else:
                self.device = "cpu"
                print(f"Loading Typhoon model {self.model_name} on CPU...")
            
            # Load model and processor from Hugging Face
            print("Loading model from Hugging Face...")
            
            # Determine torch dtype based on device and memory
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Try using the pipeline approach first
            try:
                print("Trying automatic speech recognition pipeline...")
                self.pipeline = self.transformers.pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=self.device
                )
                self.model = None  # We'll use the pipeline instead
                self.processor = None
                print("Successfully loaded using pipeline approach")
            except Exception as e1:
                print(f"Pipeline approach failed: {e1}")
                
                # Load processor
                try:
                    self.processor = self.transformers.AutoProcessor.from_pretrained(self.model_name)
                except Exception as e2:
                    print(f"Failed to load processor: {e2}")
                    self.processor = None
                
                # Load model - Try different model classes for speech recognition
                try:
                    # First try AutoModelForSpeechSeq2Seq
                    self.model = self.transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True
                    ).to(self.device)
                except Exception as e3:
                    print(f"AutoModelForSpeechSeq2Seq failed: {e3}")
                    try:
                        # Try AutoModelForCTC
                        self.model = self.transformers.AutoModelForCTC.from_pretrained(
                            self.model_name,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        ).to(self.device)
                    except Exception as e4:
                        print(f"AutoModelForCTC failed: {e4}")
                        try:
                            # Try generic AutoModel
                            self.model = self.transformers.AutoModel.from_pretrained(
                                self.model_name,
                                torch_dtype=torch_dtype,
                                low_cpu_mem_usage=True
                            ).to(self.device)
                        except Exception as e5:
                            print(f"AutoModel failed: {e5}")
                            raise Exception(f"Failed to load model with all attempted methods: {e1}, {e2}, {e3}, {e4}, {e5}")
            
            # Optimize for GPU if available
            if self.device == "cuda":
                self.model = self.gpu_manager.optimize_model_for_gpu(
                    self.model, 
                    use_half_precision=(torch_dtype == torch.float16)
                )
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            print(f"Typhoon model loaded successfully in {self.load_time:.2f}s")
            print(f"Model device: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading Typhoon model: {e}")
            self.is_loaded = False
            return False
    
    def transcribe(self, audio_path: str, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using Typhoon ASR
        
        Args:
            audio_path: Path to audio file
            language: Language code (Typhoon is primarily for Thai)
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
        
        # Use provided language or default (Typhoon is primarily for Thai)
        lang = language or self.language
        
        # Get audio duration
        audio_duration = self.get_audio_duration(audio_path)
        
        # Preprocess audio
        temp_path = None
        try:
            temp_path = self.preprocess_audio(audio_path, target_sample_rate=self.sample_rate)
            audio_to_transcribe = temp_path
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            audio_to_transcribe = audio_path
        
        try:
            start_time = time.time()
            
            # If we have a pipeline, use it directly
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                print(f"Transcribing with Typhoon ASR pipeline (language: {lang})...")
                try:
                    result = self.pipeline(
                        audio_to_transcribe,
                        generate_kwargs={"language": lang, "task": "transcribe"}
                    )
                    transcription = result.get("text", "")
                except Exception as e:
                    print(f"Pipeline transcription failed: {e}")
                    # Try without language specification
                    try:
                        result = self.pipeline(audio_to_transcribe)
                        transcription = result.get("text", "")
                    except Exception as e2:
                        print(f"Pipeline without language failed: {e2}")
                        transcription = "Pipeline transcription failed"
            else:
                # Manual model approach
                # Load and prepare audio
                waveform, sample_rate = self.torchaudio.load(audio_to_transcribe)
                
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sample_rate != self.sample_rate:
                    resampler = self.torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Move to device
                waveform = waveform.to(self.device)
                
                # Process audio
                if self.processor:
                    inputs = self.processor(
                        waveform.squeeze().numpy(),
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    # If no processor, try direct audio input
                    inputs = waveform.squeeze().numpy()
                
                # Generate transcription
                print(f"Transcribing with Typhoon ASR model (language: {lang})...")
                
                with torch.no_grad():
                    try:
                        if self.processor:
                            # Try standard generation first
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=448,  # Typical max length for speech
                                num_beams=2,
                                early_stopping=True
                            )
                            
                            # Decode transcription
                            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        else:
                            # Try forward pass without processor
                            outputs = self.model(inputs)
                            if hasattr(outputs, 'logits'):
                                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                                transcription = str(predicted_ids[0].tolist())
                            else:
                                transcription = "Model output format not recognized"
                    except Exception as e:
                        print(f"Manual transcription failed: {e}")
                        transcription = "Manual transcription failed - model not compatible"
            
            # Track performance
            processing_time = self._track_transcription(start_time, audio_duration)
            
            # Create segments (Typhoon doesn't provide word-level timestamps by default)
            # We'll create a single segment for the entire audio
            segments = [{
                "start": 0.0,
                "end": audio_duration,
                "text": transcription.strip(),
                "words": []  # Empty for now
            }]
            
            return {
                "text": transcription.strip(),
                "segments": segments,
                "language": lang,
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
        """Check if GPU is available for Typhoon"""
        return self.gpu_manager.gpu_available
    
    def transcribe_with_timestamps(self, audio_path: str, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio with forced alignment for better timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Dict containing transcription with detailed timestamps
        """
        # For now, use the standard transcription method
        # In a full implementation, this would use forced alignment techniques
        return self.transcribe(audio_path, language, **kwargs)
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of language codes (Typhoon is primarily for Thai)
        """
        # Typhoon ASR is primarily designed for Thai
        return ["th"]
    
    def set_language(self, language: str):
        """
        Set default language for transcription
        
        Args:
            language: Language code
        """
        if language in self.get_supported_languages():
            self.language = language
        else:
            print(f"Warning: Typhoon ASR primarily supports Thai, got {language}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        base_info = super().get_model_info()
        
        # Add Typhoon-specific information
        base_info.update({
            "model_type": "Typhoon ASR Realtime",
            "primary_language": "Thai",
            "sample_rate": self.sample_rate,
            "processor": "AutoProcessor" if self.processor else None
        })
        
        return base_info
    
    def benchmark_model(self, audio_path: str, num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            audio_path: Path to test audio file
            num_runs: Number of benchmark runs
            
        Returns:
            Dict containing benchmark results
        """
        if not self.validate_audio_file(audio_path):
            return {"error": "Invalid audio file"}
        
        audio_duration = self.get_audio_duration(audio_path)
        processing_times = []
        
        print(f"Benchmarking Typhoon model with {num_runs} runs...")
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            
            start_time = time.time()
            result = self.transcribe(audio_path)
            end_time = time.time()
            
            if "error" not in result:
                processing_times.append(end_time - start_time)
            else:
                print(f"  Error in run {i+1}: {result['error']}")
        
        if not processing_times:
            return {"error": "All benchmark runs failed"}
        
        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        avg_rtf = avg_time / audio_duration if audio_duration > 0 else 0
        
        return {
            "model": self.model_name,
            "device": str(self.device),
            "audio_duration": audio_duration,
            "num_runs": len(processing_times),
            "avg_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "avg_real_time_factor": avg_rtf,
            "processing_times": processing_times
        }