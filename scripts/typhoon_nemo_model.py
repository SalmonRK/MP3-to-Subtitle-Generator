#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typhoon ASR NeMo Model Implementation
Extends the base ASRModel class for SCB10X Typhoon ASR Realtime in NeMo format
"""

import os
import sys
import time
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from scripts.base_asr import ASRModel
from utils.gpu_manager import get_gpu_manager


class TyphoonNemoModel(ASRModel):
    """
    Typhoon ASR NeMo model implementation for Thai speech recognition
    """
    
    def __init__(self, model_name: str = "scb10x/typhoon-asr-realtime", use_gpu: bool = True, **kwargs):
        """
        Initialize Typhoon ASR NeMo model
        
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
        self.asr_model = None
        
        # Check if dependencies are available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        missing_deps = []
        
        try:
            import torch
            self.torch = torch
        except ImportError:
            missing_deps.append("torch")
        
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            missing_deps.append("librosa")
        
        try:
            import soundfile
            self.soundfile = soundfile
        except ImportError:
            missing_deps.append("soundfile")
        
        try:
            import nemo
            import nemo.collections.asr as nemo_asr
            self.nemo = nemo
            self.nemo_asr = nemo_asr
        except ImportError:
            missing_deps.append("nemo_toolkit[asr]")
        
        if missing_deps:
            raise ImportError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )
    
    def _prepare_audio(self, input_path: str, output_path: Optional[str] = None, target_sr: int = 16000) -> tuple:
        """
        Prepare audio file for Typhoon ASR Real-Time processing
        
        Args:
            input_path: Source audio file path
            output_path: Processed output path (auto-generated if None)
            target_sr: Target sample rate for the model
            
        Returns:
            tuple: (success: bool, output_path: str, info: dict)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return False, None, {"error": f"File not found: {input_path}"}
        
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
        if input_path.suffix.lower() not in supported_formats:
            return False, None, {
                "error": f"Unsupported format: {input_path.suffix}",
                "supported": supported_formats
            }
        
        if output_path is None:
            output_path = f"processed_{input_path.stem}.wav"
        
        print(f"Processing audio: {input_path.name}")
        
        # Load and resample audio
        # Ensure mono conversion explicitly
        try:
            # First try to load with librosa
            y, sr = self.librosa.load(str(input_path), sr=None, mono=True)
            if y is None:
                return False, None, {"error": "Failed to load audio file."}
            
            # Ensure we have a 1D array (mono)
            if y.ndim > 1:
                print(f"   Converting from {y.ndim}D to mono (1D)")
                y = self.librosa.to_mono(y)
        except Exception as e:
            print(f"   Error with librosa: {e}")
            # Fallback to soundfile
            try:
                data, sr = self.soundfile.read(str(input_path))
                # Handle different audio formats
                if data.ndim > 1:
                    print(f"   Converting from {data.ndim}D to mono (1D)")
                    # Average channels to create mono
                    y = data.mean(axis=1)
                else:
                    y = data
                
                # Resample if needed
                if sr != target_sr:
                    y = self.librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
            except Exception as e2:
                print(f"   Error with soundfile: {e2}")
                return False, None, {"error": f"Failed to load audio with both methods: {e}, {e2}"}
        
        duration = len(y) / sr
        print(f"   Original: {sr} Hz, {duration:.1f}s")
        
        if sr != target_sr:
            y = self.librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            print(f"   Resampled: {sr} Hz to {target_sr} Hz")
        
        # Normalize audio
        y = y / (max(abs(y)) + 1e-8)
        
        # Save processed audio
        self.soundfile.write(output_path, y, target_sr)
        if not os.path.exists(output_path):
            return False, None, {"error": f"Failed to write processed audio to {output_path}"}
        
        print(f"Processed: {output_path}")
        
        return True, output_path, {
            "original_sr": sr,
            "target_sr": target_sr,
            "duration": duration,
            "output_path": output_path
        }
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load Typhoon ASR NeMo model
        
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
            
            # Load the NeMo model from Hugging Face
            print("Loading Typhoon ASR Real-Time model from Hugging Face...")
            
            # Load the model using NeMo's ASRModel API
            self.asr_model = self.nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
                map_location=self.device
            )
            
            # Set model to evaluation mode
            self.asr_model.eval()
            
            # Optimize for GPU if available
            if self.device == "cuda":
                self.asr_model = self.gpu_manager.optimize_model_for_gpu(
                    self.asr_model, 
                    use_half_precision=True
                )
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            print(f"Typhoon model loaded successfully in {self.load_time:.2f}s")
            print(f"Model device: {next(self.asr_model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading Typhoon model: {e}")
            self.is_loaded = False
            return False
    
    def transcribe(self, audio_path: str, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using Typhoon ASR NeMo model
        
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
        
        # Prepare audio
        temp_path = None
        try:
            success, processed_file, info = self._prepare_audio(
                audio_path,
                target_sr=self.sample_rate
            )
            
            if not success:
                print(f"Audio processing failed: {info.get('error', 'Unknown error')}")
                return {"error": f"Audio processing failed: {info.get('error', 'Unknown error')}"}
            
            temp_path = processed_file
            audio_to_transcribe = processed_file
            
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            audio_to_transcribe = audio_path
        
        try:
            start_time = time.time()
            
            print(f"Transcribing with Typhoon ASR NeMo model (language: {lang})...")
            
            # Transcribe using NeMo model
            with torch.no_grad():
                # NeMo models can transcribe directly from file path
                # Make sure we're using the processed audio file
                if temp_path and os.path.exists(temp_path):
                    transcriptions = self.asr_model.transcribe(audio=[temp_path])
                else:
                    transcriptions = self.asr_model.transcribe(audio=[audio_to_transcribe])
                # Handle different return types from NeMo
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, 'text'):
                        transcription = result.text
                    elif isinstance(result, str):
                        transcription = result
                    else:
                        transcription = str(result)
                else:
                    transcription = ""
            
            # Track performance
            processing_time = self._track_transcription(start_time, audio_duration)
            
            # Create segments (Typhoon doesn't provide word-level timestamps by default)
            # We'll create a single segment for the entire audio and let the text segmenter handle the rest
            segments = [{
                "start": 0.0,
                "end": audio_duration,
                "text": transcription.strip(),
                "words": []  # Empty for now
            }]
            
            # Import text segmenter here to avoid circular imports
            try:
                from utils.text_segmenter import TextSegmenter
                segmenter = TextSegmenter(
                    max_chars_per_segment=42,
                    max_duration=7.0
                )
                
                # Create a transcription result dict for the segmenter
                transcription_result = {
                    "text": transcription.strip(),
                    "segments": segments
                }
                
                # Use the text segmenter to properly segment the transcription
                segmented_result = segmenter.segment_transcription(transcription_result)
                
                # Replace the single segment with properly segmented ones
                if segmented_result:
                    segments = segmented_result
                    print(f"Text segmented into {len(segments)} subtitle segments")
                else:
                    print("Text segmentation failed, using single segment")
                    
            except ImportError as e:
                print(f"Could not import text segmenter: {e}")
                print("Using single segment for entire transcription")
            except Exception as e:
                print(f"Error during text segmentation: {e}")
                print("Using single segment for entire transcription")
            
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
            if temp_path and temp_path.startswith("processed_") and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
    
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
        
        # Prepare audio
        temp_path = None
        try:
            success, processed_file, info = self._prepare_audio(
                audio_path,
                target_sr=self.sample_rate
            )
            
            if not success:
                print(f"Audio processing failed: {info.get('error', 'Unknown error')}")
                return {"error": f"Audio processing failed: {info.get('error', 'Unknown error')}"}
            
            temp_path = processed_file
            audio_to_transcribe = processed_file
            
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            audio_to_transcribe = audio_path
        
        try:
            start_time = time.time()
            
            print(f"Transcribing with Typhoon ASR NeMo model with timestamps (language: {lang})...")
            
            # Transcribe using NeMo model with hypotheses
            with torch.no_grad():
                # Perform transcription with hypotheses
                result = self.asr_model.transcribe(audio=[audio_to_transcribe], return_hypotheses=True)
                
                transcription = ""
                if result and len(result) > 0:
                    hypothesis = result[0]
                    if hasattr(hypothesis, 'text'):
                        transcription = hypothesis.text
                    elif isinstance(hypothesis, list) and len(hypothesis) > 0:
                        transcription = hypothesis[0].text if hasattr(hypothesis[0], 'text') else str(hypothesis[0])
                    else:
                        # Fallback for unexpected structure, defaulting to basic transcription
                        basic_result = self.asr_model.transcribe(audio=[audio_to_transcribe])
                        if basic_result:
                            transcription = basic_result[0]
            
            processing_time = time.time() - start_time
            
            # Generate estimated timestamps
            timestamps = []
            if transcription and audio_duration > 0:
                words = transcription.split()
                if len(words) > 0:
                    avg_duration = audio_duration / len(words)
                    
                    for i, word in enumerate(words):
                        start_t = i * avg_duration
                        end_t = start_t + avg_duration
                        timestamps.append({
                            "word": word,
                            "start": start_t,
                            "end": end_t
                        })
            
            # Create segments with timestamps
            segments = []
            for i, ts in enumerate(timestamps):
                segments.append({
                    "start": ts["start"],
                    "end": ts["end"],
                    "text": ts["word"],
                    "words": []
                })
            
            # Import text segmenter here to avoid circular imports
            try:
                from utils.text_segmenter import TextSegmenter
                segmenter = TextSegmenter(
                    max_chars_per_segment=42,
                    max_duration=7.0
                )
                
                # Create a transcription result dict for the segmenter
                transcription_result = {
                    "text": transcription.strip(),
                    "segments": segments
                }
                
                # Use the text segmenter to properly segment the transcription
                segmented_result = segmenter.segment_transcription(transcription_result)
                
                # Replace the word segments with properly segmented ones
                if segmented_result:
                    segments = segmented_result
                    print(f"Text segmented into {len(segments)} subtitle segments")
                else:
                    print("Text segmentation failed, using word-level segments")
                    
            except ImportError as e:
                print(f"Could not import text segmenter: {e}")
                print("Using word-level segments")
            except Exception as e:
                print(f"Error during text segmentation: {e}")
                print("Using word-level segments")
            
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
            if temp_path and temp_path.startswith("processed_") and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
    
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
            "model_type": "Typhoon ASR Realtime (NeMo)",
            "primary_language": "Thai",
            "sample_rate": self.sample_rate
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
        
        print(f"Benchmarking Typhoon NeMo model with {num_runs} runs...")
        
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