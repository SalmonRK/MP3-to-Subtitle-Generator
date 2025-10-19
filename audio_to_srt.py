#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio to SRT subtitle generator
Converts audio files to SRT subtitle format with Thai language support
Enhanced with GPU support and multiple ASR model options
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from googletrans import Translator
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with:")
    print("pip install SpeechRecognition pydub googletrans==4.0.0-rc1")
    sys.exit(1)

# Import our new modules
try:
    from config import get_config
    from utils.gpu_manager import print_system_info, get_gpu_manager
    from utils.text_segmenter import TextSegmenter
    from scripts import get_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are present")
    sys.exit(1)

class AudioTranscriber:
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize the audio transcriber with enhanced GPU support and model selection
        
        Args:
            config_file: Path to configuration file
            **kwargs: Additional configuration overrides
        """
        # Load configuration
        self.config = get_config(config_file)
        self.config.update(kwargs)
        
        # Initialize components
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.source_language = self.config.get("source_language", "auto")
        self.target_language = self.config.get("target_language", "th")
        
        # GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Initialize text segmenter
        self.text_segmenter = TextSegmenter(
            max_chars_per_segment=self.config.get("max_chars_per_segment", 42),
            max_duration=self.config.get("max_segment_duration", 7.0)
        )
        
        # Print system information
        if self.config.get("verbose", True):
            print_system_info()
        
        # Initialize ASR model
        self.asr_model = None
        self._load_asr_model()
    
    def _load_asr_model(self):
        """Load the ASR model based on configuration"""
        try:
            model_type = self.config.get("asr_model", "whisper")
            use_gpu = self.config.get("use_gpu", True)
            
            print(f"Loading {model_type} model...")
            
            # Get model configuration
            if model_type == "whisper":
                model_name = self.config.get("whisper_model_size", "large")
                self.asr_model = get_model(
                    model_type,
                    model_name=model_name,
                    use_gpu=use_gpu,
                    language=self.config.get("language", "th"),
                    task=self.config.get("task", "transcribe"),
                    word_timestamps=self.config.get("word_timestamps", True)
                )
            else:
                # For future models like Typhoon
                self.asr_model = get_model(model_type, use_gpu=use_gpu)
            
            # Load the model
            if self.asr_model.load_model():
                print(f"Model loaded successfully: {self.asr_model}")
                print(f"Device: {self.asr_model.device}")
                
                # Print performance stats if available
                stats = self.asr_model.get_performance_stats()
                if stats.get("transcription_count", 0) > 0:
                    print(f"Previous performance: {stats.get('real_time_factor', 0):.2f}x real-time")
            else:
                print("Failed to load model")
                self.asr_model = None
                
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            self.asr_model = None
    
    def switch_model(self, model_type: str, **kwargs):
        """
        Switch to a different ASR model
        
        Args:
            model_type: Type of model ("whisper" or "typhoon")
            **kwargs: Additional model parameters
        """
        print(f"Switching to {model_type} model...")
        
        # Update configuration
        self.config.set("asr_model", model_type)
        for key, value in kwargs.items():
            self.config.set(key, value)
        
        # Load new model
        self._load_asr_model()
        
    def audio_to_wav(self, audio_path):
        """
        Convert audio file to WAV format for processing
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Path to converted WAV file
        """
        try:
            audio_path_str = str(audio_path)
            print(f"Converting {audio_path_str} to WAV format...")
        except:
            print("Converting audio file to WAV format...")
        
        # Get file extension
        file_ext = Path(audio_path).suffix.lower()
        
        # Convert to WAV
        audio = AudioSegment.from_file(audio_path, format=file_ext[1:])
        wav_path = audio_path.replace(file_ext, '.wav')
        audio.export(wav_path, format="wav")
        
        print(f"Converted to {wav_path}")
        return wav_path
    
    def transcribe_audio(self, audio_path, chunk_duration_ms=None):
        """
        Transcribe audio file using the configured ASR model
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration_ms (int): Duration of each chunk in milliseconds (from config)
            
        Returns:
            list: List of transcribed segments with timestamps
        """
        if self.asr_model is None:
            print("No ASR model loaded. Falling back to legacy methods...")
            return self._transcribe_with_legacy_methods(audio_path, chunk_duration_ms)
        
        try:
            # Use chunk duration from config if not provided
            if chunk_duration_ms is None:
                chunk_duration_ms = self.config.get("chunk_duration_ms", 30000)
            
            print(f"Starting transcription with {self.config.get('asr_model')} model...")
            print(f"Audio file: {audio_path}")
            print(f"Chunk duration: {chunk_duration_ms}ms")
            
            # Get audio duration for performance tracking
            audio_duration = self.asr_model.get_audio_duration(audio_path)
            print(f"Audio duration: {audio_duration:.2f}s")
            
            # Transcribe with the selected model
            result = self.asr_model.transcribe(
                audio_path,
                language=self.config.get("language", "th"),
                task=self.config.get("task", "transcribe"),
                word_timestamps=self.config.get("word_timestamps", True)
            )
            
            if "error" in result:
                print(f"Transcription error: {result['error']}")
                print("Falling back to legacy methods...")
                return self._transcribe_with_legacy_methods(audio_path, chunk_duration_ms)
            
            # Use improved text segmentation
            improved_segments = self.text_segmenter.segment_transcription(result)
            
            # Convert segments to SRT format
            segments = []
            for segment in improved_segments:
                # TextSegmenter returns times in seconds, convert to milliseconds for SRT format
                start_time = self.ms_to_srt_time(segment["start"] * 1000)
                end_time = self.ms_to_srt_time(segment["end"] * 1000)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': segment["text"]
                })
            
            # Print performance stats
            processing_time = result.get("processing_time", 0)
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            print(f"Transcription completed in {processing_time:.2f}s")
            print(f"Real-time factor: {real_time_factor:.2f}x")
            print(f"Generated {len(segments)} segments")
            
            # Print model performance stats
            stats = self.asr_model.get_performance_stats()
            print(f"Model stats: {stats}")
            
            return segments
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            print("Falling back to legacy methods...")
            return self._transcribe_with_legacy_methods(audio_path, chunk_duration_ms)
    
    def _transcribe_with_legacy_methods(self, audio_path, chunk_duration_ms=30000):
        """
        Fallback transcription method using original approach
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration_ms (int): Duration of each chunk in milliseconds
            
        Returns:
            list: List of transcribed segments with timestamps
        """
        print("Using legacy transcription method...")
        
        # Try to use Whisper directly if available
        try:
            import whisper
            import torch
            WHISPER_AVAILABLE = True
        except ImportError:
            WHISPER_AVAILABLE = False
        
        if WHISPER_AVAILABLE and not audio_path.endswith('.wav'):
            try:
                return self.transcribe_with_whisper(audio_path, chunk_duration_ms)
            except Exception as e:
                print(f"Error during Whisper transcription: {e}")
        
        # Fall back to the original chunk processing method
        return self._transcribe_with_chunks(audio_path, chunk_duration_ms)
    
    def transcribe_with_whisper(self, audio_path, chunk_duration_ms=30000):
        """
        Transcribe audio file using Whisper directly
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration_ms (int): Duration of each chunk in milliseconds
            
        Returns:
            list: List of transcribed segments with timestamps
        """
        try:
            audio_path_str = str(audio_path)
            print(f"Transcribing {audio_path_str} with Whisper...")
        except:
            print("Transcribing audio file with Whisper...")
        
        try:
            # Set console encoding to UTF-8 for Windows
            if sys.platform == "win32":
                import locale
                locale.setlocale(locale.LC_ALL, 'Thai_Thailand.UTF-8')
            
            # Transcribe the entire audio file with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="th",
                task="transcribe",
                word_timestamps=True
            )
            
            segments = []
            
            # Process Whisper segments
            for segment in result.get("segments", []):
                start_time = segment.get("start", 0) * 1000  # Convert to milliseconds
                end_time = segment.get("end", 0) * 1000  # Convert to milliseconds
                
                # Format timestamps for SRT
                start_srt = self.ms_to_srt_time(start_time)
                end_srt = self.ms_to_srt_time(end_time)
                
                # Get the transcribed text
                text = segment.get("text", "").strip()
                
                if text:
                    segments.append({
                        'start_time': start_srt,
                        'end_time': end_srt,
                        'text': text
                    })
                    
                    try:
                        print(f"Transcribed: {text}")
                    except UnicodeEncodeError:
                        print(f"Transcribed: [Thai text - encoding issue]")
            
            return segments
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return []
    
    def ms_to_srt_time(self, milliseconds):
        """
        Convert milliseconds to SRT time format
        
        Args:
            milliseconds (int): Time in milliseconds
            
        Returns:
            str: Time in SRT format (HH:MM:SS,mmm)
        """
        seconds = milliseconds / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    
    def generate_srt(self, segments, output_path):
        """
        Generate SRT file from transcribed segments
        
        Args:
            segments (list): List of transcribed segments
            output_path (str): Path to output SRT file
        """
        print(f"Generating SRT file: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{segment['start_time']} --> {segment['end_time']}\n")
                f.write(f"{segment['text']}\n\n")
        
        print(f"SRT file created successfully: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert audio file to SRT subtitles')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output SRT file path', default=None)
    parser.add_argument('-c', '--config', help='Configuration file path', default=None)
    parser.add_argument('-m', '--model', help='ASR model to use (whisper or typhoon)', default=None)
    parser.add_argument('-s', '--source', help='Source language code', default=None)
    parser.add_argument('-t', '--target', help='Target language code', default=None)
    parser.add_argument('-d', '--duration', help='Chunk duration in milliseconds', type=int, default=None)
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage if available')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--model-size', help='Whisper model size (tiny, base, small, medium, large)', default=None)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Prepare configuration overrides
    config_overrides = {}
    
    if args.model:
        config_overrides["asr_model"] = args.model
    if args.source:
        config_overrides["source_language"] = args.source
    if args.target:
        config_overrides["target_language"] = args.target
    if args.duration:
        config_overrides["chunk_duration_ms"] = args.duration
    if args.gpu:
        config_overrides["use_gpu"] = True
    if args.no_gpu:
        config_overrides["use_gpu"] = False
    if args.model_size:
        config_overrides["whisper_model_size"] = args.model_size
    if args.verbose:
        config_overrides["verbose"] = True
    
    # Set output path if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.with_suffix('.srt'))
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Update output path to be in the output directory
    output_path = Path(args.output)
    args.output = str(output_dir / output_path.name)
    
    try:
        # Create transcriber with configuration
        transcriber = AudioTranscriber(config_file=args.config, **config_overrides)
        
        # Print configuration if verbose
        if args.verbose or transcriber.config.get("verbose", False):
            print("\n" + "="*50)
            print("TRANSCRIPTION CONFIGURATION")
            print("="*50)
            print(f"ASR Model: {transcriber.config.get('asr_model')}")
            if transcriber.config.get('asr_model') == 'whisper':
                print(f"Whisper Model Size: {transcriber.config.get('whisper_model_size')}")
            print(f"Use GPU: {transcriber.config.get('use_gpu')}")
            print(f"Language: {transcriber.config.get('language')}")
            print(f"Chunk Duration: {transcriber.config.get('chunk_duration_ms')}ms")
            print("="*50)
        
        # Transcribe audio
        segments = transcriber.transcribe_audio(args.input_file)
        
        if segments:
            transcriber.generate_srt(segments, args.output)
            print(f"\nSuccessfully created {len(segments)} subtitle segments")
            print(f"Output saved to: {args.output}")
            
            # Print final performance stats
            if transcriber.asr_model:
                stats = transcriber.asr_model.get_performance_stats()
                print(f"\nPerformance Statistics:")
                print(f"  Transcriptions: {stats.get('transcription_count', 0)}")
                print(f"  Total Time: {stats.get('total_transcription_time', 0):.2f}s")
                print(f"  Audio Duration: {stats.get('total_audio_duration', 0):.2f}s")
                print(f"  Real-time Factor: {stats.get('real_time_factor', 0):.2f}x")
        else:
            print("No speech detected in the audio file.")
            print("Creating a placeholder SRT file with timestamps...")
            
            # Create placeholder segments with timestamps
            audio = AudioSegment.from_file(args.input_file)
            duration_ms = len(audio)
            placeholder_segments = []
            
            # Create segments every 5 seconds
            for start_ms in range(0, duration_ms, 5000):
                end_ms = min(start_ms + 5000, duration_ms)
                start_time = transcriber.ms_to_srt_time(start_ms)
                end_time = transcriber.ms_to_srt_time(end_ms)
                
                placeholder_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': "[ไม่มีข้อความพูด]"  # "No speech detected" in Thai
                })
            
            transcriber.generate_srt(placeholder_segments, args.output)
            print(f"Created placeholder SRT file with {len(placeholder_segments)} segments")
            print(f"Output saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()