#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio to SRT subtitle generator
Converts audio files to SRT subtitle format with Thai language support
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
    # Try to import whisper for better Thai recognition
    try:
        import whisper
        import torch
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        print("Whisper not available. Install with: pip install openai-whisper torch")
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with:")
    print("pip install SpeechRecognition pydub googletrans==4.0.0-rc1 openai-whisper torch")
    sys.exit(1)

class AudioTranscriber:
    def __init__(self, source_language='en', target_language='th'):
        """
        Initialize the audio transcriber
        
        Args:
            source_language (str): Source language code (default: 'en')
            target_language (str): Target language code for translation (default: 'th')
        """
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.source_language = source_language
        self.target_language = target_language
        
        # Initialize Whisper model if available
        if WHISPER_AVAILABLE:
            print("Loading Whisper model...")
            self.whisper_model = self.load_local_whisper_model("large")
            print("Whisper model loaded successfully")
        else:
            self.whisper_model = None
    
    def load_local_whisper_model(self, model_name="large"):
        """
        Load a Whisper model from the local models directory
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Whisper model or None if loading fails
        """
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            models_dir = script_dir / "models"
            model_path = models_dir / f"{model_name}.pt"
            
            # Check if the local model exists
            if model_path.exists():
                print(f"Loading {model_name} model from local file: {model_path}")
                model = whisper.load_model("large")  # Load the base model structure
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                return model
            else:
                print(f"Local model {model_name} not found at {model_path}")
                print("Falling back to downloading from internet...")
                return whisper.load_model(model_name)
                
        except Exception as e:
            print(f"Failed to load local model: {e}")
            print("Falling back to downloading from internet...")
            return whisper.load_model(model_name)
        
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
    
    def transcribe_audio(self, audio_path, chunk_duration_ms=30000):
        """
        Transcribe audio file in chunks
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration_ms (int): Duration of each chunk in milliseconds
            
        Returns:
            list: List of transcribed segments with timestamps
        """
        # Try to use Whisper directly if available and file is not WAV
        if self.whisper_model and not audio_path.endswith('.wav'):
            try:
                audio_path_str = str(audio_path)
                print(f"Starting transcription with Whisper for {audio_path_str}...")
                return self.transcribe_with_whisper(audio_path, chunk_duration_ms)
            except Exception as e:
                print(f"Error during Whisper transcription: {e}")
                print("Falling back to WAV conversion...")
        
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            try:
                audio_path = self.audio_to_wav(audio_path)
            except Exception as e:
                print(f"Failed to convert to WAV: {e}")
                if self.whisper_model:
                    print("Trying with Whisper directly...")
                    return self.transcribe_with_whisper(audio_path, chunk_duration_ms)
                else:
                    return []
        
        print(f"Starting transcription of {audio_path}...")
        
        # Load audio
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        segments = []
        
        # Process in chunks
        for start_ms in range(0, duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            # Save chunk to temporary file
            chunk_path = f"temp_chunk_{start_ms}.wav"
            chunk.export(chunk_path, format="wav")
            
            try:
                with sr.AudioFile(chunk_path) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.record(source)
                    
                    # Try to recognize speech with multiple language options
                    print(f"Transcribing chunk {start_ms//1000}-{end_ms//1000}s...")
                    text = None
                    
                    # Try with Thai first since the user wants Thai subtitles
                    languages_to_try = ['th-TH', 'en-US', 'ja-JP', 'ko-KR', 'zh-CN']
                    
                    # If source language is specified, try it first
                    if self.source_language != 'th':
                        languages_to_try.insert(0, self.source_language)
                    
                    text = None
                    for lang in languages_to_try:
                        try:
                            print(f"Trying to recognize with language: {lang}")
                            text = self.recognizer.recognize_google(audio_data, language=lang)
                            print(f"Successfully recognized text with language: {lang}")
                            print(f"Recognized text: {text}")
                            break
                        except sr.UnknownValueError:
                            print(f"Could not understand audio with language: {lang}")
                            continue
                        except sr.RequestError as e:
                            print(f"Error with {lang}: {e}")
                            continue
                    
                    # If all specific languages fail, try auto-detect
                    if not text:
                        try:
                            print("Trying auto-detection...")
                            text = self.recognizer.recognize_google(audio_data)
                            print(f"Auto-detected text: {text}")
                        except sr.UnknownValueError:
                            print("Auto-detection also failed")
                    
                    # If Google Speech Recognition fails completely, try Whisper
                    if not text and self.whisper_model:
                        try:
                            print("Trying Whisper transcription...")
                            # Save chunk to temporary file for Whisper
                            temp_whisper_file = f"temp_whisper_{start_ms}.wav"
                            chunk.export(temp_whisper_file, format="wav")
                            
                            # Transcribe with Whisper
                            result = self.whisper_model.transcribe(temp_whisper_file, language="th")
                            text = result["text"]
                            print(f"Whisper transcribed text: {text}")
                            
                            # Clean up temporary file
                            if os.path.exists(temp_whisper_file):
                                os.remove(temp_whisper_file)
                        except Exception as e:
                            print(f"Whisper transcription failed: {e}")
                    
                    if text:
                        # Translate to Thai if needed
                        if self.source_language != self.target_language:
                            print(f"Translating to Thai...")
                            translated = self.translator.translate(text, src=self.source_language, dest=self.target_language)
                            thai_text = translated.text
                        else:
                            thai_text = text
                    else:
                        print(f"No recognizable speech found in chunk {start_ms//1000}-{end_ms//1000}s")
                        continue
                    
                    # Calculate timestamps
                    start_time = self.ms_to_srt_time(start_ms)
                    end_time = self.ms_to_srt_time(end_ms)
                    
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': thai_text
                    })
                    
            except sr.UnknownValueError:
                print(f"Could not understand audio in chunk {start_ms//1000}-{end_ms//1000}s")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        # Clean up converted WAV file if it was converted
        if audio_path.endswith('.wav') and not os.path.basename(audio_path).startswith('temp_'):
            original_path = audio_path.replace('.wav', Path(audio_path).suffix)
            if original_path != audio_path and os.path.exists(original_path):
                os.remove(audio_path)
        
        return segments
    
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
    parser.add_argument('-s', '--source', help='Source language code (default: en)', default='en')
    parser.add_argument('-t', '--target', help='Target language code (default: th)', default='th')
    parser.add_argument('-d', '--duration', help='Chunk duration in milliseconds (default: 30000)', type=int, default=30000)
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
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
    
    # Create transcriber and process
    transcriber = AudioTranscriber(source_language=args.source, target_language=args.target)
    segments = transcriber.transcribe_audio(args.input_file, chunk_duration_ms=args.duration)
    
    if segments:
        transcriber.generate_srt(segments, args.output)
        print(f"Successfully created {len(segments)} subtitle segments")
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

if __name__ == "__main__":
    main()