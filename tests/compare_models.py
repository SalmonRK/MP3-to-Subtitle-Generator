#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple comparison script for Whisper and Typhoon ASR models
"""

import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.whisper_model import WhisperModel
    from scripts.typhoon_nemo_model import TyphoonNemoModel
    from utils.gpu_manager import get_gpu_manager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_whisper_model(audio_file, use_gpu=True):
    """Test Whisper model"""
    print("\n" + "="*50)
    print("TESTING WHISPER MODEL")
    print("="*50)
    
    # Create model
    model = WhisperModel(
        model_name="large",
        use_gpu=use_gpu
    )
    
    # Load model
    print("Loading Whisper model...")
    start_time = time.time()
    if model.load_model():
        load_time = time.time() - start_time
        print(f"Whisper model loaded in {load_time:.2f}s")
        
        # Transcribe
        print(f"Transcribing {audio_file}...")
        start_time = time.time()
        result = model.transcribe(audio_file, language="th")
        processing_time = time.time() - start_time
        
        if "error" not in result:
            print("Whisper transcription successful!")
            try:
                print(f"Text: {result.get('text', 'No text')[:100]}...")
            except UnicodeEncodeError:
                print("Text: [Thai text - encoding issue]")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Audio duration: {result.get('audio_duration', 0):.2f}s")
            
            return {
                "model": "Whisper Large",
                "load_time": load_time,
                "processing_time": processing_time,
                "audio_duration": result.get('audio_duration', 0),
                "text": result.get('text', ''),
                "success": True
            }
        else:
            print(f"Whisper transcription failed: {result['error']}")
            return {
                "model": "Whisper Large",
                "error": result['error'],
                "success": False
            }
    else:
        print("Failed to load Whisper model")
        return {
            "model": "Whisper Large",
            "error": "Failed to load model",
            "success": False
        }

def test_typhoon_model(audio_file, use_gpu=True):
    """Test Typhoon NeMo model"""
    print("\n" + "="*50)
    print("TESTING TYPHOON NEMO MODEL")
    print("="*50)
    
    # Create model
    model = TyphoonNemoModel(
        model_name="scb10x/typhoon-asr-realtime",
        use_gpu=use_gpu
    )
    
    # Load model
    print("Loading Typhoon model...")
    start_time = time.time()
    if model.load_model():
        load_time = time.time() - start_time
        print(f"Typhoon model loaded in {load_time:.2f}s")
        
        # Transcribe
        print(f"Transcribing {audio_file}...")
        start_time = time.time()
        result = model.transcribe(audio_file, language="th")
        processing_time = time.time() - start_time
        
        if "error" not in result:
            print("Typhoon transcription successful!")
            try:
                print(f"Text: {result.get('text', 'No text')[:100]}...")
            except UnicodeEncodeError:
                print("Text: [Thai text - encoding issue]")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Audio duration: {result.get('audio_duration', 0):.2f}s")
            
            return {
                "model": "Typhoon ASR Realtime (NeMo)",
                "load_time": load_time,
                "processing_time": processing_time,
                "audio_duration": result.get('audio_duration', 0),
                "text": result.get('text', ''),
                "success": True
            }
        else:
            print(f"Typhoon transcription failed: {result['error']}")
            return {
                "model": "Typhoon ASR Realtime (NeMo)",
                "error": result['error'],
                "success": False
            }
    else:
        print("Failed to load Typhoon model")
        return {
            "model": "Typhoon ASR Realtime (NeMo)",
            "error": "Failed to load model",
            "success": False
        }

def compare_models(whisper_result, typhoon_result):
    """Compare model results"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if whisper_result["success"] and typhoon_result["success"]:
        # Load time comparison
        print(f"Model Load Time:")
        print(f"  Whisper: {whisper_result['load_time']:.2f}s")
        print(f"  Typhoon: {typhoon_result['load_time']:.2f}s")
        load_winner = "Whisper" if whisper_result['load_time'] < typhoon_result['load_time'] else "Typhoon"
        print(f"  Winner: {load_winner}")
        
        # Processing time comparison
        print(f"\nProcessing Time:")
        print(f"  Whisper: {whisper_result['processing_time']:.2f}s")
        print(f"  Typhoon: {typhoon_result['processing_time']:.2f}s")
        proc_winner = "Whisper" if whisper_result['processing_time'] < typhoon_result['processing_time'] else "Typhoon"
        print(f"  Winner: {proc_winner}")
        
        # Real-time factor comparison
        whisper_rtf = whisper_result['processing_time'] / whisper_result['audio_duration']
        typhoon_rtf = typhoon_result['processing_time'] / typhoon_result['audio_duration']
        
        print(f"\nReal-time Factor (lower is better):")
        print(f"  Whisper: {whisper_rtf:.2f}x")
        print(f"  Typhoon: {typhoon_rtf:.2f}x")
        rtf_winner = "Whisper" if whisper_rtf < typhoon_rtf else "Typhoon"
        print(f"  Winner: {rtf_winner}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if typhoon_rtf < whisper_rtf:
            print("  For speed: Typhoon ASR is faster")
        else:
            print("  For speed: Whisper is faster")
        
        print("  For Thai language: Typhoon ASR is specifically designed for Thai")
        print("  For general use: Whisper supports more languages")
        
        # Save results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file": "Jasmali.MP3",
            "whisper": whisper_result,
            "typhoon": typhoon_result,
            "comparison": {
                "load_time_winner": load_winner,
                "processing_time_winner": proc_winner,
                "rtf_winner": rtf_winner,
                "whisper_rtf": whisper_rtf,
                "typhoon_rtf": typhoon_rtf
            }
        }
        
        with open("model_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to model_comparison_results.json")
    else:
        print("Cannot compare - one or both models failed")
        if not whisper_result["success"]:
            print(f"Whisper error: {whisper_result.get('error', 'Unknown')}")
        if not typhoon_result["success"]:
            print(f"Typhoon error: {typhoon_result.get('error', 'Unknown')}")

def main():
    """Main function"""
    print("MODEL COMPARISON: WHISPER vs TYPHOON ASR")
    print("="*60)
    
    # Get GPU manager
    gpu_manager = get_gpu_manager()
    print(f"GPU Available: {gpu_manager.gpu_available}")
    
    # Test audio file
    audio_file = "Jasmali.MP3"
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return
    
    # Test models
    whisper_result = test_whisper_model(audio_file, use_gpu=gpu_manager.gpu_available)
    typhoon_result = test_typhoon_model(audio_file, use_gpu=gpu_manager.gpu_available)
    
    # Compare results
    compare_models(whisper_result, typhoon_result)

if __name__ == "__main__":
    main()