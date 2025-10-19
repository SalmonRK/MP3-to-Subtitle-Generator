#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify Typhoon NeMo model is working
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.typhoon_nemo_model import TyphoonNemoModel
    from utils.gpu_manager import get_gpu_manager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_typhoon_nemo():
    """Test Typhoon NeMo model with GPU"""
    print("Testing Typhoon NeMo Model...")
    
    # Get GPU manager
    gpu_manager = get_gpu_manager()
    print(f"GPU Available: {gpu_manager.gpu_available}")
    
    # Create model
    model = TyphoonNemoModel(
        model_name="scb10x/typhoon-asr-realtime",
        use_gpu=gpu_manager.gpu_available
    )
    
    # Load model
    print("Loading model...")
    if model.load_model():
        print("Model loaded successfully!")
        
        # Test transcription
        audio_file = "Jasmali.MP3"
        if os.path.exists(audio_file):
            print(f"Transcribing {audio_file}...")
            result = model.transcribe(audio_file)
            
            if "error" not in result:
                print(f"Transcription successful!")
                try:
                    print(f"Text: {result.get('text', 'No text')}")
                except UnicodeEncodeError:
                    print("Text: [Thai text - encoding issue]")
                print(f"Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"Audio duration: {result.get('audio_duration', 0):.2f}s")
            else:
                print(f"Transcription failed: {result['error']}")
        else:
            print(f"Audio file {audio_file} not found")
    else:
        print("Failed to load model")

if __name__ == "__main__":
    test_typhoon_nemo()