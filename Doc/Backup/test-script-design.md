# Test Script Design for Model Comparison

## Overview
This document outlines the design for a comprehensive test script to compare transcription accuracy between Whisper and Typhoon ASR models.

## Test Script Structure

### 1. Model Comparison Test Script

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Test Script
Compares transcription accuracy and performance between Whisper and Typhoon ASR models
"""

import os
import sys
import time
import json
import torch
import psutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class ModelComparison:
    def __init__(self):
        self.results = {
            "whisper": {},
            "typhoon": {}
        }
        self.test_files = [
            "Jasmali.MP3",
            "ขนมครก.MP3"
        ]
        
    def detect_gpu(self):
        """Detect GPU availability and information"""
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory
            }
        return {"available": False}
    
    def load_whisper_model(self, model_size="large", use_gpu=True):
        """Load Whisper model with GPU support"""
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} model on {device}...")
        
        start_time = time.time()
        model = whisper.load_model(model_size, device=device)
        load_time = time.time() - start_time
        
        self.results["whisper"]["load_time"] = load_time
        self.results["whisper"]["device"] = device
        
        return model
    
    def load_typhoon_model(self, use_gpu=True):
        """Load Typhoon ASR model with GPU support"""
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Loading Typhoon ASR model on {device}...")
        
        start_time = time.time()
        # Implementation to be added after research
        model = None  # Placeholder
        load_time = time.time() - start_time
        
        self.results["typhoon"]["load_time"] = load_time
        self.results["typhoon"]["device"] = device
        
        return model
    
    def transcribe_with_whisper(self, model, audio_path):
        """Transcribe audio using Whisper"""
        print(f"Transcribing {audio_path} with Whisper...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        result = model.transcribe(audio_path, language="th", word_timestamps=True)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "processing_time": end_time - start_time,
            "memory_used": end_memory - start_memory
        }
    
    def transcribe_with_typhoon(self, model, audio_path):
        """Transcribe audio using Typhoon ASR"""
        print(f"Transcribing {audio_path} with Typhoon...")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # Implementation to be added after research
        result = {"text": "", "segments": []}  # Placeholder
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "processing_time": end_time - start_time,
            "memory_used": end_memory - start_memory
        }
    
    def calculate_accuracy(self, reference_text, hypothesis_text):
        """Calculate word error rate (WER)"""
        # Implementation for WER calculation
        # This would need reference transcripts for accuracy calculation
        pass
    
    def run_comparison(self, use_gpu=True):
        """Run full comparison test"""
        print("Starting model comparison test...")
        print(f"GPU available: {self.detect_gpu()['available']}")
        
        # Load models
        whisper_model = self.load_whisper_model(use_gpu=use_gpu)
        typhoon_model = self.load_typhoon_model(use_gpu=use_gpu)
        
        # Test each audio file
        for audio_file in self.test_files:
            if not os.path.exists(audio_file):
                print(f"Warning: {audio_file} not found, skipping...")
                continue
            
            print(f"\nTesting with {audio_file}...")
            
            # Test Whisper
            whisper_result = self.transcribe_with_whisper(whisper_model, audio_file)
            self.results["whisper"][audio_file] = whisper_result
            
            # Test Typhoon
            typhoon_result = self.transcribe_with_typhoon(typhoon_model, audio_file)
            self.results["typhoon"][audio_file] = typhoon_result
    
    def generate_report(self):
        """Generate comparison report"""
        report = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_info": self.detect_gpu(),
            "results": self.results
        }
        
        # Save to JSON
        with open("comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Generate summary
        self.print_summary()
        
        return report
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        
        # Model loading times
        print(f"Whisper load time: {self.results['whisper'].get('load_time', 'N/A'):.2f}s")
        print(f"Typhoon load time: {self.results['typhoon'].get('load_time', 'N/A'):.2f}s")
        
        # Processing times for each file
        for audio_file in self.test_files:
            if audio_file in self.results["whisper"] and audio_file in self.results["typhoon"]:
                whisper_time = self.results["whisper"][audio_file]["processing_time"]
                typhoon_time = self.results["typhoon"][audio_file]["processing_time"]
                
                print(f"\n{audio_file}:")
                print(f"  Whisper: {whisper_time:.2f}s")
                print(f"  Typhoon: {typhoon_time:.2f}s")
                print(f"  Speed improvement: {whisper_time/typhoon_time:.2f}x")
    
    def save_transcriptions(self):
        """Save transcriptions to files for manual review"""
        for model_name in ["whisper", "typhoon"]:
            for audio_file in self.test_files:
                if audio_file in self.results[model_name]:
                    result = self.results[model_name][audio_file]
                    output_file = f"output/{audio_file}.{model_name}.srt"
                    
                    # Generate SRT format
                    self.generate_srt(result["segments"], output_file)

def main():
    """Main function to run comparison"""
    parser = argparse.ArgumentParser(description='Compare ASR models')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--files', nargs='+', help='Specific audio files to test')
    
    args = parser.parse_args()
    
    comparison = ModelComparison()
    
    if args.files:
        comparison.test_files = args.files
    
    comparison.run_comparison(use_gpu=args.gpu)
    comparison.generate_report()
    comparison.save_transcriptions()

if __name__ == "__main__":
    main()
```

## Test Methodology

### 1. Performance Metrics
- **Model Loading Time**: Time taken to initialize each model
- **Processing Speed**: Real-time factor (RTF) - processing time / audio duration
- **Memory Usage**: Peak memory consumption during transcription
- **GPU Utilization**: GPU memory and compute usage (if applicable)

### 2. Accuracy Metrics
- **Word Error Rate (WER)**: Standard metric for ASR accuracy
- **Character Error Rate (CER)**: More relevant for Thai language
- **Semantic Accuracy**: Manual evaluation of meaning preservation
- **Proper Noun Recognition**: Accuracy for names, places, etc.

### 3. Test Files
- **Jasmali.MP3**: Promotional content with clear speech
- **ขนมครก.MP3**: Conversational content with background noise
- Additional test files can be added as needed

### 4. Test Scenarios
1. **CPU vs GPU Performance**: Same model on different devices
2. **Model Size Comparison**: Different Whisper model sizes
3. **Audio Quality Impact**: Clean vs noisy audio
4. **Language Specificity**: Thai-only vs multilingual content

## Expected Outputs

### 1. JSON Results File
```json
{
  "test_date": "2024-01-01 12:00:00",
  "gpu_info": {
    "available": true,
    "name": "NVIDIA RTX 3080",
    "memory": 10737418240
  },
  "results": {
    "whisper": {
      "load_time": 5.2,
      "device": "cuda",
      "Jasmali.MP3": {
        "processing_time": 12.5,
        "memory_used": 2048576,
        "text": "...",
        "segments": [...]
      }
    },
    "typhoon": {
      "load_time": 3.8,
      "device": "cuda",
      "Jasmali.MP3": {
        "processing_time": 8.2,
        "memory_used": 1536000,
        "text": "...",
        "segments": [...]
      }
    }
  }
}
```

### 2. SRT Files
- `output/Jasmali.MP3.whisper.srt`
- `output/Jasmali.MP3.typhoon.srt`
- `output/ขนมครก.MP3.whisper.srt`
- `output/ขนมครก.MP3.typhoon.srt`

### 3. Summary Report
- Performance comparison table
- Accuracy metrics
- Recommendations based on results

## Implementation Notes

### Dependencies
- `torch` - For model loading and GPU support
- `whisper` - OpenAI Whisper model
- `transformers` - For Typhoon ASR model
- `psutil` - For system resource monitoring
- `pandas` - For data analysis
- `matplotlib` - For visualization
- `jiwer` - For WER calculation

### Usage
```bash
# Run comparison with GPU
python compare_models.py --gpu

# Test specific files
python compare_models.py --files Jasmali.MP3 ขนมครก.MP3

# Run with CPU only
python compare_models.py
```

## Next Steps
1. Implement the test script after Typhoon ASR research
2. Create reference transcripts for accuracy calculation
3. Add visualization capabilities for results
4. Implement automated reporting
5. Integrate with CI/CD for continuous testing