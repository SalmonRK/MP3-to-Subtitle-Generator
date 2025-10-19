#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Typhoon ASR with English filename only
Tests Jasmali.MP3 with Typhoon model on GPU
"""

import os
import sys
import time
import json
from pathlib import Path

# Set console encoding to UTF-8 for Windows
if sys.platform == "win32":
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'Thai_Thailand.UTF-8')
    except:
        try:
            os.system('chcp 65001 >nul')
        except:
            pass

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Change to parent directory to access audio files
os.chdir(Path(__file__).parent)

try:
    from utils.gpu_manager import get_gpu_manager, print_system_info
    from scripts import get_model
    from utils.text_segmenter import TextSegmenter
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_transcription(model, audio_file, output_name):
    """Test transcription with the loaded model"""
    print(f"\n{'='*60}")
    print(f"TRANSCRIPTION TEST: {audio_file}")
    print(f"{'='*60}")
    
    if not model:
        print("[FAILED] No model available for transcription")
        return None
    
    if not Path(audio_file).exists():
        print(f"[ERROR] Audio file not found: {audio_file}")
        return None
    
    try:
        print(f"Starting transcription of {audio_file}...")
        
        # Get audio duration
        audio_duration = model.get_audio_duration(audio_file)
        print(f"Audio duration: {audio_duration:.2f}s")
        
        # Transcribe
        start_time = time.time()
        result = model.transcribe(audio_file, language="th")
        processing_time = time.time() - start_time
        
        if "error" not in result:
            print(f"[SUCCESS] Transcription completed in {processing_time:.2f}s")
            print(f"[SUCCESS] Real-time factor: {processing_time/audio_duration:.2f}x")
            
            # Show text preview
            text = result.get("text", "")
            if text:
                try:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"[SUCCESS] Text preview: {preview}")
                except UnicodeEncodeError:
                    print("[SUCCESS] Text preview: [Thai text - encoding issue]...")
            
            # Save SRT file
            srt_file = save_srt_file(result, output_name)
            if srt_file:
                print(f"[SUCCESS] SRT file saved: {srt_file}")
            
            return {
                "audio_file": audio_file,
                "output_name": output_name,
                "processing_time": processing_time,
                "real_time_factor": processing_time/audio_duration,
                "text_length": len(text),
                "srt_file": str(srt_file) if srt_file else None,
                "result": result
            }
        else:
            print(f"[FAILED] Transcription failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error during transcription: {e}")
        return None

def get_next_running_number(output_name):
    """Get next available running number for output file"""
    import glob
    pattern = f"output/{output_name}.typhoon.*.srt"
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return "001"
    
    # Extract existing numbers
    numbers = []
    for file in existing_files:
        parts = Path(file).stem.split('.')
        if len(parts) >= 3:
            try:
                numbers.append(int(parts[-1]))
            except ValueError:
                pass
    
    if not numbers:
        return "001"
    
    next_num = max(numbers) + 1
    return f"{next_num:03d}"

def save_srt_file(result, output_name):
    """Save transcription result as SRT file with running number"""
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Get next running number
        running_number = get_next_running_number(output_name)
        srt_file = output_dir / f"{output_name}.typhoon.{running_number}.srt"
        
        with open(srt_file, 'w', encoding='utf-8') as f:
            segments = result.get('segments', [])
            
            for i, segment in enumerate(segments, 1):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '')
                
                # Convert to SRT time format
                start_srt = format_srt_time(start_time)
                end_srt = format_srt_time(end_time)
                
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
        
        return str(srt_file)
        
    except Exception as e:
        print(f"[ERROR] Error saving SRT file: {e}")
        return None

def format_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def main():
    """Main test function"""
    print("TYPHOON ASR ENGLISH FILENAME TEST")
    print("="*60)
    
    # Test file with English filename
    test_file = {"path": "Jasmali.MP3", "output": "Jasmali"}
    
    # Check if file exists
    if not Path(test_file["path"]).exists():
        print(f"[ERROR] Audio file not found: {test_file['path']}")
        return
    
    # Print system information
    print_system_info()
    
    # Test GPU detection
    gpu_manager = get_gpu_manager()
    gpu_available = gpu_manager.gpu_available
    print(f"\nGPU Available: {gpu_available}")
    
    # Test model loading with GPU
    print(f"\nTesting with GPU={gpu_available}")
    try:
        print(f"Loading Typhoon NeMo model with GPU={gpu_available}...")
        
        # Create model - Use Typhoon NeMo implementation
        model = get_model(
            "typhoon_nemo",
            model_name="scb10x/typhoon-asr-realtime",
            use_gpu=gpu_available,
            language="th"
        )
        
        # Load model
        start_time = time.time()
        success = model.load_model()
        load_time = time.time() - start_time
        
        if success:
            print(f"[SUCCESS] Model loaded successfully in {load_time:.2f}s")
            print(f"[SUCCESS] Device: {model.device}")
        else:
            print("[FAILED] Failed to load model")
            return
             
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    
    # Test transcription
    print(f"\n\n{'#'*80}")
    print(f"PROCESSING: {test_file['path']}")
    print(f"{'#'*80}")
    
    result = test_transcription(model, test_file["path"], test_file["output"])
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"TEST SUMMARY")
    print("="*60)
    
    if result:
        print("[SUCCESS] Model loading: SUCCESS")
        print(f"  Device: {model.device}")
        print(f"  Load time: {load_time:.2f}s")
        print("[SUCCESS] Transcription: SUCCESS")
        print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"  Real-time factor: {result.get('real_time_factor', 0):.2f}x")
        print(f"  Text length: {result.get('text_length', 0)} characters")
        if result.get('srt_file'):
            print(f"[SUCCESS] SRT file: {result['srt_file']}")
        
        # Save test report
        test_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Typhoon ASR (NeMo)",
            "device": str(model.device),
            "audio_file": test_file["path"],
            "load_time": load_time,
            "processing_time": result.get('processing_time', 0),
            "real_time_factor": result.get('real_time_factor', 0),
            "text_length": result.get('text_length', 0),
            "srt_file": result.get('srt_file'),
            "success": True
        }
        
        try:
            with open("output/typhoon_english_test_report.json", 'w', encoding='utf-8') as f:
                json.dump(test_report, f, indent=2, ensure_ascii=False)
            print(f"\nTest report saved to: output/typhoon_english_test_report.json")
        except Exception as e:
            print(f"Error saving test report: {e}")
    else:
        print("[FAILED] Transcription: FAILED")
    
    print("="*60)
    print("\nTEST COMPLETED!")

if __name__ == "__main__":
    main()