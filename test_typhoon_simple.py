#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Typhoon ASR with Thai audio
Tests both Jasmali.MP3 and the Thai audio file with English filename
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

def save_srt_file(result, output_name):
    """Save transcription result as SRT file"""
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        srt_file = output_dir / f"{output_name}.typhoon.srt"
        
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
    print("TYPHOON ASR SIMPLE TEST")
    print("="*60)
    
    # Test files
    test_files = [
        {"path": "Jasmali.MP3", "output": "Jasmali"},
        {"path": "ขนมครก.MP3", "output": "thai_audio"}  # Use English output name for Thai file
    ]
    
    # Check which files exist
    existing_files = []
    for file_info in test_files:
        if Path(file_info["path"]).exists():
            existing_files.append(file_info)
        else:
            print(f"[WARNING] Audio file not found: {file_info['path']}")
    
    if not existing_files:
        print("[ERROR] No test audio files found")
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
    
    # Test each audio file
    all_results = []
    
    for file_info in existing_files:
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING: {file_info['path']}")
        print(f"{'#'*80}")
        
        # Test transcription
        result = test_transcription(model, file_info["path"], file_info["output"])
        
        if result:
            all_results.append(result)
        
        # Print summary for this file
        print(f"\n" + "="*60)
        print(f"TEST SUMMARY FOR {file_info['output']}")
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
        else:
            print("[FAILED] Transcription: FAILED")
        
        print("="*60)
    
    # Generate summary report
    if all_results:
        print(f"\n\n{'#'*80}")
        print("SUMMARY REPORT")
        print(f"{'#'*80}")
        
        successful_tests = len(all_results)
        total_tests = len(existing_files)
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        
        if successful_tests > 0:
            avg_rtf = sum(r.get("real_time_factor", 0) for r in all_results) / successful_tests
            avg_processing_time = sum(r.get("processing_time", 0) for r in all_results) / successful_tests
            total_chars = sum(r.get("text_length", 0) for r in all_results)
            
            print(f"\nAverage performance metrics:")
            print(f"  Average real-time factor: {avg_rtf:.2f}x")
            print(f"  Average processing time: {avg_processing_time:.2f}s")
            print(f"  Total characters transcribed: {total_chars}")
        
        # Save summary report
        summary_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Typhoon ASR (NeMo)",
            "device": str(model.device),
            "summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": total_tests - successful_tests
            },
            "results": all_results
        }
        
        try:
            with open("output/typhoon_simple_test_report.json", 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False)
            print(f"\nSummary report saved to: output/typhoon_simple_test_report.json")
        except Exception as e:
            print(f"Error saving summary report: {e}")
    
    print("\n\nALL TESTS COMPLETED!")

if __name__ == "__main__":
    main()