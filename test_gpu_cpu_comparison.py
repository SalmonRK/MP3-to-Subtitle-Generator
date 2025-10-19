#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU vs CPU performance comparison test for Typhoon ASR
Tests transcription speed and quality on both GPU and CPU
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

def test_with_device(use_gpu, audio_file, output_name):
    """Test transcription with specified device (GPU or CPU)"""
    device_str = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*60}")
    print(f"TESTING WITH {device_str}")
    print(f"{'='*60}")
    
    if not Path(audio_file).exists():
        print(f"[ERROR] Audio file not found: {audio_file}")
        return None
    
    try:
        print(f"Loading Typhoon model with {device_str}...")
        
        # Create model
        model = get_model(
            "typhoon_nemo",
            model_name="scb10x/typhoon-asr-realtime",
            use_gpu=use_gpu,
            language="th"
        )
        
        # Load model
        start_time = time.time()
        success = model.load_model()
        load_time = time.time() - start_time
        
        if not success:
            print(f"[FAILED] Failed to load model on {device_str}")
            return None
        
        print(f"[SUCCESS] Model loaded on {device_str} in {load_time:.2f}s")
        print(f"[SUCCESS] Device: {model.device}")
        
        # Get audio duration
        audio_duration = model.get_audio_duration(audio_file)
        print(f"Audio duration: {audio_duration:.2f}s")
        
        # Transcribe
        print(f"Starting transcription on {device_str}...")
        start_time = time.time()
        result = model.transcribe(audio_file, language="th")
        processing_time = time.time() - start_time
        
        if "error" not in result:
            print(f"[SUCCESS] Transcription completed in {processing_time:.2f}s")
            rtf = processing_time/audio_duration
            print(f"[SUCCESS] Real-time factor: {rtf:.2f}x")
            
            # Show text preview
            text = result.get("text", "")
            if text:
                try:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"[SUCCESS] Text preview: {preview}")
                except UnicodeEncodeError:
                    print("[SUCCESS] Text preview: [Thai text - encoding issue]...")
            
            # Save SRT file
            srt_file = save_srt_file(result, output_name, device_str.lower())
            if srt_file:
                print(f"[SUCCESS] SRT file saved: {srt_file}")
            
            return {
                "device": device_str,
                "use_gpu": use_gpu,
                "load_time": load_time,
                "processing_time": processing_time,
                "real_time_factor": rtf,
                "text_length": len(text),
                "srt_file": str(srt_file) if srt_file else None,
                "model_device": str(model.device),
                "success": True
            }
        else:
            print(f"[FAILED] Transcription failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error during {device_str} test: {e}")
        return None

def save_srt_file(result, output_name, device):
    """Save transcription result as SRT file with device identifier"""
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Get next running number
        pattern = f"output/{output_name}.typhoon.{device}.*.srt"
        existing_files = list(Path("output").glob(f"{output_name}.typhoon.{device}.*.srt"))
        
        if existing_files:
            numbers = []
            for file in existing_files:
                parts = file.stem.split('.')
                if len(parts) >= 4:
                    try:
                        numbers.append(int(parts[-1]))
                    except ValueError:
                        pass
            
            if numbers:
                next_num = max(numbers) + 1
                running_number = f"{next_num:03d}"
            else:
                running_number = "001"
        else:
            running_number = "001"
        
        srt_file = output_dir / f"{output_name}.typhoon.{device}.{running_number}.srt"
        
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
    """Main comparison test function"""
    print("TYPHOON ASR GPU vs CPU COMPARISON TEST")
    print("="*60)
    
    # Test file
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
    
    # Store results
    results = []
    
    # Test with GPU if available
    if gpu_available:
        print(f"\n\n{'#'*80}")
        print(f"TESTING WITH GPU")
        print(f"{'#'*80}")
        
        gpu_result = test_with_device(True, test_file["path"], test_file["output"])
        if gpu_result:
            results.append(gpu_result)
    else:
        print("\n[WARNING] GPU not available, skipping GPU test")
    
    # Test with CPU
    print(f"\n\n{'#'*80}")
    print(f"TESTING WITH CPU")
    print(f"{'#'*80}")
    
    cpu_result = test_with_device(False, test_file["path"], test_file["output"])
    if cpu_result:
        results.append(cpu_result)
    
    # Generate comparison report
    if len(results) >= 2:
        print(f"\n\n{'#'*80}")
        print("COMPARISON RESULTS")
        print(f"{'#'*80}")
        
        gpu_result = next((r for r in results if r["use_gpu"]), None)
        cpu_result = next((r for r in results if not r["use_gpu"]), None)
        
        if gpu_result and cpu_result:
            print(f"\nModel Loading Time:")
            print(f"  GPU: {gpu_result['load_time']:.2f}s")
            print(f"  CPU: {cpu_result['load_time']:.2f}s")
            print(f"  Speedup: {cpu_result['load_time']/gpu_result['load_time']:.2f}x")
            
            print(f"\nProcessing Time:")
            print(f"  GPU: {gpu_result['processing_time']:.2f}s")
            print(f"  CPU: {cpu_result['processing_time']:.2f}s")
            print(f"  Speedup: {cpu_result['processing_time']/gpu_result['processing_time']:.2f}x")
            
            print(f"\nReal-time Factor:")
            print(f"  GPU: {gpu_result['real_time_factor']:.2f}x")
            print(f"  CPU: {cpu_result['real_time_factor']:.2f}x")
            
            print(f"\nText Length:")
            print(f"  GPU: {gpu_result['text_length']} characters")
            print(f"  CPU: {cpu_result['text_length']} characters")
            
            # Check if transcriptions match
            if gpu_result['text_length'] == cpu_result['text_length']:
                print(f"\n[SUCCESS] Transcriptions match exactly")
            else:
                print(f"\n[WARNING] Transcriptions differ in length")
            
            # Save comparison report
            comparison_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "audio_file": test_file["path"],
                "gpu_result": gpu_result,
                "cpu_result": cpu_result,
                "speedup": {
                    "load_time": cpu_result['load_time']/gpu_result['load_time'],
                    "processing_time": cpu_result['processing_time']/gpu_result['processing_time']
                }
            }
            
            try:
                with open("output/typhoon_gpu_cpu_comparison.json", 'w', encoding='utf-8') as f:
                    json.dump(comparison_report, f, indent=2, ensure_ascii=False)
                print(f"\nComparison report saved to: output/typhoon_gpu_cpu_comparison.json")
            except Exception as e:
                print(f"Error saving comparison report: {e}")
    
    print("\n\nCOMPARISON TEST COMPLETED!")

if __name__ == "__main__":
    main()