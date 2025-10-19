#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Typhoon ASR with GPU acceleration
Tests the Typhoon model on Jasmali.MP3 and compares with Whisper results
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
sys.path.insert(0, str(Path(__file__).parent.parent))

# Change to parent directory to access audio files
os.chdir(Path(__file__).parent.parent)

try:
    from utils.gpu_manager import get_gpu_manager, print_system_info
    from config import get_config
    from scripts import get_model
    from audio_to_srt import AudioTranscriber
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting to install missing dependencies...")
    
    # Try to install missing packages
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "torchaudio"])
        print("Dependencies installed successfully, retrying imports...")
        
        from utils.gpu_manager import get_gpu_manager, print_system_info
        from config import get_config
        from scripts import get_model
        from audio_to_srt import AudioTranscriber
    except Exception as install_error:
        print(f"Failed to install dependencies: {install_error}")
        print("Please run: pip install torch transformers torchaudio")
        sys.exit(1)


def test_gpu_detection():
    """Test GPU detection and capabilities"""
    print("="*60)
    print("GPU DETECTION TEST")
    print("="*60)
    
    gpu_manager = get_gpu_manager()
    
    print(f"GPU Available: {gpu_manager.gpu_available}")
    print(f"Optimal Device: {gpu_manager.get_optimal_device()}")
    
    if gpu_manager.gpu_available:
        memory = gpu_manager.get_memory_usage()
        print(f"GPU Memory: {memory.get('gpu_allocated', 0) / (1024**2):.1f}MB allocated")
        print(f"GPU Free: {memory.get('gpu_free', 0) / (1024**2):.1f}MB")
    
    return gpu_manager.gpu_available


def test_typhoon_model_loading(use_gpu=True):
    """Test Typhoon model loading"""
    print("\n" + "="*60)
    print("TYPHOON MODEL LOADING TEST")
    print("="*60)
    
    try:
        print(f"Loading Typhoon NeMo model with GPU={use_gpu}...")
        
        # Create model - Use Typhoon NeMo implementation
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
        
        if success:
            print(f"[SUCCESS] Model loaded successfully in {load_time:.2f}s")
            print(f"[SUCCESS] Device: {model.device}")
            print(f"[SUCCESS] Model info: {model.get_model_info()}")
            return model
        else:
            print("[FAILED] Failed to load model")
            return None
             
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None


def test_transcription(model, audio_file):
    """Test transcription with the loaded model"""
    print(f"\n" + "="*60)
    # Handle Unicode filenames for Windows console
    try:
        print(f"TRANSCRIPTION TEST: {audio_file}")
    except UnicodeEncodeError:
        print(f"TRANSCRIPTION TEST: [Thai filename]")
    print("="*60)
    
    if not model:
        print("[FAILED] No model available for transcription")
        return None
    
    if not Path(audio_file).exists():
        print(f"[ERROR] Audio file not found: {audio_file}")
        return None
    
    try:
        # Handle Unicode filenames for Windows console
        try:
            print(f"Starting transcription of {audio_file}...")
        except UnicodeEncodeError:
            print(f"Starting transcription of [Thai filename]...")
        
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
            # Handle Unicode text for Windows console
            try:
                text_preview = result.get('text', '')[:200]
                print(f"[SUCCESS] Text preview: {text_preview}...")
            except UnicodeEncodeError:
                print("[SUCCESS] Text preview: [Thai text - encoding issue]...")
            
            # Add performance metrics
            result["processing_time"] = processing_time
            result["real_time_factor"] = processing_time / audio_duration
            
            return result
        else:
            print(f"[FAILED] Transcription failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error during transcription: {e}")
        return None


def compare_with_existing_results(typhoon_result, audio_file):
    """Compare Typhoon results with existing Whisper results"""
    print(f"\n" + "="*60)
    print("COMPARISON WITH WHISPER RESULTS")
    print("="*60)
    
    # Look for existing Whisper results
    base_name = Path(audio_file).stem
    possible_whisper_files = [
        f"output/{base_name}.thai.srt",
        f"output/{base_name}.whisper.srt",
        f"output/{base_name}.large.srt"
    ]
    
    whisper_file = None
    for file_path in possible_whisper_files:
        if Path(file_path).exists():
            whisper_file = file_path
            break
    
    if not whisper_file:
        print("No existing Whisper results found for comparison")
        return
    
    print(f"Comparing with existing results: {whisper_file}")
    
    try:
        # Read Whisper SRT
        with open(whisper_file, 'r', encoding='utf-8') as f:
            whisper_content = f.read()
        
        # Extract text from SRT (simple approach)
        whisper_lines = []
        for line in whisper_content.split('\n'):
            if line.strip() and not line.isdigit() and '-->' not in line:
                whisper_lines.append(line.strip())
        
        whisper_text = ' '.join(whisper_lines)
        
        # Extract Typhoon text
        typhoon_text = typhoon_result.get('text', '')
        
        print(f"\nWhisper text preview: {whisper_text[:200]}...")
        print(f"Typhoon text preview: {typhoon_text[:200]}...")
        
        # Simple comparison metrics
        whisper_chars = len(whisper_text)
        typhoon_chars = len(typhoon_text)
        
        print(f"\nText length comparison:")
        print(f"  Whisper: {whisper_chars} characters")
        print(f"  Typhoon: {typhoon_chars} characters")
        print(f"  Difference: {abs(whisper_chars - typhoon_chars)} characters")
        
        # Performance comparison
        typhoon_rtf = typhoon_result.get('real_time_factor', 0)
        print(f"\nPerformance comparison:")
        print(f"  Typhoon RTF: {typhoon_rtf:.2f}x")
        
        return {
            "whisper_file": whisper_file,
            "whisper_chars": whisper_chars,
            "typhoon_chars": typhoon_chars,
            "typhoon_rtf": typhoon_rtf
        }
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        return None


def save_srt_file(result, audio_file):
    """Save transcription result as SRT file"""
    try:
        # Handle Unicode filenames
        try:
            base_name = Path(audio_file).stem
        except UnicodeEncodeError:
            # Fallback for Unicode filenames
            base_name = "thai_audio"
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        srt_file = output_dir / f"{base_name}.typhoon.srt"
        
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
        
        print(f"[SUCCESS] SRT file saved: {srt_file}")
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


def save_test_report(test_results, audio_file):
    """Save test results to JSON file"""
    try:
        base_name = Path(audio_file).stem
        report_file = f"output/{base_name}_typhoon_test_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Test report saved: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"[ERROR] Error saving test report: {e}")
        return None


def main():
    """Main test function"""
    print("TYPHOON ASR GPU TEST")
    print("="*60)
    
    # Test files
    audio_files = ["Jasmali.MP3", "ขนมครก.MP3"]
    
    # Check if audio files exist
    existing_files = []
    for audio_file in audio_files:
        if Path(audio_file).exists():
            existing_files.append(audio_file)
        else:
            try:
                print(f"[WARNING] Audio file not found: {audio_file}")
            except UnicodeEncodeError:
                print(f"[WARNING] Audio file not found: [Thai filename]")
    
    if not existing_files:
        print("[ERROR] No test audio files found")
        print("Please make sure at least one of Jasmali.MP3 or ขนมครก.MP3 is in the current directory")
        return
    
    # Print system information
    print_system_info()
    
    # Test GPU detection
    gpu_available = test_gpu_detection()
    
    # Test model loading with GPU
    print(f"\nTesting with GPU={gpu_available}")
    model = test_typhoon_model_loading(use_gpu=gpu_available)
    
    if not model:
        # Try with CPU if GPU failed
        if gpu_available:
            print("\nTrying with CPU...")
            model = test_typhoon_model_loading(use_gpu=False)
    
    if not model:
        print("[ERROR] Failed to load model with both GPU and CPU")
        return
    
    # Test each audio file
    all_results = []
    
    for audio_file in existing_files:
        print(f"\n\n{'#'*80}")
        try:
            print(f"PROCESSING: {audio_file}")
        except UnicodeEncodeError:
            print(f"PROCESSING: [Thai filename]")
        print(f"{'#'*80}")
        
        # Test results container
        test_results = {
            "audio_file": audio_file,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {},
            "gpu_test": {"available": gpu_available},
            "model_loading": {
                "success": True,
                "device": str(model.device),
                "load_time": getattr(model, 'load_time', 0)
            },
            "transcription": {},
            "comparison": {},
            "files_generated": []
        }
        
        # Test transcription
        result = test_transcription(model, audio_file)
        
        if result:
            test_results["transcription"]["success"] = True
            test_results["transcription"]["processing_time"] = result.get("processing_time")
            test_results["transcription"]["real_time_factor"] = result.get("real_time_factor")
            test_results["transcription"]["text_length"] = len(result.get("text", ""))
            test_results["transcription"]["text_preview"] = result.get("text", "")[:200]
            
            # Save SRT file
            srt_file = save_srt_file(result, audio_file)
            if srt_file:
                test_results["files_generated"].append(srt_file)
            
            # Compare with existing results
            comparison = compare_with_existing_results(result, audio_file)
            if comparison:
                test_results["comparison"] = comparison
        else:
            test_results["transcription"]["success"] = False
        
        # Save test report
        save_test_report(test_results, audio_file)
        all_results.append(test_results)
        
        # Print summary for this file
        print(f"\n" + "="*60)
        try:
            print(f"TEST SUMMARY FOR {audio_file}")
        except UnicodeEncodeError:
            print(f"TEST SUMMARY FOR [Thai filename]")
        print("="*60)
        
        if test_results["model_loading"]["success"]:
            print("[SUCCESS] Model loading: SUCCESS")
            print(f"  Device: {test_results['model_loading'].get('device', 'Unknown')}")
            print(f"  Load time: {test_results['model_loading'].get('load_time', 0):.2f}s")
        else:
            print("[FAILED] Model loading: FAILED")
        
        if test_results["transcription"]["success"]:
            print("[SUCCESS] Transcription: SUCCESS")
            print(f"  Processing time: {test_results['transcription'].get('processing_time', 0):.2f}s")
            print(f"  Real-time factor: {test_results['transcription'].get('real_time_factor', 0):.2f}x")
            print(f"  Text length: {test_results['transcription'].get('text_length', 0)} characters")
        else:
            print("[FAILED] Transcription: FAILED")
        
        if test_results["files_generated"]:
            print(f"[SUCCESS] Files generated: {len(test_results['files_generated'])}")
            for file_path in test_results["files_generated"]:
                print(f"  - {file_path}")
        else:
            print("[FAILED] No files generated")
        
        print("="*60)
    
    # Generate combined report
    if len(existing_files) > 1:
        generate_combined_report(all_results)
    
    print("\n\nALL TESTS COMPLETED!")


def generate_combined_report(all_results):
    """Generate a combined report for all audio files"""
    print(f"\n\n{'#'*80}")
    print("COMBINED TEST REPORT")
    print(f"{'#'*80}")
    
    successful_tests = sum(1 for r in all_results if r["transcription"]["success"])
    total_tests = len(all_results)
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    
    if successful_tests > 0:
        avg_rtf = sum(r["transcription"].get("real_time_factor", 0) for r in all_results if r["transcription"]["success"]) / successful_tests
        avg_processing_time = sum(r["transcription"].get("processing_time", 0) for r in all_results if r["transcription"]["success"]) / successful_tests
        total_chars = sum(r["transcription"].get("text_length", 0) for r in all_results if r["transcription"]["success"])
        
        print(f"\nAverage performance metrics:")
        print(f"  Average real-time factor: {avg_rtf:.2f}x")
        print(f"  Average processing time: {avg_processing_time:.2f}s")
        print(f"  Total characters transcribed: {total_chars}")
    
    # Save combined report
    combined_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tests": total_tests,
            "successful": successful_tests,
            "failed": total_tests - successful_tests
        },
        "results": all_results
    }
    
    try:
        with open("output/typhoon_combined_test_report.json", 'w', encoding='utf-8') as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)
        print(f"\nCombined report saved to: output/typhoon_combined_test_report.json")
    except Exception as e:
        print(f"Error saving combined report: {e}")


if __name__ == "__main__":
    main()