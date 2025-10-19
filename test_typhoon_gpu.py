#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Typhoon ASR model on GPU
Tests both Typhoon implementations (Transformers and NeMo) with GPU acceleration
"""

import os
import sys
import time
import json
import glob
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts import get_model
from utils.gpu_manager import get_gpu_manager, print_system_info
from utils.text_segmenter import TextSegmenter
from audio_to_srt import AudioTranscriber

def get_next_running_number(audio_basename, model_name):
    """Get next available running number for output file"""
    pattern = f"output/{audio_basename}.{model_name}.*.srt"
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

def test_typhoon_model(audio_file, model_type, use_gpu=True, model_name=None):
    """
    Test transcription with specified Typhoon model
    
    Args:
        audio_file: Path to audio file
        model_type: Type of Typhoon model ("typhoon" or "typhoon_nemo")
        use_gpu: Whether to use GPU
        model_name: Specific model name (optional)
        
    Returns:
        Dict containing test results
    """
    print(f"\n{'='*60}")
    print(f"TESTING {model_type.upper()} MODEL")
    try:
        print(f"Audio File: {audio_file}")
    except UnicodeEncodeError:
        print(f"Audio File: [Thai filename]")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"{'='*60}")
    
    # Configuration
    config_overrides = {
        "asr_model": model_type,
        "use_gpu": use_gpu,
        "verbose": True
    }
    
    if model_name:
        config_overrides["typhoon_model"] = model_name
    
    # Create transcriber
    transcriber = AudioTranscriber(**config_overrides)
    
    # Get output path with running number
    audio_path = Path(audio_file)
    audio_basename = audio_path.stem
    
    running_number = get_next_running_number(audio_basename, model_type)
    output_path = f"output/{audio_basename}.{model_type}.{running_number}.srt"
    
    # Transcribe
    start_time = time.time()
    segments = transcriber.transcribe_audio(audio_file)
    total_time = time.time() - start_time
    
    # Generate SRT
    if segments:
        transcriber.generate_srt(segments, output_path)
        
        # Collect metrics
        metrics = {
            "model": model_type,
            "model_name": model_name or transcriber.config.get("typhoon_model"),
            "audio_file": audio_file,
            "output_file": output_path,
            "total_time": total_time,
            "segments_count": len(segments),
            "success": True,
            "use_gpu": use_gpu
        }
        
        # Get model stats
        if transcriber.asr_model:
            stats = transcriber.asr_model.get_performance_stats()
            metrics.update(stats)
        
        # Assess quality
        quality = assess_subtitle_quality(segments)
        metrics["quality"] = quality
        
        return metrics
    else:
        return {
            "model": model_type,
            "model_name": model_name or transcriber.config.get("typhoon_model"),
            "audio_file": audio_file,
            "error": "No segments generated",
            "success": False,
            "use_gpu": use_gpu
        }

def assess_subtitle_quality(segments):
    """Assess the quality of generated subtitles"""
    if not segments:
        return {"score": 0, "issues": ["No segments generated"]}
    
    quality_score = 100
    issues = []
    
    # Check segment durations
    durations = []
    char_counts = []
    
    for seg in segments:
        # Parse start and end times
        start_time = seg.get("start_time", "")
        end_time = seg.get("end_time", "")
        text = seg.get("text", "")
        
        # Convert SRT time to seconds
        try:
            start_secs = srt_time_to_seconds(start_time)
            end_secs = srt_time_to_seconds(end_time)
            duration = end_secs - start_secs
            durations.append(duration)
        except:
            issues.append("Invalid timestamp format")
            quality_score -= 10
        
        char_counts.append(len(text))
    
    # Analyze durations
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Check if durations are within optimal range (2-7 seconds)
        if avg_duration < 2:
            issues.append("Average segment duration too short")
            quality_score -= 15
        elif avg_duration > 7:
            issues.append("Average segment duration too long")
            quality_score -= 15
        
        # Check for very short or very long segments
        very_short = sum(1 for d in durations if d < 1)
        very_long = sum(1 for d in durations if d > 10)
        
        if very_short > len(durations) * 0.2:
            issues.append(f"Too many very short segments ({very_short})")
            quality_score -= 10
        
        if very_long > 0:
            issues.append(f"Very long segments detected ({very_long})")
            quality_score -= 10
    
    # Analyze character counts
    if char_counts:
        avg_chars = sum(char_counts) / len(char_counts)
        max_chars = max(char_counts)
        
        # Check if segments have appropriate character count
        if max_chars > 84:  # Double line limit
            issues.append(f"Segment too long: {max_chars} characters")
            quality_score -= 10
        
        if avg_chars > 42:  # Single line limit
            issues.append(f"Average segment too long: {avg_chars:.1f} characters")
            quality_score -= 5
    
    return {
        "score": max(0, quality_score),
        "issues": issues,
        "segment_count": len(segments),
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "avg_chars": sum(char_counts) / len(char_counts) if char_counts else 0
    }

def srt_time_to_seconds(srt_time):
    """Convert SRT time format to seconds"""
    # Format: HH:MM:SS,mmm
    time_str = srt_time.replace(',', '.')
    parts = time_str.split(':')
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    return hours * 3600 + minutes * 60 + seconds

def compare_gpu_cpu_performance(audio_file, model_type):
    """
    Compare performance between GPU and CPU for the same model
    
    Args:
        audio_file: Path to audio file
        model_type: Type of model to test
        
    Returns:
        Dict containing comparison results
    """
    print(f"\n{'#'*80}")
    print(f"COMPARING GPU vs CPU PERFORMANCE")
    print(f"Model: {model_type}")
    print(f"Audio: {audio_file}")
    print(f"{'#'*80}")
    
    gpu_manager = get_gpu_manager()
    
    if not gpu_manager.gpu_available:
        print("GPU not available, skipping CPU vs GPU comparison")
        return None
    
    # Test with GPU
    print("\n1. Testing with GPU...")
    gpu_result = test_typhoon_model(audio_file, model_type, use_gpu=True)
    
    # Clear GPU cache before CPU test
    gpu_manager.clear_gpu_cache()
    
    # Test with CPU
    print("\n2. Testing with CPU...")
    cpu_result = test_typhoon_model(audio_file, model_type, use_gpu=False)
    
    # Calculate comparison
    comparison = {
        "audio_file": audio_file,
        "model_type": model_type,
        "gpu_result": gpu_result,
        "cpu_result": cpu_result
    }
    
    if gpu_result.get("success") and cpu_result.get("success"):
        gpu_time = gpu_result.get("total_time", 0)
        cpu_time = cpu_result.get("total_time", 0)
        
        if cpu_time > 0:
            speedup = cpu_time / gpu_time
            comparison["speedup"] = speedup
            comparison["faster"] = "GPU" if speedup > 1 else "CPU"
        
        gpu_quality = gpu_result.get("quality", {}).get("score", 0)
        cpu_quality = cpu_result.get("quality", {}).get("score", 0)
        comparison["quality_difference"] = gpu_quality - cpu_quality
    
    return comparison

def main():
    """Main test function"""
    print("TYPHOON ASR GPU TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    print_system_info()
    gpu_manager = get_gpu_manager()
    
    # Test files
    audio_files = ["Jasmali.MP3", "ขนมครก.MP3"]
    
    # Store results
    all_results = []
    gpu_cpu_comparisons = []
    
    # Test each audio file with each model
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"\nWarning: Audio file {audio_file} not found, skipping...")
            continue
        
        print(f"\n\n{'#'*80}")
        try:
            print(f"PROCESSING: {audio_file}")
        except UnicodeEncodeError:
            print(f"PROCESSING: [Thai filename]")
        print(f"{'#'*80}")
        
        # Test Typhoon (Transformers)
        print("\n--- Testing Typhoon (Transformers) ---")
        typhoon_result = test_typhoon_model(
            audio_file, 
            "typhoon",
            use_gpu=gpu_manager.gpu_available
        )
        all_results.append(typhoon_result)
        
        # Test Typhoon NeMo
        print("\n--- Testing Typhoon NeMo ---")
        typhoon_nemo_result = test_typhoon_model(
            audio_file,
            "typhoon_nemo",
            use_gpu=gpu_manager.gpu_available
        )
        all_results.append(typhoon_nemo_result)
        
        # Compare GPU vs CPU if GPU is available
        if gpu_manager.gpu_available:
            print("\n--- GPU vs CPU Comparison ---")
            
            # Compare Typhoon (Transformers)
            gpu_cpu_typhoon = compare_gpu_cpu_performance(audio_file, "typhoon")
            if gpu_cpu_typhoon:
                gpu_cpu_comparisons.append(gpu_cpu_typhoon)
            
            # Compare Typhoon NeMo
            gpu_cpu_nemo = compare_gpu_cpu_performance(audio_file, "typhoon_nemo")
            if gpu_cpu_nemo:
                gpu_cpu_comparisons.append(gpu_cpu_nemo)
    
    # Generate comprehensive report
    generate_test_report(all_results, gpu_cpu_comparisons)
    
    print("\n\nTEST COMPLETED!")
    print("="*60)
    print("Check typhoon_test_report.json for detailed results")

def generate_test_report(results, gpu_cpu_comparisons):
    """Generate comprehensive test report"""
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_tests": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False))
        },
        "results": results,
        "gpu_cpu_comparisons": gpu_cpu_comparisons,
        "analysis": {}
    }
    
    # Group results by audio file
    by_audio = {}
    for result in results:
        if result.get("success"):
            audio = result["audio_file"]
            if audio not in by_audio:
                by_audio[audio] = {}
            by_audio[audio][result["model"]] = result
    
    # Compare models for each audio file
    for audio, models in by_audio.items():
        if "typhoon" in models and "typhoon_nemo" in models:
            typhoon = models["typhoon"]
            typhoon_nemo = models["typhoon_nemo"]
            
            comparison = {
                "audio_file": audio,
                "typhoon_time": typhoon.get("total_time", 0),
                "typhoon_nemo_time": typhoon_nemo.get("total_time", 0),
                "typhoon_segments": typhoon.get("segments_count", 0),
                "typhoon_nemo_segments": typhoon_nemo.get("segments_count", 0),
                "typhoon_quality": typhoon.get("quality", {}).get("score", 0),
                "typhoon_nemo_quality": typhoon_nemo.get("quality", {}).get("score", 0)
            }
            
            # Determine winners
            if comparison["typhoon_time"] < comparison["typhoon_nemo_time"]:
                comparison["speed_winner"] = "typhoon"
            else:
                comparison["speed_winner"] = "typhoon_nemo"
            
            if comparison["typhoon_quality"] > comparison["typhoon_nemo_quality"]:
                comparison["quality_winner"] = "typhoon"
            else:
                comparison["quality_winner"] = "typhoon_nemo"
            
            report["analysis"][audio] = comparison
    
    # Add GPU vs CPU analysis
    if gpu_cpu_comparisons:
        gpu_speedups = []
        for comp in gpu_cpu_comparisons:
            if comp.get("speedup"):
                gpu_speedups.append(comp["speedup"])
        
        if gpu_speedups:
            report["gpu_performance"] = {
                "avg_speedup": sum(gpu_speedups) / len(gpu_speedups),
                "max_speedup": max(gpu_speedups),
                "min_speedup": min(gpu_speedups),
                "comparisons_count": len(gpu_speedups)
            }
    
    # Save report
    with open("typhoon_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    
    for audio, comp in report["analysis"].items():
        print(f"\n{audio}:")
        print(f"  Speed Winner: {comp['speed_winner']}")
        print(f"  Quality Winner: {comp['quality_winner']}")
        print(f"  Typhoon Quality: {comp['typhoon_quality']}/100")
        print(f"  Typhoon NeMo Quality: {comp['typhoon_nemo_quality']}/100")
    
    if "gpu_performance" in report:
        gpu_perf = report["gpu_performance"]
        print(f"\nGPU Performance:")
        print(f"  Average Speedup: {gpu_perf['avg_speedup']:.2f}x")
        print(f"  Maximum Speedup: {gpu_perf['max_speedup']:.2f}x")
        print(f"  Minimum Speedup: {gpu_perf['min_speedup']:.2f}x")

if __name__ == "__main__":
    main()