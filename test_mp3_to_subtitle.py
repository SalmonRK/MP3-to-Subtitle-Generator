#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for MP3 to subtitle conversion
Tests both Whisper Large and Typhoon ASR models
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

from audio_to_srt import AudioTranscriber
from utils.gpu_manager import get_gpu_manager
from utils.text_segmenter import TextSegmenter

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

def test_model_transcription(audio_file, model_type, model_size=None, use_gpu=True):
    """Test transcription with specified model"""
    print(f"\n{'='*60}")
    print(f"TESTING {model_type.upper()} MODEL")
    print(f"Audio File: {audio_file}")
    print(f"{'='*60}")
    
    # Configuration
    config_overrides = {
        "asr_model": model_type,
        "use_gpu": use_gpu,
        "verbose": True
    }
    
    if model_type == "whisper" and model_size:
        config_overrides["whisper_model_size"] = model_size
    
    # Create transcriber
    transcriber = AudioTranscriber(**config_overrides)
    
    # Get output path with running number
    audio_path = Path(audio_file)
    audio_basename = audio_path.stem
    
    if model_type == "whisper":
        model_identifier = f"whisper.{model_size}"
    else:
        model_identifier = "typhoon"
    
    running_number = get_next_running_number(audio_basename, model_identifier)
    output_path = f"output/{audio_basename}.{model_identifier}.{running_number}.srt"
    
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
            "model_size": model_size,
            "audio_file": audio_file,
            "output_file": output_path,
            "total_time": total_time,
            "segments_count": len(segments),
            "success": True
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
            "model_size": model_size,
            "audio_file": audio_file,
            "error": "No segments generated",
            "success": False
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

def main():
    """Main test function"""
    print("MP3 TO SUBTITLE TEST: WHISPER LARGE vs TYPHOON")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    gpu_manager = get_gpu_manager()
    print(f"GPU Available: {gpu_manager.gpu_available}")
    
    # Test files
    audio_files = ["Jasmali.MP3", "ขนมครก.MP3"]
    
    # Store results
    all_results = []
    
    # Test each audio file
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"\nWarning: Audio file {audio_file} not found, skipping...")
            continue
        
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING: {audio_file}")
        print(f"{'#'*80}")
        
        # Test Whisper Large
        whisper_result = test_model_transcription(
            audio_file, 
            "whisper", 
            model_size="large",
            use_gpu=gpu_manager.gpu_available
        )
        all_results.append(whisper_result)
        
        # Test Typhoon
        typhoon_result = test_model_transcription(
            audio_file,
            "typhoon",
            use_gpu=gpu_manager.gpu_available
        )
        all_results.append(typhoon_result)
    
    # Generate report
    generate_test_report(all_results)
    
    print("\n\nTEST COMPLETED!")
    print("="*60)
    print("Check test_report.json for detailed results")

def generate_test_report(results):
    """Generate comprehensive test report"""
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_tests": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False))
        },
        "results": results,
        "comparison": {}
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
        if "whisper" in models and "typhoon" in models:
            whisper = models["whisper"]
            typhoon = models["typhoon"]
            
            comparison = {
                "audio_file": audio,
                "whisper_time": whisper.get("total_time", 0),
                "typhoon_time": typhoon.get("total_time", 0),
                "whisper_segments": whisper.get("segments_count", 0),
                "typhoon_segments": typhoon.get("segments_count", 0),
                "whisper_quality": whisper.get("quality", {}).get("score", 0),
                "typhoon_quality": typhoon.get("quality", {}).get("score", 0)
            }
            
            # Determine winners
            if comparison["whisper_time"] < comparison["typhoon_time"]:
                comparison["speed_winner"] = "whisper"
            else:
                comparison["speed_winner"] = "typhoon"
            
            if comparison["whisper_quality"] > comparison["typhoon_quality"]:
                comparison["quality_winner"] = "whisper"
            else:
                comparison["quality_winner"] = "typhoon"
            
            report["comparison"][audio] = comparison
    
    # Save report
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    
    for audio, comp in report["comparison"].items():
        print(f"\n{audio}:")
        print(f"  Speed Winner: {comp['speed_winner']}")
        print(f"  Quality Winner: {comp['quality_winner']}")
        print(f"  Whisper Quality: {comp['whisper_quality']}/100")
        print(f"  Typhoon Quality: {comp['typhoon_quality']}/100")

if __name__ == "__main__":
    main()