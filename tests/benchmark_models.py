#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Benchmark Script
Compares transcription accuracy and performance between Whisper and Typhoon ASR models
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.gpu_manager import get_gpu_manager, print_system_info
    from config import get_config
    from scripts import get_model
    from audio_to_srt import AudioTranscriber
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are properly installed")
    sys.exit(1)


class ModelBenchmark:
    """
    Comprehensive benchmark for ASR models
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize benchmark
        
        Args:
            config_file: Path to configuration file
        """
        self.config = get_config(config_file)
        self.gpu_manager = get_gpu_manager()
        self.results = {}
        
    def find_test_audio_files(self) -> List[str]:
        """
        Find available test audio files
        
        Returns:
            List of audio file paths
        """
        test_files = []
        extensions = ['.mp3', '.MP3', '.wav', '.WAV', '.m4a', '.M4A']
        
        for ext in extensions:
            test_files.extend(Path('.').glob(f'*{ext}'))
        
        # Filter out system files and directories
        test_files = [f for f in test_files if f.is_file() and not f.name.startswith('.')]
        
        return [str(f) for f in test_files]
    
    def benchmark_model(self, model_type: str, model_config: Dict, audio_files: List[str]) -> Dict[str, Any]:
        """
        Benchmark a single model
        
        Args:
            model_type: Type of model ("whisper" or "typhoon")
            model_config: Model configuration
            audio_files: List of audio files to test
            
        Returns:
            Dict containing benchmark results
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        results = {
            "model_type": model_type,
            "model_config": model_config,
            "files": {},
            "summary": {},
            "errors": []
        }
        
        try:
            # Create model
            print(f"Creating {model_type} model...")
            model = get_model(model_type, **model_config)
            
            # Load model
            print(f"Loading model...")
            load_start = time.time()
            if not model.load_model():
                raise Exception(f"Failed to load {model_type} model")
            load_time = time.time() - load_start
            
            results["load_time"] = load_time
            results["model_info"] = model.get_model_info()
            
            print(f"Model loaded in {load_time:.2f}s")
            print(f"Device: {model.device}")
            
            # Test each audio file
            for audio_file in audio_files:
                print(f"\nTesting with {audio_file}...")
                file_result = self.benchmark_single_file(model, audio_file)
                results["files"][audio_file] = file_result
                
                if "error" in file_result:
                    results["errors"].append(f"{audio_file}: {file_result['error']}")
            
            # Calculate summary statistics
            results["summary"] = self.calculate_summary_stats(results["files"])
            
        except Exception as e:
            print(f"Error benchmarking {model_type}: {e}")
            results["error"] = str(e)
        
        return results
    
    def benchmark_single_file(self, model, audio_file: str) -> Dict[str, Any]:
        """
        Benchmark model with a single audio file
        
        Args:
            model: ASR model instance
            audio_file: Path to audio file
            
        Returns:
            Dict containing benchmark results for this file
        """
        result = {
            "audio_file": audio_file,
            "transcriptions": [],
            "performance": {}
        }
        
        try:
            # Get audio duration
            audio_duration = model.get_audio_duration(audio_file)
            result["audio_duration"] = audio_duration
            
            # Run multiple transcriptions for consistency
            num_runs = 3
            processing_times = []
            transcriptions = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                
                start_time = time.time()
                transcription_result = model.transcribe(audio_file)
                end_time = time.time()
                
                if "error" not in transcription_result:
                    processing_time = end_time - start_time
                    processing_times.append(processing_time)
                    transcriptions.append(transcription_result)
                    
                    print(f"    Time: {processing_time:.2f}s")
                    print(f"    Text: {transcription_result.get('text', '')[:100]}...")
                else:
                    print(f"    Error: {transcription_result['error']}")
                    result["error"] = transcription_result["error"]
                    return result
            
            if not transcriptions:
                result["error"] = "All transcription runs failed"
                return result
            
            # Calculate performance metrics
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            avg_rtf = avg_time / audio_duration if audio_duration > 0 else 0
            
            result["performance"] = {
                "num_runs": len(processing_times),
                "avg_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "avg_real_time_factor": avg_rtf,
                "processing_times": processing_times
            }
            
            # Store transcriptions
            result["transcriptions"] = transcriptions
            
            # Calculate transcription consistency
            if len(transcriptions) > 1:
                consistency = self.calculate_transcription_consistency(transcriptions)
                result["consistency"] = consistency
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def calculate_transcription_consistency(self, transcriptions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate consistency between multiple transcriptions
        
        Args:
            transcriptions: List of transcription results
            
        Returns:
            Dict containing consistency metrics
        """
        if len(transcriptions) < 2:
            return {"score": 1.0, "note": "Only one transcription"}
        
        texts = [t.get("text", "") for t in transcriptions]
        
        # Simple consistency check based on text similarity
        # In a full implementation, you might use more sophisticated metrics
        base_text = texts[0]
        similarities = []
        
        for text in texts[1:]:
            # Simple character-level similarity
            common_chars = set(base_text) & set(text)
            total_chars = set(base_text) | set(text)
            similarity = len(common_chars) / len(total_chars) if total_chars else 0
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        return {
            "score": avg_similarity,
            "similarities": similarities,
            "note": "Character-level similarity"
        }
    
    def calculate_summary_stats(self, file_results: Dict) -> Dict[str, Any]:
        """
        Calculate summary statistics across all files
        
        Args:
            file_results: Results for each file
            
        Returns:
            Dict containing summary statistics
        """
        summary = {
            "total_files": len(file_results),
            "successful_files": 0,
            "failed_files": 0,
            "total_audio_duration": 0,
            "total_processing_time": 0,
            "avg_real_time_factor": 0,
            "avg_consistency": 0
        }
        
        total_rtf = 0
        total_consistency = 0
        rtf_count = 0
        consistency_count = 0
        
        for file_path, result in file_results.items():
            if "error" in result:
                summary["failed_files"] += 1
                continue
            
            summary["successful_files"] += 1
            summary["total_audio_duration"] += result.get("audio_duration", 0)
            summary["total_processing_time"] += result["performance"].get("avg_processing_time", 0)
            
            rtf = result["performance"].get("avg_real_time_factor", 0)
            if rtf > 0:
                total_rtf += rtf
                rtf_count += 1
            
            consistency = result.get("consistency", {}).get("score", 0)
            if consistency > 0:
                total_consistency += consistency
                consistency_count += 1
        
        if rtf_count > 0:
            summary["avg_real_time_factor"] = total_rtf / rtf_count
        
        if consistency_count > 0:
            summary["avg_consistency"] = total_consistency / consistency_count
        
        return summary
    
    def compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results between models
        
        Args:
            results: Results from all models
            
        Returns:
            Dict containing comparison metrics
        """
        comparison = {
            "models": list(results.keys()),
            "performance_comparison": {},
            "accuracy_comparison": {},
            "recommendations": []
        }
        
        # Compare performance metrics
        models = list(results.keys())
        if len(models) >= 2:
            model1, model2 = models[0], models[1]
            
            summary1 = results[model1].get("summary", {})
            summary2 = results[model2].get("summary", {})
            
            comparison["performance_comparison"] = {
                "load_time": {
                    model1: results[model1].get("load_time", 0),
                    model2: results[model2].get("load_time", 0),
                    "winner": model1 if results[model1].get("load_time", 0) < results[model2].get("load_time", 0) else model2
                },
                "avg_rtf": {
                    model1: summary1.get("avg_real_time_factor", 0),
                    model2: summary2.get("avg_real_time_factor", 0),
                    "winner": model1 if summary1.get("avg_real_time_factor", 0) < summary2.get("avg_real_time_factor", 0) else model2
                },
                "success_rate": {
                    model1: summary1.get("successful_files", 0) / max(summary1.get("total_files", 1), 1),
                    model2: summary2.get("successful_files", 0) / max(summary2.get("total_files", 1), 1),
                    "winner": model1 if summary1.get("successful_files", 0) > summary2.get("successful_files", 0) else model2
                }
            }
            
            # Generate recommendations
            self.generate_recommendations(comparison, results)
        
        return comparison
    
    def generate_recommendations(self, comparison: Dict, results: Dict):
        """
        Generate recommendations based on comparison
        
        Args:
            comparison: Comparison metrics
            results: Full results
        """
        recommendations = []
        
        perf_comp = comparison.get("performance_comparison", {})
        
        # Speed recommendation
        if "avg_rtf" in perf_comp:
            rtf_winner = perf_comp["avg_rtf"]["winner"]
            # Get the other model
            models = list(perf_comp["avg_rtf"].keys())
            models.remove("winner")
            other_model = models[0] if models else None
            
            if other_model:
                rtf_diff = abs(perf_comp["avg_rtf"][rtf_winner] - perf_comp["avg_rtf"][other_model])
                if rtf_diff > 0.5:  # Significant difference
                    recommendations.append(
                        f"For speed: {rtf_winner} is {rtf_diff:.2f}x faster on average"
                    )
        
        # Reliability recommendation
        if "success_rate" in perf_comp:
            success_winner = perf_comp["success_rate"]["winner"]
            success_rate = perf_comp["success_rate"][success_winner]
            if success_rate > 0.9:
                recommendations.append(
                    f"For reliability: {success_winner} has {success_rate*100:.1f}% success rate"
                )
        
        # Thai language specialization
        if "typhoon" in results:
            recommendations.append(
                "For Thai language: Typhoon ASR is specifically designed for Thai and may provide better accuracy"
            )
        
        # GPU utilization
        if self.gpu_manager.gpu_available:
            recommendations.append(
                "GPU acceleration is available and recommended for faster processing"
            )
        
        comparison["recommendations"] = recommendations
    
    def save_results(self, results: Dict, output_file: str = "benchmark_results.json"):
        """
        Save benchmark results to file
        
        Args:
            results: Benchmark results
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: Dict):
        """
        Print benchmark summary
        
        Args:
            results: Benchmark results
        """
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for model_type, model_results in results.items():
            if model_type == "comparison":
                continue
            
            print(f"\n{model_type.upper()} MODEL:")
            print("-" * 40)
            
            if "error" in model_results:
                print(f"  Error: {model_results['error']}")
                continue
            
            summary = model_results.get("summary", {})
            print(f"  Load Time: {model_results.get('load_time', 0):.2f}s")
            print(f"  Files Tested: {summary.get('total_files', 0)}")
            print(f"  Successful: {summary.get('successful_files', 0)}")
            print(f"  Failed: {summary.get('failed_files', 0)}")
            print(f"  Avg Real-time Factor: {summary.get('avg_real_time_factor', 0):.2f}")
            print(f"  Avg Consistency: {summary.get('avg_consistency', 0):.2f}")
        
        # Print comparison if available
        if "comparison" in results:
            comparison = results["comparison"]
            print(f"\nCOMPARISON:")
            print("-" * 40)
            
            perf_comp = comparison.get("performance_comparison", {})
            for metric, data in perf_comp.items():
                if "winner" in data:
                    print(f"  {metric}: {data['winner']} wins")
            
            recommendations = comparison.get("recommendations", [])
            if recommendations:
                print(f"\nRECOMMENDATIONS:")
                for rec in recommendations:
                    print(f"  • {rec}")
        
        print("="*80)


def main():
    """Main function to run benchmarks"""
    parser = argparse.ArgumentParser(description='Benchmark ASR models')
    parser.add_argument('--config', help='Configuration file path', default=None)
    parser.add_argument('--models', nargs='+', help='Models to test (whisper, typhoon)', 
                       default=['whisper', 'typhoon'])
    parser.add_argument('--output', help='Output file for results', default='benchmark_results.json')
    parser.add_argument('--whisper-size', help='Whisper model size', default='base')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    # Initialize benchmark
    benchmark = ModelBenchmark(args.config)
    
    # Find test audio files
    audio_files = benchmark.find_test_audio_files()
    
    if not audio_files:
        print("No audio files found for testing")
        print("Please place audio files in the current directory")
        sys.exit(1)
    
    print(f"\nFound {len(audio_files)} audio files for testing:")
    for f in audio_files:
        try:
            print(f"  - {f}")
        except UnicodeEncodeError:
            # Handle Unicode filenames
            print(f"  - {Path(f).name.encode('ascii', 'replace').decode('ascii')}")
    
    # Prepare model configurations
    model_configs = {
        "whisper": {
            "model_name": args.whisper_size,
            "use_gpu": benchmark.gpu_manager.gpu_available and not args.no_gpu,
            "language": "th"
        },
        "typhoon": {
            "model_name": "scb10x/typhoon-asr-realtime",
            "use_gpu": benchmark.gpu_manager.gpu_available and not args.no_gpu,
            "language": "th"
        }
    }
    
    # Run benchmarks
    results = {}
    
    for model_type in args.models:
        if model_type in model_configs:
            model_result = benchmark.benchmark_model(
                model_type, 
                model_configs[model_type], 
                audio_files
            )
            results[model_type] = model_result
        else:
            print(f"Unknown model type: {model_type}")
    
    # Compare models if multiple were tested
    if len(results) > 1:
        results["comparison"] = benchmark.compare_models(results)
    
    # Save and print results
    benchmark.save_results(results, args.output)
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()