#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GPU support implementation
Tests the new architecture with GPU detection and model optimization
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
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


def test_gpu_manager():
    """Test GPU manager functionality"""
    print("="*60)
    print("TESTING GPU MANAGER")
    print("="*60)
    
    gpu_manager = get_gpu_manager()
    
    # Test GPU detection
    print(f"GPU Available: {gpu_manager.gpu_available}")
    print(f"Optimal Device: {gpu_manager.get_optimal_device()}")
    
    # Test memory estimation
    models = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large"]
    print("\nModel Memory Requirements:")
    for model in models:
        required_mb = gpu_manager.estimate_model_memory(model) / (1024*1024)
        can_fit = gpu_manager.can_fit_model(model)
        print(f"  {model}: {required_mb:.1f}MB - {'✓ Fit' if can_fit else '✗ Too large'}")
    
    # Test memory usage
    memory = gpu_manager.get_memory_usage()
    print(f"\nCurrent Memory Usage:")
    if "gpu_allocated" in memory:
        print(f"  GPU Allocated: {memory['gpu_allocated'] / (1024**2):.1f}MB")
        print(f"  GPU Free: {memory['gpu_free'] / (1024**2):.1f}MB")
    if "cpu_total" in memory:
        print(f"  CPU Total: {memory['cpu_total'] / (1024**3):.1f}GB")
        print(f"  CPU Available: {memory['cpu_available'] / (1024**3):.1f}GB")
    
    return gpu_manager


def test_configuration():
    """Test configuration system"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    config = get_config()
    
    # Test default values
    print(f"Default ASR Model: {config.get('asr_model')}")
    print(f"Default Whisper Size: {config.get('whisper_model_size')}")
    print(f"Use GPU: {config.get('use_gpu')}")
    print(f"Language: {config.get('language')}")
    
    # Test model configuration
    model_config = config.get_model_config()
    print(f"\nModel Config: {model_config}")
    
    # Test GPU configuration
    gpu_config = config.get_gpu_config()
    print(f"GPU Config: {gpu_config}")
    
    # Test configuration updates
    config.set("whisper_model_size", "base")
    config.set("use_gpu", False)
    print(f"\nAfter updates:")
    print(f"Whisper Size: {config.get('whisper_model_size')}")
    print(f"Use GPU: {config.get('use_gpu')}")
    
    return config


def test_whisper_model():
    """Test Whisper model with GPU support"""
    print("\n" + "="*60)
    print("TESTING WHISPER MODEL")
    print("="*60)
    
    gpu_manager = get_gpu_manager()
    
    # Test with different model sizes
    model_sizes = ["base", "small"]  # Use smaller models for testing
    
    for size in model_sizes:
        print(f"\nTesting Whisper {size} model...")
        
        try:
            # Create model
            model = get_model(
                "whisper",
                model_name=size,
                use_gpu=gpu_manager.gpu_available,
                language="th"
            )
            
            # Load model
            start_time = time.time()
            if model.load_model():
                load_time = time.time() - start_time
                print(f"  ✓ Model loaded in {load_time:.2f}s")
                print(f"  Device: {model.device}")
                print(f"  Model info: {model.get_model_info()}")
                
                # Test with a short audio file if available
                test_files = ["Jasmali.MP3", "ขนมครก.MP3"]
                for audio_file in test_files:
                    if Path(audio_file).exists():
                        print(f"  Testing with {audio_file}...")
                        try:
                            result = model.transcribe(audio_file)
                            if "error" not in result:
                                print(f"    ✓ Transcription successful")
                                print(f"    Processing time: {result.get('processing_time', 0):.2f}s")
                                print(f"    Text preview: {result.get('text', '')[:100]}...")
                            else:
                                print(f"    ✗ Transcription failed: {result['error']}")
                        except Exception as e:
                            print(f"    ✗ Error: {e}")
                        break  # Only test with first available file
            else:
                print(f"  ✗ Failed to load model")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")


def test_audio_transcriber():
    """Test the enhanced AudioTranscriber class"""
    print("\n" + "="*60)
    print("TESTING AUDIO TRANSCRIBER")
    print("="*60)
    
    try:
        # Test with different configurations
        configs = [
            {"asr_model": "whisper", "whisper_model_size": "base"},
            {"asr_model": "whisper", "whisper_model_size": "small"},
        ]
        
        for config_dict in configs:
            print(f"\nTesting with config: {config_dict}")
            
            transcriber = AudioTranscriber(**config_dict)
            
            if transcriber.asr_model:
                print(f"  ✓ Transcriber initialized with {transcriber.asr_model.model_name}")
                
                # Test with available audio file
                test_files = ["Jasmali.MP3", "ขนมครก.MP3"]
                for audio_file in test_files:
                    if Path(audio_file).exists():
                        print(f"  Testing transcription of {audio_file}...")
                        start_time = time.time()
                        
                        try:
                            segments = transcriber.transcribe_audio(audio_file)
                            transcription_time = time.time() - start_time
                            
                            if segments:
                                print(f"    ✓ Transcription successful")
                                print(f"    Time: {transcription_time:.2f}s")
                                print(f"    Segments: {len(segments)}")
                                print(f"    Preview: {segments[0]['text'][:50]}...")
                            else:
                                print(f"    ✗ No segments generated")
                        except Exception as e:
                            print(f"    ✗ Error: {e}")
                        break
            else:
                print(f"  ✗ Failed to initialize transcriber")
    
    except Exception as e:
        print(f"Error testing transcriber: {e}")


def test_command_line_interface():
    """Test command line interface with new options"""
    print("\n" + "="*60)
    print("TESTING COMMAND LINE INTERFACE")
    print("="*60)
    
    # Test files
    test_files = ["Jasmali.MP3", "ขนมครก.MP3"]
    test_file = None
    
    for audio_file in test_files:
        if Path(audio_file).exists():
            test_file = audio_file
            break
    
    if not test_file:
        print("No test audio files found")
        return
    
    print(f"Using test file: {test_file}")
    
    # Test different command line arguments
    test_commands = [
        [sys.executable, "audio_to_srt.py", test_file, "--model-size", "base", "--verbose"],
        [sys.executable, "audio_to_srt.py", test_file, "--model-size", "small", "--no-gpu", "--verbose"],
    ]
    
    for cmd in test_commands:
        print(f"\nTesting command: {' '.join(cmd[2:])}")
        
        try:
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode == 0:
                print("  ✓ Command executed successfully")
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"    {line}")
            else:
                print(f"  ✗ Command failed with return code {result.returncode}")
                print(f"    Error: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print("  ✗ Command timed out")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    """Run all tests"""
    print("SRT Generator GPU Support Test Suite")
    print("="*60)
    
    # Print system information
    print_system_info()
    
    # Run tests
    try:
        test_gpu_manager()
        test_configuration()
        test_whisper_model()
        test_audio_transcriber()
        test_command_line_interface()
        
        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print("="*60)
        print("Check the output above for any errors or issues")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()