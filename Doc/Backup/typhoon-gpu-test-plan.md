# Typhoon ASR GPU Test Plan

## Objective
Test the Typhoon ASR model with GPU acceleration on the Jasmali.MP3 file to evaluate performance and accuracy compared to Whisper.

## Test Requirements

### Prerequisites
1. GPU with CUDA support
2. Updated requirements.txt with transformers and torchaudio
3. Jasmali.MP3 file available for testing
4. Existing Whisper SRT output for comparison

### Test Steps

#### 1. Environment Setup
```bash
# Activate virtual environment
call srt_env\Scripts\activate.bat

# Install required packages
pip install transformers>=4.30.0 torchaudio>=2.0.0
```

#### 2. GPU Detection Test
```bash
python test_gpu_support.py
```

#### 3. Typhoon Model Test
Create a test script `test_typhoon_gpu.py` that will:
- Detect GPU availability
- Load Typhoon model with GPU support
- Transcribe Jasmali.MP3
- Compare performance with existing Whisper output

#### 4. Command Line Test
```bash
python audio_to_srt.py "Jasmali.MP3" --model typhoon --gpu --verbose
```

#### 5. Benchmark Comparison
```bash
python benchmark_models.py --models whisper typhoon --gpu --whisper-size base
```

## Expected Results

### Performance Metrics
- Model loading time
- Transcription speed (real-time factor)
- Memory usage (GPU vs CPU)
- Total processing time

### Quality Metrics
- Transcription accuracy
- Text completeness
- Language detection accuracy
- Timestamp precision

### Comparison Points
- Speed improvement with GPU
- Accuracy comparison with Whisper
- Resource utilization
- Error handling and fallbacks

## Test Script Structure

The test script should include:

1. **System Information**
   - GPU detection and capabilities
   - Memory availability
   - Driver information

2. **Model Loading**
   - Typhoon model initialization
   - GPU memory allocation
   - Loading time measurement

3. **Transcription Test**
   - Audio file validation
   - Transcription execution
   - Performance tracking
   - Error handling

4. **Results Analysis**
   - Performance metrics calculation
   - Quality assessment
   - Comparison with baseline
   - Recommendations

## Success Criteria

1. **Functional Requirements**
   - GPU is detected and utilized
   - Typhoon model loads successfully
   - Transcription completes without errors
   - Output SRT file is generated

2. **Performance Requirements**
   - GPU acceleration provides speed improvement
   - Memory usage is within acceptable limits
   - Processing time is reasonable for file size

3. **Quality Requirements**
   - Transcription accuracy is comparable or better than Whisper
   - Thai language recognition is accurate
   - Timestamps are correctly generated

## Troubleshooting Guide

### Common Issues
1. **GPU Not Detected**
   - Check CUDA installation
   - Verify driver compatibility
   - Check PyTorch CUDA support

2. **Model Loading Failures**
   - Internet connectivity for Hugging Face download
   - Sufficient disk space for model cache
   - Memory availability

3. **Transcription Errors**
   - Audio file format compatibility
   - Memory limitations
   - Model inference issues

### Solutions
1. Update GPU drivers
2. Increase GPU memory allocation
3. Fall back to CPU if GPU fails
4. Check audio file integrity

## Documentation

### Test Report Template
1. System configuration
2. Test execution summary
3. Performance metrics
4. Quality assessment
5. Comparison with baseline
6. Recommendations

### Output Files
- `output/Jasmali.typhoon.srt` - Typhoon transcription
- `output/Jasmali.whisper.srt` - Whisper transcription (existing)
- `typhoon_gpu_test_report.json` - Test results

## Next Steps

1. Execute the test plan
2. Analyze results
3. Document findings
4. Update recommendations
5. Optimize configuration based on results

## Implementation Notes

The test should be implemented as a standalone script that can be run independently or as part of the benchmark suite. It should provide clear output and logging for troubleshooting purposes.

Key considerations:
- Graceful fallback to CPU if GPU fails
- Clear error messages and debugging information
- Comprehensive performance tracking
- Comparison with existing baseline results