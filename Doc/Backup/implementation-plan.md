# Typhoon ASR Integration Implementation Plan

## Overview
This document outlines the implementation plan for integrating the Typhoon ASR Realtime model into the SRT Generator project, along with GPU support enhancements.

## Architecture Changes

### 1. Model Abstraction Layer
Create a model abstraction to support multiple ASR engines:

```python
class ASRModel(ABC):
    @abstractmethod
    def load_model(self, model_path: str, device: str = "auto") -> None:
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "th") -> Dict:
        pass
    
    @abstractmethod
    def is_gpu_available(self) -> bool:
        pass

class WhisperModel(ASRModel):
    # Current Whisper implementation

class TyphoonModel(ASRModel):
    # New Typhoon implementation
```

### 2. GPU Detection and Management
Implement comprehensive GPU support:

```python
class GPUManager:
    @staticmethod
    def detect_gpu() -> bool:
        """Check if CUDA GPU is available"""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU memory and capability info"""
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory,
                "free_memory": torch.cuda.memory_reserved(0)
            }
        return {"available": False}
    
    @staticmethod
    def optimize_model_for_gpu(model):
        """Optimize model for GPU inference"""
        if torch.cuda.is_available():
            model = model.cuda()
            # Additional optimizations
        return model
```

### 3. Configuration Management
Add configuration options for model selection and GPU usage:

```python
# config.py
DEFAULT_CONFIG = {
    "asr_model": "whisper",  # or "typhoon"
    "model_size": "large",   # for Whisper
    "use_gpu": True,
    "gpu_memory_fraction": 0.7,
    "language": "th",
    "chunk_duration_ms": 30000
}
```

## Implementation Steps

### Phase 1: GPU Support Implementation
1. Create GPUManager class
2. Update model loading in audio_to_srt.py
3. Add GPU detection to setup_models.py
4. Test GPU acceleration with existing Whisper models

### Phase 2: Typhoon ASR Integration
1. Research Typhoon ASR API and dependencies
2. Create TyphoonModel class
3. Update requirements.txt with new dependencies
4. Implement model loading and transcription methods
5. Add Typhoon model download functionality

### Phase 3: Testing and Benchmarking
1. Create comprehensive test suite
2. Implement performance benchmarks
3. Compare transcription quality between models
4. Document resource usage and speed improvements

### Phase 4: Documentation and Deployment
1. Update README.md with new features
2. Create user guide for GPU usage
3. Update batch files to support GPU options
4. Add troubleshooting section for GPU issues

## File Changes Required

### New Files
- `SRT-Generator/models/base_asr.py` - Abstract base class for ASR models
- `SRT-Generator/models/whisper_model.py` - Whisper implementation
- `SRT-Generator/models/typhoon_model.py` - Typhoon implementation
- `SRT-Generator/utils/gpu_manager.py` - GPU detection and management
- `SRT-Generator/config.py` - Configuration management
- `SRT-Generator/tests/benchmark.py` - Performance benchmarking script

### Modified Files
- `audio_to_srt.py` - Main script updates for new architecture
- `setup_models.py` - Add GPU support and Typhoon model download
- `requirements.txt` - Add new dependencies
- `install.bat` - Update installation process
- `generate_srt.bat` - Add GPU options
- `generate_srt_custom.bat` - Add GPU and model selection options
- `.gitignore` - Add Doc folder (pending mode switch)

## GPU Implementation Details

### Memory Management
- Implement automatic batch size adjustment based on GPU memory
- Add memory cleanup between transcription chunks
- Handle out-of-memory errors gracefully

### Performance Optimization
- Use mixed precision inference (FP16) for faster processing
- Implement model quantization if supported
- Add streaming capability for real-time transcription

### Fallback Mechanisms
- Automatic fallback to CPU if GPU fails
- Graceful degradation when GPU memory is insufficient
- Clear error messages for GPU-related issues

## Typhoon ASR Integration Details

### Dependencies (to be confirmed)
- transformers library
- torch
- torchaudio
- Additional Thai-specific processing libraries

### Model Loading
```python
def load_typhoon_model(model_path: str = None, use_gpu: bool = True):
    """Load Typhoon ASR model with optional GPU support"""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    
    model_id = "scb10x/typhoon-asr-realtime"
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor
```

### Transcription Method
```python
def transcribe_with_typhoon(model, processor, audio_path: str, language: str = "th"):
    """Transcribe audio using Typhoon ASR"""
    import torchaudio
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:  # Typhoon might require specific sample rate
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Process with model
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return {"text": transcription}
```

## Testing Strategy

### Unit Tests
- Test GPU detection and management
- Test model loading for both Whisper and Typhoon
- Test transcription accuracy with known audio samples

### Integration Tests
- Test end-to-end SRT generation with both models
- Test GPU fallback mechanisms
- Test performance with different audio lengths

### Performance Benchmarks
- Measure transcription speed (real-time factor)
- Compare memory usage
- Test GPU vs CPU performance
- Compare accuracy between models

## Success Criteria

### Performance Metrics
- GPU acceleration should provide at least 2x speed improvement
- Typhoon ASR should show equal or better accuracy than Whisper large
- Memory usage should be optimized for GPU constraints

### User Experience
- Easy switching between models via configuration
- Clear feedback about GPU usage
- Graceful handling of GPU unavailable scenarios
- Maintained compatibility with existing batch files

## Timeline Estimate
- Phase 1 (GPU Support): 2-3 days
- Phase 2 (Typhoon Integration): 3-5 days
- Phase 3 (Testing): 2-3 days
- Phase 4 (Documentation): 1-2 days

Total estimated time: 8-13 days

## Risks and Mitigations
1. **Typhoon API Changes**: Monitor for updates and maintain flexibility in implementation
2. **GPU Compatibility Issues**: Test on multiple GPU configurations
3. **Dependency Conflicts**: Use virtual environments and version pinning
4. **Performance Regression**: Maintain backward compatibility with Whisper

## Next Steps
1. Get approval for implementation plan
2. Switch to Code mode to begin implementation
3. Start with Phase 1: GPU Support Implementation
4. Update .gitignore to include Doc folder