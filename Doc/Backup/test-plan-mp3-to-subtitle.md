# Test Plan: MP3 to Subtitle Conversion for Large and Typhoon Models

## Overview
This document outlines the plan to test the MP3 to subtitle functionality for both Whisper Large and Typhoon ASR models, focusing on:
1. Proper subtitle timing and tracking
2. Thai text segmentation quality
3. Output file naming with model identification and running numbers

## Test Objectives
1. Verify that both models can accurately transcribe Thai audio
2. Check that subtitle timestamps align properly with spoken content
3. Ensure text segmentation creates readable subtitle segments
4. Generate uniquely named output files to avoid conflicts

## Test Files
- `Jasmali.MP3` - Thai audio file 1
- `ขนมครก.MP3` - Thai audio file 2 (Thai filename)

## Test Script Design

### 1. Test Script Structure
The test script (`test_mp3_to_subtitle.py`) will:
- Load both Whisper Large and Typhoon models
- Process each audio file with both models
- Generate SRT files with proper naming convention
- Collect performance metrics and quality indicators

### 2. Output Naming Convention
Output files will follow this pattern:
- `{audio_basename}.whisper.large.{running_number}.srt`
- `{audio_basename}.typhoon.{running_number}.srt`

Example:
- `Jasmali.whisper.large.001.srt`
- `Jasmali.typhoon.001.srt`
- `ขนมครก.whisper.large.001.srt`
- `ขนมครก.typhoon.001.srt`

### 3. Test Procedure

#### Step 1: Environment Setup
- Activate `srt_env` virtual environment
- Import required modules
- Initialize GPU manager
- Check model availability

#### Step 2: Model Loading
- Load Whisper Large model with GPU support
- Load Typhoon ASR model with GPU support
- Record model loading times

#### Step 3: Audio Processing
For each audio file:
1. Get audio duration
2. Transcribe with Whisper Large
3. Transcribe with Typhoon
4. Generate SRT files with proper naming
5. Record processing metrics

#### Step 4: Quality Assessment
For each transcription:
1. Check subtitle timing accuracy
2. Verify text segmentation quality
3. Count number of segments
4. Calculate average segment duration
5. Check for Thai text encoding issues

#### Step 5: Report Generation
Create a comprehensive report including:
- Model performance comparison
- Processing time metrics
- Quality indicators
- Recommendations

## Implementation Details

### Key Functions
1. `get_next_running_number()` - Get next available running number
2. `test_model_transcription()` - Test a single model with audio
3. `generate_srt_with_proper_naming()` - Create SRT with correct naming
4. `assess_subtitle_quality()` - Evaluate subtitle quality
5. `generate_test_report()` - Create final report

### Quality Metrics
1. **Timing Accuracy**: Check if subtitle timestamps align with speech
2. **Segment Duration**: Ensure segments are within optimal range (2-7 seconds)
3. **Text Length**: Verify segments don't exceed character limits
4. **Thai Encoding**: Check for proper Thai text display
5. **Natural Breaks**: Ensure segmentation respects Thai language patterns

### Expected Outputs
1. SRT files for each model/audio combination
2. Performance metrics JSON file
3. Quality assessment report
4. Summary comparison table

## Test Commands

### Running the Test
```bash
# Activate environment
srt_env\Scripts\activate

# Run test script
python test_mp3_to_subtitle.py

# Deactivate
deactivate
```

### Alternative: Individual Model Testing
```bash
# Test Whisper Large
srt_env\Scripts\python audio_to_srt.py Jasmali.MP3 --model whisper --model-size large --verbose

# Test Typhoon
srt_env\Scripts\python audio_to_srt.py Jasmali.MP3 --model typhoon --verbose
```

## Success Criteria
1. Both models successfully transcribe audio files
2. Generated SRT files have proper timestamps
3. Thai text is correctly encoded and displayed
4. Subtitle segments are of appropriate length and duration
5. Output files are uniquely named without conflicts

## Troubleshooting
1. **Model Loading Issues**: Check GPU availability and model paths
2. **Encoding Problems**: Ensure UTF-8 encoding for Thai text
3. **Timing Issues**: Verify audio preprocessing and sample rates
4. **Segmentation Problems**: Adjust text segmenter parameters

## Timeline
1. Script creation: 30 minutes
2. Testing with Jasmali.MP3: 15 minutes
3. Testing with ขนมครก.MP3: 15 minutes
4. Quality assessment: 20 minutes
5. Report generation: 20 minutes

Total estimated time: 1 hour 40 minutes