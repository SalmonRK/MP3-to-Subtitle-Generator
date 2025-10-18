# Audio to SRT Generator (Portable Version)

This tool converts audio files to SRT subtitle format with Thai language support. This portable version includes all Whisper models locally, allowing it to work without an internet connection after initial setup.

## Features

- Converts MP3 and other audio formats to SRT subtitles
- Transcribes audio using OpenAI Whisper models (offline, portable)
- Transcribes text to Thai language
- Generates properly formatted SRT files with timestamps
- **NEW:** Works completely offline after initial model download

## Requirements

- Python 3.6 or higher
- FFmpeg (required by pydub for audio processing)
- Internet connection (only for initial model download)

## Portable Setup

### First Time Setup (with Internet)

1. Install FFmpeg:
   - Download from https://ffmpeg.org/download.html
   - Add FFmpeg to your system PATH

2. Run the installation batch file to set up everything:
   ```
   install.bat
   ```
   
   This will:
   - Create a virtual environment
   - Install all required dependencies
   - Download Whisper models to the `models` folder (this may take a while)
   
3. After installation, run the generation script:
   ```
   generate_srt.bat
   ```
   
   This will process `Jasmali.MP3` and create Thai subtitles

### Portable Usage (without Internet)

After the first-time setup, the entire SRT-Generator folder is completely portable. You can:
- Copy the entire folder to another computer
- Run `generate_srt.bat` without an internet connection
- All models are included locally in the `models` folder

## Usage

### Quick Start (Windows)

First, make sure you've run `install.bat` at least once to set up the environment.

Then run the generation script:
```
generate_srt.bat
```

This will process `Jasmali.MP3` and create `output\Jasmali.thai.srt` with Thai subtitles.

For custom audio files, you can use:
```
generate_srt_custom.bat your_audio_file.mp3
```

### Unicode and Thai Filename Support
The generator now supports Unicode filenames, including Thai characters:
```
generate_srt_custom.bat "ชื่อไฟล์ภาษาไทย.mp3"
```
The output SRT file will be named with the same base name as the input file.

### Manual Usage

Run the Python script directly:
```
python audio_to_srt.py input_file.mp3 -o output_file.srt -s source_lang -t target_lang
```

Note: All output files will be saved in the `output` directory.
```

#### Parameters

- `input_file`: Path to the input audio file (required)
- `-o, --output`: Path to the output SRT file (optional, defaults to input filename with .srt extension)
- `-s, --source`: Source language code (optional, default: "en" for English)
- `-t, --target`: Target language code for translation (optional, default: "th" for Thai)
- `-d, --duration`: Chunk duration in milliseconds (optional, default: 30000)

#### Examples

1. Convert English MP3 to Thai SRT:
   ```
   python audio_to_srt.py "Jasmali.MP3" -o "Jasmali.thai.srt" -s en -t th
   ```
   The output will be saved to `output/Jasmali.thai.srt`

2. Process with different chunk size (for better accuracy with long files):
   ```
   python audio_to_srt.py "audio.mp3" -d 15000
   ```

3. Create subtitles in the same language as the audio:
   ```
   python audio_to_srt.py "thai_audio.mp3" -s th -t th
   ```

## Output

The script generates an SRT file with Thai subtitles that can be used with video players that support subtitle loading.

## Portable Version Details

### Model Information

The portable version includes the following Whisper models in the `models` folder:
- tiny.pt (~39MB) - Fastest, least accurate
- base.pt (~74MB) - Good balance
- small.pt (~244MB) - More accurate
- medium.pt (~769MB) - Very accurate
- large.pt (~1550MB) - Most accurate (used by default)

### Folder Structure

After setup, your SRT-Generator folder will contain:
```
SRT-Generator/
├── models/               # Contains Whisper model files
│   ├── tiny.pt
│   ├── base.pt
│   ├── small.pt
│   ├── medium.pt
│   ├── large.pt
│   └── model_info.txt
├── srt_env/             # Python virtual environment
├── output/              # Generated SRT files
├── audio_to_srt.py      # Main script
├── setup_models.py      # Model download script
├── install.bat          # Installation script
├── generate_srt.bat     # Generation script for Jasmali.MP3
├── generate_srt_custom.bat # Generation script for custom files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Making It Portable

To make the application portable:
1. Run `install.bat` once with an internet connection to download all models
2. The entire folder can now be copied to any computer
3. Run `generate_srt.bat` on the new computer without an internet connection

## Troubleshooting

- If you get an error about FFmpeg, make sure it's installed and added to your system PATH
- For large files, consider reducing the chunk duration with the `-d` parameter
- If the transcription quality is poor, ensure the audio has clear speech and minimal background noise
- If models are not found locally, the script will attempt to download them (requires internet)
- If you're having issues with the portable version, delete the `models` folder and run the setup again

## Supported Languages

The script supports all languages supported by Whisper. Common language codes include:

- English: `en`
- Thai: `th`
- Chinese: `zh`
- Japanese: `ja`
- Korean: `ko`
- Spanish: `es`
- French: `fr`
- German: `de`