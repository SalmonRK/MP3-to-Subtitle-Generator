@echo off
echo SRT Generator - Creating Subtitles
echo.

REM Check if virtual environment exists
if not exist "srt_env" (
    echo Error: Virtual environment not found.
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\large.pt" (
    echo Error: Whisper models not found.
    echo Please run install.bat first to download the models.
    echo.
    pause
    exit /b 1
)

REM Check if input file exists
if not exist "Jasmali.MP3" (
    echo Error: Input file Jasmali.MP3 not found.
    echo Please make sure the audio file is in the same directory.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call srt_env\Scripts\activate.bat

echo.
echo Generating SRT subtitles for Jasmali.MP3...
python audio_to_srt.py "Jasmali.MP3" -o "Jasmali.thai.srt" -s en -t th

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Process completed. Check output\Jasmali.thai.srt for the generated subtitles.
pause