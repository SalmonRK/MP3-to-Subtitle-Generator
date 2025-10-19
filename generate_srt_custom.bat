@echo off
chcp 65001 >nul
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

REM Set default values
set INPUT_FILE=audio_file.mp3
set OUTPUT_FILE=audio_file.thai.srt
set SOURCE_LANG=th
set TARGET_LANG=th

REM Check if input file was provided as argument
if not "%1"=="" (
    set INPUT_FILE=%1
    set OUTPUT_FILE=%~n1.thai.srt
)

REM Check if input file exists
if not exist "%INPUT_FILE%" (
    echo Error: Input file %INPUT_FILE% not found.
    echo Please make sure the audio file is in the same directory.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call srt_env\Scripts\activate.bat
set PYTHONIOENCODING=utf-8

echo.
echo Generating SRT subtitles for %INPUT_FILE%...
python audio_to_srt.py "%INPUT_FILE%" -o "%OUTPUT_FILE%" -s %SOURCE_LANG% -t %TARGET_LANG%

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Process completed. Check output\%OUTPUT_FILE% for the generated subtitles.
pause