@echo off
chcp 65001 >nul
echo Testing MP3 to Subtitle Conversion
echo =====================================
echo.

REM Check if virtual environment exists
if not exist "srt_env" (
    echo Error: Virtual environment not found.
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Check if test script exists
if not exist "test_mp3_to_subtitle.py" (
    echo Error: Test script not found.
    echo Please create test_mp3_to_subtitle.py first.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call srt_env\Scripts\activate.bat
set PYTHONIOENCODING=utf-8

echo.
echo Running tests for Whisper Large and Typhoon models...
echo This will test both Jasmali.MP3 and ขนมครก.MP3
echo.

python test_mp3_to_subtitle.py

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Test completed!
echo Check the output/ directory for generated SRT files
echo Check test_report.json for detailed results
echo.
pause