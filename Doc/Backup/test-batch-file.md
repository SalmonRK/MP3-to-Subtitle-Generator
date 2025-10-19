# Test Batch File Content

## Instructions
Create a file named `test_models.bat` in the SRT-Generator directory with the following content:

```batch
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
```

## How to Use

1. Save the content above as `test_models.bat`
2. Double-click the file to run the tests
3. Or run from command line: `test_models.bat`

## What It Does

1. Checks if the virtual environment exists
2. Activates the srt_env virtual environment
3. Runs the test script `test_mp3_to_subtitle.py`
4. Tests both Whisper Large and Typhoon models on:
   - Jasmali.MP3
   - ขนมครก.MP3
5. Generates SRT files with proper naming
6. Creates a test report (test_report.json)