@echo off
echo Portable SRT Generator Installation
echo.

if not exist "models" (
    echo Creating models directory...
    mkdir models
)

if not exist "models\large.pt" (
    echo.
    echo Whisper models not found locally.
    echo Setting up models for portable use...
    echo.
    
    if not exist "srt_env" (
        echo Creating virtual environment...
        python -m venv srt_env
    )
    
    echo.
    echo Activating virtual environment...
    call srt_env\Scripts\activate.bat
    
    echo.
    echo Installing required packages...
    pip install -r requirements.txt
    
    echo.
    echo Downloading Whisper models (this may take a while)...
    python scripts\setup_models.py
    
    echo.
    echo Models downloaded successfully!
) else (
    echo.
    echo Local Whisper models found.
    
    if not exist "srt_env" (
        echo Creating virtual environment...
        python -m venv srt_env
    )
    
    echo.
    echo Activating virtual environment...
    call srt_env\Scripts\activate.bat
    
    echo.
    echo Installing required packages...
    pip install -r requirements.txt
)

echo.
echo Installation completed successfully!
echo You can now run generate_srt.bat to create subtitles.
echo.
pause