#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script to download Whisper models locally for portable use
"""

import os
import sys
from pathlib import Path

try:
    import whisper
    import torch
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages with:")
    print("pip install openai-whisper torch")
    sys.exit(1)

def download_model(model_name, models_dir):
    """
    Download a specific Whisper model to the local models directory
    
    Args:
        model_name (str): Name of the model to download
        models_dir (str): Directory to save the model
    """
    print(f"Downloading {model_name} model...")
    
    try:
        # Create the models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Load the model (this will download it if not present)
        model = whisper.load_model(model_name)
        
        # Save the model to our local directory
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        print(f"Successfully downloaded and saved {model_name} to {model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False

def main():
    """
    Main function to download all required Whisper models
    """
    print("Setting up Whisper models for portable SRT Generator...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    
    # List of models to download (from smallest to largest)
    models_to_download = [
        "tiny",      # ~39MB, fastest, least accurate
        "base",      # ~74MB, good balance
        "small",     # ~244MB, more accurate
        "medium",    # ~769MB, very accurate
        "large"      # ~1550MB, most accurate (used in the script)
    ]
    
    print(f"Models will be saved to: {models_dir}")
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, models_dir):
            success_count += 1
    
    print(f"\nSetup complete! Downloaded {success_count}/{len(models_to_download)} models.")
    print("The SRT Generator can now work without an internet connection.")
    
    # Create a model info file
    info_file = models_dir / "model_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Whisper Models for Portable SRT Generator\n")
        f.write("==========================================\n\n")
        f.write("This directory contains the Whisper models needed for offline operation.\n\n")
        f.write("Available models:\n")
        for model_name in models_to_download:
            model_file = models_dir / f"{model_name}.pt"
            if model_file.exists():
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                f.write(f"- {model_name}.pt ({size_mb:.1f}MB)\n")
            else:
                f.write(f"- {model_name}.pt (not downloaded)\n")
    
    print(f"Model information saved to: {info_file}")

if __name__ == "__main__":
    main()