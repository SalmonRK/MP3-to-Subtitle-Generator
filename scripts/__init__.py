#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Model implementations for SRT Generator
"""

from .base_asr import ASRModel
from .whisper_model import WhisperModel
from .typhoon_model import TyphoonModel
from .typhoon_nemo_model import TyphoonNemoModel

__all__ = [
    'ASRModel',
    'WhisperModel',
    'TyphoonModel',
    'TyphoonNemoModel'
]

def get_model(model_type: str, **kwargs):
    """
    Factory function to get ASR model instance
    
    Args:
        model_type: Type of model ("whisper", "typhoon", or "typhoon_nemo")
        **kwargs: Additional arguments for model initialization
        
    Returns:
        ASRModel instance
    """
    if model_type.lower() == "whisper":
        return WhisperModel(**kwargs)
    elif model_type.lower() == "typhoon":
        return TyphoonModel(**kwargs)
    elif model_type.lower() in ["typhoon_nemo", "typhoon-nemo"]:
        return TyphoonNemoModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")