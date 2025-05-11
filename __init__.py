"""
ComfyUI Audio Quality Enhancer - Initialization
This module initializes the Audio Quality Enhancer and Audio Effects nodes for ComfyUI.
"""

import os
import sys
import importlib.util

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the module files
audio_enhancer_path = os.path.join(current_dir, "audio_enhancer.py")
audio_effects_path = os.path.join(current_dir, "audio_effects.py")

# Import the modules using importlib
try:
    # Import audio_enhancer.py
    spec_enhancer = importlib.util.spec_from_file_location("audio_enhancer", audio_enhancer_path)
    audio_enhancer = importlib.util.module_from_spec(spec_enhancer)
    spec_enhancer.loader.exec_module(audio_enhancer)
    
    # Import audio_effects.py
    spec_effects = importlib.util.spec_from_file_location("audio_effects", audio_effects_path)
    audio_effects = importlib.util.module_from_spec(spec_effects)
    spec_effects.loader.exec_module(audio_effects)
    
    # Get the classes
    AudioQualityEnhancer = audio_enhancer.AudioQualityEnhancer
    AudioQualityEffects = audio_effects.AudioQualityEffects
    
    # Get the mappings
    enhancer_mappings = audio_enhancer.NODE_CLASS_MAPPINGS
    enhancer_display_mappings = audio_enhancer.NODE_DISPLAY_NAME_MAPPINGS
    effects_mappings = audio_effects.NODE_CLASS_MAPPINGS
    effects_display_mappings = audio_effects.NODE_DISPLAY_NAME_MAPPINGS
    
    # Merge the dictionaries
    NODE_CLASS_MAPPINGS = {**enhancer_mappings, **effects_mappings}
    NODE_DISPLAY_NAME_MAPPINGS = {**enhancer_display_mappings, **effects_display_mappings}
    
    print("ComfyUI Audio Quality Enhancer: Successfully loaded nodes")
    print(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
    
except Exception as e:
    print(f"ComfyUI Audio Quality Enhancer: Error loading nodes: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback to empty mappings if imports failed
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}