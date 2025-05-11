"""
Enhanced SoX Audio Quality Effects for ComfyUI
Includes pitch, speed, reverb, echo, and correct gain control
Cross-platform compatible for Windows and Linux/WSL
"""

import os
import tempfile
import numpy as np
import torch
import subprocess
import soundfile as sf
import shutil
import platform

class AudioQualityEffects:
    """
    ComfyUI Node for applying audio effects (pitch, speed, reverb, echo, gain) to TTS output
    Cross-platform compatible for Windows and Linux/WSL environments
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Set default SoX path based on platform
        if platform.system() == 'Windows':
            default_sox_path = "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"
        else:
            # Default for Linux/WSL2
            default_sox_path = "sox"  # Use system PATH on Linux
        
        return {
            "required": {
                "audio": ("AUDIO",),
                "pitch_shift": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, 
                              "display": "slider", "label": "Pitch Shift (semitones)"}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, 
                               "display": "slider", "label": "Speed Factor"})
            },
            "optional": {
                "sox_path": ("STRING", {"default": default_sox_path}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.5, 
                         "display": "slider", "label": "Gain (dB)"}),
                "use_limiter": ("BOOLEAN", {"default": True, "label": "Use Limiter for Gain"}),
                "normalize_audio": ("BOOLEAN", {"default": False, "label": "Normalize Audio"}),
                "add_reverb": ("BOOLEAN", {"default": False, "label": "Add Reverb"}),
                "reverb_amount": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 5, 
                                "display": "slider", "label": "Reverb Amount"}),
                "reverb_room_scale": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 5, 
                                    "display": "slider", "label": "Room Size"}),
                "add_echo": ("BOOLEAN", {"default": False, "label": "Add Echo"}),
                "echo_delay": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, 
                             "display": "slider", "label": "Echo Delay (seconds)"}),
                "echo_decay": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1, 
                             "display": "slider", "label": "Echo Decay"})
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/effects"

    def find_sox_executable(self, provided_path):
        """Find the SoX executable on different platforms"""
        # First, check if the provided path exists
        if os.path.exists(provided_path) and os.access(provided_path, os.X_OK):
            return provided_path
            
        # For Linux/WSL, check if sox is in the PATH
        if platform.system() != "Windows":
            # Try using 'which' command to find sox
            try:
                sox_path = subprocess.check_output(["which", "sox"], text=True).strip()
                if sox_path:
                    print(f"Found SoX at: {sox_path}")
                    return sox_path
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            # Check common Linux locations
            linux_paths = [
                "/usr/bin/sox",
                "/usr/local/bin/sox",
                "/bin/sox"
            ]
            for path in linux_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    print(f"Found SoX at: {path}")
                    return path
                    
            # If sox is not found but is specified as just "sox", return it anyway
            # (it might be in a PATH location we couldn't directly verify)
            if provided_path == "sox":
                return provided_path
        else:
            # Windows paths to check
            windows_paths = [
                "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe",
                "C:\\Program Files\\sox-14-4-2\\sox.exe",
                "C:\\Program Files (x86)\\sox-14.4.2\\sox.exe",
                "C:\\Program Files\\sox-14.4.2\\sox.exe",
                "C:\\Program Files (x86)\\sox\\sox.exe",
                "C:\\Program Files\\sox\\sox.exe"
            ]
            
            for path in windows_paths:
                if os.path.exists(path):
                    print(f"Found SoX at: {path}")
                    return path
                    
        # If we get here, we couldn't find SoX
        return None
    
    def process_audio(self, audio, pitch_shift=0.0, speed_factor=1.0, sox_path=None,
                     gain_db=0.0, use_limiter=True, normalize_audio=False,
                     add_reverb=False, reverb_amount=50, reverb_room_scale=50,
                     add_echo=False, echo_delay=0.5, echo_decay=0.5):
        """Apply audio effects to audio using SoX"""
        if audio is None:
            print("No audio data to process")
            return (None,)
        
        # Check if no effects are needed
        if (abs(pitch_shift) < 0.01 and 
            abs(speed_factor - 1.0) < 0.01 and 
            abs(gain_db) < 0.01 and
            not normalize_audio and
            not add_reverb and 
            not add_echo):
            print("No effects to apply, returning original audio")
            return (audio,)
        
        # If sox_path is None (which shouldn't happen), set platform-specific default
        if sox_path is None:
            if platform.system() == "Windows":
                sox_path = "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"
            else:
                sox_path = "sox"  # Use system PATH on Linux
                
        # Verify and find SoX executable
        print(f"Trying to find SoX at: {sox_path}")
        sox_executable = self.find_sox_executable(sox_path)
        
        if not sox_executable:
            print("SoX executable not found. Please install SoX:")
            if platform.system() == "Windows":
                print("- Windows: Download and install from https://sourceforge.net/projects/sox/")
                print("- Then provide the correct path to sox.exe")
            else:
                print("- Linux/WSL: Run 'sudo apt-get install sox'")
            return (audio,)
        
        print(f"Using SoX executable: {sox_executable}")
        
        try:
            # Extract audio data and sample rate
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy for processing
            # If waveform is [batch, channels, samples]
            if waveform.dim() == 3:
                audio_np = waveform[0, 0].cpu().numpy()
            # If waveform is [channels, samples]
            elif waveform.dim() == 2:
                audio_np = waveform[0].cpu().numpy()
            else:
                audio_np = waveform.cpu().numpy()
            
            print(f"Processing audio with SoX")
            print(f"- Pitch shift: {pitch_shift} semitones")
            print(f"- Speed factor: {speed_factor}x")
            print(f"- Gain: {gain_db} dB")
            print(f"- Normalize: {normalize_audio}")
            if add_reverb:
                print(f"- Reverb: amount={reverb_amount}, room_scale={reverb_room_scale}")
            if add_echo:
                print(f"- Echo: delay={echo_delay}s, decay={echo_decay}")
            
            # Create a temporary directory for this operation
            temp_dir = tempfile.mkdtemp(prefix="orpheus_sox_")
            
            try:
                # Create input and output file paths
                input_path = os.path.join(temp_dir, "input.wav")
                output_path = os.path.join(temp_dir, "output.wav")
                
                print(f"Writing input audio to {input_path}")
                sf.write(input_path, audio_np, sample_rate)
                
                # Verify the input file was written successfully
                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    print("Failed to write input audio file")
                    return (audio,)
                
                # Create the SoX command with explicit parameters
                sox_cmd = [sox_executable, input_path, output_path]
                
                # Add effects in the order they should be applied
                effects = []
                
                # 1. Normalize (if requested) - must be applied first
                if normalize_audio:
                    effects.extend(['gain', '-n'])
                    print("Added normalize effect")
                
                # 2. Gain adjustment (if requested) with clipping prevention
                if abs(gain_db) >= 0.01:
                    if gain_db > 0 and use_limiter:
                        # For positive gain with limiter, use -l option
                        effects.extend(['gain', '-l', str(gain_db)])
                        print(f"Added gain effect with limiter: {gain_db} dB")
                    else:
                        # For negative gain or no limiter
                        effects.extend(['gain', str(gain_db)])
                        print(f"Added gain effect: {gain_db} dB")
                
                # 3. Pitch shift (if requested)
                if pitch_shift != 0:
                    # Convert semitones to cents
                    pitch_cents = int(pitch_shift * 100)
                    effects.extend(['pitch', str(pitch_cents)])
                    print(f"Added pitch effect: {pitch_cents} cents")
                
                # 4. Speed/tempo adjustment (if requested)
                if speed_factor != 1.0:
                    effects.extend(['tempo', '-s', str(speed_factor)])
                    print(f"Added tempo effect: {speed_factor}x")
                
                # 5. Reverb (if requested)
                if add_reverb:
                    # SoX reverb parameters: 
                    # reverberance(0-100) HF-damping(0-100) room-scale(0-100) stereo-depth(0-100) pre-delay(0-500) wet-gain(-10-10)
                    reverberance = int(reverb_amount)
                    hf_damping = 50  # default
                    room_scale = int(reverb_room_scale)
                    stereo_depth = 50  # default
                    pre_delay = 20  # default
                    wet_gain = 0  # default
                    
                    effects.extend([
                        'reverb',
                        str(reverberance),
                        str(hf_damping),
                        str(room_scale),
                        str(stereo_depth),
                        str(pre_delay),
                        str(wet_gain)
                    ])
                    print(f"Added reverb effect: {reverberance}% reverberance, {room_scale}% room scale")
                
                # 6. Echo (if requested)
                if add_echo:
                    # SoX echo parameters: gain-in gain-out delay decay
                    gain_in = 0.8  # Input gain
                    gain_out = 0.9  # Output gain
                    delay_seconds = echo_delay
                    delay_ms = int(delay_seconds * 1000)  # Convert to milliseconds
                    decay = echo_decay
                    
                    effects.extend([
                        'echo',
                        str(gain_in),
                        str(gain_out),
                        str(delay_ms),
                        str(decay)
                    ])
                    print(f"Added echo effect: {delay_ms}ms delay, {decay} decay")
                
                # Add the effects to the command if there are any
                if effects:
                    sox_cmd.extend(effects)
                
                # Print the full SoX command
                print("Executing:", " ".join(sox_cmd))
                
                # Run SoX command and capture output
                process = subprocess.run(
                    sox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Print SoX output for debugging
                if process.stdout and len(process.stdout.strip()) > 0:
                    print("SoX stdout:", process.stdout)
                if process.stderr and len(process.stderr.strip()) > 0:
                    print("SoX stderr:", process.stderr)
                
                # Check if SoX command succeeded
                if process.returncode != 0:
                    print(f"SoX command failed with return code {process.returncode}")
                    return (audio,)
                
                # Check if output file exists and has content
                if not os.path.exists(output_path):
                    print("Output file does not exist")
                    return (audio,)
                
                if os.path.getsize(output_path) == 0:
                    print("Output file is empty")
                    return (audio,)
                
                print(f"Reading processed audio from {output_path}")
                
                # Read processed audio
                processed_audio, new_sample_rate = sf.read(output_path)
                
                # Convert to tensor with proper dimensions
                processed_tensor = torch.tensor(processed_audio.astype(np.float32))
                
                # Ensure proper dimensions [batch, channels, samples]
                if processed_tensor.dim() == 1:
                    processed_tensor = processed_tensor.unsqueeze(0)  # Add channel dim
                
                if processed_tensor.dim() == 2:
                    processed_tensor = processed_tensor.unsqueeze(0)  # Add batch dim
                
                print(f"Processed audio shape: {processed_tensor.shape}")
                
                # Create result
                result_audio = {
                    "waveform": processed_tensor,
                    "sample_rate": new_sample_rate
                }
                
                return (result_audio,)
                
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")
                
        except Exception as e:
            print(f"Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "AudioQualityEffects": AudioQualityEffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioQualityEffects": "AI Audio Effects"
}