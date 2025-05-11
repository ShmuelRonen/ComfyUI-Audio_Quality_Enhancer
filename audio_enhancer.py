"""
ComfyUI Enhanced Audio Quality Node with Demucs Source Separation
================================================================
Processes audio with Demucs source separation and targeted enhancements
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from scipy import signal

# Try to import Demucs
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# Try to import librosa (useful for audio processing)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Try to import pedalboard (optional for effects)
try:
    from pedalboard import Pedalboard, Compressor, LowShelfFilter, HighShelfFilter
    from pedalboard import Gain, LowpassFilter, HighpassFilter, PeakFilter
    from pedalboard import NoiseGate, Limiter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

class AudioQualityEnhancer:
    """
    ComfyUI Node for advanced audio enhancement with Demucs source separation
    """
    @classmethod
    def INPUT_TYPES(cls):
        models = ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"]
        if not DEMUCS_AVAILABLE:
            models = ["Not Available - Install Demucs"]
            
        return {
            "required": {
                "audio": ("AUDIO",),
                "enhancement_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                   "display": "slider", "label": "Enhancement Level"})
            },
            "optional": {
                # Source separation controls
                "use_source_separation": ("BOOLEAN", {"default": True, "label": "Use Source Separation"}),
                "demucs_model": (models, {"default": "htdemucs"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                
                # Individual stem enhancement controls
                "vocals_enhance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                 "display": "slider", "label": "Vocals Enhancement"}),
                "drums_enhance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, 
                               "display": "slider", "label": "Drums Enhancement"}),
                "bass_enhance": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, 
                              "display": "slider", "label": "Bass Enhancement"}),
                "other_enhance": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, 
                               "display": "slider", "label": "Other Enhancement"}),
                
                # General audio enhancement controls
                "clarity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, 
                          "display": "slider", "label": "Clarity"}),
                "dynamics": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, 
                           "display": "slider", "label": "Dynamics"}),
                "warmth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                         "display": "slider", "label": "Warmth"}),
                "air": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                      "display": "slider", "label": "Air & Brilliance"}),
                
                # Stereo enhancement
                "dolby_effect": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                               "display": "slider", "label": "Dolby-like Stereo Effect"}),
                
                # Fallback controls for when source separation isn't available/enabled
                "simple_mode": (["Standard", "Aggressive"], {"default": "Standard"}),
                
                # Output controls
                "apply_limiter": ("BOOLEAN", {"default": True, "label": "Apply Limiter"})
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance_audio"
    CATEGORY = "audio/effects"

    def __init__(self):
        """Initialize the node and load models if available"""
        self.demucs_model = None
        self.source_separation_available = DEMUCS_AVAILABLE
        
        # Try to initialize Demucs if available
        if DEMUCS_AVAILABLE:
            try:
                # Load the model - we'll use htdemucs as default (hybrid transformer Demucs)
                # Only initialize the model when it's needed to save memory
                self.demucs_model_name = "htdemucs"
                print("Demucs source separation available")
            except Exception as e:
                print(f"Failed to initialize Demucs: {e}")
                self.demucs_model = None
    
    def _load_demucs_model(self, model_name="htdemucs", device="cuda"):
        """Load Demucs model on demand"""
        if not DEMUCS_AVAILABLE:
            return None
            
        try:
            # Only load if not already loaded or if model name changed
            if self.demucs_model is None or self.demucs_model_name != model_name:
                print(f"Loading Demucs model: {model_name}")
                self.demucs_model = get_model(model_name)
                self.demucs_model_name = model_name
                
                # Move to specified device
                self.demucs_model.to(device)
                
            return self.demucs_model
        except Exception as e:
            print(f"Error loading Demucs model: {e}")
            return None
    
    def enhance_audio(self, audio: Dict, enhancement_level: float = 0.5,
                     use_source_separation: bool = True, demucs_model: str = "htdemucs",
                     device: str = "cuda",
                     vocals_enhance: float = 0.5, drums_enhance: float = 0.6,
                     bass_enhance: float = 0.4, other_enhance: float = 0.4,
                     clarity: float = 0.4, dynamics: float = 0.3, 
                     warmth: float = 0.2, air: float = 0.3,
                     dolby_effect: float = 0.0,
                     simple_mode: str = "Standard",
                     apply_limiter: bool = True) -> Tuple[Dict]:
        """
        Apply advanced audio enhancement with Demucs source separation
        
        Args:
            audio: Audio dictionary with waveform and sample_rate
            enhancement_level: Master control for overall enhancement
            use_source_separation: Whether to use source separation
            demucs_model: Which Demucs model to use for separation
            device: Which device to use for inference (cuda or cpu)
            vocals_enhance: Strength of vocal enhancement
            drums_enhance: Strength of drum enhancement
            bass_enhance: Strength of bass enhancement
            other_enhance: Strength of other instruments enhancement
            clarity: Mid-frequency clarity enhancement
            dynamics: Dynamic range control
            warmth: Low-frequency enhancement
            air: High-frequency "air" enhancement
            simple_mode: "Standard" or "Aggressive" for fallback mode
            apply_limiter: Whether to apply a limiter to prevent clipping
            
        Returns:
            Enhanced audio dictionary
        """
        if audio is None:
            print("No audio data to process")
            return (None,)
            
        # Extract audio data and sample rate
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Check if no effects are needed
        if enhancement_level < 0.01:
            print("No enhancement to apply, returning original audio")
            return (audio,)
        
        print(f"Enhancing audio with level: {enhancement_level}")
        
        try:
            # Convert to numpy for processing
            # Handle different tensor dimensions
            if waveform.dim() == 3:  # [batch, channels, samples]
                audio_np = waveform[0].cpu().numpy()
                has_batch_dim = True
                num_channels = audio_np.shape[0]
            elif waveform.dim() == 2:  # [channels, samples]
                audio_np = waveform.cpu().numpy()
                has_batch_dim = False
                num_channels = audio_np.shape[0]
            else:  # [samples]
                audio_np = waveform.cpu().numpy()
                has_batch_dim = False
                num_channels = 1
                audio_np = audio_np.reshape(1, -1)  # Add channel dimension
            
            print(f"Processing audio: channels={num_channels}, sample_rate={sample_rate}")
            
            # Scale parameters based on enhancement level
            vocals_enhance *= enhancement_level
            drums_enhance *= enhancement_level
            bass_enhance *= enhancement_level
            other_enhance *= enhancement_level
            clarity *= enhancement_level
            dynamics *= enhancement_level
            warmth *= enhancement_level
            air *= enhancement_level
            
            # Determine whether to use source separation
            can_separate = (self.source_separation_available and 
                           use_source_separation and
                           demucs_model != "Not Available - Install Demucs" and
                           audio_np.shape[1] > sample_rate * 3)  # Minimum 3 seconds for separation
            
            if can_separate:
                print(f"Using Demucs source separation (model: {demucs_model}) for enhanced processing")
                enhanced_audio = self._process_with_demucs(
                    audio_np, 
                    sample_rate,
                    model_name=demucs_model,
                    device=device,
                    vocals_level=vocals_enhance,
                    drums_level=drums_enhance,
                    bass_level=bass_enhance,
                    other_level=other_enhance,
                    clarity=clarity,
                    dynamics=dynamics,
                    warmth=warmth,
                    air=air
                )
            else:
                print("Using fallback processing without source separation")
                # Process each channel independently
                enhanced_channels = []
                for ch in range(num_channels):
                    channel_data = audio_np[ch]
                    enhanced = self._process_without_separation(
                        channel_data,
                        sample_rate,
                        level=enhancement_level,
                        mode=simple_mode,
                        clarity=clarity,
                        dynamics=dynamics,
                        warmth=warmth,
                        air=air
                    )
                    enhanced_channels.append(enhanced)
                
                enhanced_audio = np.stack(enhanced_channels)
            
            # Final processing: apply limiter to avoid clipping if enabled
            if apply_limiter and PEDALBOARD_AVAILABLE:
                for ch in range(enhanced_audio.shape[0]):
                    # Create a simple pedalboard with just a limiter
                    board = Pedalboard([
                        Limiter(
                            threshold_db=-0.5,  # Just below 0dB
                            release_ms=50.0
                        )
                    ])
                    
                    # Normalize if needed
                    max_val = np.max(np.abs(enhanced_audio[ch]))
                    if max_val > 1.0:
                        enhanced_audio[ch] = enhanced_audio[ch] / max_val
                    
                    # Apply limiter
                    enhanced_audio[ch] = board.process(
                        enhanced_audio[ch],
                        sample_rate=sample_rate
                    )
            else:
                # Simple peak normalization without limiter
                max_val = np.max(np.abs(enhanced_audio))
                if max_val > 0.98:
                    enhanced_audio = enhanced_audio * (0.98 / max_val)
            
            # Convert back to tensor with same dimensions as input
            enhanced_tensor = torch.tensor(enhanced_audio.astype(np.float32))
            
            # Add batch dimension if it was present in the input
            if has_batch_dim:
                enhanced_tensor = enhanced_tensor.unsqueeze(0)
            
            # Create result dictionary
            result_audio = {
                "waveform": enhanced_tensor,
                "sample_rate": sample_rate
            }
            
            print(f"Audio enhancement complete. Output shape: {enhanced_tensor.shape}")
            return (result_audio,)
            
        except Exception as e:
            print(f"Error in audio enhancement: {e}")
            import traceback
            traceback.print_exc()
            # Return original audio if any error occurs
            return (audio,)
    
    def _process_with_demucs(self, audio, sample_rate, 
                            model_name="htdemucs", device="cuda",
                            vocals_level=0.5, drums_level=0.6, 
                            bass_level=0.4, other_level=0.4,
                            clarity=0.4, dynamics=0.3,
                            warmth=0.2, air=0.3):
        """
        Process audio using Demucs source separation for targeted enhancements
        """
        try:
            # Load Demucs model
            model = self._load_demucs_model(model_name, device)
            
            if model is None:
                raise Exception("Failed to load Demucs model")
            
            # Prepare audio for Demucs
            # Demucs expects audio as [batch, channels, samples]
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            if len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Move to correct device
            audio_tensor = audio_tensor.to(device)
            
            # Get the model sample rate (usually 44.1kHz for Demucs)
            model_sample_rate = model.samplerate
            
            # Resample audio if necessary
            if sample_rate != model_sample_rate and LIBROSA_AVAILABLE:
                print(f"Resampling audio from {sample_rate}Hz to {model_sample_rate}Hz for Demucs")
                resampled_audio = []
                for ch in range(audio.shape[0]):
                    resampled = librosa.resample(
                        audio[ch], 
                        orig_sr=sample_rate, 
                        target_sr=model_sample_rate
                    )
                    resampled_audio.append(resampled)
                audio_tensor = torch.tensor(np.stack(resampled_audio), dtype=torch.float32).unsqueeze(0)
                audio_tensor = audio_tensor.to(device)
                working_sample_rate = model_sample_rate
            else:
                working_sample_rate = sample_rate
            
            # Apply Demucs separation
            with torch.no_grad():
                sources = apply_model(model, audio_tensor)
            
            # sources is [batch, sources, channels, time]
            # Convert back to numpy arrays
            sources_np = sources.cpu().numpy()[0]  # Remove batch dimension
            
            # Get stem names (depends on the model, but typically: drums, bass, vocals, other)
            stem_names = model.sources
            
            # Create a dictionary to hold our stems
            stems = {}
            for i, name in enumerate(stem_names):
                stems[name] = sources_np[i]
                
                # If needed, resample back to original sample rate
                if working_sample_rate != sample_rate and LIBROSA_AVAILABLE:
                    resampled_stem = []
                    for ch in range(stems[name].shape[0]):
                        resampled = librosa.resample(
                            stems[name][ch], 
                            orig_sr=working_sample_rate, 
                            target_sr=sample_rate
                        )
                        resampled_stem.append(resampled)
                    stems[name] = np.stack(resampled_stem)
            
            # Process each stem with specific enhancements
            enhanced_stems = {}
            
            # Process vocals if present in the model
            if 'vocals' in stems:
                enhanced_stems['vocals'] = self._enhance_vocals(
                    stems['vocals'], 
                    sample_rate, 
                    vocals_level, 
                    clarity, 
                    air
                )
            
            # Process drums if present
            if 'drums' in stems:
                enhanced_stems['drums'] = self._enhance_drums(
                    stems['drums'], 
                    sample_rate, 
                    drums_level, 
                    dynamics, 
                    air
                )
            
            # Process bass if present
            if 'bass' in stems:
                enhanced_stems['bass'] = self._enhance_bass(
                    stems['bass'], 
                    sample_rate, 
                    bass_level, 
                    warmth
                )
            
            # Process other if present
            if 'other' in stems:
                enhanced_stems['other'] = self._enhance_other(
                    stems['other'], 
                    sample_rate, 
                    other_level, 
                    clarity, 
                    warmth, 
                    air
                )
            
            # Mix stems back together with appropriate levels
            result = np.zeros_like(audio)
            for name, stem in enhanced_stems.items():
                # Ensure the stem is the right shape
                if stem.shape[1] > result.shape[1]:
                    stem = stem[:, :result.shape[1]]
                elif stem.shape[1] < result.shape[1]:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, result.shape[1] - stem.shape[1]))
                    stem = np.pad(stem, pad_width, mode='constant')
                
                result += stem
            
            # Normalize to avoid clipping
            max_val = np.max(np.abs(result))
            if max_val > 0.98:
                result = result * (0.98 / max_val)
            
            return result
            
        except Exception as e:
            print(f"Error in Demucs processing: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to simple processing
            return self._process_without_separation(
                audio, 
                sample_rate, 
                level=max(vocals_level, drums_level, bass_level, other_level),
                mode="Standard",
                clarity=clarity,
                dynamics=dynamics,
                warmth=warmth,
                air=air
            )
    
    def _enhance_vocals(self, vocals, sample_rate, level=0.5, clarity=0.4, air=0.3):
        """
        Enhance vocals with focus on clarity and presence
        """
        if level <= 0:
            return vocals
            
        result = vocals.copy()
        
        # Add presence (2-5kHz boost)
        if clarity > 0 and PEDALBOARD_AVAILABLE:
            board = Pedalboard([
                PeakFilter(
                    cutoff_frequency_hz=3500,
                    gain_db=clarity * 6,  # Up to 6dB boost
                    q=1.0
                ),
                # De-ess slightly
                PeakFilter(
                    cutoff_frequency_hz=7500,
                    gain_db=-clarity * 2,  # Cut sibilance
                    q=2.0
                ),
                # Add air
                HighShelfFilter(
                    cutoff_frequency_hz=10000,
                    gain_db=air * 4,  # Up to 4dB boost
                    q=0.7
                )
            ])
            
            for ch in range(result.shape[0]):
                result[ch] = board.process(result[ch], sample_rate=sample_rate)
        
        # Simple alternative without pedalboard
        else:
            # Enhance presence
            nyquist = sample_rate / 2
            presence_center = 3500 / nyquist
            presence_width = 1000 / nyquist
            presence_low = presence_center - presence_width/2
            presence_high = presence_center + presence_width/2
            
            presence_sos = signal.butter(2, [presence_low, presence_high], btype='bandpass', output='sos')
            presence = signal.sosfilt(presence_sos, result)
            
            # Blend
            result = result + (presence * clarity * 0.5)
        
        return result
    
    def _enhance_drums(self, drums, sample_rate, level=0.6, dynamics=0.3, air=0.3):
        """
        Enhance drums with focus on transients and high-end
        """
        if level <= 0:
            return drums
            
        result = drums.copy()
        
        # Transient enhancement
        # Calculate the envelope
        for ch in range(result.shape[0]):
            # Fast envelope follower
            env = np.abs(result[ch])
            
            # Smooth it slightly
            win_size = int(0.01 * sample_rate)  # 10ms window
            if win_size > 1:
                env = np.convolve(env, np.ones(win_size)/win_size, mode='same')
            
            # Calculate the derivative to find transients
            env_diff = np.zeros_like(env)
            env_diff[1:] = env[1:] - env[:-1]
            
            # Create a transient mask
            transient_mask = env_diff > 0.01  # Positive slope above threshold
            
            # Apply transient boost
            result[ch, transient_mask] = result[ch, transient_mask] * (1.0 + level * 0.7)
        
        # Add high-end air to cymbals
        if air > 0 and PEDALBOARD_AVAILABLE:
            board = Pedalboard([
                HighShelfFilter(
                    cutoff_frequency_hz=10000,
                    gain_db=air * 6,  # Up to 6dB boost
                    q=0.7
                )
            ])
            
            for ch in range(result.shape[0]):
                result[ch] = board.process(result[ch], sample_rate=sample_rate)
        
        return result
    
    def _enhance_bass(self, bass, sample_rate, level=0.4, warmth=0.2):
        """
        Enhance bass with focus on warmth and definition
        """
        if level <= 0:
            return bass
            
        result = bass.copy()
        
        # Add harmonics for definition
        if warmth > 0 and PEDALBOARD_AVAILABLE:
            board = Pedalboard([
                LowShelfFilter(
                    cutoff_frequency_hz=100,
                    gain_db=warmth * 4,  # Up to 4dB boost
                    q=0.7
                ),
                # Add some low-mids for definition
                PeakFilter(
                    cutoff_frequency_hz=250,
                    gain_db=level * 3,  # Up to 3dB boost
                    q=1.0
                )
            ])
            
            for ch in range(result.shape[0]):
                result[ch] = board.process(result[ch], sample_rate=sample_rate)
        
        # Simple saturation for definition
        else:
            drive = 1.0 + (level * 2.0)
            result = np.tanh(result * drive) / drive
        
        return result
    
    def _enhance_other(self, other, sample_rate, level=0.4, clarity=0.4, warmth=0.2, air=0.3):
        """
        Enhance other instruments with a balanced approach
        """
        if level <= 0:
            return other
            
        result = other.copy()
        
        # General enhancement with balanced EQ
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([
                # Warmth
                LowShelfFilter(
                    cutoff_frequency_hz=120,
                    gain_db=warmth * 3,  # Up to 3dB boost
                    q=0.7
                ),
                # Clarity
                PeakFilter(
                    cutoff_frequency_hz=2000,
                    gain_db=clarity * 3,  # Up to 3dB boost
                    q=1.0
                ),
                # Air
                HighShelfFilter(
                    cutoff_frequency_hz=8000,
                    gain_db=air * 4,  # Up to 4dB boost
                    q=0.7
                )
            ])
            
            for ch in range(result.shape[0]):
                result[ch] = board.process(result[ch], sample_rate=sample_rate)
        
        return result
    
    def _process_without_separation(self, audio, sample_rate, level=0.5, mode="Standard",
                                  clarity=0.4, dynamics=0.3, warmth=0.2, air=0.3):
        """
        Process audio without source separation - fallback method
        """
        if level <= 0:
            return audio
            
        result = audio.copy()
        
        # Different processing based on mode
        if mode == "Aggressive":
            # More heavy-handed processing for problematic audio
            # Multiband enhancement
            
            # 1. High-end enhancement
            nyquist = sample_rate / 2
            high_cutoff = 6000 / nyquist
            high_sos = signal.butter(2, high_cutoff, btype='highpass', output='sos')
            high_band = signal.sosfilt(high_sos, result)
            
            # Apply some excitement to high frequencies
            high_band = np.tanh(high_band * (1.0 + air * 3.0)) / (1.0 + air)
            
            # 2. Mid enhancement
            mid_low = 500 / nyquist
            mid_high = 6000 / nyquist
            mid_sos = signal.butter(2, [mid_low, mid_high], btype='bandpass', output='sos')
            mid_band = signal.sosfilt(mid_sos, result)
            
            # Add some presence to mids
            mid_band = mid_band * (1.0 + clarity * 0.5)
            
            # 3. Low enhancement
            low_cutoff = 500 / nyquist
            low_sos = signal.butter(2, low_cutoff, btype='lowpass', output='sos')
            low_band = signal.sosfilt(low_sos, result)
            
            # Add some warmth to low end
            low_band = np.tanh(low_band * (1.0 + warmth * 2.0)) / (1.0 + warmth * 0.5)
            
            # Mix bands with scaling to avoid clipping
            result = (low_band + mid_band + high_band) / 2.5
            
            # Transient enhancement
            envelope = np.abs(result)
            envelope_diff = np.zeros_like(envelope)
            envelope_diff[1:] = envelope[1:] - envelope[:-1]
            transient_mask = envelope_diff > 0.01
            result[transient_mask] = result[transient_mask] * (1.0 + dynamics * 0.8)
            
        else:  # Standard mode
            # More gentle, balanced processing
            
            if PEDALBOARD_AVAILABLE:
                # Use pedalboard for more refined processing
                board = Pedalboard([
                    # Add warmth
                    LowShelfFilter(
                        cutoff_frequency_hz=100,
                        gain_db=warmth * 3,
                        q=0.7
                    ),
                    # Add clarity
                    PeakFilter(
                        cutoff_frequency_hz=2500,
                        gain_db=clarity * 4,
                        q=1.0
                    ),
                    # Add air
                    HighShelfFilter(
                        cutoff_frequency_hz=10000,
                        gain_db=air * 5,
                        q=0.7
                    ),
                    # Gentle compression for dynamics
                    Compressor(
                        threshold_db=-20,
                        ratio=1.5 + (dynamics * 1.5),
                        attack_ms=5.0,
                        release_ms=50.0
                    ),
                    # Make up gain
                    Gain(dynamics * 3)
                ])
                
                result = board.process(result, sample_rate=sample_rate)
            else:
                # Simple enhancement without pedalboard
                # High shelf for air
                nyquist = sample_rate / 2
                high_cutoff = 8000 / nyquist
                high_sos = signal.butter(2, high_cutoff, btype='highpass', output='sos')
                high_band = signal.sosfilt(high_sos, result) * (1.0 + air * 2.0)
                
                # Mix in enhanced high frequencies
                result = result * 0.7 + high_band * 0.3
        
        # Final peak normalization
        max_val = np.max(np.abs(result))
        if max_val > 0.98:
            result = result * (0.98 / max_val)
        
        return result

    def _apply_dolby_like_effect(self, audio, sample_rate, amount=0.5):
        """
        Apply a Dolby-like stereo enhancement effect
        
        Args:
            audio: Numpy array of audio data [channels, samples]
            sample_rate: Audio sample rate
            amount: Effect strength (0.0-1.0)
            
        Returns:
            Enhanced audio data
        """
        try:
            # Check if audio is stereo
            if audio.shape[0] < 2:
                # Convert mono to stereo first
                audio = np.vstack([audio, audio])
                
            # Create mid and side signals
            mid = (audio[0] + audio[1]) * 0.5
            side = (audio[0] - audio[1]) * 0.5
            
            # 1. Enhance stereo width with frequency-dependent processing
            nyquist = sample_rate / 2
            
            # Process side channel to enhance stereo width
            # Low-cut filter to ensure bass remains centered (below 150Hz)
            low_cut = 150 / nyquist
            side_sos = signal.butter(2, low_cut, btype='highpass', output='sos')
            filtered_side = signal.sosfilt(side_sos, side)
            
            # Boost upper mids and highs in the side channel (above 2kHz)
            high_boost = 2000 / nyquist
            high_sos = signal.butter(2, high_boost, btype='highpass', output='sos')
            high_side = signal.sosfilt(high_sos, filtered_side)
            
            # Apply mild saturation to add presence
            high_side = np.tanh(high_side * (1.0 + amount * 1.5)) / (1.0 + amount * 0.5)
            
            # Create an enhanced side signal with more width
            enhanced_side = filtered_side + (high_side * amount * 1.5)
            
            # 2. Apply Haas Effect (small delay to one channel) for increased width
            if amount > 0.2:  # Only apply for moderate to high effect amounts
                delay_samples = int(sample_rate * 0.015 * amount)  # Max 15ms delay
                if delay_samples > 0:
                    delay_padding = np.zeros(delay_samples)
                    
                    # Delay side component by adding zeros at the start
                    delayed_side = np.concatenate([delay_padding, enhanced_side[:-delay_samples]])
                    enhanced_side = delayed_side
            
            # 3. Bass Management
            # Filter to get just low frequencies from mid
            bass_cutoff = 150 / nyquist
            bass_sos = signal.butter(2, bass_cutoff, btype='lowpass', output='sos')
            bass = signal.sosfilt(bass_sos, mid)
            
            # Apply mild harmonics to bass (controlled saturation)
            enhanced_bass = np.tanh(bass * (1.0 + amount * 0.8)) / (1.0 + amount * 0.2)
            
            # 4. Presence & Clarity Enhancement in the mid channel
            # Focus on vocal and presence frequencies (1kHz-5kHz)
            presence_low = 1000 / nyquist
            presence_high = 5000 / nyquist
            presence_sos = signal.butter(2, [presence_low, presence_high], btype='bandpass', output='sos')
            presence = signal.sosfilt(presence_sos, mid)
            
            # Blend enhanced presence back into mid
            enhanced_mid = mid + (presence * amount * 0.4)
            
            # 5. Create an "air" band above 10kHz
            if sample_rate > 30000:  # Only if we have enough high frequencies
                air_cutoff = 10000 / nyquist
                air_sos = signal.butter(2, air_cutoff, btype='highpass', output='sos')
                air_band = signal.sosfilt(air_sos, mid)
                
                # Add subtle air to both channels
                enhanced_mid = enhanced_mid + (air_band * amount * 0.5)
            
            # 6. Recombine signals into stereo
            # Increase the side level for more width based on amount
            side_level = 1.0 + (amount * 1.0)  # Up to 2x side boost at max
            
            # Recombine with adjustable mid-side balance
            left = enhanced_mid + (enhanced_side * side_level) + enhanced_bass
            right = enhanced_mid - (enhanced_side * side_level) + enhanced_bass
            
            # 7. Apply final stereo saturation for cohesion if amount is high
            if amount > 0.5:
                saturation = 1.0 + ((amount - 0.5) * 0.6)  # Ranges from 1.0 to 1.3
                left = np.tanh(left * saturation) / saturation
                right = np.tanh(right * saturation) / saturation
            
            # Combine channels and normalize
            result = np.vstack([left, right])
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 0.98:
                result = result * (0.98 / max_val)
                
            return result
            
        except Exception as e:
            print(f"Error in Dolby-like effect: {e}")
            # Return original audio on error
            return audio
    
    def _process_without_separation(self, audio, sample_rate, level=0.5, mode="Standard",
                                  clarity=0.4, dynamics=0.3, warmth=0.2, air=0.3):
        """
        Process audio without source separation - fallback method
        """

# Register nodes
NODE_CLASS_MAPPINGS = {
    "AudioQualityEnhancer": AudioQualityEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioQualityEnhancer": "AI Audio Enhancer Pro"
}