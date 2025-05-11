# ComfyUI-Audio-Quality-Enhancer

This extension adds advanced audio processing capabilities to ComfyUI with professional-grade audio effects and AI-powered audio enhancement.

![image](https://github.com/user-attachments/assets/7b0f0744-f890-40fd-baeb-8be5a878c750)

#### Use With ACE Step
![image](https://github.com/user-attachments/assets/82f446b3-4c30-4175-8f92-00fb199658ae)



## Features

### 🎛️ AI Audio Effects Node
- **Pitch Shifting**: Adjust pitch from -12 to +12 semitones
- **Speed Adjustment**: Modify playback speed from 0.5x to 2.0x
- **Volume Control**: Professional gain control with anti-clipping protection
- **Audio Normalization**: Automatic level balancing
- **Reverb**: Studio-quality reverb with adjustable room size and amount
- **Echo**: Configurable delay and decay for spatial effects
- **Cross-platform**: Works on Windows, Linux/WSL, and macOS using SoX

### 🔊 AI Audio Enhancer Pro Node
- **Source Separation**: Powered by Demucs to enhance specific audio elements
- **Targeted Enhancement**: Individually process vocals, drums, bass, and other instruments
- **Audio Quality Controls**:
  - Enhancement Level: Master control for overall processing intensity
  - Clarity: Mid-frequency enhancement for improved definition
  - Dynamics: Adjustable compression and transient enhancement
  - Warmth: Low-frequency enhancement for richness
  - Air & Brilliance: High-frequency enhancement for sparkle
  - Dolby-like Stereo Effect: Enhanced stereo imaging
- **Fallback Processing**: Works even without source separation libraries

## Installation

### 1. Install the Extension

Clone this repository into your ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-Audio-Quality-Enhancer.git
```

### 2. Install Required Python Dependencies

```bash
cd ComfyUI-Audio-Quality-Enhancer
pip install -r requirements.txt
```

### 3. Install SoX (Required for Audio Effects)

#### Windows

1. Download SoX for Windows from the [official SourceForge page](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
   - Download the `.exe` installer (e.g., `sox-14.4.2-win32.exe`)

2. Run the installer:
   - Follow the installation prompts
   - **Important**: Note the installation directory (default is usually `C:\Program Files (x86)\sox-14-4-2\`)
   
3. No need to add to PATH - the extension uses the direct path to SoX

#### WSL 2 (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install sox
```

#### macOS

```bash
brew install sox
```

### 4. Optional: Install Advanced Audio Libraries

For full functionality of the Audio Enhancer Pro node, install these additional packages:

```bash
pip install demucs pedalboard
```

These are optional - the node will work without them but with reduced functionality.

### 5. Restart ComfyUI

After installing all required components, restart ComfyUI to load the extension.

## Nodes

### AI Audio Effects

Applies high-quality audio processing to any audio input.

**Inputs:**
- `audio`: Audio data from any audio-generating node
- `pitch_shift`: Semitone adjustment (-12 to +12)
- `speed_factor`: Playback speed modifier (0.5x to 2.0x)
- `sox_path` (optional): Custom path to SoX executable
- `gain_db` (optional): Volume adjustment in decibels
- `use_limiter` (optional): Enable/disable limiter for positive gain
- `normalize_audio` (optional): Enable/disable audio normalization
- `add_reverb` (optional): Enable/disable reverb effect
- `reverb_amount` (optional): Reverb intensity
- `reverb_room_scale` (optional): Size of virtual space
- `add_echo` (optional): Enable/disable echo effect
- `echo_delay` (optional): Time between echo repetitions
- `echo_decay` (optional): How quickly echo fades

**Outputs:**
- `audio`: Processed audio data

### AI Audio Enhancer Pro

Enhances audio quality using source separation and targeted processing.

**Inputs:**
- `audio`: Audio data from any audio-generating node
- `enhancement_level`: Master control for overall enhancement intensity
- `use_source_separation` (optional): Enable/disable Demucs separation
- `demucs_model` (optional): Model choice for source separation
- `device` (optional): Processing device (CUDA/CPU)
- `vocals_enhance` (optional): Vocals enhancement level
- `drums_enhance` (optional): Drums enhancement level
- `bass_enhance` (optional): Bass enhancement level
- `other_enhance` (optional): Other instruments enhancement level
- `clarity` (optional): Mid-frequency clarity enhancement
- `dynamics` (optional): Dynamic range processing
- `warmth` (optional): Low-frequency enhancement
- `air` (optional): High-frequency "air" enhancement
- `dolby_effect` (optional): Stereo width enhancement
- `simple_mode` (optional): Processing mode without source separation
- `apply_limiter` (optional): Final limiter to prevent clipping

**Outputs:**
- `audio`: Enhanced audio data

## Audio Effect Tips

### Volume Control

- **Gain Control**: Use `gain_db` to increase or decrease volume without distortion
  - Positive values (0 to +20 dB): Increase volume with automatic clipping prevention
  - Negative values (-20 to 0 dB): Decrease volume
  - For best results with multiple effects, set gain last in your workflow

- **Normalization**: Enable `normalize_audio` to automatically balance levels
  - Great for ensuring consistent volume across different audio samples
  - Applied before other effects for best results

### Reverb

Reverb adds a sense of space to your audio. Here are some suggested settings:

- **Small Room**: reverb_amount = 20, reverb_room_scale = 25
- **Medium Room**: reverb_amount = 40, reverb_room_scale = 50
- **Large Hall**: reverb_amount = 70, reverb_room_scale = 80
- **Cathedral**: reverb_amount = 90, reverb_room_scale = 95

### Echo

Echo creates repeating sound reflections. Good settings to try:

- **Subtle Echo**: echo_delay = 0.3, echo_decay = 0.3
- **Moderate Echo**: echo_delay = 0.5, echo_decay = 0.5
- **Canyon Echo**: echo_delay = 1.0, echo_decay = 0.7

### Effect Combinations

- **Phone Call**: pitch_shift = 0, speed_factor = 1.0, add_reverb = True, reverb_amount = 10, reverb_room_scale = 10
- **Radio Announcer**: pitch_shift = -2, speed_factor = 0.9, add_reverb = True, reverb_amount = 20, gain_db = 3
- **Stadium Announcement**: pitch_shift = 0, speed_factor = 1.0, add_reverb = True, reverb_amount = 60, add_echo = True, echo_delay = 0.8
- **Child Voice**: pitch_shift = 4, speed_factor = 1.1, gain_db = 2
- **Deep Voice**: pitch_shift = -4, speed_factor = 0.9, gain_db = -2

## Audio Enhancer Tips

### Source Separation Modes

The `use_source_separation` option dramatically changes how the Audio Enhancer Pro works:

- **With Source Separation (Recommended)**: 
  - Individual processing of vocals, drums, bass, and other instruments
  - Best for music and complex audio
  - Requires more processing power and the Demucs library

- **Without Source Separation**:
  - Simpler, frequency-based enhancement
  - Faster processing
  - Works without additional libraries
  - Two processing modes available: "Standard" (gentle) and "Aggressive" (stronger)

### Enhancement Presets

Here are some effective enhancement combinations:

- **Vocal Clarity**: vocals_enhance = 0.8, clarity = 0.6, dynamics = 0.4, air = 0.5
- **Bass Boost**: bass_enhance = 0.9, warmth = 0.7, dynamics = 0.5
- **Full Mix Master**: enhancement_level = 0.6, clarity = 0.5, dynamics = 0.6, warmth = 0.4, air = 0.5
- **Lo-Fi Effect**: enhancement_level = 0.3, warmth = 0.8, air = 0.1, simple_mode = "Aggressive"
- **Podcast Voice**: vocals_enhance = 0.7, clarity = 0.7, dynamics = 0.6, warmth = 0.3

## Usage Examples

### Basic Audio Processing

1. Add any audio-generating node (TTS, audio loader, etc.)
2. Add "AI Audio Effects"
3. Connect the audio output to the effects node input
4. Adjust pitch, speed, reverb, or other settings
5. Connect to "Preview Audio" node to hear the result

### Advanced Audio Enhancement

1. Add any audio-generating node
2. Add "AI Audio Enhancer Pro"
3. Enable source separation for best quality
4. Adjust enhancement parameters for vocals, bass, etc.
5. Connect to "Preview Audio" node

### Combined Processing

For maximum quality, you can chain both nodes:

1. Add any audio-generating node
2. Add "AI Audio Enhancer Pro" for quality enhancement
3. Add "AI Audio Effects" for creative effects
4. Connect in sequence: Audio Source → Enhancer → Effects → Preview
5. Use Enhancer for quality improvement and Effects for creative sound design

## Cross-Platform Compatibility

This extension has been tested and works on:

- Windows 10/11
- Linux (including WSL 2 on Windows)
- macOS

Different environments may require specific setup steps:

### Windows Notes
- SoX is automatically located in standard installation directories
- If installed elsewhere, provide the full path in the effects node
- Performance is best with CUDA-enabled GPUs for the Enhancer node

### WSL 2 Notes
- SoX is automatically located through the system PATH
- Enhancer node works well with CPU mode if CUDA isn't available in WSL

### macOS Notes
- Install SoX via Homebrew for best compatibility
- Enhancer node defaults to CPU mode

## SoX Troubleshooting

### Windows

If you encounter issues with SoX:

1. Verify the SoX path in the "AI Audio Effects" node:
   - Default: `C:\Program Files (x86)\sox-14-4-2\sox.exe`
   - If your installation is in a different location, provide the full path to sox.exe

2. Check if SoX is installed correctly:
   - Open Command Prompt
   - Run `"C:\Program Files (x86)\sox-14-4-2\sox.exe" --version`
   - If you get an error, reinstall SoX

### WSL 2 (Ubuntu)

1. Verify SoX installation:
   ```bash
   sox --version
   ```

2. If SoX is not found, install it:
   ```bash
   sudo apt-get update
   sudo apt-get install sox
   ```

## Enhanced Audio Processing

The AI Audio Enhancer Pro node uses several techniques for high-quality processing:

- **Source Separation**: Uses Demucs to separate audio into stems for targeted processing
- **Transient Enhancement**: Improves attack and clarity of percussion and rhythmic elements
- **Harmonic Processing**: Enhances tonal quality of musical elements
- **Frequency-Specific Processing**: Tailored enhancement for different parts of the spectrum
- **Adaptive Dynamics**: Intelligent compression and expansion based on audio content

## License

This project is provided under the MIT License. See LICENSE file for details.

## Credits

- SoX audio processing library: [SoX - Sound eXchange](http://sox.sourceforge.net/)
- Demucs source separation by [Meta Research](https://github.com/facebookresearch/demucs)
- ComfyUI: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
