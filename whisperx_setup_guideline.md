# WhisperX Transcription Setup Guide

This guide will help you set up WhisperX and run the transcription script for audio processing with word-level timestamps and speaker diarization.

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or newer
- **Operating System**: Windows 10/11 (this guide is Windows-focused)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 5GB free space for models
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for performance)

### Required Software

#### 1. Python
Download and install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation
- Verify installation: `python --version`

#### 2. Visual C++ Redistributable
Required for faster-whisper backend.
- Download from [Microsoft's website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- Install the X64 version

#### 3. Git
Download from [git-scm.com](https://git-scm.com/downloads)

## 🚀 Installation

### Step 1: Create Virtual Environment
```powershell
# Create virtual environment
python -m venv whisperx_env

# Activate virtual environment
.\whisperx_env\Scripts\Activate.ps1
```

You should see `(whisperx_env)` at the beginning of your command prompt.

### Step 2: Install PyTorch
**For NVIDIA GPU users (recommended):**
```powershell
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only installation:**
```powershell
pip install torch torchvision torchaudio
```

### Step 3: Install WhisperX and Dependencies
```powershell
# Install WhisperX from PyPI (recommended)
pip install whisperx

# Install additional required packages
pip install pandas soundfile librosa huggingface_hub
```

### Step 4: Verify Installation
```powershell
# Test if everything is working
python -c "import whisperx; print('WhisperX installed successfully!')"
```

## 🎯 Usage

### Basic Transcription
```powershell
# Transcribe a single audio file
python transcribe_using_whisperx.py "path/to/your/audio.wav"

# Transcribe all audio files in a folder
python transcribe_using_whisperx.py "path/to/audio/folder"
```

### Advanced Options
```powershell
# Use a larger model for better accuracy
python transcribe_using_whisperx.py "audio.wav" --model large-v2

# Specify output directory
python transcribe_using_whisperx.py "audio.wav" --output_dir "my_transcriptions"

# Use CPU instead of GPU
python transcribe_using_whisperx.py "audio.wav" --compute_type float32

# Disable word-level alignment
python transcribe_using_whisperx.py "audio.wav" --no_align

# Enable speaker diarization (requires HF token)
python transcribe_using_whisperx.py "audio.wav" --diarize --hf_token YOUR_TOKEN
```

## 📝 Examples

### Example 1: Basic Transcription
```powershell
# Transcribe a single file with default settings
python transcribe_using_whisperx.py "C:\Users\YourUsername\Desktop\audio\meeting.wav"
```

### Example 2: Batch Processing
```powershell
# Transcribe all audio files in a folder
python transcribe_using_whisperx.py "C:\Users\YourUsername\Desktop\audio_files"
```

### Example 3: High-Quality Transcription
```powershell
# Use large model for best accuracy
python transcribe_using_whisperx.py "important_meeting.wav" --model large-v2 --output_dir "high_quality_transcripts"
```

### Example 4: Speaker Diarization
```powershell
# Enable speaker identification (requires Hugging Face token)
python transcribe_using_whisperx.py "meeting.wav" --diarize --hf_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 🔧 Speaker Diarization Setup (Optional)

If you want to identify different speakers in your audio:

1. **Create Hugging Face Account**
   - Go to [huggingface.co/join](https://huggingface.co/join)
   - Create a free account

2. **Accept Model Agreements**
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept

3. **Get Access Token**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token

4. **Use in Script**
   ```powershell
   python transcribe_using_whisperx.py "audio.wav" --diarize --hf_token YOUR_TOKEN
   ```

## 📊 Output Format

The script generates CSV files with the following columns:

**With alignment (default):**
- `start`: Start time in seconds
- `end`: End time in seconds  
- `word`: Individual word
- `speaker`: Speaker ID (if diarization enabled)

**Without alignment (`--no_align`):**
- `start`: Start time in seconds
- `end`: End time in seconds
- `text`: Full text segment
- `speaker`: Speaker ID (if diarization enabled)

## 🛠️ Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
```powershell
# Reduce batch size
python transcribe_using_whisperx.py "audio.wav" --batch_size 8

# Use CPU instead
python transcribe_using_whisperx.py "audio.wav" --compute_type float32
```

**2. "Audio loading failed"**
- Ensure audio file is not corrupted
- Check file permissions
- Try converting to WAV format first
- The script supports WAV, MP3, M4A, FLAC formats via soundfile and librosa

**3. "Model download failed"**
- Check your internet connection
- Try using a VPN if you're behind a firewall

**4. "Diarization failed"**
- Verify your Hugging Face token is correct
- Ensure you've accepted the model agreements
- Check if the token has the necessary permissions

### Performance Tips

- **GPU Memory**: Use `--compute_type int8` if you have limited GPU memory
- **Batch Size**: Reduce `--batch_size` if you encounter memory issues
- **Model Size**: Use smaller models (`tiny`, `base`, `small`) for faster processing
- **Audio Quality**: Higher quality audio generally produces better results

## ✨ Features

### Supported Audio Formats
- **WAV, MP3, M4A, FLAC** - Supported via soundfile and librosa libraries
- **Automatic resampling** to 16kHz for optimal Whisper performance
- **No external FFmpeg required** - Audio processing handled by Python libraries

### Available Models
- `tiny`: Fastest, least accurate
- `base`: Good balance of speed/accuracy
- `small`: Recommended for most use cases
- `medium`: Higher accuracy, slower
- `large-v2`: Best accuracy, slowest

### Supported Languages
- Auto-detection for most languages
- Manual specification with `--language` parameter

