**THIS PROJECT NEEDS AN UPDATED READ ME AND AN UPDATED TEST AND MODIFICATION TO ENSURE THE PROJECT WORKS PROPERLY.**

# Audio Transcript Analyzer and Cleaner

A comprehensive audio transcription tool that combines WhisperX for high-quality transcription, speaker diarization for identifying speakers, and intelligent text cleaning to produce readable, professional transcripts.

## Features

- 🎙️ **High-Quality Transcription**: Uses WhisperX with multiple model sizes (tiny to large-v2)
- 👥 **Speaker Identification**: Identifies different speakers in the audio (requires Hugging Face token)
- 🧹 **Intelligent Text Cleaning**: Removes fillers, fixes stutters, corrects errors, and improves punctuation
- 📦 **Long Audio Support**: Automatically chunks long audio files for processing
- 🎵 **Multiple Audio Formats**: Supports MP3, WAV, M4A, FLAC, OGG, WMA, AAC
- 📊 **Multiple Output Formats**: Save as TXT, JSON, or CSV
- ⚡ **GPU/CPU Support**: Automatically uses GPU if available, falls back to CPU

## Installation

### Prerequisites

- Python 3.8 or newer
- For GPU support: NVIDIA GPU with CUDA (optional but recommended)

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (NVIDIA), install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only installation:
pip install torch torchvision torchaudio
```

## Usage

### Basic Transcription

```bash
# Transcribe an audio file
python Cleaner.py audio.mp3

# This will create audio_transcript.txt with the cleaned transcript
```

### With Speaker Identification

To identify different speakers, you need a Hugging Face token:

1. **Create a Hugging Face account** at [huggingface.co/join](https://huggingface.co/join)

2. **Accept model agreements**:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept

3. **Get your token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

4. **Use with diarization**:
```bash
python Cleaner.py audio.mp3 --diarize --hf-token YOUR_TOKEN_HERE
```

### Advanced Options

```bash
# Use a specific model (faster but less accurate)
python Cleaner.py audio.wav --model small

# Specify language (auto-detected if not specified)
python Cleaner.py audio.mp3 --language en

# Save as JSON with timestamps
python Cleaner.py audio.wav --output transcript.json --format json

# Save as CSV
python Cleaner.py audio.mp3 --format csv

# Process long audio files with custom chunk duration (in seconds)
python Cleaner.py long_audio.mp3 --chunk-duration 600

# Disable word-level alignment (faster processing)
python Cleaner.py audio.mp3 --no-alignment

# Force CPU usage
python Cleaner.py audio.mp3 --device cpu

# Use different compute type for GPU memory optimization
python Cleaner.py audio.mp3 --compute-type float16
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `audio_file` | Path to audio file (required) | - |
| `--model` | WhisperX model: tiny, base, small, medium, large-v2 | large-v2 |
| `--language` | Language code (e.g., 'en', 'es', 'fr') | Auto-detect |
| `--output`, `-o` | Output file path | `<audio_name>_transcript.txt` |
| `--format` | Output format: txt, json, csv | txt |
| `--chunk-duration` | Chunk duration in seconds for long files | 300 |
| `--diarize` | Enable speaker identification | False |
| `--hf-token` | Hugging Face token for diarization | None |
| `--no-alignment` | Disable word-level timestamps | False |
| `--device` | Device: cuda or cpu | Auto-detect |
| `--compute-type` | Compute type: float32, float16, int8 | float32 |

## Output Formats

### TXT Format
Plain text with speaker labels (if diarization enabled) and cleanup statistics.

### JSON Format
Structured data with timestamps, speaker information, and statistics:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is a test.",
      "speaker": "SPEAKER_00",
      "start_time": "0:00:00",
      "end_time": "0:00:05"
    }
  ],
  "stats": { ... }
}
```

### CSV Format
Spreadsheet-friendly format with columns:
- Start Time, End Time
- Start Seconds, End Seconds
- Speaker
- Transcript

## Text Cleaning Features

The transcript cleaner automatically:

- ✅ Removes filler words (um, uh, like, you know, etc.)
- ✅ Fixes stutters and repetitions
- ✅ Corrects common transcription errors
- ✅ Improves punctuation and capitalization
- ✅ Removes incomplete sentences
- ✅ Normalizes quotes and spacing

## Examples

### Example 1: Basic Transcription
```bash
python Cleaner.py meeting_recording.mp3
```

### Example 2: High-Quality with Speakers
```bash
python Cleaner.py interview.wav --model large-v2 --diarize --hf-token hf_xxxxxxxxxxxx --format json
```

### Example 3: Batch Processing Script
```bash
for file in *.mp3; do
    python Cleaner.py "$file" --output "${file%.mp3}_transcript.txt"
done
```

## Troubleshooting

### CUDA Out of Memory
- Use a smaller model: `--model small`
- Use CPU: `--device cpu`
- Use lower precision: `--compute-type float16` or `--compute-type int8`
- Increase chunk duration: `--chunk-duration 600`

### Audio Loading Failed
- Ensure the audio file is not corrupted
- Try converting to WAV format first
- Check file permissions

### Speaker Diarization Failed
- Verify your Hugging Face token is correct
- Ensure you've accepted the model agreements
- Check token permissions

### Model Download Issues
- Check your internet connection
- Try using a VPN if behind a firewall
- Models are cached after first download

## Performance Tips

- **GPU**: Use GPU for 5-10x faster processing
- **Model Size**: 
  - `tiny`: Fastest, least accurate
  - `base`: Good balance
  - `small`: Recommended for most cases
  - `medium`: Higher accuracy
  - `large-v2`: Best accuracy, slowest
- **Chunk Duration**: Larger chunks (600s) are faster but use more memory
- **Alignment**: Disable with `--no-alignment` for faster processing if word-level timestamps aren't needed

## License

This project uses WhisperX and other open-source libraries. Please refer to their respective licenses.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure your audio file is in a supported format

