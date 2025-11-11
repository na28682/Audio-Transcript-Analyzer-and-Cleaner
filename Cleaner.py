#!/usr/bin/env python3
"""
Audio Transcript Analyzer and Cleaner with WhisperX Integration

This script processes audio files (MP3, WAV, M4A, FLAC, etc.), transcribes them using WhisperX,
identifies speakers, and cleans the transcripts by removing noise, fillers, and errors.
"""

import re
import string
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import os
from datetime import timedelta
import torch
import numpy as np
from tqdm import tqdm

# WhisperX imports
try:
    import whisperx
    import pandas as pd
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: WhisperX not available. Install with: pip install whisperx pandas")

# Note: WhisperX handles audio loading internally, no need for pydub

@dataclass
class CleanupStats:
    """Statistics about the cleaning process"""
    original_length: int
    cleaned_length: int
    removed_fillers: int
    removed_repetitions: int
    fixed_punctuation: int
    sentences_cleaned: int

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed audio with speaker information"""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[Dict]] = None

class WhisperXProcessor:
    """Class for processing audio files using WhisperX"""
    
    def __init__(self, model_name: str = "large-v2", device: Optional[str] = None, 
                 compute_type: str = "float32", hf_token: Optional[str] = None):
        if not WHISPERX_AVAILABLE:
            raise ImportError("WhisperX is required. Install with: pip install whisperx pandas")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        self.model_name = model_name
        self.model = None
        self.align_model = None
        self.diarize_model = None
        
        # Supported audio formats
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac']
        
        print(f"Initializing WhisperX with model: {model_name}")
        print(f"Using device: {device}")
        
    def load_model(self):
        """Load the WhisperX model"""
        if self.model is None:
            try:
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                self.model = whisperx.load_model(
                    self.model_name, 
                    device=self.device,
                    compute_type=self.compute_type,
                    vad_options={"use_vad": True}
                )
                print(f"✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                if "Unauthorized" in str(e) or "Invalid credentials" in str(e):
                    print("\nAuthentication error with Hugging Face Hub.")
                    print("You may need to login: huggingface-cli login")
                raise
    
    def load_alignment_model(self, language: str):
        """Load alignment model for word-level timestamps"""
        if self.align_model is None:
            try:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=language, 
                    device=self.device
                )
                print(f"✅ Alignment model loaded for language: {language}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load alignment model: {str(e)}")
                self.align_model = None
                self.metadata = None
    
    def load_diarization_model(self):
        """Load speaker diarization model"""
        if self.diarize_model is None and self.hf_token:
            try:
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
                print("✅ Speaker diarization model loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load diarization model: {str(e)}")
                print("   Speaker identification will be disabled.")
                self.diarize_model = None
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None, 
                        chunk_duration: int = 300, enable_diarization: bool = False,
                        enable_alignment: bool = True) -> List[TranscriptionSegment]:
        """
        Transcribe audio file with optional speaker diarization and word-level alignment
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en'). If None, will auto-detect
            chunk_duration: Duration of chunks in seconds for processing long files
            enable_diarization: Whether to identify speakers
            enable_alignment: Whether to get word-level timestamps
            
        Returns:
            List of TranscriptionSegment objects
        """
        self.load_model()
        
        # Load audio
        print(f"\n🎧 Loading audio: {os.path.basename(audio_path)}")
        audio = whisperx.load_audio(audio_path)
        sample_rate = 16000
        
        # Auto-detect language if not provided
        if language is None:
            print("🔍 Detecting language...")
            # Transcribe a small sample for language detection
            sample_length = min(len(audio), sample_rate * 30)  # 30 seconds max
            audio_sample = audio[:sample_length]
            language_result = self.model.transcribe(audio_sample, language=None)
            language = language_result.get("language", "en")
            print(f"✅ Detected language: {language}")
        
        # Process in chunks for long audio files
        total_duration = len(audio) / sample_rate
        num_chunks = int(np.ceil(total_duration / chunk_duration))
        
        all_segments = []
        
        if num_chunks > 1:
            print(f"📦 Processing {num_chunks} chunks ({chunk_duration}s each)...")
            
            for i in tqdm(range(num_chunks), desc="Transcribing chunks"):
                chunk_start_sec = i * chunk_duration
                chunk_end_sec = min((i + 1) * chunk_duration, total_duration)
                
                start_idx = int(chunk_start_sec * sample_rate)
                end_idx = int(chunk_end_sec * sample_rate)
                
                audio_chunk = audio[start_idx:end_idx]
                
                # Transcribe chunk
                result = self.model.transcribe(audio_chunk, language=language, batch_size=16)
                
                # Align if enabled
                if enable_alignment:
                    if self.align_model is None:
                        self.load_alignment_model(language)
                    if self.align_model is not None and self.metadata is not None:
                        try:
                            result = whisperx.align(
                                result["segments"], 
                                self.align_model, 
                                self.metadata, 
                                audio_chunk, 
                                self.device, 
                                return_char_alignments=False
                            )
                        except Exception as e:
                            print(f"⚠️  Warning: Alignment failed for chunk {i+1}: {str(e)}")
                
                # Adjust timestamps for chunk position
                for segment in result.get("segments", []):
                    adjusted_start = chunk_start_sec + segment.get("start", 0)
                    adjusted_end = chunk_start_sec + segment.get("end", 0)
                    
                    all_segments.append({
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "text": segment.get("text", "").strip(),
                        "words": segment.get("words", [])
                    })
        else:
            # Process entire file at once
            print("📝 Transcribing audio...")
            result = self.model.transcribe(audio, language=language, batch_size=16)
            
            # Align if enabled
            if enable_alignment:
                if self.align_model is None:
                    self.load_alignment_model(language)
                if self.align_model is not None and self.metadata is not None:
                    try:
                        result = whisperx.align(
                            result["segments"], 
                            self.align_model, 
                            self.metadata, 
                            audio, 
                            self.device, 
                            return_char_alignments=False
                        )
                    except Exception as e:
                        print(f"⚠️  Warning: Alignment failed: {str(e)}")
            
            for segment in result.get("segments", []):
                all_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "words": segment.get("words", [])
                })
        
        # Speaker diarization
        if enable_diarization and self.hf_token:
            print("👥 Identifying speakers...")
            self.load_diarization_model()
            
            if self.diarize_model is not None:
                try:
                    # Diarization needs the full audio file path
                    diarize_segments = self.diarize_model(audio_path)
                    # Assign speakers to transcription segments
                    all_segments = whisperx.assign_word_speakers(diarize_segments, all_segments)
                    print("✅ Speaker identification complete")
                except Exception as e:
                    print(f"⚠️  Warning: Speaker diarization failed: {str(e)}")
                    print("   Continuing without speaker identification...")
        
        # Convert to TranscriptionSegment objects
        segments = []
        for seg in all_segments:
            # Skip empty segments
            text = seg.get("text", "").strip()
            if not text:
                continue
                
            speaker = seg.get("speaker", None)
            if speaker:
                # Handle both string and numeric speaker IDs
                if isinstance(speaker, (int, float)):
                    speaker = f"SPEAKER_{int(speaker):02d}"
                else:
                    speaker = f"SPEAKER_{speaker}" if not speaker.startswith("SPEAKER_") else speaker
            
            segments.append(TranscriptionSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=text,
                speaker=speaker,
                words=seg.get("words", [])
            ))
        
        return segments

class TranscriptAnalyzer:
    """Main class for analyzing and cleaning audio transcripts"""
    
    def __init__(self, whisperx_processor: Optional[WhisperXProcessor] = None):
        self.whisperx_processor = whisperx_processor
        
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'mm', 'hmm', 'like', 'you know', 
            'so', 'well', 'actually', 'basically', 'literally',
            'right', 'okay', 'ok', 'yeah', 'yes', 'no', 'sure'
        }
        
        self.stutter_patterns = [
            r'\b(\w+)\s+\1\b',  
            r'\b(\w+)-(\w+)\b',  
            r'\b(\w+)\s*\.\s*\1\b',  
        ]
        
        self.error_corrections = {
            'there': ['their', 'theyre'],
            'you': ['u', 'ya'],
            'are': ['r'],
            'your': ['ur', 'youre'],
            'because': ['cuz', 'cause', 'bc'],
            'with': ['w/'],
            'without': ['w/o'],
            'and': ['&', 'n'],
            'or': ['/'],
            'to': ['2'],
            'for': ['4'],
            'be': ['b'],
            'see': ['c'],
            'before': ['b4'],
            'tonight': ['2nite'],
            'today': ['2day'],
            'tomorrow': ['2morrow'],
        }
        
        self.incomplete_indicators = [
            r'\b(and|but|so|because|if|when|where|how|what|why|who)\s*$',
            r'\b(the|a|an|this|that|these|those)\s*$',
            r'\b(is|are|was|were|will|would|could|should|can|may)\s*$',
        ]

    def clean_text(self, text: str) -> Tuple[str, CleanupStats]:
        """Clean and improve transcript text quality"""
        original_length = len(text)
        stats = CleanupStats(
            original_length=original_length,
            cleaned_length=0,
            removed_fillers=0,
            removed_repetitions=0,
            fixed_punctuation=0,
            sentences_cleaned=0
        )
        
        cleaned_text = self._normalize_text(text)
        cleaned_text, filler_count = self._remove_fillers(cleaned_text)
        stats.removed_fillers = filler_count
        cleaned_text, repetition_count = self._fix_stutters(cleaned_text)
        stats.removed_repetitions = repetition_count
        cleaned_text = self._fix_transcription_errors(cleaned_text)
        cleaned_text, punct_count = self._fix_punctuation(cleaned_text)
        stats.fixed_punctuation = punct_count
        cleaned_text, sentence_count = self._remove_incomplete_sentences(cleaned_text)
        stats.sentences_cleaned = sentence_count
        cleaned_text = self._final_cleanup(cleaned_text)
        stats.cleaned_length = len(cleaned_text)
        return cleaned_text, stats

    def process_audio_file(self, audio_file_path: str, language: Optional[str] = None,
                          chunk_duration: int = 300, enable_diarization: bool = False,
                          enable_alignment: bool = True) -> Tuple[List[TranscriptionSegment], CleanupStats]:
        """
        Process audio file: transcribe, identify speakers, and clean transcript
        
        Returns:
            Tuple of (list of cleaned segments, overall cleanup stats)
        """
        if self.whisperx_processor is None:
            raise RuntimeError("WhisperX processor not initialized")
        
        # Transcribe audio
        segments = self.whisperx_processor.transcribe_audio(
            audio_file_path,
            language=language,
            chunk_duration=chunk_duration,
            enable_diarization=enable_diarization,
            enable_alignment=enable_alignment
        )
        
        # Clean each segment
        total_stats = CleanupStats(0, 0, 0, 0, 0, 0)
        
        for segment in segments:
            if not segment.text.strip():
                continue
                
            cleaned_text, stats = self.clean_text(segment.text)
            segment.text = cleaned_text
            
            # Aggregate stats
            total_stats.original_length += stats.original_length
            total_stats.cleaned_length += stats.cleaned_length
            total_stats.removed_fillers += stats.removed_fillers
            total_stats.removed_repetitions += stats.removed_repetitions
            total_stats.fixed_punctuation += stats.fixed_punctuation
            total_stats.sentences_cleaned += stats.sentences_cleaned
        
        # Filter out empty segments after cleaning
        segments = [seg for seg in segments if seg.text.strip()]
        
        return segments, total_stats

    def _normalize_text(self, text: str) -> str:
        """Basic text normalization"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'["""]', '"', text)  # Normalize double quotes
        text = re.sub(r"[''`]", "'", text)  # Normalize single quotes
        return text.strip()

    def _remove_fillers(self, text: str) -> Tuple[str, int]:
        """Remove filler words and phrases"""
        words = text.split()
        filtered_words = []
        removed_count = 0
        
        for word in words:
            word_lower = word.lower().strip(string.punctuation)
            if word_lower not in self.filler_words:
                filtered_words.append(word)
            else:
                removed_count += 1
        
        return ' '.join(filtered_words), removed_count

    def _fix_stutters(self, text: str) -> Tuple[str, int]:
        """Fix stuttering and repetitions"""
        fixed_count = 0
        
        # Count and fix duplicate words
        duplicate_pattern = r'\b(\w+)\s+\1\b'
        matches = re.findall(duplicate_pattern, text, flags=re.IGNORECASE)
        if matches:
            fixed_count += len(matches)
            text = re.sub(duplicate_pattern, r'\1', text, flags=re.IGNORECASE)
        
        # Fix hyphenated stutters (e.g., "the-the")
        hyphen_pattern = r'\b(\w+)-\1\b'
        matches = re.findall(hyphen_pattern, text, flags=re.IGNORECASE)
        if matches:
            fixed_count += len(matches)
            text = re.sub(hyphen_pattern, r'\1', text, flags=re.IGNORECASE)
        
        return text, fixed_count

    def _fix_transcription_errors(self, text: str) -> str:
        """Fix common transcription errors"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower().strip(string.punctuation)
            corrected = False
            
            for correct_word, errors in self.error_corrections.items():
                if word_lower in errors:
                    # Preserve capitalization
                    if word[0].isupper():
                        corrected_words.append(correct_word.capitalize())
                    else:
                        corrected_words.append(correct_word)
                    corrected = True
                    break
            
            if not corrected:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def _fix_punctuation(self, text: str) -> Tuple[str, int]:
        """Fix punctuation issues"""
        fixed_count = 0
        
        # Fix missing spaces after punctuation
        matches_before = len(re.findall(r'([.!?])([A-Za-z])', text))
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        fixed_count += matches_before
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix capitalization at sentence start
        sentences = re.split(r'([.!?]\s+)', text)
        fixed_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence and sentence.strip():
                # Check if this should be capitalized
                if i == 0 or (i > 0 and sentences[i-1].strip().endswith(('.', '!', '?'))):
                    if sentence[0].isalpha() and sentence[0].islower():
                        sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                        fixed_count += 1
            fixed_sentences.append(sentence)
        
        text = ''.join(fixed_sentences)
        return text, fixed_count

    def _remove_incomplete_sentences(self, text: str) -> Tuple[str, int]:
        """Remove incomplete sentences"""
        sentences = re.split(r'([.!?]\s+)', text)
        complete_sentences = []
        removed_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            is_incomplete = False
            for pattern in self.incomplete_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_incomplete = True
                    break
            
            if not is_incomplete:
                complete_sentences.append(sentence)
            else:
                removed_count += 1
        
        return ''.join(complete_sentences), removed_count

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)
        return text

def format_timestamp(seconds: float) -> str:
    """Format seconds to readable timestamp"""
    return str(timedelta(seconds=int(seconds)))

def save_transcript(segments: List[TranscriptionSegment], output_path: str, 
                   format: str = "txt", stats: Optional[CleanupStats] = None):
    """Save transcript to file in various formats"""
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                if segment.speaker:
                    f.write(f"[{segment.speaker}] ")
                f.write(f"{segment.text}\n\n")
            
            if stats:
                f.write("\n" + "="*50 + "\n")
                f.write("CLEANUP STATISTICS\n")
                f.write("="*50 + "\n")
                f.write(f"Original length: {stats.original_length} characters\n")
                f.write(f"Cleaned length: {stats.cleaned_length} characters\n")
                f.write(f"Removed fillers: {stats.removed_fillers}\n")
                f.write(f"Fixed repetitions: {stats.removed_repetitions}\n")
                f.write(f"Fixed punctuation: {stats.fixed_punctuation}\n")
                f.write(f"Removed incomplete sentences: {stats.sentences_cleaned}\n")
    
    elif format == "json":
        data = {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "start_time": format_timestamp(seg.start),
                    "end_time": format_timestamp(seg.end)
                }
                for seg in segments
            ],
            "stats": {
                "original_length": stats.original_length if stats else 0,
                "cleaned_length": stats.cleaned_length if stats else 0,
                "removed_fillers": stats.removed_fillers if stats else 0,
                "removed_repetitions": stats.removed_repetitions if stats else 0,
                "fixed_punctuation": stats.fixed_punctuation if stats else 0,
                "sentences_cleaned": stats.sentences_cleaned if stats else 0
            } if stats else None
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        if not WHISPERX_AVAILABLE:
            raise ImportError("CSV format requires pandas. Install with: pip install pandas")
        data = []
        for seg in segments:
            data.append({
                "Start Time": format_timestamp(seg.start),
                "End Time": format_timestamp(seg.end),
                "Start Seconds": seg.start,
                "End Seconds": seg.end,
                "Speaker": seg.speaker or "UNKNOWN",
                "Transcript": seg.text
            })
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    print(f"✅ Transcript saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and clean audio files using WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python Cleaner.py audio.mp3
  
  # With speaker identification
  python Cleaner.py audio.mp3 --diarize --hf_token YOUR_TOKEN
  
  # Specify model and output format
  python Cleaner.py audio.wav --model large-v2 --output transcript.json --format json
  
  # Process with custom chunk duration
  python Cleaner.py long_audio.mp3 --chunk-duration 600
        """
    )
    
    parser.add_argument("audio_file", help="Path to audio file (MP3, WAV, M4A, FLAC, etc.)")
    parser.add_argument("--model", default="large-v2", 
                       choices=["tiny", "base", "small", "medium", "large-v2"],
                       help="WhisperX model to use (default: large-v2)")
    parser.add_argument("--language", default=None,
                       help="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not specified")
    parser.add_argument("--output", "-o", default=None,
                       help="Output file path (default: <audio_name>_transcript.txt)")
    parser.add_argument("--format", choices=["txt", "json", "csv"], default="txt",
                       help="Output format (default: txt)")
    parser.add_argument("--chunk-duration", type=int, default=300,
                       help="Chunk duration in seconds for long audio files (default: 300)")
    parser.add_argument("--diarize", action="store_true",
                       help="Enable speaker diarization (requires --hf-token)")
    parser.add_argument("--hf-token", default=None,
                       help="Hugging Face token for speaker diarization")
    parser.add_argument("--no-alignment", action="store_true",
                       help="Disable word-level alignment")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--compute-type", default="float32",
                       choices=["float32", "float16", "int8"],
                       help="Compute type (default: float32)")
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Check file extension (warning only, WhisperX may still support it)
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac']
    file_ext = os.path.splitext(args.audio_file)[1].lower()
    if file_ext and file_ext not in valid_extensions:
        print(f"⚠️  Warning: File extension '{file_ext}' may not be supported.")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        print("   Attempting to process anyway...")
    
    # Check if diarization is requested without token
    if args.diarize and not args.hf_token:
        print("⚠️  Warning: --diarize requires --hf-token. Speaker identification disabled.")
        args.diarize = False
    
    # Initialize WhisperX processor
    try:
        processor = WhisperXProcessor(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            hf_token=args.hf_token
        )
    except Exception as e:
        print(f"❌ Error initializing WhisperX: {str(e)}")
        return 1
    
    # Initialize analyzer
    analyzer = TranscriptAnalyzer(whisperx_processor=processor)
    
    # Process audio
    try:
        print("\n" + "="*60)
        print("🎙️  AUDIO TRANSCRIPTION AND CLEANING")
        print("="*60)
        
        segments, stats = analyzer.process_audio_file(
            args.audio_file,
            language=args.language,
            chunk_duration=args.chunk_duration,
            enable_diarization=args.diarize,
            enable_alignment=not args.no_alignment
        )
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
            ext = {"txt": ".txt", "json": ".json", "csv": ".csv"}[args.format]
            output_path = f"{base_name}_transcript{ext}"
        
        # Save transcript
        save_transcript(segments, output_path, format=args.format, stats=stats)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"Total segments: {len(segments)}")
        if stats:
            print(f"Original text length: {stats.original_length} characters")
            print(f"Cleaned text length: {stats.cleaned_length} characters")
            print(f"Removed fillers: {stats.removed_fillers}")
            print(f"Fixed repetitions: {stats.removed_repetitions}")
            print(f"Fixed punctuation: {stats.fixed_punctuation}")
            print(f"Removed incomplete sentences: {stats.sentences_cleaned}")
        
        speakers = set(seg.speaker for seg in segments if seg.speaker)
        if speakers:
            print(f"Identified speakers: {len(speakers)}")
            for speaker in sorted(speakers):
                print(f"  - {speaker}")
        
        print(f"\n✅ Processing complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
