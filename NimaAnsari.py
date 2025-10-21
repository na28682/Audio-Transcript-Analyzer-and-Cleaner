#!/usr/bin/env python3
"""
Audio Transcript Analyzer and Cleaner with Audio Import

This script analyzes audio transcripts and removes abnormal behavior,
errors, and improves overall text quality. Supports direct audio file processing.
"""

import re
import string
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import os
import tempfile

# Audio processing imports (optional)
try:
    import speech_recognition as sr
    import pydub
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio processing not available. Install speech_recognition and pydub for audio support.")

@dataclass
class CleanupStats:
    """Statistics about the cleaning process"""
    original_length: int
    cleaned_length: int
    removed_fillers: int
    removed_repetitions: int
    fixed_punctuation: int
    sentences_cleaned: int

class AudioProcessor:
    """Class for processing audio files and converting to text"""
    
    def __init__(self):
        if not AUDIO_AVAILABLE:
            raise ImportError("Audio processing requires speech_recognition and pydub packages")
        
        self.recognizer = sr.Recognizer()
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
    
    def convert_audio_to_text(self, audio_file_path: str, language: str = 'en-US') -> str:
        try:
            audio = AudioSegment.from_file(audio_file_path)
            
            if not audio_file_path.lower().endswith('.wav'):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    audio.export(temp_file.name, format='wav')
                    temp_path = temp_file.name
            else:
                temp_path = audio_file_path
            
            with sr.AudioFile(temp_path) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio_data, language=language)
            
            if temp_path != audio_file_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return text
            
        except sr.UnknownValueError:
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            raise ConnectionError(f"Could not request results from speech recognition service: {e}")
        except Exception as e:
            raise Exception(f"Error processing audio file: {e}")
    
    def split_audio_for_processing(self, audio_file_path: str, chunk_length_ms: int = 30000) -> List[str]:
        audio = AudioSegment.from_file(audio_file_path)
        chunks = []
        
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                chunk.export(temp_file.name, format='wav')
                temp_path = temp_file.name
            
            try:
                with sr.AudioFile(temp_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.record(source)
                
                text = self.recognizer.recognize_google(audio_data)
                chunks.append(text)
                
            except (sr.UnknownValueError, sr.RequestError):
                chunks.append("")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return chunks

class TranscriptAnalyzer:
    """Main class for analyzing and cleaning audio transcripts"""
    
    def __init__(self, enable_audio: bool = True):
        self.enable_audio = enable_audio and AUDIO_AVAILABLE
        if self.enable_audio:
            self.audio_processor = AudioProcessor()
        
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

    def process_audio_file(self, audio_file_path: str, language: str = 'en-US', 
                          chunk_processing: bool = False) -> Tuple[str, CleanupStats]:
        if not self.enable_audio:
            raise RuntimeError("Audio processing not enabled. Install required packages or enable audio support.")
        
        if chunk_processing:
            text_chunks = self.audio_processor.split_audio_for_processing(audio_file_path)
            text = ' '.join(text_chunks)
        else:
            text = self.audio_processor.convert_audio_to_text(audio_file_path, language)
        
        return self.clean_text(text)

    def _normalize_text(self, text: str) -> str:
        """Basic text normalization"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'["“”]', '"', text)  # Normalize double quotes
        text = re.sub(r"[‘’`]", "'", text)  # Normalize single quotes
        return text.strip()

    # (Keep the rest of your methods as-is: _remove_fillers, _fix_stutters, _fix_transcription_errors,
    # _fix_punctuation, _remove_incomplete_sentences, _final_cleanup, analyze_transcript)

# Main function and argparse code remain unchanged
# Copy the rest of your original script from here on (main() function)
