import os
import whisperx
import pandas as pd
from datetime import timedelta
import torch
import numpy as np
from tqdm import tqdm
import huggingface_hub

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
parent_dir = "/Users/Desktop/whisperX/audio_data_whisper"
output_dir = "/Users/Desktop/whisperX/transcriptions"

os.makedirs(output_dir, exist_ok=True)

# Load WhisperX model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print("Attempting to download model from Hugging Face Hub...")
    model = whisperx.load_model("large-v2", compute_type="float32", device=device, 
                               vad_options={"use_vad": True}, download_root=None)
except (RuntimeError, huggingface_hub.errors.LocalEntryNotFoundError, 
        huggingface_hub.errors.HfHubHTTPError) as e:
    print(f"Error loading model: {str(e)}")
    
    if "Unauthorized" in str(e) or "Invalid credentials" in str(e):
        print("\nAuthentication error with Hugging Face Hub.")
        print("You need to login to Hugging Face first. Run this command:")
        print("huggingface-cli login")
        print("Or use a different model that doesn't require authentication.")
        exit(1)
    elif "local_files_only" in str(e):
        print("\nCannot find model locally and network access is restricted.")
        print("Try running with internet access or download the model manually.")
        exit(1)
    else:
        print("\nThis appears to be a PyTorch/torchvision compatibility issue.")
        print("Try reinstalling compatible versions with:")
        print("pip uninstall -y torch torchvision torchaudio")
        print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2")
        exit(1)

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def process_audio_in_chunks(audio_path, chunk_duration=300): 
    print(f"\n🎧 Processing file: {os.path.basename(audio_path)}")

    audio = whisperx.load_audio(audio_path)
    sample_rate = 16000  
    
    total_duration = len(audio) / sample_rate
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    
    all_transcripts = []
    
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        chunk_start_sec = i * chunk_duration
        chunk_end_sec = min((i + 1) * chunk_duration, total_duration)
        
        start_idx = int(chunk_start_sec * sample_rate)
        end_idx = int(chunk_end_sec * sample_rate)
        
        audio_chunk = audio[start_idx:end_idx]
        
        result = model.transcribe(audio_chunk)
        
        for segment in result["segments"]:
            adjusted_start = chunk_start_sec + segment["start"]
            adjusted_end = chunk_start_sec + segment["end"]
            
            all_transcripts.append({
                "Chunk": i + 1,
                "Chunk Start": format_timestamp(chunk_start_sec),
                "Chunk End": format_timestamp(chunk_end_sec),
                "Start Time": format_timestamp(adjusted_start),
                "End Time": format_timestamp(adjusted_end),
                "Start Seconds": adjusted_start,
                "End Seconds": adjusted_end,
                "Transcript": segment["text"].strip()
            })
    
    return all_transcripts

for filename in sorted(os.listdir(parent_dir)):
    if not filename.lower().endswith((".wav", ".mp3", ".m4a")):
        continue

    file_path = os.path.join(parent_dir, filename)
    base_name = os.path.splitext(filename)[0]
    
    try:
        transcript_data = process_audio_in_chunks(file_path)
        
        output_csv = os.path.join(output_dir, f"transcriptions_{base_name}.csv")
        df = pd.DataFrame(transcript_data)
        
        df = df.sort_values(by="Start Seconds")
        
        df.drop(columns=["Start Seconds", "End Seconds"], inplace=True)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved: {output_csv}")
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")

