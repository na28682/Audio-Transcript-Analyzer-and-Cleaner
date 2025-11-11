import os
import random
from pydub import AudioSegment

def chunk_audio(input_file, output_dir, chunk_duration=30, segment_duration=180, start_hours=0, max_hours=None):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(4):
        subfolder = os.path.join(output_dir, f"folder_{i+1}")
        os.makedirs(subfolder, exist_ok=True)
    
    # Load the audio file
    print(f"Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)

    total_duration_sec = len(audio) / 1000
    print(f"Total audio duration: {total_duration_sec} seconds")

    start_sec = start_hours * 3600
    print(f"Starting from: {start_hours} hours ({start_sec} seconds)")

    if max_hours is None:
        max_duration_sec = total_duration_sec - start_sec
    else:
        max_duration_sec = min(max_hours * 3600, total_duration_sec - start_sec)
    
    print(f"Processing {max_duration_sec/3600:.2f} hours of audio")
    num_segments = int(max_duration_sec // segment_duration)
    print(f"Number of {segment_duration}-second segments: {num_segments}")
    
    # Convert durations to milliseconds
    chunk_duration_ms = chunk_duration * 1000
    segment_duration_ms = segment_duration * 1000
    start_ms = start_sec * 1000
    
    # Process each segment
    for i in range(num_segments):
        segment_start_ms = start_ms + (i * segment_duration_ms)
        segment_end_ms = segment_start_ms + segment_duration_ms

        max_chunk_start_ms = segment_end_ms - chunk_duration_ms
 
        chunk_start_ms = random.randint(segment_start_ms, max_chunk_start_ms)
        chunk_end_ms = chunk_start_ms + chunk_duration_ms
   
        chunk = audio[chunk_start_ms:chunk_end_ms]

        start_sec_chunk = chunk_start_ms // 1000
        end_sec_chunk = chunk_end_ms // 1000
        subfolder_num = (i % 4) + 1
        subfolder = os.path.join(output_dir, f"folder_{subfolder_num}")
        output_file = os.path.join(subfolder, f"D0016T_{start_sec_chunk}_{end_sec_chunk}.wav")
        print(f"Saving chunk from segment {i+1} to folder_{subfolder_num}: {output_file}")
        chunk.export(output_file, format="wav")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract random 30-second chunks from consecutive 3-minute segments of a long audio file.")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("output_dir", help="Directory to save the output chunks")
    parser.add_argument("--chunk-duration", type=int, default=30, help="Duration of each chunk in seconds (default: 30)")
    parser.add_argument("--segment-duration", type=int, default=180, help="Duration of each segment to sample from in seconds (default: 180)")
    parser.add_argument("--start-hours", type=float, default=0, help="Starting point in hours from the beginning of the audio (default: 0)")
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum number of hours to process after the starting point (default: None, process until the end)")
    
    args = parser.parse_args()
    
    chunk_audio(args.input_file, args.output_dir, args.chunk_duration, args.segment_duration, args.start_hours, args.max_hours)

