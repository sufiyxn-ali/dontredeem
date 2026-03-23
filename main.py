import sys
import soundfile as sf
import librosa
from audio import audio_model
from text import transcribe, text_model
from metadata import parse_metadata
from fusion import fuse_scores, final_decision
import torch
torch.set_default_device("cpu")


import warnings
warnings.filterwarnings("ignore")

def run_pipeline(audio_path, metadata_path):
    # 1. Load Metadata
    print(f"Loading metadata from {metadata_path}...")
    M_t, meta_inf = parse_metadata(metadata_path)
    print(f"Metadata Score: {M_t:.2f}")
    
    # 2. Load Audio
    print(f"Loading audio from {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=16000) # Whisper works best at 16kHz
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    duration = librosa.get_duration(y=y, sr=sr)
    chunk_length = 5.0 # seconds
    samples_per_chunk = int(chunk_length * sr)
    
    scores = []
    
    # Reasoning accumulators
    all_audio_inf = set()
    all_sus_words = set()
    
    print("\nStarting Sliding Window Processing...\n")
    
    mfcc_shape_printed = False
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        
        if len(chunk) < sr: 
            # Skip chunks shorter than 1 sec to avoid issues
            continue
            
        start_time = i / sr
        end_time = min((i + samples_per_chunk) / sr, duration)
        
        # Audio Pipeline
        A_t, a_inf, mfcc_shape = audio_model(chunk, sr)
        
        if not mfcc_shape_printed:
            print(f"MFCC Shape: {mfcc_shape}")
            mfcc_shape_printed = True
        
        # Print Audio Score per window
        print(f"Window [{start_time:.1f}s - {end_time:.1f}s] Audio Score: {A_t:.2f}")
        
        if a_inf:
            all_audio_inf.update(a_inf.split(", "))
            
        T_t = 0.0
        # Text Pipeline (conditional)
        threshold = 0.3
        if A_t > threshold:
            transcript = transcribe(chunk, sr)
            if transcript:
                print(f"Transcript triggered: '{transcript}'")
                T_t, t_inf = text_model(transcript)
                print(f"Text Score: {T_t:.2f}")
                
                if "Words That was sus:" in t_inf:
                    words = t_inf.replace("Words That was sus: ", "").split(", ")
                    all_sus_words.update(words)
        
        # Fusion
        S_t = fuse_scores(A_t, T_t, M_t)
        scores.append(S_t)
        
        print(f"[{int(start_time)}-{int(end_time)} sec] -> {S_t:.2f}")
        print("-" * 40)
        
    # Final Decision
    if not scores:
        print("No audio chunks processed.")
        return
        
    S_final = sum(scores) / len(scores)
    decision = final_decision(S_final)
    
    print(f"\nFinal Score: {S_final:.2f} ({decision})")
    print("\nReasons:")
    
    # Format Audio Reasoning
    audio_str = ", ".join(list(all_audio_inf)) if all_audio_inf else "Normal"
    print(f"1) Tone/ Rate of speech/ other audio inferences: {audio_str}")
    
    # Format Text Reasoning
    text_str = ", ".join(list(all_sus_words)) if all_sus_words else "None"
    print(f"2) Words That was sus: {text_str}")
    
    # Format Meta Reasoning
    print(f"3) Other inferences: {meta_inf}")
    

if __name__ == "__main__":
    import os
    import numpy as np
    
    # Ensure dummy files exist for testing
    if not os.path.exists("metadata.txt"):
        with open("metadata.txt", "w") as f:
            f.write("12/03/2026 23:45, unsaved")
            
    if not os.path.exists("audio.wav"):
        # Create a dummy 10-second audio
        from soundfile import write
        sr=16000
        y = np.random.randn(sr * 10).astype(np.float32)
        write("audio.wav", y, sr)
        
    run_pipeline("sample_audio.wav", "metadata.txt")
