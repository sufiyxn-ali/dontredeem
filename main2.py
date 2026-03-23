import sys
import os
import warnings
import numpy as np
import librosa
import soundfile as sf
import torch

# 🔥 FIX 1: Suppress ONLY annoying HF warnings
warnings.filterwarnings("ignore", message="A custom logits processor")

# 🔥 FIX 2: Force CPU safely (NO meta bug)
device = torch.device("cpu")

# Your modules
from audio import audio_model
from text import transcribe, text_model
from metadata import parse_metadata
from fusion import fuse_scores, final_decision


def run_pipeline(audio_path, metadata_path):
    # 1. Load Metadata
    print(f"Loading metadata from {metadata_path}...")
    M_t, meta_inf = parse_metadata(metadata_path)
    print(f"Metadata Score: {M_t:.2f}")
    
    # 2. Load Audio
    print(f"Loading audio from {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    duration = librosa.get_duration(y=y, sr=sr)
    chunk_length = 5.0
    samples_per_chunk = int(chunk_length * sr)
    
    scores = []
    
    # Reason tracking
    all_audio_inf = set()
    all_sus_words = set()
    
    print("\nStarting Sliding Window Processing...\n")
    
    mfcc_shape_printed = False
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        
        if len(chunk) < sr:
            continue
            
        start_time = i / sr
        end_time = min((i + samples_per_chunk) / sr, duration)
        
        # 🎧 Audio
        A_t, a_inf, mfcc_shape = audio_model(chunk, sr)
        
        if not mfcc_shape_printed:
            print(f"MFCC Shape: {mfcc_shape}")
            mfcc_shape_printed = True
        
        print(f"Window [{start_time:.1f}s - {end_time:.1f}s] Audio Score: {A_t:.2f}")
        
        if a_inf:
            all_audio_inf.update(a_inf.split(", "))
            
        # 🧠 Text (only if needed)
        T_t = 0.0
        if A_t > 0.3:
            transcript = transcribe(chunk, sr)
            
            if transcript and transcript.strip():
                print(f"Transcript: '{transcript}'")
                
                T_t, t_inf = text_model(transcript)
                print(f"Text Score: {T_t:.2f}")
                
                if "Words That was sus:" in t_inf:
                    words = t_inf.replace("Words That was sus: ", "").split(", ")
                    all_sus_words.update(words)
        
        # 🔗 Fusion
        S_t = fuse_scores(A_t, T_t, M_t)
        scores.append(S_t)
        
        print(f"[{int(start_time)}-{int(end_time)} sec] -> {S_t:.2f}")
        print("-" * 40)
    
    if not scores:
        print("No audio chunks processed.")
        return
        
    # 📊 Final
    S_final = sum(scores) / len(scores)
    decision = final_decision(S_final)
    
    print("\n" + "="*50)
    print(f"FINAL SCORE: {S_final:.2f} → {decision}")
    print("="*50)
    
    print("\nReasons:")
    
    print(f"1) Audio Inference: {', '.join(all_audio_inf) if all_audio_inf else 'Normal'}")
    print(f"2) Suspicious Words: {', '.join(all_sus_words) if all_sus_words else 'None'}")
    print(f"3) Metadata Flags: {meta_inf}")


if __name__ == "__main__":
    # Ensure test files
    if not os.path.exists("metadata.txt"):
        with open("metadata.txt", "w") as f:
            f.write("12/03/2026 23:45, unsaved")
            
    if not os.path.exists("sample_audio.wav"):
        sr = 16000
        y = np.random.randn(sr * 10).astype(np.float32)
        sf.write("audio.wav", y, sr)
        
    run_pipeline("sample_audio.wav", "metadata.txt")