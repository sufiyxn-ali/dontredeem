"""Benchmark all audio files through the pipeline and collect results."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from audio import audio_model
from text import transcribe, text_model
from metadata import parse_metadata
from fusion import fuse_scores, final_decision
from analytics import SessionStateManager

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# All audio files
audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
meta_path = os.path.join(data_dir, "metadata.txt")

print("="*70)
print("BENCHMARK: Running all audio files through the pipeline")
print("="*70)

for audio_file in sorted(audio_files):
    audio_path = os.path.join(data_dir, audio_file)
    print(f"\n{'='*70}")
    print(f"FILE: {audio_file}")
    print(f"{'='*70}")
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Duration: {duration:.1f}s")
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue
    
    M_t, meta_inf = parse_metadata(meta_path)
    session_manager = SessionStateManager()
    
    chunk_length = 5.0
    samples_per_chunk = int(chunk_length * sr)
    full_transcript = []
    
    t_start = time.time()
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        if len(chunk) < sr:
            continue
        
        A_t, a_inf, _ = audio_model(chunk, sr)
        
        T_t = 0.0
        suspicious_tokens = []
        transcript = transcribe(chunk, sr)
        if transcript:
            full_transcript.append(transcript)
            T_t, t_inf, suspicious_tokens = text_model(transcript)
        
        S_raw = fuse_scores(A_t, T_t, M_t)
        S_smoothed = session_manager.process_window(S_raw, suspicious_tokens)
    
    elapsed = time.time() - t_start
    
    session_data = session_manager.get_session_summary()
    if session_data['total_windows'] == 0:
        print("  No windows processed.")
        continue
    
    final_ema = session_manager.aggregator.ema_score
    decision = final_decision(final_ema)
    
    tokens_str = ", ".join(session_data['all_tokens']) if session_data['all_tokens'] else "None"
    
    print(f"  Windows: {session_data['total_windows']}")
    print(f"  Final EMA: {final_ema:.4f}")
    print(f"  Max Spike: {session_data['max_spike']:.4f}")
    print(f"  Decision: {decision}")
    print(f"  Tokens: {tokens_str}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Full Transcript: {' '.join(full_transcript)[:200]}")
