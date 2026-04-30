"""Benchmark all audio files through the pipeline and collect metrics."""
import sys, os, time
# Add root project dir to path so 'from src...' works
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

import librosa
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

from src.audio import audio_model
from src.text import transcribe, text_model
from src.metadata import parse_metadata
from src.fusion import fuse_scores, final_decision
from src.analytics import SessionStateManager

data_dir = os.path.join(root_dir, 'data')

def get_true_label(filename):
    """Determine ground truth label from filename. 1 for Scam, 0 for Safe."""
    fname_lower = filename.lower()
    if "nonscam" in fname_lower:
        return 0
    elif "scam" in fname_lower:
        return 1
    return 0


# Get files and labels
benchmark_files = []
y_true = []

for f in os.listdir(data_dir):
    if not f.endswith('.wav'):
        continue
    label = get_true_label(f)
    if label is not None:
        benchmark_files.append(f)
        y_true.append(label)

meta_path = os.path.join(data_dir, "metadata.txt")

print("="*70)
print(f"BENCHMARK: Running pipeline on {len(benchmark_files)} labeled audio files")
print("="*70)

y_pred = []
processing_times = []

for idx, audio_file in enumerate(sorted(benchmark_files)):
    audio_path = os.path.join(data_dir, audio_file)
    print(f"\n[{idx+1}/{len(benchmark_files)}] FILE: {audio_file}")
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Duration: {duration:.1f}s | True Label: {'SCAM' if y_true[idx] == 1 else 'SAFE'}")
    except Exception as e:
        print(f"  ERROR loading: {e}")
        y_pred.append(0) # Default to safe on error to keep list synced
        processing_times.append(0)
        continue
    
    M_t, meta_inf = parse_metadata(meta_path)
    session_manager = SessionStateManager()
    
    chunk_length = 5.0
    samples_per_chunk = int(chunk_length * sr)
    full_transcript = []
    
    # Simple Diarization Stub since it's failing in pyannote right now
    # We will just pass the audio
    
    t_start = time.time()
    
    # Process windows
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
            # Use rolling transcript to simulate real pipeline behavior
            rolling_text = " ".join(full_transcript[-3:])
            T_t, t_inf, suspicious_tokens = text_model(rolling_text)
            
        S_raw = fuse_scores(A_t, T_t, M_t)
        S_smoothed = session_manager.process_window(S_raw, suspicious_tokens)
        
    elapsed = time.time() - t_start
    processing_times.append(elapsed)
    
    session_data = session_manager.get_session_summary()
    if session_data['total_windows'] == 0:
        print("  No windows processed.")
        y_pred.append(0)
        continue
        
    final_ema = session_manager.aggregator.ema_score
    decision = final_decision(final_ema)
    
    # Convert decision to binary prediction
    # Likely Scam -> 1, Suspicious -> 1 (conservative), Safe -> 0
    is_scam_pred = 1 if decision in ["Likely Scam", "Suspicious"] else 0
    y_pred.append(is_scam_pred)
    
    print(f"  Final EMA: {final_ema:.4f} => Decision: {decision}")
    print(f"  Prediction: {'SCAM' if is_scam_pred == 1 else 'SAFE'} " +
          f"({'CORRECT' if is_scam_pred == y_true[idx] else 'INCORRECT'})")
    print(f"  Time: {elapsed:.1f}s ({duration/elapsed:.1f}x real-time)")

print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"Total Files Analyzed: {len(benchmark_files)}")
print(f"Average Processing Time: {np.mean(processing_times):.2f}s per file")
print("\nMetrics:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print("\nConfusion Matrix:")
print("                 Predicted Safe | Predicted Scam")
print(f"  Actual Safe  |       {cm[0][0]:<8} |      {cm[0][1]:<9}")
print(f"  Actual Scam  |       {cm[1][0]:<8} |      {cm[1][1]:<9}")
print("="*70)
