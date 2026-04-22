import sys
import os
import soundfile as sf
import librosa
import torch
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_default_device("cpu")

# Import custom modules
from audio import audio_model
from text import transcribe, text_model
from metadata import parse_metadata
from fusion import fuse_scores, final_decision
from analytics import SessionStateManager

def load_diarization():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("\n[!] WARNING: 'HF_TOKEN' environment variable not set.")
        print("    Speaker Diarization bypassed. Analyzing entire audio track unconditionally.")
        return None
        
    try:
        from pyannote.audio import Pipeline
        print(f"\n[!] Authenticating Pyannote Diarization with HF_TOKEN...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        # Send to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as e:
        print(f"\n[!] Failed to load Pyannote Pipeline: {e}")
        return None

def run_pipeline(audio_path, metadata_path):
    print("\n" + "="*60)
    print("*** UNIFIED ENSEMBLE SCAM DETECTION PIPELINE ***")
    print("="*60)
    
    # 1. Initialize Analytics Engine
    session_manager = SessionStateManager()
    
    # 2. Metadata
    print(f"\n[1/4] Loading metadata from {metadata_path}...")
    M_t, meta_inf = parse_metadata(metadata_path)
    
    # 3. Audio Extraction
    print(f"\n[2/4] Loading audio from {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"      Error loading audio: {e}")
        return
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 4. Speaker Diarization
    print(f"\n[3/4] Initializing Speaker Diarization...")
    diarization_model = load_diarization()
    speaker_turns = []
    
    if diarization_model:
        try:
            print("      Diarizing audio file (this may take a moment)...")
            diar_result = diarization_model(audio_path)
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
                speaker_turns.append({"speaker": speaker, "start": turn.start, "end": turn.end})
            print(f"      Found {len(speaker_turns)} speaker turns.")
        except Exception as e:
            print(f"      Diarization inference failed: {e}. Falling back to full synthesis.")

    # 5. Inference
    chunk_length = 5.0 # seconds
    samples_per_chunk = int(chunk_length * sr)
    
    print("\n[4/4] Starting Sliding Window Inference...")
    mfcc_shape_printed = False
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        if len(chunk) < sr: continue
            
        start_time = i / sr
        end_time = min((i + samples_per_chunk) / sr, duration)
        
        print(f"\n- Window [{start_time:.1f}s - {end_time:.1f}s]")
        
        # Diarization Validation: Check if this chunk is just the victim validating
        # If diarization is active, track the dominant speaker.
        if speaker_turns:
            # check which speaker dominates the chunk
            active_speakers = [s['speaker'] for s in speaker_turns if s['start'] < end_time and s['end'] > start_time]
            if active_speakers:
                print(f"  [Diarization]: Active speakers in chunk -> {set(active_speakers)}")
            else:
                print(f"  [Diarization]: Silence.")
                
        # Audio Pipeline
        A_t, a_inf, mfcc_shape = audio_model(chunk, sr)
        print(f"  [Audio] Score: {A_t:.4f}")
        print(f"  [Details]: {a_inf[:100]}")
            
        T_t = 0.0
        t_inf = ""
        suspicious_tokens = []
        
        # Text Pipeline (conditional)
        if A_t > 0.1 or True: 
            transcript = transcribe(chunk, sr)
            if transcript:
                print(f"  [Transcript]: '{transcript}'")
                T_t, t_inf, suspicious_tokens = text_model(transcript)
                print(f"  [Score] Text: {T_t:.4f}")
                print(f"  [Analysis]: {t_inf}")
        
        # 6. Risk Fusion & Aggregation
        # Initial fusion score
        S_raw = fuse_scores(A_t, T_t, M_t)
        
        # Feed to Session State (EMA Smoothing)
        S_smoothed = session_manager.process_window(S_raw, suspicious_tokens)
        
        print(f"  => Raw Score: {S_raw:.4f} | EMA Aggregated Score: {S_smoothed:.4f}")
        print("-" * 60)
        
    # Final Decision
    session_data = session_manager.get_session_summary()
    if session_data['total_windows'] == 0:
        print("No audio chunks processed.")
        return
        
    final_ema = session_manager.aggregator.ema_score
    decision = final_decision(final_ema)
    
    print(f"\n{'='*60}")
    print(f"FINAL DECISION: {decision}")
    print(f"Final EMA Score: {final_ema:.4f}")
    print(f"{'='*60}")
    print("\n[Unified Analysis Breakdown]\n")
    print(f"Total Windows Tracked: {session_data['total_windows']}")
    print(f"Max Threat Spike: {session_data['max_spike']:.4f}")
    
    tokens_str = ", ".join(session_data['all_tokens']) if session_data['all_tokens'] else "None"
    print(f"Suspicious Vocabulary Detected: {tokens_str}")


if __name__ == "__main__":
    # Ensure env variable is set if debugging manually
    # os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE" 
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    meta_path = os.path.join(data_dir, "metadata.txt")
    audio_path = os.path.join(data_dir, "sample_1audio.wav")
    
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f: f.write("12/03/2026 23:45, unsaved")
            
    if not os.path.exists(audio_path):
        from soundfile import write
        sr=16000
        y = np.random.randn(sr * 10).astype(np.float32)
        write(audio_path, y, sr)
        
    # Start pipeline
    run_pipeline(audio_path, meta_path)
