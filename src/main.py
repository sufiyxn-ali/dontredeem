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
    try:
        from pyannote.audio import Pipeline
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'models', 'pyannote', 'speaker-diarization-3.1', 'config.yaml'
        ))
        
        if not os.path.exists(config_path):
            print("\n[!] WARNING: Offline Pyannote config not found.")
            print("    Please run 'python utils_and_tests/localize_pyannote.py' first.")
            print("    Speaker Diarization bypassed. Analyzing entire audio track unconditionally.")
            return None
            
        print(f"\n[!] Loading localized offline Pyannote Diarization...")
        pipeline = Pipeline.from_pretrained(config_path)
        
        # Send to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as e:
        print(f"\n[!] Failed to load Pyannote Pipeline locally: {e}")
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
    
    # Rolling Transcript Buffer: concatenate last N windows to catch split sentences
    TRANSCRIPT_BUFFER_SIZE = 3
    transcript_buffer = []
    
    # Diarization: identify the first speaker as the "victim" (phone owner)
    victim_speaker = None
    
    print("\n[4/4] Starting Sliding Window Inference...")
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        if len(chunk) < sr: continue
            
        start_time = i / sr
        end_time = min((i + samples_per_chunk) / sr, duration)
        
        print(f"\n- Window [{start_time:.1f}s - {end_time:.1f}s]")
        
        # Diarization Validation: identify dominant speaker and suppress victim-only chunks
        is_victim_only = False
        if speaker_turns:
            active_speakers = [s['speaker'] for s in speaker_turns if s['start'] < end_time and s['end'] > start_time]
            if active_speakers:
                # First speaker encountered is assumed to be the victim (they answered the phone)
                if victim_speaker is None:
                    victim_speaker = active_speakers[0]
                    
                dominant = max(set(active_speakers), key=active_speakers.count)
                print(f"  [Diarization]: Active -> {set(active_speakers)} | Dominant: {dominant}")
                
                # If only the victim is speaking, suppress the threat score
                if len(set(active_speakers)) == 1 and dominant == victim_speaker:
                    is_victim_only = True
                    print(f"  [Diarization]: Victim-only chunk. Suppressing threat score.")
            else:
                print(f"  [Diarization]: Silence.")
                
        # Audio Pipeline
        A_t, a_inf, mfcc_shape = audio_model(chunk, sr)
        print(f"  [Audio] Score: {A_t:.4f}")
        print(f"  [Details]: {a_inf[:100]}")
            
        T_t = 0.0
        t_inf = ""
        suspicious_tokens = []
        
        # Text Pipeline: always transcribe for the rolling buffer
        transcript = transcribe(chunk, sr)
        if transcript:
            transcript_buffer.append(transcript)
            # Keep only the last N chunks in the buffer
            if len(transcript_buffer) > TRANSCRIPT_BUFFER_SIZE:
                transcript_buffer.pop(0)
            
            # Use the rolling concatenated transcript for analysis
            rolling_transcript = " ".join(transcript_buffer)
            print(f"  [Transcript]: '{transcript}'")
            T_t, t_inf, suspicious_tokens = text_model(rolling_transcript)
            print(f"  [Score] Text: {T_t:.4f}")
            print(f"  [Analysis]: {t_inf}")
        
        # Diarization suppression: if only victim is speaking, zero the text score
        if is_victim_only:
            T_t = T_t * 0.1  # Heavily suppress, don't fully zero (victim might be reading back scam text)
            print(f"  [Suppressed] Victim-only text score reduced to {T_t:.4f}")
        
        # 6. Risk Fusion & Aggregation
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
    # Note: HF_TOKEN is no longer required due to localization.

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    meta_path = os.path.join(data_dir, "metadata.txt")
    audio_path = os.path.join(data_dir, "sample_ScamConvo.wav")
    
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f: f.write("12/03/2026 23:45, unsaved")
            
    if not os.path.exists(audio_path):
        from soundfile import write
        sr=16000
        y = np.random.randn(sr * 10).astype(np.float32)
        write(audio_path, y, sr)
        
    # Start pipeline
    run_pipeline(audio_path, meta_path)
