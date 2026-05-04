import os
import argparse
import librosa
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_default_device("cpu")

# Import custom modules
from audio import audio_model
from text import transcribe, text_model
from metadata import parse_metadata
from fusion import fuse_scores, final_decision
from analytics import SessionStateManager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_diarization():
    """Load local pyannote diarization when available.

    The pipeline is optional. If pyannote cannot initialize in the current
    environment, return None so the rest of scam detection still runs.
    """
    config_path = os.path.join(ROOT_DIR, "models", "pyannote", "speaker-diarization-3.1", "config.yaml")
    if not os.path.exists(config_path):
        print("\n[!] Local pyannote diarization config not found.")
        print(f"    Expected: {config_path}")
        print("    Continuing without diarization.")
        return None

    try:
        from pyannote.audio import Pipeline

        print("\n[*] Loading local pyannote diarization pipeline...")
        pipeline = Pipeline.from_pretrained(config_path)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        print("    [OK] Pyannote diarization loaded.")
        return pipeline
    except Exception as e:
        print(f"\n[!] Failed to load local pyannote diarization: {e}")
        print("    Continuing without diarization.")
        return None

def extract_speaker_turns(diarization_model, y, sr):
    """Run diarization on an in-memory waveform and return normalized turns."""
    if diarization_model is None:
        return []

    try:
        waveform = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
        result = diarization_model({"waveform": waveform, "sample_rate": sr})
        annotation = getattr(result, "speaker_diarization", result)

        speaker_turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_turns.append({"speaker": speaker, "start": turn.start, "end": turn.end})
        return speaker_turns
    except Exception as e:
        print(f"      Diarization inference failed: {e}. Continuing without speaker filtering.")
        return []

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
    speaker_turns = extract_speaker_turns(diarization_model, y, sr)
    print(f"      Found {len(speaker_turns)} speaker turns.")

    # 5. Inference
    chunk_length = 5.0 # seconds
    samples_per_chunk = int(chunk_length * sr)
    transcript_buffer = []
    transcript_buffer_size = 3
    victim_speaker = None
    
    print("\n[4/4] Starting Sliding Window Inference...")
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i+samples_per_chunk]
        if len(chunk) < sr: continue
            
        start_time = i / sr
        end_time = min((i + samples_per_chunk) / sr, duration)
        
        print(f"\n- Window [{start_time:.1f}s - {end_time:.1f}s]")
        
        is_victim_only = False
        if speaker_turns:
            active_speakers = [s['speaker'] for s in speaker_turns if s['start'] < end_time and s['end'] > start_time]
            if active_speakers:
                if victim_speaker is None:
                    victim_speaker = active_speakers[0]

                dominant = max(set(active_speakers), key=active_speakers.count)
                print(f"  [Diarization]: Active -> {set(active_speakers)} | Dominant: {dominant}")

                if len(set(active_speakers)) == 1 and dominant == victim_speaker:
                    is_victim_only = True
                    print("  [Diarization]: Victim-only chunk. Suppressing threat score.")
            else:
                print(f"  [Diarization]: Silence.")
                
        # Audio Pipeline
        A_t, a_inf, mfcc_shape = audio_model(chunk, sr)
        print(f"  [Audio] Score: {A_t:.4f}")
        print(f"  [Details]: {a_inf[:100]}")
            
        T_t = 0.0
        t_inf = ""
        suspicious_tokens = []
        
        transcript = transcribe(chunk, sr)
        if transcript:
            transcript_buffer.append(transcript)
            if len(transcript_buffer) > transcript_buffer_size:
                transcript_buffer.pop(0)

            rolling_transcript = " ".join(transcript_buffer)
            print(f"  [Transcript]: '{transcript}'")
            T_t, t_inf, suspicious_tokens = text_model(rolling_transcript)
            print(f"  [Score] Text: {T_t:.4f}")
            print(f"  [Analysis]: {t_inf}")

        if is_victim_only and T_t > 0:
            T_t *= 0.1
            print(f"  [Suppressed] Victim-only text score reduced to {T_t:.4f}")
        
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
    return {
        "decision": decision,
        "final_score": final_ema,
        "summary": session_data,
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local multimodal scam detection pipeline.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=os.path.join(ROOT_DIR, "data", "sample_ScamConvo.wav"),
        help="Path to a .wav file to analyze.",
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join(ROOT_DIR, "data", "metadata.txt"),
        help="Path to metadata.txt formatted as 'dd/mm/yyyy hh:mm, saved|unsaved'.",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.audio_path):
        print(f"Audio file not found: {args.audio_path}")
        return 1

    if not os.path.exists(args.metadata):
        print(f"Metadata file not found: {args.metadata}")
        return 1

    run_pipeline(args.audio_path, args.metadata)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
