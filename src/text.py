import sys
import os
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
import torch
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Load BiLSTM Loader
try:
    from src.bilstm_loader import ScamDetectionModel
    print("Loading localized BiLSTM Scam Detector...")
    bilstm_model = ScamDetectionModel()
except Exception as e:
    print(f"Warning: Failed to load BiLSTM model ({e})")
    bilstm_model = None

# Load MiniLM Semantic Embeddings
try:
    print("Loading MiniLM Semantic Embeddings...")
    minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-embed scam concept vectors
    scam_concepts = [
        "I need you to wire the money immediately",
        "Your account has been compromised and blocked",
        "Provide your social security number and password",
        "You will be arrested or deported if you do not pay",
        "Click the link to verify your identity"
    ]
    concept_embeddings = minilm_model.encode(scam_concepts, convert_to_tensor=True)
except Exception as e:
    print(f"Warning: Failed to load MiniLM ({e})")
    minilm_model = None
    concept_embeddings = None

# Localized Speech Recognition
print("Loading localized speech recognition (faster-whisper-small)...")
try:
    from faster_whisper import WhisperModel
    whisper_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'faster-whisper-small')
    if os.path.exists(whisper_path):
        asr_pipeline = WhisperModel(whisper_path, device="cpu", compute_type="int8")
    else:
        print("Warning: Offline Faster-Whisper not found.")
        asr_pipeline = None
except Exception as e:
    print(f"Warning: Failed to load ASR pipeline ({e})")
    asr_pipeline = None

# Gemma Lazy Loader
gemma_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gemma-4-E2B')
gemma_pipeline = None

# Critical highly-specific fallbacks (for fuzzy matching)
CRITICAL_FALLBACKS = ["emirates id", "mrets", "deported", "warrant", "arrest"]

def transcribe(y, sr):
    if asr_pipeline is None:
        return ""
    try:
        segments, _ = asr_pipeline.transcribe(y, beam_size=1, language="en")
        return " ".join([segment.text for segment in segments]).strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def text_model(transcript):
    global gemma_pipeline
    
    if not transcript:
        return 0.0, "No speech detected", []
        
    transcript_lower = transcript.lower()
    
    # 1. BiLSTM Base Score
    base_score = 0.0
    inference_parts = []
    suspicious_tokens = []
    
    if bilstm_model is not None:
        pred = bilstm_model.predict(transcript)
        if pred is not None:
            base_score = pred
            inference_parts.append(f"BiLSTM: {base_score:.2f}")

    # 2. Semantic Similarity Score (MiniLM)
    semantic_boost = 0.0
    if minilm_model is not None and concept_embeddings is not None:
        transcript_emb = minilm_model.encode(transcript, convert_to_tensor=True)
        cos_scores = util.cos_sim(transcript_emb, concept_embeddings)[0]
        max_sim = torch.max(cos_scores).item()
        
        if max_sim > 0.4:  # Threshold for semantic similarity
            semantic_boost = max_sim * 0.5  # Scale boost with similarity strength
            inference_parts.append(f"MiniLM Sim: {max_sim:.2f}")
    
    # 3. Fuzzy Matching (Catch ASR typos on critical words)
    fuzzy_boost = 0.0
    for critical in CRITICAL_FALLBACKS:
        if fuzz.partial_ratio(transcript_lower, critical) > 85:  # High fuzzy threshold on full transcript
            fuzzy_boost = 0.4
            inference_parts.append(f"Fuzzy Critical: {critical}")
            suspicious_tokens.append((critical, 1.0))
            break

    # Combine scores
    combined_score = min(base_score + semantic_boost + fuzzy_boost, 1.0)
    inference = " | ".join(inference_parts) if inference_parts else "No clear signals"
    
    # 4. INVERTED WAKE-UP HYBRID ARCHITECTURE
    # Gemma acts as a safety net for FALSE POSITIVES.
    # If the combined score is NOT SAFE (>= 0.5), we ask Gemma to verify.
    # If the combined score is SAFE (< 0.5), we trust it and don't wake Gemma.
    
    if combined_score >= 0.5:
        print(f"\n  [Wake-Up] BiLSTM/MiniLM says SCAM (Score: {combined_score:.2f}). Waking up Gemma to verify...")
        if gemma_pipeline is None and os.path.exists(gemma_path):
            print("  [Memory Check] Loading Gemma into RAM (float16)...")
            try:
                gemma_pipeline = pipeline("text-generation", model=gemma_path, model_kwargs={"torch_dtype": torch.float16})
            except Exception as e:
                print(f"  [Gemma Load Error]: {e}")
        
        if gemma_pipeline is not None:
            prompt = (
                "Analyze this phone call transcript. If the caller uses financial manipulation, tech support scams, or high-pressure tactics, answer SCAM. If not, answer SAFE.\n"
                f"Transcript: \"{transcript}\"\n"
                "Decision:"
            )
            try:
                res = gemma_pipeline(prompt, max_new_tokens=5, do_sample=False)
                gen_text = res[0]['generated_text'][len(prompt):].strip().upper()
                print(f"  [Gemma Outputs]: {gen_text}")
                if "SCAM" in gen_text:
                    combined_score = 1.0 # Confirmed
                    inference = f"Gemma Confirms: SCAM! ({inference})"
                else:
                    combined_score = 0.4 # Overruled to Safe
                    inference = f"Gemma Overrules: SAFE. ({inference})"
            except Exception as e:
                print(f"  [Gemma Execution Error]: {e}")
    else:
        print(f"\n  [No Wake-Up] BiLSTM/MiniLM says SAFE (Score: {combined_score:.2f}). Trusting model, skipping Gemma.")
        inference = f"Confirmed Safe ({inference})"
        
    return combined_score, inference, suspicious_tokens
