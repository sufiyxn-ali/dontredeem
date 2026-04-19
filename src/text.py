import sys
import os
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline

# Cross-Link Local DistillBertini
distillbert_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'DistillBertini', 'files')
sys.path.insert(0, distillbert_path)

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

print("Loading localized Gemma 4 E2B...")
try:
    gemma_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gemma-4-E2B')
    if os.path.exists(gemma_path):
        gemma_pipeline = pipeline("text-generation", model=gemma_path)
    else:
        gemma_pipeline = None
except Exception as e:
    print(f"Warning: Failed to load Gemma pipeline ({e})")
    gemma_pipeline = None

try:
    from infer import ScamScorer
    model_dir = os.path.join(distillbert_path, 'model')
    distilbert_scorer = ScamScorer(model_dir=model_dir, threshold=0.4, explain=True)
    print("DistilBERT model loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load DistilBERT model ({e})")
    distilbert_scorer = None

CRITICAL_KEYWORDS = [
    "deported", "deport", "mrets", "emirates id", "arrest", "warrant", 
    "police", "penalty", "fine"
]

SUSPICIOUS_KEYWORDS = [
    "urgent", "bank", "transfer", "otp", "password", 
    "account", "blocked", "compromised", "verify", 
    "card", "security", "claim", "confirm", "urgent action", "immediate",
    "click link", "updated information", "verify identity",
    "suspend", "virus", "malware", "credit", "debit",
    "social security", "ssn", "federal", "irs", "tax", 
    "refund", "prize", "inheritance", "won", "congratulations"
]

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
    if not transcript:
        return 0.0, "No speech detected", []
        
    transcript_lower = transcript.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in transcript_lower]
    found_critical = [kw for kw in CRITICAL_KEYWORDS if kw in transcript_lower]
    
    if distilbert_scorer is not None:
        try:
            result = distilbert_scorer.score(transcript)
            score = result.scam_probability
            label = result.label
            
            # Boost DistilBERT confidence with hardcoded Regex
            keyword_boost = 0.0
            if found_critical:
                keyword_boost = 0.90
            elif len(found_keywords) >= 3: keyword_boost = 0.3
            elif len(found_keywords) == 2: keyword_boost = 0.2
            elif len(found_keywords) >= 1: keyword_boost = 0.15
            
            boosted_score = min(score + keyword_boost, 1.0)
            keyword_str = f" | Keywords: {', '.join(found_keywords)}" if found_keywords else ""
            inference = f"DistilBERT: {label.upper()} (Model: {score:.3f}, Boosted: {boosted_score:.3f}){keyword_str}"
            
            # === WAKE-UP HYBRID ARCHITECTURE ===
            if boosted_score > 0.30 and gemma_pipeline is not None:
                print("\n  [Wake-Up] Suspicion > 30%. Waking up Gemma for contextual validation...")
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
                        boosted_score = 1.0
                        inference = f"Gemma Confirmed SCAM! (DistilBERT Context: {inference})"
                    else:
                        boosted_score = min(score, 0.4) # Soften the blow if Gemma disagrees
                        inference = f"Gemma Overrules: SAFE. (DistilBERT Context: {inference})"
                except Exception as e:
                    print(f"  [Gemma Execution Error]: {e}")
                    
            # Extract Tokens
            model_tokens = result.suspicious_tokens if result.suspicious_tokens else []
            all_tokens = list(model_tokens)
            if found_critical:
                all_tokens.extend([(kw, 1.0) for kw in found_critical[:5]])
            if found_keywords:
                all_tokens.extend([(kw, 0.8) for kw in found_keywords[:5]])
                
            all_tokens = sorted(all_tokens, key=lambda x: x[1], reverse=True)[:10]
            
            return boosted_score, inference, all_tokens
        except Exception as e:
            return 0.3, f"DistilBERT error: {str(e)[:50]}", []
            
    # Fallback Architecture: Pure keyword math
    score = 0.0
    tokens = []
    
    if found_critical:
        score = 0.90
        inference = f"Fallback Critical: {', '.join(found_critical)}"
        tokens.extend([(kw, 1.0) for kw in found_critical[:5]])
    elif len(found_keywords) > 0:
        score = min(len(found_keywords) * 0.25, 1.0)
        inference = f"Fallback Keywords: {', '.join(found_keywords)}"
        tokens.extend([(kw, 0.7) for kw in found_keywords[:5]])
    else:
        inference = "No suspicious content detected"
        
    # === WAKE-UP HYBRID ARCHITECTURE (Fallback) ===
    if score > 0.30 and gemma_pipeline is not None:
        print("\n  [Wake-Up] Fallback suspicion > 30%. Waking up Gemma for contextual validation...")
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
                score = 1.0
                inference = f"Gemma Confirmed SCAM! ({inference})"
            else:
                score = min(score, 0.4)
                inference = f"Gemma Overrules: SAFE. ({inference})"
        except Exception as e:
            print(f"  [Gemma Execution Error]: {e}")

    return score, inference, tokens
