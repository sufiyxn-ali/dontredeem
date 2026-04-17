import sys
import os
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline

# Cross-Link Local DistillBertini
distillbert_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'DistillBertini', 'files')
sys.path.insert(0, distillbert_path)

print("Loading speech recognition pipeline (openai/whisper-tiny)...")
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
except Exception as e:
    print(f"Warning: Failed to load ASR pipeline ({e})")
    asr_pipeline = None

try:
    from infer import ScamScorer
    model_dir = os.path.join(distillbert_path, 'model')
    distilbert_scorer = ScamScorer(model_dir=model_dir, threshold=0.4, explain=True)
    print("DistilBERT model loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load DistilBERT model ({e})")
    distilbert_scorer = None

SUSPICIOUS_KEYWORDS = [
    "urgent", "bank", "transfer", "otp", "password", 
    "account", "blocked", "compromised", "verify", 
    "police", "arrest", "warrant", "card", "security",
    "claim", "confirm", "urgent action", "immediate",
    "click link", "updated information", "verify identity",
    "suspend", "virus", "malware", "credit", "debit",
    "social security", "ssn", "federal", "irs", "tax", 
    "refund", "prize", "inheritance", "won", "congratulations"
]

def transcribe(y, sr):
    if asr_pipeline is None:
        return ""
    try:
        result = asr_pipeline({"array": y, "sampling_rate": sr}, generate_kwargs={"task": "transcribe", "language": "en"})
        return result.get('text', '').strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def text_model(transcript):
    if not transcript:
        return 0.0, "No speech detected", []
        
    transcript_lower = transcript.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in transcript_lower]
    
    if distilbert_scorer is not None:
        try:
            result = distilbert_scorer.score(transcript)
            score = result.scam_probability
            label = result.label
            
            # Boost DistilBERT confidence with hardcoded Regex
            keyword_boost = 0.0
            if len(found_keywords) >= 3: keyword_boost = 0.3
            elif len(found_keywords) == 2: keyword_boost = 0.2
            elif len(found_keywords) >= 1: keyword_boost = 0.15
            
            boosted_score = min(score + keyword_boost, 1.0)
            keyword_str = f" | Keywords: {', '.join(found_keywords)}" if found_keywords else ""
            inference = f"DistilBERT: {label.upper()} (Model: {score:.3f}, Boosted: {boosted_score:.3f}){keyword_str}"
            
            # Extract Tokens
            model_tokens = result.suspicious_tokens if result.suspicious_tokens else []
            all_tokens = list(model_tokens)
            if found_keywords:
                all_tokens.extend([(kw, 0.8) for kw in found_keywords[:5]])
                
            all_tokens = sorted(all_tokens, key=lambda x: x[1], reverse=True)[:10]
            
            return boosted_score, inference, all_tokens
        except Exception as e:
            return 0.3, f"DistilBERT error: {str(e)[:50]}", []
            
    # Fallback Architecture: Pure keyword math
    score = 0.0
    tokens = []
    if len(found_keywords) > 0:
        score = min(len(found_keywords) * 0.25, 1.0)
        inference = f"Fallback Keywords: {', '.join(found_keywords)}"
        tokens = [(kw, 0.7) for kw in found_keywords[:5]]
    else:
        inference = "No suspicious content detected"
    return score, inference, tokens
