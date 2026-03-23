import warnings
from transformers import pipeline
warnings.filterwarnings('ignore')

# Initialize ASR Pipeline
print("Loading speech recognition pipeline (openai/whisper-tiny)...")
asr_pipeline = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny"
)

SUSPICIOUS_KEYWORDS = [
    "urgent", "bank", "transfer", "otp", "password", 
    "account", "blocked", "compromised", "verify", 
    "police", "arrest", "warrant", "card", "security"
]

def transcribe(y, sr):
    """
    Transcribes a 5-second audio chunk.
    HuggingFace pipeline expects a dictionary with 'array' and 'sampling_rate'.
    """
    try:
        # Pass the raw numpy array to pipeline. 
        # Make sure the audio is normalized between -1 and 1
        result = asr_pipeline(
            {"array": y, "sampling_rate": sr},
            generate_kwargs={"task": "transcribe", "language": "en"}
        )
        return result.get('text', '').strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def text_model(transcript):
    """
    Evaluates transcript for keywords and returns a score and inferences.
    """
    if not transcript:
        return 0.0, "No speech detected"
        
    transcript_lower = transcript.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in transcript_lower]
    
    score = 0.0
    inferences = []
    
    if len(found_keywords) > 0:
        score += min(len(found_keywords) * 0.25, 1.0) # 0.25 per keyword, cap at 1.0
        inferences.append(f"Words That was sus: {', '.join(found_keywords)}")
    else:
        inferences.append("No suspicious keywords")
        
    return score, ", ".join(inferences)

if __name__ == "__main__":
    # Test script
    import numpy as np
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)
    t = transcribe(dummy_audio, 16000)
    print("Transcript:", t)
    print("Score:", text_model(t))
