import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Model path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "minilm", "best")
MINILM_ENABLED = os.environ.get("ENABLE_MINILM_FALLBACK", "1").lower() not in {"0", "false", "no"}

# Global instances
_tokenizer = None
_model = None
_load_attempted = False

def is_available() -> bool:
    """Return whether the optional local MiniLM fallback can be attempted."""
    required_files = ("config.json", "tokenizer.json")
    return MINILM_ENABLED and os.path.isdir(MODEL_DIR) and all(
        os.path.exists(os.path.join(MODEL_DIR, name)) for name in required_files
    )

def load_model():
    global _tokenizer, _model, _load_attempted
    if _load_attempted:
        return _model is not None and _tokenizer is not None

    _load_attempted = True
    if not is_available():
        return False

    if _tokenizer is None or _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            _model.eval()
        except Exception as e:
            print(f"\n[!] Optional MiniLM fallback could not be loaded from {MODEL_DIR}: {e}")
            _tokenizer = None
            _model = None
            return False

    return True

def predict_chunk(text: str) -> dict:
    """
    Predicts if a ~5-second text chunk is a scam or normal.
    Returns: { "label": "scam" or "normal", "confidence": float, "risk": "low", "medium", or "high" }
    """
    if not load_model():
        return {"label": "error", "confidence": 0.0, "risk": "low"}
        
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    scam_prob = probs[0][1].item()
    
    label = "scam" if scam_prob > 0.5 else "normal"
    
    if scam_prob > 0.75:
        risk = "high"
    elif scam_prob > 0.4:
        risk = "medium"
    else:
        risk = "low"
        
    return {
        "label": label,
        "confidence": round(scam_prob, 4),
        "risk": risk
    }

if __name__ == "__main__":
    test_texts = [
        "Hello sir, your bank account has been blocked, please verify OTP.",
        "Hey bro are you free tonight?",
        "This is customer support, confirm your Emirates ID now.",
        "Can you pick up milk on the way home?"
    ]
    for text in test_texts:
        res = predict_chunk(text)
        print(f"[{res['risk'].upper()}] {res['label']} ({res['confidence']:.2%}): {text}")
