import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Model path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "minilm", "best")

# Global instances
_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            _model.eval()
        except Exception as e:
            print(f"Error loading MiniLM model from {MODEL_DIR}. Error: {e}")

def predict_chunk(text: str) -> dict:
    """
    Predicts if a ~5-second text chunk is a scam or normal.
    Returns: { "label": "scam" or "normal", "confidence": float, "risk": "low", "medium", or "high" }
    """
    load_model()
    if _model is None or _tokenizer is None:
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
