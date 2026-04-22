import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, pipeline
from safetensors.torch import load_file

# ==================== BiLSTM Model Definition ====================
class BiLSTMScamDetector(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=768, hidden_dim=768, output_dim=2):
        super(BiLSTMScamDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        output = self.fc(last_hidden)
        return output

# ==================== Load BiLSTM Model ====================
print("[*] Loading BiLSTM Scam Detection Model...")
bilstm_model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Load model weights from safetensors
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.safetensors')
    model_weights = load_file(model_path)
    
    # Initialize model architecture
    bilstm_model = BiLSTMScamDetector(vocab_size=30522, embedding_dim=768, 
                                      hidden_dim=768, output_dim=2)
    bilstm_model.load_state_dict(model_weights)
    bilstm_model.to(device)
    bilstm_model.eval()
    print("    ✓ BiLSTM model loaded successfully from model.safetensors")
except Exception as e:
    print(f"    ✗ Warning: Failed to load BiLSTM model ({e})")
    bilstm_model = None

# Load tokenizer
try:
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'DistillBertini', 'files', 'model')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    print("    ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"    ✗ Warning: Failed to load tokenizer ({e})")
    tokenizer = None

# Load ASR pipeline for transcription
print("[*] Loading speech recognition pipeline (openai/whisper-tiny)...")
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    print("    ✓ ASR pipeline loaded successfully")
except Exception as e:
    print(f"    ✗ Warning: Failed to load ASR pipeline ({e})")
    asr_pipeline = None

# ==================== Keyword Lists ====================
CRITICAL_KEYWORDS = [
    "deported", "deport", "mrets", "emirates id", "arrest", "warrant", 
    "police", "penalty", "fine", "suspended", "urgent", "immediately", "action required"
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

# ==================== Transcription Function ====================
def transcribe(y, sr):
    """Convert audio chunk to text using Whisper ASR."""
    if asr_pipeline is None:
        return ""
    try:
        result = asr_pipeline({"array": y, "sampling_rate": sr}, 
                            generate_kwargs={"task": "transcribe", "language": "en"})
        return result.get('text', '').strip()
    except Exception as e:
        print(f"      [ASR Error]: {e}")
        return ""

# ==================== BiLSTM Inference Function ====================
def bilstm_inference(transcript):
    """Run BiLSTM model inference on transcript."""
    if bilstm_model is None or tokenizer is None:
        return None
    
    try:
        # Tokenize input (max_length=512 to match model training)
        inputs = tokenizer(transcript, max_length=512, truncation=True, 
                          padding='max_length', return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = bilstm_model(input_ids)
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)
        scam_probability = probabilities[0, 1].item()  # Class 1 = scam
        non_scam_probability = probabilities[0, 0].item()  # Class 0 = non_scam
        
        predicted_label = "scam" if scam_probability > 0.5 else "non_scam"
        
        return {
            'scam_probability': scam_probability,
            'non_scam_probability': non_scam_probability,
            'label': predicted_label,
            'logits': logits[0].cpu().numpy()
        }
    except Exception as e:
        print(f"      [BiLSTM Error]: {e}")
        return None

# ==================== Text Model (Main Detection Function) ====================
def text_model(transcript):
    """
    Analyze transcript using BiLSTM model + keyword detection.
    Returns: (score, inference_text, suspicious_tokens)
    """
    if not transcript:
        return 0.0, "No speech detected", []
    
    transcript_lower = transcript.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in transcript_lower]
    found_critical = [kw for kw in CRITICAL_KEYWORDS if kw in transcript_lower]
    
    # ==================== BiLSTM Model Inference ====================
    bilstm_result = bilstm_inference(transcript)
    
    if bilstm_result is not None:
        model_score = bilstm_result['scam_probability']
        model_label = bilstm_result['label']
        
        # Adaptive keyword boosting
        keyword_boost = 0.0
        if found_critical:
            keyword_boost = 0.25  # Moderate boost for critical keywords
        elif len(found_keywords) >= 4:
            keyword_boost = 0.15
        elif len(found_keywords) >= 2:
            keyword_boost = 0.08
        
        # Final score: combine model + keywords
        final_score = min(model_score + keyword_boost, 1.0)
        
        # Build inference string
        keyword_str = f" | Keywords: {', '.join(found_keywords[:5])}" if found_keywords else ""
        critical_str = f" | ⚠ CRITICAL: {', '.join(found_critical)}" if found_critical else ""
        inference = (f"BiLSTM: {model_label.upper()} "
                    f"(Model: {model_score:.3f}, Boosted: {final_score:.3f}){critical_str}{keyword_str}")
        
        # Extract suspicious tokens (top keywords + model info)
        suspicious_tokens = []
        if found_critical:
            suspicious_tokens.extend([(kw, 1.0) for kw in found_critical[:3]])
        if found_keywords:
            suspicious_tokens.extend([(kw, 0.7 + (model_score * 0.2)) for kw in found_keywords[:5]])
        
        suspicious_tokens = sorted(suspicious_tokens, key=lambda x: x[1], reverse=True)[:10]
        
        return final_score, inference, suspicious_tokens
    
    # ==================== Fallback: Keyword-Only Detection ====================
    print("      [*] Using keyword-based fallback detection")
    score = 0.0
    tokens = []
    
    if found_critical:
        score = min(0.8 + (len(found_critical) * 0.1), 1.0)
        inference = f"CRITICAL Keywords Detected: {', '.join(found_critical)}"
        tokens = [(kw, 1.0) for kw in found_critical[:5]]
    elif found_keywords:
        score = min(len(found_keywords) * 0.15, 1.0)
        inference = f"Suspicious Keywords: {', '.join(found_keywords[:5])}"
        tokens = [(kw, 0.7) for kw in found_keywords[:5]]
    else:
        inference = "No suspicious content detected"
        score = 0.1
    
    return score, inference, tokens

