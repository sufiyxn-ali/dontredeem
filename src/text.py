import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from transformers import pipeline

# ==================== BiLSTM Model Definition ====================
class BiLSTMScamDetector(nn.Module):
    """BiLSTM with attention for scam detection (98.33% accuracy)"""
    def __init__(self, vocab_size=4729, embedding_dim=128, hidden_dim=256, output_dim=2):
        super(BiLSTMScamDetector, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True, dropout=0.1)
        
        # Classification head (individual layers for compatibility)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch, seq_len, hidden*2)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, hidden*2)
        
        # Use attention-weighted output
        weights = torch.softmax(torch.norm(attn_out, dim=2, keepdim=True), dim=1)
        weighted_out = (attn_out * weights).sum(dim=1)  # (batch, hidden*2)
        
        # Classification
        x = self.relu(self.fc1(weighted_out))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)
        return output


# ==================== Custom Tokenizer ====================
class ScamDetectionTokenizer:
    """Custom lightweight tokenizer for scam detection (98 KB vs 268 MB BERT)"""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.special_tokens = {}
        
    def load(self, filepath):
        """Load tokenizer from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.token_to_id = data.get('token_to_id', {})
            self.id_to_token = data.get('id_to_token', {})
            self.vocab_size = data.get('vocab_size', len(self.token_to_id))
            self.special_tokens = data.get('special_tokens', {})
    
    def encode(self, text, max_length=512):
        """Encode text to token IDs"""
        tokens = []
        words = text.lower().split()
        
        for word in words:
            # Check special markers
            is_urgent = any(marker in text.lower() for marker in ['urgent', 'immediately', 'action required'])
            is_money = any(marker in text.lower() for marker in ['transfer', 'bank', 'payment', 'money', 'card'])
            is_threat = any(marker in text.lower() for marker in ['arrest', 'deport', 'police', 'suspend'])
            is_verify = any(marker in text.lower() for marker in ['verify', 'confirm', 'update', 'validate'])
            is_personal = any(marker in text.lower() for marker in ['ssn', 'password', 'otp', 'id', 'name'])
            is_account = any(marker in text.lower() for marker in ['account', 'blocked', 'compromised', 'breached'])
            
            # Add special tokens (get ID from special_tokens dict)
            if is_urgent and '[URGENT]' in self.special_tokens:
                tokens.append(self.special_tokens['[URGENT]'])
            if is_money and '[MONEY]' in self.special_tokens:
                tokens.append(self.special_tokens['[MONEY]'])
            if is_threat and '[THREAT]' in self.special_tokens:
                tokens.append(self.special_tokens['[THREAT]'])
            if is_verify and '[VERIFY]' in self.special_tokens:
                tokens.append(self.special_tokens['[VERIFY]'])
            if is_personal and '[PERSONAL]' in self.special_tokens:
                tokens.append(self.special_tokens['[PERSONAL]'])
            if is_account and '[ACCOUNT]' in self.special_tokens:
                tokens.append(self.special_tokens['[ACCOUNT]'])
            
            # Add word token (1 = [UNK] for unknown tokens)
            token_id = self.token_to_id.get(word, 1)
            tokens.append(token_id)
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return torch.tensor([tokens], dtype=torch.long)  # Return as tensor with batch dimension


# ==================== Model Loader ====================
class ScamDetectionModel:
    """Unified loader for BiLSTM model + tokenizer"""
    
    def __init__(self, model_dir='models/DistillBertini/files/model'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = {}
        self.model_dir = model_dir
        
        print(f"[*] Initializing Scam Detection Model (device: {self.device})")
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load trained BiLSTM model"""
        try:
            # Look for model file (best_model.pt or bilstm_model.pt)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_paths = [
                os.path.join(root_dir, self.model_dir, 'bilstm_model.pt'),
                os.path.join(root_dir, self.model_dir, 'best_model.pt'),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print(f"    [!] No model file found. Checked: {model_paths}")
                return
            
            # Load config
            config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Initialize and load model
            vocab_size = self.config.get('vocab_size', 4729)
            embedding_dim = self.config.get('embedding_dim', 128)
            hidden_dim = self.config.get('hidden_dim', 256)
            
            self.model = BiLSTMScamDetector(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=2
            )
            
            # Load checkpoint (may have wrapper with model_state_dict, optimizer_state_dict, etc.)
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"    [OK] BiLSTM model loaded: {os.path.basename(model_path)} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"    [!] Failed to load model: {e}")
            self.model = None
    
    def _load_tokenizer(self):
        """Load custom tokenizer"""
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tokenizer_path = os.path.join(root_dir, self.model_dir, 'scam_tokenizer.pkl')
            
            if not os.path.exists(tokenizer_path):
                print(f"    [!] Tokenizer not found: {tokenizer_path}")
                return
            
            self.tokenizer = ScamDetectionTokenizer()
            self.tokenizer.load(tokenizer_path)
            print(f"    [OK] Tokenizer loaded: {os.path.basename(tokenizer_path)} (vocab: {len(self.tokenizer.token_to_id)})")
        except Exception as e:
            print(f"    [!] Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def predict(self, text):
        """Get scam probability for text"""
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            with torch.no_grad():
                input_ids = self.tokenizer.encode(text)
                input_ids = input_ids.to(self.device)
                logits = self.model(input_ids)
                probs = torch.softmax(logits, dim=1)
                scam_prob = probs[0, 1].item()
                return scam_prob
        except Exception as e:
            print(f"      [Error] Prediction failed: {e}")
            return None


# ==================== Initialize Global Model ====================
import sys
sys.path.append(os.path.dirname(__file__))
import minilm_infer

scam_detector = None
asr_pipeline = None

try:
    # Warm up MiniLM
    minilm_infer.load_model()
    scam_detector = minilm_infer
except Exception as e:
    print(f"[!] Warning: MiniLM Scam detector initialization failed: {e}")

# Load ASR pipeline for transcription
print("[*] Loading speech recognition pipeline (openai/whisper-tiny)...")
try:
    asr_device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=asr_device)
    print(f"    [OK] ASR pipeline loaded (device: {'GPU' if asr_device == 0 else 'CPU'})")
except Exception as e:
    print(f"    [!] Warning: Failed to load ASR pipeline: {e}")
    asr_pipeline = None

# ==================== Keyword Detection ====================
CRITICAL_KEYWORDS = {
    'deported', 'deport', 'arrest', 'warrant', 'police', 'penalty', 
    'fine', 'suspended', 'jail', 'prosecution', 'legal action',
    'emirates id', 'mrets', 'expired'  # UAE-specific scam patterns
}

SCAM_INDICATORS = {
    'urgent': 0.12, 'immediately': 0.12, 'action required': 0.15,
    'bank': 0.05, 'transfer': 0.08, 'otp': 0.15, 'password': 0.15,
    'account': 0.05, 'blocked': 0.10, 'compromised': 0.10,
    'verify': 0.05, 'confirm': 0.05, 'click link': 0.15, 
    'update information': 0.10, 'virus': 0.10, 'malware': 0.10,
    'refund': 0.10, 'prize': 0.15, 'won': 0.10, 'congratulations': 0.08
}

# ==================== Legitimacy Signals ====================
STRONG_LEGITIMACY_PHRASES = [
    # Formal business/academic emails
    'pleased to inform',
    'we are pleased',
    'we were impressed',
    'impressed with your',
    'schedule an interview',
    'interview with',
    'position in our',
    'lab position',
    'research assistant',
    'we look forward',
    'looking forward',
    'our team',
    'our department',
    'our company',
    'our organization',
    'position available',
    'available position',
    'thank you for applying',
    'thank you for your',
    'application',
    'recruiter',
    'hr department',
    'human resources',
    'background check',
    'opportunity',
    'let us know',
    'let me know',
    'availability',
    'convenient time',
    'speak with you',
    'talking with you',
    'hearing from you',
    'this position',
    'your qualifications',
    'your background',
    'experience',
    'credentials',
    'impressed',
    'congratulations',  # "Congratulations on being selected" vs "Congratulations you won"
    'selected for',
    'qualified for',
    'meeting',
    'discussion',
]

WEAK_LEGITIMACY_PHRASES = [
    'on our system',
    'your dashboard',
    'your profile',
    'through our app',
    'mobile app',
    'update your information',
    'avoid delays',
    'avoid issues',
    'processing delay',
    'account settings'
]

OFFICIAL_SENDER_PATTERNS = [
    'john hopkins',
    'johns hopkins',
    'university',
    'college',
    'research institute',
    'laboratory',
    'department',
    '.edu',
    'recruiting',
    'recruitment'
]

def _detect_keywords(text):
    """Extract detected keywords with context awareness for false positive reduction"""
    text_lower = text.lower()
    detected = []
    critical = []
    
    # Check for different legitimacy levels
    has_strong_legitimacy = any(signal in text_lower for signal in STRONG_LEGITIMACY_PHRASES)
    has_weak_legitimacy = any(signal in text_lower for signal in WEAK_LEGITIMACY_PHRASES)
    has_official_sender = any(pattern in text_lower for pattern in OFFICIAL_SENDER_PATTERNS)
    
    # Dangerous scam action keywords - what scammers ask for
    scam_dangerous_actions = [
        'send ',
        'provide ',
        'give me',
        'tell me',
        'click link',
        'click here',
        'verify via',
        'via sms',
        'via email',
        'call us',
        'text us'
    ]
    
    has_dangerous_action = any(action in text_lower for action in scam_dangerous_actions)
    
    # Special case: "congratulations" is only legitimate if NOT paired with dangerous actions
    # If it says "congratulations... send us your passport/SSN" = SCAM
    if 'congratulations' in text_lower and has_dangerous_action:
        # This is likely a fake job offer scam - override legitimacy
        has_strong_legitimacy = False
        has_weak_legitimacy = False
    
    has_legitimacy_context = has_strong_legitimacy or has_weak_legitimacy or has_official_sender
    
    # Context-aware keyword detection
    uae_scam_keywords = {
        'emirates id': 1.0,
        'mrets': 1.0,
        'deported': 1.0,
        'deport': 1.0,
        'warrant': 1.0,
        'arrest': 1.0,
        'police': 0.9,
        'suspended': 0.8,
    }
    
    # Add UAE keywords (always critical unless legitimacy signal)
    for keyword, weight in uae_scam_keywords.items():
        if keyword in text_lower:
            # If it's "expired" + legitimate context, don't flag it
            if keyword == 'expired' and has_legitimacy_context:
                pass  # Skip - this is legitimate
            else:
                critical.append(keyword)
    
    # Add other critical keywords with context
    for keyword in CRITICAL_KEYWORDS:
        if keyword in text_lower:
            # "passport" is only critical if asked to SEND it, not just update it
            if keyword == 'passport':
                if has_dangerous_action and not has_legitimacy_context:
                    critical.append(keyword)
            # "expired" is only critical with dangerous action context
            elif keyword == 'expired':
                if has_dangerous_action and not has_legitimacy_context:
                    critical.append(keyword)
            else:
                # All other critical keywords are always flagged
                critical.append(keyword)
    
    # Extract suspicious indicators (with reduced weight for legitimate contexts)
    for indicator, weight in SCAM_INDICATORS.items():
        if indicator in text_lower:
            # Reduce weight if strong legitimacy present AND no dangerous actions
            if has_strong_legitimacy and not has_dangerous_action:
                adjusted_weight = weight * 0.3
            elif has_weak_legitimacy and not has_dangerous_action:
                adjusted_weight = weight * 0.5
            else:
                adjusted_weight = weight
            detected.append((indicator, adjusted_weight))
    
    return list(set(critical)), detected  # Remove duplicates


# ==================== Transcription Function ====================
def transcribe(y, sr):
    """Convert audio to text using Whisper ASR"""
    if asr_pipeline is None:
        return ""
    
    try:
        result = asr_pipeline({"array": y, "sampling_rate": sr}, 
                            generate_kwargs={"task": "transcribe", "language": "en"})
        return result.get('text', '').strip()
    except Exception as e:
        print(f"      [ASR Error]: {e}")
        return ""


# ==================== Main Text Analysis Function ====================
def text_model(transcript):
    """
    Multimodal scam detection analysis with false positive reduction
    
    Returns:
        (score, analysis_text, suspicious_tokens)
    """
    if not transcript or not isinstance(transcript, str):
        return 0.0, "No speech detected", []
    
    # Detect keywords and critical indicators
    critical_keywords, scam_indicators = _detect_keywords(transcript)
    
    # Check for different legitimacy levels
    text_lower = transcript.lower()
    has_strong_legitimacy = any(signal in text_lower for signal in STRONG_LEGITIMACY_PHRASES)
    has_weak_legitimacy = any(signal in text_lower for signal in WEAK_LEGITIMACY_PHRASES)
    has_official_sender = any(pattern in text_lower for pattern in OFFICIAL_SENDER_PATTERNS)
    
    # Dangerous actions that override legitimacy signals
    scam_dangerous_actions = [
        'send ',
        'provide ',
        'give me',
        'tell me',
        'click link',
        'click here',
        'verify via',
        'via sms',
        'via email',
        'call us',
        'text us'
    ]
    has_dangerous_action = any(action in text_lower for action in scam_dangerous_actions)
    
    # CRITICAL: "congratulations" is ONLY legitimate if NOT paired with dangerous actions
    # Scammers abuse congratulations messages with "send us your passport/SSN/bank details"
    if has_dangerous_action:
        if 'congratulations' in text_lower or 'congratulation' in text_lower:
            # This is a fake job offer or prize scam - override all legitimacy signals
            has_strong_legitimacy = False
            has_weak_legitimacy = False
    
    has_any_legitimacy = has_strong_legitimacy or has_weak_legitimacy or has_official_sender
    
    # Get MiniLM model prediction
    model_score = 0.5
    if scam_detector is not None:
        try:
            res = scam_detector.predict_chunk(transcript)
            model_score = res['confidence']
        except Exception as e:
            print(f"      [Error] MiniLM prediction failed: {e}")
    
    # ===== FALSE POSITIVE REDUCTION LOGIC =====
    # If strong legitimacy signals are present, cap the BiLSTM score
    if has_strong_legitimacy:
        # Strong signals like "schedule an interview" + "impressed with your background"
        # should override a high BiLSTM score
        if model_score > 0.3:
            print(f"      [FP Reduction] Strong legitimacy detected. Capping BiLSTM: {model_score:.4f} -> 0.15")
            model_score = 0.15  # Cap at very low score
    elif has_weak_legitimacy and model_score > 0.3:
        # Weak signals: moderate reduction
        print(f"      [FP Reduction] Weak legitimacy detected. Reducing BiLSTM: {model_score:.4f} -> {model_score * 0.4:.4f}")
        model_score = model_score * 0.4
    
    # Confidence adjustment: Very high BiLSTM scores (>0.95) are often false positives
    # Regression to the mean approach
    if model_score > 0.95 and has_any_legitimacy:
        print(f"      [Confidence Adjustment] High BiLSTM score with legitimacy context")
        model_score = 0.35
    elif model_score > 0.90 and not critical_keywords:
        # High score but no critical keywords detected?
        # This suggests model overconfidence
        print(f"      [Confidence Adjustment] High BiLSTM score but no critical keywords")
        model_score = min(model_score * 0.5, 0.45)
    
    # Keyword-based boosting (with legitimacy context)
    keyword_boost = 0.0
    
    # Check for dangerous actions paired with suspicious keywords
    if has_dangerous_action and len(scam_indicators) >= 2:
        # "send us" + "immediately" + "bank" = RED FLAG
        keyword_boost = 0.27  # Significant boost for dangerous patterns
        model_score = max(model_score, 0.35)
    elif critical_keywords:
        # Critical keywords are strong signals
        if has_any_legitimacy:
            # Reduced boost if this sounds like legitimate business
            keyword_boost = 0.05  # Very light boost only
            model_score = max(model_score, 0.15)  # Lower baseline
        else:
            # No legitimacy signal = real alert
            keyword_boost = 0.30
            model_score = max(model_score, 0.40)
    elif len(scam_indicators) >= 4:
        if has_any_legitimacy:
            keyword_boost = 0.05
        else:
            keyword_boost = 0.15
    elif len(scam_indicators) >= 2:
        if has_any_legitimacy:
            keyword_boost = 0.02
        else:
            keyword_boost = 0.08
    
    # Combine scores
    final_score = min(model_score + keyword_boost, 1.0)
    
    # Build analysis text
    analysis_parts = []
    
    if scam_detector is not None:
        analysis_parts.append(f"MiniLM: {model_score:.1%}")
    else:
        analysis_parts.append("MiniLM: N/A")
    
    if has_strong_legitimacy:
        analysis_parts.append("[Legitimate Business Email]")
    elif has_weak_legitimacy:
        analysis_parts.append("[Partial Legitimacy Signals]")
    
    if critical_keywords:
        analysis_parts.append(f"⚠ CRITICAL: {', '.join(critical_keywords[:3])}")
    
    if scam_indicators:
        top_keywords = sorted(scam_indicators, key=lambda x: x[1], reverse=True)[:3]
        keyword_names = [kw for kw, _ in top_keywords]
        analysis_parts.append(f"Keywords: {', '.join(keyword_names)}")
    
    analysis_text = " | ".join(analysis_parts)
    
    # Build suspicious tokens
    suspicious_tokens = []
    if critical_keywords:
        suspicious_tokens.extend([(kw, 1.0) for kw in critical_keywords[:3]])
    if scam_indicators:
        suspicious_tokens.extend(scam_indicators[:5])
    
    suspicious_tokens = sorted(suspicious_tokens, key=lambda x: x[1], reverse=True)[:10]
    
    return final_score, analysis_text, suspicious_tokens

