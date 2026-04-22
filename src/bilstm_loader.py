import os
import torch
import torch.nn as nn
import pickle
import json

class BiLSTMScamDetector(nn.Module):
    """BiLSTM with attention for scam detection"""
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
        
        # Classification head
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

class ScamDetectionTokenizer:
    """Custom lightweight tokenizer for scam detection"""
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
        # Check special markers on the full text once
        is_urgent = any(marker in text.lower() for marker in ['urgent', 'immediately', 'action required'])
        is_money = any(marker in text.lower() for marker in ['transfer', 'bank', 'payment', 'money', 'card'])
        is_threat = any(marker in text.lower() for marker in ['arrest', 'deport', 'police', 'suspend'])
        is_verify = any(marker in text.lower() for marker in ['verify', 'confirm', 'update', 'validate'])
        is_personal = any(marker in text.lower() for marker in ['ssn', 'password', 'otp', 'id', 'name'])
        is_account = any(marker in text.lower() for marker in ['account', 'blocked', 'compromised', 'breached'])
        
        for word in words:
            # Add special tokens
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
            
            token_id = self.token_to_id.get(word, 1) # 1 is UNK
            tokens.append(token_id)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return torch.tensor([tokens], dtype=torch.long)

class ScamDetectionModel:
    """Unified loader for BiLSTM model + tokenizer"""
    def __init__(self, model_dir='models/BiLSTM'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = {}
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(root_dir, model_dir)
        
        print(f"[*] Initializing Scam Detection Model (device: {self.device})")
        self._load_tokenizer()
        self._load_model()
        
    def _load_tokenizer(self):
        try:
            tokenizer_path = os.path.join(self.model_dir, 'scam_tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                self.tokenizer = ScamDetectionTokenizer()
                self.tokenizer.load(tokenizer_path)
            else:
                print(f"    [!] Tokenizer not found at {tokenizer_path}")
        except Exception as e:
            print(f"    [!] Tokenizer load error: {e}")

    def _load_model(self):
        try:
            model_paths = [
                os.path.join(self.model_dir, 'bilstm_model.pt'),
                os.path.join(self.model_dir, 'best_model.pt'),
                os.path.join(self.model_dir, 'latest_model.pt'),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print(f"    [!] No model file found in {self.model_dir}")
                return
            
            config_path = os.path.join(self.model_dir, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            vocab_size = self.config.get('vocab_size', 4729)
            if self.tokenizer and self.tokenizer.vocab_size > 0:
                vocab_size = self.tokenizer.vocab_size
                
            embedding_dim = self.config.get('embedding_dim', 128)
            hidden_dim = self.config.get('hidden_dim', 256)
            
            self.model = BiLSTMScamDetector(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=2
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print("    [OK] BiLSTM model loaded successfully.")
        except Exception as e:
            print(f"    [!] Model load error: {e}")

    def predict(self, text):
        if not self.model or not self.tokenizer:
            return None
        
        try:
            with torch.no_grad():
                input_ids = self.tokenizer.encode(text).to(self.device)
                logits = self.model(input_ids)
                probs = torch.softmax(logits, dim=1)
                scam_prob = probs[0, 1].item()
                return scam_prob
        except Exception as e:
            print(f"      [Error] Prediction failed: {e}")
            return None
