"""
Lightweight Tokenizer for Scam Detection - Optimized for Mobile Deployment
- Simple vocab-based tokenizer (not BERT)
- Small file size (~50KB vs 268MB for DistilBERT)
- Fast inference on mobile devices
- Includes special tokens for scam detection patterns
"""

import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple

OUTPUT_DIR = Path("models/DistillBertini/files/model")
OUTPUT_DIR.mkdir(exist_ok=True)

class ScamDetectionTokenizer:
    """Lightweight tokenizer optimized for scam detection on mobile"""
    
    def __init__(self, vocab_size=5000, max_seq_length=256):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Special tokens for scam patterns
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[URGENT]': 4,      # Scam indicator
            '[MONEY]': 5,       # Money-related
            '[THREAT]': 6,      # Threat language
            '[VERIFY]': 7,      # Verify/confirm
            '[PERSONAL]': 8,    # Personal info request
            '[ACCOUNT]': 9,     # Account-related
        }
        
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.word_freq = Counter()
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)
        
        # Add top vocab_size words
        common_words = self.word_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        current_id = len(self.special_tokens)
        for word, freq in common_words:
            self.token_to_id[word] = current_id
            self.id_to_token[current_id] = word
            current_id += 1
        
        print(f"Vocabulary size: {len(self.token_to_id)}")
        print(f"Special tokens: {len(self.special_tokens)}")
        print(f"Regular vocab: {len(self.token_to_id) - len(self.special_tokens)}")
        
        return len(self.token_to_id)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        # Simple whitespace tokenization
        tokens = text.split()
        return tokens
    
    def _add_scam_markers(self, tokens: List[str]) -> List[str]:
        """Add special tokens for scam patterns"""
        scam_keywords = {
            '[URGENT]': ['urgent', 'immediately', 'now', 'asap', 'immediately', 'hurry', 'emergency'],
            '[MONEY]': ['money', 'payment', 'transfer', 'fee', 'charge', 'credit', 'debit', 'cash', 'wire'],
            '[THREAT]': ['legal', 'lawsuit', 'suspended', 'blocked', 'cancel', 'illegal', 'criminal', 'warrant'],
            '[VERIFY]': ['verify', 'confirm', 'validate', 'authenticate', 'check', 'approve', 'authorization'],
            '[PERSONAL]': ['ssn', 'social security', 'password', 'pin', 'credit card', 'bank account', 'email'],
            '[ACCOUNT]': ['account', 'login', 'username', 'access', 'credentials']
        }
        
        text_lower = ' '.join(tokens).lower()
        enhanced_tokens = []
        
        for token in tokens:
            enhanced_tokens.append(token)
            
            # Check for scam patterns
            for marker, keywords in scam_keywords.items():
                if token in keywords:
                    enhanced_tokens.append(marker)
                    break
        
        return enhanced_tokens
    
    def encode(self, text: str, add_scam_markers=True) -> Tuple[List[int], List[int]]:
        """
        Encode text to token IDs
        Returns: (token_ids, attention_mask)
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add scam detection markers
        if add_scam_markers:
            tokens = self._add_scam_markers(tokens)
        
        # Add CLS token at start
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id['[UNK]'])
        
        # Truncate if needed
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        
        # Pad if needed
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < self.max_seq_length:
            token_ids.append(self.token_to_id['[PAD]'])
            attention_mask.append(0)
        
        return token_ids, attention_mask
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, path: Path = None):
        """Save tokenizer to disk (very small file)"""
        if path is None:
            path = OUTPUT_DIR / 'scam_tokenizer.pkl'
        
        tokenizer_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'special_tokens': self.special_tokens,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        # Also save as JSON for inspection
        json_path = path.with_suffix('.json')
        json_tokenizer = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'special_tokens': self.special_tokens,
        }
        with open(json_path, 'w') as f:
            json.dump(json_tokenizer, f, indent=2)
        
        file_size_kb = path.stat().st_size / 1024
        print(f"✓ Tokenizer saved: {path}")
        print(f"  File size: {file_size_kb:.1f} KB (vs 268 MB for DistilBERT)")
        
        return path
    
    @staticmethod
    def load(path: Path = None):
        """Load tokenizer from disk"""
        if path is None:
            path = OUTPUT_DIR / 'scam_tokenizer.pkl'
        
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = ScamDetectionTokenizer(
            vocab_size=tokenizer_data['vocab_size'],
            max_seq_length=tokenizer_data['max_seq_length']
        )
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        
        return tokenizer


def build_tokenizer_from_dataset():
    """Build and save tokenizer from preprocessed dataset"""
    import pandas as pd
    
    print("="*60)
    print("TOKENIZER BUILDER")
    print("="*60 + "\n")
    
    # Load training data
    dataset_csv = Path("models/DistillBertini/files/output/full_dataset.csv")
    if not dataset_csv.exists():
        print(f"Error: Dataset not found at {dataset_csv}")
        print("Run prepare_dataset.py first!")
        return
    
    print(f"[1/3] Loading dataset from {dataset_csv}...")
    df = pd.read_csv(dataset_csv)
    texts = df['text'].tolist()
    print(f"   Loaded {len(texts)} texts")
    
    # Create tokenizer
    print(f"\n[2/3] Building tokenizer...")
    tokenizer = ScamDetectionTokenizer(vocab_size=5000, max_seq_length=256)
    tokenizer.build_vocab(texts)
    
    # Test tokenization
    print(f"\n[3/3] Testing tokenizer...")
    test_texts = [
        "Your account has been suspended. Verify immediately by clicking the link.",
        "Thank you for calling. Your order has been shipped.",
        "URGENT: Legal action filed on your social security number. Wire money now!"
    ]
    
    for text in test_texts:
        token_ids, attention_mask = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        print(f"\n   Original: {text[:60]}...")
        print(f"   Encoded: {len(token_ids)} tokens")
        print(f"   Decoded: {decoded[:60]}...")
    
    # Save tokenizer
    print(f"\n\nSaving tokenizer...")
    tokenizer.save()
    
    # Create config file for model
    config = {
        'tokenizer_type': 'scam_detection',
        'vocab_size': tokenizer.vocab_size,
        'max_seq_length': tokenizer.max_seq_length,
        'model_type': 'bilstm',
        'special_tokens': tokenizer.special_tokens,
    }
    
    config_path = OUTPUT_DIR / 'tokenizer_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}\n")

if __name__ == "__main__":
    build_tokenizer_from_dataset()
