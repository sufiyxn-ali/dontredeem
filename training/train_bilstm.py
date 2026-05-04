"""
BiLSTM Model Training for Scam Detection
- Desktop version (full model)
- Will be distilled for mobile deployment
- Includes validation and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import pickle

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path("models/DistillBertini/files/output")
MODEL_DIR = Path("models/DistillBertini/files/model")
MODEL_DIR.mkdir(exist_ok=True)


class ScamDetectionDataset(Dataset):
    """Dataset for scam detection with tokenization"""
    
    def __init__(self, csv_path, tokenizer, max_seq_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        
        # Tokenize
        token_ids, attention_mask = self.tokenizer.encode(text, add_scam_markers=True)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BiLSTMScamDetector(nn.Module):
    """
    Bidirectional LSTM for scam detection
    Architecture optimized for both accuracy and mobile deployment
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BiLSTMScamDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Attention
        if attention_mask is not None:
            # Convert attention_mask for multi-head attention
            attn_mask = (1 - attention_mask).bool()
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask)
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out
        
        # Global average pooling
        mask_expanded = attention_mask.unsqueeze(-1).float() if attention_mask is not None else 1
        pooled = (combined * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits


class ModelTrainer:
    """Training loop for BiLSTM"""
    
    def __init__(self, model, tokenizer, device=DEVICE, batch_size=32, learning_rate=0.001):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        self.best_val_accuracy = 0
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("BILSTM TRAINING")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(is_best=True)
                print(f"  ✓ Best model saved (accuracy: {val_accuracy:.4f})")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }
        
        path = MODEL_DIR / ('best_model.pt' if is_best else 'latest_model.pt')
        torch.save(checkpoint, path)
    
    def save_training_history(self):
        """Save training history for analysis"""
        history_file = OUTPUT_DIR / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def train_model():
    """Main training pipeline"""
    from build_tokenizer import ScamDetectionTokenizer, build_tokenizer_from_dataset
    
    # Step 1: Build tokenizer if not exists
    tokenizer_path = MODEL_DIR / 'scam_tokenizer.pkl'
    if not tokenizer_path.exists():
        print("Tokenizer not found. Building...\n")
        build_tokenizer_from_dataset()
    
    # Step 2: Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ScamDetectionTokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer.token_to_id)})\n")
    
    # Step 3: Create datasets
    print("Creating datasets...")
    train_dataset = ScamDetectionDataset(
        OUTPUT_DIR / 'train.csv',
        tokenizer
    )
    val_dataset = ScamDetectionDataset(
        OUTPUT_DIR / 'val.csv',
        tokenizer
    )
    test_dataset = ScamDetectionDataset(
        OUTPUT_DIR / 'test.csv',
        tokenizer
    )
    
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}\n")
    
    # Step 4: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Step 5: Create model
    print("Creating BiLSTM model...")
    model = BiLSTMScamDetector(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Device: {DEVICE}\n")
    
    # Step 6: Train model
    trainer = ModelTrainer(model, tokenizer, device=DEVICE, batch_size=32, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=10)
    
    # Step 7: Test
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    test_loss, test_accuracy = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    # Step 8: Save
    trainer.save_checkpoint()
    trainer.save_training_history()
    
    # Save model config
    model_config = {
        'vocab_size': len(tokenizer.token_to_id),
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'num_parameters': num_params,
        'test_accuracy': test_accuracy,
    }
    
    with open(MODEL_DIR / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✓ Training complete!")
    print(f"✓ Model saved to {MODEL_DIR}")
    print(f"✓ Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train_model()
