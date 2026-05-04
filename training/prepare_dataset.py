"""
Dataset preparation pipeline for BiLSTM scam detection
- Cleans text data
- Removes duplicates and noise
- Combines multiple data sources
- Creates balanced train/val/test splits
- Optimized for phone deployment
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# Configuration
OUTPUT_DIR = Path("models/DistillBertini/files/output")
DATA_OLD_DIR = Path("D:/ScamDetectProj/OLD/Scam & Non Scan")
OUTPUT_DIR.mkdir(exist_ok=True)

class TextCleaner:
    @staticmethod
    def clean_text(text):
        """Remove noise, placeholders, and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove placeholders like [Company], [Name], [Title], etc
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove __eou__ markers
        text = re.sub(r'__eou__', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove phone numbers (optional - keep for scam detection)
        # text = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\"\-]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text

    @staticmethod
    def is_valid(text, min_length=5):
        """Check if text is valid for training"""
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # Check minimum length
        if len(text) < min_length:
            return False
        
        # Reject text that's mostly placeholders
        if len(text) < 10:
            return False
        
        # Reject if too many non-alphanumeric characters
        alpha_count = sum(1 for c in text if c.isalnum())
        if alpha_count / len(text) < 0.5:
            return False
        
        return True

class DatasetBuilder:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.scam_texts = []
        self.legit_texts = []
        self.conversation_texts = []
    
    def load_scam_texts(self):
        """Load scam detection texts"""
        print("[1/5] Loading scam texts...")
        scam_file = DATA_OLD_DIR / "English_Scam.txt"
        
        if scam_file.exists():
            with open(scam_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Split by numbered items
                items = re.split(r'^\d+\.\s+', content, flags=re.MULTILINE)
                for item in items[1:]:
                    cleaned = self.cleaner.clean_text(item.strip())
                    if self.cleaner.is_valid(cleaned, min_length=10):
                        self.scam_texts.append(cleaned)
        
        print(f"   Loaded {len(self.scam_texts)} scam texts")
        return len(self.scam_texts)
    
    def load_legit_texts(self):
        """Load legitimate/non-scam texts"""
        print("[2/5] Loading legitimate texts...")
        legit_file = DATA_OLD_DIR / "English_NonScam.txt"
        
        if legit_file.exists():
            with open(legit_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    cleaned = self.cleaner.clean_text(line.strip())
                    if self.cleaner.is_valid(cleaned, min_length=10):
                        self.legit_texts.append(cleaned)
        
        print(f"   Loaded {len(self.legit_texts)} legitimate texts")
        return len(self.legit_texts)
    
    def load_conversation_data(self):
        """Load conversation data from CSV"""
        print("[3/5] Loading conversation data...")
        
        # Load BETTER30.csv
        csv_file = DATA_OLD_DIR / "BETTER30.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
                # Extract text column (usually first column)
                if len(df.columns) > 0:
                    text_col = df.iloc[:, 0]
                    for text in text_col:
                        cleaned = self.cleaner.clean_text(str(text))
                        if self.cleaner.is_valid(cleaned, min_length=10):
                            self.conversation_texts.append((cleaned, 0))  # Assume legitimate
                print(f"   Loaded {len(self.conversation_texts)} from BETTER30.csv")
            except Exception as e:
                print(f"   Warning: Could not load BETTER30.csv: {e}")
        
        # Load conversation with labels
        conv_file = DATA_OLD_DIR / "gen_conver_noIdentifier_1000.csv"
        if conv_file.exists():
            try:
                df = pd.read_csv(conv_file, encoding='utf-8', on_bad_lines='skip')
                if 'TEXT' in df.columns and 'LABEL' in df.columns:
                    for _, row in df.iterrows():
                        text = self.cleaner.clean_text(str(row['TEXT']))
                        # Map label: slightly_suspicious/suspicious -> 1, else -> 0
                        label = 1 if 'suspicious' in str(row['LABEL']).lower() else 0
                        if self.cleaner.is_valid(text, min_length=10):
                            self.conversation_texts.append((text, label))
                print(f"   Total conversation samples: {len(self.conversation_texts)}")
            except Exception as e:
                print(f"   Warning: Could not load conversation data: {e}")
    
    def balance_dataset(self):
        """Create balanced dataset"""
        print("[4/5] Balancing dataset...")
        
        # Create label for scam texts
        scam_labeled = [(text, 1) for text in self.scam_texts]
        legit_labeled = [(text, 0) for text in self.legit_texts]
        
        # Combine all data
        all_data = scam_labeled + legit_labeled + self.conversation_texts
        
        # Remove duplicates while preserving order
        seen = set()
        unique_data = []
        for text, label in all_data:
            if text not in seen:
                seen.add(text)
                unique_data.append((text, label))
        
        print(f"   Total unique samples: {len(unique_data)}")
        
        # Count by label
        scam_count = sum(1 for _, label in unique_data if label == 1)
        legit_count = len(unique_data) - scam_count
        print(f"   Scam: {scam_count}, Legitimate: {legit_count}")
        
        # Balance if needed
        if scam_count > legit_count * 3 or legit_count > scam_count * 3:
            print(f"   Imbalance detected. Balancing...")
            min_count = min(scam_count, legit_count)
            scam_data = [d for d in unique_data if d[1] == 1][:min_count]
            legit_data = [d for d in unique_data if d[1] == 0][:min_count]
            unique_data = scam_data + legit_data
            print(f"   Balanced to {len(unique_data)} samples")
        
        return unique_data
    
    def create_splits(self, data, train_ratio=0.7, val_ratio=0.15):
        """Create train/val/test splits"""
        print("[5/5] Creating train/val/test splits...")
        
        # Shuffle
        data_array = np.array(data, dtype=object)
        indices = np.random.permutation(len(data_array))
        data_array = data_array[indices]
        
        # Split
        n = len(data_array)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = data_array[:train_end]
        val_data = data_array[train_end:val_end]
        test_data = data_array[val_end:]
        
        print(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data, val_data, test_data):
        """Save splits to CSV"""
        for split_name, data in [('train.csv', train_data), ('val.csv', val_data), ('test.csv', test_data)]:
            df = pd.DataFrame(data, columns=['text', 'label'])
            filepath = OUTPUT_DIR / split_name
            df.to_csv(filepath, index=False)
            print(f"   Saved {split_name}: {len(df)} samples")
        
        # Save full dataset
        full_data = np.concatenate([train_data, val_data, test_data])
        df_full = pd.DataFrame(full_data, columns=['text', 'label'])
        df_full.to_csv(OUTPUT_DIR / 'full_dataset.csv', index=False)
        print(f"   Saved full_dataset.csv: {len(df_full)} samples")
    
    def generate_report(self, data):
        """Generate preprocessing report"""
        print("\n" + "="*60)
        print("PREPROCESSING REPORT")
        print("="*60)
        
        texts = [t for t, _ in data]
        labels = [l for _, l in data]
        
        stats = {
            "total_samples": len(data),
            "scam_samples": sum(1 for l in labels if l == 1),
            "legitimate_samples": sum(1 for l in labels if l == 0),
            "avg_text_length": np.mean([len(t.split()) for t in texts]),
            "min_text_length": min([len(t.split()) for t in texts]),
            "max_text_length": max([len(t.split()) for t in texts]),
        }
        
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Scam: {stats['scam_samples']} ({100*stats['scam_samples']/stats['total_samples']:.1f}%)")
        print(f"Legitimate: {stats['legitimate_samples']} ({100*stats['legitimate_samples']/stats['total_samples']:.1f}%)")
        print(f"Avg Text Length: {stats['avg_text_length']:.1f} tokens")
        print(f"Min/Max Length: {stats['min_text_length']}/{stats['max_text_length']} tokens")
        print("="*60 + "\n")
        
        # Save report
        with open(OUTPUT_DIR / 'preprocessing_report.txt', 'w') as f:
            f.write("SCAM DETECTION DATASET PREPROCESSING REPORT\n")
            f.write("="*60 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("="*60 + "\n")
        
        return stats
    
    def build(self):
        """Execute full pipeline"""
        print("\n" + "="*60)
        print("DATASET PREPARATION PIPELINE")
        print("="*60 + "\n")
        
        self.load_scam_texts()
        self.load_legit_texts()
        self.load_conversation_data()
        
        data = self.balance_dataset()
        train_data, val_data, test_data = self.create_splits(data)
        self.save_splits(train_data, val_data, test_data)
        
        full_data = np.concatenate([train_data, val_data, test_data])
        self.generate_report(full_data)

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
    print("✓ Dataset preparation complete!")
