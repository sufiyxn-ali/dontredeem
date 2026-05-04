import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "Dataset/full_dataset.csv"
OUTPUT_DIR = "models/minilm"

class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    
    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.15, random_state=42, stratify=df['label'].tolist()
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}, Test size: {len(test_texts)}")

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    train_dataset = ScamDataset(train_encodings, train_labels)
    val_dataset = ScamDataset(val_encodings, val_labels)
    test_dataset = ScamDataset(test_encodings, test_labels)

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test Results: {test_results}")

    logger.info(f"Saving final model to {OUTPUT_DIR}/best")
    model.save_pretrained(f"{OUTPUT_DIR}/best")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
