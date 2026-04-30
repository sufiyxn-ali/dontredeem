import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "minilm", "best")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "full_dataset.csv")

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

def main():
    print("Loading test dataset...")
    df = pd.read_csv(DATA_PATH).dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    
    # Same split as training
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.15, random_state=42, stratify=df['label'].tolist()
    )
    
    print(f"Loading model and tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    print("Evaluating...")
    encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1).numpy()
        scam_probs = probs[:, 1].numpy()
        
    acc = accuracy_score(test_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary')
    roc_auc = roc_auc_score(test_labels, scam_probs)
    conf_mat = confusion_matrix(test_labels, preds)
    
    print("-" * 40)
    print("EVALUATION METRICS")
    print("-" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=["Normal", "Scam"]))

if __name__ == "__main__":
    main()
