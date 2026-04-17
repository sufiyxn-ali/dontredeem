"""
=============================================================================
DistilBERT Fine-Tuning for Scam Detection
=============================================================================
Reads the preprocessed train.csv / test.csv produced by preprocess_dataset.py
and fine-tunes distilbert-base-uncased as a binary classifier.

Output:
  ./model/          – saved HuggingFace model + tokenizer
  ./model/metrics.json – eval results

Usage (single GPU or CPU):
  python train_distilbert.py \
      --train_csv output/train.csv \
      --test_csv  output/test.csv  \
      --model_dir model/           \
      --epochs    4                \
      --batch     16               \
      --lr        2e-5

GPU recommended. The script auto-detects CUDA / MPS / CPU.
=============================================================================
"""

import os
import json
import argparse
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 256   # tokens — covers most scam call transcripts
WARMUP_RATIO = 0.06  # 6 % of steps used for LR warm-up


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScamDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Using CPU (training will be slow)")
    return device


def train_epoch(model, loader, optimizer, scheduler, device, epoch, total):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 50 == 0 or step == len(loader):
            avg = total_loss / step
            log.info(f"  Epoch {epoch}/{total}  step {step}/{len(loader)}  loss={avg:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        probs  = torch.softmax(out.logits, dim=1)[:, 1]   # P(scam)
        preds  = torch.argmax(out.logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "f1":        round(f1_score(y_true, y_pred),        4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred),    4),
        "roc_auc":   round(roc_auc_score(y_true, y_probs),  4),
    }
    return metrics, y_true, y_pred, y_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for scam detection")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv",  required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--epochs",    type=int,   default=4)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=2e-5)
    parser.add_argument("--max_len",   type=int,   default=MAX_LEN)
    args = parser.parse_args()

    device = get_device()

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)
    log.info(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = ScamDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        args.max_len,
    )
    test_ds = ScamDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        args.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    # Label metadata stored in config for inference
    model.config.id2label = {0: "non_scam", 1: "scam"}
    model.config.label2id = {"non_scam": 0, "scam": 1}

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    log.info(
        f"Training: epochs={args.epochs}  batch={args.batch}  "
        f"lr={args.lr}  steps={total_steps}  warmup={warmup_steps}"
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_f1     = 0.0
    best_epoch  = 0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs)
        log.info(f"Epoch {epoch} finished — avg train loss: {avg_loss:.4f}")

        metrics, y_true, y_pred, _ = evaluate(model, test_loader, device)
        log.info(
            f"  Eval → acc={metrics['accuracy']}  f1={metrics['f1']}  "
            f"prec={metrics['precision']}  rec={metrics['recall']}  "
            f"auc={metrics['roc_auc']}"
        )
        print(classification_report(y_true, y_pred, target_names=["non_scam", "scam"]))

        if metrics["f1"] > best_f1:
            best_f1      = metrics["f1"]
            best_epoch   = epoch
            best_metrics = metrics
            # Save best checkpoint
            os.makedirs(args.model_dir, exist_ok=True)
            model.save_pretrained(args.model_dir)
            tokenizer.save_pretrained(args.model_dir)
            log.info(f"  ✓ New best model saved (f1={best_f1})")

    # ── Final report ──────────────────────────────────────────────────────
    best_metrics["best_epoch"] = best_epoch
    best_metrics["total_epochs"] = args.epochs
    best_metrics["model"]      = MODEL_NAME
    best_metrics["max_len"]    = args.max_len

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(best_metrics, fh, indent=2)
    log.info(f"Metrics saved → {metrics_path}")
    log.info(f"Best F1 = {best_f1}  (epoch {best_epoch})")
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
