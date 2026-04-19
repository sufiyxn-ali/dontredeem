# Scam Detection — DistilBERT Pipeline

## Requirements

```
torch>=2.0.0
transformers>=4.38.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
```

Install:
```bash
pip install torch transformers scikit-learn pandas numpy
```

---

## File Layout

```
your_project/
├── data/
│   ├── metadata.csv          ← FTC Robocall dataset
│   ├── spam_ham.txt          ← SMS Spam Collection
│   └── dialogues.txt         ← DailyDialog
│
├── preprocess_dataset.py     ← Step 1: clean & merge data
├── train_distilbert.py       ← Step 2: fine-tune DistilBERT
├── infer.py                  ← Step 3: run inference / get T_t score
│
├── output/                   ← created by preprocess_dataset.py
│   ├── full_dataset.csv
│   ├── train.csv
│   ├── test.csv
│   └── preprocessing_report.txt
│
└── model/                    ← created by train_distilbert.py
    ├── config.json
    ├── pytorch_model.bin (or model.safetensors)
    ├── tokenizer_config.json
    ├── vocab.txt
    └── metrics.json
```

---

## Step 1 — Preprocess

```bash
python preprocess_dataset.py \
    --ftc_csv       data/metadata.csv \
    --sms_txt       data/spam_ham.txt \
    --dialog_txt    data/dialogues.txt \
    --output_dir    output/
```

What it does:
- FTC transcripts → label 1 (scam)
- SMS spam → label 1, SMS ham → label 0
- DailyDialog utterances → label 0
- Filters non-scam texts that contain scam keywords (to avoid label noise)
- Drops any text shorter than 3 words
- Balances classes via random undersampling
- Saves train.csv (90%) and test.csv (10%)

---

## Step 2 — Train

```bash
python train_distilbert.py \
    --train_csv output/train.csv \
    --test_csv  output/test.csv  \
    --model_dir model/           \
    --epochs    4                \
    --batch     16               \
    --lr        2e-5
```

- Auto-detects GPU (CUDA) / Apple Silicon (MPS) / CPU
- Saves best checkpoint (by F1) to model/
- Prints classification report after each epoch
- Saves model/metrics.json with final evaluation scores

Recommended hardware: GPU with ≥ 8 GB VRAM (or Google Colab free tier).

---

## Step 3 — Inference (T_t score)

### CLI
```bash
# Single text
python infer.py \
    --model_dir model/ \
    --text "Your Social Security number has been suspended. Press 1 now."

# File with one text per line
python infer.py \
    --model_dir model/ \
    --file      transcripts.txt \
    --json_out  scores.json
```

### Python API
```python
from infer import ScamScorer

scorer = ScamScorer("model/")

# Single score (T_t)
result = scorer.score("Congratulations! You have won a $1000 gift card.")
print(result.scam_probability)   # → 0.9714  (T_t score fed to multimodal system)
print(result.label)              # → "scam"
print(result.confidence)         # → "high"

# Batch
results = scorer.score_batch(["text1", "text2", ...])
```

**Output schema:**
```json
{
  "text": "Congratulations! You have won ...",
  "scam_probability": 0.9714,
  "label": "scam",
  "confidence": "high"
}
```

`scam_probability` is the **T_t score** — a value in [0, 1] representing
the model's confidence that the input text is a scam transcript.

---

## Notes on Data Format

### FTC metadata.csv
```
file_name,language,transcript,case_details,case_pdf
audio.wav,en,"Your Medicare benefits will expire...",…,…
```
Only the `transcript` column is used. Non-English rows are skipped.

### SMS spam_ham.txt
```
ham\tSmile in Pleasure...
spam\tPlease call our customer service...
```
Tab-separated: label first, then message.

### DailyDialog dialogues.txt
```
utterance1 __eou__ utterance2 __eou__\t<act>\t<emotion>\t<topic>
```
Tab-separated columns; only the first column (the dialogue) is parsed.
`__eou__` tokens split individual utterances.
