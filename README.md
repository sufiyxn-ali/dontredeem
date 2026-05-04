# DontRedeem Scam Detection System

Offline-first multimodal scam detection for recorded phone-call audio.

The current architecture uses a BiLSTM model as the primary text scam detector, Faster Whisper for local speech-to-text, Pyannote for speaker diarization, HuBERT speech emotion recognition when the local model is present, and MiniLM only as an optional fallback for uncertain text cases. Gemma 4 has been removed.

## Pipeline

```text
.wav audio
  -> librosa 16 kHz loading
  -> local Pyannote diarization when available
  -> 5 second windows
  -> HuBERT SER + acoustic urgency features
  -> local Faster Whisper ASR
  -> BiLSTM primary scam score
  -> optional MiniLM fallback near the BiLSTM boundary
  -> keyword and legitimacy-context rules
  -> metadata score
  -> weighted fusion
  -> asymmetric EMA ratchet
  -> Safe / Suspicious / Likely Scam
```

## Working Model Paths

```text
models/BiLSTM/best_model.pt
models/BiLSTM/scam_tokenizer.pkl
models/faster-whisper-small
models/minilm/best                 optional fallback
models/pyannote/...                speaker diarization
models/hubert-large-superb-er      speech emotion recognition
```

## Quick Start

```bash
pip install -r requirements.txt
python src/main.py "data/sample_ScamConvo.wav" --metadata data/metadata.txt
```

Text-only smoke test:

```bash
python utils_and_tests/test_bilstm.py
```

Full benchmark:

```bash
python utils_and_tests/benchmark_all.py
```

Disable MiniLM fallback:

```bash
set ENABLE_MINILM_FALLBACK=0
```

## Current Benchmark

Stable benchmark on the local 19-file audio test set:

```text
Total files: 19
Scam files: 11
Safe files: 8
Accuracy: 73.68%
Precision: 68.75%
Recall: 100.00%
F1-score: 81.48%
Average processing time: 22.28s/file
```

Confusion matrix:

```text
                 Predicted Safe | Predicted Scam
Actual Safe   |        3       |        5
Actual Scam   |        0       |       11
```

The current configuration is conservative: it catches all benchmark scam files, but flags some safe files as suspicious. This behavior is intentional for the current warning-oriented prototype, where missed scam calls are considered more harmful than cautionary false positives.

Verified local components:

```text
BiLSTM text detector        working
Faster Whisper ASR          working
Pyannote diarization        working with local model files
HuBERT SER                  working when models/hubert-large-superb-er is present
MiniLM fallback             optional, boundary/failure cases only
```

## Documentation

Current architecture and model status:

```text
docs/PROJECT_INFO_CURRENT.md
```

Current working state and benchmark results:

```text
docs/CURRENT_WORKING_RESULTS.md
```

Baseline-vs-Codexx comparison and final design decisions:

```text
docs/BASELINE_CODEXX_COMPARISON.md
```
