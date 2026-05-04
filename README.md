# DontRedeem Scam Detection System

Offline-first multimodal scam detection for recorded phone-call audio.

The final architecture uses the newer Codexx BiLSTM model as the primary text scam detector, restores the stronger baseline session logic, and keeps MiniLM only as an optional fallback for uncertain text cases. Gemma 4 has been removed.

## Pipeline

```text
.wav audio
  -> librosa 16 kHz loading
  -> optional local pyannote diarization
  -> 5 second windows
  -> local Faster Whisper ASR
  -> BiLSTM primary scam score
  -> optional MiniLM fallback near the BiLSTM boundary
  -> keyword and legitimacy-context rules
  -> audio heuristic score
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
models/pyannote/...                optional diarization
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

## Latest Benchmark

Run on the local 19-file audio benchmark after the final refactor:

```text
Accuracy: 73.68%
Precision: 68.75%
Recall: 100.00%
F1: 81.48%
Average processing time: 22.28s/file
```

The current configuration is conservative: it catches all benchmark scam files, but flags some safe files as suspicious.

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
