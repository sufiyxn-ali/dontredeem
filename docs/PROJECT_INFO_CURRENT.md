# DontRedeem Project Information

Last updated: 2026-05-05

## 1. Project Summary

DontRedeem is a local multimodal scam detection pipeline for recorded phone-call audio. It analyzes a `.wav` file in short windows, transcribes speech, scores the transcript for scam intent, scores acoustic urgency/stress signals, adds metadata risk, fuses those signals, and returns a final call-level decision.

The current project direction is offline-first. The working runtime uses local model files where available and avoids cloud API calls during normal inference.

Current active text model: BiLSTM scam detector.

MiniLM is optional and only used as a fallback for uncertain or failed BiLSTM text predictions. Gemma 4 has been removed.

## 2. Repository Source

Primary referenced repo:

```text
https://github.com/Codexx121/dontredeem.git
```

Local workspace:

```text
C:\Users\sufiy\Downloads\DP Scam\dontredeem
```

The local repo has the `codexx` remote configured and fetched. Local `main` already contains the Codexx upstream history plus local changes for offline/model integration.

## 3. Current Runtime Pipeline

The main flow is implemented across:

```text
src/main.py
src/audio.py
src/text.py
src/metadata.py
src/fusion.py
src/analytics.py
```

High-level flow:

```text
Audio file (.wav)
  -> librosa load at 16 kHz
  -> 5 second sliding windows
  -> audio heuristic scoring
  -> Faster Whisper transcription
  -> BiLSTM text scam score
  -> keyword/rule scam score adjustments
  -> metadata score
  -> weighted fusion
  -> asymmetric EMA session smoothing
  -> final decision
```

Final labels are produced by `src/fusion.py`:

```text
score > 0.75          -> Likely Scam
0.49 < score <= 0.75 -> Suspicious
score <= 0.49        -> Safe
```

Benchmark conversion treats both `Likely Scam` and `Suspicious` as scam predictions.

## 4. Model Status

### Working: BiLSTM Scam Detector

Status: Working.

Location:

```text
models/BiLSTM/best_model.pt
models/BiLSTM/scam_tokenizer.pkl
models/BiLSTM/model_config.json
```

Loaded by:

```text
src/text.py
```

Purpose:

```text
Scores transcript text as scam probability.
```

Architecture in code:

```text
Embedding
2-layer bidirectional LSTM
4-head multihead attention
Dense classifier: 512 -> 128 -> 64 -> 2
Softmax output: non-scam / scam
```

Current model loader behavior:

```text
1. Locates local BiLSTM checkpoint.
2. Loads tokenizer pickle.
3. Uses tokenizer vocabulary size when available.
4. Runs on CUDA if available, otherwise CPU.
```

Observed local load test:

```text
BiLSTM model loaded: best_model.pt
Tokenizer loaded: scam_tokenizer.pkl
Device: CPU in the tested environment
```

### Working: Faster Whisper ASR

Status: Working.

Location:

```text
models/faster-whisper-small
```

Loaded by:

```text
src/text.py
```

Purpose:

```text
Transcribes each 5 second audio chunk into English text.
```

Implementation:

```text
from faster_whisper import WhisperModel
WhisperModel(models/faster-whisper-small, device="cuda" or "cpu", compute_type="int8")
```

Observed local load test:

```text
Faster Whisper ASR loaded successfully.
```

### Working: Audio Heuristics

Status: Working.

Location:

```text
src/audio.py
```

Purpose:

```text
Scores acoustic indicators such as pitch, speech/onset rate, and spectral centroid.
```

Signals:

```text
High pitch
Fast urgency / high onset rate
Bright or piercing tone
```

This layer runs even when the speech emotion model is unavailable.

### Optional Fallback: MiniLM

Status: Optional fallback in the current pipeline.

Location:

```text
src/minilm_infer.py
models/minilm
```

Current behavior:

```text
src/text.py uses ScamDetectionModel / BiLSTM as the primary scorer.
MiniLM is used only when BiLSTM cannot score or when BiLSTM is near the decision boundary.
```

MiniLM can be disabled with `ENABLE_MINILM_FALLBACK=0`.

### Removed: Gemma 4

Status: Removed from the local project.

Gemma 4 is no longer part of the active architecture, and `models/gemma-4-E2B` has been removed.

### Working: Speaker Diarization / Pyannote

Status: Working from local files with safe fallback.

Local files exist:

```text
models/pyannote/speaker-diarization-3.1
models/pyannote/segmentation-3.0
models/pyannote/wespeaker-voxceleb-resnet34-LM
```

Current behavior:

```text
src/main.py loads local pyannote from `models/pyannote/speaker-diarization-3.1/config.yaml` when the environment supports it.
Audio is passed to pyannote as an in-memory waveform tensor.
```

Reason:

```text
Pyannote runs on an in-memory waveform tensor and returns speaker turns with
speaker labels, start times, and end times. If pyannote cannot initialize or run,
the pipeline logs the failure and continues without diarization.
```

### Working: Speech Emotion Recognition

Status: Working when the local HuBERT SER model is present.

Expected location:

```text
models/hubert-large-superb-er
```

Current behavior:

```text
src/audio.py loads the local HuBERT feature extractor and sequence classifier.
It extracts the aggression/stress probability for each audio window and blends
that learned emotion score with acoustic heuristics. If the folder is missing,
the system falls back to acoustic heuristics.
```

## 5. Active Score Components

### Text Score

Implemented in:

```text
src/text.py
```

Inputs:

```text
Transcript from Faster Whisper
```

Outputs:

```text
text_score: float from 0.0 to 1.0
analysis_text: explanation string
suspicious_tokens: weighted token list
```

Text scoring combines:

```text
BiLSTM probability
critical keyword detection
scam indicator keywords
false-positive reduction logic for legitimate business context
dangerous action detection
UAE-specific scam patterns such as Emirates ID and deportation threats
```

### Audio Score

Implemented in:

```text
src/audio.py
```

Outputs:

```text
audio_score: float from 0.0 to 1.0
audio_inferences: explanation string
mfcc_shape: debug feature shape
```

Audio scoring currently uses:

```text
HuBERT SER aggression/stress score when the local model is available
pitch
speech/onset rate
spectral centroid
```

If HuBERT SER is unavailable, audio scoring continues with the heuristic features.

### Metadata Score

Implemented in:

```text
src/metadata.py
```

Expected metadata format:

```text
dd/mm/yyyy hh:mm, unsaved
```

Examples:

```text
12/03/2026 23:45, unsaved
12/03/2026 14:30, saved
```

Scoring:

```text
unsaved contact -> +0.5
late-night call from 23:00 to 05:00 -> +0.5
```

### Fusion Score

Implemented in:

```text
src/fusion.py
```

Default weights:

```text
audio:   0.3
text:    0.5
metadata: 0.2
```

Special behavior:

```text
If text_score > 0.85:
  text weight increases to 0.7
  audio weight drops to 0.1

If audio and text are both very low:
  metadata influence is reduced

If text is extremely safe:
  aggressive audio influence is capped
```

### Session Smoothing

Implemented in:

```text
src/analytics.py
```

Current behavior:

```text
RiskAggregator uses an asymmetric exponential moving average with a peak-hold floor.
SessionStateManager tracks total windows, risk history, max spike, and suspicious tokens.
```

Current EMA settings:

```text
alpha_rise: 0.6
alpha_decay: 0.2
peak_hold_ratio: 0.7
```

## 6. Main Entry Points

### Run Main Pipeline

```bash
python src/main.py
```

Important note:

```text
The default audio path is data/sample_ScamConvo.wav and the default metadata path
is data/metadata.txt. For real testing, pass an explicit audio path and metadata
path through the CLI.
```

### Run Benchmark

```bash
python utils_and_tests/benchmark_all.py
```

Benchmark behavior:

```text
Loads every .wav file in data/
Labels files based on filename
Processes each file through audio, ASR, text, metadata, fusion, and EMA
Prints accuracy, precision, recall, F1, and confusion matrix
```

Label parser:

```text
nonscam / non scam / not scam / not-scam / notscam -> safe
scam -> scam
```

### Text-Only Smoke Test

```bash
python -c "import sys; sys.path.insert(0, 'src'); import text; print(text.text_model('urgent send your emirates id and bank otp now')[0])"
```

Observed result in local environment:

```text
BiLSTM loaded successfully.
Faster Whisper loaded successfully.
Returned scam score around 0.73 for the suspicious sample text.
```

## 7. Benchmark Notes

The benchmark was run after confirming:

```text
torch/librosa/sklearn imports work
faster_whisper imports work
pyannote loads locally and runs on in-memory waveform tensors
HuBERT SER loads locally when models/hubert-large-superb-er is present
BiLSTM loads from local files
Faster Whisper loads from local files
```

Final benchmark after the BiLSTM/MiniLM/pyannote refactor, corrected filename labels, and restored asymmetric EMA:

```text
Total Files Analyzed: 19
Average Processing Time: 22.28s per file
Accuracy: 73.68%
Precision: 68.75%
Recall: 100.00%
F1 Score: 81.48%
Confusion Matrix:
  Actual Safe: 3 predicted safe, 5 predicted scam
  Actual Scam: 0 predicted safe, 11 predicted scam
```

Important caveat:

```text
The current thresholding is conservative: recall is excellent on the benchmark, but some safe calls are flagged as suspicious.
```

To regenerate metrics, run:

```bash
python utils_and_tests/benchmark_all.py
```

## 8. Dependencies

Primary dependencies:

```text
torch
transformers
faster-whisper
librosa
numpy
scikit-learn
soundfile
pyannote.audio
```

Install:

```bash
pip install -r requirements.txt
```

Current requirements update:

```text
faster-whisper==1.2.1
```

## 9. Current Offline Status

Offline working:

```text
BiLSTM text scam model
BiLSTM tokenizer
Faster Whisper ASR
Pyannote diarization
HuBERT speech emotion recognition
audio heuristics
metadata parser
fusion logic
asymmetric EMA session smoothing
benchmark harness
```

Ignored/not active:

```text
Gemma 4
```

Conclusion:

```text
The project is runnable offline for the core scam detection pipeline. Diarization
and HuBERT SER are active when their local model folders are present, and both
layers retain clean fallback behavior so the pipeline can still run if an
environment-specific model load fails.
```

## 10. Known Issues and Risks

### 1. Diarization runtime compatibility

`load_diarization()` uses local pyannote files and passes an in-memory waveform
tensor, but pyannote can still fail if the local environment has incompatible
torch, torchcodec, or FFmpeg dependencies.

Recommended improvement:

```text
Keep the clean fallback path, and pin a pyannote/torch/torchcodec/FFmpeg combination if real diarization is required on every machine.
```

### 2. Benchmark labels depend on filenames

The benchmark infers labels from filenames rather than a manifest.

```text
Create data/labels.csv with filename,label and read labels from that file.
```

### 3. Full multimodal benchmark speed

Running Pyannote and HuBERT SER over every file is slower on CPU than the earlier
heuristic-only benchmark.

Recommended improvement:

```text
Profile the pipeline and add a benchmark mode that can toggle diarization and SER for speed comparisons.
```

### 4. README/docs drift

Some older docs mention removed Gemma behavior, old MiniLM-first behavior, DistillBertini paths, or production readiness in ways that do not match the current active path.

Recommended improvement:

```text
Use this document as the current source of truth, then update or archive stale docs.
```

## 11. Suggested Next Steps

1. Tune thresholds to reduce false positives while preserving high recall.
2. Pin and validate the pyannote runtime if real diarization is required for this version.
3. Add `data/labels.csv` so benchmarks do not depend on filename parsing.
4. Profile CPU latency with diarization and HuBERT enabled.
5. Replace older stale docs with links to this document and `docs/BASELINE_CODEXX_COMPARISON.md`.

## 12. Quick Status Table

| Component | Current Status | Active in Pipeline | Offline |
|---|---:|---:|---:|
| BiLSTM scam detector | Working | Yes | Yes |
| BiLSTM tokenizer | Working | Yes | Yes |
| Faster Whisper ASR | Working | Yes | Yes |
| Keyword scam rules | Working | Yes | Yes |
| Audio heuristics | Working | Yes | Yes |
| Metadata parser | Working | Yes | Yes |
| Fusion and EMA | Working | Yes | Yes |
| MiniLM | Optional fallback | Boundary/failure only | Yes, if local files load |
| Gemma 4 | Removed | No | Not relevant |
| Pyannote diarization | Working/fallback | Yes | Environment-dependent |
| HuBERT SER | Working/fallback | Yes, when local model is present | Yes |
