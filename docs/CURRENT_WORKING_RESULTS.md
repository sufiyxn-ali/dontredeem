# Current Working State and Results

Last updated: 2026-05-05

## Summary

The project is currently a local, offline-first multimodal scam detection
pipeline for recorded phone-call audio. It processes `.wav` files in short
windows, transcribes each window, scores scam intent from text, evaluates
acoustic and emotion cues, uses call metadata, and fuses the evidence into a
final call-level decision.

The active labels are:

- `Safe`
- `Suspicious`
- `Likely Scam`

For benchmark reporting, both `Suspicious` and `Likely Scam` are counted as
scam predictions because the system is designed as a conservative warning
pipeline.

## Active Runtime Pipeline

```text
.wav audio
  -> librosa audio loading at 16 kHz
  -> local Pyannote speaker diarization
  -> 5 second sliding windows
  -> HuBERT speech emotion recognition + acoustic features
  -> local Faster Whisper transcription
  -> rolling transcript buffer
  -> BiLSTM primary scam detector
  -> optional MiniLM fallback for uncertain BiLSTM cases
  -> keyword and false-positive reduction rules
  -> metadata risk score
  -> weighted fusion
  -> asymmetric EMA session smoothing
  -> Safe / Suspicious / Likely Scam
```

## Working Components

| Component | Status | Notes |
|---|---:|---|
| BiLSTM scam detector | Working | Primary text scam classifier loaded from local model files. |
| Custom tokenizer | Working | Lightweight tokenizer used by the BiLSTM model. |
| Faster Whisper ASR | Working | Local `faster-whisper-small` model transcribes audio windows. |
| Pyannote diarization | Working | Local pipeline loads and returns speaker turns from waveform tensors. |
| HuBERT SER | Working | Local HuBERT emotion model returns stress/aggression probabilities. |
| Audio heuristics | Working | Pitch, onset rate, and spectral centroid supplement HuBERT SER. |
| Metadata parser | Working | Scores unsaved contacts and late-night calls. |
| Score fusion | Working | Combines text, audio/emotion, and metadata signals. |
| Session smoothing | Working | Uses asymmetric EMA with a peak-hold floor. |
| MiniLM fallback | Optional | Used only when BiLSTM fails or is near the decision boundary. |

## Verified Model Behavior

### Pyannote Speaker Diarization

The local Pyannote diarization pipeline was validated on a real scam audio file.
It loaded successfully and returned speaker turns with speaker labels, start
times, and end times. The runtime passes audio to Pyannote as an in-memory
waveform tensor, which avoids the audio decoding issues that can occur on some
Windows environments.

### HuBERT Speech Emotion Recognition

The local HuBERT SER model was downloaded into `models/hubert-large-superb-er`
and validated through the project audio module. It loaded successfully and
returned an aggression/stress score for an audio window. The audio risk score
therefore uses both learned emotion recognition and acoustic heuristics.

## Benchmark Results

Latest stable 19-file benchmark:

```text
Total Files Analyzed: 19
Average Processing Time: 22.28s per file
Accuracy: 73.68%
Precision: 68.75%
Recall: 100.00%
F1 Score: 81.48%
```

Confusion matrix:

|  | Predicted Safe | Predicted Scam |
|---|---:|---:|
| Actual Safe | 3 | 5 |
| Actual Scam | 0 | 11 |

## Interpretation

The current configuration is intentionally conservative. It detected all scam
audio files in the benchmark, giving 100% recall. The tradeoff is that several
safe calls were marked as suspicious, especially when they contained vocabulary
that overlaps with real scams, such as bank, account, verify, Amazon, Microsoft,
or Emirates ID.

This behavior is acceptable for a warning-oriented prototype because missed scam
calls are more harmful than cautionary false positives. The main improvement
area is precision tuning.

## Current Limitations

- The benchmark set is small and should be expanded.
- Benchmark labels are still inferred from filenames rather than a manifest.
- Full diarization plus HuBERT benchmarking is slower on CPU.
- Some safe calls with financial, identity, or technical-support vocabulary can
  be flagged as suspicious.
- Mobile deployment still needs model conversion and latency optimization.

## Recommended Next Steps

1. Add a `data/labels.csv` manifest to replace filename-based benchmark labels.
2. Tune thresholds and false-positive rules while preserving high recall.
3. Add more safe calls containing banking, delivery, identity, and technical
   support language.
4. Convert the BiLSTM model for mobile deployment with TorchScript, ONNX, or
   TensorFlow Lite.
5. Build a streaming Android prototype that processes microphone or uploaded
   audio in short windows.
