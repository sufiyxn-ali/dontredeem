# Baseline vs Codexx Comparison

Last updated: 2026-05-04

## Repositories Compared

Baseline:

```text
https://github.com/sufiyxn-ali/dontredeem.git
remote ref: origin/main
commit: d799a9d
```

Newer Codexx implementation:

```text
https://github.com/Codexx121/dontredeem.git
remote ref: codexx/main
commit: a16e2b5
```

## Baseline Architecture

The baseline project had a multimodal scam detection architecture focused on local/private analysis:

```text
audio ingestion
  -> pyannote diarization attempt
  -> 5 second windows
  -> faster-whisper transcription
  -> BiLSTM/MiniLM/Gemma-era text scoring
  -> audio stress heuristics
  -> metadata score
  -> fusion
  -> asymmetric EMA ratchet
```

Useful baseline components:

```text
asymmetric EMA with peak-hold floor
rolling transcript buffer across recent windows
victim-only diarization suppression
local faster-whisper path
local pyannote config loading
privacy-first architecture direction
```

Weaknesses in the baseline:

```text
Gemma wake-up path was heavy and difficult to maintain locally
MiniLM semantic similarity was used too broadly
some code paths referenced external model downloads
main.py relied on hardcoded sample paths
test scripts contained stale absolute Windows paths
```

## Codexx Architecture

Codexx added the strongest model and data improvements:

```text
BiLSTM scam detector and tokenizer
BiLSTM training and deployment scripts
expanded keyword/context rules
false-positive reduction for legitimate business contexts
benchmark harnesses
dataset and model documentation
offline model folders
```

Useful Codexx components:

```text
BiLSTM text model architecture
custom tokenizer
UAE-specific scam patterns
context-aware false-positive reduction
benchmark_all.py
training pipeline
clear model docs
```

Weaknesses in Codexx:

```text
text.py still initialized MiniLM as the effective scorer in parts of the code
local ASR wiring was inconsistent before the refactor
analytics regressed from asymmetric EMA to a simple EMA
main.py lost rolling transcript and victim-only suppression
benchmark labels were inferred incorrectly for "not scam" filenames
Gemma artifacts were not useful for the final lightweight pipeline
```

## Final Design Decision

The final implementation keeps the best pieces from both:

```text
BiLSTM is the primary text scam detector.
MiniLM is optional and only runs as a fallback for failed or uncertain BiLSTM scores.
Gemma 4 is removed.
Faster Whisper is loaded from the local model directory.
Pyannote is loaded from the local config and receives in-memory waveform tensors.
Rolling transcript context is restored.
Victim-only diarization suppression is restored.
Asymmetric EMA with peak-hold is restored.
Benchmark filename labeling is corrected.
```

## Evaluation

Performance:

```text
Best: final implementation.
Reason: BiLSTM is much lighter than Gemma, faster than a transformer/LLM wake-up path, and MiniLM is lazy instead of always-on.
```

Readability:

```text
Best: final implementation.
Reason: model roles are explicit: BiLSTM primary, MiniLM fallback, Gemma removed, pyannote optional.
```

Scalability:

```text
Best: final implementation.
Reason: the pipeline remains modular across src/audio.py, src/text.py, src/main.py, src/fusion.py, and src/analytics.py.
```

Maintainability:

```text
Best: final implementation.
Reason: fewer heavyweight models, corrected tests, documented model status, and graceful fallbacks for optional local models.
```

## Validation

Completed checks:

```text
BiLSTM local load smoke test
MiniLM optional fallback smoke test
Faster Whisper local load smoke test
Pyannote local load smoke test
Pyannote inference on data/scam - 1 Microsoft.wav returned 30 speaker turns
Full benchmark completed after label parser correction
```

Latest full benchmark after restoring asymmetric EMA:

```text
Total files: 19
Average processing time: 22.28s per file
Accuracy: 73.68%
Precision: 68.75%
Recall: 100.00%
F1: 81.48%
Confusion matrix:
  Actual safe: 3 safe, 5 scam
  Actual scam: 0 safe, 11 scam
```

This confirms the ratchet improves scam recall and F1, while pushing the system toward a conservative false-positive profile.
