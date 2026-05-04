# Current Pipeline Flowchart

Last updated: 2026-05-04

This map shows the current final DontRedeem runtime after reconciling the baseline repository with the newer Codexx implementation.

```mermaid
flowchart TD
    A["Input .wav call audio"] --> B["Load audio with librosa at 16 kHz"]
    M0["metadata.txt<br/>dd/mm/yyyy hh:mm, saved|unsaved"] --> M1["Parse metadata<br/>src/metadata.py"]
    M1 --> M2["Metadata risk score<br/>unsaved + late-night signals"]

    B --> D0{"Local pyannote config exists?"}
    D0 -->|Yes| D1["Load local pyannote pipeline<br/>models/pyannote/speaker-diarization-3.1/config.yaml"]
    D0 -->|No| D4["Continue without diarization"]
    D1 --> D2["Run diarization on in-memory waveform tensor"]
    D2 --> D3{"Speaker turns found?"}
    D3 -->|Yes| D5["Track active speakers per window"]
    D3 -->|No| D4

    B --> W["Split into 5 second windows"]
    D5 --> W
    D4 --> W

    W --> A1["Audio model<br/>src/audio.py"]
    A1 --> A2{"HuBERT SER model available?"}
    A2 -->|Yes| A3["SER emotion/stress score"]
    A2 -->|No| A4["Acoustic heuristics only<br/>pitch, onset rate, spectral centroid"]
    A3 --> A5["Audio risk score"]
    A4 --> A5

    W --> T0["Transcribe window<br/>local Faster Whisper"]
    T0 --> T1["Rolling transcript buffer<br/>last 3 windows"]
    T1 --> T2["BiLSTM primary scam detector<br/>models/BiLSTM/best_model.pt"]
    T2 --> T3{"BiLSTM failed or near boundary?<br/>0.40 <= score <= 0.60"}
    T3 -->|Yes| T4["Optional MiniLM fallback<br/>models/minilm/best"]
    T3 -->|No| T5["Use BiLSTM score"]
    T4 --> T6["Blend score<br/>80% BiLSTM + 20% MiniLM"]
    T5 --> T7["Keyword/context rules"]
    T6 --> T7
    T7 --> T8["False-positive reduction<br/>legitimacy signals, dangerous actions"]
    T8 --> T9["Text risk score + suspicious tokens"]

    D5 --> V0{"Window is victim-only?"}
    V0 -->|Yes| V1["Suppress text score to 10%"]
    V0 -->|No| V2["Keep text score"]
    T9 --> V0
    V1 --> F0["Score fusion<br/>src/fusion.py"]
    V2 --> F0
    A5 --> F0
    M2 --> F0

    F0 --> F1["Weighted fusion<br/>text 0.5, audio 0.3, metadata 0.2"]
    F1 --> F2{"Critical text override?"}
    F2 -->|Text > 0.85| F3["Boost text weight<br/>text 0.7, audio 0.1"]
    F2 -->|No| F4["Use default / veto-adjusted weights"]
    F3 --> S0["Raw window risk score"]
    F4 --> S0

    S0 --> S1["SessionStateManager<br/>src/analytics.py"]
    S1 --> S2["Asymmetric EMA ratchet<br/>fast rise, slow decay, peak-hold floor"]
    S2 --> S3["Track max spike and suspicious tokens"]
    S3 --> Z{"Final EMA score"}

    Z -->|Score > 0.75| Z1["Likely Scam"]
    Z -->|0.49 < Score <= 0.75| Z2["Suspicious"]
    Z -->|Score <= 0.49| Z3["Safe"]
```

## Component Map

```text
src/main.py
  orchestrates audio loading, diarization, windows, rolling transcript,
  victim-only suppression, fusion, and final report

src/text.py
  loads Faster Whisper, BiLSTM, optional MiniLM fallback, keyword rules,
  legitimacy checks, and text risk scoring

src/audio.py
  computes acoustic heuristic score and optionally uses HuBERT SER if present

src/metadata.py
  converts call metadata into a risk score

src/fusion.py
  combines audio, text, and metadata scores with overrides/vetoes

src/analytics.py
  applies asymmetric EMA ratchet and tracks session-level evidence

utils_and_tests/benchmark_all.py
  runs the full benchmark over data/*.wav
```

## Runtime Model Hierarchy

```text
Primary scam detector:
  BiLSTM

Optional fallback:
  MiniLM only if BiLSTM fails or lands near the boundary

Removed:
  Gemma 4

Optional environment-dependent modules:
  pyannote diarization
  HuBERT speech emotion recognition
```

