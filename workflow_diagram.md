# Zero-Trust Scam Detection Overlay — System Workflow

## Complete Pipeline Network Map

```mermaid
flowchart TD
    A["Raw Audio Input\n(.wav file)"] --> B["Librosa Resampling\n16 kHz Mono"]
    A --> C["Pyannote 3.1\nSpeaker Diarization"]
    C --> D["Speaker Turn Map\nCaller vs Victim"]
    B --> E["5-Second Sliding\nWindow Processor"]
    D --> E

    E --> F["Text Engine\nWeight 50%"]
    E --> G["Audio Engine\nWeight 30%"]
    E --> H["Metadata Engine\nWeight 20%"]

    F --> F1["Faster-Whisper ASR\nSpeech to Text"]
    F1 --> F2["BiLSTM + Attention\n98.33% Accuracy"]
    F1 --> F3["Keyword Detection\nCritical + Indicators"]
    F2 --> F4["FP Reduction\nLegitimacy Signals"]
    F3 --> F4
    F4 --> F5["Text Score T"]

    G --> G1["HuBERT SER\nEmotion Detection"]
    G --> G2["Librosa DSP\nMFCC Pitch Rate"]
    G1 --> G3["Ensemble Arbiter\n70% SER 30% DSP"]
    G2 --> G3
    G3 --> G4["Audio Score A"]

    H --> H1["Contact Status\nSaved or Unsaved"]
    H --> H2["Call Timing\nLate Night Check"]
    H1 --> H3["Meta Score M"]
    H2 --> H3

    F5 --> I["Multimodal Fusion\nS = 0.5T + 0.3A + 0.2M"]
    G4 --> I
    H3 --> I

    I --> J["Asymmetric EMA\nRatchet Engine"]
    J --> K{"Final\nDecision"}
    K -->|"S > 0.75"| L["LIKELY SCAM"]
    K -->|"0.49 < S"| M["SUSPICIOUS"]
    K -->|"S < 0.49"| N["SAFE"]
```

---

## How It Works — Step by Step

### Stage 1: Audio Ingestion
The pipeline receives a raw `.wav` audio file. **Librosa** resamples the audio to 16 kHz mono format for consistent processing across all downstream models.

### Stage 2: Speaker Diarization
**Pyannote 3.1** analyzes the audio to identify and separate speakers. This produces a speaker turn map that distinguishes the **caller** (potential scammer) from the **victim**. This is critical because the system must suppress threat scoring when only the victim is talking to avoid false positives.

### Stage 3: Sliding Window Processing
The audio is divided into overlapping **5-second chunks**. Each chunk is independently processed through three parallel analysis engines. This sliding window approach enables real-time processing and temporal tracking of threat levels across the call.

### Stage 4: Three Parallel Engines

**Text Engine (50% weight):**
- Faster-Whisper ASR transcribes speech to text
- BiLSTM with 4-head attention classifies scam probability (98.33% accuracy)
- Critical keyword detector flags terms like "deport", "emirates id", "arrest"
- False positive reduction logic checks for legitimacy signals (job offers, business emails)

**Audio Engine (30% weight):**
- HuBERT SER transformer detects emotional states (anger, stress)
- Librosa extracts acoustic features: MFCC, pitch (>280 Hz = stress), onset rate (>4.5/s = urgency), spectral centroid
- Ensemble arbiter combines SER (70%) with DSP heuristics (30%)

**Metadata Engine (20% weight):**
- Checks if the caller is a saved contact (+0.5 if unsaved)
- Checks call timing (+0.5 if late night between 23:00–05:00)

### Stage 5: Multimodal Fusion
The three engine scores are fused using the weighted formula:

```
S(t) = 0.5 × T(t) + 0.3 × A(t) + 0.2 × M
```

Dynamic adjustments apply:
- If text score > 0.85 → text weight increases to 0.7 (critical text override)
- If text score < 0.1 → audio capped at 0.4 (semantic veto)
- If both audio and text < 0.15 → metadata weight drops to 0.05

### Stage 6: Asymmetric EMA Ratchet
The fused score passes through an Exponential Moving Average smoother (α = 0.5) that:
- **Escalates rapidly** on threat spikes
- **Decays slowly** on safe segments
- Maintains a **peak-hold floor** (70% of max spike) so scammers cannot dilute the final score by adding safe silence at the end

### Stage 7: Final Decision
```
S > 0.75  →  LIKELY SCAM
0.49 < S ≤ 0.75  →  SUSPICIOUS
S ≤ 0.49  →  SAFE
```

---

## BiLSTM Model Architecture

```mermaid
flowchart LR
    A["Input Text"] --> B["Custom Tokenizer\n+ Semantic Markers"]
    B --> C["Embedding\n4729 x 128"]
    C --> D["2-Layer BiLSTM\nhidden 256"]
    D --> E["4-Head\nSelf-Attention"]
    E --> F["FC 512-128\nDropout 0.3"]
    F --> G["FC 128-64\nDropout 0.2"]
    G --> H["Output\nScam or Legit"]
```

The tokenizer injects semantic marker tokens like `[URGENT]`, `[MONEY]`, `[THREAT]`, `[VERIFY]`, and `[PERSONAL]` based on detected patterns in the input text. These markers give the BiLSTM additional context about the type of language being used.

---

## Fusion Logic Detail

```mermaid
flowchart LR
    A["Default Weights\n0.5T 0.3A 0.2M"] --> E["Fused Score S"]
    B["Critical Override\nT>0.85: W_t=0.7"] --> E
    C["Semantic Veto\nT<0.1: cap A=0.4"] --> E
    D["Low Signal\nT,A<0.15: W_m=0.05"] --> E
    E --> F["EMA Smoothing\na*new + 1-a*prev"]
    F --> G["Peak Hold Floor\n70% of Max Spike"]
    G --> H["Decision\nScam Suspicious Safe"]
```

---

## Edge Case Countermeasures

| Attack Vector | System Response |
|---|---|
| **Temporal Evasion** — adding safe silence to dilute score | Asymmetric EMA ratchet locks onto spikes, refuses full decay |
| **Victim False Positives** — victim's own stressed speech | Pyannote diarization suppresses scoring on victim-only segments |
| **Cold-Tone Intimidation** — calm/polite scammer voice | Text weighted at 50%, semantic intent detected regardless of tone |
| **Fast-Talking Evasion** — confusing ASR with rapid speech | Librosa onset rate > 4.0/s triggers urgency penalty independently |
| **Fake Job Scams** — "congratulations, send your passport" | Dangerous action keywords override legitimacy signals |
