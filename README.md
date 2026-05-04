# 🎯 Multilingual Scam Detection System

Production-ready AI pipeline for detecting phone/SMS scam attempts using multimodal analysis with **98.33% accuracy** BiLSTM model.

**Key Stats:**
- ✅ Accuracy: **98.33%** on balanced test set
- 🚀 Speed: **50-150ms** per inference
- 📱 Size: **47 MB** model (mobile-optimized)
- 🎯 Special Detection: UAE fraud patterns (Emirates ID, deportation threats)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/main.py

# Test text analysis
python -c "from src.text import text_model; score, _, _ = text_model('send your passport now'); print(f'{score:.0%}')"
```

## 🎯 Features

- **BiLSTM Model**: Neural network learns scam patterns (formatting, urgency, social engineering)
- **Critical Keywords**: 100% flagged terms (passport, deport, emirates id)
- **Audio Analysis**: Speech emotion recognition + urgency detection
- **Context Awareness**: Reduces false positives on legitimate requests
- **Multimodal Fusion**: Combines text + audio + metadata for robust detection
- **Privacy-First**: 100% on-device, no cloud calls, GDPR compliant
