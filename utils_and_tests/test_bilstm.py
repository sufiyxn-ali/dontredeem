#!/usr/bin/env python
"""Quick smoke test for the current BiLSTM-first text pipeline."""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from text import scam_detector, text_model

print("=" * 60)
print("BiLSTM Scam Detection - Quick Test")
print("=" * 60)

print("\n[TEST 1] Model Loading Status:")
if scam_detector is not None and scam_detector.model is not None:
    print("  OK BiLSTM model loaded successfully")
else:
    print("  FAIL BiLSTM model failed to load")

if scam_detector is not None and scam_detector.tokenizer is not None:
    print("  OK Tokenizer loaded successfully")
else:
    print("  FAIL Tokenizer failed to load")

test_samples = [
    ("Your account has been blocked. Click here to verify your identity immediately.", "SCAM"),
    ("Hello, how are you doing today?", "LEGITIMATE"),
    ("You have won a prize! Transfer money to claim your inheritance.", "SCAM"),
    ("The weather is nice today.", "LEGITIMATE"),
]

print("\n[TEST 2] Text Inference:")
for text, expected_label in test_samples:
    score, inference, tokens = text_model(text)
    print(f"\n  Text: '{text}'")
    print(f"  Expected: {expected_label}")
    print(f"  Score: {score:.3f}")
    print(f"  Result: {inference}")
    if tokens:
        print(f"  Top Tokens: {tokens[:3]}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
