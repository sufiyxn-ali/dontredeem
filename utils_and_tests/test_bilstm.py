#!/usr/bin/env python
"""Quick test of BiLSTM scam detection model."""
import sys
sys.path.insert(0, 'd:\\ScamDetectProj\\dontredeem-main\\src')

print("=" * 60)
print("BiLSTM Scam Detection - Quick Test")
print("=" * 60)

from text import bilstm_model, tokenizer, bilstm_inference, text_model

# Test 1: Check if model loaded
print("\n[TEST 1] Model Loading Status:")
if bilstm_model is not None:
    print("  ✓ BiLSTM model loaded successfully")
else:
    print("  ✗ BiLSTM model failed to load")
    
if tokenizer is not None:
    print("  ✓ Tokenizer loaded successfully")
else:
    print("  ✗ Tokenizer failed to load")

# Test 2: Inference on text samples
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
print("Test Complete!")
print("=" * 60)
