#!/usr/bin/env python
"""
Quick examples showing how to use the enhanced ScamScorer with explanations.
"""

from infer import ScamScorer
import json

print("=" * 80)
print("EXAMPLE 1: Single Text with Explanation (Attention-Based)")
print("=" * 80)

scorer = ScamScorer("model/", explain=True)
text = "Hi there. I am calling from the technical support team. Your computer has a virus. Please provide your information."

result = scorer.score(text)
print(f"\nText: {text}")
print(f"\nPrediction: {result.label.upper()}")
print(f"Confidence: {result.scam_probability:.2%}")
print(f"\nTop Suspicious Tokens (by model attention weight):")
if result.suspicious_tokens:
    for i, (token, weight) in enumerate(result.suspicious_tokens[:8], 1):
        bar = "█" * int(weight * 500)  # visual bar
        print(f"  {i:2}. {token:15} {weight:.4f} {bar}")

print("\n" + "=" * 80)
print("EXAMPLE 2: Batch Processing (Faster)")
print("=" * 80)

# Initialize without explain for batch (faster)
batch_scorer = ScamScorer("model/", explain=False)

texts = [
    "Your account will be suspended. Click link to verify.",
    "Hey! Can we meet for coffee this weekend?",
    "URGENT: Bank transfer failed. Call immediately.",
    "The weather is nice today.",
    "Claim your free iTunes gift card now!",
]

print("\nScoring 5 texts (fast, no explanations):")
results = batch_scorer.score_batch(texts)
for text, result in zip(texts, results):
    emoji = "🚨" if result.label == "scam" else "✓"
    prob = f"{result.scam_probability:.1%}".rjust(6)
    print(f"{emoji} {prob} scam  | {text[:55]:55}")

print("\n" + "=" * 80)
print("EXAMPLE 3: Different Thresholds")
print("=" * 80)

conservative_scorer = ScamScorer("model/", threshold=0.8, explain=False)
lenient_scorer = ScamScorer("model/", threshold=0.2, explain=False)

ambiguous = "Click here for special offer on products"
result_normal = scorer.score(ambiguous)
result_conservative = conservative_scorer.score(ambiguous)
result_lenient = lenient_scorer.score(ambiguous)

print(f"\nText: {ambiguous}")
print(f"Scam probability: {result_normal.scam_probability:.4f}\n")
print(f"Threshold 0.5 (normal):      {result_normal.label:10} ← Standard")
print(f"Threshold 0.8 (conservative): {result_conservative.label:10} ← Strict (fewer false positives)")
print(f"Threshold 0.2 (lenient):      {result_lenient.label:10} ← Loose (fewer false negatives)")

print("\n" + "=" * 80)
print("EXAMPLE 4: JSON Export (for integration)")
print("=" * 80)

scorer_json = ScamScorer("model/", explain=True)
result = scorer_json.score("Congratulations! You won a prize!")
print(json.dumps(result.to_dict(), indent=2))

print("\n" + "=" * 80)
print("PERFORMANCE TIPS")
print("=" * 80)
print("""
1. Batch processing: 3-5x faster than single texts
   results = scorer.score_batch(["text1", "text2", ...])

2. Disable explanations if not needed: 10-15% faster
   scorer = ScamScorer("model/", explain=False)

3. Reuse scorer object: Initialize once, score many times
   scorer = ScamScorer("model/")
   for text in large_list:
       result = scorer.score(text)

4. Use GPU if available: Model auto-detects CUDA
   # Automatic, no code needed!

5. For 1000+ texts: batch + no explain = ~5-10ms per text
""")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
Attention weights show which tokens the model attends to, but remember:

✓ High attention = Token influenced the prediction
⚠ Doesn't mean the token CAUSED the prediction (correlation vs causation)
⚠ Multiple tokens can collectively cause a scam classification
⚠ Some tokens matter through their combinations with others

Example: "Click here" might be neutral, but combined with urgency language
becomes a scam indicator. Attention captures this interaction implicitly.
""")
