"""
Example: Running Scam Detector on Mobile/Edge Devices
"""

import onnxruntime as ort
import numpy as np
import pickle

# Load tokenizer
with open('scam_tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

# Create ONNX session (automatic CPU/GPU selection)
sess = ort.InferenceSession('scam_detector_int8.onnx')

def predict_scam(text):
    """Predict if text is scam or legitimate"""
    
    # Tokenize (same as training)
    tokens = text.lower().split()
    token_ids = []
    for token in tokens:
        if token in tokenizer_data['token_to_id']:
            token_ids.append(tokenizer_data['token_to_id'][token])
        else:
            token_ids.append(tokenizer_data['token_to_id']['[UNK]'])
    
    # Pad to max length
    max_len = 256
    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(0)
        attention_mask.append(0)
    token_ids = token_ids[:max_len]
    attention_mask = attention_mask[:max_len]
    
    # Inference
    ort_inputs = {
        'input_ids': np.array([token_ids], dtype=np.int64),
        'attention_mask': np.array([attention_mask], dtype=np.int64)
    }
    logits = sess.run(None, ort_inputs)[0]
    
    # Get prediction probabilities
    probs = softmax(logits[0])
    prediction = np.argmax(probs)
    confidence = float(probs[prediction])
    
    return {
        'prediction': 'SCAM' if prediction == 1 else 'LEGITIMATE',
        'confidence': confidence,
        'scam_probability': float(probs[1])
    }

def softmax(x):
    """Compute softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Example usage
texts = [
    "Your account has been suspended. Verify immediately!",
    "Thank you for calling our support center."
]

for text in texts:
    result = predict_scam(text)
    print(f"Text: {text[:50]}...")
    print(f"Result: {result}\n")
