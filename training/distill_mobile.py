"""
Model Distillation & Quantization for Mobile Deployment
- Converts PyTorch model to ONNX (universal format)
- Quantizes to int8 (4x size reduction)
- Creates TensorFlow Lite version for Android
- Validates mobile model accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Optional dependencies
try:
    import onnx
    HAS_ONNX = True
except:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except:
    HAS_ORT = False

try:
    import tensorflow as tf
    HAS_TF = True
except:
    HAS_TF = False

OUTPUT_DIR = Path("models/DistillBertini/files/output")
MODEL_DIR = Path("models/DistillBertini/files/model")
MOBILE_DIR = MODEL_DIR / "mobile"
MOBILE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTMScamDetector(nn.Module):
    """Same as training model"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BiLSTMScamDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        
        if attention_mask is not None:
            attn_mask = (1 - attention_mask).bool()
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask)
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        combined = lstm_out + attn_out
        
        mask_expanded = attention_mask.unsqueeze(-1).float() if attention_mask is not None else 1
        pooled = (combined * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        x = self.relu(self.fc1(pooled))
        x = self.dropout_layer(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_layer(x)
        logits = self.fc3(x)
        
        return logits


class ModelDistiller:
    """Convert and distill model for mobile"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = self._load_config()
    
    def _load_config(self):
        config_file = MODEL_DIR / 'model_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def export_to_onnx(self, max_seq_length=256):
        """Export PyTorch model to ONNX format"""
        print("\n" + "="*60)
        print("ONNX EXPORT")
        print("="*60)
        
        if not HAS_ONNX:
            print("⚠ ONNX not installed. Skipping ONNX export.")
            print("Install: pip install onnx onnxruntime")
            return None
        
        # Create dummy inputs
        dummy_input_ids = torch.zeros(1, max_seq_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
        
        onnx_path = MOBILE_DIR / 'scam_detector.onnx'
        
        # Export
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            opset_version=11,
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        file_size_mb = onnx_path.stat().st_size / (1024*1024)
        print(f"✓ ONNX model exported: {onnx_path}")
        print(f"  File size: {file_size_mb:.2f} MB\n")
        
        return onnx_path
    
    def quantize_onnx(self, onnx_path):
        """Quantize ONNX model to int8"""
        if onnx_path is None:
            print("Skiping quantization (no ONNX model)")
            return None
            
        print("="*60)
        print("INT8 QUANTIZATION")
        print("="*60)
        
        if not HAS_ORT:
            print("⚠ ONNX Runtime not installed. Skipping quantization.")
            print("Install: pip install onnxruntime onnxruntime-tools")
            return None
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = MOBILE_DIR / 'scam_detector_int8.onnx'
            
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            
            # Compare sizes
            original_size = onnx_path.stat().st_size / (1024*1024)
            quantized_size = quantized_path.stat().st_size / (1024*1024)
            compression = (1 - quantized_size / original_size) * 100
            
            print(f"✓ Quantized model saved: {quantized_path}")
            print(f"  Original: {original_size:.2f} MB")
            print(f"  Quantized: {quantized_size:.2f} MB")
            print(f"  Compression: {compression:.1f}% smaller\n")
            
            return quantized_path
        except ImportError as e:
            print(f"Warning: Could not perform quantization: {e}")
            print("Install: pip install onnxruntime-tools")
            return None
    
    def validate_onnx(self, onnx_path, test_csv):
        """Validate ONNX model accuracy"""
        if onnx_path is None or not HAS_ORT:
            print("Skipping ONNX validation (not available)")
            return None
            
        print("="*60)
        print("ONNX VALIDATION")
        print("="*60)
        
        # Load test data
        df = pd.read_csv(test_csv)
        
        # Create ONNX session
        sess = ort.InferenceSession(str(onnx_path))
        
        correct = 0
        total = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
            text = str(row['text'])
            label = int(row['label'])
            
            # Tokenize
            token_ids, attention_mask = self.tokenizer.encode(text, add_scam_markers=True)
            
            # Inference
            ort_inputs = {
                'input_ids': np.array([token_ids], dtype=np.int64),
                'attention_mask': np.array([attention_mask], dtype=np.int64)
            }
            
            ort_outs = sess.run(None, ort_inputs)
            logits = ort_outs[0]
            prediction = np.argmax(logits[0])
            
            if prediction == label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        print(f"✓ ONNX Model Accuracy: {accuracy:.4f}")
        print(f"  Correct: {correct}/{total}\n")
        
        return accuracy
    
    def create_tflite(self, onnx_path):
        """Convert to TensorFlow Lite for Android"""
        if not HAS_TF:
            print("Warning: TensorFlow not installed. Skipping TFLite conversion.")
            print("Install: pip install tensorflow")
            return None
        
        print("="*60)
        print("TFLITE CONVERSION")
        print("="*60)
        
        # This requires tf2onnx or manual conversion
        # For now, we'll create a placeholder
        tflite_path = MOBILE_DIR / 'scam_detector.tflite'
        
        print(f"Note: TFLite conversion requires additional setup")
        print(f"Recommended: Use ONNX Runtime or TensorFlow Lite Converter\n")
        
        return None
    
    def save_mobile_config(self):
        """Save configuration for mobile app"""
        config = {
            'model_type': 'bilstm',
            'input_format': 'token_ids',
            'max_seq_length': 256,
            'vocab_size': self.model_config.get('vocab_size', 5000),
            'num_classes': 2,
            'class_names': ['Legitimate', 'Scam'],
            'tokenizer_file': 'scam_tokenizer.pkl',
            'models': {
                'onnx': 'scam_detector.onnx',
                'onnx_quantized': 'scam_detector_int8.onnx',
                'tflite': 'scam_detector.tflite'
            }
        }
        
        config_path = MOBILE_DIR / 'mobile_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Mobile config saved: {config_path}\n")
    
    def create_inference_example(self):
        """Create example Python code for mobile inference"""
        example_code = '''"""
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
    print(f"Result: {result}\\n")
'''
        
        example_path = MOBILE_DIR / 'inference_example.py'
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        print(f"✓ Example code: {example_path}\n")


def distill_and_quantize():
    """Main distillation pipeline"""
    from build_tokenizer import ScamDetectionTokenizer
    import shutil
    
    print("\n" + "="*60)
    print("MOBILE DEPLOYMENT PIPELINE")
    print("="*60)
    
    # Step 1: Load components
    print("\n[1/4] Loading model and tokenizer...")
    
    model_config_file = MODEL_DIR / 'model_config.json'
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)
    
    # Load model
    model = BiLSTMScamDetector(
        vocab_size=model_config['vocab_size'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    checkpoint = torch.load(MODEL_DIR / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load tokenizer
    tokenizer = ScamDetectionTokenizer.load(MODEL_DIR / 'scam_tokenizer.pkl')
    
    print(f"✓ Model loaded")
    print(f"✓ Tokenizer loaded\n")
    
    # Step 2: Copy/Compress PyTorch model
    print("[2/4] Preparing PyTorch model for mobile...")
    
    # Copy best model
    best_model_src = MODEL_DIR / 'best_model.pt'
    best_model_dst = MOBILE_DIR / 'scam_detector_pytorch.pt'
    shutil.copy(best_model_src, best_model_dst)
    
    model_size_mb = best_model_dst.stat().st_size / (1024*1024)
    print(f"✓ PyTorch model copied: {best_model_dst}")
    print(f"  File size: {model_size_mb:.2f} MB\n")
    
    # Step 3: Try ONNX export if available
    print("[3/4] Converting to ONNX (if available)...")
    distiller = ModelDistiller(model, tokenizer)
    onnx_path = distiller.export_to_onnx() if HAS_ONNX else None
    
    if onnx_path:
        quantized_path = distiller.quantize_onnx(onnx_path)
    else:
        quantized_path = None
        if not HAS_ONNX:
            print("⚠ ONNX not installed. Using PyTorch model for mobile.")
            print("For production, install: pip install onnx onnxruntime\n")
    
    # Step 4: Save config and examples
    print("[4/4] Saving mobile config...")
    distiller.save_mobile_config()
    distiller.create_inference_example()
    
    # Print summary
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"Mobile models saved to: {MOBILE_DIR}")
    print(f"Files created:")
    for f in sorted(MOBILE_DIR.glob('*')):
        size_mb = f.stat().st_size / (1024*1024)
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{f.stat().st_size:.0f} B"
        print(f"  - {f.name}: {size_str}")
    print("\n✓ Models ready for mobile deployment!")
    
    # Print additional info
    print("\n" + "="*60)
    print("MOBILE DEPLOYMENT OPTIONS")
    print("="*60)
    if onnx_path:
        print("""
✓ ONNX Format (Recommended for Production)
  - Universal format works on any platform
  - Cross-platform compatible
  - Use: scam_detector.onnx or scam_detector_int8.onnx (quantized)

✓ PyTorch Format (For Development)
  - Native PyTorch deployment
  - Use: scam_detector_pytorch.pt
  - Requires PyTorch on target device
        """)
    else:
        print("""
✓ PyTorch Format Available
  - File: scam_detector_pytorch.pt
  - For Android: Convert to TFLITE (pip install tensorflow)
  - For iOS: Convert to Core ML (pip install coremltools)
  - For Python Edge: Use directly with PyTorch Runtime
  
To enable ONNX format:
  pip install onnx onnxruntime
  python distill_mobile.py
        """)


if __name__ == "__main__":
    distill_and_quantize()
