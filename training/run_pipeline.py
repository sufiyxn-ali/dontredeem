"""
Complete End-to-End Pipeline for BiLSTM Scam Detection
Desktop Training → Mobile Distillation

Run this script to:
1. Prepare dataset from raw data
2. Build lightweight tokenizer
3. Train BiLSTM on desktop
4. Distill & quantize for mobile
5. Generate mobile deployment files
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path.cwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ Error running {script_name}")
            return False
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SCAM DETECTION PIPELINE - DESKTOP TO MOBILE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    steps = [
        ("prepare_dataset.py", "STEP 1: PREPARE DATASET FROM RAW DATA"),
        ("build_tokenizer.py", "STEP 2: BUILD LIGHTWEIGHT TOKENIZER"),
        ("train_bilstm.py", "STEP 3: TRAIN BILSTM MODEL ON DESKTOP"),
        ("distill_mobile.py", "STEP 4: DISTILL & QUANTIZE FOR MOBILE"),
    ]
    
    completed = []
    
    for script, description in steps:
        success = run_command(script, description)
        
        if not success:
            print(f"\n❌ Pipeline stopped at {script}")
            print(f"Completed steps: {completed}")
            sys.exit(1)
        
        completed.append(script)
        print(f"\n✓ {description} - COMPLETE")
    
    # Final summary
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PIPELINE COMPLETE - READY FOR MOBILE DEPLOYMENT".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\n📊 DELIVERABLES:")
    print("""
    Desktop Training:
    ├── models/DistillBertini/files/output/
    │   ├── train.csv (training data)
    │   ├── val.csv (validation data)
    │   ├── test.csv (test data)
    │   └── full_dataset.csv (complete dataset)
    ├── models/DistillBertini/files/model/
    │   ├── best_model.pt (trained BiLSTM)
    │   ├── scam_tokenizer.pkl (vocabulary)
    │   ├── model_config.json (architecture)
    │   └── training_history.json (metrics)
    
    Mobile Deployment:
    └── models/DistillBertini/files/model/mobile/
        ├── scam_detector.onnx (full model, universal format)
        ├── scam_detector_int8.onnx (quantized, 4x smaller)
        ├── scam_detector.tflite (Android format, optional)
        ├── mobile_config.json (deployment config)
        └── inference_example.py (example code for mobile)
    """)
    
    print("\n🚀 NEXT STEPS:")
    print("""
    For Android:
    1. Use TFLite model with Android Neural Networks API
    2. Use ONNX Runtime for Android
    
    For iOS:
    1. Convert ONNX to Core ML
    2. Use ARM optimized quantization
    
    For Python on Edge:
    1. Use ONNX Runtime directly
    2. Quantized model runs fast on CPU
    """)
    
    print("\n📱 MODEL SPECIFICATIONS:")
    model_config_file = Path("models/DistillBertini/files/model/model_config.json")
    if model_config_file.exists():
        with open(model_config_file, 'r') as f:
            config = json.load(f)
            print(f"""
    Architecture: BiLSTM with Attention
    Parameters: {config.get('num_parameters', 'N/A'):,}
    Vocab Size: {config.get('vocab_size', 'N/A'):,}
    Embedding Dim: {config.get('embedding_dim', 'N/A')}
    Hidden Dim: {config.get('hidden_dim', 'N/A')}
    Test Accuracy: {config.get('test_accuracy', 'N/A'):.4f}
    """)
    
    print("✓ All steps completed successfully!")


if __name__ == "__main__":
    main()
