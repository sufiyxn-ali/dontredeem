import os
import torch
import librosa
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Load Speech Emotion Recognition Model (prefer local, fallback to HF cache)
try:
    ser_local_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hubert-large-superb-er')
    if os.path.exists(ser_local_path):
        print("Loading localized SER model (hubert-large-superb-er)...")
        audio_pipeline = pipeline("audio-classification", model=ser_local_path, trust_remote_code=True)
    else:
        print("Loading SER model from HF cache (superb/hubert-large-superb-er)...")
        audio_pipeline = pipeline("audio-classification", model="superb/hubert-large-superb-er", trust_remote_code=True)
except Exception as e:
    print(f"Warning: Failed to load SER Transformer ({e}).")
    audio_pipeline = None

def extract_audio_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0 = f0[f0 > 0]
    avg_pitch = np.nanmean(f0) if len(f0) > 0 else 0
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    rate_of_speech = len(onsets) / duration if duration > 0 else 0
    
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_centroid = np.mean(cent)
    
    return {
        'mfcc_shape': mfcc.shape,
        'pitch': avg_pitch,
        'rate': rate_of_speech,
        'centroid': avg_centroid
    }

def audio_model(y, sr):
    features = extract_audio_features(y, sr)
    inferences = []
    
    # -------------------------
    # Base Acoustic Heuristics
    # -------------------------
    heuristic_score = 0.0
    if features['pitch'] > 250:
        heuristic_score += 0.2
        inferences.append("High pitch (stress marker)")
    elif features['pitch'] > 150:
        heuristic_score += 0.05
        
    if features['rate'] > 4.0:
        heuristic_score += 0.25
        inferences.append(f"Fast urgency ({features['rate']:.1f} ons/s)")
    elif features['rate'] > 2.5:
        heuristic_score += 0.1
        
    if features['centroid'] > 3000:
        heuristic_score += 0.15
        inferences.append("Bright/Piercing Tone")
    elif features['centroid'] > 1500:
        heuristic_score += 0.05
    heuristic_score = min(max(heuristic_score, 0.0), 1.0)
    
    # -------------------------
    # 1. SER Transformer Score
    # -------------------------
    transformer_score = 0.0
    has_transformer = False
    
    if audio_pipeline is not None:
        try:
            results = audio_pipeline(y)
            # Find the probability of 'ang' (anger/aggression/high-stress)
            for res in results:
                # IEMOCAP Labels: neu, hap, ang, sad. We extract 'ang' as stress inducer.
                # If they are angry/aggressive, it correlates strongly with scam pressure techniques.
                if res['label'] == 'ang':
                    transformer_score = res['score']
                    has_transformer = True
                    break
            if transformer_score > 0.4:
                inferences.append(f"SER (Aggression/Stress): {transformer_score:.2f}")
            else:
                inferences.append(f"SER (Calm/Neutral)")
        except Exception as e:
            inferences.append(f"SER offline ({e})")
            
    # -------------------------
    # ENSEMBLE ARBITER
    # -------------------------
    if has_transformer:
        # Heavily weight the semantic audio model (70%) vs physical librosa features (30%)
        final_score = (transformer_score * 0.70) + (heuristic_score * 0.30)
    else:
        # Fallback to pure physical acoustic calculation
        final_score = heuristic_score
        
    final_score = min(max(final_score, 0.0), 1.0)
    return final_score, ", ".join(inferences), features['mfcc_shape']
