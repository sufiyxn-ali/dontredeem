import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# 1. Load Speech Emotion Recognition Model via direct inference (bypasses torchcodec)
#    Uses Wav2Vec2FeatureExtractor + HubertForSequenceClassification instead of pipeline()
#    to avoid the broken torchcodec audio decoding path on Windows.
ser_model = None
ser_feature_extractor = None
SER_LABELS = ['neu', 'hap', 'ang', 'sad']  # IEMOCAP emotion labels

try:
    ser_local_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hubert-large-superb-er')
    ser_model_name = ser_local_path if os.path.exists(ser_local_path) else "superb/hubert-large-superb-er"
    
    label = "localized" if os.path.exists(ser_local_path) else "HF cache"
    print(f"Loading SER model ({label}) via direct inference...")
    
    ser_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ser_model_name)
    ser_model = HubertForSequenceClassification.from_pretrained(ser_model_name)
    ser_model.eval()
    print("    [OK] SER model loaded successfully (direct inference mode).")
except Exception as e:
    print(f"Warning: Failed to load SER Transformer ({e}).")
    ser_model = None
    ser_feature_extractor = None

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
    # 1. SER Transformer Score (Direct Inference — bypasses torchcodec)
    # -------------------------
    transformer_score = 0.0
    has_transformer = False
    
    if ser_model is not None and ser_feature_extractor is not None:
        try:
            # Feed raw numpy array directly into the feature extractor
            inputs = ser_feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = ser_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            
            # Extract probability for each IEMOCAP emotion
            emotion_scores = {SER_LABELS[i]: probs[i].item() for i in range(len(SER_LABELS))}
            
            # We extract 'ang' (anger/aggression) as the stress inducer
            transformer_score = emotion_scores.get('ang', 0.0)
            has_transformer = True
            
            if transformer_score > 0.4:
                inferences.append(f"SER (Aggression/Stress): {transformer_score:.2f}")
            else:
                top_emotion = max(emotion_scores, key=emotion_scores.get)
                inferences.append(f"SER ({top_emotion}: {emotion_scores[top_emotion]:.2f})")
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
