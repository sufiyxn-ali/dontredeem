import librosa
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    print("Loading audio classification pipeline (emvo-ai/voiceSHIELD-small)...")
    audio_pipeline = pipeline("audio-classification", model="emvo-ai/voiceSHIELD-small", trust_remote_code=True)
except Exception as e:
    print(f"Warning: Failed to load voiceSHIELD-small ({e}). Audio scoring will rely on heuristics only.")
    audio_pipeline = None

def extract_audio_features(y, sr):
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 2. Pitch (Fundamental Frequency)
    # Using librosa.yin to estimate fundamental frequency
    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0 = f0[f0 > 0] # Filter out unvoiced frames
    avg_pitch = np.nanmean(f0) if len(f0) > 0 else 0
    
    # 3. Rate of speech (estimated by onset detection)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    rate_of_speech = len(onsets) / duration if duration > 0 else 0
    
    # 4. Tone / Energy (Spectral Centroid)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_centroid = np.mean(cent)
    
    return {
        'mfcc_shape': mfcc.shape,
        'pitch': avg_pitch,
        'rate': rate_of_speech,
        'centroid': avg_centroid
    }

def audio_model(y, sr):
    """
    Returns a score between 0 and 1, and an inferences dictionary
    """
    features = extract_audio_features(y, sr)
    
    pitch = features['pitch']
    rate = features['rate']
    centroid = features['centroid']
    
    # Simple heuristics to compute score and descriptions
    score = 0.0
    inferences = []
    
    # Pitch heuristics
    if pitch > 250:
        score += 0.3
        inferences.append("High pitch (often indicates stress/urgency)")
    elif pitch > 150:
        score += 0.1
        inferences.append("Moderate pitch")
    else:
        inferences.append("Normal/low pitch")
        
    # Rate of speech heuristics
    if rate > 4.0:
        score += 0.4
        inferences.append(f"Fast rate of speech ({rate:.2f} onsets/sec)")
    elif rate > 2.5:
        score += 0.2
        inferences.append(f"Moderate rate of speech ({rate:.2f} onsets/sec)")
    else:
        inferences.append(f"Slow rate of speech ({rate:.2f} onsets/sec)")
        
    # Tone heuristics (spectral centroid)
    if centroid > 3000:
        score += 0.3
        inferences.append("Bright/Piercing tone")
    elif centroid > 1500:
        score += 0.1
        inferences.append("Normal tone")
    else:
        inferences.append("Muffled/Dark tone")
        
    # Normalize score
    score = min(max(score, 0.0), 1.0)
    
    # -------------------------
    # AI Model Scoring
    # -------------------------
    if audio_pipeline is not None:
        try:
            results = audio_pipeline({"array": y, "sampling_rate": sr})
            model_score = 0.0
            
            for res in results:
                if res['label'] == 'malicious':
                    model_score = res['score']
                    break
                    
            # Combine the model score (70% weight) and heuristic score (30% weight)
            score = (score * 0.3) + (model_score * 0.7)
            inferences.append(f"AI Scan (malicious): {model_score:.2f}")
        except Exception as e:
            inferences.append(f"AI classification failed")
            
    # Combine inferences
    detailed_inferences = ", ".join(inferences)
    
    return score, detailed_inferences, features['mfcc_shape']

if __name__ == "__main__":
    # Test script if run directly
    print("Testing audio pipeline with a dummy signal...")
    sr = 16000
    y = np.random.randn(sr * 5) # 5 seconds of noise
    score, inf, shape = audio_model(y, sr)
    print(f"Score: {score}")
    print(f"Inferences: {inf}")
    print(f"MFCC Shape: {shape}")
