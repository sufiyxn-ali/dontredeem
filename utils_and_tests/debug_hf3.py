import librosa
from transformers import pipeline

try:
    print("Loading functioning Transformer (superb/hubert-large-superb-er)...")
    audio_pipeline = pipeline('audio-classification', model='superb/hubert-large-superb-er', trust_remote_code=True)
    y, sr = librosa.load('data/sample_audio.wav', sr=16000)
    results = audio_pipeline(y)
    print("SUCCESS: ", results)
except Exception as e:
    print('ERROR MSG:', e)
