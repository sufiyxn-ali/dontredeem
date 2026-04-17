import librosa
from transformers import pipeline

audio_pipeline = pipeline('audio-classification', model='emvo-ai/voiceSHIELD-small', trust_remote_code=True)
y, sr = librosa.load('data/sample_audio.wav', sr=16000)

try:
    results = audio_pipeline('data/sample_audio.wav')
    print(results)
except Exception as e:
    print('ERROR MSG:', e)
