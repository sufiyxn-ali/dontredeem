import librosa
from transformers import pipeline, AutoFeatureExtractor

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_pipeline = pipeline(
        'audio-classification', 
        model='emvo-ai/voiceSHIELD-small', 
        feature_extractor=feature_extractor,
        trust_remote_code=True
    )
    y, sr = librosa.load('data/sample_audio.wav', sr=16000)
    results = audio_pipeline({'array': y[:sr*5], 'sampling_rate': sr})
    print("SUCCESS: ", results)
except Exception as e:
    print('ERROR MSG:', e)
