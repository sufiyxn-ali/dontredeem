try:
    from pyannote.audio import Pipeline
    # Attempt to load open-source diarization
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("Pyannote loaded successfully!")
except Exception as e:
    print(f"FAILED: {e}")
