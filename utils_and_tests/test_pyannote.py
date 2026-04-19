import os
try:
    from pyannote.audio import Pipeline
    
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'models', 'pyannote', 'speaker-diarization-3.1', 'config.yaml'
    ))
    
    if os.path.exists(config_path):
        print(f"Attempting to load OFFLINE Pyannote from: {config_path}")
        pipeline = Pipeline.from_pretrained(config_path)
        print("Pyannote offline loaded successfully!")
    else:
        print(f"FAILED: Offline config not found at {config_path}")
        print("Please run `python utils_and_tests/localize_pyannote.py` first.")
except Exception as e:
    print(f"FAILED: {e}")
