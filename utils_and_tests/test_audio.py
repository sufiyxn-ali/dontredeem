import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import run_pipeline

def test_audiotest_folder():
    audiotest_dir = os.path.join(os.path.dirname(__file__), '..', 'audiotest')
    meta_path = os.path.join(audiotest_dir, 'metadata.txt')
    
    if not os.path.exists(meta_path):
        with open(meta_path, 'w') as f:
            f.write("12/03/2026 23:45, unsaved")
            
    files = [f for f in os.listdir(audiotest_dir) if f.endswith('.wav')]
    print(f"Found {len(files)} audio files to test.")
    
    for f in files:
        audio_path = os.path.join(audiotest_dir, f)
        print(f"\n\nTesting {audio_path}")
        try:
            run_pipeline(audio_path, meta_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    test_audiotest_folder()
