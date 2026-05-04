import os
from huggingface_hub import snapshot_download, login

def main():
    print("="*60)
    print("   Offline Deployment - Models Localizer")
    print("="*60)
    print("\nThis script will download the Text Pipeline Models locally.")
    print("It fetches Faster-Whisper-Small for local ASR.\n")
    
    try:
        token = input("Enter your Hugging Face Token if needed, otherwise press Enter: ").strip()
        if token:
            print("\n[!] Authenticating...")
            login(token)
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 1. Download Faster Whisper Small
        whisper_repo = "Systran/faster-whisper-small"
        whisper_dir = os.path.join(models_dir, "faster-whisper-small")
        print(f"\n[+] Downloading {whisper_repo} into {whisper_dir}...")
        snapshot_download(repo_id=whisper_repo, local_dir=whisper_dir, local_dir_use_symlinks=False, token=token or None)
            
        print(f"\n" + "="*60)
        print(f"SUCCESS: Models are now 100% Offline and Localized.")
        print("Your pipeline can now run local ASR with the BiLSTM scam detector.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[X] Error during localization: {e}")
        print("Check your network connection, token, and Hugging Face access.")

if __name__ == "__main__":
    main()
