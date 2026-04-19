import os
from huggingface_hub import snapshot_download, login

def main():
    print("="*60)
    print("   Offline Deployment - Models Localizer")
    print("="*60)
    print("\nThis script will download the Text Pipeline Models locally.")
    print("It fetches Faster-Whisper-Small and Gemma-4-E2B.\n")
    
    try:
        # Prompt for token (Gemma requires accepted terms on HF)
        token = input("Enter your Hugging Face Token (starts with 'hf_'): ").strip()
        if not token:
            print("No token provided. Exiting.")
            return
            
        print("\n[!] Authenticating...")
        login(token)
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 1. Download Faster Whisper Small
        whisper_repo = "Systran/faster-whisper-small"
        whisper_dir = os.path.join(models_dir, "faster-whisper-small")
        print(f"\n[+] Downloading {whisper_repo} into {whisper_dir}...")
        snapshot_download(repo_id=whisper_repo, local_dir=whisper_dir, local_dir_use_symlinks=False, token=token)

        # 2. Download Gemma 4 E2B
        gemma_repo = "google/gemma-4-E2B"
        gemma_dir = os.path.join(models_dir, "gemma-4-E2B")
        print(f"\n[+] Downloading {gemma_repo} into {gemma_dir}...")
        snapshot_download(repo_id=gemma_repo, local_dir=gemma_dir, local_dir_use_symlinks=False, token=token)
            
        print(f"\n" + "="*60)
        print(f"SUCCESS: Models are now 100% Offline and Localized.")
        print("Your pipeline can now run the Wake-Up Hybrid architecture locally.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[X] Error during localization: {e}")
        print("Ensure you have accepted the Gemma terms of service on Hugging Face!")

if __name__ == "__main__":
    main()
